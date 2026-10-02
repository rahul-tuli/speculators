# Deliverable 4 — Deploying Quantized DFlash/DSpark Drafters in vLLM

What has to change to serve a compressed-tensors–quantized (FP8 W8A8 / NVFP4 W4A4 / FP8_DYNAMIC) speculator checkpoint, and how to validate it. Verified against vllm @ `2e2fdaa99a`. Patch sketches are design-level; line numbers are from the pinned commit.

## 0. Baseline serving (works today, bf16 drafter)

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve Qwen/Qwen3-8B \
  --speculative-config '{"method":"dflash",
      "model":"RedHatAI/Qwen3-8B-speculator.dflash",
      "num_speculative_tokens":7}'
```

- The V2 runner is **required for DSpark** (no dspark branch in the V1 switch, `vllm/v1/worker/gpu_model_runner.py:628-700`); DFlash works on both but some features (mixed sliding/full attention) are V2-only (`vllm/model_executor/models/qwen3_dflash.py:122-130`). Standardize on V2.
- Draft quant config is honored independently of the target's (`get_draft_quant_config`, `vllm/model_executor/models/utils.py:952-971`; draft ModelConfig built at `vllm/config/speculative.py:1269-1293`). Mixed deployments (FP8 target + bf16 draft and vice versa) are already e2e-tested for the `draft_model` method (`tests/v1/e2e/spec_decode/draft_model/test_draft_model.py:128-149`).

## 1. Required change A — DFlash context-KV precompute (the blocker)

**Problem.** `DFlashQwen3Model` precomputes a fused context-KV buffer by reading **raw weights** and running plain `F.linear`:

- `_build_context_kv_buffers` / `_project_context_kv` read `a.qkv_proj.weight[a.q_size:]` (`qwen3_dflash.py:472-478, 545-547`), consumed by `precompute_and_store_context_kv` (`:572-643`).
- A compressed-tensors FP8/NVFP4 linear has no usable `.weight` — weights live in `weight_packed` (uint8), `weight_scale` (fp8e4m3), `weight_global_scale` (fp32) (e.g. `CompressedTensorsW4A4Fp4.create_weights`, `compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py:53-80`). Result today: AttributeError at load, or silently wrong K/V if `.weight` exists but is stale.

**Local hack (~20 lines).** In `_build_context_kv_buffers`, when the qkv_proj carries a quant method, dequantize the K/V slice once at load and build the fused buffer from that bf16 copy:

```python
w = a.qkv_proj
if hasattr(w, "quant_method") and w.quant_method is not None:
    full = w.quant_method.dequantize(w)          # or manual:
    # full = (w.weight_packed viewed as fp4 * w.weight_scale) * w.weight_global_scale
    kv_w = full[a.q_size:]                        # bf16 copy, one-time
else:
    kv_w = w.weight[a.q_size:]
```

Cost: one dequant at load; the K/V slice is a few hundred MB at most; zero runtime overhead (the fused buffer is what the hot path uses anyway).

**Upstream PR design.** Materialize the fused context-KV buffer at `process_weights_after_loading` time using the quant method's dequantized weight, rather than reading `.weight` lazily. Open design question for maintainers: add a standard `get_dequantized_weight()` accessor to the scheme interface vs. special-casing in the DFlash model. Check first whether `qwen3_dspark.py` shares this pattern (5-min grep for `_build_context_kv_buffers`/`F.linear`) — DSpark is otherwise closer to working (its Markov head already has an NVFP4 W4A16 path, `qwen3_dspark.py:109-153`).

## 2. Required change B — V2 DFlash loader quant-config override (3 lines, verify first)

DSpark's loader re-asserts the draft quant config after post-init clobbers it:

```python
# dspark/utils.py:79-81 — VllmConfig post-init restores the target's quant config
draft_vllm_config.quant_config = get_draft_quant_config(vllm_config)
```

The V2 **DFlash** loader lacks this (`dflash/utils.py:26-42`) and relies on the model class calling `get_draft_quant_config` internally (`qwen3_dflash.py:390`). That covers Linears the model constructs explicitly, but any vLLM-internal consumer of `vllm_config.quant_config` during draft load would see the *target's* config.

**Action:** port the DSpark override into `dflash/utils.py` symmetrically. **Verify necessity first:** serve NVFP4 draft + bf16 target; if scheme resolution misfires, this is the fix. Cheap either way.

## 3. Conditional change C — quantized reduced-vocab lm_head

Only if we quantize a **reduced-vocab** drafter's own serialized lm_head (dflash 32k-vocab case): `DFlashQwen3ForCausalLM.lm_head` is built **without** `quant_config` (`qwen3_dflash.py:705-709`), unlike EAGLE3 (test-enforced at `tests/model_executor/test_eagle_quantization.py:118-160`). Full-vocab DSpark shares the target's lm_head (`dspark/utils.py:100-113`) — nothing to do. Skip unless the reduced-vocab experiment lands.

## 4. Optional telemetry changes (for the experiment, not for correctness)

- **Per-position denominators**: export `num_draft_tokens_per_pos` (tracked in `SpecDecodingStats`, `vllm/v1/spec_decode/metrics.py:30-31`) as a Prometheus counter mirroring `vllm:spec_decode_num_accepted_tokens_per_pos` (`metrics.py:247-264`). ~15 lines, upstreamable.
- **Draft/verify time split**: `time.perf_counter` spans around `DFlashSpeculator.propose` (`vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`) and the target verify forward, gated by an env flag, exported as histograms. Hack locally; upstream only if the data earns it.

## 5. Hardware/kernel reality (verified in vLLM kernel registry)

| Drafter scheme            | H100 (sm90)                                                                                        | Blackwell (sm100/120)                                                      |
| ------------------------- | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| bf16                      | ✓                                                                                                  | ✓                                                                          |
| FP8 W8A8                  | ✓ native (`_is_fp8_w8a8_sm90`, `compressed_tensors.py:607`)                                        | ✓                                                                          |
| FP8_DYNAMIC               | ✓                                                                                                  | ✓                                                                          |
| NVFP4 weight-only (W4A16) | ✓ via Marlin FP4 (`kernels/linear/nvfp4/marlin.py:18-39`; warns "no native FP4") — memory win only | ✓ faster paths                                                             |
| NVFP4 **W4A4**            | ⚠ no native FP4 GEMM → emulation/fallback; not perf-representative                                 | ✓ native (CUTLASS/FlashInfer, `nvfp4/cutlass.py`, `flashinfer.py:112-193`) |

Run acceptance evals and FP8 perf on H100s; reserve Blackwell for NVFP4 W4A4 draft-speed/end-to-end numbers. Sanity-check one arm's acceptance on both to rule out kernel-numerics drift.

## 6. Deployment validation checklist (per quantized checkpoint)

1. **Load test**: `vllm serve` with the quantized drafter starts; log shows the expected scheme per module (`fc`, heads respected per the ignore list).
2. **Context-KV correctness**: with patch A, compare draft logits vs. the in-memory QDQ model on a fixed input (cosine sim / top-1 agreement) — catches silent fused-buffer corruption.
3. **Acceptance sanity**: greedy MAL within a few % of the bf16 drafter on GSM8K/HumanEval slice before any full benchmark (model-card reference values: dflash-8B MAL 3.41 HumanEval / 3.74 math; dspark-4B 4.79 / 5.37).
4. **Full Tier-1 sweep**: `vllm bench serve --save-result` × depths × benches × concurrency {1,8}; record `spec_decode_*` JSON + Prometheus snapshot.
5. Record everything in `quant_run_manifest.json` (written by `quantize_drafter.py`) + `results/*.jsonl`.

## 7. Upstreaming split

| Change                            | Local experiment      | Upstream                                                                      |
| --------------------------------- | --------------------- | ----------------------------------------------------------------------------- |
| Context-KV dequant                | hack (§1)             | PR to vLLM — design review on accessor vs. special-case                       |
| DFlash loader quant override      | 3-line patch          | trivial PR (mirror DSpark)                                                    |
| Quantized reduced-vocab lm_head   | skip                  | PR only if reduced-vocab ships                                                |
| Telemetry (per-pos denominators)  | patch                 | small PR                                                                      |
| llm-compressor speculator example | `quantize_drafter.py` | examples/ PR + micro-PR guard for unregistered model_type in `validate_model` |
| compressed-tensors                | none                  | none                                                                          |
