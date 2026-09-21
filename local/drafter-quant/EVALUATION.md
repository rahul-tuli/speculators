# Reproducible DFlash evaluation

The canonical experiment is [Explain how drafter quantization and calibration data affect speed and acceptance](https://github.com/rahul-tuli/speculators/issues/101). Use `40_eval.sh` and `run_evals.sh` with the explicit controls below; old `eval_<label>` directories and the original automatic plots are not valid completion records.

## References and scope

`bf16-drafter` serves the original drafter with the BF16 target. `target-only` serves the BF16 target without a drafter and reports acceptance as null. The first phase measures acceptance at concurrency 1; speed measurements and DSpark have later validation gates.

Use every example in qualitative, throughput_1k, throughput_2k, and throughput_8k. The user explicitly chose the original first user turn only, including for multi-turn qualitative examples. This is not full-conversation evaluation. Preserve category boundaries and do not remove overlap matches. The maximum generated length is 4096, temperature and evaluation seed are zero, normal EOS stopping applies, and all drafter arms use seven speculative tokens.

## Freeze the data

Preserve NVIDIA's downloaded preparation script, its revision, any cache-only adaptation, logs, and materialized flat files. The helper below produces immutable category files, hashes, exact example identities, token/context bounds, first-turn counts and a normalized exact overlap report against the actual tokenized calibration cache. It refuses missing configurations or placeholder prompts.

```bash
export PYTHONPATH=/workspace/speculators/local/drafter-quant
python -m dquant.freeze_eval_data \
  --data-dir /data/fast/drafter-quant/data/speedbench \
  --output /data/fast/drafter-quant/data/speedbench-frozen \
  --tokenizer /data/fast/hf_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
  --calibration-data /data/fast/drafter-quant/data/perfectblend_2048_prepared
```

Prepared prompt files remain local because their source redistribution restrictions apply. Publish manifests and hashes with the preparation recipe. The historical hidden-state capture lacks original target-weight hashes; present-day hashes cannot repair that omission.

## Run and resume

Set `DATA_MANIFEST` to `manifest.json`, `EVAL_DATASETS` to newline-separated absolute category JSONL paths, `EVAL_REPEAT` to 0, 1 or 2, and `ARM_ORDER` to the actual position in the rotated round. All records are separated by phase, arm, repeat, content-derived run identity, concurrency and workload. Use `run_evals.sh` for the three rotated rounds; it expects full-budget checkpoints named with their scheme, recipe and calibration seed. Existing one-sample exports are pilot inputs only.

```bash
EVAL_PHASE=acceptance EVAL_REPEAT=0 ARM_ORDER=0 \
  bash local/drafter-quant/scripts/40_eval.sh \
  RedHatAI/Qwen3-8B-speculator.dflash bf16-drafter
```

The launcher resolves cached model IDs to actual local snapshots and always writes vLLM commands, patches and checkpoint hashes. The run identity includes checkpoint/config hashes, actual calibration budgets and seed policy, package/source identities, hardware, backend flags and runtime module/scale inventories. NVFP4 selects W4A4 emulation explicitly; H100 timings must never be described as native NVFP4 timings. Quantized context KV uses its actual projection kernels, including activation quantization.

Each point warms up separately, then saves raw request records and counter snapshots. Completion requires exact coverage of the frozen prompts, successful nonempty generations, consistent counters and hashes for all retained artifacts. Warm-up counts do not enter measured counters or timings. Interrupted attempts remain on disk and retry into a new attempt. A changed identity uses a new run directory; changed or missing completed artifacts fail validation instead of being silently reused.

Report draft events, proposed/accepted token totals and accepted tokens by position. Compute acceptance length as `1 + A/D`, accepted-token fraction as `A/P`, and unconditional position acceptance as `A_i/D`. Target-only acceptance is not applicable. Report repeats and calibration seeds separately; mean/range are descriptive, not confidence intervals or equivalence evidence.

Speed mode uses explicit concurrency points, with a separate counter interval per point. It must pass its own compiled-runtime and warm-up validation before full execution. Initial eager pilots validate acceptance readiness only. The driver checks for competing GPU workloads at each point’s entry and refuses DSpark until its head-scope gate is implemented.
