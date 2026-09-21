# drafter-quant — does calibration data matter for drafter quantization?

> The active wayfinder experiment uses the explicit, resumable runner described in [EVALUATION.md](EVALUATION.md). Its fixed protocol supersedes the historical evaluation defaults below, including the 2% heuristic and automatic plotting.

## Calibration input status (2026-09-21)

The real-data loader now applies stored FP8 scales, reads one sample at a time,
checks target/layer/width compatibility, and rejects missing files, misaligned
inputs, non-finite values, and sample-budget shortfalls. The preserved Qwen3-8B
cache contains **2,027 prepared samples**, not 2,048. Choose the calibration budget
explicitly; the loader no longer silently reduces it. Each successful calibration
writes actual sample, token, and loss-mask counts to `calibration_manifest.json`.
New capture paths include target, backend, layers, width, length, and sample budget.

See [validation evidence](wayfinder/calibration-validation/README.md). The broader
experiment decisions and evaluation corrections remain on the
[wayfinder map](https://github.com/rahul-tuli/speculators/issues/101); the original
experimental claims and orchestration examples below are not validated results.

## 1. What this is

One question: **when quantizing DFlash/DSpark speculator drafters (FP8 W8A8 /
NVFP4 W4A4) with llm-compressor, does the calibration dataset matter?** Real
drafter inputs are outlier-heavy target hidden states; activation observers
calibrated on Gaussian randn should mis-cover that distribution and cost
acceptance length. This repo quantizes each drafter twice — once with random
Gaussian calibration tensors (the degenerate baseline, by design), once with a
real on-policy "perfectblend" dataset — and compares acceptance + speed with
the speculators eval utility (GuideLLM + Prometheus acceptance scraping).

Arm matrix (per config, i.e. per target/drafter pair):

| Arm | Scheme | Calibration | Notes |
|---|---|---|---|
| `bf16-drafter` | original BF16 drafter | — | reference acceptance/speed |
| `target-only` | none | — | speed reference; acceptance not applicable |
| `<method>-fp8-random` | `fp8_w8a8` | random Gaussian | static W8A8, degenerate baseline |
| `<method>-fp8-perfectblend` | `fp8_w8a8` | perfectblend 2048 | static W8A8 |
| `<method>-nvfp4-random` | `nvfp4_w4a4` (+ `nvfp4_expanded_mse` weight observer) | random Gaussian | W4A4, degenerate baseline |
| `<method>-nvfp4-perfectblend` | `nvfp4_w4a4` (+ `nvfp4_expanded_mse`) | perfectblend 2048 | W4A4, calibrated `input_global_scale` |
| fp8_dynamic control | `fp8_dynamic` | none (data-free) | weight-focused control; **excluded from the random-vs-real study** — run separately via `scripts/30_quantize.sh` |

Pinned repos (read-only reference checkouts; logic in this repo is grounded in
these exact commits):

| Repo | Commit | Default path (`$DRAFTERS_QUANT_REPOS/<repo>`) |
|---|---|---|
| speculators | `40b12c9` | `/workspace/speculators` |
| llm-compressor | `ed55e6f67` | `/workspace/llm-compressor` |
| compressed-tensors | `698967f` | `/workspace/compressed-tensors` |
| vllm | `2e2fdaa99a` | `/workspace/vllm` |

## 2. Prerequisites

- **GPU:** one H100 is enough for every step. NVFP4 W4A4 *native speed* needs
  Blackwell; on H100 the NVFP4 arms still produce valid **acceptance** numbers
  (which is what this study measures) — just don't read the NVFP4 speed numbers
  as production speed.
- **Disk:** ~200 GB free (worst-case hidden states 96 GB fp8 + HF cache for
  target/drafter weights + quantized checkpoints). See §7.
- **Repos:** the 4 pinned clones above. Override the root with
  `DRAFTERS_QUANT_REPOS` (default `/workspace`).
- **Outputs:** everything the pipeline writes (datasets, hidden states,
  checkpoints, results, logs) goes under `WORK_DIR` (default
  `/data/fast/drafter-quant`); the HF weights/dataset cache goes to `HF_HOME`
  (default `/data/fast/hf_cache`). Both are env-overridable; set them in your
  shell before running if your layout differs.
- **HF access:** Qwen target/drafter checkpoints and the dataset
  `inference-optimization/Qwen3-8B-Regenerated-Collection`.

## 3. Setup

```bash
bash scripts/00_setup.sh
```

Installs (in order — order matters) compressed-tensors, llm-compressor,
speculators (`pip install -e`), plus `hs_connectors` if the speculators
install didn't pull it, then runs import + CUDA sanity checks.

## 4. Step-by-step reproduction

All step scripts source a config env file (`--config`, default
`config/dflash-qwen3-8b.env`; use `config/dspark-qwen3-4b.env` for the 4B
DSpark pair). Every script has `--help` and env-var overrides, and skips its
own outputs if they already exist.

**Step 10 — sample the perfectblend subset** (CPU, minutes; downloads the 13
source JSONLs once):

```bash
bash scripts/10_prepare_dataset.sh
# -> $WORK_DIR/data/perfectblend_2048/perfectblend_2048.jsonl + mix_manifest.json
```

**Ratio-preserving (proportional) sampling**: each task source keeps the same
share of the subset as it has in the full collection (largest-remainder
rounding, deficits redistributed to sources with spare capacity),
deterministic `--seed`. `--strategy uniform` (equal quota per source) remains
available as an ablation option.

**Step 20 — tokenize + capture hidden states offline** (1 GPU, ~hours;
preflight disk check included):

```bash
bash scripts/20_capture_hidden_states.sh
# -> $WORK_DIR/data/perfectblend_2048_prepared/          (Arrow dataset)
# -> $WORK_DIR/data/perfectblend_2048_prepared/hidden_states/hs_<i>.safetensors
```

Launches vLLM with hidden-states extraction (`--hidden-states-backend fp8`),
runs `speculators prepare-data` (server-rendered `input_ids`/`loss_mask`) and
`speculators generate-offline-data` (resumable). Server is stopped by an EXIT
trap.

**Step 30 — quantize one arm** (1 GPU, ~minutes each):

```bash
# degenerate baseline arms (random Gaussian calib — smoke/baseline by design,
# statistically wrong for activation observers; do not report alone)
bash scripts/30_quantize.sh fp8_w8a8 random random
bash scripts/30_quantize.sh nvfp4_w4a4 random random --nvfp4-weight-observer nvfp4_expanded_mse

# real-data arms
bash scripts/30_quantize.sh fp8_w8a8 $WORK_DIR/data/perfectblend_2048_prepared perfectblend
bash scripts/30_quantize.sh nvfp4_w4a4 $WORK_DIR/data/perfectblend_2048_prepared perfectblend \
    --nvfp4-weight-observer nvfp4_expanded_mse

# data-free control (excluded from the random-vs-real study)
bash scripts/30_quantize.sh fp8_dynamic none fp8dyn
# -> $WORK_DIR/out/<model_tag>-<method>-<scheme>-<label>/ + quant_run_manifest.json
```

Note: `random` calibration is the degenerate baseline **by design** — it
exists so the gap (if any) isolates the effect of real calibration data.

**Step 40 — eval one arm** (1 GPU per arm; baseline + 4 quantized arms):

```bash
bash scripts/40_eval.sh none baseline-bf16
bash scripts/40_eval.sh $WORK_DIR/out/qwen3-8b-dflash-fp8_w8a8-random dflash-fp8-random
# -> $WORK_DIR/results/eval_<label>/ : acceptance.csv, perf_results.csv, artifacts/, eval_command.txt
```

Serves the pair with `vllm serve` (`VLLM_USE_V2_MODEL_RUNNER=1`,
`--speculative-config`), then runs speculators `evaluate.py` in **throughput**
mode (acceptance length per subset) and **sweep** mode (TTFT/ITL/output-tps).

**Step 50 — plots + summary** (CPU, seconds; first run may be slow while
matplotlib builds its font cache):

```bash
bash scripts/50_plot.sh
# -> $WORK_DIR/results/plots/plot{1,2,3}_*.png + summary.csv
```

## 5. One-command run (two phases)

The pipeline is split into two phases so checkpoint generation and eval can
run at different times / on different machines (`WORK_DIR` on shared or
copied storage is the handoff):

```bash
# Phase A — dataset + hidden states + all quantized checkpoints
bash scripts/run_checkpoints.sh config/dflash-qwen3-8b.env

# Phase B — serve + eval every arm, then plots
bash scripts/run_evals.sh config/dflash-qwen3-8b.env   # use the SAME config as Phase A
```

Or both back to back:

```bash
bash scripts/run_all.sh config/dflash-qwen3-8b.env    # Qwen3-8B + DFlash
bash scripts/run_all.sh config/dspark-qwen3-4b.env    # Qwen3-4B + DSpark
```

Runs steps 10 → 50 idempotently (existing outputs are skipped with a log
line). Arms: `fp8_w8a8 nvfp4_w4a4` × {random, perfectblend} + the bf16
baseline eval. NVFP4 arms add `--nvfp4-weight-observer nvfp4_expanded_mse`.
`fp8_dynamic` is the data-free control and is excluded from the random-vs-real
study by design — run it separately (§4, step 30).

## 6. Outputs & how to read them

```
$WORK_DIR/                                        (default /data/fast/drafter-quant)
├── data/perfectblend_2048/                       perfectblend_<N>.jsonl, mix_manifest.json
├── data/perfectblend_2048_prepared/              Arrow dataset + hidden_states/hs_*.safetensors
├── out/<tag>-<method>-<scheme>-<label>/          quantized checkpoint + quant_run_manifest.json
├── results/eval_<label>/                         acceptance.csv, perf_results.csv, artifacts/, eval_command.txt
├── results/plots/                                plot1/2/3 PNGs + summary.csv
└── logs/                                         server logs
```

- `plot1_acceptance_by_subset.png` — grouped bars, acceptance length per
  subset × arm. If random-calib ≈ perfectblend-calib, calibration data is
  **not** the bottleneck at these scales. The OOD-hidden-states hypothesis
  predicts a gap mainly for NVFP4 W4A4 (calibrated `input_global_scale`), so
  read FP8 and NVFP4 separately.
- `plot2_speed_vs_acceptance.png` — arms should cluster by *scheme* (speed)
  and separate by *calib* (acceptance) if calibration matters.
- `plot3_per_position.png` — per-position acceptance curves (if the eval CSVs
  carry `acceptance_at_pos_*` columns).
- `summary.csv` — tidy table of everything parsed.

**Decision rule:** calibration data is "relevant" for a scheme if perfectblend
beats random by **>2% relative mean acceptance length on ≥3 of 5 subsets**;
otherwise ship the cheaper recipe. Expect the gap mostly in NVFP4 via the
calibrated `input_global_scale`.

Honest caveat: the random arms are a *degenerate* baseline. A third arm —
real but domain-mismatched data — would isolate "any real data" vs "matched
data" (easy add: rerun step 20 with a single-source subset, e.g. only
`ultrachat_output.jsonl`).

## 7. Space math for hidden states

Stored tensor per sample: `[seq_len, n_aux_layers + 1, hidden]` (dflash-8B:
6 layers × 4096).

| Backend | bytes/token | worst case (2048 × 2048 tok) | realistic (~700 tok avg) |
|---|---|---|---|
| `fp8` (default) | 24 KB | **96 GB** | ~33 GB |
| `file` (bf16) | 48 KB | 192 GB | ~67 GB |

Keep `BACKEND=fp8`: fp8-e4m3 per-token-scaled storage, transparently
dequantized to bf16 by the loader (`hs_connectors/fp8_utils.py`), ~½ the disk.
Caveat to flag: fp8 storage adds a second quantization layer to observer
inputs — if the random-vs-real delta is small, re-run one arm with
`BACKEND=file` to rule out storage noise.

If you can't spare even ~100 GB: online fallback via
`ArrowDataset(on_missing="generate", on_generate="delete")` against a live
server — zero persistent footprint, but regenerates per run (not comparable
across arms) and needs the server alive during calibration. Offline is better
for this study because **all arms reuse the same frozen dataset**.

## 8. SpeedBench (optional)

Default eval dataset is `RedHatAI/speculator_benchmarks` — no setup. To use
NVIDIA SpeedBench instead, pre-split the JSONLs once:

```bash
python $DRAFTERS_QUANT_REPOS/speculators/scripts/evaluate/prepare_speedbench.py
export SPEEDBENCH_DATA_DIR=<dir> DATASET=speedbench/throughput_1k
bash scripts/40_eval.sh none baseline-bf16   # or export before run_all.sh
```

(`throughput_2k/8k/32k` and `qualitative` also exist; see speculators
`docs/user_guide/tutorials/evaluating_performance.md`.)

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `evaluate.py` rejects args | Global args (`--target`, `--dataset`, `--output-dir`) must come **before** the `throughput`/`sweep` subcommand (shape verified against `examples/evaluate/example_qwen3_8b_dflash_humaneval.sh` @ 40b12c9). If your checkout's argparse differs, `--help` will say. |
| `import hs_connectors` fails | The fp8 hidden-states backend needs speculators' `hs_connectors`. `pip install -e speculators` should pull it; otherwise `pip install -e $DRAFTERS_QUANT_REPOS/speculators/hs_connectors` (00_setup.sh does this automatically). |
| "unable to find lm_head" warning during quantize | **Benign, expected.** speculators doesn't override `get_output_embeddings`, so llm-compressor's `disable_lm_head` warns and continues; `lm_head`/`verifier_lm_head` are protected by the explicit ignore list instead. |
| dynamo error on calibration batch 1 | The drafter forward is `torch.compile`-wrapped at class-def time (`dflash/core.py:485`). `dquant/quantize.py` already unwraps it (shim 3); if you bypass that module, re-apply the unwrap. |
| CollateFn / unexpected batch keys | `ArrowDataset.__getitem__` emits extra keys (`lengths`, `position_ids`); the loader wrapper in `dquant/quantize.py` drops them and synthesizes `document_ids` as zeros (single-doc packing). Custom loaders must do the same. |
| First plot run is slow | matplotlib building its font cache — one-time, benign. |
| Plot script prints `[skip]` | acceptance.csv column names vary by speculators version; `dquant/plot_results.py` parses defensively and skips rather than crashes — check the skip messages. |

## 10. Provenance

Every run leaves a manifest — keep these with the results log:

- `$WORK_DIR/out/<arm>/quant_run_manifest.json` — scheme, ignore list, calib
  source, sample count, seq len, seed, timestamp, pinned commits.
- `$WORK_DIR/results/eval_<label>/eval_command.txt` — the exact evaluate.py
  invocation.
- `$WORK_DIR/data/perfectblend_2048/mix_manifest.json` — per-source quotas and
  counts, strategy, seed, HF repo id.
