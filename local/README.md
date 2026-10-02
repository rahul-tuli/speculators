# Speed-up experiment handoff: DFlash and DSpark quantized drafters

**Purpose:** measure end-to-end generated output-token throughput for the selected quantized drafters against their BF16 drafter on the same target model, with latency and acceptance reported alongside throughput.

**Run status:** plan only. No performance experiment has been run from this handoff.

## Experiment decisions

- Hardware: one NVIDIA B200 GPU; tensor parallelism 1.
- Speculation: 7 draft tokens for every arm.
- Workload: SPEED-Bench `throughput_2k`, split into low-, mixed-, and high-entropy runs.
- Mode: the speculators `evaluate.py sweep` utility, one measured sweep for each arm and entropy tier.
- Sweep budget: five sweep points (`--sweep-rate 5`), at most 100 requests per point (`--max-requests 100`), maximum concurrency 128. The five offered rates are selected and recorded by GuideLLM; they are not fixed QPS values in this plan.
- Repeats: one pass per arm. Treat results as descriptive measurements, without run-to-run confidence intervals.
- Baseline: BF16 drafter only. This measures quantized-versus-BF16 drafter serving; it does not measure speculative decoding versus target-only decoding.

This is an 8-arm experiment: four arms for each target family. Each arm runs on the same three 512-prompt entropy files, for 24 measured sweeps total. Keep each target family separate in plots; DFlash and DSpark use different target models.

## Arms

| Family                            | Role                                                | Drafter Hugging Face ID                                                          |
| --------------------------------- | --------------------------------------------------- | -------------------------------------------------------------------------------- |
| DFlash, target `Qwen/Qwen3-8B`    | BF16 baseline                                       | `RedHatAI/Qwen3-8B-speculator.dflash`                                            |
| DFlash                            | FP8 Dynamic, data-free                              | `inference-optimization/Qwen3-8B-DFlash-FP8-DYNAMIC`                             |
| DFlash                            | FP8 Block, data-free                                | `inference-optimization/Qwen3-8B-DFlash-FP8-BLOCK`                               |
| DFlash                            | NVFP4 GPTQ + IMatrix, PerfectBlend real calibration | `inference-optimization/Qwen3-8B-DFlash-GPTQ-IMatrix-PerfectBlend-NVFP4-W4A4`    |
| DSpark, target `Qwen/Qwen3.8-27B` | BF16 baseline                                       | `RedHatAI/Qwen3.8-27B-speculator.dspark`                                         |
| DSpark                            | FP8 Dynamic, data-free                              | `inference-optimization/Qwen3.8-27B-DSpark-FP8-DYNAMIC`                          |
| DSpark                            | FP8 Block, data-free                                | `inference-optimization/Qwen3.8-27B-DSpark-FP8-BLOCK`                            |
| DSpark                            | NVFP4 GPTQ + IMatrix, PerfectBlend real calibration | `inference-optimization/Qwen3.8-27B-DSpark-GPTQ-IMatrix-PerfectBlend-NVFP4-W4A4` |

The chosen set keeps the two data-free FP8 formats, which had BF16-like DFlash acceptance, and one strong real-calibrated low-bit format. It is a practical speed comparison, not a controlled test of calibration-data necessity.

## Why `throughput_2k`

SPEED-Bench qualitative contains 880 prompts across 11 task categories and is useful for checking speculation acceptance on varied tasks. Throughput sets target system speed at fixed input-length buckets and contain low-, mixed-, and high-entropy examples. `throughput_2k` means the nominal 2k-token input bucket; it does not mean 2k generated output tokens. NVIDIA describes the workload and split in the [SPEED-Bench dataset card](https://huggingface.co/datasets/nvidia/SPEED-Bench).

The evaluator already supports SPEED-Bench preparation, entropy selection, and local JSONL input through [PR #734](https://github.com/vllm-project/speculators/pull/734). Its prepared 2k workload is split across 18 entropy/subtopic files. Pool those files into three tier files as a data-preparation step. This retains all 1,536 prompts while reducing each arm from 18 sweeps to 3; it does not require evaluator changes.

## Prepare and pool the workload

Materialize the SPEED-Bench data on the run machine. The source data has third-party licensing conditions, so keep it in local data storage and do not commit or redistribute the materialized JSONL files.

Run these commands from the speculators repository root. They assume the vLLM clone is in a sibling directory named `vllm`.

```
set -e
export SPECULATORS_ROOT="$PWD"
export SPEEDBENCH_DIR="/data/fast/datasets/speedbench"
mkdir -p "$SPEEDBENCH_DIR"
python "$SPECULATORS_ROOT/scripts/evaluate/prepare_speedbench.py" \
  --data-dir "$SPEEDBENCH_DIR" \
  --configs throughput_2k \
  --download
```

Pool the prepared files with a fixed seed. The script checks for 512 prompts per tier and writes a manifest of source files and row counts.

```
python - "$SPEEDBENCH_DIR" <<'PY'
import json
import random
import sys
from pathlib import Path

root = Path(sys.argv[1])
output = root / "throughput_2k_by_entropy"
output.mkdir(parents=True, exist_ok=True)
tiers = ("low_entropy", "mixed_entropy", "high_entropy")
seed = 20260929
manifest = {"seed": seed, "tiers": {}}

for tier in tiers:
    files = sorted(root.glob(f"throughput_2k_{tier}__*.jsonl"))
    if not files:
        raise SystemExit(f"No prepared files found for {tier}")
    rows = []
    for path in files:
        with path.open() as source:
            rows.extend(json.loads(line) for line in source if line.strip())
    if len(rows) != 512:
        raise SystemExit(f"Expected 512 rows for {tier}, found {len(rows)}")
    random.Random(seed).shuffle(rows)
    destination = output / f"throughput_2k_{tier}.jsonl"
    with destination.open("w") as target:
        for row in rows:
            target.write(json.dumps({"turns": row["turns"]}, ensure_ascii=False) + "\n")
    manifest["tiers"][tier] = {
        "rows": len(rows),
        "files": [path.name for path in files],
        "output": destination.name,
    }

with (output / "manifest.json").open("w") as target:
    json.dump(manifest, target, indent=2)
    target.write("\n")
print(f"Wrote three files / 1536 prompts under {output}")
PY
```

Record the prepared source version and hashes with the results. Preserve the NVIDIA preparation script or its source revision along with the generated manifest; fetching the live preparation script again later may produce different source data.

## Patch a fresh vLLM clone

The run machine starts with fresh clones of `speculators` and `vllm`; it will not have the modified `/workspace/vllm` checkout. The vLLM changes needed for these quantized DFlash/DSpark drafters are included in [the patch file](./patches/vllm-quantized-dflash.patch).

The patch is based on vLLM commit `98dff2a81d747d1dba01a47f939f48c3526d4206`, reachable from `origin/releases/v0.29.0`. Its SHA-256 is `f8c8774243aecc1a8f4345fb943f405badbe76d6c8b939ed10c3c854cbddebb2`. Apply it to a clean vLLM checkout at that exact base:

```
set -e
export SPECULATORS_ROOT="$PWD"
export VLLM_ROOT="$(cd ../vllm && pwd)"
git -C "$VLLM_ROOT" fetch origin refs/heads/releases/v0.29.0
git -C "$VLLM_ROOT" checkout --detach 98dff2a81d747d1dba01a47f939f48c3526d4206
git -C "$VLLM_ROOT" status --short
git -C "$VLLM_ROOT" apply --check "$SPECULATORS_ROOT/local/patches/vllm-quantized-dflash.patch"
git -C "$VLLM_ROOT" apply "$SPECULATORS_ROOT/local/patches/vllm-quantized-dflash.patch"
```

Install vLLM from this checkout using the installation instructions that match the B200 environment. Before launching, confirm the Python environment imports vLLM from this checkout rather than another installed copy.

The patch changes:

1. `vllm/model_executor/models/qwen3_dflash.py`: handles scaled FP8 weights when forming dense context K/V weights; keeps quantized QKV projections on their quantized projection path after post-load processing so activation quantization remains active; and dequantizes FP8 DFlash embeddings using their stored weight scale because the vocab-parallel embedding path does not support FP8 weights.
2. `vllm/v1/worker/gpu/spec_decode/dflash/utils.py`: supplies the drafter quantization config when constructing the DFlash drafter, rather than inheriting the target's quantization config.
3. `tests/v1/spec_decode/test_dflash_causality.py`: adds coverage for dense fusion, drafter quantization config selection, and quantized context projections.

DSpark shares the patched Qwen3 DFlash backbone/context-KV path; this patch does not add a separate DSpark loader change. The tests are included in the patch but have not been run as part of this handoff.

On B200, do not use `local/drafter-quant/dquant/evaluate_arm.py` unchanged for NVFP4: it forces the linear backend to emulation. Launch with the commands below and check vLLM startup logs to confirm the selected NVFP4 backend.

## Launch each arm

For each row in the arms table, set `DRAFTER` to that row's ID and `ARM_OUT` to a unique output directory. Keep the target-specific serving flags identical across that family's BF16 and quantized arms. The speculators launcher forwards the requested vLLM flags and captures `vllm_command.txt`, `vllm.patch`, target/drafter checkpoint hashes, and the vLLM revision in the arm directory.

DFlash:

```
export TARGET="Qwen/Qwen3-8B"
export DRAFTER="inference-optimization/Qwen3-8B-DFlash-FP8-DYNAMIC"
export ARM_OUT="/data/fast/speedup-experiment/dflash/fp8-dynamic"
mkdir -p "$ARM_OUT"
python "$SPECULATORS_ROOT/scripts/launch_vllm.py" eval "$TARGET" \
  --spec-model "$DRAFTER" \
  --spec-method dflash \
  --spec-tokens 7 \
  --provenance-dir "$ARM_OUT" \
  -- \
  --tensor-parallel-size 1 \
  --reasoning-parser qwen3
```

DSpark:

```
export TARGET="Qwen/Qwen3.8-27B"
export DRAFTER="inference-optimization/Qwen3.8-27B-DSpark-FP8-DYNAMIC"
export ARM_OUT="/data/fast/speedup-experiment/dspark/fp8-dynamic"
mkdir -p "$ARM_OUT"
python "$SPECULATORS_ROOT/scripts/launch_vllm.py" eval "$TARGET" \
  --spec-model "$DRAFTER" \
  --spec-method dspark \
  --spec-tokens 7 \
  --provenance-dir "$ARM_OUT" \
  -- \
  --tensor-parallel-size 1 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --mm-encoder-tp-mode data
```

The BF16 baseline commands use the same target and flags; set `DRAFTER` to the BF16 baseline ID from the table. In the actual sweep, only one serving process uses the B200 at a time.

## Run the three sweeps for each arm

After the server is ready at `http://127.0.0.1:8000/v1`, run this once for each arm. Each tier is a separate evaluator invocation so that output and acceptance counters have one tier label.

```
for TIER in low_entropy mixed_entropy high_entropy; do
  python "$SPECULATORS_ROOT/scripts/evaluate/evaluate.py" sweep \
    --target http://127.0.0.1:8000/v1 \
    --dataset "$SPEEDBENCH_DIR/throughput_2k_by_entropy/throughput_2k_${TIER}.jsonl" \
    --subsets "$TIER" \
    --output-dir "$ARM_OUT/eval/$TIER" \
    --data-column-mapper 'kind=generative_column_mapper,column_mappings.text_column=turns' \
    --max-concurrency 128 \
    --max-requests 100 \
    --sweep-rate 5 \
    --gen-kwargs '{"temperature":0,"seed":0}'
done
```

Sweep mode also runs an output-length estimation pass before each measured sweep. The `--max-requests 100` limit applies to each measured sweep point, not to this estimation pass. Keep all estimator and sweep settings identical across arms, and budget for the additional pass when estimating runtime.

## Read and plot the results

Use the generated `perf_results.csv` rows for the performance curves:

- Plot median generated output tokens/sec (`output_tps_median`) against GuideLLM's recorded offered load (`target_rate`). Show raw points and connect them; keep entropy tiers separate and make separate DFlash and DSpark panels.
- Report quantized/BF16 output-token throughput ratios at matched offered rates where the sweeps overlap. If rates do not match, show the raw curves and state any interpolation method; do not present unmatched rates as a paired speedup.
- Include median request latency (`latency_median_s`), median time to first token (`ttft_median_ms`), and median inter-token latency (`itl_median_ms`) against offered load.
- Summarize the highest observed median output tokens/sec and its offered rate for each arm, labeling it as the highest point observed in this one sweep rather than a repeatable saturation limit.

Each tier's `acceptance.csv` row aggregates the vLLM draft counters across the full measured sweep, not separately for each offered rate. Report accepted/proposed tokens (`A/P`) and acceptance length (`1 + accepted tokens / draft events`) from the counters. Acceptance is a speculative-efficiency measure, not task accuracy. For these new tier files, each row directly summarizes one entropy tier; if combining files later, pool raw counters before calculating rates instead of averaging percentages.

Keep target family, entropy tier, prompt set, vLLM revision/patch, and generation settings attached to every result. A one-pass result has no run-to-run uncertainty; do not claim statistical significance. The historical acceptance data are acceptance-only and do not establish speedups. It also does not support the broad statement that all quantized drafters require real calibration: DFlash data-free FP8 Dynamic and FP8 Block were near BF16 acceptance, while the measured GPTQ real-versus-Gaussian comparison also changed damping. No completed DSpark acceptance comparison is in the local results index.

______________________________________________________________________

# Speculators / Drafter Quantization — Experiment Guide & Agent Onboarding

Welcome to the **DFlash Speculative Drafter Quantization** project! This guide is written so that any new agent or researcher can immediately get up to speed on the research questions, experimental setup, model recipes, benchmark results, filesystem layout, and reproduction commands.

______________________________________________________________________

## 1. Executive Summary & Research Mission

### The Core Problem

Speculative decoding uses a small, lightweight "drafter" model to propose $K$ tokens (here $K=7$), which are then verified in parallel by a larger "target" model (here `Qwen/Qwen3-8B`). To maximize inference throughput and minimize memory footprint, we explored **quantizing the speculative drafter to 8-bit (FP8) and 4-bit (NVFP4)**.

### The Key Research Questions

1. **Can quantized drafters match unquantized BF16 acceptance rates?**
2. **Does calibration data distribution matter for speculative drafters?** Specifically: does capturing *real* target hidden-state activations outperform synthetic *Gaussian* random activations or *data-free* dynamic quantization?

### The Headline Findings

1. **Data-free FP8 showed BF16-like acceptance in these DFlash runs**: FP8 Dynamic and FP8 Block measured about **30.8% accepted-token fraction ($A/P$)** versus **30.83%** for BF16. These are one-pass descriptive results, not a statistical equivalence test.
2. **The selected real-calibrated NVFP4 GPTQ checkpoint had strong acceptance**: the IMatrix PerfectBlend arm measured **28.61% $A/P$**, while its Gaussian counterpart measured **15.11%**. The pair also differs in Hessian damping ($d=0.1$ vs. $d=0.01$), so the gap does not isolate calibration data as the cause.
3. **Model Lineage Context**: Historical static models (`Gauss8`, `Blend8`, `Gauss4`, `Blend4`) were built on an earlier drafter snapshot, explaining their lower absolute baseline. The BF16 reference, `FP8 Block`, and all `GPTQ` models are built on the pinned newer block-16 lineage (`1a11b170eb...`).

______________________________________________________________________

## 2. Published Hugging Face Collection

All 10 quantized drafter models are published under the **`inference-optimization`** organization and curated in the collection: 👉 **[Hugging Face Quantized Drafters Collection](https://huggingface.co/collections/inference-optimization/quantized-drafters-6ab5354d0b0766f2b133d182)**

| #   | Local Model Folder (`/data/fast/models/`)                                 | HF Repository Identifier                                                                                                                                                            | Format                  | Calibration Data     | Damping |
| --- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- | -------------------- | :-----: |
| 1   | `Qwen3-8B-DFlash-FP8-BLOCK` (`Block8`)                                    | [`inference-optimization/Qwen3-8B-DFlash-FP8-BLOCK`](https://huggingface.co/inference-optimization/Qwen3-8B-DFlash-FP8-BLOCK)                                                       | FP8 Block (128x128)     | Data-Free            |   N/A   |
| 2   | `Qwen3-8B-DFlash-FP8-DYNAMIC` (`Drift8`)                                  | [`inference-optimization/Qwen3-8B-DFlash-FP8-DYNAMIC`](https://huggingface.co/inference-optimization/Qwen3-8B-DFlash-FP8-DYNAMIC)                                                   | FP8 Dynamic (per-token) | Data-Free            |   N/A   |
| 3   | `Qwen3-8B-DFlash-Gauss-FP8-W8A8` (`Gauss8`)                               | [`inference-optimization/Qwen3-8B-DFlash-Gauss-FP8-W8A8`](https://huggingface.co/inference-optimization/Qwen3-8B-DFlash-Gauss-FP8-W8A8)                                             | FP8 Static W8A8         | Gaussian (synthetic) |   N/A   |
| 4   | `Qwen3-8B-DFlash-PerfectBlend-FP8-W8A8` (`Blend8`)                        | [`inference-optimization/Qwen3-8B-DFlash-PerfectBlend-FP8-W8A8`](https://huggingface.co/inference-optimization/Qwen3-8B-DFlash-PerfectBlend-FP8-W8A8)                               | FP8 Static W8A8         | PerfectBlend (real)  |   N/A   |
| 5   | `Qwen3-8B-DFlash-Gauss-NVFP4-W4A4` (`Gauss4`)                             | [`inference-optimization/Qwen3-8B-DFlash-Gauss-NVFP4-W4A4`](https://huggingface.co/inference-optimization/Qwen3-8B-DFlash-Gauss-NVFP4-W4A4)                                         | NVFP4 W4A4 Static       | Gaussian (synthetic) |   N/A   |
| 6   | `Qwen3-8B-DFlash-PerfectBlend-NVFP4-W4A4` (`Blend4`)                      | [`inference-optimization/Qwen3-8B-DFlash-PerfectBlend-NVFP4-W4A4`](https://huggingface.co/inference-optimization/Qwen3-8B-DFlash-PerfectBlend-NVFP4-W4A4)                           | NVFP4 W4A4 Static       | PerfectBlend (real)  |   N/A   |
| 7   | `Qwen3-8B-DFlash-GPTQ-Gauss-NVFP4-W4A4` (`GPTQ-Gauss4`)                   | [`inference-optimization/Qwen3-8B-DFlash-GPTQ-Gauss-NVFP4-W4A4`](https://huggingface.co/inference-optimization/Qwen3-8B-DFlash-GPTQ-Gauss-NVFP4-W4A4)                               | NVFP4 W4A4 GPTQ         | Gaussian (synthetic) |  0.01   |
| 8   | `Qwen3-8B-DFlash-GPTQ-PerfectBlend-NVFP4-W4A4` (`GPTQ-Blend4`)            | [`inference-optimization/Qwen3-8B-DFlash-GPTQ-PerfectBlend-NVFP4-W4A4`](https://huggingface.co/inference-optimization/Qwen3-8B-DFlash-GPTQ-PerfectBlend-NVFP4-W4A4)                 | NVFP4 W4A4 GPTQ         | PerfectBlend (real)  |  0.10   |
| 9   | `Qwen3-8B-DFlash-GPTQ-IMatrix-Gauss-NVFP4-W4A4` (`IMatrix-Gauss4`)        | [`inference-optimization/Qwen3-8B-DFlash-GPTQ-IMatrix-Gauss-NVFP4-W4A4`](https://huggingface.co/inference-optimization/Qwen3-8B-DFlash-GPTQ-IMatrix-Gauss-NVFP4-W4A4)               | NVFP4 W4A4 GPTQ+IMatrix | Gaussian (synthetic) |  0.01   |
| 10  | `Qwen3-8B-DFlash-GPTQ-IMatrix-PerfectBlend-NVFP4-W4A4` (`IMatrix-Blend4`) | [`inference-optimization/Qwen3-8B-DFlash-GPTQ-IMatrix-PerfectBlend-NVFP4-W4A4`](https://huggingface.co/inference-optimization/Qwen3-8B-DFlash-GPTQ-IMatrix-PerfectBlend-NVFP4-W4A4) | NVFP4 W4A4 GPTQ+IMatrix | PerfectBlend (real)  |  0.10   |

Every repository includes complete provenance: `vllm_command.txt`, `quantization_command.txt`, `checkpoint_sha256.txt`, `drafter_checkpoint_sha256.txt`, `vllm.patch`, and evaluation logs.

______________________________________________________________________

## 3. Evaluation Suites & Verified Acceptance Results

The evaluation protocol measures **220 total cells** across two diverse suites with 7 speculative tokens:

1. **SPEED-Bench Qualitative**: 11 diverse tasks (80 requests each = 880 requests).
2. **RedHatAI (`speculator_benchmarks`)**: 9 diverse subsets (924 requests, including 164 on HumanEval and 200 on tool_call).

### Summary Comparison Table (All 11 Evaluated Arms)

| Arm / Model Role                              | SPEED-Bench Mean $A/P$ | RedHatAI Mean $A/P$ | Mean Acceptance Length $L = 1 + A/D$ | Primary Finding                                 |
| --------------------------------------------- | ---------------------: | ------------------: | -----------------------------------: | ----------------------------------------------- |
| **BF16 reference (unquantized)**              |             **30.83%** |          **30.70%** |                      **3.16 / 3.15** | Upper-bound reference baseline                  |
| **FP8 dynamic · data-free**                   |             **30.81%** |          **30.74%** |                      **3.16 / 3.15** | Near BF16 observed acceptance; zero calibration |
| **FP8 block · data-free**                     |             **30.76%** |          **30.59%** |                      **3.15 / 3.14** | Near BF16 observed acceptance; 128×128 blocks   |
| **NVFP4 GPTQ · PerfectBlend (d=0.1)**         |             **28.64%** |          **28.94%** |                      **3.00 / 3.03** | **Recovers 92% of BF16 acceptance!**            |
| **NVFP4 GPTQ+IMatrix · PerfectBlend (d=0.1)** |             **28.61%** |          **28.93%** |                      **3.00 / 3.02** | Matches ordinary GPTQ with real data            |
| **FP8 static · PerfectBlend**                 |                 16.52% |              17.20% |                          2.16 / 2.20 | +10.6% gain over Gaussian control               |
| **NVFP4 static · PerfectBlend**               |                 16.19% |              16.92% |                          2.13 / 2.18 | +10.4% gain over Gaussian control               |
| **NVFP4 GPTQ+IMatrix · Gaussian (d=0.01)**    |                 15.11% |              15.79% |                          2.06 / 2.11 | Synthetic data collapses 4-bit                  |
| **NVFP4 GPTQ · Gaussian (d=0.01)**            |                 14.94% |              15.61% |                          2.05 / 2.09 | Synthetic data collapses 4-bit                  |
| **FP8 static · Gaussian**                     |                  5.90% |               6.39% |                          1.41 / 1.45 | Historical older lineage static control         |
| **NVFP4 static · Gaussian**                   |                  5.75% |               6.27% |                          1.40 / 1.44 | Historical older lineage static control         |

______________________________________________________________________

## 4. Directory Map & Where Things Live

### Local Codebase: `/workspace/speculators/local/`

```text
/workspace/speculators/local/
├── README.md                      # [You are here] Authoritative agent onboarding & experiment guide
│
├── docs/                          # In-depth technical reports, audits, and operational manuals
│   ├── EXPERIMENT_SUMMARY.md      # Full Phase 1 synthesis and methodology
│   ├── EXPERIMENT_FINDINGS_2026-09-25.md # Lineage audit and correction notes
│   ├── 03_calibration_with_real_data.md  # Hidden-state capture guide
│   ├── 04_vllm_deploy_quantized_drafter.md # vLLM deployment guide
│   └── 05_run_instructions.md    # Step-by-step execution guide
│
├── results/                       # Master evaluation hub (all 11 arms validated)
│   ├── RESULTS.md                 # Evaluation report and analysis
│   ├── acceptance_measured.csv    # 220-row database of verified counters
│   ├── manifest.json              # Provenance SHA-256 hashes for all inputs/outputs
│   ├── plot_acceptance.py         # Master validation & plotting script
│   ├── acceptance_redhatai.svg / .png    # 11-arm RedHatAI comparison chart
│   ├── acceptance_speedbench.svg / .png  # 11-arm SPEED-Bench comparison chart
│   └── historical/                # Preserved Phase 1 6-arm charts & slide decks
│
├── briefs/                        # Task briefs and specifications
│   ├── upload-models.md           # Model upload specification
│   ├── add-evals.md               # 11-arm extension brief
│   └── brief-speedbench.md        # SPEED-Bench evaluation protocol
│
├── analysis/                      # Code audits, reviews, and design consensus
├── patches/                       # Patch for reproducing the quantized DFlash/DSpark vLLM support
├── learnings/                     # Conceptual notes (e.g. response regeneration vs calibration)
└── drafter-quant/                 # Production quantization & evaluation software package
    ├── dquant/                    # Core Python package (quantize, calibrate, evaluate)
    ├── scripts/                   # Shell scripts for multi-GPU pipelines
    ├── config/                    # Model environment configurations
    └── tests/                     # Unit tests
```

### High-Speed NVMe Storage: `/data/fast/`

```text
/data/fast/
├── models/                        # Authoritative self-contained models (weights, configs, cards, provenance)
│   ├── upload_results.json        # Unified HF publication manifest and commit hashes
│   ├── Qwen3-8B-DFlash-FP8-BLOCK/            # Data-free FP8 Block (Block8)
│   ├── Qwen3-8B-DFlash-FP8-DYNAMIC/          # Data-free FP8 Dynamic (Drift8)
│   ├── Qwen3-8B-DFlash-Gauss-FP8-W8A8/       # Static FP8 Gauss (Gauss8)
│   ├── Qwen3-8B-DFlash-PerfectBlend-FP8-W8A8/# Static FP8 Real (Blend8)
│   ├── Qwen3-8B-DFlash-Gauss-NVFP4-W4A4/     # Static NVFP4 Gauss (Gauss4)
│   ├── Qwen3-8B-DFlash-PerfectBlend-NVFP4-W4A4/# Static NVFP4 Real (Blend4)
│   ├── Qwen3-8B-DFlash-GPTQ-Gauss-NVFP4-W4A4/# NVFP4 GPTQ Gauss (GPTQ-Gauss4)
│   ├── Qwen3-8B-DFlash-GPTQ-PerfectBlend-NVFP4-W4A4/      # NVFP4 GPTQ Real (GPTQ-Blend4)
│   ├── Qwen3-8B-DFlash-GPTQ-IMatrix-Gauss-NVFP4-W4A4/     # NVFP4 GPTQ+IMatrix Gauss (IMatrix-Gauss4)
│   ├── Qwen3-8B-DFlash-GPTQ-IMatrix-PerfectBlend-NVFP4-W4A4/# NVFP4 GPTQ+IMatrix Real (IMatrix-Blend4)
│   └── [Convenience symlinks: Block8, Drift8, Gauss8, Blend8, Gauss4, Blend4, GPTQ-Gauss4, ...]
│
├── evaluations/                   # Full raw evaluation runs with per-request artifacts
│   ├── redhatai/                  # 11 evaluated arms on RedHatAI (9 subsets each)
│   ├── speedbench/                # 11 evaluated arms on SPEED-Bench (11 categories each)
│   └── throughput/                # Throughput benchmark logs
│
├── datasets/                      # Retained evaluation benchmark sets
│   ├── speedbench/                # SPEED-Bench qualitative prompts (880 items)
│   └── speedbench-frozen/         # Frozen evaluation prompt set
│
└── hf_cache/                      # Cached Hugging Face base models (Qwen3-8B)
```

______________________________________________________________________

## 5. Common Operational Commands

### Regenerate All Acceptance Plots and Manifests

```bash
python3 /workspace/speculators/local/results/plot_acceptance.py
```

This inspects the raw CSV and JSON records, validates all 220 cells, recalculates acceptance ratios, updates `acceptance_measured.csv`, regenerates `acceptance_speedbench.{svg,png}` and `acceptance_redhatai.{svg,png}`, and records SHA256 checksums in `manifest.json`.

### Serving a Quantized Drafter in vLLM

To launch speculative decoding with a target model and one of the quantized drafters:

```bash
# Example: Serving Qwen3-8B with the FP8 Block drafter
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B \
  --speculative-model /data/fast/models/Qwen3-8B-DFlash-FP8-BLOCK \
  --num-speculative-tokens 7 \
  --speculative-draft-tensor-parallel-size 1 \
  --speculative-method dflash \
  --port 8000
```

### Inspecting Model Provenance

Every model directory in `/data/fast/models/<model>/provenance/` contains:

```bash
ls -la /data/fast/models/Qwen3-8B-DFlash-GPTQ-PerfectBlend-NVFP4-W4A4/provenance/
# Contents:
# checkpoint_sha256.txt        - Target model SHA256
# drafter_checkpoint_sha256.txt - Unquantized drafter source SHA256
# quantization_command.txt    - Exact CLI command and environment variables
# vllm_command.txt            - vLLM serving invocation
# vllm.patch                  - Git patch for vLLM speculative engine
```

## Appendix: full vLLM patch

The next block is the exact patch applied by the fresh-clone instructions above. It is also stored separately at `local/patches/vllm-quantized-dflash.patch` for direct use with `git apply`.

```diff
diff --git a/tests/v1/spec_decode/test_dflash_causality.py b/tests/v1/spec_decode/test_dflash_causality.py
index 0f1de86cf9..acf8e5a588 100644
--- a/tests/v1/spec_decode/test_dflash_causality.py
+++ b/tests/v1/spec_decode/test_dflash_causality.py
@@ -7,11 +7,16 @@ non-causal-capable backend, so its branch table (explicit override, SWA-derived
 per-layer causality, and the no-``layer_types`` fallback) is worth pinning.
 """
 
+from contextlib import nullcontext
 from types import SimpleNamespace
+from unittest.mock import patch
 
 import pytest
+import torch
+from torch import nn
 
 from vllm.model_executor.models.qwen3_dflash import (
+    DFlashQwen3Model,
     _dflash_layer_causal,
     _get_dflash_fc_input_size,
     dflash_has_any_non_causal,
@@ -113,3 +118,126 @@ def test_eagle_aux_layers_preserves_legacy_layer_ids(config_name):
     assert get_eagle3_aux_layers_from_config(vllm_config.speculative_config) == tuple(
         layer_ids
     )
+
+
+def test_dflash_context_kv_buffers_preserve_dense_fusion():
+    model = DFlashQwen3Model.__new__(DFlashQwen3Model)
+    nn.Module.__init__(model)
+    model.hidden_norm = SimpleNamespace(weight=torch.ones(4))
+
+    full_weight = torch.arange(16, dtype=torch.float32).reshape(4, 4)
+
+    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
+
+    qkv_proj = SimpleNamespace(
+        quant_method=UnquantizedLinearMethod(),
+        weight=full_weight,
+        input_size_per_partition=4,
+        params_dtype=torch.float32,
+    )
+    attention = SimpleNamespace(
+        q_size=2,
+        qkv_proj=qkv_proj,
+        k_norm=SimpleNamespace(weight=torch.ones(2)),
+    )
+
+    model._build_context_kv_buffers([attention], has_bias=False)
+
+    torch.testing.assert_close(model._fused_kv_weight, full_weight[2:])
+
+
+def test_dflash_loader_uses_draft_quant_config():
+    from vllm.v1.worker.gpu.spec_decode.dflash import utils as dflash_utils
+
+    target_quant_config = object()
+    draft_quant_config = object()
+    draft_model_config = SimpleNamespace(hf_config=SimpleNamespace())
+    draft_model = SimpleNamespace(model=SimpleNamespace())
+    target_model = SimpleNamespace(model=SimpleNamespace())
+    vllm_config = SimpleNamespace(
+        speculative_config=SimpleNamespace(
+            draft_model_config=draft_model_config,
+            attention_backend=None,
+            kv_cache_dtype=None,
+        ),
+        attention_config=SimpleNamespace(backend=None),
+        cache_config=SimpleNamespace(),
+        quant_config=target_quant_config,
+    )
+
+    def fake_replace(config, **changes):
+        values = vars(config).copy()
+        values.update(changes)
+        return SimpleNamespace(**values)
+
+    with (
+        patch.object(dflash_utils, "replace", side_effect=fake_replace),
+        patch.object(
+            dflash_utils,
+            "get_pp_group",
+            return_value=SimpleNamespace(world_size=1),
+        ),
+        patch.object(dflash_utils, "get_model", return_value=draft_model) as get_model,
+        patch.object(dflash_utils, "get_target_lm_head", return_value=None),
+        patch(
+            "vllm.compilation.backends.set_model_tag",
+            return_value=nullcontext(),
+        ),
+        patch(
+            "vllm.model_executor.models.qwen3_dflash.dflash_has_any_non_causal",
+            return_value=False,
+        ),
+        patch(
+            "vllm.model_executor.models.utils.get_draft_quant_config",
+            return_value=draft_quant_config,
+        ),
+    ):
+        dflash_utils.load_dflash_model(target_model, vllm_config)
+
+    loaded_config = get_model.call_args.kwargs["vllm_config"]
+    assert loaded_config.quant_config is draft_quant_config
+
+
+def test_context_kv_defers_quantized_projection_and_preserves_activation_scales():
+    """Load-time fusion must defer quantized kernels and preserve A4."""
+    model = DFlashQwen3Model.__new__(DFlashQwen3Model)
+    nn.Module.__init__(model)
+    model.hidden_norm = SimpleNamespace(weight=torch.ones(4))
+    model._rms_norm_eps = 1e-6
+
+    class QuantizedProjection(nn.Module):
+        def __init__(self):
+            super().__init__()
+            self.weight_packed = torch.arange(16, dtype=torch.float32).reshape(4, 4)
+            self.quant_method = SimpleNamespace(apply=self.apply)
+            self.input_size_per_partition = 4
+            self.params_dtype = torch.float32
+            self.ready = False
+            self.scale = 1.0
+
+        def apply(self, layer, inputs, bias=None):
+            assert self.ready, "quantization post-load processing has not run"
+            rounded = torch.round(inputs / self.scale) * self.scale
+            return torch.nn.functional.linear(rounded, self.weight_packed, bias)
+
+        def forward(self, inputs):
+            return self.apply(self, inputs), None
+
+    projection = QuantizedProjection()
+    attention = SimpleNamespace(
+        q_size=2, qkv_proj=projection, k_norm=SimpleNamespace(weight=torch.ones(1))
+    )
+    model._build_context_kv_buffers([attention], has_bias=False)
+    projection.ready = True
+    inputs = torch.tensor([[0.2, 0.6, 1.2, 1.7], [2.3, 0.4, 1.6, 0.1]])
+    with patch(
+        "vllm.model_executor.models.qwen3_dflash.ops.rms_norm",
+        side_effect=lambda out, x, *_: out.copy_(x),
+    ):
+        k, v = model._project_context_kv(inputs, 2, 1, 1, 1)
+        expected = projection(inputs)[0][:, 2:]
+        torch.testing.assert_close(k.flatten(), expected[:, 0])
+        torch.testing.assert_close(v.flatten(), expected[:, 1])
+        projection.scale = 0.5
+        changed_k, _ = model._project_context_kv(inputs, 2, 1, 1, 1)
+    assert not torch.equal(k, changed_k)
diff --git a/vllm/model_executor/models/qwen3_dflash.py b/vllm/model_executor/models/qwen3_dflash.py
index 0a481b3823..a88caf8acc 100644
--- a/vllm/model_executor/models/qwen3_dflash.py
+++ b/vllm/model_executor/models/qwen3_dflash.py
@@ -23,9 +23,14 @@ from vllm.model_executor.layers.linear import (
     QKVParallelLinear,
     ReplicatedLinear,
     RowParallelLinear,
+    UnquantizedLinearMethod,
 )
 from vllm.model_executor.layers.logits_processor import LogitsProcessor
 from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
+from vllm.model_executor.layers.quantization.utils.quant_utils import (
+    get_and_maybe_dequant_weights,
+    scaled_dequantize,
+)
 from vllm.model_executor.layers.rotary_embedding import get_rope
 from vllm.model_executor.layers.vocab_parallel_embedding import (
     ParallelLMHead,
@@ -86,6 +91,27 @@ def _get_dflash_fc_input_size(vllm_config: VllmConfig) -> int:
     return target_hidden_size * num_features_to_use
 
 
+def _get_dflash_qkv_weight(
+    qkv_proj: nn.Module, out_dtype: torch.dtype | None = None
+) -> torch.Tensor:
+    """Return a DFlash QKV weight in the dense ``[out, in]`` layout."""
+    out_dtype = out_dtype or qkv_proj.params_dtype
+    weight = getattr(qkv_proj, "weight", None)
+    weight_scale = getattr(qkv_proj, "weight_scale", None)
+    fp8_fnuz = getattr(torch, "float8_e4m3fnuz", torch.float8_e4m3fn)
+    if (
+        weight is not None
+        and weight_scale is not None
+        and weight.dtype in (torch.float8_e4m3fn, fp8_fnuz)
+        and weight_scale.numel() in (weight.shape[0], weight.shape[1])
+    ):
+        if weight_scale.numel() == weight.shape[1]:
+            weight = weight.t()
+        return scaled_dequantize(weight, weight_scale, out_dtype=out_dtype)
+
+    return get_and_maybe_dequant_weights(qkv_proj, out_dtype=out_dtype)
+
+
 def _resolve_layer_attention(
     config: Qwen3Config, layer_idx: int
 ) -> tuple[int | None, bool]:
@@ -468,14 +494,31 @@ class DFlashQwen3Model(nn.Module):
     ) -> None:
         self._hidden_norm_weight = self.hidden_norm.weight.data
 
-        # KV projection weights: [num_layers * 2 * kv_size, hidden_size]
-        kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]
-        self._fused_kv_weight = torch.cat(kv_weights, dim=0)
-        if has_bias:
-            kv_biases = [a.qkv_proj.bias[a.q_size :] for a in layers_attn]
-            self._fused_kv_bias: torch.Tensor | None = torch.cat(kv_biases, dim=0)
-        else:
+        # Quantized projections must run after post-load processing and retain
+        # activation quantization; dense materialization bypasses that treatment.
+        self._context_kv_projections = None
+        if any(
+            not isinstance(a.qkv_proj.quant_method, UnquantizedLinearMethod)
+            for a in layers_attn
+        ):
+            self._context_kv_projections = tuple(
+                (a.qkv_proj, a.q_size) for a in layers_attn
+            )
+            self._fused_kv_weight = None
             self._fused_kv_bias = None
+        else:
+            kv_weights = [
+                _get_dflash_qkv_weight(a.qkv_proj, out_dtype=a.qkv_proj.params_dtype)[
+                    a.q_size :
+                ]
+                for a in layers_attn
+            ]
+            self._fused_kv_weight = torch.cat(kv_weights, dim=0)
+            if has_bias:
+                kv_biases = [a.qkv_proj.bias[a.q_size :] for a in layers_attn]
+                self._fused_kv_bias = torch.cat(kv_biases, dim=0)
+            else:
+                self._fused_kv_bias = None
 
         # K-norm weights stacked into one contiguous [num_layers, head_dim]
         # tensor so the per-layer K-norm runs as a single grouped kernel.
@@ -542,9 +585,18 @@ class DFlashQwen3Model(nn.Module):
             self._hidden_norm_weight,
             self._rms_norm_eps,
         )
-        all_kv_flat = F.linear(
-            normed_context_states, self._fused_kv_weight, self._fused_kv_bias
-        )
+        if self._context_kv_projections is None:
+            all_kv_flat = F.linear(
+                normed_context_states, self._fused_kv_weight, self._fused_kv_bias
+            )
+        else:
+            all_kv_flat = torch.cat(
+                [
+                    projection(normed_context_states)[0][..., q_size:]
+                    for projection, q_size in self._context_kv_projections
+                ],
+                dim=-1,
+            )
         # Single contiguous copy that separates K/V and transposes to
         # layer-major layout.  Result: [2, L, num_ctx, nkv, hd] contiguous.
         # Indexing dim-0 gives contiguous [L, num_ctx, nkv, hd] for K and V.
@@ -817,6 +869,18 @@ class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):
             model_weights[name] = loaded_weight
             process_eagle_weight(self, name)
 
+        # Some compressed-tensors DFlash checkpoints quantize the embedding
+        # even though vLLM's embedding implementation does not support FP8.
+        embed_weight_name = "model.embed_tokens.weight"
+        embed_scale_name = "model.embed_tokens.weight_scale"
+        if embed_scale_name in model_weights:
+            embed_weight = model_weights[embed_weight_name]
+            embed_scale = model_weights.pop(embed_scale_name)
+            embed_dtype = self.model.embed_tokens.params_dtype
+            model_weights[embed_weight_name] = embed_weight.to(
+                embed_dtype
+            ) * embed_scale.to(embed_dtype)
+
         # Route the separately-trained mask embedding (if shipped) through the
         # standard weight loader alongside the rest of the draft weights.
         mask_embedding = self._read_mask_embedding()
diff --git a/vllm/v1/worker/gpu/spec_decode/dflash/utils.py b/vllm/v1/worker/gpu/spec_decode/dflash/utils.py
index 562966e07e..728d08a35e 100644
--- a/vllm/v1/worker/gpu/spec_decode/dflash/utils.py
+++ b/vllm/v1/worker/gpu/spec_decode/dflash/utils.py
@@ -16,6 +16,7 @@ def load_dflash_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Mo
     from vllm.model_executor.models.qwen3_dflash import (
         dflash_has_any_non_causal,
     )
+    from vllm.model_executor.models.utils import get_draft_quant_config
 
     speculative_config = vllm_config.speculative_config
     assert speculative_config is not None
@@ -38,6 +39,9 @@ def load_dflash_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Mo
             else vllm_config.cache_config
         ),
     )
+    # VllmConfig post-init restores the target's quant config, so use the
+    # drafter's config while constructing the DFlash model.
+    draft_vllm_config.quant_config = get_draft_quant_config(vllm_config)
     with set_model_tag("dflash_head"):
         dflash_model = get_model(
             vllm_config=draft_vllm_config, model_config=draft_model_config
```
