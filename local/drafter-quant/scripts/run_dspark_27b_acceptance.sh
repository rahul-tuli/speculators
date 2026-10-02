#!/usr/bin/env bash
# Run the throughput-load DSpark acceptance matrix on all frozen
# RedHatAI/speculator_benchmarks and SPEED-Bench qualitative subsets.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA_ROOT=${DATA_ROOT:-/data/fast/drafter-quant/data/dspark-qwen3.8-27b-eval}
TARGET=${TARGET:-/root/.cache/huggingface/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}
SOURCE_DSPARK=${SOURCE_DSPARK:-/root/.cache/huggingface/models--RedHatAI--Qwen3.8-27B-speculator.dspark/snapshots/7f33c272e5da240978e0d55767abab8193d74b95}
CALIBRATION_DATA=${CALIBRATION_DATA:-/data/fast/drafter-quant/data/Qwen--Qwen3.8-27B-fp8-layers-4-12-20-28-36-44-52-60-h5120-n2048-seq2048-seed0_prepared}
MODEL_ROOT=${MODEL_ROOT:-/data/fast/models}
RESULTS=${RESULTS:-/data/fast/drafter-quant/results/dspark-qwen3.8-27b-acceptance}
PORT=${PORT:-8111}
CONCURRENCY=${CONCURRENCY:-128}

[[ -f "$DATA_ROOT/manifest.json" ]] || { echo "Missing frozen manifest: $DATA_ROOT/manifest.json" >&2; exit 2; }
[[ -d "$TARGET" && -d "$SOURCE_DSPARK" ]] || { echo "Missing pinned target or BF16 drafter snapshot" >&2; exit 2; }

export PYTHONPATH="$ROOT/local/drafter-quant${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export TP=2 EXPECTED_CALIB_SAMPLES=1892 SPEC_TOKENS=8 CALIBRATION_SEED=0
export CALIBRATION_DATA
export VLLM_USE_V2_MODEL_RUNNER=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

mapfile -t DATASETS < <({
  find "$DATA_ROOT/redhatai" -maxdepth 1 -type f -name '*.jsonl' -print
  find "$DATA_ROOT/speedbench" -maxdepth 1 -type f -name 'qualitative_*.jsonl' -print
} | sort)
if [[ ${#DATASETS[@]} -ne 20 ]]; then
  echo "Expected 20 frozen dataset subsets, found ${#DATASETS[@]}" >&2
  exit 2
fi

ARMS=(
  "fp8-block|Qwen3.8-27B-DSpark-FP8-BLOCK"
  "fp8-dynamic|Qwen3.8-27B-DSpark-FP8-DYNAMIC"
  "fp8-gaussian|Qwen3.8-27B-DSpark-Gauss-FP8-W8A8"
  "fp8-real|Qwen3.8-27B-DSpark-PerfectBlend-FP8-W8A8"
  "nvfp4-gaussian|Qwen3.8-27B-DSpark-Gauss-NVFP4-W4A4"
  "nvfp4-real|Qwen3.8-27B-DSpark-PerfectBlend-NVFP4-W4A4"
  "nvfp4-gptq-gaussian|Qwen3.8-27B-DSpark-GPTQ-Gauss-NVFP4-W4A4"
  "nvfp4-gptq-real|Qwen3.8-27B-DSpark-GPTQ-PerfectBlend-NVFP4-W4A4"
  "nvfp4-gptq-imatrix-gaussian|Qwen3.8-27B-DSpark-GPTQ-IMatrix-Gauss-NVFP4-W4A4"
  "nvfp4-gptq-imatrix-real|Qwen3.8-27B-DSpark-GPTQ-IMatrix-PerfectBlend-NVFP4-W4A4"
  "bf16-drafter|$SOURCE_DSPARK"
)

mkdir -p "$RESULTS"
for index in "${!ARMS[@]}"; do
  entry=${ARMS[$index]}
  arm=${entry%%|*}
  checkpoint=${entry#*|}
  [[ "$checkpoint" == /* ]] || checkpoint="$MODEL_ROOT/$checkpoint"
  [[ -d "$checkpoint" ]] || { echo "Missing checkpoint for $arm: $checkpoint" >&2; exit 2; }
  echo "[$((index + 1))/11] $arm: $checkpoint"
  EXPECTED_CALIB_SAMPLES=$EXPECTED_CALIB_SAMPLES SPEC_TOKENS=$SPEC_TOKENS \
    TP=$TP CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES VLLM_USE_V2_MODEL_RUNNER=1 \
    "$ROOT/.venv/bin/python" -m dquant.evaluate_arm \
      --target "$TARGET" \
      --drafter "$checkpoint" \
      --arm "$arm" \
      --method dspark \
      --datasets "${DATASETS[@]}" \
      --manifest "$DATA_ROOT/manifest.json" \
      --output "$RESULTS" \
      --repeat 0 \
      --order "$index" \
      --phase acceptance \
      --concurrencies "$CONCURRENCY" \
      --warmup-requests 2 \
      --port "$PORT"
done
echo "All 11 DSpark acceptance arms completed. Results: $RESULTS"
"$ROOT/.venv/bin/python" -m dquant.plot_dspark_27b \
  --results "$RESULTS" \
  --manifest "$DATA_ROOT/manifest.json" \
  --output "$ROOT/local/results/dspark_qwen3_8_27b"
