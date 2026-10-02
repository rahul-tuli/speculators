#!/usr/bin/env bash
# Produce the five added Qwen3-8B/DFlash arms from the BF16 baseline lineage.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SPECULATORS_ROOT="$(cd "$PROJECT_ROOT/../.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-$SPECULATORS_ROOT/.venv/bin/python}

SOURCE_SNAPSHOT=${SOURCE_SNAPSHOT:-/data/fast/hf_cache/hub/models--RedHatAI--Qwen3-8B-speculator.dflash/snapshots/1a11b170eb65c8a62c80ecd01dfe22a5907298e6}
SOURCE_SHA256=${SOURCE_SHA256:-afda76166ccd6a2d3ef55b214889bd045be3b8fc41b41be330d06d25059f88e8}
TARGET_TOKENIZER=${TARGET_TOKENIZER:-/data/fast/hf_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}
CALIB_DATA=${CALIB_DATA:-/data/fast/drafter-quant/data/perfectblend_2048_prepared}
OUT_ROOT=${OUT_ROOT:-/data/fast/drafter-quant/out/add-evals-20260928}
GPTQ_RANDOM_DAMPENING_FRAC=${GPTQ_RANDOM_DAMPENING_FRAC:-0.01}
GPTQ_REAL_DAMPENING_FRAC=${GPTQ_REAL_DAMPENING_FRAC:-0.1}
GPTQ_OUTPUT_SUBDIR=${GPTQ_OUTPUT_SUBDIR:-damping${GPTQ_REAL_DAMPENING_FRAC/./p}}
NUM_SAMPLES=${NUM_SAMPLES:-2027}
SEQ_LEN=${SEQ_LEN:-2048}
SEED=${SEED:-1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
HF_HOME=${HF_HOME:-/data/fast/hf_cache}

if [ ! -f "$SOURCE_SNAPSHOT/model.safetensors" ]; then
  echo "missing BF16 source checkpoint: $SOURCE_SNAPSHOT" >&2
  exit 1
fi
if [ ! -f "$TARGET_TOKENIZER/tokenizer.json" ]; then
  echo "missing pinned target tokenizer: $TARGET_TOKENIZER" >&2
  exit 1
fi
if [ ! -x "$PYTHON_BIN" ]; then
  echo "missing experiment Python environment: $PYTHON_BIN" >&2
  exit 1
fi
ACTUAL_SHA256=$(sha256sum "$SOURCE_SNAPSHOT/model.safetensors" | cut -d' ' -f1)
if [ "$ACTUAL_SHA256" != "$SOURCE_SHA256" ]; then
  echo "BF16 source SHA-256 mismatch: $ACTUAL_SHA256" >&2
  exit 1
fi
mkdir -p "$OUT_ROOT/logs"

run_arm() {
  local scheme=$1 calib=$2 label=$3
  local output log
  local extra_args=()
  if [[ "$scheme" == nvfp4_gptq* ]]; then
    if [[ "$calib" == random ]]; then
      output="$OUT_ROOT/$scheme-$label"
      log="$OUT_ROOT/logs/$scheme-$label.log"
      extra_args=(--gptq-dampening-frac "$GPTQ_RANDOM_DAMPENING_FRAC")
    else
      output="$OUT_ROOT/$GPTQ_OUTPUT_SUBDIR/$scheme-$label"
      log="$OUT_ROOT/logs/$GPTQ_OUTPUT_SUBDIR/$scheme-$label.log"
      extra_args=(--gptq-dampening-frac "$GPTQ_REAL_DAMPENING_FRAC")
    fi
  else
    output="$OUT_ROOT/$scheme-$label"
    log="$OUT_ROOT/logs/$scheme-$label.log"
  fi
  if [ -f "$output/checkpoint_complete.json" ] &&
     [ -f "$output/quant_run_manifest.json" ] &&
     ls "$output"/*.safetensors >/dev/null 2>&1; then
    echo "[skip] complete checkpoint: $output"
    return
  fi
  if [ -d "$output" ] && [ -n "$(ls -A "$output")" ]; then
    echo "partial checkpoint needs inspection: $output" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$log")"
  echo "[start] $scheme $calib -> $output"
  (cd "$PROJECT_ROOT" &&
    env HF_HOME="$HF_HOME" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
        "$PYTHON_BIN" -m dquant.quantize \
          --scheme "$scheme" --model "$SOURCE_SNAPSHOT" \
          --output-dir "$output" --processor "$TARGET_TOKENIZER" \
          --calib-data "$calib" --num-calibration-samples "$NUM_SAMPLES" \
          --seq-len "$SEQ_LEN" --seed "$SEED" --device cuda:0 \
          "${extra_args[@]}" \
          >"$log" 2>&1)
  echo "[done] $output"
}

requested=${1:-all}
case "$requested" in
  all)
    run_arm fp8_block none datafree
    run_arm nvfp4_gptq random random
    run_arm nvfp4_gptq_imatrix random random
    run_arm nvfp4_gptq "$CALIB_DATA" real
    run_arm nvfp4_gptq_imatrix "$CALIB_DATA" real
    ;;
  fp8_block-datafree) run_arm fp8_block none datafree ;;
  nvfp4_gptq-random) run_arm nvfp4_gptq random random ;;
  nvfp4_gptq_imatrix-random) run_arm nvfp4_gptq_imatrix random random ;;
  nvfp4_gptq-real) run_arm nvfp4_gptq "$CALIB_DATA" real ;;
  nvfp4_gptq_imatrix-real) run_arm nvfp4_gptq_imatrix "$CALIB_DATA" real ;;
  *) echo "unknown arm: $requested" >&2; exit 2 ;;
esac
