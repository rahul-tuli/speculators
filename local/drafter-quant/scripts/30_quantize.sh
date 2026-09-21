#!/usr/bin/env bash
# ============================================================================
# 30_quantize.sh — quantize ONE arm: <scheme> <calib> <label> [extra flags].
#
# Wraps `python -m dquant.quantize`. The config file supplies DRAFTER, TARGET
# (used as --processor) and METHOD.
#
# Idempotent: skips if the output dir already contains .safetensors.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=../lib/common.sh
source "$PROJECT_ROOT/lib/common.sh"

usage() {
  cat <<EOF
Usage: bash scripts/30_quantize.sh [--config FILE] <scheme> <calib> <label> [extra dquant.quantize flags]

  scheme   fp8_w8a8 | nvfp4_w4a4 | fp8_dynamic
  calib    'random' (degenerate baseline / smoke only) OR path to a
           speculators-prepared dataset dir from scripts/20_capture_hidden_states.sh
  label    arm label used in the output dir name (e.g. random | perfectblend)

Output: <OUT_ROOT>/<model_tag>-<METHOD>-<scheme>-<label>/
  (+ quant_run_manifest.json with scheme, ignore list, seed, pinned commits)

Options:
  --config FILE   config env file (default: config/dflash-qwen3-8b.env);
                  provides DRAFTER, TARGET (-> --processor), METHOD.

Env overrides:
  NUM_SAMPLES           calibration samples  (default: 2048)
  SEQ_LEN               calibration seq len  (default: 2048)
  SEED                  calibration seed     (default: 0)
  OUT_ROOT              output root          (default: $WORK_DIR/out)
  MODEL_TAG             dir-name tag         (default: derived from DRAFTER id)
  CUDA_VISIBLE_DEVICES  GPU selection        (default: 0)

Examples:
  bash scripts/30_quantize.sh fp8_w8a8 random random
  bash scripts/30_quantize.sh nvfp4_w4a4 \$WORK_DIR/data/perfectblend_2048_prepared perfectblend \\
      --nvfp4-weight-observer nvfp4_expanded_mse
  bash scripts/30_quantize.sh fp8_dynamic none fp8dyn   # data-free control
EOF
}

CONFIG="$PROJECT_ROOT/config/dflash-qwen3-8b.env"
while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) break ;;
  esac
done
[ $# -ge 3 ] || { usage >&2; exit 2; }
SCHEME=$1; CALIB=$2; LABEL=$3; shift 3

load_config "$CONFIG"

NUM_SAMPLES=${NUM_SAMPLES:-2048}
SEQ_LEN=${SEQ_LEN:-2048}
SEED=${SEED:-0}
OUT_ROOT=${OUT_ROOT:-$WORK_DIR/out}
MODEL_TAG=${MODEL_TAG:-$(model_tag)}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
OUT="$OUT_ROOT/${MODEL_TAG}-${METHOD}-${SCHEME}-${LABEL}"

if [ -d "$OUT" ] && ls "$OUT"/*.safetensors >/dev/null 2>&1; then
  log "exists, skipping: $OUT"
  exit 0
fi

log "quantize: scheme=$SCHEME calib=$CALIB label=$LABEL -> $OUT"
(cd "$PROJECT_ROOT" && \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python -m dquant.quantize \
    --scheme "$SCHEME" \
    --model "$DRAFTER" --output-dir "$OUT" \
    --processor "$TARGET" --calib-data "$CALIB" \
    --num-calibration-samples "$NUM_SAMPLES" --seq-len "$SEQ_LEN" \
    --seed "$SEED" "$@")
log "done: $OUT"
