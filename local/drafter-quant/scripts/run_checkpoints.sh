#!/usr/bin/env bash
# ============================================================================
# run_checkpoints.sh — PHASE A: generate all quantized checkpoints.
#
#   [1/3] 10_prepare_dataset.sh        perfectblend subset           [CPU]
#   [2/3] 20_capture_hidden_states.sh  tokenize + hidden states      [1 GPU, ~hrs]
#   [3/3] 30_quantize.sh               RANDOM + PERFECTBLEND arms    [1 GPU, ~min each]
#
# Idempotent: existing outputs are skipped. Then run Phase B:
#   bash scripts/run_evals.sh <same-config-file>
#
#   bash scripts/run_checkpoints.sh config/dflash-qwen3-8b.env
#   bash scripts/run_checkpoints.sh config/dspark-qwen3-4b.env
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=../lib/common.sh
source "$PROJECT_ROOT/lib/common.sh"

usage() {
  cat <<EOF
Usage: bash scripts/run_checkpoints.sh [--help] <config-file>

Phase A — generate checkpoints:
  [1/3] 10_prepare_dataset.sh        perfectblend subset           [CPU]
  [2/3] 20_capture_hidden_states.sh  tokenize + hidden states      [1 GPU, ~hrs]
  [3/3] 30_quantize.sh               RANDOM + PERFECTBLEND arms    [1 GPU, ~min each]

  config-file   one of config/*.env, e.g.
                  config/dflash-qwen3-8b.env   (Qwen3-8B + DFlash drafter)
                  config/dspark-qwen3-4b.env   (Qwen3-4B + DSpark drafter)

Env overrides:
  NUM_SAMPLES   perfectblend size    (default: 2048)
  SEQ_LEN       calibration seq len  (default: 2048)
  SEED          sampling/calib seed  (default: 0)
  SCHEMES       quant schemes        (default: "fp8_w8a8 nvfp4_w4a4";
                fp8_dynamic is data-free and excluded by design — run it
                separately via scripts/30_quantize.sh as a control arm)
  WORK_DIR      all outputs root     (default: /data/fast/drafter-quant)
  DRAFTERS_QUANT_REPOS  pinned repos (default: /workspace)
  PB_DIR, PREPARED, MODEL_TAG, plus every override of the step scripts
  (see each script's --help).

Outputs (under \$WORK_DIR):
  data/perfectblend_<N>/, data/perfectblend_<N>_prepared/,
  out/<tag>-<method>-<scheme>-<calib>/ + quant_run_manifest.json
EOF
}

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
  "") usage >&2; exit 2 ;;
esac
CONFIG=$(realpath "$1")
load_config "$CONFIG"   # validates the file and loads TARGET/DRAFTER/METHOD/...

NUM_SAMPLES=${NUM_SAMPLES:-2048}
SEQ_LEN=${SEQ_LEN:-2048}
SEED=${SEED:-0}
SCHEMES=${SCHEMES:-"fp8_w8a8 nvfp4_w4a4"}   # fp8_dynamic is data-free: excluded
PB_DIR=${PB_DIR:-$WORK_DIR/data/perfectblend_${NUM_SAMPLES}}
PREPARED=${PREPARED:-$(prepared_path "$NUM_SAMPLES" "$SEQ_LEN" "$SEED")}
MODEL_TAG=${MODEL_TAG:-$(model_tag)}

# Hand shared paths/knobs to the step scripts via the environment.
export NUM_SAMPLES SEQ_LEN SEED MODEL_TAG
export OUT_DIR="$PB_DIR" MAX_SAMPLES="$NUM_SAMPLES"
export DATA="$PB_DIR/perfectblend_${NUM_SAMPLES}.jsonl" PREPARED

log "############ [1/3] perfectblend subset (${NUM_SAMPLES})"
bash "$SCRIPT_DIR/10_prepare_dataset.sh" --config "$CONFIG"

log "############ [2/3] tokenize + offline hidden states"
bash "$SCRIPT_DIR/20_capture_hidden_states.sh" --config "$CONFIG"

log "############ [3/3] quantize: RANDOM + PERFECTBLEND calibration arms"
for s in $SCHEMES; do
  for calib_src in random "$PREPARED"; do
    calib_label=perfectblend; [ "$calib_src" = "random" ] && calib_label=random
    if [ "$s" = "nvfp4_w4a4" ]; then
      bash "$SCRIPT_DIR/30_quantize.sh" --config "$CONFIG" "$s" "$calib_src" "$calib_label" \
        --nvfp4-weight-observer nvfp4_expanded_mse
    else
      bash "$SCRIPT_DIR/30_quantize.sh" --config "$CONFIG" "$s" "$calib_src" "$calib_label"
    fi
  done
done

log "PHASE A DONE. Checkpoints in $WORK_DIR/out/"
log "Next: bash scripts/run_evals.sh $CONFIG"
