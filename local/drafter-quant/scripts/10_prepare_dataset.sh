#!/usr/bin/env bash
# ============================================================================
# 10_prepare_dataset.sh — stratified "perfectblend" subset sampler (CPU only).
#
# Wraps `python -m dquant.sample_dataset`: uniform quota across the 13 task
# sources of inference-optimization/Qwen3-8B-Regenerated-Collection, with
# deterministic seed and deficit redistribution.
#
# Idempotent: skips if the combined JSONL already exists.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=../lib/common.sh
source "$PROJECT_ROOT/lib/common.sh"

usage() {
  cat <<EOF
Usage: bash scripts/10_prepare_dataset.sh [--config FILE] [-- <extra dquant.sample_dataset flags>]

Builds the even task-mixed subset of
inference-optimization/Qwen3-8B-Regenerated-Collection.

Options:
  --config FILE   config env file (default: config/dflash-qwen3-8b.env)

Env overrides:
  NUM_SAMPLES   samples to draw          (default: 2048)
  SEED          sampling seed            (default: 0)
  OUT_DIR       output dir               (default: $WORK_DIR/data/perfectblend_<NUM_SAMPLES>)
  CACHE_DIR     HF download cache        (default: $HF_HOME, i.e. /data/fast/hf_cache)

Outputs:
  <OUT_DIR>/perfectblend_<NUM_SAMPLES>.jsonl   combined subset ("source" tag per row)
  <OUT_DIR>/mix_manifest.json                  per-file quotas/counts (provenance)
EOF
}

CONFIG="$PROJECT_ROOT/config/dflash-qwen3-8b.env"
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA+=("$@"); break ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done
load_config "$CONFIG"

NUM_SAMPLES=${NUM_SAMPLES:-2048}
SEED=${SEED:-0}
OUT_DIR=${OUT_DIR:-$WORK_DIR/data/perfectblend_${NUM_SAMPLES}}
CACHE_DIR=${CACHE_DIR:-}

JSONL="$OUT_DIR/perfectblend_${NUM_SAMPLES}.jsonl"
if [ -f "$JSONL" ]; then
  log "exists, skipping: $JSONL"
  exit 0
fi

CMD=(python -m dquant.sample_dataset
     --num-samples "$NUM_SAMPLES" --seed "$SEED" --output "$OUT_DIR")
[ -n "$CACHE_DIR" ] && CMD+=(--cache-dir "$CACHE_DIR")
[ ${#EXTRA[@]} -gt 0 ] && CMD+=("${EXTRA[@]}")

log "[1/6] sampling perfectblend subset (${NUM_SAMPLES} samples) -> $OUT_DIR"
(cd "$PROJECT_ROOT" && "${CMD[@]}")
log "done: $JSONL"
