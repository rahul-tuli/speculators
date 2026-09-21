#!/usr/bin/env bash
# Run an explicitly identified arm at fixed concurrency; preserve all attempts.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$PROJECT_ROOT/lib/common.sh"
CONFIG="$PROJECT_ROOT/config/dflash-qwen3-8b.env"
if [[ "${1:-}" == --config ]]; then CONFIG=$2; shift 2; fi
if [[ $# -lt 2 ]]; then
  echo 'Usage: 40_eval.sh [--config FILE] <drafter|none> <arm> [method]'
  echo 'Required env: DATA_MANIFEST, EVAL_DATASETS (newline-separated paths), EVAL_REPEAT, ARM_ORDER'
  exit 2
fi
DRAFTER_CKPT=$1
ARM=$2
load_config "$CONFIG"
: "${DATA_MANIFEST:?Frozen prepared-data manifest required}"
: "${EVAL_DATASETS:?Newline-separated JSONL paths required}"
: "${EVAL_REPEAT:?Explicit repeat index required}"
: "${ARM_ORDER:?Explicit rotated arm order required}"
mapfile -t DATA_FILES <<< "$EVAL_DATASETS"
read -ra LOADS <<< "${CONCURRENCIES:-1}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python -m dquant.evaluate_arm --target "$TARGET" --drafter "$DRAFTER_CKPT" \
  --arm "$ARM" --method "${3:-$METHOD}" --datasets "${DATA_FILES[@]}" \
  --manifest "$DATA_MANIFEST" --repeat "$EVAL_REPEAT" --order "$ARM_ORDER" \
  --phase "${EVAL_PHASE:-acceptance}" --concurrencies "${LOADS[@]}" \
  --warmup-requests "${WARMUP_REQUESTS:-2}" --port "${PORT:-8108}" \
  --output "${RESULTS_DIR:-$WORK_DIR/results}"
