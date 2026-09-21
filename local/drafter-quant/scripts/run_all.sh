#!/usr/bin/env bash
# ============================================================================
# run_all.sh — END TO END: does calibration data matter for drafter quant?
#
# Thin wrapper that runs the two phases back to back:
#   Phase A  scripts/run_checkpoints.sh   dataset -> hidden states -> checkpoints
#   Phase B  scripts/run_evals.sh         serve + eval every arm -> plots
#
# For long runs you usually want the phases separately (checkpoint generation
# and eval can live on different machines/time windows):
#   bash scripts/run_checkpoints.sh config/dflash-qwen3-8b.env
#   bash scripts/run_evals.sh       config/dflash-qwen3-8b.env
#
# Everything is idempotent: existing outputs are skipped.
#
#   bash scripts/run_all.sh config/dflash-qwen3-8b.env
#   bash scripts/run_all.sh config/dspark-qwen3-4b.env
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=../lib/common.sh
source "$PROJECT_ROOT/lib/common.sh"

usage() {
  cat <<EOF
Usage: bash scripts/run_all.sh [--help] <config-file>

Runs Phase A (run_checkpoints.sh) then Phase B (run_evals.sh), both
idempotently. All env overrides are documented in the phase scripts' --help.

  config-file   one of config/*.env, e.g.
                  config/dflash-qwen3-8b.env   (Qwen3-8B + DFlash drafter)
                  config/dspark-qwen3-4b.env   (Qwen3-4B + DSpark drafter)

Key env overrides:
  WORK_DIR      all outputs root     (default: /data/fast/drafter-quant)
  DRAFTERS_QUANT_REPOS  pinned repos (default: /workspace)
  NUM_SAMPLES / SEQ_LEN / SEED / SCHEMES / CALIBS / RESULTS_DIR
EOF
}

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
  "") usage >&2; exit 2 ;;
esac
CONFIG=$1

log "========== PHASE A: checkpoints =========="
bash "$SCRIPT_DIR/run_checkpoints.sh" "$CONFIG"

log "========== PHASE B: evals ================"
bash "$SCRIPT_DIR/run_evals.sh" "$CONFIG"

log "ALL DONE. Plots: ${RESULTS_DIR:-$WORK_DIR/results}/plots/"
