#!/usr/bin/env bash
# ============================================================================
# 50_plot.sh — plots + summary over results/eval_* (CPU only).
#
# Wraps `python -m dquant.plot_results`: acceptance-by-subset bars,
# speed-vs-acceptance scatter, per-position curves, summary.csv.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=../lib/common.sh
source "$PROJECT_ROOT/lib/common.sh"

usage() {
  cat <<EOF
Usage: bash scripts/50_plot.sh [--config FILE]

Scans <RESULTS_DIR>/eval_* and writes plots + summary.csv to <PLOTS_OUT>.

Options:
  --config FILE   config env file (default: config/dflash-qwen3-8b.env);
                  sourced for consistency, no plot vars required.

Env overrides:
  RESULTS_DIR   results root  (default: \$WORK_DIR/results)
  PLOTS_OUT     plots output  (default: <RESULTS_DIR>/plots)
EOF
}

CONFIG="$PROJECT_ROOT/config/dflash-qwen3-8b.env"
while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
load_config "$CONFIG"

RESULTS_DIR=${RESULTS_DIR:-$WORK_DIR/results}
PLOTS_OUT=${PLOTS_OUT:-$RESULTS_DIR/plots}

log "[6/6] plots + summary: $RESULTS_DIR -> $PLOTS_OUT"
(cd "$PROJECT_ROOT" && python -m dquant.plot_results \
  --results-dir "$RESULTS_DIR" --output "$PLOTS_OUT")
log "done: $PLOTS_OUT"
