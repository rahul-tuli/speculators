#!/usr/bin/env bash
# ============================================================================
# 00_setup.sh — one-time environment setup on the GPU box.
#
# Installs the pinned repos in dependency order (compressed-tensors FIRST,
# then llm-compressor, then speculators) with `pip install -e`, then runs
# import + CUDA sanity checks.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=../lib/common.sh
source "$PROJECT_ROOT/lib/common.sh"

usage() {
  cat <<EOF
Usage: bash scripts/00_setup.sh [--help]

Installs (pip install -e, in order — order matters):
  1. compressed-tensors   2. llm-compressor   3. speculators
from the pinned clones, installs hs_connectors if the speculators install did
not pull it, then verifies torch CUDA availability and package imports.

Env overrides:
  DRAFTERS_QUANT_REPOS  pinned-repo root (default: /workspace; set in lib/common.sh)
  PIP                   pip executable     (default: pip)
EOF
}

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
  "") ;;
  *) usage >&2; exit 2 ;;
esac

DRAFTERS_QUANT_REPOS=${DRAFTERS_QUANT_REPOS:-/workspace}
PIP=${PIP:-pip}

for repo in compressed-tensors llm-compressor speculators; do
  [ -d "$DRAFTERS_QUANT_REPOS/$repo" ] || die "missing pinned clone: $DRAFTERS_QUANT_REPOS/$repo"
done

log "installing pinned repos from $DRAFTERS_QUANT_REPOS (order matters)"
"$PIP" install -e "$DRAFTERS_QUANT_REPOS/compressed-tensors"
"$PIP" install -e "$DRAFTERS_QUANT_REPOS/llm-compressor"
"$PIP" install -e "$DRAFTERS_QUANT_REPOS/speculators"

# The fp8 hidden-states backend needs speculators' hs_connectors package;
# `pip install -e speculators` should pull it — fall back to a direct install.
if ! python -c "import hs_connectors" 2>/dev/null; then
  log "hs_connectors not importable; installing speculators/hs_connectors"
  "$PIP" install -e "$DRAFTERS_QUANT_REPOS/speculators/hs_connectors"
fi

log "sanity checks"
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '(no GPU)')"
python -c "import llmcompressor, speculators; print('imports OK')"
log "setup complete"
