#!/usr/bin/env bash
# ============================================================================
# 20_capture_hidden_states.sh — tokenize the perfectblend subset and capture
# target hidden states OFFLINE (fits in 200 GB; see space estimate below).
#
# Pipeline (all speculators built-ins, verified @ 40b12c9):
#   1. launch vLLM server for the TARGET model with hidden-states extraction
#      (scripts/launch_vllm.py; sets VLLM_ENABLE_SCALE_OUT_ENDPOINTS=1)
#   2. speculators prepare-data        (conversations -> input_ids/loss_mask/seq_len,
#                                       rendered server-side)
#   3. speculators generate-offline-data (one request/sample, hidden states ->
#                                       <PREPARED>/hidden_states/hs_<i>.safetensors)
#
# Space math (dflash-8B defaults): stored tensor is [seq, n_layers+1, H] =
# [seq, 6, 4096]; fp8 backend => 24 KB/token, bf16 (file backend) => 48 KB/token.
#   worst case 2048 samples x 2048 tok x 24 KB = 96 GB (fp8) / 192 GB (bf16)
#   realistic 2048 samples x ~700 tok x 24 KB = ~33 GB (fp8)
# => offline fits in 200 GB with the fp8 backend; bf16 is tight at worst case.
# Online fallback (zero persistent footprint): use ArrowDataset with
# --on-missing generate --on-generate delete during calibration instead —
# note it regenerates per run and needs the server alive during calibration.
#
# Existing caches are reused only after identity and completeness checks.
# Invalid caches are preserved and rejected; never delete data to resume.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=../lib/common.sh
source "$PROJECT_ROOT/lib/common.sh"

usage() {
  cat <<EOF
Usage: bash scripts/20_capture_hidden_states.sh [--config FILE]

Tokenizes DATA with the target model's chat template (server-side render) and
captures target hidden states offline to <PREPARED>/hidden_states/.

Options:
  --config FILE   config env file (default: config/dflash-qwen3-8b.env);
                  provides TARGET, LAYER_IDS, HIDDEN.

Env overrides:
  DRAFTERS_QUANT_REPOS  pinned-repo root   (default: /workspace; set in lib/common.sh)
  SPECDIR         speculators clone  (default: \$DRAFTERS_QUANT_REPOS/speculators)
  BACKEND         fp8 (half disk) | file (bf16)   (default: fp8)
  PORT            vLLM server port                  (default: 8100)
  DATA            input JSONL          (default: \$WORK_DIR/data/perfectblend_<MAX_SAMPLES>/perfectblend_<MAX_SAMPLES>.jsonl)
  PREPARED        prepared output dir  (default: target/backend/layers/shape/budget-specific path)
  SEQ_LEN         max sequence length               (default: 2048)
  MAX_SAMPLES     max samples                       (default: 2048)
  CONCURRENCY     generate-offline-data concurrency (default: 32)
  DISK_BUDGET_GB  preflight disk budget             (default: 180)
  SERVER_LOG      server log path      (default: \$WORK_DIR/logs/calib_vllm_server.log)
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

DRAFTERS_QUANT_REPOS=${DRAFTERS_QUANT_REPOS:-/workspace}
SPECDIR=${SPECDIR:-$DRAFTERS_QUANT_REPOS/speculators}
BACKEND=${BACKEND:-fp8}                    # fp8 (half disk) | file (bf16)
PORT=${PORT:-8100}
SEQ_LEN=${SEQ_LEN:-2048}
MAX_SAMPLES=${MAX_SAMPLES:-2048}
DATA=${DATA:-$WORK_DIR/data/perfectblend_${MAX_SAMPLES}/perfectblend_${MAX_SAMPLES}.jsonl}
PREPARED=${PREPARED:-$(prepared_path "$MAX_SAMPLES" "$SEQ_LEN" "${SEED:-0}")}
CONCURRENCY=${CONCURRENCY:-32}
DISK_BUDGET_GB=${DISK_BUDGET_GB:-180}
SERVER_LOG=${SERVER_LOG:-$WORK_DIR/logs/calib_vllm_server.log}

RAW_HS_DIR="${RAW_HS_DIR:-${PREPARED}_capture_staging}"
mkdir -p "$WORK_DIR/logs" "$(dirname "$PREPARED")" "$RAW_HS_DIR"

if [ -d "$PREPARED/hidden_states" ]; then
  log "validating existing cache: $PREPARED"
  (cd "$PROJECT_ROOT" && python -m dquant.calibration "$PREPARED" \
    --target "$TARGET" --layers $LAYER_IDS --hidden-size "$HIDDEN")
  log "complete, compatible cache; reusing $PREPARED/hidden_states"
  exit 0
fi
[ -f "$DATA" ] || die "DATA not found: $DATA (run scripts/10_prepare_dataset.sh first)"

# ---- preflight: disk-space estimate ----------------------------------------
# shellcheck disable=SC2086
N_LAYERS=$(($(echo $LAYER_IDS | wc -w) + 1))   # +1 for last layer
BYTES_PER_EL=2; [ "$BACKEND" = "fp8" ] && BYTES_PER_EL=1
EST_GB=$(( MAX_SAMPLES * SEQ_LEN * N_LAYERS * HIDDEN * BYTES_PER_EL / 1024 / 1024 / 1024 ))
log "preflight: ${MAX_SAMPLES} x ${SEQ_LEN} tok x ${N_LAYERS} layers x ${HIDDEN} x ${BYTES_PER_EL}B"
preflight_disk_check "$EST_GB" "$DISK_BUDGET_GB" "$PREPARED"

# ---- 1. launch vLLM with hidden-states extraction --------------------------
log "[1/3] launching vLLM server for $TARGET (backend=$BACKEND, port=$PORT)"
# shellcheck disable=SC2086
python "$SPECDIR/scripts/launch_vllm.py" train "$TARGET" \
  --target-layer-ids $LAYER_IDS \
  --hidden-states-backend "$BACKEND" \
  --hidden-states-path "$RAW_HS_DIR" \
  --provenance-dir "$PREPARED" \
  -- --port "$PORT" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap stop_server EXIT

log "[1/3] waiting for server health..."
wait_for_server "$PORT" "$SERVER_PID" "$SERVER_LOG"
log "[1/3] server is up"

# ---- 2. tokenize / render (conversations -> input_ids + loss_mask) ---------
log "[2/3] prepare-data: $DATA -> $PREPARED"
speculators prepare-data \
  --model "$TARGET" \
  --data "$DATA" \
  --output "$PREPARED" \
  --max-samples "$MAX_SAMPLES" \
  --seq-length "$SEQ_LEN" \
  --render-endpoint "http://localhost:${PORT}"

# ---- 3. capture hidden states offline --------------------------------------
log "[3/3] generate-offline-data -> $PREPARED/hidden_states (resumable)"
speculators generate-offline-data \
  --endpoint "http://localhost:${PORT}/v1" \
  --model "$TARGET" \
  --preprocessed-data "$PREPARED" \
  --max-samples "$MAX_SAMPLES" \
  --concurrency "$CONCURRENCY" \
  --fail-on-error

N_HS=$(ls "$PREPARED"/hidden_states/hs_*.safetensors 2>/dev/null | wc -l | tr -d ' ')
log "[done] ${N_HS} hidden-state files in $PREPARED/hidden_states"
log "[next] quantize with real data, e.g.:"
log "  bash scripts/30_quantize.sh fp8_w8a8 $PREPARED perfectblend"
