#!/usr/bin/env bash
# lib/common.sh — shared helpers for the drafter-quant scripts.
# Source this file; do not execute it directly.
#
# Provides:
#   log <msg...>                          timestamped log line to stderr
#   die <msg...>                          log + exit 1
#   load_config <file>                    source a config env file (die if missing)
#   wait_for_server <port> <pid> <log>    block until /health answers (or die)
#   preflight_disk_check <gb> <budget> <path>
#   stop_server                           trap helper; kills $SERVER_PID if set
#   model_tag                             short lowercase tag derived from $DRAFTER
#
# Central path defaults (all overridable via env):
#   DRAFTERS_QUANT_REPOS  pinned-repo clones root   (default: /workspace)
#   WORK_DIR              ALL outputs root          (default: /data/fast/drafter-quant)
#   HF_HOME               HF download/weights cache (default: /data/fast/hf_cache)
DRAFTERS_QUANT_REPOS=${DRAFTERS_QUANT_REPOS:-/workspace}
WORK_DIR=${WORK_DIR:-/data/fast/drafter-quant}
export HF_HOME=/data/fast/hf_cache
export HF_HUB_CACHE=/data/fast/hf_cache/hub
export HF_HUB_DISABLE_XET=1
mkdir -p "$WORK_DIR" "$HF_HOME"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

die() {
  log "ERROR: $*"
  exit 1
}

# load_config <file> — source a config env file (config/*.env).
load_config() {
  local cfg=$1
  [ -f "$cfg" ] || die "config file not found: $cfg"
  log "config: $cfg"
  # shellcheck disable=SC1090
  source "$cfg"
}

# wait_for_server <port> <pid> <logfile>
# Polls http://localhost:<port>/health every 5s for up to 20 min (240 x 5s),
# dying early if the server process exits. Mirrors the verified wait loop from
# the original run_calibration_data.sh / eval_drafter.sh.
wait_for_server() {
  local port=$1 pid=$2 logf=$3 i
  for i in $(seq 1 240); do
    if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
      return 0
    fi
    sleep 5
    if ! kill -0 "$pid" 2>/dev/null; then
      die "server died; see $logf"
    fi
    if [ "$i" -eq 240 ]; then
      die "server never became healthy; see $logf"
    fi
  done
}

# preflight_disk_check <gb_est> <budget_gb> <path>
# Fails if the worst-case estimate exceeds the budget; always reports the
# space available on the filesystem holding <path>.
preflight_disk_check() {
  local est_gb=$1 budget_gb=$2 path=$3 avail_gb
  log "preflight: worst-case hidden-state footprint: ${est_gb} GB (budget ${budget_gb} GB)"
  if [ "$est_gb" -gt "$budget_gb" ]; then
    die "estimate ${est_gb} GB exceeds budget ${budget_gb} GB -> use BACKEND=fp8, lower SEQ_LEN, or lower MAX_SAMPLES"
  fi
  avail_gb=$(df -BG "$(dirname "$path")" 2>/dev/null | awk 'NR==2{print $4}' || df -g "$(dirname "$path")" 2>/dev/null | awk 'NR==2{print $4}')
  log "preflight: available on target fs: ${avail_gb} GB"
}

# stop_server — install with `trap stop_server EXIT` after SERVER_PID is set.
SERVER_PID=""
stop_server() {
  if [ -n "${SERVER_PID:-}" ]; then
    log "cleanup: stopping server (pid $SERVER_PID)"
    kill "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 10); do
      if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        SERVER_PID=""
        return 0
      fi
      sleep 1
    done
    kill -9 "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
}

# model_tag — derive a short lowercase tag from $DRAFTER, e.g.
#   RedHatAI/Qwen3-8B-speculator.dflash -> qwen3-8b
# Used for out/ checkpoint dir names and eval label bookkeeping.
model_tag() {
  basename "$DRAFTER" | sed 's/-speculator.*//' | tr 'A-Z' 'a-z'
}

# Keep target-specific captures separate even when source conversations are shared.
prepared_path() {
  local n=$1 seq=$2 seed=$3 target_tag=${TARGET//\//--} layers=${LAYER_IDS// /-}
  printf '%s/data/%s-%s-layers-%s-h%s-n%s-seq%s-seed%s_prepared\n' \
    "$WORK_DIR" "$target_tag" "${BACKEND:-fp8}" "$layers" "$HIDDEN" "$n" "$seq" "$seed"
}
