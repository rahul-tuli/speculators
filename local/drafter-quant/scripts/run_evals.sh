#!/usr/bin/env bash
# Three rotated, isolated rounds. Plotting and later model phases are separate.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$PROJECT_ROOT/lib/common.sh"
CONFIG=${1:?Usage: run_evals.sh <config>}
load_config "$CONFIG"
: "${DATA_MANIFEST:?Frozen manifest required}"
: "${EVAL_DATASETS:?Newline-separated prepared files required}"
OUT_ROOT=${OUT_ROOT:-$WORK_DIR/out}
MODEL_TAG=${MODEL_TAG:-$(model_tag)}
SEED=${CALIBRATION_SEED:-0}
ARMS=(bf16-drafter fp8-gaussian fp8-real nvfp4-gaussian nvfp4-real)
CHECKPOINTS=("$DRAFTER")
for scheme in fp8_w8a8 nvfp4_w4a4; do
  for recipe in random real; do
    ckpt="$OUT_ROOT/${MODEL_TAG}-${METHOD}-${scheme}-${recipe}-seed${SEED}"
    [[ -f "$ckpt/quant_run_manifest.json" ]] || die "Validated full-budget checkpoint missing: $ckpt"
    CHECKPOINTS+=("$ckpt")
  done
done
if [[ "${EVAL_PHASE:-acceptance}" == speed ]]; then
  ARMS+=(target-only)
  CHECKPOINTS+=(none)
  export CONCURRENCIES=${CONCURRENCIES:-"1 8 32 128"}
fi
for repeat in 0 1 2; do
  for ((order=0; order<${#ARMS[@]}; order++)); do
    index=$(((order+repeat)%${#ARMS[@]}))
    EVAL_REPEAT=$repeat ARM_ORDER=$order bash "$SCRIPT_DIR/40_eval.sh" --config "$CONFIG" \
      "${CHECKPOINTS[$index]}" "${ARMS[$index]}" "$METHOD"
  done
done
