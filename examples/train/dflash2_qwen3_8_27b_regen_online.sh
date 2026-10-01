#!/bin/bash
# Online DFlash2 Training Recipe — Qwen3.8-27B (reproduction-aligned)
#
# Trains a five-layer DFlash2 drafter for Qwen/Qwen3.8-27B on an on-policy
# regenerated dataset (assistant turns produced by the target model itself).
# The objective defaults below follow the published DFlash 2 reproduction
# recipe: hard-target cross-entropy, loss_decay_gamma=7, and the strict
# top-k selector objective (no target injection). These are the
# out-of-the-box defaults as of this PR; the flags are spelled out anyway so
# the recipe documents itself.
#
# Reference points this recipe is aligned with:
#   - incoai/Qwen3.8-27B-DFlash2 checkpoint contract: 5 layers, block_size 8,
#     conv (kernel 2, group 16), selector (rank 256, top-k 16), aux layers
#     5/19/33/47/61, mask token 248070.
#   - SpecForge recipe "qwen3.8-27b-dflash2": constant lr 5e-4,
#     loss_decay_gamma 7, hard-target CE with the strict top-k selector
#     objective, ~512 samples per optimizer step for full-convergence runs.
#
# Warm start: not required. DFlash2 with identity convolutions and the
# zero-initialized selector is exactly a DFlash model at step 0, so cold
# start is well-defined. If you want to warm-start the backbone from a
# pretrained DFlash drafter (e.g. a RedHatAI *-speculator.dflash checkpoint
# for your target), pass --from-pretrained <hf-id-or-path> and drop the
# decoder-shaping flags; note RedHatAI does not currently publish a
# Qwen3.8-27B DFlash checkpoint, so this recipe cold-starts.
#
# Usage: copy this script, set DATASET to your regenerated data, adjust GPU
# assignments, then run:
#   bash examples/train/dflash2_qwen3_8_27b_regen_online.sh

set -euo pipefail

# ============ Configuration ============
MODEL="Qwen/Qwen3.8-27B"
# On-policy regenerated dataset: local path, HF id, or preset. Must be
# pretokenized (input_ids + loss_mask) or rendered via --render-endpoint.
DATASET="/path/to/qwen3.8-27b-regen-train.jsonl"
OUTPUT_DIR="./output/dflash2_qwen3_8_27b_regen"
VLLM_PORT=8000
SEQ_LENGTH=8192
EPOCHS=1                 # one epoch over a ~1M-sample regen corpus is the reference recipe
LR=5e-4                  # constant, per the reproduction recipe
MAX_SAMPLES=-1           # set to a small number (e.g. 5000) for a pipeline smoke test

# DFlash2 architecture (matches the incoai/Qwen3.8-27B-DFlash2 contract)
SPECULATOR_TYPE="dflash2"
NUM_LAYERS=5
TARGET_LAYER_IDS="5 19 33 47 61"  # Must match vLLM's eagle_aux_hidden_state_layer_ids
BLOCK_SIZE=8
MASK_TOKEN_ID=248070     # <|audio_start|>; never produced by text
CONV_KERNEL_SIZE=2
CONV_GROUP_SIZE=16
SELECTOR_RANK=256
SELECTOR_TOP_K=16

# Objective (these are the defaults; pinned here for explicitness)
LOSS_FN="ce"
GAMMA=7.0
SELECTOR_CANDIDATE_MODE="strict-topk"
MAX_ANCHORS=3072

# GPU assignments (online training needs separate GPUs for vLLM and training).
# The reference layout for this target on one 8xH200 node is 5 capture
# servers + 3 trainer ranks; size to your hardware.
VLLM_GPUS="0,1,2,3"
TRAIN_GPUS="4,5,6,7"
NUM_TRAIN_GPUS=4
# =======================================

# Step 1: Prepare data
# The dataset is on-policy regenerated: assistant turns were produced by the
# target model and ship pretokenized (input_ids + loss_mask). prepare-data just
# packages them -- no --render-endpoint (or running vLLM server) needed here.
echo "=== Step 1: Preparing data ==="
PREPARE_ARGS=(
    --model "$MODEL"
    --data "$DATASET"
    --output "$OUTPUT_DIR"
    --seq-length "$SEQ_LENGTH"
)
if [ "$MAX_SAMPLES" -gt 0 ]; then
    PREPARE_ARGS+=(--max-samples "$MAX_SAMPLES")
fi
speculators prepare-data "${PREPARE_ARGS[@]}"

# Step 2: Launch vLLM server in the background
echo "=== Step 2: Launching vLLM server ==="
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --data-parallel-size 4 --port "$VLLM_PORT" &
VLLM_PID=$!

# Ensure vLLM is cleaned up on exit
cleanup() {
    echo "Stopping vLLM server..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server to be ready..."
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 2
done
echo "vLLM server ready."

# Step 3: Train against the live vLLM server
echo "=== Step 3: Training ==="
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    -m speculators.train \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --scheduler-type none \
    --total-seq-len "$SEQ_LENGTH" \
    --speculator-type "$SPECULATOR_TYPE" \
    --num-layers "$NUM_LAYERS" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --block-size "$BLOCK_SIZE" \
    --mask-token-id "$MASK_TOKEN_ID" \
    --conv-kernel-size "$CONV_KERNEL_SIZE" \
    --conv-group-size "$CONV_GROUP_SIZE" \
    --selector-rank "$SELECTOR_RANK" \
    --selector-top-k "$SELECTOR_TOP_K" \
    --loss-fn "$LOSS_FN" \
    --dflash-decay-gamma "$GAMMA" \
    --selector-candidate-mode "$SELECTOR_CANDIDATE_MODE" \
    --max-anchors "$MAX_ANCHORS" \
    --on-missing generate \
    --on-generate delete

echo "Done. Checkpoints saved to $OUTPUT_DIR/checkpoints/"

# Validation: track these training metrics against the published yardsticks
# (inco.ai/blog/dflash2, five-layer Qwen3-4B on GSM8K):
#   - unary_candidate_recall_at_16  ~0.995 at position 0, decaying down-block
#   - unary_top_16_oracle_accepted_length  (4.27 unary-greedy -> 6.79 oracle)
#   - teacher_forced_selector_acc  rising steadily once the selector leaves
#     its zero-initialized no-op start
#   - eal (realized greedy selector path) is the headline metric
# Ablations for comparison runs: --selector-candidate-mode inject,
# --loss-fn kl_div, --dflash-decay-gamma 4.0.
