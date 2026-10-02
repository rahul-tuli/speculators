#!/usr/bin/env bash
set -eo pipefail
source /workspace/speculators/.venv/bin/activate
export HF_HOME=/data/fast/hf_cache
export HF_HUB_CACHE=/data/fast/hf_cache/hub
export HF_HUB_DISABLE_XET=1
export PATH="/workspace/speculators/.venv/bin:$PATH"
cd /workspace/speculators/local/drafter-quant
mkdir -p logs
bash scripts/run_checkpoints.sh config/dflash-qwen3-8b.env 2>&1 | tee logs/run_checkpoints.log
echo "EXIT_CODE=$?" >> logs/run_checkpoints.log
