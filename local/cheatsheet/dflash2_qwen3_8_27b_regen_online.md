# Qwen3.8-27B DFlash2 training cheatsheet

The recipe uses the regenerated dataset at `/data/fast/OpenPerfectBlend-Qwen38-27B-regenerated/data/train.jsonl.gz`, the Speculators default learning rate (`1e-3`), and `CHECKPOINT_FREQ=0.1`. Prepared data, checkpoints, Trackio's local database, and the pipeline log use the stable output directory `/data/fast/dflash2_qwen3_8_27b_regen`.

## Start a full run in `screen`

This starts the recipe detached, with all samples enabled. The recipe uses the project `.venv` for training and `/workspace/vllm/.venv/bin/python` for vLLM.

```bash
SESSION=dflash2-qwen3-8-27b-regen
OUTPUT_DIR=/data/fast/dflash2_qwen3_8_27b_regen
mkdir -p "$OUTPUT_DIR"

screen -dmS "$SESSION" bash -lc '
set -euo pipefail
cd /workspace/speculators
source .venv/bin/activate
export OUTPUT_DIR=/data/fast/dflash2_qwen3_8_27b_regen
export MAX_SAMPLES=-1
export TRACKIO_DIR="$OUTPUT_DIR/trackio"
bash examples/train/dflash2_qwen3_8_27b_regen_online.sh 2>&1 | tee -a "$OUTPUT_DIR/pipeline.log"
'
```

Check that the session is running and follow its log:

```bash
screen -ls
tail -f /data/fast/dflash2_qwen3_8_27b_regen/pipeline.log
```

Attach to the session with `screen -r dflash2-qwen3-8-27b-regen`. Detach without stopping training using **Ctrl-A**, then **D**.

## Resume after an interruption

`CHECKPOINT_FREQ=0.1` saves a numeric checkpoint every 10% of the training epoch. Rerunning with the same output directory reuses completed prepared data and resumes from the latest numeric checkpoint. A graceful interrupt may also write an `interrupted` checkpoint, which is not selected automatically; resume uses the latest periodic numeric checkpoint. If preparation is interrupted before `.arrow` dataset files are written, data preparation runs again.

```bash
SESSION=dflash2-qwen3-8-27b-regen-resume
OUTPUT_DIR=/data/fast/dflash2_qwen3_8_27b_regen

screen -dmS "$SESSION" bash -lc '
set -euo pipefail
cd /workspace/speculators
source .venv/bin/activate
export OUTPUT_DIR=/data/fast/dflash2_qwen3_8_27b_regen
export MAX_SAMPLES=-1
export TRACKIO_DIR="$OUTPUT_DIR/trackio"
bash examples/train/dflash2_qwen3_8_27b_regen_online.sh 2>&1 | tee -a "$OUTPUT_DIR/pipeline.log"
'
```

Use the same `OUTPUT_DIR` and `TRACKIO_DIR` each time. Do not start a second copy while the original training process is still running. Check `screen -ls` first; attach with `screen -r <session>` if it is still active.

## View Trackio

Training metrics and the resolved run configuration are logged under the Trackio project `speculators`. With no `TRACKIO_SPACE_ID` configured, Trackio stores them locally in the output directory:

```bash
cd /workspace/speculators
source .venv/bin/activate
export TRACKIO_DIR=/data/fast/dflash2_qwen3_8_27b_regen/trackio
trackio show --project speculators
```

Each recipe start gets a timestamped run in the same project/dashboard. To use a Hugging Face Space dashboard instead, set the same `TRACKIO_SPACE_ID=namespace/space-name` for both training and the `trackio show` command.

The full console output from preprocessing, vLLM, and training is saved separately in `pipeline.log`.
