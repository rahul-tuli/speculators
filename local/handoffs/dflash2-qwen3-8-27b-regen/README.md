# Handoff: Qwen3.8-27B DFlash2 regenerated-data training

Last updated: 2026-10-02 (UTC)

## Objective and current status

The work is to train a DFlash2 drafter for `Qwen/Qwen3.8-27B` using the regenerated OpenPerfectBlend dataset and the repository's online hidden-state training flow. The recipe was updated and that flow completed a one-epoch, 1,000-sample end-to-end smoke run, including validation and checkpointing.

The full run completed data preparation, started training on 2026-10-01, and was stopped with SIGINT on 2026-10-02. The last observed progress was step 5,974/59,429 (about 10% of the epoch). The run had `--save-best`, which suppressed the configured sub-epoch checkpoints; graceful shutdown saved only `checkpoints/interrupted`, which is not automatically resumed. The prepared Arrow dataset remains in the output directory. With the current recipe, the next launch reuses it and starts training from scratch; periodic 10% checkpoints are enabled again. The smoke output has a checkpoint trained on only 1,000 rows and should stay separate.

## Dataset

- Hugging Face dataset: `shanjiaz/OpenPerfectBlend-Qwen38-27B-regenerated`
- Downloaded source file: `/data/fast/OpenPerfectBlend-Qwen38-27B-regenerated/data/train.jsonl.gz`
- The recipe reads this gzip JSONL directly. It uses `prepare-data` on the actual dataset; the smoke test did not create a separate/synthetic input.
- Dataset scan: 1,884,616 rows. `input_ids` length quantiles were p50 1,244, p90 9,217, p95 16,253, p99 24,874, p99.9 33,237, maximum 65,537 tokens.

## Sequence length and vLLM capacity decision

The agreed target is p95 coverage. The recipe sets `SEQ_LENGTH=16384`, which covers the p95 of 16,253 tokens; longer examples are truncated to this training length by preparation. Rows that have no supervised tokens remaining after truncation can be filtered by the preparation flow, so this does not promise that every long row is unchanged.

vLLM is configured with `--max-model-len 16385`: the prepared input can be 16,384 tokens and the hidden-state extraction request reserves one generated token. At the model-config default of 262,144, one-GPU vLLM estimated a 272.46-GiB KV-cache requirement against 18.57 GiB available and failed. With one-GPU replicas at a 16,385 limit, vLLM reported 17,437 KV-cache tokens and 1.06x concurrency per replica. Data parallelism uses two one-GPU replicas; it does not combine those caches. This configuration started and served the smoke run.

## Recipe settings

Recipe: [`examples/train/dflash2_qwen3_8_27b_regen_online.sh`](../../../examples/train/dflash2_qwen3_8_27b_regen_online.sh)

- Dataset default: the downloaded path above.
- Full-run output default: `/data/fast/dflash2_qwen3_8_27b_regen`.
- `MAX_SAMPLES` defaults to `-1`; the smoke command overrode it to `1000`.
- vLLM Python: `/workspace/vllm/.venv/bin/python`.
- Training uses `python` after activating `/workspace/speculators/.venv`.
- GPUs: vLLM `0,1` with data-parallel size 2; training `2,3,4,5,6,7` with six distributed ranks.
- DFlash2 target layers passed to training/vLLM: `5 19 33 47 61`. The final target layer is appended by the vLLM launch flow; do not add it to this list.
- Learning rate is omitted from the training CLI so it uses the Speculators default `1e-3` (Muon LR inherits that value).
- `CHECKPOINT_FREQ` remains configured as `0.1` and is passed to the trainer. It saves checkpoints every 10% of the training epoch, allowing mid-epoch resume from the latest numeric checkpoint.
- The trainer resumes automatically from an existing numeric epoch checkpoint in the selected output directory. A graceful mid-epoch interruption writes an `interrupted` checkpoint, which is not selected for automatic resume; the latest periodic numeric checkpoint is used instead.
- Trackio local data is rooted at `${OUTPUT_DIR}/trackio`; project is `speculators`. Keep the same output and Trackio directories when resuming to continue in the same local dashboard/database.
- vLLM launch passes `--provenance-dir "$OUTPUT_DIR"`, as required by the repository's provenance guidance.

The initial 16K training attempt OOMed on the first step with 3,072 max anchors, before gradient checkpointing was enabled. The failure was in `_build_attention_mask` while allocating a 480-MiB tensor on a training GPU. The recipe now defaults to `MAX_ANCHORS=512` and enables `--gradient-checkpointing`; the smoke run completed with those settings. The lower anchor limit reduces the number of sampled anchors per example compared with 3,072.

## Smoke run result

- Command was run from `/workspace/speculators` after activating `.venv`:

  ```bash
  source .venv/bin/activate
  MAX_SAMPLES=1000 \
    OUTPUT_DIR=/data/fast/dflash2_qwen3_8_27b_regen_smoke1000_16k \
    TRACKIO_DIR=/data/fast/dflash2_qwen3_8_27b_regen_smoke1000_16k/trackio \
    bash examples/train/dflash2_qwen3_8_27b_regen_online.sh
  ```

- Prepared exactly 1,000 rows. The configured 90/10 split yielded 31 training steps and 3 validation batches.

- Training finished epoch 1/1 at global step 31; validation finished with `val/loss_epoch=7.639962`. No OOM or traceback occurred in this successful attempt. The process exited 0 and all eight GPUs were back at 0 MiB used.

- This smoke run used the earlier `5e-4` learning rate and periodic checkpoint setup. It validates the pipeline and memory settings; it did not exercise the current default `1e-3` learning rate.

- Checkpoint: `/data/fast/dflash2_qwen3_8_27b_regen_smoke1000_16k/checkpoints/0`. It includes model weights, optimizer state, `training_state.json` with `global_step: 31`, validation metrics, and training provenance.

- vLLM provenance/checkpoint hash are under `/data/fast/dflash2_qwen3_8_27b_regen_smoke1000_16k/` (`vllm_command.txt`, `vllm.patch`, and `checkpoint_sha256.txt`).

- Trackio DB: `/data/fast/dflash2_qwen3_8_27b_regen_smoke1000_16k/trackio/speculators.db`. Successful run name: `dflash2-qwen3-8-27b-regen-2026-10-01T19:20:12+00:00`.

- Log: `/data/fast/dflash2_qwen3_8_27b_regen_smoke1000_16k.log`.

The log is append-only and also contains an earlier failed attempt at the same output path (the 3,072-anchor OOM). When reviewing it, distinguish that earlier traceback from the later successful retry and its exit code 0.

## Starting, resuming, and viewing the full run

See [`local/cheatsheet/dflash2_qwen3_8_27b_regen_online.md`](../../cheatsheet/dflash2_qwen3_8_27b_regen_online.md) for the screen commands. It uses `/data/fast/dflash2_qwen3_8_27b_regen` for the full run and explains how to resume with the same output directory. Before launching, inspect that directory for an existing process/checkpoint so a second trainer is not started beside an active one.

For the successful smoke dashboard, activate `.venv` and run:

```bash
export TRACKIO_DIR=/data/fast/dflash2_qwen3_8_27b_regen_smoke1000_16k/trackio
trackio show --project speculators
```

For the full run, use the same command with `TRACKIO_DIR=/data/fast/dflash2_qwen3_8_27b_regen/trackio`.

## Working tree and related implementation changes

The branch changes for this run include:

- `examples/train/dflash2_qwen3_8_27b_regen_online.sh` — recipe settings above.
- `src/speculators/data_generation/preprocessing.py` — recognizes local `.jsonl.gz` files and directories containing them.
- `docs/cli/prepare_data.md` — documents `.jsonl.gz` input support.
- `local/` — this handoff and the screen/Trackio cheatsheet.

Keep the gzip loader/docs changes with this dataset workflow. Inspect the current diff and status before committing; do not assume all worktree changes belong to this handoff.

## Credential handling

The dataset required an access token during download. The file is already on disk at the path above. The token is deliberately not copied into this note, shell commands, or committed files. If redownloading becomes necessary, obtain the credential through the approved secret mechanism and pass it via the environment or Hugging Face login rather than embedding it in a script.
