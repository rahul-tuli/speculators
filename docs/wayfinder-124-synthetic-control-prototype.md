# DSpark synthetic acceptance and benchmark recipe

Status: prototype complete. The user selected eight draft tokens and the proposed random token-ID workload. The two-arm H100 smoke completed on 2026-10-08.

## Question answered

Can the map's eight-position synthetic acceptance vector be passed to the DSpark server unchanged, and what fixed `vllm bench serve` workload should be used for the depth sweep?

## Findings

- The `/workspace/vllm` checkout is at `295ac4e52e8a35772b2a63c028f6510e221a65cf`. The installed server imported vLLM from `/usr/local/lib/python3.12/dist-packages/vllm`; its build string was `0.29.1.dev0+g98dff2a81.d20260917`, while Python package metadata and the launch provenance report `0.29.0`. The installed wheel's direct URL points to `/vllm-workspace/dist/vllm-0.29.0-cp38-abi3-linux_x86_64.whl`. The installed `config/speculative.py`, `v1/sample/rejection_sampler.py`, and `benchmarks/serve.py` were byte-identical to those in the checked-out source. Keep both the checkout SHA and runtime build string when recording this environment.
- `SpeculativeConfig` accepts `rejection_sample_method="synthetic"` with exactly one of `synthetic_acceptance_rates` or `synthetic_acceptance_length`. A rates vector must contain exactly `num_speculative_tokens` entries, each in `[0,1]` and non-increasing. Entries are unconditional per-position probabilities: entry `i` is the chance that the first `i+1` proposed tokens are all accepted. The rejection sampler converts these values to conditional rates before sampling.
- The nine workload rows in the RedHatAI model card average, to three decimal places, to `[0.815, 0.646, 0.511, 0.404, 0.322, 0.259, 0.208, 0.170]`. The implied expected acceptance length is `1 + sum(rates) = 4.335`.
- The model card's deployment example says `num_speculative_tokens: 7`, while its checkpoint config says `block_size: 8` and proposal `speculative_tokens: 8`, and its acceptance table has eight positions. The user chose eight tokens, matching the checkpoint and table.
- `vllm bench serve` reads speculative counters around the measured request interval and saves the acceptance length, accepted-token fraction, and per-position rates in its JSON result. Warmup requests are outside that interval.

Sources: `/workspace/vllm/vllm/config/speculative.py`, `/workspace/vllm/vllm/v1/sample/rejection_sampler.py`, `/workspace/vllm/vllm/benchmarks/serve.py`, and the [RedHatAI model card](https://huggingface.co/RedHatAI/Qwen3.8-27B-speculator.dspark).

## Fixed workload for the depth sweep

The user selected this workload. It uses vLLM's seeded `random` dataset to hold token-ID prompts and lengths constant across arms. It is a compute-cost workload, not a quality or chat-template evaluation.

| Setting                  | Value                                                                                        |
| ------------------------ | -------------------------------------------------------------------------------------------- |
| Dataset and endpoint     | `random`, `/v1/completions`                                                                  |
| Input and output         | 512 / 128 tokens, `--random-range-ratio 0.0`, `--ignore-eos`                                 |
| Requests and client seed | 64 requests, seed 42                                                                         |
| Server seed              | 1234                                                                                         |
| Load                     | `--request-rate inf`, `--max-concurrency 16`                                                 |
| Warmup                   | 10 requests before each timed invocation                                                     |
| Repeats                  | At least 3 timed invocations per arm                                                         |
| Report                   | Output-token throughput; TTFT, TPOT, and ITL at P50/P90/P99; speculative acceptance counters |
| Client thread limits     | `RAYON_NUM_THREADS=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`                               |

Each run requests 8,192 output tokens. The first client attempt without thread limits failed while Rayon created its thread pool (`EAGAIN`, resource temporarily unavailable). The same benchmark succeeded with the limits above. Keep the thread limits in the benchmark command. The smoke's first invocation verifies the recipe; its single timing is not a performance conclusion.

Keep each server process alive across its timed repetitions, and retain the 10 client warmup requests before each measured invocation. Use identical benchmark arguments and seed for target-only, the original DSpark, each dummy depth, and quantized arms. Give each arm and repetition a unique result filename.

## Pinned models and server commands

```bash
TARGET=Qwen/Qwen3.8-27B
TARGET_REV=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
DRAFTER=RedHatAI/Qwen3.8-27B-speculator.dspark
DRAFTER_REV=87ca2fdc67f316f6c1f9ebc7ef7bbb0ad299a4ce
OUT=/data/fast/drafter-quant/results/dspark-qwen3.8-27b-acceptance/phase2-control-prototype-20261008
```

Target-only:

```bash
/usr/bin/python scripts/launch_vllm.py eval "$TARGET" \
  --provenance-dir "$OUT/target-only" \
  -- --revision "$TARGET_REV" --tensor-parallel-size 2 \
     --max-model-len 4096 --dtype bfloat16 --port 8000 --seed 1234
```

Original DSpark with the selected synthetic profile:

```bash
SPEC_CONFIG="{\"rejection_sample_method\":\"synthetic\",\"synthetic_acceptance_rates\":[0.815,0.646,0.511,0.404,0.322,0.259,0.208,0.170],\"revision\":\"$DRAFTER_REV\"}"
/usr/bin/python scripts/launch_vllm.py eval "$TARGET" \
  --spec-model "$DRAFTER" --spec-tokens 8 --spec-method dspark \
  --provenance-dir "$OUT/dspark-synthetic" \
  -- --revision "$TARGET_REV" --tensor-parallel-size 2 \
     --max-model-len 4096 --dtype bfloat16 --port 8000 --seed 1234 \
     --speculative-config "$SPEC_CONFIG"
```

Both server commands passed `--provenance-dir`. The launch artifacts are in the run directory above. The model references are pinned to immutable Hugging Face commits. The `checkpoint_sha256.txt` files record that the model IDs were remote references; the pinned revisions supply the immutable model identity.

## Benchmark command

Use the same arguments for every arm and repetition; change only the result directory, label, and filename.

```bash
RAYON_NUM_THREADS=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
vllm bench serve \
  --host 127.0.0.1 --port 8000 \
  --backend openai --endpoint /v1/completions \
  --dataset-name random \
  --model Qwen/Qwen3.8-27B \
  --num-prompts 64 --seed 42 \
  --input-len 512 --output-len 128 --random-range-ratio 0.0 \
  --request-rate inf --max-concurrency 16 --num-warmups 10 \
  --ignore-eos \
  --percentile-metrics ttft,tpot,itl --metric-percentiles 50,90,99 \
  --save-result --result-dir "$OUT/target-only" \
  --label target-only --result-filename target-only-r1.json
```

## H100 smoke result

Hardware: 2×H100 80 GB, tensor parallelism 2. The target-only and DSpark servers used target revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; the DSpark server used drafter revision `87ca2fdc67f316f6c1f9ebc7ef7bbb0ad299a4ce`. Each arm completed all 64 requests with 0 failures, 32,768 input tokens, and 8,192 output tokens.

| Arm              | Output tokens/s | TTFT P50 / P90 / P99 (ms) | TPOT P50 / P90 / P99 (ms) | ITL P50 / P90 / P99 (ms) |
| ---------------- | --------------: | ------------------------- | ------------------------- | ------------------------ |
| Target-only      |          947.27 | 383.49 / 420.21 / 432.19  | 14.03 / 15.71 / 16.38     | 13.65 / 14.09 / 15.65    |
| DSpark synthetic |          974.79 | 246.38 / 454.85 / 456.65  | 10.71 / 27.90 / 31.05     | 23.35 / 88.18 / 107.34   |

The DSpark result reported acceptance length `4.33455` and these per-position acceptance rates:

```text
[0.81889, 0.64823, 0.50992, 0.39614, 0.31785, 0.26044, 0.20981, 0.17328]
```

All eight positions were present, and the measured rates and acceptance length were close to the configured `[0.815, 0.646, 0.511, 0.404, 0.322, 0.259, 0.208, 0.170]` profile and expected length `4.335`. This confirms the synthetic control reached the sampler without truncation. These single-run throughput and latency values are smoke outputs only; ticket #125 must use repeated measurements before making a performance claim.

vLLM logged that DSpark could not identify a draft KV-cache group for this multimodal target and disabled prefix-cache reuse. The fixed random completions workload does not reuse prompt prefixes, so this did not prevent the smoke.

The committed results and provenance are available alongside this report:

- [Target-only benchmark result](wayfinder-124-artifacts/target-only/target-only-smoke.json), [launch command](wayfinder-124-artifacts/target-only/vllm_command.txt), [vLLM patch record](wayfinder-124-artifacts/target-only/vllm.patch), and [checkpoint reference](wayfinder-124-artifacts/target-only/checkpoint_sha256.txt).
- [DSpark benchmark result](wayfinder-124-artifacts/dspark-synthetic/dspark-synthetic-smoke.json), [launch command](wayfinder-124-artifacts/dspark-synthetic/vllm_command.txt), [vLLM patch record](wayfinder-124-artifacts/dspark-synthetic/vllm.patch), [target checkpoint reference](wayfinder-124-artifacts/dspark-synthetic/checkpoint_sha256.txt), and [drafter checkpoint reference](wayfinder-124-artifacts/dspark-synthetic/drafter_checkpoint_sha256.txt).

The same files remain in the local run directory `/data/fast/drafter-quant/results/dspark-qwen3.8-27b-acceptance/phase2-control-prototype-20261008/`.

The initial unbounded benchmark client failed before sending requests; retrying with the listed thread limits succeeded. No depth or quantization arms were run in this prototype.
