# PROTOTYPE — DSpark synthetic acceptance and benchmark recipe

Status: draft for human review. Static configuration and CLI behavior have been checked; the live H100 smoke is pending.

## Question this prototype answers

Can the map's eight-position synthetic acceptance vector be passed to the DSpark server unchanged, and what fixed `vllm bench serve` workload should be used for the depth sweep?

## Findings from the checked-out code and model card

- The active `/workspace/vllm` checkout is commit `295ac4e52e8a35772b2a63c028f6510e221a65cf`; package metadata reports vLLM `0.29.0`.
- `SpeculativeConfig` accepts `rejection_sample_method="synthetic"` with exactly one of `synthetic_acceptance_rates` or `synthetic_acceptance_length`. A rates vector must contain `num_speculative_tokens` entries, each in `[0,1]`, and be non-increasing. Its entries are *unconditional* per-position rates: entry `i` means the probability that the first `i+1` proposed tokens are all accepted. The rejection sampler converts these rates to conditional rates before sampling.
- The RedHatAI model card's nine workload rows average, to three decimal places, to `[0.815, 0.646, 0.511, 0.404, 0.322, 0.259, 0.208, 0.170]`. The implied expected acceptance length is `1 + sum(rates) = 4.335`.
- `vllm bench serve` collects speculative counters before and after measured requests and saves acceptance length, accepted-token fraction, and per-position rates in its result JSON. Warmup requests run before the measurement interval.
- The model card's deployment example says `num_speculative_tokens: 7`, while its checkpoint config says `block_size: 8` and `speculative_tokens: 8`, and the acceptance table has eight positions. This prototype follows the map's eight-token control; the serving smoke should settle this documentation mismatch.

Sources: `/workspace/vllm/vllm/config/speculative.py`, `/workspace/vllm/vllm/v1/sample/rejection_sampler.py`, `/workspace/vllm/vllm/benchmarks/serve.py`, and the [RedHatAI model card](https://huggingface.co/RedHatAI/Qwen3.8-27B-speculator.dspark).

## Proposed fixed measurement workload

Use vLLM's deterministic `random` dataset to keep token counts and prompts identical across arms. It sends token-ID prompts through the OpenAI completions API; this is a compute-cost workload, not a quality or chat-template evaluation.

| Setting              | Proposed value                                                                                                    |
| -------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Dataset and endpoint | `random`, `/v1/completions`                                                                                       |
| Input / output       | 512 / 128 tokens, `--random-range-ratio 0.0`, `--ignore-eos`                                                      |
| Requests and seed    | 64 requests, seed 42                                                                                              |
| Load                 | `--request-rate inf`, `--max-concurrency 16`                                                                      |
| Warmup               | 10 requests before each timed repetition                                                                          |
| Repeats              | At least 3 timed invocations per arm                                                                              |
| Report               | Output-token throughput; TTFT, TPOT, and ITL at P50/P90/P99; speculative acceptance counters for speculative arms |

This gives each run 8,192 requested output tokens while keeping the request set small enough for the depth ladder. `--save-result` should write a separate JSON file for every arm and repetition. Reuse the exact arguments and seed for target-only, the original DSpark, each dummy depth, and the quantized arms.

## Proposed server commands

```bash
TARGET=Qwen/Qwen3.8-27B
TARGET_REV=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
DRAFTER=RedHatAI/Qwen3.8-27B-speculator.dspark
DRAFTER_REV=87ca2fdc67f316f6c1f9ebc7ef7bbb0ad299a4ce
OUT=/data/fast/drafter-quant/results/dspark-qwen3.8-27b-acceptance/phase2-control-prototype-20261008
```

Target-only (current workspace's `launch_vllm.py` supports omitting `--spec-model`):

```bash
/usr/bin/python scripts/launch_vllm.py eval "$TARGET" \
  --provenance-dir "$OUT/target-only" \
  -- --revision "$TARGET_REV" --tensor-parallel-size 2 \
     --max-model-len 4096 --dtype bfloat16 --port 8000 --seed 1234
```

Original DSpark with the synthetic control:

```bash
SPEC_CONFIG="{\"rejection_sample_method\":\"synthetic\",\"synthetic_acceptance_rates\":[0.815,0.646,0.511,0.404,0.322,0.259,0.208,0.170],\"revision\":\"$DRAFTER_REV\"}"
/usr/bin/python scripts/launch_vllm.py eval "$TARGET" \
  --spec-model "$DRAFTER" --spec-tokens 8 --spec-method dspark \
  --provenance-dir "$OUT/dspark-synthetic" \
  -- --revision "$TARGET_REV" --tensor-parallel-size 2 \
     --max-model-len 4096 --dtype bfloat16 --port 8000 --seed 1234 \
     --speculative-config "$SPEC_CONFIG"
```

The target-only command relies on a pending change in the main worktree that makes `--spec-model` optional for `launch_vllm.py eval`; I did not edit that file. Both commands include `--provenance-dir` as required by the repo guide. The model references are pinned to immutable Hugging Face commits.

## Proposed client command

Change `--label`, `--result-dir`, and `--result-filename` per arm/repetition; keep the workload arguments unchanged.

```bash
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

## Smoke-test status

Not run. At inspection time, another vLLM server was already launching the 10-layer depth variant on both H100s and then held about 35 GiB on each GPU. I stopped my target-only launch while it was still hashing provenance, before it loaded a model or used the GPUs. The existing server was left untouched. The scratch run directory is `/data/fast/drafter-quant/results/dspark-qwen3.8-27b-acceptance/phase2-control-prototype-20261008/`.

## Human review needed

Please react to these choices before this ticket is resolved:

1. Keep the proposed 512-in / 128-out random token-ID workload and `/v1/completions` endpoint for the depth-cost experiment, or use chat-formatted prompts instead?
2. Keep eight speculative tokens to match the checkpoint config, eight-position table, and fixed vector, despite the model card's deployment example showing seven?

The live two-arm smoke (target-only and original DSpark) still needs to run when the H100 pair is free. Its acceptance counters must show all eight positions and be broadly consistent with the configured profile; a small smoke is a configuration check, not evidence for a throughput conclusion.
