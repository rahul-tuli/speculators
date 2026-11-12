---
title: "Synthetic per-position DSpark acceptance controls in vLLM"
tags:
  - DSpark
  - vLLM
  - Speculative decoding
---

<div class="entry-byline" markdown>
[![rahul-tuli](https://github.com/rahul-tuli.png?size=64){ .entry-avatar }](https://github.com/rahul-tuli)
**[@rahul-tuli](https://github.com/rahul-tuli)** posted on **2026-11-12**
</div>

# Synthetic per-position DSpark acceptance controls in vLLM

Context: [research ticket #131](https://github.com/rahul-tuli/speculators/issues/131) · [ablation map #129](https://github.com/rahul-tuli/speculators/issues/129).

This note checks the requested serving controls against the vLLM checkout at revision `db9527a46873454610df6dbedf79a36d6bf1a7f6`.

## Summary

Use `rejection_sample_method: "synthetic"` inside `--speculative-config`, then provide exactly one of `synthetic_acceptance_rates` or `synthetic_acceptance_length`. For the per-position profiles in this ablation, use `synthetic_acceptance_rates`: vLLM interprets each entry as the unconditional probability that the prefix through that draft position is fully accepted. The list must have one non-increasing value in `[0, 1]` per speculative token.

vLLM converts those prefix probabilities to conditional probabilities and accepts each candidate randomly, without testing whether its token agrees with the target. A rejection emits a target token, and accepting the full draft adds the usual bonus token. These settings model acceptance behavior for performance experiments; they do not preserve target-model output correctness. [Config and validation](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/config/speculative.py#L530-L599), [rate conversion](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/spec_decode/utils.py#L848-L851), [synthetic acceptance/rejection](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py#L584-L595), [bonus token after full acceptance](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/sample/rejection_sampler.py#L774-L779).

## Acceptance profile semantics

For `n = num_speculative_tokens`, pass `n` values `p1, ..., pn`, where `pi = P(the first i draft tokens are all accepted)`. The per-position conditional probability used by the rejection loop is `ci = pi / p(i-1)`, with `p0 = 1`; once a token is rejected, verification stops. Therefore, if a source reports conditional per-position acceptance probabilities instead, convert them to cumulative products before putting them in `synthetic_acceptance_rates`.

For the requested `n = 8`, the configuration has this shape. Replace the eight placeholders with the profile established from each model card, and replace the draft identifier with the corresponding hs1/hs5/hs10/hs15 stub:

```json
{
  "method": "dspark",
  "model": "<drafter-stub>",
  "num_speculative_tokens": 8,
  "rejection_sample_method": "synthetic",
  "synthetic_acceptance_rates": [p1, p2, p3, p4, p5, p6, p7, p8]
}
```

The rate vector must be non-increasing, contain exactly eight numbers, and use values from zero to one. Set the same family profile on that family's four hidden-state variants so the drafter identifier is the variable under study. `synthetic_acceptance_length` is an alternative when only the mean is known: it accepts a value in `[1, n + 1]` and is expanded to a minimum-variance schedule. For example, length `4.4` with eight draft tokens resolves to `[1, 1, 1, 0.4, 0, 0, 0, 0]`. Because vLLM computes `length - 1` as the expected accepted draft-token count, the reported mean includes the one target replacement/bonus token. [Length conversion and validation](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/config/speculative.py#L556-L599).

## Corrected launch fragments

`--speculative-config` accepts the JSON object above. `--no-enable-prefix-caching` is the generated negative form of vLLM's `--enable-prefix-caching` Boolean option; use it explicitly so model defaults cannot turn caching on. [CLI option parsing](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/engine/arg_utils.py#L372-L380), [option registration](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/engine/arg_utils.py#L1338-L1344), [speculative JSON parsing](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/engine/arg_utils.py#L1788-L1794).

Qwen3.8-27B, using the hs5 drafter as an example:

```bash
vllm serve Qwen/Qwen3.8-27B \
  --tensor-parallel-size 1 \
  --no-enable-prefix-caching \
  --speculative-config '{"method":"dspark","model":"RelaxingSnorlax/Qwen3.8-27B-speculator.dspark-hs5","num_speculative_tokens":8,"rejection_sample_method":"synthetic","synthetic_acceptance_rates":[p1,p2,p3,p4,p5,p6,p7,p8]}' \
  --language-model-only
```

Nemotron 3.5, also using hs5 as an example:

```bash
VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random \
vllm serve nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 \
  --trust-remote-code \
  --limit-mm-per-prompt '{"video":1,"image":1}' \
  --video-pruning-rate 0.5 \
  --media-io-kwargs '{"video":{"fps":2,"num_frames":256}}' \
  --tensor-parallel-size 4 \
  --speculative-config '{"method":"dspark","model":"RelaxingSnorlax/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B.dspark-hs5","num_speculative_tokens":8,"draft_tensor_parallel_size":1,"draft_sample_method":"greedy","rejection_sample_method":"synthetic","synthetic_acceptance_rates":[p1,p2,p3,p4,p5,p6,p7,p8]}' \
  --block-size 64 \
  --compilation-config '{"max_cudagraph_capture_size":192}' \
  --enable-expert-parallel \
  --max-num-seqs 128 \
  --max-model-len 114688 \
  --max-num-batched-tokens 16384 \
  --no-enable-prefix-caching \
  --async-scheduling \
  --mamba-backend flashinfer \
  --mamba-ssm-cache-dtype float16 \
  --enable-mamba-cache-stochastic-rounding \
  --mamba-cache-philox-rounds 5 \
  --chat-template-content-format string \
  --mamba-cache-mode none
```

The `uniform_random` environment setting selects vLLM's routing simulator, whose strategy chooses expert IDs from random uniform scores. vLLM warns that simulated/custom routing strategies may not produce correct model outputs; retain this setting for the requested Nemotron benchmark but treat it as a serving-performance configuration, not a correctness run. [Environment variable](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/envs.py#L1793-L1800), [router selection](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/layers/fused_moe/router/router_factory.py#L70-L78), [uniform strategy](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/layers/fused_moe/router/routing_simulator_router.py#L122-L141).

With prefix caching disabled, omit `--prefix-match-unit 64`: vLLM defines it only as the matching granularity for prefix-cache hits. Set `--mamba-cache-mode none`; vLLM documents `none` as the mode when prefix caching is disabled and `align` as the prefix-caching mode. The physical `--block-size 64` remains in the launch because it controls KV-cache block size, separate from prefix-cache match granularity. [Prefix-match-unit meaning](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/config/cache.py#L91-L101), [Mamba cache modes](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/config/cache.py#L190-L197), [block-size definition and option](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/config/cache.py#L73-L75), [CLI flag](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/engine/arg_utils.py#L1322-L1325).

## Sources

- vLLM [`SpeculativeConfig`](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/config/speculative.py) and [speculative-decoding configuration docs](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/docs/features/speculative_decoding/README.md).
- vLLM rejection sampler [rate conversion](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/spec_decode/utils.py) and [GPU acceptance loop](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py).
- vLLM [CLI argument definitions](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/engine/arg_utils.py), [cache configuration](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/config/cache.py), and [MoE routing simulator](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/layers/fused_moe/router/routing_simulator_router.py).
