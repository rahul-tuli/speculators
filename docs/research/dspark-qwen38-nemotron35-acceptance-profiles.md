---
title: "Acceptance profiles in the Qwen3.8 and Nemotron 3.5 Super VL DSpark cards"
tags:
  - DSpark
  - Qwen3
  - Nemotron 3.5
  - Speculative decoding
---

# Acceptance profiles in the Qwen3.8 and Nemotron 3.5 Super VL DSpark cards

## Summary

The [Qwen3.8 DSpark card](https://huggingface.co/RedHatAI/Qwen3.8-27B-speculator.dspark/blob/87ca2fdc67f316f6c1f9ebc7ef7bbb0ad299a4ce/README.md#acceptance-rates) reports one eight-position empirical acceptance-rate vector for each of nine workloads. Its `math_reasoning` row is `0.896, 0.799, 0.701, 0.612, 0.537, 0.459, 0.396, 0.338`, with mean acceptance length 5.74. The card does not state the sampling temperature for this table.

The matching five-layer Nemotron card is RedHatAI's [NVIDIA Nemotron 3.5 Super VL DSpark card](https://huggingface.co/RedHatAI/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B.dspark/blob/6ae7c251d475a6fc2ec2066b28bbef0e69ce236c/README.md#acceptance-rates). It reports eight-position vectors for nine workloads at both temperature 0 and temperature 1. For `math_reasoning`, the vectors are `0.949, 0.886, 0.810, 0.742, 0.675, 0.603, 0.543, 0.474` (mean length 6.68) at temperature 0, and `0.926, 0.843, 0.766, 0.693, 0.623, 0.561, 0.494, 0.435` (mean length 6.34) at temperature 1.

For an eight-position synthetic setting, these measured position rates are the best-supported profiles. Apply the relevant model-family profile to all four auxiliary-hidden-state variants; keep the Nemotron temperature explicit. The auxiliary-hidden-state counts (1, 5, 10, and 15) are a separate dimension from the speculative token positions. These profiles encode the trained five-layer card's measured acceptance behavior; they are not measured behavior of the random-weight stub variants.

## Context

This resolves [issue #130](https://github.com/rahul-tuli/speculators/issues/130) for the [auxiliary hidden-state ablation map, issue #129](https://github.com/rahul-tuli/speculators/issues/129).

The Qwen card describes a five-layer Qwen3-style drafter with a maximum draft length of eight and auxiliary target layers `4, 12, 20, 28, 36, 44, 52, 60`. Its acceptance section says evaluation used vLLM 0.29.0 and the dense `Qwen/Qwen3.8-27B` verifier. The same card's deployment examples set `num_speculative_tokens` to 7 even though its acceptance table has positions 0–7; the card does not explain this mismatch. The Nemotron card describes a five-layer Qwen3-style drafter with a maximum draft length of eight and auxiliary target layers `17, 37, 59, 79, 87`. Its acceptance section says evaluation used vLLM 0.31.0, the `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` verifier, and Spec-Bench; its serving example sets `num_speculative_tokens` to 8.

The Nemotron card is the five-layer Super VL speculator card. NVIDIA also publishes a separate Lightning 30B DSpark card; it is a different target/checkpoint and its evaluation table reports category mean acceptance lengths without position-wise values. Do not substitute that table for the Super VL profile.

## What the measurements mean

The cards' `Pos 0` through `Pos 7` columns are empirical per-position acceptance rates. They are not the confidence head's predictions. The rate at position *i* is the fraction of verification steps in which the draft prefix through position *i* was accepted; the rates are therefore marginal/unconditional prefix-acceptance probabilities. This matches vLLM's per-position metric, which divides accepted tokens at each position by the number of draft steps. In both cards, `1 + sum(Pos 0 ... Pos 7)` reproduces the reported acceptance length to rounding.

Acceptance length is a different statistic: the observed mean number of tokens emitted per verification step, including the one target-model bonus token. vLLM defines it as `1 + accepted draft tokens / verification steps`. For example, Qwen math reasoning has position rates summing to 4.738, so the card's mean length is 5.738, displayed as 5.74. These values describe observed runs, not deterministic lengths and not percentages.

Current vLLM source defines `synthetic_acceptance_rates` as an unconditional position vector: entry *i* is the marginal probability the first *i + 1* draft tokens are all accepted. The list length must equal `num_speculative_tokens`, entries must lie in `[0, 1]`, and they must be non-increasing. That is the right representation for the card vectors below. The separate scalar `synthetic_acceptance_length` is resolved to a minimum-variance, front-loaded vector; it preserves a mean but does not preserve the card's measured position profile. This source interpretation is pinned to vLLM commit `94febce7563ae2d67a062dc0f8ab408cd817f66c`; the companion controls research should confirm availability and command syntax in the ablation's pinned runtime.

## Qwen3.8: reported per-position rates

The card's Acceptance Rates section contains these values (positions are percentages). The condition is not labeled with a temperature in that section.

| Workload       | Mean acceptance length | Pos 0 | Pos 1 | Pos 2 | Pos 3 | Pos 4 | Pos 5 | Pos 6 | Pos 7 |
| -------------- | ---------------------: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: |
| HumanEval      |                   4.42 | 82.5% | 65.4% | 51.4% | 40.9% | 33.0% | 27.0% | 22.7% | 19.5% |
| math_reasoning |                   5.74 | 89.6% | 79.9% | 70.1% | 61.2% | 53.7% | 45.9% | 39.6% | 33.8% |
| qa             |                   4.22 | 79.0% | 62.0% | 49.1% | 39.1% | 31.3% | 25.2% | 19.9% | 15.9% |
| question       |                   3.80 | 78.0% | 58.2% | 43.0% | 31.9% | 24.1% | 18.6% | 14.6% | 11.6% |
| rag            |                   4.81 | 83.9% | 69.6% | 57.3% | 47.4% | 39.4% | 32.9% | 27.3% | 22.9% |
| summarization  |                   3.94 | 81.8% | 63.3% | 48.1% | 35.2% | 25.3% | 18.1% | 12.8% |  9.4% |
| tool_call      |                   3.76 | 77.9% | 57.6% | 42.4% | 31.4% | 23.6% | 18.0% | 14.1% | 11.2% |
| translation    |                   4.55 | 83.4% | 68.1% | 55.5% | 44.6% | 35.7% | 28.7% | 22.2% | 17.3% |
| writing        |                   3.77 | 77.7% | 57.6% | 42.6% | 31.7% | 23.9% | 18.4% | 14.4% | 11.4% |

The card separately reports mean acceptance length on MRCR 8-Needle with eight speculative tokens, by prompt length. These are means, not position vectors:

| Prompt length | Unique requests | Mean acceptance length |
| ------------- | --------------: | ---------------------: |
| 4K–8K         |              29 |                  4.745 |
| 8K–16K        |              24 |                  4.041 |
| 16K–32K       |              21 |                  4.627 |
| 32K–65K       |              25 |                  4.484 |
| 65K–131K      |              23 |                  4.345 |
| 131K–262K     |              26 |                  4.843 |
| 262K–524K     |              24 |                  4.671 |
| 524K–1M       |              38 |                  4.400 |
| 1M+           |              14 |                  4.297 |
| Overall       |             224 |                  4.492 |

## Nemotron 3.5 Super VL: reported per-position rates

The card reports both temperature conditions. Positions are percentages.

### Temperature 0

| Workload       | Mean acceptance length | Pos 0 | Pos 1 | Pos 2 | Pos 3 | Pos 4 | Pos 5 | Pos 6 | Pos 7 |
| -------------- | ---------------------: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: |
| HumanEval      |                   5.73 | 91.1% | 80.9% | 70.4% | 61.0% | 52.4% | 45.2% | 39.0% | 33.5% |
| math_reasoning |                   6.68 | 94.9% | 88.6% | 81.0% | 74.2% | 67.5% | 60.3% | 54.3% | 47.4% |
| qa             |                   3.63 | 79.4% | 59.1% | 42.5% | 30.4% | 20.8% | 14.2% |  9.9% |  6.7% |
| question       |                   3.96 | 80.7% | 61.7% | 45.7% | 34.3% | 26.3% | 19.9% | 15.2% | 12.0% |
| rag            |                   4.72 | 86.3% | 72.6% | 59.4% | 47.1% | 37.0% | 29.5% | 22.9% | 17.6% |
| summarization  |                   4.38 | 84.4% | 68.7% | 54.6% | 42.6% | 32.2% | 24.3% | 18.4% | 13.2% |
| tool_call      |                   4.71 | 85.1% | 69.8% | 56.3% | 45.3% | 36.8% | 30.3% | 25.7% | 21.8% |
| translation    |                   4.58 | 84.9% | 70.9% | 57.1% | 45.5% | 35.5% | 27.7% | 21.0% | 15.7% |
| writing        |                   3.92 | 80.0% | 61.0% | 44.8% | 33.8% | 25.9% | 19.6% | 15.1% | 11.9% |

### Temperature 1

| Workload       | Mean acceptance length | Pos 0 | Pos 1 | Pos 2 | Pos 3 | Pos 4 | Pos 5 | Pos 6 | Pos 7 |
| -------------- | ---------------------: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: |
| HumanEval      |                   5.32 | 87.8% | 75.6% | 64.6% | 54.7% | 46.6% | 39.6% | 34.0% | 28.8% |
| math_reasoning |                   6.34 | 92.6% | 84.3% | 76.6% | 69.3% | 62.3% | 56.1% | 49.4% | 43.5% |
| qa             |                   3.43 | 76.5% | 55.5% | 39.0% | 27.0% | 18.4% | 12.6% |  8.4% |  5.5% |
| question       |                   3.77 | 78.2% | 58.3% | 42.6% | 31.7% | 23.9% | 18.0% | 13.9% | 10.8% |
| rag            |                   4.37 | 83.6% | 67.3% | 53.3% | 41.3% | 32.5% | 25.0% | 19.2% | 14.6% |
| summarization  |                   4.15 | 82.3% | 65.8% | 51.0% | 38.7% | 28.9% | 21.0% | 15.7% | 11.1% |
| tool_call      |                   4.05 | 80.2% | 62.0% | 47.4% | 36.0% | 27.5% | 21.2% | 16.9% | 13.5% |
| translation    |                   4.35 | 83.3% | 67.8% | 53.3% | 41.7% | 32.1% | 24.5% | 18.3% | 13.5% |
| writing        |                   3.77 | 78.2% | 58.3% | 42.6% | 31.7% | 23.9% | 18.0% | 13.9% | 10.8% |

The card separately reports mean acceptance length on MRCR 8-Needle with eight speculative tokens, by prompt length. These are means, not position vectors:

| Prompt length | Unique requests | Mean acceptance length |
| ------------- | --------------: | ---------------------: |
| 4K–8K         |              97 |                  6.276 |
| 8K–16K        |              93 |                  5.754 |
| 16K–32K       |              99 |                  5.922 |
| 32K–65K       |              94 |                  5.817 |
| 65K–131K      |             108 |                  5.794 |
| 131K–262K     |              96 |                  6.018 |
| 262K–524K     |             102 |                  6.303 |
| 524K–1M       |              81 |                  4.082 |
| 1M+           |               9 |                  3.561 |
| Overall       |             779 |                  5.719 |

## Synthetic profiles for the ablation

For an eight-position synthetic profile, use these vectors for `math_reasoning`:

- Qwen3.8, with temperature unspecified in its card: `[0.896, 0.799, 0.701, 0.612, 0.537, 0.459, 0.396, 0.338]` (card mean acceptance length 5.74).
- Nemotron 3.5 Super VL at temperature 0: `[0.949, 0.886, 0.810, 0.742, 0.675, 0.603, 0.543, 0.474]` (card mean acceptance length 6.68).
- Nemotron 3.5 Super VL at temperature 1: `[0.926, 0.843, 0.766, 0.693, 0.623, 0.561, 0.494, 0.435]` (card mean acceptance length 6.34).

Reuse the selected family/temperature profile across that family's 1-, 5-, 10-, and 15-auxiliary-layer variants. That holds synthetic acceptance behavior constant while varying the hidden-state configuration. The Nemotron card supplies both sampling conditions; choose the one matching the ablation's serving temperature rather than averaging them. The Qwen card does not expose its table's temperature, so record that condition as unknown and do not claim the two families are temperature-matched.

Before translating the Qwen vector into a launch argument, resolve the card's 7-versus-8 mismatch against the pinned vLLM version: current vLLM requires `synthetic_acceptance_rates` to contain exactly `num_speculative_tokens` entries. Do not drop position 7 or change the draft budget silently. The companion controls research covers the pinned runtime and command behavior.

These profiles are specific to positions 0–7. The cards describe an eight-token maximum and provide no evidence for later positions. The number of auxiliary hidden-state layers does not add positions to the acceptance vector. If the planned draft budget differs from eight, only a prefix of these measurements is available for a shorter budget; a longer budget would require new evidence or an explicitly labeled extrapolation.

## Sources

- [Qwen3.8 DSpark model card, pinned revision](https://huggingface.co/RedHatAI/Qwen3.8-27B-speculator.dspark/blob/87ca2fdc67f316f6c1f9ebc7ef7bbb0ad299a4ce/README.md): Model Overview identifies the five-layer drafter and eight-token maximum; Acceptance Rates gives the nine workload rows; Long-Context Acceptance gives the MRCR request counts and means; Deployment contains the `num_speculative_tokens: 7` example.
- [Nemotron 3.5 Super VL DSpark model card, pinned revision](https://huggingface.co/RedHatAI/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B.dspark/blob/6ae7c251d475a6fc2ec2066b28bbef0e69ce236c/README.md): Model Overview identifies the five-layer drafter and eight-token maximum; Evaluation → Acceptance Rates gives both temperature tables; Evaluation → Long-Context Acceptance gives the MRCR request counts and means.
- [vLLM speculative config, pinned commit](https://github.com/vllm-project/vllm/blob/94febce7563ae2d67a062dc0f8ab408cd817f66c/vllm/config/speculative.py#L3289-L3413): synthetic rate semantics, validation, and scalar-to-minimum-variance conversion.
- [vLLM speculative metrics, pinned commit](https://github.com/vllm-project/vllm/blob/94febce7563ae2d67a062dc0f8ab408cd817f66c/vllm/v1/spec_decode/metrics.py#L904-L910): definitions of mean acceptance length and the per-position acceptance-rate vector.
