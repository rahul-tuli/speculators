# BF16 DSpark depth-ladder benchmark

Work item: [Measure the BF16 DSpark depth ladder and find the bottleneck](https://github.com/rahul-tuli/speculators/issues/125)

Run date: 2026-10-08 UTC

## Findings

The 5-layer drafter is the comparison baseline at **1,449.0 output tokens/s**, the median of five 1,000-request runs. The 10L, 15L, and 20L medians are within 1.3% of that reference, so treat them as a shallow-depth plateau rather than a strict ranking. 25L is 0.4% below 5L. The median drops 2.9% at 30L, 4.8% at 35L, and 14.3% at 60L.

The separate target-only control median is 973.9 tok/s from three 1,000-request runs. Even the deepest measured arm, 60L at 1,241.1 tok/s, remains 27.4% faster than target-only. The issue's target-only break-even point was not reached through 60L; 5L is the comparison baseline for the requested table.

With the synthetic acceptance profile held constant, average observed acceptance length was 4.330 tokens across the ladder. Meanwhile, median p90 TPOT increased from 11.58 ms at 5L to 13.40 ms at 60L. This pattern is consistent with the extra draft computation becoming the bottleneck as depth grows. It is an inference from these measurements; the run does not profile individual draft and target kernels.

If a downstream choice allows at most a 5% throughput loss relative to 5L, 35L is the deepest tested variant within that budget (−4.8%). 40L is −6.5%. This is a decision threshold for comparison, not a threshold specified by the benchmark.

## Throughput and checkpoint size

Each throughput value is one timed run with 1,000 requests. The median and percentage change use the five completed runs per arm. The baseline is the five-run 5L median. Checkpoint size is the on-disk `model.safetensors` file in GiB; config and small metadata files are excluded.

| Drafter | Checkpoint size |         Output throughput by repeat (tok/s) | Five-run median | Change vs. 5L |
| ------: | --------------: | ------------------------------------------: | --------------: | ------------: |
|      5L |       3.704 GiB | 1,472.0, 1,478.8, 1,449.0, 1,439.5, 1,444.5 |     **1,449.0** |      baseline |
|     10L |       6.780 GiB | 1,468.4, 1,446.3, 1,373.7, 1,481.2, 1,481.8 |     **1,468.4** |         +1.3% |
|     15L |       9.856 GiB | 1,455.1, 1,424.8, 1,437.1, 1,458.6, 1,467.2 |     **1,455.1** |         +0.4% |
|     20L |      12.933 GiB | 1,451.4, 1,451.7, 1,374.4, 1,462.5, 1,461.4 |     **1,451.7** |         +0.2% |
|     25L |      16.009 GiB | 1,444.6, 1,446.7, 1,427.8, 1,443.9, 1,432.9 |     **1,443.9** |         −0.4% |
|     30L |      19.085 GiB | 1,412.8, 1,409.9, 1,406.1, 1,406.5, 1,325.2 |     **1,406.5** |         −2.9% |
|     35L |      22.161 GiB | 1,386.3, 1,387.6, 1,379.3, 1,347.8, 1,348.4 |     **1,379.3** |         −4.8% |
|     40L |      25.238 GiB | 1,356.6, 1,356.7, 1,355.2, 1,285.2, 1,305.6 |     **1,355.2** |         −6.5% |
|     45L |      28.314 GiB | 1,320.2, 1,328.8, 1,325.8, 1,295.9, 1,323.7 |     **1,323.7** |         −8.6% |
|     50L |      31.390 GiB | 1,322.7, 1,323.7, 1,319.7, 1,270.4, 1,301.7 |     **1,319.7** |         −8.9% |
|     55L |      34.467 GiB | 1,278.9, 1,286.8, 1,251.8, 1,255.9, 1,237.8 |     **1,255.9** |        −13.3% |
|     60L |      37.543 GiB | 1,241.1, 1,243.5, 1,253.2, 1,239.5, 1,238.0 |     **1,241.1** |        −14.3% |

All 60,000 timed requests across the 12 drafters completed with zero request failures. Five-run spreads vary by depth; medians are used to limit the effect of individual low or high runs. Small differences among 5L–25L should not be treated as a decisive ranking.

## Latency summary

Values are the median across the five run-level p90 measurements for each metric, in milliseconds. Each raw result also retains p50 and p99.

| Drafter | TTFT p90 | TPOT p90 | ITL p90 |
| ------: | -------: | -------: | ------: |
|      5L |    235.6 |    11.58 |   83.07 |
|     10L |    229.0 |    11.47 |   81.50 |
|     15L |    231.2 |    11.47 |   81.47 |
|     20L |    227.8 |    11.54 |   80.69 |
|     25L |    222.7 |    11.58 |   79.45 |
|     30L |    223.1 |    11.87 |   80.45 |
|     35L |    224.6 |    12.11 |   80.67 |
|     40L |    228.6 |    12.31 |   81.07 |
|     45L |    229.7 |    12.63 |   81.19 |
|     50L |    224.9 |    12.61 |   79.93 |
|     55L |    237.3 |    13.21 |   82.91 |
|     60L |    236.0 |    13.40 |   82.99 |

## Benchmark protocol

- Hardware: 2 × NVIDIA H100 80GB HBM3; driver 610.57.04.
- Software: vLLM `0.29.1.dev0+g98dff2a81.d20260917`, PyTorch `2.13.0+cu130`, Transformers `5.16.1`.
- Target: `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Drafters: BF16 `RedHatAI/Qwen3.8-27B-speculator.dspark`, source revision `7f33c272e5da240978e0d55767abab8193d74b95`; variants from 5L through 60L in +5-layer increments.
- Serving: tensor parallel size 2, BF16, maximum model length 4096, server seed 1234, `--max-num-seqs 16`, DSpark with 8 speculative tokens.
- Synthetic acceptance rates at positions 0–7: `[0.815, 0.646, 0.511, 0.404, 0.322, 0.259, 0.208, 0.170]` for every speculative arm.
- Client: `vllm bench serve`, random dataset, 1,000 prompts per timed repetition, seed 42, 512 input tokens, 128 output tokens, random range ratio 0, infinite request rate, maximum concurrency 16, 10 warm-up requests, EOS ignored. The tool also runs an initial single-prompt check before warmup. Results include TTFT, TPOT, and ITL percentiles at 50, 90, and 99.
- Benchmark history: an initial exploratory pass used 64 requests per timed repetition; those results are retained in the older artifact directories but are excluded from this report's estimates. The formal 1,000-request pass started with three timed repetitions per arm. Two more repetitions were then run for each DSpark depth, yielding five timed repetitions (5,000 successful requests) per depth. Repeats 1–3 ran on the first server launch for each arm; repeats 4–5 ran on a second server launch. Each repetition had its own 10 warmups. Arms ran sequentially. Treat the five timed repetitions as the repeat-level sample; the 5,000 requests within an arm are not 5,000 independent benchmark runs.
- The initial target-only control was measured separately at three repetitions of 1,000 prompts: 949.3, 973.9, and 974.8 tok/s, median 973.9 tok/s. It is context only; 5L is the comparison baseline in the table.
- No 1,000-request 65L result is included; the final requested ladder ends at 60L. The earlier 64-request exploratory pass included 65L, but that result is excluded from the final table.

The common server cap of 16 sequences was applied to every formal arm. An earlier 20L startup using vLLM's default `max_num_seqs=1024` exceeded the 979 available Mamba cache blocks; the cap allowed the 20L model to load and kept server settings consistent across arms.

## Checkpoints and evidence

The 5L model was read from the pinned Hugging Face snapshot. The 10L model is at `/data/fast/models/dspark-depth-ladder/`; 15L and 20L are under `docs/wayfinder-125-dspark-bf16-depth-ladder-20261008/checkpoints/`; generated 25L–60L variants are under `/tmp/wayfinder-issue-125/checkpoints/`. Depth variants were generated in +5 increments with `local/drafter-quant/dquant/build_depth_variants.py`. Model weights are large and are not copied into this report.

Per-run benchmark JSON, client logs, server logs, `run.json`, launch commands, vLLM provenance, target and drafter checkpoint hashes are in the [formal run artifacts](../../docs/wayfinder-125-dspark-bf16-depth-ladder-20261008/artifacts/num-prompts-1000-max-num-seqs-16/).

Each DSpark depth arm has five `*-rN.json` result files; target-only has three. Server launches recorded `--provenance-dir`; each arm's artifact directory includes `vllm_command.txt`, `vllm.patch`, and checkpoint hashes. The per-arm `run.json` records the exact client command for every repetition.
