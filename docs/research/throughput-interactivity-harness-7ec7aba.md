---
title: "Pinned throughput_interactivity harness behavior at 7ec7aba"
tags:
  - GuideLLM
  - vLLM
---

# Pinned throughput_interactivity harness behavior at 7ec7aba

## Context

This note resolves [Wayfinder ticket #132](https://github.com/rahul-tuli/speculators/issues/132) for the [auxiliary hidden-state ablation map, issue #129](https://github.com/rahul-tuli/speculators/issues/129). It records the behavior of `scripts/evaluate/throughput_interactivity.py` at Speculators commit `7ec7aba9450b533334e8f7992baf4e93be3c1a57`, not the behavior of a later version.

The exact first-party source was recovered from the Git object with:

```bash
git show 7ec7aba9450b533334e8f7992baf4e93be3c1a57:scripts/evaluate/throughput_interactivity.py
```

[View the pinned source](https://github.com/rahul-tuli/speculators/blob/7ec7aba9450b533334e8f7992baf4e93be3c1a57/scripts/evaluate/throughput_interactivity.py).

## CLI

The script has four subcommands:

- `collect` runs GuideLLM, samples vLLM `/metrics`, writes raw output and a CSV, then validates the sweep.
- `parse JSON_DIR --label LABEL --csv CSV` rebuilds a CSV from JSON files.
- `validate CSV` prints per-point summaries and warnings.
- `plot --series CSV[:NAME[:#COLOR]] --out FILE` draws one or more CSV series. Optional `--x itl|little` (default `itl`), `--title`, `--subtitle`, and `--note` control the chart labels.

For `collect`, required options are `--target`, `--model`, `--streams`, `--subset`, `--max-tokens`, `--out-dir`, and `--label`. The server base URL must also serve `/metrics`; a trailing `/v1` is stripped. Optional controls are:

- `--tokenizer` (default: `--model`); repeatable `--extra-body KEY=JSON` request-body fields.
- `--dataset` (default: `RedHatAI/speculator_benchmarks`), `--prompt-column` (default: `prompt`), `--dataset-repeat` (default: `0`, automatic sizing), and `--guidellm-bin` (default: `guidellm`).
- `--repeats` (default: 3), `--max-seconds` (default: 100), and `--warmup-seconds` (default: 30).
- `--csv` (default: `<out-dir>/<label>.csv`), `--dry-run`, `--overwrite`, and `--keep-going`.

The request backend is OpenAI-compatible `/v1/chat/completions`. If `--extra-body` is omitted, the server's sampling defaults apply. The harness requires vLLM metrics and refuses to collect if prefix caching is enabled, because prompts are repeated. [CLI declarations and collection setup](https://github.com/rahul-tuli/speculators/blob/7ec7aba9450b533334e8f7992baf4e93be3c1a57/scripts/evaluate/throughput_interactivity.py#L720-L815) [subcommand and option declarations](https://github.com/rahul-tuli/speculators/blob/7ec7aba9450b533334e8f7992baf4e93be3c1a57/scripts/evaluate/throughput_interactivity.py#L1202-L1343).

## math_reasoning data and coverage

With the defaults, `--subset math_reasoning` resolves to `RedHatAI/speculator_benchmarks/math_reasoning.jsonl`. The loader supports a local JSONL file or directory, otherwise fetches `https://huggingface.co/datasets/<dataset>/resolve/main/<subset>.jsonl`. It parses every nonblank JSONL line; it does not sample rows or apply a split or row cap. The default prompt field is `prompt`, and each row is materialized as a prompt-only record.

The URL explicitly resolves the Hub's mutable `main` revision. The harness does not pin a dataset commit, hash, or fixed `math_reasoning` row count. It prints the loaded row count at runtime. Thus “full subset” means all records in the JSONL fetched for that invocation are included in the materialized data; it does not promise that every timed GuideLLM point consumes every materialized prompt. If a point exhausts its input before its time limit, validation reports `requests_exhausted`.

When `--dataset-repeat <= 0`, the harness chooses `K = ceil(max(5000, 20 * max(--streams)) / source_row_count)` (at least one), duplicates all rows K times, then shuffles with `random.Random(0)`. The resulting file is `<out-dir>/_data/math_reasoning_x<K>.jsonl`. An explicit positive `--dataset-repeat` supplies K directly. [Dataset source, row handling, sizing, and materialization](https://github.com/rahul-tuli/speculators/blob/7ec7aba9450b533334e8f7992baf4e93be3c1a57/scripts/evaluate/throughput_interactivity.py#L558-L610). The example at the same commit lists `math_reasoning` among the benchmark subsets it sweeps. [Example subset and workload settings](https://github.com/rahul-tuli/speculators/blob/7ec7aba9450b533334e8f7992baf4e93be3c1a57/examples/evaluate/example_qwen3_8_27b_dspark_throughput_interactivity.sh#L35-L60).

## Workload and load controls

Each `--streams N` value creates a closed-loop concurrent point: N requests are kept in flight, and GuideLLM starts the next request when one completes. The GuideLLM duration constraint is `max-seconds + warmup-seconds`; its warmup value is excluded from reported measurements. Defaults are a 30-second warmup plus a 100-second measurement window. The script itself does not set vLLM scheduler limits such as `max-num-seqs` or `max-num-batched-tokens`; those are server-launch settings. [GuideLLM command construction and sweep loop](https://github.com/rahul-tuli/speculators/blob/7ec7aba9450b533334e8f7992baf4e93be3c1a57/scripts/evaluate/throughput_interactivity.py#L771-L815).

## Metrics and output schema

The main throughput value is `aggregate_output_tps`: output tokens generated during the measurement window divided by its duration, prorating requests that overlap the window. Mean active concurrency is total request-in-flight time clipped to the window divided by its duration. The default InferenceX x-axis is `interactivity_itl_tps_user = 1000 / mean_itl_ms`, using successful requests completed within the window. The alternate `interactivity_tps_user` is aggregate output throughput divided by mean active concurrency. Request-level mean/median/p99 metrics are calculated from successful requests completed inside the measurement window. [Measurement-window math](https://github.com/rahul-tuli/speculators/blob/7ec7aba9450b533334e8f7992baf4e93be3c1a57/scripts/evaluate/throughput_interactivity.py#L153-L230).

The CSV header is fixed and ordered as follows:

```text
model,point,streams,repeat,source_json,measured_duration_s,successful_requests,errored_requests,incomplete_requests,measured_requests,completed_rps,aggregate_output_tps,mean_active_concurrency,interactivity_tps_user,mean_itl_ms,interactivity_itl_tps_user,mean_output_tokens,mean_prompt_tokens,mean_output_tokens_per_iteration,median_ttft_ms,p99_ttft_ms,median_itl_ms,acceptance_length,num_drafts,num_accepted_tokens,mean_running_requests,mean_waiting_requests,mean_kv_cache_usage,stop_reason
```

The raw GuideLLM file is `conc<N>_r<R>.json`; its sibling `conc<N>_r<R>.metrics.json` contains `metrics_url`, `window`, `before`, `after`, `delta`, `gauges`, `whole_run`, `samples`, and the sampled `series`. The window delta records draft counts, accepted token counts, acceptance length `1 + accepted_tokens / num_drafts`, per-position acceptance, and prefix-cache query movement. Gauge summaries contain mean, max, and sample count for running requests, waiting requests, and KV-cache use. The CSV carries acceptance length and counts plus gauge means, but not per-position acceptance. [CSV columns and sidecar fields](https://github.com/rahul-tuli/speculators/blob/7ec7aba9450b533334e8f7992baf4e93be3c1a57/scripts/evaluate/throughput_interactivity.py#L76-L108) [sidecar sampling and acceptance schema](https://github.com/rahul-tuli/speculators/blob/7ec7aba9450b533334e8f7992baf4e93be3c1a57/scripts/evaluate/throughput_interactivity.py#L312-L490).

Collection appends `bench_command.txt` with a timestamp, full invocation, cwd, Python and package versions, git SHA, and server `/version` and `/v1/models` responses. [Provenance writer](https://github.com/rahul-tuli/speculators/blob/7ec7aba9450b533334e8f7992baf4e93be3c1a57/scripts/evaluate/throughput_interactivity.py#L644-L682).

## Repetition and validation

There are three runs per stream count by default. Files are named `conc<N>_r<R>.json`; without `--overwrite`, existing point files are skipped, allowing a sweep to resume in the same output directory. Each collect invocation materializes one shared prompt file and uses seed 0 for its shuffle, so repeats within that invocation use the same prompt order. The CSV includes one row per parsed benchmark, with the repeat inferred from the filename; validation and plotting average repeat metrics by point. `validate` flags a dataset-exhausted point, a steady-state mismatch above 10%, throughput spread above 5% across repeats, or possible queueing based on median TTFT and the vLLM waiting gauge. [Repeat loop and resume behavior](https://github.com/rahul-tuli/speculators/blob/7ec7aba9450b533334e8f7992baf4e93be3c1a57/scripts/evaluate/throughput_interactivity.py#L749-L815) [repeat aggregation and validation checks](https://github.com/rahul-tuli/speculators/blob/7ec7aba9450b533334e8f7992baf4e93be3c1a57/scripts/evaluate/throughput_interactivity.py#L858-L1027).
