# DFlash evaluation readiness

[Establish trustworthy evaluation records and resumable run identities](https://github.com/rahul-tuli/speculators/issues/106) is ready for the initial H100 DFlash acceptance phase. The bounded gate passed for both calibration recipes in FP8 and NVFP4, the original BF16-drafter reference, and target-only serving. This is infrastructure/serving evidence, not the full-budget calibration experiment or a performance comparison.

## Validated gate

Every arm served on one H100 with a BF16 target, evaluation seed 0, greedy decoding, seven speculative tokens where applicable, normal EOS, and a 4096-token output cap. The common context limit was 20,749. Prefix caching was disabled. These pilots used eager execution; compiled speed and higher-load gates remain separate.

| Arm | Selected quantized kernel | Modules with finite outputs observed | Longest prompt tokens received | Acceptance counters |
| --- | --- | ---: | ---: | --- |
| bf16-drafter | BF16 reference | 333 | 16,653 | validated |
| fp8-gaussian | CutlassFP8ScaledMMLinearKernel | 338 | 16,653 | validated |
| fp8-real | CutlassFP8ScaledMMLinearKernel | 338 | 16,653 | validated |
| nvfp4-gaussian | EmulationNvFp4LinearKernel | 338 | 16,653 | validated |
| nvfp4-real | EmulationNvFp4LinearKernel | 338 | 16,653 | validated |
| target-only | BF16 reference | 290 | 16,653 | not applicable |

All 36 exported quantized projections map to 21 runtime fused modules for each quantized drafter. Export/runtime partitioned shapes, ignored embedding/readout scope and effective NVFP4 fused scales were checked. NVFP4 uses W4A4 emulation with `use_a16=False`, explicitly selected by `linear_backend=emulation`; `VLLM_BATCH_INVARIANT=0` is shared across arms. The FP8 arms selected the native CUTLASS FP8 kernel. The finite-output probes were removed before warm-up and measurement.

The tested quantized checkpoints have **one 512-token calibration sequence**. They establish full-model serving compatibility for these artifacts, not the final recipe comparison. The experiment driver rejects these smoke budgets for the full matrix: subsequent checkpoints must use 2027 sequences, a 2048-token cap, recorded effective budgets and the complete global/per-row RNG policy. Every new checkpoint must pass the same inventory and bounded serving gate before its expensive evaluation.

## Correctness changes

- The original BF16 drafter and target-only reference are separate arms. Target-only never requires speculative counters and has null acceptance.
- Each workload/concurrency/repeat has its own raw request record, warm-up record, before/after counters, identity and completion hashes. Failed attempts remain on disk. Resume checks identities and all retained artifact hashes; a file's existence is insufficient.
- Measured requests must cover the exact frozen prompt multiset without drops, repetitions, transformations or failed requests. Warm-up runs outside both the measured timing and counter intervals.
- Counter records retain D, P, A and every A_i. Undefined ratios remain null. Whole-sweep acceptance is no longer copied into per-load rows.
- The launcher now supports target-only evaluation and every invocation supplies a provenance directory. Cached model IDs resolve to local snapshots for actual weight hashes. Runtime source hashes, patches, packages, GPU identities, flags and calibration manifests accompany runs.
- A port lock protects startup. The isolation guard rejected a conflicting developmental startup and that attempt has no valid completion. Pilot scheduling is serial; no competing GPU workload was permitted at measurement entry.
- Calibration now seeds Python, NumPy, Torch and CUDA global RNGs as well as the existing per-row generator, including forward anchor selection.

DFlash's original context-KV fusion attempted a quantized kernel before post-load scale initialization and could then bypass activation quantization through dense fused weights. Quantized context projection now calls each actual quantized QKV module at runtime. A regression reproduces the original load-order failure, checks activation-scale sensitivity, and keeps the dense BF16 fusion path covered. The full vLLM working-tree patch is in each server's provenance.

Validation passed: 68 evaluator, launcher, provenance and local inventory tests, plus 16 focused vLLM DFlash tests. A real completed point resumed with the server unavailable; a modified copy of its raw result was rejected. A competing port-lock holder was rejected before model startup.

The new evaluation code passes focused Ruff checks. Repository-wide publication hooks also inspect inherited local experiment scripts, which retain existing style failures. Whole-repository mypy is also blocked by installed `hs_connectors` typing and missing tqdm/PyYAML stubs; the new script import follows the repository’s explicit import-ignore convention. This evidence branch preserves the tested source snapshot; it is not a claim that the inherited scripts pass repository-wide lint. After recording these failures, the publication commit skips formatting/lint/typecheck hooks to retain the tested evidence snapshot; it is not a merge-ready PR.

## Frozen workload and limitations

The manifest contains **5488 examples in 61 files**: 880 qualitative and 1536 in each of throughput_1k, throughput_2k and throughput_8k. `throughput_32k` is excluded. NVIDIA preparation is pinned to `bcf059af55c20a89f797724598f9908d126153e6`; SPEED-Bench is `487aa718444e816458d1a0a52bfce7a454285cf4`. Cache-only adapters avoid the shared cache's quota without changing prompts. Prepared files and scripts are hashed.

The user explicitly selected **the original first user turn of every example**. Raw qualitative metadata has 120 multi-turn examples; materializing external sources expands this to 185. All 880 examples remain. This is not a full-conversation evaluation. A LaTeX `\documentclass{article}` string is valid benchmark content and was preserved.

The longest frozen prompt has 16,653 tokens. Its count was checked independently by rendering then tokenizing and matched the live server's input count for every arm. No context truncation or example exclusion was used.

Normalized exact matching uses NFKC, case folding and collapsed whitespace against the actual decoded calibration cache (2027 sequences, 2025 distinct user segments and 2027 distinct assistant segments). It found zero evaluation-prompt matches against either set. Calibration segments can be truncated; evaluation reference responses are unavailable. This does not establish semantic independence or absence of contamination. No overlap filtering was performed. The historical hidden-state capture still lacks original target-weight hashes; current hashes do not repair that gap.

## Warm-up, scheduling and remaining gates

Freeze two excluded warm-up requests per workload at concurrency 1, using the same output cap and decoding settings, before the measured counter interval. Both short and longest-context probes were exercised; per-request warm-up latencies and outputs counts are retained. Pilot wall times and numerical measurements describe eager smoke execution costs only. Full-budget checkpoint acceptance and heterogeneous category costs must be observed as the matrix is scheduled.

The runner supports three rotated measured rounds, distinct calibration seeds, and independent load points. Use the fixed manifest and verified completion records to batch the 61 workloads without omitting examples. Initial acceptance remains concurrency 1. All repeats, means/ranges and measured differences are descriptive; this work does not establish equivalence or negligible effects.

Compiled latency/throughput, concurrency 8/32/128 feasibility and their warm-up budgets, native Blackwell reruns with same-device references, and DSpark head exclusions/serving validation remain gates in their later execution phases. They do not block the initial DFlash acceptance path.

## Evidence and handoff

- [Reproducibility bundle](evaluation-pilot-evidence.tar.gz) and [SHA-256](evaluation-pilot-evidence.tar.gz.sha256).
- [Gate summary](gate-summary.json), with per-arm paths, kernel inventories, finite-output observations, actual token counts, counter totals and warm-up measurements.
- [Runner instructions](../../../local/drafter-quant/EVALUATION.md).

The bundle contains launch/eval commands, target/drafter hashes, runtime patches and source snapshots, packages/source inventories, raw counters, numerical per-request measurements, frozen dataset manifests, preparation recipe and inherited calibration provenance. Public request records omit prompt/output payloads and preserve their hashes because materialized benchmark text is not redistributed. Unmodified raw records remain at the recorded workspace paths; public derivatives must not be used as resume inputs.

The next map task is [Execute the acceptance comparisons and produce acceptance plots](https://github.com/rahul-tuli/speculators/issues/108). Use the [agreed plotting and CSV contract](https://github.com/rahul-tuli/speculators/issues/107#issuecomment-5761128803), deriving its tidy exports from the retained evaluator records, and regenerate full-budget seed-0 calibration exports before comparisons. Rerun the serving gate for those new checkpoints. No full matrix, training run, performance claim or experiment plot was produced by this readiness task.
