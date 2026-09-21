# Drafter quantization: hardware and kernel gate

Research for [Determine which hardware and kernels support defensible quantization measurements](https://github.com/rahul-tuli/speculators/issues/103), 2026-09-21. Scope: existing Gaussian versus real calibration, acceptance first, then latency and throughput. No model servers, full benchmarks, installs, or checkpoint changes were run.

## Decision

Use H100 for BF16/FP8 acceptance and subsequent native FP8 speed measurements after checkpoint smoke checks. **Do not accept H100's default NVFP4 dispatch as a W4A4 calibration experiment:** the installed vLLM selects Marlin W4A16, which omits activation quantization. Explicit NVFP4 emulation is a candidate acceptance-only path, subject to model-level numerical and loading validation. Reserve native NVFP4 speed comparisons for a compatible Blackwell device and compiled kernel; rerun all speed reference arms on that same device.

## Observed environment and revision drift

`nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader` reports two NVIDIA H100 80GB HBM3 devices, both SM90. A read-only Python support probe reports Torch `2.13.0+cu130`, CUDA 13.0, and installed vLLM `0.29.1.dev0+g98dff2a81.d20260917.precompiled`.

| Repository         | README pin   | Actual checkout HEAD                       |
| ------------------ | ------------ | ------------------------------------------ |
| speculators        | `40b12c9`    | `fd69f5326d695e98434196b3860956494f4c0004` |
| vllm               | `2e2fdaa99a` | `98dff2a81d747d1dba01a47f939f48c3526d4206` |
| compressed-tensors | `698967f`    | `a2a210e9001d6ac45b6f294f8a66a87384c7da55` |
| llm-compressor     | `ed55e6f67`  | `0349b68f4a7c5b0b535596df8e8d171911c8acee` |

Sources: local `git rev-parse HEAD`, package metadata, and `/workspace/speculators/local/drafter-quant/README.md`. The experiment files are untracked local inputs. vLLM has pre-existing edits to `qwen3_dflash.py`, `v1/worker/gpu/spec_decode/dflash/utils.py`, and `tests/v1/spec_decode/test_dflash_causality.py`; HEAD alone does not identify that runtime. Byte comparisons found installed kernel registry, NVFP4 emulation, Qwen3 DSpark, and Qwen3 DFlash files identical to their current checkout counterparts. Installed compressed-tensors/llmcompressor versions encode the actual commits above.

## Concrete support result

The probe imported kernel classes and called `is_supported()` plus `init_nvfp4_linear_kernel()` without allocating a model:

| Probe                                           | Result on H100                         |
| ----------------------------------------------- | -------------------------------------- |
| `CutlassNvFp4LinearKernel.is_supported()`       | false: CUTLASS FP4 kernels unavailable |
| `MarlinNvFp4LinearKernel.is_supported()`        | true                                   |
| `EmulationNvFp4LinearKernel.is_supported()`     | true                                   |
| `CutlassFP8ScaledMMLinearKernel.is_supported()` | true                                   |
| Default `init_nvfp4_linear_kernel()`            | `MarlinNvFp4LinearKernel`              |

At vLLM revision above, [`kernels/linear/__init__.py`](https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm/model_executor/kernels/linear/__init__.py) puts Marlin before emulation, including when `use_a16=False`. [`nvfp4/marlin.py`](https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm/model_executor/kernels/linear/nvfp4/marlin.py) calls a weight-only GEMM and does not use the calibrated activation global scale. Thus automatic fallback changes the experiment's numerical treatment, not just its speed.

[`nvfp4/emulation.py`](https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm/model_executor/kernels/linear/nvfp4/emulation.py) and [`nvfp4_emulation_utils.py:456`](https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py#L456) instead quantize/dequantize activations with their global scale, dequantize packed weights, and multiply in the input dtype. The registry supports `--linear-backend emulation`; verify selected kernels in the eventual launch because this is a serving-wide backend option. Source inspection establishes intended W4A4 semantics, **not** empirical equivalence to Blackwell or successful model serving. Label any resulting acceptance plots “NVFP4 W4A4 emulation, H100”; do not call its latency native NVFP4 performance.

Native CUTLASS NVFP4 requires CUDA runtime at least 12.8 and a compiled SM100/SM120 family kernel, per [`nvfp4_scaled_mm_entry.cu:71`](https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_entry.cu#L71). Blackwell access is not present in the observed two-device environment. NVIDIA's [NVFP4 architecture description](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) confirms native microscaled FP4 support on Blackwell. FP8 H100 hardware acceleration is documented in [vLLM's FP8 guide](https://github.com/vllm-project/vllm/blob/main/docs/features/quantization/llm_compressor/fp8.md); runtime kernel execution still needs the model smoke gate.

## Checkpoint and shape gates before either model enters plots

1. Freeze actual revisions and patches, exported checkpoint/config hashes, and runtime kernel names. The local quantizer selects compressed-tensors `FP8` (static per-tensor W8A8) or `NVFP4` (group-16 weights and locally dynamic group-16 activations with a calibrated global scale). Source: `/workspace/compressed-tensors/src/compressed_tensors/quantization/quant_scheme.py:180,346` at the recorded revision; `/workspace/speculators/local/drafter-quant/dquant/quantize.py:270`.
2. Validate each checkpoint's actual quantized module inventory and input/output dimensions. NVFP4's loader allocates packed weights at K/2 and scales at K/16, so require group-compatible input widths after partitioning. Native CUTLASS pads dimensions; current FP8 CUTLASS pads K/N to multiples of 16. Do not infer all checkpoint shapes from the target model name. Sources: `/workspace/vllm/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py:38` and `kernels/linear/scaled_mm/cutlass.py:156` at recorded revision.
3. Check fused QKV and gate/up global scales. NVFP4 loading takes the maximum across fused tensors and warns about differing scales; llm-compressor calls `fuse_weight_observers(model)`, but success for these custom model classes must be checked in the saved artifact. Sources: NVFP4 scheme file above, lines 96–138; `/workspace/llm-compressor/src/llmcompressor/modifiers/quantization/quantization/mixin.py:240` at recorded revision.
4. DFlash obtains the drafter's own quantization config and propagates it to FC, attention, and MLP projections; its loader uses `AutoWeightsLoader`. This is wiring support, not proof that either exported checkpoint loads. Sources: current local `/workspace/vllm/vllm/model_executor/models/qwen3_dflash.py:419,465,832` (dirty file) and `models/utils.py:929`.
5. **DSpark heads need a separate compatibility check.** Its confidence projection is constructed in FP32 without a quantization config; its selected-row Markov path directly indexes `markov_w2.weight` and uses `baddbmm_`, bypassing the quantized linear kernel. The training-side Markov projection is an `nn.Linear`, so the experiment's default `targets="Linear"` can quantize it. An all-Linear DSpark checkpoint cannot be presumed supported merely because its backbone is supported. Either explicitly exclude unsupported heads (record the partial quantization scope consistently across arms) or implement and validate compatible serving. Sources: `/workspace/vllm/vllm/model_executor/models/qwen3_dspark.py:89,113` at recorded revision; `/workspace/speculators/src/speculators/models/dspark/model_definitions.py:37`; local quantizer `--keep-heads-bf16` option.

Completion gate: per-model BF16 plus both calibrated FP8/NVFP4 checkpoints must load with expected tensors/scales, produce finite outputs under the intended activation quantization, and expose comparable acceptance counters. This investigation deliberately did not run those serving checks or measure acceptance. The hardware decision is settled; checkpoint integration remains an execution prerequisite.
