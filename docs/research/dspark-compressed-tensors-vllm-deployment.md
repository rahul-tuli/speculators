# DSpark compressed-tensors deployment in vLLM

Investigation for [Determine compressed-tensors DSpark deployment support in vLLM](https://github.com/rahul-tuli/speculators/issues/152). This examines the local vLLM checkout at `4ba787ec9959e566d3b587a796dff07673324da5` and the target model at Hugging Face revision `87ca2fdc67f316f6c1f9ebc7ef7bbb0ad299a4ce`. Source analysis and the config probe below do **not** establish end-to-end deployment of quantized checkpoints; no such checkpoint was generated or served here.

## Model and ordinary deployment

The [model card](https://huggingface.co/RedHatAI/Qwen3.8-27B-speculator.dspark/blob/87ca2fdc67f316f6c1f9ebc7ef7bbb0ad299a4ce/README.md) identifies a BF16 five-layer DSpark drafter, target `Qwen/Qwen3.8-27B`, eight auxiliary layers, a vanilla Markov head and an enabled confidence head. It provides an ordinary `vllm serve` example using `method: dspark`, the published draft model and seven speculative tokens. That is a reference for the original checkpoint, not evidence for quantized variants.

Its [config](https://huggingface.co/RedHatAI/Qwen3.8-27B-speculator.dspark/blob/87ca2fdc67f316f6c1f9ebc7ef7bbb0ad299a4ce/config.json) is speculators-format: it has a top-level `speculators_model_type` of `dspark`, and a nested `transformer_layer_config`. The original does not carry quantization metadata.

## Concrete metadata-preservation gap

[vLLM's speculators config adapter](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/transformers_utils/configs/speculators/base.py#L46-L63) starts the runtime config from `transformer_layer_config` and applies an algorithm-specific updater. The [DSpark updater](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/transformers_utils/configs/speculators/algos.py#L152-L211) copies selected top-level DSpark fields but not top-level `quantization_config`. Therefore a compressed-tensors config saved at the **outer** level is dropped along this conversion path. This is conditional on the produced quantized checkpoint actually placing its metadata there; that should be checked against the final example's saved artifact.

A direct probe using the pinned public config with a synthetic top-level metadata field returned `False` for preservation and `['Qwen3DSparkModel']` for architecture:

```python
import json
import urllib.request
from vllm.transformers_utils.configs.speculators.base import SpeculatorsConfig

url = "https://huggingface.co/RedHatAI/Qwen3.8-27B-speculator.dspark/resolve/87ca2fdc67f316f6c1f9ebc7ef7bbb0ad299a4ce/config.json"
with urllib.request.urlopen(url) as response:
    config = json.load(response)
config["quantization_config"] = {"quant_method": "compressed-tensors", "probe": "synthetic"}
result = SpeculatorsConfig.extract_transformers_pre_trained_config(config)
print("quantization_config" in result, result["architectures"])
```

This tests config extraction only: it does not construct a valid complete compressed-tensors config, load weights, select kernels, or demonstrate a real checkpoint failure. A candidate vLLM change would preserve the relevant metadata through adaptation, with a regression covering an actual serialized checkpoint configuration; the design and implementation remain to be decided.

## Existing quantized draft paths

[vLLM constructs a separate draft `ModelConfig`](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/config/speculative.py#L1395-L1428), and the [draft quantization helper](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/model_executor/models/utils.py#L985-L1008) resolves its quantization config from the draft. The [Qwen DFlash backbone used by DSpark](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/model_executor/models/qwen3_dflash.py#L420-L488) passes it into backbone decoder layers and the auxiliary hidden-state projection. [DSpark constructs its Markov head using that config](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/model_executor/models/qwen3_dspark.py#L192-L231); its separate `lm_head` is constructed without the draft quantization config at this revision. The requested recipes should ignore embeddings and Markov/confidence heads; this investigation does not independently test exclusions.

Compressed-tensors [selects its linear method from matching targets and ignores](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py#L157-L181) and [matches fused-module constituents](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/model_executor/layers/quantization/compressed_tensors/utils.py#L48-L120). The [draft name mapper](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/model_executor/models/utils.py#L1010-L1061) applies prefix and decoder-layer offset transformations explicitly to ModelOpt configs; other backends only get `exclude_modules` aliases when that attribute is a list. Compressed-tensors instead uses `ignore` and `target_scheme_map`. Consequently explicit path-based targets or ignore patterns merit investigation against actual emitted metadata and runtime names; source inspection alone does not establish whether the requested recipe's patterns fail.

There is an existing [synthetic mixed ModelOpt DSpark end-to-end test](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/tests/quantization/test_modelopt.py#L730-L855), including draft token generation. It uses ModelOpt metadata and is not evidence that compressed-tensors DSpark checkpoints work.

## Quantization and hardware

The compressed-tensors implementation includes [FP8 W8A8 and NVFP4 scheme dispatch](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py#L794-L804); the FP8 implementation [handles both block-structured and non-block weights](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py#L58-L86). Generic scheme availability does not prove this DSpark checkpoint loads or that every shape/kernel combination is supported.

The local GPU query reported one H100 (compute capability 9.0); no Blackwell GPU was visible at inspection. [NVFP4 kernel selection](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/model_executor/kernels/linear/__init__.py#L1191-L1309) depends on the platform and installed kernels; the [Marlin backend is W4A16 weight-only](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/model_executor/kernels/linear/nvfp4/marlin.py#L18-L50), while the [emulation backend](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/model_executor/kernels/linear/nvfp4/emulation.py#L14-L42) emulates NVFP4 and performs a BF16 matmul. Do not equate a successful H100 fallback with native W4A4 throughput. Native NVFP4 execution requires suitable hardware and a supported installed backend; record the actual selected kernel during a future smoke test.

## Open questions for the decision ticket

- What metadata does the example's actual saved output contain, and where? Preserve it in vLLM if it is outer-level and exercise the real configuration through the full loader.
- Do the emitted targets, packed QKV/gate-up shards and scale names map to DSpark runtime names? Verify compatibility using appropriate loading and serving tests, without adding separate exclusion checks to the requested validation.
- Which FP8 and NVFP4 deployment commands and GPU/backend combinations are proven by smoke tests? Include the supporting vLLM PR reference and disclose any hardware-blocked checks.
