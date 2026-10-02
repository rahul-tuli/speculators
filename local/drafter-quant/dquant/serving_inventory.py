"""Validate exported quantized modules against the loaded DFlash inventory."""

from collections import defaultdict
from pathlib import Path

from safetensors import safe_open


def validate_export_inventory(
    checkpoint: Path, modules: dict, tensor_parallel_size: int = 1
) -> dict:
    tolerance = 1e-6
    groups = defaultdict(list)
    quantized = {name: m for name, m in modules.items() if m["scheme"] != "NoneType"}
    report = {"module_mapping": {}, "effective_scales": {}}
    with safe_open(
        checkpoint / "model.safetensors", framework="pt", device="cpu"
    ) as tensors:
        for key in tensors.keys():  # noqa: SIM118
            if not key.endswith(".weight_scale"):
                continue
            prefix = key.removesuffix(".weight_scale")
            runtime = "model." + prefix
            for name in ["q_proj", "k_proj", "v_proj"]:
                runtime = runtime.replace("." + name, ".qkv_proj")
            for name in ["gate_proj", "up_proj"]:
                runtime = runtime.replace("." + name, ".gate_up_proj")
            groups[runtime].append(prefix)
        if set(groups) != set(quantized):
            raise ValueError("Export/runtime quantized module inventory differs")
        for runtime, prefixes in groups.items():
            parameters = quantized[runtime]["parameters"]
            packed = prefixes[0] + ".weight_packed" in tensors.keys()  # noqa: SIM118
            suffix = ".weight_packed" if packed else ".weight"
            shapes = [
                tensors.get_slice(prefix + suffix).get_shape() for prefix in prefixes
            ]
            expected = [sum(shape[0] for shape in shapes), shapes[0][1]]
            actual = parameters["weight"]["shape"]
            module_class = quantized[runtime]["class"]
            shard_axis = {
                "RowParallelLinear": 1,
                "ColumnParallelLinear": 0,
                "MergedColumnParallelLinear": 0,
                "QKVParallelLinear": 0,
            }.get(module_class)
            local_expected = expected.copy()
            if shard_axis is not None and tensor_parallel_size > 1:
                if expected[shard_axis] % tensor_parallel_size:
                    raise ValueError(
                        f"Checkpoint shape is not divisible by tensor parallel size: "
                        f"{runtime}: {expected}, TP={tensor_parallel_size}"
                    )
                local_expected[shard_axis] //= tensor_parallel_size
            if actual != local_expected and actual != local_expected[::-1]:
                raise ValueError(
                    f"Unexpected partitioned weight shape: {runtime}: "
                    f"runtime={actual}, checkpoint={expected}, "
                    f"expected-local={local_expected}, TP={tensor_parallel_size}"
                )
            if packed:
                input_inverse = max(
                    float(tensors.get_tensor(prefix + ".input_global_scale").max())
                    for prefix in prefixes
                )
                weight_inverse = max(
                    float(tensors.get_tensor(prefix + ".weight_global_scale").max())
                    for prefix in prefixes
                )
                if (
                    abs(parameters["input_global_scale_inv"]["max"] / input_inverse - 1)
                    >= tolerance
                ):
                    raise ValueError(f"Unexpected effective input scale: {runtime}")
                if (
                    abs(parameters["weight_global_scale"]["max"] * weight_inverse - 1)
                    >= tolerance
                ):
                    raise ValueError(f"Unexpected effective weight scale: {runtime}")
                if quantized[runtime]["use_a16"] is not False:
                    raise ValueError(f"W4A4 activation quantization missing: {runtime}")
                report["effective_scales"][runtime] = {
                    "input_inverse": input_inverse,
                    "weight": 1 / weight_inverse,
                }
    for name, module in modules.items():
        if ("embed_tokens" in name or "lm_head" in name) and module[
            "scheme"
        ] != "NoneType":
            raise ValueError(f"Excluded module was quantized: {name}")
    report["module_mapping"] = dict(groups)
    report["export_quantized_modules"] = sum(map(len, groups.values()))
    report["runtime_quantized_modules"] = len(quantized)
    return report


def validate_dspark_head_scope(checkpoint: Path, modules: dict) -> dict:
    """Require BF16 exported heads and the vLLM-prescribed runtime head dtypes."""
    expected_prefixes = ("markov_head.", "confidence_head.")
    exported = {}
    with safe_open(
        checkpoint / "model.safetensors", framework="pt", device="cpu"
    ) as tensors:
        for key in tensors.keys():  # noqa: SIM118
            if key.startswith(expected_prefixes) and key.endswith((".weight", ".bias")):
                dtype = tensors.get_slice(key).get_dtype()
                exported[key] = dtype
                if dtype != "BF16":
                    raise ValueError(
                        f"DSpark head tensor must stay BF16: {key} ({dtype})"
                    )
            if key.startswith(expected_prefixes) and key.endswith(
                (
                    ".weight_scale",
                    ".weight_packed",
                    ".input_global_scale",
                    ".weight_zero_point",
                )
            ):
                raise ValueError(f"DSpark head must not have quantized tensors: {key}")

    for required in (
        "markov_head.markov_w1.weight",
        "markov_head.markov_w2.weight",
        "confidence_head.proj.weight",
    ):
        if required not in exported:
            raise ValueError(f"Required DSpark head tensor is absent: {required}")

    runtime = {
        name: module
        for name, module in modules.items()
        if any(part in name for part in ("markov_head", "confidence_head"))
    }
    if not runtime:
        raise ValueError("Loaded runtime inventory does not contain DSpark heads")
    for name, module in runtime.items():
        if module["scheme"] != "NoneType":
            raise ValueError(f"DSpark head is quantized at runtime: {name}")
        expected_dtype = (
            "torch.float32" if "confidence_head" in name else "torch.bfloat16"
        )
        if any(
            parameter["dtype"] != expected_dtype
            for parameter in module["parameters"].values()
        ):
            raise ValueError(
                f"Unexpected DSpark runtime head dtype: {name}; "
                f"expected {expected_dtype}"
            )

    return {
        "status": "passed",
        "checkpoint_head_tensors": exported,
        "runtime_head_modules": {
            name: {
                "class": module["class"],
                "scheme": module["scheme"],
                "expected_dtype": (
                    "torch.float32" if "confidence_head" in name else "torch.bfloat16"
                ),
                "parameters": module["parameters"],
            }
            for name, module in runtime.items()
        },
    }
