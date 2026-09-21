"""Validate exported quantized modules against the loaded DFlash inventory."""

from collections import defaultdict
from pathlib import Path

from safetensors import safe_open


def validate_export_inventory(checkpoint: Path, modules: dict) -> dict:  # noqa: C901
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
            if actual != expected and actual != expected[::-1]:
                raise ValueError(
                    f"Unexpected partitioned weight shape: {runtime}: "
                    f"{actual}, {expected}"
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
