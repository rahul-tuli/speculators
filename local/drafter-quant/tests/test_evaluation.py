import sys  # noqa: INP001
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dquant.runtime_probe import RuntimeProbe
from dquant.serving_inventory import validate_export_inventory


def test_inventory_hashes_scalar_fused_scales():
    target = torch.nn.Linear(2, 2)
    drafter = torch.nn.Linear(2, 2)
    drafter.register_parameter(
        "input_global_scale", torch.nn.Parameter(torch.tensor(2.0))
    )
    probe = RuntimeProbe()
    probe.model_runner = SimpleNamespace(
        model=target, speculator=SimpleNamespace(model=drafter)
    )
    result = probe.quant_inventory()
    scale = result["models"]["drafter"][""]["parameters"]["input_global_scale"]
    assert scale["values"] == 2.0  # noqa: PLR2004, S101
    assert len(scale["sha256"]) == 64  # noqa: PLR2004, S101
    assert scale["finite"]  # noqa: S101


def test_export_inventory_checks_fused_shape_and_scale_reduction(tmp_path):
    tensors = {}
    for index, projection in enumerate(["q_proj", "k_proj", "v_proj"], 1):
        prefix = "layers.0.self_attn." + projection
        tensors[prefix + ".weight_packed"] = torch.ones(2, 2, dtype=torch.uint8)
        tensors[prefix + ".weight_scale"] = torch.ones(2, 1)
        tensors[prefix + ".input_global_scale"] = torch.tensor(float(index))
        tensors[prefix + ".weight_global_scale"] = torch.tensor(float(index * 2))
    save_file(tensors, tmp_path / "model.safetensors")
    module = {
        "scheme": "CompressedTensorsW4A4Fp4",
        "use_a16": False,
        "parameters": {
            "weight": {"shape": [6, 2]},
            "input_global_scale_inv": {"max": 3.0},
            "weight_global_scale": {"max": 1 / 6},
        },
    }
    modules = {"model.layers.0.self_attn.qkv_proj": module}
    report = validate_export_inventory(tmp_path, modules)
    assert report["export_quantized_modules"] == 3  # noqa: S101, PLR2004
    assert report["runtime_quantized_modules"] == 1  # noqa: S101
    module["parameters"]["input_global_scale_inv"]["max"] = 1.0
    with pytest.raises(ValueError, match="effective input scale"):
        validate_export_inventory(tmp_path, modules)
    module["parameters"]["input_global_scale_inv"]["max"] = 3.0
    module["parameters"]["weight"]["shape"] = [2, 2]
    with pytest.raises(ValueError, match="partitioned weight shape"):
        validate_export_inventory(tmp_path, modules)
