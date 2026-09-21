"""Read-only checkpoint inventory and bounded linear-kernel checks; no server.

Run with the experiment's Python environment. Writes JSON to --output only.
Uses four pre-existing one-sample calibration exports, never modifies them.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

import torch
from safetensors import safe_open


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def digest(path):
    return hashlib.file_digest(open(path, "rb"), "sha256").hexdigest()


def inventory(directory):
    tensors, scales = {}, {}
    for path in sorted(directory.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                view = handle.get_slice(name)
                tensors[name] = {"shape": view.get_shape(), "dtype": view.get_dtype()}
                if "scale" in name:
                    value = handle.get_tensor(name).float()
                    tensors[name]["finite_positive"] = bool(
                        (torch.isfinite(value) & (value > 0)).all()
                    )
                    if value.numel() < 4:
                        scales[name] = value.tolist()
    config = json.loads((directory / "config.json").read_text())
    groups = []
    for i in range(5):
        for names in [
            [f"layers.{i}.self_attn.{p}_proj" for p in ("q", "k", "v")],
            [f"layers.{i}.mlp.{p}_proj" for p in ("gate", "up")],
        ]:
            for suffix in ("input_global_scale", "weight_global_scale", "input_scale", "weight_scale"):
                keys = [f"{name}.{suffix}" for name in names]
                if all(k in scales for k in keys):
                    values = [scales[k] for k in keys]
                    groups.append({"keys": keys, "values": values, "shared": all(v == values[0] for v in values)})
    modules = []
    for name, tensor in tensors.items():
        if name.endswith("weight_packed"):
            n, half_k = tensor["shape"]
            prefix = name.removesuffix("weight_packed")
            modules.append({"module": prefix[:-1], "N": n, "K": half_k * 2,
                            "group16_shape_valid": tensors[prefix + "weight_scale"]["shape"] == [n, half_k * 2 // 16] and half_k * 2 % 16 == 0})
        elif name.endswith("weight") and tensor["dtype"].startswith("F8"):
            n, k = tensor["shape"]
            modules.append({"module": name.removesuffix(".weight"), "N": n, "K": k,
                            "aligned16": n % 16 == k % 16 == 0})
    return {"directory": str(directory), "config_sha256": digest(directory / "config.json"),
            "quantization_config": config["quantization_config"], "tensors": tensors,
            "scalar_scales": scales, "fused_groups": groups, "quantized_modules": modules}


def load_module(directory, prefix):
    values = {}
    for path in directory.glob("*.safetensors"):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(prefix + "."):
                    values[key.removeprefix(prefix + ".")] = handle.get_tensor(key).cuda()
    return values


@torch.inference_mode()
def kernel_checks(root):
    from vllm.model_executor.kernels.linear import init_fp8_linear_kernel, init_nvfp4_linear_kernel
    from vllm.model_executor.kernels.linear.nvfp4.cutlass import CutlassNvFp4LinearKernel
    from vllm.model_executor.kernels.linear.nvfp4.emulation import EmulationNvFp4LinearKernel
    from vllm.model_executor.kernels.linear.nvfp4.marlin import MarlinNvFp4LinearKernel
    from vllm.model_executor.kernels.linear.scaled_mm.cutlass import CutlassFP8ScaledMMLinearKernel
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4 import CompressedTensorsW4A4Fp4
    from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8StaticTensorSym

    result = {"support": {cls.__name__: cls.is_supported() for cls in
                         (CutlassNvFp4LinearKernel, EmulationNvFp4LinearKernel, MarlinNvFp4LinearKernel, CutlassFP8ScaledMMLinearKernel)}}
    result["ambient_nvfp4_dispatch"] = type(init_nvfp4_linear_kernel()).__name__
    # Intercept only backend-config lookup, keeping the real registry and hardware checks.
    # Explicitly unset batch-invariant for these two branches; record ambient env separately.
    with patch.dict(os.environ, {"VLLM_BATCH_INVARIANT": "0"}):
        result["dispatch"] = {}
        for backend in ("auto", "emulation"):
            with patch("vllm.model_executor.kernels.linear._get_linear_backend", return_value=backend):
                result["dispatch"][backend] = type(init_nvfp4_linear_kernel()).__name__
        results = []
        for arm in ("random", "real"):
            for backend in ("auto", "emulation"):
                directory = root / f"nvfp4_w4a4-{arm}"
                values = load_module(directory, "layers.1.self_attn.k_proj")
                n, half_k = values["weight_packed"].shape
                with patch("vllm.model_executor.kernels.linear._get_linear_backend", return_value=backend):
                    scheme = CompressedTensorsW4A4Fp4()
                    layer = torch.nn.Module()
                    # Assemble the documented per-layer state directly: create_weights
                    # needs a distributed TP group, which this no-server probe avoids.
                    # This checks post-load processing/GEMM, not AutoWeightsLoader.
                    layer.logical_widths = [n]
                    layer.params_dtype = torch.bfloat16
                    layer.input_size_per_partition = half_k * 2
                    layer.output_size_per_partition = n
                    for name, value in values.items():
                        layer.register_parameter(name, torch.nn.Parameter(value, requires_grad=False))
                    scheme.process_weights_after_loading(layer)
                    torch.manual_seed(103)
                    x = torch.randn(2, half_k * 2, device="cuda", dtype=torch.bfloat16)
                    y = scheme.apply_weights(layer, x)
                    # Compare own K scale with the fused-QKV max that serving uses.
                    with safe_open(next(directory.glob("*.safetensors")), framework="pt", device="cpu") as handle:
                        fused_scale = max(float(handle.get_tensor(f"layers.1.self_attn.{p}_proj.input_global_scale").item()) for p in ("q", "k", "v"))
                    own_scale = float(layer.input_global_scale_inv.item())
                    layer.input_global_scale_inv.data.fill_(fused_scale)
                    fused_y = scheme.apply_weights(layer, x)
                    layer.input_global_scale_inv.data.fill_(own_scale * 100)
                    perturbed_y = scheme.apply_weights(layer, x)
                    torch.cuda.synchronize()
                    results.append({"arm": arm, "backend": type(scheme.kernel).__name__, "module": "layers.1.self_attn.k_proj",
                                    "input_shape": list(x.shape), "output_shape": list(y.shape), "finite": bool(torch.isfinite(y).all()),
                                    "fused_scale_finite": bool(torch.isfinite(fused_y).all()), "perturbed_scale_finite": bool(torch.isfinite(perturbed_y).all()),
                                    "own_input_global_scale_inv": own_scale, "fused_input_global_scale_inv": fused_scale,
                                    "fused_scale_output_max_abs_delta": float((y.float() - fused_y.float()).abs().max()),
                                    "scale_times_100_output_max_abs_delta": float((y.float() - perturbed_y.float()).abs().max())})
        result["nvfp4_single_layer"] = results
        result["fp8_single_layer"] = []
        for arm in ("random", "real"):
            values = load_module(root / f"fp8_w8a8-{arm}", "layers.1.self_attn.k_proj")
            n, k = values["weight"].shape
            kernel = init_fp8_linear_kernel(activation_quant_key=kFp8StaticTensorSym, weight_quant_key=kFp8StaticTensorSym,
                                           input_dtype=torch.bfloat16, out_dtype=torch.bfloat16, weight_shape=(n, k))
            layer = torch.nn.Module()
            for name, value in values.items():
                # The serving allocator creates FP32 scales and copies the saved
                # BF16 scale values into them. Preserve that dtype conversion.
                layer.register_parameter(name, torch.nn.Parameter(value.t() if name == "weight" else value.float(), requires_grad=False))
            kernel.process_weights_after_loading(layer)
            torch.manual_seed(103)
            x = torch.randn(2, k, device="cuda", dtype=torch.bfloat16)
            y = kernel.apply_weights(layer, x)
            torch.cuda.synchronize()
            result["fp8_single_layer"].append({"arm": arm, "backend": type(kernel).__name__, "module": "layers.1.self_attn.k_proj",
                                                "input_shape": list(x.shape), "output_shape": list(y.shape), "finite": bool(torch.isfinite(y).all())})
    with patch.dict(os.environ, {"VLLM_BATCH_INVARIANT": "1"}):
        result["batch_invariant_nvfp4_dispatch"] = type(init_nvfp4_linear_kernel()).__name__
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("/data/fast/drafter-quant/validation/calibration-input-20260921"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import vllm
    result = {"utc": command("date", "-u", "+%FT%TZ"), "torch": torch.__version__, "cuda": torch.version.cuda,
              "packages": {p: importlib.metadata.version(p) for p in ("vllm", "compressed-tensors", "llmcompressor", "safetensors")},
              "python": command("which", "python"), "vllm_path": vllm.__file__,
              "gpu": command("nvidia-smi", "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"),
              "env": {k: v for k, v in os.environ.items() if k in
                      {"VLLM_BATCH_INVARIANT", "VLLM_DISABLED_KERNELS", "CUDA_VISIBLE_DEVICES"}
                      or k.startswith(("VLLM_USE_", "VLLM_ENABLE_", "VLLM_MARLIN_", "VLLM_TEST_FORCE_"))}}
    result["checkpoints"] = {d.name: inventory(d) for d in sorted(args.checkpoint_root.iterdir()) if (d / "config.json").exists()}
    # Save inventory even if a runtime check raises; the log captures the exception.
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        result["kernels"] = kernel_checks(args.checkpoint_root)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
