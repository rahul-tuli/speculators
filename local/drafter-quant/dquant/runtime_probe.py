"""Worker extension for bounded serving validation; hooks are removed before timing."""

import hashlib
import os

import torch


class RuntimeProbe:
    def quant_inventory(self):  # noqa: C901
        runner = self.model_runner
        models = {"target": runner.model}
        drafter = getattr(runner, "drafter", None) or getattr(
            runner, "speculator", None
        )
        if drafter is not None:
            models["drafter"] = drafter.model
        result = {
            "environment": {
                k: v
                for k, v in os.environ.items()
                if k.startswith(("VLLM_", "CUDA_VISIBLE_DEVICES"))
            },
            "models": {},
        }
        for label, model in models.items():
            modules = {}
            for name, module in model.named_modules():
                params = dict(module.named_parameters(recurse=False))
                if not params:
                    continue
                method = getattr(module, "quant_method", None)
                scheme = getattr(module, "scheme", None)
                kernels = {}
                for owner_name, owner in [
                    ("module", module),
                    ("method", method),
                    ("scheme", scheme),
                ]:
                    if owner is not None:
                        for attr, value in vars(owner).items():
                            if "kernel" in attr or attr in ("scheme", "fp8_linear"):
                                kernels[f"{owner_name}.{attr}"] = type(value).__name__
                info = {
                    "class": type(module).__name__,
                    "method": type(method).__name__,
                    "scheme": type(scheme).__name__,
                    "use_a16": getattr(scheme, "use_a16", None),
                    "kernels": kernels,
                    "parameters": {},
                }
                for pname, tensor in params.items():
                    t = tensor.detach().contiguous()
                    entry = {"shape": list(t.shape), "dtype": str(t.dtype)}
                    if "scale" in pname:
                        f = t.float()
                        entry.update(
                            finite=bool(torch.isfinite(f).all()),
                            min=float(f.min()),
                            max=float(f.max()),
                            sha256=hashlib.sha256(
                                t.reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
                            ).hexdigest(),
                        )
                        if t.numel() <= 64:  # noqa: PLR2004
                            entry["values"] = f.cpu().tolist()
                    info["parameters"][pname] = entry
                modules[name] = info
            result["models"][label] = modules
        return result

    def begin_finite_probe(self):
        self._quant_probe_results = {}
        self._quant_probe_handles = []
        runner = self.model_runner
        models = {"target": runner.model}
        drafter = getattr(runner, "drafter", None) or getattr(
            runner, "speculator", None
        )
        if drafter is not None:
            models["drafter"] = drafter.model
        for label, model in models.items():
            for name, module in model.named_modules():
                if not dict(module.named_parameters(recurse=False)):
                    continue
                key = f"{label}.{name}"

                def hook(_module, _inputs, output, key=key):
                    tensors = output if isinstance(output, (tuple, list)) else [output]
                    values = [t for t in tensors if isinstance(t, torch.Tensor)]
                    if values:
                        item = self._quant_probe_results.setdefault(
                            key, {"calls": 0, "finite": True}
                        )
                        item["calls"] += 1
                        item["finite"] &= all(
                            bool(torch.isfinite(t.float()).all()) for t in values
                        )

                self._quant_probe_handles.append(module.register_forward_hook(hook))
        return {"installed": len(self._quant_probe_handles)}

    def end_finite_probe(self):
        for handle in self._quant_probe_handles:
            handle.remove()
        self._quant_probe_handles = []
        return self._quant_probe_results
