"""Bounded-memory, fail-closed access to captured calibration inputs."""

from __future__ import annotations

import hashlib
import json
import random
import shlex
from pathlib import Path

import torch
from datasets import load_from_disk
from hs_connectors.transfer import FP8Transfer
from safetensors import safe_open

from speculators.train.data import ArrowDataset


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_identity(path):
    provenance = path / "vllm_command.txt"
    if not provenance.exists():
        raise ValueError(f"cache lacks capture identity: {provenance}")
    commands = [
        line
        for line in provenance.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    for command in reversed(commands):
        argv = shlex.split(command)
        if "serve" not in argv:
            continue
        target = argv[argv.index("serve") + 1]
        for flag in ("--speculative_config", "--speculative-config"):
            if flag in argv:
                config = json.loads(argv[argv.index(flag) + 1])
                layers = config["draft_model_config"]["hf_config"][
                    "eagle_aux_hidden_state_layer_ids"
                ]
                if len(layers) < 2:
                    raise ValueError(
                        "cache requires auxiliary and final hidden-state layers"
                    )
                return {
                    "target": target,
                    "captured_layers": layers,
                    "capture_command_sha256": sha256_file(provenance),
                }
    raise ValueError("cache provenance does not identify target and captured layers")


def inspect_cache(
    datapath, *, expected_target=None, expected_layers=None, expected_hidden_size=None
):
    """Inventory all rows using tensor headers, without decoding hidden states.

    The inventory digest covers file headers/stat information, not tensor payloads.
    Token equality and numerical validity are checked on every consumed sample.
    """
    path = Path(datapath).resolve()
    identity = capture_identity(path)
    if expected_target is not None and identity["target"] != expected_target:
        raise ValueError(
            f"target mismatch: cache={identity['target']}, model={expected_target}"
        )
    if expected_layers is not None and identity["captured_layers"][:-1] != list(
        expected_layers
    ):
        raise ValueError("auxiliary layer order mismatch between cache and drafter")
    data = load_from_disk(str(path)).with_format(None)
    if len(data) == 0:
        raise ValueError("cache has no prepared samples")
    inventory = hashlib.sha256()
    lengths, widths, dtypes = [], set(), set()
    for i, row in enumerate(data):
        length = len(row["input_ids"])
        if length <= 0 or row["seq_len"] != length or len(row["loss_mask"]) != length:
            raise ValueError(f"sample {i}: input length / seq_len / mask mismatch")
        if not all(v in (0, 1) for v in row["loss_mask"]):
            raise ValueError(f"sample {i}: loss mask must be binary")
        file = path / "hidden_states" / f"hs_{i}.safetensors"
        if not file.is_file() or Path(str(file) + ".lock").exists():
            raise ValueError(f"sample {i}: missing or unfinished cache file {file}")
        with safe_open(str(file), framework="pt") as handle:
            if not {"hidden_states", "token_ids"}.issubset(handle.keys()):
                raise ValueError(f"sample {i}: missing cache tensors")
            hs = handle.get_slice("hidden_states")
            shape, dtype = hs.get_shape(), hs.get_dtype()
            if len(shape) != 3 or shape[:2] != [
                length,
                len(identity["captured_layers"]),
            ]:
                raise ValueError(
                    f"sample {i}: cache hidden-state shape mismatch: {shape}"
                )
            if handle.get_slice("token_ids").get_shape() != [length]:
                raise ValueError(f"sample {i}: cache token shape mismatch")
            if dtype == "F8_E4M3":
                if "hidden_states_scales" not in handle.keys():
                    raise ValueError(f"sample {i}: FP8 cache lacks scales")
                if handle.get_slice("hidden_states_scales").get_shape() != [
                    length,
                    1,
                    1,
                ]:
                    raise ValueError(f"sample {i}: FP8 scale shape mismatch")
            elif (
                dtype not in ("BF16", "F16", "F32")
                or "hidden_states_scales" in handle.keys()
            ):
                raise ValueError(f"sample {i}: unsupported cache dtype/scales: {dtype}")
            headers = [
                (
                    key,
                    handle.get_slice(key).get_shape(),
                    handle.get_slice(key).get_dtype(),
                )
                for key in handle.keys()
            ]
        if expected_hidden_size is not None and shape[-1] != expected_hidden_size:
            raise ValueError(
                f"sample {i}: hidden size mismatch: {shape[-1]} != {expected_hidden_size}"
            )
        stat = file.stat()
        inventory.update(
            json.dumps([file.name, stat.st_size, stat.st_mtime_ns, headers]).encode()
        )
        widths.add(shape[-1])
        dtypes.add(dtype)
        lengths.append(length)
    if len(widths) != 1 or len(dtypes) != 1:
        raise ValueError("cache contains mixed hidden sizes or storage dtypes")
    return data, {
        **identity,
        "cache_path": str(path),
        "available_samples": len(data),
        "available_tokens": sum(lengths),
        "hidden_size": widths.pop(),
        "storage_dtype": dtypes.pop(),
        "cache_header_stat_sha256": inventory.hexdigest(),
        "dataset_sha256": {
            p.name: sha256_file(p) for p in sorted(path.glob("*.arrow"))
        },
    }


class CheckedFP8Transfer(FP8Transfer):
    def _dequantize(self, sample):
        if sample is not None:
            states = sample["hidden_states"]
            scales = sample.get("hidden_states_scales")
            if states.dtype == torch.float8_e4m3fn and scales is None:
                raise ValueError("FP8 cache lacks scales")
            if scales is not None and (
                not torch.isfinite(scales).all() or not (scales > 0).all()
            ):
                raise ValueError("cache scales must be positive and finite")
        return super()._dequantize(sample)


class RealBatchDataset:
    """Decode at most one sample per item; retain only Arrow data and row indices."""

    def __init__(self, datapath, seq_len, num_samples, seed, **expected):
        if seq_len <= 0 or num_samples <= 0:
            raise ValueError("seq_len and num_samples must be positive")
        data, self.manifest = inspect_cache(datapath, **expected)
        if num_samples > len(data):
            raise ValueError(
                f"requested {num_samples} calibration samples; available {len(data)}. "
                "Choose an explicit available budget; no silent shortfall."
            )
        self.indices = list(range(len(data)))
        random.Random(seed).shuffle(self.indices)
        self.indices = self.indices[:num_samples]
        self.seq_len = seq_len
        self.manifest.update(
            {
                "requested_samples": num_samples,
                "actual_samples": len(self.indices),
                "requested_token_capacity": num_samples * seq_len,
                "actual_tokens": sum(
                    min(data[i]["seq_len"], seq_len) for i in self.indices
                ),
                "loss_mask_tokens": sum(
                    sum(data[i]["loss_mask"][:seq_len]) for i in self.indices
                ),
                "selected_indices": self.indices,
                "seq_len": seq_len,
                "seed": seed,
                "decoded_dtype": "bfloat16",
            }
        )
        self.source = ArrowDataset(
            max_len=seq_len,
            datapath=datapath,
            transfer=CheckedFP8Transfer(Path(datapath) / "hidden_states"),
            on_missing="raise",
            train_ratio=1.0,
            split="train",
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_index = self.indices[idx]
        item = self.source[sample_index]
        if not isinstance(item, dict):
            raise ValueError(
                f"sample {sample_index}: cached tokens do not match prepared input"
            )
        t = min(item["input_ids"].shape[0], self.seq_len)
        batch = {
            key: item[key][:t].unsqueeze(0).long() for key in ("input_ids", "loss_mask")
        }
        for key in ("hidden_states", "verifier_last_hidden_states"):
            tensor = item[key][:t].to(torch.bfloat16)
            if not torch.isfinite(tensor).all():
                raise ValueError(f"sample {sample_index}: {key} must be finite")
            batch[key] = tensor.unsqueeze(0)
        batch["document_ids"] = torch.zeros(1, t, dtype=torch.long)
        return batch


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datapath")
    parser.add_argument("--target", required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    args = parser.parse_args()
    _, manifest = inspect_cache(
        args.datapath,
        expected_target=args.target,
        expected_layers=args.layers,
        expected_hidden_size=args.hidden_size,
    )
    print(json.dumps(manifest, indent=2))
