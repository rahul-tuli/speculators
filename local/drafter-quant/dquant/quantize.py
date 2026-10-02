#!/usr/bin/env python3
"""Quantize DFlash/DSpark speculator drafters with llm-compressor.

Schemes (selected by --scheme):
  fp8_w8a8    FP8 static W8A8 (weights + activations)  -> oneshot + calibration data
  nvfp4_w4a4  NVFP4 W4A4 (fp4 group-16, LOCAL-dynamic  -> oneshot + calibration data
              activations w/ calibrated global scale)
  fp8_dynamic FP8 per-channel weights + dynamic         -> model_free_ptq (data-free,
              per-token activations (weight-focused)        no model load, no shims)
  fp8_block   FP8 128x128 weight blocks + dynamic       -> model_free_ptq (data-free)
              activation groups of 128
  nvfp4_gptq  NVFP4 W4A4 with GPTQ and expanded MSE     -> oneshot + calibration
  nvfp4_gptq_imatrix  As above, with E[x^2]-weighted    -> oneshot + calibration
              weight scale selection

The run manifest records the actual source repository revisions and the frozen
model checkpoint hash, so later library changes can be distinguished.

Why oneshot works for a drafter without framework changes:
  - DFlashDraftModel IS a transformers.PreTrainedModel (speculators/model.py:213)
  - oneshot(model=<instance>, dataset=<DataLoader>) skips AutoModel path loading
    (llm-compressor entrypoints/utils.py:61-63)
  - pipeline="basic" splats batch dicts straight into model(**batch)
    (pipelines/basic/pipeline.py:62-73) -- so target hidden states enter the real
    drafter forward as ordinary batch keys.

CALIBRATION DATA: --calib-data random feeds Gaussian tensors shaped like
drafter inputs. This is the measured random-control arm of the calibration
study. It tests how observers behave when their data distribution differs from
the real target hidden states. For the matched-data arm, capture target hidden
states with scripts/20_capture_hidden_states.sh and pass the prepared directory.
"""

# ruff: noqa: PLC0415
# Heavy model libraries are imported only on execution paths; --dry-run stays light.

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("dquant.quantize")

# ---------------------------------------------------------------------------
# Defaults (overridable via CLI)
# ---------------------------------------------------------------------------
DEFAULT_IGNORE = ["lm_head", "verifier_lm_head", "re:.*embed_tokens.*"]
# lm_head/verifier_lm_head are frozen verifier copies and are NOT caught by
# llm-compressor's disable_lm_head (speculators doesn't override
# get_output_embeddings -> warn-and-continue), so they must be ignored
# explicitly or they would be quantized. embed_tokens (also a frozen verifier
# copy) is nn.Embedding -- oneshot's targets="Linear" never matches it, but
# model_free_ptq quantizes any 2D weight tensor, so it MUST be ignored
# explicitly for the fp8_dynamic arm.

DATA_FREE_SCHEMES = frozenset({"fp8_dynamic", "fp8_block"})
GPTQ_SCHEMES = frozenset({"nvfp4_gptq", "nvfp4_gptq_imatrix"})
SCHEMES = (
    "fp8_w8a8",
    "nvfp4_w4a4",
    "fp8_dynamic",
    "fp8_block",
    "nvfp4_gptq",
    "nvfp4_gptq_imatrix",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quantize a DFlash/DSpark speculator drafter "
        "(FP8 static/dynamic/block | NVFP4 RTN/GPTQ/GPTQ+IMatrix)."
    )
    p.add_argument("--scheme", required=True, choices=SCHEMES)
    p.add_argument(
        "--model",
        required=True,
        help="HF id or local path of the speculator checkpoint, e.g. "
        "RedHatAI/Qwen3-8B-speculator.dflash",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--processor",
        default=None,
        help="Tokenizer/processor for oneshot pre_process. Default: the "
        "verifier recorded in the speculator config. Required by "
        "llm-compressor when a dataset is passed (oneshot arms).",
    )
    # calibration (oneshot arms only)
    p.add_argument(
        "--calib-data",
        default="random",
        help="'random' (Gaussian calibration control) OR path to a "
        "speculators-prepared dataset dir (Arrow dataset + "
        "hidden_states/ sibling from `speculators prepare-data` + "
        "`generate-offline-data`). See "
        "scripts/20_capture_hidden_states.sh.",
    )
    p.add_argument(
        "--num-calibration-samples",
        type=int,
        default=16,
        help="Random-mode smoke default is 16; use up to 2048 with real data.",
    )
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    # recipe shaping
    p.add_argument(
        "--ignore",
        action="append",
        default=[],
        help="Extra ignore entries (names or 're:...' regexes). Repeatable.",
    )
    p.add_argument(
        "--keep-fc-bf16",
        action="store_true",
        help="Do not quantize the fuse projection (fc) -- the most "
        "outlier-exposed module (sees raw target hidden states).",
    )
    p.add_argument(
        "--keep-heads-bf16",
        action="store_true",
        help="DSpark: do not quantize Markov/confidence head Linears.",
    )
    p.add_argument(
        "--nvfp4-weight-observer",
        default=None,
        choices=["nvfp4_expanded_mse", "nvfp4_expanded_imatrix"],
        help="Better weight scales for NVFP4 at some calibration cost.",
    )
    p.add_argument(
        "--gptq-dampening-frac",
        type=float,
        default=0.01,
        help="GPTQ Hessian diagonal dampening fraction (matched added arms use 0.1).",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="model_free_ptq shard workers (data-free arms).",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved plan and exit (no GPU/deps needed).",
    )
    return p.parse_args()


def build_ignore_list(args: argparse.Namespace) -> list[str]:
    ignore = list(DEFAULT_IGNORE)
    if args.keep_fc_bf16:
        ignore.append("re:.*\\.fc$")
    if args.keep_heads_bf16:
        ignore += ["re:.*markov.*", "re:.*confidence.*"]
    ignore += args.ignore
    return ignore


def build_recipe(
    scheme: str,
    ignore: list[str],
    nvfp4_weight_observer=None,
    gptq_dampening_frac: float = 0.01,
):
    """Build a recipe using the installed llm-compressor/CT preset definitions.

    The GPTQ pair uses identical expanded MSE search ranges. The IMatrix arm
    additionally weights errors by E[input_channel^2]. ``strict`` prevents an
    unobserved module from silently falling back to ordinary MSE.
    """
    if scheme in GPTQ_SCHEMES:
        from compressed_tensors.quantization import preset_name_to_scheme
        from llmcompressor.modifiers.gptq import GPTQModifier

        nvfp4 = preset_name_to_scheme("NVFP4", ["Linear"])
        nvfp4.weights.observer = (
            "nvfp4_expanded_imatrix"
            if scheme == "nvfp4_gptq_imatrix"
            else "nvfp4_expanded_mse"
        )
        if scheme == "nvfp4_gptq_imatrix":
            nvfp4.weights.observer_kwargs = {"strict": True}
        return GPTQModifier(
            config_groups={"group_0": nvfp4},
            ignore=ignore,
            dampening_frac=gptq_dampening_frac,
        )

    from llmcompressor.modifiers.quantization import QuantizationModifier

    preset = {"fp8_w8a8": "FP8", "nvfp4_w4a4": "NVFP4"}[scheme]
    kwargs = {"targets": "Linear", "scheme": preset, "ignore": ignore}
    if scheme == "nvfp4_w4a4" and nvfp4_weight_observer:
        kwargs["weight_observer"] = nvfp4_weight_observer
    return QuantizationModifier(**kwargs)


# ---------------------------------------------------------------------------
# Random calibration data (measured Gaussian control -- see docstring)
# ---------------------------------------------------------------------------
class RandomBatchDataset:
    """On-demand dataset producing batch dicts keyed to DFlash/DSpark forward kwargs.

    Keys match speculators dflash/core.py:486-500:
      hidden_states [1, T, N_target_layers*H]  (concatenated target hidden states)
      input_ids [1, T]
      loss_mask [1, T]                          (drives anchor selection -> all ones)
      verifier_last_hidden_states [1, T, H]
      document_ids [1, T]                       (single doc -> all zeros)
    Dims are read off the instantiated model, not config attrs, for robustness.
    Generates batches on-the-fly to avoid holding ~200 GB in memory.
    """

    def __init__(self, model, num_samples: int, seq_len: int, seed: int, dtype):
        if num_samples <= 0 or seq_len <= 0:
            raise ValueError("seq_len and num_samples must be positive")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.n_concat = model.fc.in_features  # N_target_layers * H
        self.hidden = model.verifier_lm_head.weight.shape[1]  # target hidden size H
        self.vocab = model.embed_tokens.num_embeddings  # verifier vocab
        self.seed = seed
        self.dtype = dtype

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        import torch

        g = torch.Generator().manual_seed(self.seed + idx)
        return {
            "hidden_states": torch.randn(
                1, self.seq_len, self.n_concat, generator=g
            ).to(self.dtype),
            "input_ids": torch.randint(0, self.vocab, (1, self.seq_len), generator=g),
            "loss_mask": torch.ones(1, self.seq_len, dtype=torch.long),
            "verifier_last_hidden_states": torch.randn(
                1, self.seq_len, self.hidden, generator=g
            ).to(self.dtype),
            "document_ids": torch.zeros(1, self.seq_len, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Real calibration data (speculators-prepared dataset + captured hidden states)
# ---------------------------------------------------------------------------
def make_real_loader(
    datapath: str, seq_len: int, num_samples: int, seed: int, **expected
):
    """Read scaled, aligned calibration batches lazily; reject incomplete budgets."""
    from torch.utils.data import DataLoader

    from dquant.calibration import RealBatchDataset

    dataset = RealBatchDataset(datapath, seq_len, num_samples, seed, **expected)
    return DataLoader(dataset, batch_size=None, num_workers=0), len(dataset)


# ---------------------------------------------------------------------------
# oneshot path (fp8_w8a8, nvfp4_w4a4)
# ---------------------------------------------------------------------------
def run_oneshot_arm(args: argparse.Namespace, ignore: list[str]) -> dict | None:  # noqa: C901
    import torch
    from llmcompressor import oneshot
    from llmcompressor.entrypoints.oneshot import Oneshot
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    try:
        from speculators import SpeculatorModel
    except ImportError:  # older layouts
        from speculators.model import SpeculatorModel

    # Forward anchor selection uses global RNGs as well as the per-row generator.
    import random

    import numpy as np

    random.seed(args.seed)
    np.random.seed(args.seed)  # noqa: NPY002 - drafter forward uses global NumPy RNG
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # -- Shim 1 (REQUIRED): AutoConfig probe crashes on model_type="speculator_model"
    #    (llm-compressor entrypoints/oneshot.py:277-315). Input ckpt is known
    #    unquantized, so the probe is safe to skip.
    Oneshot.validate_model = lambda _self, _model: None

    # -- Load drafter via the speculators library (bypasses AutoModel entirely).
    model = SpeculatorModel.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    calibration_max_anchors = max(1, args.seq_len // model.config.block_size)
    args.calibration_max_anchors = calibration_max_anchors

    # -- Shim 2 (REQUIRED): keep _name_or_path empty. If set, llm-compressor's
    #    resave_config copies the ORIGINAL config.json over the output
    #    (compressed_tensors_utils.py:276-341) and corrupts the saved config.
    model.config._name_or_path = ""  # noqa: SLF001 - llm-compressor config shim

    # -- Shim 3 (recommended): forward is torch.compile-wrapped at class-def time
    #    (dflash/core.py:485). Unwrap to avoid dynamo/hook friction during
    #    calibration.
    fwd = model.forward
    orig = getattr(fwd, "_torchdynamo_orig_callable", None)
    if orig is not None:
        model.forward = orig.__get__(model)

    # DSpark.forward defaults max_anchors to 3,072 (a training-time setting).
    # During this uncompiled calibration pass flex_attention materializes its
    # dense score matrix, so that padded anchor count can allocate tens of GB
    # even when the calibration sequence is short. Use the maximum number of
    # complete blocks available in each sequence instead.
    forward = model.forward

    def forward_with_calibration_anchors(*forward_args, **forward_kwargs):
        forward_kwargs["max_anchors"] = calibration_max_anchors
        return forward(*forward_args, **forward_kwargs)

    model.forward = forward_with_calibration_anchors

    # llm-compressor normally moves a model's output head to the meta device
    # during calibration. DSpark consumes drafter logits before its metric
    # calculation (the Markov head adds a bias), so keep the real BF16 head
    # available while explicit ignore rules protect it from quantization.
    output_embeddings_method = model.get_output_embeddings
    model.get_output_embeddings = lambda: None

    # -- Optional: skip loss/metric math (pipeline discards outputs anyway;
    #    pipelines/basic/pipeline.py:73). Defensive: attribute may be a method.
    for modname in (
        "speculators.models.dflash.core",
        "speculators.models.dflash.metrics",
        "speculators.models.dspark.metrics",
    ):
        try:
            import importlib

            mod = importlib.import_module(modname)
            if hasattr(mod, "compute_metrics"):
                mod.compute_metrics = lambda *_args, **_kwargs: (None, {})
        except ImportError:
            continue

    model.to(args.device)

    processor_src = args.processor
    if processor_src is None:
        # Verifier path recorded in the speculator config
        # (speculators_config.verifier.name_or_path); used for weight reload.
        try:
            processor_src = model.config.speculators_config.verifier.name_or_path
        except AttributeError:
            sys.exit(
                "--processor not given and verifier path not found in config; "
                "pass --processor <verifier hf id> explicitly."
            )
    tokenizer = AutoTokenizer.from_pretrained(processor_src)

    recipe = build_recipe(
        args.scheme, ignore, args.nvfp4_weight_observer, args.gptq_dampening_frac
    )
    gptq_audit = {}
    if args.scheme in GPTQ_SCHEMES:
        original_summary = recipe._log_rtn_fallback_summary  # noqa: SLF001

        def record_gptq_fallbacks():
            # GPTQModifier clears these fields in on_finalize, so snapshot them
            # immediately before its own fallback summary runs.
            gptq_audit.update(
                {
                    "compressed_modules": recipe._num_compressed_modules,  # noqa: SLF001
                    "rtn_fallback_modules": list(recipe._rtn_fallback_module_names),  # noqa: SLF001
                }
            )
            return original_summary()

        object.__setattr__(recipe, "_log_rtn_fallback_summary", record_gptq_fallbacks)

    if args.calib_data == "random":
        batches = RandomBatchDataset(
            model, args.num_calibration_samples, args.seq_len, args.seed, torch.bfloat16
        )
        # batch_size=None: each pre-built [1, T, ...] dict is yielded as-is.
        loader = DataLoader(batches, batch_size=None)
        n_calib = len(batches)
    else:
        loader, n_calib = make_real_loader(
            args.calib_data,
            args.seq_len,
            args.num_calibration_samples,
            args.seed,
            expected_target=model.config.speculators_config.verifier.name_or_path,
            expected_layers=model.config.aux_hidden_state_layer_ids,
            expected_hidden_size=model.verifier_lm_head.weight.shape[1],
        )
    if args.calib_data == "random":
        calibration_manifest = {
            "source": "random",
            "requested_samples": args.num_calibration_samples,
            "actual_samples": n_calib,
            "requested_token_capacity": n_calib * args.seq_len,
            "actual_tokens": n_calib * args.seq_len,
            "loss_mask_tokens": n_calib * args.seq_len,
            "seq_len": args.seq_len,
            "seed": args.seed,
            "rng_policy": "python-numpy-torch-cuda-and-per-row",
            "decoded_dtype": "bfloat16",
        }
    else:
        calibration_manifest = dict(loader.dataset.manifest)
    calibration_manifest["max_anchors"] = calibration_max_anchors
    calibration_manifest["block_size"] = model.config.block_size
    logger.info(
        "[calib] source=%s samples=%s tokens=%s",
        args.calib_data,
        n_calib,
        calibration_manifest["actual_tokens"],
    )

    # BasicPipeline invokes sequential_epoch_end once after all batches, which
    # triggers GPTQ compression without FX tracing the non-causal drafter.
    # It does not propagate compression error between layers.
    try:
        oneshot(
            model=model,
            dataset=loader,
            recipe=recipe,
            pipeline="basic",  # no fx tracing of the non-causal forward
            processor=tokenizer,  # required when dataset is provided
            moe_calibrate_all_experts=False,
            output_dir=args.output_dir,
        )
    finally:
        model.get_output_embeddings = output_embeddings_method
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "calibration_manifest.json").write_text(
        json.dumps(calibration_manifest, indent=2) + "\n"
    )
    if args.scheme in GPTQ_SCHEMES:
        if gptq_audit.get("compressed_modules", 0) == 0:
            raise RuntimeError("GPTQ modifier did not report any compressed modules")
        gptq_audit["rtn_fallback_count"] = len(gptq_audit["rtn_fallback_modules"])
        gptq_audit["gptq_dampening_frac"] = args.gptq_dampening_frac
        (Path(args.output_dir) / "gptq_audit.json").write_text(
            json.dumps(gptq_audit, indent=2) + "\n"
        )
        if gptq_audit["rtn_fallback_count"]:
            (Path(args.output_dir) / "run_status.json").write_text(
                json.dumps(
                    {
                        "status": "excluded_rtn_fallback",
                        "gptq_dampening_frac": args.gptq_dampening_frac,
                        "rtn_fallback_count": gptq_audit["rtn_fallback_count"],
                        "rtn_fallback_modules": gptq_audit["rtn_fallback_modules"],
                    },
                    indent=2,
                )
                + "\n"
            )
            raise RuntimeError(
                "GPTQ used RTN fallback for "
                + ", ".join(gptq_audit["rtn_fallback_modules"])
            )
    logger.info("[done] %s checkpoint at %s", args.scheme, args.output_dir)
    return gptq_audit or None


# ---------------------------------------------------------------------------
# model_free_ptq path (fp8_dynamic, fp8_block) -- data-free safetensors
# ---------------------------------------------------------------------------
def run_model_free_arm(args: argparse.Namespace, ignore: list[str]) -> None:
    from llmcompressor import model_free_ptq

    # Hub training snapshots also contain optimizer_state_dict.pt (~2 GiB).
    # model_free_ptq copies every non-safetensors file from its input stub, so
    # point it at a lean temporary view of the frozen checkpoint instead.
    source = Path(args.model)
    with tempfile.TemporaryDirectory(prefix="dquant-model-free-") as temp:
        stub = Path(temp)
        patterns = (
            "*.safetensors",
            "*.safetensors.index.json",
            "config.json",
            "config.py",
            "tokenizer*",
            "chat_template*",
            "train_command.txt",
            "speculators.patch",
        )
        for pattern in patterns:
            for file in source.glob(pattern):
                if file.is_file():
                    (stub / file.name).symlink_to(file.resolve())
        if not any(stub.glob("*.safetensors")) or not (stub / "config.json").exists():
            raise ValueError(f"model-free source lacks config or safetensors: {source}")
        model_free_ptq(
            model_stub=stub,
            save_directory=args.output_dir,
            scheme={"fp8_dynamic": "FP8_DYNAMIC", "fp8_block": "FP8_BLOCK"}[
                args.scheme
            ],
            ignore=ignore,
            max_workers=args.max_workers,
            device=args.device,
        )
    logger.info("[done] %s checkpoint at %s", args.scheme, args.output_dir)


def resolve_source(model: str, *, download: bool) -> tuple[str, dict]:
    """Freeze an HF ID to a snapshot and fingerprint its safetensors weights."""
    import hashlib

    path = Path(model)
    if not path.is_dir():
        if not download:
            return model, {"source_revision": None, "source_checkpoint_sha256": None}
        from huggingface_hub import snapshot_download

        path = Path(
            snapshot_download(
                repo_id=model,
                allow_patterns=[
                    "*.safetensors",
                    "*.safetensors.index.json",
                    "config.json",
                    "config.py",
                    "tokenizer*",
                    "chat_template*",
                    "train_command.txt",
                    "speculators.patch",
                ],
            )
        )
    path = path.resolve()
    weights = sorted(path.glob("*.safetensors"))
    if not weights:
        raise ValueError(f"source has no safetensors checkpoint: {path}")
    digest = hashlib.sha256()
    for file in weights:
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    revision = path.name if path.parent.name == "snapshots" else None
    return str(path), {
        "source_revision": revision,
        "source_checkpoint_sha256": digest.hexdigest(),
        "source_weight_files": [file.name for file in weights],
    }


def verify_checkpoint(output_dir: Path, scheme: str) -> dict:  # noqa: C901
    """Require a readable checkpoint and the intended compressed-tensors scheme."""
    from safetensors import safe_open

    config = json.loads((output_dir / "config.json").read_text())
    qconfig = config["quantization_config"]
    expected = {
        "fp8_block": (8, "block", None),
        "nvfp4_gptq": (4, "tensor_group", "nvfp4_expanded_mse"),
        "nvfp4_gptq_imatrix": (4, "tensor_group", "nvfp4_expanded_imatrix"),
    }
    if scheme in expected:
        bits, strategy, observer = expected[scheme]
        groups = qconfig["config_groups"]
        if not groups:
            raise ValueError("quantized config contains no groups")
        for group in groups.values():
            weights = group["weights"]
            if (weights["num_bits"], weights["strategy"], weights["observer"]) != (
                bits,
                strategy,
                observer,
            ):
                raise ValueError(f"wrong quantization scheme in {output_dir}")
    safetensors = sorted(output_dir.glob("*.safetensors"))
    if not safetensors:
        raise ValueError(f"checkpoint has no safetensors: {output_dir}")
    count = 0
    all_keys = set()
    for file in safetensors:
        with safe_open(file, framework="pt", device="cpu") as reader:
            keys = set(reader.keys())
            if all_keys & keys:
                raise ValueError("duplicate safetensors keys across shards")
            all_keys.update(keys)
            count += len(keys)
            if (
                scheme == "fp8_block"
                and "fc.weight" in keys
                and reader.get_slice("fc.weight").get_dtype() != "F8_E4M3"
            ):
                raise ValueError("fc weights were not quantized to FP8")
    if count == 0:
        raise ValueError("checkpoint has no tensors")
    if (
        scheme in GPTQ_SCHEMES
        and not {"fc.weight_packed", "fc.weight_scale", "fc.input_global_scale"}
        <= all_keys
    ):
        raise ValueError("GPTQ NVFP4 checkpoint lacks packed fc weight/scales")
    if scheme in GPTQ_SCHEMES:
        audit = json.loads((output_dir / "gptq_audit.json").read_text())
        if audit["compressed_modules"] <= 0:
            raise ValueError("GPTQ audit contains no compressed modules")
        if audit["rtn_fallback_count"] != len(audit["rtn_fallback_modules"]):
            raise ValueError("GPTQ audit fallback count does not match its module list")
    if scheme == "fp8_block" and not {"fc.weight", "fc.weight_scale"} <= all_keys:
        raise ValueError("FP8 block checkpoint lacks quantized fc weight/scale")
    index = output_dir / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        if set(weight_map) != all_keys:
            raise ValueError(
                "safetensors index does not cover exactly the saved tensors"
            )
        if not set(weight_map.values()) <= {file.name for file in safetensors}:
            raise ValueError("safetensors index points to missing shards")
    return {"weight_files": [file.name for file in safetensors], "tensor_count": count}


def write_quant_provenance(
    output_dir: Path,
    plan: dict,
    *,
    argv: list[str] | None = None,
    backfill_note: str | None = None,
) -> None:
    """Save exact invocation and untracked experiment source with each output."""
    import hashlib

    output_dir.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parents[1]
    sources = (
        "dquant/quantize.py",
        "dquant/calibration.py",
        "scripts/30_quantize.sh",
        "scripts/run_added_quantizations.sh",
    )
    saved_hashes = {}
    for relative in sources:
        source = source_root / relative
        destination = output_dir / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        saved_hashes[relative] = hashlib.sha256(destination.read_bytes()).hexdigest()

    model_source = Path(plan["resolved_model"])
    for name in ("train_command.txt", "speculators.patch"):
        original = model_source / name
        if original.is_file():
            destination = (
                output_dir / "train_command.txt"
                if name == "train_command.txt"
                else output_dir / "source_speculators.patch"
            )
            shutil.copy2(original, destination)
            saved_hashes[f"source_model/{name}"] = hashlib.sha256(
                destination.read_bytes()
            ).hexdigest()

    repo_root = Path(os.environ.get("DRAFTERS_QUANT_REPOS", "/workspace"))
    statuses = {}
    for name in ("speculators", "llm-compressor", "compressed-tensors"):
        repo = repo_root / name
        status = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "-C", str(repo), "status", "--short"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        )
        statuses[name] = status.stdout.strip() if status.returncode == 0 else None
        patch = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "-C", str(repo), "diff", "--binary", "HEAD"],  # noqa: S607
            capture_output=True,
            check=False,
        )
        if patch.returncode == 0 and patch.stdout:
            (output_dir / f"{name}.patch").write_bytes(patch.stdout)

    packages = {}
    for name in ("llmcompressor", "compressed-tensors", "torch", "transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    env = {
        name: os.environ.get(name)
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "HF_HOME",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
        )
        if os.environ.get(name) is not None
    }
    command = shlex.join(argv if argv is not None else [sys.executable, *sys.argv])
    (output_dir / "quant_command.txt").write_text(
        "# Timestamp: "
        + plan["timestamp"]
        + "\n"
        + "# Source revisions: "
        + json.dumps(plan["source_revisions"], sort_keys=True)
        + "\n"
        + "# Source status: "
        + json.dumps(statuses, sort_keys=True)
        + "\n"
        + "# Package versions: "
        + json.dumps(packages, sort_keys=True)
        + "\n"
        + "# Source SHA-256: "
        + json.dumps(saved_hashes, sort_keys=True)
        + "\n"
        + "# Environment: "
        + json.dumps(env, sort_keys=True)
        + "\n"
        + ("# Backfill note: " + backfill_note + "\n" if backfill_note else "")
        + command
        + "\n"
    )


def restore_local_processor_snapshot(
    output_dir: Path, processor: str | None
) -> dict | None:
    """Keep exact tokenizer bytes when transformers reserializes a local snapshot."""
    import hashlib

    if processor is None or not Path(processor).is_dir():
        return None
    source = Path(processor).resolve()
    identity = {"snapshot": str(source), "revision": source.name, "files": {}}
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        original = source / name
        if not original.is_file():
            continue
        destination = output_dir / name
        before = (
            hashlib.sha256(destination.read_bytes()).hexdigest()
            if destination.is_file()
            else None
        )
        shutil.copy2(original, destination)
        after = hashlib.sha256(destination.read_bytes()).hexdigest()
        if after != hashlib.sha256(original.read_bytes()).hexdigest():
            raise ValueError(f"processor file changed during copy: {name}")
        identity["files"][name] = {
            "source_sha256": after,
            "pre_restore_export_sha256": before,
            "restored_exact_source_bytes": True,
        }
    return identity


def source_revisions():
    root = Path(os.environ.get("DRAFTERS_QUANT_REPOS", "/workspace"))
    revisions = {}
    for name in ("speculators", "llm-compressor", "compressed-tensors"):
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "-C", str(root / name), "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        )
        revisions[name] = result.stdout.strip() if result.returncode == 0 else None
    return revisions


def main() -> None:  # noqa: C901 - CLI plan, export, and completion gate
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s"
    )
    args = parse_args()
    if args.scheme in GPTQ_SCHEMES and args.nvfp4_weight_observer:
        raise ValueError(
            "GPTQ arms fix the expanded MSE observer pair; do not pass "
            "--nvfp4-weight-observer"
        )
    if args.gptq_dampening_frac <= 0:
        raise ValueError("--gptq-dampening-frac must be positive")
    args.calibration_max_anchors = None
    source_model, source_metadata = resolve_source(
        args.model, download=not args.dry_run
    )
    requested_model = args.model
    args.model = source_model
    ignore = build_ignore_list(args)

    plan = {
        "scheme": args.scheme,
        "model": requested_model,
        "resolved_model": source_model,
        **source_metadata,
        "processor": args.processor,
        "output_dir": args.output_dir,
        "ignore": ignore,
        "calibration": (
            "none (data-free)"
            if args.scheme in DATA_FREE_SCHEMES
            else (
                "random Gaussian calibration control"
                if args.calib_data == "random"
                else f"real: {args.calib_data}"
            )
        ),
        "num_calibration_samples": (
            args.num_calibration_samples if args.scheme not in DATA_FREE_SCHEMES else 0
        ),
        "seq_len": args.seq_len,
        "calibration_max_anchors": (
            args.calibration_max_anchors
            if args.scheme not in DATA_FREE_SCHEMES
            else None
        ),
        "seed": args.seed,
        "device": args.device,
        "max_workers": args.max_workers if args.scheme in DATA_FREE_SCHEMES else None,
        "algorithm": (
            "GPTQ"
            if args.scheme in GPTQ_SCHEMES
            else "model_free_ptq"
            if args.scheme in DATA_FREE_SCHEMES
            else "RTN"
        ),
        "pipeline": ("basic" if args.scheme not in DATA_FREE_SCHEMES else None),
        "propagate_compression_error": False,
        "weight_observer": (
            "nvfp4_expanded_imatrix"
            if args.scheme == "nvfp4_gptq_imatrix"
            else "nvfp4_expanded_mse"
            if args.scheme == "nvfp4_gptq"
            else args.nvfp4_weight_observer
        ),
        "gptq_dampening_frac": (
            args.gptq_dampening_frac if args.scheme in GPTQ_SCHEMES else None
        ),
        "rng_policy": "python-numpy-torch-cuda-and-per-row",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_revisions": source_revisions(),
    }
    print(json.dumps(plan, indent=2))  # noqa: T201 - CLI dry-run output
    if args.dry_run:
        return

    output = Path(args.output_dir)
    if (
        args.scheme in (GPTQ_SCHEMES | {"fp8_block"})
        and output.exists()
        and any(output.iterdir())
    ):
        raise FileExistsError(
            f"output already contains files: {output}; choose a fresh directory "
            "or inspect the existing checkpoint"
        )

    write_quant_provenance(output, plan)

    algorithm_audit = None
    if args.scheme in DATA_FREE_SCHEMES:
        run_model_free_arm(args, ignore)
    else:
        algorithm_audit = run_oneshot_arm(args, ignore)
        plan["calibration_max_anchors"] = args.calibration_max_anchors
        processor_identity = restore_local_processor_snapshot(output, args.processor)
        if processor_identity:
            plan["processor_identity"] = processor_identity

    # Run manifest alongside the checkpoint for reproducibility.
    output.mkdir(parents=True, exist_ok=True)
    verified = verify_checkpoint(output, args.scheme)
    calibration_file = output / "calibration_manifest.json"
    if calibration_file.exists():
        import hashlib

        calibration = json.loads(calibration_file.read_text())
        plan["calibration_identity"] = {
            "manifest_sha256": hashlib.sha256(
                calibration_file.read_bytes()
            ).hexdigest(),
            "actual_samples": calibration["actual_samples"],
            "actual_tokens": calibration["actual_tokens"],
            "capture_command_sha256": calibration.get("capture_command_sha256"),
            "cache_header_stat_sha256": calibration.get("cache_header_stat_sha256"),
            "dataset_sha256": calibration.get("dataset_sha256"),
        }
    if algorithm_audit:
        plan["gptq_audit"] = algorithm_audit
    with (output / "quant_run_manifest.json").open("w") as f:
        json.dump(plan, f, indent=2)
    if args.scheme in GPTQ_SCHEMES:
        (output / "run_status.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "gptq_dampening_frac": args.gptq_dampening_frac,
                    "rtn_fallback_count": algorithm_audit["rtn_fallback_count"],
                },
                indent=2,
            )
            + "\n"
        )
    if args.scheme in (GPTQ_SCHEMES | {"fp8_block"}):
        (output / "checkpoint_complete.json").write_text(
            json.dumps({"scheme": args.scheme, **verified}, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
