#!/usr/bin/env python3
"""Quantize DFlash/DSpark speculator drafters with llm-compressor.

Schemes (selected by --scheme):
  fp8_w8a8    FP8 static W8A8 (weights + activations)  -> oneshot + calibration data
  nvfp4_w4a4  NVFP4 W4A4 (fp4 group-16, LOCAL-dynamic  -> oneshot + calibration data
              activations w/ calibrated global scale)
  fp8_dynamic FP8 per-channel weights + dynamic         -> model_free_ptq (data-free,
              per-token activations (weight-focused)        no model load, no shims)

Grounding (pinned repos at $DRAFTERS_QUANT_REPOS, default
/workspace):
  speculators 40b12c9, llm-compressor ed55e6f67, compressed-tensors 698967f

Why oneshot works for a drafter without framework changes:
  - DFlashDraftModel IS a transformers.PreTrainedModel (speculators/model.py:213)
  - oneshot(model=<instance>, dataset=<DataLoader>) skips AutoModel path loading
    (llm-compressor entrypoints/utils.py:61-63)
  - pipeline="basic" splats batch dicts straight into model(**batch)
    (pipelines/basic/pipeline.py:62-73) -- so target hidden states enter the real
    drafter forward as ordinary batch keys.

CALIBRATION DATA: --calib-data random feeds RANDOM tensors (Gaussian) shaped
like real drafter inputs. That is statistically WRONG for activation observers
-- real inputs are outlier-heavy target hidden states, and scales/global-scales
calibrated on randn will mis-cover the real distribution (expect acceptance-rate
damage). It exists as the degenerate baseline of the random-vs-real study and to
smoke-test the plumbing. Use scripts/20_capture_hidden_states.sh to produce a
real prepared dataset dir and pass it via --calib-data before producing any
reported result.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
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

SCHEMES = ("fp8_w8a8", "nvfp4_w4a4", "fp8_dynamic")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quantize a DFlash/DSpark speculator drafter "
        "(FP8 W8A8 static | NVFP4 W4A4 | FP8_DYNAMIC)."
    )
    p.add_argument("--scheme", required=True, choices=SCHEMES)
    p.add_argument("--model", required=True,
                   help="HF id or local path of the speculator checkpoint, e.g. "
                        "RedHatAI/Qwen3-8B-speculator.dflash")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--processor", default=None,
                   help="Tokenizer/processor for oneshot pre_process. Default: the "
                        "verifier recorded in the speculator config. Required by "
                        "llm-compressor when a dataset is passed (oneshot arms).")
    # calibration (oneshot arms only)
    p.add_argument("--calib-data", default="random",
                   help="'random' (degenerate baseline / smoke only) OR path to a "
                        "speculators-prepared dataset dir (Arrow dataset + "
                        "hidden_states/ sibling from `speculators prepare-data` + "
                        "`generate-offline-data`). See "
                        "scripts/20_capture_hidden_states.sh.")
    p.add_argument("--num-calibration-samples", type=int, default=16,
                   help="Random-mode smoke default is 16; use up to 2048 with real data.")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    # recipe shaping
    p.add_argument("--ignore", action="append", default=[],
                   help="Extra ignore entries (names or 're:...' regexes). Repeatable.")
    p.add_argument("--keep-fc-bf16", action="store_true",
                   help="Do not quantize the fuse projection (fc) -- the most "
                        "outlier-exposed module (sees raw target hidden states).")
    p.add_argument("--keep-heads-bf16", action="store_true",
                   help="DSpark: do not quantize Markov/confidence head Linears.")
    p.add_argument("--nvfp4-weight-observer", default=None,
                   choices=["nvfp4_expanded_mse", "nvfp4_expanded_imatrix"],
                   help="Better weight scales for NVFP4 at some calibration cost.")
    p.add_argument("--max-workers", type=int, default=8,
                   help="model_free_ptq shard workers (fp8_dynamic arm).")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the resolved plan and exit (no GPU/deps needed).")
    return p.parse_args()


def build_ignore_list(args: argparse.Namespace) -> list[str]:
    ignore = list(DEFAULT_IGNORE)
    if args.keep_fc_bf16:
        ignore.append("re:.*\\.fc$")
    if args.keep_heads_bf16:
        ignore += ["re:.*markov.*", "re:.*confidence.*"]
    ignore += args.ignore
    return ignore


# ---------------------------------------------------------------------------
# Random calibration data (DEGENERATE BASELINE / SMOKE ONLY -- see docstring)
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
        self.n_concat = model.fc.in_features                       # N_target_layers * H
        self.hidden = model.verifier_lm_head.weight.shape[1]       # target hidden size H
        self.vocab = model.embed_tokens.num_embeddings             # verifier vocab
        self.seed = seed
        self.dtype = dtype

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        import torch
        g = torch.Generator().manual_seed(self.seed + idx)
        return {
            "hidden_states": torch.randn(1, self.seq_len, self.n_concat,
                                         generator=g).to(self.dtype),
            "input_ids": torch.randint(0, self.vocab, (1, self.seq_len), generator=g),
            "loss_mask": torch.ones(1, self.seq_len, dtype=torch.long),
            "verifier_last_hidden_states": torch.randn(1, self.seq_len, self.hidden,
                                                       generator=g).to(self.dtype),
            "document_ids": torch.zeros(1, self.seq_len, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Real calibration data (speculators-prepared dataset + captured hidden states)
# ---------------------------------------------------------------------------
def make_real_loader(datapath: str, seq_len: int, num_samples: int, seed: int,
                     **expected):
    """Read scaled, aligned calibration batches lazily; reject incomplete budgets."""
    from torch.utils.data import DataLoader
    from dquant.calibration import RealBatchDataset

    dataset = RealBatchDataset(datapath, seq_len, num_samples, seed, **expected)
    return DataLoader(dataset, batch_size=None, num_workers=0), len(dataset)


# ---------------------------------------------------------------------------
# oneshot path (fp8_w8a8, nvfp4_w4a4)
# ---------------------------------------------------------------------------
def run_oneshot_arm(args: argparse.Namespace, ignore: list[str]) -> None:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    from llmcompressor import oneshot
    from llmcompressor.entrypoints.oneshot import Oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    try:
        from speculators import SpeculatorModel
    except ImportError:  # older layouts
        from speculators.model import SpeculatorModel

    # Forward anchor selection uses global RNGs as well as the per-row generator.
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # -- Shim 1 (REQUIRED): AutoConfig probe crashes on model_type="speculator_model"
    #    (llm-compressor entrypoints/oneshot.py:277-315). Input ckpt is known
    #    unquantized, so the probe is safe to skip.
    Oneshot.validate_model = lambda self, model: None

    # -- Load drafter via the speculators library (bypasses AutoModel entirely).
    model = SpeculatorModel.from_pretrained(args.model, torch_dtype=torch.bfloat16)

    # -- Shim 2 (REQUIRED): keep _name_or_path empty. If set, llm-compressor's
    #    resave_config copies the ORIGINAL config.json over the output
    #    (compressed_tensors_utils.py:276-341) and corrupts the saved config.
    model.config._name_or_path = ""

    # -- Shim 3 (recommended): forward is torch.compile-wrapped at class-def time
    #    (dflash/core.py:485). Unwrap to avoid dynamo/hook friction during
    #    calibration.
    fwd = model.forward
    orig = getattr(fwd, "_torchdynamo_orig_callable", None)
    if orig is not None:
        model.forward = orig.__get__(model)

    # -- Optional: skip loss/metric math (pipeline discards outputs anyway;
    #    pipelines/basic/pipeline.py:73). Defensive: attribute may be a method.
    for modname in ("speculators.models.dflash.core",
                    "speculators.models.dflash.metrics",
                    "speculators.models.dspark.metrics"):
        try:
            import importlib
            mod = importlib.import_module(modname)
            if hasattr(mod, "compute_metrics"):
                mod.compute_metrics = lambda *a, **k: (None, {})
        except Exception:
            pass

    model.to(args.device)

    processor_src = args.processor
    if processor_src is None:
        # Verifier path recorded in the speculator config
        # (speculators_config.verifier.name_or_path); used for weight reload.
        try:
            processor_src = model.config.speculators_config.verifier.name_or_path
        except AttributeError:
            sys.exit("--processor not given and verifier path not found in config; "
                     "pass --processor <verifier hf id> explicitly.")
    tokenizer = AutoTokenizer.from_pretrained(processor_src)

    scheme = {"fp8_w8a8": "FP8", "nvfp4_w4a4": "NVFP4"}[args.scheme]
    recipe_kwargs = dict(targets="Linear", scheme=scheme, ignore=ignore)
    if args.scheme == "nvfp4_w4a4" and args.nvfp4_weight_observer:
        recipe_kwargs["weight_observer"] = args.nvfp4_weight_observer
    recipe = QuantizationModifier(**recipe_kwargs)

    if args.calib_data == "random":
        batches = RandomBatchDataset(model, args.num_calibration_samples,
                                     args.seq_len, args.seed,
                                     torch.bfloat16)
        # batch_size=None: each pre-built [1, T, ...] dict is yielded as-is.
        loader = DataLoader(batches, batch_size=None)
        n_calib = len(batches)
    else:
        loader, n_calib = make_real_loader(args.calib_data, args.seq_len,
                                           args.num_calibration_samples,
                                           args.seed,
                                           expected_target=model.config.speculators_config.verifier.name_or_path,
                                           expected_layers=model.config.aux_hidden_state_layer_ids,
                                           expected_hidden_size=model.verifier_lm_head.weight.shape[1])
    if args.calib_data == "random":
        calibration_manifest = {
            "source": "random", "requested_samples": args.num_calibration_samples,
            "actual_samples": n_calib,
            "requested_token_capacity": n_calib * args.seq_len,
            "actual_tokens": n_calib * args.seq_len,
            "loss_mask_tokens": n_calib * args.seq_len,
            "seq_len": args.seq_len, "seed": args.seed,
        "rng_policy": "python-numpy-torch-cuda-and-per-row",
            "decoded_dtype": "bfloat16",
        }
    else:
        calibration_manifest = loader.dataset.manifest
    logger.info("[calib] source=%s samples=%s tokens=%s", args.calib_data, n_calib,
                calibration_manifest["actual_tokens"])

    oneshot(
        model=model,
        dataset=loader,
        recipe=recipe,
        pipeline="basic",            # no fx tracing of the non-causal forward
        processor=tokenizer,         # required when dataset is provided
        moe_calibrate_all_experts=False,
        output_dir=args.output_dir,
    )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "calibration_manifest.json").write_text(
        json.dumps(calibration_manifest, indent=2) + "\n")
    logger.info("[done] %s checkpoint at %s", args.scheme, args.output_dir)


# ---------------------------------------------------------------------------
# model_free_ptq path (fp8_dynamic) -- data-free, operates on safetensors
# ---------------------------------------------------------------------------
def run_model_free_arm(args: argparse.Namespace, ignore: list[str]) -> None:
    from llmcompressor import model_free_ptq

    model_free_ptq(
        model_stub=args.model,
        save_directory=args.output_dir,
        scheme="FP8_DYNAMIC",
        ignore=ignore,
        max_workers=args.max_workers,
        device=args.device,
    )
    logger.info("[done] fp8_dynamic checkpoint at %s", args.output_dir)


def source_revisions():
    import os
    import subprocess
    root = Path(os.environ.get("DRAFTERS_QUANT_REPOS", "/workspace"))
    revisions = {}
    for name in ("speculators", "llm-compressor", "compressed-tensors"):
        result = subprocess.run(["git", "-C", str(root / name), "rev-parse", "HEAD"],
                                capture_output=True, text=True)
        revisions[name] = result.stdout.strip() if result.returncode == 0 else None
    return revisions


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(message)s")
    args = parse_args()
    ignore = build_ignore_list(args)

    plan = {
        "scheme": args.scheme,
        "model": args.model,
        "output_dir": args.output_dir,
        "ignore": ignore,
        "calibration": ("none (data-free)" if args.scheme == "fp8_dynamic"
                        else ("RANDOM SMOKE DATA (not valid for reported results)"
                              if args.calib_data == "random"
                              else f"real: {args.calib_data}")),
        "num_calibration_samples": (args.num_calibration_samples
                                    if args.scheme != "fp8_dynamic" else 0),
        "seq_len": args.seq_len, "seed": args.seed,
        "rng_policy": "python-numpy-torch-cuda-and-per-row",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_revisions": source_revisions(),
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return

    if args.scheme == "fp8_dynamic":
        run_model_free_arm(args, ignore)
    else:
        run_oneshot_arm(args, ignore)

    # Run manifest alongside the checkpoint for reproducibility.
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "quant_run_manifest.json", "w") as f:
        json.dump(plan, f, indent=2)


if __name__ == "__main__":
    main()
