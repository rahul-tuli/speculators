#!/usr/bin/env python3
"""Run the five added DFlash drafters on the two pinned acceptance suites.

Each subset uses the repository's ``evaluate.py throughput`` command. A fresh
vLLM process serves one arm with seven speculative tokens on the selected GPU. Raw
GuideLLM requests, commands, checkpoint hashes, source hashes, and failed
attempts remain under the arm directory. ``acceptance.csv`` appears only when
every subset has exact prompt coverage and consistent speculative counters.

Example:
    python local/drafter-quant/scripts/run_add_evals.py \
      --arm fp8-block-datafree --checkpoint /data/fast/.../fp8_block-datafree
"""

from __future__ import annotations

import argparse
import collections
import copy
import csv
import hashlib
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
TARGET = Path(
    "/data/fast/hf_cache/hub/models--Qwen--Qwen3-8B/snapshots/"
    "b968826d9c46dd6066d109eabc6255188de91218"
)
PINNED_DRAFTER = Path(
    "/data/fast/hf_cache/hub/models--RedHatAI--Qwen3-8B-speculator.dflash/"
    "snapshots/1a11b170eb65c8a62c80ecd01dfe22a5907298e6"
)
PINNED_DRAFTER_SHA256 = (
    "afda76166ccd6a2d3ef55b214889bd045be3b8fc41b41be330d06d25059f88e8"
)
PINNED_DRAFTER_REVISION = "1a11b170eb65c8a62c80ecd01dfe22a5907298e6"
SPEEDBENCH = Path("/data/fast/drafter-quant/data/speedbench-frozen")
REDHATAI = Path("/data/fast/drafter-quant/pilot/redhatai-speculator_benchmarks-2ae86af")
DEFAULT_ROOT = Path("/data/fast/drafter-quant/results/add-evals-20260928-verified")
GPTQ_BASE_ROOT = Path("/data/fast/drafter-quant/out/add-evals-20260928")
GPTQ_OUTPUT_ROOT = GPTQ_BASE_ROOT / "damping0p1"
SPEED_CATEGORIES = (
    "coding",
    "humanities",
    "math",
    "multilingual",
    "qa",
    "rag",
    "reasoning",
    "roleplay",
    "stem",
    "summarization",
    "writing",
)
REDHATAI_SUBSETS = (
    "HumanEval",
    "math_reasoning",
    "qa",
    "question",
    "rag",
    "summarization",
    "tool_call",
    "translation",
    "writing",
)
ARM_IDS = (
    "fp8-block-datafree",
    "nvfp4-gptq-imatrix-gaussian",
    "nvfp4-gptq-imatrix-perfectblend",
    "nvfp4-gptq-gaussian",
    "nvfp4-gptq-perfectblend",
)
ARM_EXPECTATIONS = {
    "fp8-block-datafree": {
        "scheme": "fp8_block",
        "calibration": "none (data-free)",
        "algorithm": "model_free_ptq",
        "weight_observer": None,
        "samples": 0,
        "seed": 0,
        "weight_format": (8, "block"),
    },
    "nvfp4-gptq-imatrix-gaussian": {
        "scheme": "nvfp4_gptq_imatrix",
        "calibration": "random Gaussian calibration control",
        "algorithm": "GPTQ",
        "weight_observer": "nvfp4_expanded_imatrix",
        "samples": 2027,
        "seed": 1,
        "dampening_frac": 0.01,
        "require_explicit_dampening": False,
        "dampening_source": "recipe_and_run_status_legacy",
        "checkpoint_root": GPTQ_BASE_ROOT,
        "checkpoint_name": "nvfp4_gptq_imatrix-random",
        "weight_format": (4, "tensor_group"),
    },
    "nvfp4-gptq-imatrix-perfectblend": {
        "scheme": "nvfp4_gptq_imatrix",
        "calibration": (
            "real: /data/fast/drafter-quant/data/perfectblend_2048_prepared"
        ),
        "algorithm": "GPTQ",
        "weight_observer": "nvfp4_expanded_imatrix",
        "samples": 2027,
        "seed": 1,
        "dampening_frac": 0.1,
        "require_explicit_dampening": True,
        "dampening_source": "manifest_recipe_audit_and_run_status",
        "checkpoint_root": GPTQ_OUTPUT_ROOT,
        "checkpoint_name": "nvfp4_gptq_imatrix-real",
        "weight_format": (4, "tensor_group"),
    },
    "nvfp4-gptq-gaussian": {
        "scheme": "nvfp4_gptq",
        "calibration": "random Gaussian calibration control",
        "algorithm": "GPTQ",
        "weight_observer": "nvfp4_expanded_mse",
        "samples": 2027,
        "seed": 1,
        "dampening_frac": 0.01,
        "require_explicit_dampening": False,
        "dampening_source": "recipe_and_run_status_legacy",
        "checkpoint_root": GPTQ_BASE_ROOT,
        "checkpoint_name": "nvfp4_gptq-random",
        "weight_format": (4, "tensor_group"),
    },
    "nvfp4-gptq-perfectblend": {
        "scheme": "nvfp4_gptq",
        "calibration": (
            "real: /data/fast/drafter-quant/data/perfectblend_2048_prepared"
        ),
        "algorithm": "GPTQ",
        "weight_observer": "nvfp4_expanded_mse",
        "samples": 2027,
        "seed": 1,
        "dampening_frac": 0.1,
        "require_explicit_dampening": True,
        "dampening_source": "manifest_recipe_audit_and_run_status",
        "checkpoint_root": GPTQ_OUTPUT_ROOT,
        "checkpoint_name": "nvfp4_gptq-real",
        "weight_format": (4, "tensor_group"),
    },
}
SPEEDBENCH_ROWS_PER_CATEGORY = 80
SPEC_TOKENS = 7
CALIBRATION_SEQ_LEN = 2048
MIN_SAFETENSORS_BYTES = 100
RUNNER_SOURCE = "local/drafter-quant/scripts/run_add_evals.py"
PROVENANCE_FILES = (
    "vllm_command.txt",
    "vllm.patch",
    "checkpoint_sha256.txt",
    "drafter_checkpoint_sha256.txt",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def verify_pinned_source() -> None:
    weights = sorted(PINNED_DRAFTER.glob("*.safetensors"))
    if [path.name for path in weights] != ["model.safetensors"]:
        raise ValueError("Pinned BF16 drafter weight files changed")
    if sha256(weights[0]) != PINNED_DRAFTER_SHA256:
        raise ValueError("Pinned BF16 drafter SHA256 mismatch")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def rows(path: Path, field: str) -> collections.Counter[str]:
    prompts: collections.Counter[str] = collections.Counter()
    with path.open() as stream:
        for line in stream:
            if line.strip():
                prompt = json.loads(line)[field]
                if not isinstance(prompt, str) or not prompt:
                    raise ValueError(f"Missing {field!r} in {path}")
                prompts[prompt] += 1
    return prompts


def input_suites() -> dict[str, list[dict]]:
    speed_manifest = json.loads((SPEEDBENCH / "manifest.json").read_text())
    red_manifest = json.loads((REDHATAI / "manifest.json").read_text())
    revision = red_manifest["dataset"]["revision"]
    if revision != "2ae86affa2cb97a972b7fc681dd51c04fbff083e":
        raise ValueError(f"Unexpected RedHatAI revision: {revision}")
    red_files = {item["name"]: item for item in red_manifest["dataset"]["source_files"]}
    suites: dict[str, list[dict]] = {
        "speedbench-qualitative": [],
        "redhatai-speculator-benchmarks": [],
    }
    for category in SPEED_CATEGORIES:
        filename = f"qualitative_{category}.jsonl"
        path = SPEEDBENCH / filename
        item = speed_manifest["files"][filename]
        actual_hash = sha256(path)
        expected_prompts = rows(path, "turns")
        if (
            actual_hash != item["sha256"]
            or sum(expected_prompts.values()) != SPEEDBENCH_ROWS_PER_CATEGORY
        ):
            raise ValueError(f"Frozen SpeedBench input mismatch: {path}")
        suites["speedbench-qualitative"].append(
            {
                "name": category,
                "label": f"speedbench/qualitative/{category}",
                "path": str(path),
                "sha256": actual_hash,
                "rows": SPEEDBENCH_ROWS_PER_CATEGORY,
                "field": "turns",
                "max_requests": SPEEDBENCH_ROWS_PER_CATEGORY,
            }
        )
    for subset in REDHATAI_SUBSETS:
        filename = f"{subset}.jsonl"
        path = REDHATAI / "raw" / filename
        item = red_files[filename]
        actual_hash = sha256(path)
        expected_prompts = rows(path, "prompt")
        if (
            actual_hash != item["sha256"]
            or sum(expected_prompts.values()) != item["row_count"]
        ):
            raise ValueError(f"Pinned RedHatAI input mismatch: {path}")
        suites["redhatai-speculator-benchmarks"].append(
            {
                "name": subset,
                "label": subset,
                "path": str(path),
                "sha256": actual_hash,
                "rows": item["row_count"],
                "field": "prompt",
                "max_requests": 200,
            }
        )
    return suites


def request_prompt(request: dict) -> str:
    args = request["request_args"]
    if isinstance(args, str):
        args = json.loads(args)
    messages = args["body"]["messages"]
    return messages[0]["content"][0]["text"]


def validate_subset(dataset: dict, directory: Path) -> dict:
    csv_path = directory / "acceptance.csv"
    raw_path = (
        directory
        / "artifacts"
        / ("run_" + dataset["label"].replace("/", "_") + ".json")
    )
    with csv_path.open(newline="") as stream:
        acceptance = list(csv.DictReader(stream))
    if len(acceptance) != 1 or acceptance[0]["subset"] != dataset["label"]:
        raise ValueError(f"Missing or mismatched acceptance row in {csv_path}")
    row = acceptance[0]
    drafts = float(row["num_drafts"])
    proposed = float(row["num_draft_tokens"])
    accepted = float(row["num_accepted_tokens"])
    length = float(row["acceptance_length"])
    position = [float(row[f"acceptance_at_pos_{index}"]) for index in range(7)]
    if not all(
        math.isfinite(v) for v in (drafts, proposed, accepted, length, *position)
    ):
        raise ValueError(f"Nonfinite acceptance metric in {csv_path}")
    if not (drafts > 0 and proposed == 7 * drafts and 0 <= accepted <= proposed):
        raise ValueError(f"Invalid seven-token speculative counters in {csv_path}")
    if not math.isclose(length, 1 + accepted / drafts, abs_tol=1e-9):
        raise ValueError(f"Invalid acceptance length in {csv_path}")
    if not math.isclose(sum(position), accepted / drafts, abs_tol=1e-6):
        raise ValueError(f"Per-position counters disagree in {csv_path}")
    raw = json.loads(raw_path.read_text())
    benchmarks = raw["benchmarks"]
    if len(benchmarks) != 1:
        raise ValueError(f"Expected one throughput benchmark in {raw_path}")
    requests = benchmarks[0]["requests"]
    successful = requests["successful"]
    counts = {
        "successful": len(successful),
        "errored": len(requests["errored"]),
        "incomplete": len(requests["incomplete"]),
    }
    expected = rows(Path(dataset["path"]), dataset["field"])
    actual = collections.Counter(request_prompt(request) for request in successful)
    if counts != {"successful": dataset["rows"], "errored": 0, "incomplete": 0}:
        raise ValueError(f"Request coverage failed in {raw_path}: {counts}")
    if actual != expected:
        missing = expected - actual
        excess = actual - expected
        raise ValueError(
            f"Prompt multiset differs in {raw_path}: "
            f"missing={sum(missing.values())}, excess={sum(excess.values())}"
        )
    if any(not request.get("output") for request in successful):
        raise ValueError(f"Empty successful output in {raw_path}")
    return {
        "label": dataset["label"],
        "acceptance": row,
        "request_counts": counts,
        "expected_rows": dataset["rows"],
        "prompt_multiset_match": True,
        "raw_path": str(raw_path),
        "raw_sha256": sha256(raw_path),
        "eval_command": str(directory / "eval_command.txt"),
    }


def validate_gptq_audit(
    path: Path, manifest: dict, expected_dampening: float, require_explicit: bool
) -> str:
    """Require observed GPTQ compression with no RTN fallback modules."""
    if not path.is_file():
        raise ValueError(f"Missing GPTQ audit: {path}")
    try:
        audit = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable GPTQ audit: {path}") from exc
    if not isinstance(audit, dict):
        raise ValueError(f"GPTQ audit must be an object: {path}")
    compressed = audit.get("compressed_modules")
    fallbacks = audit.get("rtn_fallback_modules")
    fallback_count = audit.get("rtn_fallback_count")
    if type(compressed) is not int or compressed <= 0:
        raise ValueError(f"GPTQ audit has no compressed modules: {path}")
    if (
        not isinstance(fallbacks, list)
        or any(not isinstance(module, str) or not module for module in fallbacks)
        or len(set(fallbacks)) != len(fallbacks)
    ):
        raise ValueError(f"GPTQ audit fallback module list is invalid: {path}")
    if type(fallback_count) is not int or fallback_count != len(fallbacks):
        raise ValueError(f"GPTQ audit fallback count/list mismatch: {path}")
    if manifest.get("gptq_audit") != audit:
        raise ValueError(f"GPTQ audit differs from quantization manifest: {path}")
    if (
        require_explicit and audit.get("gptq_dampening_frac") != expected_dampening
    ) or (
        not require_explicit
        and "gptq_dampening_frac" in audit
        and audit["gptq_dampening_frac"] != expected_dampening
    ):
        raise ValueError(f"GPTQ audit dampening fraction mismatch: {path}")
    if fallback_count:
        raise ValueError(
            f"GPTQ audit reports {fallback_count} RTN fallback modules: {path}"
        )
    return sha256(path)


def validate_gptq_recipe(
    path: Path, manifest: dict, expected_dampening: float, require_explicit: bool
) -> str:
    """Require the recorded GPTQ modifier to use the fixed damping fraction."""
    actual = manifest.get("gptq_dampening_frac")
    if (require_explicit or "gptq_dampening_frac" in manifest) and (
        type(actual) not in (int, float) or actual != expected_dampening
    ):
        raise ValueError(
            f"GPTQ manifest dampening fraction {actual!r} != {expected_dampening}"
        )
    if not path.is_file():
        raise ValueError(f"Missing GPTQ recipe: {path}")
    try:
        recipe = yaml.load(path.read_text(), Loader=yaml.BaseLoader)  # noqa: S506
        modifier = recipe["default_stage"]["default_modifiers"]["GPTQModifier"]
        recipe_dampening = float(modifier["dampening_frac"])
    except (OSError, yaml.YAMLError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Unreadable GPTQ damping recipe: {path}") from exc
    if not math.isfinite(recipe_dampening) or recipe_dampening != expected_dampening:
        raise ValueError(
            f"GPTQ recipe dampening fraction {recipe_dampening!r} "
            f"!= {expected_dampening}: {path}"
        )
    return sha256(path)


def validate_checkpoint_for_arm(arm: str, checkpoint: Path) -> dict:  # noqa: C901
    verify_pinned_source()
    expected = ARM_EXPECTATIONS[arm]
    if expected["algorithm"] == "GPTQ":
        intended = (expected["checkpoint_root"] / expected["checkpoint_name"]).resolve()
        if checkpoint != intended:
            raise ValueError(f"{arm} requires checkpoint path {intended}")
    manifest_path = checkpoint / "quant_run_manifest.json"
    marker_path = checkpoint / "checkpoint_complete.json"
    if not manifest_path.is_file() or not marker_path.is_file():
        raise ValueError(f"Checkpoint is not marked complete: {checkpoint}")
    manifest = json.loads(manifest_path.read_text())
    marker = json.loads(marker_path.read_text())
    for key in ("scheme", "calibration", "algorithm", "weight_observer"):
        if manifest.get(key) != expected[key]:
            raise ValueError(
                f"{arm} requires {key}={expected[key]!r}, "
                f"checkpoint reports {manifest.get(key)!r}"
            )
    if manifest.get("num_calibration_samples") != expected["samples"]:
        raise ValueError(f"Wrong calibration sample count for {arm}")
    if (
        manifest.get("seq_len") != CALIBRATION_SEQ_LEN
        or manifest.get("seed") != expected["seed"]
    ):
        raise ValueError(f"Wrong calibration length or seed for {arm}")
    if manifest.get("rng_policy") != "python-numpy-torch-cuda-and-per-row":
        raise ValueError(f"Missing complete calibration RNG policy for {arm}")
    if (
        manifest.get("source_revision") != PINNED_DRAFTER_REVISION
        or manifest.get("source_checkpoint_sha256") != PINNED_DRAFTER_SHA256
        or Path(manifest.get("resolved_model", "")).resolve()
        != PINNED_DRAFTER.resolve()
        or Path(manifest.get("output_dir", "")).resolve() != checkpoint
    ):
        raise ValueError(f"Wrong source or output identity for {arm}")
    if marker.get("scheme") != expected["scheme"]:
        raise ValueError(f"Completion marker scheme mismatch for {arm}")
    weights = sorted(checkpoint.glob("*.safetensors"))
    if not weights or marker.get("weight_files") != [path.name for path in weights]:
        raise ValueError(f"Completion marker weight list mismatch for {arm}")
    if not isinstance(marker.get("tensor_count"), int) or marker["tensor_count"] <= 0:
        raise ValueError(f"Completion marker tensor count missing for {arm}")
    if any(
        path.stat().st_size <= MIN_SAFETENSORS_BYTES
        or path.stat().st_mtime_ns > marker_path.stat().st_mtime_ns
        for path in weights
    ):
        raise ValueError(f"Checkpoint weights changed after completion: {arm}")
    config = json.loads((checkpoint / "config.json").read_text())
    groups = config["quantization_config"]["config_groups"]
    if not groups:
        raise ValueError(f"No quantization group in {arm}")
    for group in groups.values():
        weight = group["weights"]
        if (weight["num_bits"], weight["strategy"]) != expected["weight_format"]:
            raise ValueError(f"Wrong exported quantization format for {arm}")
        if weight.get("observer") != expected["weight_observer"]:
            raise ValueError(f"Wrong exported weight observer for {arm}")
    calibration_path = checkpoint / "calibration_manifest.json"
    if expected["samples"]:
        if not calibration_path.is_file():
            raise ValueError(f"Missing calibration manifest for {arm}")
        calibration = json.loads(calibration_path.read_text())
        if (
            calibration.get("actual_samples") != expected["samples"]
            or calibration.get("seq_len") != CALIBRATION_SEQ_LEN
            or calibration.get("seed") != expected["seed"]
        ):
            raise ValueError(f"Incomplete calibration budget for {arm}")
        if "gaussian" in arm:
            if calibration.get("source") != "random":
                raise ValueError(f"Wrong Gaussian calibration source for {arm}")
        elif (
            Path(calibration.get("cache_path", "")).resolve()
            != Path(
                "/data/fast/drafter-quant/data/perfectblend_2048_prepared"
            ).resolve()
        ):
            raise ValueError(f"Wrong PerfectBlend calibration cache for {arm}")
    elif calibration_path.exists():
        raise ValueError(
            f"Data-free checkpoint unexpectedly has calibration manifest: {arm}"
        )
    audit_sha256 = None
    recipe_sha256 = None
    run_status_sha256 = None
    if expected["algorithm"] == "GPTQ":
        status_path = checkpoint / "run_status.json"
        if not status_path.is_file():
            raise ValueError(f"Missing GPTQ run status: {status_path}")
        status = json.loads(status_path.read_text())
        if (
            status.get("status") != "complete"
            or status.get("gptq_dampening_frac") != expected["dampening_frac"]
            or status.get("rtn_fallback_count") != 0
        ):
            raise ValueError(
                f"GPTQ checkpoint is not complete without fallbacks: {status_path}"
            )
        run_status_sha256 = sha256(status_path)
        require_explicit = expected["require_explicit_dampening"]
        recipe_sha256 = validate_gptq_recipe(
            checkpoint / "recipe.yaml",
            manifest,
            expected["dampening_frac"],
            require_explicit,
        )
        audit_sha256 = validate_gptq_audit(
            checkpoint / "gptq_audit.json",
            manifest,
            expected["dampening_frac"],
            require_explicit,
        )
    result = {
        "manifest_sha256": sha256(manifest_path),
        "completion_marker_sha256": sha256(marker_path),
        "calibration_manifest_sha256": (
            sha256(calibration_path) if calibration_path.exists() else None
        ),
        "scheme": expected["scheme"],
        "calibration": expected["calibration"],
        "source_revision": PINNED_DRAFTER_REVISION,
        "source_checkpoint_sha256": PINNED_DRAFTER_SHA256,
    }
    if audit_sha256 is not None:
        result["gptq_audit_sha256"] = audit_sha256
        result["gptq_dampening_frac"] = expected["dampening_frac"]
        result["gptq_dampening_source"] = expected["dampening_source"]
        result["recipe_sha256"] = recipe_sha256
        result["run_status_sha256"] = run_status_sha256
    return result


def prepare_identity(args: argparse.Namespace, suites: dict[str, list[dict]]) -> dict:
    checkpoint = args.checkpoint.resolve()
    if not TARGET.is_dir() or not checkpoint.is_dir():
        raise FileNotFoundError("Pinned target or drafter checkpoint is missing")
    checkpoint_audit = validate_checkpoint_for_arm(args.arm, checkpoint)
    manifest = checkpoint / "quant_run_manifest.json"
    weights = sorted(checkpoint.glob("*.safetensors"))
    if not manifest.is_file() or not weights:
        raise ValueError(f"Checkpoint lacks quant manifest or weights: {checkpoint}")
    weight_stats = {
        path.name: {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in weights
    }
    return {
        "arm": args.arm,
        "checkpoint": str(checkpoint),
        "checkpoint_manifest_sha256": sha256(manifest),
        "checkpoint_audit": checkpoint_audit,
        "checkpoint_config_sha256": sha256(checkpoint / "config.json"),
        "checkpoint_weights_stat": weight_stats,
        "target": str(TARGET),
        "target_config_sha256": sha256(TARGET / "config.json"),
        "speedbench_manifest_sha256": sha256(SPEEDBENCH / "manifest.json"),
        "redhatai_manifest_sha256": sha256(REDHATAI / "manifest.json"),
        "redhatai_revision": "2ae86affa2cb97a972b7fc681dd51c04fbff083e",
        "input_files": dict(suites),
        "evaluation": {
            "profile": "throughput",
            "max_concurrency": 128,
            "max_tokens": 4096,
            "temperature": 0,
            "seed": 0,
        },
        "serving": {
            "spec_method": "dflash",
            "spec_tokens": 7,
            "dtype": "bfloat16",
            "seed": 0,
            "max_model_len": 16384,
            "max_num_seqs": 128,
            "gpu_memory_utilization": 0.8,
            "prefix_caching": False,
            "enforce_eager": True,
            "nvfp4_kernel_backend": "emulation"
            if args.arm.startswith("nvfp4")
            else None,
            "physical_gpu": args.gpu,
        },
        "source_sha256": {
            str(path.relative_to(REPO)): sha256(path)
            for path in (
                REPO / "scripts/launch_vllm.py",
                REPO / "scripts/evaluate/evaluate.py",
                REPO / "scripts/evaluate/perf_utils.py",
                Path(__file__).resolve(),
            )
        },
    }


def ensure_identity(directory: Path, identity: dict) -> None:
    path = directory / "run-identity.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved != identity:
            complete = (directory / "evaluation-summary.json").is_file()
            source = directory / "run_add_evals.py"
            source_hash = saved.get("source_sha256", {}).get(RUNNER_SOURCE)
            if (
                complete
                and source_hash
                and source.is_file()
                and sha256(source) == source_hash
            ):
                old = copy.deepcopy(saved)
                new = copy.deepcopy(identity)
                old["source_sha256"].pop(RUNNER_SOURCE, None)
                new["source_sha256"].pop(RUNNER_SOURCE, None)
                if old == new:
                    return
            raise ValueError(
                f"Run identity changed; refusing to mix results in {directory}"
            )
        source = directory / "run_add_evals.py"
        if (
            source.exists()
            and sha256(source) != identity["source_sha256"][RUNNER_SOURCE]
        ):
            raise ValueError(f"Runner source snapshot differs in {directory}")
        if not source.exists():
            shutil.copy2(Path(__file__).resolve(), source)
    else:
        write_json(path, identity)
        shutil.copy2(
            SPEEDBENCH / "manifest.json", directory / "speedbench_manifest.json"
        )
        shutil.copy2(REDHATAI / "manifest.json", directory / "redhatai_manifest.json")
        shutil.copy2(
            Path(identity["checkpoint"]) / "quant_run_manifest.json",
            directory / "quant_run_manifest.json",
        )
        patch = subprocess.run(  # noqa: S603
            [
                shutil.which("git") or "git",
                "diff",
                "HEAD",
                "--",
                "scripts/evaluate/evaluate.py",
                "scripts/evaluate/perf_utils.py",
                "scripts/launch_vllm.py",
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        (directory / "speculators.patch").write_text(patch)
        shutil.copy2(Path(__file__).resolve(), directory / "run_add_evals.py")


def launch_server(  # noqa: C901
    args: argparse.Namespace, provenance_dir: Path, env: dict[str, str]
) -> subprocess.Popen:
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", args.port)) == 0:
            raise RuntimeError(f"Serving port {args.port} is already occupied")
    provenance_dir.mkdir(parents=True, exist_ok=False)
    flags = [
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--dtype",
        "bfloat16",
        "--seed",
        "0",
        "--max-model-len",
        "16384",
        "--max-num-seqs",
        "128",
        "--gpu-memory-utilization",
        "0.8",
        "--no-enable-prefix-caching",
        "--enforce-eager",
    ]
    if args.arm.startswith("nvfp4"):
        flags += ["--kernel-config", '{"linear_backend":"emulation"}']
    command = [
        sys.executable,
        str(REPO / "scripts/launch_vllm.py"),
        "eval",
        str(TARGET),
        "--spec-model",
        str(args.checkpoint.resolve()),
        "--spec-method",
        "dflash",
        "--spec-tokens",
        "7",
        "--provenance-dir",
        str(provenance_dir),
        "--",
        *flags,
    ]
    write_json(provenance_dir / "launcher-argv.json", command)
    log = (provenance_dir / "server.log").open("w")
    try:
        process = subprocess.Popen(  # noqa: S603
            command,
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log.close()
    try:
        deadline = time.monotonic() + 900
        url = f"http://127.0.0.1:{args.port}/v1/models"
        while time.monotonic() < deadline:
            if process.poll() is not None:
                message = (
                    f"vLLM exited with {process.returncode}; "
                    f"see {provenance_dir / 'server.log'}"
                )
                raise RuntimeError(message)
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    if response.status == 200:  # noqa: PLR2004
                        break
            except (urllib.error.URLError, TimeoutError, OSError):
                pass
            time.sleep(5)
        else:
            raise TimeoutError(
                f"vLLM readiness timed out; see {provenance_dir / 'server.log'}"
            )
        for name in PROVENANCE_FILES:
            if not (provenance_dir / name).is_file():
                raise RuntimeError(f"Missing vLLM provenance: {provenance_dir / name}")
        command_record = (provenance_dir / "vllm_command.txt").read_text()
        if "--spec-tokens 7" not in command_record:
            raise ValueError("Recorded serving command lacks seven speculative tokens")
        if args.arm.startswith("nvfp4") and "emulation" not in command_record:
            raise ValueError(
                "Recorded NVFP4 serving command lacks H100 emulation backend"
            )
        smoke_request = urllib.request.Request(
            f"http://127.0.0.1:{args.port}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": str(TARGET),
                    "messages": [{"role": "user", "content": "Name one prime number."}],
                    "temperature": 0,
                    "seed": 0,
                    "max_tokens": 32,
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(smoke_request, timeout=300) as response:  # noqa: S310
            smoke = json.load(response)
        write_json(provenance_dir / "one-request-smoke.json", smoke)
        if (
            not smoke.get("choices")
            or smoke.get("usage", {}).get("completion_tokens", 0) <= 0
        ):
            raise ValueError("vLLM one-request smoke produced no completion")
    except Exception:
        stop_server(process)
        raise
    return process


def stop_server(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def evaluation_command(
    args: argparse.Namespace, suite: str, dataset: dict, attempt: Path
) -> list[str]:
    command = [
        sys.executable,
        str(REPO / "scripts/evaluate/evaluate.py"),
        "throughput",
        "--target",
        f"http://127.0.0.1:{args.port}/v1",
        "--subsets",
        dataset["name"],
        "--output-dir",
        str(attempt),
        "--max-concurrency",
        "128",
        "--max-requests",
        str(dataset["max_requests"]),
        "--max-tokens",
        "4096",
        "--eval-seed",
        "0",
        "--gen-kwargs",
        '{"temperature":0,"seed":0}',
    ]
    if suite == "speedbench-qualitative":
        command += [
            "--dataset",
            f"speedbench/qualitative/{dataset['name']}",
            "--speedbench-data-dir",
            str(SPEEDBENCH),
        ]
    else:
        command += [
            "--dataset",
            dataset["path"],
            "--data-column-mapper",
            "kind=generative_column_mapper,column_mappings.text_column=prompt",
        ]
    return command


def run_subset(
    args: argparse.Namespace,
    suite: str,
    dataset: dict,
    arm_dir: Path,
    env: dict[str, str],
) -> dict:
    subset_root = arm_dir / "subsets" / dataset["name"]
    selected = subset_root / "selected.json"
    if selected.exists():
        record = json.loads(selected.read_text())
        validated = validate_subset(dataset, Path(record["attempt_dir"]))
        if validated["raw_sha256"] != record["raw_sha256"]:
            raise ValueError(f"Selected raw result changed: {selected}")
        return validated
    subset_root.mkdir(parents=True, exist_ok=True)
    for index in range(1, 4):
        attempt = subset_root / f"attempt-{index}"
        if attempt.exists():
            continue
        attempt.mkdir()
        command = evaluation_command(args, suite, dataset, attempt)
        write_json(attempt / "evaluator-argv.json", command)
        print(f"[{args.arm} {suite} {dataset['name']}] attempt {index}", flush=True)  # noqa: T201
        with (attempt / "eval.log").open("w") as log:
            result = subprocess.run(  # noqa: S603
                command,
                cwd=REPO,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        (attempt / "exit-code.txt").write_text(f"{result.returncode}\n")
        try:
            if result.returncode:
                raise RuntimeError(f"Evaluator exit {result.returncode}")
            validated = validate_subset(dataset, attempt)
        except (ValueError, KeyError, FileNotFoundError, RuntimeError) as exc:
            (attempt / "validation-error.txt").write_text(
                f"{type(exc).__name__}: {exc}\n"
            )
            print(f"  failed: {exc}", flush=True)  # noqa: T201
            continue
        write_json(
            selected,
            {"attempt_dir": str(attempt), "raw_sha256": validated["raw_sha256"]},
        )
        return validated
    raise RuntimeError(
        f"Three attempts failed for {args.arm} {suite} {dataset['name']}"
    )


def finalize_suite(
    args: argparse.Namespace,
    suite: str,
    datasets: list[dict],
    arm_dir: Path,
    results: list[dict],
    *,
    provenance_dir: Path,
) -> None:
    if len(results) != len(datasets):
        raise ValueError(f"Incomplete {suite} results for {args.arm}")
    fields = list(results[0]["acceptance"])
    temporary = arm_dir / "acceptance.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["acceptance"] for result in results)
    temporary.replace(arm_dir / "acceptance.csv")
    commands = []
    artifacts = arm_dir / "artifacts"
    artifacts.mkdir(exist_ok=True)
    for result in results:
        command = Path(result["eval_command"])
        commands.append(f"# {result['label']}\n{command.read_text()}")
        raw = Path(result["raw_path"])
        destination = artifacts / raw.name
        if destination.exists():
            if sha256(destination) != result["raw_sha256"]:
                raise ValueError(f"Published raw artifact changed: {destination}")
        else:
            try:
                os.link(raw, destination)
            except OSError:
                shutil.copy2(raw, destination)
    (arm_dir / "eval_command.txt").write_text("\n".join(commands))
    for name in PROVENANCE_FILES:
        shutil.copy2(provenance_dir / name, arm_dir / name)
    write_json(
        arm_dir / "evaluation-summary.json",
        {
            "status": "complete",
            "arm": args.arm,
            "suite": suite,
            "serving_provenance": str(provenance_dir),
            "spec_tokens": 7,
            "profile": "throughput",
            "subsets": results,
            "total_requests": {
                key: sum(result["request_counts"][key] for result in results)
                for key in ("successful", "errored", "incomplete")
            },
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def validate_complete_suite(
    args: argparse.Namespace, suite: str, datasets: list[dict], arm_dir: Path
) -> None:
    summary = json.loads((arm_dir / "evaluation-summary.json").read_text())
    if (
        summary.get("status") != "complete"
        or summary.get("arm") != args.arm
        or summary.get("suite") != suite
        or summary.get("spec_tokens") != SPEC_TOKENS
        or summary.get("profile") != "throughput"
    ):
        raise ValueError(f"Invalid completion summary: {arm_dir}")
    listed = summary.get("subsets", [])
    if [item.get("label") for item in listed] != [item["label"] for item in datasets]:
        raise ValueError(f"Incomplete subset list in {arm_dir}")
    expected_rows = sum(item["rows"] for item in datasets)
    if summary.get("total_requests") != {
        "successful": expected_rows,
        "errored": 0,
        "incomplete": 0,
    }:
        raise ValueError(f"Invalid request totals in {arm_dir}")
    for item in listed:
        raw = Path(item["raw_path"])
        if sha256(raw) != item["raw_sha256"] or not item.get("prompt_multiset_match"):
            raise ValueError(f"Changed raw result or missing prompt audit in {arm_dir}")
    with (arm_dir / "acceptance.csv").open(newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    if csv_rows != [item["acceptance"] for item in listed]:
        raise ValueError(f"Aggregate CSV changed in {arm_dir}")
    for name in (*PROVENANCE_FILES, "eval_command.txt", "speculators.patch"):
        if not (arm_dir / name).is_file():
            raise ValueError(f"Missing finalized provenance in {arm_dir}: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARM_IDS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--suite",
        choices=("both", "speedbench-qualitative", "redhatai-speculator-benchmarks"),
        default="both",
    )
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--port", type=int, default=8111)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    suites = input_suites()
    selected_suites = list(suites) if args.suite == "both" else [args.suite]
    identity = prepare_identity(args, suites)
    arm_dirs = {suite: args.output_root / suite / args.arm for suite in selected_suites}
    if args.check_inputs:
        print(  # noqa: T201
            json.dumps(
                {"identity": identity, "selected_suites": selected_suites}, indent=2
            )
        )
        return
    for directory in arm_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
        ensure_identity(directory, identity)
    pending = {}
    for suite in selected_suites:
        summary = arm_dirs[suite] / "evaluation-summary.json"
        if summary.exists():
            validate_complete_suite(args, suite, suites[suite], arm_dirs[suite])
        else:
            pending[suite] = suites[suite]
    if not pending:
        print(f"[{args.arm}] all requested suites already complete", flush=True)  # noqa: T201
        return
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "HF_HOME": "/data/fast/hf_cache",
            "HF_HUB_CACHE": "/data/fast/hf_cache/hub",
            "HF_DATASETS_CACHE": "/data/fast/hf_cache/datasets",
            "HF_HUB_DISABLE_XET": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PATH": str(REPO / ".venv/bin") + os.pathsep + os.environ["PATH"],
        }
    )
    primary = arm_dirs[selected_suites[0]]
    provenance_dir = (
        primary
        / "server-attempts"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    )
    process = None
    try:
        process = launch_server(args, provenance_dir, env)
        for suite, datasets in pending.items():
            arm_dir = arm_dirs[suite]
            results = [
                run_subset(args, suite, dataset, arm_dir, env) for dataset in datasets
            ]
            finalize_suite(
                args, suite, datasets, arm_dir, results, provenance_dir=provenance_dir
            )
            print(f"[{args.arm}] {suite}: {len(results)} subsets complete", flush=True)  # noqa: T201
    finally:
        stop_server(process)


if __name__ == "__main__":
    main()
