#!/usr/bin/env python3
"""Validate and summarize the six-arm, nine-subset throughput acceptance run.

GPU0 writes clean retry arms under
``<attempt-root>/gpu0-thread-limited-retry/arms/<arm>/`` and GPU1 under
``<attempt-root>/gpu1-thread-limited-retry/<arm>/``. Each arm has one ``acceptance.csv`` and one
``eval_command.txt`` per subset, with vLLM provenance at the arm root. This
script reads result artifacts only; it writes the tidy table, figures,
provenance manifest, and Markdown report under ``local/drafter-quant`` by
default.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = Path("/data/fast/drafter-quant/results/throughput-20260924")
DEFAULT_ATTEMPT_NAME = "attempt-2"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "throughput-results-20260924"
DEFAULT_REPORT = PACKAGE_ROOT / "THROUGHPUT_RESULTS_20260924.md"
HF_DATASET_ID = "RedHatAI/speculator_benchmarks"
OBSERVED_HF_HEAD_REVISION = "2ae86affa2cb97a972b7fc681dd51c04fbff083e"
MAX_DRAFT_TOKENS = 7

SUBSETS = (
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
EXPECTED_SOURCE_ROWS = {
    "HumanEval": 164,
    "math_reasoning": 80,
    "qa": 80,
    "question": 80,
    "rag": 80,
    "summarization": 80,
    "tool_call": 200,
    "translation": 80,
    "writing": 80,
}

# Keep this order in the CSV and both plots: reference first, then the four
# static scheme/recipe arms, then the data-free FP8_DYNAMIC control.
ARMS = (
    {
        "id": "bf16-drafter",
        "label": "BF16 drafter",
        "scheme": "BF16",
        "calibration": "none",
        "color": "#303030",
        "hatch": "",
    },
    {
        "id": "fp8-static-gaussian",
        "label": "FP8 static W8A8 | Gaussian",
        "scheme": "FP8 static W8A8",
        "calibration": "Gaussian",
        "color": "#2878b5",
        "hatch": "//",
    },
    {
        "id": "fp8-static-perfectblend",
        "label": "FP8 static W8A8 | PerfectBlend",
        "scheme": "FP8 static W8A8",
        "calibration": "PerfectBlend",
        "color": "#62a7d4",
        "hatch": "",
    },
    {
        "id": "nvfp4-gaussian",
        "label": "NVFP4 W4A4 emulated | Gaussian",
        "scheme": "NVFP4 W4A4 (H100 emulation)",
        "calibration": "Gaussian",
        "color": "#c43c39",
        "hatch": "//",
    },
    {
        "id": "nvfp4-perfectblend",
        "label": "NVFP4 W4A4 emulated | PerfectBlend",
        "scheme": "NVFP4 W4A4 (H100 emulation)",
        "calibration": "PerfectBlend",
        "color": "#eb8178",
        "hatch": "",
    },
    {
        "id": "fp8-dynamic-datafree",
        "label": "FP8_DYNAMIC | data-free",
        "scheme": "FP8_DYNAMIC",
        "calibration": "none (data-free)",
        "color": "#8d66b0",
        "hatch": "xx",
    },
)
ARM_BY_ID = {arm["id"]: arm for arm in ARMS}
EXPECTED_ARM_IDS = tuple(arm["id"] for arm in ARMS)
POSITION_RE = re.compile(r"^acceptance_at_pos_(\d+)$")
REQUIRED_COUNTER_COLUMNS = (
    "num_drafts",
    "num_draft_tokens",
    "num_accepted_tokens",
    "acceptance_length",
)
ARM_PROVENANCE_FILES = (
    "vllm_command.txt",
    "vllm.patch",
    "checkpoint_sha256.txt",
    "drafter_checkpoint_sha256.txt",
)


class ValidationError(ValueError):
    """Input results do not satisfy the expected coverage or metric contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"could not read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"expected a JSON object in {path}")
    return value


def finite_number(value: str | None, *, path: Path, field: str) -> float:
    if value is None or value.strip() == "":
        raise ValidationError(f"missing {field} in {path}")
    try:
        number = float(value)
    except ValueError as exc:
        raise ValidationError(f"invalid {field}={value!r} in {path}") from exc
    if not math.isfinite(number):
        raise ValidationError(f"non-finite {field}={value!r} in {path}")
    return number


def integer_counter(value: str | None, *, path: Path, field: str) -> int:
    number = finite_number(value, path=path, field=field)
    rounded = round(number)
    if not math.isclose(number, rounded, abs_tol=1e-6, rel_tol=0):
        raise ValidationError(
            f"{field} must be an integer counter in {path}; got {number}"
        )
    return int(rounded)


def read_one_row(path: Path) -> tuple[list[str], dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            rows = list(reader)
    except OSError as exc:
        raise ValidationError(f"could not read CSV {path}: {exc}") from exc
    if not fields:
        raise ValidationError(f"CSV has no header: {path}")
    if len(rows) != 1:
        raise ValidationError(f"expected one subset row in {path}; found {len(rows)}")
    return fields, {key: (value or "").strip() for key, value in rows[0].items()}


def validate_metrics(
    path: Path,
    *,
    arm_id: str,
    expected_subset: str,
) -> dict[str, Any]:
    fields, row = read_one_row(path)
    missing = [
        name for name in ("subset", *REQUIRED_COUNTER_COLUMNS) if name not in fields
    ]
    if missing:
        raise ValidationError(f"missing columns {missing} in {path}")
    if row.get("subset") != expected_subset:
        raise ValidationError(
            f"subset label in {path} is {row.get('subset')!r}; expected {expected_subset!r}"
        )

    drafts = integer_counter(row.get("num_drafts"), path=path, field="num_drafts")
    proposed = integer_counter(
        row.get("num_draft_tokens"), path=path, field="num_draft_tokens"
    )
    accepted = integer_counter(
        row.get("num_accepted_tokens"), path=path, field="num_accepted_tokens"
    )
    reported_length = finite_number(
        row.get("acceptance_length"), path=path, field="acceptance_length"
    )
    if drafts <= 0:
        raise ValidationError(f"num_drafts must be positive in {path}; got {drafts}")
    if proposed <= 0:
        raise ValidationError(
            f"num_draft_tokens must be positive in {path}; got {proposed}"
        )
    if accepted < 0 or accepted > proposed:
        raise ValidationError(
            f"accepted draft tokens must be between zero and proposed tokens in {path}; "
            f"got accepted={accepted}, proposed={proposed}"
        )
    if proposed > MAX_DRAFT_TOKENS * drafts:
        raise ValidationError(
            f"proposed draft tokens exceed {MAX_DRAFT_TOKENS} per draft event in {path}: "
            f"{proposed} > {MAX_DRAFT_TOKENS} * {drafts}"
        )
    acceptance_length = 1.0 + accepted / drafts
    if not math.isclose(reported_length, acceptance_length, rel_tol=1e-4, abs_tol=1e-4):
        raise ValidationError(
            f"acceptance_length mismatch in {path}: CSV={reported_length:.9g}, "
            f"1 + accepted/drafts={acceptance_length:.9g}"
        )

    positions: dict[int, float] = {}
    for name in fields:
        match = POSITION_RE.fullmatch(name)
        if not match:
            continue
        index = int(match.group(1))
        rate = finite_number(row.get(name), path=path, field=name)
        if not 0 <= rate <= 1 + 1e-5:
            raise ValidationError(
                f"{name} must be a fraction in [0, 1] in {path}; got {rate}"
            )
        positions[index] = rate
    if positions:
        expected_indexes = list(range(max(positions) + 1))
        if sorted(positions) != expected_indexes:
            raise ValidationError(
                f"per-position columns are not contiguous from zero in {path}"
            )
        if max(positions) >= MAX_DRAFT_TOKENS:
            raise ValidationError(
                f"per-position column exceeds the {MAX_DRAFT_TOKENS}-token draft in {path}"
            )
        position_sum = sum(positions.values())
        accepted_per_draft = accepted / drafts
        if not math.isclose(
            position_sum, accepted_per_draft, rel_tol=1e-3, abs_tol=1e-4
        ):
            raise ValidationError(
                f"per-position rates in {path} sum to {position_sum:.9g}, but "
                f"accepted_tokens/num_drafts={accepted_per_draft:.9g}"
            )

    return {
        "arm": arm_id,
        "arm_label": ARM_BY_ID[arm_id]["label"],
        "scheme": ARM_BY_ID[arm_id]["scheme"],
        "calibration": ARM_BY_ID[arm_id]["calibration"],
        "subset": expected_subset,
        "num_drafts": drafts,
        "num_draft_tokens": proposed,
        "num_accepted_tokens": accepted,
        "acceptance_length": acceptance_length,
        "accepted_token_fraction": accepted / proposed,
        "accepted_token_fraction_percent": 100 * accepted / proposed,
        "position_acceptance": positions,
        "acceptance_csv": str(path.resolve()),
        "acceptance_csv_sha256": sha256(path),
    }


def find_run_metadata(results_root: Path) -> list[dict[str, Any]]:
    metadata = []
    paths = [
        path
        for path in results_root.rglob("gpu*-run.json")
        if any("thread-limited-retry" in part for part in path.parts)
    ]
    expected_filenames = {"gpu0-run.json", "gpu1-run.json"}
    found_filenames = {path.name for path in paths}
    if found_filenames != expected_filenames or len(paths) != 2:
        raise ValidationError(
            "expected exactly one retry metadata file for each worker "
            f"({sorted(expected_filenames)}); found {[str(path) for path in sorted(paths)]}"
        )
    for path in sorted(paths):
        value = load_json(path)
        value["_path"] = str(path.resolve())
        value["_sha256"] = sha256(path)
        metadata.append(value)
    return metadata


def check_worker_metadata(
    metadata: list[dict[str, Any]], *, attempt_root: Path
) -> dict[str, Any]:
    """Normalize GPU worker manifests and confirm a common HF evaluation contract."""
    if not metadata:
        raise ValidationError(f"no gpu*-run.json metadata found under {attempt_root}")

    signatures: set[tuple[Any, ...]] = set()
    revisions: set[str] = set()
    manifest_hashes: set[str] = set()
    dataset_snapshots: list[dict[str, dict[str, Any]]] = []
    worker_hardware: dict[str, str] = {}
    worker_arm_metadata: dict[str, list[dict[str, Any]]] = {
        arm_id: [] for arm_id in EXPECTED_ARM_IDS
    }

    for run in metadata:
        evaluation = run.get("evaluation", {})
        if not isinstance(evaluation, dict):
            raise ValidationError(f"invalid evaluation metadata in {run['_path']}")

        dataset_value = evaluation.get("dataset", evaluation.get("dataset_id"))
        if isinstance(dataset_value, dict):
            dataset_value = dataset_value.get("id", dataset_value.get("repo"))
        if dataset_value != HF_DATASET_ID:
            raise ValidationError(
                f"worker metadata {run['_path']} does not record the HF dataset "
                f"{HF_DATASET_ID!r}: got {dataset_value!r}"
            )

        subset_value = evaluation.get(
            "subset_order",
            evaluation.get("subsets", evaluation.get("subset_names")),
        )
        if subset_value is None and isinstance(evaluation.get("raw_files"), list):
            subset_value = [
                Path(item.get("name", "")).stem
                for item in evaluation["raw_files"]
                if isinstance(item, dict)
            ]
        if isinstance(subset_value, dict):
            subset_names = list(subset_value)
        elif isinstance(subset_value, str):
            subset_names = [name for name in re.split(r"[,\s]+", subset_value) if name]
        elif isinstance(subset_value, list):
            subset_names = subset_value
        else:
            raise ValidationError(
                f"worker metadata {run['_path']} does not list evaluation subsets"
            )
        if set(subset_names) != set(SUBSETS) or len(subset_names) != len(SUBSETS):
            raise ValidationError(
                f"worker metadata {run['_path']} has unexpected subset coverage: {subset_names}"
            )

        local_inputs_used = evaluation.get(
            "local_jsonl_inputs_used",
            evaluation.get("evaluation_uses_local_raw_files"),
        )
        if local_inputs_used is not False:
            raise ValidationError(
                f"worker metadata {run['_path']} does not affirm HF-backed inputs "
                "(local JSONL evaluation inputs must be false)"
            )

        snapshot_value = evaluation.get("subsets")
        if isinstance(snapshot_value, dict):
            snapshot_items = [
                {"subset": name, **entry}
                for name, entry in snapshot_value.items()
                if isinstance(entry, dict)
            ]
        elif isinstance(evaluation.get("raw_files"), list):
            snapshot_items = [
                {"subset": Path(item.get("name", "")).stem, **item}
                for item in evaluation["raw_files"]
                if isinstance(item, dict)
            ]
        else:
            snapshot_items = []
        if not snapshot_items:
            raise ValidationError(
                f"worker metadata {run['_path']} has no HF subset snapshot manifest"
            )

        snapshot: dict[str, dict[str, Any]] = {}
        for item in snapshot_items:
            subset = str(item.get("subset", ""))
            if subset not in SUBSETS or subset in snapshot:
                raise ValidationError(
                    f"worker metadata {run['_path']} has invalid or duplicate snapshot "
                    f"subset {subset!r}"
                )
            filename = str(item.get("file", item.get("name", "")))
            rows = item.get("rows")
            digest = str(item.get("sha256", ""))
            tree_oid = str(item.get("hub_tree_oid", ""))
            if filename != f"{subset}.jsonl":
                raise ValidationError(
                    f"worker metadata {run['_path']} has unexpected HF manifest file "
                    f"for {subset}: {filename!r}"
                )
            if rows != EXPECTED_SOURCE_ROWS[subset]:
                raise ValidationError(
                    f"worker metadata {run['_path']} records {rows!r} source rows for "
                    f"{subset}; expected {EXPECTED_SOURCE_ROWS[subset]}"
                )
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValidationError(
                    f"worker metadata {run['_path']} has invalid SHA-256 for {subset}: "
                    f"{digest!r}"
                )
            if tree_oid and not re.fullmatch(r"[0-9a-f]{40}", tree_oid):
                raise ValidationError(
                    f"worker metadata {run['_path']} has invalid HF tree OID for "
                    f"{subset}: {tree_oid!r}"
                )
            snapshot[subset] = {
                "file": filename,
                "rows": rows,
                "sha256": digest,
                "hub_tree_oid": tree_oid or None,
            }
        if set(snapshot) != set(SUBSETS):
            raise ValidationError(
                f"worker metadata {run['_path']} snapshot manifest has incomplete "
                f"coverage: {sorted(snapshot)}"
            )
        dataset_snapshots.append(snapshot)

        pinned_manifest = run.get("pinned_input_manifest")
        manifest_sha = (
            pinned_manifest.get("sha256") if isinstance(pinned_manifest, dict) else None
        )
        manifest_sha = manifest_sha or evaluation.get("input_manifest_sha256")
        manifest_sha = manifest_sha or run.get("input_manifest_sha256")
        if manifest_sha:
            manifest_sha = str(manifest_sha)
            if not re.fullmatch(r"[0-9a-f]{64}", manifest_sha):
                raise ValidationError(
                    f"worker metadata {run['_path']} has invalid input manifest "
                    f"SHA-256: {manifest_sha!r}"
                )
            manifest_hashes.add(manifest_sha)

        revision = evaluation.get("dataset_revision", evaluation.get("hf_revision"))
        if revision:
            revisions.add(str(revision))

        server = run.get("serving", run.get("server", {}))
        if not isinstance(server, dict):
            server = {}
        signatures.add(
            (
                dataset_value,
                evaluation.get("profile", evaluation.get("mode")),
                evaluation.get("max_requests", evaluation.get("max_requests_cap")),
                evaluation.get("max_concurrency"),
                evaluation.get("max_tokens"),
                json.dumps(evaluation.get("gen_kwargs", {}), sort_keys=True),
                evaluation.get("data_column_mapper"),
                server.get("spec_method"),
                server.get("spec_tokens"),
            )
        )

        runtime = run.get("runtime", {})
        gpu_info = runtime.get("gpu0") if isinstance(runtime, dict) else None
        gpu_info = gpu_info or run.get("gpu", {}).get("device")
        if gpu_info:
            worker_hardware[run["_path"]] = str(gpu_info)

        for arm in run.get("arms", []):
            if not isinstance(arm, dict):
                continue
            arm_name = arm.get("name", arm.get("label"))
            if arm_name not in worker_arm_metadata:
                continue
            worker_arm_metadata[arm_name].append(
                {
                    "worker_metadata": run["_path"],
                    "scheme": arm.get("scheme"),
                    "calibration": arm.get("calibration"),
                    "kernel_config": arm.get("kernel_config"),
                    "status": arm.get("status"),
                }
            )
            # GPU0 records live status. GPU1's prepared metadata can retain
            # pending labels, so its completed CSV coverage is the authority.
            if "name" in arm and arm.get("status") in ("failed", "running"):
                raise ValidationError(
                    f"worker metadata marks {arm_name} {arm.get('status')}: {run['_path']}"
                )

    if len(signatures) != 1:
        raise ValidationError(
            f"GPU workers disagree on evaluation settings: {signatures}"
        )
    if len(dataset_snapshots) != len(metadata):
        raise ValidationError("worker source-manifest metadata is incomplete")
    source_snapshot: dict[str, dict[str, Any]] = {}
    for subset in SUBSETS:
        entries = [snapshot[subset] for snapshot in dataset_snapshots]
        shared_fields = ("file", "rows", "sha256")
        if any(
            len({str(entry[field]) for entry in entries}) != 1
            for field in shared_fields
        ):
            raise ValidationError(
                f"GPU workers disagree on the HF subset source manifest for {subset}"
            )
        tree_oids = {
            entry["hub_tree_oid"] for entry in entries if entry["hub_tree_oid"]
        }
        if len(tree_oids) > 1:
            raise ValidationError(
                f"GPU workers disagree on the HF tree OID for {subset}: {sorted(tree_oids)}"
            )
        source_snapshot[subset] = {
            **{field: entries[0][field] for field in shared_fields},
            "hub_tree_oid": next(iter(tree_oids), None),
        }
    if len(manifest_hashes) > 1:
        raise ValidationError(
            f"GPU workers disagree on pinned source manifest SHA-256: {sorted(manifest_hashes)}"
        )
    missing_arms = [arm for arm, sources in worker_arm_metadata.items() if not sources]
    if missing_arms:
        raise ValidationError(
            f"worker metadata does not declare expected arms: {missing_arms}"
        )
    if not worker_hardware or any(
        "H100" not in device for device in worker_hardware.values()
    ):
        raise ValidationError(
            f"worker metadata does not consistently identify H100 hardware: {worker_hardware}"
        )
    for arm_id in ("nvfp4-gaussian", "nvfp4-perfectblend"):
        descriptors = worker_arm_metadata[arm_id]
        if not any(
            descriptor["scheme"] == "nvfp4_w4a4"
            and isinstance(descriptor["kernel_config"], dict)
            and descriptor["kernel_config"].get("linear_backend") == "emulation"
            for descriptor in descriptors
        ):
            raise ValidationError(
                f"worker metadata does not show explicit NVFP4 emulation for {arm_id}"
            )

    if revisions != {OBSERVED_HF_HEAD_REVISION}:
        raise ValidationError(
            f"worker dataset revision differs from observed HF Hub HEAD "
            f"{OBSERVED_HF_HEAD_REVISION}: {sorted(revisions)}"
        )
    if len(manifest_hashes) != 1:
        raise ValidationError(
            f"expected one shared pinned source manifest SHA-256, got {sorted(manifest_hashes)}"
        )
    signature = next(iter(signatures))
    dataset_identity = {
        "repo": HF_DATASET_ID,
        "revision": OBSERVED_HF_HEAD_REVISION,
        "revision_source": "worker metadata; Hub HEAD observed 2026-09-24",
        "revision_reported_by_workers": sorted(revisions),
        "revision_pinned_in_eval_cli": False,
        "source_manifest_sha256": next(iter(manifest_hashes), None),
        "source_snapshot_subsets": source_snapshot,
        "source_snapshot_total_rows": sum(
            entry["rows"] for entry in source_snapshot.values()
        ),
        "local_jsonl_inputs_used": False,
    }
    return {
        "dataset_identity": dataset_identity,
        "profile": signature[1],
        "max_requests": signature[2],
        "max_concurrency": signature[3],
        "max_tokens": signature[4],
        "gen_kwargs": json.loads(signature[5]),
        "data_column_mapper": signature[6],
        "spec_method": signature[7],
        "spec_tokens": signature[8],
        "worker_hardware": worker_hardware,
        "worker_arm_metadata": worker_arm_metadata,
        "metadata_files": [
            {"path": run["_path"], "sha256": run["_sha256"]} for run in metadata
        ],
    }


def resolve_arm_roots(results_root: Path) -> dict[str, Path]:
    """Resolve only clean retry trees, never the earlier preflight outputs."""
    gpu0_arms = {"bf16-drafter", "fp8-static-gaussian", "fp8-dynamic-datafree"}
    roots: dict[str, Path] = {}
    for arm_id in EXPECTED_ARM_IDS:
        if arm_id in gpu0_arms:
            path = results_root / "gpu0-thread-limited-retry" / "arms" / arm_id
        else:
            path = results_root / "gpu1-thread-limited-retry" / arm_id
        if not path.is_dir():
            raise ValidationError(
                f"missing clean retry result tree for {arm_id}: {path}"
            )
        roots[arm_id] = path
    return roots


def validate_eval_provenance(
    path: Path,
    *,
    subset: str,
    max_concurrency: int,
) -> str:
    if not path.is_file():
        raise ValidationError(f"missing eval provenance file: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValidationError(f"could not read eval provenance {path}: {exc}") from exc
    command = next(
        (line for line in reversed(lines) if line and not line.startswith("#")), None
    )
    if command is None:
        raise ValidationError(f"no argv recorded in {path}")
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        raise ValidationError(f"could not parse argv in {path}: {exc}") from exc
    if not argv or "throughput" not in argv:
        raise ValidationError(f"eval provenance is not a throughput invocation: {path}")
    flags: dict[str, str] = {}
    for flag in (
        "--dataset",
        "--subsets",
        "--max-requests",
        "--max-concurrency",
        "--max-tokens",
        "--gen-kwargs",
        "--data-column-mapper",
    ):
        if flag in argv:
            index = argv.index(flag)
            if index + 1 < len(argv):
                flags[flag] = argv[index + 1]
    if flags.get("--dataset") != HF_DATASET_ID:
        raise ValidationError(
            f"eval provenance in {path} must use HF dataset {HF_DATASET_ID!r}; "
            f"got {flags.get('--dataset')!r}"
        )
    if flags.get("--subsets") != subset:
        raise ValidationError(
            f"eval provenance in {path} labels the file as {flags.get('--subsets')!r}, "
            f"expected {subset!r}"
        )
    if flags.get("--max-requests") != "200" or flags.get("--max-tokens") != "4096":
        raise ValidationError(
            f"eval provenance in {path} has unexpected request/token limits: {flags}"
        )
    if flags.get("--max-concurrency") != str(max_concurrency):
        raise ValidationError(
            f"eval provenance in {path} has max-concurrency "
            f"{flags.get('--max-concurrency')!r}; expected {max_concurrency}"
        )
    try:
        kwargs = json.loads(flags.get("--gen-kwargs", "{}"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid gen-kwargs JSON in {path}: {exc}") from exc
    if kwargs.get("temperature") != 0 or kwargs.get("seed") != 0:
        raise ValidationError(
            f"eval provenance in {path} does not pin temperature/seed to zero"
        )
    expected_mapper = "kind=generative_column_mapper,column_mappings.text_column=prompt"
    if flags.get("--data-column-mapper") != expected_mapper:
        raise ValidationError(
            f"eval provenance in {path} has unexpected data-column mapper: "
            f"{flags.get('--data-column-mapper')!r}"
        )
    return command


def collect_results(
    attempt_root: Path,
    run_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Path]]:
    arm_roots = resolve_arm_roots(attempt_root)

    records: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    for arm_id in EXPECTED_ARM_IDS:
        arm_root = arm_roots[arm_id]
        arm_provenance = {}
        for name in ARM_PROVENANCE_FILES:
            path = arm_root / name
            if not path.is_file() or path.stat().st_size == 0:
                raise ValidationError(
                    f"missing or empty vLLM provenance artifact: {path}"
                )
            arm_provenance[name] = {"path": str(path.resolve()), "sha256": sha256(path)}
        if arm_id.startswith("nvfp4-"):
            command_text = (arm_root / "vllm_command.txt").read_text(encoding="utf-8")
            if "linear_backend" not in command_text or "emulation" not in command_text:
                raise ValidationError(
                    f"vLLM provenance does not select NVFP4 emulation for {arm_id}: "
                    f"{arm_root / 'vllm_command.txt'}"
                )
        provenance[arm_id] = arm_provenance
        provenance[arm_id]["arm_root"] = str(arm_root.resolve())

        csv_files = sorted(arm_root.rglob("acceptance.csv"))
        by_subset: dict[str, Path] = {}
        for path in csv_files:
            fields, row = read_one_row(path)
            if "subset" not in fields:
                raise ValidationError(f"acceptance.csv has no subset column: {path}")
            subset = row.get("subset", "")
            if subset not in SUBSETS:
                raise ValidationError(f"unexpected subset {subset!r} in {path}")
            if subset in by_subset:
                raise ValidationError(
                    f"duplicate acceptance.csv files for {arm_id}/{subset}: "
                    f"{by_subset[subset]} and {path}"
                )
            by_subset[subset] = path
        missing = [subset for subset in SUBSETS if subset not in by_subset]
        extra_count = len(csv_files) - len(SUBSETS)
        if missing or extra_count:
            raise ValidationError(
                f"{arm_id} coverage mismatch: found {len(csv_files)} acceptance.csv files; "
                f"missing subsets={missing}; expected exactly {len(SUBSETS)}"
            )

        arm_eval_provenance = []
        for subset in SUBSETS:
            csv_path = by_subset[subset]
            record = validate_metrics(
                csv_path,
                arm_id=arm_id,
                expected_subset=subset,
            )
            eval_path = csv_path.parent / "eval_command.txt"
            command = validate_eval_provenance(
                eval_path,
                subset=subset,
                max_concurrency=run_config["max_concurrency"],
            )
            record["dataset"] = HF_DATASET_ID
            record["dataset_revision"] = run_config["dataset_identity"]["revision"]
            record["dataset_source_manifest_sha256"] = run_config["dataset_identity"][
                "source_manifest_sha256"
            ]
            record["eval_command"] = str(eval_path.resolve())
            record["eval_command_sha256"] = sha256(eval_path)
            record["eval_argv"] = command
            record["vllm_provenance_dir"] = str(arm_root.resolve())
            records.append(record)
            arm_eval_provenance.append(
                {
                    "subset": subset,
                    "path": str(eval_path.resolve()),
                    "sha256": record["eval_command_sha256"],
                }
            )
        provenance[arm_id]["eval_commands"] = arm_eval_provenance

    if len(records) != len(ARMS) * len(SUBSETS):
        raise ValidationError(
            f"expected {len(ARMS) * len(SUBSETS)} arm/subset rows; collected {len(records)}"
        )
    add_contrasts(records)
    return records, provenance, arm_roots


def add_contrasts(records: list[dict[str, Any]]) -> None:
    by_key = {(row["arm"], row["subset"]): row for row in records}
    for row in records:
        subset = row["subset"]
        bf16 = by_key[("bf16-drafter", subset)]
        row["acceptance_length_delta_vs_bf16"] = (
            row["acceptance_length"] - bf16["acceptance_length"]
        )
        row["accepted_token_fraction_delta_pp_vs_bf16"] = (
            row["accepted_token_fraction_percent"]
            - bf16["accepted_token_fraction_percent"]
        )
        if row["arm"] == "fp8-static-perfectblend":
            gaussian = by_key[("fp8-static-gaussian", subset)]
            row["acceptance_length_delta_perfectblend_minus_gaussian"] = (
                row["acceptance_length"] - gaussian["acceptance_length"]
            )
        elif row["arm"] == "nvfp4-perfectblend":
            gaussian = by_key[("nvfp4-gaussian", subset)]
            row["acceptance_length_delta_perfectblend_minus_gaussian"] = (
                row["acceptance_length"] - gaussian["acceptance_length"]
            )
        else:
            row["acceptance_length_delta_perfectblend_minus_gaussian"] = ""


def flatten_for_csv(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    max_positions = max(
        (max(row["position_acceptance"], default=-1) for row in records),
        default=-1,
    )
    output = []
    for row in records:
        flat = {
            key: value for key, value in row.items() if key != "position_acceptance"
        }
        for index in range(max_positions + 1):
            rate = row["position_acceptance"].get(index)
            flat[f"acceptance_at_pos_{index + 1}"] = "" if rate is None else rate
        output.append(flat)
    return output


def write_summary_csv(records: list[dict[str, Any]], output: Path) -> None:
    rows = flatten_for_csv(records)
    fields = [
        "arm",
        "arm_label",
        "scheme",
        "calibration",
        "dataset",
        "dataset_revision",
        "dataset_source_manifest_sha256",
        "subset",
        "num_drafts",
        "num_draft_tokens",
        "num_accepted_tokens",
        "accepted_token_fraction",
        "accepted_token_fraction_percent",
        "acceptance_length",
        "acceptance_length_delta_vs_bf16",
        "accepted_token_fraction_delta_pp_vs_bf16",
        "acceptance_length_delta_perfectblend_minus_gaussian",
    ]
    position_fields = sorted(
        (name for name in rows[0] if re.fullmatch(r"acceptance_at_pos_\d+", name)),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    fields.extend(position_fields)
    fields.extend(
        [
            "acceptance_csv",
            "acceptance_csv_sha256",
            "eval_command",
            "eval_command_sha256",
            "vllm_provenance_dir",
            "eval_argv",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def pooled_by_arm(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for arm in ARMS:
        rows = [row for row in records if row["arm"] == arm["id"]]
        drafts = sum(row["num_drafts"] for row in rows)
        proposed = sum(row["num_draft_tokens"] for row in rows)
        accepted = sum(row["num_accepted_tokens"] for row in rows)
        output.append(
            {
                **arm,
                "num_drafts": drafts,
                "num_draft_tokens": proposed,
                "num_accepted_tokens": accepted,
                "acceptance_length": 1 + accepted / drafts,
                "accepted_token_fraction_percent": 100 * accepted / proposed,
            }
        )
    return output


def plot_results(records: list[dict[str, Any]], output_dir: Path) -> dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise ValidationError(
            "matplotlib and numpy are required to create the requested figures"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    by_key = {(row["arm"], row["subset"]): row for row in records}
    matrix = np.array(
        [
            [
                by_key[(arm["id"], subset)]["accepted_token_fraction_percent"]
                for subset in SUBSETS
            ]
            for arm in ARMS
        ],
        dtype=float,
    )

    heat_png = output_dir / "accepted_token_fraction_heatmap.png"
    heat_svg = output_dir / "accepted_token_fraction_heatmap.svg"
    fig, axis = plt.subplots(figsize=(16, 6.4))
    image = axis.imshow(matrix, cmap="viridis", vmin=0, vmax=100, aspect="auto")
    axis.set_xticks(range(len(SUBSETS)), SUBSETS, rotation=30, ha="right")
    axis.set_yticks(range(len(ARMS)), [arm["label"] for arm in ARMS])
    axis.set_xlabel("RedHatAI/speculator_benchmarks subset")
    axis.set_title(
        "Accepted-token fraction by arm and subset\n"
        "Accepted draft tokens / proposed draft tokens (A/P)"
    )
    axis.set_xticks(np.arange(-0.5, len(SUBSETS), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(ARMS), 1), minor=True)
    axis.grid(which="minor", color="white", linestyle="-", linewidth=1.8)
    axis.tick_params(which="minor", bottom=False, left=False)
    norm = image.norm
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            red, green, blue, _ = image.cmap(norm(value))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            color = "black" if luminance > 0.58 else "white"
            axis.text(
                col_index,
                row_index,
                f"{value:.1f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
                fontweight="bold",
            )
    colorbar = fig.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Accepted-token fraction (%)")
    fig.tight_layout()
    fig.savefig(heat_png, dpi=220, bbox_inches="tight")
    fig.savefig(heat_svg, bbox_inches="tight")
    plt.close(fig)

    length_png = output_dir / "acceptance_length_by_subset.png"
    length_svg = output_dir / "acceptance_length_by_subset.svg"
    fig, axis = plt.subplots(figsize=(18, 7.5))
    x = np.arange(len(SUBSETS))
    width = 0.13
    offsets = (np.arange(len(ARMS)) - (len(ARMS) - 1) / 2) * width
    for index, arm in enumerate(ARMS):
        values = [
            by_key[(arm["id"], subset)]["acceptance_length"] for subset in SUBSETS
        ]
        axis.bar(
            x + offsets[index],
            values,
            width=width * 0.95,
            label=arm["label"],
            color=arm["color"],
            edgecolor="#333333",
            linewidth=0.35,
            hatch=arm["hatch"],
        )
    axis.set_xticks(x, SUBSETS, rotation=28, ha="right")
    axis.set_ylabel("Acceptance length (tokens per draft event)")
    axis.set_xlabel("RedHatAI/speculator_benchmarks subset")
    axis.set_title("Acceptance length by subset and drafter arm (seven draft tokens)")
    axis.set_ylim(0, MAX_DRAFT_TOKENS + 1.15)
    axis.set_yticks(np.arange(0, MAX_DRAFT_TOKENS + 2, 1))
    axis.axhline(1, color="#777777", linewidth=0.8, linestyle=":")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(length_png, dpi=220, bbox_inches="tight")
    fig.savefig(length_svg, bbox_inches="tight")
    plt.close(fig)
    return {
        heat_png.name: sha256(heat_png),
        heat_svg.name: sha256(heat_svg),
        length_png.name: sha256(length_png),
        length_svg.name: sha256(length_svg),
    }


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def md_cell(value: str) -> str:
    return value.replace("|", " / ").replace("\n", " ")


def make_markdown(
    *,
    records: list[dict[str, Any]],
    pooled: list[dict[str, Any]],
    dataset_identity: dict[str, Any],
    run_config: dict[str, Any],
    provenance: dict[str, Any],
    arm_roots: dict[str, Path],
    report_manifest_path: Path,
    output_dir: Path,
    report_path: Path,
    results_root: Path,
    attempt_root: Path,
) -> str:
    by_key = {(row["arm"], row["subset"]): row for row in records}
    quantized_arms = [arm for arm in ARMS if arm["id"] != "bf16-drafter"]
    table_header = "| Subset | BF16 | FP8 static G | FP8 static PB | NVFP4 G | NVFP4 PB | FP8 dynamic |\n"
    table_divider = "|---|---:|---:|---:|---:|---:|---:|\n"
    absolute_table = [table_header, table_divider]
    fraction_table = [
        "| Subset | BF16 | FP8 static G | FP8 static PB | NVFP4 G | NVFP4 PB | FP8 dynamic |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    contrast_header = (
        "| Subset | Best quantized arm | Best quantized Δ vs BF16 | "
        "FP8 PerfectBlend − Gaussian | NVFP4 PerfectBlend − Gaussian |\n"
    )
    contrast_divider = "|---|---|---:|---:|---:|\n"
    contrast_table = [contrast_header, contrast_divider]
    for subset in SUBSETS:
        absolute_values = [
            by_key[(arm["id"], subset)]["acceptance_length"] for arm in ARMS
        ]
        absolute_table.append(
            f"| {subset} | "
            + " | ".join(fmt(value) for value in absolute_values)
            + " |\n"
        )
        fraction_values = [
            by_key[(arm["id"], subset)]["accepted_token_fraction_percent"]
            for arm in ARMS
        ]
        fraction_table.append(
            f"| {subset} | "
            + " | ".join(f"{fmt(value, 1)}%" for value in fraction_values)
            + " |\n"
        )
        bf16_length = by_key[("bf16-drafter", subset)]["acceptance_length"]
        best_arm = max(
            quantized_arms,
            key=lambda arm: by_key[(arm["id"], subset)]["acceptance_length"],
        )
        best_length = by_key[(best_arm["id"], subset)]["acceptance_length"]
        fp8_delta = (
            by_key[("fp8-static-perfectblend", subset)]["acceptance_length"]
            - by_key[("fp8-static-gaussian", subset)]["acceptance_length"]
        )
        nvfp4_delta = (
            by_key[("nvfp4-perfectblend", subset)]["acceptance_length"]
            - by_key[("nvfp4-gaussian", subset)]["acceptance_length"]
        )
        contrast_table.append(
            f"| {subset} | {md_cell(best_arm['label'])} | {fmt(best_length - bf16_length, 2)} | "
            f"{fmt(fp8_delta)} | {fmt(nvfp4_delta)} |\n"
        )

    pooled_table = [
        "| Arm | Draft events (D) | Proposed draft tokens (P) | Accepted draft tokens (A) | Acceptance length (1 + A/D) | Accepted-token fraction (A/P) |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ]
    for row in pooled:
        pooled_table.append(
            f"| {md_cell(row['label'])} | {row['num_drafts']:,} | {row['num_draft_tokens']:,} | "
            f"{row['num_accepted_tokens']:,} | {fmt(row['acceptance_length'])} | "
            f"{fmt(row['accepted_token_fraction_percent'], 1)}% |\n"
        )

    settings = (
        f"GuideLLM profile `{run_config['profile']}`, max concurrency "
        f"`{run_config['max_concurrency']}`, `max_requests={run_config['max_requests']}`, "
        f"`max_tokens={run_config['max_tokens']}`, `{run_config['spec_tokens']}` "
        f"speculative tokens (`{run_config['spec_method']}`), "
        f"generation temperature/seed `0/0`."
    )
    run_metadata_lines = "\n".join(
        f"- `{item['path']}` (SHA-256 `{item['sha256']}`)"
        for item in run_config["metadata_files"]
    )
    hardware_lines = "\n".join(
        f"- `{path}`: {device}"
        for path, device in run_config["worker_hardware"].items()
    )
    arm_prov_lines: list[str] = []
    for arm in ARMS:
        artifacts = provenance[arm["id"]]
        references = ", ".join(
            f"`{artifacts[name]['path']}` (SHA-256 `{artifacts[name]['sha256'][:12]}…`)"
            for name in ARM_PROVENANCE_FILES
        )
        arm_prov_lines.append(
            f"- **{arm['label']}** (`{arm_roots[arm['id']].resolve()}`): {references}"
        )
    layout_lines = "\n".join(
        f"- `{arm_id}` → `{path.resolve()}`" for arm_id, path in arm_roots.items()
    )

    subset_interpretations = []
    for subset in SUBSETS:
        bf16_fraction = by_key[("bf16-drafter", subset)][
            "accepted_token_fraction_percent"
        ]
        fraction_extreme = max(
            quantized_arms,
            key=lambda arm: by_key[(arm["id"], subset)][
                "accepted_token_fraction_percent"
            ],
        )
        fraction_best = by_key[(fraction_extreme["id"], subset)][
            "accepted_token_fraction_percent"
        ]
        best_length_arm = max(
            quantized_arms,
            key=lambda arm: by_key[(arm["id"], subset)]["acceptance_length"],
        )
        best_length = by_key[(best_length_arm["id"], subset)]["acceptance_length"]
        bf16_length = by_key[("bf16-drafter", subset)]["acceptance_length"]
        subset_interpretations.append(
            f"- **{subset}:** the highest quantized acceptance length was "
            f"{best_length_arm['label']} ({fmt(best_length)} tokens, "
            f"{fmt(best_length - bf16_length, 2)} tokens vs BF16); the highest quantized "
            f"accepted-token fraction was {fraction_extreme['label']} "
            f"({fmt(fraction_best, 1)}%, "
            f"{fmt(fraction_best - bf16_fraction, 1)} percentage points vs BF16)."
        )

    figure_path = Path(os.path.relpath(output_dir, report_path.parent)).as_posix()

    return f"""# Throughput acceptance results — 2026-09-24

## Experiment

This report compares **six arms: five quantized drafters plus the BF16 drafter reference**. The quantized arms are static FP8 W8A8 with Gaussian and PerfectBlend calibration, NVFP4 W4A4 with Gaussian and PerfectBlend calibration, and the data-free `FP8_DYNAMIC` control. `FP8_DYNAMIC` has one data-free arm; it is not duplicated under calibration labels.

The target/drafter pair is Qwen3-8B with DFlash. Each arm was evaluated once using the HF dataset ID `{dataset_identity["repo"]}` and all nine subset labels. The observed Hub revision was `{dataset_identity["revision"]}` ({dataset_identity["revision_source"]}); worker-reported revisions: `{dataset_identity["revision_reported_by_workers"] or "none"}`. The worker-referenced HF snapshot manifest SHA-256 is `{dataset_identity["source_manifest_sha256"] or "not captured"}` and describes {dataset_identity["source_snapshot_total_rows"]} source rows across the nine subsets. Per-file row counts, SHA-256 digests, and HF tree OIDs are recorded in the report manifest as source-snapshot metadata; they are not local evaluation inputs. The per-subset `eval_command.txt` files confirm the HF dataset ID; no local JSONL path was used.

Each invocation used {settings} The default mapper consumed `prompt`. The results are throughput-mode **acceptance counters only**; this run does not report latency or output-throughput measurements.

## Paths and provenance

- Result root: `{results_root.resolve()}`
- Selected clean attempt: `{attempt_root.resolve()}`
- Clean retry output mapping (GPU0 uses `gpu0-thread-limited-retry/arms/<arm>`; GPU1 uses `gpu1-thread-limited-retry/<arm>`):
{layout_lines}
- Subset outputs: `<arm-root>/subsets/<subset>/acceptance.csv`
- Evaluation dataset: `{dataset_identity["repo"]}`
- Observed HF revision: `{dataset_identity["revision"]}`
- Summary outputs: `{output_dir.resolve()}`
- Input/output hash index: `{report_manifest_path.resolve()}`
- Hardware reported by the workers:
{hardware_lines}

The earlier GuideLLM JSON-file loader failure is retained under
`{results_root.resolve()}/attempt-1-guidellm-jsonfile-failure/` and excluded from
this complete 54-row analysis. The pre-retry attempt-2 GPU0 and GPU1 outputs
under `{attempt_root.resolve()}/arms/` and `{attempt_root.resolve()}/gpu1/` are
also excluded; this report reads the thread-limited retry output roots listed
above. Preserved attempt-2 preflight logs are in
`{attempt_root.resolve()}/preflight-failures/gpu1/` and are excluded from the
acceptance summary.

The result tree retains each subset's `eval_command.txt` and each arm's vLLM command, patch, target checkpoint hash, and drafter checkpoint hash. The GPU worker metadata files were:
{run_metadata_lines}

Arm-level vLLM provenance artifacts:
{chr(10).join(arm_prov_lines)}

The tidy CSV includes all 54 arm/subset observations, raw counters, derived metrics, per-position columns when supplied, input paths, and SHA-256 hashes. Per-position values are unconditional `A_i/D` and are reindexed to draft positions 1 through 7. The report manifest records every source CSV and provenance artifact hash.

## Pooled counter summary

These rows pool observed counters over the nine subsets: `1 + sum(A)/sum(D)` and `sum(A)/sum(P)`. They weight categories by their observed draft and proposed-token counts; they are descriptive pooled values, not equal-category averages.

{"".join(pooled_table)}
## Per-subset acceptance length

Acceptance length is `1 + accepted_tokens / num_drafts` (A/D plus the target token). Values are tokens per draft event.

{"".join(absolute_table)}
## Per-subset accepted-token fraction

Accepted-token fraction is `accepted draft tokens / proposed draft tokens` (`A/P`); the values below are percentages.

{"".join(fraction_table)}
### Per-subset comparison

The first comparison column gives the quantized arm with the highest acceptance length in that subset and its difference from BF16. The next columns show the PerfectBlend-minus-Gaussian acceptance-length differences within static FP8 and NVFP4.

{"".join(contrast_table)}
{chr(10).join(subset_interpretations)}

These one-pass differences describe this evaluation only; they do not establish statistical significance or equivalence.

## Figures

The heatmap annotates the accepted draft-token fraction `A/P` as a percentage. The grouped chart shows acceptance length `1 + A/D` for the same six arms and nine categories.

![Annotated accepted-token fraction heatmap]({figure_path}/accepted_token_fraction_heatmap.png)

![Grouped acceptance-length chart]({figure_path}/acceptance_length_by_subset.png)

## Limits

- One pass per arm and subset; generation seed 0, with no evaluation repeats or uncertainty interval.
- GuideLLM `throughput` profile acceptance counters only. The run does not measure or compare serving speed, latency, TTFT, or ITL.
- The eval command targets the HF dataset ID without a `revision=` CLI option. The report records the observed Hub HEAD, worker-reported revision, and source-manifest hash, but the HEAD observation is not a command-level revision pin.
- NVFP4 W4A4 ran through vLLM's H100 emulation backend. These acceptance observations are not native Blackwell NVFP4 speed results and do not establish Blackwell numerical parity.
- The aggregate rows are weighted by observed counters across categories. Use the per-subset rows and charts for category-level interpretation; do not read the pooled value as category-standardized.
- Calibration comparisons change the full calibration recipe (including the Gaussian versus PerfectBlend input distribution); this single run does not isolate a causal mechanism or prove equivalence.
"""


def write_report_manifest(
    path: Path,
    *,
    results_root: Path,
    attempt_root: Path,
    arm_roots: dict[str, Path],
    dataset_identity: dict[str, Any],
    run_config: dict[str, Any],
    metadata_files: list[dict[str, Any]],
    records: list[dict[str, Any]],
    provenance: dict[str, Any],
    figure_hashes: dict[str, str],
    output_files: list[Path],
) -> None:
    value = {
        "schema": "speculators.throughput-acceptance-summary.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "results_root": str(results_root.resolve()),
        "attempt_root": str(attempt_root.resolve()),
        "arm_root_by_id": {
            arm_id: str(path.resolve()) for arm_id, path in arm_roots.items()
        },
        "arms": [arm["id"] for arm in ARMS],
        "quantized_arm_count": len(ARMS) - 1,
        "bf16_reference_count": 1,
        "subsets": list(SUBSETS),
        "dataset": dataset_identity,
        "evaluation": {
            key: value for key, value in run_config.items() if key != "metadata_files"
        },
        "worker_metadata": metadata_files,
        "acceptance_csvs": [
            {
                "arm": row["arm"],
                "subset": row["subset"],
                "path": row["acceptance_csv"],
                "sha256": row["acceptance_csv_sha256"],
            }
            for row in records
        ],
        "eval_commands": [
            {
                "arm": row["arm"],
                "subset": row["subset"],
                "path": row["eval_command"],
                "sha256": row["eval_command_sha256"],
            }
            for row in records
        ],
        "vllm_provenance": provenance,
        "figures": figure_hashes,
        "other_outputs": {
            str(output.resolve()): sha256(output)
            for output in output_files
            if output.is_file()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--attempt-root",
        type=Path,
        default=None,
        help=f"clean run attempt directory (default: <results-root>/{DEFAULT_ATTEMPT_NAME})",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = args.results_root.expanduser().resolve()
    attempt_root = (
        args.attempt_root.expanduser().resolve()
        if args.attempt_root is not None
        else (results_root / DEFAULT_ATTEMPT_NAME).resolve()
    )
    output_dir = args.output_dir.expanduser().resolve()
    report_path = args.report.expanduser().resolve()

    metadata = find_run_metadata(attempt_root)
    run_config = check_worker_metadata(
        metadata,
        attempt_root=attempt_root,
    )
    dataset_identity = run_config["dataset_identity"]
    if run_config["profile"] != "throughput":
        raise ValidationError(
            f"expected throughput profile, got {run_config['profile']!r}"
        )
    if run_config["max_concurrency"] != 128:
        raise ValidationError(
            f"expected max_concurrency=128, got {run_config['max_concurrency']!r}"
        )
    if run_config["max_requests"] != 200 or run_config["max_tokens"] != 4096:
        raise ValidationError(f"unexpected worker limits in run metadata: {run_config}")
    if (
        run_config["spec_method"] != "dflash"
        or run_config["spec_tokens"] != MAX_DRAFT_TOKENS
    ):
        raise ValidationError(
            f"expected DFlash with {MAX_DRAFT_TOKENS} speculative tokens; "
            f"got method={run_config['spec_method']!r}, tokens={run_config['spec_tokens']!r}"
        )
    if run_config["gen_kwargs"] != {"temperature": 0, "seed": 0}:
        raise ValidationError(
            f"expected temperature=0 and seed=0: {run_config['gen_kwargs']}"
        )
    if run_config["data_column_mapper"] != (
        "kind=generative_column_mapper,column_mappings.text_column=prompt"
    ):
        raise ValidationError(
            f"unexpected data mapper in run metadata: {run_config['data_column_mapper']!r}"
        )

    records, provenance, arm_roots = collect_results(attempt_root, run_config)
    pooled_rows = pooled_by_arm(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "summary.csv"
    write_summary_csv(records, summary_csv)
    figure_hashes = plot_results(records, output_dir)
    manifest_path = output_dir / "report_manifest.json"
    run_meta_refs = run_config["metadata_files"]
    write_report_manifest(
        manifest_path,
        results_root=results_root,
        attempt_root=attempt_root,
        arm_roots=arm_roots,
        dataset_identity=dataset_identity,
        run_config=run_config,
        metadata_files=run_meta_refs,
        records=records,
        provenance=provenance,
        figure_hashes=figure_hashes,
        output_files=[summary_csv],
    )
    report_text = make_markdown(
        records=records,
        pooled=pooled_rows,
        dataset_identity=dataset_identity,
        run_config=run_config,
        provenance=provenance,
        arm_roots=arm_roots,
        report_manifest_path=manifest_path,
        output_dir=output_dir,
        report_path=report_path,
        results_root=results_root,
        attempt_root=attempt_root,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "complete",
                "arm_subset_rows": len(records),
                "arms": list(EXPECTED_ARM_IDS),
                "subsets": list(SUBSETS),
                "summary_csv": str(summary_csv),
                "output_dir": str(output_dir),
                "report": str(report_path),
                "report_manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except ValidationError as exc:
        print(f"validation error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
