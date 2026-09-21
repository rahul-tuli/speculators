"""One fixed-concurrency measurement, with immutable identities and verified resume.

Used by evaluate.py point. A failed attempt is retained; a retry gets a new
attempt directory. Completion is committed only after validating all requests,
counters and artifact hashes. The caller supplies model/runtime provenance.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import shlex
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from perf_utils import _accumulate, fetch_metrics, parse_prometheus_metrics

from speculators.provenance import atomic_write


def digest(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def save(path: Path, value) -> None:
    atomic_write(
        path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def acceptance(before: str, after: str, positions: int) -> dict:
    parsed_before = parse_prometheus_metrics(before)
    parsed_after = parse_prometheus_metrics(after)
    required = {
        "vllm:spec_decode_num_drafts",
        "vllm:spec_decode_num_draft_tokens",
        "vllm:spec_decode_num_accepted_tokens",
    }
    for parsed in (parsed_before, parsed_after):
        if not required.issubset({metric.name for metric in parsed}):
            raise ValueError("Missing speculative counters")
    b = _accumulate(parsed_before)
    a = _accumulate(parsed_after)
    d, p, accepted = [a[i] - b[i] for i in range(3)]
    if len(a[3]) != positions:
        raise ValueError(f"Expected {positions} position counters, found {len(a[3])}")
    counts = [value - (b[3][i] if i < len(b[3]) else 0) for i, value in enumerate(a[3])]
    if not all(math.isfinite(v) and v >= 0 for v in [d, p, accepted, *counts]):
        raise ValueError("Nonfinite or decreasing counters (server restart or reset)")
    if p > d * positions or accepted > p or any(v > d for v in counts):
        raise ValueError("Missing or inconsistent speculative counters")
    if sum(counts) != accepted or any(
        x < y for x, y in zip(counts, counts[1:], strict=False)
    ):
        raise ValueError("Position counters disagree with accepted total")
    return {
        "draft_events": d,
        "proposed_tokens": p,
        "accepted_tokens": accepted,
        "accepted_by_position": counts,
        "acceptance_length": 1 + accepted / d if d else None,
        "accepted_token_fraction": accepted / p if p else None,
        "per_position_acceptance": [v / d if d else None for v in counts],
    }


def validate_requests(
    path: Path, expected: int, prompts: list[str] | None = None
) -> dict:
    data = json.loads(path.read_text())
    if len(data.get("benchmarks", [])) != 1:
        raise ValueError("A point must contain exactly one benchmark")
    benchmark = data["benchmarks"][0]
    groups = benchmark["requests"]
    successful = groups.get("successful", [])
    if len(successful) != expected:
        raise ValueError(
            f"Expected {expected} successful requests, got {len(successful)}"
        )
    if any(groups.get(k) for k in ("errored", "incomplete", "cancelled")):
        raise ValueError("Failed/incomplete requests invalidate the point")
    for request in successful:
        tokens = request["output_metrics"]["text_tokens"]
        if not math.isfinite(tokens) or tokens <= 0:
            raise ValueError("Empty or invalid generation")
    if prompts is not None:
        actual = []
        for request in successful:
            body = json.loads(request["request_args"])["body"]
            messages = body["messages"]
            if len(messages) != 1 or messages[0]["role"] != "user":
                raise ValueError("Unexpected chat transformation")
            content = messages[0]["content"]
            text = (
                content
                if isinstance(content, str)
                else "".join(item["text"] for item in content if item["type"] == "text")
            )
            actual.append(text)
        if Counter(actual) != Counter(prompts):
            raise ValueError(
                "Request prompts differ from the frozen dataset "
                "(missing, repeated or transformed examples)"
            )
    return benchmark


def completed(root: Path, identity: dict) -> Path | None:
    marker = root / "complete.json"
    if not marker.exists():
        return None
    record = json.loads(marker.read_text())
    if record["identity"] != identity:
        raise ValueError("Output identity mismatch; select a distinct output directory")
    for name, sha in record["artifacts"].items():
        path = root / name
        if not path.is_file() or digest(path) != sha:
            raise ValueError(f"Completed artifact missing or changed: {path}")
    return root / record["attempt"]


def guidellm(args, dataset: Path, count: int, dest: Path) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "guidellm",
        "run",
        "--backend",
        canonical(
            {
                "kind": "openai_http",
                "target": args.target,
                "max_tokens": args.max_tokens,
                "extras": {"body": {"temperature": 0, "seed": args.eval_seed}},
            }
        ),
        "--data",
        canonical(
            {
                "kind": "json_file",
                "path": str(dataset),
                "load_kwargs": {
                    "split": "train",
                    "cache_dir": str(dest.parent / "dataset-cache"),
                },
            }
        ),
        "--data-column-mapper",
        "kind=generative_column_mapper,column_mappings.text_column=turns",
        "--data-loader",
        f"kind=pytorch,shuffle=false,num_workers=0,samples={count}",
        "--seed",
        f"kind=static,value={args.eval_seed}",
        "--profile",
        f"kind=throughput,max_concurrency={args.max_concurrency}",
        "--constraint",
        f"kind=max_requests,count={count}",
        "--output",
        f"kind=json,path={dest}",
    ]
    dest.with_suffix(".command.txt").write_text(shlex.join(cmd) + "\n")
    with dest.with_suffix(".log").open("w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)  # noqa: S603
    return validate_requests(
        dest,
        count,
        [
            json.loads(line)["turns"]
            for line in dataset.read_text().splitlines()
            if line.strip()
        ],
    )


def snapshot(url: str, path: Path) -> str:
    text = fetch_metrics(url)
    if text is None:
        raise ValueError("Cannot read acceptance counters")
    path.write_text(text)
    return text


def run_point(args) -> None:
    from evaluate import save_eval_provenance  # noqa: PLC0415

    if args.identity is None:
        raise ValueError("point requires --identity with model/runtime provenance")
    supplied = json.loads(args.identity.read_text())
    dataset = Path(args.dataset).resolve(strict=True)
    rows = [
        json.loads(line) for line in dataset.read_text().splitlines() if line.strip()
    ]
    if not rows or not all(
        isinstance(r.get("turns"), str) and r["turns"] for r in rows
    ):
        raise ValueError("point requires materialized single-prompt turns rows")
    if args.max_concurrency < 1 or args.warmup_requests < 1 or args.max_tokens < 1:
        raise ValueError("Concurrency, warm-up and output cap must be positive")
    identity = {
        "schema": 1,
        "run": supplied,
        "dataset_sha256": digest(dataset),
        "requests": len(rows),
        "concurrency": args.max_concurrency,
        "warmup_requests": args.warmup_requests,
        "max_tokens": args.max_tokens,
        "evaluation_seed": args.eval_seed,
        "target_only": args.target_only,
        "temperature": 0,
        "mode": "point",
        "source": {
            p.name: digest(p)
            for p in [
                Path(__file__),
                Path(__file__).with_name("perf_utils.py"),
                Path(__file__).with_name("evaluate.py"),
            ]
        },
    }
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if completed(root, identity):
            print(f"Validated completed point: {root}")
            return
        identity_path = root / "identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError("Partial run identity mismatch; use a distinct directory")
        save(identity_path, identity)
        attempt = root / f"attempt-{time.time_ns()}"
        attempt.mkdir()
        save_eval_provenance(attempt)
        save(attempt / "identity.json", identity)
        try:
            warmup = attempt / "warmup-input.jsonl"
            warmup.write_text(
                "".join(
                    json.dumps(rows[i % len(rows)]) + "\n"
                    for i in range(args.warmup_requests)
                )
            )
            guidellm(args, warmup, args.warmup_requests, attempt / "warmup.json")
            url = args.target.rstrip("/").removesuffix("/v1") + "/metrics"
            before = (
                snapshot(url, attempt / "metrics-before.txt")
                if not args.target_only
                else None
            )
            benchmark = guidellm(args, dataset, len(rows), attempt / "measurement.json")
            after = (
                snapshot(url, attempt / "metrics-after.txt")
                if not args.target_only
                else None
            )
            result = {
                "acceptance": acceptance(before, after, supplied.get("spec_tokens", 7))
                if before is not None
                else None,
                "requests": len(rows),
                "concurrency": args.max_concurrency,
                "metrics": benchmark["metrics"],
                "acceptance_scope": "this_point"
                if before is not None
                else "not_applicable",
            }
            save(attempt / "result.json", result)
            artifacts = {
                str(p.relative_to(root)): digest(p)
                for p in attempt.iterdir()
                if p.is_file()
            }
            save(
                root / "complete.json",
                {"identity": identity, "attempt": attempt.name, "artifacts": artifacts},
            )
        except Exception as exc:
            save(
                attempt / "failure.json",
                {"type": type(exc).__name__, "message": str(exc)},
            )
            raise
