"""Aggregate and plot per-subset speculative-acceptance point results.

The input tree is produced by ``dquant.evaluate_arm`` when it is given the
nine frozen subset JSONL files.  Raw counters are pooled before ratios are
derived; category ratios are never averaged.  The report distinguishes
acceptance length (``1 + A/D``) from accepted-token fraction (``A/P``).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ARMS = (
    ("bf16-drafter", "BF16 drafter baseline", "#202020", "-", "o"),
    ("fp8-dynamic-random", "FP8 dynamic | random label", "#9467bd", "--", "s"),
    ("fp8-dynamic-real", "FP8 dynamic | real label", "#c5a0d5", ":", "s"),
    ("fp8-static-random", "FP8 static W8A8 | Gaussian", "#1f77b4", "--", "o"),
    ("fp8-static-real", "FP8 static W8A8 | PerfectBlend", "#65a9d7", "-", "o"),
    ("nvfp4-random", "NVFP4 W4A4 emulated | Gaussian", "#d62728", "--", "^"),
    ("nvfp4-real", "NVFP4 W4A4 emulated | PerfectBlend", "#e3776f", "-", "^"),
)
ARM_INFO = {
    arm: (label, color, linestyle, marker)
    for arm, label, color, linestyle, marker in ARMS
}


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def arm_from_identity(identity: dict) -> str:
    run = identity.get("run", identity)
    arm = run.get("arm") if isinstance(run, dict) else None
    if arm not in ARM_INFO:
        raise ValueError(f"unsupported arm in {identity!r}")
    return arm


def validate_complete(path: Path, root: Path) -> dict:
    marker = load(path)
    if not isinstance(marker, dict) or not isinstance(marker.get("identity"), dict):
        raise ValueError(f"invalid completion marker: {path}")
    for relative, expected in marker.get("artifacts", {}).items():
        artifact = path.parent / relative
        if not artifact.is_file() or digest(artifact) != expected:
            raise ValueError(f"artifact hash mismatch: {artifact}")
    attempt_name = marker.get("attempt")
    if not isinstance(attempt_name, str):
        raise ValueError(f"completion marker has no attempt: {path}")
    result = path.parent / attempt_name / "result.json"
    if not result.is_file():
        raise ValueError(f"missing result: {result}")
    payload = load(result)
    acceptance = payload.get("acceptance")
    if not isinstance(acceptance, dict):
        raise ValueError(f"missing acceptance counters: {result}")
    d = int(acceptance["draft_events"])
    p = int(acceptance["proposed_tokens"])
    a = int(acceptance["accepted_tokens"])
    positions = [int(value) for value in acceptance["accepted_by_position"]]
    if d <= 0 or p <= 0 or a < 0 or sum(positions) != a:
        raise ValueError(f"invalid acceptance counters: {result}")
    return {
        "arm": arm_from_identity(marker["identity"]),
        "subset": path.parent.name,
        "draft_events": d,
        "proposed_tokens": p,
        "accepted_tokens": a,
        "accepted_by_position": positions,
        "complete_path": str(path.resolve()),
        "result_path": str(result.resolve()),
        "result_sha256": digest(result),
        "root": str(root.resolve()),
    }


def collect(root: Path) -> list[dict]:
    records = []
    seen = set()
    for marker in sorted(root.rglob("complete.json")):
        record = validate_complete(marker, root)
        key = (record["arm"], record["subset"])
        if key in seen:
            raise ValueError(f"duplicate completed point for {key}")
        seen.add(key)
        records.append(record)
    return records


def derive(record: dict) -> dict:
    d = record["draft_events"]
    p = record["proposed_tokens"]
    a = record["accepted_tokens"]
    return {
        **record,
        "label": ARM_INFO[record["arm"]][0],
        "acceptance_length": 1 + a / d,
        "accepted_token_fraction": a / p,
        **{
            f"per_position_{index + 1}": value / d
            for index, value in enumerate(record["accepted_by_position"])
        },
    }


def write_csv(records: list[dict], output: Path) -> None:
    positions = max(len(row["accepted_by_position"]) for row in records)
    fields = [
        "arm",
        "label",
        "subset",
        "draft_events",
        "proposed_tokens",
        "accepted_tokens",
        "acceptance_length",
        "accepted_token_fraction",
        *[f"per_position_{i}" for i in range(1, positions + 1)],
        "complete_path",
        "result_path",
        "result_sha256",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def pooled(records: list[dict]) -> list[dict]:
    output = []
    for arm, *_ in ARMS:
        group = [row for row in records if row["arm"] == arm]
        if not group:
            continue
        positions = [
            sum(row["accepted_by_position"][index] for row in group)
            for index in range(len(group[0]["accepted_by_position"]))
        ]
        d = sum(row["draft_events"] for row in group)
        p = sum(row["proposed_tokens"] for row in group)
        a = sum(row["accepted_tokens"] for row in group)
        output.append(
            {
                "arm": arm,
                "label": ARM_INFO[arm][0],
                "subsets": len(group),
                "draft_events": d,
                "proposed_tokens": p,
                "accepted_tokens": a,
                "acceptance_length": 1 + a / d,
                "accepted_token_fraction": a / p,
                "accepted_by_position": positions,
                "per_position_acceptance": [value / d for value in positions],
            }
        )
    return output


def write_pooled(rows: list[dict], output: Path) -> None:
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "arm",
                "label",
                "subsets",
                "draft_events",
                "proposed_tokens",
                "accepted_tokens",
                "acceptance_length",
                "accepted_token_fraction",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["arm"],
                    row["label"],
                    row["subsets"],
                    row["draft_events"],
                    row["proposed_tokens"],
                    row["accepted_tokens"],
                    row["acceptance_length"],
                    row["accepted_token_fraction"],
                ]
            )


def plot(records: list[dict], pooled_rows: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subsets = sorted({row["subset"] for row in records})
    fig, axes = plt.subplots(2, 1, figsize=(15, 11), sharex=True)
    for arm, label, color, linestyle, marker in ARMS:
        points = [row for row in records if row["arm"] == arm]
        if not points:
            continue
        by_subset = {row["subset"]: row for row in points}
        xs = list(range(len(subsets)))
        axes[0].plot(
            xs,
            [by_subset[name]["acceptance_length"] for name in subsets],
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=2,
        )
        axes[1].plot(
            xs,
            [100 * by_subset[name]["accepted_token_fraction"] for name in subsets],
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=2,
        )
    axes[0].set_ylabel("Acceptance length (1 + A/D)\n(tokens per draft event)")
    axes[1].set_ylabel("Accepted-token fraction (A/P) [%]")
    axes[1].set_xlabel("RedHatAI/speculator_benchmarks subset")
    axes[1].set_xticks(range(len(subsets)), subsets, rotation=25, ha="right")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    axes[0].set_title("Seven drafter arms: speculative acceptance by evaluation subset")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output / "acceptance_by_subset.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / "acceptance_by_subset.svg", bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 6))
    positions = range(1, len(pooled_rows[0]["per_position_acceptance"]) + 1)
    for row in pooled_rows:
        label, color, linestyle, marker = ARM_INFO[row["arm"]]
        axis.plot(
            positions,
            [100 * value for value in row["per_position_acceptance"]],
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=2,
        )
    axis.set_title("Pooled unconditional per-position acceptance")
    axis.set_xlabel("Draft position")
    axis.set_ylabel("Accepted at position / draft events [%]")
    axis.set_xticks(list(positions))
    axis.grid(alpha=0.25)
    axis.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "per_position_acceptance.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    records = [derive(row) for row in collect(args.results_root)]
    expected = {
        (arm, subset) for arm, *_ in ARMS for subset in {r["subset"] for r in records}
    }
    actual = {(row["arm"], row["subset"]) for row in records}
    missing = sorted(expected - actual)
    if missing and not args.allow_missing:
        raise SystemExit(f"missing completed points: {missing}")
    if not records:
        raise SystemExit("no completed points found")
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(records, args.output / "subset_results.csv")
    pooled_rows = pooled(records)
    write_pooled(pooled_rows, args.output / "pooled_results.csv")
    plot(records, pooled_rows, args.output)
    manifest = {
        "schema": "speculators.dflash-subset-acceptance-report.v1",
        "arms": [arm for arm, *_ in ARMS],
        "subsets": sorted({row["subset"] for row in records}),
        "completed_points": len(records),
        "missing_points": missing,
        "results_root": str(args.results_root.resolve()),
        "raw_result_sha256": {
            row["result_path"]: row["result_sha256"] for row in records
        },
    }
    (args.output / "report_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
