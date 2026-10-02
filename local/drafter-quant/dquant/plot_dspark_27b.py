"""Aggregate and plot the Qwen3.8-27B DSpark acceptance matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ARM_LABELS = {
    "bf16-drafter": "BF16 baseline",
    "fp8-block": "FP8 block (data-free)",
    "fp8-dynamic": "FP8 dynamic (data-free)",
    "fp8-gaussian": "FP8 W8A8 (Gaussian)",
    "fp8-real": "FP8 W8A8 (PerfectBlend)",
    "nvfp4-gaussian": "NVFP4 W4A4 (Gaussian)",
    "nvfp4-real": "NVFP4 W4A4 (PerfectBlend)",
    "nvfp4-gptq-gaussian": "NVFP4 GPTQ (Gaussian)",
    "nvfp4-gptq-real": "NVFP4 GPTQ (PerfectBlend)",
    "nvfp4-gptq-imatrix-gaussian": "NVFP4 GPTQ + IMatrix (Gaussian)",
    "nvfp4-gptq-imatrix-real": "NVFP4 GPTQ + IMatrix (PerfectBlend)",
}

COLORS = {
    "bf16-drafter": "#202020",
    "fp8-block": "#9467bd",
    "fp8-dynamic": "#c5a0d5",
    "fp8-gaussian": "#1f77b4",
    "fp8-real": "#65a9d7",
    "nvfp4-gaussian": "#d62728",
    "nvfp4-real": "#e3776f",
    "nvfp4-gptq-gaussian": "#2ca02c",
    "nvfp4-gptq-real": "#98df8a",
    "nvfp4-gptq-imatrix-gaussian": "#ff7f0e",
    "nvfp4-gptq-imatrix-real": "#ffbb78",
}


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect(root: Path) -> list[dict]:
    rows = []
    seen = set()
    for marker_path in sorted(root.rglob("complete.json")):
        marker = read(marker_path)
        for relative, expected in marker["artifacts"].items():
            artifact = marker_path.parent / relative
            if not artifact.is_file() or digest(artifact) != expected:
                raise ValueError(f"Missing or changed result artifact: {artifact}")
        point_identity = marker["identity"]
        run_identity = point_identity["run"]
        arm = run_identity["arm"]
        if arm not in ARM_LABELS:
            raise ValueError(f"Unknown DSpark arm: {arm}")
        subset = marker_path.parent.name
        key = (arm, subset)
        if key in seen:
            raise ValueError(f"Duplicate completed point: {key}")
        seen.add(key)
        attempt = marker_path.parent / marker["attempt"]
        result_path = attempt / "result.json"
        result = read(result_path)
        acceptance = result["acceptance"]
        d = int(acceptance["draft_events"])
        p = int(acceptance["proposed_tokens"])
        a = int(acceptance["accepted_tokens"])
        positions = [int(value) for value in acceptance["accepted_by_position"]]
        if d <= 0 or p != 8 * d or not 0 <= a <= p or len(positions) != 8:
            raise ValueError(f"Invalid eight-token acceptance counters: {result_path}")
        if sum(positions) != a or any(x < y for x, y in zip(positions, positions[1:])):
            raise ValueError(f"Inconsistent per-position counters: {result_path}")
        label = (
            "SPEED-Bench qualitative"
            if subset.startswith("qualitative_")
            else "RedHatAI"
        )
        rows.append(
            {
                "arm": arm,
                "label": ARM_LABELS[arm],
                "suite": label,
                "subset": subset.removeprefix("qualitative_"),
                "dataset_file": subset,
                "requests": int(result["requests"]),
                "draft_events": d,
                "proposed_tokens": p,
                "accepted_tokens": a,
                "acceptance_length": 1 + a / d,
                "accepted_token_fraction": a / p,
                "accepted_by_position": positions,
                "per_position_acceptance": [value / d for value in positions],
                "result_path": str(result_path.resolve()),
                "result_sha256": digest(result_path),
            }
        )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    fields = [
        "arm",
        "label",
        "suite",
        "subset",
        "requests",
        "draft_events",
        "proposed_tokens",
        "accepted_tokens",
        "acceptance_length",
        "accepted_token_fraction",
        "result_path",
        "result_sha256",
    ] + [f"position_{i}_accepted" for i in range(1, 9)]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    **{
                        f"position_{i}_accepted": value
                        for i, value in enumerate(row["accepted_by_position"], 1)
                    },
                }
            )


def pooled(rows: list[dict], suite: str | None = None) -> list[dict]:
    output = []
    for arm, label in ARM_LABELS.items():
        selected = [
            r
            for r in rows
            if r["arm"] == arm and (suite is None or r["suite"] == suite)
        ]
        if not selected:
            continue
        drafts = sum(r["draft_events"] for r in selected)
        proposed = sum(r["proposed_tokens"] for r in selected)
        accepted = sum(r["accepted_tokens"] for r in selected)
        output.append(
            {
                "arm": arm,
                "label": label,
                "suite": suite or "combined",
                "subsets": len(selected),
                "requests": sum(r["requests"] for r in selected),
                "draft_events": drafts,
                "proposed_tokens": proposed,
                "accepted_tokens": accepted,
                "acceptance_length": 1 + accepted / drafts,
                "accepted_token_fraction": accepted / proposed,
            }
        )
    return output


def write_pooled(rows: list[dict], path: Path) -> None:
    fields = [
        "suite",
        "arm",
        "label",
        "subsets",
        "requests",
        "draft_events",
        "proposed_tokens",
        "accepted_tokens",
        "acceptance_length",
        "accepted_token_fraction",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for suite in ("RedHatAI", "SPEED-Bench qualitative"):
        subset_names = sorted({row["subset"] for row in rows if row["suite"] == suite})
        if not subset_names:
            continue
        fig, axes = plt.subplots(
            2, 1, figsize=(max(15, len(subset_names) * 1.35), 10), sharex=True
        )
        for arm, label in ARM_LABELS.items():
            selected = {
                r["subset"]: r for r in rows if r["suite"] == suite and r["arm"] == arm
            }
            if not selected:
                continue
            xs = list(range(len(subset_names)))
            axes[0].plot(
                xs,
                [selected[name]["acceptance_length"] for name in subset_names],
                marker="o",
                linewidth=1.8,
                label=label,
                color=COLORS[arm],
            )
            axes[1].plot(
                xs,
                [
                    100 * selected[name]["accepted_token_fraction"]
                    for name in subset_names
                ],
                marker="o",
                linewidth=1.8,
                label=label,
                color=COLORS[arm],
            )
        axes[0].set_ylabel("Acceptance length (1 + A/D), tokens")
        axes[1].set_ylabel("Accepted-token fraction (A/P), %")
        axes[1].set_xticks(
            range(len(subset_names)), subset_names, rotation=30, ha="right"
        )
        axes[1].set_xlabel("Evaluation subset")
        axes[0].set_title(f"Qwen3.8-27B DSpark acceptance by subset — {suite}")
        for axis in axes:
            axis.grid(axis="y", alpha=0.25)
            axis.set_axisbelow(True)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.5, -0.06),
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.13, 1, 1))
        stem = "redhatai" if suite == "RedHatAI" else "speedbench_qualitative"
        fig.savefig(output / f"{stem}_acceptance.png", dpi=180, bbox_inches="tight")
        fig.savefig(output / f"{stem}_acceptance.svg", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-points", type=int, default=220)
    args = parser.parse_args()
    rows = collect(args.results)
    if len(rows) != args.expected_points:
        raise ValueError(
            f"Expected {args.expected_points} complete points; found {len(rows)}"
        )
    data_manifest = read(args.manifest)
    expected_files = set(data_manifest["files"])
    expected_subsets = {Path(filename).stem for filename in expected_files}
    arms = set(ARM_LABELS)
    if {row["arm"] for row in rows} != arms:
        raise ValueError(
            "The result matrix does not contain exactly the 11 requested arms"
        )
    if {row["dataset_file"] for row in rows} != expected_subsets:
        raise ValueError("The result matrix does not match all frozen dataset subsets")
    by_arm = defaultdict(set)
    expected_counts = {
        Path(name).stem: item["rows"] for name, item in data_manifest["files"].items()
    }
    for row in rows:
        by_arm[row["arm"]].add(row["dataset_file"])
        if row["requests"] != expected_counts[row["dataset_file"]]:
            raise ValueError(
                f"Request count mismatch for {row['arm']} / {row['dataset_file']}"
            )
    if any(subsets != expected_subsets for subsets in by_arm.values()):
        raise ValueError("Each arm must complete every frozen subset")
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output / "acceptance_by_subset.csv")
    pooled_rows = (
        pooled(rows)
        + pooled(rows, "RedHatAI")
        + pooled(rows, "SPEED-Bench qualitative")
    )
    write_pooled(pooled_rows, args.output / "pooled_acceptance.csv")
    plot(rows, args.output)
    print(
        json.dumps(
            {
                "complete_points": len(rows),
                "arms": len({r["arm"] for r in rows}),
                "subsets": len({r["dataset_file"] for r in rows}),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
