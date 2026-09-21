#!/usr/bin/env python3
"""Plot calibration-data relevance results: random-calibrated vs
perfectblend-calibrated quantized drafters (+ bf16 baseline).

Scans a results directory for eval_<label>/ subdirs produced by
scripts/40_eval.sh, reads:
  - acceptance.csv / *.csv with acceptance metrics (per subset; produced by
    the speculators evaluate.py throughput mode)
  - perf_results.csv (sweep mode: subset, target_rate, output_tps_median,
    ttft_median_ms, itl_median_ms, ...)

Produces (in --output):
  plot1_acceptance_by_subset.png  grouped bars: acceptance length per subset x arm
  plot2_speed_vs_acceptance.png   scatter: output_tps vs acceptance length per arm
  plot3_per_position.png          per-position acceptance curves (if columns exist)
  summary.csv                     tidy table of all parsed metrics

Label convention (from scripts/run_all.sh): <family>-<scheme>-<calib>, e.g.
  baseline-bf16, dflash-fp8-random, dflash-fp8-perfectblend,
  dflash-nvfp4-random, dflash-nvfp4-perfectblend
The arm grouping is derived from the label; unknown labels are kept as-is.

Note: matplotlib is imported lazily (after argument parsing) so that
`python -m dquant.plot_results --help` works on machines without it.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path

logger = logging.getLogger("dquant.plot_results")

plt = None  # populated lazily by _import_matplotlib()


def _import_matplotlib() -> None:
    global plt
    if plt is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        plt = _plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results-dir", default="results")
    p.add_argument("--output", default="results/plots")
    return p.parse_args()


def read_csvs(eval_dir: Path) -> list[dict]:
    rows = []
    for f in sorted(eval_dir.rglob("*.csv")):
        try:
            with open(f) as fh:
                reader = csv.DictReader(fh)
                for r in reader:
                    r["__file"] = f.name
                    rows.append(r)
        except Exception as e:
            logger.warning("[warn] could not parse %s: %s", f, e)
    return rows


def fnum(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def collect(results_dir: Path) -> dict[str, dict]:
    """-> {label: {"acceptance": {subset: x}, "per_pos": {pos: x},
                   "output_tps": x, "ttft": x, "itl": x}}"""
    arms: dict[str, dict] = {}
    for eval_dir in sorted(results_dir.glob("eval_*")):
        label = eval_dir.name[len("eval_"):]
        arm = arms.setdefault(label, {"acceptance": {}, "per_pos": {}})
        for r in read_csvs(eval_dir):
            subset = r.get("subset") or r.get("data") or "overall"
            al = fnum(r.get("acceptance_length"))
            if al is not None:
                arm["acceptance"][subset] = al
            for k, v in r.items():
                m = re.match(r"acceptance_at_pos_?(\d+)", k or "")
                if m and fnum(v) is not None:
                    arm["per_pos"][int(m.group(1))] = fnum(v)
            for key, col in (("output_tps", "output_tps_median"),
                             ("ttft", "ttft_median_ms"),
                             ("itl", "itl_median_ms")):
                x = fnum(r.get(col))
                if x is not None:
                    arm.setdefault(key, []).append(x)
    for arm in arms.values():
        for key in ("output_tps", "ttft", "itl"):
            if isinstance(arm.get(key), list) and arm[key]:
                arm[key] = sum(arm[key]) / len(arm[key])
    return arms


def plot_acceptance(arms: dict, out: Path) -> None:
    subsets = sorted({s for a in arms.values() for s in a["acceptance"]})
    if not subsets:
        logger.info("[skip] plot1: no acceptance_length data found")
        return
    labels = list(arms)
    w = 0.8 / max(len(labels), 1)
    fig, ax = plt.subplots(figsize=(max(8, len(subsets) * 1.2), 5))
    for i, label in enumerate(labels):
        xs = range(len(subsets))
        ys = [arms[label]["acceptance"].get(s) for s in subsets]
        ax.bar([x + i * w for x in xs],
               [y if y is not None else 0 for y in ys],
               width=w, label=label)
    ax.set_xticks([x + 0.4 for x in range(len(subsets))])
    ax.set_xticklabels(subsets, rotation=30, ha="right")
    ax.set_ylabel("acceptance length")
    ax.set_title("Acceptance length by subset: random vs perfectblend calibration")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "plot1_acceptance_by_subset.png", dpi=150)
    logger.info("[ok] plot1_acceptance_by_subset.png")


def plot_speed_vs_acceptance(arms: dict, out: Path) -> None:
    pts = [(a.get("output_tps"),
            (sum(a["acceptance"].values()) / len(a["acceptance"])
             if a["acceptance"] else None), label)
           for label, a in arms.items()]
    pts = [p for p in pts if p[0] is not None and p[1] is not None]
    if not pts:
        logger.info("[skip] plot2: need both output_tps and acceptance data")
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    for x, y, label in pts:
        ax.scatter(x, y, s=80)
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(6, 6), fontsize=8)
    ax.set_xlabel("output tokens/s (median, sweep)")
    ax.set_ylabel("mean acceptance length")
    ax.set_title("Speed vs acceptance — is calibration data worth it?")
    fig.tight_layout()
    fig.savefig(out / "plot2_speed_vs_acceptance.png", dpi=150)
    logger.info("[ok] plot2_speed_vs_acceptance.png")


def plot_per_position(arms: dict, out: Path) -> None:
    if not any(a["per_pos"] for a in arms.values()):
        logger.info("[skip] plot3: no per-position acceptance columns found")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, a in arms.items():
        if not a["per_pos"]:
            continue
        xs = sorted(a["per_pos"])
        ax.plot(xs, [a["per_pos"][x] for x in xs], marker="o", label=label)
    ax.set_xlabel("draft position")
    ax.set_ylabel("acceptance rate")
    ax.set_title("Per-position acceptance")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "plot3_per_position.png", dpi=150)
    logger.info("[ok] plot3_per_position.png")


def write_summary(arms: dict, out: Path) -> None:
    path = out / "summary.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "subset", "acceptance_length",
                    "output_tps_median", "ttft_median_ms", "itl_median_ms"])
        for label, a in arms.items():
            for subset, al in sorted(a["acceptance"].items()):
                w.writerow([label, subset, al, a.get("output_tps", ""),
                            a.get("ttft", ""), a.get("itl", "")])
    logger.info("[ok] %s", path)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(message)s")
    args = parse_args()
    _import_matplotlib()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    arms = collect(Path(args.results_dir))
    if not arms:
        raise SystemExit(f"no eval_* dirs under {args.results_dir} — "
                         "run scripts/40_eval.sh first")
    logger.info("[collect] arms: %s", ", ".join(arms))
    plot_acceptance(arms, out)
    plot_speed_vs_acceptance(arms, out)
    plot_per_position(arms, out)
    write_summary(arms, out)


if __name__ == "__main__":
    main()
