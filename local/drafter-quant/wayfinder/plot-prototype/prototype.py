"""THROWAWAY: three scientific figure layouts using synthetic counters only.

Run: python local/drafter-quant/wayfinder/plot-prototype/prototype.py
Open index.html; variants A/B/C answer which acceptance view should lead.
"""

# Standalone throwaway script; long text is figure captions and embedded HTML.
# ruff: noqa: INP001, E501, T201

import csv
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
CONFIGS = ["qualitative", "throughput_1k", "throughput_2k", "throughput_8k"]
ARMS = [
    "BF16 drafter",
    "FP8 · Gaussian",
    "FP8 · real",
    "NVFP4 · Gaussian",
    "NVFP4 · real",
]
COLORS = ["#555555", "#3978ad", "#dc812e", "#3978ad", "#dc812e"]
VALUES = np.array(
    [
        [4.8, 4.45, 4.61, 3.77, 4.08],
        [5.2, 4.91, 5.06, 4.1, 4.49],
        [4.5, 4.21, 4.34, 3.49, 3.79],
        [3.9, 3.48, 3.74, 2.92, np.nan],
    ]
)
REPEATS = np.array([-0.07, 0.02, 0.05])
plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
)


def frame(title, rows=2, cols=2, figsize=(13, 8)):
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    fig.suptitle("SYNTHETIC — LAYOUT REVIEW ONLY\n" + title, fontsize=15, y=0.985)
    fig.text(
        0.5,
        0.915,
        "Qwen3-8B / DFlash · H100 · NVFP4 = W4A4 emulation · calibration seed 0",
        ha="center",
        fontsize=10,
    )
    return fig, axes.ravel()


def save(fig, name, note):
    fig.text(0.025, 0.022, note, fontsize=9, va="bottom")
    fig.tight_layout(rect=(0, 0.085, 1, 0.90), h_pad=2, w_pad=3)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=160)
    svg = OUT / f"{name}.svg"
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    plt.close(fig)


def observations(config, arm):
    value = VALUES[config, arm]
    # Vary evaluation repeats without implying an empirical distribution.
    return value + REPEATS * (1 + arm * 0.15)


def interval(ax, values, y, color):
    ax.plot([min(values), max(values)], [y, y], color=color, lw=2)
    ax.scatter(values, y + np.array([-0.065, 0, 0.065]), s=15, color=color, alpha=0.65)
    ax.scatter(np.mean(values), y, marker="D", s=42, color=color, zorder=5)


fig, axes = frame("A · Absolute acceptance, with all three evaluation repeats")
for c, ax in enumerate(axes):
    for a in range(5):
        if np.isfinite(VALUES[c, a]):
            interval(ax, observations(c, a), a, COLORS[a])
        else:
            ax.text(
                1.2,
                a,
                "Missing: unsupported (illustrative)",
                color="#777777",
                va="center",
            )
    ax.set(
        yticks=range(5),
        yticklabels=ARMS,
        xlim=(1, 8),
        ylim=(4.6, -0.6),
        title=CONFIGS[c],
        xlabel="Acceptance length = 1 + ΣA / ΣD [tokens/draft event]",
    )
    ax.grid(axis="x", alpha=0.18)
save(
    fig,
    "A_absolute",
    "Concurrency 1 · Per-config pooled counters within each repeat; no cross-config average.\nDots = repeats; diamond = repeat mean; segment = min–max. Descriptive range, not a confidence interval. Missing ≠ zero.",
)

fig, axes = frame("B · Change from the BF16-drafter reference")
for c, ax in enumerate(axes):
    ax.axvline(0, color="#777777", ls="--", lw=1)
    for a in range(1, 5):
        if np.isfinite(VALUES[c, a]):
            interval(ax, observations(c, a) - observations(c, 0), a - 1, COLORS[a])
        else:
            ax.text(
                -1.35, a - 1, "Missing arm: no difference", color="#777777", va="center"
            )
    ax.set(
        yticks=range(4),
        yticklabels=ARMS[1:],
        xlim=(-1.45, 0.25),
        ylim=(3.6, -0.6),
        title=CONFIGS[c],
        xlabel="Acceptance-length difference [tokens/draft event]",
    )
    ax.grid(axis="x", alpha=0.18)
save(
    fig,
    "B_reference_delta",
    "Concurrency 1 · Each difference uses the corresponding evaluation round's BF16 reference; no prompt-paired inference.\nDots = round differences; diamond = mean; segment = min–max. BF16 absolute acceptance remains in companion A / CSV.",
)

fig, axes = frame("C · Complete calibration-recipe comparison")
for c, ax in enumerate(axes):
    for y, (g, r) in enumerate(((1, 2), (3, 4))):
        interval(ax, observations(c, g), y - 0.09, COLORS[g])
        if np.isfinite(VALUES[c, r]):
            interval(ax, observations(c, r), y + 0.09, COLORS[r])
            ax.plot(
                [VALUES[c, g], VALUES[c, r]],
                [y - 0.09, y + 0.09],
                color="#aaaaaa",
                zorder=0,
            )
            ax.text(
                7.85,
                y,
                f"real − Gaussian\n{VALUES[c, r] - VALUES[c, g]:+.2f}",
                ha="right",
                va="center",
                fontsize=9,
            )
        else:
            ax.text(
                4.5,
                y,
                "Real missing; contrast unavailable",
                fontsize=9,
                color="#777777",
            )
    ax.axvline(VALUES[c, 0], color="#555555", ls="--", label="BF16 mean")
    ax.set(
        yticks=[0, 1],
        yticklabels=["FP8", "NVFP4 (emulated)"],
        xlim=(1, 8),
        ylim=(1.5, -0.5),
        title=CONFIGS[c],
        xlabel="Acceptance length [tokens/draft event]",
    )
    ax.grid(axis="x", alpha=0.18)
save(
    fig,
    "C_recipe_pairs",
    "Concurrency 1 · Blue = Gaussian; orange = real; dashed = BF16 mean (full repeats in A / CSV).\nDots = repeats; diamond = mean; segment = min–max. Recipe effects combine tokens, masks, lengths, and hidden states.",
)

fig, axes = frame("Companion acceptance views · one illustrative workload")
ax = axes[0]
for a in range(5):
    interval(ax, (observations(0, a) - 1) / 7, a, COLORS[a])
ax.set(
    yticks=range(5),
    yticklabels=ARMS,
    ylim=(4.6, -0.6),
    xlim=(0, 1),
    xlabel="Accepted-token fraction = ΣA / ΣP",
    title="qualitative · concurrency 1",
)
ax = axes[1]
for a in range(5):
    weights = np.array([1, 0.9, 0.8, 0.65, 0.5, 0.35, 0.2])
    rates = weights * (VALUES[0, a] - 1) / weights.sum()
    ax.plot(
        range(1, 8),
        rates,
        marker="o",
        color=COLORS[a],
        ls="--" if ARMS[a].startswith("NVFP4") else "-",
        label=ARMS[a],
    )
ax.set(
    xlabel="Draft position",
    ylabel="ΣAᵢ / ΣD (all draft events)",
    ylim=(0, 1),
    title="Unconditional per-position acceptance",
)
ax.legend(fontsize=8)
ax = axes[2]
for s in range(3):
    for a in (1, 2):
        vals = observations(0, a) + s * (0.03 if a == 1 else -0.025)
        x = s + (-0.1 if a == 1 else 0.1)
        ax.plot([x, x], [min(vals), max(vals)], color=COLORS[a])
        ax.scatter([x] * 3, vals, color=COLORS[a], s=15)
        ax.scatter(x, np.mean(vals), color=COLORS[a], marker="D")
ax.set(
    xticks=range(3),
    xlabel="Calibration seed (same real examples)",
    ylabel="Acceptance length",
    title="FP8 · seeds kept separate",
)
axes[3].axis("off")
axes[3].text(
    0,
    0.9,
    "Coverage is part of each figure\n\n• Target-only acceptance: not applicable\n• Unsupported: blank value + explicit reason\n• Incomplete manifest: no pooled summary\n• All repeats visible; never labelled CI\n• Same layouts for category detail pages\n• Hardware / backend / model get separate pages",
    va="top",
    fontsize=12,
)
save(
    fig,
    "acceptance_companions",
    "SYNTHETIC examples only · D = draft events; A = accepted draft tokens; P = proposed draft tokens.\nPer-position curves illustrate means; final plots retain three repeats and range. Seed 0 first; seeds 1–2 are later panels.",
)

fig, axes = frame(
    "Later performance views · qualitative, fixed matched operating points",
    figsize=(14, 9),
)
loads = np.array([1, 8, 32, 128])
tps = np.array(
    [
        [80, 410, 830, 1000],
        [95, 510, 960, 1100],
        [98, 530, 1010, 1160],
        [45, 230, 390, 440],
        [49, 255, 425, np.nan],
        [42, 270, 630, 850],
    ],
    dtype=float,
)
labels = ARMS + ["Target-only"]
colors = COLORS + ["#9971ac"]
for a in range(6):
    y = tps[a]
    axes[0].plot(
        loads,
        y,
        color=colors[a],
        marker="o",
        ls="--" if a in (3, 4) else "-",
        label=labels[a],
    )
    axes[0].errorbar(loads, y, yerr=y * 0.03, fmt="none", ecolor=colors[a], alpha=0.6)
axes[0].set(
    xscale="log",
    xticks=loads,
    xticklabels=loads,
    xlabel="Concurrency",
    ylabel="Output tokens / measured wall second",
    title="Absolute throughput; gaps stay missing",
)
axes[0].legend(fontsize=7, ncol=2)
latencies = np.array([12500, 10900, 10500, 21800, 20300, 24200])
for a in range(6):
    interval(axes[1], latencies[a] * np.array([0.97, 1, 1.03]), a, colors[a])
axes[1].set(
    yticks=range(6),
    yticklabels=labels,
    ylim=(5.6, -0.6),
    xlabel="Request end-to-end latency [ms]",
    title="Concurrency 1 · per-repeat request median",
)
for a in range(1, 5):
    for offset, ref, marker in [(-0.1, 0, "o"), (0.1, 5, "s")]:
        axes[2].scatter(
            tps[a, 2] / tps[ref, 2], a + offset, marker=marker, color=colors[a]
        )
axes[2].axvline(1, color="#777777", ls="--")
axes[2].set(
    yticks=range(1, 5),
    yticklabels=ARMS[1:],
    ylim=(4.6, 0.4),
    xlabel="Output-throughput ratio [×]",
    title="Concurrency 32 · two named speed references",
)
axes[2].text(
    0.02,
    0.03,
    "○ / BF16 drafter    □ / target-only",
    transform=axes[2].transAxes,
    fontsize=9,
)
for a in range(5):
    # These stand for fresh counters at concurrency 32, not reused C=1 values.
    al32 = VALUES[0, a] - 0.17 + 0.025 * a
    axes[3].errorbar(
        al32, tps[a, 2], xerr=0.08, yerr=tps[a, 2] * 0.03, fmt="o", color=colors[a]
    )
    axes[3].annotate(
        ARMS[a],
        (al32, tps[a, 2]),
        xytext=(4, 5),
        textcoords="offset points",
        fontsize=8,
    )
axes[3].set(
    xlabel="Acceptance length measured at concurrency 32",
    ylabel="Output tokens / measured wall second",
    title="Same workload, hardware, load and run interval",
)
save(
    fig,
    "performance_companions",
    "SYNTHETIC · Speed panels never average loads. NVFP4 H100 timings describe emulation only.\nFinal figures show three repeats, mean and range; request median is a separate statistic. TTFT / ITL get companion pages. Target-only has no scatter acceptance.",
)

rows = []
for c, config in enumerate(CONFIGS):
    for a, arm in enumerate(ARMS):
        for repeat in range(3):
            missing = not np.isfinite(VALUES[c, a])
            d = 10000
            accepted = None if missing else round((observations(c, a)[repeat] - 1) * d)
            rows.append(
                {
                    "synthetic": "true",
                    "run_id": f"synthetic-{c}-{a}-{repeat}",
                    "config": config,
                    "aggregation": "pooled_config",
                    "arm": arm,
                    "concurrency": 1,
                    "calibration_seed": "" if a == 0 else 0,
                    "evaluation_repeat": repeat,
                    "status": "unsupported" if missing else "complete",
                    "draft_events": "" if missing else d,
                    "accepted_draft_tokens": "" if missing else accepted,
                    "proposed_draft_tokens": "" if missing else 7 * d,
                    "acceptance_length": "" if missing else 1 + accepted / d,
                }
            )
with (OUT / "synthetic_layout_data.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

html = r"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Synthetic plot prototype</title>
<style>body{font:17px system-ui;margin:24px auto;max-width:1350px;padding:0 20px 100px;color:#222}img{width:100%}nav{position:fixed;bottom:15px;left:50%;transform:translateX(-50%);background:#182a3c;color:white;padding:14px;border-radius:12px}button{padding:8px;margin:0 8px}a{color:#176197}.warning{background:#fff0c7;padding:14px}</style>
<h1>Acceptance plot layout review</h1><p class="warning"><b>SYNTHETIC DATA — NO EXPERIMENT RESULTS.</b> Throwaway prototype. Three layouts, one selected experiment protocol.</p>
<p>Proposed default: A for absolute results, C for recipe comparisons, B in the supplement. Full category detail accompanies per-configuration summaries.</p>
<p><a href="schema-proposal.md">Proposed aggregation, CSV schema and publication contract</a> · <a href="synthetic_layout_data.csv">Synthetic layout input</a></p>
<img id="main" alt="Synthetic acceptance figure"><pre id="state"></pre>
<h2>Acceptance companions</h2><img src="acceptance_companions.png" alt="Synthetic acceptance companion plots">
<h2>Later latency and throughput</h2><img src="performance_companions.png" alt="Synthetic speed companion plots">
<nav><button onclick="change(-1)">←</button><span id="label"></span><button onclick="change(1)">→</button></nav>
<script>const keys=['A','B','C'];const names=['Absolute acceptance','Difference from BF16','Recipe pairs'];const files=['A_absolute','B_reference_delta','C_recipe_pairs'];let i=Math.max(0,keys.indexOf(new URLSearchParams(location.search).get('variant')));function render(){document.getElementById('main').src=files[i]+'.png';document.getElementById('label').textContent=keys[i]+' · '+names[i];document.getElementById('state').textContent='Layout '+keys[i]+' | synthetic | Qwen3-8B/DFlash | H100 | NVFP4 emulated | concurrency 1 | seed 0\nSummary: per-config pooled counters within repeat; 3 repeat values + mean + range; missing kept blank.';}function change(d){i=(i+d+3)%3;history.replaceState(null,'','?variant='+keys[i]);render();}addEventListener('keydown',e=>{if(['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName)||e.target.isContentEditable)return;if(e.key==='ArrowLeft')change(-1);if(e.key==='ArrowRight')change(1);});render();</script></html>"""
(OUT / "index.html").write_text(html)
print(OUT / "index.html")
