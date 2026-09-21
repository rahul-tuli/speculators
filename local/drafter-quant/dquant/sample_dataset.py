#!/usr/bin/env python3
"""Build an even, task-mixed 2048-sample subset of
inference-optimization/Qwen3-8B-Regenerated-Collection ("perfectblend-style").

The source dataset is 13 task-source JSONL files (verified via the HF API):
  autoif, evol_codealpaca, lmsys_arena, magpie, metamathqa, nemotron_chat,
  nemotron_math, nemotron_stem, orca_math, tulu3, ultrachat, ultrafeedback,
  ultrainteract   -- all regenerated on-policy with Qwen3-8B.

Each row has a ShareGPT-style `conversations` column ({from, value} turns),
which is exactly what `speculators prepare-data` consumes (natural-language
path; render endpoint required downstream).

Sampling (--strategy proportional, the default): each source file keeps the
same share of the subset as it has in the full collection — quota_i =
num_samples * rows_i / total_rows, largest-remainder rounding, capped at the
file's available rows with deficits redistributed to files with spare
capacity. Deterministic via --seed, without replacement. `--strategy uniform`
gives the old equal-quota behavior (2048 // 13 = 157/158 per file) for
ablation.

Output:
  <output>/perfectblend_<N>.jsonl   combined subset (one row per line, with a
                                    "source" tag added per row)
  <output>/mix_manifest.json        per-file quotas/counts for provenance

Then run scripts/20_capture_hidden_states.sh to tokenize (prepare-data) and
capture hidden states (generate-offline-data).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

logger = logging.getLogger("dquant.sample_dataset")

REPO_ID = "inference-optimization/Qwen3-8B-Regenerated-Collection"

SOURCE_FILES = [
    "autoif_train_Qwen3-8B.jsonl",
    "evol_codealpaca_train_Qwen3-8B.jsonl",
    "lmsys_arena_train_Qwen3-8B.jsonl",
    "magpie_output.jsonl",
    "metamathqa_train_Qwen3-8B.jsonl",
    "nemotron_chat_Qwen3-8B.jsonl",
    "nemotron_math_Qwen3-8B.jsonl",
    "nemotron_stem_Qwen3-8B.jsonl",
    "orca_math_train_Qwen3-8B.jsonl",
    "tulu3_Qwen3-8B.jsonl",
    "ultrachat_output.jsonl",
    "ultrafeedback_train_sft_Qwen3-8B.jsonl",
    "ultrainteract_train_Qwen3-8B.jsonl",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--num-samples", type=int, default=2048)
    p.add_argument("--strategy", choices=["proportional", "uniform"],
                   default="proportional",
                   help="proportional (default): preserve each task source's "
                        "share of the full collection; uniform: equal quota "
                        "per source (ablation option).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="/data/fast/drafter-quant/data/perfectblend_2048")
    p.add_argument("--cache-dir", default=None,
                   help="HF download cache (defaults to ~/.cache/huggingface). "
                        "Point at a big disk if $HOME is small.")
    return p.parse_args()


def load_source(fname: str, cache_dir: str | None) -> list[dict]:
    import shutil
    import urllib.request
    from pathlib import Path
    from huggingface_hub import get_token, hf_hub_download

    try:
        path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset",
                               filename=fname, cache_dir=cache_dir)
    except OSError as e:
        if "Consistency check failed" not in str(e):
            raise
        # Upstream repo metadata has a size mismatch with the served blob.
        # Fall back to direct HTTP streaming download.
        target_dir = Path(cache_dir) if cache_dir else Path("/data/fast/hf_cache/direct_downloads")
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / fname
        if not path.exists() or path.stat().st_size == 0:
            url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{fname}"
            req = urllib.request.Request(url)
            token = get_token()
            if token:
                req.add_header("Authorization", f"Bearer {token}")
            tmp_path = path.with_suffix(".tmp")
            with urllib.request.urlopen(req) as resp, open(tmp_path, "wb") as out:
                shutil.copyfileobj(resp, out)
            tmp_path.replace(path)

    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_quotas(counts: dict[str, int], num_samples: int,
                   strategy: str) -> dict[str, int]:
    """Per-file sample quotas.

    proportional: quota_i = num_samples * rows_i / total_rows with
    largest-remainder rounding; capped at rows_i, deficits redistributed to
    files with spare capacity.
    uniform: num_samples // n_files, remainder to the first files (legacy).
    """
    if strategy == "uniform":
        base, rem = divmod(num_samples, len(counts))
        files = list(counts)
        return {f: base + (1 if i < rem else 0) for i, f in enumerate(files)}

    total = sum(counts.values())
    raw = {f: num_samples * c / total for f, c in counts.items()}
    quotas = {f: min(int(raw[f]), counts[f]) for f in counts}
    remainder = num_samples - sum(quotas.values())
    # Largest fractional parts first; skip files already at capacity.
    order = sorted(counts, key=lambda f: raw[f] - int(raw[f]), reverse=True)
    i = 0
    while remainder > 0 and any(quotas[f] < counts[f] for f in counts):
        f = order[i % len(order)]
        if quotas[f] < counts[f]:
            quotas[f] += 1
            remainder -= 1
        i += 1
    return quotas


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(message)s")
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load every source once; counts drive the proportional quotas.
    source_rows = {f: load_source(f, args.cache_dir) for f in SOURCE_FILES}
    counts = {f: len(rows) for f, rows in source_rows.items()}
    total_rows = sum(counts.values())
    quotas = compute_quotas(counts, args.num_samples, args.strategy)

    manifest = {"repo": REPO_ID, "num_samples_requested": args.num_samples,
                "strategy": args.strategy, "seed": args.seed,
                "total_rows_in_collection": total_rows, "sources": {}}
    selected: list[dict] = []

    for fname in SOURCE_FILES:
        rows = source_rows[fname]
        take = min(quotas[fname], len(rows))
        sample = rng.sample(rows, take)
        for r in sample:
            r["source"] = fname
        selected.extend(sample)
        manifest["sources"][fname] = {
            "available": len(rows),
            "collection_share": round(len(rows) / total_rows, 6),
            "quota": quotas[fname], "taken": take,
            "subset_share": round(take / args.num_samples, 6),
        }
        logger.info("[mix] %s: available=%d (%.1f%% of collection) quota=%d taken=%d",
                    fname, len(rows), 100 * len(rows) / total_rows,
                    quotas[fname], take)

    rng.shuffle(selected)
    out_path = out_dir / f"perfectblend_{args.num_samples}.jsonl"
    with open(out_path, "w") as f:
        for r in selected:
            f.write(json.dumps(r) + "\n")

    manifest["num_samples_actual"] = len(selected)
    manifest["output"] = str(out_path)
    with open(out_dir / "mix_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("[done] wrote %d rows -> %s", len(selected), out_path)
    logger.info("[next] run: bash scripts/20_capture_hidden_states.sh  (DATA=%s)",
                out_path)


if __name__ == "__main__":
    main()
