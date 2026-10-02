#!/usr/bin/env python3
"""Extract 2,048 equi-ratio (stratified) calibration samples from
shanjiaz/OpenPerfectBlend-Qwen38-27B-regenerated.

Preserves the exact original fields (id, primary_id, text, metadata,
input_ids, loss_mask) as stored in the source dataset.
Maintains the proportional ratio across the 8 dataset sources using
stratified reservoir sampling with seed=0.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import random
import time
from collections import Counter
from pathlib import Path

logger = logging.getLogger("dquant.sample_dataset_27b")

REPO_ID = "shanjiaz/OpenPerfectBlend-Qwen38-27B-regenerated"
SOURCE_REPO_ID = "mlabonne/open-perfectblend"

# Exact counts in mlabonne/open-perfectblend (1,420,909 total rows)
SOURCE_COUNTS = {
    "meta-math/MetaMathQA": 386043,
    "openbmb/UltraInteract_sft": 272266,
    "mlabonne/ultrachat_200k_sft": 207865,
    "HuggingFaceH4/orca-math-word-problems-200k": 199707,
    "HuggingFaceH4/ultrafeedback_binarized": 125255,
    "theblackcat102/evol-codealpaca-v1": 111272,
    "Post-training-Data-Flywheel/AutoIF-instruct-61k": 61492,
    "mlabonne/lmsys-arena-human-preference-55k-sharegpt": 57009,
}

SOURCE_CATEGORIES = {
    "meta-math/MetaMathQA": "math",
    "openbmb/UltraInteract_sft": "reasoning_math_code",
    "mlabonne/ultrachat_200k_sft": "general_chat",
    "HuggingFaceH4/orca-math-word-problems-200k": "math",
    "HuggingFaceH4/ultrafeedback_binarized": "chat_preference",
    "theblackcat102/evol-codealpaca-v1": "coding",
    "Post-training-Data-Flywheel/AutoIF-instruct-61k": "instruction_following",
    "mlabonne/lmsys-arena-human-preference-55k-sharegpt": "chat_preference",
}


def compute_quotas(counts: dict[str, int], num_samples: int) -> dict[str, int]:
    """Compute per-source quotas using largest-remainder rounding."""
    total = sum(counts.values())
    raw = {f: num_samples * c / total for f, c in counts.items()}
    quotas = {f: min(int(raw[f]), counts[f]) for f in counts}
    remainder = num_samples - sum(quotas.values())
    order = sorted(counts, key=lambda f: raw[f] - int(raw[f]), reverse=True)
    i = 0
    while remainder > 0 and any(quotas[f] < counts[f] for f in counts):
        f = order[i % len(order)]
        if quotas[f] < counts[f]:
            quotas[f] += 1
            remainder -= 1
        i += 1
    return quotas


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="/data/fast/datasets/shanjiaz_raw/data/train.jsonl.gz",
        help="Path to downloaded train.jsonl.gz",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/fast/drafter-quant/data/perfectblend_27b_2048",
        help="Output directory",
    )
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    quotas = compute_quotas(SOURCE_COUNTS, args.num_samples)
    total_quota = sum(quotas.values())
    assert total_quota == args.num_samples, (
        f"Quota sum {total_quota} != {args.num_samples}"
    )

    logger.info("Target sampling quotas for %d samples:", args.num_samples)
    for src, q in quotas.items():
        ratio = SOURCE_COUNTS[src] / sum(SOURCE_COUNTS.values())
        logger.info(
            "  %s: quota=%d (target=%.4f%%, allocated=%.4f%%)",
            src,
            q,
            ratio * 100,
            (q / args.num_samples) * 100,
        )

    # Stratified Reservoir Sampling
    rng = random.Random(args.seed)
    reservoirs: dict[str, list[str]] = {src: [] for src in quotas}
    seen_counts: dict[str, int] = Counter()
    all_source_counts: dict[str, int] = Counter()

    logger.info(
        "Starting single-pass stratified reservoir sampling from %s...", input_path
    )
    t0 = time.time()
    total_processed = 0

    needle = '"source": "'
    with gzip.open(input_path, "rt", encoding="utf-8") as f:
        for line in f:
            total_processed += 1
            if total_processed % 200000 == 0:
                elapsed = time.time() - t0
                speed = total_processed / elapsed
                logger.info(
                    "Processed %d rows (%.1f rows/sec)...", total_processed, speed
                )

            # Fast source extraction
            pos = line.find(needle)
            if pos == -1:
                continue
            end = line.find('"', pos + 11)
            if end == -1:
                continue
            src = line[pos + 11 : end]

            all_source_counts[src] += 1
            if src not in quotas:
                continue

            k = quotas[src]
            seen_counts[src] += 1
            idx = seen_counts[src] - 1

            if len(reservoirs[src]) < k:
                reservoirs[src].append(line)
            else:
                j = rng.randint(0, idx)
                if j < k:
                    reservoirs[src][j] = line

    elapsed = time.time() - t0
    logger.info(
        "Completed pass: %d total rows in %.2fs (%.1f rows/sec)",
        total_processed,
        elapsed,
        total_processed / elapsed,
    )

    logger.info("Source counts across full dataset (%d rows):", total_processed)
    for src, count in all_source_counts.most_common():
        pct = (count / total_processed) * 100
        logger.info("  %s: %d (%.4f%%)", src, count, pct)

    # Collect and shuffle
    selected_lines: list[str] = []
    for src in sorted(quotas.keys()):
        sample = reservoirs[src]
        assert len(sample) == quotas[src], (
            f"Reservoir for {src} has {len(sample)} != {quotas[src]}"
        )
        selected_lines.extend(sample)

    assert len(selected_lines) == args.num_samples, (
        f"Total sampled {len(selected_lines)} != {args.num_samples}"
    )
    rng.shuffle(selected_lines)

    # Write output calibration_data.jsonl
    out_file = out_dir / "calibration_data.jsonl"
    logger.info("Writing %d samples to %s...", len(selected_lines), out_file)
    with open(out_file, "w", encoding="utf-8") as f:
        for line in selected_lines:
            f.write(line if line.endswith("\n") else line + "\n")

    # Compute checksum
    sha256_hash = compute_sha256(out_file)

    # Verify rows & check breakdown
    sampled_counts = Counter()
    sampled_indices = set()
    sampled_primary_ids = set()
    sample_categories = Counter()

    for line in selected_lines:
        row = json.loads(line)
        src = row["metadata"]["source"]
        sampled_counts[src] += 1
        sampled_indices.add(row["metadata"]["idx"])
        sampled_primary_ids.add(row["primary_id"])
        sample_categories[SOURCE_CATEGORIES.get(src, "other")] += 1

    manifest = {
        "dataset": REPO_ID,
        "source_dataset": SOURCE_REPO_ID,
        "num_samples_requested": args.num_samples,
        "num_samples_actual": len(selected_lines),
        "seed": args.seed,
        "total_rows_in_collection": total_processed,
        "unique_sampled_source_indices": len(sampled_indices),
        "unique_sampled_primary_ids": len(sampled_primary_ids),
        "output_file": str(out_file),
        "sha256": sha256_hash,
        "sources": {},
        "categories": {cat: count for cat, count in sample_categories.most_common()},
    }

    for src in SOURCE_COUNTS:
        avail_mlabonne = SOURCE_COUNTS[src]
        avail_actual = all_source_counts.get(src, 0)
        taken = sampled_counts.get(src, 0)
        manifest["sources"][src] = {
            "category": SOURCE_CATEGORIES.get(src, "unknown"),
            "available_full_collection": avail_actual,
            "collection_share": round(avail_actual / total_processed, 6),
            "available_source_mlabonne": avail_mlabonne,
            "source_mlabonne_share": round(
                avail_mlabonne / sum(SOURCE_COUNTS.values()), 6
            ),
            "quota": quotas[src],
            "taken": taken,
            "subset_share": round(taken / args.num_samples, 6),
        }

    manifest_file = out_dir / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Wrote manifest to %s", manifest_file)
    logger.info("Done! Verification:")
    logger.info("  File: %s", out_file)
    logger.info("  Size: %d bytes", out_file.stat().st_size)
    logger.info("  SHA256: %s", sha256_hash)
    logger.info("  Total rows: %d", len(selected_lines))
    logger.info("  Unique source indices: %d", len(sampled_indices))
    logger.info("  Unique primary IDs: %d", len(sampled_primary_ids))


if __name__ == "__main__":
    main()
