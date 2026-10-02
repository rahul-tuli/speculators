#!/usr/bin/env python3
"""Freeze RedHatAI and SPEED-Bench inputs for the Qwen3.8-27B DSpark study."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PYTHON = ROOT / ".venv/bin/python"
OUT = Path("/data/fast/drafter-quant/data/dspark-qwen3.8-27b-eval")
SPEEDBENCH_SOURCE = Path("/data/fast/drafter-quant/data/speedbench")
CALIBRATION = Path(
    "/data/fast/datasets/Qwen--Qwen3.8-27B-fp8-layers-4-12-20-28-36-44-52-60-"
    "h5120-n2048-seq2048-seed0_prepared"
)
CALIBRATION_SOURCE_MANIFEST = Path(
    "/data/fast/drafter-quant/data/perfectblend_27b_2048/manifest.json"
)
TARGET_REPO = "Qwen/Qwen3.8-27B"
TARGET_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
TOKENIZER = Path(
    "/root/.cache/huggingface/models--Qwen--Qwen3.8-27B/snapshots/" + TARGET_REVISION
)
REDHATAI_REPO = "RedHatAI/speculator_benchmarks"
REDHATAI_REVISION = "2ae86affa2cb97a972b7fc681dd51c04fbff083e"
REDHATAI_SNAPSHOT = Path(
    "/data/fast/hf_cache/hub/datasets--RedHatAI--speculator_benchmarks/"
    "snapshots/" + REDHATAI_REVISION
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_immutable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"Refusing to overwrite changed frozen file: {path}")
        return
    path.write_text(content)


def normalized_hash(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def main() -> None:
    for path in (
        PYTHON,
        TOKENIZER / "tokenizer.json",
        CALIBRATION / "dataset_info.json",
        CALIBRATION_SOURCE_MANIFEST,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    for path in (SPEEDBENCH_SOURCE / "qualitative.jsonl", REDHATAI_SNAPSHOT):
        if not path.exists():
            raise FileNotFoundError(path)

    speed_root = OUT / "speedbench"
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(ROOT / "local/drafter-quant") + os.pathsep + env.get("PYTHONPATH", "")
    )
    subprocess.run(
        [
            str(PYTHON),
            "-m",
            "dquant.freeze_eval_data",
            "--data-dir",
            str(SPEEDBENCH_SOURCE),
            "--output",
            str(speed_root),
            "--tokenizer",
            str(TOKENIZER),
            "--calibration-data",
            str(CALIBRATION),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )

    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    target_config = AutoConfig.from_pretrained(TOKENIZER, local_files_only=True)
    max_prompt_tokens = 0
    files: dict[str, dict] = {}
    redhat_prompt_hashes: dict[str, list[str]] = {}

    from datasets import load_from_disk

    calibration = load_from_disk(str(CALIBRATION))
    calibration_prompts: set[str] = set()
    calibration_responses: set[str] = set()
    for row in calibration:
        decoded = tokenizer.decode(row["input_ids"], skip_special_tokens=False)
        for role, content in re.findall(
            r"<\|im_start\|>(user|assistant)\n(.*?)(?:<\|im_end\|>|$)",
            decoded,
            re.S,
        ):
            (calibration_prompts if role == "user" else calibration_responses).add(
                normalized_hash(content)
            )

    for subset in REDHATAI_SUBSETS:
        source = REDHATAI_SNAPSHOT / f"{subset}.jsonl"
        output = OUT / "redhatai" / f"{subset}.jsonl"
        prompts = []
        for line in source.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Missing prompt in {source}")
            prompts.append(prompt)
        content = "".join(
            json.dumps({"turns": prompt}, ensure_ascii=False) + "\n"
            for prompt in prompts
        )
        save_immutable(output, content)
        for prompt in prompts:
            tokenized = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
            max_prompt_tokens = max(max_prompt_tokens, len(tokenized))
        prompt_hashes = [normalized_hash(prompt) for prompt in prompts]
        redhat_prompt_hashes[subset] = prompt_hashes
        prompt_matches = sum(value in calibration_prompts for value in prompt_hashes)
        response_matches = sum(
            value in calibration_responses for value in prompt_hashes
        )
        files[output.name] = {
            "sha256": sha256(output),
            "rows": len(prompts),
            "suite": "redhatai-speculator-benchmarks",
            "subset": subset,
            "source_sha256": sha256(source),
            "calibration_prompt_matches": prompt_matches,
            "calibration_response_matches": response_matches,
        }

    speed_manifest_path = speed_root / "manifest.json"
    speed_manifest = json.loads(speed_manifest_path.read_text())
    speedbench_prompt_hashes: dict[str, list[str]] = {}
    speed_preparation = json.loads(
        (SPEEDBENCH_SOURCE / "preparation-provenance.json").read_text()
    )
    calibration_source = json.loads(CALIBRATION_SOURCE_MANIFEST.read_text())
    for subset in speed_manifest["configurations"]:
        if subset != "qualitative":
            continue
        for filename, item in speed_manifest["files"].items():
            if not filename.startswith("qualitative_"):
                continue
            path = speed_root / filename
            category = filename.removeprefix("qualitative_").removesuffix(".jsonl")
            speedbench_prompt_hashes[category] = [
                example["normalized_prompt_sha256"]
                for example in speed_manifest["examples"]
                if example["config"] == "qualitative" and example["file"] == filename
            ]
            for line in path.read_text().splitlines():
                row = json.loads(line)
                tokenized = tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["turns"]}],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=False,
                )
                max_prompt_tokens = max(max_prompt_tokens, len(tokenized))
            files[filename] = {
                **item,
                "suite": "speedbench-qualitative-first-user-turn",
                "subset": filename.removeprefix("qualitative_").removesuffix(".jsonl"),
                "source_sha256": sha256(SPEEDBENCH_SOURCE / "qualitative.jsonl"),
            }

    max_model_len = max(16384, max_prompt_tokens + 4096)
    redhat_all = [value for values in redhat_prompt_hashes.values() for value in values]
    speedbench_all = [
        value for values in speedbench_prompt_hashes.values() for value in values
    ]
    redhat_counts = Counter(redhat_all)
    speedbench_counts = Counter(speedbench_all)
    redhat_subset_sets = {
        name: set(values) for name, values in redhat_prompt_hashes.items()
    }
    speedbench_subset_sets = {
        name: set(values) for name, values in speedbench_prompt_hashes.items()
    }
    redhat_cross_subset_matches = sum(
        len(redhat_subset_sets[left] & redhat_subset_sets[right])
        for index, left in enumerate(redhat_subset_sets)
        for right in list(redhat_subset_sets)[index + 1 :]
    )
    speedbench_cross_subset_matches = sum(
        len(speedbench_subset_sets[left] & speedbench_subset_sets[right])
        for index, left in enumerate(speedbench_subset_sets)
        for right in list(speedbench_subset_sets)[index + 1 :]
    )
    model_context = int(
        getattr(target_config, "max_position_embeddings", max_model_len)
    )
    if max_model_len > model_context:
        raise ValueError(
            f"Required max_model_len {max_model_len} exceeds target context {model_context}"
        )
    tokenizer_files = [
        TOKENIZER / name
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "vocab.json",
            "merges.txt",
        )
        if (TOKENIZER / name).is_file()
    ]
    manifest = {
        "schema": 1,
        "experiment": "Qwen3.8-27B DSpark quantization acceptance",
        "target": {
            "repo": TARGET_REPO,
            "revision": TARGET_REVISION,
            "tokenizer_path": str(TOKENIZER),
            "tokenizer_files": {p.name: sha256(p) for p in tokenizer_files},
            "chat_template_sha256": hashlib.sha256(
                tokenizer.get_chat_template().encode()
            ).hexdigest(),
        },
        "datasets": {
            "calibration": {
                "dataset": calibration_source["dataset"],
                "source_dataset": calibration_source["source_dataset"],
                "source_manifest": str(CALIBRATION_SOURCE_MANIFEST),
                "source_manifest_sha256": sha256(CALIBRATION_SOURCE_MANIFEST),
                "raw_requested_samples": calibration_source["num_samples_requested"],
                "prepared_available_samples": len(calibration),
                "calibration_used_by_arms": 1892,
                "sequence_length_cap": 2048,
                "sampling_seed": calibration_source["seed"],
                "description": "Local proportional sample from the regenerated OpenPerfectBlend-Qwen38-27B collection; do not treat as the unmodified official PerfectBlend dataset.",
            },
            "redhatai": {
                "repo": REDHATAI_REPO,
                "revision": REDHATAI_REVISION,
                "source_snapshot": str(REDHATAI_SNAPSHOT),
                "source_file_hashes": {
                    subset + ".jsonl": sha256(REDHATAI_SNAPSHOT / (subset + ".jsonl"))
                    for subset in REDHATAI_SUBSETS
                },
                "prompt_overlap": {
                    "prompt_matches": sum(
                        value["calibration_prompt_matches"]
                        for value in files.values()
                        if value["suite"] == "redhatai-speculator-benchmarks"
                    ),
                    "response_matches": sum(
                        value["calibration_response_matches"]
                        for value in files.values()
                        if value["suite"] == "redhatai-speculator-benchmarks"
                    ),
                    "normalization": "NFKC, casefold, collapse whitespace; exact hash match",
                },
            },
            "speedbench": {
                "repo": "nvidia/SPEED-Bench",
                "revision": speed_preparation["benchmark_revision"],
                "preparation_repository": speed_preparation["preparation_repository"],
                "preparation_revision": speed_preparation["preparation_revision"],
                "preparation_provenance": str(
                    SPEEDBENCH_SOURCE / "preparation-provenance.json"
                ),
                "transformation": "First user turn only; qualitative config; all categories retained.",
                "freeze_manifest_sha256": sha256(speed_manifest_path),
                "prompt_overlap": speed_manifest["overlap"],
            },
        },
        "files": files,
        "evaluation_prompt_overlap": {
            "normalization": "NFKC, casefold, collapse whitespace; exact SHA-256 matches",
            "redhatai_duplicate_rows_within_subset": sum(
                sum(count - 1 for count in Counter(values).values() if count > 1)
                for values in redhat_prompt_hashes.values()
            ),
            "redhatai_cross_subset_matching_pairs": redhat_cross_subset_matches,
            "speedbench_duplicate_rows_within_category": sum(
                sum(count - 1 for count in Counter(values).values() if count > 1)
                for values in speedbench_prompt_hashes.values()
            ),
            "speedbench_cross_category_matching_pairs": speedbench_cross_subset_matches,
            "redhatai_speedbench_shared_unique_prompts": len(
                set(redhat_all) & set(speedbench_all)
            ),
            "redhatai_speedbench_matching_occurrences": sum(
                min(redhat_counts[key], speedbench_counts[key])
                for key in set(redhat_counts) & set(speedbench_counts)
            ),
        },
        "max_prompt_tokens": max_prompt_tokens,
        "max_output_tokens": 4096,
        "max_model_len": max_model_len,
    }
    save_immutable(
        OUT / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    print(
        json.dumps(
            {
                "output": str(OUT),
                "subsets": len(files),
                "max_prompt_tokens": max_prompt_tokens,
                "max_model_len": max_model_len,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
