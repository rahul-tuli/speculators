"""Freeze every selected SPEED-Bench first prompt, its identity and overlap audit."""

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/evaluate"))
from point import digest, save  # noqa: E402

CONFIGS = ["qualitative", "throughput_1k", "throughput_2k", "throughput_8k"]


def text_hash(text):
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def main():  # noqa: C901
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--calibration-data", type=Path, required=True)
    args = p.parse_args()
    from datasets import load_from_disk  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    template = tokenizer.get_chat_template()
    calibration = load_from_disk(str(args.calibration_data))
    calibration_prompts, calibration_responses = set(), set()
    for row in calibration:
        # Audit the actual cached/truncated sequences, not the original mixture.
        decoded = tokenizer.decode(row["input_ids"], skip_special_tokens=False)
        messages = [
            {"role": role, "content": content}
            for role, content in re.findall(
                r"<\|im_start\|>(user|assistant)\n(.*?)(?:<\|im_end\|>|$)",
                decoded,
                re.S,
            )
        ]
        for message in messages:
            value = message.get("content", "")
            if isinstance(value, str):
                (
                    calibration_responses
                    if message["role"] == "assistant"
                    else calibration_prompts
                ).add(text_hash(value))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": 1,
        "configurations": CONFIGS,
        "preparation_revision": "bcf059af55c20a89f797724598f9908d126153e6",
        "transformation": "Original first user turn of every example; later turns excluded by user decision.",  # noqa: E501
        "normalization": "NFKC, casefold, collapse whitespace; exact hash match, no exclusions",  # noqa: E501
        "calibration_data": str(args.calibration_data),
        "calibration_rows": len(calibration),
        "calibration_text_columns": calibration.column_names,
        "calibration_text_reconstruction": "Decode actual cached input_ids with target tokenizer; retain user/assistant segments including truncated final segment.",  # noqa: E501
        "calibration_files": {
            p.name: digest(p) for p in args.calibration_data.glob("*.arrow")
        },
        "calibration_prompt_hash_count": len(calibration_prompts),
        "calibration_response_hash_count": len(calibration_responses),
        "files": {},
        "sources": {},
        "examples": [],
        "multiturn_counts": {},
        "tokenizer": args.tokenizer,
        "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "max_output_tokens": 4096,
        "max_prompt_tokens": 0,
        "original_capture_limitation": "Original target weight hashes were not recorded; current hashes do not repair this.",  # noqa: E501
    }
    for script in ["prepare_nvidia.py", "prepare_nvidia_memory.py"]:
        manifest["sources"][script] = digest(args.data_dir / script)
    for config in CONFIGS:
        source = args.data_dir / f"{config}.jsonl"
        manifest["sources"][source.name] = digest(source)
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        buckets = defaultdict(list)
        turn_counts = Counter()
        for index, row in enumerate(rows):
            messages = row["messages"]
            if (
                not messages
                or messages[0]["role"] != "user"
                or not messages[0]["content"]
            ):
                raise ValueError(f"Missing first prompt: {config} row {index}")
            text = messages[0]["content"]
            if "FULL BENCHMARK DATA SHOULD BE FETCHED" in text:
                raise ValueError(f"Unmaterialized prompt: {config} row {index}")
            turn_counts[len(messages)] += 1
            cat = row["category"].replace(" ", "_").replace("/", "_")
            sub = (
                (row.get("sub_category") or "unknown")
                .replace(" ", "_")
                .replace("/", "_")
            )
            name = (
                f"{config}_{cat}"
                + (f"__{sub}" if config != "qualitative" else "")
                + ".jsonl"
            )
            buckets[name].append({"turns": text})
            prompt_tokens = len(
                tokenizer.apply_chat_template(
                    [messages[0]],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=False,
                )
            )
            manifest["max_prompt_tokens"] = max(
                manifest["max_prompt_tokens"], prompt_tokens
            )
            sha = text_hash(text)
            manifest["examples"].append(
                {
                    "config": config,
                    "source_index": index,
                    "question_id": row.get("question_id"),
                    "file": name,
                    "file_index": len(buckets[name]) - 1,
                    "first_prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "normalized_prompt_sha256": sha,
                    "turns_in_source": len(messages),
                    "prompt_tokens": prompt_tokens,
                    "matches_calibration_prompt": sha in calibration_prompts,
                    "matches_calibration_response": sha in calibration_responses,
                }
            )
        manifest["multiturn_counts"][config] = dict(turn_counts)
        for name, prompts in buckets.items():
            dest = args.output / name
            content = "".join(
                json.dumps(row, ensure_ascii=False) + "\n" for row in prompts
            )
            if dest.exists() and dest.read_text() != content:
                raise ValueError(f"Refusing to overwrite changed frozen file: {dest}")
            dest.write_text(content)
            manifest["files"][name] = {"sha256": digest(dest), "rows": len(prompts)}
    manifest["max_model_len"] = max(16384, manifest["max_prompt_tokens"] + 4096)
    manifest["overlap"] = {
        "prompt_matches": sum(
            x["matches_calibration_prompt"] for x in manifest["examples"]
        ),
        "response_matches": sum(
            x["matches_calibration_response"] for x in manifest["examples"]
        ),
        "evaluation_responses": "No reference responses in prepared SPEED-Bench; response-to-response overlap unavailable.",  # noqa: E501
    }
    if not calibration_prompts and not calibration_responses:
        raise ValueError(
            "Calibration messages unavailable; do not report a false zero-overlap audit"
        )
    save(args.output / "manifest.json", manifest)
    print(  # noqa: T201
        json.dumps(
            {k: v for k, v in manifest.items() if k not in ["examples", "files"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
