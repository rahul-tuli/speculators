#!/usr/bin/env python3
"""Upload staged DSpark drafters and add them to the existing collection."""

import json
from pathlib import Path

from huggingface_hub import HfApi

STAGE = Path("/data/fast/drafter-quant/hf-publication-dspark-qwen3.8-27b")


def main() -> None:
    manifest_path = STAGE / "publication-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    api = HfApi()
    user = api.whoami()
    print(f"Authenticated as {user['name']}")
    results_path = STAGE / "upload-results.json"
    results = (
        json.loads(results_path.read_text())
        if results_path.exists()
        else {"models": []}
    )
    completed = {
        record["repo_id"] for record in results["models"] if record.get("verified")
    }
    for repo_id in manifest["models"]:
        if repo_id in completed:
            print(f"already uploaded and verified: {repo_id}")
            continue
        prefix = "inference-optimization/Qwen3.8-27B-DSpark-"
        if not repo_id.startswith(prefix):
            raise ValueError(f"Unexpected staged repository name: {repo_id}")
        folder_name = repo_id.removeprefix(prefix)
        folder = STAGE / folder_name
        if (
            not (folder / "model.safetensors").is_file()
            or not (folder / "README.md").is_file()
        ):
            raise FileNotFoundError(f"Incomplete staged model bundle: {folder}")
        api.create_repo(
            repo_id=repo_id, repo_type="model", private=False, exist_ok=True
        )
        commit = api.upload_folder(
            folder_path=str(folder),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Publish Qwen3.8-27B DSpark quantized drafter with evaluation provenance",
        )
        api.add_collection_item(
            collection_slug=manifest["collection"],
            item_id=repo_id,
            item_type="model",
            exists_ok=True,
        )
        remote = api.list_repo_files(repo_id=repo_id, repo_type="model")
        required = {
            "README.md",
            "model.safetensors",
            "acceptance_by_subset.csv",
            "config.json",
            "quant_run_manifest.json",
            "provenance/quantization/calibration-source-manifest.json",
            "speculators.patch",
            "provenance/evaluation/eval-data-manifest.json",
            "provenance/evaluation/speedbench-preparation-provenance.json",
            "provenance/evaluation/serving/vllm_command.txt",
            "provenance/evaluation/serving/vllm.patch",
            "provenance/evaluation/serving/checkpoint_sha256.txt",
            "provenance/evaluation/serving/drafter_checkpoint_sha256.txt",
            "provenance/evaluation/serving/dspark-head-scope-audit.json",
            "provenance/evaluation/points/HumanEval/eval_command.txt",
        }
        missing = sorted(required - set(remote))
        if missing:
            raise ValueError(
                f"Remote verification failed for {repo_id}: missing {missing}"
            )
        result = {
            "repo_id": repo_id,
            "url": f"https://huggingface.co/{repo_id}",
            "commit": commit.oid,
            "remote_file_count": len(remote),
            "verified": True,
        }
        results["models"] = [
            row for row in results["models"] if row.get("repo_id") != repo_id
        ] + [result]
        results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"uploaded and verified: {repo_id} ({commit.oid})")

    collection = api.get_collection(manifest["collection"])
    collection_ids = {item.item_id for item in collection.items}
    missing = sorted(set(manifest["models"]) - collection_ids)
    if missing:
        raise ValueError(f"Models missing from collection: {missing}")
    print(
        f"All {len(manifest['models'])} models are uploaded and in {manifest['collection']}."
    )


if __name__ == "__main__":
    main()
