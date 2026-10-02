#!/usr/bin/env python3
"""Upload DSpark quantized drafters with quantization provenance only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

STAGE = Path("/data/fast/drafter-quant/hf-publication-dspark-qwen3.8-27b")
RESULTS = STAGE / "upload-pending-results.json"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    manifest_path = STAGE / "publication-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("publication_status") != "evaluation_pending":
        raise ValueError("This uploader only publishes a pending-evaluation bundle")
    repo_ids = manifest["models"]
    if len(repo_ids) != 10 or len(set(repo_ids)) != 10:
        raise ValueError(
            f"Expected ten unique quantized model repos, got {len(repo_ids)}"
        )

    api = HfApi()
    identity = api.whoami()
    print(f"Authenticated as {identity['name']}")
    results = (
        json.loads(RESULTS.read_text(encoding="utf-8"))
        if RESULTS.is_file()
        else {"models": []}
    )
    verified = {row["repo_id"] for row in results["models"] if row.get("verified")}

    common_required = {
        "README.md",
        "publication_status.json",
        "model.safetensors",
        "config.json",
        "quant_run_manifest.json",
        "train_command.txt",
        "quant_command.txt",
        "provenance/quantization/calibration-source-manifest.json",
        "provenance/quantization/drafter_checkpoint_sha256.txt",
        "provenance/quantization/model_checkpoint_sha256.json",
        "provenance/quantization/speculators.patch",
    }
    for repo_id in repo_ids:
        prefix = "inference-optimization/Qwen3.8-27B-DSpark-"
        if not repo_id.startswith(prefix):
            raise ValueError(f"Unexpected model repository: {repo_id}")
        folder = STAGE / repo_id.removeprefix(prefix)
        if (
            not (folder / "model.safetensors").is_file()
            or not (folder / "README.md").is_file()
        ):
            raise FileNotFoundError(f"Incomplete staged model: {folder}")
        if repo_id in verified:
            print(f"already uploaded and verified: {repo_id}")
            continue

        api.create_repo(
            repo_id=repo_id, repo_type="model", private=False, exist_ok=True
        )
        commit = api.upload_folder(
            folder_path=str(folder),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Publish Qwen3.8-27B DSpark quantized drafter and quantization provenance",
        )
        remote_files = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
        missing = sorted(common_required - remote_files)
        if missing:
            raise ValueError(
                f"Remote verification failed for {repo_id}; missing {missing}"
            )
        if (
            "GPTQ-" in repo_id
            and "provenance/quantization/gptq_audit.json" not in remote_files
        ):
            raise ValueError(f"GPTQ audit is missing remotely for {repo_id}")
        if (
            "PerfectBlend-" in repo_id
            and "provenance/quantization/calibration_manifest.json" not in remote_files
        ):
            raise ValueError(f"Calibration manifest is missing remotely for {repo_id}")

        local_hash = sha256(folder / "model.safetensors")
        remote_model = api.get_paths_info(
            repo_id, paths=["model.safetensors"], repo_type="model"
        )[0]
        if remote_model.lfs is None or remote_model.lfs.sha256 != local_hash:
            raise ValueError(f"Remote checkpoint SHA-256 does not match for {repo_id}")
        status_path = hf_hub_download(
            repo_id, "publication_status.json", repo_type="model"
        )
        status = json.loads(Path(status_path).read_text(encoding="utf-8"))
        if (
            status.get("status") != "evaluation_pending"
            or status.get("evaluation_results_included") is not False
        ):
            raise ValueError(f"Remote publication status is not pending for {repo_id}")
        readme_path = hf_hub_download(repo_id, "README.md", repo_type="model")
        if "Evaluation is pending." not in Path(readme_path).read_text(
            encoding="utf-8"
        ):
            raise ValueError(
                f"Remote model card does not state evaluation is pending: {repo_id}"
            )

        api.add_collection_item(
            collection_slug=manifest["collection"],
            item_id=repo_id,
            item_type="model",
            exists_ok=True,
        )
        result = {
            "repo_id": repo_id,
            "url": f"https://huggingface.co/{repo_id}",
            "commit": commit.oid,
            "checkpoint_sha256": local_hash,
            "remote_file_count": len(remote_files),
            "verified": True,
            "publication_status": "evaluation_pending",
        }
        results["models"] = [
            row for row in results["models"] if row.get("repo_id") != repo_id
        ] + [result]
        RESULTS.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"uploaded and verified: {repo_id} ({commit.oid})")

    collection = api.get_collection(manifest["collection"])
    collection_ids = {item.item_id for item in collection.items}
    missing = sorted(set(repo_ids) - collection_ids)
    if missing:
        raise ValueError(f"Models missing from collection: {missing}")
    print(
        f"All {len(repo_ids)} models are uploaded, verified, and in {manifest['collection']}."
    )


if __name__ == "__main__":
    main()
