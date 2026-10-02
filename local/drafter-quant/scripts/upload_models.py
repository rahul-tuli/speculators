import json
from pathlib import Path

from huggingface_hub import HfApi

STAGE_ROOT = Path("/data/fast/drafter-quant/hf-publication-20260928")
COLLECTION_SLUG = "inference-optimization/quantized-drafters-6ab5354d0b0766f2b133d182"

MODELS = [
    {
        "nickname": "Block8",
        "repo_id": "inference-optimization/Qwen3-8B-DFlash-FP8-BLOCK",
        "folder": STAGE_ROOT / "Block8",
    },
    {
        "nickname": "GPTQ-Gauss4",
        "repo_id": "inference-optimization/Qwen3-8B-DFlash-GPTQ-Gauss-NVFP4-W4A4",
        "folder": STAGE_ROOT / "GPTQ-Gauss4",
    },
    {
        "nickname": "GPTQ-Blend4",
        "repo_id": "inference-optimization/Qwen3-8B-DFlash-GPTQ-PerfectBlend-NVFP4-W4A4",
        "folder": STAGE_ROOT / "GPTQ-Blend4",
    },
    {
        "nickname": "IMatrix-Gauss4",
        "repo_id": "inference-optimization/Qwen3-8B-DFlash-GPTQ-IMatrix-Gauss-NVFP4-W4A4",
        "folder": STAGE_ROOT / "IMatrix-Gauss4",
    },
    {
        "nickname": "IMatrix-Blend4",
        "repo_id": "inference-optimization/Qwen3-8B-DFlash-GPTQ-IMatrix-PerfectBlend-NVFP4-W4A4",
        "folder": STAGE_ROOT / "IMatrix-Blend4",
    },
]


def main():
    api = HfApi()
    user = api.whoami()
    print(f"Authenticated as: {user['name']} ({user.get('fullname')})")

    results = {
        "collection": COLLECTION_SLUG,
        "models": [],
    }

    for m in MODELS:
        nick = m["nickname"]
        repo_id = m["repo_id"]
        folder = m["folder"]

        print("\n=======================================================")
        print(f"Processing {nick} -> {repo_id}")
        print(f"Folder: {folder}")
        print("=======================================================")

        # 1. Create repo
        print(f"1. Ensuring repo exists: {repo_id}...")
        repo_url = api.create_repo(
            repo_id=repo_id, repo_type="model", exist_ok=True, private=False
        )
        print(f"   Repo URL: {repo_url}")

        # Count local files
        local_files = [p for p in folder.rglob("*") if p.is_file()]
        print(f"   Local files to upload: {len(local_files)}")

        # 2. Upload folder
        print(f"2. Uploading folder {folder} to {repo_id}...")
        commit_info = api.upload_folder(
            folder_path=str(folder),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {nick} quantized drafter ({repo_id.split('/')[-1]}) with provenance",
        )
        print(f"   Upload complete! Commit SHA: {commit_info.oid}")

        # 3. Add to collection
        print(f"3. Adding to collection {COLLECTION_SLUG}...")
        in_coll = False
        try:
            api.add_collection_item(
                collection_slug=COLLECTION_SLUG,
                item_id=repo_id,
                item_type="model",
                exists_ok=True,
            )
            print(f"   Successfully added {repo_id} to collection.")
            in_coll = True
        except Exception as e:
            print(f"   Warning adding to collection: {e}")
            # Check if it's already in collection
            try:
                coll = api.get_collection(COLLECTION_SLUG)
                for item in coll.items:
                    if item.item_id == repo_id:
                        in_coll = True
                        print("   Confirmed item already present in collection.")
                        break
            except Exception as e2:
                print(f"   Error checking collection: {e2}")

        # 4. Verify remote files
        print("4. Verifying remote files...")
        remote_files = api.list_repo_files(repo_id=repo_id, repo_type="model")
        print(f"   Remote files count: {len(remote_files)}")

        record = {
            "nickname": nick,
            "repo_id": repo_id,
            "url": f"https://huggingface.co/{repo_id}",
            "commit": commit_info.oid,
            "local_files_count": len(local_files),
            "remote_files_count": len(remote_files),
            "has_safetensors": "model.safetensors" in remote_files,
            "in_collection": in_coll,
        }
        results["models"].append(record)

        # Write progress
        out_json = STAGE_ROOT / "upload_results.json"
        out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"   Saved progress to {out_json}")

    print(
        "\n================ All uploads and collection updates complete! ================"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
