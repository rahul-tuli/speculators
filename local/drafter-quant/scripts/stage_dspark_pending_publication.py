#!/usr/bin/env python3
"""Stage DSpark quantized drafters with quantization provenance, before evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODELS = Path("/data/fast/models")
CALIBRATION_SOURCE_MANIFEST = Path(
    "/data/fast/drafter-quant/data/perfectblend_27b_2048/manifest.json"
)
STAGE = Path("/data/fast/drafter-quant/hf-publication-dspark-qwen3.8-27b")
COLLECTION = "inference-optimization/quantized-drafters-6ab5354d0b0766f2b133d182"
TARGET_REPO = "Qwen/Qwen3.8-27B"
TARGET_REV = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
SOURCE_REPO = "RedHatAI/Qwen3.8-27B-speculator.dspark"
SOURCE_REV = "7f33c272e5da240978e0d55767abab8193d74b95"

# (arm id, checkpoint directory, HF suffix, label, description)
ARMS = [
    (
        "fp8-block",
        "Qwen3.8-27B-DSpark-FP8-BLOCK",
        "FP8-BLOCK",
        "FP8 block, data-free",
        "Data-free FP8_BLOCK export; no calibration data was used.",
    ),
    (
        "fp8-dynamic",
        "Qwen3.8-27B-DSpark-FP8-DYNAMIC",
        "FP8-DYNAMIC",
        "FP8 dynamic, data-free",
        "Data-free FP8_DYNAMIC export; no calibration data was used.",
    ),
    (
        "fp8-gaussian",
        "Qwen3.8-27B-DSpark-Gauss-FP8-W8A8",
        "Gauss-FP8-W8A8",
        "Static FP8 W8A8, Gaussian",
        "Static FP8 W8A8 using seeded Gaussian calibration, 1,892 aligned calibration records, and sequence cap 2,048.",
    ),
    (
        "fp8-real",
        "Qwen3.8-27B-DSpark-PerfectBlend-FP8-W8A8",
        "PerfectBlend-FP8-W8A8",
        "Static FP8 W8A8, target-calibrated",
        "Static FP8 W8A8 using 1,892 aligned target hidden-state calibration records and sequence cap 2,048.",
    ),
    (
        "nvfp4-gaussian",
        "Qwen3.8-27B-DSpark-Gauss-NVFP4-W4A4",
        "Gauss-NVFP4-W4A4",
        "Static NVFP4 W4A4, Gaussian",
        "Static NVFP4 W4A4 using seeded Gaussian calibration, 1,892 aligned calibration records, and sequence cap 2,048.",
    ),
    (
        "nvfp4-real",
        "Qwen3.8-27B-DSpark-PerfectBlend-NVFP4-W4A4",
        "PerfectBlend-NVFP4-W4A4",
        "Static NVFP4 W4A4, target-calibrated",
        "Static NVFP4 W4A4 using 1,892 aligned target hidden-state calibration records and sequence cap 2,048.",
    ),
    (
        "nvfp4-gptq-gaussian",
        "Qwen3.8-27B-DSpark-GPTQ-Gauss-NVFP4-W4A4",
        "GPTQ-Gauss-NVFP4-W4A4",
        "NVFP4 GPTQ, Gaussian",
        "NVFP4 GPTQ using expanded-MSE observer, 0.1 Hessian damping, and seeded Gaussian calibration (1,892 aligned records).",
    ),
    (
        "nvfp4-gptq-real",
        "Qwen3.8-27B-DSpark-GPTQ-PerfectBlend-NVFP4-W4A4",
        "GPTQ-PerfectBlend-NVFP4-W4A4",
        "NVFP4 GPTQ, target-calibrated",
        "NVFP4 GPTQ using expanded-MSE observer, 0.1 Hessian damping, and 1,892 aligned target hidden-state calibration records.",
    ),
    (
        "nvfp4-gptq-imatrix-gaussian",
        "Qwen3.8-27B-DSpark-GPTQ-IMatrix-Gauss-NVFP4-W4A4",
        "GPTQ-IMatrix-Gauss-NVFP4-W4A4",
        "NVFP4 GPTQ + IMatrix, Gaussian",
        "NVFP4 GPTQ using strict expanded-IMatrix observer, 0.1 Hessian damping, and seeded Gaussian calibration (1,892 aligned records).",
    ),
    (
        "nvfp4-gptq-imatrix-real",
        "Qwen3.8-27B-DSpark-GPTQ-IMatrix-PerfectBlend-NVFP4-W4A4",
        "GPTQ-IMatrix-PerfectBlend-NVFP4-W4A4",
        "NVFP4 GPTQ + IMatrix, target-calibrated",
        "NVFP4 GPTQ using strict expanded-IMatrix observer, 0.1 Hessian damping, and 1,892 aligned target hidden-state calibration records.",
    ),
]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def copy_file(source: Path, destination: Path, *, hardlink: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(source) != sha256(destination):
            raise ValueError(f"Staged artifact changed: {destination}")
        return
    if hardlink:
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def model_card(repo_id: str, label: str, description: str, nvfp4: bool) -> str:
    tags = ["speculative-decoding", "dspark", "qwen3", "qwen3.8", "quantized"]
    tags.append("nvfp4" if nvfp4 else "fp8")
    tags.append("data-free" if "data-free" in label else "calibration-data")
    if "GPTQ" in label:
        tags.append("gptq")
    if "IMatrix" in label:
        tags.append("imatrix")
    if "Gaussian" in label:
        tags.append("gaussian-calibration")
    elif "target-calibrated" in label:
        tags.append("openperfectblend-derived-calibration")
    tag_block = "\n".join(f"- {tag}" for tag in tags)
    command = [
        "vllm serve Qwen/Qwen3.8-27B \\",
        f"  --spec-model {repo_id} \\",
        "  --spec-method dspark \\",
        "  --spec-tokens 8" + (" \\" if nvfp4 else ""),
    ]
    if nvfp4:
        command.append('  --kernel-config \'{"linear_backend":"emulation"}\'')
    if "data-free" in label:
        calibration_note = "No calibration data was used for this data-free export."
    elif "Gaussian" in label:
        calibration_note = "This arm used seeded Gaussian calibration values rather than prompts. The quantization manifest records 1,892 calibration records and a sequence cap of 2,048."
    else:
        calibration_note = "This arm used a local proportional sample from `shanjiaz/OpenPerfectBlend-Qwen38-27B-regenerated`, derived from `mlabonne/open-perfectblend`; it is not the unmodified upstream PerfectBlend collection. The sample requested 2,048 examples; 1,892 aligned hidden-state records were available and used, with a sequence cap of 2,048."
    backend_note = (
        "For NVFP4, this example selects vLLM's emulation backend; it makes no claim of native H100 NVFP4 support."
        if nvfp4
        else "This command is an example; this checkpoint has not completed serving validation."
    )
    article = "An" if label.startswith(("FP8", "NVFP4")) else "A"
    return "\n".join(
        [
            "---",
            "library_name: speculators",
            "base_model:",
            f"- {SOURCE_REPO}",
            f"- {TARGET_REPO}",
            "license: apache-2.0",
            "pipeline_tag: text-generation",
            "tags:",
            tag_block,
            "---",
            "",
            f"# {repo_id.rsplit('/', 1)[-1]}",
            "",
            f"{article} {label} DSpark drafter derived from [{SOURCE_REPO}](https://huggingface.co/{SOURCE_REPO}) at revision `{SOURCE_REV}`. Pair it with [{TARGET_REPO}](https://huggingface.co/{TARGET_REPO}) at revision `{TARGET_REV}`. This repository contains a drafter component, not a standalone chat model.",
            "",
            "## Quantization",
            "",
            description,
            "",
            calibration_note,
            "Quantization commands, manifests, source files, patches, and the exported checkpoint checksum are in `provenance/quantization/`.",
            "",
            "## Example serving command",
            "",
            "```bash",
            *command,
            "```",
            "",
            backend_note,
            "",
            "## Evaluation status",
            "",
            "Evaluation is pending. No completed acceptance, speed, or quality results are included with this publication. The checkpoint has quantization provenance only; serving/runtime validation and the planned evaluation matrix have not completed.",
            "",
            "## Reproducibility",
            "",
            "`provenance/quantization/` includes the training and quantization commands, quantization manifest, calibration metadata where applicable, source scripts and patches, and the SHA-256 digest of the published drafter weights. Calibration prompt data is not redistributed.",
            "",
        ]
    )


def main() -> None:
    STAGE.mkdir(parents=True, exist_ok=True)
    (STAGE / ".gitattributes").write_text(
        "*.safetensors filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8"
    )
    source_manifest_sha = sha256(CALIBRATION_SOURCE_MANIFEST)
    repo_ids = []
    for arm, checkpoint_name, suffix, label, description in ARMS:
        checkpoint = MODELS / checkpoint_name
        model_file = checkpoint / "model.safetensors"
        quant_manifest = checkpoint / "quant_run_manifest.json"
        if not model_file.is_file() or not quant_manifest.is_file():
            raise FileNotFoundError(f"Incomplete checkpoint: {checkpoint}")
        repo_id = f"inference-optimization/Qwen3.8-27B-DSpark-{suffix}"
        repo_ids.append(repo_id)
        stage_dir = STAGE / suffix
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / ".gitattributes").write_text(
            "*.safetensors filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8"
        )

        for source in checkpoint.iterdir():
            if source.is_file():
                copy_file(
                    source,
                    stage_dir / source.name,
                    hardlink=source.suffix == ".safetensors",
                )
            elif source.is_dir() and source.name == "source":
                for item in source.rglob("*"):
                    if item.is_file():
                        copy_file(
                            item,
                            stage_dir
                            / "provenance/quantization/source"
                            / item.relative_to(source),
                        )

        quant_prov = stage_dir / "provenance/quantization"
        for name in (
            "train_command.txt",
            "quant_command.txt",
            "quant_run_manifest.json",
            "calibration_manifest.json",
            "gptq_audit.json",
            "run_status.json",
            "checkpoint_complete.json",
            "speculators.patch",
            "source_speculators.patch",
            "compressed-tensors.patch",
            "llm-compressor.patch",
        ):
            path = checkpoint / name
            if path.is_file():
                copy_file(path, quant_prov / name)

        digest = sha256(model_file)
        copy_file(
            CALIBRATION_SOURCE_MANIFEST, quant_prov / "calibration-source-manifest.json"
        )
        (quant_prov / "drafter_checkpoint_sha256.txt").write_text(
            f"{digest}  model.safetensors\n", encoding="utf-8"
        )
        (quant_prov / "model_checkpoint_sha256.json").write_text(
            json.dumps({"file": "model.safetensors", "sha256": digest}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        (stage_dir / "publication_status.json").write_text(
            json.dumps(
                {
                    "status": "evaluation_pending",
                    "evaluation_results_included": False,
                    "serving_validation_complete": False,
                    "arm": arm,
                    "repo_id": repo_id,
                    "target_repo": TARGET_REPO,
                    "target_revision": TARGET_REV,
                    "drafter_source_repo": SOURCE_REPO,
                    "drafter_source_revision": SOURCE_REV,
                    "calibration_source_manifest_sha256": source_manifest_sha,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (stage_dir / "README.md").write_text(
            model_card(repo_id, label, description, "NVFP4" in label), encoding="utf-8"
        )
        print(f"staged pending evaluation: {arm} ({digest})")

    (STAGE / "publication-manifest.json").write_text(
        json.dumps(
            {
                "collection": COLLECTION,
                "publication_status": "evaluation_pending",
                "target_repo": TARGET_REPO,
                "target_revision": TARGET_REV,
                "drafter_source_repo": SOURCE_REPO,
                "drafter_source_revision": SOURCE_REV,
                "calibration_source_manifest_sha256": source_manifest_sha,
                "models": repo_ids,
                "quantized_arms": len(repo_ids),
                "evaluation_baseline": "bf16-drafter",
                "evaluation_results_included": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Staged {len(repo_ids)} quantized DSpark models at {STAGE}")


if __name__ == "__main__":
    main()
