#!/usr/bin/env python3
"""Stage DSpark model cards and provenance after the complete eval matrix."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODELS = Path("/data/fast/models")
RESULTS = Path("/data/fast/drafter-quant/results/dspark-qwen3.8-27b-acceptance")
DATA = Path("/data/fast/drafter-quant/data/dspark-qwen3.8-27b-eval")
CALIBRATION_SOURCE_MANIFEST = Path(
    "/data/fast/drafter-quant/data/perfectblend_27b_2048/manifest.json"
)
CHARTS = ROOT / "local/results/dspark_qwen3_8_27b"
STAGE = Path("/data/fast/drafter-quant/hf-publication-dspark-qwen3.8-27b")
TARGET_REV = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
SOURCE_REV = "7f33c272e5da240978e0d55767abab8193d74b95"
COLLECTION = "inference-optimization/quantized-drafters-6ab5354d0b0766f2b133d182"

# (arm id, checkpoint directory, HF repo suffix, display label, description)
ARMS = [
    (
        "fp8-block",
        "Qwen3.8-27B-DSpark-FP8-BLOCK",
        "FP8-BLOCK",
        "FP8 block, data-free",
        "Data-free FP8_BLOCK; no calibration data was used.",
    ),
    (
        "fp8-dynamic",
        "Qwen3.8-27B-DSpark-FP8-DYNAMIC",
        "FP8-DYNAMIC",
        "FP8 dynamic, data-free",
        "Data-free FP8_DYNAMIC; no calibration data was used.",
    ),
    (
        "fp8-gaussian",
        "Qwen3.8-27B-DSpark-Gauss-FP8-W8A8",
        "Gauss-FP8-W8A8",
        "Static FP8 W8A8, Gaussian",
        "Static FP8 W8A8 with seeded Gaussian calibration, 1,892 batches, sequence cap 2,048.",
    ),
    (
        "fp8-real",
        "Qwen3.8-27B-DSpark-PerfectBlend-FP8-W8A8",
        "PerfectBlend-FP8-W8A8",
        "Static FP8 W8A8, target-calibrated",
        "Static FP8 W8A8 with 1,892 target hidden-state calibration samples, sequence cap 2,048.",
    ),
    (
        "nvfp4-gaussian",
        "Qwen3.8-27B-DSpark-Gauss-NVFP4-W4A4",
        "Gauss-NVFP4-W4A4",
        "Static NVFP4 W4A4, Gaussian",
        "Static NVFP4 W4A4 with seeded Gaussian calibration, 1,892 batches, sequence cap 2,048.",
    ),
    (
        "nvfp4-real",
        "Qwen3.8-27B-DSpark-PerfectBlend-NVFP4-W4A4",
        "PerfectBlend-NVFP4-W4A4",
        "Static NVFP4 W4A4, target-calibrated",
        "Static NVFP4 W4A4 with 1,892 target hidden-state calibration samples, sequence cap 2,048.",
    ),
    (
        "nvfp4-gptq-gaussian",
        "Qwen3.8-27B-DSpark-GPTQ-Gauss-NVFP4-W4A4",
        "GPTQ-Gauss-NVFP4-W4A4",
        "NVFP4 GPTQ, Gaussian",
        "NVFP4 GPTQ using expanded-MSE observer, 0.1 Hessian damping, and seeded Gaussian calibration (1,892 batches).",
    ),
    (
        "nvfp4-gptq-real",
        "Qwen3.8-27B-DSpark-GPTQ-PerfectBlend-NVFP4-W4A4",
        "GPTQ-PerfectBlend-NVFP4-W4A4",
        "NVFP4 GPTQ, target-calibrated",
        "NVFP4 GPTQ using expanded-MSE observer, 0.1 Hessian damping, and 1,892 target hidden-state calibration samples.",
    ),
    (
        "nvfp4-gptq-imatrix-gaussian",
        "Qwen3.8-27B-DSpark-GPTQ-IMatrix-Gauss-NVFP4-W4A4",
        "GPTQ-IMatrix-Gauss-NVFP4-W4A4",
        "NVFP4 GPTQ + IMatrix, Gaussian",
        "NVFP4 GPTQ using strict expanded-IMatrix observer, 0.1 Hessian damping, and seeded Gaussian calibration (1,892 batches).",
    ),
    (
        "nvfp4-gptq-imatrix-real",
        "Qwen3.8-27B-DSpark-GPTQ-IMatrix-PerfectBlend-NVFP4-W4A4",
        "GPTQ-IMatrix-PerfectBlend-NVFP4-W4A4",
        "NVFP4 GPTQ + IMatrix, target-calibrated",
        "NVFP4 GPTQ using strict expanded-IMatrix observer, 0.1 Hessian damping, and 1,892 target hidden-state calibration samples.",
    ),
]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def copy_file(source: Path, dest: Path, hardlink: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if sha256(source) != sha256(dest):
            raise ValueError(f"Staged artifact changed: {dest}")
        return
    if hardlink:
        try:
            os.link(source, dest)
            return
        except OSError:
            pass
    shutil.copy2(source, dest)


def markdown_table(rows: list[dict], suite: str) -> str:
    selected = sorted(
        (r for r in rows if r["suite"] == suite), key=lambda r: r["subset"]
    )
    lines = [
        "| Subset | Requests | Acceptance length | Accepted-token fraction |",
        "|---|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| {row['subset']} | {row['requests']} | {float(row['acceptance_length']):.3f} tokens | {100 * float(row['accepted_token_fraction']):.2f}% |"
        )
    return "\n".join(lines)


def model_card(
    repo_id: str,
    arm: str,
    label: str,
    description: str,
    rows: list[dict],
    pooled_rows: list[dict],
    nvfp4: bool,
) -> str:
    model_rows = [row for row in rows if row["arm"] == arm]
    means = {row["suite"]: row for row in pooled_rows if row["arm"] == arm}
    tags = [
        "speculative-decoding",
        "dspark",
        "qwen3",
        "qwen3.8",
        "quantized",
        "nvfp4" if nvfp4 else "fp8",
    ]
    tags.append("data-free" if "data-free" in label else "calibration-data")
    if "GPTQ" in label:
        tags.append("gptq")
    if "IMatrix" in label:
        tags.append("imatrix")
    if "Gaussian" in label:
        tags.append("gaussian-calibration")
    elif "target-calibrated" in label:
        tags.append("perfectblend-calibration")
    tag_block = "\n".join(f"- {tag}" for tag in tags)
    speed = means["SPEED-Bench qualitative"]
    redhat = means["RedHatAI"]
    command = [
        "vllm serve Qwen/Qwen3.8-27B \\",
        f"  --spec-model {repo_id} \\",
        "  --spec-method dspark \\",
        "  --spec-tokens 8" + (" " + chr(92) if nvfp4 else ""),
    ]
    if nvfp4:
        command.append('  --kernel-config \'{"linear_backend":"emulation"}\'')
    return "\n".join(
        [
            "---",
            "library_name: speculators",
            "base_model:",
            "- RedHatAI/Qwen3.8-27B-speculator.dspark",
            "- Qwen/Qwen3.8-27B",
            "license: apache-2.0",
            "pipeline_tag: text-generation",
            "tags:",
            tag_block,
            "---",
            "",
            f"# {repo_id.rsplit('/', 1)[-1]}",
            "",
            f"A {label} DSpark drafter for the [Qwen3.8-27B target](https://huggingface.co/Qwen/Qwen3.8-27B), derived from [RedHatAI/Qwen3.8-27B-speculator.dspark](https://huggingface.co/RedHatAI/Qwen3.8-27B-speculator.dspark) at revision `{SOURCE_REV}`. This is a drafter component, not a standalone chat model.",
            "",
            "## Quantization",
            "",
            description,
            "",
            "The DSpark Markov and confidence heads remain BF16; the checkpoint/runtime head-scope audit verifies this. Target-calibrated arms use a local proportional sample from `shanjiaz/OpenPerfectBlend-Qwen38-27B-regenerated`, derived from `mlabonne/open-perfectblend`; it is not the unmodified upstream PerfectBlend collection. The raw sample requested 2,048 examples; 1,892 aligned hidden-state records were available and used.",
            "",
            "## Use with vLLM",
            "",
            "Pair this drafter with the pinned Qwen3.8-27B target and a DSpark-capable vLLM build:",
            "",
            "```bash",
            *command,
            "```",
            "",
            "## Acceptance evaluation",
            "",
            "GuideLLM throughput profile at maximum concurrency 128; temperature 0, seed 0, and up to 4,096 generated tokens. Acceptance length is `1 + accepted_tokens / draft_events`; accepted-token fraction is `accepted_tokens / proposed_tokens`. H100 W4A4 acceptance used vLLM NVFP4 emulation, so the results are not native NVFP4 speed claims.",
            "",
            f"Pooled by draft events — SPEED-Bench qualitative: **{float(speed['acceptance_length']):.3f} tokens**, {100 * float(speed['accepted_token_fraction']):.2f}% accepted-token fraction; RedHatAI: **{float(redhat['acceptance_length']):.3f} tokens**, {100 * float(redhat['accepted_token_fraction']):.2f}%.",
            "",
            "### RedHatAI/speculator_benchmarks",
            "",
            markdown_table(model_rows, "RedHatAI"),
            "",
            "### SPEED-Bench qualitative",
            "",
            markdown_table(model_rows, "SPEED-Bench qualitative"),
            "",
            "## Reproducibility",
            "",
            "`provenance/` contains training and quantization commands/manifests, quantizer sources and patches, vLLM command/patch and target/drafter hashes, the DSpark BF16 head-scope audit, frozen dataset revisions and hashes, per-subset eval commands, aggregate GuideLLM outputs, counter snapshots, and identities. Frozen prompt files and raw request records remain local; prompts are not republished.",
            "",
        ]
    )


def main() -> None:
    results_csv = CHARTS / "acceptance_by_subset.csv"
    pooled_csv = CHARTS / "pooled_acceptance.csv"
    rows = list(csv.DictReader(results_csv.open(newline="", encoding="utf-8")))
    pooled_rows = list(csv.DictReader(pooled_csv.open(newline="", encoding="utf-8")))
    if len(rows) != 220 or len({row["arm"] for row in rows}) != 11:
        raise ValueError("Need all 220 completed points before publication staging")
    manifest = read(DATA / "manifest.json")
    expected = sorted(manifest["files"])
    repo_ids = []
    STAGE.mkdir(parents=True, exist_ok=True)
    copy_file(DATA / "manifest.json", STAGE / "eval-data-manifest.json")
    copy_file(CALIBRATION_SOURCE_MANIFEST, STAGE / "calibration-source-manifest.json")
    copy_file(results_csv, STAGE / "acceptance_by_subset.csv")
    copy_file(pooled_csv, STAGE / "pooled_acceptance.csv")
    (STAGE / ".gitattributes").write_text(
        "*.safetensors filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8"
    )

    for arm, checkpoint_name, suffix, label, description in ARMS:
        repo_id = f"inference-optimization/Qwen3.8-27B-DSpark-{suffix}"
        repo_ids.append(repo_id)
        checkpoint = MODELS / checkpoint_name
        if (
            not (checkpoint / "quant_run_manifest.json").is_file()
            or not (checkpoint / "model.safetensors").is_file()
        ):
            raise FileNotFoundError(f"Incomplete checkpoint: {checkpoint}")
        run_roots = sorted(
            path
            for path in (RESULTS / "acceptance" / arm / "repeat-0").glob("*")
            if (path / "point-identity.json").is_file()
        )
        if len(run_roots) != 1:
            raise ValueError(
                f"Expected one acceptance run identity for {arm}; found {len(run_roots)}"
            )
        run_root = run_roots[0]
        stage_dir = STAGE / suffix
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / ".gitattributes").write_text(
            "*.safetensors filter=lfs diff=lfs merge=lfs -text\n",
            encoding="utf-8",
        )

        for source in checkpoint.iterdir():
            destination = (
                stage_dir / source.name
                if source.is_file()
                else stage_dir / "provenance/quantization/source"
            )
            if source.is_file():
                copy_file(source, destination, hardlink=source.suffix == ".safetensors")
            elif source.name == "source":
                for item in source.rglob("*"):
                    if item.is_file():
                        copy_file(item, destination / item.relative_to(source))
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

        server_dirs = sorted(
            path
            for path in run_root.glob("server-*")
            if (path / "finished.json").is_file()
        )
        if len(server_dirs) != 1:
            raise ValueError(
                f"Expected one finished serving run for {arm}; found {len(server_dirs)}"
            )
        server = server_dirs[0]
        serving = stage_dir / "provenance/evaluation/serving"
        for name in (
            "vllm_command.txt",
            "vllm.patch",
            "checkpoint_sha256.txt",
            "drafter_checkpoint_sha256.txt",
            "runtime-inventory.json",
            "dspark-head-scope-audit.json",
            "environment.json",
            "export-runtime-audit.json",
            "dspark-export-runtime-audit.json",
        ):
            path = server / name
            if path.is_file():
                copy_file(path, serving / name)
        point_identity = read(run_root / "point-identity.json")
        copy_file(
            run_root / "point-identity.json",
            stage_dir / "provenance/evaluation/point-identity.json",
        )
        repos = point_identity.get("environment", {}).get("repos", {})
        for repo_name, record in repos.items():
            patch_text = record.get("patch")
            if patch_text:
                (stage_dir / "provenance/evaluation" / f"{repo_name}.patch").write_text(
                    patch_text, encoding="utf-8"
                )
        for filename in expected:
            point = run_root / "concurrency-128" / Path(filename).stem
            marker_path = point / "complete.json"
            if not marker_path.is_file():
                raise ValueError(f"Missing completed subset {arm}/{filename}")
            marker = read(marker_path)
            for relative, wanted in marker["artifacts"].items():
                path = point / relative
                if not path.is_file() or sha256(path) != wanted:
                    raise ValueError(f"Eval artifact hash mismatch: {path}")
            attempt = point / marker["attempt"]
            dest = stage_dir / "provenance/evaluation/points" / Path(filename).stem
            public_hashes = {}
            for name in (
                "eval_command.txt",
                "measurement.command.txt",
                "result.json",
                "metrics-before.txt",
                "metrics-after.txt",
                "identity.json",
            ):
                path = attempt / name
                if path.is_file():
                    copy_file(path, dest / name)
                    public_hashes[name] = sha256(path)
                elif name not in ("metrics-before.txt", "metrics-after.txt"):
                    raise ValueError(f"Missing eval provenance: {path}")
            (dest / "published-point-manifest.json").write_text(
                json.dumps(
                    {
                        "dataset_file": filename,
                        "requests": read(attempt / "result.json")["requests"],
                        "local_completion_marker_sha256": sha256(marker_path),
                        "local_artifact_sha256": marker["artifacts"],
                        "published_artifact_sha256": public_hashes,
                        "omitted_local_artifacts": sorted(
                            set(marker["artifacts"]) - set(public_hashes)
                        ),
                        "omission_reason": "Raw GuideLLM request and warm-up artifacts can contain benchmark prompts; frozen data remains local.",
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

        model_rows = [row for row in rows if row["arm"] == arm]
        if {row["dataset_file"] for row in model_rows} != {
            Path(name).stem for name in expected
        }:
            raise ValueError(f"Subset result mismatch for {arm}")
        with (stage_dir / "acceptance_by_subset.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(model_rows)
        copy_file(
            STAGE / "eval-data-manifest.json",
            stage_dir / "provenance/evaluation/eval-data-manifest.json",
        )
        copy_file(
            CALIBRATION_SOURCE_MANIFEST,
            stage_dir / "provenance/quantization/calibration-source-manifest.json",
        )
        preparation_provenance = Path(
            manifest["datasets"]["speedbench"]["preparation_provenance"]
        )
        copy_file(
            preparation_provenance,
            stage_dir / "provenance/evaluation/speedbench-preparation-provenance.json",
        )
        (stage_dir / "README.md").write_text(
            model_card(
                repo_id, arm, label, description, rows, pooled_rows, "NVFP4" in label
            ),
            encoding="utf-8",
        )
        (stage_dir / "publication.json").write_text(
            json.dumps(
                {
                    "repo_id": repo_id,
                    "collection": COLLECTION,
                    "arm": arm,
                    "checkpoint_sha256": sha256(checkpoint / "model.safetensors"),
                    "target_revision": TARGET_REV,
                    "drafter_source_revision": SOURCE_REV,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"staged {arm}: {stage_dir}")

    (STAGE / "publication-manifest.json").write_text(
        json.dumps(
            {
                "collection": COLLECTION,
                "target_revision": TARGET_REV,
                "drafter_source_revision": SOURCE_REV,
                "models": repo_ids,
                "quantized_arms": 10,
                "evaluation_baseline": "bf16-drafter",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
