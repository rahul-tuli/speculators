import hashlib
import os
import shutil
from pathlib import Path

STAGE_ROOT = Path("/data/fast/drafter-quant/hf-publication-20260928")
BASE_OUT = Path("/data/fast/drafter-quant/out/add-evals-20260928")
EVAL_BASE = Path("/data/fast/drafter-quant/results/add-evals-20260928-verified")
GITATTRIBUTES_SRC = Path(
    "/data/fast/drafter-quant/hf-publication-20260924/Drift8/.gitattributes"
)

MODELS = [
    {
        "nickname": "Block8",
        "repo_id": "inference-optimization/Qwen3-8B-DFlash-FP8-BLOCK",
        "title": "Block8-FP8-BLOCK — Data-free FP8 Block quantization",
        "tags": [
            "speculative-decoding",
            "dflash",
            "qwen3",
            "quantized",
            "fp8",
            "block-quantization",
            "data-free",
        ],
        "ckpt_dir": BASE_OUT / "fp8_block-datafree",
        "eval_redhat": EVAL_BASE
        / "redhatai-speculator-benchmarks"
        / "fp8-block-datafree",
        "eval_speed": EVAL_BASE / "speedbench-qualitative" / "fp8-block-datafree",
        "variant_desc": '- **Quantization:** FP8 Block (`model_free_ptq(scheme="FP8_BLOCK")` with static 128x128 blocks and dynamic 128 groups).\n- **Calibration:** Data-free quantization; no calibration data or samples were used.\n- **Source snapshot:** Pinned newer BF16 drafter snapshot `1a11b170eb65c8a62c80ecd01dfe22a5907298e6` (block size 16, draft vocabulary 151,936, sliding-window attention).',
        "speed_l": "3.1535",
        "speed_ratio": "30.76%",
        "redhat_l": "3.1410",
        "redhat_ratio": "30.59%",
    },
    {
        "nickname": "GPTQ-Gauss4",
        "repo_id": "inference-optimization/Qwen3-8B-DFlash-GPTQ-Gauss-NVFP4-W4A4",
        "title": "GPTQ-Gauss4-NVFP4-W4A4 — NVFP4 W4A4 with GPTQ (Gaussian calibration)",
        "tags": [
            "speculative-decoding",
            "dflash",
            "qwen3",
            "quantized",
            "nvfp4",
            "gptq",
            "gaussian-calibration",
        ],
        "ckpt_dir": BASE_OUT / "nvfp4_gptq-random",
        "eval_redhat": EVAL_BASE
        / "redhatai-speculator-benchmarks"
        / "nvfp4-gptq-gaussian",
        "eval_speed": EVAL_BASE / "speedbench-qualitative" / "nvfp4-gptq-gaussian",
        "variant_desc": "- **Quantization:** NVFP4 W4A4 via `GPTQModifier` (`nvfp4_expanded_mse`).\n- **Calibration:** Synthetic Gaussian batches (2,027 batches, sequence length 2,048, seed 0).\n- **Hessian Damping:** 0.01 (36/36 modules quantized, 0 RTN fallbacks).\n- **Source snapshot:** Pinned newer BF16 drafter snapshot `1a11b170eb65c8a62c80ecd01dfe22a5907298e6` (block size 16, draft vocabulary 151,936, sliding-window attention).",
        "speed_l": "2.0459",
        "speed_ratio": "14.94%",
        "redhat_l": "2.0926",
        "redhat_ratio": "15.61%",
    },
    {
        "nickname": "GPTQ-Blend4",
        "repo_id": "inference-optimization/Qwen3-8B-DFlash-GPTQ-PerfectBlend-NVFP4-W4A4",
        "title": "GPTQ-Blend4-NVFP4-W4A4 — NVFP4 W4A4 with GPTQ (PerfectBlend real calibration)",
        "tags": [
            "speculative-decoding",
            "dflash",
            "qwen3",
            "quantized",
            "nvfp4",
            "gptq",
            "perfectblend-calibration",
        ],
        "ckpt_dir": BASE_OUT / "damping0p1" / "nvfp4_gptq-real",
        "eval_redhat": EVAL_BASE
        / "redhatai-speculator-benchmarks"
        / "nvfp4-gptq-perfectblend",
        "eval_speed": EVAL_BASE / "speedbench-qualitative" / "nvfp4-gptq-perfectblend",
        "variant_desc": "- **Quantization:** NVFP4 W4A4 via `GPTQModifier` (`nvfp4_expanded_mse`).\n- **Calibration:** Real target hidden states from proportional sample of `inference-optimization/Qwen3-8B-Regenerated-Collection` (2,027 prepared rows, 3,462,008 actual tokens, seed 0).\n- **Hessian Damping:** 0.1 (adjusted to avoid Cholesky failure on layers.0.self_attn.q_proj; 36/36 modules quantized with 0 RTN fallbacks).\n- **Source snapshot:** Pinned newer BF16 drafter snapshot `1a11b170eb65c8a62c80ecd01dfe22a5907298e6` (block size 16, draft vocabulary 151,936, sliding-window attention).",
        "speed_l": "3.0048",
        "speed_ratio": "28.64%",
        "redhat_l": "3.0260",
        "redhat_ratio": "28.94%",
    },
    {
        "nickname": "IMatrix-Gauss4",
        "repo_id": "inference-optimization/Qwen3-8B-DFlash-GPTQ-IMatrix-Gauss-NVFP4-W4A4",
        "title": "IMatrix-Gauss4-NVFP4-W4A4 — NVFP4 W4A4 with GPTQ + IMatrix (Gaussian calibration)",
        "tags": [
            "speculative-decoding",
            "dflash",
            "qwen3",
            "quantized",
            "nvfp4",
            "gptq",
            "imatrix",
            "gaussian-calibration",
        ],
        "ckpt_dir": BASE_OUT / "nvfp4_gptq_imatrix-random",
        "eval_redhat": EVAL_BASE
        / "redhatai-speculator-benchmarks"
        / "nvfp4-gptq-imatrix-gaussian",
        "eval_speed": EVAL_BASE
        / "speedbench-qualitative"
        / "nvfp4-gptq-imatrix-gaussian",
        "variant_desc": "- **Quantization:** NVFP4 W4A4 via `GPTQModifier` with Importance Matrix (`nvfp4_expanded_imatrix`, `strict: true`).\n- **Calibration:** Synthetic Gaussian batches (2,027 batches, sequence length 2,048, seed 0).\n- **Hessian Damping:** 0.01 (36/36 modules quantized, 0 RTN fallbacks).\n- **Source snapshot:** Pinned newer BF16 drafter snapshot `1a11b170eb65c8a62c80ecd01dfe22a5907298e6` (block size 16, draft vocabulary 151,936, sliding-window attention).",
        "speed_l": "2.0579",
        "speed_ratio": "15.11%",
        "redhat_l": "2.1055",
        "redhat_ratio": "15.79%",
    },
    {
        "nickname": "IMatrix-Blend4",
        "repo_id": "inference-optimization/Qwen3-8B-DFlash-GPTQ-IMatrix-PerfectBlend-NVFP4-W4A4",
        "title": "IMatrix-Blend4-NVFP4-W4A4 — NVFP4 W4A4 with GPTQ + IMatrix (PerfectBlend real calibration)",
        "tags": [
            "speculative-decoding",
            "dflash",
            "qwen3",
            "quantized",
            "nvfp4",
            "gptq",
            "imatrix",
            "perfectblend-calibration",
        ],
        "ckpt_dir": BASE_OUT / "damping0p1" / "nvfp4_gptq_imatrix-real",
        "eval_redhat": EVAL_BASE
        / "redhatai-speculator-benchmarks"
        / "nvfp4-gptq-imatrix-perfectblend",
        "eval_speed": EVAL_BASE
        / "speedbench-qualitative"
        / "nvfp4-gptq-imatrix-perfectblend",
        "variant_desc": "- **Quantization:** NVFP4 W4A4 via `GPTQModifier` with Importance Matrix (`nvfp4_expanded_imatrix`, `strict: true`).\n- **Calibration:** Real target hidden states from proportional sample of `inference-optimization/Qwen3-8B-Regenerated-Collection` (2,027 prepared rows, 3,462,008 actual tokens, seed 0).\n- **Hessian Damping:** 0.1 (adjusted to avoid Cholesky failure on layers.0.self_attn.q_proj; 36/36 modules quantized with 0 RTN fallbacks).\n- **Source snapshot:** Pinned newer BF16 drafter snapshot `1a11b170eb65c8a62c80ecd01dfe22a5907298e6` (block size 16, draft vocabulary 151,936, sliding-window attention).",
        "speed_l": "3.0025",
        "speed_ratio": "28.61%",
        "redhat_l": "3.0250",
        "redhat_ratio": "28.93%",
    },
]


def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def make_readme(m):
    tags_str = "\n".join(f"- {t}" for t in m["tags"])
    return f"""---
library_name: speculators
base_model:
- RedHatAI/Qwen3-8B-speculator.dflash
- Qwen/Qwen3-8B
license: apache-2.0
pipeline_tag: text-generation
tags:
{tags_str}
---

# {m["title"]}

**{m["nickname"]}** is a quantized DFlash drafter for the [Qwen3-8B target](https://huggingface.co/Qwen/Qwen3-8B), derived from [RedHatAI/Qwen3-8B-speculator.dflash](https://huggingface.co/RedHatAI/Qwen3-8B-speculator.dflash) at pinned revision `1a11b170eb65c8a62c80ecd01dfe22a5907298e6`. This repository contains the drafter component; it is not a standalone chat model.

## Variant

{m["variant_desc"]}

## Use with vLLM

Pair this drafter with the Qwen3-8B target and a DFlash-capable vLLM build:

```bash
vllm serve Qwen/Qwen3-8B \\
  --spec-model {m["repo_id"]} \\
  --spec-tokens 7 \\
  --spec-method dflash
```

`config.py` provides the custom drafter configuration. Serving command and runtime patch are preserved in `provenance/evaluation/`.

## Evaluation Performance

Evaluated with 7 speculative tokens per draft event against the `Qwen/Qwen3-8B` target:

| Benchmark | Mean Acceptance Length ($L = 1 + A/D$) | Mean Accepted Ratio ($A/P$) |
|---|---|---|
| **SPEED-Bench Qualitative** (11 categories, 880 requests) | {m["speed_l"]} tokens | {m["speed_ratio"]} |
| **RedHatAI / speculator_benchmarks** (9 subsets, 924 requests) | {m["redhat_l"]} tokens | {m["redhat_ratio"]} |

## Reproducibility & Provenance

Full provenance and reproducibility artifacts are preserved:
- Root directory contains model weights, configuration, tokenizer, quantization manifests, and recipe.
- `provenance/` contains:
  - `train_command.txt`: Source model training command.
  - `train_command_scope.txt`: Clarification of source drafter snapshot lineage.
  - `quantization_command.txt`: Exact command and environment used to run quantization.
  - `quantization_source/`: Source code snapshot of quantizer utilities.
  - `evaluation/`: Contains vLLM serving commands (`vllm_command.txt`), vLLM runtime patch (`vllm.patch`), speculators patch (`speculators.patch`), target/drafter SHA-256 hashes, and per-subset evaluation commands.
  - `publication_sha256.txt`: SHA-256 hashes of all published files.

The source drafter lists Apache-2.0 licensing on its [Hugging Face model card](https://huggingface.co/RedHatAI/Qwen3-8B-speculator.dflash).
"""


def stage_all():
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)

    for m in MODELS:
        nick = m["nickname"]
        out_dir = STAGE_ROOT / nick
        print(f"\n================ Staging {nick} -> {out_dir} ================")
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Copy .gitattributes
        shutil.copy2(GITATTRIBUTES_SRC, out_dir / ".gitattributes")

        # 2. Hard-link or copy files from checkpoint dir
        ckpt_dir = m["ckpt_dir"]
        for f in ckpt_dir.iterdir():
            if f.is_file():
                dst = out_dir / f.name
                if dst.exists():
                    dst.unlink()
                # Hard link large safetensors, copy small files
                if f.suffix in [".safetensors"]:
                    os.link(f, dst)
                else:
                    shutil.copy2(f, dst)

        # 3. Write README.md
        readme_content = make_readme(m)
        (out_dir / "README.md").write_text(readme_content, encoding="utf-8")

        # 4. Create provenance directory
        prov_dir = out_dir / "provenance"
        prov_dir.mkdir(parents=True, exist_ok=True)

        # 4a. train_command.txt
        if (ckpt_dir / "train_command.txt").exists():
            shutil.copy2(ckpt_dir / "train_command.txt", prov_dir / "train_command.txt")

        # 4b. train_command_scope.txt
        scope_text = (
            "This command was captured from the source RedHatAI/Qwen3-8B-speculator.dflash snapshot revision "
            "`1a11b170eb65c8a62c80ecd01dfe22a5907298e6` (model.safetensors SHA-256 `afda76166ccd6a2d3ef55b214889bd045be3b8fc41b41be330d06d25059f88e8`). "
            "This quantized checkpoint was produced directly from that pinned BF16 snapshot (block size 16, draft vocabulary 151,936, sliding-window attention).\n"
        )
        (prov_dir / "train_command_scope.txt").write_text(scope_text, encoding="utf-8")

        # 4c. quantization_command.txt
        if (ckpt_dir / "quant_command.txt").exists():
            shutil.copy2(
                ckpt_dir / "quant_command.txt", prov_dir / "quantization_command.txt"
            )

        # 4d. quantization_source/
        src_dir = ckpt_dir / "source"
        if src_dir.exists() and src_dir.is_dir():
            prov_src = prov_dir / "quantization_source"
            if prov_src.exists():
                shutil.rmtree(prov_src)
            shutil.copytree(src_dir, prov_src)

        # 4e. evaluation/
        eval_dir = prov_dir / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)

        # From RedHat eval:
        er = m["eval_redhat"]
        es = m["eval_speed"]

        if (er / "checkpoint_sha256.txt").exists():
            shutil.copy2(
                er / "checkpoint_sha256.txt", eval_dir / "checkpoint_sha256.txt"
            )
        if (er / "drafter_checkpoint_sha256.txt").exists():
            shutil.copy2(
                er / "drafter_checkpoint_sha256.txt",
                eval_dir / "drafter_checkpoint_sha256.txt",
            )
        if (er / "vllm_command.txt").exists():
            shutil.copy2(er / "vllm_command.txt", eval_dir / "vllm_command.txt")
        if (er / "vllm.patch").exists():
            shutil.copy2(er / "vllm.patch", eval_dir / "vllm.patch")
        if (er / "speculators.patch").exists():
            shutil.copy2(er / "speculators.patch", eval_dir / "speculators.patch")

        if (er / "eval_command.txt").exists():
            shutil.copy2(
                er / "eval_command.txt", eval_dir / "redhatai_eval_command.txt"
            )
        if (es / "eval_command.txt").exists():
            shutil.copy2(
                es / "eval_command.txt", eval_dir / "speedbench_eval_command.txt"
            )

        # Per-subset eval commands
        eval_cmds_dir = eval_dir / "eval_commands"
        eval_cmds_dir.mkdir(parents=True, exist_ok=True)

        # RedHat subsets
        rh_sub = er / "subsets"
        if rh_sub.exists():
            for s in rh_sub.iterdir():
                if s.is_dir():
                    cmd_file = s / "attempt-1" / "eval_command.txt"
                    if cmd_file.exists():
                        target_sub = eval_cmds_dir / "redhatai" / s.name
                        target_sub.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(cmd_file, target_sub / "eval_command.txt")

        # Speedbench subsets
        sb_sub = es / "subsets"
        if sb_sub.exists():
            for s in sb_sub.iterdir():
                if s.is_dir():
                    # check attempt-1 or attempt-2
                    cmd_file = s / "attempt-1" / "eval_command.txt"
                    if not cmd_file.exists():
                        cmd_file = s / "attempt-2" / "eval_command.txt"
                    if cmd_file.exists():
                        target_sub = eval_cmds_dir / "speedbench" / s.name
                        target_sub.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(cmd_file, target_sub / "eval_command.txt")

        # 5. Compute publication_sha256.txt (excluding publication_sha256.txt itself)
        pub_sha_file = prov_dir / "publication_sha256.txt"
        if pub_sha_file.exists():
            pub_sha_file.unlink()

        all_files = sorted([p for p in out_dir.rglob("*") if p.is_file()])
        lines = []
        for p in all_files:
            rel = p.relative_to(out_dir)
            if str(rel) == "provenance/publication_sha256.txt":
                continue
            h = sha256_file(p)
            lines.append(f"{h}  {rel}")

        pub_sha_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Staged {len(lines) + 1} files in {out_dir}")


if __name__ == "__main__":
    stage_all()
