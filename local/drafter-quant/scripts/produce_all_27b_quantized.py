#!/usr/bin/env python3
"""Produce all 10 quantized drafters for RedHatAI/Qwen3.8-27B-speculator.dspark.

Excludes lm_head, verifier_lm_head, embed_tokens, and all markov/confidence heads
using --keep-heads-bf16.

Runs two GPU queues (GPU 0 and GPU 1). Calibrated arms use all 1,892 prepared
examples capped at 2,048 tokens, and their forward pass limits anchors to the
number of complete blocks per sequence. Both GPTQ calibration pairs use
matching damping.
"""

import os
import subprocess
import sys
from pathlib import Path

PYTHON_BIN = "/workspace/speculators/.venv/bin/python"
SOURCE_DRAFTER = "/root/.cache/huggingface/models--RedHatAI--Qwen3.8-27B-speculator.dspark/snapshots/7f33c272e5da240978e0d55767abab8193d74b95"
PROCESSOR = "Qwen/Qwen3.8-27B"
CALIB_DATA = "/data/fast/drafter-quant/data/Qwen--Qwen3.8-27B-fp8-layers-4-12-20-28-36-44-52-60-h5120-n2048-seq2048-seed0_prepared"
CALIB_SAMPLES = 1892
SEQ_LEN = 2048
GPTQ_DAMPENING_FRAC = "0.1"
OUT_ROOT = Path("/data/fast/models")
LOG_DIR = Path("/data/fast/drafter-quant/logs/quantize_27b")
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# 10 drafter specifications
MODELS = [
    # GPU 0 queue
    {
        "gpu": "0",
        "name": "Qwen3.8-27B-DSpark-FP8-BLOCK",
        "scheme": "fp8_block",
        "calib": "none",
        "extra_args": [],
    },
    {
        "gpu": "0",
        "name": "Qwen3.8-27B-DSpark-Gauss-FP8-W8A8",
        "scheme": "fp8_w8a8",
        "calib": "random",
        "extra_args": [],
    },
    {
        "gpu": "0",
        "name": "Qwen3.8-27B-DSpark-PerfectBlend-FP8-W8A8",
        "scheme": "fp8_w8a8",
        "calib": CALIB_DATA,
        "extra_args": [],
    },
    {
        "gpu": "0",
        "name": "Qwen3.8-27B-DSpark-GPTQ-Gauss-NVFP4-W4A4",
        "scheme": "nvfp4_gptq",
        "calib": "random",
        "extra_args": ["--gptq-dampening-frac", GPTQ_DAMPENING_FRAC],
    },
    {
        "gpu": "0",
        "name": "Qwen3.8-27B-DSpark-GPTQ-PerfectBlend-NVFP4-W4A4",
        "scheme": "nvfp4_gptq",
        "calib": CALIB_DATA,
        "extra_args": ["--gptq-dampening-frac", GPTQ_DAMPENING_FRAC],
    },
    # GPU 1 queue
    {
        "gpu": "1",
        "name": "Qwen3.8-27B-DSpark-FP8-DYNAMIC",
        "scheme": "fp8_dynamic",
        "calib": "none",
        "extra_args": [],
    },
    {
        "gpu": "1",
        "name": "Qwen3.8-27B-DSpark-Gauss-NVFP4-W4A4",
        "scheme": "nvfp4_w4a4",
        "calib": "random",
        "extra_args": [],
    },
    {
        "gpu": "1",
        "name": "Qwen3.8-27B-DSpark-PerfectBlend-NVFP4-W4A4",
        "scheme": "nvfp4_w4a4",
        "calib": CALIB_DATA,
        "extra_args": [],
    },
    {
        "gpu": "1",
        "name": "Qwen3.8-27B-DSpark-GPTQ-IMatrix-Gauss-NVFP4-W4A4",
        "scheme": "nvfp4_gptq_imatrix",
        "calib": "random",
        "extra_args": ["--gptq-dampening-frac", GPTQ_DAMPENING_FRAC],
    },
    {
        "gpu": "1",
        "name": "Qwen3.8-27B-DSpark-GPTQ-IMatrix-PerfectBlend-NVFP4-W4A4",
        "scheme": "nvfp4_gptq_imatrix",
        "calib": CALIB_DATA,
        "extra_args": ["--gptq-dampening-frac", GPTQ_DAMPENING_FRAC],
    },
]


def is_complete(out_dir: Path, scheme: str) -> bool:
    manifest = out_dir / "quant_run_manifest.json"
    weights = list(out_dir.glob("*.safetensors"))
    if not manifest.is_file() or not weights:
        return False
    if scheme in ("fp8_block", "nvfp4_gptq", "nvfp4_gptq_imatrix"):
        marker = out_dir / "checkpoint_complete.json"
        if not marker.is_file():
            return False
    return True


def run_quantization(m: dict) -> None:
    name = m["name"]
    out_dir = OUT_ROOT / name
    log_file = LOG_DIR / f"{name}.log"
    gpu = m["gpu"]
    scheme = m["scheme"]
    calib = m["calib"]
    extra = m["extra_args"]

    if is_complete(out_dir, scheme):
        print(f"[{name}] ALREADY COMPLETE at {out_dir}. Skipping.")
        return

    print(f"[{name}] Starting on GPU {gpu} -> {out_dir}")
    cmd = [
        PYTHON_BIN,
        "-m",
        "dquant.quantize",
        "--scheme",
        scheme,
        "--model",
        SOURCE_DRAFTER,
        "--output-dir",
        str(out_dir),
        "--processor",
        PROCESSOR,
        "--calib-data",
        calib,
        "--num-calibration-samples",
        str(CALIB_SAMPLES),
        "--seq-len",
        str(SEQ_LEN),
        "--seed",
        "0",
        "--keep-heads-bf16",
        "--device",
        "cuda:0",
        *extra,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = "/workspace/speculators/local/drafter-quant"

    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            cmd,
            cwd="/workspace/speculators/local/drafter-quant",
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
        )
        ret = proc.wait()

    if ret != 0:
        print(f"[{name}] FAILED with code {ret}. See {log_file}")
        raise RuntimeError(f"Quantization failed for {name}, see {log_file}")

    print(f"[{name}] FINISHED successfully -> {out_dir}")


def run_worker_queue(gpu_id: str) -> None:
    queue = [m for m in MODELS if m["gpu"] == gpu_id]
    for m in queue:
        run_quantization(m)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", choices=["0", "1", "all"], default="all")
    args = parser.parse_args()

    if args.gpu in ("0", "1"):
        run_worker_queue(args.gpu)
    else:
        # Launch both workers in parallel
        p0 = subprocess.Popen([PYTHON_BIN, __file__, "--gpu", "0"])
        p1 = subprocess.Popen([PYTHON_BIN, __file__, "--gpu", "1"])
        ret0 = p0.wait()
        ret1 = p1.wait()
        if ret0 != 0 or ret1 != 0:
            sys.exit(1)
        print("All 10 quantizations completed successfully!")
