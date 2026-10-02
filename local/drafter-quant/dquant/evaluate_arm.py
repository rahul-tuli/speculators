"""Serve one arm and run immutable fixed-load points; called by 40_eval.sh.

All model paths are resolved before launching so provenance hashes real weights.
Repeated invocations validate completed points and retain interrupted attempts.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/evaluate"))
from point import canonical, digest, save  # noqa: E402

from dquant.serving_inventory import (  # noqa: E402
    validate_dspark_head_scope,
    validate_export_inventory,
)


def resolved_model(value):
    path = Path(value)
    if path.is_dir():
        return path.resolve()
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    return Path(
        snapshot_download(
            value,
            local_files_only=True,
            allow_patterns=[
                "*.safetensors",
                "*.json",
                "*.py",
                "*.jinja",
                "*.model",
                "merges.txt",
                "vocab.txt",
            ],
        )
    ).resolve()


def model_identity(path):
    files = sorted(
        p
        for p in path.iterdir()
        if p.is_file() and p.suffix in {".safetensors", ".json", ".jinja", ".py"}
    )
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError(f"No checkpoint weights in {path}")
    return {"path": str(path), "files": {p.name: digest(p) for p in files}}


def rpc(port, method):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/collective_rpc",
        data=json.dumps({"method": method}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310
        return json.load(response)


def inventory_check(inventory, arm):  # noqa: C901
    models = inventory["results"][0]["models"]
    if arm == "target-only":
        if "drafter" in models:
            raise ValueError("Target-only unexpectedly has a drafter")
        return
    drafter = models["drafter"]
    quantized = [m for m in drafter.values() if m["scheme"] != "NoneType"]
    if arm.startswith(("fp8", "nvfp4")) and not quantized:
        raise ValueError("Expected quantized drafter modules")
    for m in quantized:
        kernels = list(m["kernels"].values())
        if arm.startswith("nvfp4") and "EmulationNvFp4LinearKernel" not in kernels:
            raise ValueError(f"NVFP4 requires explicit W4A4 emulation: {kernels}")
        if arm.startswith("nvfp4") and m["use_a16"] is not False:
            raise ValueError("NVFP4 activation quantization is not enabled")
        if arm.startswith("fp8") and not any(
            "Kernel" in k and ("Fp8" in k or "FP8" in k) for k in kernels
        ):
            raise ValueError(f"FP8 kernel inventory incomplete: {kernels}")
        for name, param in m["parameters"].items():
            if "scale" in name and (not param["finite"] or param["min"] <= 0):
                raise ValueError(f"Invalid runtime scale: {name}")
    if arm == "bf16-drafter" and quantized:
        raise ValueError("BF16 reference unexpectedly quantized")


def environment():
    names = [
        "vllm",
        "torch",
        "transformers",
        "compressed-tensors",
        "llmcompressor",
        "speculators",
        "guidellm",
        "datasets",
    ]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    repos = {}
    for name in ["speculators", "vllm", "compressed-tensors", "llm-compressor"]:
        path = ROOT.parent / name
        sha = subprocess.check_output(  # noqa: S603
            ["git", "-C", str(path), "rev-parse", "HEAD"],  # noqa: S607
            text=True,
        ).strip()
        diff = subprocess.check_output(  # noqa: S603
            ["git", "-C", str(path), "diff", "HEAD", "--binary"],  # noqa: S607
            text=True,
        )
        repos[name] = {"sha": sha, "patch": diff}
    sources = {}
    for folder in [ROOT / "scripts/evaluate", ROOT / "local/drafter-quant/dquant"]:
        for p in sorted(folder.glob("*.py")):
            sources[str(p.relative_to(ROOT))] = digest(p)
    sources["scripts/launch_vllm.py"] = digest(ROOT / "scripts/launch_vllm.py")
    runtime_sources = {}
    for package in ["vllm", "compressed_tensors", "guidellm"]:
        spec = importlib.util.find_spec(package)
        folder = Path(spec.origin).parent
        hashes = {
            str(p.relative_to(folder)): digest(p) for p in sorted(folder.rglob("*.py"))
        }
        runtime_sources[package] = {
            "path": str(folder),
            "sha256": hashlib.sha256(canonical(hashes).encode()).hexdigest(),
            "files": hashes,
        }
    hardware = subprocess.check_output(
        [  # noqa: S607
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
    )
    return {
        "hardware": hardware,
        "runtime_sources": runtime_sources,
        "versions": versions,
        "repos": repos,
        "sources": sources,
        "flags": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("VLLM_", "CUDA_VISIBLE_DEVICES"))
        },
        "thread_limits": {
            key: os.environ.get(key)
            for key in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "TOKENIZERS_PARALLELISM",
            )
        },
    }


def main():  # noqa: C901
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", required=True)
    p.add_argument(
        "--drafter", required=True, help="Local checkpoint, cached HF ID, or none"
    )
    p.add_argument("--arm", required=True)
    p.add_argument("--method", default="dflash")
    p.add_argument("--datasets", nargs="+", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeat", type=int, required=True)
    p.add_argument("--order", type=int, required=True)
    p.add_argument(
        "--phase", choices=["pilot", "acceptance", "speed"], default="acceptance"
    )
    p.add_argument("--concurrencies", nargs="+", type=int, default=[1])
    p.add_argument("--warmup-requests", type=int, default=2)
    p.add_argument("--port", type=int, default=8108)
    args = p.parse_args()
    # Claim the port before hashing/loading, when no server is listening yet.
    port_lock = Path(f"/tmp/speculators-evaluation-port-{args.port}.lock").open("w")  # noqa: SIM115, S108
    fcntl.flock(port_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    target = resolved_model(args.target)
    drafter = None if args.drafter == "none" else resolved_model(args.drafter)
    if (drafter is None) != (args.arm == "target-only"):
        raise ValueError("Use distinct bf16-drafter and target-only references")
    if args.phase == "acceptance" and (
        drafter is None or any(c < 1 for c in args.concurrencies)
    ):
        raise ValueError(
            "Acceptance requires a drafter and positive concurrency values"
        )
    if args.method not in ("dflash", "dspark"):
        raise ValueError(
            f"Unsupported method: {args.method}. Expected 'dflash' or 'dspark'."
        )
    os.environ["VLLM_BATCH_INVARIANT"] = "0"
    os.environ["VLLM_SERVER_DEV_MODE"] = "1"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    tp = int(os.environ.get("TP", "2" if "27B" in str(target) else "1"))
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = (
            ",".join(str(i) for i in range(tp)) if tp > 1 else "0"
        )
    os.environ["PYTHONPATH"] = (
        str(ROOT / "local/drafter-quant")
        + os.pathsep
        + os.environ.get("PYTHONPATH", "")
    )
    calibration = {}
    # FP8_DYNAMIC and FP8_BLOCK are data-free model-free PTQ arms. Neither
    # has a calibration manifest or consumes the random/real datasets.
    data_free = args.arm.startswith(("fp8-dynamic", "fp8-block"))
    if drafter and args.arm.startswith(("fp8", "nvfp4")):
        manifest_names = ["quant_run_manifest.json"]
        if not data_free:
            manifest_names.insert(0, "calibration_manifest.json")
        calibration = {
            name: json.loads((drafter / name).read_text()) for name in manifest_names
        }
        if args.method == "dspark":
            expected_schemes = {
                "fp8-block": "fp8_block",
                "fp8-dynamic": "fp8_dynamic",
                "fp8-gaussian": "fp8_w8a8",
                "fp8-real": "fp8_w8a8",
                "nvfp4-gaussian": "nvfp4_w4a4",
                "nvfp4-real": "nvfp4_w4a4",
                "nvfp4-gptq-gaussian": "nvfp4_gptq",
                "nvfp4-gptq-real": "nvfp4_gptq",
                "nvfp4-gptq-imatrix-gaussian": "nvfp4_gptq_imatrix",
                "nvfp4-gptq-imatrix-real": "nvfp4_gptq_imatrix",
            }
            run = calibration["quant_run_manifest.json"]
            expected_scheme = expected_schemes.get(args.arm)
            if expected_scheme is None or run.get("scheme") != expected_scheme:
                raise ValueError(
                    f"Quantization scheme mismatch for {args.arm}: "
                    f"{run.get('scheme')} != {expected_scheme}"
                )
            if data_free:
                if (
                    run.get("calibration") != "none (data-free)"
                    or run.get("num_calibration_samples") != 0
                ):
                    raise ValueError(f"{args.arm} must be data-free")
            else:
                budget = calibration["calibration_manifest.json"]
                if args.arm.endswith("gaussian"):
                    if budget.get("source") != "random":
                        raise ValueError(f"{args.arm} requires Gaussian calibration")
                else:
                    expected_calibration = Path(
                        os.environ.get(
                            "CALIBRATION_DATA",
                            "/data/fast/drafter-quant/data/"
                            "Qwen--Qwen3.8-27B-fp8-layers-4-12-20-28-36-44-52-60-"
                            "h5120-n2048-seq2048-seed0_prepared",
                        )
                    ).resolve()
                    reported_calibration = run.get("calibration", "")
                    if (
                        not reported_calibration.startswith("real: ")
                        or Path(reported_calibration.removeprefix("real: ")).resolve()
                        != expected_calibration
                    ):
                        raise ValueError(f"{args.arm} calibration path mismatch")
                    if (
                        Path(budget.get("cache_path", "")).resolve()
                        != expected_calibration
                    ):
                        raise ValueError(f"{args.arm} calibration cache mismatch")
                if (
                    budget.get("actual_samples") != 1892
                    or budget.get("seq_len") != 2048
                    or budget.get("seed")
                    != int(os.environ.get("CALIBRATION_SEED", "0"))
                    or budget.get("block_size") != 8
                    or budget.get("max_anchors") != 256
                ):
                    raise ValueError(
                        f"Invalid DSpark calibration budget/settings for {args.arm}"
                    )
            if expected_scheme in ("nvfp4_gptq", "nvfp4_gptq_imatrix"):
                expected_observer = (
                    "nvfp4_expanded_imatrix"
                    if expected_scheme.endswith("imatrix")
                    else "nvfp4_expanded_mse"
                )
                if (
                    run.get("weight_observer") != expected_observer
                    or run.get("gptq_dampening_frac") != 0.1
                ):
                    raise ValueError(f"Unexpected GPTQ observer/damping for {args.arm}")
        if args.phase != "pilot" and not data_free:
            budget = calibration["calibration_manifest.json"]
            run = calibration["quant_run_manifest.json"]
            expected_samples = int(
                os.environ.get(
                    "EXPECTED_CALIB_SAMPLES",
                    "2048" if args.method == "dspark" else "2027",
                )
            )
            if (
                budget["actual_samples"] != expected_samples
                or budget["seq_len"] != 2048
            ):  # noqa: PLR2004
                raise ValueError(
                    f"Full experiment requires {expected_samples} calibration sequences capped at 2048 tokens"  # noqa: E501
                )
            if run.get("rng_policy") != "python-numpy-torch-cuda-and-per-row":
                raise ValueError("Checkpoint predates complete calibration RNG control")
            if run["seed"] != int(os.environ.get("CALIBRATION_SEED", "0")):
                raise ValueError("Calibration seed mismatch")
    manifest = json.loads(args.manifest.read_text())
    for path in args.datasets:
        if manifest["files"].get(path.name, {}).get("sha256") != digest(path):
            raise ValueError(
                f"Dataset absent from or differs from frozen manifest: {path}"
            )
    launch_flags = [
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--dtype",
        "bfloat16",
        "--seed",
        "0",
        "--max-model-len",
        str(manifest["max_model_len"]),
        "--max-num-seqs",
        "128",
        "--gpu-memory-utilization",
        "0.8",
        "--no-enable-prefix-caching",
        "--worker-extension-cls",
        "dquant.runtime_probe.RuntimeProbe",
    ]
    # Eager is an explicit initial validation/acceptance treatment. Speed is a later gate.  # noqa: E501
    if tp > 1 and "--tensor-parallel-size" not in launch_flags:
        launch_flags += ["--tensor-parallel-size", str(tp)]
    if args.phase != "speed":
        launch_flags.append("--enforce-eager")
    if args.arm.startswith("nvfp4"):
        launch_flags += ["--kernel-config", '{"linear_backend":"emulation"}']
    spec_tokens = int(
        os.environ.get("SPEC_TOKENS", "8" if args.method == "dspark" else "7")
    )
    identity = {
        "arm": args.arm,
        "calibration": calibration,
        "target": model_identity(target),
        "drafter": model_identity(drafter) if drafter else None,
        "manifest": manifest,
        "phase": args.phase,
        "repeat": args.repeat,
        "order": args.order,
        "spec_tokens": spec_tokens if drafter else None,
        "launch_flags": launch_flags,
        "environment": environment(),
    }
    run_id = hashlib.sha256(canonical(identity).encode()).hexdigest()[:20]
    root = args.output / args.phase / args.arm / f"repeat-{args.repeat}" / run_id
    root.mkdir(parents=True, exist_ok=True)
    save(root / "run-identity.json", identity)
    # Refuse to share a serving port with another run.
    import socket  # noqa: PLC0415

    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", args.port)) == 0:
            raise RuntimeError(f"Port {args.port} is already in use")
    server_dir = root / f"server-{time.time_ns()}"
    server_dir.mkdir()
    save(server_dir / "environment.json", identity["environment"])
    for name, sha in identity["environment"]["sources"].items():
        destination = server_dir / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
        if digest(destination) != sha:
            raise ValueError("Source changed during provenance capture")
    cmd = [
        sys.executable,
        str(ROOT / "scripts/launch_vllm.py"),
        "eval",
        str(target),
        "--provenance-dir",
        str(server_dir),
    ]
    if drafter:
        cmd += [
            "--spec-model",
            str(drafter),
            "--spec-method",
            args.method,
            "--spec-tokens",
            str(spec_tokens),
        ]
    cmd += ["--", *launch_flags]
    with (server_dir / "server.log").open("w") as log:
        process = subprocess.Popen(  # noqa: S603
            cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            for _ in range(240):
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited; see {server_dir}")
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{args.port}/health", timeout=2
                    ):
                        break
                except OSError:
                    time.sleep(5)
            else:
                raise TimeoutError("Server readiness timed out")
            for name in ["vllm_command.txt", "vllm.patch", "checkpoint_sha256.txt"] + (
                ["drafter_checkpoint_sha256.txt"] if drafter else []
            ):
                if not (server_dir / name).exists():
                    raise RuntimeError(f"Missing mandatory provenance: {name}")
            inventory = rpc(args.port, "quant_inventory")
            save(server_dir / "runtime-inventory.json", inventory)
            inventory_check(inventory, args.arm)
            if args.arm.startswith(("fp8", "nvfp4")):
                audit = validate_export_inventory(
                    drafter,
                    inventory["results"][0]["models"]["drafter"],
                    tensor_parallel_size=tp,
                )
            if args.method == "dspark":
                head_audit = validate_dspark_head_scope(
                    drafter, inventory["results"][0]["models"]["drafter"]
                )
                save(server_dir / "dspark-head-scope-audit.json", head_audit)
                if args.arm.startswith(("fp8", "nvfp4")):
                    audit["dspark_head_scope"] = head_audit
                    save(server_dir / "export-runtime-audit.json", audit)
            if args.phase == "pilot":
                rpc(args.port, "begin_finite_probe")
                request = urllib.request.Request(
                    f"http://127.0.0.1:{args.port}/v1/chat/completions",
                    data=json.dumps(
                        {
                            "model": str(target),
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "Write a short Python factorial function.",  # noqa: E501
                                }
                            ],
                            "temperature": 0,
                            "seed": 0,
                            "max_tokens": 64,
                        }
                    ).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310
                    save(server_dir / "finite-response.json", json.load(response))
                finite = rpc(args.port, "end_finite_probe")
                save(server_dir / "finite-probe.json", finite)
                observations = finite["results"][0]
                if not observations or not all(
                    v["finite"] for v in observations.values()
                ):
                    raise ValueError("Nonfinite runtime outputs")
                if drafter and not any(k.startswith("drafter.") for k in observations):
                    raise ValueError("Finite probe did not execute drafter")
                for name, module in (
                    inventory["results"][0]["models"].get("drafter", {}).items()
                ):
                    if (
                        module["scheme"] != "NoneType"
                        and f"drafter.{name}" not in observations
                    ):
                        raise ValueError(f"Quantized module not exercised: {name}")
            # The inventory is part of resume identity; server timestamps are not.
            point_identity = dict(identity, inventory=inventory)
            save(root / "point-identity.json", point_identity)
            for concurrency in args.concurrencies:
                for dataset in args.datasets:
                    dest = root / f"concurrency-{concurrency}" / dataset.stem
                    eval_cmd = [
                        sys.executable,
                        str(ROOT / "scripts/evaluate/evaluate.py"),
                        "point",
                        "--target",
                        f"http://127.0.0.1:{args.port}/v1",
                        "--dataset",
                        str(dataset),
                        "--identity",
                        str(root / "point-identity.json"),
                        "--output-dir",
                        str(dest),
                        "--max-concurrency",
                        str(concurrency),
                        "--warmup-requests",
                        str(args.warmup_requests),
                    ]
                    if drafter is None:
                        eval_cmd.append("--target-only")
                    active_pids = subprocess.check_output(
                        [  # noqa: S607
                            "nvidia-smi",
                            "--query-compute-apps=pid",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                    ).split()
                    if any(os.getpgid(int(pid)) != process.pid for pid in active_pids):
                        raise RuntimeError(
                            "Another GPU workload violates measurement isolation"
                        )
                    subprocess.run(eval_cmd, check=True)  # noqa: S603
            save(
                server_dir / "finished.json",
                {
                    "status": "complete",
                    "run_id": run_id,
                    "datasets": [str(p) for p in args.datasets],
                    "concurrencies": args.concurrencies,
                },
            )
        except Exception as exc:
            save(
                server_dir / "failure.json",
                {"type": type(exc).__name__, "message": str(exc)},
            )
            raise
        finally:
            os.killpg(process.pid, signal.SIGTERM) if process.poll() is None else None
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    print(root)  # noqa: T201


if __name__ == "__main__":
    main()
