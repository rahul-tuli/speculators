#!/usr/bin/env python3
"""Restore exact pinned tokenizer files after an in-flight quantizer export.

Use this for runs launched before quantize.py copied local processor files back
after transformers.save_pretrained(). The launch-time source bundle is preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from dquant.quantize import restore_local_processor_snapshot, verify_checkpoint


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--processor", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    processor = args.processor.resolve()
    manifest_path = output / "quant_run_manifest.json"
    plan = json.loads(manifest_path.read_text())
    if plan["scheme"] not in ("nvfp4_gptq", "nvfp4_gptq_imatrix"):
        raise ValueError("post-export tokenizer restoration is for GPTQ arms")
    if Path(plan["processor"]).resolve() != processor:
        raise ValueError("processor differs from the launch-time manifest")
    completion_marker = output / "checkpoint_complete.json"
    if not completion_marker.is_file():
        raise ValueError("checkpoint has no completion marker")
    launch_source = output / "source/dquant/quantize.py"
    if not launch_source.is_file():
        raise ValueError("checkpoint lacks its launch-time quantizer source")
    completion_marker.unlink()

    source_dir = output / "post_export_source"
    source_dir.mkdir(exist_ok=True)
    saved_source_hashes = {}
    for source in (Path(__file__), Path(__file__).with_name("quantize.py")):
        destination = source_dir / source.name
        shutil.copy2(source, destination)
        saved_source_hashes[source.name] = sha256(destination)

    identity = restore_local_processor_snapshot(output, str(processor))
    if not identity or not identity["files"]:
        raise ValueError("local processor snapshot had no tokenizer files")
    record = {
        "action": "post-export tokenizer restoration",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": "transformers.save_pretrained changed ByteLevel tokenizer metadata",
        "processor_identity": identity,
        "launch_quantize_sha256": sha256(launch_source),
        "post_export_source_sha256": saved_source_hashes,
        "source_revisions_at_launch": plan["source_revisions"],
        "launch_source_preserved": True,
    }
    verified = verify_checkpoint(output, plan["scheme"])
    plan["processor_identity"] = identity
    plan["post_export_tokenizer_restoration"] = record
    manifest_path.write_text(json.dumps(plan, indent=2) + "\n")
    (output / "post_export_tokenizer_restoration.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    (output / "post_export_command.txt").write_text(
        "# Timestamp: "
        + record["timestamp"]
        + "\n# Post-export source SHA-256: "
        + json.dumps(saved_source_hashes, sort_keys=True)
        + "\n"
        + shlex.join([sys.executable, *sys.argv])
        + "\n"
    )
    completion_marker.write_text(
        json.dumps({"scheme": plan["scheme"], **verified}, indent=2) + "\n"
    )
    print(  # noqa: T201 - CLI result
        json.dumps({"output": str(output), "processor_revision": identity["revision"]})
    )


if __name__ == "__main__":
    main()
