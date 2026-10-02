"""Build a manifest for the frozen per-subset evaluation files.

The bounded pilot was materialized as one combined ``selected.jsonl`` file,
while the frozen directory also retains one JSONL file per benchmark subset.
The point evaluator validates every input against a manifest, so this helper
derives a manifest that covers both representations and verifies that the
subset files are exactly the rows selected by the canonical manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--selected-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subset-data-dir", type=Path)
    parser.add_argument("--expected-per-file", type=int, default=8)
    args = parser.parse_args()

    base = json.loads(args.base_manifest.read_text(encoding="utf-8"))
    canonical = json.loads(args.canonical_manifest.read_text(encoding="utf-8"))
    selected = canonical.get("selection", {}).get("selected_rows", [])
    if not selected:
        raise ValueError("canonical manifest has no selected rows")

    expected: dict[str, list[int]] = defaultdict(list)
    for row in selected:
        try:
            expected[row["source_file"]].append(int(row["source_index"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed selected row: {row!r}") from exc

    files = dict(base.get("files", {}))
    subset_names = sorted(expected)
    if len(subset_names) != 9:
        raise ValueError(f"expected 9 benchmark subsets, found {len(subset_names)}")

    selected_path = args.selected_file or args.data_dir / "selected.jsonl"
    selected_rows = rows(selected_path)
    if len(selected_rows) != len(selected):
        raise ValueError("selected.jsonl row count differs from canonical manifest")

    for name in subset_names:
        source_path = args.data_dir / name
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source_rows = rows(source_path)
        materialized = []
        for row in source_rows:
            if isinstance(row.get("prompt"), str):
                materialized.append({"turns": row["prompt"]})
            elif (
                isinstance(row.get("messages"), list)
                and row["messages"]
                and row["messages"][0].get("role") == "user"
                and isinstance(row["messages"][0].get("content"), str)
            ):
                materialized.append({"turns": row["messages"][0]["content"]})
            else:
                raise ValueError(f"{name}: cannot materialize source row")
        source_rows = materialized
        source_indices = expected[name]
        if len(source_indices) != args.expected_per_file:
            raise ValueError(
                f"{name}: expected {args.expected_per_file} selected rows, "
                f"found {len(source_indices)}"
            )
        if any(index >= len(source_rows) for index in source_indices):
            raise ValueError(f"{name}: selected row exceeds source file")
        offset = min(
            int(row["materialized_index"])
            for row in selected
            if row["source_file"] == name
        )
        actual = [source_rows[index] for index in source_indices]
        for index, row in enumerate(actual):
            if row != selected_rows[offset + index]:
                raise ValueError(f"{name}: row {index} differs from selected.jsonl")
        subset_dir = args.subset_data_dir or args.output.parent / "subset-data"
        subset_dir.mkdir(parents=True, exist_ok=True)
        path = subset_dir / name
        content = "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in actual
        )
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError(f"refusing to overwrite changed subset file: {path}")
        path.write_text(content, encoding="utf-8")
        files[name] = {"path": name, "rows": len(actual), "sha256": digest(path)}

    files["selected.jsonl"] = {
        "path": "selected.jsonl",
        "rows": len(selected_rows),
        "sha256": digest(selected_path),
    }

    result = dict(base)
    result["manifest_type"] = "speculators.evaluation_subset_manifest"
    result["files"] = files
    result["subset_files"] = subset_names
    result["subset_rows"] = {name: len(expected[name]) for name in subset_names}
    result["source_manifest"] = {
        "base_manifest": str(args.base_manifest),
        "base_manifest_sha256": digest(args.base_manifest),
        "canonical_manifest": str(args.canonical_manifest),
        "canonical_manifest_sha256": digest(args.canonical_manifest),
        "builder": "local/drafter-quant/dquant/build_subset_manifest.py",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "subsets": subset_names}, indent=2))


if __name__ == "__main__":
    main()
