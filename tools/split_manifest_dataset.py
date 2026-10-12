"""Split an existing manifest dataset into train and validation manifests.

The generated manifests reference the original image files with relative paths,
so the split does not duplicate the images on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path


def _write_manifest(samples: list[dict], source_root: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    rewritten = []
    for sample in samples:
        source_file = source_root / str(sample["file"])
        if not source_file.is_file():
            raise FileNotFoundError(f"Manifest file does not exist: {source_file}")
        rewritten.append({
            "file": Path(os.path.relpath(source_file, output_root)).as_posix(),
            "label": int(sample["label"]),
        })
    (output_root / "manifest.json").write_text(json.dumps(rewritten, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a manifest dataset without copying images.")
    parser.add_argument("--source", type=Path, required=True, help="Existing dataset directory containing manifest.json")
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--val-output", type=Path, required=True)
    parser.add_argument("--val-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest_path = args.source / "manifest.json"
    samples = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not 0 < args.val_count < len(samples):
        raise ValueError(f"--val-count must be in [1, {len(samples) - 1}].")
    if args.train_output.resolve() == args.val_output.resolve():
        raise ValueError("--train-output and --val-output must be different directories.")
    if any(path.exists() and any(path.iterdir()) for path in (args.train_output, args.val_output)):
        raise FileExistsError("Output directories must be absent or empty to prevent overwriting a split.")

    indices = list(range(len(samples)))
    random.Random(args.seed).shuffle(indices)
    val_indices = set(indices[:args.val_count])
    train = [sample for index, sample in enumerate(samples) if index not in val_indices]
    validation = [sample for index, sample in enumerate(samples) if index in val_indices]
    _write_manifest(train, args.source, args.train_output)
    _write_manifest(validation, args.source, args.val_output)
    print(f"Wrote {len(train)} train and {len(validation)} validation samples.")


if __name__ == "__main__":
    main()
