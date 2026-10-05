"""Download a small, deterministic ImageNet-1K validation subset from Hugging Face."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("--count must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("ILSVRC/imagenet-1k", split=args.split, streaming=True)
    examples = dataset.shuffle(seed=args.seed, buffer_size=10_000).take(args.count)
    manifest = []
    for index, example in enumerate(examples):
        label = int(example["label"])
        class_dir = args.output / f"{label:04d}"
        class_dir.mkdir(exist_ok=True)
        filename = f"{index:05d}.jpg"
        image = example["image"].convert("RGB")
        image.save(class_dir / filename, format="JPEG", quality=95)
        manifest.append({"file": f"{label:04d}/{filename}", "label": label})
    if len(manifest) != args.count:
        raise RuntimeError(f"Expected {args.count} examples, received {len(manifest)}")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(manifest)} ImageNet {args.split} images to {args.output}")


if __name__ == "__main__":
    main()
