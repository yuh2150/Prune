"""Local-cache-only classification dataloaders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset


@dataclass(frozen=True)
class ClassificationDataLoaders:
    train: DataLoader
    validation: DataLoader


def create_classification_dataloaders(config: Any, *, seed: int = 42) -> ClassificationDataLoaders:
    """Build deterministic MNIST/FashionMNIST/EMNIST loaders without downloading data.

    Dataset acquisition belongs to the caller; failing clearly on a missing
    local cache prevents an experiment configuration from causing a surprise
    network download.
    """
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise RuntimeError("Classification datasets require torchvision.") from exc
    name = str(config.name).lower()
    image_padding = int(getattr(config, "image_padding", 0))
    if name == "imagenet1k_subset":
        normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
        validation = _limit(_ManifestImageNetDataset(config.root, transform), config.val_limit, seed)
        train = _limit(_ManifestImageNetDataset(getattr(config, "train_root", None) or config.root, transform), config.train_limit, seed)
        return ClassificationDataLoaders(train=DataLoader(train, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, generator=torch.Generator().manual_seed(seed)), validation=DataLoader(validation, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers))
    transform_steps = [transforms.Pad(image_padding)] if image_padding else []
    transform_steps.append(transforms.ToTensor())
    transform = transforms.Compose(transform_steps)
    if name == "custom_47labels":
        return _create_custom_47labels_dataloaders(config, transform, seed)
    classes = {
        "mnist": datasets.MNIST,
        "fashionmnist": datasets.FashionMNIST,
        "fashion_mnist": datasets.FashionMNIST,
        "emnist": datasets.EMNIST,
    }
    if name not in classes:
        raise ValueError("classification dataset.name must be 'mnist', 'fashionmnist', 'emnist', or 'custom_47labels'.")
    factory = classes[name]
    dataset_kwargs = {}
    if name == "emnist":
        split = str(getattr(config, "emnist_split", "balanced")).lower()
        if split not in datasets.EMNIST.splits:
            choices = ", ".join(datasets.EMNIST.splits)
            raise ValueError(f"dataset.emnist_split must be one of: {choices}.")
        dataset_kwargs["split"] = split
        # torchvision EMNIST Letters uses labels 1..26; CrossEntropy needs 0..25.
        if split == "letters":
            dataset_kwargs["target_transform"] = _zero_based_emnist_letters
    try:
        train = factory(root=config.root, train=True, transform=transform, download=False, **dataset_kwargs)
        validation = factory(root=config.root, train=False, transform=transform, download=False, **dataset_kwargs)
    except (RuntimeError, FileNotFoundError) as exc:
        suffix = f" (split {dataset_kwargs['split']})" if name == "emnist" else ""
        raise FileNotFoundError(
            f"{factory.__name__}{suffix} is not available in local cache '{config.root}'; download it explicitly before running."
        ) from exc
    train = _limit(train, config.train_limit, seed)
    validation = _limit(validation, config.val_limit, seed + 1)
    generator = torch.Generator().manual_seed(seed)
    return ClassificationDataLoaders(
        train=DataLoader(train, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, generator=generator),
        validation=DataLoader(validation, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers),
    )


def _limit(dataset, limit: int | None, seed: int):
    if limit is None or limit >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:limit].tolist()
    return Subset(dataset, indices)


def _zero_based_emnist_letters(label: int) -> int:
    return int(label) - 1


class _ImageTextLabelDataset(Dataset):
    """Grayscale images paired with one integer class id per text file."""

    def __init__(self, root: str, transform):
        self.root = Path(root)
        images_dir = self.root / "images"
        labels_dir = self.root / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise FileNotFoundError(
                f"custom_47labels needs '{images_dir}' and '{labels_dir}' directories."
            )
        extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        self.samples = []
        for image_path in sorted(path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in extensions):
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing label file for '{image_path.name}': '{label_path}'.")
            try:
                text = label_path.read_text(encoding="utf-8").strip()
                label = int(text)
            except ValueError as exc:
                raise ValueError(f"Label file '{label_path}' must contain exactly one integer class id.") from exc
            if not 0 <= label < 47:
                raise ValueError(f"Label in '{label_path}' must be in [0, 46], got {label}.")
            self.samples.append((image_path, label))
        if not self.samples:
            raise FileNotFoundError(f"No supported image files found in '{images_dir}'.")
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from PIL import Image

        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("L")
            return self.transform(image), label


class _ManifestImageNetDataset(Dataset):
    """ImageNet subset saved by ``download_imagenet_val_subset.py``."""

    def __init__(self, root: str, transform):
        import json
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"ImageNet subset manifest not found: '{manifest_path}'.")
        self.samples = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not self.samples or any(not 0 <= int(sample["label"]) < 1000 for sample in self.samples):
            raise ValueError("ImageNet subset manifest must contain ImageNet-1K labels in [0, 999].")
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from PIL import Image
        sample = self.samples[index]
        with Image.open(self.root / sample["file"]) as image:
            return self.transform(image.convert("RGB")), int(sample["label"])


def _create_custom_47labels_dataloaders(config: Any, transform, seed: int) -> ClassificationDataLoaders:
    """Split the flat custom dataset into deterministic, stratified train/validation subsets."""
    dataset = _ImageTextLabelDataset(config.root, transform)
    validation_fraction = float(getattr(config, "validation_fraction", 0.2))
    train_indices, validation_indices = _stratified_split(
        [label for _, label in dataset.samples], validation_fraction, seed
    )
    generator = torch.Generator().manual_seed(seed)
    return ClassificationDataLoaders(
        train=DataLoader(Subset(dataset, train_indices), batch_size=config.batch_size, shuffle=True,
                         num_workers=config.num_workers, generator=generator),
        validation=DataLoader(Subset(dataset, validation_indices), batch_size=config.batch_size,
                              shuffle=False, num_workers=config.num_workers),
    )


def _stratified_split(labels: list[int], validation_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    by_label: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        by_label.setdefault(label, []).append(index)
    generator = torch.Generator().manual_seed(seed)
    train, validation = [], []
    for label in sorted(by_label):
        indices = by_label[label]
        order = torch.randperm(len(indices), generator=generator).tolist()
        shuffled = [indices[index] for index in order]
        # Keep singleton classes in training so calibration always has a target.
        val_count = min(len(shuffled) - 1, max(1, round(len(shuffled) * validation_fraction))) if len(shuffled) > 1 else 0
        validation.extend(shuffled[:val_count])
        train.extend(shuffled[val_count:])
    if not train or not validation:
        raise ValueError("custom_47labels needs at least two images across one or more classes to form train/validation splits.")
    return train, validation
