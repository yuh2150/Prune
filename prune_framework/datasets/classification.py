"""Local-cache-only MNIST-family and EMNIST classification dataloaders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset


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
    classes = {
        "mnist": datasets.MNIST,
        "fashionmnist": datasets.FashionMNIST,
        "fashion_mnist": datasets.FashionMNIST,
        "emnist": datasets.EMNIST,
    }
    if name not in classes:
        raise ValueError("classification dataset.name must be 'mnist', 'fashionmnist', or 'emnist'.")
    factory = classes[name]
    image_padding = int(getattr(config, "image_padding", 0))
    transform_steps = [transforms.Pad(image_padding)] if image_padding else []
    transform_steps.append(transforms.ToTensor())
    transform = transforms.Compose(transform_steps)
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
