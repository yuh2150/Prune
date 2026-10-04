"""Minimal model-agnostic classification evaluation."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from prune_framework.contracts.evaluation import EvaluationResult


def evaluate_classification(
    model: nn.Module,
    dataloader: Iterable,
    device: torch.device | str,
    loss_fn: nn.Module | None = None,
) -> EvaluationResult:
    """Return mean cross-entropy loss and top-1 accuracy without mutation."""
    device = torch.device(device)
    criterion = loss_fn or nn.CrossEntropyLoss()
    was_training = model.training
    total_loss = 0.0
    correct = 0
    samples = 0
    try:
        model.eval()
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device=device, dtype=next(model.parameters()).dtype)
                labels = labels.to(device=device, dtype=torch.long)
                logits = model(images)
                if logits.ndim != 2 or logits.shape[0] != labels.shape[0]:
                    raise ValueError("Classification model must return [batch, num_classes] logits.")
                loss = criterion(logits, labels)
                total_loss += float(loss.detach()) * labels.numel()
                correct += int(logits.argmax(dim=1).eq(labels).sum())
                samples += labels.numel()
    finally:
        model.train(was_training)
    if samples == 0:
        raise ValueError("Classification evaluation dataloader yielded no samples.")
    return EvaluationResult(
        metrics={"loss": total_loss / samples, "accuracy": correct / samples},
        num_samples=samples,
        metadata={"task": "classification"},
    )
