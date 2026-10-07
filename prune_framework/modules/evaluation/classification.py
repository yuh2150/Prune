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
    """Return loss, accuracy and macro P/R/F1 over observed classes.

    Classes with neither targets nor predictions are excluded. Undefined
    per-class precision/recall is zero; detection IoU mAP is not applicable.
    """
    device = torch.device(device)
    criterion = loss_fn or nn.CrossEntropyLoss()
    was_training = model.training
    total_loss = 0.0
    correct = 0
    samples = 0
    target_counts = predicted_counts = true_positive_counts = None
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
                predictions = logits.argmax(dim=1)
                matches = predictions.eq(labels)
                correct += int(matches.sum())
                num_classes = logits.shape[1]
                if target_counts is None:
                    target_counts = torch.zeros(num_classes, dtype=torch.long, device=device)
                    predicted_counts = torch.zeros_like(target_counts)
                    true_positive_counts = torch.zeros_like(target_counts)
                elif num_classes != target_counts.numel():
                    raise ValueError("Classification model must return a consistent number of classes.")
                target_counts += torch.bincount(labels, minlength=num_classes)
                predicted_counts += torch.bincount(predictions, minlength=num_classes)
                true_positive_counts += torch.bincount(labels[matches], minlength=num_classes)
                samples += labels.numel()
    finally:
        model.train(was_training)
    if samples == 0:
        raise ValueError("Classification evaluation dataloader yielded no samples.")
    observed = (target_counts + predicted_counts) > 0
    precision = (true_positive_counts.double() / predicted_counts.clamp_min(1))[observed].mean()
    recall = (true_positive_counts.double() / target_counts.clamp_min(1))[observed].mean()
    # Average per-class F1, rather than the harmonic mean of macro P/R.
    f1 = (2 * true_positive_counts.double() /
          (target_counts + predicted_counts).clamp_min(1))[observed].mean()
    return EvaluationResult(
        metrics={"loss": total_loss / samples, "accuracy": correct / samples,
                 "precision": float(precision), "recall": float(recall), "f1": float(f1)},
        num_samples=samples,
        metadata={"task": "classification", "average": "macro",
                  "class_scope": "targets_or_predictions", "zero_division": 0},
    )
