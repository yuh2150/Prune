"""Classification callbacks for local-cache MNIST-family experiments."""

from __future__ import annotations

import itertools
import math
from typing import Any, Iterable

import torch
import torch.nn as nn

from prune_framework.datasets import create_classification_dataloaders
from prune_framework.modules.calibration.context import CalibrationContext
from prune_framework.modules.evaluation.classification import evaluate_classification
from prune_framework.modules.model.masks import MaskManager


def _loaders(config):
    return create_classification_dataloaders(config.dataset, seed=config.experiment.seed)


def _loss(model: nn.Module, batch: Any, device: torch.device, temperature: float = 1.0) -> torch.Tensor:
    images, labels = batch
    images = images.to(device=device, dtype=next(model.parameters()).dtype)
    labels = labels.to(device=device, dtype=torch.long)
    return nn.CrossEntropyLoss()(model(images) / temperature, labels)


def calibrate(model, config, device, dataloader: Iterable | None = None,
              temperature: float = 1.0, microbatch_size: int | None = None,
              parameter_scope: str | None = None, **_unused) -> CalibrationContext:
    """Provide fixed classification batches and scalar CrossEntropy calibration loss."""
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Calibration temperature must be finite and positive.")
    if microbatch_size is not None and (not isinstance(microbatch_size, int) or microbatch_size < 1):
        raise ValueError("Calibration microbatch_size must be a positive integer.")
    if dataloader is None:
        dataloader = _loaders(config).train
    batches = list(itertools.islice(dataloader, config.pruning.calibration_batches))
    if not batches:
        raise ValueError("Classification calibration requires at least one training batch.")
    if microbatch_size is not None:
        batches = [(images[start:start + microbatch_size], labels[start:start + microbatch_size])
                   for images, labels in batches for start in range(0, len(labels), microbatch_size)]
    device = torch.device(device)
    return CalibrationContext(
        batches=batches,
        loss_fn=lambda candidate, batch: _loss(candidate, batch, device, temperature),
        seed=config.pruning.calibration_seed,
        sample_count=sum(labels.numel() for _, labels in batches),
        device=str(device),
        parameter_scope=parameter_scope or getattr(config.pruning, "parameter_scope", "weights"),
        batch_weights=[float(labels.numel()) for _, labels in batches],
    ).validate()


def evaluate(model, config, device, dataloader: Iterable | None = None, **_unused):
    """Evaluate classification accuracy/loss using validation data or an injected loader."""
    if dataloader is None:
        dataloader = _loaders(config).validation
    return evaluate_classification(model, dataloader, device)


def recover(
    model,
    config,
    device,
    stage="recovery",
    *,
    dataloader: Iterable | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
    regularization=None,
    **_unused,
):
    """Fine-tune a pruned classifier with CrossEntropyLoss and persistent masks."""
    if stage not in {"recovery", "regularization"}:
        raise ValueError("Classification recovery supports only recovery or regularization stages.")
    if config.recovery.epochs < 1:
        raise ValueError("recovery.epochs must be at least 1 for classification recovery.")
    if dataloader is None:
        dataloader = _loaders(config).train
    device = torch.device(device)
    model.to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Classification recovery model exposes no trainable parameters.")
    options = dict(config.recovery.optimizer)
    name = str(options.pop("name", "sgd")).lower()
    if optimizer is None:
        if name == "sgd":
            optimizer = torch.optim.SGD(parameters, lr=float(options.pop("lr", 0.01)), momentum=float(options.pop("momentum", 0.0)), weight_decay=float(options.pop("weight_decay", 0.0)))
        elif name == "adam":
            optimizer = torch.optim.Adam(parameters, lr=float(options.pop("lr", 0.001)), weight_decay=float(options.pop("weight_decay", 0.0)))
        else:
            raise ValueError("classification recovery.optimizer.name must be 'sgd' or 'adam'.")
    if options:
        raise ValueError(f"Unsupported classification recovery optimizer options: {sorted(options)}")
    optimizer_parameters = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if any(id(parameter) not in optimizer_parameters for parameter in parameters):
        raise ValueError("Classification recovery optimizer must contain every pruned-model parameter.")
    MaskManager.attach_optimizer(model, optimizer)
    was_training, losses, steps = model.training, [], 0
    try:
        model.train()
        for _ in range(config.recovery.epochs):
            for batch in dataloader:
                optimizer.zero_grad(set_to_none=True)
                loss = _loss(model, batch, device)
                if regularization is not None:
                    loss, _ = regularization.augment_loss(loss)
                loss.backward()
                optimizer.step()
                if regularization is not None:
                    regularization.advance()
                MaskManager.enforce(model)
                losses.append(float(loss.detach()))
                steps += 1
                if max_batches is not None and steps >= max_batches:
                    break
            if max_batches is not None and steps >= max_batches:
                break
    finally:
        model.train(was_training)
    return {
        "model": model,
        "metrics": {
            "recovery_loss": sum(losses) / len(losses) if losses else None,
            "recovery_steps": steps,
            "recovery_epochs": config.recovery.epochs,
        },
        "metadata": {"task": "classification", "optimizer": type(optimizer).__name__},
    }
