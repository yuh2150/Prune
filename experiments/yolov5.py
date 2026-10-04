"""YOLO callbacks used by the unified pruning pipeline."""
from __future__ import annotations

from typing import Any, Callable, Iterable

import torch

from prune_framework.contracts.evaluation import normalize_yolo_evaluation
from prune_framework.modules.calibration.context import CalibrationContext
from prune_framework.modules.model.masks import MaskManager


def evaluate(model, data, dataloader=None, **kwargs):
    from test import test
    accepted = {'batch_size', 'imgsz', 'conf_thres', 'iou_thres', 'single_cls', 'half_precision'}
    options = {k: v for k, v in kwargs.items() if k in accepted}
    value = test(data=data, model=model, dataloader=dataloader, plots=False, **options)
    samples = len(dataloader.dataset) if dataloader is not None and hasattr(dataloader, 'dataset') else 0
    return normalize_yolo_evaluation(value, samples)


def calibrate_taylor(model, config, device, data=None, dataloader=None, **kwargs):
    """Prepare a fixed local-data subset; never silently replace it with noise."""
    import itertools
    from utils.general import check_dataset
    from utils.datasets import create_dataloader
    from utils.loss import ComputeLoss
    if dataloader is None:
        if data is None:
            raise ValueError('YOLO calibration requires data YAML or an explicit dataloader')
        dataset = check_dataset(data, autodownload=False)
        stride = int(model.stride.max())
        dataloader = create_dataloader(dataset['val'], config.model.input_shape[-1],
                                      config.pruning.calibration_batch_size, stride, workers=0)[0]
    batches = [(images.to(device=device, dtype=next(model.parameters()).dtype) / 255., targets.to(device))
               for images, targets, *_ in itertools.islice(dataloader, config.pruning.calibration_batches)]
    if not hasattr(model, "hyp"):
        raise ValueError("YOLO calibration requires model.hyp with training loss hyperparameters")
    compute_loss = ComputeLoss(model)
    def loss_fn(candidate, batch):
        return compute_loss(candidate(batch[0]), batch[1])[0]
    return CalibrationContext(batches, loss_fn, config.pruning.calibration_seed,
                              sum(len(images) for images, _ in batches), str(device)).validate()


def recover(
    model,
    config,
    device,
    stage="recovery",
    *,
    data=None,
    dataloader=None,
    optimizer=None,
    scheduler=None,
    loss_fn: Callable[[torch.nn.Module, Any], torch.Tensor] | None = None,
    batch_size: int = 16,
    workers: int = 0,
    imgsz: int | None = None,
    max_batches: int | None = None,
    lr: float = 1e-3,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    optimizer_name: str = "sgd",
    scheduler_name: str = "none",
    scheduler_step_size: int = 1,
    scheduler_gamma: float = 0.1,
    regularization=None,
    **_unused,
):
    """Fine-tune the exact model instance received after pruning.

    ``dataloader`` and ``loss_fn`` are injectable for deterministic tests. A
    normal YOLO run supplies ``data`` and uses the repository's ``ComputeLoss``.
    The optimizer is constructed after pruning, so it cannot retain dense-model
    parameters. Persistent masks are attached and explicitly enforced.
    """
    if stage not in {"recovery", "regularization"}:
        raise ValueError(f"YOLO recovery does not support stage {stage!r}.")
    if not isinstance(model, torch.nn.Module):
        raise TypeError("YOLO recovery requires an nn.Module.")
    if config.recovery.epochs < 1:
        raise ValueError("recovery.epochs must be at least 1 for recovery.")
    device = torch.device(device)
    model.to(device)
    was_training = model.training

    if dataloader is None:
        if data is None:
            raise ValueError("YOLO recovery requires recovery.kwargs.data or an explicit dataloader.")
        from utils.datasets import create_dataloader
        from utils.general import check_dataset

        dataset = check_dataset(data, autodownload=False)
        if not dataset.get("train"):
            raise ValueError("YOLO recovery dataset has no train split.")
        stride = int(torch.as_tensor(getattr(model, "stride", [32])).max())
        dataloader = create_dataloader(
            dataset["train"], int(imgsz or config.model.input_shape[-1]), int(batch_size), stride, workers=int(workers)
        )[0]

    if loss_fn is None:
        if not hasattr(model, "hyp"):
            raise ValueError("YOLO recovery requires model.hyp with training loss hyperparameters.")
        from utils.loss import ComputeLoss

        compute_loss = ComputeLoss(model)

        def loss_fn(candidate, batch):
            images, targets = _prepare_yolo_batch(candidate, batch, device)
            return compute_loss(candidate(images), targets)[0]

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("YOLO recovery model exposes no trainable parameters.")
    if optimizer is None:
        if optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(parameters, lr=float(lr), momentum=float(momentum), weight_decay=float(weight_decay))
        elif optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(parameters, lr=float(lr), weight_decay=float(weight_decay))
        else:
            raise ValueError("recovery.optimizer_name must be 'sgd' or 'adam'.")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("YOLO recovery optimizer must be a torch.optim.Optimizer.")
    optimizer_parameters = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if any(id(parameter) not in optimizer_parameters for parameter in parameters):
        raise ValueError("Recovery optimizer must be created from every trainable parameter of the pruned model.")
    MaskManager.attach_optimizer(model, optimizer)

    if scheduler is None:
        if scheduler_name == "none":
            scheduler = None
        elif scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(scheduler_step_size), gamma=float(scheduler_gamma))
        elif scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.recovery.epochs)
        else:
            raise ValueError("recovery.scheduler_name must be 'none', 'step', or 'cosine'.")
    if scheduler is not None and not hasattr(scheduler, "step"):
        raise TypeError("YOLO recovery scheduler must provide step().")

    losses: list[float] = []
    steps = 0
    try:
        model.train()
        for _epoch in range(config.recovery.epochs):
            for batch in dataloader:
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model, batch)
                if regularization is not None:
                    loss, _report = regularization.augment_loss(loss)
                if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
                    raise TypeError("YOLO recovery loss_fn must return a scalar Tensor.")
                loss.reshape(()).backward()
                optimizer.step()
                MaskManager.enforce(model)
                if regularization is not None:
                    regularization.advance()
                losses.append(float(loss.detach().cpu()))
                steps += 1
                if max_batches is not None and steps >= int(max_batches):
                    break
            if scheduler is not None:
                scheduler.step()
            if max_batches is not None and steps >= int(max_batches):
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
        "metadata": {
            "stage": stage,
            "optimizer": type(optimizer).__name__,
            "scheduler": type(scheduler).__name__ if scheduler is not None else None,
            "regularization": type(regularization.term).__name__ if regularization is not None else None,
        },
    }


def _prepare_yolo_batch(model, batch: Iterable[Any], device: torch.device):
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise TypeError("YOLO recovery dataloader must yield images and targets.")
    images, targets = batch[:2]
    if not isinstance(images, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError("YOLO recovery images and targets must be tensors.")
    dtype = next(model.parameters()).dtype
    images = images.to(device=device, dtype=dtype)
    if images.numel() and images.detach().max() > 1:
        images = images / 255.0
    return images, targets.to(device)
