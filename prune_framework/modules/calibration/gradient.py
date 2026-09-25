"""Deterministic, side-effect-free gradient collection for saliency criteria."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Sequence

import torch
import torch.nn as nn

from prune_framework.contracts.targets import PrunableTarget


LossFunction = Callable[[nn.Module, Any], torch.Tensor]


@dataclass
class CalibrationResult:
    """Detached, target-keyed gradients captured during calibration."""

    gradients: Dict[str, torch.Tensor] = field(default_factory=dict)
    losses: List[float] = field(default_factory=list)
    batches: int = 0
    accumulated: bool = True

    def context_for(self, target: PrunableTarget) -> Dict[str, torch.Tensor]:
        try:
            return {"grad": self.gradients[target.name]}
        except KeyError as exc:
            raise KeyError(f"Calibration did not collect a gradient for target '{target.name}'.") from exc

    def describe(self) -> Dict[str, Any]:
        return {
            "batches": self.batches,
            "accumulated": self.accumulated,
            "mean_loss": sum(self.losses) / len(self.losses) if self.losses else None,
            "targets": {name: list(gradient.shape) for name, gradient in self.gradients.items()},
        }


@dataclass
class HigherOrderCalibrationResult(CalibrationResult):
    """Detached Hessian-gradient products collected for higher-order criteria.

    ``gradients`` contains ``H @ g`` for each requested target, where ``g`` is
    the gradient of the mean calibration loss with respect to all model
    parameters.  The first-order gradients are retained only as detached
    diagnostic data; neither mapping holds an autograd graph.
    """

    first_order_gradients: Dict[str, torch.Tensor] = field(default_factory=dict)

    def describe(self) -> Dict[str, Any]:
        description = super().describe()
        description["higher_order"] = True
        description["first_order_targets"] = {
            name: list(gradient.shape) for name, gradient in self.first_order_gradients.items()
        }
        return description


class GradientCalibrationRunner:
    """Run forward/loss/backward calibration without changing model state.

    The runner always clears gradients before calibration.  It snapshots and
    restores pre-existing parameter gradients, module training state, buffers,
    and RNG state afterwards.  Saliency algorithms consume the returned
    detached gradients rather than relying on residual ``parameter.grad``.
    """

    def __init__(self, *, seed: int = 42, accumulate: bool = True):
        self.seed = int(seed)
        self.accumulate = bool(accumulate)

    def run(
        self,
        model: nn.Module,
        targets: Sequence[PrunableTarget],
        batches: Iterable[Any],
        loss_fn: LossFunction,
    ) -> CalibrationResult:
        batch_list = list(batches)
        if not batch_list:
            raise ValueError("Gradient calibration requires at least one batch.")
        if not targets:
            raise ValueError("Gradient calibration requires at least one target.")

        prior_training = model.training
        prior_module_training = {id(module): module.training for module in model.modules()}
        prior_gradients = {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
        }
        prior_requires_grad = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
        prior_buffers = {name: buffer.detach().clone() for name, buffer in model.named_buffers()}
        python_state = random.getstate()
        torch_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        result = CalibrationResult(batches=len(batch_list), accumulated=self.accumulate)

        try:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            model.train()
            # Exported YOLO checkpoints may carry frozen parameters. Calibration
            # needs a backward graph but must leave the original freeze policy
            # untouched afterwards.
            for parameter in model.parameters():
                parameter.requires_grad_(True)
            model.zero_grad(set_to_none=True)
            collected: Dict[str, torch.Tensor] = {}
            # Both modes return the mean gradient across batches. ``accumulate``
            # only controls whether PyTorch retains gradients between backward
            # calls or the runner sums detached per-batch gradients.
            scale = float(len(batch_list))
            for batch in batch_list:
                if not self.accumulate:
                    model.zero_grad(set_to_none=True)
                loss = loss_fn(model, batch)
                if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
                    raise TypeError("Calibration loss_fn must return a scalar or single-element torch.Tensor.")
                loss = loss.reshape(())
                result.losses.append(float(loss.detach().cpu()))
                (loss / scale).backward()
                if not self.accumulate:
                    self._add_target_gradients(collected, targets)

            if self.accumulate:
                self._add_target_gradients(collected, targets)
            result.gradients = collected
            return result
        finally:
            # Calibration is deliberately isolated from normal training. Restore
            # all mutable state even when forward/backward raises.
            model.zero_grad(set_to_none=True)
            parameter_map = dict(model.named_parameters())
            for name, gradient in prior_gradients.items():
                parameter_map[name].grad = None if gradient is None else gradient
                parameter_map[name].requires_grad_(prior_requires_grad[name])
            buffer_map = dict(model.named_buffers())
            for name, value in prior_buffers.items():
                if name in buffer_map:
                    buffer_map[name].copy_(value)
            # Preserve intentionally mixed train/eval submodule states as well
            # as the root model mode.
            model.train(prior_training)
            for module in model.modules():
                module.training = prior_module_training[id(module)]
            random.setstate(python_state)
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    @staticmethod
    def _add_target_gradients(
        collected: Dict[str, torch.Tensor], targets: Sequence[PrunableTarget]
    ) -> None:
        for target in targets:
            gradient = target.module.weight.grad
            if gradient is None:
                raise RuntimeError(f"Calibration produced no gradient for target '{target.name}'.")
            value = gradient.detach().clone()
            collected[target.name] = value if target.name not in collected else collected[target.name] + value


class HigherOrderCalibrationRunner:
    """Compute detached Hessian-gradient products for GraSP safely.

    GraSP preserves gradient flow by computing ``H @ g`` from the mean task
    loss.  The implementation builds the higher-order graph only inside this
    runner and releases it before returning.  Like first-order calibration, it
    restores parameter gradients, ``requires_grad`` flags, buffers, modes and
    RNG state even if the loss function fails.

    Memory scales with the number and size of retained calibration forward
    graphs.  Callers should keep the supplied calibration batch count and
    batch size intentionally small.
    """

    def __init__(self, *, seed: int = 42):
        self.seed = int(seed)

    def run(
        self,
        model: nn.Module,
        targets: Sequence[PrunableTarget],
        batches: Iterable[Any],
        loss_fn: LossFunction,
    ) -> HigherOrderCalibrationResult:
        batch_list = list(batches)
        if not batch_list:
            raise ValueError("Higher-order calibration requires at least one batch.")
        if not targets:
            raise ValueError("Higher-order calibration requires at least one target.")

        prior_training = model.training
        prior_module_training = {id(module): module.training for module in model.modules()}
        prior_gradients = {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
        }
        prior_requires_grad = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
        prior_buffers = {name: buffer.detach().clone() for name, buffer in model.named_buffers()}
        python_state = random.getstate()
        torch_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        result = HigherOrderCalibrationResult(batches=len(batch_list), accumulated=True)

        # Retain every parameter in the vector used for H @ g.  This preserves
        # cross-parameter Hessian terms from protected heads into safe targets.
        parameters = list(model.parameters())
        parameter_names = [name for name, _ in model.named_parameters()]

        try:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            model.train()
            for parameter in parameters:
                parameter.requires_grad_(True)
            model.zero_grad(set_to_none=True)

            mean_loss: torch.Tensor | None = None
            scale = float(len(batch_list))
            for batch in batch_list:
                loss = loss_fn(model, batch)
                if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
                    raise TypeError("Higher-order calibration loss_fn must return a scalar or single-element torch.Tensor.")
                loss = loss.reshape(()) / scale
                result.losses.append(float(loss.detach().cpu() * scale))
                mean_loss = loss if mean_loss is None else mean_loss + loss

            assert mean_loss is not None  # guarded by the non-empty batch check
            raw_first_order = torch.autograd.grad(
                mean_loss, parameters, create_graph=True, allow_unused=True
            )
            # ``parameter * 0`` gives unused parameters a differentiable zero,
            # allowing a mixed model to produce an HVP without special cases.
            first_order = tuple(
                gradient if gradient is not None else parameter * 0.0
                for parameter, gradient in zip(parameters, raw_first_order)
            )
            squared_gradient_norm = sum((gradient.square().sum() * 0.5) for gradient in first_order)
            raw_hvp = torch.autograd.grad(squared_gradient_norm, parameters, allow_unused=True)

            parameter_to_hvp = {
                name: (hvp if hvp is not None else torch.zeros_like(parameter)).detach().clone()
                for name, parameter, hvp in zip(parameter_names, parameters, raw_hvp)
            }
            parameter_to_first_order = {
                name: gradient.detach().clone()
                for name, gradient in zip(parameter_names, first_order)
            }
            for target in targets:
                target_weight = target.module.weight
                target_name = next(
                    (name for name, parameter in model.named_parameters() if parameter is target_weight), None
                )
                if target_name is None:
                    raise RuntimeError(f"Calibration target '{target.name}' is not a model parameter.")
                result.gradients[target.name] = parameter_to_hvp[target_name]
                result.first_order_gradients[target.name] = parameter_to_first_order[target_name]
            return result
        finally:
            # Drop model-visible gradients and all higher-order references before
            # restoring normal training state. The returned tensors are clones.
            model.zero_grad(set_to_none=True)
            parameter_map = dict(model.named_parameters())
            for name, gradient in prior_gradients.items():
                parameter_map[name].grad = None if gradient is None else gradient
                parameter_map[name].requires_grad_(prior_requires_grad[name])
            buffer_map = dict(model.named_buffers())
            for name, value in prior_buffers.items():
                if name in buffer_map:
                    buffer_map[name].copy_(value)
            model.train(prior_training)
            for module in model.modules():
                module.training = prior_module_training[id(module)]
            random.setstate(python_state)
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
