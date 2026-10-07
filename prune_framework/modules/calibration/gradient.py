"""Deterministic, side-effect-free gradient collection for saliency criteria."""

from __future__ import annotations

import random
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Sequence

import torch
import torch.nn as nn

from prune_framework.contracts.targets import PrunableTarget
from prune_framework.modules.model.masks import MaskManager


LossFunction = Callable[[nn.Module, Any], torch.Tensor]


@dataclass
class CalibrationResult:
    """Detached, target-keyed gradients captured during calibration."""

    gradients: Dict[str, torch.Tensor] = field(default_factory=dict)
    losses: List[float] = field(default_factory=list)
    batches: int = 0
    accumulated: bool = True
    aggregation: str = "signed_mean"

    def context_for(self, target: PrunableTarget) -> Dict[str, torch.Tensor]:
        try:
            return {"grad": self.gradients[target.name]}
        except KeyError as exc:
            raise KeyError(f"Calibration did not collect a gradient for target '{target.name}'.") from exc

    def describe(self) -> Dict[str, Any]:
        return {
            "batches": self.batches,
            "accumulated": self.accumulated,
            "aggregation": self.aggregation,
            "mean_loss": sum(self.losses) / len(self.losses) if self.losses else None,
            "targets": {name: list(gradient.shape) for name, gradient in self.gradients.items()},
        }


@dataclass
class HigherOrderCalibrationResult(CalibrationResult):
    """Detached Hessian-gradient products collected for higher-order criteria.

    ``gradients`` contains ``H @ g`` for each requested target, where ``g`` is
    the gradient of the mean calibration loss with respect to the selected
    parameter scope (Conv2d/Linear weights by default).  The first-order gradients are retained only as detached
    diagnostic data; neither mapping holds an autograd graph.
    """

    first_order_gradients: Dict[str, torch.Tensor] = field(default_factory=dict)
    parameter_scope: str = "weights"
    parameter_names: List[str] = field(default_factory=list)
    target_statistics: Dict[str, Any] = field(default_factory=dict)
    batch_weights: List[float] = field(default_factory=list)

    def context_for(self, target: PrunableTarget) -> Dict[str, torch.Tensor]:
        return {"hvp": self.gradients[target.name], "first_order_grad": self.first_order_gradients[target.name]}

    def describe(self) -> Dict[str, Any]:
        description = super().describe()
        description["higher_order"] = True
        description["algorithm"] = "two_pass_hvp"
        description["hvp_objective"] = "grad(dot(g, stop_grad(g)))"
        description["batch_weights"] = self.batch_weights
        description["parameter_scope"] = self.parameter_scope
        description["parameter_names"] = self.parameter_names
        description["target_statistics"] = self.target_statistics
        description["mean_loss"] = sum(loss * weight for loss, weight in zip(self.losses, self.batch_weights)) if self.losses else None
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

    def __init__(self, *, seed: int = 42, accumulate: bool = True, aggregation: str = "signed_mean"):
        self.seed = int(seed)
        self.accumulate = bool(accumulate)
        if aggregation not in {"signed_mean", "abs_mean"}:
            raise ValueError("Gradient aggregation must be 'signed_mean' or 'abs_mean'.")
        self.aggregation = aggregation

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
        result = CalibrationResult(
            batches=len(batch_list), accumulated=self.accumulate, aggregation=self.aggregation
        )

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
            # ``signed_mean`` preserves the established SNIP semantics:
            # abs(W * mean(gradient)). ``abs_mean`` is an explicit variant
            # using mean(abs(gradient)); it must collect each batch before
            # signed gradients can cancel.
            scale = float(len(batch_list))
            if self.aggregation == "abs_mean":
                for batch in batch_list:
                    model.zero_grad(set_to_none=True)
                    loss = self._loss(loss_fn, model, batch, result)
                    (loss / scale).backward()
                    self._add_target_gradients(collected, targets, absolute=True)
            else:
                for batch in batch_list:
                    if not self.accumulate:
                        model.zero_grad(set_to_none=True)
                    loss = self._loss(loss_fn, model, batch, result)
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
    def _loss(loss_fn: LossFunction, model: nn.Module, batch: Any, result: CalibrationResult) -> torch.Tensor:
        loss = loss_fn(model, batch)
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
            raise TypeError("Calibration loss_fn must return a scalar or single-element torch.Tensor.")
        loss = loss.reshape(())
        result.losses.append(float(loss.detach().cpu()))
        return loss

    @staticmethod
    def _add_target_gradients(
        collected: Dict[str, torch.Tensor], targets: Sequence[PrunableTarget], *, absolute: bool = False
    ) -> None:
        for target in targets:
            gradient = MaskManager.original_weight(target.module).grad
            if gradient is None:
                raise RuntimeError(f"Calibration produced no gradient for target '{target.name}'.")
            value = gradient.detach().abs().clone() if absolute else gradient.detach().clone()
            collected[target.name] = value if target.name not in collected else collected[target.name] + value


class HigherOrderCalibrationRunner:
    """Compute detached Hessian-gradient products for GraSP safely.

    GraSP preserves gradient flow by computing ``H @ g`` from the mean task
    loss.  The implementation builds the higher-order graph only inside this
    runner and releases it before returning.  Like first-order calibration, it
    restores parameter gradients, ``requires_grad`` flags, buffers, modes and
    RNG state even if the loss function fails.

    Two passes accumulate a detached mean gradient, then sum per-batch
    Hessian-vector products. Only one batch's autograd graph is retained.
    """

    def __init__(self, *, seed: int = 42, parameter_scope: str = "weights"):
        self.seed = int(seed)
        if parameter_scope not in {"all", "weights"}:
            raise ValueError("Higher-order parameter_scope must be 'all' or 'weights'.")
        self.parameter_scope = parameter_scope

    def run(
        self,
        model: nn.Module,
        targets: Sequence[PrunableTarget],
        batches: Iterable[Any],
        loss_fn: LossFunction,
        batch_weights: Sequence[float] | None = None,
    ) -> HigherOrderCalibrationResult:
        batch_list = list(batches)
        if not batch_list:
            raise ValueError("Higher-order calibration requires at least one batch.")
        if not targets:
            raise ValueError("Higher-order calibration requires at least one target.")
        weights = list(batch_weights) if batch_weights is not None else [1.0] * len(batch_list)
        if len(weights) != len(batch_list) or any(not math.isfinite(w) or w <= 0 for w in weights):
            raise ValueError("Higher-order batch_weights must be finite positive weights for every batch.")
        normalized_weights = [weight / sum(weights) for weight in weights]

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
        result = HigherOrderCalibrationResult(batches=len(batch_list), accumulated=True, parameter_scope=self.parameter_scope)

        # The GraSP reference differentiates Conv2d/Linear weights, including
        # protected classifier weights, but excludes biases and BN parameters.
        weight_ids = {id(MaskManager.original_weight(module)) for module in model.modules()
                      if isinstance(module, (nn.Conv2d, nn.Linear))}
        named_parameters = [(name, parameter) for name, parameter in model.named_parameters()
                            if self.parameter_scope == "all" or id(parameter) in weight_ids]
        parameter_names = [name for name, _ in named_parameters]
        parameters = [parameter for _, parameter in named_parameters]
        result.parameter_names = parameter_names
        result.batch_weights = normalized_weights

        try:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            model.train()
            for parameter in parameters:
                parameter.requires_grad_(True)
            model.zero_grad(set_to_none=True)

            first_order = [torch.zeros_like(parameter) for parameter in parameters]
            replay_states = []
            for batch, batch_weight in zip(batch_list, normalized_weights):
                replay_states.append((random.getstate(), torch.random.get_rng_state(),
                                      torch.cuda.get_rng_state_all() if cuda_states is not None else None,
                                      {name: buffer.detach().clone() for name, buffer in model.named_buffers()}))
                loss = loss_fn(model, batch)
                if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
                    raise TypeError("Higher-order calibration loss_fn must return a scalar or single-element torch.Tensor.")
                loss = loss.reshape(())
                result.losses.append(float(loss.detach().cpu()))
                loss = loss * batch_weight
                gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
                for accumulated, gradient in zip(first_order, gradients):
                    if gradient is not None:
                        accumulated.add_(gradient.detach())
            raw_hvp = [torch.zeros_like(parameter) for parameter in parameters]
            for batch, batch_weight, (py_rng, cpu_rng, gpu_rng, buffers) in zip(batch_list, normalized_weights, replay_states):
                # Replay the same stochastic forward and buffer state so both
                # passes differentiate the same calibration objective.
                random.setstate(py_rng)
                torch.random.set_rng_state(cpu_rng)
                if gpu_rng is not None:
                    torch.cuda.set_rng_state_all(gpu_rng)
                with torch.no_grad():
                    for name, buffer in model.named_buffers():
                        buffer.copy_(buffers[name])
                loss = loss_fn(model, batch)
                if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
                    raise TypeError("Higher-order calibration loss_fn must return a scalar or single-element torch.Tensor.")
                gradients = torch.autograd.grad(loss.reshape(()) * batch_weight, parameters, create_graph=True, allow_unused=True)
                # A differentiable zero also handles unused parameters and
                # constant first derivatives (whose Hessian is exactly zero).
                objective = sum(((gradient if gradient is not None else parameter * 0) * vector).sum()
                                + (parameter * 0).sum()
                                for parameter, gradient, vector in zip(parameters, gradients, first_order))
                products = torch.autograd.grad(objective, parameters, allow_unused=True)
                for accumulated, product in zip(raw_hvp, products):
                    if product is not None:
                        accumulated.add_(product.detach())

            parameter_to_hvp = {
                name: (hvp if hvp is not None else torch.zeros_like(parameter)).detach().clone()
                for name, parameter, hvp in zip(parameter_names, parameters, raw_hvp)
            }
            parameter_to_first_order = {
                name: gradient.detach().clone()
                for name, gradient in zip(parameter_names, first_order)
            }
            for target in targets:
                target_weight = MaskManager.original_weight(target.module)
                target_name = next(
                    (name for name, parameter in model.named_parameters() if parameter is target_weight), None
                )
                if target_name is None:
                    raise RuntimeError(f"Calibration target '{target.name}' is not a model parameter.")
                result.gradients[target.name] = parameter_to_hvp[target_name]
                result.first_order_gradients[target.name] = parameter_to_first_order[target_name]
                hvp = result.gradients[target.name]
                score = -(target.module.weight.detach() * hvp)
                result.target_statistics[target.name] = {
                    "numel": score.numel(), "raw_score_min": float(score.min()),
                    "raw_score_max": float(score.max()), "raw_score_mean": float(score.mean()),
                    "raw_score_mean_abs": float(score.abs().mean()),
                    "raw_score_std": float(score.std(unbiased=False)),
                    "first_order_grad_norm": float(result.first_order_gradients[target.name].norm()),
                    "hvp_norm": float(hvp.norm()),
                }
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
