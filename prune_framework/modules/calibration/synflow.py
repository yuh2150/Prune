"""Data-free, state-isolated SynFlow calibration."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Sequence

import torch
import torch.nn as nn

from prune_framework.contracts.targets import PrunableTarget


OutputReducer = Callable[[Any], torch.Tensor]
InputFactory = Callable[[], Any]


@dataclass
class SynFlowResult:
    """Detached SynFlow gradients keyed by their typed pruning target."""

    gradients: Dict[str, torch.Tensor] = field(default_factory=dict)

    def context_for(self, target: PrunableTarget) -> Dict[str, torch.Tensor]:
        try:
            return {"grad": self.gradients[target.name]}
        except KeyError as exc:
            raise KeyError(f"SynFlow did not collect a gradient for target '{target.name}'.") from exc

    def describe(self) -> Dict[str, Any]:
        return {"targets": {name: list(gradient.shape) for name, gradient in self.gradients.items()}}


class SynFlowCalibrationRunner:
    """Calculate SynFlow gradients without retaining any model-side change.

    Canonical SynFlow linearizes the network temporarily by replacing every
    floating-point parameter with its absolute value, evaluates an all-ones
    input, and differentiates a scalar output sum.  A full parameter snapshot,
    rather than sign multiplication, restores zeros and unusual parameter
    values exactly even when the run fails part way through.
    """

    def run(
        self,
        model: nn.Module,
        targets: Sequence[PrunableTarget],
        input_factory: InputFactory,
        output_reducer: OutputReducer,
        snapshot_extra_state: Callable[[], Any] | None = None,
        restore_extra_state: Callable[[Any], None] | None = None,
    ) -> SynFlowResult:
        if not targets:
            raise ValueError("SynFlow requires at least one Conv2d or Linear weight target.")

        prior_training = model.training
        prior_module_training = {id(module): module.training for module in model.modules()}
        prior_parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        prior_gradients = {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
        }
        prior_requires_grad = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
        prior_buffers = {name: buffer.detach().clone() for name, buffer in model.named_buffers()}
        python_state = random.getstate()
        torch_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        extra_state = snapshot_extra_state() if snapshot_extra_state is not None else None

        try:
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(True)
                if parameter.is_floating_point() or parameter.is_complex():
                    parameter.data.abs_()
            model.zero_grad(set_to_none=True)

            synthetic_input = input_factory()
            output = model(synthetic_input)
            objective = output_reducer(output)
            if not isinstance(objective, torch.Tensor) or objective.numel() != 1:
                raise TypeError("SynFlow output reducer must return a scalar torch.Tensor.")
            objective.reshape(()).backward()

            gradients: Dict[str, torch.Tensor] = {}
            for target in targets:
                gradient = target.module.weight.grad
                if gradient is None:
                    raise RuntimeError(f"SynFlow produced no gradient for target '{target.name}'.")
                gradients[target.name] = gradient.detach().clone()
            return SynFlowResult(gradients=gradients)
        finally:
            # No calibration state is allowed to leak into recovery or training.
            model.zero_grad(set_to_none=True)
            parameter_map = dict(model.named_parameters())
            for name, value in prior_parameters.items():
                parameter = parameter_map[name]
                with torch.no_grad():
                    parameter.copy_(value)
                parameter.grad = prior_gradients[name]
                parameter.requires_grad_(prior_requires_grad[name])
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
            if restore_extra_state is not None:
                restore_extra_state(extra_state)
