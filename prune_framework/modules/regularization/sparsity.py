"""Architecture-neutral train-time channel-sparsity regularization."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from prune_framework.contracts import BaseModelAdapter, ChannelSparsityTarget


@dataclass(frozen=True)
class RegularizationReport:
    """Detached accounting emitted whenever a term augments a task loss."""

    strength: float
    raw_penalty: float
    weighted_penalty: float
    target_count: int


class RegularizationTerm(ABC):
    """A differentiable regularizer over adapter-approved sparsity targets."""

    @abstractmethod
    def targets(self, adapter: BaseModelAdapter) -> List[ChannelSparsityTarget]:
        raise NotImplementedError

    @abstractmethod
    def raw_penalty(self, adapter: BaseModelAdapter) -> torch.Tensor:
        raise NotImplementedError

    def setup(self, adapter: BaseModelAdapter) -> None:
        """Install any train-time state required by this term."""

    def teardown(self) -> None:
        """Remove train-time state before physical structural mutation."""

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        del state


class L1BatchNormScaleRegularization(RegularizationTerm):
    """Network-Slimming penalty ``sum(|gamma|)`` over Conv-associated BN scales."""

    def targets(self, adapter: BaseModelAdapter) -> List[ChannelSparsityTarget]:
        return adapter.get_channel_sparsity_targets()

    def raw_penalty(self, adapter: BaseModelAdapter) -> torch.Tensor:
        targets = self.targets(adapter)
        if not targets:
            raise RuntimeError(
                f"{type(adapter).__name__} exposes no BatchNorm scale targets for channel-sparsity regularization."
            )
        return torch.stack([target.parameter.abs().sum() for target in targets]).sum()


class HardConcreteChannelGate(nn.Module):
    """One stochastic Hard-Concrete gate for every channel of a BN output."""

    def __init__(
        self,
        channels: int,
        *,
        beta: float = 2.0 / 3.0,
        gamma: float = -0.1,
        zeta: float = 1.1,
        log_alpha_init: float = 0.0,
    ):
        super().__init__()
        if channels < 1 or beta <= 0 or gamma >= 0 or zeta <= 1:
            raise ValueError("Invalid Hard-Concrete parameters.")
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.zeta = float(zeta)
        self.log_alpha = nn.Parameter(torch.full((channels,), float(log_alpha_init)))

    def sample_gate(self) -> torch.Tensor:
        """Sample a differentiable gate in [0, 1] using the current RNG."""
        uniform = torch.rand_like(self.log_alpha).clamp_(1e-6, 1.0 - 1e-6)
        concrete = torch.sigmoid((uniform.log() - (1.0 - uniform).log() + self.log_alpha) / self.beta)
        return (concrete * (self.zeta - self.gamma) + self.gamma).clamp(0.0, 1.0)

    def deterministic_gate(self) -> torch.Tensor:
        """Mean-logit deterministic gate used for evaluation and plan creation."""
        concrete = torch.sigmoid(self.log_alpha)
        return (concrete * (self.zeta - self.gamma) + self.gamma).clamp(0.0, 1.0)

    def expected_l0(self) -> torch.Tensor:
        """Expected number of non-zero gates from the Hard-Concrete CDF."""
        offset = self.beta * math.log(-self.gamma / self.zeta)
        return torch.sigmoid(self.log_alpha - offset).sum()

    def forward(self, output: torch.Tensor) -> torch.Tensor:
        if output.ndim < 2 or output.shape[1] != self.log_alpha.numel():
            raise ValueError("Hard-Concrete gate output shape does not match its channel count.")
        gate = self.sample_gate() if self.training else self.deterministic_gate()
        return output * gate.reshape(1, -1, *([1] * (output.ndim - 2)))


@dataclass
class _GateBinding:
    target: ChannelSparsityTarget
    gate: HardConcreteChannelGate
    hook: Any


class L0HardConcreteRegularization(RegularizationTerm):
    """Expected-L0 regularization with stochastic channel-output gates.

    Gate parameters are registered on the adapter-approved BatchNorm module so
    they appear in ``model.parameters()`` and ``model.state_dict()``. A forward
    hook multiplies the complete BN output, including its bias, without changing
    tensor dimensions during regularized recovery.
    """

    _GATE_ATTRIBUTE = "_prune_hard_concrete_gate"

    def __init__(
        self,
        *,
        beta: float = 2.0 / 3.0,
        gamma: float = -0.1,
        zeta: float = 1.1,
        log_alpha_init: float = 0.0,
    ):
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.zeta = float(zeta)
        self.log_alpha_init = float(log_alpha_init)
        self._bindings: Dict[str, _GateBinding] = {}

    def setup(self, adapter: BaseModelAdapter) -> None:
        if self._bindings:
            return
        targets = self.targets(adapter)
        if not targets:
            raise RuntimeError(
                f"{type(adapter).__name__} exposes no BatchNorm scale targets for Hard-Concrete channel gates."
            )
        for target in targets:
            existing = getattr(target.module, self._GATE_ATTRIBUTE, None)
            if existing is not None and not isinstance(existing, HardConcreteChannelGate):
                raise RuntimeError(f"{target.name} already has an incompatible channel gate.")
            gate = existing or HardConcreteChannelGate(
                target.parameter.numel(),
                beta=self.beta,
                gamma=self.gamma,
                zeta=self.zeta,
                log_alpha_init=self.log_alpha_init,
            ).to(device=target.parameter.device, dtype=target.parameter.dtype)
            if existing is None:
                setattr(target.module, self._GATE_ATTRIBUTE, gate)
            hook = target.module.register_forward_hook(
                lambda _module, _inputs, output, channel_gate=gate: channel_gate(output)
            )
            self._bindings[target.name] = _GateBinding(target=target, gate=gate, hook=hook)

    def teardown(self) -> None:
        for binding in self._bindings.values():
            binding.hook.remove()
            if getattr(binding.target.module, self._GATE_ATTRIBUTE, None) is binding.gate:
                delattr(binding.target.module, self._GATE_ATTRIBUTE)
        self._bindings.clear()

    def targets(self, adapter: BaseModelAdapter) -> List[ChannelSparsityTarget]:
        return adapter.get_channel_sparsity_targets()

    def raw_penalty(self, adapter: BaseModelAdapter) -> torch.Tensor:
        del adapter
        if not self._bindings:
            raise RuntimeError("Hard-Concrete gates must be installed before computing expected L0.")
        return torch.stack([binding.gate.expected_l0() for binding in self._bindings.values()]).sum()

    def deterministic_scores(self, adapter: BaseModelAdapter) -> Dict[str, torch.Tensor]:
        del adapter
        if not self._bindings:
            raise RuntimeError("Hard-Concrete gates are unavailable for pruning decisions.")
        return {name: binding.gate.deterministic_gate().detach() for name, binding in self._bindings.items()}

    def pruning_indices(
        self, adapter: BaseModelAdapter, *, threshold: float, min_channels: int
    ) -> Dict[str, List[int]]:
        if not 0 <= threshold <= 1:
            raise ValueError("Hard-Concrete pruning threshold must be in [0, 1].")
        decisions: Dict[str, List[int]] = {}
        for name, scores in self.deterministic_scores(adapter).items():
            binding = self._bindings[name]
            maximum = max(0, binding.target.channel_target.module.out_channels - int(min_channels))
            selected = torch.nonzero(scores <= threshold, as_tuple=False).flatten().tolist()
            # Keep the least-active gates when a threshold would violate the
            # structural minimum. Stable sorting keeps equal gates deterministic.
            if len(selected) > maximum:
                selected = torch.argsort(scores, stable=True)[:maximum].tolist()
            if selected:
                decisions[name] = [int(index) for index in selected]
        return decisions

    def state_dict(self) -> Dict[str, Any]:
        return {
            "beta": self.beta,
            "gamma": self.gamma,
            "zeta": self.zeta,
            "log_alpha_init": self.log_alpha_init,
            "gates": {name: list(binding.gate.log_alpha.shape) for name, binding in self._bindings.items()},
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        for key in ("beta", "gamma", "zeta"):
            if key in state and not math.isclose(float(state[key]), float(getattr(self, key))):
                raise ValueError(f"Hard-Concrete checkpoint {key} does not match this controller.")


class RegularizationController:
    """Schedule and checkpoint state for adding a regularizer during recovery.

    The controller deliberately does not own an optimizer or dataloader. A
    recovery loop calls :meth:`augment_loss` before ``backward`` and
    :meth:`advance` after its optimizer step. This keeps the primitive useful
    for any training system while making state serialization explicit.
    """

    _SCHEDULES = {"constant", "linear_warmup", "cosine_decay"}

    def __init__(
        self,
        term: RegularizationTerm,
        adapter: BaseModelAdapter,
        *,
        strength: float,
        schedule: str = "constant",
        warmup_steps: int = 0,
        total_steps: int = 0,
    ):
        if strength < 0:
            raise ValueError("Regularization strength must be non-negative.")
        if schedule not in self._SCHEDULES:
            raise ValueError(f"Unsupported regularization schedule '{schedule}'.")
        if warmup_steps < 0 or total_steps < 0:
            raise ValueError("Regularization warmup_steps and total_steps must be non-negative.")
        self.term = term
        self.adapter = adapter
        self.strength = float(strength)
        self.schedule = schedule
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.step = 0
        self.term.setup(adapter)

    def current_strength(self, step: Optional[int] = None, total_steps: Optional[int] = None) -> float:
        step = self.step if step is None else int(step)
        total = self.total_steps if total_steps is None else int(total_steps)
        if self.strength == 0:
            return 0.0
        if self.schedule == "constant":
            return self.strength
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.strength * float(step + 1) / float(self.warmup_steps)
        if self.schedule == "linear_warmup":
            return self.strength
        # Cosine starts after optional warmup and reaches zero on the final
        # configured step. If a total is absent, preserve the full strength.
        if total <= self.warmup_steps:
            return self.strength
        progress = min(1.0, max(0.0, (step - self.warmup_steps) / max(1, total - self.warmup_steps - 1)))
        return self.strength * 0.5 * (1.0 + math.cos(math.pi * progress))

    def augment_loss(
        self,
        task_loss: torch.Tensor,
        *,
        step: Optional[int] = None,
        total_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, RegularizationReport]:
        if not isinstance(task_loss, torch.Tensor) or task_loss.numel() != 1:
            raise TypeError("Regularization can only augment a scalar task loss.")
        raw = self.term.raw_penalty(self.adapter)
        strength = self.current_strength(step=step, total_steps=total_steps)
        weighted = raw * strength
        total = task_loss.reshape(()) + weighted
        report = RegularizationReport(
            strength=strength,
            raw_penalty=float(raw.detach().cpu()),
            weighted_penalty=float(weighted.detach().cpu()),
            target_count=len(self.term.targets(self.adapter)),
        )
        return total, report

    def advance(self, steps: int = 1) -> None:
        if steps < 0:
            raise ValueError("Regularization progress cannot move backwards.")
        self.step += int(steps)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "term": type(self.term).__name__,
            "strength": self.strength,
            "schedule": self.schedule,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "step": self.step,
            "targets": [target.describe() for target in self.term.targets(self.adapter)],
            "term_state": self.term.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state.get("term") != type(self.term).__name__:
            raise ValueError("Regularization checkpoint term does not match this controller.")
        self.step = int(state.get("step", 0))
        self.term.load_state_dict(dict(state.get("term_state", {})))

    def prune_indices(self, *, threshold: float, min_channels: int) -> Dict[str, List[int]]:
        if not isinstance(self.term, L0HardConcreteRegularization):
            raise RuntimeError("Only Hard-Concrete regularization produces gate pruning decisions.")
        return self.term.pruning_indices(self.adapter, threshold=threshold, min_channels=min_channels)

    def remove_train_time_state(self) -> None:
        """Remove stochastic gates only after a structural plan has validated."""
        self.term.teardown()
