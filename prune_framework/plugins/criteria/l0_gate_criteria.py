"""Deterministic Hard-Concrete gate importance for structured plans."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from prune_framework.core.interfaces import BaseImportanceCriterion
from prune_framework.core.registry import register_criterion
from prune_framework.modules.regularization import HardConcreteChannelGate


@register_criterion("l0_gate")
@register_criterion("hard_concrete")
class HardConcreteGateCriterion(BaseImportanceCriterion):
    """Score channels by their deterministic Hard-Concrete activation gate."""

    def score(self, module: nn.Module, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        del context
        batch_norm = module if isinstance(module, nn.BatchNorm2d) else getattr(module, "bn", None)
        gate = getattr(batch_norm, "_prune_hard_concrete_gate", None)
        if not isinstance(gate, HardConcreteChannelGate):
            raise ValueError("HardConcreteGateCriterion requires a BatchNorm2d module with an installed channel gate.")
        return gate.deterministic_gate().detach()
