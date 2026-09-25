"""Canonical SynFlow saliency criterion."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from prune_framework.contracts.targets import TargetType
from prune_framework.core.interfaces import BaseImportanceCriterion
from prune_framework.core.registry import register_criterion


@register_criterion("synflow")
class SynFlowCriterion(BaseImportanceCriterion):
    """Element-wise SynFlow saliency: ``abs(W * d(sum(output))/dW)``."""

    requires_synflow_calibration = True
    global_selection = True
    calibration_target_types = {TargetType.CONV_WEIGHT, TargetType.LINEAR_WEIGHT}

    def score(self, module: nn.Module, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            raise ValueError(f"SynFlowCriterion supports Conv2d and Linear, got {type(module).__name__}.")
        gradient = context.get("grad") if context else None
        if gradient is None:
            raise RuntimeError("SynFlowCriterion requires a detached gradient from SynFlow calibration.")
        if tuple(gradient.shape) != tuple(module.weight.shape):
            raise ValueError(
                f"SynFlow gradient shape {tuple(gradient.shape)} does not match "
                f"weight shape {tuple(module.weight.shape)}."
            )
        return (module.weight.detach() * gradient.detach()).abs().detach()
