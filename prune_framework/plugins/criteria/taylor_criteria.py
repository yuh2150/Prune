import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from prune_framework.core.interfaces import BaseImportanceCriterion
from prune_framework.core.registry import register_criterion


@register_criterion("taylor")
@register_criterion("taylor_first_order")
class TaylorFirstOrderCriterion(BaseImportanceCriterion):
    """
    First-Order Taylor Expansion Importance Score:
    Score = |Weight * Gradient| summed over input channels and spatial dimensions.
    Requires gradients to be computed via backward pass before scoring.
    """

    def score(self, module: nn.Module, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        grad = context.get("grad") if context else None
        g = grad if grad is not None else getattr(module.weight, "grad", None)
        if g is None:
            raise RuntimeError(f"TaylorFirstOrderCriterion requires gradients on module {module}. Perform backward pass prior to pruning.")
        if isinstance(module, nn.Conv2d):
            return (module.weight.data * g).abs().sum(dim=[1, 2, 3])
        elif isinstance(module, nn.Linear):
            return (module.weight.data * g).abs().sum(dim=1)
        raise ValueError(f"TaylorFirstOrderCriterion does not support module type {type(module)}")

    def compute_scores(self, module: nn.Module, grad: Optional[torch.Tensor] = None) -> torch.Tensor:
        context = {"grad": grad} if grad is not None else None
        return self.score(module, context=context)

