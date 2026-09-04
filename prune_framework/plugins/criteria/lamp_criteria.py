import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from prune_framework.core.interfaces import BaseImportanceCriterion
from prune_framework.core.registry import register_criterion


@register_criterion("lamp")
class LAMPCriterion(BaseImportanceCriterion):
    """
    Layer-Adaptive Magnitude Pruning (LAMP) Score.
    Normalizes squared weight magnitudes relative to the cumulative weight distribution in the layer.
    """

    def score(self, module: nn.Module, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        if isinstance(module, nn.Conv2d):
            w = module.weight.data.view(module.weight.size(0), -1)
            w_abs_sq = torch.sort(w.abs() ** 2, dim=1, descending=True)[0]
            sum_res = torch.cumsum(w_abs_sq, dim=1)
            return (w_abs_sq / (sum_res + 1e-8)).sum(dim=1)
        raise ValueError(f"LAMPCriterion requires Conv2d module, got {type(module)}")

    def compute_scores(self, module: nn.Module, grad: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.score(module)

