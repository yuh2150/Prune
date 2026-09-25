import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from prune_framework.core.interfaces import BaseImportanceCriterion
from prune_framework.core.registry import register_criterion
from prune_framework.modules.analysis.importance import LayerAdaptiveMagnitudeNormalizer


@register_criterion("lamp")
class LAMPCriterion(BaseImportanceCriterion):
    """
    Canonical element-wise Layer-Adaptive Magnitude Pruning (LAMP) score.

    Each eligible Conv2d or Linear weight tensor is normalized independently;
    the unstructured pruner then makes the global pruning decision.
    """

    provides_elementwise_scores = True
    global_selection = True

    def score(self, module: nn.Module, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            raise ValueError(f"LAMPCriterion supports Conv2d and Linear, got {type(module).__name__}")
        return LayerAdaptiveMagnitudeNormalizer.scores(module.weight)

    def compute_scores(self, module: nn.Module, grad: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.score(module)
