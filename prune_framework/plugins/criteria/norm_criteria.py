import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from prune_framework.core.interfaces import BaseImportanceCriterion
from prune_framework.core.registry import register_criterion


@register_criterion("l1")
@register_criterion("l1_norm")
class L1NormCriterion(BaseImportanceCriterion):
    """L1-Norm weight magnitude criterion (Sum of absolute values per output channel)."""

    def score(self, module: nn.Module, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        if isinstance(module, nn.Conv2d):
            return torch.norm(module.weight.data, p=1, dim=[1, 2, 3])
        elif isinstance(module, nn.Linear):
            return torch.norm(module.weight.data, p=1, dim=1)
        raise ValueError(f"L1NormCriterion does not support module type {type(module)}")

    def compute_scores(self, module: nn.Module, grad: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.score(module)


@register_criterion("l2")
@register_criterion("l2_norm")
class L2NormCriterion(BaseImportanceCriterion):
    """L2-Norm weight magnitude criterion (Euclidean norm per output channel)."""

    def score(self, module: nn.Module, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        if isinstance(module, nn.Conv2d):
            return torch.norm(module.weight.data, p=2, dim=[1, 2, 3])
        elif isinstance(module, nn.Linear):
            return torch.norm(module.weight.data, p=2, dim=1)
        raise ValueError(f"L2NormCriterion does not support module type {type(module)}")

    def compute_scores(self, module: nn.Module, grad: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.score(module)


@register_criterion("magnitude")
class MagnitudeCriterion(BaseImportanceCriterion):
    """Absolute weight magnitude importance score."""

    def score(self, module: nn.Module, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            return module.weight.data.abs().mean(dim=[i for i in range(1, module.weight.data.ndim)])
        raise ValueError(f"MagnitudeCriterion does not support module type {type(module)}")

    def compute_scores(self, module: nn.Module, grad: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.score(module)


@register_criterion("random")
class RandomCriterion(BaseImportanceCriterion):
    """Random importance score (used for ablation baseline)."""

    def score(self, module: nn.Module, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            num_channels = module.out_channels if isinstance(module, nn.Conv2d) else module.out_features
            return torch.rand(num_channels, device=module.weight.device)
        raise ValueError(f"RandomCriterion does not support module type {type(module)}")

    def compute_scores(self, module: nn.Module, grad: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.score(module)

