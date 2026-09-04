import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from prune_framework.core.interfaces import BaseImportanceCriterion
from prune_framework.core.registry import register_criterion


@register_criterion("bn_scale")
@register_criterion("bn_gamma")
class BNScaleCriterion(BaseImportanceCriterion):
    """
    Network Slimming Criterion based on absolute BatchNorm gamma magnitude (|gamma|).
    """

    def score(self, module: nn.Module, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        if isinstance(module, nn.BatchNorm2d):
            return module.weight.data.abs()
        elif hasattr(module, "bn") and isinstance(module.bn, nn.BatchNorm2d):
            return module.bn.weight.data.abs()
        raise ValueError("BNScaleCriterion requires a BatchNorm2d module or container with .bn attribute.")

    def compute_scores(self, module: nn.Module, grad: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.score(module)


@register_criterion("bn_l1_combined")
class BNL1CombinedCriterion(BaseImportanceCriterion):
    """
    Combined criterion multiplying BatchNorm gamma magnitude by Conv L1-norm.
    """

    def score(self, module: nn.Module, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        if hasattr(module, "conv") and hasattr(module, "bn"):
            conv_norm = torch.norm(module.conv.weight.data, p=1, dim=[1, 2, 3])
            bn_scale = module.bn.weight.data.abs()
            return conv_norm * bn_scale
        raise ValueError("BNL1CombinedCriterion requires module with .conv and .bn attributes.")

    def compute_scores(self, module: nn.Module, grad: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.score(module)

