from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class BaseImportanceCriterion(ABC):
    """Abstract Base Class for Importance Criteria."""

    supported_pruning_modes = frozenset({"structured", "unstructured", "depth"})
    uses_bn_wrapper = False
    supported_module_types = (nn.Conv2d, nn.Linear)

    @abstractmethod
    def score(self, module: nn.Module, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Computes 1D tensor of importance scores for channels, weights, or blocks."""
        pass

    def compute_scores(self, module: nn.Module, grad: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Alias method for backward compatibility."""
        context = {"grad": grad} if grad is not None else None
        return self.score(module, context=context)
