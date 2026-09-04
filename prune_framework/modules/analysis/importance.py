import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from prune_framework.core.registry import PluginRegistry


class ImportanceScorer:
    """Helper module to compute layer importance distributions."""

    def __init__(self, criterion_name: str = "l1"):
        self.criterion_cls = PluginRegistry.get_criterion(criterion_name)

    def score_layer(self, module: nn.Module, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        criterion = self.criterion_cls()
        return criterion.score(module, context)
