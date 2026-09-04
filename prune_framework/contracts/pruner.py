from abc import ABC, abstractmethod
import torch.nn as nn
from typing import Dict, Any
from .adapter import BaseModelAdapter
from .criterion import BaseImportanceCriterion
from .granularity import BaseGranularity


class BasePruner(ABC):
    """Abstract Base Class for Pruners."""

    @abstractmethod
    def prune(
        self,
        model_adapter: BaseModelAdapter,
        criterion: BaseImportanceCriterion,
        granularity: BaseGranularity,
        config: Dict[str, Any]
    ) -> nn.Module:
        """Executes structural, unstructured, or layer pruning."""
        pass
