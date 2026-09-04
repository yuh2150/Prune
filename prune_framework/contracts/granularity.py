from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import List


class BaseGranularity(ABC):
    """Abstract Base Class for Granularities."""

    @abstractmethod
    def extract_indices(
        self,
        scores: torch.Tensor,
        prune_ratio: float,
        module: nn.Module
    ) -> List[int]:
        """Extracts indices to prune based on scores, ratio, and granularity logic."""
        pass
