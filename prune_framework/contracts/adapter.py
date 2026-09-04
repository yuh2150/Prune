from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Any


class BaseModelAdapter(ABC):
    """Abstract Base Class for Model Adapters."""

    def __init__(self, model: nn.Module):
        self.model = model

    @classmethod
    def load_model(cls, weights_path: str, device: torch.device) -> Tuple[nn.Module, Optional[Any]]:
        """Loads model instance and optional checkpoint dictionary."""
        raise NotImplementedError("Subclass must implement load_model classmethod.")

    @abstractmethod
    def get_pruneable_modules(self) -> List[Tuple[str, nn.Module]]:
        """Returns name and module pairs for prunable layers."""
        pass

    @abstractmethod
    def get_pruneable_blocks(self) -> List[Tuple[int, nn.Module]]:
        """Returns block index and module pairs for depth/block pruning."""
        pass

    @abstractmethod
    def get_dummy_input(self, device: torch.device) -> torch.Tensor:
        """Returns dummy tensor for tracing and validation."""
        pass

    def prepare_for_pruning(self):
        """Optional pre-pruning initialization hook."""
        pass

    def post_prune_cleanup(self):
        """Optional post-pruning cleanup or head re-indexing hook."""
        pass
