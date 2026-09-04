import torch
import torch.nn as nn
from typing import Tuple, Any, Optional
from prune_framework.core.registry import PluginRegistry


class ModelLoader:
    """Helper module to load heterogeneous models using registered adapters."""

    @staticmethod
    def load(model_name: str, weights_path: str, device: torch.device) -> Tuple[nn.Module, Optional[Any]]:
        adapter_cls = PluginRegistry.get_model_adapter(model_name)
        if hasattr(adapter_cls, "load_model"):
            return adapter_cls.load_model(weights_path, device)
        raise NotImplementedError(
            f"Model adapter '{model_name}' ({adapter_cls.__name__}) does not implement 'load_model'."
        )
