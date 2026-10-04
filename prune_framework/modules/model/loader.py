import inspect

import torch
import torch.nn as nn
from typing import Tuple, Any, Optional
from prune_framework.core.registry import PluginRegistry


class ModelLoader:
    """Helper module to load heterogeneous models using registered adapters."""

    @staticmethod
    def load(
        model_name: str,
        weights_path: str,
        device: torch.device,
        **model_options: Any,
    ) -> Tuple[nn.Module, Optional[Any]]:
        adapter_cls = PluginRegistry.get_model_adapter(model_name)
        if hasattr(adapter_cls, "load_model"):
            load_model = adapter_cls.load_model
            parameters = inspect.signature(load_model).parameters
            accepts_kwargs = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
            supported_options = {
                name: value for name, value in model_options.items()
                if accepts_kwargs or name in parameters
            }
            return load_model(weights_path, device, **supported_options)
        raise NotImplementedError(
            f"Model adapter '{model_name}' ({adapter_cls.__name__}) does not implement 'load_model'."
        )
