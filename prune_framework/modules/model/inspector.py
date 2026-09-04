import torch.nn as nn
from typing import Dict, Any
from prune_framework.core.registry import PluginRegistry


class ModelInspector:
    """Inspects prunable layer counts, shapes, and architecture details."""

    @staticmethod
    def inspect(model: nn.Module, model_name: str) -> Dict[str, Any]:
        adapter_cls = PluginRegistry.get_model_adapter(model_name)
        adapter = adapter_cls(model)
        modules = adapter.get_pruneable_modules()
        blocks = adapter.get_pruneable_blocks()

        return {
            "total_pruneable_modules": len(modules),
            "total_pruneable_blocks": len(blocks),
            "module_names": [name for name, _ in modules]
        }
