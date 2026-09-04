import torch
import torch.nn as nn
import torch.nn.utils.prune as prune_api
from typing import Dict, Any
from prune_framework.core.interfaces import BasePruner, BaseModelAdapter, BaseImportanceCriterion
from prune_framework.core.registry import register_pruner


@register_pruner("unstructured")
@register_pruner("unstructured_weight")
class UnstructuredPruner(BasePruner):
    """
    Unstructured Weight Sparsity Pruner.
    Applies element-wise weight masking using PyTorch native pruning API.
    Zeroes out weights without altering network tensor shapes.
    """

    def prune(
        self,
        model_adapter: BaseModelAdapter,
        criterion: BaseImportanceCriterion,
        config: Dict[str, Any]
    ) -> nn.Module:
        model = model_adapter.model
        prunable_layers = model_adapter.get_pruneable_conv_layers()
        pruning_params = config.get("pruning_params", config.get("amount", 0.3))

        if isinstance(pruning_params, float):
            pruning_params = [(i, pruning_params) for i in range(len(prunable_layers))]

        for layer_idx, rate in pruning_params:
            if layer_idx < 0 or layer_idx >= len(prunable_layers):
                continue

            layer_name, target_conv = prunable_layers[layer_idx]
            if rate <= 0:
                continue

            # Apply L1 unstructured or random unstructured based on criterion type name
            criterion_name = config.get("criterion_name", "l1_norm")
            if criterion_name == "random":
                prune_api.random_unstructured(target_conv, name="weight", amount=rate)
            else:
                prune_api.l1_unstructured(target_conv, name="weight", amount=rate)

            # Make mask permanent
            prune_api.remove(target_conv, "weight")
            print(f" - Unstructured Pruner: Layer {layer_idx} ({layer_name}) applied {rate * 100:.1f}% sparsity mask.")

        return model
