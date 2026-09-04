import torch
import torch.nn as nn
import torch_pruning as tp
from typing import Dict, Any, List
from prune_framework.core.interfaces import BasePruner, BaseModelAdapter, BaseImportanceCriterion, BaseGranularity
from prune_framework.core.registry import register_pruner
from prune_framework.core.utils import make_divisible


@register_pruner("structured")
@register_pruner("structured_channel")
class StructuredPruner(BasePruner):
    """
    Structured Channel (Width) Pruner.
    Uses DependencyGraph from torch_pruning to trace inter-layer dependencies
    and physically shrink output channels and dependent input channels.
    """

    def prune(
        self,
        model_adapter: BaseModelAdapter,
        criterion: BaseImportanceCriterion,
        granularity: BaseGranularity = None,
        config: Dict[str, Any] = None
    ) -> nn.Module:
        if config is None:
            config = {}
        model = model_adapter.model
        device = next(model.parameters()).device

        pruning_params = config.get("pruning_params", config.get("amount", 0.3))
        if hasattr(model_adapter, "get_pruneable_modules"):
            prunable_layers = model_adapter.get_pruneable_modules()
        elif hasattr(model_adapter, "get_pruneable_conv_layers"):
            prunable_layers = model_adapter.get_pruneable_conv_layers()
        else:
            prunable_layers = []

        if isinstance(pruning_params, float):
            pruning_params = [(i, pruning_params) for i in range(len(prunable_layers))]

        pruning_params = sorted(pruning_params, key=lambda x: x[0])

        # Temporarily enable requires_grad for tracing dependency graph
        grad_states = {name: p.requires_grad for name, p in model.named_parameters()}
        for p in model.parameters():
            p.requires_grad = True

        dummy_input = model_adapter.get_dummy_input(device)
        DG = tp.DependencyGraph().build_dependency(model, dummy_input)

        # Restore requires_grad states
        for name, p in model.named_parameters():
            p.requires_grad = grad_states[name]

        with torch.no_grad():
            for layer_idx, rate in pruning_params:
                if layer_idx < 0 or layer_idx >= len(prunable_layers):
                    continue

                layer_name, target_module = prunable_layers[layer_idx]
                if not isinstance(target_module, nn.Conv2d):
                    continue

                out_channels = target_module.out_channels

                if hasattr(criterion, "score"):
                    scores = criterion.score(target_module)
                elif hasattr(criterion, "compute_scores"):
                    scores = criterion.compute_scores(target_module)
                else:
                    scores = torch.norm(target_module.weight.data, p=2, dim=[1, 2, 3])

                if granularity is not None and hasattr(granularity, "extract_indices"):
                    prune_indices = granularity.extract_indices(scores, rate, target_module)
                else:
                    n_pruned = make_divisible(rate * out_channels, 2)
                    if n_pruned >= out_channels:
                        n_pruned = out_channels - 2
                    if n_pruned <= 0:
                        continue
                    prune_indices = torch.argsort(scores)[:n_pruned].tolist()

                if not prune_indices:
                    continue

                group = DG.get_pruning_group(target_module, tp.prune_conv_out_channels, prune_indices)
                if DG.check_pruning_group(group):
                    group.prune()
                    print(f" - Structured Pruner: Layer {layer_idx} ({layer_name}) pruned {len(prune_indices)}/{out_channels} channels. New out_channels={target_module.out_channels}")
                else:
                    print(f" - Warning: Dependency check failed for pruning {layer_name}")

        return model

