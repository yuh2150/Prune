import torch
import torch.nn as nn
from typing import Dict, Any
from prune_framework.core.interfaces import BasePruner, BaseModelAdapter, BaseImportanceCriterion, BaseGranularity
from prune_framework.core.registry import register_pruner


@register_pruner("depth")
@register_pruner("layer_depth")
class DepthPruner(BasePruner):
    """
    Depth / Layer Pruner.
    Physically removes Bottleneck blocks in C3 modules (YOLO) or Transformer encoder/decoder layers (RT-DETR)
    based on block-level importance scoring.
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
        pruning_params = config.get("pruning_params", config.get("drop_params", []))
        block_tuples = model_adapter.get_pruneable_blocks()
        block_map = {layer_id: module for layer_id, module in block_tuples}

        with torch.no_grad():
            if isinstance(pruning_params, list):
                for item in pruning_params:
                    if not isinstance(item, (tuple, list)) or len(item) < 2:
                        continue
                    layer_id, remove_num = item[0], item[1]

                    if layer_id not in block_map:
                        continue

                    module = block_map[layer_id]

                    # Handle C3 / BottleneckCSP modules (YOLOv5)
                    if hasattr(module, "m") and isinstance(module.m, nn.Sequential):
                        seq = module.m
                        n_orig = len(seq)
                        n_keep = max(1, n_orig - remove_num)
                        if n_keep >= n_orig:
                            continue

                        scores = []
                        for b in seq:
                            conv_target = b.cv2.conv if hasattr(b, "cv2") and hasattr(b.cv2, "conv") else b[0]
                            if hasattr(criterion, "score"):
                                s = criterion.score(conv_target).mean().item()
                            elif hasattr(criterion, "compute_scores"):
                                s = criterion.compute_scores(conv_target).mean().item()
                            else:
                                s = torch.norm(conv_target.weight.data, p=2).item()
                            scores.append(s)

                        keep_idx = sorted(torch.argsort(torch.tensor(scores), descending=True)[:n_keep].tolist())
                        module.m = nn.Sequential(*[seq[i] for i in keep_idx])
                        print(f" - DepthPruner: Layer {layer_id} depth reduced from {n_orig} to {n_keep} blocks. Kept indices: {keep_idx}")

            elif isinstance(pruning_params, float):
                ratio = pruning_params
                if hasattr(model_adapter, "prune_transformer_layers"):
                    model = model_adapter.prune_transformer_layers(ratio)
                    print(f" - DepthPruner: Applied global transformer layer pruning with ratio {ratio:.2f}")

        return model
