import torch
import torch.nn as nn
from typing import List
from prune_framework.core.interfaces import BaseGranularity
from prune_framework.core.registry import register_granularity
from prune_framework.core.utils import make_divisible


@register_granularity("channel")
class ChannelGranularity(BaseGranularity):
    """Channel-level granularity for output channel pruning."""

    def extract_indices(self, scores: torch.Tensor, prune_ratio: float, module: nn.Module) -> List[int]:
        total = scores.numel()
        n_prune = make_divisible(total * prune_ratio, 2)
        if n_prune >= total:
            n_prune = total - 2
        if n_prune <= 0:
            return []
        return torch.argsort(scores)[:n_prune].tolist()


@register_granularity("filter")
class FilterGranularity(BaseGranularity):
    """Conv2d output-filter granularity.

    A filter is ``weight[out_channel, ...]``.  The structured pruner removes
    that output channel and lets its dependency graph update bias, consumers
    and compatible normalisation layers. Grouped/depthwise kernels are
    rejected until their group-coupled dependency semantics are explicit.
    """

    conv_only = True

    def extract_indices(self, scores: torch.Tensor, prune_ratio: float, module: nn.Module) -> List[int]:
        if not isinstance(module, nn.Conv2d):
            raise TypeError("Filter pruning requires a Conv2d output-filter target.")
        if module.groups != 1:
            raise ValueError("Filter pruning does not support grouped or depthwise Conv2d modules.")
        total = scores.numel()
        n_prune = int(total * prune_ratio)
        if n_prune >= total:
            n_prune = max(0, total - 1)
        if n_prune <= 0:
            return []
        return torch.argsort(scores, stable=True)[:n_prune].tolist()


@register_granularity("weight")
class WeightGranularity(BaseGranularity):
    """Weight-level (unstructured elementwise) granularity."""

    def extract_indices(self, scores: torch.Tensor, prune_ratio: float, module: nn.Module) -> List[int]:
        return []


@register_granularity("block")
class BlockGranularity(BaseGranularity):
    """Block-level granularity for C3 / Bottleneck block removal."""

    def extract_indices(self, scores: torch.Tensor, prune_ratio: float, module: nn.Module) -> List[int]:
        total = scores.numel()
        n_remove = max(0, min(int(prune_ratio), total - 1))
        if n_remove <= 0:
            return []
        return torch.argsort(scores)[:n_remove].tolist()


@register_granularity("layer")
class LayerGranularity(BaseGranularity):
    """Layer-level granularity for dropping entire transformer or CNN layers."""

    def extract_indices(self, scores: torch.Tensor, prune_ratio: float, module: nn.Module) -> List[int]:
        total = scores.numel()
        n_remove = max(0, min(int(prune_ratio), total - 1))
        return torch.argsort(scores)[:n_remove].tolist()


@register_granularity("head")
class HeadGranularity(BaseGranularity):
    """Multi-Head Attention head granularity."""

    def extract_indices(self, scores: torch.Tensor, prune_ratio: float, module: nn.Module) -> List[int]:
        total = scores.numel()
        n_prune = int(total * prune_ratio)
        return torch.argsort(scores)[:n_prune].tolist()
