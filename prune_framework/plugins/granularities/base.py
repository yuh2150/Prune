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
    """Filter-level granularity for 2D Conv filter pruning."""

    def extract_indices(self, scores: torch.Tensor, prune_ratio: float, module: nn.Module) -> List[int]:
        total = scores.numel()
        n_prune = int(total * prune_ratio)
        if n_prune >= total:
            n_prune = max(0, total - 1)
        if n_prune <= 0:
            return []
        return torch.argsort(scores)[:n_prune].tolist()


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
