import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from prune_framework.core.registry import PluginRegistry


class ImportanceScorer:
    """Helper module to compute layer importance distributions."""

    def __init__(self, criterion_name: str = "l1"):
        self.criterion_cls = PluginRegistry.get_criterion(criterion_name)

    def score_layer(self, module: nn.Module, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        criterion = self.criterion_cls()
        return criterion.score(module, context)


class LayerAdaptiveMagnitudeNormalizer:
    """Canonical, tie-aware LAMP normalization for one weight tensor.

    For a weight with magnitude ``a_i``, the score is
    ``a_i / sqrt(sum(a_j**2 for a_j >= a_i))``.  Scores are calculated within
    one tensor only; a caller can then compare the normalized scores globally.
    """

    @staticmethod
    def scores(weight: torch.Tensor) -> torch.Tensor:
        if weight.numel() == 0:
            return torch.empty_like(weight)
        magnitudes = weight.detach().abs().reshape(-1)
        # Stable order gives reproducible placement. Equal magnitudes receive
        # the same denominator below, rather than an order-dependent score.
        order = torch.argsort(magnitudes, stable=True)
        sorted_magnitudes = magnitudes[order]
        tail_energy = torch.cumsum(sorted_magnitudes.square().flip(0), dim=0).flip(0)

        positions = torch.arange(sorted_magnitudes.numel(), device=weight.device)
        starts = torch.ones_like(positions, dtype=torch.bool)
        if starts.numel() > 1:
            starts[1:] = sorted_magnitudes[1:] != sorted_magnitudes[:-1]
        group_starts = torch.cummax(torch.where(starts, positions, torch.zeros_like(positions)), dim=0).values
        denominator = tail_energy[group_starts].sqrt()
        sorted_scores = torch.where(
            denominator > 0,
            sorted_magnitudes / denominator,
            torch.zeros_like(sorted_magnitudes),
        )
        scores = torch.empty_like(magnitudes)
        scores[order] = sorted_scores
        return scores.reshape_as(weight).detach()
