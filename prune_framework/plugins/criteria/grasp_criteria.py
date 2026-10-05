"""GraSP saliency criterion for one-shot, element-wise pruning."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from prune_framework.contracts.targets import TargetType
from prune_framework.core.interfaces import BaseImportanceCriterion
from prune_framework.core.registry import register_criterion


@register_criterion("grasp")
class GraSPCriterion(BaseImportanceCriterion):
    supported_pruning_modes = frozenset(['unstructured'])
    uses_bn_wrapper = False
    """Element-wise GraSP score: ``-W * (H @ grad(loss))``.

    The higher-order runner supplies a detached Hessian-gradient product in
    ``context``.  Negative scores are intentional: global pruning removes the
    lowest scores, as prescribed by the gradient-flow preservation objective.
    """

    requires_higher_order_calibration = True
    global_selection = True
    calibration_target_types = {TargetType.CONV_WEIGHT, TargetType.LINEAR_WEIGHT}

    def score(self, module: nn.Module, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            raise ValueError(f"GraSPCriterion supports Conv2d and Linear, got {type(module).__name__}.")
        hessian_gradient_product = context.get("grad") if context else None
        if hessian_gradient_product is None:
            raise RuntimeError("GraSPCriterion requires a detached Hessian-gradient product from higher-order calibration.")
        if tuple(hessian_gradient_product.shape) != tuple(module.weight.shape):
            raise ValueError(
                f"GraSP HVP shape {tuple(hessian_gradient_product.shape)} does not match "
                f"weight shape {tuple(module.weight.shape)}."
            )
        return (-(module.weight.detach() * hessian_gradient_product.detach())).detach()


@register_criterion("grasp_layer_normalized")
class GraSPLayerNormalizedCriterion(GraSPCriterion):
    """GraSP with per-layer score-scale normalization before global ranking.

    Raw GraSP scores are still computed exactly as :class:`GraSPCriterion`.
    The unstructured pruner divides every target's scores by its mean absolute
    score before its global comparison and records raw/normalized statistics in
    the pruning plan.  This is intentionally a separately named variant, not
    a change to the plain ``grasp`` baseline.
    """

    score_normalization = "mean_abs"


@register_criterion("grasp_conv1_capped")
class GraSPConv1CappedCriterion(GraSPCriterion):
    """Plain GraSP ranking with a maximum 10% budget for ``conv1.conv``.

    The cap prevents the first feature extractor from absorbing the global
    pruning budget. Remaining eligible weights are still ranked globally by
    the unmodified GraSP score.
    """

    max_pruning_fraction_by_target = {"conv1.conv": 0.10}
