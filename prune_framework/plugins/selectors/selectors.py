from typing import Dict, Union, Any, Optional
from prune_framework.core.interfaces import BaseSelector
from prune_framework.core.registry import register_selector
from prune_framework.contracts.sensitivity_result import (
    SensitivityResult,
    SelectionResult,
    SelectedLayer,
)


@register_selector("sensitivity")
class SensitivitySelector(BaseSelector):
    """
    Selects layer pruning rates based on metric degradation relative to baseline.
    Only rates that keep relative metric drop within max_allowed_relative_drop (default 5%)
    and stay under target_sparsity (per-layer cap) are chosen.
    """

    def select_layers(
        self,
        sensitivity_data: Union[SensitivityResult, Dict[int, Dict[float, float]]],
        target_sparsity: float,
        **kwargs: Any
    ) -> SelectionResult:
        if isinstance(sensitivity_data, dict):
            res = SensitivityResult.from_dict(sensitivity_data)
        else:
            res = sensitivity_data

        max_allowed_rel_drop = kwargs.get("max_allowed_relative_drop", kwargs.get("max_allowed_drop", 0.05))

        selected = []
        for idx in sorted(res.profiles.keys()):
            profile = res.profiles[idx]
            best_rate = 0.0
            best_metric = res.baseline_score
            best_drop = 0.0

            valid_pts = [
                pt for pt in profile.points.values()
                if pt.is_valid and pt.rate <= target_sparsity
            ]
            valid_pts.sort(key=lambda p: p.rate)

            for pt in valid_pts:
                if pt.relative_drop <= max_allowed_rel_drop:
                    best_rate = pt.rate
                    best_metric = pt.score
                    best_drop = pt.metric_drop

            selected.append(
                SelectedLayer(
                    layer_idx=idx,
                    layer_name=profile.layer_name,
                    target_rate=best_rate,
                    expected_metric=best_metric,
                    expected_drop=best_drop
                )
            )

        return SelectionResult(selected, strategy_name="sensitivity", target_sparsity=target_sparsity)


@register_selector("greedy")
class GreedySelector(BaseSelector):
    """
    Greedy Layer Selection algorithm that selects the maximum pruning rate per layer under target_sparsity
    that achieves the best metric score (minimal degradation).
    """

    def select_layers(
        self,
        sensitivity_data: Union[SensitivityResult, Dict[int, Dict[float, float]]],
        target_sparsity: float,
        **kwargs: Any
    ) -> SelectionResult:
        if isinstance(sensitivity_data, dict):
            res = SensitivityResult.from_dict(sensitivity_data)
        else:
            res = sensitivity_data

        selected = []
        for idx in sorted(res.profiles.keys()):
            profile = res.profiles[idx]
            best_rate = 0.0
            best_metric = res.baseline_score
            best_drop = 0.0

            valid_pts = [
                pt for pt in profile.points.values()
                if pt.is_valid and pt.rate <= target_sparsity
            ]

            if valid_pts:
                # Find point with highest metric score (minimal drop) among non-zero candidates
                # Tie-breaker: higher rate
                best_pt = max(valid_pts, key=lambda p: (p.score, p.rate))
                if best_pt.score > 0:
                    best_rate = best_pt.rate
                    best_metric = best_pt.score
                    best_drop = best_pt.metric_drop

            selected.append(
                SelectedLayer(
                    layer_idx=idx,
                    layer_name=profile.layer_name,
                    target_rate=best_rate,
                    expected_metric=best_metric,
                    expected_drop=best_drop
                )
            )

        return SelectionResult(selected, strategy_name="greedy", target_sparsity=target_sparsity)


@register_selector("threshold")
class ThresholdSelector(BaseSelector):
    """
    Threshold-based Layer Selection strategy.
    Selects the maximum pruning rate per layer under target_sparsity that satisfies either
    an absolute drop threshold (max_allowed_drop) or a relative drop threshold (max_allowed_relative_drop).
    """

    def select_layers(
        self,
        sensitivity_data: Union[SensitivityResult, Dict[int, Dict[float, float]]],
        target_sparsity: float,
        **kwargs: Any
    ) -> SelectionResult:
        if isinstance(sensitivity_data, dict):
            res = SensitivityResult.from_dict(sensitivity_data)
        else:
            res = sensitivity_data

        max_allowed_drop: Optional[float] = kwargs.get("max_allowed_drop")
        max_allowed_rel_drop: Optional[float] = kwargs.get("max_allowed_relative_drop", 0.05)

        selected = []
        for idx in sorted(res.profiles.keys()):
            profile = res.profiles[idx]
            best_rate = 0.0
            best_metric = res.baseline_score
            best_drop = 0.0

            valid_pts = [
                pt for pt in profile.points.values()
                if pt.is_valid and pt.rate <= target_sparsity
            ]
            valid_pts.sort(key=lambda p: p.rate)

            for pt in valid_pts:
                passes_abs = max_allowed_drop is None or pt.metric_drop <= max_allowed_drop
                passes_rel = max_allowed_rel_drop is None or pt.relative_drop <= max_allowed_rel_drop

                if passes_abs and passes_rel:
                    best_rate = pt.rate
                    best_metric = pt.score
                    best_drop = pt.metric_drop

            selected.append(
                SelectedLayer(
                    layer_idx=idx,
                    layer_name=profile.layer_name,
                    target_rate=best_rate,
                    expected_metric=best_metric,
                    expected_drop=best_drop
                )
            )

        return SelectionResult(selected, strategy_name="threshold", target_sparsity=target_sparsity)

