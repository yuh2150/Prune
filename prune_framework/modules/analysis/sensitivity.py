import copy
from prune_framework.modules.model.snapshot import ModelSnapshot
import torch
import torch.nn as nn
from typing import List, Callable, Optional, Dict, Any
from prune_framework.core.engine import PruningEngine
from prune_framework.core.logging import get_logger
from prune_framework.contracts.sensitivity_result import (
    SensitivityResult,
    LayerSensitivityProfile,
    SensitivityPoint,
)

logger = get_logger("prune_framework.sensitivity")


class SensitivityAnalyzer:
    """Evaluates layer-by-layer pruning sensitivity across pruning rates."""

    def __init__(self, model_name: str, pruner_name: str, criterion_name: str, granularity_name: str = "channel"):
        self.model_name = model_name
        self.pruner_name = pruner_name
        self.criterion_name = criterion_name
        self.granularity_name = granularity_name
        self.engine = PruningEngine(model_name, pruner_name, criterion_name, granularity_name)

    def analyze(
        self,
        model: nn.Module,
        pruning_rates: List[float],
        eval_fn: Callable[[nn.Module], float],
        metric_direction: str = "higher_is_better",
        calibration=None,
        config=None,
        baseline=None,
    ) -> SensitivityResult:
        adapter = self.engine.adapter_cls(model)
        probe_config = dict(config or {})
        probe_config['allow_noop'] = True  # small ratios may round to zero
        # Capability checks precede any probe/evaluation.
        probe_config['pruning_params'] = [(0, pruning_rates[0])] if pruning_rates else []
        self.engine.validate_compatibility(probe_config)
        if self.engine.pruner_cls.pruning_mode == 'depth':
            raise ValueError('Depth pruning does not support layer-wise sensitivity')
        pruneable_modules = [(target.name, target.module) for target in self.engine.targets(model)]
        snapshot = baseline or ModelSnapshot(model)

        baseline_score = eval_fn(snapshot.restore())
        logger.info(
            f"Sensitivity Analysis starting for {len(pruneable_modules)} layers across {len(pruning_rates)} rates. "
            f"Baseline metric: {baseline_score:.4f}"
        )

        profiles: Dict[int, LayerSensitivityProfile] = {}

        for idx, (layer_name, _) in enumerate(pruneable_modules):
            profile = LayerSensitivityProfile(layer_idx=idx, layer_name=layer_name)
            for rate in pruning_rates:
                model_copy = snapshot.restore()
                execution_config = {**probe_config, "pruning_params": [(idx, rate)]}
                if calibration:
                    execution_config.update(calibration)
                try:
                    self.engine.execute(model_copy, execution_config, verify_forward=False)
                    score = eval_fn(model_copy)
                    if metric_direction == "higher_is_better":
                        drop = baseline_score - score
                    else:
                        drop = score - baseline_score
                    rel_drop = drop / abs(baseline_score) if baseline_score != 0 else (float("inf") if drop > 0 else 0.0)

                    pt = SensitivityPoint(
                        rate=rate,
                        score=score,
                        metric_drop=drop,
                        relative_drop=rel_drop,
                        is_valid=True
                    )
                    logger.info(
                        f" - Layer {idx:2d} ({layer_name}) @ Rate {rate:.2f} -> "
                        f"Metric: {score:.4f} (Drop: {drop:+.4f}, RelDrop: {rel_drop:.2%})"
                    )
                except Exception as e:
                    pt = SensitivityPoint(
                        rate=rate,
                        score=0.0,
                        metric_drop=float("inf") if metric_direction == "higher_is_better" else float("-inf"),
                        relative_drop=1.0,
                        is_valid=False,
                        error_message=str(e)
                    )
                    logger.warning(
                        f" - Layer {idx:2d} ({layer_name}) @ Rate {rate:.2f} -> Failed evaluation: {e}"
                    )
                profile.points[rate] = pt
            profiles[idx] = profile

        metadata = {
            "model_name": self.model_name,
            "pruner_name": self.pruner_name,
            "criterion_name": self.criterion_name,
            "granularity_name": self.granularity_name,
            "metric_direction": metric_direction,
            "num_pruneable_layers": len(pruneable_modules),
            "rates_evaluated": list(pruning_rates)
        }

        return SensitivityResult(
            baseline_score=baseline_score,
            profiles=profiles,
            metadata=metadata
        )

