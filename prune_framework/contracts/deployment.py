"""Deployment constraints are independent of requested pruning sparsity."""
from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class DeploymentTargets:
    parameter_reduction: Optional[float] = None
    flops_reduction: Optional[float] = None
    max_map50_drop: Optional[float] = None
    max_map50_95_drop: Optional[float] = None
    max_latency_ms: Optional[float] = None
    max_memory_mb: Optional[float] = None

    def validate(self):
        for key, value in vars(self).items():
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f'targets.{key} must be finite and non-negative')
            if value is not None and ('reduction' in key or 'drop' in key) and value > 1:
                raise ValueError(f'targets.{key} must be in [0, 1]')


@dataclass
class TargetCheckResult:
    reached: bool
    violations: list[str] = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)


class TargetChecker:
    @staticmethod
    def check(baseline, current, targets, baseline_cost=None, current_cost=None):
        targets.validate()
        baseline = getattr(baseline, 'metrics', baseline) or {}
        current = getattr(current, 'metrics', current) or {}
        violations, unavailable = [], []
        def compare(name, measured, limit, lower=True):
            if limit is None:
                return
            if measured is None or not math.isfinite(measured):
                unavailable.append(name)
            elif (measured > limit + 1e-12 if lower else measured + 1e-12 < limit):
                violations.append(f'{name}: measured={measured:g}, target={limit:g}')
        for name, field_name in [('parameter_reduction', 'params'), ('flops_reduction', 'flops')]:
            before = getattr(baseline_cost, field_name, None)
            after = getattr(current_cost, field_name, None)
            reduction = 1 - after / before if before is not None and before > 0 and after is not None else None
            compare(name, reduction, getattr(targets, name), lower=False)
        for key, target in [('map50', 'max_map50_drop'), ('map50_95', 'max_map50_95_drop')]:
            before = baseline.get(key, baseline.get('map') if key == 'map50_95' else None)
            after = current.get(key, current.get('map') if key == 'map50_95' else None)
            compare(target, before - after if before is not None and after is not None else None, getattr(targets, target))
        compare('max_latency_ms', current.get('total_ms'), targets.max_latency_ms)
        # Peak memory is an explicit measurement, never parameter storage size.
        compare('max_memory_mb', current.get('memory_mb'), targets.max_memory_mb)
        return TargetCheckResult(not violations and not unavailable, violations, unavailable)
