"""Plan-based persistent dense fallback implementations for sparse patterns."""
from __future__ import annotations
from typing import Any, Dict, List
import torch.nn as nn
from prune_framework.contracts.targets import PrunableTarget, PruningGroup, PruningPlan, TargetType
from prune_framework.core.interfaces import BaseModelAdapter, BasePruner
from prune_framework.core.registry import register_pruner
from prune_framework.modules.model.masks import MaskManager
from prune_framework.modules.model.pattern_masks import PatternMask


class _PatternPruner(BasePruner):
    kind = "pattern"
    def _targets(self, adapter: BaseModelAdapter) -> List[PrunableTarget]:
        return list(adapter.get_prunable_targets({TargetType.CONV_WEIGHT, TargetType.LINEAR_WEIGHT}))
    def validate_plan(self, plan, model_adapter, config=None):
        valid = True
        for group in plan.groups:
            mask = group.dependencies[0].get("mask") if group.dependencies else None
            ok = isinstance(group.primary, PrunableTarget) and isinstance(group.primary.module, (nn.Conv2d, nn.Linear)) and mask is not None and tuple(mask.shape) == tuple(group.primary.module.weight.shape)
            group.validated, group.dependency_count = ok, 1
            if not ok: group.validation_error, valid = "Invalid sparse pattern mask.", False
        return valid
    def apply_plan(self, plan, model_adapter, config=None):
        for group in plan.groups:
            if not group.validated: raise RuntimeError(f"Cannot apply unvalidated sparse plan for {group.primary.name}.")
            MaskManager.apply(group.primary.module, group.dependencies[0]["mask"])
        if (config or {}).get("optimizer") is not None: MaskManager.attach_optimizer(model_adapter.model, config["optimizer"])
        return model_adapter.model


@register_pruner("nm_sparsity")
@register_pruner("n_sparsity")
class NMSparsityPruner(_PatternPruner):
    kind = "nm"
    def create_plan(self, model_adapter, criterion=None, granularity=None, config: Dict[str, Any] | None=None):
        config = config or {}; n, m = config.get("n"), config.get("m")
        plan = PruningPlan(pruner_name="nm_sparsity", metadata={"n": n, "m": m, "inference": "dense_fallback"})
        for target in self._targets(model_adapter):
            pattern = PatternMask.nm(target.module.weight, n, m)
            plan.groups.append(PruningGroup(primary=target, operation="mask_nm", dependencies=[{"mask": pattern.mask, "layout": pattern.layout}]))
        return plan


@register_pruner("block_sparse")
class BlockSparsePruner(_PatternPruner):
    kind = "block_sparse"
    def create_plan(self, model_adapter, criterion=None, granularity=None, config: Dict[str, Any] | None=None):
        config = config or {}; size = tuple(config.get("block_size", (4, 4))); amount = float(config.get("amount", 0.5))
        plan = PruningPlan(pruner_name="block_sparse", metadata={"block_size": list(size), "amount": amount, "inference": "dense_fallback"})
        for target in self._targets(model_adapter):
            pattern = PatternMask.blocks(target.module.weight, size, amount)
            plan.groups.append(PruningGroup(primary=target, operation="mask_blocks", dependencies=[{"mask": pattern.mask, "layout": pattern.layout}]))
        return plan
