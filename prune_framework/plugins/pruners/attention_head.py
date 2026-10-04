"""Adapter-driven masked multi-head-attention pruning."""

from __future__ import annotations

from typing import Any, Dict

import torch

from prune_framework.contracts.targets import AttentionHeadTarget, PruningGroup, PruningPlan
from prune_framework.core.interfaces import BaseImportanceCriterion, BaseModelAdapter, BasePruner
from prune_framework.core.registry import register_pruner


@register_pruner("attention_head")
@register_pruner("head")
class AttentionHeadPruner(BasePruner):
    """Mask complete attention heads; this does not physically shrink embed dim."""

    pruning_mode = "head"
    supports_layerwise_policy = False
    _MAGNITUDE_CRITERIA = {"magnitude", "l1", "l1_norm", "l2", "l2_norm"}

    def validate_config(self, criterion: BaseImportanceCriterion, config: Dict[str, Any]) -> None:
        amount = config.get("amount", 0.3)
        if not isinstance(amount, (int, float)) or not 0 <= float(amount) < 1:
            raise ValueError("Attention-head pruning amount must be in [0, 1).")
        if config.get("pruning_params") not in (None, amount):
            raise ValueError("Attention-head pruning does not support layer-wise policy.")
        name = str(config.get("criterion_name", type(criterion).__name__)).lower()
        if name not in self._MAGNITUDE_CRITERIA:
            raise ValueError("Attention-head pruning currently supports magnitude, L1, or L2 scoring only.")

    def create_plan(self, model_adapter: BaseModelAdapter, criterion: BaseImportanceCriterion, granularity=None,
                    config: Dict[str, Any] | None = None) -> PruningPlan:
        del granularity
        config = config or {}
        self.validate_config(criterion, config)
        amount = float(config.get("amount", 0.3))
        plan = PruningPlan(pruner_name="attention_head", metadata={"semantics": "masked_head", "physical_compression": False})
        for target in model_adapter.get_attention_head_targets():
            scores = model_adapter.score_attention_heads(target)
            if tuple(scores.shape) != (target.num_heads,):
                raise ValueError(f"Attention score for {target.name} must have one value per head.")
            count = int(round(target.num_heads * amount))
            count = min(count, target.num_heads - 1)
            if count:
                indices = torch.argsort(scores.detach(), stable=True)[:count].tolist()
                plan.groups.append(PruningGroup(primary=target, operation="mask_attention_head", indices=[int(i) for i in indices]))
        return plan

    def validate_plan(self, plan: PruningPlan, model_adapter: BaseModelAdapter, config=None) -> bool:
        del config
        valid = True
        for group in plan.groups:
            if not isinstance(group.primary, AttentionHeadTarget) or group.operation != "mask_attention_head":
                group.validated, group.validation_error, valid = False, "Invalid attention-head action.", False
                continue
            error = model_adapter.validate_attention_head_plan(group.primary, group.indices)
            group.validated = error is None
            group.validation_error = error
            group.dependency_count = 4
            group.dependencies = [{"semantics": "Q/K/V output slices + output-projection input slices"}]
            valid = valid and group.validated
        return valid

    def apply_plan(self, plan: PruningPlan, model_adapter: BaseModelAdapter, config=None):
        del config
        for group in plan.groups:
            if not group.validated:
                raise RuntimeError(f"Cannot apply invalid attention-head action for {group.primary.name}.")
            model_adapter.apply_attention_head_mask(group.primary, group.indices)
        return model_adapter.model
