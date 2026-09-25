"""Adapter-defined layer and structural-block pruning."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Sequence

import torch

from prune_framework.contracts.targets import PruningGroup, PruningPlan, StructuralBlockTarget
from prune_framework.core.interfaces import BaseGranularity, BaseImportanceCriterion, BaseModelAdapter, BasePruner
from prune_framework.core.registry import register_pruner


@register_pruner("depth")
@register_pruner("layer")
@register_pruner("layer_depth")
@register_pruner("structural_block")
class DepthPruner(BasePruner):
    """Remove only block targets explicitly declared by an adapter.

    The pruner never infers how a model should be rewired. It selects targets,
    validates the complete operation on a deep-copied model, then delegates
    removal and repair to adapter hooks.
    """

    def create_plan(
        self,
        model_adapter: BaseModelAdapter,
        criterion: BaseImportanceCriterion,
        granularity: BaseGranularity | None = None,
        config: Dict[str, Any] | None = None,
    ) -> PruningPlan:
        del granularity
        config = config or {}
        targets = list(model_adapter.get_structural_block_targets())
        requested = self._select_targets(targets, model_adapter, criterion, config)
        return PruningPlan(
            pruner_name="structural_block",
            groups=[PruningGroup(primary=target, operation="remove_structural_block") for target in requested],
            metadata={
                "selection": "adapter_declared",
                "available_targets": [target.name for target in targets],
                "requested_targets": [target.name for target in requested],
            },
        )

    def validate_plan(
        self,
        plan: PruningPlan,
        model_adapter: BaseModelAdapter,
        config: Dict[str, Any] | None = None,
    ) -> bool:
        del config
        targets = [group.primary for group in plan.groups]
        if not all(isinstance(target, StructuralBlockTarget) for target in targets):
            for group in plan.groups:
                group.validated = False
                group.validation_error = "Structural block plan contains a non-structural target."
            return False
        errors = model_adapter.validate_structural_block_plan(targets)
        if not errors:
            dry_run_error = self._dry_run(model_adapter, targets)
            if dry_run_error is not None:
                errors = {target.name: dry_run_error for target in targets}
        for group in plan.groups:
            error = errors.get(group.primary.name)
            group.validated = error is None
            group.validation_error = error
            group.dependency_count = 0
            group.dependencies = [{
                "owner_name": group.primary.owner_name,
                "block_type": group.primary.block_type,
                "index": group.primary.index,
            }]
        return not errors

    def apply_plan(
        self,
        plan: PruningPlan,
        model_adapter: BaseModelAdapter,
        config: Dict[str, Any] | None = None,
    ) -> torch.nn.Module:
        del config
        if any(not group.validated for group in plan.groups):
            invalid = next(group.primary.name for group in plan.groups if not group.validated)
            raise RuntimeError(f"Cannot apply unvalidated structural block action for {invalid}.")
        targets = [group.primary for group in plan.groups]
        for target in model_adapter.order_structural_block_removals(targets):
            model_adapter.remove_structural_block(target)
        model_adapter.repair_structural_block_links(targets)
        invariant_error = model_adapter.validate_structural_block_invariants()
        if invariant_error is not None:
            raise RuntimeError(f"Structural block repair violated adapter invariants: {invariant_error}")
        return model_adapter.model

    @staticmethod
    def _select_targets(
        targets: Sequence[StructuralBlockTarget],
        model_adapter: BaseModelAdapter,
        criterion: BaseImportanceCriterion,
        config: Dict[str, Any],
    ) -> List[StructuralBlockTarget]:
        named = config.get("block_names")
        if named is not None:
            available = {target.name: target for target in targets}
            missing = [str(name) for name in named if str(name) not in available]
            if missing:
                raise ValueError(f"Adapter does not declare requested structural blocks: {', '.join(missing)}")
            return [available[str(name)] for name in named]

        amount = config.get("amount", config.get("pruning_params", config.get("drop_params", 0.0)))
        if not isinstance(amount, (int, float)):
            raise ValueError("Structural block pruning requires a numeric amount or block_names.")
        if not 0 <= float(amount) < 1:
            raise ValueError("Structural block pruning amount must be in [0, 1).")
        count = int(round(len(targets) * float(amount)))
        if count == 0:
            return []
        scored = [
            (float(model_adapter.score_structural_block(target, criterion).detach().cpu()), target.name, target)
            for target in targets
        ]
        return [target for _, _, target in sorted(scored, key=lambda item: (item[0], item[1]))[:count]]

    @staticmethod
    def _dry_run(model_adapter: BaseModelAdapter, targets: Sequence[StructuralBlockTarget]) -> str | None:
        """Prove the adapter removal plan forwards on a clone before mutation."""
        if not targets:
            return None
        try:
            clone = copy.deepcopy(model_adapter.model)
            clone_adapter = type(model_adapter)(clone)
            clone_targets = {target.name: target for target in clone_adapter.get_structural_block_targets()}
            remapped = []
            for target in targets:
                if target.name not in clone_targets:
                    return f"Dry-run adapter cannot resolve declared target '{target.name}'."
                remapped.append(clone_targets[target.name])
            clone_errors = clone_adapter.validate_structural_block_plan(remapped)
            if clone_errors:
                return next(iter(clone_errors.values()))
            for target in clone_adapter.order_structural_block_removals(remapped):
                clone_adapter.remove_structural_block(target)
            clone_adapter.repair_structural_block_links(remapped)
            invariant_error = clone_adapter.validate_structural_block_invariants()
            if invariant_error is not None:
                return invariant_error
            device = next(clone.parameters()).device
            with torch.no_grad():
                clone(clone_adapter.get_dummy_input(device))
            return None
        except Exception as exc:
            return f"Structural block dry-run failed: {exc}"
