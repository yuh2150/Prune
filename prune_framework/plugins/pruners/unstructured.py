import torch
import torch.nn as nn
from typing import Dict, Any, List

from prune_framework.contracts.targets import PrunableTarget, PruningGroup, PruningPlan, TargetType
from prune_framework.core.interfaces import BasePruner, BaseModelAdapter, BaseImportanceCriterion
from prune_framework.core.registry import register_pruner
from prune_framework.modules.analysis.global_selection import GlobalScoreSelector
from prune_framework.modules.model.masks import MaskManager


@register_pruner("unstructured")
@register_pruner("unstructured_weight")
class UnstructuredPruner(BasePruner):
    """Plan-based persistent element-wise pruning for Conv2d and Linear."""

    def create_plan(self, model_adapter: BaseModelAdapter, criterion: BaseImportanceCriterion,
                    granularity=None, config: Dict[str, Any] = None) -> PruningPlan:
        config = config or {}
        targets = self._weight_targets(model_adapter)
        pruning_params = config.get("pruning_params", config.get("amount", 0.3))
        if self._uses_global_selection(criterion, config):
            return self._create_global_plan(targets, criterion, config)
        selections = [(index, pruning_params) for index in range(len(targets))] if isinstance(pruning_params, float) else list(pruning_params or [])
        plan = PruningPlan(pruner_name="unstructured", metadata={"criterion": config.get("criterion_name")})
        for target_index, rate in selections:
            if target_index < 0 or target_index >= len(targets) or not 0 < rate < 1:
                continue
            target = targets[target_index]
            scores = self._weight_scores(target, criterion, config)
            count = min(max(int(round(target.module.weight.numel() * float(rate))), 0), target.module.weight.numel())
            if count == 0:
                continue
            indices = torch.argsort(scores.flatten(), stable=True)[:count].tolist()
            plan.groups.append(PruningGroup(
                primary=target, operation="mask_weight", indices=[int(index) for index in indices]
            ))
        return plan

    def _create_global_plan(
        self,
        targets: List[PrunableTarget],
        criterion: BaseImportanceCriterion,
        config: Dict[str, Any],
    ) -> PruningPlan:
        amount = config.get("amount", config.get("pruning_params", 0.3))
        if not isinstance(amount, (int, float)):
            raise ValueError("Global element-wise pruning requires one numeric amount.")
        score_items = [(target, self._weight_scores(target, criterion, config)) for target in targets]
        selection = GlobalScoreSelector.select_lowest(score_items, float(amount))
        plan = PruningPlan(
            pruner_name="unstructured",
            metadata={
                "criterion": config.get("criterion_name"),
                "selection": "global",
                "requested_amount": float(amount),
                "total_elements": selection.total_elements,
                "selected_elements": selection.selected_elements,
            },
        )
        for target in targets:
            indices = selection.indices.get(target.name, [])
            if indices:
                plan.groups.append(PruningGroup(primary=target, operation="mask_weight", indices=indices))
        return plan

    def validate_plan(self, plan: PruningPlan, model_adapter: BaseModelAdapter,
                      config: Dict[str, Any] = None) -> bool:
        all_valid = True
        for action in plan.groups:
            module = action.primary.module
            valid = (
                action.primary.is_weight and isinstance(module, (nn.Conv2d, nn.Linear)) and bool(action.indices)
                and len(action.indices) <= module.weight.numel() and min(action.indices) >= 0
                and max(action.indices) < module.weight.numel() and len(set(action.indices)) == len(action.indices)
            )
            action.validated = valid
            action.dependency_count = 1
            if not valid:
                action.validation_error = "Invalid element-wise mask indices."
                all_valid = False
        return all_valid

    def apply_plan(self, plan: PruningPlan, model_adapter: BaseModelAdapter,
                   config: Dict[str, Any] = None) -> nn.Module:
        for action in plan.groups:
            if not action.validated:
                raise RuntimeError(f"Cannot apply unvalidated mask action for {action.primary.name}.")
            module = action.primary.module
            mask = torch.ones_like(module.weight)
            mask.flatten()[action.indices] = 0
            MaskManager.apply(module, mask)
            print(f" - Unstructured Pruner: {action.primary.name} masked {len(action.indices)}/{module.weight.numel()} weights.")
        optimizer = (config or {}).get("optimizer")
        if optimizer is not None:
            MaskManager.attach_optimizer(model_adapter.model, optimizer)
        return model_adapter.model

    @staticmethod
    def _weight_targets(model_adapter: BaseModelAdapter) -> List[PrunableTarget]:
        if hasattr(model_adapter, "get_prunable_targets"):
            return list(model_adapter.get_prunable_targets({TargetType.CONV_WEIGHT, TargetType.LINEAR_WEIGHT}))
        return [PrunableTarget(name, module, TargetType.CONV_WEIGHT)
                for name, module in model_adapter.get_pruneable_modules() if isinstance(module, nn.Conv2d)]

    @staticmethod
    def _weight_scores(target: PrunableTarget, criterion: BaseImportanceCriterion,
                       config: Dict[str, Any]) -> torch.Tensor:
        if (
            getattr(criterion, "requires_gradients", False)
            or getattr(criterion, "requires_synflow_calibration", False)
            or getattr(criterion, "requires_higher_order_calibration", False)
        ):
            calibration_key = (
                "synflow_calibration" if getattr(criterion, "requires_synflow_calibration", False)
                else "higher_order_calibration" if getattr(criterion, "requires_higher_order_calibration", False)
                else "gradient_calibration"
            )
            calibration = config.get(calibration_key)
            if calibration is None:
                raise RuntimeError(f"This unstructured criterion requires {calibration_key}.")
            scores = criterion.score(target.module, calibration.context_for(target))
            if tuple(scores.shape) != tuple(target.module.weight.shape):
                raise ValueError(
                    f"{criterion.__class__.__name__} must return element-wise scores for {target.name}."
                )
            return scores.detach()
        if getattr(criterion, "provides_elementwise_scores", False):
            scores = criterion.score(target.module)
            if tuple(scores.shape) != tuple(target.module.weight.shape):
                raise ValueError(
                    f"{criterion.__class__.__name__} must return element-wise scores for {target.name}."
                )
            return scores.detach()
        criterion_name = str(config.get("criterion_name", criterion.__class__.__name__)).lower()
        if "random" in criterion_name:
            return torch.rand_like(target.module.weight)
        # Element-wise magnitude is the P0 interpretation of L1, L2, and
        # Magnitude criteria. Structured criteria are not silently reinterpreted.
        if not any(name in criterion_name for name in ("magnitude", "l1", "l2", "norm")):
            raise ValueError(f"{criterion.__class__.__name__} does not provide element-wise scores for unstructured pruning.")
        return target.module.weight.detach().abs()

    @staticmethod
    def _uses_global_selection(criterion: BaseImportanceCriterion, config: Dict[str, Any]) -> bool:
        return bool(config.get("global_pruning", False) or getattr(criterion, "global_selection", False))
