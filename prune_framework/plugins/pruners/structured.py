import torch
import torch.nn as nn
import torch_pruning as tp
from typing import Dict, Any, List, Tuple

from prune_framework.contracts.targets import PrunableTarget, PruningGroup, PruningPlan, TargetType
from prune_framework.core.interfaces import BasePruner, BaseModelAdapter, BaseImportanceCriterion, BaseGranularity
from prune_framework.core.registry import register_pruner
from prune_framework.core.utils import make_divisible
from .structural_operations import STRUCTURAL_OPERATIONS, StructuralOperation, get_structural_operation


@register_pruner("structured")
@register_pruner("structured_channel")
class StructuredPruner(BasePruner):
    """Dependency-graph validated structural output-dimension pruning.

    Every physical mutation is selected from :mod:`structural_operations` by
    target type.  This deliberately covers only Conv2d output channels and
    Linear output features.  Adapters remain the authority on which concrete
    modules may become roots in a particular model family.
    """

    def create_plan(self, model_adapter: BaseModelAdapter, criterion: BaseImportanceCriterion,
                    granularity: BaseGranularity = None, config: Dict[str, Any] = None) -> PruningPlan:
        config = config or {}
        targets = self._structural_targets(model_adapter, criterion)
        pruning_params = config.get("pruning_params", config.get("amount", 0.3))
        min_features = int(config.get("min_features", config.get("min_channels", 2)))
        forced_indices = config.get("forced_channel_indices", {}) or {}
        forced_indices = {**forced_indices, **(config.get("forced_structural_indices", {}) or {})}
        use_forced_indices = "forced_channel_indices" in config
        use_forced_indices = use_forced_indices or "forced_structural_indices" in config
        plan = PruningPlan(
            pruner_name="structured",
            metadata={
                # Keep the legacy key in artifacts/config consumers while
                # describing the generic output-dimension constraint as well.
                "min_channels": min_features,
                "min_features": min_features,
                "global_pruning": bool(config.get("global_pruning", False)),
            },
        )

        global_indices: Dict[int, List[int]] = {}
        if use_forced_indices:
            selections = [
                (index, 0.0) for index, target in enumerate(targets) if target.name in forced_indices
            ]
        elif config.get("global_pruning", False) and isinstance(pruning_params, float):
            candidates: List[Tuple[float, int, int]] = []
            caps: Dict[int, int] = {}
            total = 0
            for index, target in enumerate(targets):
                scores = self._scores(model_adapter, criterion, target, config)
                total += self._output_size(target)
                caps[index] = max(0, self._output_size(target) - min_features)
                candidates.extend((float(score), index, channel) for channel, score in enumerate(scores.detach().cpu().tolist()))
            requested = min(int(total * pruning_params), sum(caps.values()))
            selected = {index: 0 for index in caps}
            for _score, index, channel in sorted(candidates):
                if sum(selected.values()) >= requested:
                    break
                if selected[index] >= caps[index]:
                    continue
                global_indices.setdefault(index, []).append(channel)
                selected[index] += 1
            selections = [(index, 0.0) for index in sorted(global_indices)]
        elif isinstance(pruning_params, float):
            selections = [(index, pruning_params) for index in range(len(targets))]
        else:
            selections = list(pruning_params or [])

        for target_index, rate in sorted(selections, key=lambda value: value[0]):
            if target_index < 0 or target_index >= len(targets):
                continue
            target = targets[target_index]
            if target.name in forced_indices:
                indices = [int(index) for index in forced_indices[target.name]]
            elif target_index in global_indices:
                indices = global_indices[target_index]
            else:
                scores = self._scores(model_adapter, criterion, target, config)
                if granularity is not None and hasattr(granularity, "extract_indices"):
                    indices = granularity.extract_indices(scores, rate, target.module)
                else:
                    count = make_divisible(rate * self._output_size(target), 2)
                    count = min(count, max(0, self._output_size(target) - min_features))
                    indices = torch.argsort(scores)[:count].tolist() if count > 0 else []
                indices = indices[: max(0, self._output_size(target) - min_features)]
            if indices:
                operation = self._operation_for(target)
                if operation is None:
                    # This is normally prevented by _structural_targets, but
                    # keeps manually constructed plans inspectable/rejectable.
                    continue
                plan.groups.append(PruningGroup(
                    primary=target,
                    operation=operation.plan_operation,
                    indices=sorted(set(int(index) for index in indices)),
                ))
        return plan

    def validate_plan(self, plan: PruningPlan, model_adapter: BaseModelAdapter,
                      config: Dict[str, Any] = None) -> bool:
        """Check every planned dependency group before the first mutation."""
        all_valid = True
        for action in plan.groups:
            try:
                operation = self._operation_for(action.primary)
                if operation is None:
                    action.validated = False
                    action.validation_error = (
                        f"Unsupported structural target type '{action.primary.target_type.value}'."
                    )
                    all_valid = False
                    continue
                output_size = self._output_size(action.primary)
                if not action.indices or min(action.indices) < 0 or max(action.indices) >= output_size:
                    action.validated = False
                    action.validation_error = "Structural pruning indices are outside the target output dimension."
                    all_valid = False
                    continue
                dependency_graph = self._build_dependency_graph(model_adapter)
                group = dependency_graph.get_pruning_group(
                    action.primary.module, operation.pruning_fn, action.indices
                )
                action.dependency_count = len(group)
                action.dependencies = self._describe_dependencies(group, model_adapter.model)
                action.validated = bool(dependency_graph.check_pruning_group(group))
                if not action.validated:
                    action.validation_error = "torch_pruning rejected the dependency group"
                    all_valid = False
            except Exception as exc:
                action.validated = False
                action.validation_error = str(exc)
                all_valid = False
        return all_valid

    def apply_plan(self, plan: PruningPlan, model_adapter: BaseModelAdapter,
                   config: Dict[str, Any] = None) -> nn.Module:
        if any(not action.validated for action in plan.groups):
            invalid = next(action.primary.name for action in plan.groups if not action.validated)
            raise RuntimeError(f"Cannot apply unvalidated structured action for {invalid}.")
        gate_controller = (config or {}).get("l0_gate_controller")
        if gate_controller is not None:
            # Gate hooks must remain live through dependency validation so the
            # plan is inspectable; remove them just before dimension mutation.
            gate_controller.remove_train_time_state()
        for action in plan.groups:
            # A dependency graph becomes stale after every structural mutation.
            operation = self._operation_for(action.primary)
            if operation is None:
                raise RuntimeError(f"No structural operation is registered for {action.primary.target_type.value}.")
            dependency_graph = self._build_dependency_graph(model_adapter)
            group = dependency_graph.get_pruning_group(
                action.primary.module, operation.pruning_fn, action.indices
            )
            if not dependency_graph.check_pruning_group(group):
                raise RuntimeError(f"Dependency group became invalid for {action.primary.name}.")
            before = self._output_size(action.primary)
            group.prune()
            print(
                f" - Structured Pruner: {action.primary.name} pruned {len(action.indices)}/{before} "
                f"{operation.dimension_name}. New output size={self._output_size(action.primary)}"
            )
        return model_adapter.model

    @staticmethod
    def _structural_targets(
        model_adapter: BaseModelAdapter, criterion: BaseImportanceCriterion
    ) -> List[PrunableTarget]:
        allowed = set(getattr(criterion, "calibration_target_types", STRUCTURAL_OPERATIONS))
        allowed &= set(STRUCTURAL_OPERATIONS)
        if hasattr(model_adapter, "get_prunable_targets"):
            return list(model_adapter.get_prunable_targets(allowed))
        return [PrunableTarget(name, module, TargetType.CONV_OUT_CHANNEL)
                for name, module in model_adapter.get_pruneable_modules() if isinstance(module, nn.Conv2d)]

    @staticmethod
    def _operation_for(target: PrunableTarget) -> StructuralOperation | None:
        operation = get_structural_operation(target.target_type)
        return operation if operation is not None and operation.supports(target.module) else None

    @staticmethod
    def _output_size(target: PrunableTarget) -> int:
        if target.output_size is None:
            raise ValueError(f"{target.target_type.value} has no structural output dimension.")
        return int(target.output_size)

    @staticmethod
    def _scores(model_adapter: BaseModelAdapter, criterion: BaseImportanceCriterion,
                target: PrunableTarget, config: Dict[str, Any] = None) -> torch.Tensor:
        config = config or {}
        calibration = config.get("gradient_calibration")
        if getattr(criterion, "requires_gradients", False):
            # Taylor must score the actual target weight. YOLO's Conv-BN wrapper
            # is still used for BN criteria, but has no direct weight gradient.
            context = calibration.context_for(target) if calibration is not None else None
            scores = criterion.score(target.module, context=context)
        else:
            importance_module = model_adapter.get_importance_module(target.name, target.module)
            scores = criterion.score(importance_module) if hasattr(criterion, "score") else criterion.compute_scores(importance_module)
        output_size = StructuredPruner._output_size(target)
        if scores.numel() != output_size:
            raise ValueError(
                f"Criterion returned {scores.numel()} scores for {target.name}, "
                f"which has {output_size} structural output features."
            )
        return scores

    @staticmethod
    def _build_dependency_graph(model_adapter: BaseModelAdapter) -> tp.DependencyGraph:
        model = model_adapter.model
        device = next(model.parameters()).device
        grad_states = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
        try:
            for parameter in model.parameters():
                parameter.requires_grad = True
            with torch.enable_grad():
                return tp.DependencyGraph().build_dependency(model, model_adapter.get_dummy_input(device))
        finally:
            for name, parameter in model.named_parameters():
                parameter.requires_grad = grad_states[name]

    @staticmethod
    def _describe_dependencies(group, model: nn.Module) -> List[Dict[str, Any]]:
        """Serialize torch-pruning edges without retaining its stale graph."""
        module_names = {id(module): name for name, module in model.named_modules()}
        dependencies: List[Dict[str, Any]] = []
        for item in group.items:
            dependency = item.dep
            target_module = dependency.target.module
            dependencies.append({
                "target_name": module_names.get(id(target_module), type(target_module).__name__),
                "target_type": type(target_module).__name__,
                "operation": str(dependency.pruning_fn),
                "indices": [int(index) for index in item.idxs],
            })
        return dependencies
