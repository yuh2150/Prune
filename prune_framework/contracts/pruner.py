from abc import ABC, abstractmethod
import torch.nn as nn
from typing import Dict, Any
from .adapter import BaseModelAdapter
from .criterion import BaseImportanceCriterion
from .granularity import BaseGranularity
from .targets import PruningPlan


class BasePruner(ABC):
    """Abstract Base Class for Pruners."""

    pruning_mode = None
    supports_layerwise_policy = False

    def validate_config(self, criterion, config):
        mode = self.pruning_mode
        if mode and mode not in criterion.supported_pruning_modes:
            raise ValueError(f"Criterion {type(criterion).__name__} does not support pruning mode {mode}")
        params = config.get('pruning_params', config.get('amount', 0.3))
        if isinstance(params, (list, tuple)):
            if not self.supports_layerwise_policy:
                raise ValueError(f'{type(self).__name__} does not support layer-wise policy')
            if config.get('forced_channel_indices') is not None or config.get('forced_structural_indices') is not None:
                raise ValueError('Forced indices cannot override a layer-wise policy')
            if mode == 'unstructured' and (config.get('global_pruning') or getattr(criterion, 'global_selection', False)):
                raise ValueError('Global unstructured selection does not support layer-wise policy')
            seen = set()
            for index, rate in params:
                if not isinstance(index, int) or index < 0 or index in seen:
                    raise ValueError(f'Invalid or duplicated layer index: {index}')
                seen.add(index)
                if not 0 <= rate < 1:
                    raise ValueError(f'Invalid layer ratio: {rate}')
        elif not isinstance(params, (int, float)) or not 0 <= params < 1:
            raise ValueError(f'Invalid pruning ratio: {params}')

    def check_plan(self, plan, config):
        requested = config.get('pruning_params', config.get('amount', 0.3))
        positive = any(rate > 0 for _, rate in requested) if isinstance(requested, (list, tuple)) else requested > 0
        positive = positive or config.get('amount', 0) > 0
        if plan.is_empty and positive and not config.get('allow_noop', False):
            raise ValueError('Requested pruning has no executable actions; set allow_noop explicitly to permit this')
        seen = set()
        for action in plan.groups:
            name = action.primary.name
            if name in seen:
                raise ValueError(f'Duplicated pruning action for {name}')
            seen.add(name)

    def create_plan(
        self,
        model_adapter: BaseModelAdapter,
        criterion: BaseImportanceCriterion,
        granularity: BaseGranularity | None = None,
        config: Dict[str, Any] | None = None,
    ) -> PruningPlan:
        """Generate an inspectable plan without changing model parameters."""
        raise NotImplementedError(f"{type(self).__name__} does not implement plan generation.")

    def validate_plan(
        self,
        plan: PruningPlan,
        model_adapter: BaseModelAdapter,
        config: Dict[str, Any] | None = None,
    ) -> bool:
        """Validate all planned mutations before any mutation is applied."""
        return True

    def apply_plan(
        self,
        plan: PruningPlan,
        model_adapter: BaseModelAdapter,
        config: Dict[str, Any] | None = None,
    ) -> nn.Module:
        """Apply a previously validated plan."""
        raise NotImplementedError(f"{type(self).__name__} does not implement plan application.")

    def prune(
        self,
        model_adapter: BaseModelAdapter,
        criterion: BaseImportanceCriterion,
        granularity: BaseGranularity | None = None,
        config: Dict[str, Any] | None = None,
    ) -> nn.Module:
        """Compatibility wrapper for plan → validate → apply pruners."""
        config = config or {}
        self.validate_config(criterion, config)
        plan = self.create_plan(model_adapter, criterion, granularity, config)
        self.check_plan(plan, config)
        self.last_plan = plan
        if not self.validate_plan(plan, model_adapter, config):
            raise RuntimeError(f"{type(self).__name__} rejected its pruning plan.")
        return self.apply_plan(plan, model_adapter, config)
