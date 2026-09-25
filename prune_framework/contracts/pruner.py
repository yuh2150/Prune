from abc import ABC, abstractmethod
import torch.nn as nn
from typing import Dict, Any
from .adapter import BaseModelAdapter
from .criterion import BaseImportanceCriterion
from .granularity import BaseGranularity
from .targets import PruningPlan


class BasePruner(ABC):
    """Abstract Base Class for Pruners."""

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
        plan = self.create_plan(model_adapter, criterion, granularity, config)
        self.last_plan = plan
        if not self.validate_plan(plan, model_adapter, config):
            raise RuntimeError(f"{type(self).__name__} rejected its pruning plan.")
        return self.apply_plan(plan, model_adapter, config)
