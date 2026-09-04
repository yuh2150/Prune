import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from .registry import PluginRegistry
from .interfaces import BaseModelAdapter, BaseImportanceCriterion, BaseGranularity, BasePruner
from .results import PruningResult
from .logging import logger


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class PruningEngine:
    """
    Pure Pipeline Orchestrator.
    Dynamically composes ModelAdapter, Pruner, Criterion, and Granularity plugins
    to execute pruning without any hardcoded model or technique conditionals.
    """

    def __init__(
        self,
        model_name: str,
        pruner_name: str,
        criterion_name: str,
        granularity_name: str = "channel"
    ):
        self.model_name = model_name
        self.pruner_name = pruner_name
        self.criterion_name = criterion_name
        self.granularity_name = granularity_name

        self.adapter_cls = PluginRegistry.get_model_adapter(model_name)
        self.pruner_cls = PluginRegistry.get_pruner(pruner_name)
        self.criterion_cls = PluginRegistry.get_criterion(criterion_name)
        self.granularity_cls = PluginRegistry.get_granularity(granularity_name)

    def execute(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        verify_forward: bool = True
    ) -> PruningResult:
        """
        Executes end-to-end pruning.
        """
        adapter: BaseModelAdapter = self.adapter_cls(model)
        pruner: BasePruner = self.pruner_cls()
        criterion: BaseImportanceCriterion = self.criterion_cls()
        granularity: BaseGranularity = self.granularity_cls()

        logger.info("Executing Pruning Pipeline (Composition)")
        logger.info(f" - Model Adapter:     {adapter.__class__.__name__}")
        logger.info(f" - Pruner Strategy:   {pruner.__class__.__name__}")
        logger.info(f" - Importance Metric: {criterion.__class__.__name__}")
        logger.info(f" - Granularity:       {granularity.__class__.__name__}")

        adapter.prepare_for_pruning()
        params_before = count_parameters(model)

        # Execute Pruning
        pruned_model = pruner.prune(adapter, criterion, granularity, config)

        adapter.post_prune_cleanup()
        params_after = count_parameters(pruned_model)

        params_removed = params_before - params_after
        pct_removed = (params_removed / params_before * 100.0) if params_before > 0 else 0.0

        forward_ok = False
        if verify_forward:
            device = next(pruned_model.parameters()).device
            try:
                dummy = adapter.get_dummy_input(device)
                pruned_model.eval()
                with torch.no_grad():
                    _ = pruned_model(dummy)
                forward_ok = True
                logger.info("Forward pass shape verification: SUCCESS ✅")
            except Exception as e:
                logger.warning(f"Forward pass shape verification WARNING: {e} ⚠️")

        logger.info("Pruning Completed Statistics:")
        logger.info(f" - Parameters Before: {params_before:,}")
        logger.info(f" - Parameters After:  {params_after:,}")
        logger.info(f" - Reduction:         {params_removed:,} ({pct_removed:.2f}%)")

        return PruningResult(
            model_name=self.model_name,
            pruner_name=self.pruner_name,
            criterion_name=self.criterion_name,
            granularity_name=self.granularity_name,
            params_before=params_before,
            params_after=params_after,
            params_reduction_pct=pct_removed,
            forward_verified=forward_ok
        )
