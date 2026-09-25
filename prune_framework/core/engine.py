import copy
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from .registry import PluginRegistry
from .interfaces import BaseModelAdapter, BaseImportanceCriterion, BaseGranularity, BasePruner
from .results import PruningResult
from .logging import logger
from prune_framework.modules.evaluation.validator import ModelValidator


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
        config = dict(config or {})
        config.setdefault("criterion_name", self.criterion_name)
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

        validation = None
        forward_ok = False
        if verify_forward:
            device = next(pruned_model.parameters()).device
            dummy = adapter.get_dummy_input(device)
            validation = ModelValidator.validate_architecture(
                pruned_model, dummy, params_before=params_before, verify_checkpoint=True
            )
            # Preserve the legacy meaning of ``forward_verified``. Checkpoint
            # round-trip is recorded as a separate architecture signal because
            # third-party detector objects are not always deepcopy-safe.
            forward_ok = validation.forward_verified and validation.parameter_count_verified
            if forward_ok:
                logger.info("Forward pass shape verification: SUCCESS ✅")
            if not validation.valid:
                logger.warning(f"Architecture validation WARNING: {validation.errors} ⚠️")

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
            forward_verified=forward_ok,
            pruning_plan=getattr(pruner, "last_plan", None).describe() if getattr(pruner, "last_plan", None) else {},
            architecture_validation=validation.describe() if validation is not None else {},
        )

    def build_dependency_graph(self, model: nn.Module):
        """Trace the current model for a structured-pruning safety preflight.

        The graph must be rebuilt after each physical pruning operation because
        module dimensions and graph edges change. This method deliberately does
        not cache a graph across pruning steps.
        """
        try:
            import torch_pruning as tp
        except ImportError as exc:
            raise RuntimeError("Structured pruning requires the 'torch_pruning' package.") from exc

        adapter: BaseModelAdapter = self.adapter_cls(model)
        device = next(model.parameters()).device
        grad_states = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
        try:
            for parameter in model.parameters():
                parameter.requires_grad = True
            return tp.DependencyGraph().build_dependency(model, adapter.get_dummy_input(device))
        finally:
            for name, parameter in model.named_parameters():
                parameter.requires_grad = grad_states[name]
