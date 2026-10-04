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

    def targets(self, model):
        adapter, pruner, criterion = self.adapter_cls(model), self.pruner_cls(), self.criterion_cls()
        if pruner.pruning_mode == "structured":
            return pruner._structural_targets(adapter, criterion)
        if pruner.pruning_mode in {"unstructured", "nm", "block_sparse"}:
            # Constrained masks share the adapter-approved weight target
            # boundary with unstructured pruning; they do not expose their own
            # broader module discovery policy.
            from prune_framework.plugins.pruners.unstructured import UnstructuredPruner
            return UnstructuredPruner._weight_targets(adapter)
        if pruner.pruning_mode == "head":
            return adapter.get_attention_head_targets()
        return adapter.get_structural_block_targets()

    def validate_compatibility(self, config):
        self.pruner_cls().validate_config(self.criterion_cls(), config)

    def validate_model_compatibility(self, model, config):
        self.validate_compatibility(config)
        if self.pruner_cls.pruning_mode in {'depth', 'head'}:
            return
        criterion = self.criterion_cls()
        adapter = self.adapter_cls(model)
        targets = self.targets(model)
        if not targets and config.get('amount', .3) > 0 and not config.get('allow_noop', False):
            raise ValueError('Requested pruning has no executable actions: adapter exposes no compatible targets')
        for target in targets:
            module = target.module
            if criterion.uses_bn_wrapper:
                module = adapter.get_importance_module(target.name, module)
                valid = isinstance(module, nn.BatchNorm2d) or isinstance(getattr(module, 'bn', None), nn.BatchNorm2d)
            else:
                valid = isinstance(module, criterion.supported_module_types)
            if not valid:
                raise ValueError(f'Criterion {type(criterion).__name__} does not support pruning mode '
                                 f'{self.pruner_cls.pruning_mode} for module type {type(target.module).__name__}'
                                 + (' (requires BatchNorm2d ownership)' if criterion.uses_bn_wrapper else ''))

    def build_plan(self, model, config):
        config = dict(config or {})
        config.setdefault("criterion_name", self.criterion_name)
        self.validate_model_compatibility(model, config)
        adapter, pruner = self.adapter_cls(model), self.pruner_cls()
        criterion = self.criterion_cls()
        pruner.validate_config(criterion, config)
        adapter.prepare_for_pruning()
        plan = pruner.create_plan(adapter, criterion, self.granularity_cls(), config)
        pruner.check_plan(plan, config)
        if not pruner.validate_plan(plan, adapter, config):
            errors = [g.validation_error for g in plan.groups if not g.validated]
            raise ValueError(f"Invalid pruning plan: {errors}")
        plan.metadata["criterion"] = self.criterion_name
        return plan

    def execute(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        verify_forward: bool = True,
        plan=None,
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
        pruner.validate_config(criterion, config)
        plan = plan if plan is not None else self.build_plan(model, config)
        pruner.check_plan(plan, config)
        modules = dict(model.named_modules())
        if any(modules.get(group.primary.name) is not group.primary.module for group in plan.groups):
            raise ValueError('Plan targets do not belong to this model; rebuild the plan')
        pruner.last_plan = plan
        pruned_model = pruner.apply_plan(plan, adapter, config)

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
        from prune_framework.plugins.pruners.structured import StructuredPruner
        return StructuredPruner._build_dependency_graph(self.adapter_cls(model))
