"""Compatibility entry point for the unified pruning pipeline."""

from __future__ import annotations

from typing import Optional

from prune_framework.core.config import FrameworkConfig
from prune_framework.core.results import ExperimentResult, PruningResult
from .unified import EvaluationCallback, RecoveryCallback, UnifiedPruningPipeline


def run_unified_pruning_pipeline(
    config: FrameworkConfig,
    evaluator: Optional[EvaluationCallback] = None,
    recovery: Optional[RecoveryCallback] = None,
) -> ExperimentResult:
    return UnifiedPruningPipeline(config, evaluator=evaluator, recovery=recovery).run()


def run_pruning_pipeline(config: FrameworkConfig) -> PruningResult:
    """Run the unified pipeline while preserving the original return type."""
    return run_unified_pruning_pipeline(config).pruning
