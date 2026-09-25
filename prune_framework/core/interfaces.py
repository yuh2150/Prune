"""
Legacy interfaces module.
All abstract contracts have been migrated to `prune_framework.contracts`.
Re-exported here for backward compatibility.
"""
from prune_framework.contracts import (
    BaseModelAdapter,
    BasePruner,
    BaseImportanceCriterion,
    BaseGranularity,
    BaseSelector,
    ChannelSparsityTarget,
    StructuralBlockTarget,
    PrunableTarget,
    PruningGroup,
    PruningPlan,
    TargetType,
)

__all__ = [
    "BaseModelAdapter",
    "BasePruner",
    "BaseImportanceCriterion",
    "BaseGranularity",
    "BaseSelector",
    "ChannelSparsityTarget",
    "StructuralBlockTarget",
    "PrunableTarget",
    "PruningGroup",
    "PruningPlan",
    "TargetType",
]
