"""Reusable descriptions of model parameters that pruning may affect.

The objects in this module deliberately keep model-family policy out of
pruners.  Adapters decide which modules are safe to expose, while pruners
operate on typed targets and inspectable plans.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

import torch.nn as nn


class TargetType(str, Enum):
    """The supported P0 target kinds."""

    CONV_WEIGHT = "conv_weight"
    CONV_OUT_CHANNEL = "conv_out_channel"
    LINEAR_WEIGHT = "linear_weight"
    LINEAR_OUT_FEATURE = "linear_out_feature"


_EXPECTED_MODULES = {
    TargetType.CONV_WEIGHT: nn.Conv2d,
    TargetType.CONV_OUT_CHANNEL: nn.Conv2d,
    TargetType.LINEAR_WEIGHT: nn.Linear,
    TargetType.LINEAR_OUT_FEATURE: nn.Linear,
}


@dataclass(frozen=True)
class PrunableTarget:
    """A typed parameter or structural feature exposed by a model adapter."""

    name: str
    module: nn.Module = field(repr=False, compare=False)
    target_type: TargetType = TargetType.CONV_WEIGHT
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        expected = _EXPECTED_MODULES[self.target_type]
        if not isinstance(self.module, expected):
            raise TypeError(
                f"{self.target_type.value} requires {expected.__name__}, got {type(self.module).__name__}."
            )

    @property
    def is_weight(self) -> bool:
        return self.target_type in {TargetType.CONV_WEIGHT, TargetType.LINEAR_WEIGHT}

    @property
    def is_output_feature(self) -> bool:
        return self.target_type in {TargetType.CONV_OUT_CHANNEL, TargetType.LINEAR_OUT_FEATURE}

    @property
    def output_size(self) -> Optional[int]:
        if self.target_type is TargetType.CONV_OUT_CHANNEL:
            return self.module.out_channels
        if self.target_type is TargetType.LINEAR_OUT_FEATURE:
            return self.module.out_features
        return None

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "target_type": self.target_type.value,
            "module_type": type(self.module).__name__,
            "weight_shape": list(self.module.weight.shape),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ChannelSparsityTarget:
    """A trainable scalar vector that maps one-to-one to Conv output channels.

    Adapters create these mappings.  A regularizer therefore never needs to
    infer architecture policy from module names, while the structured pruner
    continues to mutate the associated :class:`PrunableTarget` through its
    dependency graph.
    """

    name: str
    channel_target: PrunableTarget = field(repr=False, compare=False)
    parameter: nn.Parameter = field(repr=False, compare=False)
    module: nn.Module = field(repr=False, compare=False)
    source: str = "batchnorm_scale"

    def __post_init__(self) -> None:
        if not self.channel_target.is_output_feature or not isinstance(self.channel_target.module, nn.Conv2d):
            raise TypeError("Channel sparsity targets must map to a Conv2d output-channel target.")
        if self.parameter.ndim != 1 or self.parameter.numel() != self.channel_target.module.out_channels:
            raise ValueError(
                f"Sparsity parameter for '{self.name}' must have one value per Conv output channel."
            )
        if not isinstance(self.module, nn.BatchNorm2d):
            raise TypeError("Channel sparsity targets currently require BatchNorm2d ownership.")

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "channel_target": self.channel_target.name,
            "source": self.source,
            "module_type": type(self.module).__name__,
            "shape": list(self.parameter.shape),
        }


@dataclass(frozen=True)
class StructuralBlockTarget:
    """An adapter-declared removable architecture block.

    Unlike tensor pruning targets, this describes a complete module whose
    removal and any topology repair are owned by the model adapter.  ``index``
    is its position in the adapter-declared owner collection before mutation.
    """

    name: str
    module: nn.Module = field(repr=False, compare=False)
    block_type: str = "structural_block"
    owner_name: str = ""
    index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "block_type": self.block_type,
            "module_type": type(self.module).__name__,
            "owner_name": self.owner_name,
            "index": self.index,
            "metadata": dict(self.metadata),
        }


@dataclass
class PruningGroup:
    """A primary target and the model targets coupled to its mutation.

    For structured Conv pruning, ``dependency_count`` comes from the
    torch-pruning dependency group built during validation.  The dependency
    graph itself is intentionally not retained because it becomes stale after
    any structural mutation.
    """

    primary: PrunableTarget | StructuralBlockTarget
    related_targets: List[PrunableTarget | StructuralBlockTarget] = field(default_factory=list)
    operation: str = "mask_weight"
    indices: List[int] = field(default_factory=list)
    dependencies: List[Dict[str, Any]] = field(default_factory=list)
    dependency_count: int = 0
    validated: bool = False
    validation_error: Optional[str] = None

    def describe(self) -> Dict[str, Any]:
        return {
            "primary": self.primary.describe(),
            "related_targets": [target.describe() for target in self.related_targets],
            "operation": self.operation,
            "indices": list(self.indices),
            "dependencies": [dict(dependency) for dependency in self.dependencies],
            "dependency_count": self.dependency_count,
            "validated": self.validated,
            "validation_error": self.validation_error,
        }


@dataclass
class PruningPlan:
    """An inspectable collection of mutations produced before model changes."""

    pruner_name: str
    groups: List[PruningGroup] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.groups

    def describe(self) -> Dict[str, Any]:
        return {
            "pruner": self.pruner_name,
            "groups": [group.describe() for group in self.groups],
            "metadata": dict(self.metadata),
        }

    def target_names(self) -> Iterable[str]:
        return (group.primary.name for group in self.groups)
