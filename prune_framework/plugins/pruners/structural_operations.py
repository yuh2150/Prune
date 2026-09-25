"""Typed ``torch_pruning`` operations used for physical width pruning.

The registry keeps the mapping from a framework target type to a third-party
mutation operation in one place.  Adapters still decide which targets are safe
for a model family; this module only describes operations whose module and
dimension semantics are known to the generic structured pruner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Type

import torch.nn as nn
import torch_pruning as tp

from prune_framework.contracts.targets import TargetType


@dataclass(frozen=True)
class StructuralOperation:
    """One safe root operation supported by ``StructuredPruner``."""

    target_type: TargetType
    module_type: Type[nn.Module]
    pruning_fn: Callable
    plan_operation: str
    dimension_name: str

    def supports(self, module: nn.Module) -> bool:
        return isinstance(module, self.module_type)


STRUCTURAL_OPERATIONS: Dict[TargetType, StructuralOperation] = {
    TargetType.CONV_OUT_CHANNEL: StructuralOperation(
        target_type=TargetType.CONV_OUT_CHANNEL,
        module_type=nn.Conv2d,
        pruning_fn=tp.prune_conv_out_channels,
        plan_operation="prune_conv_out_channels",
        dimension_name="channels",
    ),
    TargetType.LINEAR_OUT_FEATURE: StructuralOperation(
        target_type=TargetType.LINEAR_OUT_FEATURE,
        module_type=nn.Linear,
        pruning_fn=tp.prune_linear_out_channels,
        plan_operation="prune_linear_out_features",
        dimension_name="features",
    ),
}


def get_structural_operation(target_type: TargetType) -> Optional[StructuralOperation]:
    """Return the registered physical-pruning operation for ``target_type``."""

    return STRUCTURAL_OPERATIONS.get(target_type)
