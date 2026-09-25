"""Reusable train-time sparsity regularization primitives."""

from .sparsity import (
    L1BatchNormScaleRegularization,
    L0HardConcreteRegularization,
    HardConcreteChannelGate,
    RegularizationController,
    RegularizationReport,
    RegularizationTerm,
)

__all__ = [
    "L1BatchNormScaleRegularization",
    "L0HardConcreteRegularization",
    "HardConcreteChannelGate",
    "RegularizationController",
    "RegularizationReport",
    "RegularizationTerm",
]
