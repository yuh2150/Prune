"""Deterministic validators for constrained weight-mask patterns."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def weight_matrix(module: nn.Module | torch.Tensor) -> torch.Tensor:
    """Return the documented 2-D grouping view without copying weight data.

    A two-dimensional tensor is accepted to make runtime validators operate on
    candidate masks without constructing a synthetic module.
    """
    if isinstance(module, torch.Tensor):
        if module.ndim < 2:
            raise TypeError(f"Constrained sparsity expects a matrix-like tensor, got shape {tuple(module.shape)}.")
        return module.reshape(module.shape[0], -1)
    if isinstance(module, nn.Linear):
        return module.weight.reshape(module.out_features, module.in_features)
    if isinstance(module, nn.Conv2d):
        return module.weight.reshape(module.out_channels, -1)
    raise TypeError(f"Constrained sparsity supports Conv2d and Linear, got {type(module).__name__}.")


def validate_nm_pattern(values: torch.Tensor, n: int, m: int, *, exact: bool = True) -> bool:
    """Check N non-zero entries per contiguous input-dimension group of M."""
    if n <= 0 or m <= 0 or n > m or values.ndim != 2 or values.shape[1] % m:
        return False
    nonzero = values.ne(0).reshape(values.shape[0], -1, m).sum(dim=-1)
    return bool(torch.all(nonzero == n if exact else nonzero <= n))


def validate_block_sparse_pattern(values: torch.Tensor, block_shape: Sequence[int]) -> bool:
    """Each rectangular block must be entirely retained or entirely masked."""
    if len(block_shape) != 2:
        return False
    rows, cols = (int(block_shape[0]), int(block_shape[1]))
    if rows <= 0 or cols <= 0 or values.ndim != 2 or values.shape[0] % rows or values.shape[1] % cols:
        return False
    blocks = values.ne(0).reshape(values.shape[0] // rows, rows, values.shape[1] // cols, cols)
    block_counts = blocks.permute(0, 2, 1, 3).reshape(-1, rows * cols).sum(dim=1)
    return bool(torch.all((block_counts == 0) | (block_counts == rows * cols)))
