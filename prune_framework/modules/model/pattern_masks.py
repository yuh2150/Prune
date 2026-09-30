"""Validated dense masks for semi-structured and block-sparse pruning.

Masks remain ordinary dense tensors on purpose: they are serializable through
``MaskManager`` and inference remains a correct dense fallback until a sparse
backend is selected by an application.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import torch


@dataclass(frozen=True)
class PatternMask:
    kind: str
    mask: torch.Tensor
    layout: dict

    @staticmethod
    def nm(weight: torch.Tensor, n: int, m: int) -> "PatternMask":
        if not (isinstance(n, int) and isinstance(m, int) and 0 < n <= m):
            raise ValueError("N:M sparsity requires integers satisfying 0 < N <= M.")
        if weight.numel() % m:
            raise ValueError(f"Weight has {weight.numel()} elements, not divisible by M={m}.")
        flat = weight.detach().abs().reshape(-1, m)
        keep = torch.zeros_like(flat)
        # argsort(stable=True) makes equal magnitudes reproducible.
        keep.scatter_(1, torch.argsort(flat, dim=1, descending=True, stable=True)[:, :n], 1)
        return PatternMask("nm", keep.reshape_as(weight), {"n": n, "m": m, "axis": "flattened_row_major"})

    @staticmethod
    def blocks(weight: torch.Tensor, block: Tuple[int, int], amount: float) -> "PatternMask":
        if len(block) != 2 or not all(isinstance(v, int) and v > 0 for v in block):
            raise ValueError("block_size must contain two positive integers.")
        if not 0 <= amount < 1:
            raise ValueError("block sparse amount must be in [0, 1).")
        rows, cols = weight.shape[0], weight[0].numel()
        br, bc = block
        if rows % br or cols % bc:
            raise ValueError(f"Flattened weight layout ({rows}, {cols}) is not divisible by block_size={block}.")
        matrix = weight.detach().abs().reshape(rows, cols)
        scores = matrix.reshape(rows // br, br, cols // bc, bc).mean((1, 3)).reshape(-1)
        drop = int(scores.numel() * amount)
        keep = torch.ones_like(scores)
        if drop:
            keep[torch.argsort(scores, stable=True)[:drop]] = 0
        expanded = keep.reshape(rows // br, cols // bc).repeat_interleave(br, 0).repeat_interleave(bc, 1)
        return PatternMask("block_sparse", expanded.reshape_as(weight), {"block_size": list(block), "layout": [rows, cols]})
