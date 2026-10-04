"""Magnitude-based rectangular block-sparse weight masking."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
import torch.nn as nn

from prune_framework.contracts.targets import PruningGroup, PruningPlan
from prune_framework.core.interfaces import BaseImportanceCriterion, BaseModelAdapter, BasePruner
from prune_framework.core.registry import register_pruner
from prune_framework.modules.model.masks import MaskManager
from prune_framework.modules.model.sparsity_patterns import validate_block_sparse_pattern, weight_matrix
from prune_framework.plugins.pruners.unstructured import UnstructuredPruner


@register_pruner("block_sparse")
@register_pruner("block_sparsity")
class BlockSparsePruner(BasePruner):
    """Mask whole blocks in the documented two-dimensional weight view."""

    pruning_mode = "block_sparse"
    supports_layerwise_policy = False

    def validate_config(self, criterion: BaseImportanceCriterion, config: Dict[str, Any]) -> None:
        shape = config.get("block_shape")
        if not isinstance(shape, (list, tuple)) or len(shape) != 2 or any(not isinstance(v, int) or v <= 0 for v in shape):
            raise ValueError("Block-sparse pruning requires block_shape=[positive_rows, positive_columns].")
        if config.get("global_pruning", False):
            raise ValueError("Block-sparse pruning is per-tensor and does not support global selection.")
        if isinstance(config.get("pruning_params"), (list, tuple)):
            raise ValueError("Block-sparse pruning does not support a layer-wise pruning policy.")
        amount = config.get("amount", 0.3)
        if not isinstance(amount, (int, float)) or not 0 <= float(amount) < 1:
            raise ValueError("Block-sparse pruning amount must be in [0, 1).")
        name = str(config.get("criterion_name", type(criterion).__name__)).lower()
        if name not in {"magnitude", "l1", "l1_norm", "l2", "l2_norm"}:
            raise ValueError("Block-sparse pruning currently supports magnitude, L1, or L2 scoring only.")

    def create_plan(
        self,
        model_adapter: BaseModelAdapter,
        criterion: BaseImportanceCriterion,
        granularity=None,
        config: Dict[str, Any] | None = None,
    ) -> PruningPlan:
        del granularity
        config = config or {}
        self.validate_config(criterion, config)
        shape = tuple(int(value) for value in config["block_shape"])
        amount = float(config.get("amount", 0.3))
        plan = PruningPlan(
            pruner_name="block_sparse",
            metadata={"semantics": "masked_rectangular_weight_blocks", "block_shape": list(shape), "block_score": "mean_abs"},
        )
        for target in UnstructuredPruner._weight_targets(model_adapter):
            matrix = weight_matrix(target.module)
            self._validate_shape(target.name, matrix, shape)
            existing = self._existing_matrix_mask(target.module)
            if not validate_block_sparse_pattern(existing, shape):
                raise ValueError(f"Existing mask for {target.name} is not aligned with block_shape={list(shape)}.")
            block_scores = self._block_scores(matrix.detach().abs(), shape)
            count = int(round(block_scores.numel() * amount))
            if count == 0:
                continue
            selected = torch.argsort(block_scores.reshape(-1), stable=True)[:count].tolist()
            mask = torch.ones_like(matrix)
            block_rows, block_cols = matrix.shape[0] // shape[0], matrix.shape[1] // shape[1]
            for flat_index in selected:
                row, column = divmod(flat_index, block_cols)
                mask[row * shape[0]:(row + 1) * shape[0], column * shape[1]:(column + 1) * shape[1]] = 0
            mask.mul_(existing)
            if not validate_block_sparse_pattern(mask, shape):
                raise RuntimeError(f"Internal block-sparse mask generation failed for {target.name}.")
            indices = torch.nonzero(mask.reshape(-1).eq(0), as_tuple=False).flatten().tolist()
            plan.groups.append(PruningGroup(
                primary=target,
                operation="mask_weight",
                indices=[int(index) for index in indices],
                dependencies=[{"pattern": "rectangular_block", "block_shape": list(shape)}],
            ))
        return plan

    def validate_plan(self, plan: PruningPlan, model_adapter: BaseModelAdapter, config=None) -> bool:
        del model_adapter
        config = config or {}
        shape = tuple(int(value) for value in config["block_shape"])
        valid = True
        for group in plan.groups:
            module = group.primary.module
            candidate = torch.ones_like(module.weight)
            candidate.reshape(-1)[group.indices] = 0
            if MaskManager.has_mask(module):
                candidate.mul_(MaskManager.mask(module))
            pattern_ok = validate_block_sparse_pattern(weight_matrix(candidate), shape)
            group.validated = (
                group.operation == "mask_weight"
                and isinstance(module, (nn.Conv2d, nn.Linear))
                and bool(group.indices)
                and len(set(group.indices)) == len(group.indices)
                and min(group.indices) >= 0
                and max(group.indices) < module.weight.numel()
                and pattern_ok
            )
            group.dependency_count = 1
            group.validation_error = None if group.validated else "Invalid block-sparse weight-mask action."
            valid = valid and group.validated
        return valid

    def apply_plan(self, plan: PruningPlan, model_adapter: BaseModelAdapter, config=None) -> nn.Module:
        for group in plan.groups:
            if not group.validated:
                raise RuntimeError(f"Cannot apply invalid block-sparse mask action for {group.primary.name}.")
            mask = torch.ones_like(group.primary.module.weight)
            mask.reshape(-1)[group.indices] = 0
            MaskManager.apply(group.primary.module, mask)
        optimizer = (config or {}).get("optimizer")
        if optimizer is not None:
            MaskManager.attach_optimizer(model_adapter.model, optimizer)
        return model_adapter.model

    @staticmethod
    def _validate_shape(name: str, matrix: torch.Tensor, shape: Sequence[int]) -> None:
        if matrix.shape[0] % shape[0] or matrix.shape[1] % shape[1]:
            raise ValueError(
                f"Block-sparse pruning requires {name} matrix shape {tuple(matrix.shape)} to be divisible by block_shape={list(shape)}."
            )

    @staticmethod
    def _block_scores(matrix: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
        rows, cols = shape
        blocks = matrix.reshape(matrix.shape[0] // rows, rows, matrix.shape[1] // cols, cols)
        return blocks.permute(0, 2, 1, 3).reshape(-1, rows * cols).mean(dim=1)

    @staticmethod
    def _existing_matrix_mask(module: nn.Module) -> torch.Tensor:
        if MaskManager.has_mask(module):
            return weight_matrix(MaskManager.mask(module)).detach()
        return torch.ones_like(weight_matrix(module))
