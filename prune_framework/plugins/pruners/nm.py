"""Magnitude-based N:M semi-structured weight masking."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from prune_framework.contracts.targets import PruningGroup, PruningPlan
from prune_framework.core.interfaces import BaseImportanceCriterion, BaseModelAdapter, BasePruner
from prune_framework.core.registry import register_pruner
from prune_framework.modules.model.masks import MaskManager
from prune_framework.modules.model.sparsity_patterns import validate_nm_pattern, weight_matrix
from prune_framework.plugins.pruners.unstructured import UnstructuredPruner


@register_pruner("nm")
@register_pruner("nm_sparsity")
class NMSparsityPruner(BasePruner):
    """Keep exactly ``n`` values in each contiguous group of ``m`` values.

    Groups are contiguous along the second axis of the documented two-dimensional
    weight view: ``[out_features, in_features]`` for Linear and
    ``[out_channels, in_channels/groups * kH * kW]`` for Conv2d.  A pre-existing
    mask is never relaxed, so it may leave fewer than ``n`` retained values.
    """

    pruning_mode = "nm"
    supports_layerwise_policy = False

    def validate_config(self, criterion: BaseImportanceCriterion, config: Dict[str, Any]) -> None:
        n, m = config.get("n"), config.get("m")
        if not isinstance(n, int) or not isinstance(m, int) or n <= 0 or m <= 0 or n > m:
            raise ValueError("N:M pruning requires integer n and m with 0 < n <= m.")
        if config.get("global_pruning", False):
            raise ValueError("N:M pruning is topology-constrained and does not support global selection.")
        if isinstance(config.get("pruning_params"), (list, tuple)):
            raise ValueError("N:M pruning does not support a layer-wise pruning policy.")
        name = str(config.get("criterion_name", type(criterion).__name__)).lower()
        if name not in {"magnitude", "l1", "l1_norm", "l2", "l2_norm"}:
            raise ValueError("N:M pruning currently supports magnitude, L1, or L2 scoring only.")

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
        n, m = int(config["n"]), int(config["m"])
        plan = PruningPlan(
            pruner_name="nm",
            metadata={
                "semantics": "n_nonzero_per_m",
                "n": n,
                "m": m,
                "implied_weight_sparsity": 1.0 - (n / m),
                "selection": "per_group_magnitude",
                "skipped_incompatible_targets": [],
            },
        )
        for target in UnstructuredPruner._weight_targets(model_adapter):
            scores = weight_matrix(target.module).detach().abs()
            if scores.shape[1] % m:
                # CNN stems (e.g. RGB 7x7 convolutions) often cannot express a
                # 2:4 pattern. Preserve their dense weights and apply the
                # documented pattern to all compatible targets.
                plan.metadata["skipped_incompatible_targets"].append({
                    "name": target.name, "input_width": int(scores.shape[1]), "m": m,
                })
                continue
            existing = self._existing_matrix_mask(target.module)
            # Vectorized group-wise top-k: avoids millions of Python loops on
            # ImageNet-scale convolution weights while preserving existing masks.
            grouped_scores = scores.reshape(scores.shape[0], -1, m)
            active = existing.bool().reshape_as(grouped_scores)
            ranked = grouped_scores.masked_fill(~active, float("-inf"))
            selected = torch.argsort(ranked, dim=-1, descending=True, stable=True)[..., :n]
            selected_active = active.gather(-1, selected)
            grouped_mask = torch.zeros_like(grouped_scores)
            grouped_mask.scatter_(-1, selected, selected_active.to(grouped_mask.dtype))
            mask = grouped_mask.reshape_as(scores)
            if not validate_nm_pattern(mask, n, m, exact=False):
                raise RuntimeError(f"Internal N:M mask generation failed for {target.name}.")
            pruned = torch.nonzero(mask.reshape(-1).eq(0), as_tuple=False).flatten().tolist()
            if pruned:
                plan.groups.append(PruningGroup(
                    primary=target,
                    operation="mask_weight",
                    indices=[int(index) for index in pruned],
                    dependencies=[{"pattern": f"{n}:{m}", "axis": "input_dimension"}],
                ))
        return plan

    def validate_plan(self, plan: PruningPlan, model_adapter: BaseModelAdapter, config=None) -> bool:
        del model_adapter
        config = config or {}
        n, m = int(config["n"]), int(config["m"])
        valid = True
        for group in plan.groups:
            module = group.primary.module
            shape = tuple(module.weight.shape)
            candidate = torch.ones(shape, device=module.weight.device, dtype=module.weight.dtype)
            candidate.reshape(-1)[group.indices] = 0
            if MaskManager.has_mask(module):
                candidate.mul_(MaskManager.mask(module))
            pattern_ok = validate_nm_pattern(weight_matrix(candidate), n, m, exact=False)
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
            group.validation_error = None if group.validated else "Invalid N:M weight-mask action."
            valid = valid and group.validated
        return valid

    def apply_plan(self, plan: PruningPlan, model_adapter: BaseModelAdapter, config=None) -> nn.Module:
        for group in plan.groups:
            if not group.validated:
                raise RuntimeError(f"Cannot apply invalid N:M mask action for {group.primary.name}.")
            module = group.primary.module
            mask = torch.ones_like(module.weight)
            mask.reshape(-1)[group.indices] = 0
            MaskManager.apply(module, mask)
        optimizer = (config or {}).get("optimizer")
        if optimizer is not None:
            MaskManager.attach_optimizer(model_adapter.model, optimizer)
        return model_adapter.model

    @staticmethod
    def _existing_matrix_mask(module: nn.Module) -> torch.Tensor:
        if MaskManager.has_mask(module):
            return weight_matrix(MaskManager.mask(module)).detach()
        return torch.ones_like(weight_matrix(module))
