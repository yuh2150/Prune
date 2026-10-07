"""Adapter for the local LeNet-5 classification reference model."""

from __future__ import annotations

import os
import pickle
from typing import Any, Iterable, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from prune_framework.contracts.targets import PrunableTarget, TargetType
from prune_framework.core.interfaces import BaseModelAdapter
from prune_framework.core.registry import register_model
from prune_framework.models import LeNet5
from prune_framework.modules.model.masks import MaskManager


@register_model("lenet5")
@register_model("lenet")
class LeNet5Adapter(BaseModelAdapter):
    """Expose safe LeNet Conv/Linear targets while protecting the class head."""

    @classmethod
    def load_model(
        cls,
        weights_path: str,
        device: torch.device,
        *,
        num_classes: Optional[int] = None,
    ) -> Tuple[nn.Module, Optional[Any]]:
        if not weights_path or weights_path.lower() in {"none", "random"}:
            return LeNet5(num_classes=num_classes or 10).to(device), None
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"LeNet-5 checkpoint not found: {weights_path}")
        try:
            payload = torch.load(weights_path, map_location=device, weights_only=True)
        except (TypeError, RuntimeError, pickle.UnpicklingError):
            # Framework exports retain an nn.Module so structurally pruned
            # topologies can be restored without inferring their old shapes.
            payload = torch.load(weights_path, map_location=device, weights_only=False)
        state = payload.get("model", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
        if isinstance(state, nn.Module):
            checkpoint_classes = int(state.classifier[-1].out_features)
            if num_classes is not None and num_classes != checkpoint_classes:
                raise ValueError(
                    f"LeNet-5 checkpoint has {checkpoint_classes} classes but model.num_classes={num_classes}."
                )
            return state.to(device), payload if isinstance(payload, dict) else None
        if not isinstance(state, dict):
            raise TypeError("LeNet-5 checkpoint must contain a model state dict.")
        checkpoint_classes = cls._checkpoint_num_classes(state)
        if num_classes is not None and num_classes != checkpoint_classes:
            raise ValueError(
                f"LeNet-5 checkpoint has {checkpoint_classes} classes but model.num_classes={num_classes}."
            )
        model = LeNet5(num_classes=num_classes or checkpoint_classes).to(device)
        cls._attach_serialized_masks(model, state)
        model.load_state_dict(state)
        return model, payload if isinstance(payload, dict) else None

    @staticmethod
    def _checkpoint_num_classes(state: dict[str, Any]) -> int:
        weight = state.get("classifier.5.weight")
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise KeyError("LeNet-5 checkpoint is missing classifier.5.weight needed to determine class count.")
        return int(weight.shape[0])

    @staticmethod
    def _attach_serialized_masks(model: nn.Module, state: dict[str, Any]) -> None:
        modules = dict(model.named_modules())
        suffix = ".parametrizations.weight.0.mask"
        for key, mask in state.items():
            if key.endswith(suffix):
                name = key[:-len(suffix)]
                if name not in modules:
                    raise KeyError(f"Checkpoint mask references missing LeNet module '{name}'.")
                MaskManager.apply(modules[name], mask)

    def supported_target_types(self) -> Set[TargetType]:
        return {
            TargetType.CONV_WEIGHT,
            TargetType.CONV_OUT_CHANNEL,
            TargetType.LINEAR_WEIGHT,
            TargetType.LINEAR_OUT_FEATURE,
        }

    def get_pruneable_modules(self) -> List[Tuple[str, nn.Module]]:
        return [
            (name, module)
            for name, module in self.model.named_modules()
            if isinstance(module, (nn.Conv2d, nn.Linear)) and not self.is_protected_module(name, module)
        ]

    def get_prunable_targets(self, target_types: Optional[Iterable[TargetType]] = None) -> List[PrunableTarget]:
        return super().get_prunable_targets(target_types)

    @staticmethod
    def is_protected_module(name: str, module: nn.Module) -> bool:
        del module
        return name == "classifier.5"

    def get_pruneable_blocks(self):
        return []

    def get_dummy_input(self, device: torch.device) -> torch.Tensor:
        parameter = next(self.model.parameters())
        return torch.randn(1, 1, 28, 28, device=device, dtype=parameter.dtype)
