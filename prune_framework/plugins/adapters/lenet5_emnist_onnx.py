"""Adapter that imports the checked-in 47-class LeNet-5 ONNX weights."""

from __future__ import annotations

import os
import pickle
from typing import Any, Iterable, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from prune_framework.contracts.targets import PrunableTarget, TargetType
from prune_framework.core.interfaces import BaseModelAdapter
from prune_framework.core.registry import register_model
from prune_framework.models import LeNet5EMNIST
from prune_framework.modules.model.masks import MaskManager


@register_model("lenet5_emnist_onnx")
@register_model("lenet5_onnx")
class LeNet5EMNISTONNXAdapter(BaseModelAdapter):
    """Make the local ONNX LeNet usable by PyTorch pruning pipelines."""

    _WEIGHT_SHAPES = {
        "conv1.conv.weight": (6, 1, 5, 5),
        "conv1.conv.bias": (6,),
        "conv2.conv.weight": (16, 6, 5, 5),
        "conv2.conv.bias": (16,),
        "conv3.conv.weight": (120, 16, 5, 5),
        "conv3.conv.bias": (120,),
        "fc1.weight": (84, 120),
        "fc1.bias": (84,),
        "fc2.weight": (47, 84),
        "fc2.bias": (47,),
    }

    @classmethod
    def load_model(
        cls,
        weights_path: str,
        device: torch.device,
        *,
        num_classes: Optional[int] = None,
    ) -> Tuple[nn.Module, Optional[Any]]:
        if not weights_path or weights_path.lower() in {"none", "random"}:
            return LeNet5EMNIST(num_classes=num_classes or 47).to(device), None
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"LeNet-5 EMNIST checkpoint not found: {weights_path}")
        if weights_path.lower().endswith(".onnx"):
            return cls._load_onnx(weights_path, device, num_classes), None
        return cls._load_state_dict(weights_path, device, num_classes)

    @classmethod
    def _load_onnx(cls, weights_path: str, device: torch.device, num_classes: Optional[int]) -> nn.Module:
        try:
            import onnx
            from onnx import numpy_helper
        except ImportError as exc:
            raise RuntimeError("Loading LeNet5_Numbers&Characters_FP32.onnx requires the 'onnx' package.") from exc
        graph = onnx.load(weights_path).graph
        if len(graph.input) != 1:
            raise ValueError("Expected exactly one ONNX input for the EMNIST LeNet model.")
        dimensions = graph.input[0].type.tensor_type.shape.dim
        static_shape = tuple(dimension.dim_value for dimension in dimensions[1:])
        if static_shape != (1, 32, 32):
            raise ValueError(f"Expected ONNX input [batch, 1, 32, 32], got {static_shape}.")
        initializers = {item.name: numpy_helper.to_array(item) for item in graph.initializer}
        cls._validate_onnx_initializers(initializers)
        checkpoint_classes = int(initializers["fc2.weight"].shape[0])
        if num_classes is not None and num_classes != checkpoint_classes:
            raise ValueError(
                f"LeNet-5 ONNX checkpoint has {checkpoint_classes} classes but model.num_classes={num_classes}."
            )
        model = LeNet5EMNIST(num_classes=checkpoint_classes).to(device)
        state = model.state_dict()
        for name in cls._WEIGHT_SHAPES:
            state[name] = torch.from_numpy(initializers[name].copy()).to(
                dtype=state[name].dtype, device=device
            )
        model.load_state_dict(state)
        return model

    @classmethod
    def _validate_onnx_initializers(cls, initializers: dict[str, Any]) -> None:
        missing = set(cls._WEIGHT_SHAPES) - set(initializers)
        if missing:
            raise KeyError(f"ONNX LeNet checkpoint is missing initializers: {sorted(missing)}")
        malformed = {
            name: tuple(initializers[name].shape)
            for name, shape in cls._WEIGHT_SHAPES.items()
            if tuple(initializers[name].shape) != shape
        }
        if malformed:
            raise ValueError(f"ONNX LeNet initializer shapes do not match the expected topology: {malformed}")

    @classmethod
    def _load_state_dict(
        cls, weights_path: str, device: torch.device, num_classes: Optional[int]
    ) -> Tuple[nn.Module, Optional[Any]]:
        try:
            payload = torch.load(weights_path, map_location=device, weights_only=True)
        except (TypeError, RuntimeError, pickle.UnpicklingError):
            # Structured pruning changes Conv/Linear dimensions.  Framework
            # exports therefore retain the exact safe model topology alongside
            # its state dict; load that topology instead of forcing original
            # ONNX dimensions onto a pruned checkpoint.
            payload = torch.load(weights_path, map_location=device, weights_only=False)
        state = payload.get("model", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
        if isinstance(state, nn.Module):
            checkpoint_classes = int(getattr(getattr(state, "fc2", None), "out_features", 0))
            if checkpoint_classes < 1:
                raise TypeError("Structured LeNet-5 checkpoint model is missing the final fc2 classifier.")
            if num_classes is not None and num_classes != checkpoint_classes:
                raise ValueError(
                    f"LeNet-5 checkpoint has {checkpoint_classes} classes but model.num_classes={num_classes}."
                )
            return state.to(device), payload if isinstance(payload, dict) else None
        if not isinstance(state, dict):
            raise TypeError("LeNet-5 EMNIST checkpoint must contain a model state dict.")
        weight = state.get("fc2.weight")
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise KeyError("LeNet-5 EMNIST checkpoint is missing fc2.weight needed to determine class count.")
        checkpoint_classes = int(weight.shape[0])
        if num_classes is not None and num_classes != checkpoint_classes:
            raise ValueError(
                f"LeNet-5 EMNIST checkpoint has {checkpoint_classes} classes but model.num_classes={num_classes}."
            )
        model = LeNet5EMNIST(num_classes=checkpoint_classes).to(device)
        cls._attach_serialized_masks(model, state)
        model.load_state_dict(state)
        return model, payload if isinstance(payload, dict) else None

    @staticmethod
    def _attach_serialized_masks(model: nn.Module, state: dict[str, Any]) -> None:
        modules = dict(model.named_modules())
        suffix = ".parametrizations.weight.0.mask"
        for key, mask in state.items():
            if key.endswith(suffix):
                name = key[:-len(suffix)]
                if name not in modules:
                    raise KeyError(f"Checkpoint mask references missing EMNIST LeNet module '{name}'.")
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
        return name == "fc2"

    def get_pruneable_blocks(self):
        return []

    def get_dummy_input(self, device: torch.device) -> torch.Tensor:
        parameter = next(self.model.parameters())
        return torch.randn(1, 1, 32, 32, device=device, dtype=parameter.dtype)
