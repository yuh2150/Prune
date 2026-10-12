"""Torchvision ResNet-18 adapter for ImageNet classification pruning."""
from __future__ import annotations

import os
import pickle
from typing import Any, Iterable, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from prune_framework.contracts.targets import ChannelSparsityTarget, PrunableTarget, StructuralBlockTarget, TargetType
from prune_framework.core.interfaces import BaseModelAdapter
from prune_framework.core.registry import register_model


@register_model("resnet18")
@register_model("torchvision_resnet18")
@register_model("resnet18_cifar10")
class ResNet18Adapter(BaseModelAdapter):
    @classmethod
    def load_model(cls, weights_path: str, device: torch.device, *, num_classes: Optional[int] = None) -> Tuple[nn.Module, Optional[Any]]:
        from torchvision.models import ResNet18_Weights, resnet18
        if num_classes == 10 and str(weights_path).startswith("hf_hub:"):
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise RuntimeError("CIFAR-10 ResNet-18 requires huggingface_hub. Install it with 'pip install huggingface_hub'.") from exc
            repo_id = str(weights_path).removeprefix("hf_hub:")
            checkpoint_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
            # CIFAR ResNet-18 has a 3x3/stride-1 stem and no max-pool, unlike
            # torchvision's ImageNet ResNet-18 (7x7/stride-2 stem + max-pool).
            model = resnet18(weights=None, num_classes=10)
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
            model.load_state_dict(state_dict, strict=True)
            return model.to(device), {"state_dict": state_dict, "source": repo_id}
        if num_classes not in (None, 1000):
            raise ValueError("ResNet-18 supports num_classes=10 only with an hf_hub: CIFAR-10 checkpoint.")
        if weights_path and os.path.isfile(weights_path):
            try:
                payload = torch.load(weights_path, map_location=device, weights_only=True)
            except (TypeError, RuntimeError, pickle.UnpicklingError):
                payload = torch.load(weights_path, map_location=device, weights_only=False)
            model = payload.get("model", payload) if isinstance(payload, dict) else payload
            if not isinstance(model, nn.Module):
                raise TypeError("ResNet-18 framework checkpoints must contain a serialized model module.")
            return model.to(device), payload if isinstance(payload, dict) else None
        weights = ResNet18_Weights.DEFAULT if str(weights_path).lower() in {"default", "imagenet1k", "imagenet-1k"} else None
        if weights is None:
            raise ValueError("ResNet18Adapter currently supports weights='DEFAULT' only.")
        return resnet18(weights=weights).to(device), None

    def supported_target_types(self) -> Set[TargetType]:
        return {TargetType.CONV_WEIGHT, TargetType.CONV_OUT_CHANNEL, TargetType.LINEAR_WEIGHT, TargetType.LINEAR_OUT_FEATURE}

    def get_pruneable_modules(self) -> List[Tuple[str, nn.Module]]:
        return [(name, module) for name, module in self.model.named_modules()
                if isinstance(module, (nn.Conv2d, nn.Linear)) and not self.is_protected_module(name, module)]

    def get_prunable_targets(self, target_types: Optional[Iterable[TargetType]] = None) -> List[PrunableTarget]:
        return super().get_prunable_targets(target_types)

    @staticmethod
    def is_protected_module(name: str, module: nn.Module) -> bool:
        del module
        return name == "fc" or name.startswith("fc.")

    def get_importance_module(self, name: str, module: nn.Module) -> nn.Module:
        parent = dict(self.model.named_modules()).get(name.rsplit(".", 1)[0]) if "." in name else None
        return parent if parent is not None and isinstance(getattr(parent, "bn1", None), nn.BatchNorm2d) else module

    def supports_channel_sparsity_regularization(self) -> bool:
        return True

    def get_channel_sparsity_targets(self) -> List[ChannelSparsityTarget]:
        modules = dict(self.model.named_modules())
        targets = []
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.BatchNorm2d):
                continue
            conv_name = "conv1" if name == "bn1" else name.rsplit(".", 1)[0] + ".conv" + name[-1]
            conv = modules.get(conv_name)
            if isinstance(conv, nn.Conv2d) and not self.is_protected_module(conv_name, conv):
                targets.append(ChannelSparsityTarget(name=name, module=module, parameter=module.weight,
                    channel_target=PrunableTarget(conv_name, conv, TargetType.CONV_OUT_CHANNEL)))
        return targets

    def get_structural_block_targets(self) -> List[StructuralBlockTarget]:
        targets = []
        for layer_name in ("layer1", "layer2", "layer3", "layer4"):
            layer = getattr(self.model, layer_name)
            # Removing the final same-shape BasicBlock preserves residual topology.
            index = len(layer) - 1
            targets.append(StructuralBlockTarget(f"{layer_name}.{index}", layer[index], "resnet_basicblock", layer_name, index))
        return targets

    def validate_structural_block_plan(self, targets):
        return {target.name: "ResNet layer must retain one BasicBlock" for target in targets
                if target.index != len(getattr(self.model, target.owner_name)) - 1 or len(getattr(self.model, target.owner_name)) <= 1}

    def order_structural_block_removals(self, targets):
        return sorted(targets, key=lambda target: (target.owner_name, -target.index))

    def remove_structural_block(self, target):
        layer = getattr(self.model, target.owner_name)
        retained = [block for index, block in enumerate(layer) if index != target.index]
        setattr(self.model, target.owner_name, nn.Sequential(*retained))

    def get_pruneable_blocks(self):
        return [(target.index, target.module) for target in self.get_structural_block_targets()]

    def get_dummy_input(self, device: torch.device) -> torch.Tensor:
        parameter = next(self.model.parameters())
        spatial_size = 32 if getattr(self.model.conv1, "kernel_size", None) == (3, 3) else 224
        return torch.randn(1, 3, spatial_size, spatial_size, device=device, dtype=parameter.dtype)
