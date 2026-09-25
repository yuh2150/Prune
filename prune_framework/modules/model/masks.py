"""Persistent, serializable weight masks for Conv2d and Linear modules."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

import torch
import torch.nn as nn
from torch.nn.utils import parametrize


class WeightMask(nn.Module):
    """A parametrization that leaves the original parameter trainable.

    The effective module weight is always ``original * mask``.  This avoids
    materializing zeros during recovery while preserving PyTorch state-dict
    support for both the original parameter and the mask buffer.
    """

    def __init__(self, mask: torch.Tensor):
        super().__init__()
        self.register_buffer("mask", mask.detach().clone())

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight * self.mask


class MaskManager:
    """Apply, persist, enforce, and explicitly materialize weight masks."""

    _SUPPORTED = (nn.Conv2d, nn.Linear)

    @classmethod
    def apply(cls, module: nn.Module, mask: torch.Tensor) -> None:
        if not isinstance(module, cls._SUPPORTED):
            raise TypeError(f"Persistent masks support Conv2d and Linear, got {type(module).__name__}.")
        if tuple(mask.shape) != tuple(module.weight.shape):
            raise ValueError(
                f"Mask shape {tuple(mask.shape)} does not match weight shape {tuple(module.weight.shape)}."
            )
        mask = mask.to(device=module.weight.device, dtype=module.weight.dtype)
        if cls.has_mask(module):
            cls._mask_module(module).mask.copy_(mask)
        else:
            parametrize.register_parametrization(module, "weight", WeightMask(mask))
            original = module.parametrizations.weight.original
            # Optimizers normally see ``original``. The hook prevents ordinary
            # gradient updates from reviving masked elements between steps.
            original.register_hook(lambda grad, owner=module: grad * cls.mask(owner))
        cls.enforce_module(module)

    @classmethod
    def has_mask(cls, module: nn.Module) -> bool:
        return hasattr(module, "parametrizations") and "weight" in module.parametrizations

    @classmethod
    def _mask_module(cls, module: nn.Module) -> WeightMask:
        if not cls.has_mask(module) or not isinstance(module.parametrizations.weight[0], WeightMask):
            raise ValueError(f"Module {type(module).__name__} has no framework weight mask.")
        return module.parametrizations.weight[0]

    @classmethod
    def mask(cls, module: nn.Module) -> torch.Tensor:
        return cls._mask_module(module).mask

    @classmethod
    def original_weight(cls, module: nn.Module) -> torch.nn.Parameter:
        if cls.has_mask(module):
            return module.parametrizations.weight.original
        return module.weight

    @classmethod
    def enforce_module(cls, module: nn.Module) -> None:
        """Zero masked values in the original parameter after an optimizer step."""
        if cls.has_mask(module):
            with torch.no_grad():
                cls.original_weight(module).mul_(cls.mask(module))

    @classmethod
    def enforce(cls, model: nn.Module) -> None:
        for module in model.modules():
            if isinstance(module, cls._SUPPORTED) and cls.has_mask(module):
                cls.enforce_module(module)

    @classmethod
    def attach_optimizer(cls, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        """Enforce masks after every optimizer step, including weight decay.

        PyTorch optimizers expose post-step hooks on supported versions.  A
        caller on an older PyTorch can call :meth:`enforce` after ``step()``.
        """
        if not hasattr(optimizer, "register_step_post_hook"):
            raise RuntimeError("Optimizer post-step hooks are unavailable; call MaskManager.enforce(model) after step().")
        optimizer.register_step_post_hook(lambda _optimizer, _args, _kwargs: cls.enforce(model))

    @classmethod
    def state_dict(cls, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Return a compact, name-keyed mask state independent of module layout."""
        return {
            name: cls.mask(module).detach().cpu().clone()
            for name, module in model.named_modules()
            if isinstance(module, cls._SUPPORTED) and cls.has_mask(module)
        }

    @classmethod
    def load_state_dict(cls, model: nn.Module, masks: Mapping[str, torch.Tensor]) -> None:
        modules = dict(model.named_modules())
        for name, mask in masks.items():
            if name not in modules:
                raise KeyError(f"Mask references missing module '{name}'.")
            cls.apply(modules[name], mask)

    @classmethod
    def materialize(cls, model: nn.Module) -> None:
        """Make zeros permanent and remove parametrizations for export only."""
        for module in model.modules():
            if isinstance(module, cls._SUPPORTED) and cls.has_mask(module):
                cls.enforce_module(module)
                parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
