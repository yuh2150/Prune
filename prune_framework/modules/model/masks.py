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
            # Pruning is monotonic: a later plan may add zeros but cannot
            # silently revive weights removed by an earlier plan.
            cls._mask_module(module).mask.mul_(mask)
        else:
            parametrize.register_parametrization(module, "weight", WeightMask(mask))
            original = module.parametrizations.weight.original
            # Optimizers normally see ``original``. The hook prevents ordinary
            # gradient updates from reviving masked elements between steps.
            # Frozen detector parameters do not accept autograd hooks. They
            # still receive persistent masking and post-step enforcement.
            if original.requires_grad:
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
        ParameterMaskManager.enforce(model)

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
        ParameterMaskManager.materialize(model)


class ParameterMaskManager:
    """Persistent masks for explicit non-``weight`` parameters.

    Attention-head masking needs to constrain fused QKV tensors and projection
    biases as well as ordinary Linear weights.  This manager deliberately uses
    an explicit parameter name so it cannot accidentally mask arbitrary model
    state through a broad module scan.
    """

    @staticmethod
    def _parametrization(module: nn.Module, parameter_name: str) -> WeightMask:
        if not hasattr(module, "parametrizations") or parameter_name not in module.parametrizations:
            raise ValueError(f"{type(module).__name__}.{parameter_name} has no framework parameter mask.")
        mask = module.parametrizations[parameter_name][0]
        if not isinstance(mask, WeightMask):
            raise ValueError(f"{type(module).__name__}.{parameter_name} has an incompatible parametrization.")
        return mask

    @classmethod
    def has_mask(cls, module: nn.Module, parameter_name: str) -> bool:
        return hasattr(module, "parametrizations") and parameter_name in module.parametrizations

    @classmethod
    def apply(cls, module: nn.Module, parameter_name: str, mask: torch.Tensor) -> None:
        parameter = getattr(module, parameter_name, None)
        if not isinstance(parameter, torch.Tensor):
            raise TypeError(f"{type(module).__name__}.{parameter_name} is not a tensor parameter.")
        if tuple(mask.shape) != tuple(parameter.shape):
            raise ValueError(f"Mask shape {tuple(mask.shape)} does not match {parameter_name} shape {tuple(parameter.shape)}.")
        mask = mask.to(device=parameter.device, dtype=parameter.dtype)
        if cls.has_mask(module, parameter_name):
            stored = cls._parametrization(module, parameter_name).mask
            stored.mul_(mask)
        else:
            parametrize.register_parametrization(module, parameter_name, WeightMask(mask))
            original = module.parametrizations[parameter_name].original
            original.register_hook(lambda grad, owner=module, name=parameter_name: grad * cls.mask(owner, name))
        cls.enforce_module(module, parameter_name)

    @classmethod
    def mask(cls, module: nn.Module, parameter_name: str) -> torch.Tensor:
        return cls._parametrization(module, parameter_name).mask

    @classmethod
    def original(cls, module: nn.Module, parameter_name: str) -> torch.nn.Parameter:
        return module.parametrizations[parameter_name].original if cls.has_mask(module, parameter_name) else getattr(module, parameter_name)

    @classmethod
    def enforce_module(cls, module: nn.Module, parameter_name: str) -> None:
        if cls.has_mask(module, parameter_name):
            with torch.no_grad():
                cls.original(module, parameter_name).mul_(cls.mask(module, parameter_name))

    @classmethod
    def enforce(cls, model: nn.Module) -> None:
        for module in model.modules():
            for name in getattr(module, "parametrizations", {}):
                if cls.has_mask(module, name):
                    cls.enforce_module(module, name)

    @classmethod
    def state_dict(cls, model: nn.Module) -> Dict[str, torch.Tensor]:
        return {
            f"{module_name}:{name}": cls.mask(module, name).detach().cpu().clone()
            for module_name, module in model.named_modules()
            for name in getattr(module, "parametrizations", {})
            if cls.has_mask(module, name)
        }

    @classmethod
    def load_state_dict(cls, model: nn.Module, masks: Mapping[str, torch.Tensor]) -> None:
        modules = dict(model.named_modules())
        for key, mask in masks.items():
            if ":" not in key:
                raise ValueError(f"Invalid parameter-mask key {key!r}.")
            module_name, parameter_name = key.rsplit(":", 1)
            if module_name not in modules:
                raise KeyError(f"Parameter mask references missing module '{module_name}'.")
            cls.apply(modules[module_name], parameter_name, mask)

    @classmethod
    def materialize(cls, model: nn.Module) -> None:
        for module in model.modules():
            for name in list(getattr(module, "parametrizations", {}).keys()):
                if cls.has_mask(module, name):
                    cls.enforce_module(module, name)
                    parametrize.remove_parametrizations(module, name, leave_parametrized=True)
