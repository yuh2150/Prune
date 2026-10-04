from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Callable, Dict, Iterable, List, Tuple, Optional, Any, Set

from .targets import AttentionHeadTarget, ChannelSparsityTarget, PrunableTarget, StructuralBlockTarget, TargetType


class BaseModelAdapter(ABC):
    """Abstract Base Class for Model Adapters."""

    def __init__(self, model: nn.Module):
        self.model = model

    @classmethod
    def load_model(cls, weights_path: str, device: torch.device) -> Tuple[nn.Module, Optional[Any]]:
        """Loads model instance and optional checkpoint dictionary."""
        raise NotImplementedError("Subclass must implement load_model classmethod.")

    @abstractmethod
    def get_pruneable_modules(self) -> List[Tuple[str, nn.Module]]:
        """Returns name and module pairs for prunable layers."""
        pass

    @abstractmethod
    def get_pruneable_blocks(self) -> List[Tuple[int, nn.Module]]:
        """Returns block index and module pairs for depth/block pruning."""
        pass

    @abstractmethod
    def get_dummy_input(self, device: torch.device) -> torch.Tensor:
        """Returns dummy tensor for tracing and validation."""
        pass

    def get_importance_module(self, name: str, module: nn.Module) -> nn.Module:
        """Returns the module whose parameters represent ``module``'s importance.

        Most architectures score the pruned module itself. Adapters may return a
        Conv-BN wrapper when a criterion needs BatchNorm parameters.
        """
        return module

    def supported_target_types(self) -> Set[TargetType]:
        """Return the target kinds this adapter has explicitly approved.

        The base implementation preserves legacy adapters by exposing the
        Conv2d targets represented by ``get_pruneable_modules``. New adapters
        should override this rather than teaching pruners architecture-specific
        module-name rules.
        """
        return {TargetType.CONV_WEIGHT, TargetType.CONV_OUT_CHANNEL}

    def get_prunable_targets(
        self, target_types: Optional[Iterable[TargetType]] = None
    ) -> List[PrunableTarget]:
        """Enumerate typed targets from adapter-approved modules.

        This is the P0 boundary between model policy and pruning mechanics.
        The legacy ``get_pruneable_modules`` API remains available for existing
        callers and structural-pruning code.
        """
        allowed = set(target_types) if target_types is not None else self.supported_target_types()
        allowed &= self.supported_target_types()
        targets: List[PrunableTarget] = []
        for name, module in self.get_pruneable_modules():
            if isinstance(module, nn.Conv2d):
                if TargetType.CONV_WEIGHT in allowed:
                    targets.append(PrunableTarget(name, module, TargetType.CONV_WEIGHT))
                if TargetType.CONV_OUT_CHANNEL in allowed:
                    targets.append(PrunableTarget(name, module, TargetType.CONV_OUT_CHANNEL))
            elif isinstance(module, nn.Linear):
                if TargetType.LINEAR_WEIGHT in allowed:
                    targets.append(PrunableTarget(name, module, TargetType.LINEAR_WEIGHT))
                if TargetType.LINEAR_OUT_FEATURE in allowed:
                    targets.append(PrunableTarget(name, module, TargetType.LINEAR_OUT_FEATURE))
        return targets

    def supports_gradient_calibration(self) -> bool:
        """Whether this adapter can supply batches and a task loss internally."""
        return False

    def get_gradient_calibration_batches(
        self, batches: int, device: torch.device, seed: int, batch_size: int = 1
    ) -> Iterable[Any]:
        raise NotImplementedError(f"{type(self).__name__} does not provide gradient calibration batches.")

    def build_gradient_calibration_loss(self) -> Callable[[nn.Module, Any], torch.Tensor]:
        raise NotImplementedError(f"{type(self).__name__} does not provide a gradient calibration loss.")

    def supports_channel_sparsity_regularization(self) -> bool:
        """Whether this adapter approves train-time channel-sparsity targets."""
        return False

    def get_channel_sparsity_targets(self) -> List[ChannelSparsityTarget]:
        """Map safe Conv output targets to their trainable BN scales.

        The default only acts after an adapter opts in. It uses the existing
        adapter importance mapping, so Conv-BN wrappers can expose their own
        relationship without regularizers depending on architecture names.
        """
        if not self.supports_channel_sparsity_regularization():
            return []
        targets: List[ChannelSparsityTarget] = []
        for target in self.get_prunable_targets({TargetType.CONV_OUT_CHANNEL}):
            importance_module = self.get_importance_module(target.name, target.module)
            batch_norm = (
                importance_module
                if isinstance(importance_module, nn.BatchNorm2d)
                else getattr(importance_module, "bn", None)
            )
            if not isinstance(batch_norm, nn.BatchNorm2d) or batch_norm.weight is None:
                continue
            targets.append(
                ChannelSparsityTarget(
                    name=target.name,
                    channel_target=target,
                    parameter=batch_norm.weight,
                    module=batch_norm,
                )
            )
        return targets

    def build_regularized_training_loss(self, regularization: Any) -> Callable[..., Any]:
        """Wrap the adapter's task loss with a reusable regularization term.

        Recovery code owns data loading and optimizer steps. This wrapper keeps
        normal training unchanged while giving adapters with an integrated task
        loss (currently YOLO) a direct, architecture-safe entry point.
        """
        if not self.supports_gradient_calibration():
            raise NotImplementedError(f"{type(self).__name__} has no integrated task loss for regularized recovery.")
        task_loss = self.build_gradient_calibration_loss()

        def regularized_loss(model: nn.Module, batch: Any, step: int | None = None, total_steps: int | None = None):
            return regularization.augment_loss(task_loss(model, batch), step=step, total_steps=total_steps)

        return regularized_loss

    def get_structural_block_targets(self) -> List[StructuralBlockTarget]:
        """Return adapter-approved removable blocks or layers.

        Empty by default: structural deletion is unsafe unless an adapter also
        implements validation, removal and repair hooks below.
        """
        return []

    def get_attention_head_targets(self) -> List[AttentionHeadTarget]:
        """Return only adapter-approved attention modules; default is unsupported."""
        return []

    def score_attention_heads(self, target: AttentionHeadTarget) -> torch.Tensor:
        """Magnitude score combining Q/K/V outputs and output-projection inputs."""
        module, heads, width = target.module, target.num_heads, target.head_dim
        scores = []
        if target.layout == "fused_qkv":
            qkv = module.in_proj_weight.detach().reshape(3, heads, width, -1)
            scores.extend(qkv.abs().mean(dim=(2, 3)))
            scores.append(module.out_proj.weight.detach().reshape(-1, heads, width).abs().mean(dim=(0, 2)))
        else:
            for name in ("q_proj", "k_proj", "v_proj"):
                projection = getattr(module, name)
                scores.append(projection.weight.detach().reshape(heads, width, -1).abs().mean(dim=(1, 2)))
            scores.append(getattr(module, "o_proj").weight.detach().reshape(-1, heads, width).abs().mean(dim=(0, 2)))
        return torch.stack(scores).mean(dim=0)

    def validate_attention_head_plan(self, target: AttentionHeadTarget, indices: List[int]) -> Optional[str]:
        if target not in self.get_attention_head_targets():
            return f"Adapter does not declare attention target '{target.name}'."
        if not indices or len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= target.num_heads:
            return "Attention-head indices are invalid."
        if len(indices) >= target.num_heads:
            return "Attention pruning must retain at least one head."
        return None

    def apply_attention_head_mask(self, target: AttentionHeadTarget, indices: List[int]) -> None:
        """Mask Q/K/V output slices and output-projection input slices consistently."""
        from prune_framework.modules.model.masks import ParameterMaskManager

        module, heads, width = target.module, target.num_heads, target.head_dim
        selected = torch.as_tensor(indices, device=next(module.parameters()).device)
        offsets = selected[:, None] * width + torch.arange(width, device=selected.device)[None, :]
        rows = offsets.reshape(-1)
        if target.layout == "fused_qkv":
            weight_mask = torch.ones_like(module.in_proj_weight)
            for projection in range(3):
                weight_mask[rows + projection * heads * width, :] = 0
            ParameterMaskManager.apply(module, "in_proj_weight", weight_mask)
            if module.in_proj_bias is not None:
                bias_mask = torch.ones_like(module.in_proj_bias)
                for projection in range(3):
                    bias_mask[rows + projection * heads * width] = 0
                ParameterMaskManager.apply(module, "in_proj_bias", bias_mask)
            output = module.out_proj
        else:
            for name in ("q_proj", "k_proj", "v_proj"):
                projection = getattr(module, name)
                weight_mask = torch.ones_like(projection.weight)
                weight_mask[rows, :] = 0
                ParameterMaskManager.apply(projection, "weight", weight_mask)
                if projection.bias is not None:
                    bias_mask = torch.ones_like(projection.bias)
                    bias_mask[rows] = 0
                    ParameterMaskManager.apply(projection, "bias", bias_mask)
            output = module.o_proj
        output_mask = torch.ones_like(output.weight)
        output_mask[:, rows] = 0
        ParameterMaskManager.apply(output, "weight", output_mask)

    def score_structural_block(
        self, target: StructuralBlockTarget, criterion: Any = None
    ) -> torch.Tensor:
        """Return a scalar block importance for deterministic selection."""
        values = [parameter.detach().abs().mean() for parameter in target.module.parameters()]
        if not values:
            return torch.zeros((), device=next(self.model.parameters()).device)
        return torch.stack(values).mean()

    def validate_structural_block_plan(
        self, targets: List[StructuralBlockTarget]
    ) -> Dict[str, str]:
        """Return target-name keyed errors before any structural mutation."""
        if not targets:
            return {}
        return {target.name: "Adapter does not declare structural block removal for this target." for target in targets}

    def order_structural_block_removals(
        self, targets: List[StructuralBlockTarget]
    ) -> List[StructuralBlockTarget]:
        return sorted(targets, key=lambda target: target.name)

    def remove_structural_block(self, target: StructuralBlockTarget) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not implement structural block removal.")

    def repair_structural_block_links(self, removed: List[StructuralBlockTarget]) -> None:
        """Repair model-family metadata after all requested blocks are removed."""

    def validate_structural_block_invariants(self) -> Optional[str]:
        """Return an invariant error, or ``None`` when the architecture is sound."""
        return None

    def get_synflow_input(self, device: torch.device) -> Any:
        """Return the all-ones synthetic input used by data-free SynFlow.

        Tensor-only adapters inherit this implementation. Adapters with
        keyword-only or multi-input forwards should override it explicitly.
        """
        dummy_input = self.get_dummy_input(device)
        if not isinstance(dummy_input, torch.Tensor):
            raise NotImplementedError(
                f"{type(self).__name__} must implement get_synflow_input for a non-Tensor model input."
            )
        return torch.ones_like(dummy_input)

    def reduce_synflow_output(self, output: Any) -> torch.Tensor:
        """Map a known model output structure to a SynFlow scalar objective."""
        if isinstance(output, torch.Tensor):
            return output.sum()
        raise ValueError(
            f"{type(self).__name__} does not define a SynFlow output reduction for {type(output).__name__}."
        )

    def snapshot_synflow_state(self) -> Any:
        """Snapshot non-state-dict forward state, when an adapter has any."""
        return None

    def restore_synflow_state(self, snapshot: Any) -> None:
        """Restore state returned by :meth:`snapshot_synflow_state`."""
        return None

    def prepare_for_pruning(self):
        """Optional pre-pruning initialization hook."""
        pass

    def post_prune_cleanup(self):
        """Optional post-pruning cleanup or head re-indexing hook."""
        pass
