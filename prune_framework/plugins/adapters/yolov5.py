import torch
import torch.nn as nn
from typing import Callable, Iterable, List, Tuple, Optional, Any, Set
from prune_framework.core.interfaces import BaseModelAdapter
from prune_framework.core.registry import register_model
from prune_framework.contracts.targets import PrunableTarget, StructuralBlockTarget, TargetType


@register_model("yolov5")
@register_model("yolov7")
class YOLOv5Adapter(BaseModelAdapter):
    """Adapter for YOLOv5 and YOLOv7 detection models."""

    @classmethod
    def load_model(cls, weights_path: str, device: torch.device) -> Tuple[nn.Module, Optional[Any]]:
        from models.experimental import attempt_download
        attempt_download(weights_path)
        try:
            ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(weights_path, map_location=device)
        model = ckpt["ema" if ckpt.get("ema") else "model"].to(device)
        return model, ckpt

    def get_pruneable_modules(self) -> List[Tuple[str, nn.Module]]:
        pruneable = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                if self._is_protected_name(name):
                    continue
                pruneable.append((name, module))
        return pruneable

    @staticmethod
    def _is_protected_name(name: str) -> bool:
        return any(head in name.lower() for head in ["detect", "segment", "anchor", "m.0", "m.1", "m.2"])

    def supported_target_types(self) -> Set[TargetType]:
        """YOLO exposes Conv and non-head Linear weights in the P0 target API.

        Only Conv output features are structurally mutable today; Linear output
        features are declared so future structural support need not change the
        adapter contract.
        """
        return {
            TargetType.CONV_WEIGHT,
            TargetType.CONV_OUT_CHANNEL,
            TargetType.LINEAR_WEIGHT,
            TargetType.LINEAR_OUT_FEATURE,
        }

    def get_prunable_targets(self, target_types: Optional[Iterable[TargetType]] = None) -> List[PrunableTarget]:
        allowed = set(target_types) if target_types is not None else self.supported_target_types()
        allowed &= self.supported_target_types()
        targets: List[PrunableTarget] = []
        for name, module in self.model.named_modules():
            if self._is_protected_name(name):
                continue
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

    def get_pruneable_conv_layers(self) -> List[Tuple[str, nn.Conv2d]]:
        """Alias for get_pruneable_modules for Conv2d layers."""
        return [(name, mod) for name, mod in self.get_pruneable_modules() if isinstance(mod, nn.Conv2d)]

    def supports_gradient_calibration(self) -> bool:
        return True

    def supports_channel_sparsity_regularization(self) -> bool:
        """YOLO Conv wrappers expose affine BatchNorm scales for slimming."""
        return True

    def get_gradient_calibration_batches(self, batches: int, device: torch.device, seed: int, batch_size: int = 1):
        """Create deterministic no-label batches for first-order calibration.

        The standard YOLO loss still contains objectness terms for empty labels,
        so this is a valid self-contained calibration path. Experiments that
        need dataset-representative saliency can call the runner directly with
        real YOLO dataloader batches without changing the pruning API.
        """
        if batches < 1 or batch_size < 1:
            raise ValueError("YOLO gradient calibration requires at least one batch.")
        first_param = next(self.model.parameters())
        generator = torch.Generator(device=device.type)
        generator.manual_seed(seed)
        shape = (batch_size, 3, 640, 640)
        return [
            (
                torch.randn(shape, generator=generator, device=device, dtype=first_param.dtype),
                torch.zeros((0, 6), device=device, dtype=first_param.dtype),
            )
            for _ in range(batches)
        ]

    def build_gradient_calibration_loss(self) -> Callable[[nn.Module, Any], torch.Tensor]:
        """Reuse the repository's normal YOLO detection loss without training."""
        from utils.loss import ComputeLoss

        compute_loss = ComputeLoss(self.model)

        def loss_fn(model: nn.Module, batch: Any) -> torch.Tensor:
            images, targets = batch
            predictions = model(images)
            loss, _ = compute_loss(predictions, targets)
            return loss

        return loss_fn

    def reduce_synflow_output(self, output: Any) -> torch.Tensor:
        """Use YOLO's raw prediction tensors, never decoded inference outputs."""
        if isinstance(output, torch.Tensor):
            return output.sum()
        tensors = output
        # YOLO inference returns ``(decoded_predictions, raw_predictions)``.
        if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], (list, tuple)):
            tensors = output[1]
        if isinstance(tensors, (list, tuple)) and tensors and all(isinstance(item, torch.Tensor) for item in tensors):
            return sum(item.sum() for item in tensors)
        raise ValueError(
            "YOLO SynFlow requires a Tensor, raw prediction list, or "
            "(decoded_predictions, raw_predictions) output."
        )

    def snapshot_synflow_state(self) -> Any:
        """Preserve Detect grids, which YOLO evaluation may update lazily."""
        snapshot = {}
        for name, module in self.model.named_modules():
            if not (hasattr(module, "grid") and hasattr(module, "anchor_grid")):
                continue
            snapshot[name] = (self._clone_synflow_value(module.grid), self._clone_synflow_value(module.anchor_grid))
        return snapshot

    def restore_synflow_state(self, snapshot: Any) -> None:
        for name, (grid, anchor_grid) in (snapshot or {}).items():
            module = dict(self.model.named_modules()).get(name)
            if module is not None:
                module.grid = grid
                module.anchor_grid = anchor_grid

    @staticmethod
    def _clone_synflow_value(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if isinstance(value, list):
            return [YOLOv5Adapter._clone_synflow_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(YOLOv5Adapter._clone_synflow_value(item) for item in value)
        return value

    def get_importance_module(self, name: str, module: nn.Module) -> nn.Module:
        """Use a YOLO Conv wrapper for BN-based importance when available."""
        if "." not in name:
            return module
        parent_name = name.rsplit(".", 1)[0]
        parent = dict(self.model.named_modules()).get(parent_name)
        if parent is not None and getattr(parent, "conv", None) is module and hasattr(parent, "bn"):
            return parent
        return module

    def get_pruneable_blocks(self) -> List[Tuple[int, nn.Module]]:
        pruneable = []
        if hasattr(self.model, "model") and isinstance(self.model.model, nn.Sequential):
            for i, module in enumerate(self.model.model):
                cls_name = module.__class__.__name__
                if cls_name in ["C3", "BottleneckCSP", "C3TR"] and hasattr(module, "m"):
                    pruneable.append((i, module))
        return pruneable

    def get_structural_block_targets(self) -> List[StructuralBlockTarget]:
        """Expose only internal C3/CSP Bottlenecks with a safe sequential owner.

        Removing one of these blocks preserves the enclosing C3 tensor shape;
        deleting a top-level detection graph node is deliberately unsupported.
        """
        targets: List[StructuralBlockTarget] = []
        supported = {"C3", "BottleneckCSP"}
        for name, module in self.model.named_modules():
            blocks = getattr(module, "m", None)
            if type(module).__name__ not in supported or not isinstance(blocks, nn.Sequential) or len(blocks) <= 1:
                continue
            owner_name = f"{name}.m" if name else "m"
            for index, block in enumerate(blocks):
                targets.append(
                    StructuralBlockTarget(
                        name=f"{owner_name}.{index}",
                        module=block,
                        block_type="yolo_csp_bottleneck",
                        owner_name=owner_name,
                        index=index,
                    )
                )
        return targets

    def score_structural_block(self, target: StructuralBlockTarget, criterion=None) -> torch.Tensor:
        """Aggregate criterion scores over a Bottleneck's projection modules."""
        scores = []
        for module in target.module.modules():
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue
            try:
                score = criterion.score(module) if criterion is not None else None
                if score is not None:
                    scores.append(score.detach().float().mean())
            except (RuntimeError, ValueError, AttributeError):
                continue
        if scores:
            return torch.stack(scores).mean()
        return super().score_structural_block(target, criterion)

    def validate_structural_block_plan(self, targets: List[StructuralBlockTarget]) -> dict[str, str]:
        errors: dict[str, str] = {}
        owners: dict[str, list[StructuralBlockTarget]] = {}
        modules = dict(self.model.named_modules())
        for target in targets:
            owner = modules.get(target.owner_name)
            if target.block_type != "yolo_csp_bottleneck" or not isinstance(owner, nn.Sequential):
                errors[target.name] = "YOLO target is not an adapter-declared C3/CSP bottleneck sequence entry."
                continue
            if target.index < 0 or target.index >= len(owner) or owner[target.index] is not target.module:
                errors[target.name] = "YOLO structural block no longer matches its declared sequential owner."
                continue
            owners.setdefault(target.owner_name, []).append(target)
        for owner_name, selected in owners.items():
            owner = modules[owner_name]
            if len(owner) - len(selected) < 1:
                for target in selected:
                    errors[target.name] = "YOLO C3/CSP owner must retain at least one Bottleneck block."
        return errors

    def order_structural_block_removals(self, targets: List[StructuralBlockTarget]) -> List[StructuralBlockTarget]:
        return sorted(targets, key=lambda target: (target.owner_name, -target.index))

    def remove_structural_block(self, target: StructuralBlockTarget) -> None:
        owner = dict(self.model.named_modules()).get(target.owner_name)
        if not isinstance(owner, nn.Sequential) or target.index >= len(owner) or owner[target.index] is not target.module:
            raise RuntimeError(f"Cannot remove stale YOLO structural block '{target.name}'.")
        retained = [block for index, block in enumerate(owner) if index != target.index]
        was_training = owner.training
        owner.__init__(*retained)
        owner.train(was_training)

    def validate_structural_block_invariants(self) -> Optional[str]:
        for name, module in self.model.named_modules():
            if type(module).__name__ not in {"C3", "BottleneckCSP"}:
                continue
            blocks = getattr(module, "m", None)
            if not isinstance(blocks, nn.Sequential) or len(blocks) < 1:
                return f"YOLO structural owner '{name}.m' must remain a non-empty Sequential."
        return None

    def get_dummy_input(self, device: torch.device) -> torch.Tensor:
        first_param = next(self.model.parameters())
        return torch.randn(1, 3, 640, 640, device=device, dtype=first_param.dtype)
