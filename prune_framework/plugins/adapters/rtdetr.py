import math
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Any, Set
from prune_framework.core.interfaces import BaseModelAdapter
from prune_framework.core.registry import register_model
from prune_framework.contracts.targets import StructuralBlockTarget, TargetType


@register_model("rtdetr")
class RTDETRAdapter(BaseModelAdapter):
    """Adapter for RT-DETR Hugging Face object detection models."""

    @classmethod
    def load_model(cls, weights_path: str, device: torch.device) -> Tuple[nn.Module, Optional[Any]]:
        from transformers import RTDetrForObjectDetection
        model = RTDetrForObjectDetection.from_pretrained(weights_path)
        model.to(device)
        return model, None

    def get_ignored_patterns(self) -> List[str]:
        return [
            "model.backbone.model.embedder",
            "shortcut",
            "model.encoder_input_proj",
            "model.decoder_input_proj",
            "model.encoder.aifi",
            "model.decoder",
            "model.enc_score_head",
            "model.enc_bbox_head",
            "model.enc_output",
            "model.denoising_class_embed"
        ]

    def supported_target_types(self) -> Set[TargetType]:
        """Only the existing safe Conv path is exposed in P0.

        Decoder, attention, and all Linear projections remain excluded until a
        model-specific dependency implementation exists.
        """
        return {TargetType.CONV_WEIGHT, TargetType.CONV_OUT_CHANNEL}

    def supports_channel_sparsity_regularization(self) -> bool:
        """RT-DETR has no supported pruning-aware recovery path yet."""
        return False

    def get_pruneable_modules(self) -> List[Tuple[str, nn.Module]]:
        patterns = self.get_ignored_patterns()
        pruneable = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                if not any(p in name for p in patterns):
                    pruneable.append((name, module))
        return pruneable

    def get_pruneable_conv_layers(self) -> List[Tuple[str, nn.Conv2d]]:
        return [(name, mod) for name, mod in self.get_pruneable_modules() if isinstance(mod, nn.Conv2d)]

    def get_pruneable_blocks(self) -> List[Tuple[int, nn.Module]]:
        return []

    def get_structural_block_targets(self) -> List[StructuralBlockTarget]:
        """Expose transformer stack entries only when they use ModuleList stacks."""
        nested = getattr(self.model, "model", None)
        targets: List[StructuralBlockTarget] = []
        for kind in ("encoder", "decoder"):
            stack = getattr(getattr(nested, kind, None), "layers", None)
            if not isinstance(stack, nn.ModuleList) or len(stack) <= 1:
                continue
            owner_name = f"model.{kind}.layers"
            for index, layer in enumerate(stack):
                targets.append(
                    StructuralBlockTarget(
                        name=f"{owner_name}.{index}",
                        module=layer,
                        block_type=f"rtdetr_{kind}_layer",
                        owner_name=owner_name,
                        index=index,
                        metadata={"stack": kind},
                    )
                )
        return targets

    def validate_structural_block_plan(self, targets: List[StructuralBlockTarget]) -> dict[str, str]:
        errors: dict[str, str] = {}
        stacks: dict[str, list[StructuralBlockTarget]] = {}
        nested = getattr(self.model, "model", None)
        for target in targets:
            kind = target.metadata.get("stack")
            layers = getattr(getattr(nested, kind, None), "layers", None)
            if kind not in {"encoder", "decoder"} or not isinstance(layers, nn.ModuleList):
                errors[target.name] = "RT-DETR structural target does not belong to a declared ModuleList stack."
                continue
            if target.index < 0 or target.index >= len(layers) or layers[target.index] is not target.module:
                errors[target.name] = "RT-DETR structural layer no longer matches its declared stack position."
                continue
            stacks.setdefault(kind, []).append(target)
        for kind, selected in stacks.items():
            layers = getattr(getattr(nested, kind), "layers")
            if len(layers) - len(selected) < 1:
                for target in selected:
                    errors[target.name] = f"RT-DETR {kind} stack must retain at least one layer."
        return errors

    def order_structural_block_removals(self, targets: List[StructuralBlockTarget]) -> List[StructuralBlockTarget]:
        return sorted(targets, key=lambda target: (target.owner_name, -target.index))

    def remove_structural_block(self, target: StructuralBlockTarget) -> None:
        kind = target.metadata.get("stack")
        nested = getattr(self.model, "model", None)
        layers = getattr(getattr(nested, kind, None), "layers", None)
        if not isinstance(layers, nn.ModuleList) or target.index >= len(layers) or layers[target.index] is not target.module:
            raise RuntimeError(f"Cannot remove stale RT-DETR structural layer '{target.name}'.")
        del layers[target.index]

    def repair_structural_block_links(self, removed: List[StructuralBlockTarget]) -> None:
        del removed
        nested = getattr(self.model, "model", None)
        for kind, field in (("encoder", "num_encoder_layers"), ("decoder", "num_decoder_layers")):
            layers = getattr(getattr(nested, kind, None), "layers", None)
            if not isinstance(layers, nn.ModuleList):
                continue
            for owner in (self.model, nested, getattr(self.model, "config", None)):
                if owner is not None and hasattr(owner, field):
                    setattr(owner, field, len(layers))
            config_field = "encoder_layers" if kind == "encoder" else "decoder_layers"
            config = getattr(self.model, "config", None)
            if config is not None and hasattr(config, config_field):
                setattr(config, config_field, len(layers))

    def validate_structural_block_invariants(self) -> Optional[str]:
        nested = getattr(self.model, "model", None)
        for kind, field in (("encoder", "num_encoder_layers"), ("decoder", "num_decoder_layers")):
            layers = getattr(getattr(nested, kind, None), "layers", None)
            if layers is None:
                continue
            if not isinstance(layers, nn.ModuleList) or len(layers) < 1:
                return f"RT-DETR {kind} layers must remain a non-empty ModuleList."
            for owner in (self.model, nested):
                if owner is not None and hasattr(owner, field) and getattr(owner, field) != len(layers):
                    return f"RT-DETR {field} is inconsistent with its ModuleList length."
        return None

    def get_dummy_input(self, device: torch.device) -> torch.Tensor:
        first_param = next(self.model.parameters())
        return torch.randn(1, 3, 640, 640, device=device, dtype=first_param.dtype)

    def reduce_synflow_output(self, output: Any) -> torch.Tensor:
        """Reduce only the documented RT-DETR detection outputs explicitly."""
        logits = getattr(output, "logits", None)
        boxes = getattr(output, "pred_boxes", None)
        if isinstance(output, dict):
            logits = output.get("logits", logits)
            boxes = output.get("pred_boxes", boxes)
        if isinstance(logits, torch.Tensor) and isinstance(boxes, torch.Tensor):
            return logits.sum() + boxes.sum()
        raise ValueError(
            "RT-DETR SynFlow requires an output with Tensor 'logits' and 'pred_boxes'; "
            f"received {type(output).__name__}."
        )

    def prepare_for_pruning(self):
        try:
            from transformers.models.rt_detr.modeling_rt_detr import RTDetrFrozenBatchNorm2d
        except ImportError:
            return

        def set_module(parent, name, child):
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], child)

        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, RTDetrFrozenBatchNorm2d):
                bn = nn.BatchNorm2d(module.weight.shape[0]).to(device=device, dtype=dtype)
                if hasattr(module, "weight") and module.weight is not None:
                    bn.weight.data.copy_(module.weight.data)
                if hasattr(module, "bias") and module.bias is not None:
                    bn.bias.data.copy_(module.bias.data)
                if hasattr(module, "running_mean") and module.running_mean is not None:
                    bn.running_mean.data.copy_(module.running_mean.data)
                if hasattr(module, "running_var") and module.running_var is not None:
                    bn.running_var.data.copy_(module.running_var.data)
                set_module(self.model, name, bn)
                count += 1
        if count > 0:
            print(f" - RTDETRAdapter: Replaced {count} FrozenBatchNorm2d layers with BatchNorm2d.")

    def prune_transformer_layers(self, prune_ratio: float = 0.2, criterion=None) -> nn.Module:
        """Remove the least important encoder/decoder blocks deterministically.

        Layer deletion cannot use a channel dependency graph, but it must still
        be reproducible and criterion-driven. We score all eligible linear or
        convolution projections in each transformer block and retain blocks with
        the greatest average score. Criteria that cannot score a block fall back
        to its parameter L2 norm.
        """
        if not hasattr(self.model, "model"):
            return self.model

        nested = self.model.model
        if not hasattr(nested, "encoder") or not hasattr(nested, "decoder"):
            return self.model

        encoder, decoder = nested.encoder, nested.decoder
        orig_enc = len(encoder.layers) if hasattr(encoder, "layers") else 0
        orig_dec = len(decoder.layers) if hasattr(decoder, "layers") else 0

        enc_drop = math.ceil(orig_enc * prune_ratio) if orig_enc > 0 else 0
        dec_drop = math.ceil(orig_dec * prune_ratio) if orig_dec > 0 else 0

        enc_drop = min(enc_drop, orig_enc - 1) if orig_enc > 1 else 0
        dec_drop = min(dec_drop, orig_dec - 1) if orig_dec > 1 else 0

        if orig_enc > 0 and enc_drop > 0:
            keep_enc = self._select_blocks_to_keep(encoder.layers, orig_enc - enc_drop, criterion)
            encoder.layers = nn.ModuleList([encoder.layers[i] for i in keep_enc])
            print(f" - RTDETRAdapter: Kept encoder layers {keep_enc}")

        if orig_dec > 0 and dec_drop > 0:
            keep_dec = self._select_blocks_to_keep(decoder.layers, orig_dec - dec_drop, criterion)
            decoder.layers = nn.ModuleList([decoder.layers[i] for i in keep_dec])
            print(f" - RTDETRAdapter: Kept decoder layers {keep_dec}")

        self.model.num_encoder_layers = len(encoder.layers)
        self.model.num_decoder_layers = len(decoder.layers)
        if hasattr(self.model, "config"):
            self.model.config.encoder_layers = len(encoder.layers)
            self.model.config.decoder_layers = len(decoder.layers)

        return self.model

    @staticmethod
    def _select_blocks_to_keep(blocks, keep_count: int, criterion) -> List[int]:
        scored_blocks = []
        for index, block in enumerate(blocks):
            scores = []
            for module in block.modules():
                if not isinstance(module, (nn.Linear, nn.Conv2d)):
                    continue
                try:
                    score = criterion.score(module) if criterion is not None else None
                    if score is not None:
                        scores.append(float(score.detach().mean()))
                except (RuntimeError, ValueError):
                    continue
            if not scores:
                parameter_norms = [parameter.detach().norm().item() for parameter in block.parameters()]
                scores = [sum(parameter_norms) / max(1, len(parameter_norms))]
            scored_blocks.append((sum(scores) / len(scores), index))
        # Preserve module order after selecting the top-scoring blocks.
        return sorted(index for _, index in sorted(scored_blocks, key=lambda item: (-item[0], item[1]))[:keep_count])
