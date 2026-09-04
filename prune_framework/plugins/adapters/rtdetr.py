import math
import random
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Any
from prune_framework.core.interfaces import BaseModelAdapter
from prune_framework.core.registry import register_model


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

    def get_dummy_input(self, device: torch.device) -> torch.Tensor:
        first_param = next(self.model.parameters())
        return torch.randn(1, 3, 640, 640, device=device, dtype=first_param.dtype)

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

    def prune_transformer_layers(self, prune_ratio: float = 0.2) -> nn.Module:
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
            keep_enc = sorted(random.sample(range(orig_enc), orig_enc - enc_drop))
            encoder.layers = nn.ModuleList([encoder.layers[i] for i in keep_enc])
            print(f" - RTDETRAdapter: Kept encoder layers {keep_enc}")

        if orig_dec > 0 and dec_drop > 0:
            keep_dec = sorted(random.sample(range(orig_dec), orig_dec - dec_drop))
            decoder.layers = nn.ModuleList([decoder.layers[i] for i in keep_dec])
            print(f" - RTDETRAdapter: Kept decoder layers {keep_dec}")

        self.model.num_encoder_layers = len(encoder.layers)
        self.model.num_decoder_layers = len(decoder.layers)
        if hasattr(self.model, "config"):
            self.model.config.encoder_layers = len(encoder.layers)
            self.model.config.decoder_layers = len(decoder.layers)

        return self.model
