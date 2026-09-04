import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Any
from prune_framework.core.interfaces import BaseModelAdapter
from prune_framework.core.registry import register_model


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
                if any(head in name.lower() for head in ["detect", "segment", "anchor", "m.0", "m.1", "m.2"]):
                    continue
                pruneable.append((name, module))
        return pruneable

    def get_pruneable_conv_layers(self) -> List[Tuple[str, nn.Conv2d]]:
        """Alias for get_pruneable_modules for Conv2d layers."""
        return [(name, mod) for name, mod in self.get_pruneable_modules() if isinstance(mod, nn.Conv2d)]

    def get_pruneable_blocks(self) -> List[Tuple[int, nn.Module]]:
        pruneable = []
        if hasattr(self.model, "model") and isinstance(self.model.model, nn.Sequential):
            for i, module in enumerate(self.model.model):
                cls_name = module.__class__.__name__
                if cls_name in ["C3", "BottleneckCSP", "C3TR"] and hasattr(module, "m"):
                    pruneable.append((i, module))
        return pruneable

    def get_dummy_input(self, device: torch.device) -> torch.Tensor:
        first_param = next(self.model.parameters())
        return torch.randn(1, 3, 640, 640, device=device, dtype=first_param.dtype)
