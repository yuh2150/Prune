import torch
import torch.nn as nn


class ModelValidator:
    """Validates structural integrity and forward pass compatibility of pruned models."""

    @staticmethod
    def validate_forward(model: nn.Module, dummy_input: torch.Tensor) -> bool:
        model.eval()
        try:
            with torch.no_grad():
                out = model(dummy_input)
            return out is not None
        except Exception as e:
            print(f"ModelValidator Validation Failed: {e}")
            return False
