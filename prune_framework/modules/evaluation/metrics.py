import torch
import torch.nn as nn
from typing import Dict, Any


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def compute_sparsity(model: nn.Module) -> float:
    total = 0
    zeros = 0
    for p in model.parameters():
        total += p.numel()
        zeros += (p == 0).sum().item()
    return (zeros / total * 100.0) if total > 0 else 0.0


def compute_model_size_mb(model: nn.Module) -> float:
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)
