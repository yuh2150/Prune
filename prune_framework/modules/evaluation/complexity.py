"""Model-cost measurements kept separate from latency benchmarking."""

from __future__ import annotations

import torch
import torch.nn as nn

from prune_framework.core.results import ComplexityResult
from .metrics import compute_model_size_mb, compute_sparsity


def measure_complexity(model: nn.Module, dummy_input: torch.Tensor, include_flops: bool = True) -> ComplexityResult:
    """Measure parameters always and FLOPs when THOP can trace this architecture.

    Some dynamic detection models cannot be traced by THOP. In that case the
    result deliberately records ``None`` rather than reporting a misleading 0.
    """
    params = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    result = ComplexityResult(
        params=params,
        trainable_params=trainable,
        model_size_mb=compute_model_size_mb(model),
        sparsity_pct=compute_sparsity(model),
    )
    if not include_flops:
        return result
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
        result.macs = int(macs)
        result.flops = int(macs * 2)
    except Exception:
        # Dynamic model control flow and unsupported custom ops are expected for
        # some detectors; the pipeline records the unavailable measurement.
        result.macs = None
        result.flops = None
    return result
