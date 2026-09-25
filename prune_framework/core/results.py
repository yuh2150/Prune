from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class BenchmarkResult:
    inference_latency_ms: float = 0.0
    nms_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    fps: float = 0.0
    latency_std_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0


@dataclass
class ComplexityResult:
    params: int = 0
    trainable_params: int = 0
    flops: Optional[int] = None
    macs: Optional[int] = None
    model_size_mb: float = 0.0
    sparsity_pct: float = 0.0


@dataclass
class PruningResult:
    model_name: str
    pruner_name: str
    criterion_name: str
    granularity_name: str
    params_before: int
    params_after: int
    params_reduction_pct: float
    forward_verified: bool = False
    benchmark: Optional[BenchmarkResult] = None
    pruning_plan: Dict[str, Any] = field(default_factory=dict)
    architecture_validation: Dict[str, Any] = field(default_factory=dict)
    extra_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Result of the unified pipeline and paths of its reproducible artifacts."""

    run_dir: str
    pruning: PruningResult
    baseline_metrics: Optional[Dict[str, float]] = None
    final_metrics: Optional[Dict[str, float]] = None
    sensitivity: Optional[Dict[str, Any]] = None
    selection: Optional[Dict[str, Any]] = None
    complexity_before: Optional[ComplexityResult] = None
    complexity_after: Optional[ComplexityResult] = None
    artifacts: Dict[str, str] = field(default_factory=dict)
