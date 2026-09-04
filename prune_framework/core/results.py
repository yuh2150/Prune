from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class BenchmarkResult:
    inference_latency_ms: float = 0.0
    nms_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    fps: float = 0.0


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
    extra_metrics: Dict[str, Any] = field(default_factory=dict)
