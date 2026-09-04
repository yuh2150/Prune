import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from .exceptions import ConfigValidationException


@dataclass
class ModelConfig:
    name: str = "yolov5"
    weights: str = "yolov5s.pt"
    device: str = "cuda"


@dataclass
class PruningConfig:
    pruner: str = "structured"
    criterion: str = "l1"
    granularity: str = "channel"
    amount: float = 0.3
    layer_params: Optional[List[Any]] = None


@dataclass
class AnalysisConfig:
    sensitivity: bool = False
    layer_selection: bool = False


@dataclass
class BenchmarkConfig:
    enabled: bool = True
    runs: int = 100
    warmup: int = 10


@dataclass
class ExportConfig:
    enabled: bool = False
    format: str = "onnx"
    output_path: str = "pruned_model.onnx"


@dataclass
class FrameworkConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    output_path: str = "pruned_checkpoint.pt"

    @classmethod
    def from_yaml(cls, path: str) -> "FrameworkConfig":
        if not os.path.exists(path):
            raise ConfigValidationException(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        model_cfg = ModelConfig(**data.get("model", {}))
        prune_cfg = PruningConfig(**data.get("pruning", {}))
        analysis_cfg = AnalysisConfig(**data.get("analysis", {}))
        bench_cfg = BenchmarkConfig(**data.get("benchmark", {}))
        export_cfg = ExportConfig(**data.get("export", {}))

        return cls(
            model=model_cfg,
            pruning=prune_cfg,
            analysis=analysis_cfg,
            benchmark=bench_cfg,
            export=export_cfg,
            output_path=data.get("output_path", "pruned_checkpoint.pt")
        )
