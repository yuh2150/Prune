"""Typed, backwards-compatible configuration for pruning experiments."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .exceptions import ConfigValidationException


@dataclass
class ModelConfig:
    name: str = "yolov5"
    weights: str = "yolov5s.pt"
    device: str = "cuda"
    input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640)


@dataclass
class PruningConfig:
    """Pruner settings. Legacy names remain the stored canonical fields."""

    pruner: str = "structured"
    criterion: str = "l1"
    granularity: str = "channel"
    amount: float = 0.3
    global_pruning: bool = False
    layer_params: Optional[List[Any]] = None
    iterative_steps: int = 1
    min_channels: int = 2
    calibration_callback: Optional[str] = None
    calibration_batches: int = 1
    calibration_batch_size: int = 1
    calibration_seed: int = 42
    calibration_accumulate: bool = True

    @property
    def method(self) -> str:
        return self.pruner

    @method.setter
    def method(self, value: str) -> None:
        self.pruner = value

    @property
    def structure(self) -> str:
        return self.granularity

    @structure.setter
    def structure(self, value: str) -> None:
        self.granularity = value

    @property
    def target_ratio(self) -> float:
        return self.amount

    @target_ratio.setter
    def target_ratio(self, value: float) -> None:
        self.amount = value


@dataclass
class AnalysisConfig:
    """Compatibility section retained for the original YAML files."""

    sensitivity: bool = False
    layer_selection: bool = False


@dataclass
class SensitivityConfig:
    enabled: bool = False
    rates: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5])
    selector: str = "sensitivity"
    max_allowed_relative_drop: float = 0.05
    metric_direction: str = "higher_is_better"


@dataclass
class EvaluationConfig:
    enabled: bool = False
    callback: Optional[str] = None
    metric: str = "map"
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryConfig:
    enabled: bool = False
    epochs: int = 0
    callback: Optional[str] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegularizationConfig:
    """Train-time sparsity settings run before structural mutation."""

    enabled: bool = False
    term: str = "l1_bn_scale"
    strength: float = 0.0
    schedule: str = "constant"
    warmup_steps: int = 0
    total_steps: int = 0
    pruning_criterion: str = "bn_scale"
    gate_beta: float = 2.0 / 3.0
    gate_gamma: float = -0.1
    gate_zeta: float = 1.1
    gate_log_alpha_init: float = 0.0
    gate_threshold: float = 0.5


@dataclass
class BenchmarkConfig:
    enabled: bool = True
    latency: bool = True
    flops: bool = True
    params: bool = True
    runs: int = 100
    warmup: int = 10


@dataclass
class ExportConfig:
    enabled: bool = False
    onnx: bool = False
    format: str = "onnx"
    output_path: str = "pruned_model.onnx"
    opset_version: int = 12

    def wants_onnx(self) -> bool:
        return self.enabled and (self.onnx or self.format.lower() == "onnx")


@dataclass
class ExperimentConfig:
    name: str = "pruning"
    output_dir: str = "artifacts"
    seed: int = 42
    deterministic: bool = True


@dataclass
class FrameworkConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    output_path: str = "pruned_checkpoint.pt"

    def validate(self) -> None:
        if not 0.0 <= self.pruning.amount < 1.0:
            raise ConfigValidationException("pruning.target_ratio/amount must be in [0, 1).")
        if self.pruning.iterative_steps < 1:
            raise ConfigValidationException("pruning.iterative_steps must be at least 1.")
        if self.pruning.min_channels < 1:
            raise ConfigValidationException("pruning.min_channels must be at least 1.")
        if self.pruning.calibration_batches < 1:
            raise ConfigValidationException("pruning.calibration_batches must be at least 1.")
        if self.pruning.calibration_batch_size < 1:
            raise ConfigValidationException("pruning.calibration_batch_size must be at least 1.")
        if self.recovery.enabled and self.recovery.epochs < 1:
            raise ConfigValidationException("recovery.epochs must be at least 1 when recovery is enabled.")
        if self.regularization.strength < 0:
            raise ConfigValidationException("regularization.strength must be non-negative.")
        if self.regularization.schedule not in {"constant", "linear_warmup", "cosine_decay"}:
            raise ConfigValidationException("regularization.schedule must be constant, linear_warmup, or cosine_decay.")
        if self.regularization.warmup_steps < 0 or self.regularization.total_steps < 0:
            raise ConfigValidationException("regularization warmup_steps and total_steps must be non-negative.")
        if self.regularization.gate_beta <= 0 or self.regularization.gate_gamma >= 0 or self.regularization.gate_zeta <= 1:
            raise ConfigValidationException("Hard-Concrete requires gate_beta > 0, gate_gamma < 0, and gate_zeta > 1.")
        if not 0 <= self.regularization.gate_threshold <= 1:
            raise ConfigValidationException("regularization.gate_threshold must be in [0, 1].")
        if self.sensitivity.enabled and not self.sensitivity.rates:
            raise ConfigValidationException("sensitivity.rates cannot be empty when sensitivity is enabled.")
        if any(rate <= 0 or rate >= 1 for rate in self.sensitivity.rates):
            raise ConfigValidationException("Every sensitivity rate must be in (0, 1).")
        if self.benchmark.runs < 1 or self.benchmark.warmup < 0:
            raise ConfigValidationException("benchmark.runs must be positive and benchmark.warmup non-negative.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str) -> "FrameworkConfig":
        if not os.path.exists(path):
            raise ConfigValidationException(f"Configuration file not found: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ConfigValidationException("The root YAML value must be a mapping.")
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrameworkConfig":
        sections = dict(data)
        pruning = dict(sections.get("pruning", {}))
        cls._rename(pruning, "method", "pruner")
        cls._rename(pruning, "structure", "granularity")
        cls._rename(pruning, "target_ratio", "amount")
        cls._rename(pruning, "global", "global_pruning")

        export = dict(sections.get("export", {}))
        if "onnx" in export and "enabled" not in export:
            export["enabled"] = bool(export["onnx"])

        analysis = dict(sections.get("analysis", {}))
        sensitivity = dict(sections.get("sensitivity", {}))
        if "enabled" not in sensitivity:
            sensitivity["enabled"] = bool(analysis.get("sensitivity", False))

        config = cls(
            model=ModelConfig(**dict(sections.get("model", {}))),
            pruning=PruningConfig(**pruning),
            analysis=AnalysisConfig(**analysis),
            sensitivity=SensitivityConfig(**sensitivity),
            evaluation=EvaluationConfig(**dict(sections.get("evaluation", {}))),
            recovery=RecoveryConfig(**dict(sections.get("recovery", {}))),
            regularization=RegularizationConfig(**dict(sections.get("regularization", {}))),
            benchmark=BenchmarkConfig(**dict(sections.get("benchmark", {}))),
            export=ExportConfig(**export),
            experiment=ExperimentConfig(**dict(sections.get("experiment", {}))),
            output_path=sections.get("output_path", "pruned_checkpoint.pt"),
        )
        config.validate()
        return config

    @staticmethod
    def _rename(values: Dict[str, Any], old: str, new: str) -> None:
        if old in values:
            if new in values and values[new] != values[old]:
                raise ConfigValidationException(f"Specify only one of '{old}' and '{new}'.")
            values[new] = values.pop(old)
