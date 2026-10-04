from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Mapping


@dataclass(frozen=True)
class Detection:
    """Standardized framework representation of a single object detection prediction."""
    image_id: int
    category_id: int
    score: float
    bbox: Tuple[float, float, float, float]  # [x_min, y_min, width, height]


@dataclass(frozen=True)
class EvaluationResult:
    """Immutable contract representing evaluation metrics and metadata."""
    metrics: Dict[str, float]
    num_samples: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get(self, metric_name: str, default: float = 0.0) -> float:
        return self.metrics.get(metric_name, default)

    @property
    def map(self) -> float:
        return self.metrics.get("map", 0.0)

    @property
    def map50(self) -> float:
        return self.metrics.get("map50", 0.0)


class BasePreProcessor(ABC):
    """Abstract Base Class for model-specific input pre-processing."""

    @abstractmethod
    def process(self, images: Any, **kwargs: Any) -> Any:
        """Transforms raw images/batch into model input tensors."""
        pass


class BasePostProcessor(ABC):
    """Abstract Base Class for model-specific prediction post-processing."""

    @abstractmethod
    def process(self, outputs: Any, **kwargs: Any) -> List[List[Detection]]:
        """Transforms raw model output logits/boxes into standardized Detection objects per image."""
        pass


class BaseEvaluator(ABC):
    """Abstract Base Class for metric evaluation against ground truth."""

    @abstractmethod
    def evaluate(
        self,
        predictions: List[Detection],
        targets: Any,
        **kwargs: Any
    ) -> EvaluationResult:
        """Calculates evaluation metrics comparing predictions against ground truth targets."""
        pass


def normalize_evaluation(value, metric='map'):
    """Validate framework callbacks; detector tuples require an explicit adapter."""
    import math
    from numbers import Real
    if isinstance(value, EvaluationResult):
        metrics = value.metrics
    elif isinstance(value, Real) and not isinstance(value, bool):
        metrics = {metric: value}
    elif isinstance(value, dict):
        metrics = value
    else:
        raise TypeError('An evaluation callback must return a float, metric dict, or EvaluationResult.')
    if not metrics or any(not isinstance(k, str) or not isinstance(v, Real) or isinstance(v, bool)
                          or not math.isfinite(v) for k, v in metrics.items()):
        raise ValueError('Evaluation metrics must be a non-empty mapping of names to finite numbers.')
    normalized = {k: float(v) for k, v in metrics.items()}
    if 'map' in normalized and 'map50_95' in normalized and normalized['map'] != normalized['map50_95']:
        raise ValueError('map and map50_95 must agree')
    if 'map' in normalized:
        normalized.setdefault('map50_95', normalized['map'])
    if 'map50_95' in normalized:
        normalized.setdefault('map', normalized['map50_95'])
    return EvaluationResult(normalized, value.num_samples if isinstance(value, EvaluationResult) else 0,
                            value.metadata if isinstance(value, EvaluationResult) else {})


def normalize_yolo_evaluation(value, num_samples=0):
    """Strict bridge for the local YOLO evaluator's results/maps/times contract."""
    import numpy as np
    import math
    from numbers import Real
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError('YOLO evaluation must return (results, maps, times)')
    results, maps, times = value
    if not isinstance(results, (tuple, list)) or len(results) != 7:
        raise ValueError('YOLO results must contain exactly 7 metrics/losses')
    if not isinstance(times, (tuple, list)) or len(times) != 6:
        raise ValueError('YOLO times must contain exactly 6 values')
    if any(not isinstance(v, Real) or isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in times):
        raise ValueError('YOLO times must contain finite non-negative numbers')
    if not isinstance(maps, np.ndarray) or maps.ndim != 1 or not np.isfinite(maps).all():
        raise ValueError('YOLO maps must be a finite one-dimensional ndarray')
    keys = ('precision', 'recall', 'map50', 'map50_95', 'box_loss', 'obj_loss', 'cls_loss')
    metrics = dict(zip(keys, results))
    metrics.update(zip(('inference_ms', 'nms_ms', 'total_ms'), times[:3]))
    return normalize_evaluation(EvaluationResult(metrics, num_samples, {'per_class_ap': maps.tolist()}))
