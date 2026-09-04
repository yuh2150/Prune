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
