from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterator, Any


@dataclass(frozen=True)
class SensitivityPoint:
    """Represents a single sensitivity evaluation point for a layer at a specific pruning rate."""
    rate: float
    score: float
    metric_drop: float
    relative_drop: float
    is_valid: bool = True
    error_message: Optional[str] = None


@dataclass
class LayerSensitivityProfile:
    """Holds sensitivity points across multiple pruning rates for a specific model layer."""
    layer_idx: int
    layer_name: str
    points: Dict[float, SensitivityPoint] = field(default_factory=dict)


@dataclass
class SensitivityResult:
    """Immutable domain representation of complete sensitivity analysis results."""
    baseline_score: float
    profiles: Dict[int, LayerSensitivityProfile] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[int, Dict[float, float]]:
        """Converts to legacy nested dictionary format: layer_idx -> rate -> score."""
        return {
            idx: {rate: pt.score for rate, pt in profile.points.items() if pt.is_valid}
            for idx, profile in self.profiles.items()
        }

    @classmethod
    def from_dict(
        cls,
        raw_dict: Dict[int, Dict[float, float]],
        baseline_score: float = 1.0,
        metric_direction: str = "higher_is_better"
    ) -> "SensitivityResult":
        """Factory method to convert legacy raw nested dictionary into SensitivityResult."""
        profiles = {}
        for idx, rates_dict in raw_dict.items():
            profile = LayerSensitivityProfile(layer_idx=idx, layer_name=f"layer_{idx}")
            for rate, score in rates_dict.items():
                if metric_direction == "higher_is_better":
                    drop = baseline_score - score
                else:
                    drop = score - baseline_score
                rel_drop = drop / abs(baseline_score) if baseline_score != 0 else 0.0
                profile.points[rate] = SensitivityPoint(
                    rate=rate,
                    score=score,
                    metric_drop=drop,
                    relative_drop=rel_drop,
                    is_valid=True
                )
            profiles[idx] = profile
        return cls(baseline_score=baseline_score, profiles=profiles)


@dataclass(frozen=True)
class SelectedLayer:
    """Represents the pruning selection decision for a single layer."""
    layer_idx: int
    layer_name: str
    target_rate: float
    expected_metric: float
    expected_drop: float


class SelectionResult:
    """
    Immutable selection result container.
    Implements tuple iteration for 100% backward compatibility with legacy List[Tuple[int, float]] consumers.
    """

    def __init__(
        self,
        selected_layers: List[SelectedLayer],
        strategy_name: str = "",
        target_sparsity: float = 0.0
    ):
        self.selected_layers = selected_layers
        self.strategy_name = strategy_name
        self.target_sparsity = target_sparsity

    def __iter__(self) -> Iterator[Tuple[int, float]]:
        """Allows iterating as tuples of (layer_idx, target_rate) for backward compatibility."""
        for item in self.selected_layers:
            yield (item.layer_idx, item.target_rate)

    def __getitem__(self, index: int) -> Tuple[int, float]:
        """Allows index access as (layer_idx, target_rate) tuple for backward compatibility."""
        item = self.selected_layers[index]
        return (item.layer_idx, item.target_rate)

    def __len__(self) -> int:
        return len(self.selected_layers)

    def summary(self) -> Dict[str, Any]:
        """Returns a summarized dictionary representation of the selection outcome."""
        return {
            "strategy": self.strategy_name,
            "target_sparsity": self.target_sparsity,
            "num_selected_layers": len(self.selected_layers),
            "layers": [
                {
                    "layer_idx": l.layer_idx,
                    "layer_name": l.layer_name,
                    "rate": l.target_rate,
                    "expected_metric": l.expected_metric,
                    "expected_drop": l.expected_drop
                }
                for l in self.selected_layers
            ]
        }
