from abc import ABC, abstractmethod
from typing import Dict, Union, Any
from prune_framework.contracts.sensitivity_result import SensitivityResult, SelectionResult


class BaseSelector(ABC):
    """Abstract Base Class for Layer Selection Strategies."""

    @abstractmethod
    def select_layers(
        self,
        sensitivity_data: Union[SensitivityResult, Dict[int, Dict[float, float]]],
        target_sparsity: float,
        **kwargs: Any
    ) -> SelectionResult:
        """Determines layer indices and pruning ratios based on sensitivity data."""
        pass

