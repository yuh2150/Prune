from typing import Dict, Union, Any
from prune_framework.core.registry import PluginRegistry
from prune_framework.contracts.sensitivity_result import SensitivityResult, SelectionResult


class LayerSelectorModule:
    """Module managing layer selection strategies based on sensitivity outputs."""

    def __init__(self, selector_name: str = "sensitivity"):
        self.selector_name = selector_name
        self.selector_cls = PluginRegistry.get_selector(selector_name)

    def select(
        self,
        sensitivity_data: Union[SensitivityResult, Dict[int, Dict[float, float]]],
        target_sparsity: float,
        **kwargs: Any
    ) -> SelectionResult:
        selector = self.selector_cls()
        return selector.select_layers(sensitivity_data, target_sparsity, **kwargs)

