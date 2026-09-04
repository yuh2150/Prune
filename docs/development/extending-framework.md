# Developer & Extension Guide

This guide explains how to extend the framework by adding new model adapters, layer selectors, pruners, importance criteria, and running test suites.

---

## 1. Adding a New Model Adapter

To support a new model architecture (e.g. `mobilenet`):

1. Create a new file under `prune_framework/plugins/adapters/mobilenet.py`.
2. Inherit from `BaseModelAdapter` and decorate with `@register_model_adapter("mobilenet")`.
3. Implement required abstract methods:

```python
import torch
import torch.nn as nn
from typing import List, Tuple, Any
from prune_framework.contracts import BaseModelAdapter
from prune_framework.core.registry import register_model_adapter


@register_model_adapter("mobilenet")
class MobileNetAdapter(BaseModelAdapter):
    """Adapter for MobileNet architecture."""

    def get_pruneable_modules(self) -> List[Tuple[str, nn.Module]]:
        """Returns list of (name, module) pairs that can be pruned."""
        pruneable = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                pruneable.append((name, module))
        return pruneable

    def get_dummy_input(self, device: torch.device) -> Any:
        """Returns dummy tensor for shape verification and profiling."""
        return torch.randn(1, 3, 224, 224, device=device)

    @classmethod
    def load_model(cls, weights: str, device: torch.device) -> Tuple[nn.Module, Any]:
        """Loads weights and returns (model, checkpoint_data)."""
        # Load weights logic...
        model = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=True).to(device)
        return model, None
```

4. Export your module in `prune_framework/plugins/adapters/__init__.py`.

---

## 2. Adding a New Layer Selector

To add a custom selection strategy (e.g., `pareto`):

1. Create or edit `prune_framework/plugins/selectors/selectors.py`.
2. Inherit from `BaseSelector` and decorate with `@register_selector("pareto")`.
3. Implement `select_layers`:

```python
from typing import Union, Dict, Any
from prune_framework.contracts import BaseSelector, SensitivityResult, SelectionResult, SelectedLayer
from prune_framework.core.registry import register_selector


@register_selector("pareto")
class ParetoSelector(BaseSelector):
    """Custom Pareto-frontier layer selection algorithm."""

    def select_layers(
        self,
        sensitivity_data: Union[SensitivityResult, Dict[int, Dict[float, float]]],
        target_sparsity: float,
        **kwargs: Any
    ) -> SelectionResult:
        if isinstance(sensitivity_data, dict):
            res = SensitivityResult.from_dict(sensitivity_data)
        else:
            res = sensitivity_data

        selected = []
        for idx, profile in res.profiles.items():
            # Apply custom Pareto logic...
            best_rate = 0.2
            selected.append(SelectedLayer(idx, profile.layer_name, best_rate, res.baseline_score, 0.0))

        return SelectionResult(selected, strategy_name="pareto", target_sparsity=target_sparsity)
```

---

## 3. Adding a New Pruning Strategy

To add a new pruner strategy (e.g., `gradual`):

1. Create a file under `prune_framework/plugins/pruners/`.
2. Inherit from `BasePruner` and decorate with `@register_pruner("gradual")`.
3. Implement `prune(self, model, params)` method.

---

## 4. Running Tests & Architecture Validation

The repository includes unit tests, contract tests, and architecture isolation tests under `tests/`.

### Running All Tests
Activate the Conda environment and run unittest:

```bash
conda activate env_cv
python -m unittest discover -s tests
```

### Key Test Suites
* **`tests/unit/test_registry.py`**: Verifies dynamic plugin registration and retrieval.
* **`tests/unit/test_sensitivity_analyzer.py`**: Verifies SA sweep, evaluation error handling, and baseline calculations.
* **`tests/unit/test_layer_selectors.py`**: Verifies `SensitivitySelector`, `GreedySelector`, `ThresholdSelector`, and backward compatibility.
* **`tests/unit/test_evaluation_components.py`**: Verifies `EvaluationPipeline`, `PreProcessor`, `PostProcessor`, and `COCOEvaluator`.
* **`tests/contract/test_plugin_contracts.py`**: Verifies all plugins satisfy contract interfaces.
* **`tests/architecture/test_dependency_rules.py`**: Enforces strict architecture rules preventing `prune_framework` from illegally importing legacy root scripts.
