# Framework API Reference

This document provides a comprehensive API reference for all core packages, contracts, modules, and plugins in `prune_framework`.

---

## 1. Core Framework API (`prune_framework.core`)

### `PluginRegistry`
* **Import Path**: `from prune_framework.core.registry import PluginRegistry`
* **Purpose**: Centralized registry managing dynamic plugin discovery and instantiation.
* **Key Methods**:
  * `list_models() -> List[str]`: Returns names of registered model adapters.
  * `list_pruners() -> List[str]`: Returns names of registered pruners.
  * `list_criteria() -> List[str]`: Returns names of registered importance criteria.
  * `list_granularities() -> List[str]`: Returns names of registered granularities.
  * `list_selectors() -> List[str]`: Returns names of registered layer selectors.
  * `get_model_adapter(name: str) -> Type[BaseModelAdapter]`
  * `get_pruner(name: str) -> Type[BasePruner]`
  * `get_criterion(name: str) -> Type[BaseImportanceCriterion]`
  * `get_granularity(name: str) -> Type[BaseGranularity]`
  * `get_selector(name: str) -> Type[BaseSelector]`

---

### `PruningEngine`
* **Import Path**: `from prune_framework.core.engine import PruningEngine`
* **Purpose**: Orchestrates model composition, importance scoring, and pruning execution.
* **Constructor**:
  ```python
  PruningEngine(
      model_name: str,
      pruner_name: str,
      criterion_name: str,
      granularity_name: str = "channel"
  )
  ```
* **Methods**:
  * `execute(model: nn.Module, config: Dict[str, Any], verify_forward: bool = True) -> PruningResult`: Executes pruning on model and returns `PruningResult`.

---

### `FrameworkConfig`
* **Import Path**: `from prune_framework.core.config import FrameworkConfig`
* **Purpose**: Root configuration dataclass.
* **Classmethod**: `from_yaml(path: str) -> FrameworkConfig`
* **Attributes**:
  * `model: ModelConfig` (`name`, `weights`, `device`)
  * `pruning: PruningConfig` (`pruner`, `criterion`, `granularity`, `amount`, `layer_params`)
  * `analysis: AnalysisConfig` (`sensitivity`, `layer_selection`)
  * `benchmark: BenchmarkConfig` (`enabled`, `runs`, `warmup`)
  * `export: ExportConfig` (`enabled`, `format`, `output_path`)
  * `output_path: str`

---

## 2. Domain Contracts & Models (`prune_framework.contracts`)

### `SensitivityResult`
* **Import Path**: `from prune_framework.contracts import SensitivityResult`
* **Attributes**:
  * `baseline_score: float`
  * `profiles: Dict[int, LayerSensitivityProfile]`
  * `metadata: Dict[str, Any]`
* **Methods**:
  * `to_dict() -> Dict[int, Dict[float, float]]`: Converts to legacy nested dict.
  * `from_dict(raw_dict, baseline_score=1.0) -> SensitivityResult`: Factory method.

---

### `SelectionResult`
* **Import Path**: `from prune_framework.contracts import SelectionResult`
* **Attributes**:
  * `selected_layers: List[SelectedLayer]`
  * `strategy_name: str`
  * `target_sparsity: float`
* **Iteration**: Implements `__iter__` yielding `(layer_idx, target_rate)` tuples.

---

### `EvaluationResult` & `Detection`
* **Import Path**: `from prune_framework.contracts import EvaluationResult, Detection`
* **`Detection` Attributes**: `image_id: int`, `category_id: int`, `score: float`, `bbox: Tuple[float, float, float, float]`
* **`EvaluationResult` Attributes**: `metrics: Dict[str, float]`, `num_samples: int`, `metadata: Dict[str, Any]`
* **`EvaluationResult` Properties**: `.map`, `.map50`

---

## 3. Analysis & Evaluation Modules (`prune_framework.modules`)

### `SensitivityAnalyzer`
* **Import Path**: `from prune_framework.modules.analysis.sensitivity import SensitivityAnalyzer`
* **Constructor**:
  ```python
  SensitivityAnalyzer(
      model_name: str,
      pruner_name: str,
      criterion_name: str,
      granularity_name: str = "channel"
  )
  ```
* **Methods**:
  * `analyze(model: nn.Module, pruning_rates: List[float], eval_fn: Callable[[nn.Module], float], metric_direction: str = "higher_is_better") -> SensitivityResult`

---

### `LayerSelectorModule`
* **Import Path**: `from prune_framework.modules.analysis.layer_selection import LayerSelectorModule`
* **Constructor**: `LayerSelectorModule(selector_name: str = "sensitivity")`
* **Methods**:
  * `select(sensitivity_data: Union[SensitivityResult, Dict[int, Dict[float, float]]], target_sparsity: float, **kwargs: Any) -> SelectionResult`

---

### `EvaluationPipeline`
* **Import Path**: `from prune_framework.modules.evaluation.pipeline import EvaluationPipeline`
* **Constructor**:
  ```python
  EvaluationPipeline(
      preprocessor: BasePreProcessor,
      postprocessor: BasePostProcessor,
      evaluator: BaseEvaluator
  )
  ```
* **Methods**:
  * `run(model: nn.Module, dataloader: Any, device: torch.device, conf_thres: float = 0.001, half: bool = True, show_progress: bool = True) -> EvaluationResult`

---

### `RTDetrPreProcessor` & `RTDetrPostProcessor`
* **Import Path**: `from prune_framework.modules.evaluation.rtdetr_processors import RTDetrPreProcessor, RTDetrPostProcessor`
* **`RTDetrPreProcessor.process(images, device, half)`**: Prepares pixel values tensor.
* **`RTDetrPostProcessor.process(outputs, target_sizes, image_ids, threshold)`**: Returns `List[Detection]`.

---

### `COCOEvaluator`
* **Import Path**: `from prune_framework.modules.evaluation.coco_evaluator import COCOEvaluator`
* **Constructor**: `COCOEvaluator(coco_gt: Any, save_json_path: Optional[str] = None)`
* **Methods**: `evaluate(predictions: List[Detection], evaluated_img_ids: List[int]) -> EvaluationResult`
