# Getting Started & Quickstart Guide

This guide covers environment installation, repository structure, CLI execution, and Python pipeline API usage for the **Model Pruning Framework**.

---

## 1. Installation & Environment Setup

### Environment Requirements
* Linux OS (Ubuntu 20.04/22.04 recommended)
* Python 3.10+ (tested on Python 3.10 and 3.13)
* PyTorch 2.0+ with CUDA support
* Conda / Miniconda

### Activating Environment
Activate the pre-configured Conda environment containing all required dependencies (`torch`, `torchvision`, `transformers`, `pycocotools`, `pyyaml`, `tqdm`, `thop`):

```bash
conda activate env_cv
```

---

## 2. Repository Structure

The codebase is organized into a modular framework package (`prune_framework`), CLI entrypoints, dataset loaders, configuration files, and test suites:

```text
/home/huy/Projects/Prune/
├── prune_framework/                  # Core Framework Package
│   ├── contracts/                    # Base Interface Contracts & Dataclasses
│   │   ├── adapter.py                # BaseModelAdapter
│   │   ├── pruner.py                 # BasePruner
│   │   ├── criterion.py              # BaseImportanceCriterion
│   │   ├── granularity.py            # BaseGranularity
│   │   ├── selector.py               # BaseSelector
│   │   ├── sensitivity_result.py     # SensitivityResult, SelectionResult, etc.
│   │   └── evaluation.py             # Detection, EvaluationResult, BasePre/PostProcessor
│   ├── core/                         # Core Registry, Engine, Config & Logging
│   │   ├── config.py                 # FrameworkConfig dataclasses
│   │   ├── engine.py                 # PruningEngine composition logic
│   │   ├── exceptions.py             # Domain exception hierarchy
│   │   ├── logging.py                # get_logger() logging setup
│   │   ├── registry.py               # PluginRegistry & decorators
│   │   └── results.py                # PruningResult dataclass
│   ├── modules/                      # Sub-system Modules
│   │   ├── analysis/                 # SensitivityAnalyzer, LayerSelectorModule
│   │   ├── evaluation/               # RTDetrPre/PostProcessors, COCOEvaluator, Pipeline
│   │   ├── export/                   # ModelExporter (ONNX & Checkpoints)
│   │   └── model/                    # ModelLoader
│   ├── pipelines/                    # Top-Level Pipeline Orchestrators
│   │   ├── pruning.py                # run_pruning_pipeline
│   │   ├── sensitivity.py            # run_sensitivity_pipeline
│   │   └── benchmarking.py           # run_benchmark_pipeline
│   └── plugins/                      # Plugin Implementations
│       ├── adapters/                 # YOLOv5Adapter, RTDetrAdapter, YOLOv7 compatibility alias
│       ├── criteria/                 # L1Norm, L2Norm, TaylorExpansion, Random
│       ├── granularities/            # Channel, Filter, Layer
│       ├── pruners/                  # Structured, Unstructured, Depth, Taylor
│       └── selectors/                # SensitivitySelector, GreedySelector, ThresholdSelector
├── configs/                          # Sample YAML Configurations
│   ├── yolov5_structured.yaml
│   ├── yolov5_unstructured.yaml
│   └── rtdetr_structured.yaml
├── tests/                            # Comprehensive Test Suite
│   ├── unit/                         # Unit tests for registry, selectors, SA, evaluation
│   ├── contract/                     # Plugin inheritance contract verification
│   └── architecture/                 # Dependency isolation architecture rules
├── main.py                           # CLI Entrypoint for Framework
├── test_rtdetr.py                    # RT-DETR Standalone & Pipeline Evaluation CLI
├── test.py                           # YOLOv5 Validation Stub
└── train.py                          # Fine-tuning & Training Entrypoint
```

---

## 3. Command Line Interface (CLI) Quickstart

The framework provides a unified CLI entrypoint via `main.py`.

### A. List Registered Plugins
List all dynamically discovered plugins in the registry:

```bash
python main.py --list-models
python main.py --list-pruners
python main.py --list-criteria
python main.py --list-granularities
python main.py --list-selectors
```

### B. Run Pruning Using YAML Config
Execute pruning using pre-defined configuration files in `configs/`:

```bash
python main.py --config configs/yolov5_structured.yaml
```

### C. Run Pruning Using CLI Overrides
Override configuration parameters directly from command line arguments:

```bash
python main.py \
  --model yolov5 \
  --weights weights/yolov5s.pt \
  --strategy structured \
  --criterion l1 \
  --granularity channel \
  --amount 0.3 \
  --output-path weights/yolov5s_pruned.pt
```

---

## 4. Python API Quickstart

### A. End-to-End Model Pruning

```python
import torch
from prune_framework.core.config import FrameworkConfig
from prune_framework.pipelines.pruning import run_pruning_pipeline

# 1. Load configuration
config = FrameworkConfig.from_yaml("configs/yolov5_structured.yaml")
config.pruning.amount = 0.25
config.output_path = "weights/yolov5s_pruned_25.pt"

# 2. Execute pruning pipeline
result = run_pruning_pipeline(config)

print(f"Parameters Before: {result.params_before:,}")
print(f"Parameters After:  {result.params_after:,}")
print(f"Reduction:         {result.params_reduction_pct:.2f}%")
```

### B. Sensitivity Analysis & Layer Selection Pipeline

```python
from prune_framework.core.config import FrameworkConfig
from prune_framework.pipelines.sensitivity import run_sensitivity_pipeline
from prune_framework.modules.analysis.layer_selection import LayerSelectorModule

# 1. Load config
config = FrameworkConfig.from_yaml("configs/rtdetr_structured.yaml")

# 2. Define evaluation callback returning float metric (e.g. mAP)
def eval_fn(model):
    # Run validation loop and return mAP score
    return 0.85

# 3. Run sensitivity sweep across pruning rates
results = run_sensitivity_pipeline(config, eval_fn=eval_fn, rates=[0.1, 0.2, 0.3, 0.5])

print(f"Baseline mAP: {results.baseline_score:.4f}")

# 4. Perform automated layer selection (relative mAP drop <= 5%)
selector = LayerSelectorModule(selector_name="sensitivity")
selection = selector.select(results, target_sparsity=0.5, max_allowed_relative_drop=0.05)

for layer_idx, target_rate in selection:
    print(f"Layer {layer_idx:2d} -> Selected Pruning Rate: {target_rate:.2f}")
```

### C. Decoupled Evaluation Pipeline

```python
import torch
from prune_framework.modules.evaluation.rtdetr_processors import RTDetrPreProcessor, RTDetrPostProcessor
from prune_framework.modules.evaluation.coco_evaluator import COCOEvaluator
from prune_framework.modules.evaluation.pipeline import EvaluationPipeline

# 1. Instantiate model-specific processors & evaluator
preprocessor = RTDetrPreProcessor(hf_model_name="PekingU/rtdetr_r18vd")
postprocessor = RTDetrPostProcessor(hf_model_name="PekingU/rtdetr_r18vd")
evaluator = COCOEvaluator(coco_gt_obj=coco_gt)

# 2. Build composable evaluation pipeline
pipeline = EvaluationPipeline(preprocessor, postprocessor, evaluator)

# 3. Run evaluation
result = pipeline.run(
    model=model,
    dataloader=val_dataloader,
    device=torch.device("cuda:0"),
    conf_thres=0.001,
    half=True
)

print(f"Evaluation mAP@0.5:0.95: {result.map:.4f}")
print(f"Evaluation mAP@0.5:     {result.map50:.4f}")
```
