# Production Model Pruning Framework (Modular + Plugin Architecture)

A professional, enterprise-grade model pruning framework for Deep Learning architectures (**YOLOv5**, **YOLOv7**, **RT-DETR**, **ResNet**).

The framework enforces **Zero-Code-Change Core Extensibility** using a clean composition of four orthogonal plugin abstractions:

$$\text{Pruner (Plugin)} + \text{Criterion (Plugin)} + \text{Granularity (Plugin)} + \text{ModelAdapter (Plugin)}$$

---

## 🌟 Architecture Overview

```text
prune_framework/
├── core/
│   ├── registry.py            # Central PluginRegistry with decorator discovery
│   ├── interfaces.py          # Abstract Base Classes (BasePruner, BaseCriterion, etc.)
│   ├── engine.py              # Pure Orchestrator & Forward Verification
│   ├── config.py              # Configuration manager & YAML loader
│   ├── results.py             # Standardized result dataclasses
│   └── exceptions.py          # Framework exception hierarchy
├── modules/
│   ├── analysis/              # Sensitivity, Layer Selection, Importance Scorer
│   ├── evaluation/            # Metrics, Decoupled Benchmark, Validator
│   ├── model/                 # Model Loader & Inspector
│   ├── export/                # ONNX and Checkpoint Exporters
│   └── utils/                 # Logging & Profiling utilities
├── plugins/
│   ├── pruners/               # structured, unstructured, depth
│   ├── criteria/              # l1, l2, magnitude, bn_gamma, taylor, lamp, random
│   ├── granularities/         # weight, channel, filter, block, head, layer
│   ├── adapters/              # yolov5, rtdetr
│   └── selectors/             # sensitivity, greedy
├── pipelines/
│   ├── pruning.py             # End-to-end Pruning execution
│   ├── sensitivity.py         # End-to-end Sensitivity Analysis
│   └── benchmarking.py        # End-to-end Decoupled Latency Benchmark
├── configs/                   # Production YAML templates
└── main.py                    # Unified CLI Entry Point
```

---

## 🚀 Quickstart & Usage

### 1. Dynamic Listing Commands
Inspect registered plugins dynamically via `main.py`:

```bash
# List registered model adapters
python main.py --list-models

# List registered pruning strategies
python main.py --list-pruners

# List registered importance criteria
python main.py --list-criteria

# List registered granularities
python main.py --list-granularities
```

---

### 2. Configuration-Driven Pruning Execution

Execute pruning using YAML configuration files:

```bash
# Structured Channel Pruning on YOLOv5
python main.py --config configs/yolov5_structured.yaml

# Unstructured Weight Sparsity on YOLOv5
python main.py --config configs/yolov5_unstructured.yaml

# Transformer Depth Pruning on RT-DETR
python main.py --config configs/rtdetr_structured.yaml
```

---

### 3. Command Line Parameter Overrides

Override YAML options directly from the command line:

```bash
python main.py \
  --model yolov5 \
  --strategy structured \
  --criterion l1 \
  --granularity channel \
  --weights weights/yolov5s.pt \
  --amount 0.35 \
  --output-path pruned_yolov5s.pt
```

---

## 🔌 Adding a New Plugin (Zero-Code-Change Core)

To add a new importance criterion (e.g. `CustomCosine`), create a single file `prune_framework/plugins/criteria/custom.py`:

```python
import torch
import torch.nn as nn
from prune_framework.core.interfaces import BaseImportanceCriterion
from prune_framework.core.registry import register_criterion

@register_criterion("custom_cosine")
class CustomCosineCriterion(BaseImportanceCriterion):
    def score(self, module: nn.Module, context=None) -> torch.Tensor:
        # Calculate custom importance scores
        return torch.norm(module.weight.data, p=2, dim=[1, 2, 3])
```

That's it! Use `criterion: custom_cosine` in your YAML configuration or CLI flags immediately without modifying any core engine or registry code.

---

## 🧪 Running Framework Test Suite

Run unit and integration tests using Python's native test runner:

```bash
conda run -n env_cv python -m unittest discover -s tests
```
