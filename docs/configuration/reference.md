# Configuration & CLI Reference

This document provides a reference for YAML configuration parameters and CLI command line arguments.

---

## 1. YAML Configuration Reference (`FrameworkConfig`)

Sample YAML configuration file (`configs/yolov5_structured.yaml`):

```yaml
model:
  name: yolov5
  weights: weights/yolov5s.pt
  device: cuda

pruning:
  pruner: structured
  criterion: l1
  granularity: channel
  amount: 0.3
  layer_params: null

analysis:
  sensitivity: false
  layer_selection: false

benchmark:
  enabled: true
  runs: 50
  warmup: 10

export:
  enabled: false
  format: onnx
  output_path: pruned_yolov5s.onnx

output_path: pruned_yolov5s.pt
```

---

## 2. Configuration Parameter Table

### `model` Section
| Parameter | Type | Default | Required | Description |
| :--- | :--- | :--- | :--- | :--- |
| `name` | `str` | `"yolov5"` | Yes | Registered model adapter name (`yolov5`, `rtdetr`, `resnet`). |
| `weights` | `str` | `"yolov5s.pt"` | Yes | Path to weights checkpoint or HuggingFace hub model ID. |
| `device` | `str` | `"cuda"` | No | Execution device (`"cuda"`, `"cuda:0"`, or `"cpu"`). |

### `pruning` Section
| Parameter | Type | Default | Required | Description |
| :--- | :--- | :--- | :--- | :--- |
| `pruner` | `str` | `"structured"` | Yes | Pruning strategy plugin (`structured`, `unstructured`, `depth`, `taylor`). |
| `criterion` | `str` | `"l1"` | Yes | Importance metric plugin (`l1`, `l2`, `taylor`, `random`). |
| `granularity` | `str` | `"channel"` | No | Pruning granularity plugin (`channel`, `filter`, `layer`). |
| `amount` | `float` | `0.3` | Yes | Target sparsity or global pruning ratio (0.0 to 1.0). |
| `layer_params`| `list` | `null` | No | Optional per-layer pruning tuples `[(layer_idx, rate), ...]`. |

### `analysis` Section
| Parameter | Type | Default | Required | Description |
| :--- | :--- | :--- | :--- | :--- |
| `sensitivity` | `bool` | `false` | No | Whether to execute Sensitivity Analysis sweep before pruning. |
| `layer_selection` | `bool` | `false` | No | Whether to run automated Layer Selection based on sensitivity. |

### `benchmark` & `export` Sections
| Parameter | Type | Default | Required | Description |
| :--- | :--- | :--- | :--- | :--- |
| `benchmark.enabled` | `bool` | `true` | No | Whether to run post-pruning latency profiling. |
| `benchmark.runs` | `int` | `100` | No | Number of benchmark inference iterations. |
| `export.enabled` | `bool` | `false` | No | Whether to export pruned model to ONNX. |
| `export.output_path`| `str` | `"pruned.onnx"`| No | Destination path for exported ONNX model. |

---

## 3. CLI Argument Reference (`main.py`)

| CLI Argument | Type | Description |
| :--- | :--- | :--- |
| `--config` | `str` | Path to YAML configuration file. |
| `--model` | `str` | Model adapter architecture override (`yolov5`, `rtdetr`, `resnet`). |
| `--weights` | `str` | Weights checkpoint path override. |
| `--strategy` | `str` | Pruner strategy override (`structured`, `unstructured`, `depth`, `taylor`). |
| `--criterion` | `str` | Importance criterion override (`l1`, `l2`, `taylor`, `random`). |
| `--granularity` | `str` | Granularity override (`channel`, `filter`, `layer`). |
| `--amount` | `float` | Target sparsity ratio override. |
| `--output-path` | `str` | Output checkpoint path override. |
| `--list-models` | Flag | Lists all registered model adapters. |
| `--list-pruners` | Flag | Lists all registered pruners. |
| `--list-criteria` | Flag | Lists all registered criteria. |
| `--list-granularities` | Flag | Lists all registered granularities. |
| `--list-selectors` | Flag | Lists all registered layer selectors. |
