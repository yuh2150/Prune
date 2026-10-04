# Configuration reference

[Unified configuration](../pruning.md#unified-configuration) là mô tả chính. Schema thực nằm trong [config.py](../../prune_framework/core/config.py); mẫu trong [research_unified.yaml](../../configs/research_unified.yaml).

Top-level sections: `model`, `pruning`, `analysis`, `sensitivity`, `evaluation`, `targets`, `regularization`, `recovery`, `benchmark`, `export`, `experiment`; `output_path` là scalar. Không có top-level data/calibration/policy. Unknown sections bị reject.

| Alias đầu vào | Field lưu trong config |
|---|---|
| `pruning.method` | `pruner` |
| `pruning.structure` | `granularity` |
| `pruning.target_ratio` | `amount` |
| `pruning.global` | `global_pruning` |

Conflicting aliases bị reject. `iterative_steps` chỉ nhận 1. Dataset đi qua callback kwargs; calibration settings dưới `pruning`. Layer-wise policy dùng `layer_params` hoặc selector; không kết hợp global unstructured với policy.

`benchmark.enabled: false` không tự tắt params/FLOPs flags. Build-only vẫn có thể đo complexity. `recovery.callback: experiments.yolov5:recover` chỉ là placeholder. Các số target trong YAML là yêu cầu, không phải kết quả đo.

Các config YOLO khác dùng đường weights khác mẫu research; kiểm tra path trước planning. File [rtdetr_structured.yaml](../../configs/rtdetr_structured.yaml) thực tế cấu hình depth/l2/layer.
