# Quickstart: inspection and planning

Xem [tài liệu chính](../pruning.md) trước khi chuẩn bị model/data. Chạy từ repository root, trong môi trường đã có dependencies của project; không tự động tải checkpoint/dataset trong hướng dẫn này.

```bash
python main.py --config configs/research_unified.yaml --dry-run
```

Dry-run parse/validate config và mô tả orchestration; không load model/data hoặc resolve callback bằng import. Nó không chứng nhận experiment readiness.

```bash
python main.py --config configs/research_unified.yaml --build-plan-only
```

Command planning này có compute: checkpoint load, baseline evaluation/complexity, calibration, sensitivity và validation trên shadow models tùy config. Cần local weights/data và YOLO loss hyperparameters. Nó dừng trước final apply, recovery, latency benchmark và export; không cho pre-pruning regularization training.

Config mẫu bật recovery callback chưa implement. Không có full experiment/fine-tune command trong quickstart. Xem [CLI](../pruning.md#cli), [config](../pruning.md#unified-configuration) và [recovery](../pruning.md#recovery).
