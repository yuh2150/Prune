# Detector Pruning Framework

Framework tổ chức pruning detector bằng adapter, importance criterion, calibration, sensitivity-aware policy và executable pruning plan.

**Tài liệu chính: [Pruning Framework](docs/pruning.md).**

```text
Model → Baseline snapshot/evaluation → Compatibility → Calibration
→ Sensitivity → Policy → Build/validate plan → Apply → Recovery
→ Evaluation/measurement → Deployment target check → Finalize
```

Contracts và mechanisms đã **Implemented**; orchestration, evaluator và dependency safety đã **Validated** bằng unit/tiny/mock/synthetic tests. YOLOv5 recovery fine-tune callback có test CPU với data/loss injectable; full detector end-to-end pruning chưa **Experimentally validated** trong unified flow hiện tại.

Từ repository root, trong môi trường project:

```bash
python main.py --config configs/research_unified.yaml --dry-run
```

Dry-run kiểm tra config/mô tả flow, không load model hoặc data. Để chuẩn bị và validate plan:

```bash
python main.py --config configs/research_unified.yaml --build-plan-only
```

Build-plan-only có thể load model, chạy calibration, sensitivity/evaluation và graph validation; cần checkpoint/data phù hợp và có thể tốn compute. Nó dừng trước final apply/recovery/export. Config mẫu chưa phải recipe cho experiment đầy đủ.

Xem [docs index](docs/README.md), [engineering contracts](docs/architecture/pruning-contracts.md) và [historical results](REPORT.md). Kết quả lịch sử chưa được revalidate bằng unified flow hiện tại.
