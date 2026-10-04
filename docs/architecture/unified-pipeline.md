# Unified pipeline

Entry: `main.main` → `run_pruning_pipeline` → `run_unified_pruning_pipeline` → `UnifiedPruningPipeline.run`.

[Source](../../prune_framework/pipelines/unified.py) và [flow chuẩn](../pruning.md#current-flow-and-execution-boundaries) là điểm tham chiếu. Calibration chạy trước sensitivity; callback trả `CalibrationContext`, runner thực hiện gradient computation. Policy từ sensitivity được truyền vào final plan.

`describe_flow(config)` là static dry-run boundary. `run(build_plan_only=True)` dừng sau plan validation/artifacts; vẫn có thể chạy evaluation, calibration, temporary probes và complexity measurement. `regularization.enabled` bị reject ở build-only vì có training.

Sau apply: architecture validation → optional recovery → evaluation/complexity → optional PyTorch latency → target check → export/checkpoint. Target thiếu metric không được pass. Xem [implementation contracts](pruning-contracts.md).
