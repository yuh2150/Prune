# Architecture overview

[Pruning Framework](../pruning.md) là tài liệu hiện hành. Các contracts nằm trong [contracts](../../prune_framework/contracts), orchestration trong [unified.py](../../prune_framework/pipelines/unified.py), plan build/apply trong [engine.py](../../prune_framework/core/engine.py).

Adapter expose typed targets; criterion tính importance; pruner thực hiện mechanism. Sensitivity đo probes trên baseline clones và selector tạo policy; engine build/validate exact mutation plan trước apply. Taylor là criterion, không phải một pruning strategy riêng.

Xem [flow và boundaries](../pruning.md#current-flow-and-execution-boundaries), [dependency safety](../pruning.md#structured-pruning-and-dependency-safety) và [implementation notes](pruning-contracts.md). [Flow audit](pruning-flow-audit.md) mô tả lịch sử trước sửa contracts, không thay thế trạng thái hiện tại.
