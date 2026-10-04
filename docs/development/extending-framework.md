# Extending the framework

Đọc [contracts](../../prune_framework/contracts) và [compatibility matrix](../pruning.md#compatibility-matrix) trước khi thêm plugin. Registry discovery nằm trong [registry.py](../../prune_framework/core/registry.py).

1. Adapter kế thừa `BaseModelAdapter`, expose typed targets và bảo vệ task outputs; đăng ký bằng shorthand `register_model` hoặc `PluginRegistry.register_model_adapter`.
2. Criterion kế thừa `BaseImportanceCriterion`, implement `score(module, context=None)` và khai báo pruning modes/module types/calibration needs đúng thực tế. Không dùng metadata rộng hơn implementation.
3. Pruner kế thừa `BasePruner`, implement create/validate/apply plan; khai báo `pruning_mode` và `supports_layerwise_policy`. Validation phải kiểm tra indices, dependencies và action sequence phù hợp topology.
4. Callback evaluation trả normalized-compatible result; calibration trả `CalibrationContext`; recovery cần implementation thật, không metric giả hoặc fixed accuracy.
5. Kiểm tra unit/tiny/mock cho contract trước khi gắn nhãn Validated. Chỉ ghi Experimentally validated khi đã có detector + dataset thật end-to-end với protocol/artifacts rõ ràng.

Các signatures cụ thể xem [API reference](../api/framework-api.md). Không thêm một criterion vào registry rồi mặc định coi mọi strategy/adapter tương thích.
