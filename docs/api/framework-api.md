# Framework API reference

Xem [tài liệu chính](../pruning.md) để biết compatibility và validation boundaries.

| Symbol | Source / contract |
|---|---|
| `FrameworkConfig.from_dict`, `from_yaml` | [config.py](../../prune_framework/core/config.py) |
| `UnifiedPruningPipeline.describe_flow(config)` | Static orchestration description, không load detector |
| `UnifiedPruningPipeline(config).run(build_plan_only=True)` | Planning có compute, trả `PlanBuildResult` |
| `PruningEngine.build_plan(model, config)` | Build/validate plan |
| `PruningEngine.execute` | Apply qua engine, có thể nhận supplied plan |
| `ModelSnapshot` | [snapshot.py](../../prune_framework/modules/model/snapshot.py), deepcopy/restore |
| `CalibrationContext` | [context.py](../../prune_framework/modules/calibration/context.py), batches + loss_fn |
| `EvaluationResult`, `normalize_evaluation`, `normalize_yolo_evaluation` | [evaluation.py](../../prune_framework/contracts/evaluation.py) |
| `DeploymentTargets`, `TargetChecker.check` | [deployment.py](../../prune_framework/contracts/deployment.py) |

[Pipeline source](../../prune_framework/pipelines/unified.py), [engine source](../../prune_framework/core/engine.py), [results](../../prune_framework/core/results.py) chứa signatures đầy đủ.

`BasePruner` tách `create_plan`, `validate_plan`, `apply_plan`; `prune(model_adapter, criterion, granularity=None, config=None)` là compatibility wrapper. Custom callers phải giữ validation trước apply. Không coi abstract/default contract là bằng chứng custom pruner đã validate đúng.

Registry dùng `PluginRegistry.get_model_adapter` và các getter/list methods theo plugin type. Shorthand decorators gồm `register_model`, `register_pruner`, `register_criterion`, `register_granularity`, `register_selector`; xem [registry.py](../../prune_framework/core/registry.py).
