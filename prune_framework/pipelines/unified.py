"""The architecture-agnostic, stage-based pruning experiment pipeline."""

from __future__ import annotations

import importlib
import inspect
import math
from dataclasses import asdict
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn

from prune_framework.contracts.evaluation import EvaluationResult, normalize_evaluation
from prune_framework.contracts.deployment import TargetChecker
from prune_framework.modules.model.snapshot import ModelSnapshot
from prune_framework.modules.calibration.context import CalibrationContext
from prune_framework.core.config import FrameworkConfig
from prune_framework.core.engine import PruningEngine
from prune_framework.core.exceptions import ConfigValidationException, PruningExecutionError
from prune_framework.core.experiment import ExperimentArtifacts, seed_everything
from prune_framework.core.results import ExperimentResult, PlanBuildResult
from prune_framework.modules.analysis.layer_selection import LayerSelectorModule
from prune_framework.modules.analysis.sensitivity import SensitivityAnalyzer
from prune_framework.modules.evaluation.benchmark import LatencyBenchmark, attach_quality_metrics
from prune_framework.modules.evaluation.complexity import measure_complexity
from prune_framework.modules.evaluation.validator import ModelValidator
from prune_framework.modules.export.exporter import ModelExporter
from prune_framework.modules.calibration import GradientCalibrationRunner, HigherOrderCalibrationRunner, SynFlowCalibrationRunner
from prune_framework.modules.model.loader import ModelLoader
from prune_framework.modules.regularization import (
    L0HardConcreteRegularization,
    L1BatchNormScaleRegularization,
    RegularizationController,
)
from prune_framework.contracts.targets import TargetType


EvaluationCallback = Callable[..., float | Dict[str, float] | EvaluationResult]
RecoveryCallback = Callable[..., nn.Module | None | Dict[str, Any]]


class UnifiedPruningPipeline:
    """Runs a reproducible pruning experiment without model-family branches.

    Adapters own model loading, tracing input and protected modules; user supplied
    callbacks own dataset-specific evaluation and recovery. This prevents the
    core pipeline from assuming YOLO, COCO, Hugging Face or a particular loss.
    """

    @staticmethod
    def describe_flow(config: FrameworkConfig) -> Dict[str, Any]:
        """Describe the current run order without constructing a pipeline.

        This is static inspection, not a validated pruning plan. In particular,
        callbacks are not imported: importing user code can itself perform work.
        Known orchestration gaps are reported rather than hidden by a dry-run.
        """
        config.validate()
        cfg = config
        criterion = ("l0_gate" if cfg.regularization.term == "l0_hard_concrete"
                     else cfg.regularization.pruning_criterion) if cfg.regularization.enabled else cfg.pruning.criterion
        engine = PruningEngine(cfg.model.name, cfg.pruning.pruner, criterion, cfg.pruning.granularity)
        criterion_cls = engine.criterion_cls
        calibration = any(getattr(criterion_cls, flag, False) for flag in (
            "requires_gradients", "requires_higher_order_calibration", "requires_synflow_calibration"))
        compatibility = {
            "amount": cfg.pruning.amount,
            "global_pruning": cfg.pruning.global_pruning,
            "n": cfg.pruning.n,
            "m": cfg.pruning.m,
            "block_shape": cfg.pruning.block_shape,
        }
        if (cfg.sensitivity.enabled and _uses_layerwise_sensitivity(engine)) or cfg.pruning.layer_params is not None:
            compatibility['pruning_params'] = cfg.pruning.layer_params or [(0, cfg.pruning.amount)]
        if cfg.sensitivity.enabled and cfg.regularization.enabled and cfg.regularization.term == 'l0_hard_concrete':
            raise ConfigValidationException('L0 forced indices cannot override sensitivity layer-wise policy')
        engine.validate_compatibility(compatibility)
        stages = ["seed_and_create_artifacts", "load_model_and_select_plugins", "baseline_snapshot"]
        if cfg.benchmark.params or cfg.benchmark.flops or cfg.targets.parameter_reduction is not None or cfg.targets.flops_reduction is not None:
            stages.append("baseline_complexity")
        needs_eval = cfg.evaluation.enabled or cfg.sensitivity.enabled or cfg.targets.max_map50_drop is not None or cfg.targets.max_map50_95_drop is not None
        if needs_eval:
            stages.append("baseline_evaluation_callback")
        stages.append("deployment_targets_and_compatibility")
        if cfg.regularization.enabled:
            stages.append("pre_pruning_regularization_callback")
        if engine.pruner_cls.pruning_mode == 'structured':
            stages.append("dependency_preflight")
        if calibration:
            stages.append("importance_calibration")
        if cfg.sensitivity.enabled and _uses_layerwise_sensitivity(engine):
            stages.extend(["sensitivity_probes", "select_layer_ratios"])
        else:
            stages.append(f"technique_specific_{_policy_kind(engine)}_policy")
        stages.extend(["requested_policy", "build_validate_plan", "BUILD_PLAN_ONLY_STOP", "apply_plan", "architecture_validation"])
        if cfg.recovery.enabled:
            stages.append("recovery_callback")
        if needs_eval:
            stages.append("final_evaluation_callback")
        if cfg.benchmark.params or cfg.benchmark.flops or cfg.targets.parameter_reduction is not None or cfg.targets.flops_reduction is not None:
            stages.append("final_complexity")
        if cfg.benchmark.enabled and cfg.benchmark.latency:
            stages.append("pytorch_latency_benchmark")
        stages.append("target_check")
        stages.append("save_checkpoint_reload_validate")
        if cfg.export.wants_onnx():
            stages.append("optional_onnx_export_validate")
        stages.append("save_result")
        gaps = ["Callbacks, checkpoint, dataset and architecture compatibility are NOT executed by this dry-run."]
        if cfg.recovery.callback == 'experiments.yolov5:recover' and cfg.recovery.enabled:
            gaps.append("YOLO recovery requires a local training split or injected dataloader/loss; dry-run does not validate either.")
        return {
            "dry_run": True,
            "executed_stages": [],
            "model": cfg.model.name,
            "strategy": cfg.pruning.pruner,
            "criterion": criterion,
            "configured_run_order": stages,
            "callbacks_not_imported": {
                "evaluation": cfg.evaluation.callback,
                "recovery": cfg.recovery.callback,
                "calibration": cfg.pruning.calibration_callback,
            },
            "gaps": gaps,
        }

    def __init__(
        self,
        config: FrameworkConfig,
        evaluator: Optional[EvaluationCallback] = None,
        recovery: Optional[RecoveryCallback] = None,
    ):
        config.validate()
        self.config = config
        self.evaluator = evaluator or self._load_callback(config.evaluation.callback)
        self.recovery = recovery or self._load_callback(config.recovery.callback)
        self.calibration_callback = self._load_callback(config.pruning.calibration_callback)

    def run(self, build_plan_only: bool = False) -> ExperimentResult | PlanBuildResult:
        cfg = self.config
        seed_everything(cfg.experiment.seed, cfg.experiment.deterministic)
        artifacts = ExperimentArtifacts(cfg.experiment.output_dir, cfg.experiment.name)
        artifact_paths = {"config": artifacts.write_json("config.resolved.json", cfg.to_dict())}
        stages: Dict[str, Any] = {}

        device = torch.device(cfg.model.device if torch.cuda.is_available() else "cpu")
        model, checkpoint = ModelLoader.load(
            cfg.model.name,
            cfg.model.weights,
            device,
            **({"num_classes": cfg.model.num_classes} if cfg.model.num_classes is not None else {}),
        )
        if cfg.regularization.enabled and cfg.regularization.term == "l0_hard_concrete":
            active_criterion = "l0_gate"
        else:
            active_criterion = (
                cfg.regularization.pruning_criterion if cfg.regularization.enabled else cfg.pruning.criterion
            )
        engine = PruningEngine(
            model_name=cfg.model.name,
            pruner_name=cfg.pruning.pruner,
            criterion_name=active_criterion,
            granularity_name=cfg.pruning.granularity,
        )
        compatibility_config = {
            "amount": cfg.pruning.amount,
            "global_pruning": cfg.pruning.global_pruning,
            "n": cfg.pruning.n,
            "m": cfg.pruning.m,
            "block_shape": cfg.pruning.block_shape,
        }
        if (cfg.sensitivity.enabled and _uses_layerwise_sensitivity(engine)) or cfg.pruning.layer_params is not None:
            compatibility_config['pruning_params'] = cfg.pruning.layer_params or [(0, cfg.pruning.amount)]
        if cfg.sensitivity.enabled and cfg.regularization.enabled and cfg.regularization.term == 'l0_hard_concrete':
            raise ConfigValidationException('L0 forced indices cannot override sensitivity layer-wise policy')
        engine.validate_compatibility(compatibility_config)
        if build_plan_only and cfg.regularization.enabled:
            raise ConfigValidationException('build-plan-only cannot run pre-pruning regularization training; provide a prepared checkpoint')
        baseline_snapshot = ModelSnapshot(model)
        adapter = engine.adapter_cls(model)
        dummy_input = adapter.get_dummy_input(device)
        stages["load_model"] = {"device": str(device), "model": cfg.model.name}

        complexity_before = None
        if cfg.benchmark.params or cfg.benchmark.flops or cfg.targets.parameter_reduction is not None or cfg.targets.flops_reduction is not None:
            complexity_before = measure_complexity(baseline_snapshot.restore(), dummy_input, include_flops=cfg.benchmark.flops or cfg.targets.flops_reduction is not None)
            stages["baseline_cost"] = asdict(complexity_before)

        baseline_metrics = self._evaluate(baseline_snapshot.restore(), "baseline") if self._needs_evaluation() else None
        if baseline_metrics is not None:
            stages["baseline_evaluation"] = baseline_metrics
            baseline_error = _invalid_baseline_reason(cfg, baseline_metrics)
            if baseline_error:
                artifacts.write_json("result.json", {"status": "INVALID_BASELINE", "failure_reason": baseline_error,
                    "baseline_valid": False, "baseline_metrics": baseline_metrics})
                raise PruningExecutionError(f"INVALID_BASELINE: {baseline_error}")

        regularization = None
        if cfg.regularization.enabled:
            if cfg.pruning.pruner.lower() not in {"structured", "structured_channel"}:
                raise ConfigValidationException("L1 channel-sparsity regularization requires a structured channel pruner.")
            if not adapter.supports_channel_sparsity_regularization():
                raise ConfigValidationException(
                    f"{cfg.model.name} does not provide a supported Conv-BN recovery path for channel-sparsity regularization."
                )
            if cfg.regularization.term not in {"l1_bn_scale", "l0_hard_concrete"}:
                raise ConfigValidationException(f"Unsupported regularization term '{cfg.regularization.term}'.")
            if self.recovery is None:
                raise ConfigValidationException(
                    "regularization.enabled requires a recovery callback to run train-time sparsity regularization."
                )
            term = (
                L0HardConcreteRegularization(
                    beta=cfg.regularization.gate_beta,
                    gamma=cfg.regularization.gate_gamma,
                    zeta=cfg.regularization.gate_zeta,
                    log_alpha_init=cfg.regularization.gate_log_alpha_init,
                )
                if cfg.regularization.term == "l0_hard_concrete"
                else L1BatchNormScaleRegularization()
            )
            regularization = RegularizationController(
                term,
                adapter,
                strength=cfg.regularization.strength,
                schedule=cfg.regularization.schedule,
                warmup_steps=cfg.regularization.warmup_steps,
                total_steps=cfg.regularization.total_steps,
            )
            recovery_output = self._invoke(
                self.recovery,
                model,
                "regularization",
                cfg.recovery.kwargs,
                adapter=adapter,
                regularization=regularization,
            )
            model, recovery_details = self._resolve_recovery_output(recovery_output, model)
            if model is not adapter.model:
                adapter = engine.adapter_cls(model)
                regularization.adapter = adapter
            regularization_summary = regularization.state_dict()
            if recovery_details:
                regularization_summary["recovery"] = recovery_details
            artifact_paths["regularization"] = artifacts.write_json("regularization.json", regularization_summary)
            stages["regularization"] = regularization_summary

        engine.validate_model_compatibility(model, compatibility_config)
        if cfg.pruning.pruner.lower() in {"structured", "structured_channel"}:
            # The preflight proves that this adapter and its dummy input can be
            # traced before any weights are physically modified.
            engine.build_dependency_graph(model)
            stages["dependency_graph"] = {"status": "built"}

        calibration_result = None
        criterion_cls = engine.criterion_cls
        calibration_context = None
        if self.calibration_callback is not None:
            calibration_context = self._invoke(self.calibration_callback, model, 'calibration',
                                               cfg.pruning.calibration_kwargs, adapter=adapter)
            if not isinstance(calibration_context, CalibrationContext):
                raise TypeError('Calibration callback must return CalibrationContext')
            calibration_context.validate()
            if not getattr(criterion_cls, 'requires_gradients', False) and not getattr(criterion_cls, 'requires_higher_order_calibration', False):
                raise ConfigValidationException('A batch/loss calibration callback is only supported for gradient/HVP criteria')
        def calibration_batches():
            if calibration_context is not None:
                return calibration_context.batches
            return adapter.get_gradient_calibration_batches(
                cfg.pruning.calibration_batches, device, cfg.pruning.calibration_seed,
                cfg.pruning.calibration_batch_size)
        def calibration_loss():
            return calibration_context.loss_fn if calibration_context is not None else adapter.build_gradient_calibration_loss()
        if getattr(criterion_cls, "requires_synflow_calibration", False):
            target_types = getattr(criterion_cls, "calibration_target_types", {TargetType.CONV_WEIGHT, TargetType.LINEAR_WEIGHT})
            targets = adapter.get_prunable_targets(target_types)
            if not targets:
                supported = ", ".join(sorted(target.value for target in target_types))
                raise ConfigValidationException(
                    f"{cfg.model.name} exposes no safe SynFlow targets for: {supported}."
                )
            calibration_result = SynFlowCalibrationRunner().run(
                model=model,
                targets=targets,
                input_factory=lambda: adapter.get_synflow_input(device),
                output_reducer=adapter.reduce_synflow_output,
                snapshot_extra_state=adapter.snapshot_synflow_state,
                restore_extra_state=adapter.restore_synflow_state,
            )
            calibration_summary = calibration_result.describe()
            artifact_paths["synflow_calibration"] = artifacts.write_json("synflow_calibration.json", calibration_summary)
            stages["importance_calibration"] = {"method": "synflow", **calibration_summary}
        elif getattr(criterion_cls, "requires_higher_order_calibration", False):
            if calibration_context is None and not adapter.supports_gradient_calibration():
                raise ConfigValidationException(
                    f"{cfg.model.name} does not provide an integrated task loss for higher-order gradient calibration."
                )
            target_types = getattr(criterion_cls, "calibration_target_types", {TargetType.CONV_WEIGHT, TargetType.LINEAR_WEIGHT})
            targets = adapter.get_prunable_targets(target_types)
            if not targets:
                supported = ", ".join(sorted(target.value for target in target_types))
                raise ConfigValidationException(
                    f"{cfg.model.name} exposes no safe higher-order-calibration targets for: {supported}."
                )
            calibration_result = HigherOrderCalibrationRunner(
                seed=calibration_context.seed if calibration_context is not None else cfg.pruning.calibration_seed,
                parameter_scope=calibration_context.parameter_scope if calibration_context is not None else cfg.pruning.parameter_scope,
            ).run(
                model=model,
                targets=targets,
                batches=calibration_batches(),
                loss_fn=calibration_loss(),
                batch_weights=calibration_context.batch_weights if calibration_context is not None else None,
            )
            calibration_summary = calibration_result.describe()
            artifact_paths["grasp_calibration"] = artifacts.write_json("grasp_calibration.json", calibration_summary)
            stages["importance_calibration"] = {"method": "grasp", **calibration_summary}
        elif getattr(criterion_cls, "requires_gradients", False):
            if calibration_context is None and not adapter.supports_gradient_calibration():
                raise ConfigValidationException(
                    f"{cfg.model.name} does not provide an integrated task loss for gradient calibration."
                )
            target_types = getattr(criterion_cls, "calibration_target_types", {TargetType.CONV_OUT_CHANNEL})
            targets = adapter.get_prunable_targets(target_types)
            if not targets:
                supported = ", ".join(sorted(target.value for target in target_types))
                raise ConfigValidationException(
                    f"{cfg.model.name} exposes no safe gradient-calibration targets for: {supported}."
                )
            runner = GradientCalibrationRunner(
                seed=calibration_context.seed if calibration_context is not None else cfg.pruning.calibration_seed,
                accumulate=cfg.pruning.calibration_accumulate,
                aggregation=cfg.snip.gradient_aggregation if active_criterion.lower() == "snip" else "signed_mean",
            )
            calibration_result = runner.run(
                model=model,
                targets=targets,
                batches=calibration_batches(),
                loss_fn=calibration_loss(),
            )
            calibration_summary = calibration_result.describe()
            artifact_paths["calibration"] = artifacts.write_json("calibration.json", calibration_summary)
            stages["importance_calibration"] = calibration_summary

        calibration_payload = {}
        if calibration_result is not None:
            key = ('synflow_calibration' if getattr(criterion_cls, 'requires_synflow_calibration', False)
                   else 'higher_order_calibration' if getattr(criterion_cls, 'requires_higher_order_calibration', False)
                   else 'gradient_calibration')
            calibration_payload[key] = calibration_result

        selection_summary = None
        sensitivity_summary = None
        pruning_params = cfg.pruning.layer_params if cfg.pruning.layer_params is not None else cfg.pruning.amount
        if cfg.sensitivity.enabled and _uses_layerwise_sensitivity(engine):
            self._require_evaluator("sensitivity analysis")
            analyzer = SensitivityAnalyzer(
                model_name=cfg.model.name,
                pruner_name=cfg.pruning.pruner,
                criterion_name=active_criterion,
                granularity_name=cfg.pruning.granularity,
            )
            result = analyzer.analyze(
                model=model,
                pruning_rates=cfg.sensitivity.rates,
                eval_fn=lambda candidate: self._metric(self._evaluate(candidate, "sensitivity")),
                metric_direction=cfg.sensitivity.metric_direction,
                calibration=calibration_payload,
                baseline=ModelSnapshot(model) if cfg.regularization.enabled else baseline_snapshot,
                config={"min_channels": cfg.pruning.min_channels, "global_pruning": False},
            )
            if not result.profiles or not any(pt.is_valid for profile in result.profiles.values() for pt in profile.points.values()):
                raise PruningExecutionError('Sensitivity produced no valid probes; refusing to build policy')
            selector = LayerSelectorModule(cfg.sensitivity.selector)
            selection = selector.select(
                result,
                target_sparsity=cfg.pruning.amount,
                max_allowed_relative_drop=cfg.sensitivity.max_allowed_relative_drop,
            )
            pruning_params = list(selection)
            sensitivity_summary = _sensitivity_to_dict(result)
            selection_summary = selection.summary()
            artifact_paths["sensitivity"] = artifacts.write_json("sensitivity.json", sensitivity_summary)
            artifact_paths["selection"] = artifacts.write_json("selection.json", selection_summary)
            stages["sensitivity"] = {"profiles": len(result.profiles), "selector": cfg.sensitivity.selector}
        else:
            # Each non-layerwise mechanism owns a valid policy domain: global
            # saliency for unstructured, adapter structural scores for depth,
            # head scores for attention and constrained patterns for N:M/block.
            # Do not force these mechanisms through synthetic layer probes.
            policy_kind = _policy_kind(engine)
            selection_summary = {
                "policy_kind": policy_kind,
                "source": "technique_specific_scoring",
                "global_pruning": cfg.pruning.global_pruning,
                "amount": cfg.pruning.amount,
            }
            stages["technique_policy"] = selection_summary
            artifact_paths["selection"] = artifacts.write_json("selection.json", selection_summary)

        stages["importance_estimation"] = {"criterion": active_criterion}
        requested_pruning_plan = {
            "method": cfg.pruning.pruner,
            "granularity": cfg.pruning.granularity,
            "target_ratio": cfg.pruning.amount,
            "global": cfg.pruning.global_pruning,
            "targets": pruning_params,
        }
        stages["requested_pruning_plan"] = requested_pruning_plan

        pruning_config = {
            "allow_noop": cfg.pruning.allow_noop,
            "amount": cfg.pruning.amount,
            "pruning_params": pruning_params,
            "global_pruning": cfg.pruning.global_pruning,
            "iterative_steps": cfg.pruning.iterative_steps,
            "min_channels": cfg.pruning.min_channels,
            "n": cfg.pruning.n,
            "m": cfg.pruning.m,
            "block_shape": cfg.pruning.block_shape,
        }
        if cfg.regularization.enabled and cfg.regularization.term == "l0_hard_concrete":
            gate_indices = regularization.prune_indices(
                threshold=cfg.regularization.gate_threshold,
                min_channels=cfg.pruning.min_channels,
            )
            pruning_config["forced_channel_indices"] = gate_indices
            pruning_config["l0_gate_controller"] = regularization
            stages["gate_pruning_decisions"] = {
                "threshold": cfg.regularization.gate_threshold,
                "indices": gate_indices,
            }
        if calibration_result is not None:
            if getattr(criterion_cls, "requires_synflow_calibration", False):
                pruning_config["synflow_calibration"] = calibration_result
                pruning_config["synflow_recalibrate"] = lambda: SynFlowCalibrationRunner().run(
                    model=model, targets=targets,
                    input_factory=lambda: adapter.get_synflow_input(device),
                    output_reducer=adapter.reduce_synflow_output,
                    snapshot_extra_state=adapter.snapshot_synflow_state,
                    restore_extra_state=adapter.restore_synflow_state,
                )
            elif getattr(criterion_cls, "requires_higher_order_calibration", False):
                pruning_config["higher_order_calibration"] = calibration_result
            else:
                pruning_config["gradient_calibration"] = calibration_result
        plan = engine.build_plan(model, pruning_config)
        artifact_paths['pruning_plan'] = artifacts.write_json('pruning_plan.json', plan.describe())
        artifact_paths['baseline'] = artifacts.write_json('baseline.json', baseline_metrics)
        if build_plan_only:
            return PlanBuildResult(plan, model, baseline_metrics, selection_summary, artifact_paths)
        pruning_result = engine.execute(model, pruning_config, verify_forward=True, plan=plan)
        # Persist the resolved target groups and validation status generated by
        # the pruner, not only the configuration-level pruning intent.
        artifact_paths["pruning_plan"] = artifacts.write_json("pruning_plan.json", pruning_result.pruning_plan)
        stages["pruning_plan"] = pruning_result.pruning_plan
        if not pruning_result.forward_verified or not ModelValidator.validate_forward(model, adapter.get_dummy_input(device)):
            raise PruningExecutionError("Pruned model failed architecture validation; recovery and export were skipped.")
        stages["apply_pruning"] = asdict(pruning_result)
        stages["architecture_validation"] = {"status": "passed"}

        if cfg.recovery.enabled:
            if self.recovery is None:
                raise ConfigValidationException("recovery.enabled requires recovery.callback or a recovery callable.")
            recovery_output = self._invoke(
                self.recovery, model, "recovery", cfg.recovery.kwargs, adapter=adapter
            )
            model, recovery_details = self._resolve_recovery_output(recovery_output, model)
            if model is not adapter.model:
                adapter = engine.adapter_cls(model)
            stages["recovery"] = {"epochs": cfg.recovery.epochs, "callback": cfg.recovery.callback}
            if recovery_details:
                stages["recovery"].update(recovery_details)
                artifact_paths["recovery"] = artifacts.write_json("recovery.json", recovery_details)

        final_metrics = self._evaluate(model, "final") if self._needs_evaluation() else None
        if final_metrics is not None:
            stages["final_evaluation"] = final_metrics
        diagnostics = _model_diagnostics(model)
        stages["post_pruning_diagnostics"] = diagnostics
        if not diagnostics["finite"]:
            raise PruningExecutionError("FAILED: pruned model contains NaN or Inf weights.")

        complexity_after = None
        if cfg.benchmark.params or cfg.benchmark.flops or cfg.targets.parameter_reduction is not None or cfg.targets.flops_reduction is not None:
            complexity_after = measure_complexity(
                model,
                adapter.get_dummy_input(device),
                include_flops=cfg.benchmark.flops or cfg.targets.flops_reduction is not None,
            )
            stages["final_cost"] = asdict(complexity_after)
        if cfg.benchmark.enabled and cfg.benchmark.latency:
            benchmark = LatencyBenchmark(cfg.benchmark.warmup, cfg.benchmark.runs).benchmark(
                model, adapter.get_dummy_input(device)
            )
            attach_quality_metrics(benchmark, final_metrics)
            pruning_result.benchmark = benchmark
            stages["latency"] = asdict(benchmark)

        target_metrics = dict(final_metrics or {})
        if pruning_result.benchmark is not None:
            target_metrics['total_ms'] = pruning_result.benchmark.total_latency_ms
        target_check = TargetChecker.check(baseline_metrics, target_metrics, cfg.targets,
                                          complexity_before, complexity_after)
        stages['target_check'] = asdict(target_check)
        artifact_paths['target_check'] = artifacts.write_json('target_check.json', target_check)
        if not target_check.reached:
            raise PruningExecutionError(f'Deployment targets not reached: {target_check.violations}; unavailable: {target_check.unavailable}')

        ModelExporter.export_checkpoint(model, cfg.output_path, checkpoint)
        artifact_paths["checkpoint"] = cfg.output_path
        checkpoint_status = self._validate_checkpoint_artifact(cfg.output_path, device)
        stages["artifact"] = {"checkpoint_saved": True, **checkpoint_status}
        artifact_paths["artifact_validation"] = artifacts.write_json("artifact_validation.json", stages["artifact"])
        if not checkpoint_status["checkpoint_reload_validated"]:
            raise PruningExecutionError(
                f"Checkpoint was saved but reload validation failed: {checkpoint_status['checkpoint_validation_reason']}"
            )

        export_status = {
            "onnx_exported": False,
            "onnx_validated": False,
            "onnx_validation_status": "DISABLED",
            "onnx_validation_reason": None,
        }
        if cfg.export.wants_onnx():
            try:
                export_status = ModelExporter.export_onnx(
                    model,
                    adapter.get_dummy_input(device),
                    cfg.export.output_path,
                    opset_version=cfg.export.opset_version,
                )
                artifact_paths["onnx"] = cfg.export.output_path
            except Exception as exc:
                # Checkpoint is deliberately saved first.  An optional ONNX
                # failure is recorded as an artifact status, never allowed to
                # erase the successfully-pruned PyTorch deliverable.
                export_status = {
                    "onnx_exported": False,
                    "onnx_validated": False,
                    "onnx_validation_status": "FAILED",
                    "onnx_validation_reason": str(exc),
                }
        stages["export"] = export_status
        pruning_result.extra_metrics.update({"stages": stages, "run_dir": str(artifacts.run_dir)})
        summary = _benchmark_summary(baseline_metrics, final_metrics, complexity_after, cfg.pruning.amount, diagnostics)
        artifact_paths["result"] = artifacts.write_json(
            "result.json",
            {
                **summary,
                "pruning": pruning_result,
                "baseline_metrics": baseline_metrics,
                "final_metrics": final_metrics,
                "complexity_before": complexity_before,
                "complexity_after": complexity_after,
                "stages": stages,
            },
        )
        return ExperimentResult(
            run_dir=str(artifacts.run_dir),
            pruning=pruning_result,
            baseline_metrics=baseline_metrics,
            final_metrics=final_metrics,
            sensitivity=sensitivity_summary,
            selection=selection_summary,
            complexity_before=complexity_before,
            complexity_after=complexity_after,
            artifacts=artifact_paths,
        )

    def _needs_evaluation(self) -> bool:
        return (self.config.evaluation.enabled or self.config.sensitivity.enabled
                or self.config.targets.max_map50_drop is not None or self.config.targets.max_map50_95_drop is not None)

    def _require_evaluator(self, stage: str) -> None:
        if self.evaluator is None:
            raise ConfigValidationException(
                f"{stage} requires evaluation.callback or an evaluator passed to UnifiedPruningPipeline."
            )

    def _evaluate(self, model: nn.Module, stage: str) -> Dict[str, float]:
        self._require_evaluator(stage)
        value = self._invoke(self.evaluator, model, stage, self.config.evaluation.kwargs)
        return normalize_evaluation(value, self.config.evaluation.metric).metrics

    def _metric(self, metrics: Dict[str, float]) -> float:
        try:
            return metrics[self.config.evaluation.metric]
        except KeyError as exc:
            raise ConfigValidationException(
                f"Evaluation result does not contain configured metric '{self.config.evaluation.metric}'."
            ) from exc

    @staticmethod
    def _load_callback(path: Optional[str]) -> Optional[Callable[..., Any]]:
        if not path:
            return None
        if ":" not in path:
            raise ConfigValidationException("Callbacks must use 'package.module:function' notation.")
        module_name, attribute = path.split(":", 1)
        try:
            callback = getattr(importlib.import_module(module_name), attribute)
            if not callable(callback):
                raise ConfigValidationException(f"Callback '{path}' is not callable")
            return callback
        except (ImportError, AttributeError) as exc:
            raise ConfigValidationException(f"Could not load callback '{path}': {exc}") from exc

    def _invoke(
        self,
        callback: Callable[..., Any],
        model: nn.Module,
        stage: str,
        kwargs: Dict[str, Any],
        *,
        adapter: Optional[Any] = None,
        regularization: Optional[Any] = None,
    ) -> Any:
        signature = inspect.signature(callback)
        candidates = {"model": model, "config": self.config, "device": next(model.parameters()).device, "stage": stage}
        if adapter is not None:
            candidates["adapter"] = adapter
        if regularization is not None:
            candidates["regularization"] = regularization
        candidates.update(kwargs)
        accepts_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
        named = {key: value for key, value in candidates.items() if accepts_kwargs or key in signature.parameters}
        if named:
            return callback(**named)
        if signature.parameters:
            return callback(model)
        return callback()

    @staticmethod
    def _resolve_recovery_output(value: Any, current_model: nn.Module) -> tuple[nn.Module, Dict[str, Any]]:
        """Accept legacy callbacks and structured recovery results without ambiguity."""
        if value is None:
            return current_model, {}
        if isinstance(value, nn.Module):
            return value, {}
        if not isinstance(value, dict):
            raise TypeError("Recovery callback must return None, an nn.Module, or a mapping containing 'model'.")
        model = value.get("model", current_model)
        if not isinstance(model, nn.Module):
            raise TypeError("Recovery result 'model' must be an nn.Module.")
        return model, {key: item for key, item in value.items() if key != "model"}

    def _validate_checkpoint_artifact(self, path: str, device: torch.device) -> Dict[str, Any]:
        """Reload the emitted artifact through the public adapter boundary."""
        try:
            options = {"num_classes": self.config.model.num_classes} if self.config.model.num_classes is not None else {}
            restored, _ = ModelLoader.load(self.config.model.name, path, device, **options)
            engine = PruningEngine(
                self.config.model.name,
                self.config.pruning.pruner,
                self.config.pruning.criterion,
                self.config.pruning.granularity,
            )
            restored_adapter = engine.adapter_cls(restored)
            forward_ok = ModelValidator.validate_forward(restored, restored_adapter.get_dummy_input(device))
            if not forward_ok:
                return {
                    "checkpoint_reload_validated": False,
                    "checkpoint_validation_reason": "reloaded model failed forward validation",
                }
            # The reloaded artifact is evaluated with the final evaluator contract.
            # ``final`` is deliberately retained as the callback stage for backward
            # compatibility; the enclosing artifact result identifies this pass as
            # the reload validation rather than a second in-memory final metric.
            reloaded_metrics = self._evaluate(restored, "final") if self._needs_evaluation() else None
            return {
                "checkpoint_reload_validated": True,
                "checkpoint_validation_reason": None,
                "reloaded_metrics": reloaded_metrics,
            }
        except Exception as exc:
            return {
                "checkpoint_reload_validated": False,
                "checkpoint_validation_reason": f"{type(exc).__name__}: {exc}",
            }


def _policy_kind(engine: PruningEngine) -> str:
    """Name the scoring/policy semantics without duplicating pruner logic."""
    mode = engine.pruner_cls.pruning_mode
    if mode == "structured":
        return "structural_sensitivity"
    if mode == "unstructured":
        return "global_weight_score" if getattr(engine.criterion_cls, "global_selection", False) else "weight_score"
    if mode == "depth":
        return "adapter_structural_score"
    if mode == "head":
        return "attention_head_score"
    if mode == "nm":
        return "nm_pattern_score"
    if mode == "block_sparse":
        return "block_pattern_score"
    return "pruner_defined_score"


def _uses_layerwise_sensitivity(engine: PruningEngine) -> bool:
    """Only mechanisms with a meaningful layer-ratio policy run generic probes."""
    return engine.pruner_cls.pruning_mode == "structured"


def _sensitivity_to_dict(result) -> Dict[str, Any]:
    return {
        "baseline_score": result.baseline_score,
        "metadata": result.metadata,
        "profiles": {
            str(index): {
                "layer_name": profile.layer_name,
                "points": {str(rate): asdict(point) for rate, point in profile.points.items()},
            }
            for index, profile in result.profiles.items()
        },
    }


def _invalid_baseline_reason(cfg: FrameworkConfig, metrics: Dict[str, float]) -> str | None:
    """Reject known-invalid random/native classification baselines before mutation."""
    if any(not math.isfinite(float(value)) for value in metrics.values()):
        return "baseline metrics contain NaN or Inf"
    accuracy = metrics.get("accuracy")
    # The native LeNet configs must not silently benchmark random weights on
    # the 47-class dataset. A trained model is expected to clear this floor.
    if cfg.model.name.lower() in {"lenet5", "lenet"} and cfg.dataset.name == "custom_47labels":
        if str(cfg.model.weights).lower() in {"random", "none", ""}:
            return "native LeNet-5 uses random/no weights for custom_47labels"
        if accuracy is not None and accuracy < 0.5:
            return f"native LeNet-5 accuracy {accuracy:.4f} is below required 0.50"
    return None


def _model_diagnostics(model: nn.Module) -> Dict[str, Any]:
    layers, zero, total, nonfinite = {}, 0, 0, 0
    for name, parameter in model.named_parameters():
        value = parameter.detach()
        count = value.numel()
        zeros = int(value.eq(0).sum())
        invalid = int((~torch.isfinite(value)).sum())
        layers[name] = {"zero": zeros, "total": count, "sparsity_pct": 100.0 * zeros / max(1, count),
                        "min": float(value.min()), "max": float(value.max()), "mean": float(value.mean()), "std": float(value.std()), "nan_inf": invalid}
        zero += zeros; total += count; nonfinite += invalid
    return {"finite": nonfinite == 0, "nan_inf": nonfinite, "actual_sparsity": zero / max(1, total),
            "remaining_weight_pct": 100.0 * (total - zero) / max(1, total), "layers": layers}


def _benchmark_summary(baseline, final, complexity, requested_sparsity, diagnostics) -> Dict[str, Any]:
    baseline, final = baseline or {}, final or {}
    deltas = {f"{name}_delta": final[name] - baseline[name] for name in ("accuracy", "loss", "precision", "recall", "f1", "map50", "map50_95") if name in baseline and name in final}
    quality_key = "accuracy" if "accuracy_delta" in deltas else "map50_95"
    drop = -deltas.get(f"{quality_key}_delta", 0.0)
    detection = quality_key == "map50_95"
    limits = (0.01, 0.03, 0.10, 0.20 if detection else 0.30)
    epsilon = 1e-12
    status = ("PASS" if drop <= limits[0] + epsilon else "GOOD" if drop <= limits[1] + epsilon else
              "DEGRADED" if drop <= limits[2] + epsilon else "SEVERE" if drop <= limits[3] + epsilon else "FAILED")
    if "map50_95_delta" in deltas:
        deltas["map5095_delta"] = deltas["map50_95_delta"]
    return {"baseline_valid": True, "requested_sparsity": requested_sparsity,
            "actual_sparsity": diagnostics["actual_sparsity"], "status": status,
            "failure_reason": None if status != "FAILED" else "quality collapse after pruning",
            **{name: final.get(name) for name in ("precision", "recall", "f1", "map50", "map50_95")}, **deltas}
