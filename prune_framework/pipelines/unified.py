"""The architecture-agnostic, stage-based pruning experiment pipeline."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import asdict
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn

from prune_framework.contracts.evaluation import EvaluationResult
from prune_framework.core.config import FrameworkConfig
from prune_framework.core.engine import PruningEngine
from prune_framework.core.exceptions import ConfigValidationException, PruningExecutionError
from prune_framework.core.experiment import ExperimentArtifacts, seed_everything
from prune_framework.core.results import ExperimentResult
from prune_framework.modules.analysis.layer_selection import LayerSelectorModule
from prune_framework.modules.analysis.sensitivity import SensitivityAnalyzer
from prune_framework.modules.evaluation.benchmark import LatencyBenchmark
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

    def run(self) -> ExperimentResult:
        cfg = self.config
        seed_everything(cfg.experiment.seed, cfg.experiment.deterministic)
        artifacts = ExperimentArtifacts(cfg.experiment.output_dir, cfg.experiment.name)
        artifact_paths = {"config": artifacts.write_json("config.resolved.json", cfg.to_dict())}
        stages: Dict[str, Any] = {}

        device = torch.device(cfg.model.device if torch.cuda.is_available() else "cpu")
        model, checkpoint = ModelLoader.load(cfg.model.name, cfg.model.weights, device)
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
        adapter = engine.adapter_cls(model)
        dummy_input = adapter.get_dummy_input(device)
        stages["load_model"] = {"device": str(device), "model": cfg.model.name}

        complexity_before = None
        if cfg.benchmark.params or cfg.benchmark.flops:
            complexity_before = measure_complexity(model, dummy_input, include_flops=cfg.benchmark.flops)
            stages["baseline_cost"] = asdict(complexity_before)

        baseline_metrics = self._evaluate(model, "baseline") if self._needs_evaluation() else None
        if baseline_metrics is not None:
            stages["baseline_evaluation"] = baseline_metrics

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
            recovered = self._invoke(
                self.recovery,
                model,
                "regularization",
                cfg.recovery.kwargs,
                adapter=adapter,
                regularization=regularization,
            )
            if isinstance(recovered, nn.Module):
                model = recovered
                adapter = engine.adapter_cls(model)
                regularization.adapter = adapter
            regularization_summary = regularization.state_dict()
            artifact_paths["regularization"] = artifacts.write_json("regularization.json", regularization_summary)
            stages["regularization"] = regularization_summary

        selection_summary = None
        sensitivity_summary = None
        pruning_params = cfg.pruning.layer_params or cfg.pruning.amount
        if cfg.sensitivity.enabled:
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
            )
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

        if cfg.pruning.pruner.lower() in {"structured", "structured_channel"}:
            # The preflight proves that this adapter and its dummy input can be
            # traced before any weights are physically modified.
            engine.build_dependency_graph(model)
            stages["dependency_graph"] = {"status": "built"}

        calibration_result = None
        criterion_cls = engine.criterion_cls
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
            if not adapter.supports_gradient_calibration():
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
            calibration_result = HigherOrderCalibrationRunner(seed=cfg.pruning.calibration_seed).run(
                model=model,
                targets=targets,
                batches=adapter.get_gradient_calibration_batches(
                    cfg.pruning.calibration_batches,
                    device,
                    cfg.pruning.calibration_seed,
                    cfg.pruning.calibration_batch_size,
                ),
                loss_fn=adapter.build_gradient_calibration_loss(),
            )
            calibration_summary = calibration_result.describe()
            artifact_paths["grasp_calibration"] = artifacts.write_json("grasp_calibration.json", calibration_summary)
            stages["importance_calibration"] = {"method": "grasp", **calibration_summary}
        elif getattr(criterion_cls, "requires_gradients", False):
            if not adapter.supports_gradient_calibration():
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
                seed=cfg.pruning.calibration_seed,
                accumulate=cfg.pruning.calibration_accumulate,
            )
            calibration_result = runner.run(
                model=model,
                targets=targets,
                batches=adapter.get_gradient_calibration_batches(
                    cfg.pruning.calibration_batches,
                    device,
                    cfg.pruning.calibration_seed,
                    cfg.pruning.calibration_batch_size,
                ),
                loss_fn=adapter.build_gradient_calibration_loss(),
            )
            calibration_summary = calibration_result.describe()
            artifact_paths["calibration"] = artifacts.write_json("calibration.json", calibration_summary)
            stages["importance_calibration"] = calibration_summary

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
            "amount": cfg.pruning.amount,
            "pruning_params": pruning_params,
            "global_pruning": cfg.pruning.global_pruning,
            "iterative_steps": cfg.pruning.iterative_steps,
            "min_channels": cfg.pruning.min_channels,
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
            elif getattr(criterion_cls, "requires_higher_order_calibration", False):
                pruning_config["higher_order_calibration"] = calibration_result
            else:
                pruning_config["gradient_calibration"] = calibration_result
        pruning_result = engine.execute(model, pruning_config, verify_forward=True)
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
            recovered = self._invoke(self.recovery, model, "recovery", cfg.recovery.kwargs)
            if isinstance(recovered, nn.Module):
                model = recovered
                adapter = engine.adapter_cls(model)
            stages["recovery"] = {"epochs": cfg.recovery.epochs, "callback": cfg.recovery.callback}

        final_metrics = self._evaluate(model, "final") if self._needs_evaluation() else None
        if final_metrics is not None:
            stages["final_evaluation"] = final_metrics

        complexity_after = None
        if cfg.benchmark.params or cfg.benchmark.flops:
            complexity_after = measure_complexity(
                model,
                adapter.get_dummy_input(device),
                include_flops=cfg.benchmark.flops,
            )
            stages["final_cost"] = asdict(complexity_after)
        if cfg.benchmark.enabled and cfg.benchmark.latency:
            benchmark = LatencyBenchmark(cfg.benchmark.warmup, cfg.benchmark.runs).benchmark(
                model, adapter.get_dummy_input(device)
            )
            pruning_result.benchmark = benchmark
            stages["latency"] = asdict(benchmark)

        if cfg.export.wants_onnx():
            ModelExporter.export_onnx(
                model,
                adapter.get_dummy_input(device),
                cfg.export.output_path,
                opset_version=cfg.export.opset_version,
            )
            artifact_paths["onnx"] = cfg.export.output_path
            stages["export"] = {"onnx": cfg.export.output_path}

        ModelExporter.export_checkpoint(model, cfg.output_path, checkpoint)
        artifact_paths["checkpoint"] = cfg.output_path
        pruning_result.extra_metrics.update({"stages": stages, "run_dir": str(artifacts.run_dir)})
        artifact_paths["result"] = artifacts.write_json(
            "result.json",
            {
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
        return self.config.evaluation.enabled or self.config.sensitivity.enabled

    def _require_evaluator(self, stage: str) -> None:
        if self.evaluator is None:
            raise ConfigValidationException(
                f"{stage} requires evaluation.callback or an evaluator passed to UnifiedPruningPipeline."
            )

    def _evaluate(self, model: nn.Module, stage: str) -> Dict[str, float]:
        self._require_evaluator(stage)
        value = self._invoke(self.evaluator, model, stage, self.config.evaluation.kwargs)
        if isinstance(value, EvaluationResult):
            return dict(value.metrics)
        if isinstance(value, (int, float)):
            return {self.config.evaluation.metric: float(value)}
        if isinstance(value, dict) and all(isinstance(v, (int, float)) for v in value.values()):
            return {str(key): float(metric) for key, metric in value.items()}
        raise TypeError("An evaluation callback must return a float, metric dict, or EvaluationResult.")

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
            return getattr(importlib.import_module(module_name), attribute)
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
