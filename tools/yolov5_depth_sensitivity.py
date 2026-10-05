"""Sensitivity-guided structural depth pruning for YOLOv5 C3/CSP blocks.

The generic sensitivity stage varies a layer's channel pruning ratio.  A depth
operation is different: its atomic action is removal of a declared Bottleneck
inside a C3/CSP module.  This runner ablates every such action independently,
measures validation mAP, then applies the safest set to a fresh model.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

import torch

# Allow ``python tools/yolov5_depth_sensitivity.py`` from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prune_framework.core.config import FrameworkConfig
from prune_framework.core.engine import PruningEngine
from prune_framework.core.experiment import ExperimentArtifacts, seed_everything
from prune_framework.modules.evaluation.benchmark import LatencyBenchmark
from prune_framework.modules.evaluation.complexity import measure_complexity
from prune_framework.modules.export.exporter import ModelExporter
from prune_framework.modules.model.loader import ModelLoader
from prune_framework.modules.model.snapshot import ModelSnapshot


def _evaluate(model: torch.nn.Module, cfg: FrameworkConfig) -> dict[str, float]:
    if not cfg.evaluation.callback:
        raise ValueError("YOLO depth sensitivity requires evaluation.callback.")
    if ":" not in cfg.evaluation.callback:
        raise ValueError("evaluation.callback must use 'module:function' notation.")
    module_name, function_name = cfg.evaluation.callback.split(":", 1)
    callback = getattr(__import__(module_name, fromlist=[function_name]), function_name)
    value = callback(model=model, **cfg.evaluation.kwargs)
    return dict(value.metrics)


def _relative_drop(baseline: float, candidate: float) -> float:
    if baseline == 0:
        return 0.0 if candidate == 0 else float("inf")
    return (baseline - candidate) / abs(baseline)


def _apply_named_blocks(model: torch.nn.Module, cfg: FrameworkConfig, names: list[str]):
    engine = PruningEngine(cfg.model.name, "depth", cfg.pruning.criterion, cfg.pruning.granularity)
    return engine.execute(
        model,
        {"amount": 0.0, "block_names": names, "allow_noop": not names},
        verify_forward=True,
    )


def run(cfg: FrameworkConfig, *, build_plan_only: bool = False, max_probes: int | None = None) -> dict[str, Any]:
    if cfg.pruning.pruner.lower() not in {"depth", "layer", "layer_depth", "structural_block"}:
        raise ValueError("This runner only supports pruning.pruner: depth (or layer alias).")
    if not cfg.sensitivity.enabled:
        raise ValueError("Set sensitivity.enabled: true; this runner always records ablation sensitivity.")
    if not cfg.evaluation.enabled:
        raise ValueError("Set evaluation.enabled: true; selecting safe depth blocks requires mAP evaluation.")

    seed_everything(cfg.experiment.seed, cfg.experiment.deterministic)
    device = torch.device(cfg.model.device if cfg.model.device == "cuda" and torch.cuda.is_available() else "cpu")
    model, checkpoint = ModelLoader.load(cfg.model.name, cfg.model.weights, device)
    model.eval()
    snapshot = ModelSnapshot(model)
    engine = PruningEngine(cfg.model.name, "depth", cfg.pruning.criterion, cfg.pruning.granularity)
    targets = engine.targets(model)
    if not targets:
        raise ValueError("YOLO adapter found no removable C3/CSP Bottleneck targets.")
    names = [target.name for target in targets]
    if max_probes is not None:
        if max_probes < 1:
            raise ValueError("--max-probes must be positive.")
        names = names[:max_probes]

    artifacts = ExperimentArtifacts(cfg.experiment.output_dir, cfg.experiment.name)
    artifacts.write_json("config.json", cfg.to_dict())
    artifacts.write_json("declared_targets.json", {"targets": names, "target_count": len(names)})
    if build_plan_only:
        return {"run_dir": str(artifacts.run_dir), "declared_targets": names, "probes": []}

    baseline_metrics = _evaluate(model, cfg)
    metric_name = cfg.evaluation.metric
    if metric_name not in baseline_metrics:
        raise ValueError(f"Evaluation did not return configured metric {metric_name!r}: {sorted(baseline_metrics)}")
    baseline_score = float(baseline_metrics[metric_name])
    dummy = engine.adapter_cls(model).get_dummy_input(device)
    complexity_before = measure_complexity(model, dummy, include_flops=cfg.benchmark.flops)

    probes: list[dict[str, Any]] = []
    for name in names:
        candidate = snapshot.restore().to(device).eval()
        try:
            result = _apply_named_blocks(candidate, cfg, [name])
            metrics = _evaluate(candidate, cfg)
            score = float(metrics[metric_name])
            probes.append({
                "target": name,
                "metrics": metrics,
                "score": score,
                "absolute_drop": baseline_score - score,
                "relative_drop": _relative_drop(baseline_score, score),
                "params_after": result.params_after,
                "params_removed": result.params_before - result.params_after,
                "forward_verified": result.forward_verified,
                "error": None,
            })
        except Exception as exc:  # A failed ablation is intentionally never selected.
            probes.append({"target": name, "error": f"{type(exc).__name__}: {exc}"})

    max_drop = cfg.sensitivity.max_allowed_relative_drop
    eligible = [item for item in probes if item.get("error") is None and item["relative_drop"] <= max_drop]
    desired = max(1, int(round(len(targets) * cfg.pruning.amount))) if cfg.pruning.amount > 0 else 0
    selected = sorted(
        eligible,
        key=lambda item: (item["relative_drop"], -item["params_removed"], item["target"]),
    )[:desired]
    selected_names = [item["target"] for item in selected]
    artifacts.write_json("depth_sensitivity.json", {
        "metric": metric_name,
        "baseline_metrics": baseline_metrics,
        "baseline_score": baseline_score,
        "max_allowed_relative_drop": max_drop,
        "probes": probes,
    })
    artifacts.write_json("selection.json", {
        "requested_fraction": cfg.pruning.amount,
        "requested_blocks": desired,
        "eligible_blocks": len(eligible),
        "selected_blocks": selected_names,
        "selection_rule": "lowest relative mAP drop, then largest parameter saving",
    })
    if desired and not selected_names:
        raise RuntimeError(f"No removable YOLO blocks met max_allowed_relative_drop={max_drop}.")

    pruned = snapshot.restore().to(device).eval()
    pruning = _apply_named_blocks(pruned, cfg, selected_names)
    final_metrics = _evaluate(pruned, cfg)
    complexity_after = measure_complexity(pruned, dummy, include_flops=cfg.benchmark.flops)
    if cfg.benchmark.enabled and cfg.benchmark.latency:
        pruning.benchmark = LatencyBenchmark(cfg.benchmark.warmup, cfg.benchmark.runs).benchmark(pruned, dummy)

    ModelExporter.export_checkpoint(pruned, cfg.output_path, checkpoint)
    result = {
        "baseline_metrics": baseline_metrics,
        "final_metrics": final_metrics,
        "complexity_before": asdict(complexity_before),
        "complexity_after": asdict(complexity_after),
        "pruning": asdict(pruning),
        "selected_blocks": selected_names,
        "checkpoint": cfg.output_path,
    }
    artifacts.write_json("result.json", result)
    return {"run_dir": str(artifacts.run_dir), **result}


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv5 C3/CSP depth pruning with independent block sensitivity.")
    parser.add_argument("--config", required=True, help="Framework YAML configuration")
    parser.add_argument("--build-plan-only", action="store_true", help="List adapter-declared blocks without mAP probes")
    parser.add_argument("--max-probes", type=int, default=None, help="Limit ablation probes (smoke testing only)")
    args = parser.parse_args()
    cfg = FrameworkConfig.from_yaml(args.config)
    result = run(cfg, build_plan_only=args.build_plan_only, max_probes=args.max_probes)
    print(result)


if __name__ == "__main__":
    main()
