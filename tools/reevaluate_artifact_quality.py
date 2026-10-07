"""Measure missing classifier P/R/F1 from historical checkpoints or saved plans.

Run from the repository root with ``python -m tools.reevaluate_artifact_quality
--write``. Only runs whose original accuracy/loss reproduce are updated.
No pruning decisions are recalculated, and no checkpoints are overwritten.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

import prune_framework.plugins  # register local adapters and pruners
from prune_framework.contracts.targets import PrunableTarget, PruningGroup, PruningPlan, TargetType
from prune_framework.core.config import FrameworkConfig
from prune_framework.core.experiment import seed_everything
from prune_framework.core.registry import PluginRegistry
from prune_framework.datasets import create_classification_dataloaders
from prune_framework.modules.evaluation.classification import evaluate_classification
from prune_framework.modules.model.loader import ModelLoader
from prune_framework.plugins.pruners.unstructured import UnstructuredPruner

QUALITY = ("precision", "recall", "f1")
CLASSIFIERS = {"lenet5", "lenet5_emnist_onnx", "resnet18", "torchvision_resnet18"}


def resolved_config(run_dir, imagenet_root=None):
    cfg = FrameworkConfig.from_dict(json.loads((run_dir / "config.resolved.json").read_text()))
    cfg.model.device = "cpu"
    if cfg.dataset.name == "custom_47labels" and not Path(cfg.dataset.root).exists():
        cfg.dataset.root = "datasets/Dataset_MNIST_47Labels_1000images"
    if cfg.dataset.name == "imagenet1k_subset" and imagenet_root:
        cfg.dataset.root = imagenet_root
    return cfg


def replay_plan(model, model_name, description):
    """Rebind saved decisions to fresh modules, without rerunning saliency."""
    adapter = PluginRegistry.get_model_adapter(model_name)(model)
    modules = dict(model.named_modules())
    blocks = {target.name: target for target in adapter.get_structural_block_targets()}
    groups = []
    for item in description["groups"]:
        primary = item["primary"]
        name = primary["name"]
        if item["operation"] == "remove_structural_block":
            target = blocks[name]
        else:
            target = PrunableTarget(name, modules[name], TargetType(primary["target_type"]),
                                    primary.get("metadata", {}))
        groups.append(PruningGroup(primary=target, operation=item["operation"],
                                   indices=item["indices"], validated=item["validated"]))
    plan = PruningPlan(description["pruner"], groups, copy.deepcopy(description.get("metadata", {})))
    if all(group.operation == "mask_weight" for group in groups):
        pruner = UnstructuredPruner()
    else:
        pruner = PluginRegistry.get_pruner(description["pruner"])()
    pruner.apply_plan(plan, adapter)
    return model


def verify_metrics(previous, measured, stage):
    if not previous or "accuracy" not in previous:
        raise ValueError(f"{stage}: historical accuracy is absent; cannot verify model identity")
    for name in ("accuracy", "loss"):
        if name in previous and not math.isclose(previous[name], measured[name], rel_tol=1e-4, abs_tol=1e-5):
            raise ValueError(f"{stage} {name} mismatch: old={previous[name]}, measured={measured[name]}")


def augment_result(result, baseline, final, provenance):
    """Preserve old scores/timings, add only verified new quality measurements."""
    for key, measured in (("baseline_metrics", baseline), ("final_metrics", final)):
        result[key].update({name: measured[name] for name in QUALITY})
    for name in QUALITY:
        result[name] = final[name]
        result[f"{name}_delta"] = final[name] - baseline[name]
    stages = result.get("stages", {})
    replicas = [stages, result.get("pruning", {}).get("extra_metrics", {}).get("stages", {})]
    for replica in replicas:
        for key, measured in (("baseline_evaluation", baseline), ("final_evaluation", final)):
            if isinstance(replica.get(key), dict):
                replica[key].update({name: measured[name] for name in QUALITY})
        for nested in (replica.get("apply_pruning", {}).get("benchmark"), replica.get("latency")):
            if isinstance(nested, dict):
                nested.update({name: final[name] for name in QUALITY})
    benchmark = result.get("pruning", {}).get("benchmark")
    if isinstance(benchmark, dict):
        benchmark.update({name: final[name] for name in QUALITY})
    result["quality_reevaluation"] = provenance
    return result


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--imagenet-root")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--report", default="outputs/benchmark-quality-20261007/quality_reevaluation.json")
    parser.add_argument("--backup-dir", default="/tmp/prune-quality-reevaluation-backup")
    args = parser.parse_args()
    torch.set_num_threads(4)
    cache = {}
    records = []
    timestamp = datetime.now(timezone.utc).isoformat()
    for path in sorted(Path(args.artifacts).rglob("result.json")):
        record = {"source": str(path)}
        try:
            config_path = path.parent / "config.resolved.json"
            if not config_path.exists():
                record.update(status="skipped", reason="no resolved classifier config")
                continue
            cfg = resolved_config(path.parent, args.imagenet_root)
            record["model"] = cfg.model.name
            if cfg.model.name not in CLASSIFIERS:
                record.update(status="skipped", reason="detection metrics use their original evaluator")
                continue
            result = json.loads(path.read_text())
            if cfg.recovery.enabled or cfg.regularization.enabled:
                raise ValueError("trained runs require their exact final checkpoint; plan replay is insufficient")
            key = json.dumps([asdict(cfg.dataset), asdict(cfg.model), cfg.experiment.seed], sort_keys=True)
            if key not in cache:
                seed_everything(cfg.experiment.seed, cfg.experiment.deterministic)
                loader = create_classification_dataloaders(cfg.dataset, seed=cfg.experiment.seed).validation
                # Small local classification splits: materialize once, preserve order.
                batches = list(loader)
                loader = DataLoader(TensorDataset(torch.cat([x for x, _ in batches]),
                                                  torch.cat([y for _, y in batches])),
                                    batch_size=cfg.dataset.batch_size)
                # The original pipeline initializes the model before loader iteration.
                seed_everything(cfg.experiment.seed, cfg.experiment.deterministic)
                model, _ = ModelLoader.load(cfg.model.name, cfg.model.weights, torch.device("cpu"),
                                            num_classes=cfg.model.num_classes)
                baseline = evaluate_classification(model, loader, "cpu")
                cache[key] = (model, loader, baseline)
            original, loader, baseline = cache[key]
            verify_metrics(result.get("baseline_metrics"), baseline.metrics, "baseline")
            plan_path = path.parent / "pruning_plan.json"
            plan = json.loads(plan_path.read_text())
            model = replay_plan(copy.deepcopy(original), cfg.model.name, plan)
            final = evaluate_classification(model, loader, "cpu")
            verify_metrics(result.get("final_metrics"), final.metrics, "final")
            provenance = {
                "timestamp_utc": timestamp, "device": "cpu", "torch_version": torch.__version__,
                "restoration": "original_weights_and_saved_pruning_plan", "plan": str(plan_path),
                "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                "weights": cfg.model.weights, "dataset": asdict(cfg.dataset), "seed": cfg.experiment.seed,
                "weights_sha256": (hashlib.sha256(Path(cfg.model.weights).read_bytes()).hexdigest()
                                   if Path(cfg.model.weights).is_file() else None),
                "num_samples": final.num_samples, **final.metadata,
                "verification": {"baseline": baseline.metrics, "final": final.metrics,
                                 "accuracy_loss_match": True, "relative_tolerance": 1e-4, "absolute_tolerance": 1e-5},
            }
            if args.write:
                backup = Path(args.backup_dir) / path
                backup.parent.mkdir(parents=True, exist_ok=True)
                if not backup.exists():
                    shutil.copy2(path, backup)
                atomic_json(path, augment_result(result, baseline.metrics, final.metrics, provenance))
                baseline_file = path.parent / "baseline.json"
                if baseline_file.exists():
                    old = json.loads(baseline_file.read_text())
                    if isinstance(old, dict):
                        backup_base = Path(args.backup_dir) / baseline_file
                        if not backup_base.exists():
                            shutil.copy2(baseline_file, backup_base)
                        old.update({name: baseline.metrics[name] for name in QUALITY})
                        atomic_json(baseline_file, old)
                atomic_json(path.parent / "quality_reevaluation.json", provenance)
            record.update(status="updated" if args.write else "verified", baseline=baseline.metrics,
                          final=final.metrics, num_samples=final.num_samples)
        except Exception as exc:
            record.update(status="unavailable", reason=f"{type(exc).__name__}: {exc}")
        finally:
            records.append(record)
            print(json.dumps(record), flush=True)
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, {"timestamp_utc": timestamp, "write": args.write, "runs": records})


if __name__ == "__main__":
    main()
