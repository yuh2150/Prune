"""Measure P/R/F1/AP for retained YOLO checkpoints on the original COCO-500 list."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from experiments.yolov5 import evaluate
from prune_framework.modules.model.loader import ModelLoader
import prune_framework.plugins
from tools.reevaluate_artifact_quality import atomic_json
from utils.datasets import create_dataloader

QUALITY = ("precision", "recall", "f1", "map50", "map50_95", "map")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--report", default="outputs/benchmark-quality-20261007/yolo_quality_reevaluation.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    dataset = yaml.safe_load(Path("data/coco500.yaml").read_text())
    # Historical YAML was renamed; use the retained original image list.
    dataset["path"] = str(Path("coco").resolve())
    dataset["val"] = str(Path("coco/val2017_500.txt").resolve())
    dataset.pop("train", None)
    dataset.pop("test", None)
    loader = create_dataloader(dataset["val"], 640, 8, 32, workers=0, rect=True, pad=.5)[0]
    if len(loader.dataset) != 500:
        raise ValueError("Expected original COCO-500 validation split")
    base_path = "weights/yolov5s.pt"
    original, _ = ModelLoader.load("yolov5", base_path, torch.device("cpu"))
    baseline = evaluate(original, data=dataset, dataloader=loader, batch_size=8, imgsz=640)
    records = []
    for path in sorted(Path("artifacts").rglob("result.json")):
        result = json.loads(path.read_text()) if path.stat().st_size < 25_000_000 else {}
        if result.get("pruning", {}).get("model_name") != "yolov5":
            continue
        record = {"source": str(path)}
        try:
            cfg_path = path.parent / "config.resolved.json"
            cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
            requested = Path(result.get("checkpoint") or cfg.get("output_path", ""))
            checkpoint_path = requested if requested.is_file() else Path("artifacts") / requested
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Final checkpoint missing: {requested}")
            candidate, _ = ModelLoader.load("yolov5", str(checkpoint_path), torch.device("cpu"))
            expected_params = result.get("complexity_after", {}).get("params")
            if expected_params is not None and sum(p.numel() for p in candidate.parameters()) != expected_params:
                raise ValueError("Checkpoint dimensions differ from the saved post-prune model")
            final = evaluate(candidate, data=dataset, dataloader=loader, batch_size=8, imgsz=640)
            old_base = copy.deepcopy(result["baseline_metrics"])
            old_final = copy.deepcopy(result["final_metrics"])
            provenance = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(), "device": "cpu",
                "torch_version": torch.__version__, "restoration": "saved_final_checkpoint",
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                "baseline_checkpoint": base_path, "num_samples": final.num_samples,
                "validation_list": dataset["val"],
                "validation_list_sha256": hashlib.sha256(Path(dataset["val"]).read_bytes()).hexdigest(),
                "protocol": {"imgsz": 640, "batch_size": 8, "conf_thres": .001,
                             "nms_iou_thres": .6, "average": "macro", "f1_iou": .5,
                             "operating_point": "maximum_macro_f1", "class_scope": "targets"},
                "historical_baseline_metrics": old_base, "historical_final_metrics": old_final,
                "measured_baseline": baseline.metrics, "measured_final": final.metrics,
            }
            for key, measured in (("baseline_metrics", baseline.metrics), ("final_metrics", final.metrics)):
                result[key].update({name: measured[name] for name in QUALITY})
            for name in QUALITY:
                result[name] = final.metrics[name]
                result[f"{name}_delta"] = final.metrics[name] - baseline.metrics[name]
            replicas = [result.get("stages", {}), result["pruning"].get("extra_metrics", {}).get("stages", {})]
            for replica in replicas:
                for key, measured in (("baseline_evaluation", baseline.metrics), ("final_evaluation", final.metrics)):
                    if isinstance(replica.get(key), dict):
                        replica[key].update({name: measured[name] for name in QUALITY})
                if isinstance(replica.get("latency"), dict):
                    replica["latency"].update({name: final.metrics[name] for name in QUALITY})
            if isinstance(result["pruning"].get("benchmark"), dict):
                result["pruning"]["benchmark"].update({name: final.metrics[name] for name in QUALITY})
            result["quality_reevaluation"] = provenance
            if args.write:
                backup = Path("/tmp/prune-quality-reevaluation-backup") / path
                backup.parent.mkdir(parents=True, exist_ok=True)
                if not backup.exists():
                    shutil.copy2(path, backup)
                atomic_json(path, result)
                atomic_json(path.parent / "quality_reevaluation.json", provenance)
                baseline_path = path.parent / "baseline.json"
                if baseline_path.exists():
                    baseline_backup = Path("/tmp/prune-quality-reevaluation-backup") / baseline_path
                    if not baseline_backup.exists():
                        shutil.copy2(baseline_path, baseline_backup)
                    old = json.loads(baseline_path.read_text())
                    old.update({name: baseline.metrics[name] for name in QUALITY})
                    atomic_json(baseline_path, old)
            record.update(status="updated" if args.write else "measured", baseline=baseline.metrics,
                          final=final.metrics, num_samples=final.num_samples)
        except Exception as exc:
            record.update(status="unavailable", reason=f"{type(exc).__name__}: {exc}")
        records.append(record)
        print(json.dumps(record), flush=True)
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, {"runs": records})


if __name__ == "__main__":
    main()
