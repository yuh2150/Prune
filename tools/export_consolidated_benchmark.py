"""Create one auditable benchmark report across LeNet5, ResNet18, YOLOv5s and RT-DETR.

Only completed artifact metrics are used.  The report deliberately separates
final eligible measurements from sensitivity/smoke, invalid, failed and exact
duplicate runs; none of the latter influence rankings.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from openpyxl import Workbook

from export_benchmark_excel import NA, collect, metric, num, value, write_sheet


OUT = Path("benchmarks/pruning_benchmark_consolidated.xlsx")
MODEL_ORDER = ("LeNet5", "ResNet18", "YOLOv5s", "RT-DETR")


def canonical_model(name: str) -> str | None:
    lower = name.lower()
    if "lenet" in lower:
        return "LeNet5"
    if "resnet18" in lower:
        return "ResNet18"
    if "yolov5" in lower:
        return "YOLOv5s"
    if "rtdetr" in lower or "rt-detr" in lower:
        return "RT-DETR"
    return None


def is_detection(run: dict[str, Any]) -> bool:
    return any(metric(run["final"], key) is not None for key in ("map50", "map50_95", "map5095"))


def raw_actual(run: dict[str, Any]) -> float | None:
    return num(run["actual_pct"])


def run_signature(run: dict[str, Any]) -> tuple[Any, ...]:
    """Only exact measured duplicates share a signature; similar runs do not."""
    base = run["baseline"]
    final = run["final"]
    return (
        canonical_model(run["model"]), run["technique"], run["scope"], run["requested"], raw_actual(run),
        metric(base, "accuracy", "map50_95", "map5095"), metric(final, "accuracy", "map50_95", "map5095"),
        metric(base, "loss"), metric(final, "loss"),
    )


def sort_preference(run: dict[str, Any]) -> tuple[int, str]:
    # Validated-flow artifacts win over older artifacts; run id provides a
    # deterministic reproducible tie break.
    return (1 if run["current_final"] else 0, run["run"])


def classify(all_runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    validated_models = {
        canonical_model(run["model"])
        for run in all_runs
        if canonical_model(run["model"]) is not None and run["current_final"]
    }
    for run in all_runs:
        model = canonical_model(run["model"])
        if model is None:
            excluded.append({**run, "model_family": NA, "exclusion": "model outside requested report"})
            continue
        run["model_family"] = model
        if not run["baseline_ok"]:
            excluded.append({**run, "exclusion": "INVALID_BASELINE"})
        elif not run["phase"].startswith("Final"):
            excluded.append({**run, "exclusion": "sensitivity/smoke; not final pruning"})
        elif model in validated_models and not run["current_final"]:
            excluded.append({**run, "exclusion": "historical run superseded by validated flow for this model"})
        elif run["status"] in {"FAILED", "INVALID_BASELINE"}:
            excluded.append({**run, "exclusion": "FAILED or invalid; excluded from ranking"})
        else:
            candidates.append(run)

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for run in candidates:
        grouped[run_signature(run)].append(run)
    eligible: list[dict[str, Any]] = []
    for same in grouped.values():
        same.sort(key=sort_preference, reverse=True)
        eligible.append(same[0])
        for duplicate in same[1:]:
            excluded.append({**duplicate, "exclusion": f"duplicate of {same[0]['run']}"})
    return sorted(eligible, key=lambda item: (item["model_family"], item["technique"], str(item["requested"]))), excluded


def requested_prunable(run: dict[str, Any]) -> Any:
    return run["requested"]


def actual_prunable(run: dict[str, Any]) -> Any:
    # Earlier artifacts did not record the denominator.  Preserve their number
    # as reported but do not mislabel it as whole-model sparsity.
    return run["actual_pct"]


def requested_percent(run: dict[str, Any]) -> float | None:
    text = str(run["requested"])
    if not text.endswith("%"):
        return None
    try:
        return float(text[:-1])
    except ValueError:
        return None


def pruning_evidence(run: dict[str, Any]) -> str:
    """Whether the artifact demonstrates material pruning, independently of quality."""
    actual, requested_pct = raw_actual(run), requested_percent(run)
    structural = num(run["structural"])
    if run["scope"] in {"filter", "channel", "depth"}:
        if (structural is not None and structural >= 1.0) or (actual is not None and requested_pct is not None and actual >= requested_pct * 0.85):
            return "Material structural/actual reduction"
        return "Target not evidenced by artifact"
    if actual is not None and requested_pct is not None and actual >= requested_pct * 0.85:
        return "Actual sparsity approximately reached"
    return "Target not evidenced by artifact"


def rankable(run: dict[str, Any]) -> bool:
    return pruning_evidence(run) != "Target not evidenced by artifact"


def loss_delta(run: dict[str, Any]) -> Any:
    baseline, final = metric(run["baseline"], "loss"), metric(run["final"], "loss")
    return value(final - baseline) if baseline is not None and final is not None else NA


def gflops(raw_flops: Any) -> Any:
    raw = num(raw_flops)
    return value(raw / 1_000_000_000, 9) if raw is not None else NA


def backend(run: dict[str, Any]) -> Any:
    # Device is recorded in the resolved config; framework/backend itself was
    # not consistently recorded, so never claim more than this evidence.
    config_path = Path(run["config"])
    try:
        import json
        device = json.loads(config_path.read_text(encoding="utf-8")).get("model", {}).get("device")
        return f"device={device} (config)" if device else NA
    except (OSError, ValueError):
        return NA


def runtime_value(run: dict[str, Any], field: str) -> float | None:
    """Detection evaluator timings take precedence over generic benchmark timing."""
    if is_detection(run):
        detected = {"mean": "inference_ms", "nms": "nms_ms", "total": "total_ms"}
        if field in detected:
            return metric(run["final"], detected[field])
        return None
    generic = {"mean": "inference_latency_ms", "p95": "latency_p95_ms", "nms": "nms_latency_ms", "total": "total_latency_ms", "fps": "fps"}
    return num(run["benchmark"].get(generic[field]))


def pick_best(runs: list[dict[str, Any]], predicate=lambda r: True) -> dict[str, Any] | None:
    candidates = [run for run in runs if predicate(run)]
    if not candidates:
        return None
    return min(candidates, key=lambda run: (run["drop"] if run["drop"] is not None else float("inf"), -(raw_actual(run) or -1)))


def label(run: dict[str, Any] | None) -> str:
    if not run:
        return NA
    return f"{run['technique']} @ {run['requested']} ({run['status']})"


def summary_rows(eligible: list[dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    structured = {"Filter Pruning", "Channel Pruning", "Taylor Pruning", "Layer / Depth Pruning", "Block Sparse", "Block Pruning"}
    for model in MODEL_ORDER:
        runs = [run for run in eligible if run["model_family"] == model and rankable(run)]
        usable = [run for run in runs if run["status"] in {"PASS", "GOOD"}]
        quality = pick_best(runs)
        best30 = pick_best(runs, lambda run: str(run["requested"]).startswith("30."))
        best50 = pick_best(runs, lambda run: str(run["requested"]).startswith("50."))
        highest = max(usable, key=lambda run: (raw_actual(run) or -1, -(run["drop"] or 0))) if usable else None
        structured_best = pick_best(runs, lambda run: run["technique"] in structured)
        timed = [run for run in runs if run["status"] in {"PASS", "GOOD", "DEGRADED"} and runtime_value(run, "mean") is not None]
        fastest = min(timed, key=lambda run: runtime_value(run, "mean") or float("inf")) if timed else None
        rows.append([
            model, label(quality), label(best30), label(best50),
            f"{highest['technique']} @ {highest['actual_pct']}%" if highest else NA,
            label(structured_best), label(fastest) if fastest else NA,
        ])
    return rows


def reduction_label(run: dict[str, Any] | None, field: str) -> str:
    if not run:
        return NA
    if field == "structural":
        reduction = num(run["structural"])
    else:
        before, after = num(run["before"].get("flops")), num(run["after"].get("flops"))
        reduction = (before - after) / before * 100 if before not in (None, 0) and after is not None else None
    return f"{label(run)} — {value(reduction)}%" if reduction is not None else NA


def ranking_rows(eligible: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    for model in MODEL_ORDER:
        runs = [run for run in eligible if run["model_family"] == model and rankable(run)]
        best_quality = pick_best(runs)
        structural = [run for run in runs if num(run["structural"]) is not None and num(run["structural"]) > 0]
        best_structural = max(structural, key=lambda run: num(run["structural"]) or -1) if structural else None
        flops = [run for run in runs if num(run["before"].get("flops")) not in (None, 0) and num(run["after"].get("flops")) is not None]
        best_flops = max(flops, key=lambda run: (num(run["before"].get("flops")) - num(run["after"].get("flops"))) / (num(run["before"].get("flops")) or 1)) if flops else None
        timed = [run for run in runs if run["status"] in {"PASS", "GOOD", "DEGRADED"} and runtime_value(run, "mean") is not None]
        fastest = min(timed, key=lambda run: runtime_value(run, "mean") or float("inf")) if timed else None
        rows.append([model, label(best_quality), reduction_label(best_structural, "structural"), reduction_label(best_flops, "flops"), label(fastest)])
    return rows


def support_matrix(eligible: list[dict[str, Any]], excluded: list[dict[str, Any]]) -> list[list[Any]]:
    techniques = ["Magnitude", "SNIP", "GraSP", "SynFlow", "LAMP", "L1 Regularization", "L0 Regularization", "Filter Pruning", "Channel Pruning", "Taylor Pruning", "Layer / Depth Pruning", "Block Pruning", "Block Sparse", "N:M Sparsity", "Attention Head Pruning"]
    rows = []
    for technique in techniques:
        row = [technique]
        for model in MODEL_ORDER:
            valid = [run for run in eligible if run["model_family"] == model and run["technique"] == technique]
            failed = [run for run in excluded if run.get("model_family") == model and run["technique"] == technique and run["status"] == "FAILED"]
            observed = [run for run in excluded if run.get("model_family") == model and run["technique"] == technique]
            if valid:
                priority = {"PASS": 0, "GOOD": 1, "DEGRADED": 2, "SEVERE": 3}
                row.append(min(valid, key=lambda run: (priority.get(run["status"], 99), -(raw_actual(run) or -1)))["status"])
            elif failed:
                row.append("FAILED")
            elif technique == "Attention Head Pruning" and model in {"LeNet5", "ResNet18", "YOLOv5s"}:
                row.append("NOT_APPLICABLE")
            elif observed:
                row.append("NOT_RUN")
            else:
                row.append("NOT_RUN")
        rows.append(row)
    return rows


def main() -> None:
    all_runs, discovery = collect()
    eligible, excluded = classify(all_runs)
    workbook = Workbook()
    workbook.active.title = "Read Me"
    write_sheet(workbook.active, "Consolidated Pruning Benchmark", ["Field", "Value"], [
        ["Data policy", "Completed artifact metrics only. No config-only run and no synthetic metric is included."],
        ["Quality inclusion", "Final pruning + valid baseline + status not FAILED; exact duplicates collapsed to one preferred artifact."],
        ["Excluded audit", "Sensitivity/smoke, INVALID_BASELINE, FAILED, and duplicates are retained in Excluded Audit."],
        ["Sparsity caveat", "Actual sparsity is the artifact-reported value. Its denominator is not recorded in old artifacts; Whole-model sparsity is N/A."],
        ["RT-DETR", "No completed RT-DETR artifact was discovered; report cells are NOT_RUN/N/A."],
        ["Eligible final runs", len(eligible)], ["Excluded completed runs", len(excluded)], ["Artifact directories discovered", len(discovery)],
    ])

    classification = [run for run in eligible if run["model_family"] in {"LeNet5", "ResNet18"} and not is_detection(run)]
    detection = [run for run in eligible if run["model_family"] in {"YOLOv5s", "RT-DETR"} and is_detection(run)]
    write_sheet(workbook.create_sheet("Classification Quality"), "1. Classification Quality Benchmark",
        ["Model", "Technique", "Scope", "Requested Prunable Sparsity", "Actual Sparsity (artifact-reported, %)", "Pruning Target Evidence", "Baseline Accuracy", "Pruned Accuracy", "Accuracy Drop", "Baseline Loss", "Pruned Loss", "Status", "Artifact"],
        ([r["model_family"], r["technique"], r["scope"], requested_prunable(r), actual_prunable(r), pruning_evidence(r), value(metric(r["baseline"], "accuracy")), value(metric(r["final"], "accuracy")), value(r["drop"]), value(metric(r["baseline"], "loss")), value(metric(r["final"], "loss")), r["status"], r["artifact"]] for r in classification))
    write_sheet(workbook.create_sheet("Detection Quality"), "2. Detection Quality Benchmark",
        ["Model", "Technique", "Scope", "Requested Prunable Sparsity", "Actual Sparsity (artifact-reported, %)", "Pruning Target Evidence", "Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95", "mAP Drop", "Status", "Artifact"],
        ([r["model_family"], r["technique"], r["scope"], requested_prunable(r), actual_prunable(r), pruning_evidence(r), value(metric(r["final"], "precision")), value(metric(r["final"], "recall")), value(metric(r["final"], "map50")), value(metric(r["final"], "map50_95", "map5095")), value(r["drop"]), r["status"], r["artifact"]] for r in detection))
    write_sheet(workbook.create_sheet("Complexity"), "3. Complexity Benchmark",
        ["Model", "Technique", "Requested Prunable Sparsity", "Actual Sparsity (artifact-reported, %)", "Whole-model Sparsity", "Pruning Target Evidence", "Parameters Before", "Parameters After", "Non-zero Params", "Structural Reduction (%)", "GFLOPs Before", "GFLOPs After (artifact-reported)", "Model Size After (MB)", "Artifact"],
        ([r["model_family"], r["technique"], requested_prunable(r), actual_prunable(r), NA, pruning_evidence(r), value(r["before"].get("params"), 0), value(r["after"].get("params"), 0), NA, value(r["structural"]), gflops(r["before"].get("flops")), gflops(r["after"].get("flops")), value(r["after"].get("model_size_mb")), r["artifact"]] for r in eligible))
    write_sheet(workbook.create_sheet("Runtime"), "4. Runtime Benchmark",
        ["Model", "Technique", "Requested Prunable Sparsity", "Mean Latency (ms)", "P95 (ms)", "NMS (ms)", "Total Latency (ms)", "FPS", "Backend / Device", "Artifact"],
        ([r["model_family"], r["technique"], requested_prunable(r), value(runtime_value(r, "mean")), value(runtime_value(r, "p95")), value(runtime_value(r, "nms")), value(runtime_value(r, "total")), value(runtime_value(r, "fps")), backend(r), r["artifact"]] for r in eligible))
    write_sheet(workbook.create_sheet("Summary Matrix"), "5. Summary Matrix", ["Model", "Best Quality Retention", "Best 30%", "Best 50%", "Highest Usable Sparsity (PASS/GOOD)", "Best Structured Method", "Best Runtime"], summary_rows(eligible))
    write_sheet(workbook.create_sheet("Ranking"), "Ranking From Eligible, Material-Pruning Runs",
        ["Model", "Best Quality Retention", "Best Structural Parameter Reduction", "Best Artifact-reported FLOP Reduction", "Best Runtime With Acceptable Quality"], ranking_rows(eligible))
    write_sheet(workbook.create_sheet("Technique Support"), "6. Technique Support Matrix", ["Technique", *MODEL_ORDER], support_matrix(eligible, excluded))
    write_sheet(workbook.create_sheet("Excluded Audit"), "Excluded / Failed / Duplicate / Non-final Completed Runs",
        ["Model", "Technique", "Phase", "Requested Sparsity", "Actual Sparsity (%)", "Baseline Score", "Pruned Score", "Drop", "Status", "Exclusion", "Artifact"],
        ([r.get("model_family", NA), r["technique"], r["phase"], r["requested"], r["actual_pct"], value(metric(r["baseline"], "accuracy", "map50_95", "map5095")), value(metric(r["final"], "accuracy", "map50_95", "map5095")), value(r["drop"]), r["status"], r["exclusion"], r["artifact"]] for r in excluded))
    write_sheet(workbook.create_sheet("Discovery"), "Artifact Discovery", ["Run ID", "Completion", "Config", "Model", "Phase"], discovery)
    try:
        workbook.save(OUT)
        destination = OUT
    except PermissionError:
        destination = OUT.with_name(f"{OUT.stem}_updated{OUT.suffix}")
        workbook.save(destination)
    print(f"Exported {len(eligible)} eligible and {len(excluded)} excluded completed runs to {destination}")


if __name__ == "__main__":
    main()
