"""Build a traceable Excel report from pruning artifacts.

The exporter never invents a measurement. A missing measurement is rendered as
``N/A`` and each row retains its artifact path for auditability. Large N:M
result files are read only through their top-level summary fields.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


ROOT = Path("artifacts")
OUT = Path("benchmarks/pruning_benchmark_report.xlsx")
NA = "N/A"


def read_json(path: Path | None) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path else {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def object_at_start(text: str, key: str) -> dict[str, Any]:
    """Extract a top-level object without parsing a multi-hundred MB plan."""
    marker = re.search(rf'"{re.escape(key)}"\s*:\s*\{{', text)
    if not marker:
        return {}
    start = marker.end() - 1
    depth, quoted, escaped = 0, False, False
    for index in range(start, len(text)):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return {}
    return {}


def scalar_at_start(text: str, key: str) -> Any:
    pattern = rf'"{re.escape(key)}"\s*:\s*(null|true|false|-?[0-9.eE+]+|"(?:[^"\\]|\\.)*")'
    match = re.search(pattern, text)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def result_summary(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            text = handle.read(512 * 1024)
    except OSError:
        return {}
    return {
        "baseline_metrics": object_at_start(text, "baseline_metrics"),
        "final_metrics": object_at_start(text, "final_metrics"),
        "complexity_before": object_at_start(text, "complexity_before"),
        "complexity_after": object_at_start(text, "complexity_after"),
        "benchmark": object_at_start(text, "benchmark"),
        "pruning": object_at_start(text, "pruning"),
        "baseline_valid": scalar_at_start(text, "baseline_valid"),
        "actual_sparsity": scalar_at_start(text, "actual_sparsity"),
        "status": scalar_at_start(text, "status"),
    }


def find_config(run_dir: Path) -> Path | None:
    for filename in ("config.resolved.json", "config.json"):
        path = run_dir / filename
        if path.is_file():
            return path
    return None


def num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def value(value_: Any, digits: int = 6) -> Any:
    number = num(value_)
    return round(number, digits) if number is not None else NA


def metric(metrics: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        measured = num(metrics.get(key))
        if measured is not None:
            return measured
    return None


def technique(pruning: dict[str, Any], regularization: dict[str, Any]) -> str:
    if regularization.get("enabled"):
        return "L0 Regularization" if "l0" in str(regularization.get("term", "")).lower() else "L1 Regularization"
    pruner = str(pruning.get("pruner", "")).lower()
    criterion = str(pruning.get("criterion", "")).lower()
    granularity = str(pruning.get("granularity", "")).lower()
    if pruner == "nm":
        return "N:M Sparsity"
    if "block_sparse" in pruner or "block_sparse" in criterion:
        return "Block Sparse"
    if pruner in {"depth", "layer"} or granularity == "layer":
        return "Layer / Depth Pruning"
    if criterion == "taylor":
        return "Taylor Pruning"
    if granularity == "channel":
        return "Channel Pruning"
    if granularity == "filter":
        return "Filter Pruning"
    return {"magnitude": "Magnitude", "l1": "Magnitude", "snip": "SNIP", "grasp": "GraSP", "synflow": "SynFlow", "lamp": "LAMP"}.get(criterion, f"{pruner or 'Unknown'} / {criterion or 'Unknown'}")


def pruning_scope(pruning: dict[str, Any]) -> str:
    pruner = str(pruning.get("pruner", "")).lower()
    granularity = str(pruning.get("granularity", "")).lower()
    if pruner == "nm":
        return "N:M"
    if "block_sparse" in pruner:
        return "block"
    if pruner in {"depth", "layer"} or granularity == "layer":
        return "depth"
    if granularity in {"channel", "filter", "block"}:
        return granularity
    return "global" if pruning.get("global_pruning") else "layerwise"


def run_phase(run_dir: Path, config: dict[str, Any]) -> str:
    name = run_dir.name.lower()
    if "smoke" in name:
        return "Calibration / smoke test"
    if "sensitivity" in name or (run_dir / "sensitivity.json").is_file() or config.get("analysis", {}).get("sensitivity"):
        return "Sensitivity experiment"
    return "Pruned after fine-tuning" if config.get("recovery", {}).get("enabled") else "Final pruning (no fine-tuning)"


def valid_baseline(model: str, baseline: dict[str, Any], marker: Any) -> bool:
    if marker is False:
        return False
    accuracy = metric(baseline, "accuracy")
    # This explicitly catches the observed random native-LeNet baseline; it is
    # not applied to unrelated models/datasets.
    if "lenet" in model.lower() and accuracy is not None and accuracy <= 0.05:
        return False
    return bool(baseline)


def status(base: float | None, final: float | None, is_detection: bool) -> tuple[str, float | None]:
    if base is None or final is None:
        return "N/A", None
    drop = base - final
    limits = ((0.01, "PASS"), (0.03, "GOOD"), (0.10, "DEGRADED"), (0.20 if is_detection else 0.30, "SEVERE"))
    for limit, label in limits:
        if drop <= limit:
            return label, drop
    return "FAILED", drop


def requested(pruning: dict[str, Any]) -> str:
    if str(pruning.get("pruner", "")).lower() == "nm":
        return f"{pruning.get('n')}:{pruning.get('m')}" if pruning.get("n") is not None and pruning.get("m") is not None else NA
    amount = num(pruning.get("amount"))
    return f"{amount * 100:.3f}%" if amount is not None else NA


def collect() -> tuple[list[dict[str, Any]], list[list[Any]]]:
    completed: list[dict[str, Any]] = []
    discovery: list[list[Any]] = []
    for directory in sorted(path for path in ROOT.iterdir() if path.is_dir()):
        config_path = find_config(directory)
        result_path = directory / "result.json"
        config = read_json(config_path)
        if not result_path.is_file():
            discovery.append([directory.name, "no result.json", config_path.as_posix() if config_path else NA, NA, NA])
            continue
        result = result_summary(result_path)
        baseline, final = result["baseline_metrics"], result["final_metrics"]
        model = str(config.get("model", {}).get("name") or result["pruning"].get("model_name") or "Unknown")
        phase = run_phase(directory, config)
        is_complete = bool(baseline and final)
        discovery.append([directory.name, "completed" if is_complete else "result lacks baseline/final metrics", config_path.as_posix() if config_path else NA, model, phase])
        if not is_complete:
            continue
        prune = config.get("pruning", {})
        baseline_ok = valid_baseline(model, baseline, result["baseline_valid"])
        is_detection = any(metric(final, key) is not None for key in ("map50", "map50_95", "map5095"))
        base_score = metric(baseline, "map50_95", "map5095") if is_detection else metric(baseline, "accuracy")
        final_score = metric(final, "map50_95", "map5095") if is_detection else metric(final, "accuracy")
        run_status, drop = status(base_score, final_score, is_detection)
        if not baseline_ok or result["status"] == "INVALID_BASELINE":
            run_status = "INVALID_BASELINE"
        before, after = result["complexity_before"], result["complexity_after"]
        actual = num(result["actual_sparsity"])
        # Keep sub-percent structured measurements visible (for example a
        # requested 10% run which only achieved 0.000318%).
        actual_pct = value(actual * 100 if actual is not None else after.get("sparsity_pct"), 6)
        params_before, params_after = num(before.get("params")), num(after.get("params"))
        structural = (params_before - params_after) / params_before * 100 if params_before not in (None, 0) and params_after is not None else None
        current_final = baseline_ok and phase.startswith("Final") and result["baseline_valid"] is True
        reason = "Included" if current_final else ("invalid baseline" if not baseline_ok else "not a final-pruning run" if not phase.startswith("Final") else "historical run: no validated-flow baseline flag")
        completed.append({
            "run": directory.name, "artifact": directory.as_posix(), "config": config_path.as_posix() if config_path else NA,
            "model": model, "dataset": config.get("dataset", {}).get("name", NA), "technique": technique(prune, config.get("regularization", {})),
            "scope": pruning_scope(prune), "phase": phase, "requested": requested(prune), "actual_pct": actual_pct,
            "baseline": baseline, "final": final, "before": before, "after": after,
            "benchmark": result["benchmark"] or result["pruning"].get("benchmark", {}), "prune": prune,
            "status": run_status, "drop": drop, "baseline_ok": baseline_ok, "current_final": current_final,
            "reason": reason, "structural": structural,
        })
    return completed, discovery


def sensitivity_rows() -> list[list[Any]]:
    """Return every recorded layer/rate point; these are never final quality rows."""
    rows: list[list[Any]] = []
    for directory in sorted(path for path in ROOT.iterdir() if path.is_dir()):
        path = directory / "sensitivity.json"
        data = read_json(path) if path.is_file() else {}
        if not data:
            continue
        config = read_json(find_config(directory))
        model = config.get("model", {}).get("name", NA)
        baseline = num(data.get("baseline_score"))
        profiles = data.get("profiles", {})
        if not isinstance(profiles, dict):
            continue
        for profile in profiles.values():
            layer = profile.get("layer_name", NA)
            points = profile.get("points", {})
            if not isinstance(points, dict):
                continue
            for point in points.values():
                rows.append([
                    model, directory.name, layer, value(baseline), value(point.get("rate")),
                    value(point.get("score")), value(point.get("metric_drop")),
                    value(point.get("relative_drop")), point.get("is_valid", NA),
                    point.get("error_message") or NA, directory.as_posix(),
                ])
    return rows


def write_sheet(ws, title: str, headers: list[str], rows: Iterable[Iterable[Any]]) -> None:
    ws.append([title])
    ws["A1"].font = Font(bold=True, color="FFFFFF", size=14)
    ws["A1"].fill = PatternFill("solid", fgColor="1F4E78")
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(headers))
    ws.append(headers)
    for cell in ws[2]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="5B9BD5")
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    for row in rows:
        ws.append(list(row))
    ws.freeze_panes = "A3"
    ws.auto_filter.ref = f"A2:{get_column_letter(len(headers))}{max(ws.max_row, 2)}"
    ws.row_dimensions[1].height, ws.row_dimensions[2].height = 24, 32
    for column in range(1, len(headers) + 1):
        longest = max(len(str(ws.cell(row, column).value or "")) for row in range(1, ws.max_row + 1))
        ws.column_dimensions[get_column_letter(column)].width = min(48, max(12, longest + 2))


def main() -> None:
    all_runs, discovery = collect()
    finals = [run for run in all_runs if run["current_final"]]
    historical = [run for run in all_runs if not run["current_final"]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    write_sheet(workbook.active, "Pruning Benchmark Report", ["Field", "Value"], [
        ["Generated (UTC)", datetime.now(timezone.utc).isoformat(timespec="seconds")],
        ["Scope", "Artifact-derived results only; this exporter does not rerun experiments."],
        ["Final inclusion", "Completed final pruning + valid baseline + baseline_valid=true from validated flow."],
        ["Excluded from final", "Sensitivity, smoke/calibration, invalid-baseline, and older runs without the validated-flow flag."],
        ["Missing metric", "N/A; no values are inferred from a requested ratio or documentation."],
        ["Actual sparsity", "Artifact actual_sparsity, else artifact complexity_after.sparsity_pct. Structural reduction is separate."],
        ["Traceability", "Every result row includes the artifact directory."],
        ["Final runs", len(finals)], ["Historical/debug completed runs", len(historical)], ["Artifact directories discovered", len(discovery)],
    ])
    workbook.active.title = "Read Me"
    classification = [run for run in finals if metric(run["final"], "accuracy") is not None]
    detection = [run for run in finals if run not in classification]
    write_sheet(workbook.create_sheet("Quality Classification"), "1. Quality Benchmark — Classification",
        ["Model", "Technique", "Scope", "Requested Sparsity", "Actual Sparsity (%)", "Phase", "Baseline Accuracy", "Pruned Accuracy", "Accuracy Drop", "Baseline Loss", "Pruned Loss", "Loss Delta", "Precision (macro)", "Recall (macro)", "F1 (macro)", "mAP@0.5", "mAP@0.5:0.95", "Status", "Artifact"],
        ([r["model"], r["technique"], r["scope"], r["requested"], r["actual_pct"], r["phase"], value(metric(r["baseline"], "accuracy")), value(metric(r["final"], "accuracy")), value(r["drop"]), value(metric(r["baseline"], "loss")), value(metric(r["final"], "loss")), value(metric(r["final"], "loss") - metric(r["baseline"], "loss")) if metric(r["final"], "loss") is not None and metric(r["baseline"], "loss") is not None else NA, value(metric(r["final"], "precision")) if metric(r["final"], "precision") is not None else "NR", value(metric(r["final"], "recall")) if metric(r["final"], "recall") is not None else "NR", value(metric(r["final"], "f1")) if metric(r["final"], "f1") is not None else "NR", NA, NA, r["status"], r["artifact"]] for r in classification))
    write_sheet(workbook.create_sheet("Quality Detection"), "2. Quality Benchmark — Detection",
        ["Model", "Technique", "Scope", "Requested Sparsity", "Actual Sparsity (%)", "Precision", "Recall", "F1 (macro)", "mAP@0.5", "mAP@0.5:0.95", "mAP Drop", "Status", "Artifact"],
        ([r["model"], r["technique"], r["scope"], r["requested"], r["actual_pct"], value(metric(r["final"], "precision")), value(metric(r["final"], "recall")), value(metric(r["final"], "f1")) if metric(r["final"], "f1") is not None else "NR", value(metric(r["final"], "map50")), value(metric(r["final"], "map50_95", "map5095")), value(r["drop"]), r["status"], r["artifact"]] for r in detection))
    write_sheet(workbook.create_sheet("Sparsity"), "3. Sparsity Benchmark",
        ["Model", "Technique", "Requested Sparsity", "Actual Sparsity (%)", "Prunable Parameters", "Zero Prunable Weights", "Non-zero Prunable Weights", "N", "M", "Group Count", "Invalid Group Count", "Block Shape", "Zero Blocks", "Total Blocks", "Block Sparsity", "Constraint Valid", "Status", "Artifact"],
        ([r["model"], r["technique"], r["requested"], r["actual_pct"], NA, NA, NA, r["prune"].get("n", NA), r["prune"].get("m", NA), NA, NA, str(r["prune"].get("block_shape")) if r["prune"].get("block_shape") else NA, NA, NA, NA, NA, r["status"], r["artifact"]] for r in finals))
    write_sheet(workbook.create_sheet("Complexity Runtime"), "4. Complexity & Runtime Benchmark",
        ["Model", "Technique", "Actual Sparsity (%)", "Dense Parameters", "Non-zero Parameters", "Structural Reduction (%)", "Dense FLOPs", "Artifact-reported After FLOPs", "Dense Model Size (MB)", "After Model Size (MB)", "Latency (ms)", "P50 (ms)", "P95 (ms)", "FPS", "Artifact"],
        ([r["model"], r["technique"], r["actual_pct"], value(r["before"].get("params"), 0), NA, value(r["structural"]), value(r["before"].get("flops"), 0), value(r["after"].get("flops"), 0), value(r["before"].get("model_size_mb")), value(r["after"].get("model_size_mb")), value(r["benchmark"].get("inference_latency_ms")), value(r["benchmark"].get("latency_p50_ms")), value(r["benchmark"].get("latency_p95_ms")), value(r["benchmark"].get("fps")), r["artifact"]] for r in finals))
    summary_rows = []
    for key in sorted({(r["model"], r["technique"]) for r in classification}):
        choices = [r for r in classification if (r["model"], r["technique"]) == key and r["status"] in {"PASS", "GOOD"}]
        # Select the highest observed sparsity preserving PASS/GOOD; then use
        # the smallest observed drop as a deterministic tie-breaker.
        choices.sort(key=lambda r: (num(r["actual_pct"]) or -1, -(r["drop"] or 0)), reverse=True)
        selected = choices[0] if choices else None
        summary_rows.append([
            key[0], key[1], len([r for r in classification if (r["model"], r["technique"]) == key]),
            selected["actual_pct"] if selected else NA,
            value(metric(selected["final"], "accuracy")) if selected else NA,
            value(selected["drop"]) if selected else NA,
            selected["status"] if selected else "No PASS/GOOD run",
            selected["artifact"] if selected else NA,
        ])
    write_sheet(workbook.create_sheet("Technique Summary"), "5. Best Observed Quality-Preserving Result",
        ["Model", "Technique", "Completed Final Runs", "Highest PASS/GOOD Actual Sparsity (%)", "Observed Pruned Accuracy", "Observed Accuracy Drop", "Status", "Artifact"], summary_rows)
    coverage = [
        ("Unstructured", "Magnitude Pruning", "Magnitude"),
        ("Unstructured", "SNIP", "SNIP"),
        ("Unstructured", "GraSP", "GraSP"),
        ("Unstructured", "SynFlow", "SynFlow"),
        ("Unstructured", "LAMP", "LAMP"),
        ("Structured", "L1 Regularization", "L1 Regularization"),
        ("Structured", "L0 Regularization", "L0 Regularization"),
        ("Structured", "Channel Pruning", "Channel Pruning"),
        ("Structured", "Filter Pruning", "Filter Pruning"),
        ("Structured", "Layer Pruning", "Layer / Depth Pruning"),
        ("Structured", "Block Pruning", "Block Pruning"),
        ("Structured", "Taylor-based Gradient Pruning", "Taylor Pruning"),
        ("Structured", "Attention Head Pruning", "Attention Head Pruning"),
        ("Semi-Structured", "N:M Sparsity", "N:M Sparsity"),
        ("Semi-Structured", "Block Sparse", "Block Sparse"),
    ]
    coverage_rows = []
    for family, label, normalized in coverage:
        observed_final = [r for r in finals if r["technique"] == normalized]
        observed_historical = [r for r in historical if r["technique"] == normalized]
        if observed_final:
            availability = "Final validated run available"
        elif observed_historical:
            availability = "Historical run only — excluded from final"
        else:
            availability = "No completed artifact"
        paths = observed_final or observed_historical
        coverage_rows.append([family, label, availability, len(observed_final), len(observed_historical), paths[0]["artifact"] if paths else NA])
    write_sheet(workbook.create_sheet("Technique Coverage"), "Technique Coverage — Requested 14 Techniques",
        ["Family", "Requested Technique", "Availability", "Validated Final Runs", "Historical Completed Runs", "Example Artifact"], coverage_rows)
    write_sheet(workbook.create_sheet("Sensitivity"), "Sensitivity Results — Separate From Final Quality Benchmark",
        ["Model", "Run ID", "Layer", "Baseline Score", "Pruning Rate", "Score", "Metric Drop", "Relative Drop", "Valid", "Error", "Artifact"], sensitivity_rows())
    write_sheet(workbook.create_sheet("Historical Debug"), "Historical / Debug / Excluded Completed Runs",
        ["Model", "Technique", "Phase", "Baseline Valid", "Requested Sparsity", "Actual Sparsity (%)", "Baseline Accuracy/mAP", "Pruned Accuracy/mAP", "Drop", "Status", "Exclusion Reason", "Artifact"],
        ([r["model"], r["technique"], r["phase"], r["baseline_ok"], r["requested"], r["actual_pct"], value(metric(r["baseline"], "accuracy", "map50_95", "map5095")), value(metric(r["final"], "accuracy", "map50_95", "map5095")), value(r["drop"]), r["status"], r["reason"], r["artifact"]] for r in historical))
    write_sheet(workbook.create_sheet("Artifact Discovery"), "Artifact Discovery", ["Run ID", "Completion", "Config", "Model", "Phase"], discovery)
    destination = OUT
    try:
        workbook.save(destination)
    except PermissionError:
        destination = OUT.with_name(f"{OUT.stem}_updated{OUT.suffix}")
        workbook.save(destination)
        print(f"Note: {OUT} is open/locked; wrote the updated workbook to {destination}.")
    print(f"Exported {len(finals)} final and {len(historical)} historical completed runs to {destination}")


if __name__ == "__main__":
    main()
