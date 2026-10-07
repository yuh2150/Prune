"""Compare pretrained GraSP and its guarded hybrid on the same LeNet split.

Run from the repository root: python -m tools.benchmark_grasp_guarded
Every run uses a fresh checkpoint, fixed training calibration and no recovery.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from prune_framework.core.config import FrameworkConfig
from prune_framework.pipelines.unified import UnifiedPruningPipeline


def main():
    output = Path("artifacts/grasp_guarded_comparison_20261006")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for method, ratios in (
        ("grasp", (.10, .30)),
        ("grasp_magnitude_guarded", (.01, .05, .10, .30)),
        ("magnitude", (.10, .30)),
    ):
        for ratio in ratios:
            cfg = FrameworkConfig.from_yaml("configs/lenet5_custom_47labels_grasp.yaml")
            cfg.pruning.criterion = method
            cfg.pruning.amount = ratio
            cfg.experiment.name = f"lenet5_guarded_comparison_{method}_{int(ratio * 100):02d}"
            cfg.output_path = str(output / (cfg.experiment.name + ".pt"))
            if method == "magnitude":
                cfg.pruning.calibration_callback = None
            result = UnifiedPruningPipeline(cfg).run()
            saved = json.loads(Path(result.artifacts["result"]).read_text())
            plan = saved["pruning"]["pruning_plan"]
            counts = {g["primary"]["name"]: len(g["indices"]) for g in plan["groups"]}
            sizes = {"conv1.conv": 150, "conv2.conv": 2400, "conv3.conv": 48000, "fc1": 10080}
            assert set(counts) <= set(sizes), "Unexpected or protected target pruned"
            assert sum(counts.values()) == round(sum(sizes.values()) * ratio)
            record = {
                "criterion": method, "ratio": ratio,
                "baseline": saved["baseline_metrics"], "final": saved["final_metrics"],
                "selected_elements": sum(counts.values()),
                "total_prunable_elements": plan["metadata"]["total_elements"],
                "whole_model_sparsity_pct": saved["complexity_after"]["sparsity_pct"],
                "layers": {name: {"numel": size, "pruned": counts.get(name, 0),
                                  "fraction": counts.get(name, 0) / size}
                           for name, size in sizes.items()},
                "magnitude_guard": plan["metadata"].get("magnitude_guard"),
                "result_path": result.artifacts["result"],
            }
            records.append(record)
            (output / "comparison.json").write_text(json.dumps(records, indent=2) + "\n")
            print(json.dumps(record), flush=True)
    (output / "protocol.json").write_text(json.dumps({
        "checkpoint_sha256": hashlib.sha256(Path(cfg.model.weights).read_bytes()).hexdigest(),
        "config": "configs/lenet5_custom_47labels_grasp.yaml",
        "seed": 42, "validation_samples": 200, "calibration_batches": 4,
        "calibration_batch_size": 64, "microbatch_size": 20, "temperature": 200,
        "candidate_multiplier": 1.5, "fine_tuning": False,
    }, indent=2) + "\n")
    lines = [
        "# GraSP magnitude guard — LeNet5 pretrained, 06/10/2026", "",
        "Cùng checkpoint ONNX, split seed 42, validation 200 ảnh. Calibration dùng "
        "4 batches train × 64, temperature 200, microbatch 20. Không fine-tune. "
        "Guard lấy 1.5× ngân sách weights nhỏ nhất theo magnitude; GraSP có dấu, "
        "chuẩn hóa mean_abs từng lớp, quyết định phần bỏ trong nhóm này.", "",
        "| Criterion | Prune eligible | Baseline accuracy | Post accuracy | Loss | Conv1 | Conv2 | Conv3 | fc1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        fractions = " | ".join(f'{t["fraction"]:.2%}' for t in r["layers"].values())
        lines.append(f'| {r["criterion"]} | {r["ratio"]:.0%} | {r["baseline"]["accuracy"]:.2%} '
                     f'| {r["final"]["accuracy"]:.2%} | {r["final"]["loss"]:.6f} | {fractions} |')
    lines += ["", "Mỗi tỷ lệ lớp dùng numel của chính lớp làm mẫu số. Global eligible "
              "dùng 60,630 weights; classifier fc2 và bias được bảo vệ. Tensor dimensions "
              "giữ nguyên; không suy ra speedup hoặc giảm dense parameters.", "",
              "Đây là hybrid cho checkpoint pretrained, không phải GraSP gốc. Khi nhóm "
              "ứng viên phủ hết weights (ratio ≥ 2/3), guard không còn giới hạn magnitude. "
              "Kết quả một seed và validation nhỏ chưa chứng minh khả năng tổng quát; "
              "không khẳng định tốt hơn magnitude.", "", "## Artifacts", ""]
    lines += [f'- {r["criterion"]} {r["ratio"]:.0%}: `{r["result_path"]}`' for r in records]
    (output / "comparison.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
