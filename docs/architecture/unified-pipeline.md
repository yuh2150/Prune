# Unified pruning experiment pipeline

`UnifiedPruningPipeline` is the single research workflow. It keeps the core independent of YOLO, RT-DETR, COCO, a trainer, and a loss implementation; an adapter describes the model while user callbacks provide dataset-specific evaluation, Taylor calibration, and recovery.

```text
Load model → dependency-graph preflight → baseline evaluation
→ optional sensitivity sweep → layer/group selection → pruning plan
→ physical pruning → architecture validation → optional recovery
→ final evaluation → latency/FLOPs/parameters → export → artifacts
```

The canonical schema is demonstrated by [research_unified.yaml](../../configs/research_unified.yaml). Replace its `experiments.yolov5:*` callback paths with callbacks for the dataset and trainer in use. `method`, `structure`, `target_ratio`, and `global` are aliases accepted by `FrameworkConfig`; the older `pruner`, `granularity`, and `amount` keys continue to work.

An evaluation callback is a Python function such as `experiments.yolov5:evaluate`. It receives named arguments it declares from `model`, `config`, `device`, `stage`, and `evaluation.kwargs`, and returns a float, a metrics dictionary, or `EvaluationResult`. A recovery callback receives the same context and may update the model in place or return a replacement model. Taylor requires `pruning.calibration_callback`; that function must run a representative loss backward pass so weights have gradients.

Each run writes a resolved config, sensitivity profile, selection, and final result into a timestamped directory under `experiment.output_dir`. The pipeline seeds Python, NumPy, and PyTorch before loading the model. `global: true` selects channels from one cross-layer importance ranking while retaining `min_channels` in every eligible layer.
