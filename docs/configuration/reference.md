# Configuration reference

`FrameworkConfig` accepts the original keys (`pruner`, `granularity`, `amount`) and the research schema (`method`, `structure`, `target_ratio`). Do not mix an alias with its original name in the same section.

```yaml
model:
  name: yolov5
  weights: weights/yolov5s.pt
  device: cuda

pruning:
  method: structured       # alias: pruner
  criterion: l1
  structure: channel       # alias: granularity
  target_ratio: 0.30       # alias: amount
  global: true             # stored as global_pruning
  min_channels: 8

sensitivity:
  enabled: true
  rates: [0.1, 0.2, 0.3]
  selector: sensitivity
  max_allowed_relative_drop: 0.05

evaluation:
  enabled: true
  callback: experiments.yolov5:evaluate
  metric: map

recovery:
  enabled: true
  epochs: 10
  callback: experiments.yolov5:recover

benchmark:
  latency: true
  flops: true
  params: true
  warmup: 20
  runs: 100

export:
  onnx: true
  output_path: artifacts/model.onnx

experiment:
  name: yolov5_l1_30
  output_dir: artifacts
  seed: 42
  deterministic: true
```

`evaluation.callback`, `recovery.callback`, and `pruning.calibration_callback` use `package.module:function` notation. The evaluation callback returns a metric float, a metric dictionary, or `EvaluationResult`. A recovery callback receives the pruned model and returns either that model or `None` after in-place training. Taylor requires a calibration callback that performs a representative backward pass before the pruning stage.

The supported model names are the registered entries shown by `python main.py --list-models`; currently these are `yolov5`, a compatibility alias `yolov7`, and `rtdetr`. ResNet is not registered.

The original `analysis.sensitivity` key remains accepted and maps to `sensitivity.enabled`. Its `analysis.layer_selection` flag is retained for old configuration compatibility; the unified pipeline runs its configured selector whenever sensitivity is enabled.
