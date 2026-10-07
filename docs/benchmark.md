# Benchmark quality metrics

Benchmark reports include `precision`, `recall`, `f1`, `map50` (mAP@0.5) and
`map50_95` (mAP@0.5:0.95), alongside latency, FPS and complexity. Values
are fractions in JSON/CSV and percentages in Excel. Missing measurements are
`null` in JSON and `NR` in quality reports. `N/A` means not applicable.

- Classification (LeNet5 / ResNet18): accuracy, cross-entropy loss, macro
  Precision, Recall and F1. Average over classes appearing in targets or
  predictions across the entire validation loader; undefined per-class ratios
  contribute zero. Macro F1 averages per-class F1; it is not the harmonic
  mean of macro Precision and Recall. IoU-based detection mAP is not applicable.
- YOLO: use the existing evaluator's P/R and AP protocol unchanged. Macro F1
  averages the evaluator's per-class F1 at its maximum macro-F1
  operating point, IoU 0.5, over classes with validation targets.
- COCO / RT-DETR: mAP comes from `pycocotools.COCOeval`. P/R uses its
  non-ignored matches at IoU 0.5, all object areas, up to 100 detections,
  confidence >= 0.25. Average over classes with non-ignored ground truth.
  The result metadata records this protocol. The RT-DETR CLI accepts
  `--precision-recall-conf-thres`; keep `--conf-thres` low when measuring AP.
  AP50 is not Precision, and COCO average recall is not this Recall.

COCO matching reference: [official COCOeval implementation](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py).
Do not compare P/R across evaluators without aligning their protocols.

Enable the configured dataset evaluator to measure quality:

```yaml
evaluation:
  enabled: true
  callback: experiments.classification:evaluate
  metric: accuracy
```

The unified pipeline saves baseline/final metrics, final quality values and
before/after deltas to `result.json`. When latency benchmarking runs, its
`pruning.benchmark` also contains final quality metrics. The standalone
`run_benchmarking_pipeline` evaluates the loaded model when evaluation is
enabled or an evaluator is passed, and prints all five quality labels.
Quality metrics require labeled data; timing on a dummy input cannot supply them.

To measure missing classification P/R/F1 in existing artifacts:

```bash
python -m tools.reevaluate_artifact_quality --write
# If the original ImageNet subset was relocated:
python -m tools.reevaluate_artifact_quality --write --imagenet-root /path/to/subset
```

The tool restores original weights and replays each saved pruning plan, then
verifies baseline/final accuracy and loss against the historical result before
adding measured quality values. It preserves old scores and timings and records
the dataset, sample count, seed, plan hash and verification in each updated run.
Runs with unavailable data or non-reproducing scores remain unchanged, with a
reason in `outputs/benchmark-quality-20261007/quality_reevaluation.json`.

For retained YOLO checkpoints and the original local COCO-500 validation list:

```bash
python -m tools.reevaluate_yolo_artifact_quality --write
```

This updates detection P/R/F1/AP from CPU inference. Historical detector
quality scores and the new measurements are retained in the reevaluation
metadata; original latency measurements remain unchanged. The legacy YOLO
`(results, maps, times)` return contract is preserved.

To regenerate the Excel workbook and quality CSV from saved artifacts:

```bash
node tools/build_benchmark_workbook.mjs outputs/benchmark-quality-20261007
```

The builder uses the workspace runtime's `@oai/artifact-tool` dependency
through `/tmp/prune-benchmark-workbook/node_modules`. It writes
`pruning_benchmark.xlsx` and `quality_benchmark.csv`. Saved results remain
unchanged. Historical runs lacking P/R need evaluation again to populate those
columns; accuracy alone cannot recover them. Raw runs retain every saved result,
including diagnostic/smoke runs, and headline tables consolidate identical rows.
