import torch
from prune_framework.core.config import FrameworkConfig
from prune_framework.modules.model.loader import ModelLoader
from prune_framework.core.registry import PluginRegistry
from prune_framework.modules.evaluation.benchmark import LatencyBenchmark, attach_quality_metrics
from prune_framework.pipelines.unified import UnifiedPruningPipeline


def run_benchmarking_pipeline(config: FrameworkConfig, post_process_fn=None, evaluator=None):
    evaluation = UnifiedPruningPipeline(config, evaluator=evaluator) if config.evaluation.enabled or evaluator is not None else None
    if evaluation is not None:
        evaluation._require_evaluator("Benchmark quality evaluation")
    device = torch.device(config.model.device if torch.cuda.is_available() else "cpu")
    model, _ = ModelLoader.load(
        config.model.name,
        config.model.weights,
        device,
        **({"num_classes": config.model.num_classes} if config.model.num_classes is not None else {}),
    )

    adapter_cls = PluginRegistry.get_model_adapter(config.model.name)
    adapter = adapter_cls(model)
    dummy_input = adapter.get_dummy_input(device)

    benchmarker = LatencyBenchmark(warmup_runs=config.benchmark.warmup, test_runs=config.benchmark.runs)
    result = benchmarker.benchmark(model, dummy_input, post_process_fn=post_process_fn)
    if evaluation is not None:
        attach_quality_metrics(result, evaluation._evaluate(model, "benchmark"))

    print("\n==================================================")
    print(f"Decoupled Latency Benchmark Report ({config.model.name})")
    print(f" - Inference Latency: {result.inference_latency_ms:.3f} ms")
    print(f" - Post-Process (NMS):{result.nms_latency_ms:.3f} ms")
    print(f" - Total Latency:     {result.total_latency_ms:.3f} ms")
    print(f" - Throughput (FPS):  {result.fps:.2f}")
    for label, name in (("Precision", "precision"), ("Recall", "recall"), ("F1", "f1"),
                        ("mAP@0.5", "map50"), ("mAP@0.5:0.95", "map50_95")):
        value = getattr(result, name)
        print(f" - {label}: {value:.4f}" if value is not None else f" - {label}: N/A")
    print("==================================================\n")

    return result
