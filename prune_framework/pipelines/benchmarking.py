import torch
from prune_framework.core.config import FrameworkConfig
from prune_framework.modules.model.loader import ModelLoader
from prune_framework.core.registry import PluginRegistry
from prune_framework.modules.evaluation.benchmark import LatencyBenchmark


def run_benchmarking_pipeline(config: FrameworkConfig, post_process_fn=None):
    device = torch.device(config.model.device if torch.cuda.is_available() else "cpu")
    model, _ = ModelLoader.load(config.model.name, config.model.weights, device)

    adapter_cls = PluginRegistry.get_model_adapter(config.model.name)
    adapter = adapter_cls(model)
    dummy_input = adapter.get_dummy_input(device)

    benchmarker = LatencyBenchmark(warmup_runs=config.benchmark.warmup, test_runs=config.benchmark.runs)
    result = benchmarker.benchmark(model, dummy_input, post_process_fn=post_process_fn)

    print("\n==================================================")
    print(f"Decoupled Latency Benchmark Report ({config.model.name})")
    print(f" - Inference Latency: {result.inference_latency_ms:.3f} ms")
    print(f" - Post-Process (NMS):{result.nms_latency_ms:.3f} ms")
    print(f" - Total Latency:     {result.total_latency_ms:.3f} ms")
    print(f" - Throughput (FPS):  {result.fps:.2f}")
    print("==================================================\n")

    return result
