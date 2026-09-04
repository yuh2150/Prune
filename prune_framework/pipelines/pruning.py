import torch
from prune_framework.core.config import FrameworkConfig
from prune_framework.core.engine import PruningEngine
from prune_framework.modules.model.loader import ModelLoader
from prune_framework.modules.export.exporter import ModelExporter
from prune_framework.modules.evaluation.benchmark import LatencyBenchmark


def run_pruning_pipeline(config: FrameworkConfig):
    device = torch.device(config.model.device if torch.cuda.is_available() else "cpu")
    print(f"Loading model '{config.model.name}' from {config.model.weights}")
    model, ckpt = ModelLoader.load(config.model.name, config.model.weights, device)

    engine = PruningEngine(
        model_name=config.model.name,
        pruner_name=config.pruning.pruner,
        criterion_name=config.pruning.criterion,
        granularity_name=config.pruning.granularity
    )

    exec_config = {
        "amount": config.pruning.amount,
        "pruning_params": config.pruning.layer_params or config.pruning.amount
    }

    result = engine.execute(model, exec_config, verify_forward=True)

    if config.benchmark.enabled:
        benchmarker = LatencyBenchmark(warmup_runs=config.benchmark.warmup, test_runs=config.benchmark.runs)
        adapter = engine.adapter_cls(model)
        dummy_input = adapter.get_dummy_input(device)
        result.benchmark = benchmarker.benchmark(model, dummy_input)
        print(f"Benchmark Results: {result.benchmark}")

    if config.export.enabled:
        adapter = engine.adapter_cls(model)
        dummy_input = adapter.get_dummy_input(device)
        ModelExporter.export_onnx(model, dummy_input, config.export.output_path)

    ModelExporter.export_checkpoint(model, config.output_path, ckpt)
    return result
