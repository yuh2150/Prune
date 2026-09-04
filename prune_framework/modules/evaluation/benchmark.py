import time
import torch
import torch.nn as nn
from prune_framework.core.results import BenchmarkResult


class LatencyBenchmark:
    """Decoupled latency benchmark module measuring raw inference and post-processing."""

    def __init__(self, warmup_runs: int = 10, test_runs: int = 100):
        self.warmup_runs = warmup_runs
        self.test_runs = test_runs

    def benchmark(
        self,
        model: nn.Module,
        dummy_input: torch.Tensor,
        post_process_fn=None
    ) -> BenchmarkResult:
        device = dummy_input.device
        model.eval()

        with torch.no_grad():
            for _ in range(self.warmup_runs):
                pred = model(dummy_input)
                if post_process_fn:
                    _ = post_process_fn(pred)
            if device.type == "cuda":
                torch.cuda.synchronize()

        inf_times = []
        with torch.no_grad():
            for _ in range(self.test_runs):
                t0 = time.perf_counter()
                pred = model(dummy_input)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                inf_times.append((t1 - t0) * 1000.0)

        nms_times = []
        if post_process_fn and pred is not None:
            for _ in range(self.test_runs):
                t0 = time.perf_counter()
                _ = post_process_fn(pred)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                nms_times.append((t1 - t0) * 1000.0)

        avg_inf_ms = sum(inf_times) / len(inf_times)
        avg_nms_ms = sum(nms_times) / len(nms_times) if nms_times else 0.0
        total_ms = avg_inf_ms + avg_nms_ms
        fps = 1000.0 / total_ms if total_ms > 0 else 0.0

        return BenchmarkResult(
            inference_latency_ms=round(avg_inf_ms, 3),
            nms_latency_ms=round(avg_nms_ms, 3),
            total_latency_ms=round(total_ms, 3),
            fps=round(fps, 2)
        )
