import contextlib
import io
import unittest
from dataclasses import asdict
from unittest.mock import patch

import torch

from prune_framework.core.config import FrameworkConfig
from prune_framework.core.results import BenchmarkResult
from prune_framework.modules.evaluation.benchmark import attach_quality_metrics
from prune_framework.pipelines.benchmarking import run_benchmarking_pipeline
from prune_framework.pipelines.unified import _benchmark_summary


class TestBenchmarkQuality(unittest.TestCase):
    def test_detection_quality_alias_is_attached_and_serializable(self):
        metrics = {"precision": .8, "recall": .7, "map50": .6, "map": .4}
        result = asdict(attach_quality_metrics(BenchmarkResult(), metrics))
        self.assertEqual([result[k] for k in ("precision", "recall", "map50", "map50_95")],
                         [.8, .7, .6, .4])

    def test_missing_map_is_none_and_real_zero_is_preserved(self):
        result = attach_quality_metrics(BenchmarkResult(), {"accuracy": .1, "precision": 0., "recall": 0., "f1": 0.})
        self.assertEqual(result.precision, 0.)
        self.assertEqual(result.f1, 0.)
        self.assertIsNone(result.map50)
        self.assertIsNone(result.map50_95)

    def test_summary_contains_final_quality_and_deltas(self):
        baseline = {"precision": .9, "recall": .8, "f1": .85, "map50": .7, "map50_95": .5}
        final = {"precision": .8, "recall": .7, "f1": .75, "map50": .6, "map50_95": .48}
        summary = _benchmark_summary(baseline, final, None, .3, {"actual_sparsity": .3})
        for name, value in final.items():
            self.assertEqual(summary[name], value)
            self.assertAlmostEqual(summary[f"{name}_delta"], value - baseline[name])
        self.assertEqual(summary["status"], "GOOD")

    def test_standalone_benchmark_evaluates_real_model_and_prints_quality(self):
        config = FrameworkConfig.from_dict({
            "model": {"name": "lenet5", "weights": "random", "device": "cpu"},
            "benchmark": {"warmup": 0, "runs": 1},
        })
        model = torch.nn.Linear(2, 2)
        stages = []

        def evaluator(model, stage):
            stages.append(stage)
            return {"precision": .5, "recall": .6, "map50": .7, "map50_95": .4}

        output = io.StringIO()
        with patch("prune_framework.pipelines.benchmarking.ModelLoader.load", return_value=(model, None)), \
                patch("prune_framework.pipelines.benchmarking.PluginRegistry.get_model_adapter") as adapter, \
                contextlib.redirect_stdout(output):
            adapter.return_value.return_value.get_dummy_input.return_value = torch.zeros(1, 2)
            result = run_benchmarking_pipeline(config, evaluator=evaluator)
        self.assertEqual(stages, ["benchmark"])
        self.assertEqual(result.map50_95, .4)
        self.assertIn("Precision: 0.5000", output.getvalue())
        self.assertIn("mAP@0.5:0.95: 0.4000", output.getvalue())
