import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from prune_framework.core.config import FrameworkConfig
from prune_framework.core.exceptions import ConfigValidationException
from prune_framework.pipelines.unified import UnifiedPruningPipeline
from prune_framework.plugins.adapters.yolov5 import YOLOv5Adapter


class TinyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 12, 3, padding=1),
        )

    def forward(self, images):
        return self.features(images)


class TestUnifiedPruningPipeline(unittest.TestCase):
    def test_rtdetr_taylor_reports_missing_integrated_calibration(self):
        config = FrameworkConfig.from_dict(
            {
                "model": {"name": "rtdetr", "weights": "unused", "device": "cpu"},
                "pruning": {"method": "structured", "criterion": "taylor", "structure": "channel", "target_ratio": 0.25},
                "benchmark": {"enabled": False, "latency": False, "flops": False, "params": False, "runs": 1},
            }
        )
        with patch("prune_framework.pipelines.unified.ModelLoader.load", return_value=(TinyDetector(), None)):
            with self.assertRaisesRegex(ConfigValidationException, "does not provide an integrated task loss"):
                UnifiedPruningPipeline(config).run()

    def test_taylor_uses_integrated_calibration_without_callback(self):
        with tempfile.TemporaryDirectory() as directory:
            config = FrameworkConfig.from_dict(
                {
                    "model": {"name": "yolov5", "weights": "unused.pt", "device": "cpu"},
                    "pruning": {
                        "method": "structured",
                        "criterion": "taylor",
                        "structure": "channel",
                        "target_ratio": 0.25,
                        "calibration_batches": 1,
                        "calibration_seed": 9,
                    },
                    "benchmark": {"enabled": False, "latency": False, "flops": False, "params": False, "runs": 1},
                    "export": {"enabled": False},
                    "experiment": {"name": "taylor", "output_dir": directory, "seed": 9},
                    "output_path": str(Path(directory) / "model.pt"),
                }
            )
            calibration_batch = torch.randn(1, 3, 16, 16)
            with patch("prune_framework.pipelines.unified.ModelLoader.load", return_value=(TinyDetector(), None)), patch.object(
                YOLOv5Adapter, "get_gradient_calibration_batches", return_value=[calibration_batch]
            ), patch.object(
                YOLOv5Adapter,
                "build_gradient_calibration_loss",
                return_value=lambda model, batch: model(batch).square().mean(),
            ):
                result = UnifiedPruningPipeline(config).run()

            self.assertTrue(result.pruning.forward_verified)
            self.assertIn("calibration", result.artifacts)
            self.assertTrue(Path(result.artifacts["calibration"]).is_file())

    def test_stage_pipeline_saves_reproducible_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            config = FrameworkConfig.from_dict(
                {
                    "model": {"name": "yolov5", "weights": "unused.pt", "device": "cpu"},
                    "pruning": {"method": "structured", "structure": "channel", "target_ratio": 0.25},
                    "sensitivity": {"enabled": True, "rates": [0.1], "selector": "sensitivity"},
                    "evaluation": {"enabled": True, "metric": "map"},
                    "recovery": {"enabled": True, "epochs": 1},
                    "benchmark": {"enabled": False, "latency": False, "flops": False, "params": True, "runs": 1},
                    "export": {"enabled": False},
                    "experiment": {"name": "unit", "output_dir": directory, "seed": 7},
                    "output_path": str(Path(directory) / "model.pt"),
                }
            )
            recovered = []

            def evaluator(model, stage):
                self.assertIn(stage, {"baseline", "sensitivity", "final"})
                return {"map": 0.8}

            def recovery(model, config, stage):
                self.assertEqual(stage, "recovery")
                recovered.append(config.recovery.epochs)
                return model

            with patch(
                "prune_framework.pipelines.unified.ModelLoader.load",
                return_value=(TinyDetector(), None),
            ):
                result = UnifiedPruningPipeline(config, evaluator=evaluator, recovery=recovery).run()

            self.assertTrue(result.pruning.forward_verified)
            self.assertEqual(result.baseline_metrics, {"map": 0.8})
            self.assertEqual(result.final_metrics, {"map": 0.8})
            self.assertEqual(recovered, [1])
            self.assertTrue(Path(result.artifacts["config"]).is_file())
            self.assertTrue(Path(result.artifacts["sensitivity"]).is_file())
            self.assertTrue(Path(result.artifacts["selection"]).is_file())
            self.assertTrue(Path(result.artifacts["pruning_plan"]).is_file())
            self.assertTrue(Path(result.artifacts["result"]).is_file())
            self.assertTrue(Path(result.artifacts["checkpoint"]).is_file())


if __name__ == "__main__":
    unittest.main()
