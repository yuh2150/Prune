"""CPU recovery contracts for the YOLO callback without a detector dataset."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from experiments.yolov5 import recover
from prune_framework.core.config import FrameworkConfig
from prune_framework.core.engine import PruningEngine
from prune_framework.modules.model.masks import MaskManager
from prune_framework.pipelines.unified import UnifiedPruningPipeline


class TinyRecoveryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 1, bias=False)
        self.head = nn.Conv2d(4, 2, 1, bias=False)

    def forward(self, images):
        return self.head(torch.relu(self.conv(images)))


class TinyPipelineRecoveryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1, bias=False)
        self.head = nn.Conv2d(4, 2, 1, bias=False)

    def forward(self, images):
        return self.head(torch.relu(self.conv(images)))


def _config(epochs=2):
    return FrameworkConfig.from_dict(
        {
            "model": {"device": "cpu", "input_shape": [1, 1, 8, 8]},
            "recovery": {"enabled": True, "epochs": epochs},
            "benchmark": {"enabled": False, "latency": False, "flops": False, "params": False},
            "export": {"enabled": False},
        }
    )


class TestYOLOv5Recovery(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(51)
        self.model = TinyRecoveryModel()
        self.batches = [torch.randn(2, 1, 8, 8), torch.randn(2, 1, 8, 8)]

    @staticmethod
    def loss(candidate, batch):
        return candidate(batch).square().mean()

    def test_recovery_updates_the_received_model_and_returns_metadata(self):
        before = self.model.conv.weight.detach().clone()
        self.model.eval()
        result = recover(
            self.model, _config(), "cpu", dataloader=self.batches, loss_fn=self.loss,
            max_batches=3, lr=0.1, scheduler_name="step",
        )
        self.assertIs(result["model"], self.model)
        self.assertEqual(result["metrics"]["recovery_steps"], 3)
        self.assertEqual(result["metrics"]["recovery_epochs"], 2)
        self.assertEqual(result["metadata"]["optimizer"], "SGD")
        self.assertFalse(self.model.training)
        self.assertFalse(torch.equal(before, self.model.conv.weight))

    def test_unstructured_masks_survive_recovery_and_state_round_trip(self):
        mask = torch.ones_like(self.model.conv.weight)
        mask.flatten()[::2] = 0
        MaskManager.apply(self.model.conv, mask)
        result = recover(
            self.model, _config(), "cpu", dataloader=self.batches, loss_fn=self.loss,
            max_batches=4, optimizer_name="adam", lr=0.01,
        )
        self.assertIs(result["model"], self.model)
        original = MaskManager.original_weight(self.model.conv)
        self.assertTrue(torch.equal(original[mask == 0], torch.zeros_like(original[mask == 0])))

        with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
            torch.save({"model": self.model.state_dict(), "masks": MaskManager.state_dict(self.model)}, handle.name)
            checkpoint = torch.load(handle.name, weights_only=True)
            restored = TinyRecoveryModel()
            MaskManager.load_state_dict(restored, checkpoint["masks"])
            restored.load_state_dict(checkpoint["model"])
        self.assertTrue(torch.equal(MaskManager.mask(restored.conv), mask))
        self.assertTrue(torch.equal(self.model(self.batches[0]), restored(self.batches[0])))

    def test_structured_model_can_recover_after_parameter_graph_changes(self):
        model = nn.Sequential(nn.Conv2d(3, 8, 1), nn.ReLU(), nn.Conv2d(8, 4, 1))
        PruningEngine("yolov5", "structured", "l1", "channel").execute(
            model, {"amount": 0.25, "min_channels": 2}
        )
        pruned = model
        before_ids = {id(parameter) for parameter in pruned.parameters()}
        result = recover(
            pruned, _config(epochs=1), "cpu", dataloader=[torch.randn(2, 3, 8, 8)],
            loss_fn=lambda candidate, batch: candidate(batch).square().mean(), max_batches=1,
        )
        self.assertIs(result["model"], pruned)
        self.assertEqual(before_ids, {id(parameter) for parameter in pruned.parameters()})
        expected_channels = pruned[2].out_channels
        self.assertEqual(tuple(pruned(torch.randn(1, 3, 8, 8)).shape), (1, expected_channels, 8, 8))
        restored = copy.deepcopy(pruned)
        restored.load_state_dict(pruned.state_dict())
        self.assertEqual(tuple(restored(torch.randn(1, 3, 8, 8)).shape), (1, expected_channels, 8, 8))

    def test_pipeline_recovery_result_contract_preserves_legacy_none(self):
        model, details = UnifiedPruningPipeline._resolve_recovery_output(None, self.model)
        self.assertIs(model, self.model)
        self.assertEqual(details, {})
        model, details = UnifiedPruningPipeline._resolve_recovery_output(
            {"model": self.model, "metrics": {"loss": 1.0}}, self.model
        )
        self.assertIs(model, self.model)
        self.assertEqual(details["metrics"], {"loss": 1.0})

    def test_pipeline_runs_prune_recovery_checkpoint_with_injected_tiny_training_data(self):
        with tempfile.TemporaryDirectory() as directory:
            batches = [torch.randn(2, 3, 8, 8)]
            config = FrameworkConfig.from_dict(
                {
                    "model": {"name": "yolov5", "weights": "unused.pt", "device": "cpu"},
                    "pruning": {"method": "unstructured", "criterion": "magnitude", "structure": "weight", "target_ratio": 0.25},
                    "recovery": {
                        "enabled": True,
                        "epochs": 1,
                    },
                    "benchmark": {"enabled": False, "latency": False, "flops": False, "params": False, "runs": 1},
                    "export": {"enabled": False},
                    "experiment": {"name": "tiny_recovery", "output_dir": directory, "seed": 17},
                    "output_path": str(Path(directory) / "model.pt"),
                }
            )
            def injected_recovery(model, config, device, stage):
                return recover(model, config, device, stage, dataloader=batches, loss_fn=self.loss, max_batches=1, lr=0.01)

            with patch("prune_framework.pipelines.unified.ModelLoader.load", return_value=(TinyPipelineRecoveryModel(), None)):
                result = UnifiedPruningPipeline(config, recovery=injected_recovery).run()
            self.assertTrue(result.pruning.forward_verified)
            self.assertIn("recovery", result.artifacts)
            self.assertTrue(Path(result.artifacts["checkpoint"]).is_file())


if __name__ == "__main__":
    unittest.main()
