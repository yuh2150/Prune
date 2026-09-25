"""Hard-Concrete L0 channel-gate tests over the shared structured pruner."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from prune_framework.core.config import FrameworkConfig
from prune_framework.core.exceptions import ConfigValidationException
from prune_framework.modules.regularization import (
    HardConcreteChannelGate,
    L0HardConcreteRegularization,
    RegularizationController,
)
from prune_framework.pipelines.unified import UnifiedPruningPipeline
from prune_framework.plugins.adapters.yolov5 import YOLOv5Adapter
from prune_framework.plugins.criteria.l0_gate_criteria import HardConcreteGateCriterion
from prune_framework.plugins.granularities.base import ChannelGranularity
from prune_framework.plugins.pruners.structured import StructuredPruner


class ConvBN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU()

    def forward(self, images):
        return self.act(self.bn(self.conv(images)))


class ConvBNDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(ConvBN(3, 8), ConvBN(8, 8))

    def forward(self, images):
        return self.features(images)


class TestL0HardConcrete(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(61)
        self.model = ConvBNDetector()
        self.adapter = YOLOv5Adapter(self.model)
        self.term = L0HardConcreteRegularization(log_alpha_init=0.0)
        self.controller = RegularizationController(self.term, self.adapter, strength=0.2, total_steps=5)

    @property
    def first_gate(self):
        return self.model.features[0].bn._prune_hard_concrete_gate

    def test_sampling_and_deterministic_gate_behavior(self):
        gate = self.first_gate
        torch.manual_seed(7)
        first = gate.sample_gate()
        second = gate.sample_gate()
        self.assertTrue(torch.all((first >= 0) & (first <= 1)))
        self.assertFalse(torch.equal(first, second))

        self.model.eval()
        deterministic = gate.deterministic_gate()
        output_one = self.model.features[0].bn(torch.ones(1, 8, 2, 2))
        output_two = self.model.features[0].bn(torch.ones(1, 8, 2, 2))
        self.assertTrue(torch.equal(output_one, output_two))
        self.assertTrue(torch.allclose(output_one[:, :, 0, 0], deterministic.reshape(1, -1)))

    def test_expected_l0_matches_reference_and_logits_receive_gradients(self):
        values = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0, -2.0, -1.0, 0.0])
        with torch.no_grad():
            self.first_gate.log_alpha.copy_(values)
        expected = torch.sigmoid(
            values - self.first_gate.beta * torch.log(torch.tensor(-self.first_gate.gamma / self.first_gate.zeta))
        ).sum()
        self.assertTrue(torch.allclose(self.first_gate.expected_l0(), expected))

        task_loss = self.model(torch.randn(2, 3, 8, 8)).sum() * 0.0
        total, report = self.controller.augment_loss(task_loss)
        total.backward()
        self.assertGreater(report.raw_penalty, 0.0)
        self.assertIsNotNone(self.first_gate.log_alpha.grad)
        self.assertTrue(torch.any(self.first_gate.log_alpha.grad != 0))

    def test_expected_active_channels_reduce_with_l0_regularization(self):
        gates = [binding.gate.log_alpha for binding in self.term._bindings.values()]
        optimizer = torch.optim.SGD(gates, lr=1.0)
        before = self.term.raw_penalty(self.adapter).detach()
        for _ in range(4):
            optimizer.zero_grad(set_to_none=True)
            total, _ = self.controller.augment_loss(self.model(torch.randn(1, 3, 8, 8)).sum() * 0.0)
            total.backward()
            optimizer.step()
            self.controller.advance()
        after = self.term.raw_penalty(self.adapter).detach()
        self.assertLess(after, before)

    def test_checkpoint_round_trip_preserves_gates_optimizer_and_controller_state(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        optimizer.zero_grad(set_to_none=True)
        total, _ = self.controller.augment_loss(self.model(torch.randn(2, 3, 8, 8)).square().mean())
        total.backward()
        optimizer.step()
        self.controller.advance(3)
        gate_before = self.first_gate.log_alpha.detach().clone()

        with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
            torch.save(
                {"model": self.model.state_dict(), "optimizer": optimizer.state_dict(), "regularization": self.controller.state_dict()},
                handle.name,
            )
            checkpoint = torch.load(handle.name, weights_only=True)
        restored = ConvBNDetector()
        restored_adapter = YOLOv5Adapter(restored)
        restored_term = L0HardConcreteRegularization(log_alpha_init=0.0)
        restored_controller = RegularizationController(restored_term, restored_adapter, strength=0.2, total_steps=5)
        restored.load_state_dict(checkpoint["model"])
        restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.1, momentum=0.9)
        restored_optimizer.load_state_dict(checkpoint["optimizer"])
        restored_controller.load_state_dict(checkpoint["regularization"])

        self.assertEqual(restored_controller.step, 3)
        self.assertTrue(torch.equal(restored.features[0].bn._prune_hard_concrete_gate.log_alpha, gate_before))
        self.assertEqual(len(restored_optimizer.state), len(optimizer.state))

    def test_deterministic_gate_decisions_build_validated_structural_plan_then_forward(self):
        with torch.no_grad():
            for binding in self.term._bindings.values():
                binding.gate.log_alpha.copy_(torch.tensor([-10.0, -10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]))
        scores = HardConcreteGateCriterion().score(self.model.features[0])
        self.assertTrue(torch.all(scores[:2] <= 0.5))
        decisions = self.controller.prune_indices(threshold=0.5, min_channels=2)
        self.assertEqual(decisions["features.0.conv"], [0, 1])
        pruner = StructuredPruner()
        plan = pruner.create_plan(
            self.adapter,
            HardConcreteGateCriterion(),
            ChannelGranularity(),
            {"amount": 0.25, "min_channels": 2, "forced_channel_indices": decisions, "l0_gate_controller": self.controller},
        )
        self.assertTrue(pruner.validate_plan(plan, self.adapter))
        self.assertTrue(hasattr(self.model.features[0].bn, "_prune_hard_concrete_gate"))
        pruner.apply_plan(plan, self.adapter, {"l0_gate_controller": self.controller})
        self.assertFalse(hasattr(self.model.features[0].bn, "_prune_hard_concrete_gate"))
        self.assertEqual(tuple(self.model(torch.randn(1, 3, 8, 8)).shape), (1, 6, 8, 8))

    def test_yolo_pipeline_l0_then_structural_pruning_and_rtdetr_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            config = FrameworkConfig.from_dict(
                {
                    "model": {"name": "yolov5", "weights": "unused.pt", "device": "cpu"},
                    "pruning": {"method": "structured", "criterion": "l1", "structure": "channel", "target_ratio": 0.25},
                    "regularization": {
                        "enabled": True, "term": "l0_hard_concrete", "strength": 0.1,
                        "gate_threshold": 0.5, "total_steps": 1,
                    },
                    "benchmark": {"enabled": False, "latency": False, "flops": False, "params": False, "runs": 1},
                    "export": {"enabled": False},
                    "experiment": {"name": "l0", "output_dir": directory, "seed": 6},
                    "output_path": str(Path(directory) / "model.pt"),
                }
            )

            def recovery(model, stage, regularization):
                self.assertEqual(stage, "regularization")
                for binding in regularization.term._bindings.values():
                    with torch.no_grad():
                        binding.gate.log_alpha.copy_(torch.tensor([-10.0, -10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]))
                return model

            with patch("prune_framework.pipelines.unified.ModelLoader.load", return_value=(ConvBNDetector(), None)):
                result = UnifiedPruningPipeline(config, recovery=recovery).run()
            self.assertTrue(result.pruning.forward_verified)
            self.assertEqual(result.pruning.criterion_name, "l0_gate")
            self.assertIn("regularization", result.artifacts)
            self.assertIn("gate_pruning_decisions", result.pruning.extra_metrics["stages"])

        unsupported = FrameworkConfig.from_dict(
            {
                "model": {"name": "rtdetr", "weights": "unused", "device": "cpu"},
                "pruning": {"method": "structured", "structure": "channel", "target_ratio": 0.25},
                "regularization": {"enabled": True, "term": "l0_hard_concrete", "strength": 0.1},
                "benchmark": {"enabled": False, "latency": False, "flops": False, "params": False, "runs": 1},
            }
        )
        with patch("prune_framework.pipelines.unified.ModelLoader.load", return_value=(ConvBNDetector(), None)):
            with self.assertRaisesRegex(ConfigValidationException, "Conv-BN recovery path"):
                UnifiedPruningPipeline(unsupported, recovery=lambda model: model).run()


if __name__ == "__main__":
    unittest.main()
