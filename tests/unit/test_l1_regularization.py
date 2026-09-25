"""L1 BatchNorm-scale regularization and structural-pruning integration tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from prune_framework.core.config import FrameworkConfig
from prune_framework.core.exceptions import ConfigValidationException
from prune_framework.modules.regularization import L1BatchNormScaleRegularization, RegularizationController
from prune_framework.pipelines.unified import UnifiedPruningPipeline
from prune_framework.plugins.adapters.yolov5 import YOLOv5Adapter
from prune_framework.plugins.criteria.bn_criteria import BNScaleCriterion
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


class TestL1Regularization(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(53)
        self.model = ConvBNDetector()
        self.adapter = YOLOv5Adapter(self.model)
        self.term = L1BatchNormScaleRegularization()

    def _controller(self, **kwargs):
        return RegularizationController(self.term, self.adapter, strength=0.2, **kwargs)

    def test_l1_penalty_is_added_and_reaches_bn_scale_gradients(self):
        with torch.no_grad():
            self.model.features[0].bn.weight.copy_(torch.tensor([1.0, -2.0, 3.0, -4.0, 1.0, -2.0, 3.0, -4.0]))
            self.model.features[1].bn.weight.fill_(0.5)
        controller = self._controller()
        task_loss = self.model(torch.randn(2, 3, 8, 8)).sum() * 0.0
        total, report = controller.augment_loss(task_loss)
        expected_raw = sum(target.parameter.detach().abs().sum() for target in self.adapter.get_channel_sparsity_targets())

        self.assertTrue(torch.allclose(total.detach(), task_loss.detach() + expected_raw * 0.2))
        self.assertEqual(report.target_count, 2)
        self.assertEqual(report.strength, 0.2)
        total.backward()
        expected_gradient = 0.2 * self.model.features[0].bn.weight.detach().sign()
        self.assertTrue(torch.allclose(self.model.features[0].bn.weight.grad, expected_gradient))

    def test_regularization_drives_bn_scales_down_and_schedule_is_deterministic(self):
        controller = self._controller(schedule="linear_warmup", warmup_steps=2)
        optimizer = torch.optim.SGD([target.parameter for target in self.adapter.get_channel_sparsity_targets()], lr=0.25)
        before = self.model.features[0].bn.weight.detach().abs().mean()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            zero_task_loss = self.model(torch.randn(2, 3, 8, 8)).sum() * 0.0
            total, _ = controller.augment_loss(zero_task_loss)
            total.backward()
            optimizer.step()
            controller.advance()
        after = self.model.features[0].bn.weight.detach().abs().mean()

        self.assertLess(after, before)
        self.assertAlmostEqual(controller.current_strength(step=0), 0.1)
        self.assertAlmostEqual(controller.current_strength(step=1), 0.2)
        cosine = self._controller(schedule="cosine_decay", warmup_steps=1, total_steps=5)
        self.assertAlmostEqual(cosine.current_strength(step=0), 0.2)
        self.assertGreater(cosine.current_strength(step=1), cosine.current_strength(step=4))

    def test_zero_strength_preserves_task_loss_and_gradients(self):
        baseline = ConvBNDetector()
        baseline.load_state_dict(self.model.state_dict())
        baseline_images = torch.randn(2, 3, 8, 8)
        loss = baseline(baseline_images).square().mean()
        loss.backward()
        expected_gradients = {name: parameter.grad.detach().clone() for name, parameter in baseline.named_parameters()}

        controller = RegularizationController(self.term, self.adapter, strength=0.0)
        total, report = controller.augment_loss(self.model(baseline_images).square().mean())
        total.backward()
        self.assertEqual(report.weighted_penalty, 0.0)
        self.assertTrue(torch.equal(total.detach(), loss.detach()))
        for name, parameter in self.model.named_parameters():
            self.assertTrue(torch.allclose(parameter.grad, expected_gradients[name]))

    def test_yolo_adapter_wraps_its_integrated_task_loss_without_training_loop_changes(self):
        controller = self._controller()
        batch = torch.randn(2, 3, 8, 8)
        base_loss = lambda model, images: model(images).square().mean()
        with patch.object(self.adapter, "build_gradient_calibration_loss", return_value=base_loss):
            regularized_loss = self.adapter.build_regularized_training_loss(controller)
            total, report = regularized_loss(self.model, batch, step=0)
        expected = base_loss(self.model, batch) + self.term.raw_penalty(self.adapter) * 0.2
        self.assertTrue(torch.allclose(total, expected))
        self.assertEqual(report.target_count, 2)

    def test_structured_pruning_uses_learned_bn_scales_and_forwards(self):
        with torch.no_grad():
            self.model.features[0].bn.weight.copy_(torch.arange(1, 9, dtype=torch.float32))
            self.model.features[1].bn.weight.copy_(torch.arange(1, 9, dtype=torch.float32))
        pruner = StructuredPruner()
        plan = pruner.create_plan(
            self.adapter, BNScaleCriterion(), ChannelGranularity(), {"amount": 0.25, "min_channels": 2}
        )
        self.assertTrue(plan.groups)
        self.assertTrue(pruner.validate_plan(plan, self.adapter))
        self.assertTrue(all(group.validated for group in plan.groups))
        pruner.apply_plan(plan, self.adapter)
        self.assertLess(self.model.features[0].conv.out_channels, 8)
        self.assertEqual(tuple(self.model(torch.randn(1, 3, 8, 8)).shape), (1, 6, 8, 8))

    def test_checkpoint_round_trip_restores_model_optimizer_and_regularization_progress(self):
        controller = self._controller(schedule="cosine_decay", total_steps=4)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        optimizer.zero_grad(set_to_none=True)
        total, _ = controller.augment_loss(self.model(torch.randn(2, 3, 8, 8)).square().mean())
        total.backward()
        optimizer.step()
        controller.advance(3)

        with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
            torch.save(
                {"model": self.model.state_dict(), "optimizer": optimizer.state_dict(), "regularization": controller.state_dict()},
                handle.name,
            )
            checkpoint = torch.load(handle.name, weights_only=True)
        restored = ConvBNDetector()
        restored.load_state_dict(checkpoint["model"])
        restored_adapter = YOLOv5Adapter(restored)
        restored_controller = RegularizationController(
            L1BatchNormScaleRegularization(), restored_adapter, strength=0.2, schedule="cosine_decay", total_steps=4
        )
        restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.1, momentum=0.9)
        restored_optimizer.load_state_dict(checkpoint["optimizer"])
        restored_controller.load_state_dict(checkpoint["regularization"])

        self.assertEqual(restored_controller.step, 3)
        self.assertEqual(len(restored_optimizer.state), len(optimizer.state))
        for original, reloaded in zip(self.model.parameters(), restored.parameters()):
            self.assertTrue(torch.equal(original, reloaded))

    def test_yolo_recovery_receives_regularizer_then_prunes_bn_channels_and_rtdetr_rejects(self):
        with tempfile.TemporaryDirectory() as directory:
            config = FrameworkConfig.from_dict(
                {
                    "model": {"name": "yolov5", "weights": "unused.pt", "device": "cpu"},
                    "pruning": {"method": "structured", "criterion": "l1", "structure": "channel", "target_ratio": 0.25},
                    "regularization": {"enabled": True, "strength": 0.1, "schedule": "constant", "total_steps": 2},
                    "benchmark": {"enabled": False, "latency": False, "flops": False, "params": False, "runs": 1},
                    "export": {"enabled": False},
                    "experiment": {"name": "regularized", "output_dir": directory, "seed": 4},
                    "output_path": str(Path(directory) / "model.pt"),
                }
            )
            received = []

            def recovery(model, stage, regularization, adapter):
                self.assertEqual(stage, "regularization")
                received.append(regularization)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    total, _ = regularization.augment_loss(model(torch.randn(1, 3, 16, 16)).square().mean())
                    total.backward()
                    optimizer.step()
                    regularization.advance()
                return model

            with patch("prune_framework.pipelines.unified.ModelLoader.load", return_value=(ConvBNDetector(), None)):
                result = UnifiedPruningPipeline(config, recovery=recovery).run()
            self.assertEqual(len(received), 1)
            self.assertTrue(result.pruning.forward_verified)
            self.assertIn("regularization", result.artifacts)
            self.assertEqual(result.pruning.criterion_name, "bn_scale")

        unsupported = FrameworkConfig.from_dict(
            {
                "model": {"name": "rtdetr", "weights": "unused", "device": "cpu"},
                "pruning": {"method": "structured", "structure": "channel", "target_ratio": 0.25},
                "regularization": {"enabled": True, "strength": 0.1},
                "benchmark": {"enabled": False, "latency": False, "flops": False, "params": False, "runs": 1},
            }
        )
        with patch("prune_framework.pipelines.unified.ModelLoader.load", return_value=(ConvBNDetector(), None)):
            with self.assertRaisesRegex(ConfigValidationException, "Conv-BN recovery path"):
                UnifiedPruningPipeline(unsupported, recovery=lambda model: model).run()


if __name__ == "__main__":
    unittest.main()
