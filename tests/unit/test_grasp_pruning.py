"""Focused GraSP tests for the shared unstructured pruning path."""

from __future__ import annotations

import copy
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from prune_framework.contracts import BaseModelAdapter, TargetType
from prune_framework.core.config import FrameworkConfig
from prune_framework.core.exceptions import ConfigValidationException
from prune_framework.modules.calibration import HigherOrderCalibrationResult, HigherOrderCalibrationRunner
from prune_framework.modules.model.masks import MaskManager
from prune_framework.pipelines.unified import UnifiedPruningPipeline
from prune_framework.plugins.adapters.yolov5 import YOLOv5Adapter
from prune_framework.plugins.criteria.grasp_criteria import GraSPCriterion
from prune_framework.plugins.granularities.base import WeightGranularity
from prune_framework.plugins.pruners.unstructured import UnstructuredPruner


class ConvLinearFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=2, bias=False)
        self.linear = nn.Linear(18, 4, bias=False)

    def forward(self, images):
        return self.linear(self.conv(images).flatten(1))


class StatefulConvFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=2, bias=False)
        self.bn = nn.BatchNorm2d(2)

    def forward(self, images):
        return self.bn(self.conv(images))


class FixtureAdapter(BaseModelAdapter):
    @classmethod
    def load_model(cls, weights_path, device):
        raise NotImplementedError

    def get_pruneable_modules(self):
        return [("conv", self.model.conv), ("linear", self.model.linear)]

    def get_pruneable_blocks(self):
        return []

    def get_dummy_input(self, device):
        return torch.ones(2, 1, 4, 4, device=device)

    def supported_target_types(self):
        return {
            TargetType.CONV_WEIGHT,
            TargetType.CONV_OUT_CHANNEL,
            TargetType.LINEAR_WEIGHT,
            TargetType.LINEAR_OUT_FEATURE,
        }


class TinyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 12, 3, padding=1)
        )

    def forward(self, images):
        return self.features(images)


class TestGraSPPruning(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.model = ConvLinearFixture()
        self.adapter = FixtureAdapter(self.model)
        self.targets = self.adapter.get_prunable_targets({TargetType.CONV_WEIGHT, TargetType.LINEAR_WEIGHT})
        self.images = torch.randn(2, 1, 4, 4)

    @staticmethod
    def loss_fn(model, batch):
        return model(batch).square().mean()

    def _calibrate(self, batches=None):
        return HigherOrderCalibrationRunner(seed=41).run(
            self.model, self.targets, batches or [self.images], self.loss_fn
        )

    def test_scores_match_direct_hessian_gradient_reference_for_conv_and_linear(self):
        reference = copy.deepcopy(self.model)
        parameters = list(reference.parameters())
        loss = self.loss_fn(reference, self.images)
        first_order = torch.autograd.grad(loss, parameters, create_graph=True)
        half_squared_norm = sum(gradient.square().sum() * 0.5 for gradient in first_order)
        hessian_gradient = torch.autograd.grad(half_squared_norm, parameters)
        expected = {
            "conv": -(reference.conv.weight.detach() * hessian_gradient[0].detach()),
            "linear": -(reference.linear.weight.detach() * hessian_gradient[1].detach()),
        }

        result = self._calibrate()
        criterion = GraSPCriterion()
        for target in self.targets:
            score = criterion.score(target.module, result.context_for(target))
            self.assertTrue(torch.allclose(score, expected[target.name]))
            self.assertEqual(tuple(score.shape), tuple(target.module.weight.shape))
            self.assertFalse(score.requires_grad)
            self.assertIsNone(score.grad_fn)

    def test_higher_order_calibration_is_deterministic_and_restores_all_visible_state(self):
        model = StatefulConvFixture()
        target = self_target = self.targets[0]
        target = type(self_target)("conv", model.conv, TargetType.CONV_WEIGHT)
        model.eval()
        model.bn.train()  # Preserve intentionally mixed module modes.
        model.conv.weight.requires_grad_(False)
        original_grad = torch.full_like(model.conv.weight, 7.0)
        model.conv.weight.grad = original_grad.clone()
        state_before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        batches = [torch.randn(2, 1, 4, 4), torch.randn(2, 1, 4, 4)]
        rng_before = torch.random.get_rng_state()
        python_before = random.getstate()

        first = HigherOrderCalibrationRunner(seed=8).run(
            model, [target], batches, lambda candidate, batch: candidate(batch).square().mean()
        )
        second = HigherOrderCalibrationRunner(seed=8).run(
            model, [target], batches, lambda candidate, batch: candidate(batch).square().mean()
        )

        self.assertTrue(torch.equal(first.gradients["conv"], second.gradients["conv"]))
        self.assertTrue(torch.equal(first.first_order_gradients["conv"], second.first_order_gradients["conv"]))
        self.assertFalse(first.gradients["conv"].requires_grad)
        self.assertIsNone(first.gradients["conv"].grad_fn)
        self.assertFalse(first.first_order_gradients["conv"].requires_grad)
        self.assertIsNone(first.first_order_gradients["conv"].grad_fn)
        self.assertFalse(model.training)
        self.assertTrue(model.bn.training)
        self.assertFalse(model.conv.weight.requires_grad)
        self.assertTrue(torch.equal(model.conv.weight.grad, original_grad))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))
        self.assertEqual(random.getstate(), python_before)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, state_before[name]))

    def test_global_plan_ties_are_stable_exact_and_masks_remain_effective(self):
        with torch.no_grad():
            self.model.conv.weight.fill_(1.0)
            self.model.linear.weight.fill_(1.0)
        zero_hvp = {target.name: torch.zeros_like(target.module.weight) for target in self.targets}
        calibration = HigherOrderCalibrationResult(gradients=zero_hvp, batches=1)
        pruner = UnstructuredPruner()
        config = {"amount": 0.25, "criterion_name": "grasp", "higher_order_calibration": calibration}
        first_plan = pruner.create_plan(self.adapter, GraSPCriterion(), WeightGranularity(), config)
        second_plan = pruner.create_plan(self.adapter, GraSPCriterion(), WeightGranularity(), config)

        self.assertEqual(first_plan.metadata["selection"], "global")
        self.assertEqual(first_plan.metadata["total_elements"], 80)
        self.assertEqual(first_plan.metadata["selected_elements"], 20)
        expected = [("conv", list(range(8))), ("linear", list(range(12)))]
        self.assertEqual([(group.primary.name, group.indices) for group in first_plan.groups], expected)
        self.assertEqual(
            [(group.primary.name, group.indices) for group in second_plan.groups], expected
        )
        self.assertTrue(pruner.validate_plan(first_plan, self.adapter))
        pruner.apply_plan(first_plan, self.adapter)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, weight_decay=0.1)
        MaskManager.attach_optimizer(self.model, optimizer)
        self.model(self.images).sum().backward()
        optimizer.step()
        self.assertEqual(sum(int((MaskManager.mask(module) == 0).sum()) for module in (self.model.conv, self.model.linear)), 20)
        for module in (self.model.conv, self.model.linear):
            mask = MaskManager.mask(module)
            self.assertTrue(torch.equal(MaskManager.original_weight(module)[mask == 0], torch.zeros_like(MaskManager.original_weight(module)[mask == 0])))
        self.assertEqual(tuple(self.model(self.images).shape), (2, 4))

    def test_yolo_pipeline_uses_higher_order_loss_and_rtdetr_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            config = FrameworkConfig.from_dict(
                {
                    "model": {"name": "yolov5", "weights": "unused.pt", "device": "cpu"},
                    "pruning": {
                        "method": "unstructured", "criterion": "grasp", "structure": "weight",
                        "target_ratio": 0.25, "calibration_batches": 1, "calibration_batch_size": 1,
                        "calibration_seed": 9,
                    },
                    "benchmark": {"enabled": False, "latency": False, "flops": False, "params": False, "runs": 1},
                    "export": {"enabled": False},
                    "experiment": {"name": "grasp", "output_dir": directory, "seed": 9},
                    "output_path": str(Path(directory) / "model.pt"),
                }
            )
            calibration_batch = torch.randn(1, 3, 16, 16)
            with patch("prune_framework.pipelines.unified.ModelLoader.load", return_value=(TinyDetector(), None)), patch.object(
                YOLOv5Adapter, "get_gradient_calibration_batches", return_value=[calibration_batch]
            ), patch.object(
                YOLOv5Adapter, "build_gradient_calibration_loss", return_value=lambda model, batch: model(batch).square().mean()
            ):
                result = UnifiedPruningPipeline(config).run()
            self.assertTrue(result.pruning.forward_verified)
            self.assertIn("grasp_calibration", result.artifacts)

        unsupported = FrameworkConfig.from_dict(
            {
                "model": {"name": "rtdetr", "weights": "unused", "device": "cpu"},
                "pruning": {"method": "unstructured", "criterion": "grasp", "structure": "weight", "target_ratio": 0.25},
                "benchmark": {"enabled": False, "latency": False, "flops": False, "params": False, "runs": 1},
            }
        )
        with patch("prune_framework.pipelines.unified.ModelLoader.load", return_value=(TinyDetector(), None)):
            with self.assertRaisesRegex(ConfigValidationException, "integrated task loss for higher-order"):
                UnifiedPruningPipeline(unsupported).run()


if __name__ == "__main__":
    unittest.main()
