import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from prune_framework.contracts import BaseModelAdapter, TargetType
from prune_framework.core.config import FrameworkConfig
from prune_framework.core.exceptions import ConfigValidationException
from prune_framework.modules.calibration import CalibrationResult, GradientCalibrationRunner
from prune_framework.modules.model.masks import MaskManager
from prune_framework.pipelines.unified import UnifiedPruningPipeline
from prune_framework.plugins.adapters.yolov5 import YOLOv5Adapter
from prune_framework.plugins.criteria.snip_criteria import SNIPCriterion
from prune_framework.plugins.granularities.base import WeightGranularity
from prune_framework.plugins.pruners.unstructured import UnstructuredPruner


class ConvLinearFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=2, bias=False)
        self.linear = nn.Linear(18, 4, bias=False)

    def forward(self, images):
        return self.linear(self.conv(images).flatten(1))


class FixtureAdapter(BaseModelAdapter):
    @classmethod
    def load_model(cls, weights_path, device):
        raise NotImplementedError

    def get_pruneable_modules(self):
        return [("conv", self.model.conv), ("linear", self.model.linear)]

    def get_pruneable_blocks(self):
        return []

    def get_dummy_input(self, device):
        return torch.randn(2, 1, 4, 4, device=device)

    def supported_target_types(self):
        return {TargetType.CONV_WEIGHT, TargetType.CONV_OUT_CHANNEL, TargetType.LINEAR_WEIGHT, TargetType.LINEAR_OUT_FEATURE}


class TinyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 12, 3, padding=1))

    def forward(self, images):
        return self.features(images)


class TestSNIPPruning(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.model = ConvLinearFixture()
        self.adapter = FixtureAdapter(self.model)
        self.targets = self.adapter.get_prunable_targets({TargetType.CONV_WEIGHT, TargetType.LINEAR_WEIGHT})
        self.images = torch.randn(2, 1, 4, 4)

    @staticmethod
    def loss_fn(model, batch):
        return model(batch).square().mean()

    def _calibrate(self):
        return GradientCalibrationRunner(seed=23, accumulate=True).run(
            self.model, self.targets, [self.images], self.loss_fn
        )

    def test_snip_matches_elementwise_reference_for_conv_and_linear(self):
        self.model.eval()
        prior_state = {name: value.detach().clone() for name, value in self.model.state_dict().items()}
        prior_rng = torch.random.get_rng_state()
        prior_grad = torch.full_like(self.model.conv.weight, 5.0)
        self.model.conv.weight.grad = prior_grad.clone()

        result = self._calibrate()
        criterion = SNIPCriterion()
        for target in self.targets:
            score = criterion.score(target.module, result.context_for(target))
            expected = (target.module.weight.detach() * result.gradients[target.name]).abs()
            self.assertTrue(torch.equal(score, expected))
            self.assertEqual(tuple(score.shape), tuple(target.module.weight.shape))
            self.assertFalse(score.requires_grad)
            self.assertIsNone(score.grad_fn)

        self.assertFalse(self.model.training)
        self.assertTrue(torch.equal(self.model.conv.weight.grad, prior_grad))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), prior_rng))
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, prior_state[name]))

    def test_calibration_and_scores_are_deterministic(self):
        first = self._calibrate()
        second = self._calibrate()
        criterion = SNIPCriterion()
        for target in self.targets:
            first_score = criterion.score(target.module, first.context_for(target))
            second_score = criterion.score(target.module, second.context_for(target))
            self.assertTrue(torch.equal(first.gradients[target.name], second.gradients[target.name]))
            self.assertTrue(torch.equal(first_score, second_score))

    def test_global_selection_is_exact_and_ties_use_target_then_flat_index_order(self):
        with torch.no_grad():
            self.model.conv.weight.fill_(1)
            self.model.linear.weight.fill_(1)
        gradients = {target.name: torch.ones_like(target.module.weight) for target in self.targets}
        calibration = CalibrationResult(gradients=gradients, batches=1)
        pruner = UnstructuredPruner()
        plan = pruner.create_plan(
            self.adapter, SNIPCriterion(), WeightGranularity(), {"amount": 0.25, "gradient_calibration": calibration, "criterion_name": "snip"}
        )

        self.assertEqual(plan.metadata["selection"], "global")
        self.assertEqual(plan.metadata["total_elements"], 80)
        self.assertEqual(plan.metadata["selected_elements"], 20)
        self.assertEqual([(group.primary.name, group.indices) for group in plan.groups], [("conv", list(range(8))), ("linear", list(range(12)))])
        self.assertFalse(MaskManager.has_mask(self.model.conv))
        self.assertTrue(pruner.validate_plan(plan, self.adapter))
        pruner.apply_plan(plan, self.adapter)
        self.assertEqual(int((MaskManager.mask(self.model.conv) == 0).sum()), 8)
        self.assertEqual(int((MaskManager.mask(self.model.linear) == 0).sum()), 12)

    def test_snip_masks_survive_optimizer_and_round_trip(self):
        pruner = UnstructuredPruner()
        plan = pruner.create_plan(
            self.adapter, SNIPCriterion(), WeightGranularity(), {"amount": 0.25, "gradient_calibration": self._calibrate(), "criterion_name": "snip"}
        )
        self.assertTrue(pruner.validate_plan(plan, self.adapter))
        pruner.apply_plan(plan, self.adapter)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, weight_decay=0.1)
        MaskManager.attach_optimizer(self.model, optimizer)
        self.model(self.images).sum().backward()
        optimizer.step()
        for module in (self.model.conv, self.model.linear):
            mask = MaskManager.mask(module)
            self.assertTrue(torch.equal(MaskManager.original_weight(module)[mask == 0], torch.zeros_like(MaskManager.original_weight(module)[mask == 0])))

        with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
            torch.save({"model": self.model.state_dict(), "masks": MaskManager.state_dict(self.model)}, handle.name)
            checkpoint = torch.load(handle.name, weights_only=True)
            restored = ConvLinearFixture()
            MaskManager.load_state_dict(restored, checkpoint["masks"])
            restored.load_state_dict(checkpoint["model"])
        for original, reloaded in ((self.model.conv, restored.conv), (self.model.linear, restored.linear)):
            self.assertTrue(torch.equal(MaskManager.mask(original), MaskManager.mask(reloaded)))
        self.assertTrue(torch.equal(self.model(self.images), restored(self.images)))

    def test_yolo_adapter_masked_forward_works(self):
        model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(), nn.Conv2d(8, 8, 3))
        adapter = YOLOv5Adapter(model)
        targets = adapter.get_prunable_targets({TargetType.CONV_WEIGHT, TargetType.LINEAR_WEIGHT})
        images = torch.randn(1, 3, 32, 32)
        calibration = GradientCalibrationRunner(seed=7).run(
            model, targets, [images], lambda candidate, batch: candidate(batch).square().mean()
        )
        pruner = UnstructuredPruner()
        plan = pruner.create_plan(adapter, SNIPCriterion(), WeightGranularity(), {"amount": 0.25, "gradient_calibration": calibration, "criterion_name": "snip"})
        self.assertTrue(pruner.validate_plan(plan, adapter))
        pruner.apply_plan(plan, adapter)
        self.assertIsNotNone(model(images))

    def test_pipeline_calibrates_snip_and_rtdetr_remains_explicitly_unsupported(self):
        with tempfile.TemporaryDirectory() as directory:
            config = FrameworkConfig.from_dict(
                {
                    "model": {"name": "yolov5", "weights": "unused.pt", "device": "cpu"},
                    "pruning": {"method": "unstructured", "criterion": "snip", "structure": "weight", "target_ratio": 0.25, "calibration_batches": 1, "calibration_seed": 9},
                    "benchmark": {"enabled": False, "latency": False, "flops": False, "params": False, "runs": 1},
                    "export": {"enabled": False},
                    "experiment": {"name": "snip", "output_dir": directory, "seed": 9},
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
            self.assertIn("calibration", result.artifacts)

        unsupported = FrameworkConfig.from_dict(
            {
                "model": {"name": "rtdetr", "weights": "unused", "device": "cpu"},
                "pruning": {"method": "unstructured", "criterion": "snip", "structure": "weight", "target_ratio": 0.25},
                "benchmark": {"enabled": False, "latency": False, "flops": False, "params": False, "runs": 1},
            }
        )
        with patch("prune_framework.pipelines.unified.ModelLoader.load", return_value=(TinyDetector(), None)):
            with self.assertRaisesRegex(ConfigValidationException, "does not provide an integrated task loss"):
                UnifiedPruningPipeline(unsupported).run()


if __name__ == "__main__":
    unittest.main()
