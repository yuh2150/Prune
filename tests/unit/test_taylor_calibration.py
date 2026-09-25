import unittest

import torch
import torch.nn as nn

from prune_framework.contracts import PrunableTarget, TargetType
from prune_framework.modules.calibration import GradientCalibrationRunner
from prune_framework.plugins.adapters.yolov5 import YOLOv5Adapter
from prune_framework.plugins.criteria.taylor_criteria import TaylorFirstOrderCriterion
from prune_framework.plugins.granularities.base import ChannelGranularity
from prune_framework.plugins.pruners.structured import StructuredPruner


class TinyCalibrationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 3, kernel_size=2, bias=False)

    def forward(self, images):
        return self.conv(images)


class StatefulCalibrationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 3, kernel_size=2, bias=False)
        self.bn = nn.BatchNorm2d(3)

    def forward(self, images):
        return self.bn(self.conv(images))


class TestTaylorCalibration(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.model = TinyCalibrationModel()
        self.target = PrunableTarget("conv", self.model.conv, TargetType.CONV_OUT_CHANNEL)
        self.images = torch.randn(2, 1, 4, 4)

    @staticmethod
    def loss_fn(model, batch):
        return model(batch).square().mean()

    def test_runner_matches_manual_taylor_score_and_restores_state(self):
        self.model.eval()
        prior_grad = torch.full_like(self.model.conv.weight, 3.0)
        self.model.conv.weight.grad = prior_grad.clone()

        self.model.zero_grad(set_to_none=True)
        manual_loss = self.loss_fn(self.model, self.images)
        manual_loss.backward()
        manual_grad = self.model.conv.weight.grad.detach().clone()
        expected_score = (self.model.conv.weight.detach() * manual_grad).abs().sum(dim=(1, 2, 3))
        self.model.conv.weight.grad = prior_grad.clone()

        result = GradientCalibrationRunner(seed=5, accumulate=True).run(
            self.model, [self.target], [self.images], self.loss_fn
        )
        actual_score = TaylorFirstOrderCriterion().score(self.model.conv, result.context_for(self.target))

        self.assertTrue(torch.allclose(actual_score, expected_score))
        self.assertTrue(torch.equal(self.model.conv.weight.grad, prior_grad))
        self.assertFalse(self.model.training)
        self.assertFalse(result.gradients["conv"].requires_grad)
        self.assertIsNone(result.gradients["conv"].grad_fn)
        self.assertEqual(result.batches, 1)

    def test_runner_is_deterministic_and_averages_multiple_batches(self):
        batches = [self.images, self.images * 0.5]
        runner = GradientCalibrationRunner(seed=17, accumulate=True)
        first = runner.run(self.model, [self.target], batches, self.loss_fn)
        second = runner.run(self.model, [self.target], batches, self.loss_fn)
        independent = GradientCalibrationRunner(seed=17, accumulate=False).run(
            self.model, [self.target], batches, self.loss_fn
        )

        self.assertTrue(torch.equal(first.gradients["conv"], second.gradients["conv"]))
        self.assertTrue(torch.allclose(first.gradients["conv"], independent.gradients["conv"]))

    def test_runner_restores_buffers_and_mixed_module_modes(self):
        model = StatefulCalibrationModel()
        model.train()
        model.bn.eval()  # a mixed state that model.train() would otherwise overwrite
        target = PrunableTarget("conv", model.conv, TargetType.CONV_OUT_CHANNEL)
        running_mean = model.bn.running_mean.detach().clone()
        running_var = model.bn.running_var.detach().clone()
        batch_counter = model.bn.num_batches_tracked.detach().clone()
        original_grad = torch.full_like(model.conv.weight, 2.0)
        model.conv.weight.requires_grad_(False)
        model.conv.weight.grad = original_grad.clone()

        GradientCalibrationRunner(seed=3).run(
            model, [target], [torch.randn(2, 1, 4, 4)], lambda candidate, batch: candidate(batch).mean()
        )

        self.assertTrue(model.training)
        self.assertFalse(model.bn.training)
        self.assertTrue(torch.equal(model.bn.running_mean, running_mean))
        self.assertTrue(torch.equal(model.bn.running_var, running_var))
        self.assertTrue(torch.equal(model.bn.num_batches_tracked, batch_counter))
        self.assertTrue(torch.equal(model.conv.weight.grad, original_grad))
        self.assertFalse(model.conv.weight.requires_grad)

    def test_taylor_plan_validates_then_prunes_yolo_adapter_forward_fixture(self):
        torch.manual_seed(13)
        model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(), nn.Conv2d(8, 8, 3))
        adapter = YOLOv5Adapter(model)
        targets = adapter.get_prunable_targets({TargetType.CONV_OUT_CHANNEL})
        images = torch.randn(1, 3, 32, 32)
        result = GradientCalibrationRunner(seed=13).run(
            model,
            targets,
            [images],
            lambda candidate, batch: candidate(batch).square().mean(),
        )

        pruner = StructuredPruner()
        criterion = TaylorFirstOrderCriterion()
        plan = pruner.create_plan(
            adapter,
            criterion,
            ChannelGranularity(),
            {"amount": 0.25, "gradient_calibration": result},
        )
        before = [module.out_channels for _, module in adapter.get_pruneable_modules()]
        self.assertTrue(plan.groups)
        self.assertTrue(pruner.validate_plan(plan, adapter))
        self.assertEqual(before, [module.out_channels for _, module in adapter.get_pruneable_modules()])
        self.assertTrue(all(group.validated for group in plan.groups))

        pruner.apply_plan(plan, adapter)
        with torch.no_grad():
            output = model(adapter.get_dummy_input(torch.device("cpu")))
        self.assertIsNotNone(output)
        self.assertLess(model[0].out_channels, 8)


if __name__ == "__main__":
    unittest.main()
