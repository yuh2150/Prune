"""Focused GraSP tests for the shared unstructured pruning path."""

from __future__ import annotations

import copy
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import torch
import torch.nn as nn

from prune_framework.contracts import BaseModelAdapter, PrunableTarget, TargetType
from prune_framework.core.config import FrameworkConfig
from prune_framework.core.exceptions import ConfigValidationException
from prune_framework.modules.calibration import HigherOrderCalibrationResult, HigherOrderCalibrationRunner
from prune_framework.modules.model.masks import MaskManager
from prune_framework.pipelines.unified import UnifiedPruningPipeline
from prune_framework.plugins.adapters.yolov5 import YOLOv5Adapter
from prune_framework.plugins.criteria.grasp_criteria import (
    GraSPConv1CappedCriterion,
    GraSPCriterion,
    GraSPLayerNormalizedCriterion,
    GraSPMagnitudeGuardedCriterion,
)
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

    def test_two_pass_weight_hvp_matches_joint_loss_with_dropout_and_biases(self):
        model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Dropout(.3), nn.Linear(4, 2))
        reference = copy.deepcopy(model)
        batches = [torch.randn(3, 3), torch.randn(2, 3)]
        targets = [PrunableTarget(str(i), model[i], TargetType.LINEAR_WEIGHT) for i in (0, 3)]
        parameters = [reference[0].weight, reference[3].weight]
        torch.manual_seed(17)
        loss = sum(reference(x).square().mean() for x in batches) / len(batches)
        g = torch.autograd.grad(loss, parameters, create_graph=True)
        expected = torch.autograd.grad(sum(v.square().sum() / 2 for v in g), parameters)
        result = HigherOrderCalibrationRunner(seed=17, parameter_scope="weights").run(
            model, targets, batches, lambda candidate, x: candidate(x).square().mean()
        )
        for target, hvp in zip(targets, expected):
            self.assertTrue(torch.allclose(result.gradients[target.name], hvp, atol=1e-6))
        self.assertEqual(result.describe()["algorithm"], "two_pass_hvp")
        self.assertEqual(result.describe()["parameter_scope"], "weights")

    def test_classification_reference_temperature_and_microbatches(self):
        from experiments.classification import calibrate
        images = torch.randn(45, 3)
        labels = torch.arange(45) % 2
        config = SimpleNamespace(pruning=SimpleNamespace(calibration_batches=1, calibration_seed=17))
        context = calibrate(nn.Linear(3, 2), config, "cpu", dataloader=[(images, labels)],
                            temperature=200, microbatch_size=20)
        self.assertEqual([len(batch[1]) for batch in context.batches], [20, 20, 5])
        self.assertEqual(context.sample_count, 45)
        model = nn.Linear(3, 2)
        expected = nn.CrossEntropyLoss()(model(images[:20]) / 200, labels[:20])
        self.assertTrue(torch.allclose(context.loss_fn(model, context.batches[0]), expected))
        with self.assertRaisesRegex(ValueError, "temperature"):
            calibrate(model, config, "cpu", dataloader=[(images, labels)], temperature=0)

    def test_two_pass_hvp_handles_constant_first_derivatives(self):
        result = HigherOrderCalibrationRunner().run(
            self.model, self.targets, [self.images], lambda model, batch: sum(p.sum() for p in model.parameters())
        )
        for value in result.gradients.values():
            self.assertTrue(torch.equal(value, torch.zeros_like(value)))

    def test_hvp_matches_explicit_hessian_matrix(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.Tanh(), nn.Linear(2, 1)).double()
        targets = [PrunableTarget(str(i), model[i], TargetType.LINEAR_WEIGHT) for i in (0, 2)]
        x = torch.randn(3, 2, dtype=torch.float64)
        vector = torch.cat([model[0].weight.flatten(), model[2].weight.flatten()]).detach().requires_grad_()
        def objective(v):
            params = {"0.weight": v[:4].reshape(2, 2), "2.weight": v[4:].reshape(1, 2)}
            return torch.func.functional_call(model, params, (x,)).square().mean()
        hessian = torch.autograd.functional.hessian(objective, vector)
        gradient = torch.autograd.grad(objective(vector), vector)[0]
        result = HigherOrderCalibrationRunner().run(model, targets, [x], lambda m, b: m(b).square().mean())
        actual = torch.cat([result.gradients[t.name].flatten() for t in targets])
        self.assertTrue(torch.allclose(actual, hessian @ gradient, atol=1e-10))
        self.assertEqual(set(result.context_for(targets[0])), {"hvp", "first_order_grad"})

    def test_default_weights_scope_excludes_bias_batchnorm_and_extra_parameter(self):
        model = nn.Sequential(nn.Conv2d(1, 2, 2), nn.BatchNorm2d(2), nn.Flatten(), nn.Linear(8, 2))
        model.register_parameter("extra", nn.Parameter(torch.ones(1)))
        targets = [PrunableTarget("0", model[0], TargetType.CONV_WEIGHT), PrunableTarget("3", model[3], TargetType.LINEAR_WEIGHT)]
        result = HigherOrderCalibrationRunner().run(model, targets, [torch.randn(3, 1, 3, 3)], lambda m, b: m(b).square().mean())
        self.assertEqual(result.parameter_names, ["0.weight", "3.weight"])
        for tensor in [*result.gradients.values(), *result.first_order_gradients.values()]:
            self.assertFalse(tensor.requires_grad)
            self.assertIsNone(tensor.grad_fn)

    def test_microbatch_invariance_with_short_final_chunk(self):
        from experiments.classification import calibrate
        model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2)).double()
        images = torch.randn(64, 3, dtype=torch.float64)
        labels = torch.arange(64) % 2
        targets = [PrunableTarget(str(i), model[i], TargetType.LINEAR_WEIGHT) for i in (0, 2)]
        config = SimpleNamespace(pruning=SimpleNamespace(calibration_batches=1, calibration_seed=17, parameter_scope="weights"))
        results = []
        for size in (None, 16, 20):
            context = calibrate(model, config, "cpu", dataloader=[(images, labels)], temperature=200, microbatch_size=size)
            if size == 20:
                self.assertEqual([len(b[1]) for b in context.batches], [20, 20, 20, 4])
            results.append(HigherOrderCalibrationRunner(parameter_scope=context.parameter_scope).run(
                model, targets, context.batches, context.loss_fn, batch_weights=context.batch_weights))
        for result in results[1:]:
            for target in targets:
                self.assertTrue(torch.allclose(result.first_order_gradients[target.name], results[0].first_order_gradients[target.name], rtol=1e-8, atol=1e-14))
                self.assertTrue(torch.allclose(result.gradients[target.name], results[0].gradients[target.name], rtol=1e-8, atol=1e-14))

    def test_lenet_protected_head_remains_unchanged_and_global_count_is_exact(self):
        from prune_framework.models.lenet5_emnist import LeNet5EMNIST
        from prune_framework.plugins.adapters.lenet5_emnist_onnx import LeNet5EMNISTONNXAdapter
        model = LeNet5EMNIST(num_classes=47)
        adapter = LeNet5EMNISTONNXAdapter(model)
        targets = adapter.get_prunable_targets({TargetType.CONV_WEIGHT, TargetType.LINEAR_WEIGHT})
        head = model.fc2.weight.detach().clone()
        result = HigherOrderCalibrationRunner().run(model, targets, [torch.randn(2, 1, 32, 32)], lambda m, b: m(b).square().mean())
        self.assertIn("fc2.weight", result.parameter_names)
        pruner = UnstructuredPruner()
        plan = pruner.create_plan(adapter, GraSPCriterion(), WeightGranularity(), {"amount": .1, "criterion_name": "grasp", "higher_order_calibration": result})
        self.assertNotIn("fc2", [g.primary.name for g in plan.groups])
        self.assertEqual(sum(len(g.indices) for g in plan.groups), round(sum(t.module.weight.numel() for t in targets) * .1))
        self.assertTrue(pruner.validate_plan(plan, adapter))
        pruner.apply_plan(plan, adapter)
        self.assertTrue(torch.equal(model.fc2.weight, head))

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
        calibration = HigherOrderCalibrationResult(gradients=zero_hvp, first_order_gradients=zero_hvp, batches=1)
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

    def test_grasp_removes_highest_signed_scores_as_in_author_topk_rule(self):
        with torch.no_grad():
            self.model.conv.weight.fill_(1)
            self.model.linear.weight.fill_(1)
        raw_scores = torch.arange(80, dtype=torch.float32) - 40
        calibration = HigherOrderCalibrationResult(gradients={
            "conv": -raw_scores[:8].reshape_as(self.model.conv.weight),
            "linear": -raw_scores[8:].reshape_as(self.model.linear.weight),
        })
        calibration.first_order_gradients = {name: torch.zeros_like(value) for name, value in calibration.gradients.items()}
        plan = UnstructuredPruner().create_plan(
            self.adapter, GraSPCriterion(), WeightGranularity(),
            {"amount": .25, "criterion_name": "grasp", "higher_order_calibration": calibration},
        )
        self.assertEqual(plan.metadata["score_order"], "highest")
        self.assertEqual([(g.primary.name, g.indices) for g in plan.groups], [("linear", list(range(52, 72)))])

    def test_paper_algorithm2_matches_weighted_joint_loss_including_cross_batches(self):
        model = copy.deepcopy(self.model).double()
        reference = copy.deepcopy(model)
        chunks = [torch.randn(n, 1, 4, 4, dtype=torch.float64) for n in (20, 20, 20, 4)]
        labels = [torch.arange(len(x)) % 4 for x in chunks]
        batches = list(zip(chunks, labels))
        def loss(candidate, batch):
            x, y = batch
            return nn.functional.cross_entropy(candidate(x) / 200, y)
        weights = [reference.conv.weight, reference.linear.weight]
        joint_loss = sum(loss(reference, b) * len(b[0]) / 64 for b in batches)
        g = torch.autograd.grad(joint_loss, weights, create_graph=True)
        # Algorithm 2: detach the vector operand, not both operands.
        expected = torch.autograd.grad(sum((v * v.detach()).sum() for v in g), weights)
        targets = [PrunableTarget("conv", model.conv, TargetType.CONV_WEIGHT),
                   PrunableTarget("linear", model.linear, TargetType.LINEAR_WEIGHT)]
        result = HigherOrderCalibrationRunner().run(model, targets, batches, loss, batch_weights=[20,20,20,4])
        for target, gradient, hvp in zip(targets, g, expected):
            torch.testing.assert_close(result.first_order_gradients[target.name], gradient.detach(), rtol=1e-8, atol=1e-14)
            torch.testing.assert_close(result.gradients[target.name], hvp, rtol=1e-8, atol=1e-14)
        self.assertEqual(result.describe()["batch_weights"], [20/64,20/64,20/64,4/64])

    def test_paper_equation7_signed_score_matches_gradient_flow_derivative(self):
        model = nn.Linear(2, 2, bias=False).double()
        x = torch.tensor([[.3, -.7], [.8, .2]], dtype=torch.float64)
        target = PrunableTarget("linear", model, TargetType.LINEAR_WEIGHT)
        result = HigherOrderCalibrationRunner().run(model, [target], [x], lambda m, b: m(b).square().mean())
        score = GraSPCriterion().score(model, result.context_for(target)).flatten()
        original = model.weight.detach().clone()
        def flow_at(delta, index):
            candidate = copy.deepcopy(model)
            with torch.no_grad():
                candidate.weight.copy_(original)
                candidate.weight.view(-1)[index] *= 1 - delta
            g = torch.autograd.grad(candidate(x).square().mean(), candidate.weight)[0]
            return g.square().sum()
        epsilon = 1e-5
        for index in range(score.numel()):
            derivative = (flow_at(epsilon, index) - flow_at(-epsilon, index)) / (2 * epsilon)
            torch.testing.assert_close(derivative, 2 * score[index], rtol=1e-7, atol=1e-9)

    def test_two_pass_matches_reference_chunk_sum_up_to_positive_scale(self):
        # Reference ImageNet code sums mean-loss gradients and HVPs over chunks.
        # Equal chunks differ from our sample mean by K**2, preserving ranking.
        reference = copy.deepcopy(self.model).double()
        model = copy.deepcopy(reference)
        batches = [torch.randn(4, 1, 4, 4, dtype=torch.float64) for _ in range(3)]
        labels = [torch.arange(4) for _ in batches]
        weights = [reference.conv.weight, reference.linear.weight]
        def loss(candidate, batch):
            x, y = batch
            return nn.functional.cross_entropy(candidate(x) / 200, y)
        chunks = list(zip(batches, labels))
        g = [torch.zeros_like(w) for w in weights]
        for batch in chunks:
            gradients = torch.autograd.grad(loss(reference, batch), weights)
            for accumulated, gradient in zip(g, gradients):
                accumulated.add_(gradient.detach())
        hvp = [torch.zeros_like(w) for w in weights]
        for batch in chunks:
            gradients = torch.autograd.grad(loss(reference, batch), weights, create_graph=True)
            products = torch.autograd.grad(sum((v * d).sum() for v, d in zip(g, gradients)), weights)
            for accumulated, product in zip(hvp, products):
                accumulated.add_(product.detach())
        targets = [PrunableTarget("conv", model.conv, TargetType.CONV_WEIGHT),
                   PrunableTarget("linear", model.linear, TargetType.LINEAR_WEIGHT)]
        result = HigherOrderCalibrationRunner().run(model, targets, chunks, loss, batch_weights=[4]*3)
        for target, expected_g, expected_hvp in zip(targets, g, hvp):
            torch.testing.assert_close(result.first_order_gradients[target.name] * 3, expected_g)
            torch.testing.assert_close(result.gradients[target.name] * 9, expected_hvp)
        raw = torch.cat([-(w.detach() * h).flatten() for w, h in zip(weights, hvp)])
        denominator = raw.sum().abs() + 1e-10
        self.assertTrue(torch.equal(torch.argsort(raw, descending=True),
                                    torch.argsort(raw / denominator, descending=True)))

    def test_layer_normalization_preserves_tiny_score_scale_and_zero_targets(self):
        for scale in (1., 1e-10, 1e-20):
            raw = torch.arange(1, 9, dtype=torch.float32).reshape_as(self.model.conv.weight) * scale
            zeros = torch.zeros_like(self.model.linear.weight)
            normalized, statistics = UnstructuredPruner._normalise_global_scores(
                [(self.targets[0], raw), (self.targets[1], zeros)], GraSPLayerNormalizedCriterion())
            torch.testing.assert_close(normalized[0][1], raw / raw.abs().mean())
            self.assertTrue(torch.equal(normalized[1][1], zeros))
            self.assertTrue(torch.isfinite(normalized[1][1]).all())
            self.assertAlmostEqual(statistics[self.targets[0].name]["normalization_scale"] / float(raw.abs().mean()), 1.)
            # Plain GraSP remains raw, without per-layer normalization.
            baseline, baseline_statistics = UnstructuredPruner._normalise_global_scores(
                [(self.targets[0], raw)], GraSPCriterion())
            self.assertTrue(torch.equal(baseline[0][1], raw))
            self.assertEqual(baseline_statistics, {})

    def test_magnitude_guard_uses_grasp_ranking_only_within_small_weight_pool(self):
        with torch.no_grad():
            self.model.conv.weight.copy_(torch.arange(1, 9).reshape_as(self.model.conv.weight))
            self.model.linear.weight.copy_(torch.arange(9, 81).reshape_as(self.model.linear.weight))
        raw_scores = {
            "conv": torch.arange(-8, 0, dtype=torch.float32).reshape_as(self.model.conv.weight),
            "linear": torch.arange(1, 73, dtype=torch.float32).reshape_as(self.model.linear.weight),
        }
        calibration = HigherOrderCalibrationResult(
            gradients={t.name: -raw_scores[t.name] / t.module.weight.detach() for t in self.targets},
            first_order_gradients={t.name: torch.zeros_like(t.module.weight) for t in self.targets},
        )
        before = copy.deepcopy(self.model.state_dict())
        config = {"amount": .25, "higher_order_calibration": calibration}
        pruner = UnstructuredPruner()
        guarded = pruner.create_plan(self.adapter, GraSPMagnitudeGuardedCriterion(), config=config)
        plain = pruner.create_plan(self.adapter, GraSPCriterion(), config=config)
        self.assertEqual(guarded.metadata["selected_elements"], 20)
        self.assertEqual(guarded.metadata["score_order"], "highest")
        self.assertEqual(guarded.metadata["magnitude_guard"]["candidate_elements"], 30)
        self.assertEqual(guarded.metadata["magnitude_guard"]["max_candidate_magnitude"], 30)
        # GraSP removes the top scores inside the pool, not the magnitude prefix.
        self.assertEqual([(g.primary.name, g.indices) for g in guarded.groups],
                         [("linear", list(range(2, 22)))])
        self.assertEqual([(g.primary.name, g.indices) for g in plain.groups],
                         [("linear", list(range(52, 72)))])
        self.assertNotIn("magnitude_guard", plain.metadata)
        self.assertEqual(set(guarded.metadata["score_statistics"]), {"conv", "linear"})
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertTrue(pruner.validate_plan(guarded, self.adapter))
        pruner.apply_plan(guarded, self.adapter)
        for group in guarded.groups:
            self.assertTrue(torch.all(group.primary.module.weight.flatten()[group.indices] == 0))

    def test_magnitude_guard_exact_budget_and_stable_ties_at_boundary_amounts(self):
        with torch.no_grad():
            self.model.conv.weight.fill_(1)
            self.model.linear.weight.fill_(1)
        zeros = {t.name: torch.zeros_like(t.module.weight) for t in self.targets}
        calibration = HigherOrderCalibrationResult(gradients=zeros, first_order_gradients=zeros)
        for amount in (0., .0125, .25, .9):
            with self.subTest(amount=amount):
                plan = UnstructuredPruner().create_plan(
                    self.adapter, GraSPMagnitudeGuardedCriterion(),
                    config={"amount": amount, "higher_order_calibration": calibration})
                count = round(80 * amount)
                selected = [(g.primary.name, i) for g in plan.groups for i in g.indices]
                expected = [("conv", i) for i in range(8)] + [("linear", i) for i in range(72)]
                self.assertEqual(selected, expected[:count])
                self.assertEqual(plan.metadata["selected_elements"], count)
                self.assertEqual(plan.metadata["magnitude_guard"]["candidate_elements"],
                                 round(80 * min(1., amount * 1.5)))

    def test_magnitude_guard_rejects_nonfinite_scores_and_insufficient_capped_pool(self):
        zeros = {t.name: torch.zeros_like(t.module.weight) for t in self.targets}
        calibration = HigherOrderCalibrationResult(gradients=zeros, first_order_gradients=zeros)
        config = {"amount": .25, "higher_order_calibration": calibration}
        criterion = GraSPMagnitudeGuardedCriterion()
        with torch.no_grad():
            self.model.conv.weight.fill_(1)
            self.model.linear.weight.fill_(2)
        criterion.max_pruning_fraction_by_target = {"conv": 0.}
        with self.assertRaisesRegex(ValueError, "fewer candidates"):
            UnstructuredPruner().create_plan(self.adapter, criterion, config={**config, "amount": .05})
        del criterion.max_pruning_fraction_by_target
        calibration.gradients["linear"][0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "finite scores and weights"):
            UnstructuredPruner().create_plan(self.adapter, criterion, config=config)

    def test_layer_normalized_variant_records_raw_and_normalized_score_statistics(self):
        calibration = self._calibrate()
        plan = UnstructuredPruner().create_plan(
            self.adapter,
            GraSPLayerNormalizedCriterion(),
            WeightGranularity(),
            {"amount": 0.10, "criterion_name": "grasp_layer_normalized", "higher_order_calibration": calibration},
        )

        statistics = plan.metadata["score_statistics"]
        self.assertEqual(set(statistics), {"conv", "linear"})
        for details in statistics.values():
            self.assertEqual(details["normalization"], "mean_abs")
            self.assertGreater(details["normalization_scale"], 0.0)
            self.assertAlmostEqual(details["normalized"]["mean_abs"], 1.0, places=6)

    def test_conv1_capped_variant_redistributes_global_budget(self):
        with torch.no_grad():
            self.model.conv.weight.fill_(1.0)
            self.model.linear.weight.fill_(1.0)
        zero_hvp = {target.name: torch.zeros_like(target.module.weight) for target in self.targets}
        calibration = HigherOrderCalibrationResult(gradients=zero_hvp, first_order_gradients=zero_hvp, batches=1)
        criterion = GraSPConv1CappedCriterion()
        criterion.max_pruning_fraction_by_target = {"conv": 0.10}
        plan = UnstructuredPruner().create_plan(
            self.adapter,
            criterion,
            WeightGranularity(),
            {"amount": 0.25, "criterion_name": "grasp_conv1_capped", "higher_order_calibration": calibration},
        )

        selected = {group.primary.name: len(group.indices) for group in plan.groups}
        self.assertEqual(plan.metadata["selected_elements"], 20)
        self.assertLessEqual(selected["conv"], 1)  # round(8 * 10%)
        self.assertEqual(sum(selected.values()), 20)
        self.assertEqual(plan.metadata["max_pruning_fraction_by_target"], {"conv": 0.10})

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
