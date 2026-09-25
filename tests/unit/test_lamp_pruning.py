import tempfile
import unittest

import torch
import torch.nn as nn

from prune_framework.contracts import BaseModelAdapter, TargetType
from prune_framework.modules.model.masks import MaskManager
from prune_framework.plugins.criteria.lamp_criteria import LAMPCriterion
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
        return torch.ones(2, 1, 4, 4, device=device)

    def supported_target_types(self):
        return {TargetType.CONV_WEIGHT, TargetType.CONV_OUT_CHANNEL, TargetType.LINEAR_WEIGHT, TargetType.LINEAR_OUT_FEATURE}


class TestLAMPPruning(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.model = ConvLinearFixture()
        self.adapter = FixtureAdapter(self.model)
        self.criterion = LAMPCriterion()

    def test_scores_match_hand_calculated_elementwise_reference_for_conv_and_linear(self):
        conv = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
        linear = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            conv.weight.copy_(torch.tensor([[[[1.0, -3.0]]]]))
            linear.weight.copy_(torch.tensor([[1.0, -2.0], [3.0, -4.0]]))

        conv_expected = torch.tensor([[[[1.0 / (10.0 ** 0.5), 1.0]]]])
        linear_expected = torch.tensor([
            [1.0 / (30.0 ** 0.5), 2.0 / (29.0 ** 0.5)],
            [3.0 / 5.0, 1.0],
        ])
        self.assertTrue(torch.allclose(self.criterion.score(conv), conv_expected))
        self.assertTrue(torch.allclose(self.criterion.score(linear), linear_expected))

    def test_equal_magnitudes_share_scores_and_global_plan_is_deterministic(self):
        with torch.no_grad():
            self.model.conv.weight.fill_(1.0)
            self.model.linear.weight.fill_(1.0)
        conv_scores = self.criterion.score(self.model.conv)
        linear_scores = self.criterion.score(self.model.linear)
        self.assertTrue(torch.allclose(conv_scores, torch.full_like(conv_scores, 1.0 / (8.0 ** 0.5))))
        self.assertTrue(torch.allclose(linear_scores, torch.full_like(linear_scores, 1.0 / (72.0 ** 0.5))))

        pruner = UnstructuredPruner()
        config = {"amount": 0.25, "criterion_name": "lamp"}
        first = pruner.create_plan(self.adapter, self.criterion, WeightGranularity(), config)
        second = pruner.create_plan(self.adapter, self.criterion, WeightGranularity(), config)
        self.assertEqual(first.describe(), second.describe())
        self.assertEqual(first.metadata["selection"], "global")
        self.assertEqual(first.metadata["total_elements"], 80)
        self.assertEqual(first.metadata["selected_elements"], 20)
        self.assertEqual(sum(len(group.indices) for group in first.groups), 20)
        # Linear scores are lower than Conv scores; equal Linear scores are
        # resolved by the selector's stable flattened-index tie-break.
        self.assertEqual([(group.primary.name, group.indices) for group in first.groups], [("linear", list(range(20)))])

    def test_masks_survive_optimizer_reload_and_forward(self):
        pruner = UnstructuredPruner()
        plan = pruner.create_plan(
            self.adapter, self.criterion, WeightGranularity(), {"amount": 0.25, "criterion_name": "lamp"}
        )
        self.assertTrue(pruner.validate_plan(plan, self.adapter))
        pruner.apply_plan(plan, self.adapter)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, weight_decay=0.1)
        MaskManager.attach_optimizer(self.model, optimizer)
        images = self.adapter.get_dummy_input(torch.device("cpu"))
        self.model(images).sum().backward()
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
        self.assertTrue(torch.equal(self.model(images), restored(images)))


if __name__ == "__main__":
    unittest.main()
