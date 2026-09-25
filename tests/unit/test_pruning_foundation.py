import tempfile
import unittest

import torch
import torch.nn as nn

from prune_framework.contracts import BaseModelAdapter, TargetType
from prune_framework.modules.model.masks import MaskManager
from prune_framework.plugins.adapters.rtdetr import RTDETRAdapter
from prune_framework.plugins.adapters.yolov5 import YOLOv5Adapter
from prune_framework.plugins.criteria.norm_criteria import MagnitudeCriterion
from prune_framework.plugins.granularities.base import WeightGranularity
from prune_framework.plugins.pruners.unstructured import UnstructuredPruner
from prune_framework.core.engine import PruningEngine


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
        return {
            TargetType.CONV_WEIGHT,
            TargetType.CONV_OUT_CHANNEL,
            TargetType.LINEAR_WEIGHT,
            TargetType.LINEAR_OUT_FEATURE,
        }


class TestP0PruningFoundation(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = ConvLinearFixture()
        self.adapter = FixtureAdapter(self.model)

    def test_typed_targets_and_adapter_capabilities(self):
        targets = self.adapter.get_prunable_targets()
        self.assertEqual(
            {target.target_type for target in targets},
            {
                TargetType.CONV_WEIGHT,
                TargetType.CONV_OUT_CHANNEL,
                TargetType.LINEAR_WEIGHT,
                TargetType.LINEAR_OUT_FEATURE,
            },
        )
        yolo_targets = YOLOv5Adapter(self.model).get_prunable_targets(
            {TargetType.CONV_WEIGHT, TargetType.LINEAR_WEIGHT}
        )
        self.assertEqual({target.target_type for target in yolo_targets}, {TargetType.CONV_WEIGHT, TargetType.LINEAR_WEIGHT})
        self.assertEqual(
            RTDETRAdapter(self.model).supported_target_types(),
            {TargetType.CONV_WEIGHT, TargetType.CONV_OUT_CHANNEL},
        )

    def test_magnitude_plan_has_exact_conv_and_linear_mask_ratios(self):
        pruner = UnstructuredPruner()
        plan = pruner.create_plan(self.adapter, MagnitudeCriterion(), WeightGranularity(), {"amount": 0.25, "criterion_name": "magnitude"})
        self.assertEqual([group.primary.name for group in plan.groups], ["conv", "linear"])
        self.assertEqual([len(group.indices) for group in plan.groups], [2, 18])
        self.assertTrue(pruner.validate_plan(plan, self.adapter))
        self.assertTrue(all(group.validated for group in plan.groups))
        self.assertEqual(plan.describe()["groups"][0]["operation"], "mask_weight")

        pruner.apply_plan(plan, self.adapter)
        self.assertEqual(int((MaskManager.mask(self.model.conv) == 0).sum()), 2)
        self.assertEqual(int((MaskManager.mask(self.model.linear) == 0).sum()), 18)
        self.assertEqual(int((self.model.conv.weight == 0).sum()), 2)
        self.assertEqual(int((self.model.linear.weight == 0).sum()), 18)
        self.assertEqual(tuple(self.model(self.adapter.get_dummy_input(torch.device("cpu"))).shape), (2, 4))

    def test_masks_survive_optimizer_and_state_round_trip(self):
        pruner = UnstructuredPruner()
        pruner.prune(self.adapter, MagnitudeCriterion(), WeightGranularity(), {"amount": 0.25, "criterion_name": "magnitude"})
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, weight_decay=0.1)
        MaskManager.attach_optimizer(self.model, optimizer)
        self.model(self.adapter.get_dummy_input(torch.device("cpu"))).sum().backward()
        optimizer.step()

        for module in (self.model.conv, self.model.linear):
            mask = MaskManager.mask(module)
            self.assertTrue(torch.equal(MaskManager.original_weight(module)[mask == 0], torch.zeros_like(MaskManager.original_weight(module)[mask == 0])))
            self.assertTrue(torch.equal(module.weight[mask == 0], torch.zeros_like(module.weight[mask == 0])))

        with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
            torch.save({"model": self.model.state_dict(), "masks": MaskManager.state_dict(self.model)}, handle.name)
            checkpoint = torch.load(handle.name, weights_only=True)
            restored = ConvLinearFixture()
            MaskManager.load_state_dict(restored, checkpoint["masks"])
            restored.load_state_dict(checkpoint["model"])

        self.assertTrue(torch.equal(MaskManager.mask(self.model.conv), MaskManager.mask(restored.conv)))
        self.assertTrue(torch.equal(MaskManager.mask(self.model.linear), MaskManager.mask(restored.linear)))
        images = self.adapter.get_dummy_input(torch.device("cpu"))
        self.assertTrue(torch.equal(self.model(images), restored(images)))

    def test_structured_engine_records_validated_plan_and_architecture_checks(self):
        model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(), nn.Conv2d(8, 8, 3))
        result = PruningEngine("yolov5", "structured", "l1", "channel").execute(model, {"amount": 0.25})
        self.assertTrue(result.forward_verified)
        self.assertTrue(result.architecture_validation["checkpoint_verified"])
        self.assertLessEqual(result.architecture_validation["params_after"], result.architecture_validation["params_before"])
        self.assertTrue(result.pruning_plan["groups"])
        self.assertTrue(all(group["validated"] for group in result.pruning_plan["groups"]))


if __name__ == "__main__":
    unittest.main()
