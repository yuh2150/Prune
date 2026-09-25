"""Adapter-defined structural layer/block pruning tests."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from models.common import C3
from prune_framework.contracts import BaseModelAdapter, PruningGroup, PruningPlan, StructuralBlockTarget
from prune_framework.modules.evaluation.complexity import measure_complexity
from prune_framework.modules.evaluation.validator import ModelValidator
from prune_framework.plugins.adapters.rtdetr import RTDETRAdapter
from prune_framework.plugins.adapters.yolov5 import YOLOv5Adapter
from prune_framework.plugins.criteria.norm_criteria import MagnitudeCriterion
from prune_framework.plugins.pruners.depth import DepthPruner


class ToyYOLO(nn.Module):
    def __init__(self, bottlenecks: int = 2):
        super().__init__()
        self.model = nn.Sequential(nn.Conv2d(3, 8, 1), C3(8, 8, n=bottlenecks), nn.Conv2d(8, 8, 1))

    def forward(self, images):
        return self.model(images)


class ToyYOLOAdapter(YOLOv5Adapter):
    def get_dummy_input(self, device):
        return torch.randn(1, 3, 32, 32, device=device)


class TransformerStack(nn.Module):
    def __init__(self, count):
        super().__init__()
        self.layers = nn.ModuleList([nn.Conv2d(3, 3, 1) for _ in range(count)])

    def forward(self, images):
        for layer in self.layers:
            images = layer(images)
        return images


class TinyRTDETRCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TransformerStack(2)
        self.decoder = TransformerStack(2)

    def forward(self, images):
        return self.decoder(self.encoder(images))


class TinyRTDETR(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyRTDETRCore()
        self.num_encoder_layers = 2
        self.num_decoder_layers = 2
        self.config = SimpleNamespace(encoder_layers=2, decoder_layers=2)

    def forward(self, images):
        return self.model(images)


class BrokenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Conv2d(3, 3, 1)

    def forward(self, images):
        if not isinstance(self.block, nn.Conv2d):
            raise RuntimeError("required block was removed")
        return self.block(images)


class BrokenAdapter(BaseModelAdapter):
    @classmethod
    def load_model(cls, weights_path, device):
        raise NotImplementedError

    def get_pruneable_modules(self):
        return []

    def get_pruneable_blocks(self):
        return []

    def get_dummy_input(self, device):
        return torch.randn(1, 3, 8, 8, device=device)

    def get_structural_block_targets(self):
        return [StructuralBlockTarget("block", self.model.block, "broken", "", 0)]

    def validate_structural_block_plan(self, targets):
        return {}

    def remove_structural_block(self, target):
        self.model.block = nn.Identity()


class TestStructuralBlockPruning(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.model = ToyYOLO(bottlenecks=2)
        self.adapter = ToyYOLOAdapter(self.model)
        self.pruner = DepthPruner()
        self.criterion = MagnitudeCriterion()

    def test_yolo_block_selection_is_deterministic_and_prunes_one_safe_bottleneck(self):
        first = self.pruner.create_plan(self.adapter, self.criterion, config={"amount": 0.5})
        second = self.pruner.create_plan(self.adapter, self.criterion, config={"amount": 0.5})
        self.assertEqual(first.target_names().__next__(), second.target_names().__next__())
        self.assertEqual(len(first.groups), 1)
        before_parameters = sum(parameter.numel() for parameter in self.model.parameters())
        before_cost = measure_complexity(self.model, self.adapter.get_dummy_input(torch.device("cpu")))

        self.assertTrue(self.pruner.validate_plan(first, self.adapter))
        self.pruner.apply_plan(first, self.adapter)
        after_parameters = sum(parameter.numel() for parameter in self.model.parameters())
        after_cost = measure_complexity(self.model, self.adapter.get_dummy_input(torch.device("cpu")))
        self.assertEqual(len(self.model.model[1].m), 1)
        self.assertLess(after_parameters, before_parameters)
        self.assertIsNotNone(before_cost.flops)
        self.assertIsNotNone(after_cost.flops)
        self.assertLess(after_cost.flops, before_cost.flops)
        self.assertTrue(ModelValidator.validate_forward(self.model, self.adapter.get_dummy_input(torch.device("cpu"))))

    def test_yolo_pruned_block_model_saves_reloads_and_forwards(self):
        plan = self.pruner.create_plan(self.adapter, self.criterion, config={"amount": 0.5})
        self.assertTrue(self.pruner.validate_plan(plan, self.adapter))
        self.pruner.apply_plan(plan, self.adapter)
        images = self.adapter.get_dummy_input(torch.device("cpu"))
        with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
            torch.save(self.model, handle.name)
            try:
                restored = torch.load(handle.name, weights_only=False)
            except TypeError:
                restored = torch.load(handle.name)
        self.assertTrue(torch.allclose(self.model(images), restored(images)))
        self.assertTrue(ModelValidator.validate_checkpoint_reload(self.model, images))

    def test_non_removable_yolo_block_is_rejected_before_mutation(self):
        model = ToyYOLO(bottlenecks=1)
        adapter = ToyYOLOAdapter(model)
        target = StructuralBlockTarget(
            "model.1.m.0", model.model[1].m[0], "yolo_csp_bottleneck", "model.1.m", 0
        )
        plan = PruningPlan(pruner_name="structural_block", groups=[PruningGroup(primary=target, operation="remove_structural_block")])
        state_before = copy.deepcopy(model.state_dict())

        self.assertFalse(self.pruner.validate_plan(plan, adapter))
        self.assertFalse(plan.groups[0].validated)
        self.assertEqual(len(model.model[1].m), 1)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, state_before[name]))

    def test_architecture_dry_run_failure_leaves_original_model_unchanged(self):
        model = BrokenModel()
        adapter = BrokenAdapter(model)
        plan = self.pruner.create_plan(adapter, self.criterion, config={"block_names": ["block"]})
        self.assertFalse(self.pruner.validate_plan(plan, adapter))
        self.assertIn("dry-run failed", plan.groups[0].validation_error)
        self.assertIsInstance(model.block, nn.Conv2d)

    def test_rtdetr_adapter_declares_modulelist_layers_and_repairs_counts(self):
        model = TinyRTDETR()
        adapter = RTDETRAdapter(model)
        targets = adapter.get_structural_block_targets()
        self.assertEqual(len(targets), 4)
        pruner = DepthPruner()
        plan = pruner.create_plan(adapter, self.criterion, config={"block_names": ["model.encoder.layers.0"]})
        self.assertTrue(pruner.validate_plan(plan, adapter))
        pruner.apply_plan(plan, adapter)
        self.assertEqual(len(model.model.encoder.layers), 1)
        self.assertEqual(model.num_encoder_layers, 1)
        self.assertEqual(model.config.encoder_layers, 1)
        self.assertTrue(ModelValidator.validate_forward(model, adapter.get_dummy_input(torch.device("cpu"))))


if __name__ == "__main__":
    unittest.main()
