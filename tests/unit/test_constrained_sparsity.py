"""CPU contracts for N:M and rectangular block-sparse masks."""

from __future__ import annotations

import tempfile
import unittest

import torch
import torch.nn as nn

from prune_framework.contracts import BaseModelAdapter, TargetType
from prune_framework.core.engine import PruningEngine
from prune_framework.modules.model.masks import MaskManager
from prune_framework.modules.model.sparsity_patterns import (
    validate_block_sparse_pattern,
    validate_nm_pattern,
    weight_matrix,
)
from prune_framework.plugins.criteria.norm_criteria import MagnitudeCriterion
from prune_framework.plugins.pruners.block_sparse import BlockSparsePruner
from prune_framework.plugins.pruners.nm import NMSparsityPruner


class TinyWeights(nn.Module):
    def __init__(self, conv: bool = False):
        super().__init__()
        self.layer = nn.Conv2d(1, 2, kernel_size=(1, 4), bias=False) if conv else nn.Linear(4, 4, bias=False)

    def forward(self, values):
        return self.layer(values)


class TinyWeightsAdapter(BaseModelAdapter):
    @classmethod
    def load_model(cls, weights_path, device):
        raise NotImplementedError

    def get_pruneable_modules(self):
        return [("layer", self.model.layer)]

    def supported_target_types(self):
        return {TargetType.CONV_WEIGHT, TargetType.CONV_OUT_CHANNEL, TargetType.LINEAR_WEIGHT, TargetType.LINEAR_OUT_FEATURE}

    def get_pruneable_blocks(self):
        return []

    def get_dummy_input(self, device):
        return torch.randn(1, 1, 1, 4, device=device) if isinstance(self.model.layer, nn.Conv2d) else torch.randn(1, 4, device=device)


class TestNMSparsity(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        self.model = TinyWeights()
        self.adapter = TinyWeightsAdapter(self.model)
        self.pruner = NMSparsityPruner()
        with torch.no_grad():
            self.model.layer.weight.copy_(torch.tensor([
                [9.0, 8.0, 2.0, 1.0],
                [1.0, 2.0, 8.0, 9.0],
                [4.0, 4.0, 4.0, 4.0],
                [5.0, 4.0, 3.0, 2.0],
            ]))

    def test_two_of_four_is_exact_deterministic_and_persistent(self):
        config = {"n": 2, "m": 4, "criterion_name": "magnitude"}
        plan = self.pruner.create_plan(self.adapter, MagnitudeCriterion(), config=config)
        self.assertEqual(plan.metadata["semantics"], "n_nonzero_per_m")
        self.assertEqual(plan.metadata["implied_weight_sparsity"], 0.5)
        self.assertTrue(self.pruner.validate_plan(plan, self.adapter, config))
        self.pruner.apply_plan(plan, self.adapter, config)
        mask = weight_matrix(MaskManager.mask(self.model.layer))
        self.assertTrue(validate_nm_pattern(mask, 2, 4))
        self.assertTrue(torch.equal(mask[0], torch.tensor([1.0, 1.0, 0.0, 0.0])))
        # Equal scores retain the first two values due to stable sorting.
        self.assertTrue(torch.equal(mask[2], torch.tensor([1.0, 1.0, 0.0, 0.0])))

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, weight_decay=0.1)
        MaskManager.attach_optimizer(self.model, optimizer)
        self.model(torch.ones(2, 4)).sum().backward()
        optimizer.step()
        original = MaskManager.original_weight(self.model.layer)
        self.assertTrue(torch.equal(original[mask == 0], torch.zeros_like(original[mask == 0])))
        with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
            torch.save({"model": self.model.state_dict(), "masks": MaskManager.state_dict(self.model)}, handle.name)
            checkpoint = torch.load(handle.name, weights_only=True)
            restored = TinyWeights()
            MaskManager.load_state_dict(restored, checkpoint["masks"])
            restored.load_state_dict(checkpoint["model"])
        self.assertTrue(torch.equal(MaskManager.mask(restored.layer), MaskManager.mask(self.model.layer)))

    def test_one_of_two_conv_and_existing_mask_never_revives(self):
        model = TinyWeights(conv=True)
        adapter = TinyWeightsAdapter(model)
        MaskManager.apply(model.layer, torch.tensor([[[[0.0, 1.0, 1.0, 1.0]]], [[[1.0, 1.0, 1.0, 1.0]]]]))
        config = {"n": 1, "m": 2, "criterion_name": "magnitude"}
        plan = self.pruner.create_plan(adapter, MagnitudeCriterion(), config=config)
        self.assertTrue(self.pruner.validate_plan(plan, adapter, config))
        self.pruner.apply_plan(plan, adapter, config)
        mask = weight_matrix(MaskManager.mask(model.layer))
        self.assertEqual(mask[0, 0].item(), 0.0)
        self.assertTrue(validate_nm_pattern(mask, 1, 2, exact=False))
        self.assertEqual(tuple(model(torch.randn(1, 1, 1, 4)).shape), (1, 2, 1, 1))

    def test_invalid_pattern_and_nondivisible_width_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, "0 < n <= m"):
            self.pruner.create_plan(self.adapter, MagnitudeCriterion(), config={"n": 3, "m": 2, "criterion_name": "magnitude"})
        invalid_model = nn.Linear(5, 2, bias=False)
        invalid_adapter = TinyWeightsAdapter(TinyWeights())
        invalid_adapter.model.layer = invalid_model
        with self.assertRaisesRegex(ValueError, "divisible"):
            self.pruner.create_plan(invalid_adapter, MagnitudeCriterion(), config={"n": 2, "m": 4, "criterion_name": "magnitude"})


class TestBlockSparse(unittest.TestCase):
    def setUp(self):
        self.model = TinyWeights()
        self.adapter = TinyWeightsAdapter(self.model)
        self.pruner = BlockSparsePruner()
        with torch.no_grad():
            self.model.layer.weight.copy_(torch.tensor([
                [0.1, 0.1, 4.0, 4.0],
                [0.1, 0.1, 4.0, 4.0],
                [5.0, 5.0, 3.0, 3.0],
                [5.0, 5.0, 3.0, 3.0],
            ]))

    def test_rectangular_block_selection_ties_and_optimizer(self):
        config = {"amount": 0.25, "block_shape": [2, 2], "criterion_name": "magnitude"}
        plan = self.pruner.create_plan(self.adapter, MagnitudeCriterion(), config=config)
        self.assertTrue(self.pruner.validate_plan(plan, self.adapter, config))
        self.pruner.apply_plan(plan, self.adapter, config)
        mask = weight_matrix(MaskManager.mask(self.model.layer))
        self.assertTrue(validate_block_sparse_pattern(mask, [2, 2]))
        self.assertTrue(torch.equal(mask[:2, :2], torch.zeros(2, 2)))
        self.assertEqual(int(mask.eq(0).sum()), 4)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=0.1)
        MaskManager.attach_optimizer(self.model, optimizer)
        self.model(torch.randn(2, 4)).sum().backward()
        optimizer.step()
        original = MaskManager.original_weight(self.model.layer)
        self.assertTrue(torch.equal(original[mask == 0], torch.zeros_like(original[mask == 0])))

    def test_conv_mapping_and_invalid_block_shape(self):
        model = TinyWeights(conv=True)
        adapter = TinyWeightsAdapter(model)
        config = {"amount": 0.5, "block_shape": [1, 2], "criterion_name": "magnitude"}
        plan = self.pruner.create_plan(adapter, MagnitudeCriterion(), config=config)
        self.assertTrue(self.pruner.validate_plan(plan, adapter, config))
        self.pruner.apply_plan(plan, adapter, config)
        self.assertTrue(validate_block_sparse_pattern(weight_matrix(MaskManager.mask(model.layer)), [1, 2]))
        self.assertEqual(tuple(model(torch.randn(1, 1, 1, 4)).shape), (1, 2, 1, 1))

        with self.assertRaisesRegex(ValueError, "divisible"):
            self.pruner.create_plan(self.adapter, MagnitudeCriterion(), config={"amount": 0.25, "block_shape": [3, 2], "criterion_name": "magnitude"})
        with self.assertRaisesRegex(ValueError, "block_shape"):
            self.pruner.create_plan(self.adapter, MagnitudeCriterion(), config={"amount": 0.25, "block_shape": [2], "criterion_name": "magnitude"})

    def test_registered_pruners_execute_through_engine_composition(self):
        cases = [
            ("nm", {"n": 2, "m": 4}, lambda layer: validate_nm_pattern(weight_matrix(MaskManager.mask(layer)), 2, 4)),
            ("block_sparse", {"amount": 0.25, "block_shape": [2, 2]}, lambda layer: validate_block_sparse_pattern(weight_matrix(MaskManager.mask(layer)), [2, 2])),
        ]
        for pruner_name, config, check in cases:
            with self.subTest(pruner=pruner_name):
                model = nn.Sequential(nn.Linear(4, 4, bias=False))
                result = PruningEngine("yolov5", pruner_name, "magnitude", "weight").execute(
                    model, {**config, "criterion_name": "magnitude"}, verify_forward=False
                )
                self.assertEqual(result.pruning_plan["pruner"], pruner_name)
                self.assertTrue(check(model[0]))


if __name__ == "__main__":
    unittest.main()
