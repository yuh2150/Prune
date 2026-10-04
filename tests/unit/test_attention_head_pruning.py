"""Masked multi-head attention pruning contracts on CPU."""

from __future__ import annotations

import tempfile
import unittest

import torch
import torch.nn as nn

from prune_framework.contracts import AttentionHeadTarget, BaseModelAdapter
from prune_framework.modules.model.masks import MaskManager, ParameterMaskManager
from prune_framework.plugins.criteria.norm_criteria import MagnitudeCriterion
from prune_framework.plugins.adapters.rtdetr import RTDETRAdapter
from prune_framework.plugins.pruners.attention_head import AttentionHeadPruner


class TinyAttentionAdapter(BaseModelAdapter):
    @classmethod
    def load_model(cls, weights_path, device):
        raise NotImplementedError

    def get_pruneable_modules(self):
        return []

    def get_pruneable_blocks(self):
        return []

    def get_dummy_input(self, device):
        return torch.randn(2, 3, 8, device=device)

    def get_attention_head_targets(self):
        module = self.model.attention
        return [AttentionHeadTarget("attention", module, 4, 2, "fused_qkv")]


class TinyAttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(8, 4, batch_first=True)

    def forward(self, values):
        return self.attention(values, values, values, need_weights=False)[0]


class RTDetrSelfAttention(nn.Module):
    """Minimal protocol fixture; the exact class name is intentional."""

    def __init__(self):
        super().__init__()
        self.head_dim = 2
        self.q_proj = nn.Linear(8, 8)
        self.k_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)
        self.o_proj = nn.Linear(8, 8)

    def forward(self, values):
        return self.o_proj(self.q_proj(values) + self.k_proj(values) + self.v_proj(values))


class TinyRTDetrAttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = RTDetrSelfAttention()
        self.ordinary_linear = nn.Linear(8, 8)

    def forward(self, values):
        return self.attention(values) + self.ordinary_linear(values)


class TestAttentionHeadPruning(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(61)
        self.model = TinyAttentionModel()
        self.adapter = TinyAttentionAdapter(self.model)
        self.pruner = AttentionHeadPruner()

    def test_masks_complete_qkv_and_output_slices_with_deterministic_ties(self):
        with torch.no_grad():
            self.model.attention.in_proj_weight.fill_(1)
            self.model.attention.out_proj.weight.fill_(1)
        plan = self.pruner.create_plan(self.adapter, MagnitudeCriterion(), config={"amount": 0.5, "criterion_name": "magnitude"})
        self.assertEqual(plan.metadata["semantics"], "masked_head")
        self.assertFalse(plan.metadata["physical_compression"])
        self.assertEqual(plan.groups[0].indices, [0, 1])
        self.assertTrue(self.pruner.validate_plan(plan, self.adapter))
        self.pruner.apply_plan(plan, self.adapter)

        rows = torch.tensor([0, 1, 2, 3])
        qkv_mask = ParameterMaskManager.mask(self.model.attention, "in_proj_weight")
        for projection in range(3):
            self.assertTrue(torch.equal(qkv_mask[rows + projection * 8], torch.zeros_like(qkv_mask[rows])))
        output_mask = ParameterMaskManager.mask(self.model.attention.out_proj, "weight")
        self.assertTrue(torch.equal(output_mask[:, rows], torch.zeros_like(output_mask[:, rows])))
        self.assertEqual(tuple(self.model(torch.randn(2, 3, 8)).shape), (2, 3, 8))

    def test_mask_survives_optimizer_and_round_trip(self):
        plan = self.pruner.create_plan(self.adapter, MagnitudeCriterion(), config={"amount": 0.25, "criterion_name": "magnitude"})
        self.assertTrue(self.pruner.validate_plan(plan, self.adapter))
        self.pruner.apply_plan(plan, self.adapter)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=0.1)
        MaskManager.attach_optimizer(self.model, optimizer)
        self.model(torch.randn(2, 3, 8)).square().mean().backward()
        optimizer.step()
        qkv_mask = ParameterMaskManager.mask(self.model.attention, "in_proj_weight")
        original = ParameterMaskManager.original(self.model.attention, "in_proj_weight")
        self.assertTrue(torch.equal(original[qkv_mask == 0], torch.zeros_like(original[qkv_mask == 0])))

        with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
            torch.save({"model": self.model.state_dict(), "parameter_masks": ParameterMaskManager.state_dict(self.model)}, handle.name)
            checkpoint = torch.load(handle.name, weights_only=True)
            restored = TinyAttentionModel()
            ParameterMaskManager.load_state_dict(restored, checkpoint["parameter_masks"])
            restored.load_state_dict(checkpoint["model"])
        self.assertTrue(torch.equal(
            ParameterMaskManager.mask(self.model.attention, "in_proj_weight"),
            ParameterMaskManager.mask(restored.attention, "in_proj_weight"),
        ))
        inputs = torch.randn(1, 3, 8)
        self.assertTrue(torch.equal(self.model(inputs), restored(inputs)))

    def test_invalid_head_plan_and_unsupported_criterion_fail_loudly(self):
        target = self.adapter.get_attention_head_targets()[0]
        invalid = self.pruner.create_plan(self.adapter, MagnitudeCriterion(), config={"amount": 0.25, "criterion_name": "magnitude"})
        invalid.groups[0].indices = [0, 1, 2, 3]
        self.assertFalse(self.pruner.validate_plan(invalid, self.adapter))
        self.assertIn("retain at least one", invalid.groups[0].validation_error)
        with self.assertRaisesRegex(ValueError, "magnitude, L1, or L2"):
            self.pruner.create_plan(self.adapter, object(), config={"amount": 0.25, "criterion_name": "snip"})
        self.assertEqual(target.num_heads, 4)

    def test_rtdetr_adapter_discovers_only_explicit_attention_protocol(self):
        model = TinyRTDetrAttentionModel()
        adapter = RTDETRAdapter(model)
        targets = adapter.get_attention_head_targets()
        self.assertEqual([target.name for target in targets], ["attention"])
        self.assertEqual(targets[0].layout, "separate_qkv")
        plan = self.pruner.create_plan(adapter, MagnitudeCriterion(), config={"amount": 0.25, "criterion_name": "magnitude"})
        self.assertTrue(self.pruner.validate_plan(plan, adapter))
        self.pruner.apply_plan(plan, adapter)
        selected = plan.groups[0].indices
        rows = torch.tensor(selected)[:, None] * 2 + torch.arange(2)[None, :]
        rows = rows.reshape(-1)
        for projection in (model.attention.q_proj, model.attention.k_proj, model.attention.v_proj):
            self.assertTrue(torch.equal(ParameterMaskManager.mask(projection, "weight")[rows], torch.zeros_like(projection.weight[rows])))
        self.assertEqual(tuple(model(torch.randn(2, 3, 8)).shape), (2, 3, 8))


if __name__ == "__main__":
    unittest.main()
