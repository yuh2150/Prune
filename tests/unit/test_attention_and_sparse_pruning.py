import tempfile
import unittest
import torch
import torch.nn as nn

from prune_framework.contracts import AttentionHeadTarget, BaseModelAdapter, TargetType
from prune_framework.modules.model.masks import MaskManager
from prune_framework.plugins.pruners.attention import AttentionHeadPruner, CompactMultiheadAttention
from prune_framework.plugins.pruners.sparse import BlockSparsePruner, NMSparsityPruner
from prune_framework.plugins.adapters.yolov5 import YOLOv5Adapter
from models.common import TransformerLayer


class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__(); self.attn = nn.MultiheadAttention(8, 4, batch_first=True); self.norm = nn.LayerNorm(8)
    def forward(self, x):
        y, _ = self.attn(x, x, x, need_weights=False); return self.norm(x + y)


class MixedModel(nn.Module):
    def __init__(self):
        super().__init__(); self.conv = nn.Conv2d(2, 4, 2, bias=False); self.linear = nn.Linear(16, 8, bias=False)
    def forward(self, x): return self.linear(self.conv(x).flatten(1))


class Adapter(BaseModelAdapter):
    @classmethod
    def load_model(cls, weights_path, device): raise NotImplementedError
    def get_pruneable_modules(self): return [(n, m) for n, m in self.model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    def get_pruneable_blocks(self): return []
    def get_dummy_input(self, device): return torch.randn(2, 2, 3, 3, device=device)
    def supported_target_types(self): return {TargetType.CONV_WEIGHT, TargetType.LINEAR_WEIGHT}
    def get_attention_head_targets(self):
        return [AttentionHeadTarget("attn", self.model.attn, self.model, "attn")] if isinstance(getattr(self.model, "attn", None), nn.MultiheadAttention) else []


class TestAttentionHeadAndSparsePruning(unittest.TestCase):
    def test_attention_prunes_one_and_multiple_heads_and_preserves_residual_shape(self):
        model = AttentionModel(); adapter = Adapter(model); x = torch.randn(2, 5, 8)
        plan = AttentionHeadPruner().create_plan(adapter, config={"amount": .25})
        pruner = AttentionHeadPruner(); self.assertTrue(pruner.validate_plan(plan, adapter)); pruner.apply_plan(plan, adapter)
        self.assertIsInstance(model.attn, CompactMultiheadAttention); self.assertEqual(model.attn.num_heads, 3); self.assertEqual(model(x).shape, x.shape)
        model = AttentionModel(); adapter = Adapter(model); plan = pruner.create_plan(adapter, config={"amount": .5}); self.assertTrue(pruner.validate_plan(plan, adapter)); pruner.apply_plan(plan, adapter)
        self.assertEqual(model.attn.num_heads, 2); self.assertEqual(model(x).shape, x.shape)

    def test_attention_rejects_invalid_counts_and_round_trips(self):
        model = AttentionModel(); adapter = Adapter(model); pruner = AttentionHeadPruner()
        with self.assertRaises(ValueError): pruner.create_plan(adapter, config={"amount": 0})
        with self.assertRaises(ValueError): pruner.create_plan(adapter, config={"amount": 1})
        plan = pruner.create_plan(adapter, config={"amount": .25}); self.assertTrue(pruner.validate_plan(plan, adapter)); pruner.apply_plan(plan, adapter)
        x = torch.randn(2, 5, 8)
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/attention.pt"
            torch.save(model, path); restored = torch.load(path, weights_only=False)
        self.assertTrue(torch.allclose(model(x), restored(x)))

    def test_attention_rejects_adapters_without_explicit_targets(self):
        with self.assertRaisesRegex(ValueError, "no adapter-approved"):
            AttentionHeadPruner().create_plan(Adapter(MixedModel()), config={"amount": .25})

    def test_yolo_transformer_layer_adapter_exposure_and_residual_forward(self):
        model = nn.Module(); model.layer = TransformerLayer(8, 4)
        adapter = YOLOv5Adapter(model); self.assertEqual(len(adapter.get_attention_head_targets()), 1)
        pruner = AttentionHeadPruner(); plan = pruner.create_plan(adapter, config={"amount": .25})
        self.assertTrue(pruner.validate_plan(plan, adapter)); pruner.apply_plan(plan, adapter)
        self.assertEqual(model.layer(torch.randn(5, 2, 8)).shape, (5, 2, 8))

    def _sparse_case(self, pruner, config, checker):
        model = MixedModel(); adapter = Adapter(model); plan = pruner.create_plan(adapter, config=config)
        self.assertTrue(pruner.validate_plan(plan, adapter)); pruner.apply_plan(plan, adapter, {"optimizer": torch.optim.SGD(model.parameters(), .1)})
        self.assertEqual(model(torch.randn(2, 2, 3, 3)).shape, (2, 8))
        for _, module in adapter.get_pruneable_modules(): checker(MaskManager.mask(module))
        optimizer = torch.optim.SGD(model.parameters(), .1); MaskManager.attach_optimizer(model, optimizer); optimizer.zero_grad(); model(torch.randn(2,2,3,3)).sum().backward(); optimizer.step()
        for _, module in adapter.get_pruneable_modules(): self.assertTrue(torch.all(MaskManager.original_weight(module)[MaskManager.mask(module) == 0] == 0))
        state = MaskManager.state_dict(model); restored = MixedModel(); MaskManager.load_state_dict(restored, state)
        for name, module in restored.named_modules():
            if name in state: self.assertTrue(torch.equal(MaskManager.mask(module), state[name]))

    def test_nm_masks_are_exact_and_validate_shapes(self):
        def check(mask): self.assertTrue(torch.all(mask.reshape(-1, 4).sum(1) == 2))
        self._sparse_case(NMSparsityPruner(), {"n": 2, "m": 4}, check)
        with self.assertRaises(ValueError): NMSparsityPruner().create_plan(Adapter(MixedModel()), config={"n": 3, "m": 2})
        with self.assertRaises(ValueError): NMSparsityPruner().create_plan(Adapter(MixedModel()), config={"n": 2, "m": 7})

    def test_block_masks_are_whole_blocks_and_aligned(self):
        def check(mask):
            matrix = mask.reshape(mask.shape[0], -1).reshape(mask.shape[0] // 2, 2, -1 // 2 if False else mask[0].numel() // 2, 2)
            values = matrix.reshape(matrix.shape[0], 2, matrix.shape[2], 2).mean((1, 3)); self.assertTrue(torch.all((values == 0) | (values == 1)))
        self._sparse_case(BlockSparsePruner(), {"block_size": [2, 2], "amount": .5}, check)
        with self.assertRaises(ValueError): BlockSparsePruner().create_plan(Adapter(MixedModel()), config={"block_size": [3, 2], "amount": .5})


if __name__ == "__main__": unittest.main()
