"""Safe compact attention pruning for adapter-declared MultiheadAttention modules."""
from __future__ import annotations
from typing import Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from prune_framework.contracts.targets import AttentionHeadTarget, PruningGroup, PruningPlan
from prune_framework.core.interfaces import BaseModelAdapter, BasePruner
from prune_framework.core.registry import register_pruner


class CompactMultiheadAttention(nn.Module):
    """MHA with fewer internal heads but its original external embedding width."""
    def __init__(self, source: nn.MultiheadAttention, kept_heads: list[int]):
        super().__init__()
        if source.kdim != source.embed_dim or source.vdim != source.embed_dim or source.bias_k is not None or source.add_zero_attn:
            raise ValueError("Only self-attention with equal Q/K/V dimensions and no bias_k/add_zero_attn is supported.")
        self.embed_dim, self.head_dim, self.num_heads = source.embed_dim, source.head_dim, len(kept_heads)
        self.batch_first, self.dropout = source.batch_first, source.dropout
        self.inner_dim = self.head_dim * self.num_heads
        indices = torch.cat([torch.arange(h * self.head_dim, (h + 1) * self.head_dim) for h in kept_heads])
        self.q_proj = nn.Linear(self.embed_dim, self.inner_dim, bias=source.in_proj_bias is not None)
        self.k_proj, self.v_proj = nn.Linear(self.embed_dim, self.inner_dim, bias=source.in_proj_bias is not None), nn.Linear(self.embed_dim, self.inner_dim, bias=source.in_proj_bias is not None)
        self.out_proj = nn.Linear(self.inner_dim, self.embed_dim, bias=source.out_proj.bias is not None)
        with torch.no_grad():
            for i, projection in enumerate((self.q_proj, self.k_proj, self.v_proj)):
                projection.weight.copy_(source.in_proj_weight[i * self.embed_dim:(i + 1) * self.embed_dim][indices])
                if projection.bias is not None: projection.bias.copy_(source.in_proj_bias[i * self.embed_dim:(i + 1) * self.embed_dim][indices])
            self.out_proj.weight.copy_(source.out_proj.weight[:, indices]);
            if self.out_proj.bias is not None: self.out_proj.bias.copy_(source.out_proj.bias)
    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False):
        if key_padding_mask is not None or is_causal:
            raise ValueError("Compact attention does not yet support key_padding_mask or is_causal.")
        def sequence(tensor): return tensor.transpose(0, 1) if self.batch_first else tensor
        q_input, k_input, v_input = sequence(query), sequence(key), sequence(value)
        q, k, v = (projection(tensor).reshape(tensor.shape[0], tensor.shape[1], self.num_heads, self.head_dim).transpose(1, 2) for projection, tensor in ((self.q_proj, q_input), (self.k_proj, k_input), (self.v_proj, v_input)))
        scores = q @ k.transpose(-2, -1) / self.head_dim ** 0.5
        if attn_mask is not None: scores = scores + attn_mask
        weights = torch.softmax(scores, -1); out = (F.dropout(weights, p=self.dropout, training=self.training) @ v).transpose(1, 2).reshape(q_input.shape[0], q_input.shape[1], self.inner_dim)
        out = self.out_proj(out); out = out.transpose(0, 1) if self.batch_first else out
        returned = weights.mean(1) if average_attn_weights else weights
        return out, returned if need_weights else None


@register_pruner("attention_head")
class AttentionHeadPruner(BasePruner):
    def create_plan(self, model_adapter, criterion=None, granularity=None, config: Dict[str, Any] | None=None):
        config = config or {}; amount = config.get("amount", 0.25); plan = PruningPlan(pruner_name="attention_head")
        targets = model_adapter.get_attention_head_targets()
        if not targets:
            raise ValueError(f"{type(model_adapter).__name__} exposes no adapter-approved attention-head targets.")
        for target in targets:
            heads = target.module.num_heads; drop = int(heads * amount)
            if not 0 < drop < heads: raise ValueError(f"Attention target '{target.name}' requires pruning between one and {heads - 1} heads.")
            hd = target.module.head_dim; importance = target.module.in_proj_weight.detach().reshape(3, heads, hd, -1).abs().mean((0, 2, 3))
            removed = torch.argsort(importance, stable=True)[:drop].tolist()
            plan.groups.append(PruningGroup(primary=target, operation="compact_attention_heads", indices=removed))
        return plan
    def validate_plan(self, plan, model_adapter, config=None):
        valid = True
        for group in plan.groups:
            target = group.primary; ok = isinstance(target, AttentionHeadTarget) and bool(group.indices) and len(set(group.indices)) == len(group.indices) and all(0 <= i < target.module.num_heads for i in group.indices) and len(group.indices) < target.module.num_heads and getattr(target.owner, target.attribute, None) is target.module
            group.validated, group.dependency_count = ok, 4
            if not ok: group.validation_error, valid = "Invalid attention-head plan or stale owner reference.", False
        return valid
    def apply_plan(self, plan, model_adapter, config=None):
        for group in plan.groups:
            if not group.validated: raise RuntimeError(f"Cannot apply unvalidated attention plan for {group.primary.name}.")
            target = group.primary; kept = [i for i in range(target.module.num_heads) if i not in set(group.indices)]
            setattr(target.owner, target.attribute, CompactMultiheadAttention(target.module, kept))
        return model_adapter.model
