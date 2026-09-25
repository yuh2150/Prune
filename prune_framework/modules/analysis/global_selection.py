"""Deterministic global ranking for element-wise pruning scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch

from prune_framework.contracts.targets import PrunableTarget


@dataclass(frozen=True)
class GlobalSelection:
    """Indices selected from each target by a single global score ranking."""

    indices: Dict[str, list[int]]
    total_elements: int
    selected_elements: int


class GlobalScoreSelector:
    """Select the lowest element-wise scores across ordered pruning targets.

    Concatenating scores in adapter target order and using stable sorting makes
    ties deterministic: earlier targets and then lower flattened indices win.
    The selector is intentionally score-only, so SNIP, LAMP and later saliency
    criteria can share it without coupling to mask mutation.
    """

    @staticmethod
    def select_lowest(
        score_items: Sequence[Tuple[PrunableTarget, torch.Tensor]], amount: float
    ) -> GlobalSelection:
        if not 0 <= float(amount) <= 1:
            raise ValueError("Global pruning amount must be in [0, 1].")
        if not score_items:
            return GlobalSelection({}, 0, 0)

        flattened: list[torch.Tensor] = []
        offsets: list[tuple[PrunableTarget, int, int]] = []
        offset = 0
        for target, scores in score_items:
            if tuple(scores.shape) != tuple(target.module.weight.shape):
                raise ValueError(
                    f"Element-wise score shape {tuple(scores.shape)} does not match "
                    f"{target.name} weight shape {tuple(target.module.weight.shape)}."
                )
            values = scores.detach().reshape(-1).to(device="cpu")
            size = values.numel()
            flattened.append(values)
            offsets.append((target, offset, offset + size))
            offset += size

        total = offset
        selected = int(round(total * float(amount)))
        if selected == 0:
            return GlobalSelection({}, total, 0)

        # ``stable=True`` preserves concatenation order for equal saliencies.
        ranking = torch.argsort(torch.cat(flattened), stable=True)[:selected]
        chosen = torch.zeros(total, dtype=torch.bool)
        chosen[ranking] = True
        selections: Dict[str, list[int]] = {}
        for target, start, end in offsets:
            local = torch.nonzero(chosen[start:end], as_tuple=False).flatten().tolist()
            if local:
                selections[target.name] = [int(index) for index in local]
        return GlobalSelection(selections, total, selected)
