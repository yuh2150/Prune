"""Reproducibility and artifact helpers for pruning experiments."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


class ExperimentArtifacts:
    def __init__(self, output_dir: str, name: str):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.run_dir = Path(output_dir) / f"{name}_{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=False)

    def write_json(self, filename: str, value: Any) -> str:
        path = self.run_dir / filename
        with path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, default=_json_default)
        return str(path)


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")
