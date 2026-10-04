"""User-supplied calibration batches and loss, prepared once per baseline."""
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional


@dataclass
class CalibrationContext:
    batches: Iterable[Any]
    loss_fn: Callable
    seed: int = 42
    sample_count: Optional[int] = None
    device: Optional[str] = None

    def validate(self):
        if not callable(self.loss_fn):
            raise TypeError('CalibrationContext.loss_fn must be callable')
        self.batches = list(self.batches)
        if not self.batches:
            raise ValueError('CalibrationContext requires non-empty batches')
        return self
