import math
import torch


def make_divisible(x: float, divisor: int) -> int:
    """
    Returns nearest number x divisible by divisor.
    """
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor
