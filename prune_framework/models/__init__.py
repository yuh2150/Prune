"""Small reference architectures used by framework smoke tests and examples."""

from .lenet5 import LeNet5
from .lenet5_emnist import LeNet5EMNIST

__all__ = ["LeNet5", "LeNet5EMNIST"]
