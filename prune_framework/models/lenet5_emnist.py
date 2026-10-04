"""The 32x32, 47-class LeNet-5 topology stored in the local ONNX model."""

from __future__ import annotations

import torch
import torch.nn as nn


class _ConvActivation(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(inputs))


class LeNet5EMNIST(nn.Module):
    """Exact PyTorch counterpart of ``LeNet5_Numbers&Characters_FP32.onnx``.

    Input is a padded, grayscale ``[N, 1, 32, 32]`` tensor and output is 47
    EMNIST Balanced logits. Module names intentionally match the ONNX
    initializer names, allowing the adapter to copy weights without a generic
    ONNX graph converter.
    """

    def __init__(self, num_classes: int = 47):
        super().__init__()
        self.conv1 = _ConvActivation(1, 6)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = _ConvActivation(6, 16)
        self.pool_1 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = _ConvActivation(16, 120)
        self.fc1 = nn.Linear(120, 84)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        values = self.pool(self.conv1(images))
        values = self.pool_1(self.conv2(values))
        values = self.conv3(values)
        values = torch.flatten(values, 1)
        return self.fc2(self.relu(self.fc1(values)))
