"""Regression coverage for the checked-in EMNIST ONNX LeNet import path."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch

from prune_framework.core.config import FrameworkConfig
from prune_framework.core.engine import PruningEngine
from prune_framework.core.exceptions import ConfigValidationException
from prune_framework.modules.model.loader import ModelLoader
from prune_framework.plugins.adapters.lenet5_emnist_onnx import LeNet5EMNISTONNXAdapter


ROOT = Path(__file__).resolve().parents[2]
ONNX_PATH = ROOT / "LeNet5_Numbers&Characters_FP32.onnx"


class TestLeNet5EMNISTONNX(unittest.TestCase):
    def _load(self):
        return ModelLoader.load("lenet5_emnist_onnx", str(ONNX_PATH), torch.device("cpu"), num_classes=47)[0]

    def test_loads_local_onnx_and_exposes_32px_pruning_targets(self):
        model = self._load()
        adapter = LeNet5EMNISTONNXAdapter(model)
        self.assertEqual(tuple(adapter.get_dummy_input(torch.device("cpu")).shape), (1, 1, 32, 32))
        self.assertEqual([name for name, _ in adapter.get_pruneable_modules()], [
            "conv1.conv", "conv2.conv", "conv3.conv", "fc1",
        ])
        self.assertEqual(tuple(model(torch.randn(2, 1, 32, 32)).shape), (2, 47))

    @unittest.skipUnless(importlib.util.find_spec("onnxruntime"), "onnxruntime is required for equivalence test")
    def test_imported_pytorch_logits_match_onnx_runtime(self):
        import numpy as np
        import onnxruntime as ort

        torch.manual_seed(13)
        inputs = torch.randn(3, 1, 32, 32)
        model = self._load().eval()
        expected = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"]).run(
            None, {"input": inputs.numpy()}
        )[0]
        actual = model(inputs).detach().numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_structured_pruning_keeps_47_class_forward_contract(self):
        model = self._load()
        result = PruningEngine("lenet5_emnist_onnx", "structured", "l1", "channel").execute(
            model, {"amount": 0.1, "min_channels": 2}, verify_forward=True
        )
        self.assertTrue(result.forward_verified)
        self.assertEqual(tuple(model(torch.randn(1, 1, 32, 32)).shape), (1, 47))

    def test_onnx_config_requires_32px_input_and_padding(self):
        config = FrameworkConfig.from_yaml(str(ROOT / "configs/lenet5_emnist_balanced_sensitivity_full.yaml"))
        self.assertEqual(config.dataset.image_padding, 2)
        with self.assertRaisesRegex(ConfigValidationException, "image_padding=2"):
            FrameworkConfig.from_dict(
                {
                    "model": {
                        "name": "lenet5_emnist_onnx",
                        "input_shape": [1, 1, 32, 32],
                        "num_classes": 47,
                    },
                    "dataset": {"name": "emnist", "emnist_split": "balanced"},
                }
            )


if __name__ == "__main__":
    unittest.main()
