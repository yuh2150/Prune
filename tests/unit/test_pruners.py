import unittest

import torch
import torch.nn as nn

from prune_framework.core.engine import PruningEngine


class TestPrunerContractsAtRuntime(unittest.TestCase):
    def test_unstructured_uses_the_shared_pruner_signature(self):
        model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(), nn.Conv2d(8, 8, 3))
        result = PruningEngine("yolov5", "unstructured", "magnitude", "weight").execute(
            model, {"amount": 0.25}
        )
        self.assertTrue(result.forward_verified)
        self.assertEqual(result.params_before, result.params_after)
        self.assertGreater((model[0].weight == 0).sum().item(), 0)

    def test_global_structured_pruning_preserves_minimum_width(self):
        model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(), nn.Conv2d(8, 16, 3))
        result = PruningEngine("yolov5", "structured", "l1", "channel").execute(
            model, {"amount": 0.5, "global_pruning": True, "min_channels": 2}
        )
        self.assertTrue(result.forward_verified)
        self.assertGreaterEqual(model[0].out_channels, 2)
        self.assertGreaterEqual(model[2].out_channels, 2)
        self.assertLess(result.params_after, result.params_before)


if __name__ == "__main__":
    unittest.main()
