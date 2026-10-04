"""LeNet adapter target policy and structural dependency smoke tests."""

import unittest

import torch

from prune_framework.core.engine import PruningEngine
from prune_framework.models import LeNet5
from prune_framework.plugins.adapters.lenet5 import LeNet5Adapter


class TestClassificationAdapter(unittest.TestCase):
    def test_loads_configured_emnist_class_count(self):
        model, checkpoint = LeNet5Adapter.load_model("random", torch.device("cpu"), num_classes=47)
        self.assertIsNone(checkpoint)
        self.assertEqual(tuple(model(torch.randn(2, 1, 28, 28)).shape), (2, 47))

    def test_adapter_discovers_conv_linear_and_protects_class_head(self):
        adapter = LeNet5Adapter(LeNet5())
        names = [target.name for target in adapter.get_prunable_targets()]
        self.assertIn("features.0", names)
        self.assertIn("classifier.1", names)
        self.assertIn("classifier.3", names)
        self.assertNotIn("classifier.5", names)

    def test_structured_conv_pool_flatten_linear_dependency_forwards(self):
        model = LeNet5()
        result = PruningEngine("lenet5", "structured", "l1", "channel").execute(
            model, {"amount": 0.15, "min_channels": 2}, verify_forward=True
        )
        self.assertTrue(result.forward_verified)
        self.assertEqual(tuple(model(torch.randn(2, 1, 28, 28)).shape), (2, 10))

    def test_filter_plan_targets_conv_filters_only(self):
        model = LeNet5()
        plan = PruningEngine("lenet5", "structured", "l1", "filter").build_plan(
            model, {"amount": 0.2, "min_channels": 2}
        )
        self.assertTrue(plan.groups)
        self.assertTrue(all(group.primary.name.startswith("features.") for group in plan.groups))


if __name__ == "__main__":
    unittest.main()
