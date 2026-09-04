import unittest
import torch
import torch.nn as nn
from prune_framework.core.interfaces import BaseImportanceCriterion
from prune_framework.core.registry import register_criterion, PluginRegistry
from prune_framework.core.engine import PruningEngine


@register_criterion("dummy_custom")
class DummyCustomCriterion(BaseImportanceCriterion):
    """Custom user-defined criterion plugin for extensibility proof."""

    def score(self, module: nn.Module, context=None) -> torch.Tensor:
        num = module.out_channels if isinstance(module, nn.Conv2d) else module.out_features
        return torch.ones(num, device=module.weight.device)


class TestExtensibility(unittest.TestCase):
    def test_extensibility_plugin(self):
        self.assertIn("dummy_custom", PluginRegistry.list_criteria())

        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(16)
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            def forward(self, x):
                return self.conv2(self.bn1(self.conv1(x)))

        model = SimpleNet()
        engine = PruningEngine(
            model_name="yolov5",
            pruner_name="structured",
            criterion_name="dummy_custom",
            granularity_name="channel"
        )

        res = engine.execute(model, config={"pruning_params": [(0, 0.5)]})
        self.assertTrue(res.forward_verified)
        self.assertLess(res.params_after, res.params_before)


if __name__ == "__main__":
    unittest.main()
