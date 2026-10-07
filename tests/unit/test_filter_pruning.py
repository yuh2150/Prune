"""Filter pruning means Conv2d output-filter pruning, never scalar masking."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from prune_framework.contracts import BaseModelAdapter, TargetType
from prune_framework.core.engine import PruningEngine
from prune_framework.plugins.criteria.norm_criteria import L1NormCriterion
from prune_framework.plugins.granularities.base import FilterGranularity
from prune_framework.plugins.pruners.structured import StructuredPruner


class FilterFixture(nn.Module):
    def __init__(self, *, groups=1):
        super().__init__()
        self.producer = nn.Conv2d(4 if groups > 1 else 2, 4, 1, bias=True, groups=groups)
        self.consumer = nn.Conv2d(4, 3, 1, bias=False)
        # Deliberately an adapter-approved structural target: sensitivity must
        # nevertheless omit it for Conv-only filter pruning.
        self.classifier = nn.Linear(3, 2)

    def forward(self, images):
        return self.consumer(torch.relu(self.producer(images)))


class FilterAdapter(BaseModelAdapter):
    @classmethod
    def load_model(cls, weights_path, device):
        raise NotImplementedError

    def get_pruneable_modules(self):
        return [("producer", self.model.producer), ("classifier", self.model.classifier)]

    def get_pruneable_blocks(self):
        return []

    def get_dummy_input(self, device):
        return torch.randn(2, self.model.producer.in_channels, 5, 5, device=device)

    def supported_target_types(self):
        return {TargetType.CONV_OUT_CHANNEL, TargetType.LINEAR_OUT_FEATURE}


class TestFilterPruning(unittest.TestCase):
    def setUp(self):
        self.model = FilterFixture()
        self.adapter = FilterAdapter(self.model)
        with torch.no_grad():
            # L1 output-filter scores: [1, 2, 3, 4].
            for index in range(4):
                self.model.producer.weight[index].fill_(index + 1)
            self.model.producer.bias.copy_(torch.arange(4.0))
        self.pruner = StructuredPruner()

    def test_filter_ratio_selects_output_filters_and_propagates_bias_and_consumer(self):
        plan = self.pruner.create_plan(
            self.adapter, L1NormCriterion(), FilterGranularity(), {"amount": 0.5, "min_channels": 1}
        )
        self.assertEqual(plan.groups[0].primary.name, "producer")
        self.assertEqual(plan.groups[0].indices, [0, 1])
        self.assertTrue(self.pruner.validate_plan(plan, self.adapter))
        self.pruner.apply_plan(plan, self.adapter)
        self.assertEqual(self.model.producer.out_channels, 2)
        self.assertEqual(self.model.producer.bias.numel(), 2)
        self.assertEqual(self.model.consumer.in_channels, 2)
        self.assertEqual(tuple(self.model(self.adapter.get_dummy_input(torch.device("cpu"))).shape), (2, 3, 5, 5))

    def test_filter_ties_have_stable_lowest_index_selection(self):
        with torch.no_grad():
            self.model.producer.weight.fill_(1)
        plan = self.pruner.create_plan(
            self.adapter, L1NormCriterion(), FilterGranularity(), {"amount": 0.5, "min_channels": 1}
        )
        self.assertEqual(plan.groups[0].indices, [0, 1])

    def test_grouped_and_depthwise_filter_pruning_fail_loudly(self):
        grouped = FilterFixture(groups=2)
        adapter = FilterAdapter(grouped)
        with self.assertRaisesRegex(ValueError, "grouped or depthwise"):
            self.pruner.create_plan(adapter, L1NormCriterion(), FilterGranularity(), {"amount": 0.5})

    def test_engine_sensitivity_targets_match_filter_plan_targets(self):
        # Construct the engine boundary directly so this guards the target
        # indexing used by SensitivityAnalyzer, without registering a global
        # test-only plugin.
        engine = object.__new__(PruningEngine)
        engine.adapter_cls = FilterAdapter
        engine.pruner_cls = StructuredPruner
        engine.criterion_cls = L1NormCriterion
        engine.granularity_cls = FilterGranularity
        self.assertEqual([target.name for target in engine.targets(self.model)], ["producer"])


if __name__ == "__main__":
    unittest.main()
