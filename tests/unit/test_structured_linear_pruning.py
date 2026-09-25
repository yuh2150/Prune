"""DependencyGraph-backed Linear output-feature pruning coverage."""

from __future__ import annotations

import copy
import unittest

import torch
import torch.nn as nn

from prune_framework.contracts import BaseModelAdapter, PrunableTarget, PruningGroup, PruningPlan, TargetType
from prune_framework.plugins.criteria.norm_criteria import L1NormCriterion
from prune_framework.plugins.granularities.base import ChannelGranularity
from prune_framework.plugins.pruners.structured import StructuredPruner


class LinearChain(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(6, 8)
        self.output = nn.Linear(8, 4)

    def forward(self, values):
        return self.output(torch.relu(self.hidden(values)))


class ResidualLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = nn.Linear(6, 8)
        self.right = nn.Linear(6, 8)
        self.output = nn.Linear(8, 4)

    def forward(self, values):
        return self.output(torch.relu(self.left(values) + self.right(values)))


class FlattenLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(6, 8)
        self.output = nn.Linear(8, 4)

    def forward(self, values):
        # Both shape operations retain the dynamically inferred feature size.
        # DepGraph can therefore propagate hidden-output pruning to output-in.
        features = self.hidden(values).unsqueeze(1)
        return self.output(features.flatten(1))


class RootOnlyLinearAdapter(BaseModelAdapter):
    """Expose only declared safe roots; downstream modules remain traceable."""

    def __init__(self, model, root_name):
        super().__init__(model)
        self.root_name = root_name

    @classmethod
    def load_model(cls, weights_path, device):
        raise NotImplementedError

    def get_pruneable_modules(self):
        return [(self.root_name, getattr(self.model, self.root_name))]

    def get_pruneable_blocks(self):
        return []

    def get_dummy_input(self, device):
        return torch.randn(3, 6, device=device)

    def supported_target_types(self):
        return {TargetType.LINEAR_OUT_FEATURE}


class TestStructuredLinearPruning(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(89)
        self.pruner = StructuredPruner()
        self.criterion = L1NormCriterion()

    def _plan(self, adapter):
        return self.pruner.create_plan(
            adapter, self.criterion, ChannelGranularity(), {"amount": 0.25, "min_features": 2}
        )

    def test_linear_chain_propagates_to_downstream_input_and_reduces_parameters(self):
        model = LinearChain()
        adapter = RootOnlyLinearAdapter(model, "hidden")
        parameters_before = sum(parameter.numel() for parameter in model.parameters())
        plan = self._plan(adapter)

        self.assertEqual(plan.groups[0].operation, "prune_linear_out_features")
        self.assertTrue(self.pruner.validate_plan(plan, adapter))
        self.pruner.apply_plan(plan, adapter)

        self.assertEqual(model.hidden.out_features, 6)
        self.assertEqual(model.output.in_features, 6)
        self.assertLess(sum(parameter.numel() for parameter in model.parameters()), parameters_before)
        self.assertEqual(tuple(model(adapter.get_dummy_input(torch.device("cpu"))).shape), (3, 4))

    def test_linear_residual_addition_prunes_coupled_branch_and_output_projection(self):
        model = ResidualLinear()
        adapter = RootOnlyLinearAdapter(model, "left")
        plan = self._plan(adapter)

        self.assertTrue(self.pruner.validate_plan(plan, adapter))
        self.assertTrue(any(item["target_name"] == "right" for item in plan.groups[0].dependencies))
        self.pruner.apply_plan(plan, adapter)

        self.assertEqual(model.left.out_features, 6)
        self.assertEqual(model.right.out_features, 6)
        self.assertEqual(model.output.in_features, 6)
        self.assertEqual(tuple(model(adapter.get_dummy_input(torch.device("cpu"))).shape), (3, 4))

    def test_linear_output_propagates_through_dynamic_unsqueeze_and_flatten(self):
        model = FlattenLinear()
        adapter = RootOnlyLinearAdapter(model, "hidden")
        plan = self._plan(adapter)

        self.assertTrue(self.pruner.validate_plan(plan, adapter))
        self.pruner.apply_plan(plan, adapter)

        self.assertEqual(model.hidden.out_features, 6)
        self.assertEqual(model.output.in_features, 6)
        self.assertEqual(tuple(model(adapter.get_dummy_input(torch.device("cpu"))).shape), (3, 4))

    def test_non_structural_target_is_rejected_before_mutation(self):
        model = LinearChain()
        adapter = RootOnlyLinearAdapter(model, "hidden")
        state_before = copy.deepcopy(model.state_dict())
        invalid_target = PrunableTarget("hidden", model.hidden, TargetType.LINEAR_WEIGHT)
        plan = PruningPlan(
            pruner_name="structured",
            groups=[PruningGroup(primary=invalid_target, operation="prune_linear_out_features", indices=[0])],
        )

        self.assertFalse(self.pruner.validate_plan(plan, adapter))
        self.assertIn("Unsupported structural target type", plan.groups[0].validation_error)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, state_before[name]))

    def test_existing_conv_operation_is_unchanged(self):
        model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(), nn.Conv2d(8, 8, 3))
        from prune_framework.plugins.adapters.yolov5 import YOLOv5Adapter

        adapter = YOLOv5Adapter(model)
        plan = self.pruner.create_plan(adapter, self.criterion, ChannelGranularity(), {"amount": 0.25})
        self.assertTrue(all(group.operation == "prune_conv_out_channels" for group in plan.groups))
        self.assertTrue(self.pruner.validate_plan(plan, adapter))
        self.pruner.apply_plan(plan, adapter)
        self.assertEqual(tuple(model(adapter.get_dummy_input(torch.device("cpu"))).shape)[1], 6)


if __name__ == "__main__":
    unittest.main()
