import unittest
import torch
import torch.nn as nn
from prune_framework.core.registry import PluginRegistry
from prune_framework.contracts import (
    BaseModelAdapter,
    BasePruner,
    BaseImportanceCriterion,
    BaseGranularity,
    BaseSelector,
)


class TestPluginContracts(unittest.TestCase):
    def setUp(self):
        PluginRegistry._ensure_plugins_loaded()

    def test_model_adapter_contracts(self):
        for model_name in PluginRegistry.list_models():
            adapter_cls = PluginRegistry.get_model_adapter(model_name)
            self.assertTrue(
                issubclass(adapter_cls, BaseModelAdapter),
                f"Model adapter '{model_name}' ({adapter_cls}) must inherit from BaseModelAdapter."
            )
            self.assertTrue(
                hasattr(adapter_cls, "load_model"),
                f"Model adapter '{model_name}' must implement 'load_model' classmethod."
            )

    def test_pruner_contracts(self):
        for pruner_name in PluginRegistry.list_pruners():
            pruner_cls = PluginRegistry.get_pruner(pruner_name)
            self.assertTrue(
                issubclass(pruner_cls, BasePruner),
                f"Pruner '{pruner_name}' ({pruner_cls}) must inherit from BasePruner."
            )
            self.assertTrue(
                hasattr(pruner_cls, "prune"),
                f"Pruner '{pruner_name}' must implement 'prune' method."
            )

    def test_criterion_contracts(self):
        for crit_name in PluginRegistry.list_criteria():
            crit_cls = PluginRegistry.get_criterion(crit_name)
            self.assertTrue(
                issubclass(crit_cls, BaseImportanceCriterion),
                f"Criterion '{crit_name}' ({crit_cls}) must inherit from BaseImportanceCriterion."
            )
            self.assertTrue(
                hasattr(crit_cls, "score"),
                f"Criterion '{crit_name}' must implement 'score' method."
            )

    def test_granularity_contracts(self):
        for gran_name in PluginRegistry.list_granularities():
            gran_cls = PluginRegistry.get_granularity(gran_name)
            self.assertTrue(
                issubclass(gran_cls, BaseGranularity),
                f"Granularity '{gran_name}' ({gran_cls}) must inherit from BaseGranularity."
            )
            self.assertTrue(
                hasattr(gran_cls, "extract_indices"),
                f"Granularity '{gran_name}' must implement 'extract_indices' method."
            )


if __name__ == "__main__":
    unittest.main()
