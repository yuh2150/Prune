import unittest
from prune_framework.core.registry import PluginRegistry, register_pruner
from prune_framework.core.interfaces import BasePruner
from prune_framework.core.exceptions import DuplicatePluginError


class TestDuplicateRegistration(unittest.TestCase):
    def test_duplicate_registration_raises_error(self):
        @register_pruner("unique_dummy_pruner_a")
        class DummyPrunerA(BasePruner):
            def prune(self, model_adapter, criterion, granularity, config):
                pass

        with self.assertRaises(DuplicatePluginError):
            @register_pruner("unique_dummy_pruner_a")
            class DummyPrunerB(BasePruner):
                def prune(self, model_adapter, criterion, granularity, config):
                    pass


if __name__ == "__main__":
    unittest.main()
