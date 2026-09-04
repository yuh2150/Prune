import unittest
from prune_framework.core.registry import PluginRegistry
from prune_framework.core.exceptions import PluginNotFoundException


class TestRegistry(unittest.TestCase):
    def test_registry_discovery(self):
        self.assertIn("structured", PluginRegistry.list_pruners())
        self.assertIn("unstructured", PluginRegistry.list_pruners())
        self.assertIn("l1", PluginRegistry.list_criteria())
        self.assertIn("channel", PluginRegistry.list_granularities())
        self.assertIn("yolov5", PluginRegistry.list_models())

    def test_registry_missing_plugin(self):
        with self.assertRaises(PluginNotFoundException):
            PluginRegistry.get_pruner("non_existent_pruner")


if __name__ == "__main__":
    unittest.main()
