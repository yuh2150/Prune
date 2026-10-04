"""Local-only classification data policy and LeNet non-applicable mechanisms."""

import tempfile
import unittest

from prune_framework.core.config import DatasetConfig, FrameworkConfig
from prune_framework.core.exceptions import ConfigValidationException
from prune_framework.datasets import create_classification_dataloaders
from prune_framework.models import LeNet5
from prune_framework.plugins.adapters.lenet5 import LeNet5Adapter


class TestClassificationDataset(unittest.TestCase):
    def test_missing_local_dataset_fails_without_download(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "download it explicitly"):
                create_classification_dataloaders(DatasetConfig(name="mnist", root=directory, batch_size=2))

    def test_fashionmnist_uses_same_factory_contract_and_lenet_has_no_heads_or_blocks(self):
        config = DatasetConfig(name="fashionmnist", root="/not-a-local-cache", batch_size=2, train_limit=2, val_limit=2)
        self.assertEqual(config.name, "fashionmnist")
        adapter = LeNet5Adapter(LeNet5())
        self.assertEqual(adapter.get_attention_head_targets(), [])
        self.assertEqual(adapter.get_structural_block_targets(), [])
        self.assertFalse(adapter.supports_channel_sparsity_regularization())

    def test_missing_emnist_cache_fails_without_download(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "EMNIST.*split balanced.*download it explicitly"):
                create_classification_dataloaders(
                    DatasetConfig(name="emnist", emnist_split="balanced", root=directory, batch_size=2)
                )

    def test_emnist_letters_labels_are_zero_based_and_lenet_class_count_is_validated(self):
        from prune_framework.datasets.classification import _zero_based_emnist_letters

        self.assertEqual(_zero_based_emnist_letters(1), 0)
        self.assertEqual(_zero_based_emnist_letters(26), 25)
        valid = FrameworkConfig.from_dict(
            {
                "model": {"name": "lenet5", "weights": "random", "device": "cpu", "num_classes": 47},
                "dataset": {"name": "emnist", "emnist_split": "balanced"},
            }
        )
        self.assertEqual(valid.model.num_classes, 47)
        with self.assertRaisesRegex(ConfigValidationException, "requires model.num_classes=47"):
            FrameworkConfig.from_dict(
                {
                    "model": {"name": "lenet5", "weights": "random", "device": "cpu"},
                    "dataset": {"name": "emnist", "emnist_split": "balanced"},
                }
            )


if __name__ == "__main__":
    unittest.main()
