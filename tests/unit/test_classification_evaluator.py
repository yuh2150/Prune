"""Classification loss/accuracy evaluator is independent of detection outputs."""

import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from prune_framework.models import LeNet5
from prune_framework.modules.evaluation.classification import evaluate_classification


class TestClassificationEvaluator(unittest.TestCase):
    def test_returns_loss_accuracy_samples_and_restores_mode(self):
        model = LeNet5().train()
        loader = DataLoader(TensorDataset(torch.randn(7, 1, 28, 28), torch.randint(0, 10, (7,))), batch_size=3)
        result = evaluate_classification(model, loader, "cpu")
        self.assertTrue(model.training)
        self.assertEqual(result.num_samples, 7)
        self.assertIn("loss", result.metrics)
        self.assertGreaterEqual(result.metrics["accuracy"], 0.0)
        self.assertLessEqual(result.metrics["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
