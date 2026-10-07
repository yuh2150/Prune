"""Classification loss/accuracy evaluator is independent of detection outputs."""

import unittest

import torch
import torch.nn as nn
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

    def test_macro_precision_recall_aggregate_batches_and_exclude_unobserved_classes(self):
        model = nn.Linear(4, 4, bias=False).train()
        with torch.no_grad():
            model.weight.copy_(torch.eye(4))
        loader = DataLoader(TensorDataset(torch.eye(4)[[0, 0, 1, 2]],
                                         torch.tensor([0, 0, 0, 1])), batch_size=2)
        result = evaluate_classification(model, loader, "cpu")
        self.assertAlmostEqual(result.metrics["accuracy"], 0.5)
        self.assertAlmostEqual(result.metrics["precision"], 1 / 3)
        self.assertAlmostEqual(result.metrics["recall"], 2 / 9)
        self.assertAlmostEqual(result.metrics["f1"], 4 / 15)
        self.assertEqual(result.metadata["average"], "macro")
        self.assertNotIn("map50", result.metrics)
        self.assertTrue(model.training)

    def test_macro_f1_is_not_harmonic_mean_of_macro_precision_recall(self):
        model = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.eye(2))
        loader = DataLoader(TensorDataset(torch.eye(2)[[0, 1, 1, 1]],
                                         torch.tensor([0, 0, 0, 1])), batch_size=2)
        result = evaluate_classification(model, loader, "cpu")
        self.assertAlmostEqual(result.metrics["f1"], 0.5)
        p, r = result.metrics["precision"], result.metrics["recall"]
        self.assertNotAlmostEqual(result.metrics["f1"], 2 * p * r / (p + r))


if __name__ == "__main__":
    unittest.main()
