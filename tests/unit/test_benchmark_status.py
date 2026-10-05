"""Status classification boundaries for artifact benchmark summaries."""

import unittest

from prune_framework.pipelines.unified import _benchmark_summary


class TestBenchmarkStatus(unittest.TestCase):
    def test_classification_status_uses_accuracy_delta_and_inclusive_boundaries(self):
        for drop, expected in ((0.01, "PASS"), (0.03, "GOOD"), (0.10, "DEGRADED"), (0.30, "SEVERE"), (0.300001, "FAILED")):
            with self.subTest(drop=drop):
                summary = _benchmark_summary(
                    {"accuracy": 1.0, "loss": 0.5},
                    {"accuracy": 1.0 - drop, "loss": 0.5},
                    complexity=None,
                    requested_sparsity=0.1,
                    diagnostics={"actual_sparsity": 0.1},
                )
                self.assertAlmostEqual(summary["accuracy_delta"], -drop)
                self.assertEqual(summary["status"], expected)

    def test_classification_collapse_is_not_misclassified_as_pass(self):
        summary = _benchmark_summary(
            {"accuracy": 0.84}, {"accuracy": 0.01}, None, 0.1, {"actual_sparsity": 0.1}
        )
        self.assertAlmostEqual(summary["accuracy_delta"], -0.83)
        self.assertEqual(summary["status"], "FAILED")
