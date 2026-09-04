import unittest
import torch
import torch.nn as nn
from prune_framework.modules.analysis.sensitivity import SensitivityAnalyzer
from prune_framework.contracts.sensitivity_result import SensitivityResult, LayerSensitivityProfile


class SimpleTestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

    def forward(self, x):
        return self.conv2(self.bn1(self.conv1(x)))


class TestSensitivityAnalyzer(unittest.TestCase):
    def setUp(self):
        self.model = SimpleTestNet()
        self.analyzer = SensitivityAnalyzer(
            model_name="yolov5",
            pruner_name="structured",
            criterion_name="l1",
            granularity_name="channel"
        )

    def test_normal_sensitivity_analysis(self):
        def eval_fn(m):
            return 0.85

        result = self.analyzer.analyze(
            model=self.model,
            pruning_rates=[0.1, 0.2, 0.5],
            eval_fn=eval_fn
        )

        self.assertIsInstance(result, SensitivityResult)
        self.assertEqual(result.baseline_score, 0.85)
        self.assertEqual(len(result.profiles), 2)  # conv1, conv2

        profile0 = result.profiles[0]
        self.assertEqual(len(profile0.points), 3)
        self.assertTrue(profile0.points[0.1].is_valid)
        self.assertEqual(profile0.points[0.1].score, 0.85)
        self.assertAlmostEqual(profile0.points[0.1].metric_drop, 0.0)
        self.assertAlmostEqual(profile0.points[0.1].relative_drop, 0.0)

    def test_evaluation_failure_handling(self):
        call_count = 0

        def failing_eval_fn(m):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise RuntimeError("Evaluation GPU OOM error")
            return 0.90

        result = self.analyzer.analyze(
            model=self.model,
            pruning_rates=[0.1, 0.5],
            eval_fn=failing_eval_fn
        )

        self.assertEqual(result.baseline_score, 0.90)
        profile0 = result.profiles[0]

        # Second eval call should fail gracefully
        pt_failing = profile0.points[0.1]
        self.assertFalse(pt_failing.is_valid)
        self.assertEqual(pt_failing.score, 0.0)
        self.assertEqual(pt_failing.error_message, "Evaluation GPU OOM error")

    def test_empty_pruning_rates(self):
        def eval_fn(m):
            return 0.75

        result = self.analyzer.analyze(
            model=self.model,
            pruning_rates=[],
            eval_fn=eval_fn
        )

        self.assertEqual(result.baseline_score, 0.75)
        for idx, profile in result.profiles.items():
            self.assertEqual(len(profile.points), 0)

    def test_to_dict_and_from_dict_conversion(self):
        def eval_fn(m):
            return 0.80

        result = self.analyzer.analyze(
            model=self.model,
            pruning_rates=[0.25],
            eval_fn=eval_fn
        )

        raw_dict = result.to_dict()
        self.assertIn(0, raw_dict)
        self.assertIn(0.25, raw_dict[0])
        self.assertEqual(raw_dict[0][0.25], 0.80)

        reconstructed = SensitivityResult.from_dict(raw_dict, baseline_score=0.80)
        self.assertEqual(reconstructed.baseline_score, 0.80)
        self.assertEqual(reconstructed.profiles[0].points[0.25].score, 0.80)


if __name__ == "__main__":
    unittest.main()
