import unittest
from prune_framework.modules.analysis.layer_selection import LayerSelectorModule
from prune_framework.contracts.sensitivity_result import (
    SensitivityResult,
    LayerSensitivityProfile,
    SensitivityPoint,
    SelectionResult,
)


class TestLayerSelectors(unittest.TestCase):
    def setUp(self):
        # Construct synthetic sensitivity result with baseline 0.80
        # Layer 0: robust (rate 0.1 -> 0.79, 0.2 -> 0.78, 0.3 -> 0.76)
        # Layer 1: sensitive (rate 0.1 -> 0.70, 0.2 -> 0.40, 0.3 -> 0.10)
        self.baseline = 0.80
        profiles = {
            0: LayerSensitivityProfile(
                layer_idx=0,
                layer_name="conv1",
                points={
                    0.1: SensitivityPoint(rate=0.1, score=0.79, metric_drop=0.01, relative_drop=0.0125, is_valid=True),
                    0.2: SensitivityPoint(rate=0.2, score=0.78, metric_drop=0.02, relative_drop=0.025, is_valid=True),
                    0.3: SensitivityPoint(rate=0.3, score=0.76, metric_drop=0.04, relative_drop=0.050, is_valid=True),
                }
            ),
            1: LayerSensitivityProfile(
                layer_idx=1,
                layer_name="conv2",
                points={
                    0.1: SensitivityPoint(rate=0.1, score=0.70, metric_drop=0.10, relative_drop=0.125, is_valid=True),
                    0.2: SensitivityPoint(rate=0.2, score=0.40, metric_drop=0.40, relative_drop=0.500, is_valid=True),
                    0.3: SensitivityPoint(rate=0.3, score=0.10, metric_drop=0.70, relative_drop=0.875, is_valid=True),
                }
            )
        }
        self.sensitivity_result = SensitivityResult(baseline_score=self.baseline, profiles=profiles)

    def test_sensitivity_selector_degradation_threshold(self):
        selector_mod = LayerSelectorModule("sensitivity")
        # Allowed relative drop 5% (0.05).
        # Layer 0: rate 0.3 has rel drop 0.05 (<= 0.05), so rate 0.3 selected.
        # Layer 1: rate 0.1 has rel drop 0.125 (> 0.05), so no rate passes threshold -> rate 0.0 selected.
        res = selector_mod.select(self.sensitivity_result, target_sparsity=0.5, max_allowed_relative_drop=0.05)

        self.assertIsInstance(res, SelectionResult)
        rates = dict(res)
        self.assertEqual(rates[0], 0.3)
        self.assertEqual(rates[1], 0.0)

    def test_sensitivity_selector_target_sparsity_cap(self):
        selector_mod = LayerSelectorModule("sensitivity")
        # Target sparsity capped at 0.2
        res = selector_mod.select(self.sensitivity_result, target_sparsity=0.2, max_allowed_relative_drop=0.05)
        rates = dict(res)
        self.assertEqual(rates[0], 0.2)
        self.assertEqual(rates[1], 0.0)

    def test_greedy_selector(self):
        selector_mod = LayerSelectorModule("greedy")
        res = selector_mod.select(self.sensitivity_result, target_sparsity=0.3)
        rates = dict(res)
        # Layer 0: rate 0.1 has highest metric score 0.79 among candidate rates
        self.assertEqual(rates[0], 0.1)
        # Layer 1: rate 0.1 has score 0.70
        self.assertEqual(rates[1], 0.1)

    def test_threshold_selector_absolute_drop(self):
        selector_mod = LayerSelectorModule("threshold")
        # Max allowed absolute drop = 0.03
        res = selector_mod.select(self.sensitivity_result, target_sparsity=0.3, max_allowed_drop=0.03, max_allowed_relative_drop=None)
        rates = dict(res)
        # Layer 0: rate 0.2 (drop 0.02 <= 0.03), rate 0.3 (drop 0.04 > 0.03) -> 0.2 selected
        self.assertEqual(rates[0], 0.2)
        self.assertEqual(rates[1], 0.0)

    def test_legacy_dict_input_and_tuple_iteration(self):
        raw_dict = {
            0: {0.1: 0.79, 0.2: 0.78},
            1: {0.1: 0.70, 0.2: 0.40}
        }
        selector_mod = LayerSelectorModule("sensitivity")
        res = selector_mod.select(raw_dict, target_sparsity=0.5, max_allowed_relative_drop=0.05)

        # Test tuple iteration
        tuples = list(res)
        self.assertEqual(len(tuples), 2)
        self.assertEqual(tuples[0][0], 0)

        # Test index access
        self.assertEqual(res[0], tuples[0])

    def test_empty_profiles(self):
        empty_res = SensitivityResult(baseline_score=0.80, profiles={})
        selector_mod = LayerSelectorModule("sensitivity")
        res = selector_mod.select(empty_res, target_sparsity=0.5)
        self.assertEqual(len(res), 0)

    def test_invalid_evaluation_points_ignored(self):
        profiles = {
            0: LayerSensitivityProfile(
                layer_idx=0,
                layer_name="conv1",
                points={
                    0.1: SensitivityPoint(rate=0.1, score=0.0, metric_drop=0.8, relative_drop=1.0, is_valid=False, error_message="GPU error"),
                    0.2: SensitivityPoint(rate=0.2, score=0.79, metric_drop=0.01, relative_drop=0.0125, is_valid=True)
                }
            )
        }
        invalid_res = SensitivityResult(baseline_score=0.80, profiles=profiles)
        selector_mod = LayerSelectorModule("sensitivity")
        res = selector_mod.select(invalid_res, target_sparsity=0.5, max_allowed_relative_drop=0.05)
        rates = dict(res)
        self.assertEqual(rates[0], 0.2)


if __name__ == "__main__":
    unittest.main()
