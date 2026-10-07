"""CPU classification saliency/pruning smoke tests using synthetic labels."""

import unittest

import torch

from experiments.classification import calibrate
from prune_framework.core.config import FrameworkConfig
from prune_framework.core.engine import PruningEngine
from prune_framework.models import LeNet5
from prune_framework.modules.calibration import GradientCalibrationRunner, HigherOrderCalibrationRunner, SynFlowCalibrationRunner
from prune_framework.modules.model.masks import MaskManager
from prune_framework.modules.model.sparsity_patterns import validate_block_sparse_pattern, weight_matrix
from prune_framework.plugins.adapters.lenet5 import LeNet5Adapter


def _batch(size=3):
    return torch.randn(size, 1, 28, 28), torch.randint(0, 10, (size,))


def _loss(model, batch):
    return torch.nn.CrossEntropyLoss()(model(batch[0]), batch[1])


class TestLeNet5PruningSmoke(unittest.TestCase):
    def test_magnitude_snip_lamp_grasp_synflow_and_taylor_build_apply_forward(self):
        cases = ("magnitude", "lamp", "snip", "grasp", "grasp_magnitude_guarded", "synflow")
        for criterion_name in cases:
            with self.subTest(criterion=criterion_name):
                model = LeNet5()
                engine = PruningEngine("lenet5", "unstructured", criterion_name, "weight")
                config = {"amount": 0.03, "criterion_name": criterion_name}
                adapter = LeNet5Adapter(model)
                targets = engine.targets(model)
                if criterion_name == "snip":
                    config["gradient_calibration"] = GradientCalibrationRunner().run(model, targets, [_batch(), _batch()], _loss)
                elif criterion_name.startswith("grasp"):
                    config["higher_order_calibration"] = HigherOrderCalibrationRunner().run(model, targets, [_batch()], _loss)
                elif criterion_name == "synflow":
                    config["synflow_calibration"] = SynFlowCalibrationRunner().run(
                        model, targets, lambda: adapter.get_synflow_input(torch.device("cpu")), adapter.reduce_synflow_output
                    )
                result = engine.execute(model, config, verify_forward=True)
                self.assertTrue(result.forward_verified)
                self.assertEqual(tuple(model(_batch(2)[0]).shape), (2, 10))

    def test_signed_and_abs_snip_calibration_accept_classification_cross_entropy(self):
        model = LeNet5()
        targets = PruningEngine("lenet5", "unstructured", "snip", "weight").targets(model)
        signed = GradientCalibrationRunner(aggregation="signed_mean").run(model, targets, [_batch(), _batch()], _loss)
        absolute = GradientCalibrationRunner(aggregation="abs_mean").run(model, targets, [_batch(), _batch()], _loss)
        self.assertEqual(signed.aggregation, "signed_mean")
        self.assertEqual(absolute.aggregation, "abs_mean")
        self.assertEqual(set(signed.gradients), {target.name for target in targets})

    def test_taylor_structured_calibration_and_conv_flatten_dependency(self):
        model = LeNet5()
        engine = PruningEngine("lenet5", "structured", "taylor", "channel")
        targets = engine.targets(model)
        calibration = GradientCalibrationRunner().run(model, targets, [_batch()], _loss)
        result = engine.execute(
            model,
            {"amount": 0.1, "min_channels": 2, "criterion_name": "taylor", "gradient_calibration": calibration},
            verify_forward=True,
        )
        self.assertTrue(result.forward_verified)
        self.assertEqual(tuple(model(_batch(2)[0]).shape), (2, 10))

    def test_block_sparse_one_by_one_applies_to_full_lenet_and_larger_blocks_fail_clearly(self):
        model = LeNet5()
        result = PruningEngine("lenet5", "block_sparse", "magnitude", "weight").execute(
            model, {"amount": 0.03, "block_shape": [1, 1], "criterion_name": "magnitude"}, verify_forward=True
        )
        self.assertTrue(result.forward_verified)
        self.assertTrue(validate_block_sparse_pattern(weight_matrix(MaskManager.mask(model.features[0])), [1, 1]))
        with self.assertRaisesRegex(ValueError, "divisible"):
            PruningEngine("lenet5", "block_sparse", "magnitude", "weight").build_plan(
                LeNet5(), {"amount": 0.03, "block_shape": [4, 4], "criterion_name": "magnitude"}
            )

    def test_nm_two_of_four_skips_incompatible_lenet_stem_and_prunes_compatible_layers(self):
        plan = PruningEngine("lenet5", "nm", "magnitude", "weight").build_plan(
            LeNet5(), {"n": 2, "m": 4, "criterion_name": "magnitude"}
        )
        self.assertIn(
            {"name": "features.0", "input_width": 25, "m": 4},
            plan.metadata["skipped_incompatible_targets"],
        )
        self.assertTrue(plan.groups)

    def test_classification_calibration_callback_uses_injected_loader(self):
        config = FrameworkConfig.from_dict({"model": {"name": "lenet5", "device": "cpu", "input_shape": [1, 1, 28, 28]}})
        context = calibrate(LeNet5(), config, "cpu", dataloader=[_batch()])
        self.assertEqual(context.sample_count, 3)
        self.assertEqual(context.loss_fn(LeNet5(), _batch()).ndim, 0)


if __name__ == "__main__":
    unittest.main()
