import copy
import unittest

import torch

from prune_framework.models import LeNet5EMNIST
from tools.reevaluate_artifact_quality import augment_result, replay_plan, verify_metrics


class TestArtifactQualityReevaluation(unittest.TestCase):
    def test_replay_uses_saved_indices_without_reselecting_weights(self):
        model = LeNet5EMNIST()
        original = model.conv1.conv.weight.detach().clone()
        description = {"pruner": "unstructured", "groups": [{
            "primary": {"name": "conv1.conv", "target_type": "conv_weight"},
            "indices": [0, 3, 149], "operation": "mask_weight", "validated": True,
        }]}
        replay_plan(model, "lenet5_emnist_onnx", description)
        expected = original.clone().flatten()
        expected[[0, 3, 149]] = 0
        torch.testing.assert_close(model.conv1.conv.weight.flatten(), expected)

    def test_rejects_mismatch_before_updating_artifact(self):
        with self.assertRaisesRegex(ValueError, "accuracy mismatch"):
            verify_metrics({"accuracy": .84, "loss": .4}, {"accuracy": .83, "loss": .4}, "final")
        with self.assertRaisesRegex(ValueError, "historical accuracy is absent"):
            verify_metrics(None, {"accuracy": .84}, "baseline")

    def test_augmentation_preserves_old_scores_and_updates_replicas(self):
        old = {"baseline_metrics": {"accuracy": .84, "loss": .400001},
               "final_metrics": {"accuracy": .83, "loss": .410001},
               "pruning": {"benchmark": {"fps": 123}, "extra_metrics": {"stages": {
                   "baseline_evaluation": {"accuracy": .84}, "final_evaluation": {"accuracy": .83}}}},
               "stages": {"final_evaluation": {"accuracy": .83}}}
        baseline = {"precision": .9, "recall": .8, "f1": .85}
        final = {"precision": .8, "recall": .7, "f1": .75}
        result = augment_result(copy.deepcopy(old), baseline, final, {"num_samples": 200})
        self.assertEqual(result["final_metrics"]["loss"], old["final_metrics"]["loss"])
        self.assertEqual(result["pruning"]["benchmark"]["fps"], 123)
        self.assertEqual(result["pruning"]["extra_metrics"]["stages"]["final_evaluation"]["f1"], .75)
        self.assertEqual(result["stages"]["final_evaluation"]["f1"], .75)
        self.assertAlmostEqual(result["f1_delta"], -.1)


if __name__ == "__main__":
    unittest.main()
