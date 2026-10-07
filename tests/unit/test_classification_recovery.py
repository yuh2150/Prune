"""Pruned LeNet recovery and checkpoint persistence without downloaded data."""

import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.classification import recover
from prune_framework.core.config import FrameworkConfig
from prune_framework.core.engine import PruningEngine
from prune_framework.models import LeNet5
from prune_framework.modules.model.masks import MaskManager
from prune_framework.modules.model.loader import ModelLoader
from prune_framework.modules.export.exporter import ModelExporter
from prune_framework.modules.evaluation.classification import evaluate_classification
from prune_framework.pipelines.unified import UnifiedPruningPipeline


def _loader():
    return DataLoader(TensorDataset(torch.randn(12, 1, 28, 28), torch.randint(0, 10, (12,))), batch_size=4)


def _config(directory: str, *, recovery: bool = True):
    return FrameworkConfig.from_dict(
        {
            "model": {"name": "lenet5", "weights": "random", "device": "cpu", "input_shape": [1, 1, 28, 28]},
            "dataset": {"name": "mnist", "root": str(Path(directory) / "missing_data"), "batch_size": 4, "train_limit": 12, "val_limit": 12},
            "pruning": {"method": "unstructured", "criterion": "magnitude", "structure": "weight", "target_ratio": 0.1},
            "evaluation": {"enabled": True, "metric": "accuracy"},
            "recovery": {"enabled": recovery, "epochs": 1, "optimizer": {"name": "sgd", "lr": 0.01}},
            "benchmark": {"enabled": False, "latency": False, "flops": False, "params": False, "runs": 1},
            "export": {"enabled": False},
            "experiment": {"name": "lenet_synthetic", "output_dir": directory, "seed": 23},
            "output_path": str(Path(directory) / "lenet.pt"),
        }
    )


class TestClassificationRecovery(unittest.TestCase):
    def test_exported_masked_checkpoint_is_materialized_and_reloadable(self):
        with tempfile.TemporaryDirectory() as directory:
            model = LeNet5()
            PruningEngine("lenet5", "unstructured", "magnitude", "weight").execute(
                model, {"amount": 0.1, "criterion_name": "magnitude"}
            )
            inputs = torch.randn(2, 1, 28, 28)
            expected = model.eval()(inputs)
            path = str(Path(directory) / "exported.pt")

            ModelExporter.export_checkpoint(model, path)

            payload = torch.load(path, weights_only=False)
            self.assertFalse(MaskManager.has_mask(payload["model"].features[0]))
            restored, _ = ModelLoader.load("lenet5", path, torch.device("cpu"))
            torch.testing.assert_close(restored.eval()(inputs), expected)

    def test_recovery_keeps_masks_and_adapter_reloads_serialized_mask_state(self):
        with tempfile.TemporaryDirectory() as directory:
            model = LeNet5()
            PruningEngine("lenet5", "unstructured", "magnitude", "weight").execute(
                model, {"amount": 0.1, "criterion_name": "magnitude"}
            )
            config = _config(directory)
            result = recover(model, config, "cpu", dataloader=_loader(), max_batches=2)
            self.assertEqual(result["metrics"]["recovery_steps"], 2)
            masked = model.features[0]
            mask = MaskManager.mask(masked)
            original = MaskManager.original_weight(masked)
            self.assertTrue(torch.equal(original[mask == 0], torch.zeros_like(original[mask == 0])))
            torch.save(model.state_dict(), config.output_path)
            restored, _ = ModelLoader.load("lenet5", config.output_path, torch.device("cpu"))
            self.assertTrue(torch.equal(MaskManager.mask(restored.features[0]), mask))
            inputs = torch.randn(2, 1, 28, 28)
            self.assertTrue(torch.equal(model.eval()(inputs), restored.eval()(inputs)))

    def test_unified_synthetic_prune_recover_checkpoint_reload_flow(self):
        with tempfile.TemporaryDirectory() as directory:
            config = _config(directory)
            train_loader, val_loader = _loader(), _loader()

            def evaluator(model, stage):
                self.assertIn(stage, {"baseline", "final"})
                return evaluate_classification(model, val_loader, "cpu")

            def recovery(model, config, device, stage):
                return recover(model, config, device, stage, dataloader=train_loader, max_batches=2)

            result = UnifiedPruningPipeline(config, evaluator=evaluator, recovery=recovery).run()
            self.assertTrue(result.pruning.forward_verified)
            self.assertIn("recovery", result.artifacts)
            self.assertTrue(Path(config.output_path).is_file())
            restored, _ = ModelLoader.load("lenet5", config.output_path, torch.device("cpu"))
            reloaded = evaluate_classification(restored, val_loader, "cpu")
            self.assertEqual(reloaded.num_samples, 12)


if __name__ == "__main__":
    unittest.main()
