import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from prune_framework.contracts import BaseModelAdapter, TargetType
from prune_framework.core.config import FrameworkConfig
from prune_framework.modules.calibration import SynFlowCalibrationRunner
from prune_framework.modules.model.masks import MaskManager
from prune_framework.pipelines.unified import UnifiedPruningPipeline
from prune_framework.plugins.adapters.rtdetr import RTDETRAdapter
from prune_framework.plugins.adapters.yolov5 import YOLOv5Adapter
from prune_framework.plugins.criteria.synflow_criteria import SynFlowCriterion
from prune_framework.plugins.granularities.base import WeightGranularity
from prune_framework.plugins.pruners.unstructured import UnstructuredPruner


class ConvLinearFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=2, bias=False)
        self.linear = nn.Linear(18, 4, bias=False)
        self.last_input = None

    def forward(self, images):
        self.last_input = images.detach().clone()
        return self.linear(self.conv(images).flatten(1))


class FixtureAdapter(BaseModelAdapter):
    @classmethod
    def load_model(cls, weights_path, device):
        raise NotImplementedError

    def get_pruneable_modules(self):
        return [("conv", self.model.conv), ("linear", self.model.linear)]

    def get_pruneable_blocks(self):
        return []

    def get_dummy_input(self, device):
        return torch.randn(2, 1, 4, 4, device=device)

    def supported_target_types(self):
        return {TargetType.CONV_WEIGHT, TargetType.CONV_OUT_CHANNEL, TargetType.LINEAR_WEIGHT, TargetType.LINEAR_OUT_FEATURE}


class TinyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 12, 3, padding=1))

    def forward(self, images):
        return self.features(images)


class TestSynFlowPruning(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        self.model = ConvLinearFixture()
        self.adapter = FixtureAdapter(self.model)
        self.targets = self.adapter.get_prunable_targets({TargetType.CONV_WEIGHT, TargetType.LINEAR_WEIGHT})

    def _run_synflow(self):
        return SynFlowCalibrationRunner().run(
            self.model,
            self.targets,
            input_factory=lambda: self.adapter.get_synflow_input(torch.device("cpu")),
            output_reducer=self.adapter.reduce_synflow_output,
        )

    def test_signs_state_gradients_modes_and_rng_are_restored_exactly(self):
        with torch.no_grad():
            self.model.conv.weight.copy_(torch.linspace(-1.0, 1.0, self.model.conv.weight.numel()).reshape_as(self.model.conv.weight))
            self.model.linear.weight.copy_(torch.linspace(-2.0, 2.0, self.model.linear.weight.numel()).reshape_as(self.model.linear.weight))
        self.model.eval()
        self.model.conv.weight.grad = torch.full_like(self.model.conv.weight, 3.0)
        parameters_before = {name: parameter.detach().clone() for name, parameter in self.model.named_parameters()}
        state_before = {name: value.detach().clone() for name, value in self.model.state_dict().items()}
        rng_before = torch.random.get_rng_state()

        result = self._run_synflow()

        self.assertTrue(torch.equal(self.model.last_input, torch.ones_like(self.model.last_input)))
        self.assertFalse(self.model.training)
        self.assertTrue(torch.equal(self.model.conv.weight.grad, torch.full_like(self.model.conv.weight, 3.0)))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))
        for name, parameter in self.model.named_parameters():
            self.assertTrue(torch.equal(parameter, parameters_before[name]))
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, state_before[name]))
        for gradient in result.gradients.values():
            self.assertFalse(gradient.requires_grad)
            self.assertIsNone(gradient.grad_fn)

    def test_scores_are_deterministic_for_conv_and_linear(self):
        first = self._run_synflow()
        second = self._run_synflow()
        criterion = SynFlowCriterion()
        for target in self.targets:
            first_score = criterion.score(target.module, first.context_for(target))
            second_score = criterion.score(target.module, second.context_for(target))
            expected = (target.module.weight.detach() * first.gradients[target.name]).abs()
            self.assertTrue(torch.equal(first.gradients[target.name], second.gradients[target.name]))
            self.assertTrue(torch.equal(first_score, expected))
            self.assertTrue(torch.equal(first_score, second_score))
            self.assertEqual(tuple(first_score.shape), tuple(target.module.weight.shape))
            self.assertIsNone(first_score.grad_fn)

    def test_global_plan_has_exact_cardinality_and_masked_forward(self):
        result = self._run_synflow()
        pruner = UnstructuredPruner()
        plan = pruner.create_plan(
            self.adapter,
            SynFlowCriterion(),
            WeightGranularity(),
            {"amount": 0.25, "criterion_name": "synflow", "synflow_calibration": result},
        )
        self.assertEqual(plan.metadata["selection"], "global")
        self.assertEqual(plan.metadata["total_elements"], 80)
        self.assertEqual(plan.metadata["selected_elements"], 20)
        self.assertEqual(sum(len(group.indices) for group in plan.groups), 20)
        self.assertFalse(MaskManager.has_mask(self.model.conv))
        self.assertTrue(pruner.validate_plan(plan, self.adapter))
        pruner.apply_plan(plan, self.adapter)
        self.assertEqual(sum(int((MaskManager.mask(module) == 0).sum()) for module in (self.model.conv, self.model.linear)), 20)
        self.assertEqual(tuple(self.model(torch.ones(2, 1, 4, 4)).shape), (2, 4))

    def test_adapter_reducers_accept_only_known_output_structures(self):
        yolo = YOLOv5Adapter(nn.Identity())
        raw = [torch.ones(1, 3), torch.full((1, 2), 2.0)]
        self.assertEqual(float(yolo.reduce_synflow_output((torch.zeros(1), raw))), 7.0)
        with self.assertRaisesRegex(ValueError, "YOLO SynFlow requires"):
            yolo.reduce_synflow_output({"unknown": torch.ones(1)})

        rtdetr = RTDETRAdapter(nn.Identity())
        output = SimpleNamespace(logits=torch.ones(1, 2), pred_boxes=torch.full((1, 4), 2.0))
        self.assertEqual(float(rtdetr.reduce_synflow_output(output)), 10.0)
        with self.assertRaisesRegex(ValueError, "logits.*pred_boxes"):
            rtdetr.reduce_synflow_output(torch.ones(1))

    def test_yolo_adapter_restores_lazy_detect_grid_state(self):
        class DetectLike(nn.Module):
            def __init__(self):
                super().__init__()
                self.grid = [torch.tensor([1.0])]
                self.anchor_grid = [torch.tensor([2.0])]

        model = nn.Module()
        model.detect = DetectLike()
        adapter = YOLOv5Adapter(model)
        snapshot = adapter.snapshot_synflow_state()
        model.detect.grid = [torch.tensor([10.0])]
        model.detect.anchor_grid = [torch.tensor([20.0])]
        adapter.restore_synflow_state(snapshot)
        self.assertTrue(torch.equal(model.detect.grid[0], torch.tensor([1.0])))
        self.assertTrue(torch.equal(model.detect.anchor_grid[0], torch.tensor([2.0])))

    def test_pipeline_uses_data_free_synflow_without_yolo_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            config = FrameworkConfig.from_dict(
                {
                    "model": {"name": "yolov5", "weights": "unused.pt", "device": "cpu"},
                    "pruning": {"method": "unstructured", "criterion": "synflow", "structure": "weight", "target_ratio": 0.25},
                    "benchmark": {"enabled": False, "latency": False, "flops": False, "params": False, "runs": 1},
                    "export": {"enabled": False},
                    "experiment": {"name": "synflow", "output_dir": directory, "seed": 9},
                    "output_path": str(Path(directory) / "model.pt"),
                }
            )
            with patch("prune_framework.pipelines.unified.ModelLoader.load", return_value=(TinyDetector(), None)), patch.object(
                YOLOv5Adapter, "build_gradient_calibration_loss", side_effect=AssertionError("SynFlow must not use YOLO loss")
            ):
                result = UnifiedPruningPipeline(config).run()
            self.assertTrue(result.pruning.forward_verified)
            self.assertIn("synflow_calibration", result.artifacts)


if __name__ == "__main__":
    unittest.main()
