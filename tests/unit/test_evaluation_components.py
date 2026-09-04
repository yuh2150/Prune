import unittest
import torch
import torch.nn as nn
from prune_framework.contracts.evaluation import (
    Detection,
    EvaluationResult,
    BasePreProcessor,
    BasePostProcessor,
    BaseEvaluator,
)
from prune_framework.modules.evaluation.pipeline import EvaluationPipeline


class DummyPreProcessor(BasePreProcessor):
    def process(self, images, device=None, half=False, **kwargs):
        tensor = torch.zeros((len(images), 3, 640, 640), device=device)
        if half:
            tensor = tensor.half()
        return {"pixel_values": tensor}


class DummyPostProcessor(BasePostProcessor):
    def process(self, outputs, target_sizes, image_ids, threshold=0.001, **kwargs):
        detections = []
        for img_id in image_ids:
            detections.append(
                Detection(
                    image_id=img_id,
                    category_id=1,
                    score=0.95,
                    bbox=(10.0, 10.0, 50.0, 50.0)
                )
            )
        return detections


class DummyEvaluator(BaseEvaluator):
    def evaluate(self, predictions, evaluated_img_ids, **kwargs):
        return EvaluationResult(
            metrics={"map": 0.85, "map50": 0.92, "map75": 0.88, "mar": 0.90},
            num_samples=len(evaluated_img_ids),
            metadata={"predictions_count": len(predictions)}
        )


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)

    def forward(self, pixel_values):
        return {"logits": torch.zeros((pixel_values.shape[0], 100, 80)), "boxes": torch.zeros((pixel_values.shape[0], 100, 4))}


class TestEvaluationComponents(unittest.TestCase):
    def test_evaluation_contracts(self):
        det = Detection(image_id=101, category_id=1, score=0.9, bbox=(0.0, 0.0, 10.0, 10.0))
        self.assertEqual(det.image_id, 101)

        res = EvaluationResult(metrics={"map": 0.80, "map50": 0.90}, num_samples=10)
        self.assertEqual(res.map, 0.80)
        self.assertEqual(res.map50, 0.90)
        self.assertEqual(res.get("map75", 0.0), 0.0)

    def test_evaluation_pipeline_flow(self):
        prep = DummyPreProcessor()
        post = DummyPostProcessor()
        evaluator = DummyEvaluator()

        pipeline = EvaluationPipeline(prep, post, evaluator)

        dummy_dataloader = [
            (
                [torch.zeros((3, 640, 640))],
                [{"height": 640, "width": 640, "image_id": 1}]
            ),
            (
                [torch.zeros((3, 640, 640))],
                [{"height": 640, "width": 640, "image_id": 2}]
            )
        ]

        model = DummyModel()
        device = torch.device("cpu")

        result = pipeline.run(model, dummy_dataloader, device, half=False, show_progress=False)

        self.assertIsInstance(result, EvaluationResult)
        self.assertEqual(result.map, 0.85)
        self.assertEqual(result.num_samples, 2)
        self.assertEqual(result.metadata["predictions_count"], 2)


if __name__ == "__main__":
    unittest.main()
