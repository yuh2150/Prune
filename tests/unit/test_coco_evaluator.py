"""Real COCO matching on tiny annotations without model weights or downloads."""
import contextlib
import io
import unittest

from pycocotools.coco import COCO

from prune_framework.contracts.evaluation import Detection
from prune_framework.modules.evaluation.coco_evaluator import COCOEvaluator


class TestCOCOEvaluator(unittest.TestCase):
    def setUp(self):
        self.coco = COCO()
        self.coco.dataset = {
            "info": {},
            "images": [{"id": i, "height": 100, "width": 100} for i in (1, 2, 3)],
            "categories": [{"id": i, "name": str(i)} for i in (1, 2, 3)],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 10, 10], "area": 100, "iscrowd": 0},
                {"id": 2, "image_id": 1, "category_id": 1, "bbox": [50, 50, 10, 10], "area": 100, "iscrowd": 0},
                {"id": 3, "image_id": 1, "category_id": 1, "bbox": [80, 80, 10, 10], "area": 100, "iscrowd": 1},
                {"id": 4, "image_id": 2, "category_id": 2, "bbox": [10, 10, 10, 10], "area": 100, "iscrowd": 0},
            ],
        }
        with contextlib.redirect_stdout(io.StringIO()):
            self.coco.createIndex()
        self.predictions = [
            Detection(1, 1, .9, (10, 10, 10, 10)),
            Detection(1, 1, .8, (10, 10, 10, 10)),  # duplicate: false positive
            Detection(1, 1, .7, (30, 30, 10, 10)),
            Detection(1, 1, .1, (50, 50, 10, 10)),  # below P/R threshold
            Detection(1, 1, .99, (80, 80, 10, 10)),  # crowd: ignored
            Detection(2, 2, .95, (10, 10, 10, 8)),  # IoU .8
            Detection(3, 1, .6, (10, 10, 10, 10)),  # background image: false positive
            Detection(3, 3, .9, (10, 10, 10, 10)),  # class without GT excluded from macro
        ]

    def evaluate(self, threshold=.25, predictions=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return COCOEvaluator(self.coco, precision_recall_conf_thres=threshold).evaluate(
                self.predictions if predictions is None else predictions, [1, 2, 3, 1])

    def test_precision_recall_are_matched_counts_not_ap_or_average_recall(self):
        result = self.evaluate()
        self.assertAlmostEqual(result.metrics["precision"], .625)
        self.assertAlmostEqual(result.metrics["recall"], .75)
        self.assertEqual(result.metrics["map50_95"], result.metrics["map"])
        self.assertGreater(result.metrics["map50"], result.metrics["map50_95"])
        self.assertEqual(result.num_samples, 3)
        self.assertEqual(result.metadata["precision_recall_conf_thres"], .25)

    def test_confidence_threshold_changes_pr_but_leaves_ap_unchanged(self):
        high, low = self.evaluate(), self.evaluate(.05)
        self.assertAlmostEqual(low.metrics["precision"], .7)
        self.assertAlmostEqual(low.metrics["recall"], 1.)
        self.assertEqual(high.metrics["map50"], low.metrics["map50"])
        self.assertEqual(high.metrics["map50_95"], low.metrics["map50_95"])

    def test_empty_predictions_return_all_quality_metrics(self):
        result = self.evaluate(predictions=[])
        for key in ("precision", "recall", "map50", "map50_95"):
            self.assertEqual(result.metrics[key], 0.)

    def test_evaluation_failure_is_not_reported_as_zero_quality(self):
        with self.assertRaisesRegex(RuntimeError, "COCO evaluation failed"):
            COCOEvaluator(object()).evaluate(self.predictions, [1])

    def test_invalid_threshold_rejected(self):
        for threshold in (-.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                COCOEvaluator(self.coco, precision_recall_conf_thres=threshold)
