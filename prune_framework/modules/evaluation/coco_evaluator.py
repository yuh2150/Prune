import os
import json
import tempfile
import math
from collections import defaultdict
import numpy as np
from typing import List, Dict, Any, Optional
from prune_framework.contracts.evaluation import BaseEvaluator, Detection, EvaluationResult
from prune_framework.core.logging import get_logger

logger = get_logger("prune_framework.evaluator")


class COCOEvaluator(BaseEvaluator):
    """COCO AP plus macro P/R at IoU 0.5 and a fixed confidence threshold."""

    def __init__(self, coco_gt: Any, save_json_path: Optional[str] = None,
                 precision_recall_conf_thres: float = 0.25):
        if not math.isfinite(precision_recall_conf_thres) or not 0 <= precision_recall_conf_thres <= 1:
            raise ValueError("Precision/recall confidence threshold must be between 0 and 1.")
        self.coco_gt = coco_gt
        self.save_json_path = save_json_path
        self.precision_recall_conf_thres = precision_recall_conf_thres

    def evaluate(
        self,
        predictions: List[Detection],
        evaluated_img_ids: List[int],
        **kwargs: Any
    ) -> EvaluationResult:
        evaluated_img_ids = sorted(set(evaluated_img_ids))
        metadata = {"precision_recall_iou": 0.5,
                    "precision_recall_conf_thres": self.precision_recall_conf_thres,
                    "precision_recall_average": "macro_over_ground_truth_classes",
                    "max_detections": 100}
        if not predictions:
            logger.warning("COCOEvaluator received 0 predictions.")
            return EvaluationResult(
                metrics={"precision": 0.0, "recall": 0.0, "map": 0.0,
                         "map50": 0.0, "map50_95": 0.0, "map75": 0.0, "mar": 0.0},
                num_samples=len(evaluated_img_ids),
                metadata={**metadata, "status": "no_predictions"}
            )

        jdict = [
            {
                "image_id": d.image_id,
                "category_id": d.category_id,
                "bbox": list(d.bbox),
                "score": d.score
            }
            for d in predictions
        ]

        if self.save_json_path:
            json_file = self.save_json_path
            os.makedirs(os.path.dirname(os.path.abspath(json_file)), exist_ok=True)
            with open(json_file, "w") as f:
                json.dump(jdict, f)
        else:
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                json.dump(jdict, f)
                json_file = f.name

        try:
            from pycocotools.cocoeval import COCOeval

            pred_coco = self.coco_gt.loadRes(json_file)
            coco_eval = COCOeval(self.coco_gt, pred_coco, "bbox")
            coco_eval.params.imgIds = list(set(evaluated_img_ids))

            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            stats = coco_eval.stats
            precision, recall = _precision_recall(coco_eval, self.precision_recall_conf_thres)
            metrics = {
                "precision": precision,
                "recall": recall,
                "map": float(stats[0]),
                "map50_95": float(stats[0]),
                "map50": float(stats[1]),
                "map75": float(stats[2]),
                "mar": float(stats[8])
            }
            return EvaluationResult(
                metrics=metrics,
                num_samples=len(evaluated_img_ids),
                metadata={**metadata, "stats": [float(s) for s in stats]}
            )
        except Exception as e:
            logger.error(f"pycocotools COCOeval failed: {e}")
            raise RuntimeError(f"COCO evaluation failed: {e}") from e
        finally:
            if not self.save_json_path and os.path.exists(json_file):
                try:
                    os.remove(json_file)
                except OSError:
                    pass


def _precision_recall(coco_eval, confidence_threshold: float):
    """Count class-wise TP/FP from COCO matches, excluding crowd/ignored boxes.

    Use only the all-area records; other area ranges would duplicate detections.
    AP remains integrated over confidences and is computed separately by COCO.
    """
    iou_index = int(np.flatnonzero(np.isclose(coco_eval.params.iouThrs, 0.5))[0])
    all_area = coco_eval.params.areaRng[coco_eval.params.areaRngLbl.index("all")]
    counts = defaultdict(lambda: [0, 0, 0])  # true positives, false positives, targets
    for record in coco_eval.evalImgs:
        if record is None or list(record["aRng"]) != list(all_area):
            continue
        totals = counts[record["category_id"]]
        matches = np.asarray(record["dtMatches"])[iou_index] > 0
        ignored = np.asarray(record["dtIgnore"])[iou_index].astype(bool)
        selected = (np.asarray(record["dtScores"]) >= confidence_threshold) & ~ignored
        totals[0] += int(np.count_nonzero(selected & matches))
        totals[1] += int(np.count_nonzero(selected & ~matches))
        totals[2] += int(np.count_nonzero(~np.asarray(record["gtIgnore"]).astype(bool)))
    classes = [values for values in counts.values() if values[2] > 0]
    if not classes:
        return 0.0, 0.0
    precision = sum(tp / max(tp + fp, 1) for tp, fp, _ in classes) / len(classes)
    recall = sum(tp / targets for tp, _, targets in classes) / len(classes)
    return precision, recall
