import os
import json
import tempfile
from typing import List, Dict, Any, Optional
from prune_framework.contracts.evaluation import BaseEvaluator, Detection, EvaluationResult
from prune_framework.core.logging import get_logger

logger = get_logger("prune_framework.evaluator")


class COCOEvaluator(BaseEvaluator):
    """Evaluator that calculates standard COCO mAP metrics using pycocotools."""

    def __init__(self, coco_gt: Any, save_json_path: Optional[str] = None):
        self.coco_gt = coco_gt
        self.save_json_path = save_json_path

    def evaluate(
        self,
        predictions: List[Detection],
        evaluated_img_ids: List[int],
        **kwargs: Any
    ) -> EvaluationResult:
        if not predictions:
            logger.warning("COCOEvaluator received 0 predictions.")
            return EvaluationResult(
                metrics={"map": 0.0, "map50": 0.0, "map75": 0.0, "mar": 0.0},
                num_samples=len(evaluated_img_ids),
                metadata={"status": "no_predictions"}
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
            metrics = {
                "map": float(stats[0]),
                "map50": float(stats[1]),
                "map75": float(stats[2]),
                "mar": float(stats[8])
            }
            return EvaluationResult(
                metrics=metrics,
                num_samples=len(evaluated_img_ids),
                metadata={"stats": [float(s) for s in stats]}
            )
        except Exception as e:
            logger.error(f"pycocotools COCOeval failed: {e}")
            return EvaluationResult(
                metrics={"map": 0.0, "map50": 0.0, "map75": 0.0, "mar": 0.0},
                num_samples=len(evaluated_img_ids),
                metadata={"error": str(e)}
            )
        finally:
            if not self.save_json_path and os.path.exists(json_file):
                try:
                    os.remove(json_file)
                except OSError:
                    pass
