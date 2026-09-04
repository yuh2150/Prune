import torch
import torch.nn as nn
from typing import Any, List, Optional
from tqdm import tqdm
from prune_framework.contracts.evaluation import (
    BasePreProcessor,
    BasePostProcessor,
    BaseEvaluator,
    Detection,
    EvaluationResult,
)
from prune_framework.core.logging import get_logger

logger = get_logger("prune_framework.pipeline")


class EvaluationPipeline:
    """
    Decoupled Evaluation Pipeline composing PreProcessor, Model Inference,
    PostProcessor, and Evaluator.
    """

    def __init__(
        self,
        preprocessor: BasePreProcessor,
        postprocessor: BasePostProcessor,
        evaluator: BaseEvaluator
    ):
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.evaluator = evaluator

    def run(
        self,
        model: nn.Module,
        dataloader: Any,
        device: torch.device,
        conf_thres: float = 0.001,
        half: bool = True,
        show_progress: bool = True
    ) -> EvaluationResult:
        model.eval()
        if half and device.type != "cpu":
            model.half()

        all_detections: List[Detection] = []
        evaluated_img_ids: List[int] = []

        iterator = tqdm(dataloader, desc="Evaluating Pipeline") if show_progress else dataloader

        for batch_data in iterator:
            if batch_data is None:
                continue
            imgs, targets = batch_data

            # 1. Pre-process batch
            inputs = self.preprocessor.process(images=imgs, device=device, half=half)

            # 2. Model Inference
            with torch.no_grad():
                outputs = model(**inputs)

            # 3. Post-process predictions into Detection objects
            original_sizes = [(t["height"], t["width"]) for t in targets]
            target_sizes = torch.tensor(original_sizes, device=device)
            image_ids = [t["image_id"] for t in targets]

            batch_detections = self.postprocessor.process(
                outputs=outputs,
                target_sizes=target_sizes,
                image_ids=image_ids,
                threshold=conf_thres
            )

            all_detections.extend(batch_detections)
            evaluated_img_ids.extend(image_ids)

        # 4. Evaluate metrics
        result = self.evaluator.evaluate(
            predictions=all_detections,
            evaluated_img_ids=evaluated_img_ids
        )
        return result
