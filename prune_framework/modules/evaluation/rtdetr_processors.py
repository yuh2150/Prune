import torch
from typing import List, Dict, Any, Optional, Tuple
from transformers import RTDetrImageProcessor
from prune_framework.contracts.evaluation import (
    BasePreProcessor,
    BasePostProcessor,
    Detection,
)


def _coco80_to_coco91_class() -> List[int]:
    """Returns 80-class to 91-class mapping for COCO dataset."""
    return [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
        35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
    ]


class RTDetrPreProcessor(BasePreProcessor):
    """Pre-processor for RT-DETR object detection model using HuggingFace RTDetrImageProcessor."""

    def __init__(self, hf_model_name: str = "PekingU/rtdetr_r18vd"):
        try:
            self.image_processor = RTDetrImageProcessor.from_pretrained(hf_model_name, local_files_only=True)
        except Exception:
            self.image_processor = RTDetrImageProcessor.from_pretrained(hf_model_name, local_files_only=False)

    def process(self, images: List[Any], device: Optional[torch.device] = None, half: bool = False, **kwargs: Any) -> Dict[str, torch.Tensor]:
        inputs = self.image_processor(images=images, return_tensors="pt")
        if device is not None:
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if half and "pixel_values" in inputs and isinstance(inputs["pixel_values"], torch.Tensor):
            inputs["pixel_values"] = inputs["pixel_values"].half()
        return inputs


class RTDetrPostProcessor(BasePostProcessor):
    """Post-processor for RT-DETR outputs converting logits & boxes into standardized Detection objects."""

    def __init__(self, hf_model_name: str = "PekingU/rtdetr_r18vd"):
        try:
            self.image_processor = RTDetrImageProcessor.from_pretrained(hf_model_name, local_files_only=True)
        except Exception:
            self.image_processor = RTDetrImageProcessor.from_pretrained(hf_model_name, local_files_only=False)
        self.coco91class = _coco80_to_coco91_class()


    def process(
        self,
        outputs: Any,
        target_sizes: torch.Tensor,
        image_ids: List[int],
        threshold: float = 0.001,
        **kwargs: Any
    ) -> List[Detection]:
        results = self.image_processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=threshold
        )

        detections: List[Detection] = []
        for si, res in enumerate(results):
            image_id = image_ids[si]
            boxes = res["boxes"].tolist()
            scores = res["scores"].tolist()
            labels = res["labels"].tolist()

            for box, score, label in zip(boxes, scores, labels):
                x_min, y_min, x_max, y_max = box
                w_box = x_max - x_min
                h_box = y_max - y_min
                cat_id = self.coco91class[label] if label < len(self.coco91class) else label

                detections.append(
                    Detection(
                        image_id=image_id,
                        category_id=cat_id,
                        score=round(score, 5),
                        bbox=(round(x_min, 3), round(y_min, 3), round(w_box, 3), round(h_box, 3))
                    )
                )

        return detections
