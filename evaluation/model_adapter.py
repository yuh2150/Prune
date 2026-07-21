"""
Abstract base class and YOLO/DETR concrete implementations for model adapters,
as well as helpers for model loading and profiling.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
import torch
import torch.nn.functional as F
from thop import profile

from models.experimental import attempt_load
from prune import load_pruned_model
from utils.torch_utils import select_device, TracedModel
from utils.general import non_max_suppression


class ModelAdapter(ABC):
    """
    Abstract base class to standardize interface for different object detection models.
    """
    
    @abstractmethod
    def preprocess(self, img, device, half=False):
        """Preprocess batch of images."""
        pass
        
    @abstractmethod
    def forward(self, model, img, augment=False):
        """Perform forward pass on the model."""
        pass
        
    @abstractmethod
    def postprocess(self, outputs, conf_thres, iou_thres, labels=None):
        """Postprocess model outputs (e.g. NMS)."""
        pass
        
    @abstractmethod
    def get_num_classes(self, model):
        """Get the number of target classes."""
        pass

    @abstractmethod
    def get_class_names(self, model):
        """Get mapping of class indices to names."""
        pass


class YOLOAdapter(ModelAdapter):
    """
    Concrete implementation of ModelAdapter for YOLO models.
    """
    
    def preprocess(self, img, device, half=False):
        """
        Preprocess image tensor: move to device, convert to half/float, scale 0-255 to 0.0-1.0.
        """
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()
        img /= 255.0
        return img
        
    def forward(self, model, img, augment=False):
        """
        Run forward pass returning (inference_output, training_output).
        """
        return model(img, augment=augment)
        
    def postprocess(self, outputs, conf_thres, iou_thres, labels=None):
        """
        Run NMS on outputs. Returns list of detections per image.
        """
        inference_out = outputs[0] if isinstance(outputs, tuple) else outputs
        labels = labels if labels is not None else []
        return non_max_suppression(
            inference_out, 
            conf_thres=conf_thres, 
            iou_thres=iou_thres, 
            labels=labels, 
            multi_label=True
        )
        
    def get_num_classes(self, model):
        """
        Get number of classes from model attribute.
        """
        names = self.get_class_names(model)
        return len(names)

    def get_class_names(self, model):
        """
        Get class names dictionary from model or its module.
        """
        if hasattr(model, 'names'):
            return {k: v for k, v in enumerate(model.names)}
        elif hasattr(model, 'module') and hasattr(model.module, 'names'):
            return {k: v for k, v in enumerate(model.module.names)}
        return {}


class DETRAdapter(ModelAdapter):
    """
    Concrete implementation of ModelAdapter for DETR models.
    """
    
    def preprocess(self, img, device, half=False):
        """
        Preprocess image tensor: move to device, convert to half/float.
        """
        img = img.to(device)
        if half:
            img = img.half()
        return img

    def forward(self, model, img, augment=False):
        """
        DETR forward pass. Returns outputs and auxiliary training outputs if available.
        """
        outputs = model(img)
        return outputs, None

    def postprocess(self, outputs, conf_thres, iou_thres, labels=None):
        """
        Postprocess DETR outputs. Normalizes logs and scales bboxes to standard [xyxy, conf, cls] list.
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        
        # Softmax over class dimensions to get probabilities
        prob = F.softmax(out_logits, -1)
        scores, pred_labels = prob[..., :-1].max(-1)  # exclude background class (last slot)
        
        # Convert boxes from normalized cxcywh to normalized xyxy
        # cxcywh to xyxy conversion helper
        x_c, y_c, w, h = out_bbox.unbind(-1)
        boxes_xyxy = torch.stack([
            (x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)
        ], dim=-1)
        
        batch_size = out_logits.shape[0]
        detections = []
        
        for i in range(batch_size):
            img_boxes = boxes_xyxy[i]
            img_scores = scores[i]
            img_labels = pred_labels[i]
            
            # Confidence threshold filter
            keep = img_scores > conf_thres
            
            if keep.sum() > 0:
                det = torch.cat([
                    img_boxes[keep], 
                    img_scores[keep, None], 
                    img_labels[keep, None].float()
                ], dim=1)
                
                # Optional NMS filtering to align with object detectors
                if iou_thres and iou_thres < 1.0:
                    import torchvision
                    keep_nms = torchvision.ops.nms(det[:, :4], det[:, 4], iou_thres)
                    det = det[keep_nms]
            else:
                det = torch.zeros((0, 6), device=out_logits.device)
                
            detections.append(det)
            
        return detections

    def get_num_classes(self, model):
        """
        Get number of model target classes.
        """
        if hasattr(model, 'num_classes'):
            return model.num_classes
        return 80  # Default to COCO

    def get_class_names(self, model):
        """
        Get class names dictionary.
        """
        if hasattr(model, 'names'):
            return {k: v for k, v in enumerate(model.names)}
        return {}


class ModelLoader:
    """
    Handles loading and profiling of PyTorch models with support for pruning.
    """
    
    @staticmethod
    def load_model(weights, modification, pruning_params, criterion, device, 
                   imgsz=640, trace=False, half_precision=True):
        """
        Load model and apply pruning/tracing if requested.
        
        Returns:
            model, params, fs (GFLOPS)
        """
        if modification == "prune-unstructured":
            model = load_pruned_model(weights, pruning_params, device)
        elif modification == "prune-structured":
            model = load_pruned_model(weights, pruning_params, criterion, map_location=device)
        elif modification == "prune-layer":
            model = load_pruned_model(weights, pruning_params, criterion,
                                      map_location=device, modification="prune-layer")
        else:
            model = attempt_load(weights, map_location=device)
            
        gs = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        
        # Profile GFLOPS and params
        stride = gs
        sample_device = next(model.parameters()).device
        ch = model.yaml.get('ch', 3) if hasattr(model, 'yaml') and model.yaml else 3
        img = torch.zeros((1, ch, stride, stride), device=sample_device)
        try:
            flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2
            img_size = imgsz if isinstance(imgsz, list) else [imgsz, imgsz]
            fs = round(flops * img_size[0] / stride * img_size[1] / stride, 3)
        except Exception as e:
            print(f"Warning: profiling GFLOPS failed: {e}")
            fs = 0.0
            
        params = sum(x.numel() for x in model.parameters())
        
        if trace:
            model = TracedModel(model, device, imgsz)
            
        # Half precision
        half = device.type != 'cpu' and half_precision
        if half:
            model.half()
            
        model.eval()
        return model, params, fs, gs
