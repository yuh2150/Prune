"""
Abstract base class and YOLO/DETR concrete implementations for model adapters,
as well as helpers for model loading and profiling.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
import torch
import torch.nn.functional as F
from thop import profile
import torch_pruning as tp

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
        self._last_img_shape = img.shape[-2:]
        
        # Kiểm tra signature của model.forward để truyền orig_target_sizes nếu mô hình yêu cầu (RT-DETR)
        import inspect
        sig = inspect.signature(model.forward)
        if 'orig_target_sizes' in sig.parameters:
            import torch
            orig_target_sizes = torch.tensor([[img.shape[-2], img.shape[-1]]] * img.shape[0], device=img.device)
            outputs = model(img, orig_target_sizes)
        else:
            outputs = model(img)
            
        return outputs, None

    def postprocess(self, outputs, conf_thres, iou_thres, labels=None):
        """
        Postprocess DETR/RT-DETR outputs.
        """
        # Nếu outputs là tuple dạng (labels, boxes, scores) từ RT-DETR
        if isinstance(outputs, tuple) and len(outputs) == 3:
            pred_labels, boxes_xyxy, scores = outputs
            # Cần đảm bảo kiểu dữ liệu là float cho bboxes, labels và scores
            boxes_xyxy = boxes_xyxy.float()
            scores = scores.float()
            pred_labels = pred_labels.float()
        else:
            # Standard DETR
            out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
            
            # Softmax over class dimensions to get probabilities
            prob = F.softmax(out_logits, -1)
            scores, pred_labels = prob[..., :-1].max(-1)  # exclude background class (last slot)
            
            # Convert boxes from normalized cxcywh to normalized xyxy
            x_c, y_c, w, h = out_bbox.unbind(-1)
            boxes_xyxy = torch.stack([
                (x_c - 0.5 * w), (y_c - 0.5 * h),
                (x_c + 0.5 * w), (y_c + 0.5 * h)
            ], dim=-1)
            
            # Nhân tỷ lệ với kích thước ảnh đầu vào (resized input size)
            img_h, img_w = self._last_img_shape if hasattr(self, '_last_img_shape') else (800, 800)
            scale_fct = torch.tensor([img_w, img_h, img_w, img_h], device=out_bbox.device, dtype=out_bbox.dtype)
            boxes_xyxy = boxes_xyxy * scale_fct
            
            # Ánh xạ nhãn dự đoán từ coco91 sang coco80 cho standard DETR
            from utils.general import coco80_to_coco91_class
            coco91_to_coco80 = {v: k for k, v in enumerate(coco80_to_coco91_class())}
            pred_labels_mapped = torch.full_like(pred_labels, -1, dtype=torch.float32)
            for c_91, c_80 in coco91_to_coco80.items():
                pred_labels_mapped[pred_labels == c_91] = c_80
            pred_labels = pred_labels_mapped
            
        batch_size = boxes_xyxy.shape[0]
        detections = []
        
        for i in range(batch_size):
            img_boxes = boxes_xyxy[i]
            img_scores = scores[i]
            img_labels = pred_labels[i]
            
            # Lọc theo ngưỡng tin cậy và loại bỏ nhãn -1 (ngoài 80 lớp coco)
            keep = (img_scores > conf_thres) & (img_labels >= 0)
            
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
                det = torch.zeros((0, 6), device=boxes_xyxy.device)
                
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
        import os
        weights_str = str(weights[0] if isinstance(weights, list) else weights).lower()
        
        # Tự động nhận diện và tải DETR/RT-DETR từ PyTorch Hub
        if 'detr' in weights_str:
            import torch
            import sys
            
            # Lưu lại sys.path ban đầu
            original_sys_path = list(sys.path)
            hub_dir = torch.hub.get_dir()
            
            # Phân biệt DETR và RT-DETR
            is_rtdetr = 'rtdetr' in weights_str
            fb_hub_path = os.path.join(hub_dir, 'facebookresearch_detr_main')
            if is_rtdetr:
                detr_hub_path = os.path.join(hub_dir, 'lyuwenyu_RT-DETR_main')
            else:
                detr_hub_path = fb_hub_path
            
            # Ưu tiên đưa đường dẫn hub lên hàng đầu (cả hai đường dẫn nếu có để đảm bảo tìm thấy 'util')
            for path in [detr_hub_path, fb_hub_path]:
                if path in sys.path:
                    sys.path.remove(path)
                sys.path.insert(0, path)
            
            # Tạm thời loại bỏ đường dẫn thư mục hiện tại để tránh xung đột thư mục 'models'
            cwd = os.getcwd()
            local_paths = [cwd, '', '.']
            for p in local_paths:
                while p in sys.path:
                    sys.path.remove(p)
                    
            # Sao lưu các module YOLO cục bộ khỏi sys.modules để tránh xung đột namespace
            local_models_cache = {}
            yolo_keys = ['models', 'models.common', 'models.experimental', 'models.yolo', 'models.pruned_common']
            for k in list(sys.modules.keys()):
                if any(k == yk or k.startswith(yk + '.') for yk in yolo_keys):
                    local_models_cache[k] = sys.modules.pop(k)
                    
            try:
                if is_rtdetr:
                    print("Loading RT-DETR model from PyTorch Hub...")
                    if 'rtdetr_r18vd' in weights_str:
                        model = torch.hub.load('lyuwenyu/RT-DETR:main', 'rtdetr_r18vd', pretrained=True, trust_repo=True)
                    elif 'rtdetr_r34vd' in weights_str:
                        model = torch.hub.load('lyuwenyu/RT-DETR:main', 'rtdetr_r34vd', pretrained=True, trust_repo=True)
                    elif 'rtdetr_r50vd' in weights_str:
                        model = torch.hub.load('lyuwenyu/RT-DETR:main', 'rtdetr_r50vd', pretrained=True, trust_repo=True)
                    else:
                        model = torch.hub.load('lyuwenyu/RT-DETR:main', 'rtdetr_r18vd', pretrained=True, trust_repo=True)
                else:
                    print("Loading DETR model from PyTorch Hub...")
                    if 'resnet18' in weights_str:
                        from hubconf import _make_detr
                        print("Constructing DETR model with ResNet-18 backbone...")
                        model = _make_detr("resnet18", dilation=False, num_classes=91)
                    elif 'resnet34' in weights_str:
                        from hubconf import _make_detr
                        print("Constructing DETR model with ResNet-34 backbone...")
                        model = _make_detr("resnet34", dilation=False, num_classes=91)
                    else:
                        model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
            finally:
                # Khôi phục sys.path nguyên bản
                sys.path = original_sys_path
                # Thêm cả hai hub path vào cuối sys.path làm fallback vĩnh viễn
                for path in [detr_hub_path, fb_hub_path]:
                    if path not in sys.path:
                        sys.path.append(path)
                # Khôi phục các module YOLO cục bộ vào sys.modules
                for k, v in local_models_cache.items():
                    sys.modules[k] = v
            
            model._model_type = 'detr'
                
            # Load custom weights/state_dict nếu tệp tồn tại và có dung lượng thực tế (> 1MB)
            w_file = weights[0] if isinstance(weights, list) else weights
            if os.path.exists(w_file) and os.path.getsize(w_file) > 1024 * 1024:
                try:
                    ckpt = torch.load(w_file, map_location='cpu')
                    if isinstance(ckpt, dict) and 'model' in ckpt:
                        state_dict = ckpt['model'].state_dict() if hasattr(ckpt['model'], 'state_dict') else ckpt['model']
                    else:
                        state_dict = ckpt
                    model.load_state_dict(state_dict, strict=False)
                    print(f"Successfully loaded custom state_dict from {w_file}")
                except Exception as e:
                    print(f"Warning: Could not load state_dict from {w_file} ({e}). Using default weights.")
            model = model.to(device)
        elif modification == "prune-unstructured":
            model = load_pruned_model(weights, pruning_params, device)
        elif modification == "prune-structured":
            model = load_pruned_model(weights, pruning_params, criterion, map_location=device)
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


class LayerInfo:
    """
    Đóng gói thông tin metadata của một layer có thể thực hiện pruning.
    """
    def __init__(self, name: str, module: torch.nn.Module, layer_type: str, metadata: dict = None):
        self.name = name              # Tên định danh duy nhất (e.g., 'backbone.body.conv1')
        self.module = module          # Tham chiếu thực tế tới torch.nn.Module
        self.layer_type = layer_type  # Kiểu layer (e.g., 'Conv2d', 'Linear')
        self.metadata = metadata or {}  # Các siêu dữ liệu bổ sung


class PrunerAdapter(ABC):
    """
    Interface định nghĩa các thao tác phục vụ pruning trên một kiến trúc model cụ thể.
    """
    @abstractmethod
    def get_prunable_layers(self, model) -> list[LayerInfo]:
        """
        Trả về danh sách các LayerInfo có thể prune. 
        Đã tự động loại trừ các layer output/head nguy hiểm.
        """
        pass

    @abstractmethod
    def prune_layer(self, model, layer_name: str, amount: float, criterion: int) -> torch.nn.Module:
        """
        Thực hiện structured pruning trên một layer cụ thể dựa theo tên của nó.
        """
        pass


class YOLOPrunerAdapter(PrunerAdapter):
    def get_prunable_layers(self, model) -> list[LayerInfo]:
        import torch.nn as nn
        prunable = []
        
        # Nếu model sử dụng cấu hình pruning_cfg thủ công (như YOLOv7)
        if hasattr(model, 'pruning_cfg'):
            from prune import get_conv_layers
            conv_modules = get_conv_layers(model)
            for idx, conv_m in enumerate(conv_modules):
                # Ánh xạ tên để dễ tracking
                name = f"conv_layer_{idx}"
                if isinstance(conv_m, list) or isinstance(conv_m, tuple):
                    module_ref = conv_m[0]
                else:
                    module_ref = conv_m
                prunable.append(LayerInfo(name, module_ref, 'Conv2d'))
            return prunable
            
        # Đối với YOLOv5 sử dụng torch_pruning
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                # Bỏ qua các layer thuộc Detect head
                is_detect = any('detect' in parent_name.lower() for parent_name in name.split('.'))
                if not is_detect:
                    prunable.append(LayerInfo(name, m, 'Conv2d'))
        return prunable

    def prune_layer(self, model, layer_name: str, amount: float, criterion: int) -> torch.nn.Module:
        # Nếu dùng torch_pruning cho YOLOv5
        import torch_pruning as tp
        from prune import make_divisible
        import torch.nn as nn
        
        # Tìm target layer theo tên
        target_conv = None
        for name, m in model.named_modules():
            if name == layer_name:
                target_conv = m
                break
                
        if target_conv is None:
            return model
            
        out_channels = target_conv.out_channels
        n_pruned = make_divisible(amount * out_channels, 2)
        if n_pruned >= out_channels:
            n_pruned = out_channels - 2
            
        if n_pruned <= 0:
            return model
            
        # Áp dụng cơ chế chọn kênh theo criterion tương tự như prune.py
        weights = target_conv.weight.data
        importances = torch.norm(weights, p=2, dim=[1, 2, 3])
        if criterion == 1:
            importances = -importances
        elif criterion in [2, 3]:
            importances = torch.norm(weights, p=1, dim=[1, 2, 3])
            if criterion == 3:
                importances = -importances
        elif criterion == 6:
            importances = torch.rand(out_channels, device=weights.device)
            
        idxs = torch.argsort(importances)[:n_pruned].tolist()
        
        # Xây dựng dependency graph và thực hiện prune
        # Giả lập input tensor kích thước chuẩn để trace graph
        first_param = next(model.parameters())
        x = torch.randn(1, 3, 640, 640).to(device=first_param.device, dtype=first_param.dtype)
        
        # Cập nhật tạm thời requires_grad để phục vụ trace graph
        grad_states = {n: p.requires_grad for n, p in model.named_parameters()}
        for p in model.parameters():
            p.requires_grad = True
            
        DG = tp.DependencyGraph().build_dependency(model, x)
        
        for n, p in model.named_parameters():
            p.requires_grad = grad_states[n]
            
        group = DG.get_pruning_group(target_conv, tp.prune_conv_out_channels, idxs)
        if DG.check_pruning_group(group):
            group.prune()
            
        return model


class FrozenBatchNormPruner(tp.BasePruningFunc):
    def prune_out_channels(self, layer: torch.nn.Module, idxs: list[int]) -> torch.nn.Module:
        keep_idxs = list(set(range(layer.weight.shape[0])) - set(idxs))
        keep_idxs.sort()
        
        device = layer.weight.device
        keep_tensor = torch.LongTensor(keep_idxs).to(device)
        
        layer.weight = torch.index_select(layer.weight, 0, keep_tensor)
        
        if hasattr(layer, 'bias') and layer.bias is not None:
            layer.bias = torch.index_select(layer.bias, 0, keep_tensor)
            
        if hasattr(layer, 'running_mean') and layer.running_mean is not None:
            layer.running_mean = torch.index_select(layer.running_mean, 0, keep_tensor)
            
        if hasattr(layer, 'running_var') and layer.running_var is not None:
            layer.running_var = torch.index_select(layer.running_var, 0, keep_tensor)
            
        return layer
        
    prune_in_channels = prune_out_channels
    
    def get_out_channels(self, layer: torch.nn.Module):
        return layer.weight.shape[0]
        
    def get_in_channels(self, layer: torch.nn.Module):
        return layer.weight.shape[0]


class DETRPrunerAdapter(PrunerAdapter):
    def get_prunable_layers(self, model) -> list[LayerInfo]:
        import torch.nn as nn
        prunable = []
        
        # Với DETR ta có cả Conv2d (Backbone)
        for name, m in model.named_modules():
            # Chỉ prune các layer thuộc backbone
            # Loại bỏ các head dự đoán đầu ra và layer chiếu trung gian
            is_output = any(kw in name for kw in ['class_embed', 'bbox_embed', 'query_embed', 'input_proj'])
            
            if isinstance(m, nn.Conv2d) and not is_output:
                prunable.append(LayerInfo(name, m, 'Conv2d'))
        return prunable

    def prune_layer(self, model, layer_name: str, amount: float, criterion: int) -> torch.nn.Module:
        import torch_pruning as tp
        from prune import make_divisible
        import torch.nn as nn
        
        target_module = None
        for name, m in model.named_modules():
            if name == layer_name:
                target_module = m
                break
                
        if target_module is None:
            return model
            
        # Sử dụng kích thước đầu vào thích hợp cho DETR/RT-DETR
        first_param = next(model.parameters())
        
        # Kiểm tra signature của model.forward để truyền tham số đầy đủ cho DependencyGraph
        import inspect
        sig = inspect.signature(model.forward)
        if 'orig_target_sizes' in sig.parameters:
            # RT-DETR sử dụng positional embeddings cố định kích thước 640x640
            x = torch.randn(1, 3, 640, 640).to(device=first_param.device, dtype=first_param.dtype)
            orig_target_sizes = torch.tensor([[640, 640]], device=first_param.device)
            example_inputs = (x, orig_target_sizes)
        else:
            # Standard DETR hỗ trợ kích thước tùy biến 800x800
            x = torch.randn(1, 3, 800, 800).to(device=first_param.device, dtype=first_param.dtype)
            example_inputs = x
        
        if isinstance(target_module, nn.Conv2d):
            out_channels = target_module.out_channels
            pruning_fn = tp.prune_conv_out_channels
        else:
            return model
            
        n_pruned = make_divisible(amount * out_channels, 2)
        if n_pruned >= out_channels:
            n_pruned = out_channels - 2
        if n_pruned <= 0:
            return model
            
        # Áp dụng criterion để chọn index kênh prune
        weights = target_module.weight.data
        importances = torch.norm(weights, p=2, dim=list(range(1, weights.ndim)))
        if criterion == 1:
            importances = -importances
        elif criterion in [2, 3]:
            importances = torch.norm(weights, p=1, dim=list(range(1, weights.ndim)))
            if criterion == 3:
                importances = -importances
        elif criterion == 6:
            importances = torch.rand(out_channels, device=weights.device)
            
        idxs = torch.argsort(importances)[:n_pruned].tolist()
        
        grad_states = {n: p.requires_grad for n, p in model.named_parameters()}
        for p in model.parameters():
            p.requires_grad = True
            
        # Tự động tìm và đăng ký FrozenBatchNorm2d nếu có
        frozen_bn_type = None
        for m in model.modules():
            if m.__class__.__name__ == 'FrozenBatchNorm2d':
                frozen_bn_type = m.__class__
                break
                
        customized_pruners = {}
        if frozen_bn_type is not None:
            customized_pruners[frozen_bn_type] = FrozenBatchNormPruner()
            
        DG = tp.DependencyGraph().build_dependency(model, example_inputs, customized_pruners=customized_pruners)
        
        for n, p in model.named_parameters():
            p.requires_grad = grad_states[n]
            
        try:
            group = DG.get_pruning_group(target_module, pruning_fn, idxs)
            if DG.check_pruning_group(group):
                group.prune()
        except Exception as e:
            print(f"Warning: Skipping pruning for layer {layer_name} because it is not in the dependency graph or not traceable ({e})")
            
        return model


class PrunerAdapterFactory:
    """
    Tự động xác định PrunerAdapter phù hợp dựa trên cấu hình model.
    """
    @staticmethod
    def get_pruner(weights_path_or_list, model=None) -> PrunerAdapter:
        if model is not None:
            model_class = str(type(model)).lower()
            if 'detr' in model_class or getattr(model, '_model_type', '') == 'detr':
                return DETRPrunerAdapter()
                
        weights_str = ""
        if isinstance(weights_path_or_list, list) and len(weights_path_or_list) > 0:
            weights_str = str(weights_path_or_list[0]).lower()
        elif isinstance(weights_path_or_list, str):
            weights_str = weights_path_or_list.lower()
            
        if 'detr' in weights_str:
            return DETRPrunerAdapter()
            
        return YOLOPrunerAdapter()


class ModelAdapterFactory:
    """
    Factory để tự động nhận diện và cung cấp ModelAdapter phù hợp cho Inference/Evaluation.
    """
    @staticmethod
    def get_adapter(weights_path_or_list, model=None) -> ModelAdapter:
        if model is not None:
            model_class = str(type(model)).lower()
            if 'detr' in model_class or getattr(model, '_model_type', '') == 'detr':
                return DETRAdapter()
            
        weights_str = ""
        if isinstance(weights_path_or_list, list) and len(weights_path_or_list) > 0:
            weights_str = str(weights_path_or_list[0]).lower()
        elif isinstance(weights_path_or_list, str):
            weights_str = weights_path_or_list.lower()
            
        if 'detr' in weights_str:
            return DETRAdapter()
        
        return YOLOAdapter()

