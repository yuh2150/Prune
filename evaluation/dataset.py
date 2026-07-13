"""
Dataset adapters and loader helpers for evaluation tasks, supporting YOLO and DETR.
"""

from abc import ABC, abstractmethod
import yaml
import torch

from utils.datasets import create_dataloader
from utils.general import check_dataset, colorstr


class DatasetAdapter(ABC):
    """
    Abstract interface for wrapping dataloaders/datasets to standardize batch retrieval.
    """
    
    @abstractmethod
    def __iter__(self):
        """Should yield (img, targets, paths, shapes) on iteration."""
        pass

    @abstractmethod
    def __len__(self):
        """Return the number of batches."""
        pass

    @property
    @abstractmethod
    def num_classes(self):
        """Return number of classes in the dataset config."""
        pass

    @property
    @abstractmethod
    def class_names(self):
        """Return class names in the dataset config."""
        pass

    @property
    @abstractmethod
    def img_files(self):
        """Return list of image files."""
        pass


class YOLODatasetAdapter(DatasetAdapter):
    """
    Concrete implementation of DatasetAdapter for YOLO datasets.
    """
    
    def __init__(self, dataloader, data_dict, single_cls=False):
        self.dataloader = dataloader
        self.data_dict = data_dict
        self.single_cls = single_cls

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

    @property
    def num_classes(self):
        return 1 if self.single_cls else int(self.data_dict['nc'])

    @property
    def class_names(self):
        return self.data_dict.get('names', [])

    @property
    def img_files(self):
        if hasattr(self.dataloader, 'dataset') and hasattr(self.dataloader.dataset, 'img_files'):
            return self.dataloader.dataset.img_files
        return []


class DETRDatasetAdapter(DatasetAdapter):
    """
    Dataset adapter for DETR/COCO detection datasets.
    Translates DETR outputs to a standard YOLO-compatible format.
    """
    
    def __init__(self, dataloader, single_cls=False):
        self.dataloader = dataloader
        self.single_cls = single_cls
        self.dataset = getattr(dataloader, 'dataset', None)

    def __iter__(self):
        for samples, targets in self.dataloader:
            # samples is NestedTensor (has .tensors) or standard Tensor
            if hasattr(samples, 'tensors'):
                img = samples.tensors
            else:
                img = samples
                
            yolo_targets_list = []
            paths = []
            shapes = []
            
            for i, tgt in enumerate(targets):
                image_id = tgt.get("image_id", torch.tensor([i]))
                if isinstance(image_id, torch.Tensor):
                    image_id_val = image_id.item()
                else:
                    image_id_val = image_id
                paths.append(str(image_id_val))
                
                # Extract image dimensions
                if "orig_size" in tgt:
                    orig_h, orig_w = tgt["orig_size"].tolist()
                else:
                    orig_h, orig_w = img.shape[-2:]
                    
                if "size" in tgt:
                    h_new, w_new = tgt["size"].tolist()
                else:
                    h_new, w_new = img.shape[-2:]
                    
                shapes.append(((orig_h, orig_w), ((h_new, w_new), (0.0, 0.0))))
                
                if "boxes" in tgt and "labels" in tgt:
                    boxes = tgt["boxes"]  # normalized cxcywh [M, 4]
                    labels = tgt["labels"]  # class ids [M]
                    from utils.general import coco80_to_coco91_class
                    coco91_to_coco80 = {v: k for k, v in enumerate(coco80_to_coco91_class())}
                    for box, label in zip(boxes.tolist(), labels.tolist()):
                        mapped_lbl = coco91_to_coco80.get(int(label), -1)
                        if mapped_lbl != -1:
                            yolo_targets_list.append([i, mapped_lbl, *box])
                        
            if yolo_targets_list:
                yolo_targets = torch.tensor(yolo_targets_list, dtype=torch.float32)
            else:
                yolo_targets = torch.zeros((0, 6), dtype=torch.float32)
                
            yield img, yolo_targets, paths, shapes

    def __len__(self):
        return len(self.dataloader)

    @property
    def num_classes(self):
        if self.single_cls:
            return 1
        if hasattr(self.dataset, 'coco') and hasattr(self.dataset.coco, 'cats'):
            return len(self.dataset.coco.cats)
        return 80  # Default to COCO classes

    @property
    def class_names(self):
        from utils.general import coco80_to_coco91_class
        coco91_to_coco80 = {v: k for k, v in enumerate(coco80_to_coco91_class())}
        names_dict = {}
        if hasattr(self.dataset, 'coco') and hasattr(self.dataset.coco, 'cats'):
            cats = self.dataset.coco.loadCats(self.dataset.coco.getCatIds())
            for cat in cats:
                c91 = cat['id']
                if c91 in coco91_to_coco80:
                    c80 = coco91_to_coco80[c91]
                    names_dict[c80] = cat['name']
        if not names_dict:
            names_dict = {i: f'class_{i}' for i in range(self.num_classes)}
        return names_dict

    @property
    def img_files(self):
        if hasattr(self.dataset, 'ids'):
            return self.dataset.ids
        return []


def build_dataloader(data, imgsz, batch_size, gs, opt, task='val', dataloader=None, single_cls=False, model_type='yolo'):
    """
    Builds dataloader and returns appropriate wrapped DatasetAdapter.
    """
    if model_type == 'detr':
        if dataloader is None:
            try:
                import sys
                import os
                import torch
                original_sys_path = list(sys.path)
                cwd = os.getcwd()
                if cwd not in sys.path:
                    sys.path.insert(0, cwd)
                    
                # Đảm bảo đường dẫn hub của facebookresearch_detr_main nằm trong sys.path để tìm thấy 'util'
                hub_dir = torch.hub.get_dir()
                fb_hub_path = os.path.join(hub_dir, 'facebookresearch_detr_main')
                if fb_hub_path not in sys.path:
                    sys.path.insert(0, fb_hub_path)
                    
                # Cô lập thư viện datasets của Hugging Face trong site-packages
                cached_datasets = {}
                for k in list(sys.modules.keys()):
                    if k == 'datasets' or k.startswith('datasets.'):
                        cached_datasets[k] = sys.modules.pop(k)
                        
                try:
                    from datasets.coco import build as build_coco_dataset
                finally:
                    sys.path = original_sys_path
                    if fb_hub_path not in sys.path:
                        sys.path.append(fb_hub_path)
                    for k, v in cached_datasets.items():
                        sys.modules[k] = v
                        
                is_rtdetr = False
                if opt and hasattr(opt, 'weights'):
                    w_str = str(opt.weights[0] if isinstance(opt.weights, list) else opt.weights).lower()
                    is_rtdetr = 'rtdetr' in w_str
                    
                class Args:
                    coco_path = opt.data if (opt and hasattr(opt, 'data')) else './coco'
                    masks = False
                    img_size = imgsz
                    square_resize = is_rtdetr
                dataset = build_coco_dataset(task, Args())
                
                from torch.utils.data import DataLoader
                def collate_fn(batch):
                    try:
                        import importlib
                        try:
                            misc_module = importlib.import_module("util.misc")
                        except ImportError:
                            misc_module = importlib.import_module("utils.misc")
                        detr_collate = getattr(misc_module, "collate_fn")
                        return detr_collate(batch)
                    except Exception as e:
                        print(f"Fallback collate due to: {e}")
                        batch = list(zip(*batch))
                        images = torch.stack(batch[0], dim=0)
                        return images, batch[1]
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
            except Exception as e:
                print(f"Warning: Failed to build DETR dataloader: {e}")
                
        return DETRDatasetAdapter(dataloader, single_cls=single_cls)
        
    # Default YOLO loader
    if isinstance(data, str):
        with open(data, encoding='utf-8') as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)
    
    if dataloader is None:
        dataloader_task = task if task in ('train', 'val', 'test') else 'val'
        dataloader = create_dataloader(
            data[dataloader_task], 
            imgsz, 
            batch_size, 
            gs, 
            single_cls=single_cls, 
            pad=0.5, 
            rect=True,
            prefix=colorstr(f'{dataloader_task}: ')
        )[0]
        
    return YOLODatasetAdapter(dataloader, data, single_cls=single_cls)
