"""
Metric collection, accumulation, and evaluation classes for validation tasks.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import torch

from utils.metrics import ap_per_class, ConfusionMatrix
from utils.general import box_iou, xywh2xyxy, scale_coords


class MetricCollector:
    """
    Accumulates per-image detection statistics and calculates precision, recall, and mAP metrics.
    """
    
    def __init__(self, nc, iouv, device):
        """
        Args:
            nc (int): Number of classes.
            iouv (Tensor): IoU thresholds vector.
            device (torch.device): Device.
        """
        self.nc = nc
        self.iouv = iouv
        self.niou = iouv.numel()
        self.device = device
        
        self.seen = 0
        self.stats = []
        self.confusion_matrix = ConfusionMatrix(nc=nc)
        
    def process_image(self, pred, predn, labels, shapes_si, img_si_shape, plots=True):
        """
        Match predictions to targets for a single image, updating stats and confusion matrix.
        """
        nl = len(labels)
        tcls = labels[:, 0].tolist() if nl else []
        self.seen += 1
        
        if len(pred) == 0:
            if nl:
                self.stats.append((
                    torch.zeros(0, self.niou, dtype=torch.bool), 
                    torch.Tensor(), 
                    torch.Tensor(), 
                    tcls
                ))
            return
            
        # Assign all predictions as incorrect
        correct = torch.zeros(pred.shape[0], self.niou, dtype=torch.bool, device=self.device)
        if nl:
            detected = []  # target indices
            tcls_tensor = labels[:, 0]

            # Target boxes
            tbox = xywh2xyxy(labels[:, 1:5])
            scale_coords(img_si_shape, tbox, shapes_si)  # native-space labels
            if plots:
                self.confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

            # Per target class
            for cls in torch.unique(tcls_tensor):
                ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # target indices
                pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # prediction indices

                # Search for detections
                if pi.shape[0]:
                    # Prediction to target ious
                    ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                    # Append detections
                    detected_set = set()
                    for j in (ious > self.iouv[0]).nonzero(as_tuple=False):
                        d = ti[i[j]]  # detected target
                        if d.item() not in detected_set:
                            detected_set.add(d.item())
                            detected.append(d)
                            correct[pi[j]] = ious[j] > self.iouv  # iou_thres is 1xn
                            if len(detected) == nl:  # all targets already located in image
                                break
                                
        self.stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    def compute_metrics(self, plots=True, v5_metric=False, save_dir=Path(''), names=None):
        """
        Calculate precision, recall, and AP scores.
        
        Returns:
            dict containing: mp, mr, map50, map, p, r, ap, f1, ap_class, nt
        """
        stats_np = [np.concatenate(x, 0) for x in zip(*self.stats)] if self.stats else []
        
        p, r, f1, mp, mr, map50, map = 0., 0., 0., 0., 0., 0., 0.
        ap, ap_class, ap50 = [], [], []
        
        if len(stats_np) and stats_np[0].any():
            _, _, p, r, ap, f1, ap_class = ap_per_class(
                *stats_np, plot=plots, v5_metric=v5_metric, save_dir=save_dir, names=names
            )
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats_np[3].astype(np.int64), minlength=self.nc)
            
            # Save results.txt
            with open(save_dir / 'results.txt', "w") as r_ap:
                print(map, ap, file=r_ap)
        else:
            nt = np.zeros(self.nc)
            
        return {
            'mp': mp, 'mr': mr, 'map50': map50, 'map': map,
            'p': p, 'r': r, 'ap': ap, 'ap50': ap50, 'f1': f1, 'ap_class': ap_class, 'nt': nt,
            'stats': stats_np
        }


class MetricEvaluator(ABC):
    """
    Abstract interface for evaluating models and accumulating metrics.
    """
    
    @abstractmethod
    def reset(self):
        """Reset internal accumulator statistics."""
        pass
        
    @abstractmethod
    def update(self, preds, preds_native, targets, shapes, img_shape, plots=True):
        """
        Update accumulator with batch predictions and targets.
        """
        pass
        
    @abstractmethod
    def evaluate(self, plots=True, v5_metric=False, save_dir=None, names=None):
        """
        Compute final metrics.
        """
        pass


class DetectionEvaluator(MetricEvaluator):
    """
    Evaluator implementation for object detection tasks.
    """
    
    def __init__(self, nc, iouv, device):
        self.nc = nc
        self.iouv = iouv
        self.device = device
        self.reset()
        
    def reset(self):
        self.collector = MetricCollector(self.nc, self.iouv, self.device)
        
    def update(self, preds, preds_native, targets, shapes, img_shape, plots=True):
        """
        Update with batch predictions. Note that targets coordinates are already 
        expected to be converted to pixels (as in evaluation loop).
        """
        nb = len(preds)
        for si in range(nb):
            pred = preds[si]
            predn = preds_native[si]
            labels = targets[targets[:, 0] == si, 1:]
            
            self.collector.process_image(
                pred=pred,
                predn=predn,
                labels=labels,
                shapes_si=shapes[si],
                img_si_shape=img_shape,
                plots=plots
            )
            
    def evaluate(self, plots=True, v5_metric=False, save_dir=None, names=None):
        return self.collector.compute_metrics(plots=plots, v5_metric=v5_metric, save_dir=save_dir, names=names)
