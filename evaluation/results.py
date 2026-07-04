"""
Methods for reporting, printing, and saving results/predictions/plots.
"""

import json
from pathlib import Path
from threading import Thread
import torch

from utils.plots import plot_images, output_to_target
from utils.general import xyxy2xywh


# --- Savers ---

def save_label_txt(predn, shape, path, save_dir, save_conf):
    """
    Save predictions to normalized text files in YOLO format.
    """
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_plots(batch_i, img, targets, paths, out, save_dir, names, plots=True):
    """
    Trigger background threads to plot predictions and ground truth images.
    """
    if plots and batch_i < 3:
        f_labels = save_dir / f'test_batch{batch_i}_labels.jpg'
        Thread(target=plot_images, args=(img, targets, paths, f_labels, names), daemon=True).start()
        f_preds = save_dir / f'test_batch{batch_i}_pred.jpg'
        Thread(target=plot_images, args=(img, output_to_target(out), paths, f_preds, names), daemon=True).start()


def save_predictions_json(jdict, save_dir, is_coco, img_files, weights, original_map, original_map50):
    """
    Save prediction JSON compatible with pycocotools and evaluate using pycocotools if available.
    
    Returns:
        tuple: (map, map50) updated results if evaluated, otherwise original results.
    """
    if not len(jdict):
        return original_map, original_map50
        
    w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''
    anno_json = './coco/annotations/instances_val2017.json'
    pred_json = str(save_dir / f"{w}_predictions.json")
    
    print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
    with open(pred_json, 'w') as f:
        json.dump(jdict, f)

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        anno = COCO(anno_json)
        pred = anno.loadRes(pred_json)
        eval = COCOeval(anno, pred, 'bbox')
        if is_coco:
            eval.params.imgIds = [int(Path(x).stem) for x in img_files]
        eval.evaluate()
        eval.accumulate()
        eval.summarize()
        return eval.stats[0], eval.stats[1]  # mAP@0.5:0.95, mAP@0.5
    except Exception as e:
        print(f'pycocotools unable to run: {e}')
        return original_map, original_map50


# --- Reporters ---

def print_results(seen, nt, mp, mr, map50, map, ap_class, p, r, ap50, ap, names, 
                  verbose=False, nc=0, training=False):
    """
    Format and print evaluation metrics overall and per-class.
    """
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(ap_class):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))


def print_speed(t0, t1, seen, imgsz, batch_size):
    """
    Calculate and print speed statistics per image.
    
    Returns:
        tuple: (inference_ms, nms_ms, total_ms)
    """
    val_seen = max(seen, 1)
    t = tuple(x / val_seen * 1E3 for x in (t0, t1, t0 + t1))
    print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % 
          (t + (imgsz, imgsz, batch_size)))
    return t


def log_to_wandb(wandb_logger, save_dir, wandb_images, names, plots=True):
    """
    Log results, confusion matrix, and validation batch plots to weights & biases.
    """
    if plots and wandb_logger and wandb_logger.wandb:
        val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) 
                       for f in sorted(save_dir.glob('test*.jpg'))]
        wandb_logger.log({"Validation": val_batches})
        
    if wandb_images and wandb_logger:
        wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})
