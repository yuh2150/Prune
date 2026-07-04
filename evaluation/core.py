"""
Core evaluation coordinator module. Matches the functional logic of validation loop from test.py.
"""

from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from utils.general import coco80_to_coco91_class, scale_coords, xyxy2xywh
from utils.torch_utils import time_synchronized, select_device

from evaluation.config import init_evaluation_env
from evaluation.model_adapter import YOLOAdapter, DETRAdapter, ModelLoader
from evaluation.dataset import build_dataloader
from evaluation.metrics import DetectionEvaluator
from evaluation.results import save_label_txt, save_plots, save_predictions_json, print_results, print_speed, log_to_wandb


def evaluate(data,
             weights=None,
             batch_size=32,
             imgsz=640,
             conf_thres=0.001,
             iou_thres=0.6,
             save_json=False,
             single_cls=False,
             augment=False,
             verbose=False,
             model=None,
             dataloader=None,
             save_dir=Path(''),
             save_txt=False,
             save_hybrid=False,
             save_conf=False,
             plots=True,
             wandb_logger=None,
             compute_loss=None,
             half_precision=True,
             trace=False,
             is_coco=False,
             v5_metric=False,
             pruning_params=None,
             criterion=0,
             opt=None,
             model_adapter=None,
             dataset_adapter=None):
    """
    Standardized validation evaluation function. Handles loading/profiling models,
    processing dataloader iterations, calculating precision/recall/mAP metrics,
    plotting/saving prediction inputs, and logging to WandB.
    """
    if pruning_params is None:
        pruning_params = []
        
    # Auto-detect if using a DETR model
    is_detr = False
    if model is not None:
        is_detr = 'detr' in str(type(model)).lower()
    elif weights is not None:
        is_detr = any('detr' in str(w).lower() for w in (weights if isinstance(weights, list) else [weights]))

    if model_adapter is None:
        model_adapter = DETRAdapter() if is_detr else YOLOAdapter()
        
    training = model is not None
    
    if training:
        device = next(model.parameters()).device
        gs = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        params = sum(x.numel() for x in model.parameters())
        fs = 0.0
    else:
        # Resolve device
        device_str = opt.device if opt and hasattr(opt, 'device') else ''
        device = select_device(device_str, batch_size=batch_size)
        
        # Initialize output environment directories
        project = opt.project if opt and hasattr(opt, 'project') else 'runs/test'
        name = opt.name if opt and hasattr(opt, 'name') else 'exp'
        exist_ok = opt.exist_ok if opt and hasattr(opt, 'exist_ok') else False
        
        save_dir = init_evaluation_env(
            project=project,
            name=name,
            exist_ok=exist_ok,
            save_txt=save_txt
        )
        
        # Model loading & structural tuning
        modification = opt.modification if opt and hasattr(opt, 'modification') else ''
        model, params, fs, gs = ModelLoader.load_model(
            weights=weights,
            modification=modification,
            pruning_params=pruning_params,
            criterion=criterion,
            device=device,
            imgsz=imgsz,
            trace=trace,
            half_precision=half_precision
        )
        
    # Set model state
    model.eval()
    half = device.type != 'cpu' and half_precision
    if half:
        model.half()
        
    # Dataset & dataloader initialization
    if dataset_adapter is None:
        dataset_adapter = build_dataloader(
            data=data,
            imgsz=imgsz,
            batch_size=batch_size,
            gs=gs,
            opt=opt,
            task='val',
            dataloader=dataloader,
            single_cls=single_cls,
            model_type='detr' if is_detr else 'yolo'
        )
        
    # Warmup / Run once to trace or calibrate model
    if not training and device.type != 'cpu':
        dummy_img = torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))
        model_adapter.forward(model, dummy_img)
        
    if v5_metric:
        print("Testing with YOLOv5 AP metric...")
        
    # Configure variables
    nc = dataset_adapter.num_classes
    names = model_adapter.get_class_names(model)
    if not names:
        # Fallback names index-to-string mapping
        names = {i: f'class_{i}' for i in range(nc)}
        
    is_coco = is_coco or (isinstance(data, str) and data.endswith('coco.yaml'))
    
    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    evaluator = DetectionEvaluator(nc, iouv, device)
    
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
        
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    loss = torch.zeros(3, device=device)
    jdict, wandb_images = [], []
    t0, t1 = 0.0, 0.0
    
    # Validation loop
    pbar = tqdm(dataset_adapter, desc=s)
    for batch_i, (img, targets, paths, shapes) in enumerate(pbar):
        img = model_adapter.preprocess(img, device, half=half)
        targets = targets.to(device)
        nb, _, height, width = img.shape
        
        with torch.no_grad():
            t = time_synchronized()
            outputs = model_adapter.forward(model, img, augment=augment)
            t0 += time_synchronized() - t
            
            # Loss calculations if needed
            if isinstance(outputs, tuple) and len(outputs) >= 2:
                out, train_out = outputs[0], outputs[1]
            else:
                out = outputs
                train_out = None
                
            if compute_loss and train_out is not None:
                if isinstance(train_out, list):
                    loss += compute_loss([x.float() for x in train_out], targets)[1][:3]
                else:
                    loss += compute_loss(train_out, targets)[1][:3]
                    
            # Postprocess outputs (NMS)
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []
            
            t = time_synchronized()
            out = model_adapter.postprocess(out, conf_thres, iou_thres, labels=lb)
            t1 += time_synchronized() - t
            
        # Process statistics and file savings
        preds = []
        preds_native = []
        
        for si, pred in enumerate(out):
            path = Path(paths[si])
            predn = pred.clone()
            
            if len(pred) > 0:
                scale_coords(img[si].shape[1:], predn[:, :4], shapes[si])
                
            preds.append(pred)
            preds_native.append(predn)
            
            # Export labels to txt
            if save_txt and len(pred) > 0:
                save_label_txt(predn, shapes[si], path, save_dir, save_conf)
                
            # Logging bounding boxes to Weights & Biases
            if len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0:
                if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
                    box_data = [{
                        "position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                        "class_id": int(cls),
                        "box_caption": "%s %.3f" % (names[cls], conf),
                        "scores": {"class_score": conf},
                        "domain": "pixel"
                    } for *xyxy, conf, cls in pred.tolist()]
                    boxes = {"predictions": {"box_data": box_data, "class_labels": names}}
                    wandb_images.append(wandb_logger.wandb.Image(img[si], boxes=boxes, caption=path.name))
                    
            if wandb_logger and hasattr(wandb_logger, 'wandb_run') and wandb_logger.wandb_run and len(pred) > 0:
                wandb_logger.log_training_progress(predn, path, names)
                
            # Accumulate predictions in pycocotools dictionary
            if save_json and len(pred) > 0:
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left
                coco91class = coco80_to_coco91_class()
                for p_det, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({
                        'image_id': image_id,
                        'category_id': coco91class[int(p_det[5])] if is_coco else int(p_det[5]),
                        'bbox': [round(x, 3) for x in b],
                        'score': round(p_det[4], 5)
                    })
                    
        # Update metrics accumulator
        evaluator.update(
            preds=preds,
            preds_native=preds_native,
            targets=targets,
            shapes=shapes,
            img_shape=img.shape[2:],
            plots=plots
        )
        
        # Save plots
        save_plots(
            batch_i=batch_i,
            img=img,
            targets=targets,
            paths=paths,
            out=out,
            save_dir=save_dir,
            names=names,
            plots=plots
        )
        
    # Calculate final metrics
    metrics = evaluator.evaluate(plots=plots, v5_metric=v5_metric, save_dir=save_dir, names=names)
    
    mp = metrics['mp']
    mr = metrics['mr']
    map50 = metrics['map50']
    map = metrics['map']
    p = metrics['p']
    r = metrics['r']
    ap = metrics['ap']
    ap50 = metrics['ap50']
    ap_class = metrics['ap_class']
    nt = metrics['nt']
    
    seen = evaluator.collector.seen
    
    # Save predictions as JSON and run COCO evaluator
    if save_json:
        map, map50 = save_predictions_json(
            jdict=jdict,
            save_dir=save_dir,
            is_coco=is_coco,
            img_files=dataset_adapter.img_files,
            weights=weights,
            original_map=map,
            original_map50=map50
        )
        
    # Display statistics
    print_results(
        seen=seen,
        nt=nt,
        mp=mp,
        mr=mr,
        map50=map50,
        map=map,
        ap_class=ap_class,
        p=p,
        r=r,
        ap50=ap50,
        ap=ap,
        names=names,
        verbose=verbose,
        nc=nc,
        training=training
    )
    
    # Calculate execution speeds
    val_seen = max(seen, 1)
    speed_ms = tuple(x / val_seen * 1E3 for x in (t0, t1, t0 + t1))
    t = speed_ms + (imgsz, imgsz, batch_size)
    if not training:
        print_speed(t0, t1, seen, imgsz, batch_size)
        
    # Confusion matrix plots and logging
    if plots:
        evaluator.collector.confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        log_to_wandb(wandb_logger, save_dir, wandb_images, names, plots=plots)
        
    model.float()
    
    if not training:
        s_lbl = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s_lbl}")
        
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
        
    # Return matches exactly test.py returns
    if training:
        return (mp, mr, map50, map, *(loss.cpu() / len(dataset_adapter)).tolist()), maps, t
    else:
        return (mp, mr, map50, map, *(loss.cpu() / len(dataset_adapter)).tolist()), maps, t, params, fs
