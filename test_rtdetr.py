import argparse
import json
import os
import ast
from pathlib import Path
from thop import profile
from copy import deepcopy

import numpy as np
import torch
import yaml
from tqdm import tqdm

from transformers import RTDetrImageProcessor, RTDetrConfig
from utils.general import coco80_to_coco91_class, check_file, set_logging, increment_path, colorstr
from utils.torch_utils import select_device, time_synchronized

from prune_rtdetr import load_pruned_model_rtdetr, attempt_load_rtdetr, get_prunable_layers
from dataset_coco_rtdetr import CocoEvalDataset, eval_collate_fn

def test(data,
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
         plots=False,
         half_precision=True,
         pruning_params=[],
         criterion=0,
         img_dir='./coco/images/val2017',
         ann_file='./coco/annotations/instances_val2017.json',
         modification=''
         ):
    
    # Initialize/load model and set device
    set_logging()
    device = select_device(opt.device, batch_size=batch_size)
    
    # Resolve weights directory
    if isinstance(weights, list):
        weights_dir = weights[0]
    else:
        weights_dir = weights
        
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and apply pruning if requested
    if model is None:
        if modification == "prune-unstructured":
            model = load_pruned_model_rtdetr(weights_dir, pruning_params, criterion, map_location=device, structured=False)
        elif modification == "prune-structured":
            model = load_pruned_model_rtdetr(weights_dir, pruning_params, criterion, map_location=device, structured=True)
        else:
            model, _, _ = attempt_load_rtdetr(weights_dir, device)
    else:
        # If model is already loaded, we still prune it if modification and pruning_params are supplied
        if pruning_params:
            from prune_rtdetr import prune_unstructured, prune_structured_global
            if modification == "prune-unstructured":
                model = prune_unstructured(model, pruning_params, criterion)
            elif modification == "prune-structured":
                model = prune_structured_global(model, pruning_params, criterion)
                # Verify shape consistency
                print("Verifying model forward pass shape consistency...")
                try:
                    dummy = torch.randn(1, 3, imgsz, imgsz).to(device=device, dtype=next(model.parameters()).dtype)
                    with torch.no_grad():
                        _ = model(dummy)
                    print("Forward pass verification successful.")
                except Exception as e:
                    print(f"WARNING: Model forward pass failed after structured pruning: {e}")
            
    # Load image processor and config
    image_processor = RTDetrImageProcessor.from_pretrained(weights_dir, local_files_only=True)
    config = RTDetrConfig.from_pretrained(weights_dir, local_files_only=True)
    
    # Profile params and GFLOPs
    try:
        dummy_img = torch.zeros((1, 3, imgsz, imgsz), device=device, dtype=next(model.parameters()).dtype)
        flops = profile(deepcopy(model), inputs=(dummy_img,), verbose=False)[0] / 1E9 * 2
        fs = round(flops, 3)
    except Exception as e:
        print(f"Warning: thop profiling GFLOPs failed: {e}")
        fs = 0.0
    params = sum(x.numel() for x in model.parameters())
    
    # Half precision
    half = device.type != 'cpu' and half_precision
    if half:
        model.half()
        
    model.eval()
    
    # Load data dict
    if isinstance(data, str):
        with open(data) as f:
            data_dict = yaml.load(f, Loader=yaml.SafeLoader)
    else:
        data_dict = data
        
    nc = 1 if single_cls else int(data_dict['nc'])
    
    # Parse image IDs for custom subset evaluation (e.g. coco_500)
    img_ids = None
    val_path = data_dict.get('val', '')
    if isinstance(val_path, str) and val_path.endswith('.txt'):
        if os.path.exists(val_path):
            with open(val_path, 'r') as f:
                lines = f.readlines()
            img_ids = []
            for line in lines:
                line = line.strip()
                if line:
                    stem = Path(line).stem
                    if stem.isdigit():
                        img_ids.append(int(stem))
                        
    # Setup dataset & dataloader
    if dataloader is None:
        from pycocotools.coco import COCO
        coco_gt = COCO(ann_file)
        dataset = CocoEvalDataset(coco_gt_obj=coco_gt, img_dir=img_dir, img_ids=img_ids)
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=eval_collate_fn)
    else:
        coco_gt = dataloader.dataset.coco
        
    seen = 0
    jdict = []
    evaluated_img_ids = []
    t0, t1 = 0.0, 0.0
    
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    
    for batch_i, batch_data in enumerate(tqdm(dataloader, desc=s)):
        if batch_data is None:
            continue
        imgs, targets = batch_data
        
        # Warmup and preprocess batch via image_processor
        t_prep_start = time_synchronized()
        inputs = image_processor(images=imgs, return_tensors="pt").to(device)
        if half:
            inputs['pixel_values'] = inputs['pixel_values'].half()
        t_prep = time_synchronized() - t_prep_start
        
        # Inference
        t_inf_start = time_synchronized()
        with torch.no_grad():
            outputs = model(**inputs)
        t_inf = time_synchronized() - t_inf_start
        t0 += (t_prep + t_inf)
        
        # Post-process predictions
        t_post_start = time_synchronized()
        original_sizes = [(t['height'], t['width']) for t in targets]
        target_sizes = torch.tensor(original_sizes, device=device)
        results = image_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=conf_thres)
        t1 += time_synchronized() - t_post_start
        
        image_ids = [t['image_id'] for t in targets]
        evaluated_img_ids.extend(image_ids)
        
        for si, res in enumerate(results):
            seen += 1
            image_id = image_ids[si]
            boxes = res['boxes'].tolist()
            scores = res['scores'].tolist()
            labels = res['labels'].tolist()
            for box, score, label in zip(boxes, scores, labels):
                x_min, y_min, x_max, y_max = box
                w_box = x_max - x_min
                h_box = y_max - y_min
                jdict.append({
                    'image_id': image_id,
                    'category_id': coco91class[label] if label < len(coco91class) else label,
                    'bbox': [round(x_min, 3), round(y_min, 3), round(w_box, 3), round(h_box, 3)],
                    'score': round(score, 5)
                })
                
    # Run COCOeval
    map, map50 = 0.0, 0.0
    mp, mr = 0.0, 0.0
    if len(jdict) > 0:
        w = Path(weights_dir).stem
        pred_json = str(save_dir / f"{w}_predictions.json")
        print(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)
            
        try:
            from pycocotools.cocoeval import COCOeval
            
            anno = coco_gt
            pred = anno.loadRes(pred_json)
            eval_coco = COCOeval(anno, pred, 'bbox')
            eval_coco.params.imgIds = list(set(evaluated_img_ids))
            
            eval_coco.evaluate()
            eval_coco.accumulate()
            eval_coco.summarize()
            map, map50 = eval_coco.stats[0], eval_coco.stats[1]
            mp = map50
            mr = eval_coco.stats[8]
        except Exception as e:
            print(f'pycocotools unable to run: {e}')
            
    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4
    print(pf % ('all', seen, 0, mp, mr, map50, map))
    
    # Speeds
    t = tuple(x / max(seen, 1) * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)
    print('Speed: %.1f/%.1f/%.1f ms inference/postprocess/total per %gx%g image at batch-size %g' % t)
    
    # Save results to txt in save_dir
    with open(os.path.join(save_dir, 'results.txt'), "w") as r_ap:
        print(map, file=r_ap)
        
    maps = np.zeros(nc) + map
    if len(jdict) > 0:
        try:
            if 'eval_coco' in locals() and hasattr(eval_coco, 'eval'):
                # Extract class-wise AP@0.5:0.95
                prec = eval_coco.eval['precision']  # shape: [T, R, K, A, M]
                cat_ids = eval_coco.params.catIds
                for i80 in range(nc):
                    c91 = coco91class[i80] if i80 < len(coco91class) else i80
                    if c91 in cat_ids:
                        k = cat_ids.index(c91)
                        pk = prec[:, :, k, 0, 2]
                        if np.all(pk < 0):
                            maps[i80] = 0.0
                        else:
                            maps[i80] = np.mean(pk[pk >= 0])
        except Exception as e:
            print(f'Error extracting class-wise AP: {e}')
            
    return (mp, mr, map50, map, 0.0, 0.0, 0.0), maps, t, params, fs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test_rtdetr.py')
    parser.add_argument('--weights', nargs='+', type=str, default='PekingU/rtdetr_r18vd', help='model directory path(s)')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for post-processing')
    parser.add_argument('--task', default='val', help='val, pruning_sensitivity_analysis')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    
    # Flags for pruning
    parser.add_argument('--modification', default='', help='prune-structured, prune-unstructured')
    parser.add_argument('--prune-output', type=str, default='output.txt', help="file to write results of pruning to")
    parser.add_argument('--pruning-params', type=str, default='',
                        help='Pruning parameters can be defined as "[(l_1, p_1), (l_2, p_2), (l_3, p_3), ..., (l_n, p_n)]"')
    parser.add_argument('--criterion', type=int, default=0,
                        help="Importance criterion for pruning")
    parser.add_argument('--pruning-rate', type=str, default='0.5', help='Pruning rate can be either defined as a string of list of pruning rates or a single rate')
    
    # Dataset customization paths
    parser.add_argument('--img-dir', type=str, default='./coco/images/val2017', help='directory of COCO images')
    parser.add_argument('--ann-file', type=str, default='./coco/annotations/instances_val2017.json', help='COCO instances json file')

    opt = parser.parse_args()
    opt.save_json = True  # Always enable saving predictions for COCOeval
    opt.data = check_file(opt.data)
    print(opt)
    
    pruning_params_parsed = []
    if len(opt.pruning_params) > 0:
        pruning_params_parsed = ast.literal_eval(opt.pruning_params)
    opt.pruning_rate = ast.literal_eval(opt.pruning_rate)
    
    if opt.task == 'val':
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             pruning_params=pruning_params_parsed,
             criterion=opt.criterion,
             img_dir=opt.img_dir,
             ann_file=opt.ann_file,
             modification=opt.modification
             )
             
    elif opt.task == 'pruning_sensitivity_analysis':
        name = Path(opt.weights[0]).parent.parent.stem if Path(opt.weights[0]).parent.parent.stem \
            else Path(opt.weights[0]).stem
        folder = os.path.join('output', name)
        os.makedirs(folder, exist_ok=True)
        
        # Load model once to query total prunable layers dynamically
        device = select_device(opt.device, batch_size=opt.batch_size)
        from prune_rtdetr import replace_frozen_bn
        base_model, _, _ = attempt_load_rtdetr(opt.weights[0], device)
        base_model = replace_frozen_bn(base_model)
        prunable_layers = get_prunable_layers(base_model)
        conv_layers = len(prunable_layers)
        print(f"Snapshotting prunable Conv2d layers. Found {conv_layers} prunable layers.")
        
        # Create dataset & dataloader once to reuse across iterations
        import yaml
        if isinstance(opt.data, str):
            with open(opt.data) as f:
                data_dict = yaml.load(f, Loader=yaml.SafeLoader)
        else:
            data_dict = opt.data
            
        img_ids = None
        val_path = data_dict.get('val', '')
        if isinstance(val_path, str) and val_path.endswith('.txt'):
            if os.path.exists(val_path):
                with open(val_path, 'r') as f:
                    lines = f.readlines()
                img_ids = []
                for line in lines:
                    line = line.strip()
                    if line:
                        stem = Path(line).stem
                        if stem.isdigit():
                            img_ids.append(int(stem))
                            
        from pycocotools.coco import COCO
        coco_gt = COCO(opt.ann_file)
        dataset = CocoEvalDataset(coco_gt_obj=coco_gt, img_dir=opt.img_dir, img_ids=img_ids)
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0, collate_fn=eval_collate_fn)
            
        rate_list = [opt.pruning_rate] if not isinstance(opt.pruning_rate, list) else opt.pruning_rate
        for rate in rate_list:
            prune_output = os.path.join(folder, opt.prune_output.replace('.txt', '_' + str(int(100 * rate)) + '.txt'))
            with open(prune_output, "w") as f:
                print("[", file=f)
            for i in range(conv_layers):
                print(f"\n--- Pruning sensitivity analysis: Layer {i+1}/{conv_layers} with rate {rate} ---")
                model_copy = deepcopy(base_model)
                r = test(opt.data,
                         opt.weights,
                         opt.batch_size,
                         opt.img_size,
                         opt.conf_thres,
                         opt.iou_thres,
                         False,
                         opt.single_cls,
                         opt.augment,
                         opt.verbose,
                         model=model_copy,
                         dataloader=dataloader,
                         pruning_params=[(i, rate)],
                         plots=False,
                         img_dir=opt.img_dir,
                         ann_file=opt.ann_file,
                         modification=opt.modification
                         )
                (mp, mr, map50, map, _, _, _, ), maps, t, params, fs = r
                with open(prune_output, "a") as f:
                    print(f"({i}, ({mp}, {mr}, {map50}, {map}, {list(maps)}, {t}, {params}, {fs})),", file=f)
            with open(prune_output, "a") as f:
                print("]", file=f)
            print(f"Sensitivity analysis for pruning rate {rate} saved to {prune_output}")
