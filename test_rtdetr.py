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

from prune_framework.core import PluginRegistry, PruningEngine
from prune_framework.modules.model.loader import ModelLoader
from prune_framework.modules.evaluation.rtdetr_processors import RTDetrPreProcessor, RTDetrPostProcessor
from prune_framework.modules.evaluation.coco_evaluator import COCOEvaluator
from prune_framework.modules.evaluation.pipeline import EvaluationPipeline
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
        if isinstance(weights_dir, str) and (weights_dir.endswith('.pt') or weights_dir.endswith('.pth')):
            print(f"Loading pruned model checkpoint from {weights_dir}...")
            try:
                ckpt = torch.load(weights_dir, map_location=device, weights_only=False)
            except TypeError:
                ckpt = torch.load(weights_dir, map_location=device)
            if isinstance(ckpt, dict) and 'model_object' in ckpt:
                model = ckpt['model_object']
            elif isinstance(ckpt, dict) and 'model' in ckpt:
                model = ckpt['model']
            else:
                model = ckpt
        else:
            model, _ = ModelLoader.load("rtdetr", weights_dir, device)
            if pruning_params:
                pruner_type = "depth" if modification == "prune-layer" else ("unstructured" if modification == "prune-unstructured" else "structured")
                criterion_type = "l2" if criterion in [0, 1] else "l1"
                engine = PruningEngine("rtdetr", pruner_type, criterion_type)
                model = engine.execute(model, {"pruning_params": pruning_params, "amount": pruning_params if isinstance(pruning_params, float) else 0.3}).adapter_cls(model).model
    else:
        if pruning_params:
            pruner_type = "depth" if modification == "prune-layer" else ("unstructured" if modification == "prune-unstructured" else "structured")
            criterion_type = "l2" if criterion in [0, 1] else "l1"
            engine = PruningEngine("rtdetr", pruner_type, criterion_type)
            engine.execute(model, {"pruning_params": pruning_params, "amount": pruning_params if isinstance(pruning_params, float) else 0.3})

            
    # Load image processor and config
    hf_dir = 'PekingU/rtdetr_r18vd' if (isinstance(weights_dir, str) and (weights_dir.endswith('.pt') or weights_dir.endswith('.pth'))) else weights_dir
    try:
        image_processor = RTDetrImageProcessor.from_pretrained(hf_dir, local_files_only=True)
        config = RTDetrConfig.from_pretrained(hf_dir, local_files_only=True)
    except Exception:
        image_processor = RTDetrImageProcessor.from_pretrained(hf_dir, local_files_only=False)
        config = RTDetrConfig.from_pretrained(hf_dir, local_files_only=False)
    
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
    
    # Instantiate decoupled PreProcessor, PostProcessor, Evaluator and Pipeline
    preprocessor = RTDetrPreProcessor(hf_dir)
    postprocessor = RTDetrPostProcessor(hf_dir)
    w_name = Path(weights_dir).stem if weights_dir else "rtdetr"
    pred_json = str(save_dir / f"{w_name}_predictions.json") if save_dir else None
    evaluator = COCOEvaluator(coco_gt, save_json_path=pred_json)

    eval_pipeline = EvaluationPipeline(preprocessor, postprocessor, evaluator)

    print(f"\nRunning evaluation pipeline for {len(dataloader.dataset)} images...")
    eval_result = eval_pipeline.run(
        model=model,
        dataloader=dataloader,
        device=device,
        conf_thres=conf_thres,
        half=half,
        show_progress=True
    )

    map = eval_result.map
    map50 = eval_result.map50
    mp = map50
    mr = eval_result.metrics.get("mar", 0.0)
    seen = eval_result.num_samples

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4
    print(pf % ('all', seen, 0, mp, mr, map50, map))

    # Speeds placeholder/dummy speed tuple
    t = (0.0, 0.0, 0.0, imgsz, imgsz, batch_size)
    print('Speed: %.1f/%.1f/%.1f ms inference/postprocess/total per %gx%g image at batch-size %g' % t)

    # Save results to txt in save_dir
    with open(os.path.join(save_dir, 'results.txt'), "w") as r_ap:
        print(map, file=r_ap)

        
    maps = np.zeros(nc) + map
    if eval_result.num_samples > 0:
        try:
            pass
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
    parser.add_argument('--modification', default='', help='prune-structured, prune-unstructured, prune-layer')
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
        
        if opt.modification == 'prune-layer':
            print("\n--- Running Layer Pruning (Depth) Sensitivity Analysis ---")
            print(f"Rates to evaluate: {rate_list}")
            
            # Evaluate baseline first for reference
            print("\nEvaluating Baseline Model...")
            baseline_copy = deepcopy(base_model)
            r_base = test(opt.data,
                          opt.weights,
                          opt.batch_size,
                          opt.img_size,
                          opt.conf_thres,
                          opt.iou_thres,
                          False,
                          opt.single_cls,
                          opt.augment,
                          opt.verbose,
                          model=baseline_copy,
                          dataloader=dataloader,
                          pruning_params=[],
                          plots=False,
                          img_dir=opt.img_dir,
                          ann_file=opt.ann_file,
                          modification=''
                          )
            (mp_b, mr_b, map50_b, map_b, _, _, _), _, _, params_b, fs_b = r_base
            print(f"Baseline: mAP={map_b:.4f}, mAP50={map50_b:.4f}, Params={params_b:,}, GFLOPs={fs_b}")
            
            results = []
            results.append({
                'Ratio': 'Baseline',
                'Params': f"{params_b:,}",
                'Param Reduc %': '0.00',
                'GFLOPs': f"{fs_b:.2f}",
                'GFLOP Reduc %': '0.00',
                'mAP': f"{map_b:.4f}",
                'mAP50': f"{map50_b:.4f}"
            })
            
            for rate in rate_list:
                print(f"\n--- Evaluating Pruning Ratio: {rate} ---")
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
                         pruning_params=rate,
                         plots=False,
                         img_dir=opt.img_dir,
                         ann_file=opt.ann_file,
                         modification='prune-layer'
                         )
                (mp, mr, map50, map_val, _, _, _), _, _, params, fs = r
                
                param_reduc = (1.0 - params / params_b) * 100 if params_b > 0 else 0.0
                gflop_reduc = (1.0 - fs / fs_b) * 100 if fs_b > 0 else 0.0
                
                print(f"Ratio {rate}: mAP={map_val:.4f}, mAP50={map50:.4f}, Params={params:,} (Reduc {param_reduc:.2f}%), GFLOPs={fs} (Reduc {gflop_reduc:.2f}%)")
                
                results.append({
                    'Ratio': f"{rate:.2f}" if isinstance(rate, float) else str(rate),
                    'Params': f"{params:,}",
                    'Param Reduc %': f"{param_reduc:.2f}",
                    'GFLOPs': f"{fs:.2f}",
                    'GFLOP Reduc %': f"{gflop_reduc:.2f}",
                    'mAP': f"{map_val:.4f}",
                    'mAP50': f"{map50:.4f}"
                })
                
            # Create DataFrame and save
            import pandas as pd
            df = pd.DataFrame(results)
            csv_path = os.path.join(folder, 'pruning_results_summary.csv')
            df.to_csv(csv_path, index=False)
            print(f"\nSummary table saved to {csv_path}")
            print(df.to_string(index=False))
            
        else:
            prunable_layers = get_prunable_layers(base_model)
            conv_layers = len(prunable_layers)
            print(f"Snapshotting prunable Conv2d layers. Found {conv_layers} prunable layers.")
            
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
