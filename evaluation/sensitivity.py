"""
Pruning sensitivity analysis execution module.
"""

import os
from pathlib import Path
from copy import deepcopy
import torch
from utils.torch_utils import select_device
from evaluation.model_adapter import ModelLoader, PrunerAdapterFactory

def run_sensitivity_analysis(opt, evaluate_fn):
    """
    Run structured pruning sensitivity analysis across model layers and pruning rates.
    
    Args:
        opt (Namespace): Parsed command-line arguments.
        evaluate_fn (callable): Backward-compatible test/evaluate function reference.
    """
    name = Path(opt.weights[0]).parent.parent.stem if Path(opt.weights[0]).parent.parent.stem \
        else Path(opt.weights[0]).stem
    folder = os.path.join('output', name)
    os.makedirs(folder, exist_ok=True)
    
    cpu_device = torch.device('cpu')
    device = select_device(opt.device if hasattr(opt, 'device') else '')
    
    print("Loading baseline model on CPU for sensitivity analysis...")
    try:
        baseline_model, _, _, _ = ModelLoader.load_model(
            opt.weights, '', [], 0, cpu_device, imgsz=opt.img_size, trace=False, half_precision=False
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

    pruner = PrunerAdapterFactory.get_pruner(opt.weights, model=baseline_model)
    prunable_layers = pruner.get_prunable_layers(baseline_model)
    
    rate_list = [opt.pruning_rate] if not isinstance(opt.pruning_rate, list) else opt.pruning_rate
    
    for rate in rate_list:
        prune_output = os.path.join(folder, opt.prune_output.replace('.txt', '_' + str(int(100 * rate)) + '.txt'))
        
        with open(prune_output, "w") as f:
            print("[", file=f)
            
        print(f"\nEvaluating sensitivity on {len(prunable_layers)} layers at pruning rate {rate}...")
        
        for layer_info in prunable_layers:
            print(f"\n>>> Analyzing layer: {layer_info.name} ({layer_info.layer_type})")
            
            # Deepcopy on CPU to preserve GPU memory
            pruned_model = deepcopy(baseline_model)
            
            # Prune on CPU
            pruned_model = pruner.prune_layer(pruned_model, layer_info.name, rate, opt.criterion)
            
            # Move pruned model to the target device for fast evaluation
            pruned_model.to(device)
            
            # Run evaluate with plots=False, save_txt/save_hybrid/save_conf=False, save_json=False
            r = evaluate_fn(
                opt.data,
                None,  # weights = None, load model passed via parameter instead
                opt.batch_size,
                opt.img_size,
                opt.conf_thres,
                opt.iou_thres,
                False,  # save_json
                opt.single_cls,
                opt.augment,
                opt.verbose,
                save_txt=False,
                save_hybrid=False,
                save_conf=False,
                trace=not opt.no_trace,
                v5_metric=opt.v5_metric,
                pruning_params=None,
                plots=False,
                model=pruned_model,
            )
            (mp, mr, map50, map, _, _, _, ), maps, t, params, fs = r
            
            # Log results by unique layer name
            with open(prune_output, "a") as f:
                print(f"('{layer_info.name}', ({mp}, {mr}, {map50}, {map}, {list(maps)}, {t}, {params}, {fs})),", file=f)
                
        with open(prune_output, "a") as f:
            print("]", file=f)
            
        print(f"Sensitivity analysis for pruning rate {rate} saved to {prune_output}")

