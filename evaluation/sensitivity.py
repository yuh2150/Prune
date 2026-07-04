"""
Pruning sensitivity analysis execution module.
"""

import os
from pathlib import Path

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
    
    # Dynamically determine the number of convolutional layers
    from evaluation.model_adapter import ModelLoader
    from prune import get_conv_layers
    import torch
    
    device = torch.device('cpu')
    try:
        model, _, _, _ = ModelLoader.load_model(
            opt.weights, '', [], 0, device, imgsz=opt.img_size, trace=False, half_precision=False
        )
        conv_layers = len(get_conv_layers(model))
    except Exception as e:
        print(f"Warning: Could not dynamically load model to count conv layers ({e}). Falling back to default.")
        conv_layers = 55 if 'tiny' in opt.weights[0] else 89
    rate_list = [opt.pruning_rate] if not isinstance(opt.pruning_rate, list) else opt.pruning_rate
    
    for rate in rate_list:
        prune_output = os.path.join(folder, opt.prune_output.replace('.txt', '_' + str(int(100 * rate)) + '.txt'))
        with open(prune_output, "a") as f:
            print("[", file=f)
            
        for i in range(conv_layers):
            # Run evaluate with plots=False, save_txt/save_hybrid/save_conf=False, save_json=False
            r = evaluate_fn(
                opt.data,
                opt.weights,
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
                pruning_params=[(i, rate)],
                plots=False,
            )
            (mp, mr, map50, map, _, _, _, ), maps, t, params, fs = r
            with open(prune_output, "a") as f:
                print(f"({i}, ({mp}, {mr}, {map50}, {map}, {list(maps)}, {t}, {params}, {fs})),", file=f)
                
        with open(prune_output, "a") as f:
            print("]", file=f)
            
        print(f"Sensitivity analysis for pruning rate {rate} saved to {prune_output}")
