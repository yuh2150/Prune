"""
Pruning sensitivity analysis execution module.

Supports two modes, selected via opt.modification:
  - 'prune-structured' (default): iterate every Conv2d layer, apply channel pruning, record mAP.
  - 'prune-layer': iterate every prunable C3/BottleneckCSP module, remove ONE Bottleneck, record mAP.

The output file format is identical in both modes so that existing layer_selection.py scripts work
unchanged.
"""

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Layer (depth) pruning sensitivity analysis
# ---------------------------------------------------------------------------

def layer_sensitivity_analysis(opt, evaluate_fn):
    """
    Run layer-pruning sensitivity analysis across all prunable C3 modules.

    For each C3 / BottleneckCSP module where len(module.m) > 1:
      - Remove exactly ONE Bottleneck (pruning_params=[(layer_id, 1)])
      - Evaluate mAP via evaluate_fn
      - Record: layer_id, original_depth, new_depth, mAP, ΔmAP, Params, GFLOPs

    Output file format is identical to run_sensitivity_analysis so that
    layer_selection.py works without modification.

    Args:
        opt (Namespace): Parsed CLI arguments (same as sensitivity_analysis.py uses).
        evaluate_fn (callable): The evaluate wrapper (same signature as in run_sensitivity_analysis).
    """
    import torch
    from evaluation.model_adapter import ModelLoader
    from prune import get_prunable_c3_layers

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    name = Path(opt.weights[0]).parent.parent.stem or Path(opt.weights[0]).stem
    folder = os.path.join('output', name)
    os.makedirs(folder, exist_ok=True)

    # ------------------------------------------------------------------
    # Baseline evaluation (unmodified model)
    # ------------------------------------------------------------------
    print("\n[layer_sensitivity_analysis] Evaluating baseline model...")
    r_base = evaluate_fn(
        opt.data,
        opt.weights,
        opt.batch_size,
        opt.img_size,
        opt.conf_thres,
        opt.iou_thres,
        False,            # save_json
        opt.single_cls,
        opt.augment,
        opt.verbose,
        save_txt=False,
        save_hybrid=False,
        save_conf=False,
        trace=not opt.no_trace,
        v5_metric=opt.v5_metric,
        pruning_params=[],
        plots=False,
    )
    (_, _, _, map_base, _, _, _), _, _, params_base, fs_base = r_base
    print(f"[layer_sensitivity_analysis] Baseline  mAP={map_base:.4f}  "
          f"Params={params_base}  GFLOPs={fs_base}")

    # ------------------------------------------------------------------
    # Discover prunable C3 layers from a fresh model load
    # ------------------------------------------------------------------
    device = torch.device('cpu')
    try:
        model, _, _, _ = ModelLoader.load_model(
            opt.weights, '', [], 0, device, imgsz=opt.img_size,
            trace=False, half_precision=False
        )
        prunable = get_prunable_c3_layers(model)
    except Exception as e:
        print(f"[layer_sensitivity_analysis] WARNING: could not load model to discover C3 layers ({e}).")
        prunable = []

    print(f"[layer_sensitivity_analysis] Found {len(prunable)} prunable C3/CSP module(s): "
          f"{[lid for lid, _ in prunable]}")

    if not prunable:
        print("[layer_sensitivity_analysis] Nothing to prune \u2014 exiting.")
        return

    # ------------------------------------------------------------------
    # One output file per pruning-rate entry (keeps same naming convention
    # as run_sensitivity_analysis so existing parsers work).
    # For layer pruning the "rate" is always 1 removed bottleneck, but we
    # still honour a rate_list so the CLI flag is consistent.
    # ------------------------------------------------------------------
    rate_list = [opt.pruning_rate] if not isinstance(opt.pruning_rate, list) else opt.pruning_rate

    # Import C3TR once (may not exist in pure YOLOv5)
    try:
        from models.common import C3TR
    except ImportError:
        C3TR = None

    for rate in rate_list:
        # Build the output filename with the same _XX.txt suffix convention
        try:
            suffix = str(int(100 * float(rate)))
        except (TypeError, ValueError):
            suffix = str(rate)
        prune_output = os.path.join(
            folder,
            opt.prune_output.replace('.txt', f'_layer_{suffix}.txt')
        )

        with open(prune_output, 'w') as f:
            print('[', file=f)

        for layer_id, module in prunable:
            # Determine original depth
            if C3TR is not None and isinstance(module, C3TR):
                original_depth = len(module.m.tr)
            else:
                original_depth = len(module.m)
            new_depth = original_depth - 1          # always remove exactly 1

            print(f"\n[layer_sensitivity_analysis] "
                  f"C3 layer_id={layer_id}  depth {original_depth} -> {new_depth}")

            # Temporarily set modification so ModelLoader/evaluate routes correctly
            saved_mod = opt.modification
            opt.modification = 'prune-layer'

            r = evaluate_fn(
                opt.data,
                opt.weights,
                opt.batch_size,
                opt.img_size,
                opt.conf_thres,
                opt.iou_thres,
                False,
                opt.single_cls,
                opt.augment,
                opt.verbose,
                save_txt=False,
                save_hybrid=False,
                save_conf=False,
                trace=not opt.no_trace,
                v5_metric=opt.v5_metric,
                pruning_params=[(layer_id, 1)],
                plots=False,
            )
            opt.modification = saved_mod

            (mp, mr, map50, map_val, _, _, _), maps, t, params, fs = r
            delta_map = map_val - map_base

            print(f"  layer_id={layer_id}  orig_depth={original_depth}  new_depth={new_depth}  "
                  f"mAP={map_val:.4f}  \u0394mAP={delta_map:+.4f}  "
                  f"Params={params}  GFLOPs={fs}")

            # Write in the same tuple format as run_sensitivity_analysis
            with open(prune_output, 'a') as f:
                print(
                    f"({layer_id}, ({mp}, {mr}, {map50}, {map_val}, "
                    f"{list(maps)}, {t}, {params}, {fs})),",
                    file=f
                )

        with open(prune_output, 'a') as f:
            print(']', file=f)

        print(f"[layer_sensitivity_analysis] Results saved to {prune_output}")


# ---------------------------------------------------------------------------
# Structured (channel) pruning sensitivity analysis  — original implementation
# ---------------------------------------------------------------------------

def run_sensitivity_analysis(opt, evaluate_fn):
    """
    Run pruning sensitivity analysis.

    Dispatches to layer_sensitivity_analysis() when opt.modification == 'prune-layer',
    otherwise runs the original structured (channel) pruning sweep across Conv2d layers.

    Args:
        opt (Namespace): Parsed command-line arguments.
        evaluate_fn (callable): Backward-compatible evaluate wrapper function.
    """
    if getattr(opt, 'modification', '') == 'prune-layer':
        return layer_sensitivity_analysis(opt, evaluate_fn)

    # ------------------------------------------------------------------
    # Original structured-pruning sensitivity analysis
    # ------------------------------------------------------------------
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
        with open(prune_output, 'w') as f:
            print('[', file=f)

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
            (mp, mr, map50, map, _, _, _,), maps, t, params, fs = r
            with open(prune_output, 'a') as f:
                print(f"({i}, ({mp}, {mr}, {map50}, {map}, {list(maps)}, {t}, {params}, {fs})),", file=f)

        with open(prune_output, 'a') as f:
            print(']', file=f)

        print(f"Sensitivity analysis for pruning rate {rate} saved to {prune_output}")
