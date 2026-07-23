import argparse
from models.experimental import attempt_download
from utils.torch_utils import model_info
from utils.general import make_divisible
import torch
import torch.nn as nn
import yaml
import ast
import numpy as np

# Safe imports for YOLOv7 components that might be missing in YOLOv5
try:
    from models.common import RepConv
except ImportError:
    class RepConv:
        pass

try:
    from models.yolo import IDetect
except ImportError:
    class IDetect:
        pass

def prune_single_layer(model, layer_index, percentage, criterion):
    next_layers = model.pruning_cfg[layer_index][0]
    concat_slices = model.pruning_cfg[layer_index][1]

    # get all modules with convolutional layers
    conv_modules = get_conv_layers(model)

    if layer_index < 0 or layer_index > len(conv_modules) - 1:
        raise Exception(f"layer {layer_index} index exceeded maximum number of layers: {len(conv_modules) - 1}")

    # current conv module that will be pruned
    # conv_module consists of a convolutional layer, batch normalization layer and SiLU activation
    conv_module = conv_modules[layer_index]

    # extract convolutional and batch normalization layers from the current conv_module
    # RepConv has two convolutional layers which is why conv_layers and bn_layers are lists
    conv_layers, bn_layers, _ = extract_layers_from_module(conv_module)

    num_params_before_pruning = sum([c.weight.numel() for c in conv_layers])

    device = conv_layers[0].weight.device
    c_out = conv_layers[0].out_channels # number of output channels of current layer

    pruned_indices = determine_pruned_indices(conv_layers, bn_layers, percentage, criterion)

    for c in conv_layers:
        new_weights = torch.index_select(c.weight, dim=0, index=pruned_indices).contiguous()
        c.weight = torch.nn.Parameter(new_weights)
        c.out_channels = len(pruned_indices)

    for b in bn_layers:
        new_weights = torch.index_select(b.weight, dim=0, index=pruned_indices).contiguous()
        new_bias = torch.index_select(b.bias, dim=0, index=pruned_indices).contiguous()
        new_running_var = torch.index_select(b.running_var, dim=0, index=pruned_indices).contiguous()
        new_running_mean = torch.index_select(b.running_mean, dim=0, index=pruned_indices).contiguous()

        b.num_features = len(pruned_indices)
        b.weight = torch.nn.Parameter(new_weights)
        b.running_var = torch.nn.Parameter(new_running_var).detach()
        b.running_mean = torch.nn.Parameter(new_running_mean).detach()
        b.bias = torch.nn.Parameter(new_bias)

    num_params_after_pruning_current_layer = sum([c.weight.numel() for c in conv_layers])

    print(
        f"\nLayer {layer_index}: Removed {c_out - len(pruned_indices)}/{c_out} filters based on criterion {criterion} and pruned {num_params_before_pruning - num_params_after_pruning_current_layer}/{num_params_before_pruning} ({round(100 * float(num_params_before_pruning - num_params_after_pruning_current_layer) / num_params_before_pruning, 2)}%) of the parameters in this convolutional layer. Pruning this layer affects the following layers:")

    # prune all layers that are affected by pruning the layer at index layer_index
    for next_layer, slice in zip(next_layers, concat_slices):
        offset = int(np.sum(model.pruning_cfg[next_layer][2][:slice]))
        if next_layer < 0 or next_layer > len(conv_modules) - 1:
            raise Exception(f"next_layer {next_layer} index exceeded maximum number of layers: {len(conv_modules) - 1}")
        next_conv_module = conv_modules[next_layer]
        next_conv_layers, _, next_implicits = extract_layers_from_module(next_conv_module)

        num_params_n = sum([next_conv.weight.numel() for next_conv in next_conv_layers])

        n_in = next_conv_layers[0].in_channels # number of input channels of the next layer

        # update slice sizes in pruning_cfg
        if len(model.pruning_cfg[next_layer][2]) > 0:
            model.pruning_cfg[next_layer][2][slice] -= c_out - len(pruned_indices)
        # indices need to be modified in case the next layer has concatenated input channels
        if offset > n_in-c_out:
            offset = n_in-c_out                         # fixed bug
        pruned_indices_next_layer = torch.cat(
            (torch.arange(offset).to(device), pruned_indices.to(device) + offset, torch.arange(offset + c_out, n_in).to(device)))
        for next_conv in next_conv_layers:
            new_weights = torch.index_select(next_conv.weight, dim=1, index=pruned_indices_next_layer).contiguous()
            next_conv.weight = torch.nn.Parameter(new_weights)
            next_conv.in_channels = n_in - c_out + len(pruned_indices)
        for next_implicit in next_implicits:
            new_implicit = torch.index_select(next_implicit.implicit, dim=1, index=pruned_indices_next_layer).contiguous()
            next_implicit.implicit = nn.Parameter(new_implicit)
            next_implicit.channel = n_in - c_out + len(pruned_indices)
        num_params_n_pruned = sum([next_conv.weight.numel() for next_conv in next_conv_layers])
        print(
            f"    Layer {next_layer}: Removed {n_in-len(pruned_indices_next_layer)}/{n_in} input channels and pruned {num_params_n - num_params_n_pruned}/{num_params_n} ({round(100 * float(num_params_n - num_params_n_pruned) / num_params_n, 2)}%) of the parameters in this convolutional layer.")

def extract_layers_from_module(conv_module):
    from models.common import Conv
    conv_layers = []
    bn_layers = []
    implicits = []
    if isinstance(conv_module, Conv):
        conv_layers.append(conv_module.conv)
        bn_layers.append(conv_module.bn)
    elif isinstance(conv_module, RepConv):   # RepConv has two convolutional layers
        conv_layers.append(conv_module.rbr_dense[0])
        conv_layers.append(conv_module.rbr_1x1[0])
        bn_layers.append(conv_module.rbr_dense[1])
        bn_layers.append(conv_module.rbr_1x1[1])
    elif isinstance(conv_module, nn.Conv2d):
        conv_layers.append(conv_module)
    elif len(conv_module) > 1:
        conv_layers.append(conv_module[0])
        implicits.append(conv_module[1])
    return conv_layers, bn_layers, implicits

# get all modules with convolutional layers
def get_conv_layers(model):
    from models.common import Conv
    from models.yolo import Detect
    conv_layers = []
    for p in model.modules():
        if isinstance(p, Conv) or isinstance(p, RepConv):
            conv_layers.append(p)
        elif isinstance(p, Detect):
            for conv in p.m:
                conv_layers.append(conv)
        elif isinstance(p, IDetect):
            for conv, ia, im in zip(p.m, p.ia, p.im):
                conv_layers.append([conv, ia, im])
    return conv_layers

def determine_pruned_indices(conv, bn, percentage, criterion=0):
    device = conv[0].weight.device
    importances = torch.zeros_like(torch.norm(conv[0].weight, p=2, dim=[1, 2, 3]))
    if criterion == 0: # smallest L2
        for c in conv:  # loop in case conv has two convolutional layers (it usually has just one)
            importances += torch.norm(c.weight, p=2, dim=[1, 2, 3])
    elif criterion == 1: # largest L2
        for c in conv:
            importances -= torch.norm(c.weight, p=2, dim=[1, 2, 3])
    elif criterion == 2: # smallest L1
        for c in conv:
            importances += torch.norm(c.weight, p=1, dim=[1, 2, 3])
    elif criterion == 3: # largest L1
        for c in conv:
            importances -= torch.norm(c.weight, p=1, dim=[1, 2, 3])
    elif criterion == 4: # smallest bn scale factor
        for b in bn:
            importances += b.weight
    elif criterion == 5: # smallest bn scale factor * L1 norm
        for c, b in zip(conv, bn):
            importances += b.weight * torch.norm(c.weight, p=1, dim=[1, 2, 3])
    elif criterion == 6: # random
        importances += torch.rand_like(importances)

    indices = torch.argsort(importances)
    indices.to(device)
    n_to_prune = make_divisible(percentage * len(indices), 2)   # filters
    return torch.sort(indices[n_to_prune:])[0]

def num_params(model):
    a = 0
    for p in model.parameters():
        a += p.numel()
    return a

def prune_structured(model, pruning_params, criterion, tiny=False):
    import os
    pruning_cfg = 'cfg/pruning_config/yolov7-tiny_pruning_cfg.yaml' if tiny else 'cfg/pruning_config/yolov7_pruning_cfg.yaml'
    use_tp = not os.path.exists(pruning_cfg) or 'yolov5' in str(type(model)).lower()

    if type(pruning_params) == float and pruning_params < 1 and pruning_params > 0:
        num_conv_layers = len([conv for conv in model.modules() if isinstance(conv, nn.Conv2d)])
        pruning_params = [(i, pruning_params) for i in range(num_conv_layers)]
    pruning_params = sorted(pruning_params, key=lambda x: x[0])  # sorting is important to ensure that conv layers after concat layers are pruned from back to front
    print('Pruning model... ')
    num_params_before = num_params(model)

    if use_tp:
        import torch_pruning as tp
        print("Using torch_pruning for structured pruning...")
        # Get all conv modules
        conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        
        # Trace dependency graph
        # Trace dependency graph with matching device and dtype
        first_param = next(model.parameters())
        x = torch.randn(1, 3, 640, 640).to(device=first_param.device, dtype=first_param.dtype)
        
        # Save original requires_grad states and set them to True for tracing
        grad_states = {}
        for name, p in model.named_parameters():
            grad_states[name] = p.requires_grad
            p.requires_grad = True
            
        DG = tp.DependencyGraph().build_dependency(model, x)
        
        # Restore original requires_grad states
        for name, p in model.named_parameters():
            p.requires_grad = grad_states[name]
        
        with torch.no_grad():
            for layer_idx, amount in pruning_params:
                if layer_idx < 0 or layer_idx >= len(conv_layers):
                    continue
                target_conv = conv_layers[layer_idx]
                out_channels = target_conv.out_channels
                
                # Skip if it is a Detect layer head output
                is_detect = False
                for m in model.modules():
                    if 'detect' in m.__class__.__name__.lower():
                        if target_conv in m.modules():
                            is_detect = True
                            break
                if is_detect:
                    print(f"Skipping Detect layer conv at index {layer_idx}")
                    continue
                    
                n_pruned = make_divisible(amount * out_channels, 2)
                if n_pruned >= out_channels:
                    n_pruned = out_channels - 2  # keep at least 2 channels
                if n_pruned <= 0:
                    continue
                    
                # Determine indices to prune based on criterion
                weights = target_conv.weight.data
                importances = torch.zeros(out_channels, device=weights.device)
                if criterion in [0, 1]:
                    importances = torch.norm(target_conv.weight.data, p=2, dim=[1, 2, 3])
                    if criterion == 1:
                        importances = -importances
                elif criterion in [2, 3]:
                    importances = torch.norm(target_conv.weight.data, p=1, dim=[1, 2, 3])
                    if criterion == 3:
                        importances = -importances
                elif criterion == 6:
                    importances = torch.rand(out_channels, device=weights.device)
                else:
                    importances = torch.norm(target_conv.weight.data, p=2, dim=[1, 2, 3])
                    
                idxs = torch.argsort(importances)[:n_pruned].tolist()
                
                group = DG.get_pruning_group(target_conv, tp.prune_conv_out_channels, idxs)
                if DG.check_pruning_group(group):
                    group.prune()
                    print(f"Layer {layer_idx}: Pruned {n_pruned}/{out_channels} channels. New out_channels={target_conv.out_channels}")
                
        num_params_after = num_params(model)
        print(f'\nPruned {num_params_before - num_params_after}/{num_params_before} parameters in total. Global sparsity: {round(100 * float(num_params_before - num_params_after) / num_params_before, 2)}%\n')
        model_info(model)
        return model

    # Fallback to rule-based manual pruning (YOLOv7 only)
    if not hasattr(model, "pruning_cfg"):
        with open(pruning_cfg, encoding='utf-8') as f:
            model.pruning_cfg = yaml.load(f, Loader=yaml.SafeLoader)['pruning_cfg']
    conv_modules = get_conv_layers(model)

    with torch.no_grad():
        for layer, amount in pruning_params:
            if isinstance(conv_modules[layer], RepConv):
                out = conv_modules[layer].rbr_dense[0].out_channels
            else:
                out = conv_modules[layer].conv.out_channels
            n_prune = make_divisible(amount * out, 2)
            if n_prune - out:
                prune_single_layer(model, layer, amount, criterion)
    num_params_after = num_params(model)
    print(
        f'\nPruned {num_params_before - num_params_after}/{num_params_before} parameters in total. Global sparsity: {round(100 * float(num_params_before - num_params_after) / num_params_before, 2)}%\n')
    model_info(model)

    return model


def get_prunable_c3_layers(model):
    """
    Find every C3 / BottleneckCSP module in *model.model* (the flat layer list)
    whose Bottleneck Sequential has length > 1, i.e. it can have at least one
    Bottleneck removed while still leaving one remaining.

    Only iterates the top-level model.model list — this intentionally excludes
    Detect, SPPF, Conv, Upsample and Concat which are never C3 subclasses.

    Args:
        model: YOLO model instance (must have a .model attribute).

    Returns:
        list[tuple[int, nn.Module]]: (layer_id, module) pairs, where layer_id
        is the index inside model.model.
    """
    from models.common import C3, BottleneckCSP
    try:
        from models.common import C3TR
    except ImportError:
        C3TR = None

    prunable = []
    if not hasattr(model, 'model'):
        return prunable

    for i, module in enumerate(model.model):
        if not isinstance(module, (C3, BottleneckCSP)):
            continue
        if C3TR is not None and isinstance(module, C3TR):
            # C3TR stores its transformer blocks under module.m.tr
            if hasattr(module.m, 'tr') and isinstance(module.m.tr, nn.Sequential) \
                    and len(module.m.tr) >= 1:
                prunable.append((i, module))
        else:
            # Standard C3 / BottleneckCSP store bottlenecks in module.m
            if hasattr(module, 'm') and isinstance(module.m, nn.Sequential) \
                    and len(module.m) >= 1:
                prunable.append((i, module))
    return prunable


def _score_bottlenecks(seq: nn.Sequential, criterion: int) -> torch.Tensor:
    """
    Compute a scalar importance score for every Bottleneck in *seq*.
    Higher score = more important = should be KEPT.

    Uses the 3×3 conv (cv2) weight as the representative tensor, mirroring
    the same ``criterion`` enum used by ``determine_pruned_indices()`` for
    structured channel pruning.

    Criterion values:
        0  – L2 norm (keep largest)   ← default for prune-layer
        1  – L2 norm (keep smallest)
        2  – L1 norm (keep largest)
        3  – L1 norm (keep smallest)
        4  – mean absolute BN γ (keep largest)
        6  – random (ablation / baseline)
    """
    scores = []
    for block in seq:
        # cv2 is the 3×3 conv — most discriminative weight in a Bottleneck
        w = block.cv2.conv.weight.data
        if criterion == 0:    # L2, keep largest
            s = torch.norm(w, p=2)
        elif criterion == 1:  # L2, keep smallest
            s = -torch.norm(w, p=2)
        elif criterion == 2:  # L1, keep largest
            s = torch.norm(w, p=1)
        elif criterion == 3:  # L1, keep smallest
            s = -torch.norm(w, p=1)
        elif criterion == 4:  # BN gamma magnitude
            s = block.cv2.bn.weight.data.abs().mean()
        elif criterion == 6:  # random  (ablation)
            s = torch.rand(1).squeeze()
        else:                 # fallback → L1 largest
            s = torch.norm(w, p=1)
        scores.append(s.item())
    return torch.tensor(scores, dtype=torch.float32)


def prune_layers(model, pruning_params, criterion: int = 0):
    """
    Physically remove Bottleneck blocks from C3 modules using importance scoring.

    For each (layer_id, remove_num) pair the function:
      1. Scores every Bottleneck in the target C3 module via ``_score_bottlenecks``.
      2. Removes the ``remove_num`` *lowest-scoring* blocks.
      3. Keeps the remaining blocks in their original order.

    Only C3 / BottleneckCSP modules are modified.  Detect, SPPF, Conv, Upsample
    and Concat are never touched.

    Args:
        model:          YOLO model instance.
        pruning_params: List of (layer_id, remove_num) tuples.
                        layer_id  – index inside model.model.
                        remove_num – number of Bottlenecks to drop.
                        Example: [(4, 1)] drops the least-important Bottleneck
                        from C3 block #4.
        criterion:      Importance metric (same enum as determine_pruned_indices).
                        0=L2-keep-largest, 1=L2-keep-smallest,
                        2=L1-keep-largest (recommended), 3=L1-keep-smallest,
                        4=BN-gamma, 6=random.

    Returns:
        model: The same model object, modified in-place.
    """
    from models.common import C3, BottleneckCSP
    try:
        from models.common import C3TR
    except ImportError:
        C3TR = None

    if not hasattr(model, 'model'):
        return model

    with torch.no_grad():
        for layer_id, remove_num in pruning_params:
            if layer_id < 0 or layer_id >= len(model.model):
                print(f"prune_layers: layer_id {layer_id} out of range — skipping.")
                continue
            module = model.model[layer_id]
            if not isinstance(module, (C3, BottleneckCSP)):
                print(f"prune_layers: model.model[{layer_id}] is "
                      f"{type(module).__name__}, not C3/CSP — skipping.")
                continue

            # --- C3TR: no importance scoring (transformer blocks differ structurally) ---
            if C3TR is not None and isinstance(module, C3TR):
                if hasattr(module.m, 'tr') and isinstance(module.m.tr, nn.Sequential):
                    n_orig = len(module.m.tr)
                    n_new = max(0, n_orig - remove_num)
                    if n_new < n_orig:
                        # C3TR: keep first n_new (no weight-based scoring available)
                        module.m.tr = nn.Sequential(*list(module.m.tr)[:n_new])
                        print(f"  Pruned C3TR at model.model[{layer_id}]: "
                              f"{n_orig} -> {n_new} transformer layers.")
                continue

            # --- Standard C3 / BottleneckCSP: importance-based selection ---
            if not (hasattr(module, 'm') and isinstance(module.m, nn.Sequential)):
                continue

            seq = module.m
            n_orig = len(seq)
            n_new = max(0, n_orig - remove_num)
            if n_new >= n_orig:
                continue

            if remove_num >= n_orig:
                print(f"  WARNING: remove_num={remove_num} >= depth={n_orig} at "
                      f"model.model[{layer_id}]; keeping 0 blocks.")

            # Score each block; keep the n_new highest-scoring ones
            scores = _score_bottlenecks(seq, criterion)
            keep_idx = torch.argsort(scores, descending=True)[:n_new]
            keep_idx = keep_idx.sort().values.tolist()   # restore sequential order
            removed_idx = [i for i in range(n_orig) if i not in keep_idx]

            module.m = nn.Sequential(*[seq[i] for i in keep_idx])
            print(f"  Pruned C3/CSP at model.model[{layer_id}] (criterion={criterion}): "
                  f"{n_orig} -> {n_new} blocks. "
                  f"Removed indices {removed_idx}, kept {keep_idx}.")

    return model

def load_pruned_model(weights, pruning_params, criterion=0, map_location=None, save=None,
                      modification='prune-structured'):
    """
    Load a YOLO checkpoint and apply either structured channel pruning or layer
    (depth) pruning, then return the fused FP32 model ready for evaluation.

    Args:
        weights: path(s) to the .pt checkpoint.
        pruning_params: parameter list whose meaning depends on *modification*:
            - prune-structured: [(layer_idx, prune_rate), ...]
            - prune-layer:      [(layer_id,  remove_num), ...]
        criterion (int): importance criterion for structured pruning (ignored for prune-layer).
        map_location: torch.load device override.
        save: if given, save the pruned checkpoint to this path.
        modification (str): 'prune-structured' (default) | 'prune-layer' | 'prune-unstructured'.

    Returns:
        model: pruned, fused, float32, eval-mode YOLO model.
    """
    # Allow passing a device object as the third positional arg (legacy call style)
    if not isinstance(criterion, int) and map_location is None:
        map_location = criterion
        criterion = 0

    for w in weights if isinstance(weights, list) else [weights]:
        attempt_download(w)
        try:
            ckpt = torch.load(w, map_location=map_location, weights_only=False)
        except TypeError:
            ckpt = torch.load(w, map_location=map_location)
        model = ckpt['ema' if ckpt.get('ema') else 'model']

    if modification == 'prune-layer':
        pruned_model = prune_layers(model, pruning_params, criterion=criterion)
    else:
        # Default: structured channel pruning (also covers 'prune-structured')
        pruned_model = prune_structured(model, pruning_params, criterion, tiny='tiny' in weights[0])

    if save:
        ckpt['model'] = pruned_model
        torch.save(ckpt, save)

    return pruned_model.float().fuse().eval()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='prune.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--pruning-params', type=str, default='',
                        help='List of (layer_id, value) tuples. For prune-structured: value is prune rate [0,1]. '
                             'For prune-layer: value is number of Bottlenecks to remove. '
                             'Example: "[(4,1),(6,1)]"')
    parser.add_argument('--criterion', type=int, default=0,
                        help='Importance criterion for structured pruning: '
                             '0=smallest L2 | 1=largest L2 | 2=smallest L1 | 3=largest L1 | '
                             '4=smallest BN scale | 5=BN scale * L1 | 6=random')
    parser.add_argument('--modification', type=str, default='prune-structured',
                        help='prune-structured | prune-layer')
    parser.add_argument('--name', default=None, help='save pruned model to this path')
    opt = parser.parse_args()

    pruning_params_parsed = []
    if len(opt.pruning_params) > 0:
        pruning_params_parsed = ast.literal_eval(opt.pruning_params)

    # Examples:
    #   python prune.py --weights yolov5s.pt --modification prune-structured --pruning-params "[(0,0.3)]" --name pruned.pt
    #   python prune.py --weights yolov5s.pt --modification prune-layer      --pruning-params "[(4,1),(6,1)]" --name layer_pruned.pt
    load_pruned_model(opt.weights, pruning_params_parsed, opt.criterion,
                      save=opt.name, modification=opt.modification)