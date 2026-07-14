import os
import torch
import torch.nn as nn
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor, RTDetrConfig

def attempt_load_rtdetr(weights_dir, device):
    """
    Load RTDetrForObjectDetection + RTDetrImageProcessor + config from HuggingFace format.
    
    Args:
        weights_dir (str): Path to HF model directory containing model.safetensors, config.json, etc.
        device (torch.device): Device to load the model on.
        
    Returns:
        model (RTDetrForObjectDetection)
        image_processor (RTDetrImageProcessor)
        config (RTDetrConfig)
    """
    print(f"Loading RT-DETR model from {weights_dir}...")
    image_processor = RTDetrImageProcessor.from_pretrained(weights_dir, local_files_only=True)
    config = RTDetrConfig.from_pretrained(weights_dir, local_files_only=True)
    
    # Handle possible custom number of labels
    num_labels = len(config.id2label) if config.id2label else 80
    
    model = RTDetrForObjectDetection.from_pretrained(
        weights_dir,
        config=config,
        ignore_mismatched_sizes=True,
        local_files_only=True
    )
    model.to(device)
    
    print(f"Model loaded successfully on {device}.")
    print(f"Encoder layers: {getattr(config, 'encoder_layers', 'N/A')}")
    print(f"Decoder layers: {getattr(config, 'decoder_layers', 'N/A')}")
    print(f"Hidden dimension (d_model): {getattr(config, 'd_model', 'N/A')}")
    print(f"Number of queries: {getattr(config, 'num_queries', 'N/A')}")
    print(f"Number of labels: {num_labels}")
    
    return model, image_processor, config

def replace_frozen_bn(model):
    """
    Find and replace RTDetrFrozenBatchNorm2d with standard BatchNorm2d.
    
    Args:
        model (nn.Module): RTDetr model.
        
    Returns:
        model (nn.Module): RTDetr model with standard BN layers.
    """
    try:
        from transformers.models.rt_detr.modeling_rt_detr import RTDetrFrozenBatchNorm2d
    except ImportError:
        # Fallback placeholder
        class RTDetrFrozenBatchNorm2d(nn.Module):
            pass
            
    def set_module_by_name(parent, name, child):
        parts = name.split('.')
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], child)
        
    count = 0
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    for name, module in model.named_modules():
        if isinstance(module, RTDetrFrozenBatchNorm2d):
            bn = nn.BatchNorm2d(module.weight.shape[0]).to(device=device, dtype=dtype)
            
            # Copy parameters
            if hasattr(module, 'weight') and module.weight is not None:
                bn.weight.data.copy_(module.weight.data)
            if hasattr(module, 'bias') and module.bias is not None:
                bn.bias.data.copy_(module.bias.data)
            if hasattr(module, 'running_mean') and module.running_mean is not None:
                bn.running_mean.data.copy_(module.running_mean.data)
            if hasattr(module, 'running_var') and module.running_var is not None:
                bn.running_var.data.copy_(module.running_var.data)
                
            set_module_by_name(model, name, bn)
            count += 1
            
    print(f"Replaced {count} FrozenBatchNorm2d layers with BatchNorm2d.")
    return model

def get_ignored_patterns():
    """
    Get patterns to exclude sensitive layers from pruning to prevent shape mismatch.
    
    Returns:
        list of str: List of ignored module name substrings.
    """
    return [
        "model.backbone.model.embedder",  # Backbone stem
        "shortcut",                       # Downsample residual shortcuts
        "model.encoder_input_proj",       # Multi-scale input projections
        "model.decoder_input_proj",       # Decoder input projections
        "model.encoder.aifi",             # AIFI transformer layer
        "model.decoder",                  # Whole decoder module (class/bbox embed, query embeddings)
        "model.enc_score_head",           # Prediction head
        "model.enc_bbox_head",            # Prediction head
        "model.enc_output",               # Encoder output projections
        "model.denoising_class_embed"     # Denoising class embedding
    ]

def get_prunable_layers(model):
    """
    Traverse model named_modules() in deterministic order and return prunable nn.Conv2d layers.
    
    Args:
        model (nn.Module): The RT-DETR model.
        
    Returns:
        list of (str, nn.Module): List of prunable layer names and their instances.
    """
    patterns = get_ignored_patterns()
    prunable = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            ignored = False
            for pattern in patterns:
                if pattern in name:
                    ignored = True
                    break
            if not ignored:
                prunable.append((name, module))
    return prunable

def prune_structured_single_layer(model, layer_index, rate, criterion):
    """
    Prune a single Conv2d layer by its index using torch_pruning.
    """
    import torch_pruning as tp
    from utils.general import make_divisible
    
    # Store and set requires_grad to True for tracing
    grad_states = {name: p.requires_grad for name, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad = True
        
    device = next(model.parameters()).device
    x = torch.randn(1, 3, 640, 640).to(device)
    
    DG = tp.DependencyGraph().build_dependency(model, x)
    
    # Restore requires_grad
    for name, p in model.named_parameters():
        p.requires_grad = grad_states[name]
        
    prunable = get_prunable_layers(model)
    if layer_index < 0 or layer_index >= len(prunable):
        raise ValueError(f"layer_index {layer_index} out of range [0, {len(prunable)-1}]")
        
    name, target_conv = prunable[layer_index]
    out_channels = target_conv.out_channels
    
    n_pruned = make_divisible(rate * out_channels, 2)
    if n_pruned >= out_channels:
        n_pruned = out_channels - 2
    if n_pruned <= 0:
        print(f"Structured Pruning on Layer {layer_index} ({name}) skipped (rate={rate}, pruned=0).")
        return model
        
    # Determine indices to prune
    weights = target_conv.weight.data
    importances = torch.zeros(out_channels, device=device)
    if criterion in [0, 1]:
        importances = torch.norm(weights, p=2, dim=[1, 2, 3])
        if criterion == 1:
            importances = -importances
    elif criterion in [2, 3]:
        importances = torch.norm(weights, p=1, dim=[1, 2, 3])
        if criterion == 3:
            importances = -importances
    elif criterion == 6:
        importances = torch.rand(out_channels, device=device)
    else:
        importances = torch.norm(weights, p=2, dim=[1, 2, 3])
        
    idxs = torch.argsort(importances)[:n_pruned].tolist()
    
    group = DG.get_pruning_group(target_conv, tp.prune_conv_out_channels, idxs)
    if DG.check_pruning_group(group):
        group.prune()
        print(f"Pruned structured Layer {layer_index} ({name}): {n_pruned}/{out_channels} channels. New out_channels={target_conv.out_channels}")
    else:
        print(f"Warning: dependency check failed for pruning {name}.")
        
    return model

def prune_structured_global(model, pruning_params, criterion):
    """
    Prune multiple layers structured using torch_pruning.
    """
    import torch_pruning as tp
    from utils.general import make_divisible
    
    prunable = get_prunable_layers(model)
    if isinstance(pruning_params, float):
        pruning_params = [(i, pruning_params) for i in range(len(prunable))]
        
    pruning_params = sorted(pruning_params, key=lambda x: x[0])
    
    # Store and set requires_grad to True for tracing
    grad_states = {name: p.requires_grad for name, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad = True
        
    device = next(model.parameters()).device
    x = torch.randn(1, 3, 640, 640).to(device)
    
    DG = tp.DependencyGraph().build_dependency(model, x)
    
    # Restore requires_grad
    for name, p in model.named_parameters():
        p.requires_grad = grad_states[name]
        
    for layer_idx, rate in pruning_params:
        if layer_idx < 0 or layer_idx >= len(prunable):
            continue
        name, target_conv = prunable[layer_idx]
        out_channels = target_conv.out_channels
        
        n_pruned = make_divisible(rate * out_channels, 2)
        if n_pruned >= out_channels:
            n_pruned = out_channels - 2
        if n_pruned <= 0:
            continue
            
        # Determine indices to prune
        weights = target_conv.weight.data
        importances = torch.zeros(out_channels, device=device)
        if criterion in [0, 1]:
            importances = torch.norm(weights, p=2, dim=[1, 2, 3])
            if criterion == 1:
                importances = -importances
        elif criterion in [2, 3]:
            importances = torch.norm(weights, p=1, dim=[1, 2, 3])
            if criterion == 3:
                importances = -importances
        elif criterion == 6:
            importances = torch.rand(out_channels, device=device)
        else:
            importances = torch.norm(weights, p=2, dim=[1, 2, 3])
            
        idxs = torch.argsort(importances)[:n_pruned].tolist()
        
        group = DG.get_pruning_group(target_conv, tp.prune_conv_out_channels, idxs)
        if DG.check_pruning_group(group):
            group.prune()
            print(f"Pruned structured Layer {layer_idx} ({name}): {n_pruned}/{out_channels} channels. New out_channels={target_conv.out_channels}")
            
    return model

def prune_unstructured(model, pruning_params, criterion):
    """
    Prune model unstructured using standard PyTorch l1_unstructured or random_unstructured.
    """
    import torch.nn.utils.prune as prune
    
    prunable = get_prunable_layers(model)
    if isinstance(pruning_params, float):
        pruning_params = [(i, pruning_params) for i in range(len(prunable))]
        
    for layer_idx, rate in pruning_params:
        if layer_idx < 0 or layer_idx >= len(prunable):
            continue
        name, target_conv = prunable[layer_idx]
        if rate <= 0:
            continue
            
        if criterion == 6:
            prune.random_unstructured(target_conv, name='weight', amount=rate)
        else:
            prune.l1_unstructured(target_conv, name='weight', amount=rate)
            
        print(f"Pruned unstructured Layer {layer_idx} ({name}) with rate {rate} using criterion {criterion}.")
        
    return model

def load_pruned_model_rtdetr(weights_dir, pruning_params, criterion, map_location=None, structured=True):
    """
    Main entry point for loading and pruning RT-DETR models.
    
    Args:
        weights_dir (str): HF model directory path.
        pruning_params (list or float): Pruning rates specification.
        criterion (int): Importance evaluation criterion index.
        map_location (torch.device): Target load device.
        structured (bool): Perform structured pruning if True, otherwise unstructured.
        
    Returns:
        model (nn.Module): Loaded and pruned model.
    """
    model, image_processor, config = attempt_load_rtdetr(weights_dir, map_location)
    model = replace_frozen_bn(model)
    
    if structured:
        model = prune_structured_global(model, pruning_params, criterion)
        # Verify shape consistency after structured pruning via forward pass
        print("Verifying model forward pass shape consistency...")
        try:
            dummy = torch.randn(1, 3, 640, 640).to(map_location)
            with torch.no_grad():
                _ = model(dummy)
            print("Forward pass verification successful.")
        except Exception as e:
            print(f"WARNING: Model forward pass failed after structured pruning: {e}")
    else:
        model = prune_unstructured(model, pruning_params, criterion)
        
    model.eval()
    return model
