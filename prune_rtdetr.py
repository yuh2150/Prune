import os
import torch
import torch.nn as nn
import copy
import random
import math
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
    try:
        image_processor = RTDetrImageProcessor.from_pretrained(weights_dir, local_files_only=True)
        config = RTDetrConfig.from_pretrained(weights_dir, local_files_only=True)
        local = True
    except Exception:
        print(f"Local files not found for {weights_dir}. Attempting to download from Hugging Face Hub...")
        image_processor = RTDetrImageProcessor.from_pretrained(weights_dir, local_files_only=False)
        config = RTDetrConfig.from_pretrained(weights_dir, local_files_only=False)
        local = False
    
    # Handle possible custom number of labels
    num_labels = len(config.id2label) if config.id2label else 80
    
    model = RTDetrForObjectDetection.from_pretrained(
        weights_dir,
        config=config,
        ignore_mismatched_sizes=True,
        local_files_only=local
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

def prune_layers_rtdetr(model, prune_ratio=0.2):
    """
    Prune RT-DETR by removing transformer layers only.
    Correctly handles re-indexing of auxiliary prediction heads.
    """
    print(f"\n--- Starting Simplified Layer Pruning (Ratio: {prune_ratio:.2f}) ---")
    # Create a deep copy to avoid modifying the original model instance
    pruned_model = copy.deepcopy(model)
    pruned_model.eval() # Ensure model is in eval mode after copy

    # --- Structure Check for Hugging Face Model ---
    # Check for the nested 'model' attribute which contains the core components
    if not hasattr(pruned_model, 'model') or not isinstance(pruned_model.model, nn.Module):
         print("Error: Top-level model does not have a 'model' attribute. Cannot find transformer components.")
         return model # Return original if structure is wrong

    # Access the core RTDetrModel components
    nested_model = pruned_model.model
    if not hasattr(nested_model, 'encoder') or not isinstance(nested_model.encoder, nn.Module):
        print("Error: Nested 'model' attribute does not have an 'encoder' module.")
        return model
    if not hasattr(nested_model, 'decoder') or not isinstance(nested_model.decoder, nn.Module):
        print("Error: Nested 'model' attribute does not have a 'decoder' module.")
        return model

    # Get encoder and decoder modules
    encoder = nested_model.encoder
    decoder = nested_model.decoder

    # Get original layer counts safely
    original_enc_layers = len(encoder.layers) if hasattr(encoder, 'layers') and isinstance(encoder.layers, nn.ModuleList) else 0
    original_dec_layers = len(decoder.layers) if hasattr(decoder, 'layers') and isinstance(decoder.layers, nn.ModuleList) else 0

    print(f"Original encoder layers: {original_enc_layers}")
    print(f"Original decoder layers: {original_dec_layers}")

    # --- Calculate Layers to Prune ---
    if original_enc_layers == 0 and original_dec_layers == 0:
        print("Model has no encoder or decoder layers. Skipping pruning.")
        if not hasattr(pruned_model, 'num_encoder_layers'): pruned_model.num_encoder_layers = 0
        if not hasattr(pruned_model, 'num_decoder_layers'): pruned_model.num_decoder_layers = 0
        return pruned_model

    enc_layers_to_prune = math.ceil(original_enc_layers * prune_ratio) if original_enc_layers > 0 else 0
    dec_layers_to_prune = math.ceil(original_dec_layers * prune_ratio) if original_dec_layers > 0 else 0

    # Ensure we don't prune all layers if layers exist and pruning is attempted
    enc_layers_to_prune = min(enc_layers_to_prune, original_enc_layers - 1) if original_enc_layers > 1 else 0
    dec_layers_to_prune = min(dec_layers_to_prune, original_dec_layers - 1) if original_dec_layers > 1 else 0

    print(f"Targeting removal of {enc_layers_to_prune} encoder layers.")
    print(f"Targeting removal of {dec_layers_to_prune} decoder layers.")

    # --- Encoder Pruning ---
    new_encoder_layers = []
    if original_enc_layers > 0:
        new_encoder_layers = encoder.layers # Default to existing list
        if enc_layers_to_prune > 0:
            # Randomly sample indices to KEEP
            enc_indices_to_keep = sorted(random.sample(range(original_enc_layers),
                                                    original_enc_layers - enc_layers_to_prune))
            print(f"Keeping encoder layers at original indices: {enc_indices_to_keep}")
            # Create a new ModuleList with only the kept layers
            new_encoder_layers = nn.ModuleList([encoder.layers[i] for i in enc_indices_to_keep])
            encoder.layers = new_encoder_layers # Modify encoder within nested_model
    else:
        print("Skipping encoder pruning (0 layers or 0 prune count).")

    # Store actual final encoder layer count on the TOP-LEVEL model object
    pruned_model.num_encoder_layers = len(new_encoder_layers)

    # --- Decoder Pruning ---
    new_decoder_layers = []
    dec_indices_to_keep = list(range(original_dec_layers)) # Default to all indices before pruning
    if original_dec_layers > 0:
        new_decoder_layers = decoder.layers # Default to existing list
        if dec_layers_to_prune > 0:
            # Randomly sample indices to KEEP
            dec_indices_to_keep = sorted(random.sample(range(original_dec_layers),
                                                    original_dec_layers - dec_layers_to_prune))
            print(f"Keeping decoder layers at original indices: {dec_indices_to_keep}")
            # Create a new ModuleList with only the kept layers
            new_decoder_layers = nn.ModuleList([decoder.layers[i] for i in dec_indices_to_keep])
            decoder.layers = new_decoder_layers # Modify decoder within nested_model
    else:
        print("Skipping decoder pruning (0 layers or 0 prune count).")

    # Update decoder's internal layer count attribute if it exists
    if hasattr(decoder, 'num_layers'):
        decoder.num_layers = len(new_decoder_layers)
    # Store actual final decoder layer count on the TOP-LEVEL model object
    pruned_model.num_decoder_layers = len(new_decoder_layers)

    # --- Prediction Head Pruning and Re-indexing ---
    # Identify whether prediction heads are on the top-level model (Deformable DETR) or on the decoder (RT-DETR)
    target_for_heads = pruned_model
    if not hasattr(target_for_heads, 'class_embed') and hasattr(decoder, 'class_embed'):
        target_for_heads = decoder

    has_class_embed = hasattr(target_for_heads, 'class_embed') and target_for_heads.class_embed is not None
    has_bbox_embed = hasattr(target_for_heads, 'bbox_embed') and target_for_heads.bbox_embed is not None

    is_multi_head_class = has_class_embed and isinstance(target_for_heads.class_embed, nn.ModuleList) and len(target_for_heads.class_embed) > 1
    is_multi_head_bbox = has_bbox_embed and isinstance(target_for_heads.bbox_embed, nn.ModuleList) and len(target_for_heads.bbox_embed) > 1

    if is_multi_head_class or is_multi_head_bbox:
        print("\nRebuilding prediction heads ModuleList for pruned decoder...")

        if is_multi_head_class:
            original_class_embed_list = target_for_heads.class_embed
            num_original_heads = len(original_class_embed_list)
            print(f"  Processing {num_original_heads} original class prediction heads.")
            
            new_class_embed_list = nn.ModuleList()
            for i, original_idx in enumerate(dec_indices_to_keep):
                if i == len(dec_indices_to_keep) - 1:
                    # The last layer in the pruned decoder gets the original final head
                    final_head_idx = num_original_heads - 1
                    new_class_embed_list.append(original_class_embed_list[final_head_idx])
                    print(f"  Mapping pruned decoder layer {i} (original {original_idx}) to final prediction head {final_head_idx}.")
                else:
                    # Intermediate layers get their corresponding original heads
                    new_class_embed_list.append(original_class_embed_list[original_idx])
                    print(f"  Mapping pruned decoder layer {i} (original {original_idx}) to prediction head {original_idx}.")

            target_for_heads.class_embed = new_class_embed_list
            print(f"  Finished class heads. New count: {len(new_class_embed_list)}")

        elif has_class_embed:
             print("  Original class embed is single layer. Keeping it as is.")

        if is_multi_head_bbox:
            original_bbox_embed_list = target_for_heads.bbox_embed
            num_original_heads = len(original_bbox_embed_list)
            print(f"  Processing {num_original_heads} original bbox prediction heads.")
            
            new_bbox_embed_list = nn.ModuleList()
            for i, original_idx in enumerate(dec_indices_to_keep):
                if i == len(dec_indices_to_keep) - 1:
                    # The last layer in the pruned decoder gets the original final head
                    final_head_idx = num_original_heads - 1
                    new_bbox_embed_list.append(original_bbox_embed_list[final_head_idx])
                    print(f"  Mapping pruned decoder layer {i} (original {original_idx}) to final prediction head {final_head_idx}.")
                else:
                    # Intermediate layers get their corresponding original heads
                    new_bbox_embed_list.append(original_bbox_embed_list[original_idx])
                    print(f"  Mapping pruned decoder layer {i} (original {original_idx}) to prediction head {original_idx}.")

            target_for_heads.bbox_embed = new_bbox_embed_list
            print(f"  Finished bbox heads. New count: {len(new_bbox_embed_list)}")

        elif has_bbox_embed:
             print("  Original bbox embed is single layer/module. Keeping it as is.")

    elif has_class_embed:
         print("\nModel appears to have only a single final prediction head. No head pruning/re-indexing needed.")
    else:
         print("\nWarning: Model does not appear to have 'class_embed'. Skipping head processing.")

    print("--- Pruning Finished ---")
    print(f"Final layer counts: Encoder={pruned_model.num_encoder_layers}, Decoder={pruned_model.num_decoder_layers}")
    if hasattr(target_for_heads, 'class_embed') and target_for_heads.class_embed is not None:
        is_list = isinstance(target_for_heads.class_embed, nn.ModuleList)
        head_len = len(target_for_heads.class_embed) if is_list else 1
        print(f"Final class_embed: {'ModuleList' if is_list else 'Single Module'}, Length/Count: {head_len}")
    if hasattr(target_for_heads, 'bbox_embed') and target_for_heads.bbox_embed is not None:
        is_list = isinstance(target_for_heads.bbox_embed, nn.ModuleList)
        head_len = len(target_for_heads.bbox_embed) if is_list else 1
        print(f"Final bbox_embed: {'ModuleList' if is_list else 'Single Module'}, Length/Count: {head_len}")

    # Ensure the model configuration reflects the pruned layer counts (important for saving/reloading)
    if hasattr(pruned_model, 'config'):
        print("Updating model config with new layer counts...")
        pruned_model.config.encoder_layers = pruned_model.num_encoder_layers
        pruned_model.config.decoder_layers = pruned_model.num_decoder_layers
    else:
        print("Warning: Pruned model does not have a 'config' attribute to update layer counts.")

    return pruned_model

def load_pruned_model_rtdetr(weights_dir, pruning_params, criterion, map_location=None, modification='prune-structured', structured=None):
    """
    Main entry point for loading and pruning RT-DETR models.
    
    Args:
        weights_dir (str): HF model directory path.
        pruning_params (list or float): Pruning rates specification.
        criterion (int): Importance evaluation criterion index.
        map_location (torch.device): Target load device.
        modification (str): 'prune-structured' | 'prune-unstructured' | 'prune-layer'.
        structured (bool, optional): Legacy parameter to override modification.
        
    Returns:
        model (nn.Module): Loaded and pruned model.
    """
    if structured is not None:
        modification = 'prune-structured' if structured else 'prune-unstructured'
        
    model, image_processor, config = attempt_load_rtdetr(weights_dir, map_location)
    model = replace_frozen_bn(model)
    
    if modification == 'prune-layer':
        prune_ratio = 0.2  # default fallback
        if isinstance(pruning_params, float):
            prune_ratio = pruning_params
        elif isinstance(pruning_params, list) and len(pruning_params) > 0:
            if isinstance(pruning_params[0], (int, float)):
                prune_ratio = float(pruning_params[0])
            elif isinstance(pruning_params[0], (list, tuple)) and len(pruning_params[0]) > 1:
                prune_ratio = float(pruning_params[0][1])
        print(f"Applying layer pruning with ratio: {prune_ratio}")
        model = prune_layers_rtdetr(model, prune_ratio)
    elif modification == 'prune-unstructured':
        model = prune_unstructured(model, pruning_params, criterion)
    else:
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
        
    model.eval()
    return model

if __name__ == '__main__':
    import argparse
    import ast
    
    parser = argparse.ArgumentParser(prog='prune_rtdetr.py')
    parser.add_argument('--weights', type=str, default='PekingU/rtdetr_r18vd', help='HF model directory path')
    parser.add_argument('--output-path', type=str, default='rtdetr-pruned.pt', help='output path for pruned model')
    parser.add_argument('--pruning-params', type=str, default='', help='Pruning parameters as string')
    parser.add_argument('--criterion', type=int, default=0, help='Pruning criterion (0=L2, etc.)')
    parser.add_argument('--modification', type=str, default='prune-structured', help='prune-structured | prune-unstructured | prune-layer')
    
    opt = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    pruning_params_parsed = []
    if len(opt.pruning_params) > 0:
        param_str = opt.pruning_params.strip()
        open_brackets = param_str.count('[')
        close_brackets = param_str.count(']')
        if open_brackets > close_brackets:
            param_str += ']' * (open_brackets - close_brackets)
        elif close_brackets > open_brackets:
            param_str = '[' * (close_brackets - open_brackets) + param_str
            
        try:
            parsed = ast.literal_eval(param_str)
            if isinstance(parsed, list):
                if len(parsed) > 0 and isinstance(parsed[0], list):
                    if len(parsed[0]) > 0 and isinstance(parsed[0][0], (tuple, list)):
                        parsed = parsed[0]
            pruning_params_parsed = parsed
        except Exception as e:
            print(f"Error parsing pruning params: {e}")
            raise e
            
    print(f"Pruning model with parameters: {pruning_params_parsed}")
    model = load_pruned_model_rtdetr(
        weights_dir=opt.weights,
        pruning_params=pruning_params_parsed,
        criterion=opt.criterion,
        map_location=device,
        modification=opt.modification
    )
    
    # Save the model
    if opt.output_path.endswith('.pt'):
        # Save PyTorch checkpoint containing the state dict and architecture/config
        ckpt = {
            'model': model.state_dict(),
            'config': model.config if hasattr(model, 'config') else None,
            'model_object': model
        }
        torch.save(ckpt, opt.output_path)
        print(f"Pruned model checkpoint saved to {opt.output_path}")
    else:
        # Save as Hugging Face folder
        model.save_pretrained(opt.output_path)
        try:
            from transformers import RTDetrImageProcessor
            image_processor = RTDetrImageProcessor.from_pretrained(opt.weights, local_files_only=True)
            image_processor.save_pretrained(opt.output_path)
        except Exception as e:
            print(f"Warning: Could not save image processor: {e}")
        print(f"Pruned Hugging Face model directory saved to {opt.output_path}")

