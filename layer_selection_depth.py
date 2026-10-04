import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

def select_depth_layers(input_file, base_params, base_flops, params_layers, flops_layers, params_map, flops_map, save=False):
    with open(input_file, 'r') as f:
        # Sử dụng eval để parse dữ liệu Python từ file, np cần được import để parse np.float64
        data = eval(f.read())
    
    layers = []
    maps = []
    params_saved = []
    flops_saved = []
    
    for item in data:
        layer_idx = item[0]
        metrics = item[1]
        
        mAP_05 = metrics[2]  # index 2 là mAP@0.5
        pruned_params = metrics[-2]
        pruned_flops = metrics[-1]
        
        layers.append(layer_idx)
        maps.append(mAP_05)
        # Tính số param/flops tiết kiệm được
        params_saved.append(base_params - pruned_params)
        flops_saved.append(base_flops - pruned_flops)
        
    # Tính điểm số (score) theo công thức của bài báo (tiết kiệm * mAP^alpha)
    score_params = [p_saved * (m ** params_map) for p_saved, m in zip(params_saved, maps)]
    score_flops = [f_saved * (m ** flops_map) for f_saved, m in zip(flops_saved, maps)]
    
    # Chọn top K layers có điểm cao nhất
    top_params_idx = np.argsort(score_params)[-params_layers:] if params_layers > 0 else []
    top_flops_idx = np.argsort(score_flops)[-flops_layers:] if flops_layers > 0 else []
    
    selected_params = [layers[i] for i in top_params_idx]
    selected_flops = [layers[i] for i in top_flops_idx]
    
    # Gộp kết quả
    selection_result = {}
    for l in selected_flops:
        selection_result[l] = 'FLOPS'
    for l in selected_params:
        if l in selection_result:
            selection_result[l] = 'both'
        else:
            selection_result[l] = 'params'
            
    print("\n" + "="*85)
    print(f"{'DEPTH LAYER SELECTION DECISION ANALYSIS':^85}")
    print("="*85)
    print(f"{'Layer':^8} | {'Selected For':^15} | {'mAP@0.5':^10} | {'Param Score':^15} | {'FLOPS Score':^15}")
    print("-"*85)
    for i, l in enumerate(layers):
        status = selection_result.get(l, '-')
        print(f"{l:^8d} | {status:^15} | {maps[i]:^10.4f} | {score_params[i]:^15.3e} | {score_flops[i]:^15.3e}")
    print("="*85 + "\n")
    
    # Đầu ra cho --modification prune-layer: [(layer_idx, remove_num), ...]
    selected_list = [(l, 1) for l in selection_result.keys()]
    print(f"Final Selection for prune.py (Layer, Remove_Num): {selected_list}")

    if save:
        folder = os.path.join('graphs', 'SA_Depth')
        os.makedirs(folder, exist_ok=True)
        
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        
        ax[0].bar([str(l) for l in layers], score_params, color='limegreen')
        ax[0].set_title('Param Score (ParamsSaved * mAP^N)')
        ax[0].set_xlabel('Layer')
        ax[0].set_ylabel('Score')
        ax[0].grid(axis='y', linestyle='--', alpha=0.7)
        
        ax[1].bar([str(l) for l in layers], score_flops, color='royalblue')
        ax[1].set_title('FLOPS Score (FlopsSaved * mAP^N)')
        ax[1].set_xlabel('Layer')
        ax[1].set_ylabel('Score')
        ax[1].grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.suptitle('Depth Pruning Sensitivity Scores', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(folder, 'depth_scores.png')
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Graphs saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to sensitivity analysis file')
    parser.add_argument('--params', type=int, default=7227000, help='Base params of model')
    parser.add_argument('--flops', type=float, default=16.5, help='Base FLOPS of model')
    parser.add_argument('--params-layers', type=int, default=2, help='Number of layers to prune for params')
    parser.add_argument('--flops-layers', type=int, default=2, help='Number of layers to prune for FLOPS')
    parser.add_argument('--params-map', type=float, default=20, help='mAP power for param selection')
    parser.add_argument('--flops-map', type=float, default=20, help='mAP power for FLOPS selection')
    parser.add_argument('--save', action='store_true', help='Save graphs')
    opt = parser.parse_args()
    
    select_depth_layers(opt.input, opt.params, opt.flops, opt.params_layers, opt.flops_layers, opt.params_map, opt.flops_map, opt.save)
