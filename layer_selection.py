import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def select(output, parameters, flops, params_layers, flops_layers, params_map, flops_map, save=False):
    rates = []
    layer = []
    precision = []
    params = []
    flps = []
    params_val = []
    flps_val = []
    conv_layers = 55 if 'tiny' in output else 89

    def clean(line, char):
        a = []
        line = line.replace('np.float64', '').replace('np.float32', '').replace('np.float16', '')
        lis = line.split(',')
        lis.pop()
        for i in range(len(lis)):
            for c in char:
                lis[i] = lis[i].replace(c, '')
            a.append(float(lis[i]))
        return a

    layer_name_map = {}
    for idx, file in enumerate(sorted(os.listdir(output))):
        name = Path(file).stem
        rate = name.split('_')
        if len(rate) != 2 or not rate[1].isdigit() or rate[0] == 'test':
            continue
        
        temp_layer = []
        temp_precision = []
        temp_params = []
        temp_flps = []
        
        with open(os.path.join(output, file), 'r') as f:
            for line in f:
                line_str = line.strip()
                if line_str.startswith('[') or line_str.startswith(']'):
                    continue
                if line_str.endswith(','):
                    line_str = line_str[:-1]
                if len(line_str) > 2:
                    try:
                        cleaned_line = line_str.replace('np.float64', '').replace('np.float32', '').replace('np.float16', '')
                        import ast
                        val = ast.literal_eval(cleaned_line)
                        
                        layer_id = val[0]
                        metrics_tuple = val[1]
                        
                        seq_idx = len(temp_layer)
                        layer_name_map[seq_idx] = layer_id
                        
                        temp_layer.append(seq_idx)
                        temp_precision.append(metrics_tuple[2])
                        temp_params.append(parameters - metrics_tuple[6])
                        temp_flps.append(flops - metrics_tuple[7])
                    except Exception as e:
                        continue
                        
        if len(temp_layer) < 10:
            print(f"Warning: Skipping file {file} because it is empty or incomplete.")
            continue
            
        rates.append(int(rate[1])/100)
        layer.append(temp_layer)
        precision.append(temp_precision)
        params.append(temp_params)
        flps.append(temp_flps)
        params_val.append([])
        flps_val.append([])

    conv_layers = len(layer[0]) if len(layer) > 0 else conv_layers
    for idx, lay in enumerate(layer):
        if len(lay) < conv_layers:
            raise Exception(f'Sensitivity analysis for {rates[idx]} is incomplete')
        elif len(lay) > conv_layers:
            raise Exception(f'Sensitivity analysis for {rates[idx]} has more conv layers than expected')

    # Parameters selection
    for i in range(len(params_val)):
        for idx, param in enumerate(params[i]):
            params_val[i].append(param * (precision[i][idx] ** params_map))

    max_p_param = max([max(params_val[i]) for i in range(len(params_val))])
    transposed_param = [list(i) for i in zip(*params_val)]
    max_val_param = [max(transposed_param[i]) for i in range(len(transposed_param))]
    max_rate_param = [np.argmax(transposed_param[i]) for i in range(len(transposed_param))]
    
    # Robust parameter selection loop
    frac_param = 2.0
    best_number_params = []
    best_diff_p = float('inf')
    visited_fracs_p = set()
    for _ in range(5000):
        frac_rounded = round(frac_param, 4)
        if frac_rounded in visited_fracs_p:
            break
        visited_fracs_p.add(frac_rounded)
        
        current_indices = [idx for idx, val in enumerate(max_val_param) if val > max_p_param / frac_param]
        diff = abs(len(current_indices) - params_layers)
        if diff < best_diff_p:
            best_diff_p = diff
            best_number_params = current_indices
            best_frac_p = frac_param
            
        if len(current_indices) < params_layers - 1:
            frac_param += 0.001
        elif len(current_indices) > params_layers:
            frac_param -= 0.001
        else:
            best_number_params = current_indices
            best_frac_p = frac_param
            break
    number_params = best_number_params
    frac_param = best_frac_p
    rate_params = [max_rate_param[num] for num in number_params]
    params_threshold = max_p_param / frac_param

    # FLOPS selection
    for i in range(len(flps_val)):
        for idx, flp in enumerate(flps[i]):
            flps_val[i].append(flp * (precision[i][idx] ** flops_map))

    max_p_flop = max([max(flps_val[i]) for i in range(len(flps_val))])
    transposed_flop = [list(i) for i in zip(*flps_val)]
    max_val_flop = [max(transposed_flop[i]) for i in range(len(transposed_flop))]
    max_rate_flop = [np.argmax(transposed_flop[i]) for i in range(len(transposed_flop))]
    
    # Robust FLOPS selection loop
    frac_flop = 2.0
    best_number_flps = []
    best_diff_f = float('inf')
    visited_fracs_f = set()
    for _ in range(5000):
        frac_rounded = round(frac_flop, 4)
        if frac_rounded in visited_fracs_f:
            break
        visited_fracs_f.add(frac_rounded)
        
        current_indices = [idx for idx, val in enumerate(max_val_flop) if val > max_p_flop / frac_flop]
        diff = abs(len(current_indices) - flops_layers)
        if diff < best_diff_f:
            best_diff_f = diff
            best_number_flps = current_indices
            best_frac_f = frac_flop
            
        if len(current_indices) < flops_layers - 1:
            frac_flop += 0.001
        elif len(current_indices) > flops_layers:
            frac_flop -= 0.001
        else:
            best_number_flps = current_indices
            best_frac_f = frac_flop
            break
    number_flps = best_number_flps
    frac_flop = best_frac_f
    rate_flps = [max_rate_flop[num] for num in number_flps]
    flops_threshold = max_p_flop / frac_flop

    # selected layers and pruning rates:
    layers = {}
    for num, r in zip(number_flps, rate_flps):
        layers[(num, rates[r])] = 'FLOPS'
    for num, r in zip(number_params, rate_params):
        if (num, rates[r]) in layers:
            layers[(num, rates[r])] = 'both'
        else:
            layers[(num, rates[r])] = 'params'

    # Print selection analysis table to console
    print("\n" + "="*110)
    print(f"{'LAYER SELECTION DECISION ANALYSIS':^110}")
    print("="*110)
    print(f"Params Selection: Cutoff threshold = {params_threshold:,.3e} (Max score = {max_p_param:,.3e})")
    print(f"FLOPS Selection:  Cutoff threshold = {flops_threshold:,.3e} (Max score = {max_p_flop:,.3e})")
    print("-"*110)
    print(f"{'Layer Name':<45} | {'Selected For':^15} | {'Rate':^8} | {'Param Score':^15} | {'FLOPS Score':^15}")
    print("-"*110)
    for (num, rate) in sorted(layers.keys()):
        rate_idx = rates.index(rate)
        p_score = params_val[rate_idx][num]
        f_score = flps_val[rate_idx][num]
        layer_name = layer_name_map.get(num, str(num))
        print(f"{layer_name:<45} | {layers[(num, rate)]:^15} | {rate:^8.2%} | {p_score:^15.3e} | {f_score:^15.3e}")
    print("="*110 + "\n")

    # Saving SA diagrams
    if save:
        folder = os.path.join('graphs', f'SA_{Path(output).name}')
        os.makedirs(folder, exist_ok=True)
        
        plt.rcParams['font.sans-serif'] = 'Arial'
        plt.rcParams['font.family'] = 'sans-serif'
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

        # Plot 1: Parameters sensitivity
        fig1, ax1 = plt.subplots(figsize=(9, 6), dpi=300)
        for idx in range(len(layer)):
            ax1.plot(layer[idx], [p / 1E6 for p in params[idx]], label=f'Rate: {rates[idx]*100:.0f}%', linewidth=2)
        
        # Highlight selected
        selected_x_p = []
        selected_y_p = []
        for idx, val in enumerate(number_params):
            selected_x_p.append(val)
            selected_y_p.append(params[rate_params[idx]][val] / 1E6)
        ax1.scatter(selected_x_p, selected_y_p, color='limegreen', edgecolor='black', s=80, zorder=5, label='Selected (Params)')
        ax1.set_title('Sensitivity Analysis: Pruned Parameters by Layer', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Convolutional Layer Index')
        ax1.set_ylabel('Pruned Parameters (millions)')
        ax1.legend(frameon=True, facecolor='white', framealpha=0.9)
        ax1.grid(True, linestyle='--', alpha=0.6)
        fig1.savefig(os.path.join(folder, 'sa_parameters.pdf'), bbox_inches='tight')
        fig1.savefig(os.path.join(folder, 'sa_parameters.png'), bbox_inches='tight')
        plt.close(fig1)

        # Plot 2: FLOPS sensitivity
        fig2, ax2 = plt.subplots(figsize=(9, 6), dpi=300)
        for idx in range(len(layer)):
            ax2.plot(layer[idx], flps[idx], label=f'Rate: {rates[idx]*100:.0f}%', linewidth=2)
            
        selected_x_f = []
        selected_y_f = []
        for idx, val in enumerate(number_flps):
            selected_x_f.append(val)
            selected_y_f.append(flps[rate_flps[idx]][val])
        ax2.scatter(selected_x_f, selected_y_f, color='royalblue', edgecolor='black', s=80, zorder=5, label='Selected (FLOPS)')
        ax2.set_title('Sensitivity Analysis: Pruned GFLOPS by Layer', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Convolutional Layer Index')
        ax2.set_ylabel('Pruned GFLOPS')
        ax2.legend(frameon=True, facecolor='white', framealpha=0.9)
        ax2.grid(True, linestyle='--', alpha=0.6)
        fig2.savefig(os.path.join(folder, 'sa_flops.pdf'), bbox_inches='tight')
        fig2.savefig(os.path.join(folder, 'sa_flops.png'), bbox_inches='tight')
        plt.close(fig2)

        # Plot 3: mAP degradation
        fig0, ax0 = plt.subplots(figsize=(9, 6), dpi=300)
        for idx in range(len(layer)):
            ax0.plot(layer[idx], precision[idx], label=f'Rate: {rates[idx]*100:.0f}%', linewidth=2)
            
        for val in layers:
            col = 'royalblue' if layers[val] == 'FLOPS' else ('limegreen' if layers[val] == 'params' else 'red')
            rate_idx = rates.index(val[1])
            ax0.scatter(val[0], precision[rate_idx][val[0]], color=col, edgecolor='black', s=80, zorder=5)
        
        # Add dummy plots for clean legend
        ax0.scatter([], [], color='limegreen', edgecolor='black', s=80, label='Selected (Params)')
        ax0.scatter([], [], color='royalblue', edgecolor='black', s=80, label='Selected (FLOPS)')
        ax0.scatter([], [], color='red', edgecolor='black', s=80, label='Selected (Both)')
        
        ax0.set_title('Sensitivity Analysis: mAP Degradation by Layer', fontsize=12, fontweight='bold')
        ax0.set_xlabel('Convolutional Layer Index')
        ax0.set_ylabel('mAP@0.5')
        ax0.legend(frameon=True, facecolor='white', framealpha=0.9)
        ax0.grid(True, linestyle='--', alpha=0.6)
        fig0.savefig(os.path.join(folder, 'sa_map.pdf'), bbox_inches='tight')
        fig0.savefig(os.path.join(folder, 'sa_map.png'), bbox_inches='tight')
        plt.close(fig0)

        # Plot 4: NEW - Selection Scores Curve (Show why layers were selected)
        fig3, (ax_sp, ax_sf) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
        
        # Parameter scores
        for idx in range(len(layer)):
            ax_sp.plot(layer[idx], params_val[idx], label=f'Rate: {rates[idx]*100:.0f}%', linewidth=2)
        ax_sp.axhline(y=params_threshold, color='red', linestyle='--', linewidth=1.5, label='Cutoff Threshold')
        
        # Highlight selected parameter layers
        sel_score_x = []
        sel_score_y = []
        for idx, val in enumerate(number_params):
            sel_score_x.append(val)
            sel_score_y.append(params_val[rate_params[idx]][val])
        ax_sp.scatter(sel_score_x, sel_score_y, color='limegreen', edgecolor='black', s=80, zorder=5, label='Selected')
        ax_sp.set_title('Parameter Selection Score: ParamsSaved * mAP^20', fontsize=11, fontweight='bold')
        ax_sp.set_xlabel('Convolutional Layer Index')
        ax_sp.set_ylabel('Score')
        ax_sp.legend(frameon=True, facecolor='white', framealpha=0.9)
        ax_sp.grid(True, linestyle='--', alpha=0.6)
        
        # FLOPS scores
        for idx in range(len(layer)):
            ax_sf.plot(layer[idx], flps_val[idx], label=f'Rate: {rates[idx]*100:.0f}%', linewidth=2)
        ax_sf.axhline(y=flops_threshold, color='red', linestyle='--', linewidth=1.5, label='Cutoff Threshold')
        
        # Highlight selected FLOPS layers
        sel_f_x = []
        sel_f_y = []
        for idx, val in enumerate(number_flps):
            sel_f_x.append(val)
            sel_f_y.append(flps_val[rate_flps[idx]][val])
        ax_sf.scatter(sel_f_x, sel_f_y, color='royalblue', edgecolor='black', s=80, zorder=5, label='Selected')
        ax_sf.set_title('FLOPS Selection Score: FlopsSaved * mAP^20', fontsize=11, fontweight='bold')
        ax_sf.set_xlabel('Convolutional Layer Index')
        ax_sf.set_ylabel('Score')
        ax_sf.legend(frameon=True, facecolor='white', framealpha=0.9)
        ax_sf.grid(True, linestyle='--', alpha=0.6)

        plt.suptitle('Layer Selection Abstraction (Selection Scores vs. Thresholds)', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        fig3.savefig(os.path.join(folder, 'sa_selection_scores.pdf'), bbox_inches='tight')
        fig3.savefig(os.path.join(folder, 'sa_selection_scores.png'), bbox_inches='tight')
        plt.close(fig3)

        print(f'Sensitivity analysis graphs (including scores) are saved in {folder}')

    final_params = []
    for (num, rate) in sorted(layers.keys()):
        final_params.append((layer_name_map.get(num, num), rate))
    return str(final_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='layer_selection.py')
    parser.add_argument('--output', type=str, default='output/yolov7_training',
                        help='directory containing text files which are outputs of sensitivity analysis')
    parser.add_argument('--params', type=int, default=36907898, help='number of parameters')
    parser.add_argument('--flops', type=float, default=104.514, help='number of FLOPS')
    parser.add_argument('--params-layers', type=int, default=6, help='number of layers to be pruned for parameters')
    parser.add_argument('--flops-layers', type=int, default=5, help='number of layers to be pruned for FLOPS')
    parser.add_argument('--params-map', type=float, default=20, help='impact of map on parameter selection')
    parser.add_argument('--flops-map', type=float, default=20, help='impact of map on FLOPS selection')
    parser.add_argument('--save', action='store_true', help='saving sensitivity analysis graphs')
    opt = parser.parse_args()

    layers = \
        select(opt.output, opt.params, opt.flops, opt.params_layers, opt.flops_layers, opt.params_map,
               opt.flops_map, opt.save)
    print("\n>>> SELECTED PRUNING PARAMETERS:")
    print(layers)