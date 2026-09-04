import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def load_data():
    # Load unstructured data
    mag_acc = pd.read_csv('benchmarks/Benchmark(Magnitude_Accuracy).csv')
    mag_hw = pd.read_csv('benchmarks/Benchmark(Magnitude_Hardware).csv')
    
    # Load structured data
    str_acc = pd.read_csv('benchmarks/Benchmark(Structure Accuracy).csv')
    str_hw = pd.read_csv('benchmarks/Benchmark(Structure Hardware).csv')
    
    # Clean up column names (strip whitespace)
    for df in [mag_acc, mag_hw, str_acc, str_hw]:
        df.columns = df.columns.str.strip()
        
    return mag_acc, mag_hw, str_acc, str_hw

def preprocess_data(mag_acc, mag_hw, str_acc, str_hw):
    # Process mAP to percentage if it's decimal (<= 1.0)
    for df in [mag_acc, str_acc]:
        for col in ['mAP@0.5', 'mAP@0.5:0.95']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: x * 100 if x <= 1.0 else x)
                
    # ----------------------------------------------------
    # Merge Magnitude (Unstructured) Data
    # ----------------------------------------------------
    mag_acc = mag_acc.rename(columns={'Model Name': 'model_name'})
    mag_hw = mag_hw.rename(columns={'Model Name': 'model_name'})
    unstructured_df = pd.merge(mag_acc, mag_hw, on='model_name')
    
    # ----------------------------------------------------
    # Merge Structure Data
    # ----------------------------------------------------
    str_acc = str_acc.rename(columns={'Models': 'model_name'})
    str_hw = str_hw.rename(columns={'Models': 'model_name'})
    
    # Clean model names in structured data to match properly
    str_acc['model_name'] = str_acc['model_name'].str.strip()
    str_hw['model_name'] = str_hw['model_name'].str.strip()
    
    # Exclude empty rows in str_hw
    str_hw = str_hw[str_hw['model_name'].notna() & (str_hw['model_name'] != '')]
    
    structured_df = pd.merge(str_acc, str_hw, on='model_name')
    
    return unstructured_df, structured_df

def setup_plot_style():
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['font.family'] = 'sans-serif'
    style_to_use = 'default'
    for style in ['seaborn-v0_8-whitegrid', 'seaborn-whitegrid', 'ggplot']:
        if style in plt.style.available:
            style_to_use = style
            break
    plt.style.use(style_to_use)

def generate_comparison_dashboard(unstructured, structured, colors):
    color_unstructured, color_structured_filter, color_structured_layer, color_baseline = colors
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    fig.suptitle("Pruning Strategy Comparison: Structured vs Unstructured", fontsize=18, fontweight='bold', y=0.98)
    
    # Extract YOLOv5s Unstructured & Structured
    yolo_unstructured = unstructured[unstructured['model_name'].str.contains('YOLOv5s|Baseline YOLOv5s', case=False, na=False)].copy()
    yolo_unstructured['sort_key'] = yolo_unstructured['model_name'].apply(
        lambda x: 0 if 'Baseline' in x else int(x.split('Pruned')[-1].replace('%', '').strip())
    )
    yolo_unstructured = yolo_unstructured.sort_values('sort_key')
    yolo_structured = structured[structured['model_name'].str.contains('YOLOV5s', case=False, na=False)].copy()
    
    # Extract RT-DETR Unstructured & Structured
    rtdetr_unstructured = unstructured[unstructured['model_name'].str.contains('RT-DETR', case=False, na=False)].copy()
    rtdetr_unstructured['sort_key'] = rtdetr_unstructured['model_name'].apply(
        lambda x: 0 if 'Baseline' in x else int(x.split('Pruned')[-1].replace('%', '').strip())
    )
    rtdetr_unstructured = rtdetr_unstructured.sort_values('sort_key')
    rtdetr_structured = structured[structured['model_name'].str.contains('RT-DETR', case=False, na=False)].copy()
    
    # -------------------------------------------------------------------------
    # Plot 1: YOLOv5s Accuracy vs. Parameters
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.set_title("YOLOv5s: Accuracy vs. Parameters (M)", fontsize=13, fontweight='bold', pad=10)
    
    yolo_unstruct_params_m = yolo_unstructured['Parameters'] / 1E6
    ax1.plot(yolo_unstruct_params_m, yolo_unstructured['mAP@0.5'], 'o--', color=color_unstructured, linewidth=2.5, markersize=8, label='Unstructured Pruning')
    
    for idx, row in yolo_unstructured.iterrows():
        p_m = row['Parameters'] / 1E6
        label = 'Baseline' if 'Baseline' in row['model_name'] else row['model_name'].split('Pruned')[-1].strip()
        ax1.annotate(label, (p_m, row['mAP@0.5']), textcoords="offset points", xytext=(-10, 8), ha='center', fontsize=9, fontweight='bold', color=color_unstructured)
        
    for idx, row in yolo_structured.iterrows():
        p_m = row['Parameters'] / 1E6
        m_name = row['model_name']
        if 'Baseline' in m_name:
            ax1.scatter(p_m, row['mAP@0.5'], marker='*', s=250, color=color_baseline, zorder=5, label='Baseline')
        else:
            if 'Filter' in m_name:
                color = color_structured_filter
                marker = '^' if 'Fine-tuned' in m_name else 'v'
                lbl = 'Structured Filter (Fine-tuned)' if 'Fine-tuned' in m_name else 'Structured Filter'
            else:
                color = color_structured_layer
                marker = 's' if 'Fine-tuned' in m_name else 'd'
                lbl = 'Structured Layer (Fine-tuned)' if 'Fine-tuned' in m_name else 'Structured Layer'
            
            ax1.scatter(p_m, row['mAP@0.5'], marker=marker, s=150, color=color, zorder=4, label=lbl)
            short_lbl = 'Filter FT' if 'Filter Pruned Fine-tuned' in m_name else ('Filter' if 'Filter Pruned' in m_name else ('Layer FT' if 'Layer Pruned Fine-tuned' in m_name else 'Layer'))
            ax1.annotate(short_lbl, (p_m, row['mAP@0.5']), textcoords="offset points", xytext=(12, -4), ha='left', fontsize=9, fontweight='bold', color=color)
            
    ax1.set_xlabel("Parameters (Millions)", fontsize=11)
    ax1.set_ylabel("mAP@0.5 (%)", fontsize=11)
    ax1.set_ylim(-5, 75)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # -------------------------------------------------------------------------
    # Plot 2: YOLOv5s Accuracy vs. FPS
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.set_title("YOLOv5s: Accuracy vs. Speed (FPS)", fontsize=13, fontweight='bold', pad=10)
    ax2.plot(yolo_unstructured['FPS'], yolo_unstructured['mAP@0.5'], 'o--', color=color_unstructured, linewidth=2.5, markersize=8, label='Unstructured Pruning')
    
    for idx, row in yolo_unstructured.iterrows():
        label = 'Baseline' if 'Baseline' in row['model_name'] else row['model_name'].split('Pruned')[-1].strip()
        ax2.annotate(label, (row['FPS'], row['mAP@0.5']), textcoords="offset points", xytext=(-10, 8), ha='center', fontsize=9, fontweight='bold', color=color_unstructured)
        
    for idx, row in yolo_structured.iterrows():
        fps = row['FPS']
        m_name = row['model_name']
        if 'Baseline' in m_name:
            ax2.scatter(fps, row['mAP@0.5'], marker='*', s=250, color=color_baseline, zorder=5, label='Baseline')
        else:
            if 'Filter' in m_name:
                color = color_structured_filter
                marker = '^' if 'Fine-tuned' in m_name else 'v'
                lbl = 'Structured Filter (Fine-tuned)' if 'Fine-tuned' in m_name else 'Structured Filter'
            else:
                color = color_structured_layer
                marker = 's' if 'Fine-tuned' in m_name else 'd'
                lbl = 'Structured Layer (Fine-tuned)' if 'Fine-tuned' in m_name else 'Structured Layer'
                
            ax2.scatter(fps, row['mAP@0.5'], marker=marker, s=150, color=color, zorder=4, label=lbl)
            short_lbl = 'Filter FT' if 'Filter Pruned Fine-tuned' in m_name else ('Filter' if 'Filter Pruned' in m_name else ('Layer FT' if 'Layer Pruned Fine-tuned' in m_name else 'Layer'))
            ax2.annotate(short_lbl, (fps, row['mAP@0.5']), textcoords="offset points", xytext=(12, -4), ha='left', fontsize=9, fontweight='bold', color=color)
            
    ax2.set_xlabel("Frame Rate (FPS)", fontsize=11)
    ax2.set_ylabel("mAP@0.5 (%)", fontsize=11)
    ax2.set_ylim(-5, 75)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # -------------------------------------------------------------------------
    # Plot 3: RT-DETR Accuracy vs. Parameters
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.set_title("RT-DETR-R18: Accuracy vs. Parameters (M)", fontsize=13, fontweight='bold', pad=10)
    
    rtdetr_unstruct_params_m = rtdetr_unstructured['Parameters'] / 1E6
    ax3.plot(rtdetr_unstruct_params_m, rtdetr_unstructured['mAP@0.5'], 'o--', color=color_unstructured, linewidth=2.5, markersize=8, label='Unstructured Pruning')
    
    for idx, row in rtdetr_unstructured.iterrows():
        p_m = row['Parameters'] / 1E6
        label = 'Baseline' if 'Baseline' in row['model_name'] else row['model_name'].split('Pruned')[-1].strip()
        ax3.annotate(label, (p_m, row['mAP@0.5']), textcoords="offset points", xytext=(-10, 8), ha='center', fontsize=9, fontweight='bold', color=color_unstructured)
        
    for idx, row in rtdetr_structured.iterrows():
        p_m = row['Parameters'] / 1E6
        m_name = row['model_name']
        if 'Baseline' in m_name:
            ax3.scatter(p_m, row['mAP@0.5'], marker='*', s=250, color=color_baseline, zorder=5, label='Baseline')
        else:
            ax3.scatter(p_m, row['mAP@0.5'], marker='^', s=150, color=color_structured_filter, zorder=4, label='Structured Pruned')
            ax3.annotate('Structured Pruned', (p_m, row['mAP@0.5']), textcoords="offset points", xytext=(12, -4), ha='left', fontsize=9, fontweight='bold', color=color_structured_filter)
            
    ax3.set_xlabel("Parameters (Millions)", fontsize=11)
    ax3.set_ylabel("mAP@0.5 (%)", fontsize=11)
    ax3.set_ylim(-5, 80)
    ax3.grid(True, linestyle='--', alpha=0.6)
    
    # -------------------------------------------------------------------------
    # Plot 4: RT-DETR Accuracy vs. FPS
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.set_title("RT-DETR-R18: Accuracy vs. Speed (FPS)", fontsize=13, fontweight='bold', pad=10)
    ax4.plot(rtdetr_unstructured['FPS'], rtdetr_unstructured['mAP@0.5'], 'o--', color=color_unstructured, linewidth=2.5, markersize=8, label='Unstructured Pruning')
    
    for idx, row in rtdetr_unstructured.iterrows():
        label = 'Baseline' if 'Baseline' in row['model_name'] else row['model_name'].split('Pruned')[-1].strip()
        ax4.annotate(label, (row['FPS'], row['mAP@0.5']), textcoords="offset points", xytext=(-10, 8), ha='center', fontsize=9, fontweight='bold', color=color_unstructured)
        
    for idx, row in rtdetr_structured.iterrows():
        fps = row['FPS']
        m_name = row['model_name']
        if 'Baseline' in m_name:
            ax4.scatter(fps, row['mAP@0.5'], marker='*', s=250, color=color_baseline, zorder=5, label='Baseline')
        else:
            ax4.scatter(fps, row['mAP@0.5'], marker='^', s=150, color=color_structured_filter, zorder=4, label='Structured Pruned')
            ax4.annotate('Structured Pruned', (fps, row['mAP@0.5']), textcoords="offset points", xytext=(12, -4), ha='left', fontsize=9, fontweight='bold', color=color_structured_filter)
            
    ax4.set_xlabel("Frame Rate (FPS)", fontsize=11)
    ax4.set_ylabel("mAP@0.5 (%)", fontsize=11)
    ax4.set_ylim(-5, 80)
    ax4.grid(True, linestyle='--', alpha=0.6)
    
    # Legends
    for ax in [ax1, ax2, ax3, ax4]:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='lower left', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    output_path = 'benchmarks/structured_vs_unstructured_comparison.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Generated comparison dashboard at: {output_path}")
    plt.close()

def generate_efficiency_comparison(unstructured, structured, colors):
    color_unstructured, color_structured_filter, color_structured_layer, color_baseline = colors
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    fig.suptitle("Pruning Efficiency: mAP@0.5 Retention vs. Parameter Reduction (%)", fontsize=16, fontweight='bold', y=0.98)
    
    # YOLOv5s Base Parameters
    yolo_base_params = 7225885
    
    # Process YOLOv5s unstructured
    yolo_unstruct = unstructured[unstructured['model_name'].str.contains('YOLOv5s|Baseline YOLOv5s', case=False, na=False)].copy()
    yolo_unstruct['reduction_pct'] = yolo_unstruct['model_name'].apply(
        lambda x: 0.0 if 'Baseline' in x else float(x.split('Pruned')[-1].replace('%', '').strip())
    )
    yolo_unstruct = yolo_unstruct.sort_values('reduction_pct')
    
    # Process YOLOv5s structured
    yolo_struct = structured[structured['model_name'].str.contains('YOLOV5s', case=False, na=False)].copy()
    yolo_struct['reduction_pct'] = yolo_struct['Parameters'].apply(
        lambda x: (yolo_base_params - x) / yolo_base_params * 100.0
    )
    
    # Plot YOLOv5s
    ax1 = axes[0]
    ax1.set_title("YOLOv5s: mAP@0.5 vs. Parameters Pruned (%)", fontsize=12, fontweight='bold', pad=10)
    ax1.plot(yolo_unstruct['reduction_pct'], yolo_unstruct['mAP@0.5'], 'o--', color=color_unstructured, linewidth=2.5, markersize=8, label='Unstructured Pruning')
    
    for idx, row in yolo_unstruct.iterrows():
        ax1.annotate(f"{row['reduction_pct']:.0f}%", (row['reduction_pct'], row['mAP@0.5']), textcoords="offset points", xytext=(-5, 8), ha='center', fontsize=9, fontweight='bold', color=color_unstructured)
        
    for idx, row in yolo_struct.iterrows():
        m_name = row['model_name']
        if 'Baseline' in m_name:
            continue
        
        if 'Filter' in m_name:
            color = color_structured_filter
            marker = '^' if 'Fine-tuned' in m_name else 'v'
            lbl = 'Structured Filter (Fine-tuned)' if 'Fine-tuned' in m_name else 'Structured Filter'
        else:
            color = color_structured_layer
            marker = 's' if 'Fine-tuned' in m_name else 'd'
            lbl = 'Structured Layer (Fine-tuned)' if 'Fine-tuned' in m_name else 'Structured Layer'
            
        ax1.scatter(row['reduction_pct'], row['mAP@0.5'], marker=marker, s=150, color=color, zorder=4, label=lbl)
        short_lbl = 'Filter FT' if 'Filter Pruned Fine-tuned' in m_name else ('Filter' if 'Filter Pruned' in m_name else ('Layer FT' if 'Layer Pruned Fine-tuned' in m_name else 'Layer'))
        ax1.annotate(f"{short_lbl} ({row['reduction_pct']:.1f}%)", (row['reduction_pct'], row['mAP@0.5']), textcoords="offset points", xytext=(8, -4), ha='left', fontsize=9, fontweight='bold', color=color)
        
    ax1.set_xlabel("Parameter Reduction (%)", fontsize=11)
    ax1.set_ylabel("mAP@0.5 (%)", fontsize=11)
    ax1.set_xlim(-5, 85)
    ax1.set_ylim(-5, 75)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # -------------------------------------------------------------------------
    # RT-DETR
    # -------------------------------------------------------------------------
    rtdetr_base_params = 20174608
    rtdetr_unstruct = unstructured[unstructured['model_name'].str.contains('RT-DETR', case=False, na=False)].copy()
    rtdetr_unstruct['reduction_pct'] = rtdetr_unstruct['model_name'].apply(
        lambda x: 0.0 if 'Baseline' in x else float(x.split('Pruned')[-1].replace('%', '').strip())
    )
    rtdetr_unstruct = rtdetr_unstruct.sort_values('reduction_pct')
    
    rtdetr_struct = structured[structured['model_name'].str.contains('RT-DETR', case=False, na=False)].copy()
    rtdetr_struct['reduction_pct'] = rtdetr_struct['Parameters'].apply(
        lambda x: (rtdetr_base_params - x) / rtdetr_base_params * 100.0
    )
    
    ax2 = axes[1]
    ax2.set_title("RT-DETR-R18: mAP@0.5 vs. Parameters Pruned (%)", fontsize=12, fontweight='bold', pad=10)
    ax2.plot(rtdetr_unstruct['reduction_pct'], rtdetr_unstruct['mAP@0.5'], 'o--', color=color_unstructured, linewidth=2.5, markersize=8, label='Unstructured Pruning')
    
    for idx, row in rtdetr_unstruct.iterrows():
        ax2.annotate(f"{row['reduction_pct']:.0f}%", (row['reduction_pct'], row['mAP@0.5']), textcoords="offset points", xytext=(-5, 8), ha='center', fontsize=9, fontweight='bold', color=color_unstructured)
        
    for idx, row in rtdetr_struct.iterrows():
        m_name = row['model_name']
        if 'Baseline' in m_name:
            continue
        
        ax2.scatter(row['reduction_pct'], row['mAP@0.5'], marker='^', s=150, color=color_structured_filter, zorder=4, label='Structured Pruned')
        ax2.annotate(f"Structured Pruned ({row['reduction_pct']:.1f}%)", (row['reduction_pct'], row['mAP@0.5']), textcoords="offset points", xytext=(8, -4), ha='left', fontsize=9, fontweight='bold', color=color_structured_filter)
        
    ax2.set_xlabel("Parameter Reduction (%)", fontsize=11)
    ax2.set_ylabel("mAP@0.5 (%)", fontsize=11)
    ax2.set_xlim(-5, 85)
    ax2.set_ylim(-5, 80)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Legends
    for ax in [ax1, ax2]:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='lower left', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    output_path = 'benchmarks/pruning_efficiency_comparison.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Generated efficiency comparison at: {output_path}")
    plt.close()

def generate_complexity_speed_dashboard(unstructured, structured, colors):
    color_unstructured, color_structured_filter, color_structured_layer, color_baseline = colors
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    fig.suptitle("Computational Complexity vs. Inference Performance", fontsize=16, fontweight='bold', y=0.98)
    
    # Filter only baseline & structured models for YOLOv5s and RT-DETR to see the impact of structured pruning
    # unstructured pruning has constant GFLOPs so we'll show it as a line.
    
    # -------------------------------------------------------------------------
    # Plot 1: YOLOv5s GFLOPs vs. FPS
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    ax1.set_title("YOLOv5s: Speed (FPS) vs. Computational Complexity (GFLOPs)", fontsize=12, fontweight='bold', pad=10)
    
    yolo_unstruct = unstructured[unstructured['model_name'].str.contains('YOLOv5s|Baseline YOLOv5s', case=False, na=False)].copy()
    yolo_struct = structured[structured['model_name'].str.contains('YOLOV5s', case=False, na=False)].copy()
    
    # Unstructured GFLOPs are constant, so plot a vertical region or range
    ax1.scatter(yolo_unstruct['GFLOPs'], yolo_unstruct['FPS'], color=color_unstructured, s=100, label='Unstructured Models', alpha=0.7)
    for idx, row in yolo_unstruct.iterrows():
        lbl = 'Base' if 'Baseline' in row['model_name'] else row['model_name'].split('Pruned')[-1].strip()
        ax1.annotate(lbl, (row['GFLOPs'], row['FPS']), textcoords="offset points", xytext=(-10, 5), ha='center', fontsize=8, color=color_unstructured)
        
    # Structured GFLOPs actually decrease
    for idx, row in yolo_struct.iterrows():
        m_name = row['model_name']
        if 'Baseline' in m_name:
            ax1.scatter(row['GFLOPs'], row['FPS'], marker='*', s=250, color=color_baseline, zorder=5, label='Baseline')
        else:
            if 'Filter' in m_name:
                color = color_structured_filter
                marker = '^' if 'Fine-tuned' in m_name else 'v'
                lbl = 'Structured Filter (Fine-tuned)' if 'Fine-tuned' in m_name else 'Structured Filter'
            else:
                color = color_structured_layer
                marker = 's' if 'Fine-tuned' in m_name else 'd'
                lbl = 'Structured Layer (Fine-tuned)' if 'Fine-tuned' in m_name else 'Structured Layer'
            
            ax1.scatter(row['GFLOPs'], row['FPS'], marker=marker, s=150, color=color, zorder=4, label=lbl)
            short_lbl = 'Filter FT' if 'Filter Pruned Fine-tuned' in m_name else ('Filter' if 'Filter Pruned' in m_name else ('Layer FT' if 'Layer Pruned Fine-tuned' in m_name else 'Layer'))
            ax1.annotate(short_lbl, (row['GFLOPs'], row['FPS']), textcoords="offset points", xytext=(8, -4), ha='left', fontsize=9, fontweight='bold', color=color)
            
    ax1.set_xlabel("Complexity (GFLOPs)", fontsize=11)
    ax1.set_ylabel("Frame Rate (FPS)", fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_xlim(13.8, 17.0)
    ax1.set_ylim(2.5, 10.5)
    
    # -------------------------------------------------------------------------
    # Plot 2: RT-DETR GFLOPs vs. FPS
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    ax2.set_title("RT-DETR-R18: Speed (FPS) vs. Computational Complexity (GFLOPs)", fontsize=12, fontweight='bold', pad=10)
    
    rtdetr_unstruct = unstructured[unstructured['model_name'].str.contains('RT-DETR', case=False, na=False)].copy()
    rtdetr_struct = structured[structured['model_name'].str.contains('RT-DETR', case=False, na=False)].copy()
    
    ax2.scatter(rtdetr_unstruct['GFLOPs'], rtdetr_unstruct['FPS'], color=color_unstructured, s=100, label='Unstructured Models', alpha=0.7)
    for idx, row in rtdetr_unstruct.iterrows():
        lbl = 'Base' if 'Baseline' in row['model_name'] else row['model_name'].split('Pruned')[-1].strip()
        ax2.annotate(lbl, (row['GFLOPs'], row['FPS']), textcoords="offset points", xytext=(-10, 5), ha='center', fontsize=8, color=color_unstructured)
        
    for idx, row in rtdetr_struct.iterrows():
        m_name = row['model_name']
        if 'Baseline' in m_name:
            ax2.scatter(row['GFLOPs'], row['FPS'], marker='*', s=250, color=color_baseline, zorder=5, label='Baseline')
        else:
            ax2.scatter(row['GFLOPs'], row['FPS'], marker='^', s=150, color=color_structured_filter, zorder=4, label='Structured Pruned')
            ax2.annotate('Structured Pruned', (row['GFLOPs'], row['FPS']), textcoords="offset points", xytext=(8, -4), ha='left', fontsize=9, fontweight='bold', color=color_structured_filter)
            
    ax2.set_xlabel("Complexity (GFLOPs)", fontsize=11)
    ax2.set_ylabel("Frame Rate (FPS)", fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_xlim(52, 63)
    ax2.set_ylim(0.4, 3.2)
    
    # Legends
    for ax in [ax1, ax2]:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    output_path = 'benchmarks/computation_speed_comparison.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Generated GFLOPs vs Speed comparison at: {output_path}")
    plt.close()

def main():
    setup_plot_style()
    mag_acc, mag_hw, str_acc, str_hw = load_data()
    unstructured, structured = preprocess_data(mag_acc, mag_hw, str_acc, str_hw)
    
    # Colors: Unstructured (Blue), Filter (Orange), Layer (Green/Teal), Baseline (Dark Gray)
    colors = ['#2E66FF', '#FF5400', '#00C49F', '#1A1D20']
    
    generate_comparison_dashboard(unstructured, structured, colors)
    generate_efficiency_comparison(unstructured, structured, colors)
    generate_complexity_speed_dashboard(unstructured, structured, colors)
    
    print("All comparison plots successfully generated!")

if __name__ == '__main__':
    main()
