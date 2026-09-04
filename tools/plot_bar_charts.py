import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def load_data():
    mag_acc = pd.read_csv('benchmarks/Benchmark(Magnitude_Accuracy).csv')
    mag_hw = pd.read_csv('benchmarks/Benchmark(Magnitude_Hardware).csv')
    str_acc = pd.read_csv('benchmarks/Benchmark(Structure Accuracy).csv')
    str_hw = pd.read_csv('benchmarks/Benchmark(Structure Hardware).csv')
    
    for df in [mag_acc, mag_hw, str_acc, str_hw]:
        df.columns = df.columns.str.strip()
        
    return mag_acc, mag_hw, str_acc, str_hw

def preprocess_and_merge():
    mag_acc, mag_hw, str_acc, str_hw = load_data()
    
    # Normalize mAP to percentages
    for df in [mag_acc, str_acc]:
        for col in ['mAP@0.5', 'mAP@0.5:0.95']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: x * 100 if x <= 1.0 else x)
                
    # Merge
    mag = pd.merge(
        mag_acc.rename(columns={'Model Name': 'model_name'}),
        mag_hw.rename(columns={'Model Name': 'model_name'}),
        on='model_name'
    )
    
    # Exclude empty rows in str_hw
    str_hw = str_hw.rename(columns={'Models': 'model_name'})
    str_hw['model_name'] = str_hw['model_name'].str.strip()
    str_hw = str_hw[str_hw['model_name'].notna() & (str_hw['model_name'] != '')]
    
    str_acc = str_acc.rename(columns={'Models': 'model_name'})
    str_acc['model_name'] = str_acc['model_name'].str.strip()
    
    struc = pd.merge(str_acc, str_hw, on='model_name')
    
    return mag, struc

def plot_accuracy_dashboard(df_plot, title, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    x = np.arange(len(df_plot))
    width = 0.5
    
    # 1. mAP@0.5
    ax = axes[0]
    bars = ax.bar(x, df_plot['mAP05'], width, color=df_plot['color'], edgecolor='none')
    ax.set_title("mAP@0.5 (%) - Higher is Better", fontsize=11, fontweight='bold', pad=8)
    ax.set_ylabel("mAP@0.5 (%)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot['label'], rotation=15, ha='right', fontsize=9)
    ax.set_ylim(0, max(df_plot['mAP05'].max() * 1.15, 10))
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.2f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=8)
                    
    # 2. mAP@0.5:0.95
    ax = axes[1]
    bars = ax.bar(x, df_plot['mAP0595'], width, color=df_plot['color'], edgecolor='none')
    ax.set_title("mAP@0.5:0.95 (%) - Higher is Better", fontsize=11, fontweight='bold', pad=8)
    ax.set_ylabel("mAP@0.5:0.95 (%)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot['label'], rotation=15, ha='right', fontsize=9)
    ax.set_ylim(0, max(df_plot['mAP0595'].max() * 1.15, 10))
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.2f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=8)
                    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Generated accuracy dashboard at: {output_path}")
    plt.close()

def plot_hardware_dashboard(df_plot, title, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    x = np.arange(len(df_plot))
    width = 0.5
    
    # 1. Parameters
    ax = axes[0]
    bars = ax.bar(x, df_plot['params'], width, color=df_plot['color'], edgecolor='none')
    ax.set_title("Model Parameters (M) - Lower is Better", fontsize=11, fontweight='bold', pad=8)
    ax.set_ylabel("Parameters (M)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot['label'], rotation=15, ha='right', fontsize=9)
    ax.set_ylim(0, max(df_plot['params'].max() * 1.15, 1))
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.2f}M', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=8)
                    
    # 2. GFLOPs
    ax = axes[1]
    bars = ax.bar(x, df_plot['gflops'], width, color=df_plot['color'], edgecolor='none')
    ax.set_title("Complexity (GFLOPs) - Lower is Better", fontsize=11, fontweight='bold', pad=8)
    ax.set_ylabel("GFLOPs", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot['label'], rotation=15, ha='right', fontsize=9)
    ax.set_ylim(0, max(df_plot['gflops'].max() * 1.15, 1))
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=8)
                    
    # 3. FPS
    ax = axes[2]
    bars = ax.bar(x, df_plot['fps'], width, color=df_plot['color'], edgecolor='none')
    ax.set_title("Frame Rate (FPS) - Higher is Better", fontsize=11, fontweight='bold', pad=8)
    ax.set_ylabel("FPS", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot['label'], rotation=15, ha='right', fontsize=9)
    ax.set_ylim(0, max(df_plot['fps'].max() * 1.15, 1))
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=8)
                    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Generated hardware dashboard at: {output_path}")
    plt.close()

def main():
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['font.family'] = 'sans-serif'
    style_to_use = 'default'
    for style in ['seaborn-v0_8-whitegrid', 'seaborn-whitegrid', 'ggplot']:
        if style in plt.style.available:
            style_to_use = style
            break
    plt.style.use(style_to_use)
    
    mag, struc = preprocess_and_merge()
    os.makedirs('benchmarks', exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Extract YOLOv5s
    # -------------------------------------------------------------------------
    yolo_mag = mag[mag['model_name'].str.contains('YOLOv5s|Baseline YOLOv5s', case=False, na=False)].copy()
    yolo_mag['sort_key'] = yolo_mag['model_name'].apply(
        lambda x: 0 if 'Baseline' in x else int(x.split('Pruned')[-1].replace('%', '').strip())
    )
    yolo_mag = yolo_mag.sort_values('sort_key')
    
    yolo_struc = struc[struc['model_name'].str.contains('YOLOV5s', case=False, na=False)].copy()
    yolo_struc_base = yolo_struc[yolo_struc['model_name'].str.contains('Baseline', case=False, na=False)]
    yolo_struc_f = yolo_struc[yolo_struc['model_name'] == 'YOLOV5s - Filter Pruned']
    yolo_struc_fft = yolo_struc[yolo_struc['model_name'] == 'YOLOV5s - Filter Pruned Fine-tuned']
    yolo_struc_l = yolo_struc[yolo_struc['model_name'] == 'YOLOV5s - Layer Pruned']
    yolo_struc_lft = yolo_struc[yolo_struc['model_name'] == 'YOLOV5s - Layer Pruned Fine-tuned']
    
    # -------------------------------------------------------------------------
    # Extract RT-DETR
    # -------------------------------------------------------------------------
    rtdetr_mag = mag[mag['model_name'].str.contains('RT-DETR', case=False, na=False)].copy()
    rtdetr_mag['sort_key'] = rtdetr_mag['model_name'].apply(
        lambda x: 0 if 'Baseline' in x else int(x.split('Pruned')[-1].replace('%', '').strip())
    )
    rtdetr_mag = rtdetr_mag.sort_values('sort_key')
    
    rtdetr_struc = struc[struc['model_name'].str.contains('RT-DETR', case=False, na=False)].copy()
    rtdetr_struc_base = rtdetr_struc[rtdetr_struc['model_name'].str.contains('Baseline', case=False, na=False)]
    rtdetr_struc_pruned = rtdetr_struc[~rtdetr_struc['model_name'].str.contains('Baseline', case=False, na=False)]
    
    # =========================================================================
    # 1. YOLOv5s Unstructured Plots
    # =========================================================================
    yolo_unstructured_list = []
    for _, row in yolo_mag.iterrows():
        name = row['model_name']
        label = 'Baseline' if 'Baseline' in name else f"Pruned {name.split('Pruned')[-1].strip()}"
        yolo_unstructured_list.append({
            'label': label,
            'mAP05': row['mAP@0.5'],
            'mAP0595': row['mAP@0.5:0.95'],
            'params': row['Parameters'] / 1E6,
            'gflops': row['GFLOPs'],
            'fps': row['FPS'],
            'color': '#1E3A8A' if 'Baseline' in name else '#2E66FF'
        })
    df_yolo_unstruct = pd.DataFrame(yolo_unstructured_list)
    plot_accuracy_dashboard(df_yolo_unstruct, "YOLOv5s Unstructured Pruning: Accuracy (mAP)", "benchmarks/yolov5s_unstructured_accuracy_bar.png")
    plot_hardware_dashboard(df_yolo_unstruct, "YOLOv5s Unstructured Pruning: Hardware Performance", "benchmarks/yolov5s_unstructured_hardware_bar.png")
    
    # =========================================================================
    # 2. YOLOv5s Structured Plots
    # =========================================================================
    yolo_structured_list = []
    for row_df in [yolo_struc_base, yolo_struc_f, yolo_struc_fft, yolo_struc_l, yolo_struc_lft]:
        if len(row_df) > 0:
            row = row_df.iloc[0]
            name = row['model_name']
            if 'Baseline' in name:
                label = 'Baseline'
                color = '#374151'
            elif 'Filter' in name:
                label = 'Filter FT' if 'Fine-tuned' in name else 'Filter'
                color = '#FF5400' if 'Fine-tuned' in name else '#FF9E64'
            else:
                label = 'Layer FT' if 'Fine-tuned' in name else 'Layer'
                color = '#00C49F' if 'Fine-tuned' in name else '#5DF2D6'
            
            yolo_structured_list.append({
                'label': label,
                'mAP05': row['mAP@0.5'],
                'mAP0595': row['mAP@0.5:0.95'],
                'params': row['Parameters'] / 1E6,
                'gflops': row['GFLOPs'],
                'fps': row['FPS'],
                'color': color
            })
    df_yolo_struct = pd.DataFrame(yolo_structured_list)
    plot_accuracy_dashboard(df_yolo_struct, "YOLOv5s Structured Pruning: Accuracy (mAP)", "benchmarks/yolov5s_structured_accuracy_bar.png")
    plot_hardware_dashboard(df_yolo_struct, "YOLOv5s Structured Pruning: Hardware Performance", "benchmarks/yolov5s_structured_hardware_bar.png")
    
    # =========================================================================
    # 3. RT-DETR Unstructured Plots
    # =========================================================================
    rtdetr_unstructured_list = []
    for _, row in rtdetr_mag.iterrows():
        name = row['model_name']
        label = 'Baseline' if 'Baseline' in name else f"Pruned {name.split('Pruned')[-1].strip()}"
        rtdetr_unstructured_list.append({
            'label': label,
            'mAP05': row['mAP@0.5'],
            'mAP0595': row['mAP@0.5:0.95'],
            'params': row['Parameters'] / 1E6,
            'gflops': row['GFLOPs'],
            'fps': row['FPS'],
            'color': '#4C1D95' if 'Baseline' in name else '#8B5CF6'
        })
    df_rtdetr_unstruct = pd.DataFrame(rtdetr_unstructured_list)
    plot_accuracy_dashboard(df_rtdetr_unstruct, "RT-DETR-R18 Unstructured Pruning: Accuracy (mAP)", "benchmarks/rtdetr_unstructured_accuracy_bar.png")
    plot_hardware_dashboard(df_rtdetr_unstruct, "RT-DETR-R18 Unstructured Pruning: Hardware Performance", "benchmarks/rtdetr_unstructured_hardware_bar.png")
    
    # =========================================================================
    # 4. RT-DETR Structured Plots
    # =========================================================================
    rtdetr_structured_list = []
    for row_df in [rtdetr_struc_base, rtdetr_struc_pruned]:
        if len(row_df) > 0:
            row = row_df.iloc[0]
            name = row['model_name']
            label = 'Baseline' if 'Baseline' in name else 'Structured Pruned'
            color = '#374151' if 'Baseline' in name else '#EC4899'
            rtdetr_structured_list.append({
                'label': label,
                'mAP05': row['mAP@0.5'],
                'mAP0595': row['mAP@0.5:0.95'],
                'params': row['Parameters'] / 1E6,
                'gflops': row['GFLOPs'],
                'fps': row['FPS'],
                'color': color
            })
    df_rtdetr_struct = pd.DataFrame(rtdetr_structured_list)
    plot_accuracy_dashboard(df_rtdetr_struct, "RT-DETR-R18 Structured Pruning: Accuracy (mAP)", "benchmarks/rtdetr_structured_accuracy_bar.png")
    plot_hardware_dashboard(df_rtdetr_struct, "RT-DETR-R18 Structured Pruning: Hardware Performance", "benchmarks/rtdetr_structured_hardware_bar.png")
    
    print("All 8 separated dashboards successfully generated!")

if __name__ == '__main__':
    main()
