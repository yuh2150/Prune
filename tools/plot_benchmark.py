import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_benchmark_for_model(csv_file, title, output_image, colors):
    if not os.path.exists(csv_file):
        print(f"Warning: {csv_file} not found. Skipping plot for this model.")
        return False
        
    df = pd.read_csv(csv_file)
    
    # Process mAP to percentage if it's decimal
    df['mAP@0.5'] = df['mAP@0.5'].apply(lambda x: x * 100 if x <= 1.0 else x)
    df['mAP@0.5:0.95'] = df['mAP@0.5:0.95'].apply(lambda x: x * 100 if x <= 1.0 else x)
    
    # Setup styling
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['font.family'] = 'sans-serif'
    style_to_use = 'default'
    for style in ['seaborn-v0_8-whitegrid', 'seaborn-whitegrid', 'ggplot']:
        if style in plt.style.available:
            style_to_use = style
            break
    plt.style.use(style_to_use)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
    fig.suptitle(f"{title} - Pruning & Performance Benchmark", fontsize=16, fontweight='bold', y=0.98, color='#1A1D20')
    
    models = df['Model Name'].tolist()
    # Replace long names for layout cleanliness
    models_clean = [m.replace('-50epochs', '\n(Fine-tuned)') for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    # ----------------------------------------------------
    # Chart 1: Parameters & GFLOPs (Dual Axis)
    # ----------------------------------------------------
    ax1 = axes[0]
    params_m = [p / 1E6 for p in df['Parameters']]
    rects1 = ax1.bar(x - width/2, params_m, width, label='Params (M)', color=colors[0], edgecolor='none')
    
    ax1_2 = ax1.twinx()
    rects2 = ax1_2.bar(x + width/2, df['GFLOPs'], width, label='GFLOPs', color=colors[1], edgecolor='none')
    
    ax1.set_title("Model Complexity Comparison", fontsize=12, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models_clean, fontsize=10)
    ax1.set_ylabel("Parameters (Millions)", color=colors[0], fontsize=11, fontweight='bold')
    ax1_2.set_ylabel("GFLOPs", color=colors[1], fontsize=11, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add values on top of bars
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f'{h:.2f}M', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        ax1_2.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=9)
                    
    # Style twin axes spines
    ax1.spines['left'].set_color(colors[0])
    ax1_2.spines['right'].set_color(colors[1])
    ax1.spines['top'].set_visible(False)
    ax1_2.spines['top'].set_visible(False)

    # ----------------------------------------------------
    # Chart 2: Accuracy (mAP@0.5 & mAP@0.5:0.95)
    # ----------------------------------------------------
    ax2 = axes[1]
    rects3 = ax2.bar(x - width/2, df['mAP@0.5'], width, label='mAP@0.5', color=colors[2], edgecolor='none')
    rects4 = ax2.bar(x + width/2, df['mAP@0.5:0.95'], width, label='mAP@0.5:0.95', color=colors[3], edgecolor='none')
    
    ax2.set_title("Detection Accuracy (mAP %)", fontsize=12, fontweight='bold', pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models_clean, fontsize=10)
    ax2.set_ylabel("mAP (%)", fontsize=11)
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add values on top of bars
    for rect in rects3:
        h = rect.get_height()
        ax2.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=9)
    for rect in rects4:
        h = rect.get_height()
        ax2.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=9)
                    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # ----------------------------------------------------
    # Chart 3: Total Latency (ms/img)
    # ----------------------------------------------------
    ax3 = axes[2]
    bars_lat = ax3.bar(models_clean, df['Total Latency (ms/img)'], width=0.45, color=colors[4], edgecolor='none')
    
    ax3.set_title("Total Latency (ms) - Lower is Better", fontsize=12, fontweight='bold', pad=12)
    ax3.set_ylabel("Latency (ms / image)", fontsize=11)
    # Give some headroom on y-axis
    max_lat = df['Total Latency (ms/img)'].max()
    ax3.set_ylim(0, max_lat * 1.2)
    ax3.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add values on top of bars
    for bar in bars_lat:
        h = bar.get_height()
        ax3.annotate(f'{h:.1f} ms', xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=9)
                    
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
                    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(output_image, bbox_inches='tight', dpi=300)
    print(f"Successfully generated and saved: {os.path.abspath(output_image)}")
    plt.close()
    return True

def main():
    # Palettes: [Param color, GFLOPs color, mAP0.5 color, mAP0.5:0.95 color, Inference color]
    yolo_colors = ['#3A86FF', '#FFBD00', '#00C49F', '#007A87', '#FF5400']
    rtdetr_colors = ['#FF6B6B', '#8338EC', '#FFB84C', '#E27396', '#4895EF']
    
    print("Generating separate benchmark plots...")
    
    # Plot YOLOv5s
    plot_benchmark_for_model(
        csv_file='benchmarks/benchmark_results.csv',
        title='YOLOv5s Object Detector',
        output_image='benchmarks/benchmark_chart_yolo.png',
        colors=yolo_colors
    )
    
    # Plot RT-DETR
    plot_benchmark_for_model(
        csv_file='benchmarks/benchmark_results_rtdetr.csv',
        title='RT-DETR Object Detector',
        output_image='benchmarks/benchmark_chart_rtdetr.png',
        colors=rtdetr_colors
    )

if __name__ == '__main__':
    main()
