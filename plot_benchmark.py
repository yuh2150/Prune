import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    csv_file = 'benchmark_results.csv'
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Please run python run_benchmark.py first.")
        return
        
    df = pd.read_csv(csv_file)
    
    # Set style for professional look
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
    
    models = df['Model Name'].tolist()
    colors = ['#FF4B4B', '#1F77B4'] # Red for Baseline, Blue for Pruned
    
    # ----------------------------------------------------
    # Chart 1: Parameters & GFLOPs (Normalized comparison)
    # ----------------------------------------------------
    ax1 = axes[0]
    x = np.arange(len(models))
    width = 0.35
    
    # Divide parameters by 1M for cleaner axis
    params_m = [p / 1E6 for p in df['Parameters']]
    rects1 = ax1.bar(x - width/2, params_m, width, label='Params (Millions)', color='#E377C2')
    
    # Use secondary y-axis for GFLOPs
    ax1_2 = ax1.twinx()
    rects2 = ax1_2.bar(x + width/2, df['GFLOPs'], width, label='GFLOPs', color='#BCBD22')
    
    ax1.set_title("Model Complexity Comparison", fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=12)
    ax1.set_ylabel("Parameters (Millions)", color='#E377C2', fontsize=12)
    ax1_2.set_ylabel("GFLOPs", color='#BCBD22', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Add values on top of bars
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f'{h:.2f}M', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax1_2.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

    # ----------------------------------------------------
    # Chart 2: Accuracy (mAP@0.5 & mAP@0.5:0.95)
    # ----------------------------------------------------
    ax2 = axes[1]
    rects3 = ax2.bar(x - width/2, df['mAP@0.5'] * 100, width, label='mAP@0.5', color='#17BECF')
    rects4 = ax2.bar(x + width/2, df['mAP@0.5:0.95'] * 100, width, label='mAP@0.5:0.95 (COCO)', color='#1F77B4')
    
    ax2.set_title("Detection Accuracy (mAP %)", fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=12)
    ax2.set_ylabel("mAP (%)", fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Add values on top of bars
    for rect in rects3:
        h = rect.get_height()
        ax2.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
    for rect in rects4:
        h = rect.get_height()
        ax2.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

    # ----------------------------------------------------
    # Chart 3: Total Latency (ms/img)
    # ----------------------------------------------------
    ax3 = axes[2]
    rects5 = ax3.bar(models, df['Total Latency (ms/img)'], width=0.5, color=['#FF7F0E', '#2CA02C'])
    
    ax3.set_title("Total Latency on GPU (ms/image)", fontsize=14, fontweight='bold', pad=15)
    ax3.set_ylabel("Latency (ms)", fontsize=12)
    # Give some headroom on y-axis
    ax3.set_ylim(0, max(df['Total Latency (ms/img)']) * 1.3)
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    # Add values on top of bars
    for rect in rects5:
        h = rect.get_height()
        ax3.annotate(f'{h:.2f} ms', xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
                    
    # Adjust layout
    plt.tight_layout()
    chart_file = 'benchmark_chart.png'
    plt.savefig(chart_file, bbox_inches='tight')
    print(f"Successfully generated and saved comparison chart to: {os.path.abspath(chart_file)}")

if __name__ == '__main__':
    main()
