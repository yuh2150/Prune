import os
import pandas as pd
from pathlib import Path
from evaluation.core import evaluate

class DummyOptions:
    def __init__(self, device='', project='runs/test', name='exp', exist_ok=True, modification=''):
        self.device = device
        self.project = project
        self.name = name
        self.exist_ok = exist_ok
        self.modification = modification

def main():
    # Options
    data_path = 'data/coco500.yaml'
    # Use GPU if available, else CPU
    import torch
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Running benchmark on device: {device}")
    
    models = {
        'Baseline': 'yolov5s.pt',
        'Pruned': 'yolov5s-pruned.pt',
        'Fine-tuned-50epochs': 'yolov5s-pruned-finetuned.pt'
    }
    
    results_list = []
    
    for name, weights in models.items():
        print(f"\n=========================================")
        print(f"Evaluating {name} Model: {weights}")
        print(f"=========================================")
        
        opt = DummyOptions(device=device, project='runs/test', name=f'bench_{name.lower()}', exist_ok=True)
        
        try:
            # We run evaluation under standard test config (conf_thres=0.001, iou_thres=0.6)
            results, maps, speed, params, flops = evaluate(
                data=data_path,
                weights=weights,
                batch_size=1,
                imgsz=640,
                conf_thres=0.001,
                iou_thres=0.6,
                save_json=False,
                plots=False,
                opt=opt
            )
            
            # results: (mp, mr, map50, map, box_loss, obj_loss, cls_loss)
            mp, mr, map50, map_coco = results[0], results[1], results[2], results[3]
            # speed: (inference_ms, nms_ms, total_ms, imgsz, imgsz, batch_size)
            inf_speed, nms_speed, total_speed = speed[0], speed[1], speed[2]
            
            results_list.append({
                'Model Name': name,
                'Weight File': weights,
                'Parameters': params,
                'GFLOPs': flops,
                'Precision (P)': round(mp, 4),
                'Recall (R)': round(mr, 4),
                'mAP@0.5': round(map50, 4),
                'mAP@0.5:0.95': round(map_coco, 4),
                'Inference Speed (ms/img)': round(inf_speed, 2),
                'NMS Speed (ms/img)': round(nms_speed, 2),
                'Total Latency (ms/img)': round(total_speed, 2),
                'FPS': round(1000 / total_speed, 1) if total_speed > 0 else 0.0
            })
            
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            
    if not results_list:
        print("No results to save.")
        return
        
    df = pd.DataFrame(results_list)
    
    # Calculate percentage reduction for params and flops, and speedup for latency
    if len(df) == 2:
        # Add a difference row or columns
        print("\n--- Summary of Comparison ---")
        base_row = df.iloc[0]
        pruned_row = df.iloc[1]
        
        param_reduction = (base_row['Parameters'] - pruned_row['Parameters']) / base_row['Parameters'] * 100
        flops_reduction = (base_row['GFLOPs'] - pruned_row['GFLOPs']) / base_row['GFLOPs'] * 100
        speedup = (base_row['Total Latency (ms/img)'] - pruned_row['Total Latency (ms/img)']) / base_row['Total Latency (ms/img)'] * 100
        
        print(f"Parameters reduced by: {param_reduction:.2f}%")
        print(f"GFLOPs reduced by:     {flops_reduction:.2f}%")
        print(f"Latency reduced by:    {speedup:.2f}%")

    # Export to CSV
    csv_file = 'benchmark_results.csv'
    df.to_csv(csv_file, index=False)
    print(f"\nSaved CSV report to: {os.path.abspath(csv_file)}")
    
    # Export to XLSX
    xlsx_file = 'benchmark_results.xlsx'
    try:
        df.to_excel(xlsx_file, index=False)
        print(f"Saved Excel report to: {os.path.abspath(xlsx_file)}")
    except Exception as e:
        print(f"Could not save Excel file (pandas xlsx engine missing?): {e}")

if __name__ == '__main__':
    main()
