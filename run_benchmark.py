import os
import torch
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
    data_path = 'data/coco1000.yaml'
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Running benchmark on device: {device}")
    
    # Models to benchmark
    models = {
        'Baseline (YOLOv5s)': 'weights/yolov5s.pt',
        # 'Channel Pruned': 'weights/yolov5s-pruned.pt',
        # 'Channel Pruned Fine-tuned': 'weights/yolov5s-pruned-finetuned.pt',
        # 'Layer Pruned': 'weights/yolov5s-layer-pruned.pt',
        # 'Layer Pruned Fine-tuned': 'weights/yolov5s-layer-pruned-finetuned.pt',
        'Taylor Expansion Pruned': 'weights/yolov5s-taylor-pruned.pt'
    }
    
    results_list = []
    baseline_params = None
    baseline_flops = None
    baseline_latency = None

    for name, weights in models.items():
        if not os.path.exists(weights):
            print(f"Skipping {name} (File not found: {weights})")
            continue

        print(f"\n=========================================")
        print(f"Evaluating {name} Model: {weights}")
        print(f"=========================================")
        
        opt = DummyOptions(device=device, project='runs/test', name=f'bench_{name.lower().replace(" ", "_")}', exist_ok=True)
        
        try:
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
            
            mp, mr, map50, map_coco = results[0], results[1], results[2], results[3]
            inf_speed, nms_speed, total_speed = speed[0], speed[1], speed[2]
            
            if 'Baseline' in name or baseline_params is None:
                baseline_params = params
                baseline_flops = flops
                baseline_latency = total_speed

            param_red = ((baseline_params - params) / baseline_params * 100) if baseline_params else 0.0
            flops_red = ((baseline_flops - flops) / baseline_flops * 100) if baseline_flops else 0.0
            speedup = ((baseline_latency - total_speed) / baseline_latency * 100) if baseline_latency else 0.0

            results_list.append({
                'Model Name': name,
                'Weight File': weights,
                'Parameters': params,
                'Param Reduction (%)': round(param_red, 2),
                'GFLOPs': round(flops, 3),
                'FLOPs Reduction (%)': round(flops_red, 2),
                'Precision (P)': round(mp, 4),
                'Recall (R)': round(mr, 4),
                'mAP@0.5': round(map50, 4),
                'mAP@0.5:0.95': round(map_coco, 4),
                'Inference Speed (ms)': round(inf_speed, 2),
                'NMS Speed (ms)': round(nms_speed, 2),
                'Total Latency (ms)': round(total_speed, 2),
                'FPS': round(1000 / total_speed, 1) if total_speed > 0 else 0.0,
                'Speedup (%)': round(speedup, 2)
            })
            
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            
    if not results_list:
        print("No results to save.")
        return
        
    df = pd.DataFrame(results_list)
    os.makedirs("benchmarks", exist_ok=True)
    
    # Export to CSV
    csv_file = 'benchmarks/taylor_benchmark_results.csv'
    df.to_csv(csv_file, index=False)
    print(f"\n[Benchmark] Saved CSV report to: {os.path.abspath(csv_file)}")
    
    # Export to XLSX
    xlsx_file = 'benchmarks/taylor_benchmark_results.xlsx'
    try:
        df.to_excel(xlsx_file, index=False)
        print(f"[Benchmark] Saved Excel report to: {os.path.abspath(xlsx_file)}")
    except Exception as e:
        print(f"[Benchmark] Warning exporting Excel: {e}")

if __name__ == '__main__':
    main()
