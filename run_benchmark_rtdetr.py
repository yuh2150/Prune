import os
import pandas as pd
import torch
from pathlib import Path
import test_rtdetr
from test_rtdetr import test

class DummyOptions:
    def __init__(self, device='', project='runs/test', name='exp', exist_ok=True, modification='', prune_output='output.txt', pruning_rate='0.5', task='val', conf_thres=0.001, iou_thres=0.6, data='data/coco500.yaml', weights='PekingU/rtdetr_r18vd', batch_size=8, img_size=640, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, exist_ok_opt=True, no_trace=True, v5_metric=False, img_dir='./coco/images/val2017', ann_file='./coco/annotations/instances_val2017.json'):
        self.device = device
        self.project = project
        self.name = name
        self.exist_ok = exist_ok
        self.modification = modification
        self.prune_output = prune_output
        self.pruning_rate = pruning_rate
        self.task = task
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.data = data
        self.weights = [weights]
        self.batch_size = batch_size
        self.img_size = img_size
        self.single_cls = single_cls
        self.augment = augment
        self.verbose = verbose
        self.save_txt = save_txt
        self.save_hybrid = save_hybrid
        self.save_conf = save_conf
        self.save_json = save_json
        self.no_trace = no_trace
        self.v5_metric = v5_metric
        self.img_dir = img_dir
        self.ann_file = ann_file

def main():
    # Options
    data_path = 'data/coco500.yaml'
    # Use GPU if available, else CPU
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Running RT-DETR benchmark on device: {device}")
    
    # We look for rtdetr-pruned.pt, then fall back to pruned_rtdetr.pt
    pruned_weight = 'rtdetr-pruned.pt' if os.path.exists('rtdetr-pruned.pt') else 'pruned_rtdetr.pt'
    if not os.path.exists(pruned_weight):
        print(f"Warning: Pruned weight file '{pruned_weight}' not found. Will skip evaluating Pruned model.")
        
    models = {
        'Baseline': 'PekingU/rtdetr_r18vd',
        'Pruned': 'rtdetr-pruned.pt',
    }
    # if os.path.exists(pruned_weight):
    #     models['Pruned'] = pruned_weight
        
    results_list = []
    
    for name, weights in models.items():
        print(f"\n=========================================")
        print(f"Evaluating {name} Model: {weights}")
        print(f"=========================================")
        
        opt = DummyOptions(
            device=device,
            project='runs/test',
            name=f'bench_rtdetr_{name.lower()}',
            exist_ok=True,
            data=data_path,
            weights=weights,
            batch_size=8,
            ann_file='./coco/annotations/instances_val2017.json',
            img_dir='./coco/images/val2017'
        )
        test_rtdetr.opt = opt
        
        try:
            # We run evaluation under standard test config (conf_thres=0.001, iou_thres=0.6)
            results, maps, speed, params, flops = test(
                data=data_path,
                weights=weights,
                batch_size=8,
                imgsz=640,
                conf_thres=0.001,
                iou_thres=0.6,
                save_json=False,
                plots=False,
                img_dir=opt.img_dir,
                ann_file=opt.ann_file,
                modification=''
            )
            
            # results: (mp, mr, map50, map, ...)
            mp, mr, map50, map_coco = results[0], results[1], results[2], results[3]
            # speed: (prep_and_inf_ms, postprocess_ms, total_ms, imgsz, imgsz, batch_size)
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
            import traceback
            traceback.print_exc()
            
    if not results_list:
        print("No results to save.")
        return
        
    df = pd.DataFrame(results_list)
    print("\n" + df.to_string(index=False))
    
    # Calculate percentage reduction for params and flops, and speedup for latency
    if len(df) >= 2:
        print("\n--- Summary of Comparison ---")
        try:
            base_row = df[df['Model Name'] == 'Baseline'].iloc[0]
            pruned_row = df[df['Model Name'] == 'Pruned'].iloc[0]
            
            param_reduction = (base_row['Parameters'] - pruned_row['Parameters']) / base_row['Parameters'] * 100
            flops_reduction = (base_row['GFLOPs'] - pruned_row['GFLOPs']) / base_row['GFLOPs'] * 100
            speedup = (base_row['Total Latency (ms/img)'] - pruned_row['Total Latency (ms/img)']) / base_row['Total Latency (ms/img)'] * 100
            
            print(f"Parameters reduced by: {param_reduction:.2f}%")
            print(f"GFLOPs reduced by:     {flops_reduction:.2f}%")
            print(f"Latency reduced by:    {speedup:.2f}%")
        except Exception as e:
            print(f"Error printing summary: {e}")
        
    # Export to CSV
    csv_file = 'benchmark_results_rtdetr.csv'
    df.to_csv(csv_file, index=False)
    print(f"\nSaved CSV report to: {os.path.abspath(csv_file)}")
    
    # Export to XLSX
    xlsx_file = 'benchmark_results_rtdetr.xlsx'
    try:
        df.to_excel(xlsx_file, index=False)
        print(f"Saved Excel report to: {os.path.abspath(xlsx_file)}")
    except Exception as e:
        print(f"Could not save Excel file (pandas xlsx engine missing?): {e}")

if __name__ == '__main__':
    main()
