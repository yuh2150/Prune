import argparse
import ast
import os
from pathlib import Path
import numpy as np
from utils.general import check_file
from utils.plots import plot_study_txt
from evaluation.core import evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='study.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    # Flags for pruning and config
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--modification', default='', help='prune_structured')
    parser.add_argument('--pruning-params', type=str, default='',
                        help='Pruning parameters can be defined as "[(l_1, p_1), (l_2, p_2), (l_3, p_3), ..., (l_n, p_n)]" where l_i are the indices of the layers to prune and p_i are the corresponding pruning rates for each layer')
    parser.add_argument('--criterion', type=int, default=0,
                        help="Importance criterion for pruning:\n0= smallest L2-norm\n1= largest L2-norm\n2= smallest L1-norm\n3= largest L1-norm\n4= smallest batch normalization scale factor\n5= smallest batch normalization scale factor * L1-norm\n6= random")

    opt = parser.parse_args()
    opt.data = check_file(opt.data)
    
    pruning_params_parsed = []
    if len(opt.pruning_params) > 0:
        pruning_params_parsed = ast.literal_eval(opt.pruning_params)

    print(opt)

    x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
    for w in opt.weights:
        f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
        y = []  # y axis
        for i in x:  # img-size
            print(f'\nRunning {f} point {i}...')
            r, _, t = evaluate(
                data=opt.data,
                weights=w,
                batch_size=opt.batch_size,
                imgsz=i,
                conf_thres=opt.conf_thres,
                iou_thres=opt.iou_thres,
                save_json=opt.save_json,
                plots=False,
                v5_metric=opt.v5_metric,
                trace=not opt.no_trace,
                pruning_params=pruning_params_parsed,
                criterion=opt.criterion,
                opt=opt
            )
            y.append(r + t)  # results and times
        np.savetxt(f, y, fmt='%10.4g')  # save
    os.system('zip -r study.zip study_*.txt')
    plot_study_txt(x=x)  # plot
