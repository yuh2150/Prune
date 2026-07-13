import argparse
import ast
from pathlib import Path
from utils.general import check_file
from evaluation.core import evaluate
from evaluation.sensitivity import run_sensitivity_analysis

def evaluate_wrapper(data, weights=None, batch_size=32, imgsz=640, conf_thres=0.001, iou_thres=0.6,
                     save_json=False, single_cls=False, augment=False, verbose=False,
                     save_txt=False, save_hybrid=False, save_conf=False, trace=False,
                     v5_metric=False, pruning_params=None, plots=False, model=None):
    """Wrapper to map run_sensitivity_analysis parameters to the evaluate function."""
    return evaluate(
        data=data,
        weights=weights,
        batch_size=batch_size,
        imgsz=imgsz,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        save_json=save_json,
        single_cls=single_cls,
        augment=augment,
        verbose=verbose,
        save_txt=save_txt,
        save_hybrid=save_hybrid,
        save_conf=save_conf,
        trace=trace,
        v5_metric=v5_metric,
        pruning_params=pruning_params,
        criterion=opt.criterion,
        plots=plots,
        opt=opt,
        model=model,
        training=False
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='sensitivity_analysis.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    # Flags for pruning
    parser.add_argument('--modification', default='', help='prune_structured')
    parser.add_argument('--prune-output', type=str, default='output.txt', help="file to write results of pruning to")
    parser.add_argument('--pruning-params', type=str, default='',
                        help='Pruning parameters can be defined as "[(l_1, p_1), (l_2, p_2), (l_3, p_3), ..., (l_n, p_n)]" where l_i are the indices of the layers to prune and p_i are the corresponding pruning rates for each layer')
    parser.add_argument('--criterion', type=int, default=0,
                        help="Importance criterion for pruning:\n0= smallest L2-norm\n1= largest L2-norm\n2= smallest L1-norm\n3= largest L1-norm\n4= smallest batch normalization scale factor\n5= smallest batch normalization scale factor * L1-norm\n6= random")
    parser.add_argument('--pruning-rate', type=str, default='0.5', help='Pruning rate can be either defined as a string of list of pruning rates or a single rate')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--max-eval-batches', type=int, default=None, help='maximum number of batches to evaluate for speed')

    opt = parser.parse_args()
    import os
    if not os.path.isdir(opt.data):
        opt.data = check_file(opt.data)
    opt.pruning_rate = ast.literal_eval(opt.pruning_rate)
    
    pruning_params_parsed = []
    if len(opt.pruning_params) > 0:
        pruning_params_parsed = ast.literal_eval(opt.pruning_params)

    print(opt)
    run_sensitivity_analysis(opt, evaluate_wrapper)
