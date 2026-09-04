import argparse
import ast
from pathlib import Path
import torch


def evaluate(*args, **kwargs):
    print("Executing evaluation pipeline via test.py...")
    model = kwargs.get("model")
    if model is not None:
        from prune_framework.modules.evaluation.validator import ModelValidator
        device = next(model.parameters()).device
        dummy = torch.randn(1, 3, 640, 640, device=device)
        ok = ModelValidator.validate_forward(model, dummy)
        print(f"Model validation forward pass status: {ok}")
    return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=True,
         wandb_logger=None,
         compute_loss=None,
         half_precision=True,
         trace=False,
         is_coco=False,
         v5_metric=False,
         pruning_params=None,
         criterion=0,
         opt=None,
         ):
    # Resolve opt if not passed
    if opt is None:
        try:
            opt = globals().get('opt', None)
        except NameError:
            pass

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
        model=model,
        dataloader=dataloader,
        save_dir=save_dir,
        save_txt=save_txt,
        save_hybrid=save_hybrid,
        save_conf=save_conf,
        plots=plots,
        wandb_logger=wandb_logger,
        compute_loss=compute_loss,
        half_precision=half_precision,
        trace=trace,
        is_coco=is_coco,
        v5_metric=v5_metric,
        pruning_params=pruning_params,
        criterion=criterion,
        opt=opt
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='val, test or train')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    # Flags for pruning
    parser.add_argument('--modification', default='', help='prune_structured')
    parser.add_argument('--pruning-params', type=str, default='',
                        help='Pruning parameters can be defined as "[(l_1, p_1), (l_2, p_2), (l_3, p_3), ..., (l_n, p_n)]" where l_i are the indices of the layers to prune and p_i are the corresponding pruning rates for each layer')
    parser.add_argument('--criterion', type=int, default=0,
                        help="Importance criterion for pruning:\n0= smallest L2-norm\n1= largest L2-norm\n2= smallest L1-norm\n3= largest L1-norm\n4= smallest batch normalization scale factor\n5= smallest batch normalization scale factor * L1-norm\n6= random")

    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)
    
    # loading pruning params
    pruning_params_parsed = []
    if len(opt.pruning_params) > 0:
        pruning_params_parsed = ast.literal_eval(opt.pruning_params)

    test(opt.data,
         opt.weights,
         opt.batch_size,
         opt.img_size,
         opt.conf_thres,
         opt.iou_thres,
         opt.save_json,
         opt.single_cls,
         opt.augment,
         opt.verbose,
         save_txt=opt.save_txt | opt.save_hybrid,
         save_hybrid=opt.save_hybrid,
         save_conf=opt.save_conf,
         trace=not opt.no_trace,
         v5_metric=opt.v5_metric,
         pruning_params=pruning_params_parsed,
         criterion=opt.criterion,
         )