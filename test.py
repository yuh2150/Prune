import argparse
import ast
import json
import numpy as np
import yaml
from pathlib import Path
import torch
from utils.general import (check_file, check_dataset, check_img_size, non_max_suppression,
                           scale_coords, xywh2xyxy, xyxy2xywh, coco80_to_coco91_class, increment_path)
from utils.metrics import ap_per_class, box_iou, ConfusionMatrix
from utils.torch_utils import select_device, time_sync


def process_batch(detections, labels, iouv):
    """One-to-one, same-class box matches, independently at each IoU threshold."""
    correct = torch.zeros((len(detections), len(iouv)), dtype=torch.bool, device=iouv.device)
    if not len(labels) or not len(detections):
        return correct
    iou = box_iou(labels[:, 1:], detections[:, :4])
    same_class = labels[:, :1] == detections[:, 5]
    for k, threshold in enumerate(iouv):
        gt, pred = torch.where((iou >= threshold) & same_class)
        if not len(gt):
            continue
        matches = torch.stack((gt, pred, iou[gt, pred]), 1).cpu().numpy()
        matches = matches[matches[:, 2].argsort()[::-1]]
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        matches = matches[matches[:, 2].argsort()[::-1]]
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(int), k] = True
    return correct


@torch.no_grad()
def evaluate(data, weights=None, batch_size=32, imgsz=640, conf_thres=0.001,
             iou_thres=0.6, save_json=False, single_cls=False, augment=False,
             verbose=False, model=None, dataloader=None, save_dir=Path(''),
             save_txt=False, save_hybrid=False, save_conf=False, plots=True,
             wandb_logger=None, compute_loss=None, half_precision=True,
             trace=False, is_coco=False, v5_metric=False, pruning_params=None,
             criterion=0, opt=None, quality_metrics=None):
    """Validate local YOLO checkpoints or an in-memory training/EMA model.

    Returns (results, maps, times): seven floats (P, R, AP50, AP50:95,
    box/objectness/class loss), an nc-length AP50:95 array, and the legacy
    (inference_ms, nms_ms, total_ms, imgsz, imgsz, batch_size) tuple.
    Classes absent from validation inherit aggregate mAP, as expected by
    train.py's image weighting. Loss is zero when no loss callback is supplied.
    AP uses the repository's ap_per_class implementation, not COCOeval.
    Optional quality_metrics receives macro F1 at the same IoU-0.5 operating
    point as P/R, plus the image count, without changing the legacy tuple.
    Legacy trace/wandb arguments are accepted; inference remains eager.
    """
    if pruning_params:
        raise ValueError('Apply pruning before evaluation and pass the pruned model or checkpoint.')
    if batch_size < 1 or imgsz < 1:
        raise ValueError('batch_size and imgsz must be positive')
    task = getattr(opt, 'task', 'val')
    if task not in ('train', 'val', 'test'):
        raise ValueError(f'Unsupported evaluation task: {task}')
    if isinstance(data, (str, Path)):
        with open(check_file(str(data)), encoding='utf-8') as f:
            data = yaml.safe_load(f)
    data = dict(data)
    nc = 1 if single_cls else int(data['nc'])
    names = data.get('names', [str(i) for i in range(nc)])
    names = dict(enumerate(names)) if isinstance(names, (list, tuple)) else names
    if single_cls:
        names = {0: names.get(0, 'class0')}
    if model is None:
        from models.experimental import attempt_load
        weight_paths = weights if isinstance(weights, (list, tuple)) else [weights]
        for path in weight_paths:
            if path is None or not Path(path).is_file():
                raise FileNotFoundError(f'Local checkpoint not found: {path}')
        device = select_device(getattr(opt, 'device', ''), batch_size=batch_size)
        model = attempt_load([str(p) for p in weight_paths], map_location=device, fuse=False)
    device = next(model.parameters()).device
    original_dtype = next(model.parameters()).dtype
    original_modes = [(module, module.training) for module in model.modules()]
    stride = int(torch.as_tensor(getattr(model, 'stride', [32])).max())
    imgsz = check_img_size(imgsz, s=stride)
    if dataloader is None:
        from utils.datasets import create_dataloader
        data = check_dataset(data, autodownload=False)
        if not data.get(task):
            raise ValueError(f'Dataset has no {task!r} split')
        dataloader = create_dataloader(data[task], imgsz, batch_size, stride,
                                      single_cls=single_cls, pad=0.5, rect=True,
                                      workers=getattr(opt, 'workers', 0))[0]
    save_dir = Path(save_dir)
    if plots or save_txt or save_hybrid or save_json:
        save_dir.mkdir(parents=True, exist_ok=True)
    if save_txt or save_hybrid:
        (save_dir / 'labels').mkdir(exist_ok=True)
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    stats, json_rows = [], []
    loss = torch.zeros(3, device=device)
    elapsed = np.zeros(2)
    seen = batches = 0
    confusion = ConfusionMatrix(nc=nc) if plots else None
    try:
        model.eval()
        model.half() if half_precision and device.type == 'cuda' else model.float()
        dtype = next(model.parameters()).dtype
        for images, targets, paths, shapes in dataloader:
            images = images.to(device=device, dtype=dtype) / 255.0
            targets = targets.to(device).clone()
            if single_cls:
                targets[:, 1] = 0
            if len(targets) and ((targets[:, 1] < 0).any() or (targets[:, 1] >= nc).any()):
                raise ValueError('Target class outside dataset nc')
            t = time_sync()
            output = model(images, augment=True) if augment else model(images)
            prediction, raw = output if isinstance(output, tuple) else (output, None)
            elapsed[0] += time_sync() - t
            if not single_cls and prediction.shape[-1] - 5 != nc:
                raise ValueError('Model class count does not match dataset nc')
            if compute_loss is not None:
                if raw is None:
                    raise ValueError('Validation loss requires raw detection outputs (disable augment)')
                loss += compute_loss([x.float() for x in raw], targets)[1][:3]
            height, width = images.shape[2:]
            targets[:, 2:] *= targets.new_tensor([width, height, width, height])
            t = time_sync()
            predictions = non_max_suppression(prediction, conf_thres, iou_thres,
                                             multi_label=True, agnostic=single_cls)
            elapsed[1] += time_sync() - t
            batches += 1
            for si, pred in enumerate(predictions):
                seen += 1
                labels = targets[targets[:, 0] == si, 1:]
                if single_cls:
                    pred[:, 5] = 0
                native = pred.clone()
                boxes = xywh2xyxy(labels[:, 1:5])
                shape = shapes[si]
                original_shape = shape[0] if shape is not None else (height, width)
                if shape is not None:
                    scale_coords(images.shape[2:], native[:, :4], original_shape, shape[1])
                    scale_coords(images.shape[2:], boxes, original_shape, shape[1])
                native_labels = torch.cat((labels[:, :1], boxes), 1)
                correct = process_batch(native, native_labels, iouv)
                stats.append((correct.cpu().numpy(), pred[:, 4].cpu().numpy(),
                              pred[:, 5].cpu().numpy(), labels[:, 0].cpu().numpy()))
                if confusion is not None:
                    confusion.process_batch(native, native_labels)
                if save_txt or save_hybrid:
                    rows = native
                    if save_hybrid and len(labels):
                        # Export autolabels without leaking GT into measured predictions.
                        rows = torch.cat((rows, torch.cat((boxes, torch.ones_like(labels[:, :1]),
                                                          labels[:, :1]), 1)))
                    gn = pred.new_tensor([original_shape[1], original_shape[0]] * 2)
                    with (save_dir / 'labels' / (Path(paths[si]).stem + '.txt')).open('w') as f:
                        for row in rows:
                            xywh = (xyxy2xywh(row[:4].view(1, 4)) / gn).view(-1).tolist()
                            values = [row[5].item(), *xywh]
                            if save_conf:
                                values.append(row[4].item())
                            f.write(' '.join(f'{v:g}' for v in values) + '\n')
                if save_json:
                    image_id = Path(paths[si]).stem
                    image_id = int(image_id) if image_id.isnumeric() else image_id
                    category_ids = coco80_to_coco91_class() if is_coco and not single_cls else list(range(nc))
                    for row in native.cpu().tolist():
                        x1, y1, x2, y2, conf, cls = row
                        json_rows.append(dict(image_id=image_id, category_id=category_ids[int(cls)],
                                              bbox=[x1, y1, x2-x1, y2-y1], score=conf))
    finally:
        model.to(dtype=original_dtype)
        for module, training in original_modes:
            module.training = training
    if not seen:
        raise ValueError('Validation dataloader yielded no images')
    tp, conf, pred_cls, target_cls = [np.concatenate(x, axis=0) for x in zip(*stats)]
    mp = mr = mf1 = map50 = mean_ap = 0.0
    maps = np.zeros(nc)
    if len(target_cls):
        _, _, p, r, f1, ap, classes = ap_per_class(
            tp, conf, pred_cls, target_cls, plot=plots and bool(tp.any()),
            save_dir=save_dir, names=names, v5_metric=v5_metric)
        mp, mr, map50, mean_ap = float(p.mean()), float(r.mean()), float(ap[:, 0].mean()), float(ap.mean())
        mf1 = float(f1.mean())
        maps[:] = mean_ap
        maps[classes] = ap.mean(1)
        if verbose:
            for i, cls in enumerate(classes):
                print(f'{names.get(int(cls), cls)}: P={p[i]:.5f} R={r[i]:.5f} AP50={ap[i, 0]:.5f} AP={ap[i].mean():.5f}')
    if confusion is not None:
        confusion.plot(save_dir=str(save_dir), names=[names.get(i, str(i)) for i in range(nc)])
    if save_json:
        (save_dir / 'predictions.json').write_text(json.dumps(json_rows), encoding='utf-8')
    inference_ms, nms_ms = (elapsed / seen * 1000).tolist()
    times = (inference_ms, nms_ms, inference_ms + nms_ms, imgsz, imgsz, batch_size)
    results = (mp, mr, map50, mean_ap, *(loss / batches).cpu().tolist())
    if quality_metrics is not None:
        quality_metrics.update(f1=mf1, num_samples=seen)
    print(f'Images={seen} P={mp:.5f} R={mr:.5f} F1={mf1:.5f} mAP@0.5={map50:.5f} mAP@0.5:0.95={mean_ap:.5f}')
    print(f'Inference={inference_ms:.3f} ms/image NMS={nms_ms:.3f} ms/image')
    return results, maps, times

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
         quality_metrics=None,
         ):
    # Resolve opt if not passed
    if opt is None:
        try:
            opt = globals().get('opt', None)
        except NameError:
            pass

    return evaluate(
        quality_metrics=quality_metrics,
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

def main():
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', choices=['train', 'val', 'test'], help='val, test or train')
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

    parser.add_argument('--workers', type=int, default=0, help='dataloader worker count')
    parser.add_argument('--no-plots', action='store_true', help='disable metric plots')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)
    
    # loading pruning params
    pruning_params_parsed = []
    if len(opt.pruning_params) > 0:
        pruning_params_parsed = ast.literal_eval(opt.pruning_params)

    return test(opt.data,
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
         opt=opt,
         is_coco=Path(opt.data).stem.startswith('coco'),
         plots=not opt.no_plots,
         save_dir=increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok),
         )

if __name__ == '__main__':
    main()
