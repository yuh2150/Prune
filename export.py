import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from models.experimental import attempt_load
from models.yolo import Detect
from utils.torch_utils import select_device


def export_onnx(model, im, file, opset, train, dynamic, simplify):
    # ONNX export
    try:
        import onnx
        print(f'\nStarting ONNX export with opset {opset}...')
        f = str(file.with_suffix('.onnx'))

        # Prepare model
        # ONNX export requires eval mode, no gradients
        model.eval()
        for k, m in model.named_modules():
            if isinstance(m, Detect):
                m.inplace = False
                m.onnx_dynamic = dynamic
                m.export = True  # some versions use m.export to change behavior

        # Dry run
        y = model(im)
        
        # Output info
        if isinstance(y, tuple):
            print(f"Model output shape (tuple of {len(y)} elements):")
            for i, x in enumerate(y):
                if isinstance(x, torch.Tensor):
                    print(f"  [{i}]: {list(x.shape)}")
                elif isinstance(x, list):
                    print(f"  [{i}]: list of length {len(x)} with shapes {[list(t.shape) for t in x]}")
        else:
            print(f"Model output shape: {list(y.shape)}")

        # Export
        torch.onnx.export(
            model,
            im,
            f,
            verbose=False,
            opset_version=opset,
            dynamo=False,
            training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['output'] if not isinstance(y, tuple) else ['output'] + [f'output_head_{i}' for i in range(len(y[1]))],
            dynamic_axes={
                'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
                'output': {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
            } if dynamic else None
        )

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx structure

        # Simplify
        if simplify:
            try:
                import onnxsim
                print(f'Simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(
                    model_onnx,
                    dynamic_input_shape=dynamic,
                    input_shapes={'images': list(im.shape)} if dynamic else None
                )
                assert check, 'assert simplify OK'
                onnx.save(model_onnx, f)
            except Exception as e:
                print(f'Simplifier failure: {e}')

        print(f'ONNX export success, saved as {f}')
        return f
    except Exception as e:
        print(f'ONNX export failure: {e}')
        raise e


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv5 Pruned Model ONNX Exporter')
    parser.add_argument('--weights', type=str, default='yolov5s-pruned.pt', help='weights path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='image size h,w')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--opset', type=int, default=12, help='ONNX opset version')
    parser.add_argument('--train', action='store_true', help='export target model in training mode')
    parser.add_argument('--dynamic', action='store_true', help='export with dynamic axes')
    parser.add_argument('--simplify', action='store_true', help='simplify ONNX model')
    args = parser.parse_args()
    
    # format imgsz to list [h, w]
    if len(args.imgsz) == 1:
        args.imgsz = [args.imgsz[0], args.imgsz[0]]
    elif len(args.imgsz) > 2:
        args.imgsz = args.imgsz[:2]
        
    return args


def main():
    args = parse_args()
    t = time.time()
    
    # Device
    device = select_device(args.device)
    
    # Load PyTorch model
    print(f'Loading model from {args.weights}...')
    model = attempt_load(args.weights, map_location=device)  # load FP32 model
    
    # Input tensor
    im = torch.zeros(args.batch_size, 3, *args.imgsz).to(device)
    
    # Export ONNX
    export_onnx(model, im, Path(args.weights), args.opset, args.train, args.dynamic, args.simplify)
    
    print(f'Done in {time.time() - t:.2f}s.')


if __name__ == '__main__':
    main()
