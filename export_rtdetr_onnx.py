import argparse
import os
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn

# Wrap RT-DETR to restrict outputs to (logits, pred_boxes)
class RTDetrONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits, outputs.pred_boxes

def convert_double_to_float32(model_path):
    print("Converting double (float64) nodes/initializers to float32 in ONNX graph to prevent CPU execution issues...")
    import onnx
    from onnx import TensorProto
    import numpy as np
    
    model = onnx.load(model_path)
    graph = model.graph
    
    # 1. Convert initializers of type DOUBLE (11) to FLOAT (1)
    for init in graph.initializer:
        if init.data_type == TensorProto.DOUBLE:
            if init.raw_data:
                double_vals = np.frombuffer(init.raw_data, dtype=np.float64)
                float_vals = double_vals.astype(np.float32)
                init.raw_data = float_vals.tobytes()
            else:
                double_vals = np.array(init.double_data, dtype=np.float64)
                float_vals = double_vals.astype(np.float32)
                init.float_data.extend(float_vals)
                del init.double_data[:]
            init.data_type = TensorProto.FLOAT
            
    # 2. Convert value_info types
    for vi in graph.value_info:
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.elem_type == TensorProto.DOUBLE:
            vi.type.tensor_type.elem_type = TensorProto.FLOAT
            
    # 3. Convert node attributes
    for node in graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.DOUBLE:
                    attr.i = TensorProto.FLOAT
        elif node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == TensorProto.DOUBLE:
                    t = attr.t
                    if t.raw_data:
                        double_vals = np.frombuffer(t.raw_data, dtype=np.float64)
                        float_vals = double_vals.astype(np.float32)
                        t.raw_data = float_vals.tobytes()
                    else:
                        double_vals = np.array(t.double_data, dtype=np.float64)
                        float_vals = double_vals.astype(np.float32)
                        t.float_data.extend(float_vals)
                        del t.double_data[:]
                    t.data_type = TensorProto.FLOAT

    # Save the modified model
    onnx.save(model, model_path)
    print("Graph conversion from DOUBLE to FLOAT completed successfully.")

def export_onnx(weights_path, output_path, opset, batch_size, imgsz, dynamic):
    print(f"Loading checkpoint from {weights_path}...")
    try:
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)
        
    if isinstance(ckpt, dict) and "model_object" in ckpt:
        model = ckpt["model_object"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        model = ckpt["model"]
    else:
        model = ckpt
        
    # Ensure model is on CPU and in eval mode
    model = model.cpu().eval()
    
    # Wrap model
    onnx_model = RTDetrONNXWrapper(model)
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, imgsz, imgsz)
    
    # Configure output path
    if not output_path:
        output_path = str(Path(weights_path).with_suffix(".onnx"))
        
    print(f"Exporting to ONNX format (opset {opset}) at: {output_path}...")
    
    input_names = ["images"]
    output_names = ["logits", "pred_boxes"]
    
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            "images": {0: "batch"},
            "logits": {0: "batch"},
            "pred_boxes": {0: "batch"}
        }
        print("Dynamic batch axis enabled.")
        
    t0 = time.time()
    try:
        torch.onnx.export(
            onnx_model,
            dummy_input,
            output_path,
            verbose=False,
            opset_version=opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            training=torch.onnx.TrainingMode.EVAL,
            dynamo=False
        )
        duration = time.time() - t0
        print(f"ONNX export completed successfully in {duration:.2f} seconds.")
        
        # Convert DOUBLE types to FLOAT to resolve ONNX Runtime compatibility
        try:
            convert_double_to_float32(output_path)
        except Exception as e:
            print(f"Warning: Failed to convert double tensors to float32: {e}")
    except Exception as e:
        print(f"ONNX export failed: {e}")
        sys.exit(1)
        
    # Verify using onnx library
    try:
        import onnx
        print("Checking ONNX model structure...")
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model structure check passed!")
    except Exception as e:
        print(f"ONNX checker failed: {e}")
        
    # Try simplifying if onnxsim is installed
    try:
        import onnxsim
        print("onnx-simplifier detected. Simplifying ONNX graph...")
        model_simplified, check = onnxsim.simplify(output_path)
        if check:
            onnx.save(model_simplified, output_path)
            print("ONNX model simplified successfully!")
        else:
            print("Simplification check failed.")
    except ImportError:
        pass
    except Exception as e:
        print(f"Simplification failed: {e}")
        
    # Verify using onnxruntime
    try:
        import onnxruntime as ort
        import numpy as np
        print("Verifying ONNX model using ONNX Runtime...")
        session = ort.InferenceSession(output_path)
        
        # Prepare inputs
        x = np.random.randn(batch_size, 3, imgsz, imgsz).astype(np.float32)
        ort_inputs = {session.get_inputs()[0].name: x}
        
        # Run inference
        outputs = session.run(None, ort_inputs)
        print(f"ONNX Runtime inference success!")
        print(f"  Inputs:  {session.get_inputs()[0].name} (shape: {x.shape})")
        print(f"  Outputs:")
        for idx, out_meta in enumerate(session.get_outputs()):
            print(f"    - {out_meta.name}: shape {outputs[idx].shape}")
    except ImportError:
        print("onnxruntime not installed, skipping inference verification.")
    except Exception as e:
        print(f"ONNX Runtime verification failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Export RT-DETR PyTorch checkpoint to ONNX")
    parser.add_argument("--weights", type=str, required=True, help="Path to PyTorch (.pt) weights file")
    parser.add_argument("--output", type=str, default="", help="Path to save output ONNX model")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for dummy input")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (square)")
    parser.add_argument("--dynamic", action="store_true", default=True, help="Enable dynamic batch axis")
    
    args = parser.parse_args()
    export_onnx(
        weights_path=args.weights,
        output_path=args.output,
        opset=args.opset,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        dynamic=args.dynamic
    )

if __name__ == "__main__":
    main()
