import os
import sys
import argparse
import torch
import torch.nn as nn
import torch_pruning as tp
from pathlib import Path

# Add project root to sys.path
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.experimental import attempt_load
from utils.general import check_dataset
from utils.torch_utils import select_device, model_info

class TaylorYOLOPruner:
    """
    Implements First-Order Taylor Expansion (Gradient-based) Pruning for YOLOv5.
    Evaluates channel/layer importance based on I_i = |(dL / dW_i) * W_i|.
    """
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = select_device(device)
        print(f"[Taylor Pruner] Loading YOLOv5 model from: {model_path}")
        self.model = attempt_load(model_path, map_location=self.device)
        self.model.eval()

    def calibrate_gradients(self, dummy_input: torch.Tensor):
        """
        Perform forward and backward passes to compute gradients for Taylor Importance.
        """
        self.model.train()
        for p in self.model.parameters():
            p.requires_grad = True
        self.model.zero_grad()
        print("[Taylor Pruner] Running forward pass to accumulate Taylor gradients...")
        
        # Forward pass
        output = self.model(dummy_input)
        
        # Calculate loss proxy: sum of output predictions
        if isinstance(output, tuple) or isinstance(output, list):
            loss = sum(out.sum() for out in output if isinstance(out, torch.Tensor))
        else:
            loss = output.sum()
            
        # Backward pass to calculate gradients
        loss.backward()
        print("[Taylor Pruner] Backward pass complete. Gradients computed for all layers.")

    def compute_layer_importance(self, dummy_input: torch.Tensor):
        """
        Compute Taylor Expansion importance score for every Conv layer in YOLOv5.
        """
        self.calibrate_gradients(dummy_input)
        
        taylor_scores = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and module.weight.grad is not None:
                # First-Order Taylor Expansion: | grad * weight |
                importance = (module.weight.grad * module.weight).abs().sum().item()
                taylor_scores[name] = importance
                
        # Sort layers by Taylor importance (ascending: least important first)
        sorted_scores = sorted(taylor_scores.items(), key=lambda item: item[1])
        
        print("\n" + "=" * 65)
        print(" TAYLOR EXPANSION (GRADIENT) LAYER IMPORTANCE RANKING")
        print("=" * 65)
        print(f"{'Layer Name':<45} | {'Taylor Score':<15}")
        print("-" * 65)
        for name, score in sorted_scores[:15]:
            print(f"{name:<45} | {score:<15.6f}")
        print("=" * 65 + "\n")
        
        return sorted_scores

    def prune(self, pruning_ratio: float = 0.2, save_path: str = "weights/yolov5s-taylor-pruned.pt"):
        """
        Prune YOLOv5 channels using Taylor Importance and Torch-Pruning Dependency Graph.
        """
        dummy_input = torch.randn(1, 3, 640, 640, device=self.device)
        
        # 1. Rank importance
        self.compute_layer_importance(dummy_input)
        
        # 2. Setup Torch-Pruning Taylor Importance & MetaPruner
        importance = tp.importance.TaylorImportance()
        
        # Ignored layers: Detection head anchors
        ignored_layers = []
        for m in self.model.modules():
            if hasattr(m, 'anchor_grid') or type(m).__name__ in ['Detect', 'IDetect']:
                ignored_layers.append(m)

        pruner = tp.pruner.MetaPruner(
            self.model,
            example_inputs=dummy_input,
            importance=importance,
            pruning_ratio=pruning_ratio,
            ignored_layers=ignored_layers,
        )

        params_before = sum(p.numel() for p in self.model.parameters())
        print(f"[Taylor Pruner] Parameters before pruning: {params_before:,}")

        # Re-compute Taylor gradients right before pruner step
        self.model.zero_grad()
        self.calibrate_gradients(dummy_input)

        # Execute Pruning step
        pruner.step()

        params_after = sum(p.numel() for p in self.model.parameters())
        reduction = (1 - params_after / params_before) * 100
        
        print("\n" + "=" * 60)
        print(" TAYLOR PRUNING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Parameters Before: {params_before:,}")
        print(f"Parameters After:  {params_after:,}")
        print(f"Parameter Reduction: {reduction:.2f}%")
        print("=" * 60)

        # Save Pruned Model
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({'model': self.model}, save_path)
        print(f"[Taylor Pruner] Saved pruned model weights to: {save_path}\n")
        return self.model


def parse_args():
    parser = argparse.ArgumentParser(description="Taylor Expansion (Gradient) Pruning for YOLOv5")
    parser.add_argument("--weights", type=str, default="weights/yolov5s.pt", help="Path to input YOLOv5 weights")
    parser.add_argument("--ratio", type=float, default=0.2, help="Pruning ratio (0.1 to 0.5)")
    parser.add_argument("--output", type=str, default="weights/yolov5s-taylor-pruned.pt", help="Output path for pruned model")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pruner = TaylorYOLOPruner(model_path=args.weights, device=args.device)
    pruner.prune(pruning_ratio=args.ratio, save_path=args.output)
