import os
import copy
import torch
import torch.nn as nn

from prune_framework.modules.model.masks import MaskManager


class ModelExporter:
    """Exports pruned PyTorch models to ONNX or PyTorch checkpoints."""

    @staticmethod
    def export_onnx(
        model: nn.Module,
        dummy_input: torch.Tensor,
        output_path: str,
        opset_version: int = 12,
        materialize_masks: bool = True,
    ):
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        # Deployment export is an explicit materialization boundary. Work on a
        # clone so the training/recovery model retains its persistent masks.
        export_model = copy.deepcopy(model) if materialize_masks else model
        if materialize_masks:
            MaskManager.materialize(export_model)
        export_model.eval()
        torch.onnx.export(
            export_model,
            dummy_input,
            output_path,
            opset_version=opset_version,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        print(f"Exported ONNX model to: {output_path}")

    @staticmethod
    def export_checkpoint(model: nn.Module, output_path: str, ckpt: dict = None):
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        if output_path.endswith(".pt") and ckpt is not None:
            ckpt["model"] = model
            torch.save(ckpt, output_path)
            print(f"Saved pruned PyTorch checkpoint to: {output_path}")
        elif hasattr(model, "save_pretrained"):
            model.save_pretrained(output_path)
            print(f"Saved Hugging Face model folder to: {output_path}")
        else:
            torch.save(model.state_dict(), output_path)
            print(f"Saved PyTorch state dict to: {output_path}")
