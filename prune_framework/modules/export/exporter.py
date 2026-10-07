import os
import copy
from typing import Any

import numpy as np
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
    ) -> dict[str, Any]:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        # Deployment export is the final pipeline boundary, after recovery and
        # evaluation.  Materialize the live model *before* deepcopy: PyTorch's
        # parametrization API uses generated module classes, and removing a
        # parametrization only on a deepcopy can invalidate the source class.
        if materialize_masks:
            MaskManager.materialize(model)
        export_model = copy.deepcopy(model) if materialize_masks else model
        export_model.eval()
        # The legacy exporter is materially more stable with the project ONNX
        # stack than the new dynamo path for models with dynamically-pruned
        # module shapes.  Keep the requested opset explicit and validate the
        # exact file we have written below.
        torch.onnx.export(
            export_model,
            dummy_input,
            output_path,
            opset_version=opset_version,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            dynamo=False,
        )
        print(f"Exported ONNX model to: {output_path}")
        return ModelExporter.validate_onnx(export_model, dummy_input, output_path)

    @staticmethod
    def validate_onnx(
        model: nn.Module,
        dummy_input: torch.Tensor,
        output_path: str,
        *,
        rtol: float = 1e-4,
        atol: float = 1e-5,
    ) -> dict[str, Any]:
        """Validate a just-exported ONNX artifact, including ORT inference.

        Validation never assumes that every supported PyTorch model has a
        single Tensor output.  ONNX export in this framework currently uses a
        single ``output`` name, so non-Tensor outputs are reported precisely
        rather than being silently treated as a successful runtime check.
        """
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
        except Exception as exc:
            raise RuntimeError(f"ONNX structural validation failed for '{output_path}': {exc}") from exc

        try:
            import onnxruntime as ort
        except ImportError as exc:
            return {
                "onnx_exported": True,
                "onnx_validated": False,
                "onnx_validation_status": "NOT_RUN_MISSING_RESOURCE",
                "onnx_validation_reason": f"onnxruntime is required for inference validation: {exc}",
            }

        was_training = model.training
        try:
            model.eval()
            with torch.no_grad():
                expected = model(dummy_input)
        finally:
            model.train(was_training)
        if not isinstance(expected, torch.Tensor):
            return {
                "onnx_exported": True,
                "onnx_validated": False,
                "onnx_validation_status": "NOT_RUN_UNSUPPORTED_OUTPUT",
                "onnx_validation_reason": f"PyTorch output is {type(expected).__name__}, not a Tensor.",
            }
        try:
            session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
            actual = session.run(None, {session.get_inputs()[0].name: dummy_input.detach().cpu().numpy()})
            if len(actual) != 1:
                raise RuntimeError(f"expected one ONNX output, received {len(actual)}")
            actual_array = np.asarray(actual[0])
            expected_array = expected.detach().cpu().numpy()
            if actual_array.shape != expected_array.shape:
                raise RuntimeError(f"output shape mismatch: ONNX {actual_array.shape}, PyTorch {expected_array.shape}")
            if not np.allclose(actual_array, expected_array, rtol=rtol, atol=atol):
                maximum = float(np.max(np.abs(actual_array - expected_array)))
                raise RuntimeError(f"output values differ (max_abs_error={maximum:g}, rtol={rtol}, atol={atol})")
        except Exception as exc:
            raise RuntimeError(f"ONNX Runtime validation failed for '{output_path}': {exc}") from exc
        return {
            "onnx_exported": True,
            "onnx_validated": True,
            "onnx_validation_status": "PASSED",
            "onnx_validation_reason": None,
        }

    @staticmethod
    def export_checkpoint(model: nn.Module, output_path: str, ckpt: dict = None):
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        # This is called only after recovery/evaluation.  Make zero masks
        # permanent on the final model first; see ``export_onnx`` for why this
        # must precede deepcopy with PyTorch parametrizations.
        MaskManager.materialize(model)
        export_model = copy.deepcopy(model)
        if output_path.endswith(".pt") and ckpt is not None:
            # The YOLO loader intentionally prefers ``ema`` over ``model``;
            # keeping an old EMA would silently reload unpruned weights.
            payload = dict(ckpt)
            payload["model"] = export_model
            if payload.get("ema") is not None:
                payload["ema"] = export_model
            torch.save(payload, output_path)
            print(f"Saved pruned PyTorch checkpoint to: {output_path}")
        elif hasattr(model, "save_pretrained"):
            model.save_pretrained(output_path)
            print(f"Saved Hugging Face model folder to: {output_path}")
        else:
            # Structured pruning changes tensor shapes.  Preserve the exact
            # topology so the produced artifact can be loaded again.
            torch.save(
                {"format_version": 1, "model": export_model, "state_dict": export_model.state_dict()},
                output_path,
            )
            print(f"Saved reloadable PyTorch checkpoint to: {output_path}")
