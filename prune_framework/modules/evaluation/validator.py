import copy
import os
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


@dataclass
class ArchitectureValidationResult:
    forward_verified: bool
    output_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema_matches: Optional[bool] = None
    checkpoint_verified: bool = False
    params_before: Optional[int] = None
    params_after: Optional[int] = None
    parameter_count_verified: bool = True
    errors: List[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return self.forward_verified and self.checkpoint_verified and self.parameter_count_verified

    def describe(self) -> Dict[str, Any]:
        return asdict(self)


class ModelValidator:
    """Validates structural integrity and forward pass compatibility of pruned models."""

    @staticmethod
    def validate_forward(model: nn.Module, dummy_input: torch.Tensor) -> bool:
        model.eval()
        try:
            with torch.no_grad():
                out = model(dummy_input)
            return out is not None
        except Exception as e:
            print(f"ModelValidator Validation Failed: {e}")
            return False

    @staticmethod
    def output_schema(output: Any) -> Dict[str, Any]:
        """Return a JSON-friendly schema without assuming a detector output type."""
        if isinstance(output, torch.Tensor):
            return {"kind": "tensor", "shape": list(output.shape), "dtype": str(output.dtype)}
        if isinstance(output, dict) or hasattr(output, "items"):
            return {
                "kind": "mapping",
                "items": {str(key): ModelValidator.output_schema(value) for key, value in output.items()},
            }
        if isinstance(output, (list, tuple)):
            return {"kind": type(output).__name__, "items": [ModelValidator.output_schema(value) for value in output]}
        return {"kind": type(output).__name__}

    @staticmethod
    def validate_checkpoint_reload(model: nn.Module, dummy_input: torch.Tensor) -> bool:
        """Round-trip the state dict into a clone and verify the cloned forward."""
        path = None
        try:
            clone = copy.deepcopy(model)
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as handle:
                path = handle.name
            torch.save(model.state_dict(), path)
            try:
                state = torch.load(path, map_location=next(model.parameters()).device, weights_only=True)
            except TypeError:
                state = torch.load(path, map_location=next(model.parameters()).device)
            clone.load_state_dict(state)
            clone.eval()
            with torch.no_grad():
                return clone(dummy_input) is not None
        except Exception:
            return False
        finally:
            if path and os.path.exists(path):
                os.unlink(path)

    @staticmethod
    def validate_architecture(
        model: nn.Module,
        dummy_input: torch.Tensor,
        *,
        params_before: Optional[int] = None,
        expected_output_schema: Optional[Dict[str, Any]] = None,
        verify_checkpoint: bool = True,
    ) -> ArchitectureValidationResult:
        errors: List[str] = []
        output_schema: Dict[str, Any] = {}
        forward_verified = False
        try:
            model.eval()
            with torch.no_grad():
                output = model(dummy_input)
            if output is None:
                errors.append("Model forward returned None.")
            else:
                forward_verified = True
                output_schema = ModelValidator.output_schema(output)
        except Exception as exc:
            errors.append(f"Forward validation failed: {exc}")

        params_after = sum(parameter.numel() for parameter in model.parameters())
        parameter_count_verified = params_before is None or params_after <= params_before
        if not parameter_count_verified:
            errors.append(f"Parameter count increased from {params_before} to {params_after}.")
        schema_matches = None if expected_output_schema is None else output_schema == expected_output_schema
        if schema_matches is False:
            errors.append("Output schema differs from the expected schema.")
        checkpoint_verified = not verify_checkpoint or (
            forward_verified and ModelValidator.validate_checkpoint_reload(model, dummy_input)
        )
        if not checkpoint_verified:
            errors.append("Checkpoint state-dict round trip failed.")
        return ArchitectureValidationResult(
            forward_verified=forward_verified,
            output_schema=output_schema,
            output_schema_matches=schema_matches,
            checkpoint_verified=checkpoint_verified,
            params_before=params_before,
            params_after=params_after,
            parameter_count_verified=parameter_count_verified,
            errors=errors,
        )
