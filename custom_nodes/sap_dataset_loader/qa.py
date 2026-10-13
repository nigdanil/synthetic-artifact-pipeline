"""Read-only ComfyUI adapter for the SAP technical QA gate (stage 6.6)."""

from __future__ import annotations

import importlib
import math


class SAPQAValidator:
    """Emit PASS/FAIL, machine-readable report and colored-border QA preview."""

    RETURN_TYPES = ("BOOLEAN", "STRING", "IMAGE")
    RETURN_NAMES = ("qa_pass", "qa_report", "qa_preview")
    FUNCTION = "validate"
    CATEGORY = "SAP/QA"
    DESCRIPTION = (
        "Validate original/effect IMAGE, native MASK and immutable source bbox data. "
        "PASS only establishes technical checks; visual review is mandatory. "
        "Green/red border is an inspection preview, not a training artifact."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original": ("IMAGE",),
                "modified": ("IMAGE",),
                "mask": ("MASK",),
                "annotations": ("SAP_ANNOTATIONS",),
                "selection": ("SAP_ROI_SELECTION",),
                "max_new_clipping_fraction": (
                    "FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}
                ),
                "max_mask_coverage": (
                    "FLOAT", {"default": 0.60, "min": 0.0, "max": 1.0, "step": 0.01}
                ),
                "outside_tolerance": (
                    "FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}
                ),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return math.nan

    def validate(
        self,
        original,
        modified,
        mask,
        annotations,
        selection,
        max_new_clipping_fraction: float = 0.25,
        max_mask_coverage: float = 0.60,
        outside_tolerance: float = 0.0,
    ):
        try:
            import numpy as np
            import torch
        except ImportError as exc:
            raise RuntimeError("ComfyUI must provide its existing numpy and torch") from exc
        if not all(hasattr(item, "detach") for item in (original, modified, mask)):
            raise TypeError("original, modified and mask must be ComfyUI tensors")
        source = original.detach().cpu().numpy()
        result = modified.detach().cpu().numpy()
        weights = mask.detach().cpu().numpy()
        if source.ndim != 4 or source.shape[0] != 1 or source.shape[-1] != 3:
            raise ValueError("original must be one RGB IMAGE [1,H,W,3]")
        if result.ndim != 4 or result.shape[0] != 1 or result.shape[-1] != 3:
            raise ValueError("modified must be one RGB IMAGE [1,H,W,3]")
        height, width = source.shape[1:3]
        if weights.shape != (1, height, width):
            raise ValueError("MASK must be [1,H,W] matching original resolution")
        qa_api = importlib.import_module("sapipeline.qa")
        outcome = qa_api.validate_artifact(
            source[0], result[0], weights[0], annotations, selection,
            max_new_clipping_fraction=max_new_clipping_fraction,
            max_mask_coverage=max_mask_coverage,
            outside_tolerance=outside_tolerance,
        )
        # Presentation only; never overwrite or mutate the produced artifact.
        # A blank red/green status strip also works on a completely black image.
        preview = np.clip(np.nan_to_num(result[0], nan=0.0, posinf=0.0, neginf=0.0), 0, 1)
        preview = np.array(preview, dtype=np.float32, copy=True)
        thickness = min(12, max(1, min(height, width) // 20))
        color = np.array([0.07, 0.9, 0.2] if outcome.passed else [1.0, 0.1, 0.1],
                         dtype=np.float32)
        preview[:thickness] = color
        preview[-thickness:] = color
        preview[:, :thickness] = color
        preview[:, -thickness:] = color
        tensor = torch.from_numpy(preview).unsqueeze(0)
        if getattr(modified, "device", None) is not None:
            tensor = tensor.to(device=modified.device)
        return outcome.passed, outcome.to_json(), tensor
