"""ComfyUI adapter: deterministic selected-bbox mask, no image edits or I/O.

ComfyUI's torch dependency is imported only when a graph actually executes.
The SAP core is supplied by the previously executed Dataset Loader node.
"""

from __future__ import annotations

import importlib
import json
import math


class SAPMaskBuilder:
    """Build a ComfyUI MASK tensor from an in-process SAP ROI selection."""

    RETURN_TYPES = ("MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("mask", "mask_overlay", "mask_summary")
    FUNCTION = "build"
    CATEGORY = "SAP/Masks"
    DESCRIPTION = (
        "Build a float32 [B,H,W] effect mask from selected bbox instances. "
        "Output overlay is inspection-only. Does not change image or labels."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "selection": ("SAP_ROI_SELECTION",),
                "image": ("IMAGE",),
                "mask_kind": (["soft_rect", "rect"],),
                "pad_px": ("INT", {"default": 0, "min": 0, "max": 500}),
                "feather_px": ("FLOAT", {"default": 16.0, "min": 0.0, "max": 512.0, "step": 1.0}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        """Do not cache against possible upstream changes."""
        return math.nan

    def build(
        self,
        selection,
        image,
        mask_kind: str = "soft_rect",
        pad_px: int = 0,
        feather_px: float = 16.0,
        strength: float = 1.0,
    ):
        try:
            import numpy as np
            import torch
        except ImportError as exc:
            raise RuntimeError("ComfyUI must provide its existing numpy and torch") from exc

        selected_api = importlib.import_module("sapipeline.selection")
        mask_api = importlib.import_module("sapipeline.masks")
        if not isinstance(selection, selected_api.SelectionResult):
            raise TypeError("selection must come from SAP ROI Selector")
        if not hasattr(image, "detach"):
            raise TypeError("image must be a ComfyUI IMAGE torch tensor")

        source = image.detach().cpu().numpy()
        if source.ndim != 4 or source.shape[0] != 1 or source.shape[-1] != 3:
            raise ValueError("Mask Builder needs one RGB IMAGE [1,H,W,3]")
        height, width = source.shape[1:3]
        if not np.isfinite(source).all() or (source < 0).any() or (source > 1).any():
            raise ValueError("IMAGE values must be finite in [0,1]")
        if source.dtype.kind != "f":
            raise TypeError("IMAGE must contain float pixel values")

        result = mask_api.build_mask(
            width,
            height,
            selection.targets,
            kind=mask_kind,
            pad_px=pad_px,
            feather_px=feather_px,
            strength=strength,
        )
        # Standard ComfyUI MASK is [B,H,W] float32, with 1 = active.
        mask_array = np.array(result.mask, dtype=np.float32, copy=True)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)

        # Overlay visualizes the true feathered mask; it does NOT apply effects.
        original = source[0].astype(np.float32, copy=False)
        alpha = mask_array[..., None] * 0.48
        cyan = np.array([0.0, 0.9, 1.0], dtype=np.float32)
        preview = original * (1.0 - alpha) + cyan * alpha
        preview_tensor = torch.from_numpy(preview.astype(np.float32)).unsqueeze(0)
        summary = json.dumps(
            {
                "mask_kind": result.kind,
                "image_size_px": [width, height],
                "target_count": len(selection.targets),
                "selected_indices": sorted(obj.instance_index for obj in selection.targets),
                "pad_px": result.pad_px,
                "feather_px": result.feather_px,
                "strength": result.strength,
                "mask_min": float(mask_array.min()),
                "mask_max": float(mask_array.max()),
                "mask_type": "bbox-effect-mask-not-segmentation",
            },
            sort_keys=True,
        )
        return (mask_tensor, preview_tensor, summary)
