"""Classic ComfyUI adapter for the existing deterministic SAP highlight engine.

Reads IMAGE/MASK tensors without mutating them. Torch is supplied by ComfyUI;
this module is importable from the public Python project without torch.
"""

from __future__ import annotations

import importlib
import json
import math


class SAPHighlightEffect:
    """Apply the existing SAP soft-glow / hotspot to one RGB image and mask."""

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("effect_image", "effect_summary")
    FUNCTION = "apply"
    CATEGORY = "SAP/Effects"
    DESCRIPTION = (
        "Deterministic CPU photometric highlight under a ComfyUI MASK. "
        "No geometry changes; source IMAGE, MASK, annotations are untouched. "
        "Hotspot is an experimental/inspection effect, not accepted training data."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "effect_kind": (["soft_glow", "hotspot"],),
                "intensity": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
                "white_boost": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01}),
                "gamma_lift": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return math.nan

    def apply(
        self,
        image,
        mask,
        effect_kind: str = "soft_glow",
        intensity: float = 0.65,
        white_boost: float = 0.55,
        gamma_lift: float = 0.18,
    ):
        try:
            import numpy as np
            import torch
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("ComfyUI must provide its existing numpy, torch and Pillow") from exc

        if not hasattr(image, "detach") or not hasattr(mask, "detach"):
            raise TypeError("image and mask must be ComfyUI torch tensors")
        source = image.detach().cpu().numpy()
        weights = mask.detach().cpu().numpy()
        if source.ndim != 4 or source.shape[0] != 1 or source.shape[-1] != 3:
            raise ValueError("Highlight requires one RGB IMAGE [1,H,W,3]")
        height, width = source.shape[1:3]
        if weights.shape != (1, height, width):
            raise ValueError("MASK must be [1,H,W] and match IMAGE resolution")
        if source.dtype.kind != "f" or weights.dtype.kind != "f":
            raise TypeError("IMAGE and MASK must contain float pixel values")
        if not np.isfinite(source).all() or (source < 0).any() or (source > 1).any():
            raise ValueError("IMAGE values must be finite in [0,1]")
        if not np.isfinite(weights).all() or (weights < 0).any() or (weights > 1).any():
            raise ValueError("MASK values must be finite in [0,1]")
        if effect_kind not in ("soft_glow", "hotspot"):
            raise ValueError("effect_kind must be soft_glow or hotspot")

        effect_api = importlib.import_module("sapipeline.effects")
        rgb_u8 = np.rint(source[0] * 255.0).astype(np.uint8)
        rendered = effect_api.apply_highlight(
            Image.fromarray(rgb_u8, mode="RGB"),
            weights[0],
            kind=effect_kind,
            intensity=intensity,
            white_boost=white_boost,
            gamma_lift=gamma_lift,
        )
        result_pixels = np.asarray(rendered.image, dtype=np.uint8).astype(np.float32) / 255.0
        # The core is uint8-based; retain the exact original float values at zero mask.
        result_pixels = np.where(
            (weights[0, :, :, None] > 0.0) & (rendered.intensity > 0.0),
            result_pixels,
            source[0],
        )
        result_pixels = np.array(result_pixels, dtype=np.float32, copy=True)
        result_tensor = torch.from_numpy(result_pixels).unsqueeze(0)
        # Preserve the input device where it is not CPU. Default loaders use CPU.
        if getattr(image, "device", None) is not None:
            result_tensor = result_tensor.to(device=image.device)
        changed_pixels = int(np.count_nonzero(np.any(result_pixels != source[0], axis=-1)))
        summary = json.dumps(
            {
                "effect_kind": effect_kind,
                "intensity": rendered.intensity,
                "white_boost": rendered.white_boost,
                "gamma_lift": rendered.gamma_lift,
                "image_size_px": [width, height],
                "mask_active_pixels": int(np.count_nonzero(weights[0] > 0)),
                "changed_pixels": changed_pixels,
                "zero_mask_pixels_preserved": True,
                "training_approval": "pending_manual_review",
            },
            sort_keys=True,
        )
        return (result_tensor, summary)
