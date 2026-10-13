"""SAP 6.7: deterministic elliptical/directional glare classic ComfyUI node."""

from __future__ import annotations

import importlib
import json
import math


class SAPRealisticGlare:
    """Generate seeded directional illumination only inside the original MASK."""

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("effect_image", "effect_mask", "effect_summary")
    FUNCTION = "apply"
    CATEGORY = "SAP/Effects"
    DESCRIPTION = (
        "Local, smooth elliptical or oriented directional glare, clipped to SAP MASK. "
        "Not a physically modeled glass reflection or an approved training sample."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "selection": ("SAP_ROI_SELECTION",),
            "kind": (["directional", "elliptical"],),
            "angle_deg": ("FLOAT", {"default": -25.0, "min": -180.0, "max": 180.0, "step": 1.0}),
            "radius_scale": ("FLOAT", {"default": 0.85, "min": 0.20, "max": 1.50, "step": 0.05}),
            "aspect_ratio": ("FLOAT", {"default": 0.35, "min": 0.10, "max": 1.0, "step": 0.05}),
            "falloff": ("FLOAT", {"default": 1.25, "min": 0.50, "max": 4.0, "step": 0.05}),
            "intensity": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.05}),
            "seed": ("INT", {"default": 42017, "min": 0, "max": 2147483647}),
            "center_jitter": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 0.30, "step": 0.01}),
        }}

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return math.nan

    def apply(
        self, image, mask, selection, kind="directional", angle_deg=-25.0,
        radius_scale=0.85, aspect_ratio=0.35, falloff=1.25,
        intensity=0.55, seed=42017, center_jitter=0.10,
    ):
        try:
            import numpy as np
            import torch
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("ComfyUI needs its existing torch, numpy and Pillow") from exc
        if not hasattr(image, "detach") or not hasattr(mask, "detach"):
            raise TypeError("IMAGE and MASK must be ComfyUI tensors")
        original = image.detach().cpu().numpy()
        source_mask = mask.detach().cpu().numpy()
        if original.ndim != 4 or original.shape[0] != 1 or original.shape[-1] != 3:
            raise ValueError("Expected one RGB IMAGE [1,H,W,3]")
        height, width = original.shape[1:3]
        if source_mask.shape != (1, height, width):
            raise ValueError("MASK must have shape [1,H,W] matching IMAGE")
        if original.dtype.kind != "f" or not np.isfinite(original).all():
            raise ValueError("IMAGE must be finite float RGB")
        if np.any((original < 0) | (original > 1)):
            raise ValueError("IMAGE range must be [0,1]")
        engine = importlib.import_module("sapipeline.effects")
        raw = np.rint(original[0] * 255.0).astype(np.uint8)
        effect = engine.apply_realistic_glare(
            Image.fromarray(raw, mode="RGB"), source_mask[0], selection,
            kind=kind, angle_deg=angle_deg, radius_scale=radius_scale,
            aspect_ratio=aspect_ratio, falloff=falloff, intensity=intensity,
            seed=seed, center_jitter=center_jitter,
        )
        weights = effect.effect_mask
        altered = np.asarray(effect.image, dtype=np.uint8).astype(np.float32) / 255.0
        # Original floats are preserved exactly where the real effect has no weight.
        result = np.where((weights > 0)[..., None] & (intensity > 0), altered, original[0])
        result = np.array(result, dtype=np.float32, copy=True)
        effect_mask = np.array(weights, dtype=np.float32, copy=True)
        output_image = torch.from_numpy(result).unsqueeze(0)
        output_mask = torch.from_numpy(effect_mask).unsqueeze(0)
        if getattr(image, "device", None) is not None:
            output_image = output_image.to(device=image.device)
        if getattr(mask, "device", None) is not None:
            output_mask = output_mask.to(device=mask.device)
        changed = int(np.count_nonzero(np.any(result != original[0], axis=-1)))
        summary = json.dumps({
            "schema_version": "sap-glare-v1", "kind": kind, "seed": seed,
            "angle_deg": angle_deg, "radius_scale": radius_scale,
            "aspect_ratio": aspect_ratio, "falloff": falloff,
            "intensity": intensity, "center_jitter": center_jitter,
            "image_size_px": [width, height], "changed_pixels": changed,
            "effect_mask_active_pixels": int(np.count_nonzero(weights > 0)),
            "outside_effect_mask_unchanged": True,
            "training_approved": False,
            "manual_review_required": True,
            "scope": "selected_bbox_local_not_glass_plane",
        }, sort_keys=True, allow_nan=False)
        return output_image, output_mask, summary
