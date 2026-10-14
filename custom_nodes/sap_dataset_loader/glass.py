"""SAP Glass Reflection V1: perspective-plane strips for classic ComfyUI."""

from __future__ import annotations

import importlib
import json
import math


class SAPGlassReflection:
    """Create a spatial MASK and modified IMAGE across a user-defined glass plane."""

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("effect_image", "effect_mask", "effect_summary")
    FUNCTION = "apply"
    CATEGORY = "SAP/Effects"
    DESCRIPTION = (
        "Project soft reflected light bands across a normalized TL/TR/BR/BL glass plane. "
        "Does not need bbox masks; it can span product boundaries. Manual QA required."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "quad_json": ("STRING", {"default": "[[0.09,0.21],[0.98,0.21],[0.99,0.75],[0.03,0.74]]",
                                    "multiline": False,
                                    "tooltip": "Four normalized corners TL, TR, BR, BL; tune to actual glass"}),
            "angle_deg": ("FLOAT", {"default": -12.0, "min": -45.0, "max": 45.0, "step": 1.0}),
            "band_width": ("FLOAT", {"default": 0.060, "min": 0.025, "max": 0.25, "step": 0.005}),
            "band_count": ("INT", {"default": 2, "min": 1, "max": 3}),
            "band_separation": ("FLOAT", {"default": 0.23, "min": 0.05, "max": 0.40, "step": 0.01}),
            "edge_feather": ("FLOAT", {"default": 0.035, "min": 0.01, "max": 0.15, "step": 0.005}),
            "intensity": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05}),
            "seed": ("INT", {"default": 42017, "min": 0, "max": 2147483647}),
        }}

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return math.nan

    def apply(
        self, image, quad_json, angle_deg=-12.0, band_width=0.060,
        band_count=2, band_separation=0.23, edge_feather=0.035,
        intensity=0.65, seed=42017,
    ):
        try:
            import numpy as np
            import torch
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("ComfyUI requires its existing numpy, torch, Pillow") from exc
        if not hasattr(image, "detach"):
            raise TypeError("image must be a ComfyUI IMAGE tensor")
        rgb = image.detach().cpu().numpy()
        if (rgb.ndim != 4 or rgb.shape[0] != 1 or rgb.shape[-1] != 3
                or rgb.dtype.kind != "f" or not np.isfinite(rgb).all()
                or np.any((rgb < 0) | (rgb > 1))):
            raise ValueError("expected a single finite RGB IMAGE [1,H,W,3] in [0,1]")
        try:
            quad = json.loads(quad_json)
        except (TypeError, ValueError) as exc:
            raise ValueError("quad_json must be JSON: [[TLx,TLy],[TRx,TRy],[BRx,BRy],[BLx,BLy]]") from exc
        module = importlib.import_module("sapipeline.effects.glass_reflection")
        source = np.rint(rgb[0]*255).astype(np.uint8)
        result = module.apply_glass_reflection(
            Image.fromarray(source, mode="RGB"), quad_norm=quad, seed=seed,
            intensity=intensity, angle_deg=angle_deg, band_width=band_width,
            band_count=band_count, band_separation=band_separation,
            edge_feather=edge_feather,
        )
        mask = result.effect_mask.astype(np.float32, copy=True)
        changed_pixels = int(np.count_nonzero(np.any(
            np.asarray(result.image, dtype=np.uint8) != source, axis=-1,
        )))
        changed = np.asarray(result.image, dtype=np.float32) / 255.0
        # Preserve the user's float values exactly wherever the true mask is zero.
        changed = np.where(mask[..., None] > 0, changed, rgb[0]).astype(np.float32)
        output = torch.from_numpy(np.array(changed, copy=True)).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        if getattr(image, "device", None) is not None:
            output = output.to(device=image.device)
            mask_tensor = mask_tensor.to(device=image.device)
        summary = json.dumps({
            "schema_version": "sap-glass-reflection-v1",
            "kind": "projected_strips", "quad_norm": quad,
            "seed": seed, "angle_deg": angle_deg, "band_width": band_width,
            "band_count": band_count, "band_separation": band_separation,
            "edge_feather": edge_feather, "intensity": intensity,
            "image_size_px": [rgb.shape[2], rgb.shape[1]],
            "changed_pixels": changed_pixels,
            "effect_mask_active_pixels": int(np.count_nonzero(mask > 0)),
            "scope": "normalized_glass_plane_not_bbox_local",
            "outside_effect_mask_unchanged": True,
            "training_approved": False, "manual_review_required": True,
        }, allow_nan=False, sort_keys=True)
        return output, mask_tensor, summary
