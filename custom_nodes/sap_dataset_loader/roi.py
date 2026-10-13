"""Classic ComfyUI ROI selector backed by SAP's existing pure-Python selector.

A selected-ROI preview is for inspection only: it is not a new training image
or a segmentation mask. Original IMAGE/annotations are never modified.
"""

from __future__ import annotations

import importlib
import json
import math


class SAPROISelector:
    """Select original bbox instances and draw a private full-resolution preview."""

    RETURN_TYPES = ("SAP_ROI_SELECTION", "IMAGE", "STRING")
    RETURN_NAMES = ("selection", "roi_preview", "selection_summary")
    FUNCTION = "select"
    CATEGORY = "SAP/Annotations"
    DESCRIPTION = (
        "Select SKU bbox instances deterministically from SAP Dataset Loader. "
        "The preview shows cyan selected and red unselected boxes. "
        "No masks, effects, saves or geometry changes are performed."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "annotations": ("SAP_ANNOTATIONS",),
                "image": ("IMAGE",),
                "mode": (["random_n", "random_one", "by_index", "by_class"],),
                "seed": ("INT", {"default": 42017, "min": 0, "max": 2147483647}),
                "count": ("INT", {"default": 3, "min": 1, "max": 10000}),
                "instance_index": ("INT", {"default": 0, "min": 0, "max": 1000000}),
                "class_name": ("STRING", {"default": "", "multiline": False}),
                "include_generic_product": ("BOOLEAN", {"default": False}),
                "allow_service_targets": ("BOOLEAN", {"default": False}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return math.nan

    def select(
        self,
        annotations,
        image,
        mode: str = "random_n",
        seed: int = 42017,
        count: int = 3,
        instance_index: int = 0,
        class_name: str = "",
        include_generic_product: bool = False,
        allow_service_targets: bool = False,
    ):
        if not isinstance(class_name, str):
            raise TypeError("class_name must be text")
        if mode != "by_class" and class_name:
            raise ValueError("class_name is only valid for by_class mode")
        if mode == "by_class" and not class_name:
            raise ValueError("by_class mode requires class_name")
        if mode not in ("random_n", "random_one") and include_generic_product:
            raise ValueError("include_generic_product is only valid for random modes")
        if mode in ("random_n", "random_one") and allow_service_targets:
            raise ValueError("allow_service_targets requires explicit by_index/by_class")
        if mode != "by_index" and instance_index != 0:
            raise ValueError("instance_index is only valid in by_index mode")

        try:
            import numpy as np
            import torch
            from PIL import Image, ImageDraw
        except ImportError as exc:
            raise RuntimeError("ComfyUI requires its existing numpy, torch and Pillow") from exc

        from_module = importlib.import_module("sapipeline.annotations")
        selector = importlib.import_module("sapipeline.selection")
        if not isinstance(annotations, from_module.AnnotationDocument):
            raise TypeError("annotations must come from SAP Dataset Loader")
        if not hasattr(image, "detach"):
            raise TypeError("image must be a ComfyUI IMAGE torch tensor")
        pixels = image.detach().cpu().numpy()
        if pixels.ndim != 4 or pixels.shape[0] != 1 or pixels.shape[-1] != 3:
            raise ValueError("ROI preview requires a single RGB IMAGE [1,H,W,3]")
        h, w = pixels.shape[1:3]
        if (w, h) != (annotations.image_width, annotations.image_height):
            raise ValueError("IMAGE size does not match SAP_ANNOTATIONS coordinates")
        if not np.isfinite(pixels).all() or (pixels < 0).any() or (pixels > 1).any():
            raise ValueError("IMAGE values must be finite and normalized to [0,1]")

        selection = selector.select_rois(
            annotations,
            mode=mode,
            seed=seed,
            count=count if mode == "random_n" else 1,
            instance_index=instance_index if mode == "by_index" else None,
            class_name=class_name if mode == "by_class" else None,
            include_generic_product=include_generic_product,
            allow_service_targets=allow_service_targets,
        )

        # Work on a separate copy; never alter the source IMAGE.
        rgb = np.rint(pixels[0] * 255).astype(np.uint8)
        preview = Image.fromarray(rgb, mode="RGB")
        draw = ImageDraw.Draw(preview)
        marked = {obj.instance_index for obj in selection.targets}
        width_px = max(2, round(min(w, h) / 350))
        for obj in annotations.objects:
            x1, y1, x2, y2 = obj.bbox_xyxy_px
            box = (
                max(0, math.floor(x1)),
                max(0, math.floor(y1)),
                min(w - 1, math.ceil(x2) - 1),
                min(h - 1, math.ceil(y2) - 1),
            )
            if box[2] < box[0] or box[3] < box[1]:
                raise ValueError("Degenerate annotation after pixel rounding")
            selected = obj.instance_index in marked
            color = (0, 255, 255) if selected else (255, 60, 55)
            draw.rectangle(box, outline=color, width=width_px)
            if selected:
                draw.text((box[0], max(0, box[1] - 14)), str(obj.instance_index),
                          fill=color, stroke_width=1, stroke_fill=(0, 0, 0))

        out = np.asarray(preview, dtype=np.uint8).copy()
        result_image = torch.from_numpy(out).to(dtype=torch.float32).div_(255.0).unsqueeze(0)
        summary = json.dumps(
            {"mode": mode, "seed": seed, "selected_instance_indices": sorted(marked),
             "selected_count": len(marked), "total_annotations": len(annotations.objects),
             "preview": "cyan-selected-red-unselected-bbox-only"},
            sort_keys=True,
        )
        return (selection, result_image, summary)
