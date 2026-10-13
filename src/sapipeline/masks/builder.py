"""Deterministic masks on the encoded, unrotated image coordinate grid.

This is an *effect mask*, not object segmentation ground truth. The entire
pixel area outside padded target bounding boxes stays exactly zero.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from sapipeline.annotations import ObjectAnnotation

MaskKind = Literal["rect", "soft_rect"]


class MaskError(ValueError):
    """Reject malformed geometry or settings instead of producing a bad mask."""


@dataclass(frozen=True, slots=True)
class MaskBuildResult:
    mask: np.ndarray  # float32 [height,width], guaranteed in [0,1]
    padded_boxes_xyxy_px: tuple[tuple[float, float, float, float], ...]
    kind: MaskKind
    pad_px: int
    feather_px: float
    strength: float


def _valid_image_dimension(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise MaskError(f"{name} must be a positive integer")


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MaskError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise MaskError(f"{name} must be finite")
    return result


def build_mask(
    image_width: int,
    image_height: int,
    targets: Sequence[ObjectAnnotation],
    *,
    kind: MaskKind = "soft_rect",
    pad_px: int = 0,
    feather_px: float = 16.0,
    strength: float = 1.0,
) -> MaskBuildResult:
    """Build a [H,W] float32 mask by taking the maximum across selected boxes.

    - Pixels are sampled at their centers: (column+0.5, row+0.5).
    - Padding expands a box and is explicitly clipped to image bounds.
    - Soft feather is *inside* the padded bbox, not on neighboring objects.
    - Feather is capped at half the shorter bbox side to keep the center strong.
    - Zero mask values are guaranteed outside every padded target rectangle.
    """
    _valid_image_dimension(image_width, "image_width")
    _valid_image_dimension(image_height, "image_height")
    if kind not in ("rect", "soft_rect"):
        raise MaskError("kind must be rect or soft_rect")
    if type(pad_px) is not int or pad_px < 0:
        raise MaskError("pad_px must be a non-negative integer")
    feather = _finite_number(feather_px, "feather_px")
    intensity = _finite_number(strength, "strength")
    if feather < 0:
        raise MaskError("feather_px must be non-negative")
    if not 0 < intensity <= 1:
        raise MaskError("strength must be > 0 and <= 1")
    if not targets:
        raise MaskError("At least one target bbox is required")
    if len({obj.instance_index for obj in targets}) != len(targets):
        raise MaskError("Selected instance_index values must be unique")

    result = np.zeros((image_height, image_width), dtype=np.float32)
    boxes: list[tuple[float, float, float, float]] = []
    for obj in targets:
        try:
            a, b, c, d = (_finite_number(v, "bbox coordinate") for v in obj.bbox_xyxy_px)
        except (TypeError, ValueError) as exc:
            raise MaskError(f"Invalid bounding box for instance {obj.instance_index}") from exc
        if not (0 <= a < c <= image_width and 0 <= b < d <= image_height):
            raise MaskError(f"Out-of-bounds bbox for instance {obj.instance_index}")

        x1 = max(0.0, a - pad_px)
        y1 = max(0.0, b - pad_px)
        x2 = min(float(image_width), c + pad_px)
        y2 = min(float(image_height), d + pad_px)
        boxes.append((x1, y1, x2, y2))
        # A pixel participates only if its CENTER falls in the half-open bbox.
        col_begin = max(0, math.ceil(x1 - 0.5))
        col_end = min(image_width, math.ceil(x2 - 0.5))
        row_begin = max(0, math.ceil(y1 - 0.5))
        row_end = min(image_height, math.ceil(y2 - 0.5))
        if row_end <= row_begin or col_end <= col_begin:
            raise MaskError(f"Bbox too small to cover any pixel: instance {obj.instance_index}")

        section = result[row_begin:row_end, col_begin:col_end]
        if kind == "rect" or feather == 0:
            np.maximum(section, intensity, out=section)
            continue
        span = min(feather, (x2 - x1) / 2, (y2 - y1) / 2)
        # Allocate only a small ROI rather than multiple full-resolution masks.
        xx = np.arange(col_begin, col_end, dtype=np.float32)[None, :] + 0.5
        yy = np.arange(row_begin, row_end, dtype=np.float32)[:, None] + 0.5
        dx = np.minimum(xx - x1, x2 - xx)
        dy = np.minimum(yy - y1, y2 - yy)
        norm_x = np.clip(dx / span, 0.0, 1.0)
        norm_y = np.clip(dy / span, 0.0, 1.0)
        weight = np.minimum(norm_x, norm_y)
        weight = weight * weight * (3.0 - 2.0 * weight)  # smoothstep
        np.maximum(section, weight * intensity, out=section)

    return MaskBuildResult(
        mask=result,
        padded_boxes_xyxy_px=tuple(boxes),
        kind=kind,
        pad_px=pad_px,
        feather_px=feather,
        strength=intensity,
    )
