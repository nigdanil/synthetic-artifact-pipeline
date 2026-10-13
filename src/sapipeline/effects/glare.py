"""Seeded, ROI-local elliptical and directional glare with smooth edges.

A bbox is an *area of permission*, never an object segmentation mask. Glare
weights are multiplied by the existing MASK. No glass-plane geometry is inferred.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image

from sapipeline.selection import SelectionResult

from .highlight import EffectError, apply_highlight

GlareKind = Literal["elliptical", "directional"]


@dataclass(frozen=True, slots=True)
class GlareResult:
    image: Image.Image
    effect_mask: np.ndarray
    kind: GlareKind
    seed: int
    intensity: float


def _range(value: float, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise EffectError(f"{label} must be finite in [{minimum},{maximum}]")
    return number


def build_glare_mask(
    base_mask: np.ndarray,
    selection: SelectionResult,
    *,
    kind: GlareKind = "directional",
    angle_deg: float = -25.0,
    radius_scale: float = 0.85,
    aspect_ratio: float = 0.35,
    falloff: float = 1.25,
    seed: int = 42017,
    center_jitter: float = 0.10,
) -> np.ndarray:
    """Return float32 [H,W] glare weights, exactly zero outside base_mask.

    Per-ROI taper to zero at all four bbox borders prevents rectangular edges;
    the envelope is rotated in the original image coordinate frame.
    """
    if kind not in ("elliptical", "directional"):
        raise EffectError("kind must be elliptical or directional")
    angle = _range(angle_deg, "angle_deg", -180.0, 180.0)
    radius = _range(radius_scale, "radius_scale", 0.20, 1.50)
    ratio = _range(aspect_ratio, "aspect_ratio", 0.10, 1.0)
    decay = _range(falloff, "falloff", 0.50, 4.0)
    jitter = _range(center_jitter, "center_jitter", 0.0, 0.30)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2147483647:
        raise EffectError("seed must be an integer in [0,2147483647]")
    if not isinstance(base_mask, np.ndarray) or base_mask.ndim != 2:
        raise EffectError("base_mask must be a [H,W] numpy array")
    if base_mask.dtype.kind != "f" or not np.isfinite(base_mask).all():
        raise EffectError("base_mask must contain finite float values")
    if (base_mask < 0).any() or (base_mask > 1).any():
        raise EffectError("base_mask must contain values in [0,1]")
    if not isinstance(selection, SelectionResult) or not selection.targets:
        raise EffectError("selection must contain at least one SAP target")
    height, width = base_mask.shape
    if height == 0 or width == 0:
        raise EffectError("base_mask cannot be empty")

    rng = np.random.default_rng(seed)
    weights = np.zeros((height, width), dtype=np.float32)
    indices: set[int] = set()
    for obj in sorted(selection.targets, key=lambda item: item.instance_index):
        if type(obj.instance_index) is not int or obj.instance_index in indices:
            raise EffectError("selection has invalid or duplicate instance_index")
        indices.add(obj.instance_index)
        try:
            x1, y1, x2, y2 = obj.bbox_xyxy_px
            valid_box = (all(type(v) in (int, float) and math.isfinite(v)
                             for v in (x1, y1, x2, y2))
                         and 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height)
        except (TypeError, ValueError):
            valid_box = False
        if not valid_box:
            raise EffectError("selection bbox lies outside image or is invalid")
        left, top, right, bottom = (math.floor(x1), math.floor(y1),
                                    math.ceil(x2), math.ceil(y2))
        bw, bh = x2 - x1, y2 - y1
        yy, xx = np.mgrid[top:bottom, left:right]
        dx0 = xx + 0.5 - (x1 + x2) / 2.0
        dy0 = yy + 0.5 - (y1 + y2) / 2.0
        # Jitter changes the center without altering source bbox/annotation.
        dx0 -= rng.uniform(-jitter, jitter) * bw
        dy0 -= rng.uniform(-jitter, jitter) * bh
        theta = math.radians(angle + rng.uniform(-1.0, 1.0) * 12.0 * jitter)
        co, si = math.cos(theta), math.sin(theta)
        along = co * dx0 + si * dy0
        across = -si * dx0 + co * dy0
        major = max(0.5, radius * bw * (0.48 if kind == "elliptical" else 0.85))
        minor = max(0.5, radius * bh * ratio * (0.48 if kind == "elliptical" else 0.33))
        squared = (along / major) ** 2 + (across / minor) ** 2
        profile = np.exp(-0.5 * squared ** decay)

        # Smoothstep(x) is zero exactly on the bbox boundary and C1 smooth.
        x_edge = np.minimum(xx + 0.5 - x1, x2 - (xx + 0.5))
        y_edge = np.minimum(yy + 0.5 - y1, y2 - (yy + 0.5))
        x_fade = np.clip(x_edge / max(1e-6, 0.18 * bw), 0, 1)
        y_fade = np.clip(y_edge / max(1e-6, 0.18 * bh), 0, 1)
        taper = (x_fade * x_fade * (3 - 2 * x_fade)
                 * y_fade * y_fade * (3 - 2 * y_fade))
        roi = np.asarray(profile * taper, dtype=np.float32)
        # Ignore numerically negligible tails, while keeping shape continuous.
        roi[roi < 1e-5] = 0.0
        weights[top:bottom, left:right] = np.maximum(
            weights[top:bottom, left:right], roi
        )
    result = weights * base_mask  # The sole allowed region is the input MASK.
    return np.asarray(result, dtype=np.float32)


def apply_realistic_glare(
    image: Image.Image,
    base_mask: np.ndarray,
    selection: SelectionResult,
    *,
    kind: GlareKind = "directional",
    angle_deg: float = -25.0,
    radius_scale: float = 0.85,
    aspect_ratio: float = 0.35,
    falloff: float = 1.25,
    intensity: float = 0.55,
    seed: int = 42017,
    center_jitter: float = 0.10,
) -> GlareResult:
    """Render ROI glare through the existing photometric soft-glow engine."""
    if not isinstance(image, Image.Image) or image.mode != "RGB":
        raise EffectError("image must be RGB PIL image")
    strength = _range(intensity, "intensity", 0.0, 1.0)
    if base_mask.shape != (image.height, image.width):
        raise EffectError("MASK must match image size")
    effect_mask = build_glare_mask(
        base_mask, selection, kind=kind, angle_deg=angle_deg,
        radius_scale=radius_scale, aspect_ratio=aspect_ratio, falloff=falloff,
        seed=seed, center_jitter=center_jitter,
    )
    result = apply_highlight(
        image, effect_mask, kind="soft_glow", intensity=strength,
        white_boost=0.48, gamma_lift=0.16,
    )
    return GlareResult(result.image, effect_mask, kind, seed, strength)
