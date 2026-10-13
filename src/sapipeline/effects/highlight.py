"""Deterministic local highlight compositing on top of ROI-derived masks.

The engine is intentionally simple and CPU-only. It does not change geometry,
re-run detection, or use diffusion/ComfyUI. It only applies a repeatable local
photometric highlight to the encoded image grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image

EffectKind = Literal["soft_glow", "hotspot"]


class EffectError(ValueError):
    """Reject invalid image/mask/effect settings."""


@dataclass(frozen=True, slots=True)
class HighlightResult:
    image: Image.Image
    kind: EffectKind
    intensity: float
    white_boost: float
    gamma_lift: float


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EffectError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise EffectError(f"{name} must be finite")
    return result



def _validate_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if not isinstance(mask, np.ndarray):
        raise EffectError("mask must be a numpy ndarray")
    if mask.ndim != 2:
        raise EffectError("mask must be 2-dimensional [H,W]")
    if mask.shape != (height, width):
        raise EffectError(f"mask shape {mask.shape} != image size {(height, width)}")
    if not np.issubdtype(mask.dtype, np.number):
        raise EffectError("mask dtype must be numeric")
    if not np.isfinite(mask).all():
        raise EffectError("mask must contain only finite values")
    clipped = np.clip(mask.astype(np.float32, copy=False), 0.0, 1.0)
    return clipped



def apply_highlight(
    image: Image.Image,
    mask: np.ndarray,
    *,
    kind: EffectKind = "soft_glow",
    intensity: float = 0.75,
    white_boost: float = 0.65,
    gamma_lift: float = 0.18,
) -> HighlightResult:
    """Apply a deterministic local highlight based on a float32 mask in [0,1].

    The output stays in RGB uint8 and preserves the exact input resolution.
    Zero-valued mask pixels are preserved bit-for-bit.
    """
    if kind not in ("soft_glow", "hotspot"):
        raise EffectError("kind must be soft_glow or hotspot")
    if image.mode != "RGB":
        image = image.convert("RGB")
    strength = _finite_number(intensity, "intensity")
    if not 0 <= strength <= 1:
        raise EffectError("intensity must be between 0 and 1")
    boost = _finite_number(white_boost, "white_boost")
    if not 0 <= boost <= 1:
        raise EffectError("white_boost must be between 0 and 1")
    lift = _finite_number(gamma_lift, "gamma_lift")
    if not 0 <= lift <= 1:
        raise EffectError("gamma_lift must be between 0 and 1")

    width, height = image.size
    weights = _validate_mask(mask, width, height)
    pixels = np.asarray(image, dtype=np.uint8)
    if weights.max(initial=0.0) == 0.0 or strength == 0.0:
        return HighlightResult(image.copy(), kind, strength, boost, lift)

    source = pixels.astype(np.float32) / 255.0
    alpha = (weights * strength).astype(np.float32)
    if kind == "soft_glow":
        whiten = alpha * boost
        gamma = np.maximum(1e-6, 1.0 - alpha * lift)
        bright = np.power(source, gamma[..., None])
        out = bright + (1.0 - bright) * whiten[..., None]
    else:  # hotspot
        core = np.clip(np.power(alpha, 0.75) * (0.85 + 0.15 * boost), 0.0, 1.0)
        soft = np.clip(alpha * (0.35 + 0.25 * lift), 0.0, 1.0)
        out = source + (1.0 - source) * core[..., None]
        out = np.clip(out * (1.0 + soft[..., None]), 0.0, 1.0)

    # Preserve exact source pixels where the mask is zero.
    out_u8 = np.rint(np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
    zero_mask = alpha <= 0.0
    out_u8[zero_mask] = pixels[zero_mask]
    return HighlightResult(Image.fromarray(out_u8, mode="RGB"), kind, strength, boost, lift)
