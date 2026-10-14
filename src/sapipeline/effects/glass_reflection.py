"""Deterministic 2.5D glass-plane strip reflections (not true 3D illumination).

The user defines a four-corner normalized glass quadrilateral. An inverse planar
homography maps each pixel into that plane. All effect weights are zero outside
that polygon. Source pixels and SKU annotations are never modified in place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

DEFAULT_GLASS_QUAD = ((0.09, 0.21), (0.98, 0.21), (0.99, 0.75), (0.03, 0.74))


class GlassReflectionError(ValueError):
    """Invalid geometry/photometric parameters; fail before applying an effect."""


@dataclass(frozen=True, slots=True)
class GlassReflectionResult:
    image: Image.Image
    effect_mask: np.ndarray  # float32 [H,W], actual support of the effect


def _number(value: object, label: str, lo: float, hi: float) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi:
        raise GlassReflectionError(f"{label} must be finite in [{lo},{hi}]")
    return float(value)


def _quad(value: object) -> np.ndarray:
    try:
        pts = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise GlassReflectionError("quad_norm must be four numeric [x,y] points") from exc
    if pts.shape != (4, 2) or not np.isfinite(pts).all() or np.any((pts < 0) | (pts > 1)):
        raise GlassReflectionError("quad_norm must be four finite normalized [x,y] corners")
    # Require a convex simple polygon with ordered TL, TR, BR, BL corners.
    directions = np.roll(pts, -1, axis=0) - pts
    next_edges = np.roll(directions, -1, axis=0)
    crosses = directions[:, 0]*next_edges[:, 1] - directions[:, 1]*next_edges[:, 0]
    area = 0.5 * abs(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1)
                            - pts[:, 1] * np.roll(pts[:, 0], -1)))
    if np.any(crosses <= 1e-6) or area < 0.04:
        raise GlassReflectionError("quad_norm must be a sufficiently large convex clockwise-on-screen quadrilateral")
    if not (pts[0, 0] < pts[1, 0] and pts[3, 0] < pts[2, 0]
            and pts[0, 1] < pts[3, 1] and pts[1, 1] < pts[2, 1]):
        raise GlassReflectionError("quad_norm corner order must be TL, TR, BR, BL")
    return pts


def _homography(pts: np.ndarray) -> np.ndarray:
    # Map unit-plane (u,v) -> normalized image (x,y), invert for raster sampling.
    corners = ((0., 0.), (1., 0.), (1., 1.), (0., 1.))
    rows: list[list[float]] = []
    vals: list[float] = []
    for (u, v), (x, y) in zip(corners, pts, strict=True):
        rows.append([u, v, 1., 0., 0., 0., -x*u, -x*v])
        rows.append([0., 0., 0., u, v, 1., -y*u, -y*v])
        vals.extend((float(x), float(y)))
    try:
        h = np.linalg.solve(np.asarray(rows), np.asarray(vals))
        hmat = np.array([[h[0], h[1], h[2]], [h[3], h[4], h[5]],
                         [h[6], h[7], 1.]])
        return np.linalg.inv(hmat)
    except np.linalg.LinAlgError as exc:
        raise GlassReflectionError("degenerate glass-plane homography") from exc


def apply_glass_reflection(
    image: Image.Image, *, quad_norm=DEFAULT_GLASS_QUAD, seed: int = 42017,
    intensity: float = 0.65, angle_deg: float = -12.0,
    band_width: float = 0.060, band_count: int = 2,
    band_separation: float = 0.23, edge_feather: float = 0.035,
) -> GlassReflectionResult:
    """Overlay soft projected light strips onto a specified glass quadrilateral.

    This does not guess which pixels are truly glass, nor model glass refraction.
    Mask is an effect mask, not a SKU segmentation ground truth.
    """
    if not isinstance(image, Image.Image) or image.mode != "RGB":
        raise GlassReflectionError("source must be a PIL RGB image")
    if type(seed) is not int or not 0 <= seed <= 2147483647:
        raise GlassReflectionError("seed must be a nonnegative int <= 2147483647")
    if type(band_count) is not int or not 1 <= band_count <= 3:
        raise GlassReflectionError("band_count must be 1..3")
    intensity = _number(intensity, "intensity", 0.0, 1.0)
    angle_deg = _number(angle_deg, "angle_deg", -45., 45.)
    band_width = _number(band_width, "band_width", 0.025, 0.25)
    band_separation = _number(band_separation, "band_separation", 0.05, 0.40)
    edge_feather = _number(edge_feather, "edge_feather", 0.01, 0.15)
    pts = _quad(quad_norm)
    w, h = image.size
    if w < 16 or h < 16:
        raise GlassReflectionError("image must be at least 16x16")

    inverse = _homography(pts)
    y, x = np.mgrid[0:h, 0:w]
    # Compare pixel CENTERS in the normalized JPEG pixel grid, no EXIF transpose.
    xx = (x + 0.5) / w
    yy = (y + 0.5) / h
    denominator = inverse[2, 0] * xx + inverse[2, 1] * yy + inverse[2, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (inverse[0, 0] * xx + inverse[0, 1] * yy + inverse[0, 2]) / denominator
        v = (inverse[1, 0] * xx + inverse[1, 1] * yy + inverse[1, 2]) / denominator
    inside = (np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u <= 1)
              & (v >= 0) & (v <= 1))
    # Prevent NaN leakage in subsequently multiplied vectors.
    u = np.where(inside, u, 0.)
    v = np.where(inside, v, 0.)
    edge_distance = np.minimum.reduce((u, 1-u, v, 1-v))
    t = np.clip(edge_distance / edge_feather, 0., 1.)
    plane_fade = t*t*(3.-2.*t)

    rng = np.random.default_rng(seed)
    # Slope in plane coordinates; bands cross multiple products and unlabelled gaps.
    slope = math.tan(math.radians(angle_deg)) * 0.55
    weight = np.zeros((h, w), dtype=np.float64)
    for i in range(band_count):
        center = 0.59 + (i-(band_count-1)/2.)*band_separation
        center += float(rng.uniform(-0.017, 0.017))
        phase = float(rng.uniform(0, 2*math.pi))
        dist = v - center - slope * (u - 0.5)
        gaussian = np.exp(-0.5 * (dist / band_width)**2)
        along = np.exp(-0.5 * ((u - 0.5)/0.43)**8)
        texture = 0.94 + 0.06*np.cos(2*math.pi*2.5*u + phase)
        strip = gaussian * along * texture
        weight = 1.0 - (1.0-weight)*(1.0-strip)
    weight = np.clip(weight * plane_fade * inside, 0., 1.)
    # 16-bit PNG export rounds tiny values to zero; use the exact same support.
    weight[weight < 0.5/65535] = 0.0
    weight = weight.astype(np.float32)
    orig = np.asarray(image, dtype=np.uint8)
    if intensity == 0.0:
        return GlassReflectionResult(image.copy(), weight)
    base = orig.astype(np.float32)/255.0
    composite = base + (1.0-base)*weight[..., None]*intensity
    rgb = np.rint(np.clip(composite, 0., 1.)*255).astype(np.uint8)
    return GlassReflectionResult(Image.fromarray(rgb, mode="RGB"), weight)
