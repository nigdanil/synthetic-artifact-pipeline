"""Versioned, strict geometry contract for manually marked surfaces.

All coordinates refer to the encoded, unrotated image pixel grid.
Surface annotations never replace or modify product SEG/YOLO labels.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

SCHEMA_VERSION = "sap-surfaces-v1"
PIXEL_POLICY = "encoded-pixels-no-exif-transpose"
MATERIALS = frozenset({"glass", "metal", "plastic", "wood", "unknown"})
SHA_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
ID_PATTERN = re.compile(r"surface_[A-Za-z0-9_-]{1,56}\Z")


class SurfaceError(ValueError):
    """Invalid surface document or geometry."""


def _require_keys(value: Any, keys: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise SurfaceError(f"{label}: invalid or unexpected fields")


def _quad(value: Any, width: int, height: int) -> list[list[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise SurfaceError("quad_norm: exactly four corners required")

    points = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise SurfaceError("quad_norm: each corner must be [x,y]")
        if any(type(v) not in (int, float) for v in point):
            raise SurfaceError("quad_norm: coordinates must be numeric")
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in point):
            raise SurfaceError("quad_norm: coordinates outside [0,1]")
        points.append([float(point[0]), float(point[1])])

    # Order: top-left, top-right, bottom-right, bottom-left.
    a, b, c, d = points
    if not (a[0] < b[0] and d[0] < c[0]
            and a[1] < d[1] and b[1] < c[1]):
        raise SurfaceError("quad_norm: expected TL,TR,BR,BL order")

    crosses = []
    for i in range(4):
        p = points[i]
        q = points[(i + 1) % 4]
        r = points[(i + 2) % 4]
        cross = ((q[0] - p[0]) * (r[1] - q[1])
                 - (q[1] - p[1]) * (r[0] - q[0]))
        crosses.append(cross)

    if any(cross <= 1e-10 for cross in crosses):
        raise SurfaceError("quad_norm: non-convex or degenerate polygon")

    twice_area = sum(
        points[i][0] * points[(i + 1) % 4][1]
        - points[(i + 1) % 4][0] * points[i][1]
        for i in range(4)
    )

    if twice_area <= 0 or twice_area * width * height / 2 < 16:
        raise SurfaceError("quad_norm: surface area is too small")

    return points


def validate_document(
    document: Any,
    *,
    expected_sha256: str | None = None,
    expected_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Validate and normalize a surface document without mutating input."""
    _require_keys(
        document,
        {
            "schema_version", "source", "surfaces",
            "manual_review_required", "training_approved",
        },
        "document",
    )

    if document["schema_version"] != SCHEMA_VERSION:
        raise SurfaceError("unsupported surface schema version")
    if document["manual_review_required"] is not True:
        raise SurfaceError("manual review must remain required")
    if document["training_approved"] is not False:
        raise SurfaceError("surface markup cannot approve training data")

    source = document["source"]
    _require_keys(
        source,
        {"source_id", "image_sha256", "width", "height", "pixel_policy"},
        "source",
    )

    sha = source["image_sha256"]
    if not isinstance(sha, str) or not SHA_PATTERN.fullmatch(sha):
        raise SurfaceError("invalid source SHA-256")
    if source["source_id"] != f"src_{sha[:16]}":
        raise SurfaceError("source ID differs from source SHA-256")

    width, height = source["width"], source["height"]
    if (type(width) is not int or type(height) is not int
            or width < 16 or height < 16):
        raise SurfaceError("invalid source image dimensions")
    if source["pixel_policy"] != PIXEL_POLICY:
        raise SurfaceError("incompatible image orientation policy")
    if expected_sha256 is not None and sha != expected_sha256:
        raise SurfaceError("source SHA-256 mismatch")
    if expected_size is not None and (width, height) != expected_size:
        raise SurfaceError("source image dimensions mismatch")

    raw_surfaces = document["surfaces"]
    if not isinstance(raw_surfaces, list) or len(raw_surfaces) > 32:
        raise SurfaceError("surfaces must be a list of at most 32 entries")

    seen = set()
    surfaces = []

    for item in raw_surfaces:
        _require_keys(
            item,
            {"surface_id", "material", "quad_norm", "reviewed", "note"},
            "surface",
        )
        identifier = item["surface_id"]
        if not isinstance(identifier, str) or not ID_PATTERN.fullmatch(identifier):
            raise SurfaceError("invalid surface_id")
        if identifier in seen:
            raise SurfaceError("duplicate surface_id")
        seen.add(identifier)

        material = item["material"]
        if not isinstance(material, str) or material not in MATERIALS:
            raise SurfaceError("unsupported surface material")
        if type(item["reviewed"]) is not bool:
            raise SurfaceError("reviewed must be boolean")
        if material == "unknown" and item["reviewed"]:
            raise SurfaceError("unknown material cannot be confirmed")
        note = item["note"]
        if not isinstance(note, str) or len(note) > 500:
            raise SurfaceError("invalid surface note")

        surfaces.append({
            "surface_id": identifier,
            "material": material,
            "quad_norm": _quad(item["quad_norm"], width, height),
            "reviewed": item["reviewed"],
            "note": note,
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "source": dict(source),
        "surfaces": surfaces,
        "manual_review_required": True,
        "training_approved": False,
    }


def new_document(sha256: str, width: int, height: int) -> dict[str, Any]:
    """Create an empty document bound to a source image."""
    document = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "source_id": f"src_{sha256[:16]}",
            "image_sha256": sha256,
            "width": width,
            "height": height,
            "pixel_policy": PIXEL_POLICY,
        },
        "surfaces": [],
        "manual_review_required": True,
        "training_approved": False,
    }
    return validate_document(document)


def dumps_document(document: Any) -> str:
    """Produce deterministic UTF-8 JSON with a trailing newline."""
    checked = validate_document(document)
    return json.dumps(
        checked, ensure_ascii=False, indent=2,
        sort_keys=True, allow_nan=False,
    ) + "\n"


def loads_document(content: str | bytes) -> dict[str, Any]:
    """Read a surface JSON document, rejecting duplicate JSON keys."""
    if isinstance(content, bytes):
        try:
            content = content.decode("utf-8-sig")
        except UnicodeError as exc:
            raise SurfaceError("invalid JSON encoding") from exc

    if not isinstance(content, str) or len(content) > 2_000_000:
        raise SurfaceError("invalid JSON size or type")

    def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise SurfaceError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        raw = json.loads(content, object_pairs_hook=unique_keys)
    except (ValueError, TypeError) as exc:
        raise SurfaceError(f"invalid surface JSON: {exc}") from exc

    return validate_document(raw)
