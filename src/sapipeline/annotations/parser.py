"""Strict, geometry-preserving SEG / YOLO bounding-box adapters.

Coordinates use the *encoded*, unrotated pixel grid, with half-open XYXY bounds.
No image transforms, class-ID inference, clipping, or loss of SEG source fields.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


class AnnotationError(ValueError):
    """Structured rejection of invalid annotations; never silently skip a box."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True, slots=True)
class ObjectAnnotation:
    instance_index: int
    class_name: str
    class_id: int | None
    sku_id: str | None
    bbox_xyxy_px: tuple[int | float, int | float, int | float, int | float]
    raw_fields: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AnnotationDocument:
    image_width: int
    image_height: int
    image_filename: str | None
    source_format: Literal["seg", "yolo"]
    objects: tuple[ObjectAnnotation, ...]
    source_seg: dict[str, Any] | None = None


def _image_size(width: int, height: int) -> None:
    if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0
           for n in (width, height)):
        raise AnnotationError("schema_mismatch", "image width/height must be positive integers")


def _number(value: Any, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnnotationError("invalid_bbox", f"{label} must be a finite number")
    if not math.isfinite(value):
        raise AnnotationError("invalid_bbox", f"{label} must be finite")
    return value


def _bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
    label: str,
) -> tuple[int | float, int | float, int | float, int | float]:
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise AnnotationError(
            "invalid_bbox", f"{label}: bounds {(x1, y1, x2, y2)} outside {width}x{height}"
        )
    return (x1, y1, x2, y2)


def _decode_json(raw: bytes | str, label: str) -> Any:
    try:
        source = raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw
        return json.loads(source)
    except (UnicodeError, json.JSONDecodeError, TypeError) as exc:
        raise AnnotationError("invalid_json", f"{label}: {exc}") from exc


def parse_seg(
    raw: bytes | str,
    *,
    image_width: int,
    image_height: int,
    image_filename: str | None = None,
) -> AnnotationDocument:
    """Read vendor SEG JSON preserving fields, class spelling and instance order.

    No EXIF transpose is applied: dimensions must be the raw encoded JPEG size.
    """
    _image_size(image_width, image_height)
    document = _decode_json(raw, "SEG")
    if (
        not isinstance(document, dict)
        or not isinstance(document.get("filename"), str)
        or not document["filename"]
        or not isinstance(document.get("selectedList"), list)
    ):
        raise AnnotationError("schema_mismatch", "SEG requires filename and selectedList list")
    if image_filename is not None and document["filename"] != image_filename:
        raise AnnotationError(
            "schema_mismatch",
            f"SEG filename {document['filename']!r} != image filename {image_filename!r}",
        )
    objects = []
    for index, item in enumerate(document["selectedList"]):
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"]:
            raise AnnotationError("schema_mismatch", f"selectedList[{index}] missing name")
        rect = item.get("rect")
        if not isinstance(rect, dict) or not all(
            field in rect for field in ("x", "y", "width", "height")
        ):
            raise AnnotationError("schema_mismatch", f"selectedList[{index}] missing rect keys")
        x, y, w, h = (
            _number(rect[key], f"selectedList[{index}].rect.{key}")
            for key in ("x", "y", "width", "height")
        )
        xyxy = _bbox(x, y, x + w, y + h, image_width, image_height, f"item[{index}]")
        objects.append(
            ObjectAnnotation(
                instance_index=index,
                class_name=item["name"],
                class_id=None,
                sku_id=None,
                bbox_xyxy_px=xyxy,
                raw_fields=deepcopy({k: v for k, v in item.items() if k not in ("name", "rect")}),
            )
        )
    return AnnotationDocument(
        image_width=image_width,
        image_height=image_height,
        image_filename=document["filename"],
        source_format="seg",
        objects=tuple(objects),
        source_seg=deepcopy(document),
    )


def validate_class_mapping(classes: Mapping[int, str]) -> dict[int, str]:
    """A supplied mapping is authoritative; never derive IDs from label order."""
    if not isinstance(classes, Mapping) or not classes:
        raise AnnotationError("unknown_class_mapping", "class mapping cannot be empty")
    result: dict[int, str] = {}
    for index, name in classes.items():
        if type(index) is not int or index < 0 or not isinstance(name, str) or not name:
            raise AnnotationError(
                "unknown_class_mapping", "IDs must be non-negative ints; names nonempty"
            )
        if index in result or name in result.values():
            raise AnnotationError("unknown_class_mapping", "duplicate class ID or class name")
        result[index] = name
    return result


def load_class_mapping(path: Path) -> dict[int, str]:
    """Read a *local-only* JSON mapping like {\"0\":\"sample_class\"}."""
    try:
        content = Path(path).read_bytes()
    except OSError as exc:
        raise AnnotationError("unknown_class_mapping", f"cannot read mapping: {exc}") from exc
    try:
        raw = _decode_json(content, "class mapping")
    except AnnotationError as exc:
        raise AnnotationError("unknown_class_mapping", str(exc)) from exc
    if not isinstance(raw, dict):
        raise AnnotationError("unknown_class_mapping", "mapping must be a JSON object")
    classes = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.isascii() or not key.isdecimal():
            raise AnnotationError("unknown_class_mapping", f"invalid class ID key {key!r}")
        number = int(key)
        if str(number) != key or number in classes:
            raise AnnotationError("unknown_class_mapping", f"ambiguous class ID {key!r}")
        classes[number] = value
    return validate_class_mapping(classes)


def _snap_pixel(value: float) -> int | float:
    """Remove harmless floating error after normalized YOLO -> pixel conversion."""
    nearest = round(value)
    return nearest if abs(value - nearest) < 1e-8 else value


def parse_yolo(
    raw: bytes | str,
    *,
    image_width: int,
    image_height: int,
    class_mapping: Mapping[int, str],
    image_filename: str | None = None,
) -> AnnotationDocument:
    """Parse standard YOLO TXT: class_id cx cy w h (normalized [0,1])."""
    _image_size(image_width, image_height)
    mapping = validate_class_mapping(class_mapping)
    try:
        content = raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw
    except UnicodeError as exc:
        raise AnnotationError("schema_mismatch", f"YOLO TXT encoding: {exc}") from exc
    if not isinstance(content, str):
        raise AnnotationError("schema_mismatch", "YOLO TXT must be text")
    objects = []
    for line_no, line in enumerate(content.splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise AnnotationError("schema_mismatch", f"line {line_no}: expected 5 fields")
        class_id_text = parts[0]
        if not class_id_text.isascii() or not class_id_text.isdecimal():
            raise AnnotationError("schema_mismatch", f"line {line_no}: invalid class ID")
        class_id = int(class_id_text)
        if class_id not in mapping:
            raise AnnotationError("unknown_class_mapping", f"line {line_no}: unknown ID {class_id}")
        try:
            cx, cy, w, h = (float(token) for token in parts[1:])
        except ValueError as exc:
            raise AnnotationError("invalid_bbox", f"line {line_no}: nonnumeric bbox") from exc
        for value, label in zip((cx, cy, w, h), ("cx", "cy", "w", "h"), strict=True):
            _number(value, f"line {line_no}.{label}")
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise AnnotationError(
                "invalid_bbox", f"line {line_no}: normalized values out of bounds"
            )
        x1 = _snap_pixel((cx - w / 2) * image_width)
        y1 = _snap_pixel((cy - h / 2) * image_height)
        x2 = _snap_pixel((cx + w / 2) * image_width)
        y2 = _snap_pixel((cy + h / 2) * image_height)
        # Only tolerate representation noise (< 1e-8 px); never clip real invalid boxes.
        xyxy = _bbox(x1, y1, x2, y2, image_width, image_height, f"line {line_no}")
        objects.append(
            ObjectAnnotation(len(objects), mapping[class_id], class_id, None, xyxy, {})
        )
    return AnnotationDocument(
        image_width, image_height, image_filename, "yolo", tuple(objects)
    )


def to_yolo_txt(document: AnnotationDocument, class_mapping: Mapping[int, str]) -> str:
    """Convert bbox to normalized YOLO TXT with explicitly stable class IDs."""
    mapping = validate_class_mapping(class_mapping)
    name_to_id = {name: number for number, name in mapping.items()}
    _image_size(document.image_width, document.image_height)
    lines = []
    for obj in document.objects:
        if obj.class_name not in name_to_id:
            raise AnnotationError(
                "unknown_class_mapping", f"class {obj.class_name!r} has no declared ID"
            )
        x1, y1, x2, y2 = _bbox(
            *obj.bbox_xyxy_px,
            document.image_width,
            document.image_height,
            f"item[{obj.instance_index}]",
        )
        cx = (x1 + x2) / (2 * document.image_width)
        cy = (y1 + y2) / (2 * document.image_height)
        w = (x2 - x1) / document.image_width
        h = (y2 - y1) / document.image_height
        values = " ".join(format(v, ".17g") for v in (cx, cy, w, h))
        lines.append(f"{name_to_id[obj.class_name]} {values}")
    return "\n".join(lines) + ("\n" if lines else "")


def to_seg_json(document: AnnotationDocument, *, filename: str) -> bytes:
    """Render a new SEG with only its filename updated for original SEG inputs.

    Preserve all original keys/values including gender, unknown object and root fields.
    YOLO inputs create a minimal SEG with class names and pixel rectangles.
    """
    if not isinstance(filename, str) or not filename or filename in (".", "..") or any(
        separator in filename for separator in ("/", "\\", "\x00")
    ):
        raise AnnotationError("schema_mismatch", "output filename must be a basename")
    if document.source_format == "seg":
        if document.source_seg is None:
            raise AnnotationError("schema_mismatch", "missing original SEG fields")
        result = deepcopy(document.source_seg)
    else:
        result = {
            "selectedList": [
                {
                    "name": obj.class_name,
                    "rect": {
                        "x": obj.bbox_xyxy_px[0],
                        "y": obj.bbox_xyxy_px[1],
                        "width": obj.bbox_xyxy_px[2] - obj.bbox_xyxy_px[0],
                        "height": obj.bbox_xyxy_px[3] - obj.bbox_xyxy_px[1],
                    },
                }
                for obj in document.objects
            ]
        }
    result["filename"] = filename
    return (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
