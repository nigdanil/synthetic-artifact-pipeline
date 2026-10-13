"""Load a verified JPEG/SEG pair from a local ZIP and the private Task-1 index.

Read-only: no extraction, geometry transforms, or modification of the originals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import BadZipFile, ZipFile

from PIL import Image, UnidentifiedImageError

from sapipeline.annotations import AnnotationDocument, AnnotationError, parse_seg
from sapipeline.dataset.indexer import (
    MAX_ANNOTATION_BYTES,
    MAX_IMAGE_BYTES,
    DatasetIndexError,
    _read_member,
)


@dataclass(slots=True)
class LoadedPair:
    """RGB preview pixels on the *encoded image grid* and their source annotations."""

    image: Image.Image
    annotations: AnnotationDocument
    entry: dict[str, Any]


def _member(value: Any, prefix: str, suffixes: tuple[str, ...]) -> str:
    if not isinstance(value, str):
        raise DatasetIndexError("Index member path must be text")
    path = PurePosixPath(value)
    if (
        not value.startswith(prefix + "/")
        or "\\" in value
        or any(part in ("", ".", "..") for part in value.split("/"))
        or path.suffix.lower() not in suffixes
    ):
        raise DatasetIndexError(f"Unsafe index member path: {value!r}")
    return value


def read_private_index(path: Path) -> dict[str, Any]:
    """Read Task-1 metadata; never infer a missing index from other directories."""
    try:
        index = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetIndexError(f"Cannot read private dataset index: {exc}") from exc
    if (
        not isinstance(index, dict)
        or index.get("schema_version") != "1.0"
        or not isinstance(index.get("items"), list)
        or not isinstance(index.get("sources"), list)
    ):
        raise DatasetIndexError("Unsupported dataset index structure (rerun Task 1)")
    return index


def choose_index_entry(
    index: dict[str, Any], *, split: str, image_member: str | None = None,
    sample_index: int = 0,
) -> dict[str, Any]:
    """Pick an exact member, or the first globally unique, normally oriented example.

    Explicit --image can address duplicates for inspection, but automatic selection
    excludes *all* exact duplicates (especially cross-split overlaps).
    """
    if split not in ("learning", "verification"):
        raise DatasetIndexError(f"Unsupported split {split!r}")
    if isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index < 0:
        raise DatasetIndexError("sample_index must be a non-negative integer")
    if image_member is not None and sample_index != 0:
        raise DatasetIndexError("--sample-index cannot be combined with --image")
    items = index.get("items")
    if not isinstance(items, list):
        raise DatasetIndexError("Index items missing")
    if image_member is not None:
        _member(image_member, "img", (".jpg", ".jpeg", ".png"))
        matches = [x for x in items if isinstance(x, dict) and x.get("split") == split
                   and x.get("image_member") == image_member]
        if len(matches) != 1:
            raise DatasetIndexError("Requested split/image must identify exactly one indexed pair")
        return matches[0]

    counts: dict[str, int] = {}
    for entry in items:
        if not isinstance(entry, dict) or not isinstance(entry.get("image_sha256"), str):
            raise DatasetIndexError("Invalid index entry; rerun Task 1")
        key = entry["image_sha256"]
        counts[key] = counts.get(key, 0) + 1
    candidates = [x for x in items if x.get("split") == split
                  and x.get("exif_orientation") == 1
                  and isinstance(x.get("bbox_count_declared"), int)
                  and x["bbox_count_declared"] > 0
                  and counts[x["image_sha256"]] == 1]
    if not candidates:
        raise DatasetIndexError("No unique normally oriented annotated images; specify --image")
    ordered = sorted(candidates, key=lambda x: x["image_member"])
    if sample_index >= len(ordered):
        raise DatasetIndexError(f"sample_index={sample_index} out of range ({len(ordered)} eligible)")
    return ordered[sample_index]


def load_indexed_pair(
    index: dict[str, Any], entry: dict[str, Any], *, data_dir: Path
) -> LoadedPair:
    """Verify source names, original SHA256s, dimensions and SEG before returning RGB."""
    if not isinstance(entry, dict):
        raise DatasetIndexError("Invalid index entry")
    split = entry.get("split")
    archive = entry.get("archive")
    sources = index.get("sources")
    if (
        not isinstance(sources, list)
        or not isinstance(archive, str)
        or archive in ("", ".", "..")
        or Path(archive).name != archive
        or "/" in archive or "\\" in archive
        or sum(isinstance(s, dict) and s.get("split") == split
               and s.get("archive") == archive for s in sources) != 1
    ):
        raise DatasetIndexError("Index source/split/archive mismatch")
    image_name = _member(entry.get("image_member"), "img", (".jpg", ".jpeg", ".png"))
    annotation_name = _member(entry.get("annotation_member"), "seg", (".seg",))
    if PurePosixPath(image_name).with_suffix("").name.casefold() != (
        PurePosixPath(annotation_name).stem.casefold()
    ) or PurePosixPath(image_name).parts[1:-1] != PurePosixPath(annotation_name).parts[1:-1]:
        raise DatasetIndexError("Image and annotation paths do not form a pair")
    target = Path(data_dir) / archive
    try:
        with ZipFile(target, "r") as zipped:
            image_bytes = _read_member(
                zipped, zipped.getinfo(image_name), MAX_IMAGE_BYTES, image_name
            )
            annotation_bytes = _read_member(
                zipped, zipped.getinfo(annotation_name), MAX_ANNOTATION_BYTES,
                annotation_name,
            )
    except (OSError, BadZipFile, KeyError, ValueError) as exc:
        raise DatasetIndexError(f"Cannot read indexed pair from ZIP: {exc}") from exc
    if (sha256(image_bytes).hexdigest() != entry.get("image_sha256")
            or sha256(annotation_bytes).hexdigest() != entry.get("annotation_sha256")):
        raise DatasetIndexError("Source ZIP bytes do not match the private index (rerun Task 1)")
    try:
        with Image.open(BytesIO(image_bytes)) as source:
            source.load()
            orientation = source.getexif().get(274, 1)
            if orientation != entry.get("exif_orientation"):
                raise DatasetIndexError("EXIF orientation differs from indexed value")
            if orientation != 1:
                raise DatasetIndexError(
                    "Non-default EXIF orientation: manually review before ROI preview"
                )
            if (source.width, source.height) != (entry.get("width"), entry.get("height")):
                raise DatasetIndexError("Image dimensions differ from indexed size")
            pixels = source.convert("RGB")  # No EXIF rotation, resizing or source writes.
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise DatasetIndexError(f"Unable to decode source image: {exc}") from exc
    try:
        document = parse_seg(
            annotation_bytes,
            image_width=pixels.width,
            image_height=pixels.height,
            image_filename=PurePosixPath(image_name).name,
        )
    except AnnotationError as exc:
        raise DatasetIndexError(f"Source annotation invalid: {exc}") from exc
    if len(document.objects) != entry.get("bbox_count_declared"):
        raise DatasetIndexError("SEG object count differs from index")
    return LoadedPair(image=pixels, annotations=document, entry=dict(entry))
