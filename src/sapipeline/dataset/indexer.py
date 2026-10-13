"""Read-only ZIP dataset inventory; not a SEG/YOLO annotation parser.

This module deliberately never extracts or rewrites source images/annotations.
It indexes paths, image dimensions, raw SHA-256 hashes and annotation counts.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import BadZipFile, ZipFile, ZipInfo

from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_IMAGE_BYTES = 100 * 1024 * 1024
MAX_ANNOTATION_BYTES = 10 * 1024 * 1024


class DatasetIndexError(ValueError):
    """Input is incomplete, ambiguous or damaged; don't publish a success index."""


def _sha(data: bytes) -> str:
    return sha256(data).hexdigest()


def _zip_sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pairs(zipped: ZipFile, label: str) -> tuple[dict[str, ZipInfo], dict[str, ZipInfo]]:
    images: dict[str, ZipInfo] = {}
    annotations: dict[str, ZipInfo] = {}
    seen_members: set[str] = set()
    for member in zipped.infolist():
        if member.is_dir():
            continue
        name = member.filename
        path = PurePosixPath(name)
        if (
            name.startswith("/")
            or "\\" in name
            or any(part in ("", ".", "..") for part in name.split("/"))
        ):
            raise DatasetIndexError(f"{label}: unsafe ZIP member path: {name!r}")
        folded = name.casefold()
        if folded in seen_members:
            raise DatasetIndexError(f"{label}: duplicate ZIP member: {name!r}")
        seen_members.add(folded)
        if len(path.parts) < 2 or path.parts[0] not in ("img", "seg"):
            continue
        is_image = path.parts[0] == "img" and path.suffix.lower() in IMAGE_EXTENSIONS
        is_annotation = path.parts[0] == "seg" and path.suffix.lower() == ".seg"
        if not (is_image or is_annotation):
            continue
        # Relative stem preserves subdirectories, if present, and is independent of extension.
        stem = "/".join(path.parts[1:-1] + (path.stem,)).casefold()
        target = images if is_image else annotations
        if stem in target:
            raise DatasetIndexError(f"{label}: ambiguous files for {stem!r}")
        target[stem] = member
    missing_seg = sorted(images.keys() - annotations.keys())
    missing_image = sorted(annotations.keys() - images.keys())
    if missing_seg or missing_image or not images:
        raise DatasetIndexError(
            f"{label}: {len(missing_seg)} images without SEG, "
            f"{len(missing_image)} SEGs without image, {len(images)} images found; "
            f"examples: missing_seg={missing_seg[:3]}, missing_image={missing_image[:3]}"
        )
    return images, annotations


def _read_member(zipped: ZipFile, info: ZipInfo, maximum: int, label: str) -> bytes:
    if info.file_size > maximum:
        raise DatasetIndexError(f"{label}: member too large: {info.filename}")
    try:
        # Bound decompression even if a corrupt ZIP lies about uncompressed size.
        with zipped.open(info) as stream:
            data = stream.read(maximum + 1)
        if len(data) > maximum:
            raise DatasetIndexError(f"{label}: member too large: {info.filename}")
        return data
    except (BadZipFile, OSError, RuntimeError, EOFError) as exc:
        raise DatasetIndexError(f"{label}: unreadable member {info.filename}: {exc}") from exc


def _inspect_image(data: bytes, label: str) -> tuple[int, int, int]:
    try:
        with Image.open(BytesIO(data)) as picture:
            width, height = picture.size  # Raw encoded pixel coordinates (no EXIF rotation).
            orientation = picture.getexif().get(274, 1)
            picture.load()  # Force decode; header-only checks miss truncated JPEGs.
        if width <= 0 or height <= 0:
            raise ValueError("non-positive image dimensions")
        return width, height, orientation
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise DatasetIndexError(f"{label}: invalid image: {exc}") from exc


def _inspect_annotation(data: bytes, image_name: str, label: str) -> tuple[int, str]:
    try:
        raw = json.loads(data.decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetIndexError(f"{label}: invalid SEG JSON: {exc}") from exc
    if (
        not isinstance(raw, dict)
        or not isinstance(raw.get("selectedList"), list)
        or not isinstance(raw.get("filename"), str)
    ):
        raise DatasetIndexError(f"{label}: SEG must have selectedList and filename")
    if raw["filename"] != image_name:
        raise DatasetIndexError(
            f"{label}: SEG filename differs from paired image: "
            f"{raw['filename']!r} != {image_name!r}"
        )
    # Compare semantic annotation contents without the filename metadata, because
    # one exact JPEG can legitimately be stored under different names.
    comparable = {key: value for key, value in raw.items() if key != "filename"}
    # selectedList order is not meaningful to a dataset inventory: compare
    # the multiset of full annotation objects, preserving any repeats.
    comparable["selectedList"] = sorted(
        json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        for obj in comparable["selectedList"]
    )
    fingerprint = _sha(json.dumps(comparable, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return len(raw["selectedList"]), fingerprint


def build_index(archives: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a deterministic inventory and review report from read-only ZIP archives.

    `archives` maps stable split names to ZIP files. No outputs are written here.
    """
    if len(archives) < 1:
        raise DatasetIndexError("At least one ZIP archive is required")
    records: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    for split, supplied in sorted(archives.items()):
        archive = Path(supplied)
        if not archive.is_file():
            raise DatasetIndexError(f"{split}: ZIP not found: {archive}")
        sources.append(
            {"split": split, "archive": archive.name, "archive_sha256": _zip_sha(archive)}
        )
        try:
            with ZipFile(archive, "r") as zipped:
                images, annotations = _pairs(zipped, split)
                for stem in sorted(images):
                    image_info = images[stem]
                    seg_info = annotations[stem]
                    image = _read_member(zipped, image_info, MAX_IMAGE_BYTES, split)
                    seg = _read_member(zipped, seg_info, MAX_ANNOTATION_BYTES, split)
                    width, height, orientation = _inspect_image(
                        image, f"{split}/{image_info.filename}"
                    )
                    count, fingerprint = _inspect_annotation(
                        seg, PurePosixPath(image_info.filename).name, f"{split}/{seg_info.filename}"
                    )
                    digest = _sha(image)
                    records.append(
                        {
                            "split": split,
                            "archive": archive.name,
                            "image_member": image_info.filename,
                            "annotation_member": seg_info.filename,
                            "image_sha256": digest,
                            "annotation_sha256": _sha(seg),
                            "canonical_image_id": digest,
                            "annotation_fingerprint": fingerprint,
                            "width": width,
                            "height": height,
                            "exif_orientation": orientation,
                            "bbox_count_declared": count,
                        }
                    )
        except (BadZipFile, OSError) as exc:
            raise DatasetIndexError(f"{split}: failed to open ZIP: {exc}") from exc
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_hash[row["image_sha256"]].append(row)
    duplicate_groups: list[dict[str, Any]] = []
    for digest, members in sorted(by_hash.items()):
        if len(members) < 2:
            continue
        splits = sorted({member["split"] for member in members})
        fingerprints = {member["annotation_fingerprint"] for member in members}
        duplicate_groups.append(
            {
                "group_id": digest,
                "cross_split": len(splits) > 1,
                "annotation_disagreement": len(fingerprints) > 1,
                "members": [
                    {
                        "split": m["split"],
                        "archive": m["archive"],
                        "image": m["image_member"],
                        "annotation": m["annotation_member"],
                    }
                    for m in members
                ],
            }
        )
    by_split = Counter(row["split"] for row in records)
    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "total_pairs": len(records),
        "pairs_by_split": dict(sorted(by_split.items())),
        "total_bbox_entries_declared": sum(row["bbox_count_declared"] for row in records),
        "unique_images_sha256": len(by_hash),
        "exact_duplicate_groups": len(duplicate_groups),
        "cross_split_duplicate_groups": sum(g["cross_split"] for g in duplicate_groups),
        "annotation_disagreement_groups": sum(
            g["annotation_disagreement"] for g in duplicate_groups
        ),
        "exif_orientation_review_count": sum(
            row["exif_orientation"] not in (1, None) for row in records
        ),
        "duplicate_groups": duplicate_groups,
        "notes": [
            "SHA-256 detects byte-identical images only; near duplicates require separate review.",
            "Duplicate annotations are not merged; disagreement needs manual review.",
            "bbox_count_declared is a list length, not full bbox geometry validation.",
            "These splits must not be treated as an independent test set before leakage review.",
        ],
    }
    index = {"schema_version": "1.0", "sources": sources, "items": records}
    return index, summary
