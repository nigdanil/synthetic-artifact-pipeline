"""Private, read-only geometry validation and local bbox previews for source ZIPs."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
from zipfile import ZipFile

from PIL import Image, ImageDraw

from sapipeline.annotations import AnnotationError, load_class_mapping, parse_seg, to_yolo_txt
from sapipeline.dataset import DatasetIndexError, build_index
from sapipeline.dataset.indexer import MAX_ANNOTATION_BYTES, MAX_IMAGE_BYTES, _read_member

ROOT = Path(__file__).resolve().parents[1]


def _private_path(path: Path) -> Path:
    target = path.expanduser().resolve()
    report_root = (ROOT / "reports").resolve()
    if target == report_root or not target.is_relative_to(report_root):
        raise ValueError("Output must remain strictly under reports/")
    return target


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".annotation-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _preview(image_bytes: bytes, objects: tuple, *, expected_size: tuple[int, int]) -> bytes:
    with Image.open(BytesIO(image_bytes)) as photo:
        # Deliberately DO NOT apply ImageOps.exif_transpose: bbox is in encoded pixels.
        photo.load()
        if photo.size != expected_size:
            raise AnnotationError("schema_mismatch", "JPEG size differs from dataset index")
        canvas = photo.convert("RGB")
    drawing = ImageDraw.Draw(canvas)
    line_width = max(2, round(min(canvas.size) / 500))
    for obj in objects:
        x1, y1, x2, y2 = obj.bbox_xyxy_px
        drawing.rectangle((x1, y1, x2, y2), outline=(255, 40, 40), width=line_width)
        # Tiny instance numbers keep dense overlays readable; full class strings
        # remain in the private report, not painted across neighboring products.
        label = str(obj.instance_index)
        text_y = max(0, y1 - 12)
        drawing.text((max(0, x1), text_y), label, fill=(255, 255, 0), stroke_width=1,
                     stroke_fill=(0, 0, 0))
    buffer = BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate SEG bboxes against raw JPEG dimensions and draw local overlays"
        )
    )
    parser.add_argument("--learning", type=Path, default=ROOT / "data/raw/for_learning.zip")
    parser.add_argument("--verification", type=Path, default=ROOT / "data/raw/for_verification.zip")
    parser.add_argument(
        "--class-map", type=Path, help="Optional PRIVATE JSON map: validate SEG->YOLO"
    )
    parser.add_argument("--preview-count", type=int, default=5)
    parser.add_argument(
        "--report", type=Path, default=ROOT / "reports/annotation_validation_summary.json"
    )
    parser.add_argument("--preview-dir", type=Path, default=ROOT / "reports/annotation_previews")
    parser.add_argument(
        "--force", action="store_true", help="Replace existing private reports/previews"
    )
    args = parser.parse_args(argv)

    try:
        if not 0 <= args.preview_count <= 100:
            raise ValueError("--preview-count must be between 0 and 100")
        report_path = _private_path(args.report)
        preview_dir = _private_path(args.preview_dir)
        if preview_dir == report_path or report_path.is_relative_to(preview_dir):
            raise ValueError("Report and preview paths must not overlap")
        if report_path.exists() and not args.force:
            raise ValueError(f"Report exists: {report_path}; use --force")
        mapping = load_class_mapping(args.class_map) if args.class_map else None
        archives = {"learning": args.learning, "verification": args.verification}
        index, _ = build_index(archives)
        entries = index["items"]
        reviewed = []
        seen_hashes: set[str] = set()
        previews: list[tuple[Path, bytes]] = []
        total_objects = 0
        preview_examples: list[dict] = []
        for split, zip_path in sorted(archives.items()):
            with ZipFile(zip_path, "r") as source:
                for entry in entries:
                    if entry["split"] != split:
                        continue
                    seg_info = source.getinfo(entry["annotation_member"])
                    seg = _read_member(source, seg_info, MAX_ANNOTATION_BYTES,
                                       f"{split}/{entry['annotation_member']}")
                    document = parse_seg(
                        seg,
                        image_width=entry["width"],
                        image_height=entry["height"],
                        image_filename=PurePosixPath(entry["image_member"]).name,
                    )
                    if len(document.objects) != entry["bbox_count_declared"]:
                        raise AnnotationError("schema_mismatch", "annotation count changed")
                    if mapping is not None:
                        to_yolo_txt(document, mapping)  # require explicit coverage, no ID guesses
                    total_objects += len(document.objects)
                    if entry["exif_orientation"] != 1:
                        reviewed.append({
                            "split": split,
                            "image_member": entry["image_member"],
                            "exif_orientation": entry["exif_orientation"],
                        })
                    if (
                        split == "learning" and entry["exif_orientation"] == 1
                        and entry["image_sha256"] not in seen_hashes
                        and len(previews) < args.preview_count
                    ):
                        seen_hashes.add(entry["image_sha256"])
                        filename = f"learning_{len(previews) + 1:03}.png"
                        destination = preview_dir / filename
                        if destination.exists() and not args.force:
                            raise ValueError(f"Preview exists: {destination}; use --force")
                        image_info = source.getinfo(entry["image_member"])
                        image_bytes = _read_member(
                            source, image_info, MAX_IMAGE_BYTES, f"{split}/{entry['image_member']}"
                        )
                        previews.append((destination, _preview(
                            image_bytes, document.objects,
                            expected_size=(entry["width"], entry["height"]),
                        )))
                        preview_examples.append({
                            "preview_file": filename,
                            "image_member": entry["image_member"],
                            "objects": [{
                                "instance_index": obj.instance_index,
                                "class_name": obj.class_name,
                                "bbox_xyxy_px": obj.bbox_xyxy_px,
                            } for obj in document.objects],
                        })
        if len(previews) != args.preview_count:
            raise ValueError("Not enough distinct learning images for requested previews")
        result = {
            "schema_version": "1.0",
            "policy": "encoded-pixels-no-EXIF-transpose",
            "valid_pairs": len(entries),
            "valid_bbox_entries": total_objects,
            "exif_orientation_review_count": len(reviewed),
            "exif_orientation_review": reviewed,
            "preview_count": len(previews),
            "preview_examples": preview_examples,
            "class_mapping_used": mapping is not None,
            "warnings": ["Non-default EXIF metadata requires manual review before transforms"]
            if reviewed else [],
        }
        # Do not write reports or previews until every bbox has passed validation.
        for path, image in previews:
            _atomic_write(path, image)
        report_bytes = (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        _atomic_write(report_path, report_bytes)
        print(f"Validated pairs: {len(entries)} | bboxes: {total_objects}")
        print(f"EXIF review: {len(reviewed)} | local previews: {len(previews)}")
        print(f"Report: {report_path}")
        return 0
    except (AnnotationError, DatasetIndexError, ValueError, OSError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
