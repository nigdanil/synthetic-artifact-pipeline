"""Local-only bbox ROI preview. No image effects, labels export or ComfyUI edits."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from hashlib import sha256
from io import BytesIO
from pathlib import Path

from PIL import ImageDraw

from sapipeline.dataset import (
    DatasetIndexError,
    choose_index_entry,
    load_indexed_pair,
    read_private_index,
)
from sapipeline.selection import SelectionError, select_rois

ROOT = Path(__file__).resolve().parents[1]


def _private_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    report_root = (ROOT / "reports").resolve()
    if resolved == report_root or not resolved.is_relative_to(report_root):
        raise ValueError("--out-dir must be a subdirectory inside the local reports/")
    return resolved


def _draw_preview(pixels, objects, selected_indices: set[int]) -> bytes:
    canvas = pixels.copy()
    draw = ImageDraw.Draw(canvas)
    stroke = max(2, round(min(canvas.size) / 350))
    for obj in objects:
        x1, y1, x2, y2 = obj.bbox_xyxy_px
        selected = obj.instance_index in selected_indices
        color = (0, 245, 255) if selected else (255, 64, 64)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=stroke * (3 if selected else 1))
        y_label = max(0, y1 - 13)
        draw.text((max(0, x1), y_label), str(obj.instance_index), fill=color,
                  stroke_width=1, stroke_fill=(0, 0, 0))
    buffer = BytesIO()
    canvas.save(buffer, format="PNG")  # lossless, full resolution, no geometry change
    return buffer.getvalue()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preview deterministic product bbox ROI selection")
    parser.add_argument("--index", type=Path, default=ROOT / "data/work/index/dataset_index.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--split", choices=("learning", "verification"), default="learning")
    parser.add_argument("--image", help="Exact ZIP member path, e.g. img/example.jpg")
    parser.add_argument("--sample-index", type=int, default=0,
                        help="0-based position among globally unique, normally oriented examples")
    parser.add_argument("--mode", choices=("by_index", "by_class", "random_one", "random_n"),
                        default="random_one")
    parser.add_argument("--instance-index", type=int)
    parser.add_argument("--class-name")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42017)
    parser.add_argument("--include-generic-product", action="store_true")
    parser.add_argument("--allow-service-targets", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports/roi_previews")
    parser.add_argument("--force", action="store_true", help="Replace existing private preview files")
    args = parser.parse_args(argv)

    try:
        # Reject ignored options rather than silently disregarding user intent.
        if args.instance_index is not None and args.mode != "by_index":
            raise SelectionError("--instance-index requires --mode by_index")
        if args.class_name is not None and args.mode != "by_class":
            raise SelectionError("--class-name requires --mode by_class")
        if args.mode != "random_n" and args.count != 1:
            raise SelectionError("--count is only supported for --mode random_n")
        if args.mode not in ("random_one", "random_n") and args.include_generic_product:
            raise SelectionError("--include-generic-product only applies to random modes")
        if args.mode in ("random_one", "random_n") and args.allow_service_targets:
            raise SelectionError("--allow-service-targets only applies to explicit selections")
        if args.seed < 0:
            raise SelectionError("--seed must be non-negative")
        out_dir = _private_output_dir(args.out_dir)
        index = read_private_index(args.index)
        entry = choose_index_entry(index, split=args.split, image_member=args.image,
                                   sample_index=args.sample_index)
        pair = load_indexed_pair(index, entry, data_dir=args.data_dir)
        result = select_rois(
            pair.annotations,
            mode=args.mode,
            seed=args.seed,
            instance_index=args.instance_index,
            class_name=args.class_name,
            count=args.count,
            include_generic_product=args.include_generic_product,
            allow_service_targets=args.allow_service_targets,
        )
        picked = {x.instance_index for x in result.targets}
        preview = _draw_preview(pair.image, pair.annotations.objects, picked)
        # Only hash class filters; no vendor names in output filenames.
        discriminator = ""
        if args.mode == "by_index":
            discriminator = f"_i{args.instance_index}"
        elif args.mode == "by_class":
            discriminator = "_c" + sha256(args.class_name.encode("utf-8")).hexdigest()[:8]
        elif args.mode == "random_n":
            discriminator = f"_n{args.count}"
        if args.include_generic_product:
            discriminator += "_withgeneric"
        if args.allow_service_targets:
            discriminator += "_service"
        name = f"{args.split}_{entry['image_sha256'][:12]}_{args.mode}_{args.seed}{discriminator}"
        image_path = out_dir / f"{name}.png"
        report_path = out_dir / f"{name}.json"
        if not args.force and (image_path.exists() or report_path.exists()):
            raise ValueError("Preview/report already exists; specify --force to replace")
        report = {
            "schema_version": "1.0",
            "policy": "encoded-pixels-no-EXIF-transpose",
            "source": {
                "split": entry["split"], "archive": entry["archive"],
                "image_member": entry["image_member"],
                "annotation_member": entry["annotation_member"],
                "image_sha256": entry["image_sha256"],
                "annotation_sha256": entry["annotation_sha256"],
            },
            "image_size_px": [pair.image.width, pair.image.height],
            "selection": {
                "mode": result.mode, "seed": result.seed,
                "include_generic_product": args.include_generic_product,
                "allow_service_targets": args.allow_service_targets,
                "instance_indices": sorted(picked),
            },
            "objects": [{
                "instance_index": obj.instance_index,
                "class_name": obj.class_name,
                "bbox_xyxy_px": list(obj.bbox_xyxy_px),
                "selected": obj.instance_index in picked,
            } for obj in pair.annotations.objects],
            "preview_file": image_path.name,
            "qa": {"status": "pending_manual_review"},
        }
        body = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        out_dir.mkdir(parents=True, exist_ok=True)
        # Write both payloads fully before making either visible.
        with tempfile.TemporaryDirectory(prefix=".roi-", dir=out_dir) as temp:
            draft_png = Path(temp) / "image.png"
            draft_json = Path(temp) / "report.json"
            draft_png.write_bytes(preview)
            draft_json.write_bytes(body)
            os.replace(draft_png, image_path)
            os.replace(draft_json, report_path)
        print(f"Source: {entry['split']}/{entry['image_member']}")
        print(f"Objects: {len(pair.annotations.objects)} | targets: {sorted(picked)}")
        print(f"Preview: {image_path}")
        print(f"Report: {report_path}")
        return 0
    except (DatasetIndexError, SelectionError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
