"""Private-only mask PNG, overlay preview and manifest from source ZIP + ROI selection."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sapipeline.dataset import (
    DatasetIndexError,
    choose_index_entry,
    load_indexed_pair,
    read_private_index,
)
from sapipeline.masks import MaskError, build_mask
from sapipeline.selection import SelectionError, select_rois

ROOT = Path(__file__).resolve().parents[1]


def _private_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = (ROOT / "reports").resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError("--out-dir must be a subdirectory inside the local reports/")
    return resolved


def _as_png(pixels: Image.Image) -> bytes:
    payload = BytesIO()
    pixels.save(payload, format="PNG")
    return payload.getvalue()


def _overlay_preview(image: Image.Image, mask: Image.Image, targets) -> Image.Image:
    # Visualization ONLY: do not export this as an augmented training image.
    cyan = Image.new("RGB", image.size, (0, 225, 240))
    alpha = mask.point(lambda value: round(value * 0.45))
    preview = Image.composite(cyan, image, alpha)
    draw = ImageDraw.Draw(preview)
    stroke = max(2, round(min(image.size) / 350))
    for obj in targets:
        x1, y1, x2, y2 = obj.bbox_xyxy_px
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 255), width=stroke)
        draw.text((max(0, x1), max(0, y1 - 13)), str(obj.instance_index),
                  fill=(0, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))
    return preview


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preview private bbox-derived effect masks")
    parser.add_argument("--index", type=Path, default=ROOT / "data/work/index/dataset_index.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--split", choices=("learning", "verification"), default="learning")
    parser.add_argument("--image", help="Exact private ZIP image member, such as img/example.jpg")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--mode", choices=("by_index", "by_class", "random_one", "random_n"),
                        default="random_one")
    parser.add_argument("--instance-index", type=int)
    parser.add_argument("--class-name")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42017)
    parser.add_argument("--include-generic-product", action="store_true")
    parser.add_argument("--allow-service-targets", action="store_true")
    parser.add_argument("--mask-kind", choices=("rect", "soft_rect"), default="soft_rect")
    parser.add_argument("--pad-px", type=int, default=0)
    parser.add_argument("--feather-px", type=float, default=16.0)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports/mask_previews")
    parser.add_argument("--force", action="store_true", help="Overwrite existing local artifacts")
    args = parser.parse_args(argv)

    try:
        if args.instance_index is not None and args.mode != "by_index":
            raise SelectionError("--instance-index requires --mode by_index")
        if args.class_name is not None and args.mode != "by_class":
            raise SelectionError("--class-name requires --mode by_class")
        if args.mode != "random_n" and args.count != 1:
            raise SelectionError("--count is only supported for --mode random_n")
        if args.mode not in ("random_one", "random_n") and args.include_generic_product:
            raise SelectionError("--include-generic-product applies only to random modes")
        if args.mode in ("random_one", "random_n") and args.allow_service_targets:
            raise SelectionError("--allow-service-targets applies only to explicit selections")
        output_dir = _private_output_dir(args.out_dir)
        index = read_private_index(args.index)
        entry = choose_index_entry(index, split=args.split, image_member=args.image,
                                   sample_index=args.sample_index)
        pair = load_indexed_pair(index, entry, data_dir=args.data_dir)
        selected = select_rois(
            pair.annotations,
            mode=args.mode, instance_index=args.instance_index,
            class_name=args.class_name, count=args.count, seed=args.seed,
            include_generic_product=args.include_generic_product,
            allow_service_targets=args.allow_service_targets,
        )
        generated = build_mask(
            pair.image.width, pair.image.height, selected.targets,
            kind=args.mask_kind, pad_px=args.pad_px,
            feather_px=args.feather_px, strength=args.strength,
        )
        mask_bytes = np.rint(generated.mask * 255).astype(np.uint8)
        mask_image = Image.fromarray(mask_bytes)
        overlay = _overlay_preview(pair.image, mask_image, selected.targets)

        parameters = {
            "selection_mode": args.mode,
            "seed": args.seed,
            "indices": sorted(obj.instance_index for obj in selected.targets),
            "include_generic_product": args.include_generic_product,
            "allow_service_targets": args.allow_service_targets,
            "mask_kind": generated.kind,
            "pad_px": generated.pad_px,
            "feather_px": generated.feather_px,
            "strength": generated.strength,
        }
        # Filename does not include vendor class names; parameters distinguish runs.
        key = sha256(json.dumps(parameters, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        name = f"{entry['split']}_{entry['image_sha256'][:12]}_mask_{key}"
        mask_path = output_dir / f"{name}_mask.png"
        preview_path = output_dir / f"{name}_overlay.png"
        report_path = output_dir / f"{name}.json"
        targets = [
            {
                "instance_index": obj.instance_index,
                "class_name": obj.class_name,
                "bbox_xyxy_px": list(obj.bbox_xyxy_px),
                "padded_bbox_xyxy_px": list(box),
            }
            for obj, box in zip(selected.targets, generated.padded_boxes_xyxy_px,
                                strict=True)
        ]
        report = {
            "schema_version": "1.0",
            "mask_definition": "bbox-effect-weights-not-segmentation",
            "orientation_policy": "encoded-pixels-no-EXIF-transpose",
            "source": {
                "split": entry["split"], "archive": entry["archive"],
                "image_member": entry["image_member"],
                "annotation_member": entry["annotation_member"],
                "image_sha256": entry["image_sha256"],
                "annotation_sha256": entry["annotation_sha256"],
            },
            "image_size_px": [pair.image.width, pair.image.height],
            "parameters": parameters,
            "targets": targets,
            "compositing": "per-pixel maximum",
            "mask_format": "8-bit grayscale PNG; float32 mask in memory",
            "mask_file": mask_path.name,
            "preview_file": preview_path.name,
            "qa": {"status": "pending_manual_review"},
        }
        json_bytes = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        if not args.force and any(path.exists() for path in (mask_path, preview_path, report_path)):
            raise ValueError("Mask/preview/report already exists; specify --force to replace")
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".mask-", dir=output_dir) as temporary:
            temp = Path(temporary)
            draft_mask = temp / "mask.png"
            draft_preview = temp / "preview.png"
            draft_report = temp / "report.json"
            draft_mask.write_bytes(_as_png(mask_image))
            draft_preview.write_bytes(_as_png(overlay))
            draft_report.write_bytes(json_bytes)
            os.replace(draft_mask, mask_path)
            os.replace(draft_preview, preview_path)
            os.replace(draft_report, report_path)
        print(f"Source: {entry['split']}/{entry['image_member']}")
        print(f"Targets: {parameters['indices']} | kind: {generated.kind}")
        print(f"Mask: {mask_path}")
        print(f"Overlay: {preview_path}")
        print(f"Report: {report_path}")
        return 0
    except (DatasetIndexError, SelectionError, MaskError, OSError, ValueError,
            TypeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
