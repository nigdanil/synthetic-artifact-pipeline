"""Generate private static HTML for exported SAP runs; no web server needed.

Usage: python scripts/build_batch_gallery.py [--overwrite] [--project-root ...]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sapipeline.review.gallery import BatchGalleryError, build_batch_gallery


def publish(root: Path, *, overwrite: bool = False) -> tuple[str, Path, int]:
    if not root.is_absolute() or root.is_symlink() or not (root / "pyproject.toml").is_file():
        raise BatchGalleryError("project_root must be an absolute SAP checkout")
    reports = root / "reports"
    output = reports / "batch_gallery"
    if reports.is_symlink() or output.is_symlink():
        raise BatchGalleryError("Unsafe symlinked gallery directory")
    if output.exists() and not overwrite:
        raise BatchGalleryError("Gallery exists. Use --overwrite to refresh the snapshot")
    gallery = build_batch_gallery(root)
    reports.mkdir(exist_ok=True)
    temp: Path | None = Path(tempfile.mkdtemp(prefix=".gallery_pending_", dir=reports))
    backup: Path | None = None
    try:
        (temp / "index.html").write_bytes(gallery.html_page)
        (temp / "gallery_summary.json").write_text(
            json.dumps(gallery.summary, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (temp / "thumbs").mkdir()
        for relative, data in gallery.thumbnails.items():
            (temp / relative).write_bytes(data)
        if output.exists():
            if not output.is_dir():
                raise BatchGalleryError("Gallery target exists but is not a directory")
            backup = reports / (".gallery_previous_" + uuid.uuid4().hex)
            os.replace(output, backup)
        try:
            os.replace(temp, output)
            temp = None
        except Exception:
            if backup is not None:
                os.replace(backup, output)
                backup = None
            raise
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
    finally:
        if temp is not None:
            shutil.rmtree(temp, ignore_errors=True)
    return "CREATED", output / "index.html", gallery.summary["runs"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SAP Stage 8.2 private, offline Batch Gallery")
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--overwrite", action="store_true", help="Replace derived gallery snapshot")
    args = parser.parse_args(argv)
    try:
        result, output, runs = publish(Path(args.project_root).resolve(), overwrite=args.overwrite)
        print(f"Batch Gallery: {result}; runs={runs}; {output}")
        print("Read-only gallery. Reviews and original exported bundles unchanged.")
        return 0
    except (BatchGalleryError, OSError, ValueError) as exc:
        print(f"Batch Gallery ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
