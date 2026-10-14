"""Offline glass-plane guides, manually saved corners and 2-effect pilot previews."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sapipeline.batch.glass_preview import (
    create_geometry_guides,
    generate_glass_previews,
    set_scene_quad,
)
from sapipeline.batch.runner import BatchRunError
from sapipeline.dataset import DatasetIndexError
from sapipeline.effects.glass_reflection import GlassReflectionError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Private SAP Glass Pilot geometry and preview")
    parser.add_argument("--project-root", default=str(ROOT))
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("guide", "preview"):
        command = sub.add_parser(action)
        command.add_argument("--start", type=int, default=0)
        command.add_argument("--count", type=int, default=5)
    setting = sub.add_parser("set-quad")
    setting.add_argument("--scene", type=int, required=True)
    setting.add_argument("--quad", required=True, help="JSON [[TLx,TLy], [TRx,TRy], [BRx,BRy], [BLx,BLy]]")
    setting.add_argument("--replace", action="store_true", help="Replace an existing quad with backup")
    args = parser.parse_args(argv)
    try:
        root = Path(args.project_root).absolute()
        if args.action == "guide":
            for path in create_geometry_guides(root, start=args.start, count=args.count):
                print(f"PRIVATE GLASS GUIDE: {path}")
        elif args.action == "set-quad":
            try:
                coords = json.loads(args.quad)
            except json.JSONDecodeError as exc:
                raise BatchRunError("--quad must be a JSON array of four coordinate pairs") from exc
            path = set_scene_quad(root, scene=args.scene, quad=coords, replace=args.replace)
            print(f"SAVED MANUAL QUAD for scene {args.scene}: {path}")
        else:
            for folder, doc in generate_glass_previews(root, start=args.start, count=args.count):
                statuses = ", ".join(
                    f"{x['variant']}={x['technical_qa']['status']}" for x in doc["variants"]
                )
                print(f"PRIVATE GLASS PREVIEW: {folder} [{statuses}]")
        print("No Stage-7 export or training approval. Human visual review required.")
        return 0
    except (BatchRunError, DatasetIndexError, GlassReflectionError,
            OSError, ValueError, TypeError) as exc:
        print(f"GLASS PREVIEW ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
