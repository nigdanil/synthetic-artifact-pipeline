"""Plan and validate private glass reflection Pilot 100 (no image generation)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sapipeline.batch.glass_pilot import (
    build_glass_plan,
    load_glass_plan,
    save_glass_plan,
)
from sapipeline.batch.runner import BatchRunError
from sapipeline.dataset import DatasetIndexError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Private glass-reflection 100-image plan only")
    parser.add_argument("--project-root", default=str(ROOT))
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("plan", help="Reserve sources and effects, without generating images")
    p.add_argument("--scenes", type=int, default=60)
    p.add_argument("--seed", type=int, default=850001)
    sub.add_parser("check", help="Validate existing plan and count missing glass quads")
    args = parser.parse_args(argv)
    try:
        root = Path(args.project_root).absolute()
        if args.action == "plan":
            plan = build_glass_plan(root, scenes=args.scenes, seed=args.seed)
            output = save_glass_plan(root, plan)
            print(f"GLASS PLAN: {len(plan['sources'])} distinct new sources, "
                  f"{len(plan['runs'])} candidates, target=100; {output}")
        else:
            plan, missing = load_glass_plan(root)
            print(f"GLASS PLAN VALID: sources={len(plan['sources'])}, "
                  f"candidates={len(plan['runs'])}, missing_quads={missing}")
        print("PLAN ONLY. No images exported. No automatic glass localization or QA approval.")
        return 0
    except (BatchRunError, OSError, ValueError, TypeError, DatasetIndexError) as exc:
        print(f"GLASS PLAN ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
