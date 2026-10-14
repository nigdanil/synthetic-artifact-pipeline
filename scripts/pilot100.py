"""Deadline-safe private 100-image SAP pilot (not automatically training-ready)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sapipeline.batch.pilot import (
    build_pilot_plan,
    pack_pilot,
    run_pilot,
    save_pilot_plan,
)
from sapipeline.batch.runner import BatchRunError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SAP private 100-candidate technical QA pilot")
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--scenes", type=int, default=60)
    p.add_argument("--seed", type=int, default=650001)
    r = sub.add_parser("run")
    r.add_argument("--max-new", type=int, default=100)
    sub.add_parser("pack")
    parser.add_argument("--project-root", default=str(ROOT))
    args = parser.parse_args(argv)
    try:
        root = Path(args.project_root).absolute()
        if args.action == "plan":
            doc = build_pilot_plan(root, scenes=args.scenes, seed=args.seed)
            dst = save_pilot_plan(root, doc)
            print(f"PILOT PLAN: {len(doc['sources'])} new source scenes, "
                  f"{len(doc['runs'])} candidate variants, target=100; {dst}")
            print("Only bbox-local directional/elliptical glare; NO glass reflections; no auto approval")
        elif args.action == "run":
            made, previous, qa_failed = run_pilot(root, max_new=args.max_new)
            print(f"PILOT: NEW={len(made)} EXISTING_VERIFIED={previous} QA_REJECTED={len(qa_failed)}")
            print("Safe to resume. Original ZIPs, review decisions and old runs unchanged.")
        else:
            dst, count = pack_pilot(root)
            print(f"PRIVATE QA-PREVIEW ZIP: {count} images + SEG; {dst}")
            print("Technical QA PASS only. Not manually or training approved.")
        return 0
    except (BatchRunError, OSError, ValueError, TypeError) as exc:
        print(f"PILOT ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
