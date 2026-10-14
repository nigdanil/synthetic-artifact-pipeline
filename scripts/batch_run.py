"""Stage 8.4: create a private deterministic plan or export up to N new runs.

Examples (Git Bash):
  python scripts/batch_run.py plan --scenes 4
  python scripts/batch_run.py run --max-new 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sapipeline.batch import BatchRunError, build_plan, execute_plan, save_plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Private SAP Stage-8.4 serial batch runner")
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("plan", help="Create private plan (no source or output writes)")
    p.add_argument("--scenes", type=int, default=4, help="Number of NEW unique learning scenes")
    p.add_argument("--seed", type=int, default=42017)
    r = sub.add_parser("run", help="Export next N new runs, with QA and immutable bundles")
    r.add_argument("--max-new", type=int, default=1)
    parser.add_argument("--project-root", default=str(ROOT))
    args = parser.parse_args(argv)
    try:
        root = Path(args.project_root).absolute()
        if args.action == "plan":
            doc = build_plan(root, scenes=args.scenes, seed=args.seed)
            plan_file = save_plan(root, doc)
            print(f"PLAN CREATED (private): scenes={len(doc['sources'])}; "
                  f"planned_runs={len(doc['runs'])}; {plan_file}")
            print("Glass runs REQUIRE scene-specific glass_quad_norm before exporting.")
        else:
            created, skipped = execute_plan(root, max_new=args.max_new)
            print(f"BATCH: new={len(created)}; already_present={len(skipped)}")
            for run_id in created:
                print(f"EXPORTED QA PASS (not manually approved): {run_id}")
            print("All Stage-7 bundles stay immutable; manual reviews unchanged.")
        return 0
    except (BatchRunError, OSError, ValueError, TypeError) as exc:
        print(f"Batch ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
