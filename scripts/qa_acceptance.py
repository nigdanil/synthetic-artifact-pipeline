"""Stage-8 private batch audit and manual review preparation (Git Bash compatible).

Usage: python scripts/qa_acceptance.py init|audit --project-root ...
Never writes inside artifact bundles or the public repository tree.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sapipeline.qa.acceptance import (
    AcceptanceError,
    audit_batch,
    make_review_template,
)


def _private_path(root: Path, filename: str) -> Path:
    path = Path(filename)
    if not path.is_absolute():
        path = root / path
    reports = root / "reports"
    if reports.is_symlink() or path.is_symlink():
        raise AcceptanceError("symlinked report path")
    canonical = path.resolve(strict=False)
    if not canonical.is_relative_to(reports.resolve(strict=False)) or canonical == reports:
        raise AcceptanceError("reports must be inside project_root/reports")
    return canonical


def _write_private(path: Path, payload: dict, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise AcceptanceError(f"report already exists: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temp: Path | None = None
    try:
        fd, name = tempfile.mkstemp(prefix=".acceptance_", suffix=".tmp", dir=path.parent)
        temp = Path(name)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        if path.exists() and not overwrite:
            raise AcceptanceError(f"report already exists: {path.name}")
        temp.replace(path)
        temp = None
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Private SAP Stage-8 acceptance")
    parser.add_argument("action", choices=["init", "audit"])
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--reviews", default="reports/qa_acceptance/reviews.json")
    parser.add_argument("--report", default="reports/qa_acceptance/acceptance.json")
    parser.add_argument("--min-outputs", type=int, default=20)
    parser.add_argument("--min-unique-sources", type=int, default=5)
    parser.add_argument("--overwrite-report", action="store_true")
    args = parser.parse_args(argv)
    try:
        root = Path(args.project_root).resolve(strict=True)
        review_file = _private_path(root, args.reviews)
        report_file = _private_path(root, args.report)
        if args.action == "init":
            if review_file.exists():
                raise AcceptanceError("review file already exists; do not overwrite human decisions")
            result = audit_batch(root, min_outputs=args.min_outputs,
                                 min_unique_sources=args.min_unique_sources)
            template = make_review_template(result)
            _write_private(review_file, template)
            print(f"Prepared {len(template['reviews'])} pending reviews: {review_file}")
            print("All initial verdicts are needs_review, not approved")
            return 0
        reviews = None
        if review_file.exists():
            reviews = json.loads(review_file.read_text(encoding="utf-8"))
        result = audit_batch(root, reviews=reviews,
                             min_outputs=args.min_outputs,
                             min_unique_sources=args.min_unique_sources)
        _write_private(report_file, result.report, overwrite=args.overwrite_report)
        totals = result.report["totals"]
        print(f"Stage 8: {result.report['status']}; outputs={totals['outputs']}; "
              f"valid={totals['valid']}; approved={totals['manually_approved']}; "
              f"unique_sources={totals['unique_sources']}")
        print(f"Private report: {report_file}")
        return 0 if result.accepted else 1
    except (AcceptanceError, OSError, ValueError, TypeError) as exc:
        print(f"Stage 8 error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
