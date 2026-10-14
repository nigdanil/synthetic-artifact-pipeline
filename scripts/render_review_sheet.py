"""Private Stage 8.1 comparison PNG + metrics JSON for Stage-7 artifact bundles.

Examples:
  python scripts/render_review_sheet.py --run-id glass_reflection_0001
  python scripts/render_review_sheet.py --all
  python scripts/render_review_sheet.py --run-id glass_reflection_0001 --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sapipeline.review import ReviewSheetError, build_review_sheet

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")


def _discover_runs(runs: Path) -> list[str]:
    """Ignore unrelated entries, but fail closed for damaged or unsafe run folders."""
    names: list[str] = []
    for entry in sorted(runs.iterdir(), key=lambda item: item.name):
        if entry.is_symlink():
            raise ReviewSheetError(f"Unsafe symlink in runs directory: {entry.name}")
        if not entry.is_dir():
            print(f"SKIPPED non-directory entry: {entry.name}")
            continue
        manifest = entry / "manifest.json"
        if not _ID.fullmatch(entry.name):
            if manifest.exists() or manifest.is_symlink():
                raise ReviewSheetError(f"Invalid run ID with manifest: {entry.name}")
            print(f"SKIPPED non-run directory: {entry.name}")
            continue
        if manifest.is_symlink() or not manifest.is_file():
            raise ReviewSheetError(f"Incomplete exported run without safe manifest: {entry.name}")
        names.append(entry.name)
    return names


def _save(root: Path, run_id: str, *, overwrite: bool) -> str:
    reports = root / "reports"
    parent = reports / "visual_review"
    output = parent / run_id
    if any(p.is_symlink() for p in (reports, parent, output)):
        raise ReviewSheetError("Symlinked private report path")
    if output.exists() and not overwrite:
        return "EXISTS"
    if output.exists() and not output.is_dir():
        raise ReviewSheetError("Review target is not a directory")
    # All source validation/rendering happens before any writes.
    sheet = build_review_sheet(root, run_id)
    data = dict(sheet.metrics)
    data["review_sheet_sha256"] = hashlib.sha256(sheet.png).hexdigest()
    parent.mkdir(parents=True, exist_ok=True)
    temp: Path | None = None
    try:
        temp = Path(tempfile.mkdtemp(prefix=".visual_review_", dir=parent))
        (temp / "review_sheet.png").write_bytes(sheet.png)
        (temp / "review_metrics.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if output.exists():
            # Regeneration is explicitly requested; replace ONLY derived review files.
            for filename in ("review_sheet.png", "review_metrics.json"):
                os.replace(temp / filename, output / filename)
            temp.rmdir()
            temp = None
        else:
            temp.rename(output)
            temp = None
    finally:
        if temp is not None:
            for item in temp.iterdir():
                item.unlink()
            temp.rmdir()
    return "CREATED"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only SAP Stage 8.1 visual review sheets")
    choose = parser.add_mutually_exclusive_group(required=True)
    choose.add_argument("--run-id", help="One exported SAP run identifier")
    choose.add_argument("--all", action="store_true", help="Render all Stage-7 run directories")
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--overwrite", action="store_true", help="Refresh derived reports only")
    args = parser.parse_args(argv)
    try:
        root = Path(args.project_root).resolve(strict=True)
        if not (root / "pyproject.toml").is_file():
            raise ReviewSheetError("project_root must be an SAP checkout")
        runs = root / "outputs/sap_runs"
        if runs.is_symlink() or not runs.is_dir():
            raise ReviewSheetError("Missing or unsafe runs directory")
        names = [args.run_id] if args.run_id else _discover_runs(runs)
        if not names:
            raise ReviewSheetError("No exported runs found")
        for run_id in names:
            if not _ID.fullmatch(run_id):
                raise ReviewSheetError(f"Invalid run ID: {run_id}")
            action = _save(root, run_id, overwrite=args.overwrite)
            print(f"{run_id}: {action} -- {root / 'reports/visual_review' / run_id}")
        print("Visual reports only; original bundles unchanged; manual approval unchanged")
        return 0
    except (ReviewSheetError, OSError, ValueError) as exc:
        print(f"Visual review ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
