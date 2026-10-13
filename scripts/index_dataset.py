"""Index local source ZIP archives without extracting or changing original data."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from sapipeline.dataset import DatasetIndexError, build_index

ROOT = Path(__file__).resolve().parents[1]


def _protected_output(path: Path, allowed_root: Path) -> Path:
    result = path.expanduser().resolve()
    if not result.is_relative_to(allowed_root.resolve()) or result == allowed_root.resolve():
        raise DatasetIndexError(f"Output must be under {allowed_root}: {path}")
    return result


def _stage_json(destination: Path, data: dict) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Private report/index; never world-readable on POSIX when permissions apply.
    fd, name = tempfile.mkstemp(prefix=".index-", suffix=".tmp", dir=destination.parent)
    path = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Index learning and verification ZIP datasets (read-only)"
    )
    parser.add_argument("--learning", type=Path, default=ROOT / "data/raw/for_learning.zip")
    parser.add_argument("--verification", type=Path, default=ROOT / "data/raw/for_verification.zip")
    parser.add_argument("--index", type=Path, default=ROOT / "data/work/index/dataset_index.json")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/dataset_index_summary.json")
    parser.add_argument(
        "--force", action="store_true", help="Replace previously generated local index/report"
    )
    args = parser.parse_args()
    staged: list[Path] = []
    try:
        index_path = _protected_output(args.index, ROOT / "data/work")
        report_path = _protected_output(args.report, ROOT / "reports")
        for destination in (index_path, report_path):
            if destination.exists() and not args.force:
                raise DatasetIndexError(
                    f"Output already exists: {destination}. Use --force to rebuild"
                )
        index, summary = build_index({"learning": args.learning, "verification": args.verification})
        staged = [_stage_json(index_path, index), _stage_json(report_path, summary)]
        for temporary, destination in zip(staged, (index_path, report_path), strict=True):
            os.replace(temporary, destination)
        print(
            f"Indexed pairs: {summary['total_pairs']} | "
            f"unique images: {summary['unique_images_sha256']}"
        )
        print(
            f"Duplicate groups: {summary['exact_duplicate_groups']} | "
            f"cross-split: {summary['cross_split_duplicate_groups']}"
        )
        print(f"Index: {index_path}")
        print(f"Report: {report_path}")
        return 0
    except (DatasetIndexError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        for temporary in staged:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
