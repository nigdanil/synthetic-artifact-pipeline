"""Private, manifest-bound manual QA decisions for a local-only review UI.

The exported bundles remain immutable. Manual decisions never imply training approval.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from sapipeline.review.gallery import BatchGalleryError, build_batch_gallery

_REVIEW_CHECKS = ("product_identity", "glare_naturalness", "bbox_consistency")
_RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_SCHEMA = "sap-manual-reviews-v1"


class ManualReviewError(ValueError):
    """User-friendly, fail-closed review update error."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _paths(project_root: Path) -> tuple[Path, Path]:
    root = Path(project_root).resolve(strict=True)
    if not (root / "pyproject.toml").is_file():
        raise ManualReviewError("project_root must be a SAP checkout")
    reports = root / "reports"
    directory = reports / "qa_acceptance"
    review_file = directory / "reviews.json"
    if any(p.is_symlink() for p in (reports, directory, review_file)):
        raise ManualReviewError("Symlinked review location not allowed")
    return root, review_file


def _load(path: Path) -> tuple[dict[str, Any], str]:
    if not path.exists():
        return {"schema_version": _SCHEMA, "reviews": []}, _sha(b"")
    if path.is_symlink() or not path.is_file():
        raise ManualReviewError("Unsafe reviews.json")
    content = path.read_bytes()
    if len(content) > 2_000_000:
        raise ManualReviewError("reviews.json exceeds size limit")
    try:
        data = json.loads(content)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ManualReviewError("Malformed reviews.json") from exc
    if (not isinstance(data, dict) or data.get("schema_version") != _SCHEMA
            or not isinstance(data.get("reviews"), list)):
        raise ManualReviewError("Invalid manual reviews schema")
    seen: set[str] = set()
    for review in data["reviews"]:
        if not isinstance(review, dict) or not isinstance(review.get("run_id"), str):
            raise ManualReviewError("Invalid review entry")
        run_id = review["run_id"]
        if not _RUN.fullmatch(run_id) or run_id in seen:
            raise ManualReviewError("Unsafe or duplicate review ID")
        if review.get("verdict") not in ("approved", "rejected", "needs_review"):
            raise ManualReviewError("Invalid saved verdict")
        if not isinstance(review.get("manifest_sha256"), str):
            raise ManualReviewError("Invalid saved manifest hash")
        seen.add(run_id)
    return data, _sha(content)


def read_state(project_root: Path) -> dict[str, Any]:
    """Return validated exported records with manifest-bound pending decisions."""
    root, review_path = _paths(project_root)
    try:
        gallery = build_batch_gallery(root)
    except BatchGalleryError as exc:
        raise ManualReviewError(str(exc)) from exc
    reviews, etag = _load(review_path)
    saved = {entry["run_id"]: entry for entry in reviews["reviews"]}
    records = []
    for row in gallery.summary["records"]:
        run_id = row["run_id"]
        manifest = root / "outputs" / "sap_runs" / run_id / "manifest.json"
        manifest_sha = _sha(manifest.read_bytes())
        review = saved.get(run_id)
        stale = bool(review and review["manifest_sha256"] != manifest_sha)
        records.append({
            "run_id": run_id,
            "qa": row["qa"],
            "effect": row["effect"],
            "source_split": row["source_split"],
            "changed_fraction_percent": row["changed_fraction_percent"],
            "mask_coverage_percent": row["mask_coverage_percent"],
            "manifest_sha256": manifest_sha,
            "review_status": "stale_review" if stale else row["manual"],
            # Stale decisions are not pre-selected for another image version.
            "review": (review if review and not stale else {
                "verdict": "needs_review", "reviewer": "", "reason": "",
                "checks": {key: False for key in _REVIEW_CHECKS},
            }),
        })
    return {
        "schema_version": "sap-review-ui-v1",
        "runs": records,
        "reviews_etag": etag,
        "totals": {"runs": len(records), "approved": sum(r["review_status"] == "approved" for r in records)},
        "training_approved": False,
    }


def _clean_text(value: Any, name: str, maximum: int) -> str:
    if not isinstance(value, str) or len(value) > maximum or any(ord(c) < 32 for c in value):
        raise ManualReviewError(f"Invalid {name}")
    return value.strip()


def _decision(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {
        "run_id", "manifest_sha256", "reviews_etag", "verdict", "reviewer", "reason", "checks",
    }:
        raise ManualReviewError("Unexpected review fields")
    run_id = raw["run_id"]
    if not isinstance(run_id, str) or not _RUN.fullmatch(run_id):
        raise ManualReviewError("Invalid run_id")
    verdict = raw["verdict"]
    if verdict not in ("approved", "rejected", "needs_review"):
        raise ManualReviewError("Invalid verdict")
    checks = raw["checks"]
    if not isinstance(checks, dict) or set(checks) != set(_REVIEW_CHECKS) or any(type(v) is not bool for v in checks.values()):
        raise ManualReviewError("Exactly three boolean checks are required")
    reviewer = _clean_text(raw["reviewer"], "reviewer", 80)
    reason = _clean_text(raw["reason"], "reason", 1000)
    if verdict in ("approved", "rejected") and not reviewer:
        raise ManualReviewError("Reviewer is required")
    if verdict == "approved" and not all(checks.values()):
        raise ManualReviewError("All three checks must be true to approve")
    if verdict == "rejected" and len(reason) < 8:
        raise ManualReviewError("Rejection reason must contain at least 8 characters")
    if not isinstance(raw["manifest_sha256"], str) or not _SHA.fullmatch(raw["manifest_sha256"]):
        raise ManualReviewError("Invalid manifest SHA-256")
    if not isinstance(raw["reviews_etag"], str) or not _SHA.fullmatch(raw["reviews_etag"]):
        raise ManualReviewError("Invalid review version")
    return {
        "run_id": run_id, "manifest_sha256": raw["manifest_sha256"],
        "verdict": verdict, "reviewer": reviewer, "reason": reason, "checks": checks,
    }


def save_review(project_root: Path, raw: Any) -> dict[str, Any]:
    """Explicit one-run save, with optimistic lock, backup and atomic replacement."""
    entry = _decision(raw)
    root, path = _paths(project_root)
    state = read_state(root)  # Re-check all immutable source hashes and reviews.
    if raw["reviews_etag"] != state["reviews_etag"]:
        raise ManualReviewError("Reviews changed since page load; refresh before saving")
    match = next((r for r in state["runs"] if r["run_id"] == entry["run_id"]), None)
    if match is None:
        raise ManualReviewError("Unknown run ID")
    if entry["manifest_sha256"] != match["manifest_sha256"]:
        raise ManualReviewError("Manifest changed; refresh and review the new image")
    if entry["verdict"] == "approved" and match["qa"] != "PASS":
        raise ManualReviewError("Technical QA must PASS before approval")
    reviews, current_etag = _load(path)
    if current_etag != raw["reviews_etag"]:
        raise ManualReviewError("Reviews changed during save")
    entries = reviews["reviews"]
    for i, previous in enumerate(entries):
        if previous["run_id"] == entry["run_id"]:
            entries[i] = entry
            break
    else:
        entries.append(entry)
    target_dir = path.parent
    if target_dir.parent.is_symlink() or target_dir.is_symlink() or path.is_symlink():
        raise ManualReviewError("Symlinked review output is forbidden")
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(reviews, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    # Preserve prior human decisions in a local backup before a replacement.
    if path.exists():
        previous_bytes = path.read_bytes()
        if _sha(previous_bytes) != current_etag:
            raise ManualReviewError("Reviews changed before backup")
        backup_dir = target_dir / "backups"
        if backup_dir.is_symlink():
            raise ManualReviewError("Symlinked backup location is forbidden")
        backup_dir.mkdir(exist_ok=True)
        backup = backup_dir / f"reviews_{current_etag}.json"
        if not backup.exists():
            with backup.open("xb") as f:
                f.write(previous_bytes)
                f.flush()
                os.fsync(f.fileno())
    fd, temp = tempfile.mkstemp(prefix=".pending_reviews_", suffix=".json", dir=target_dir)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        if path.exists() and _sha(path.read_bytes()) != current_etag:
            raise ManualReviewError("Reviews changed before atomic replacement")
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return {"run_id": entry["run_id"], "verdict": entry["verdict"], "reviews_etag": _sha(payload), "training_approved": False}
