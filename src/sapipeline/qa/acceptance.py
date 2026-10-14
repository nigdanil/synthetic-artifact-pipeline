"""Read-only Stage-8 acceptance audit for private, QA-gated SAP run bundles.

A technical PASS does not imply training approval. The batch gate requires
explicit, manifest-bound human review of every exported candidate.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sapipeline.annotations import parse_seg
from sapipeline.dataset import choose_index_entry, load_indexed_pair, read_private_index

_RUN_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_REQUIRED = {"original", "image", "annotation", "effect_mask", "qa_report"}
_REVIEW_CHECKS = ("product_identity", "glare_naturalness", "bbox_consistency")


class AcceptanceError(ValueError):
    """Unsafe arguments or malformed review file; never silently approve."""


@dataclass(frozen=True, slots=True)
class AcceptanceResult:
    report: dict[str, Any]

    @property
    def accepted(self) -> bool:
        return self.report["status"] == "ACCEPTED"

    def to_json(self) -> str:
        return json.dumps(self.report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise AcceptanceError("expected JSON object")
    return obj


def _bundle_file(bundle: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name:
        raise AcceptanceError("unsafe artifact path")
    relative = Path(name)
    if relative.is_absolute() or any(part in (".", "..", "") for part in relative.parts):
        raise AcceptanceError("unsafe artifact path")
    file = bundle / relative
    # Do not follow arbitrary symlinks even when they resolve within the project.
    if any(p.is_symlink() for p in (file, *file.parents) if p != bundle.parent):
        raise AcceptanceError("symlinked artifact path")
    if not file.is_file():
        raise AcceptanceError("missing artifact file")
    return file


def _run_check(bundle: Path, project_root: Path, index: dict) -> dict[str, Any]:
    issues: list[str] = []
    result: dict[str, Any] = {"run_id": bundle.name, "checks": {}, "issues": issues}

    def require(name: str, ok: bool) -> None:
        result["checks"][name] = bool(ok)
        if not ok:
            issues.append(name)

    if bundle.is_symlink() or not bundle.is_dir() or not _RUN_ID.fullmatch(bundle.name):
        require("safe_run_directory", False)
        return result
    try:
        manifest_file = _bundle_file(bundle, "manifest.json")
        manifest_bytes = manifest_file.read_bytes()
        manifest = json.loads(manifest_bytes)
        if not isinstance(manifest, dict):
            raise AcceptanceError("manifest must be an object")
        result["manifest_sha256"] = _digest(manifest_bytes)
        require("manifest_schema", manifest.get("schema_version") == "sap-artifact-v1")
        require("manifest_run_id", manifest.get("run_id") == bundle.name)
        require("geometry_unchanged", manifest.get("geometry_changed") is False)
        require("training_not_preapproved", manifest.get("qa", {}).get("training_approved") is False)
        require("label_policy", manifest.get("label_policy") == "source_seg_geometry_unchanged_no_new_glint_labels")
        source = manifest.get("source")
        if not isinstance(source, dict):
            raise AcceptanceError("missing source provenance")
        source_sha = source.get("image_sha256")
        split = source.get("split")
        require("source_sha256", isinstance(source_sha, str) and bool(_SHA.fullmatch(source_sha)))
        require("source_split", split in ("learning", "verification"))
        result["source_sha256"] = source_sha if isinstance(source_sha, str) else ""
        result["source_split"] = split if isinstance(split, str) else ""
        output = manifest.get("output")
        hashes = manifest.get("output_sha256")
        if not isinstance(output, dict) or set(output) != _REQUIRED or not isinstance(hashes, dict):
            raise AcceptanceError("incomplete output manifest")
        require("hash_key_consistency", set(hashes) == _REQUIRED)
        paths = {key: _bundle_file(bundle, relative) for key, relative in output.items()}
        require("distinct_artifact_paths", len(set(paths.values())) == len(paths))
        for key, file in paths.items():
            expected = hashes.get(key)
            require("sha256_" + key, isinstance(expected, str) and bool(_SHA.fullmatch(expected))
                    and _digest(file.read_bytes()) == expected)
        qa = _read_json(paths["qa_report"])
        require("technical_qa_pass", qa.get("schema_version") == "sap-qa-v1"
                and qa.get("status") == "PASS" and not qa.get("failures")
                and all(v is True for v in qa.get("checks", {}).values()))
        require("manifest_qa_pass", manifest.get("qa", {}).get("status") == "PASS")
        with Image.open(paths["original"]) as pic:
            original = np.asarray(pic.convert("RGB"))
            source_mode = pic.mode
        with Image.open(paths["image"]) as pic:
            modified = np.asarray(pic.convert("RGB"))
            modified_mode = pic.mode
        with Image.open(paths["effect_mask"]) as pic:
            mask = np.asarray(pic)
        require("rgb_encoding", source_mode == modified_mode == "RGB")
        width, height = original.shape[1], original.shape[0]
        require("image_geometry", original.shape == modified.shape
                and manifest.get("image_size_px") == [width, height])
        require("mask_16bit_geometry", mask.shape == (height, width)
                and mask.dtype.kind in "iu" and mask.dtype.itemsize >= 2
                and bool(np.all((mask >= 0) & (mask <= 65535))))
        if original.shape == modified.shape and mask.shape == (height, width):
            changed = np.any(original != modified, axis=2)
            active = mask > 0
            require("visible_effect", bool((changed & active).any()))
            require("outside_mask_unchanged", not bool((changed & ~active).any()))
            metrics = qa.get("metrics", {})
            require("qa_changed_pixels_consistent", isinstance(metrics, dict)
                    and isinstance(metrics.get("changed_pixels"), int)
                    and abs(metrics["changed_pixels"] - int((changed & active).sum()))
                    <= max(3, int(changed.sum() * 0.02)))
            result["observed_changed_pixels"] = int(changed.sum())
            result["observed_mask_coverage_fraction"] = round(float(active.mean()), 8)
        annotation = parse_seg(paths["annotation"].read_bytes(), image_width=width,
                               image_height=height,
                               image_filename=Path(output["image"]).name)
        targets = manifest.get("targets", [])
        if not isinstance(targets, list):
            raise AcceptanceError("invalid target list")
        lookup = {obj.instance_index: obj for obj in annotation.objects}
        require("target_bbox_and_class", bool(targets) and all(
            isinstance(target, dict)
            and type(target.get("instance_index")) is int
            and target["instance_index"] in lookup
            and target.get("class_name") == lookup[target["instance_index"]].class_name
            and target.get("bbox_xyxy_px") == list(lookup[target["instance_index"]].bbox_xyxy_px)
            for target in targets
        ))
        # Verify original source JPEG + all SEG fields against the private ZIP/index.
        if result["checks"].get("source_sha256") and result["checks"].get("source_split"):
            entry = choose_index_entry(index, split=split, image_member=source.get("image_member"))
            for key in ("archive", "image_member", "annotation_member", "image_sha256",
                        "annotation_sha256", "split"):
                require("source_provenance_" + key, entry.get(key) == source.get(key))
            pair = load_indexed_pair(index, entry, data_dir=project_root / "data/raw")
            require("source_rgb_same", np.array_equal(np.asarray(pair.image.convert("RGB")), original))
            require("source_seg_same", annotation.objects == pair.annotations.objects
                    and annotation.source_seg == {
                        **pair.annotations.source_seg,
                        "filename": Path(output["image"]).name,
                    })
            duplicate_splits = {
                item.get("split") for item in index.get("items", [])
                if item.get("image_sha256") == source_sha
            }
            require("no_index_cross_split_duplicate", len(duplicate_splits) == 1)
    except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
        # Error text is intentionally not included: ZIP paths and class names are private.
        require("bundle_readable_and_complete", False)
        result["error_type"] = type(exc).__name__
    return result


def parse_reviews(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if data.get("schema_version") != "sap-manual-reviews-v1":
        raise AcceptanceError("invalid review schema")
    entries = data.get("reviews")
    if not isinstance(entries, list):
        raise AcceptanceError("reviews must be a list")
    result: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("run_id"), str):
            raise AcceptanceError("review has no run_id")
        run_id = entry["run_id"]
        if not _RUN_ID.fullmatch(run_id) or run_id in result:
            raise AcceptanceError("duplicate or unsafe review run_id")
        if entry.get("verdict") not in ("approved", "rejected", "needs_review"):
            raise AcceptanceError("invalid review verdict")
        result[run_id] = entry
    return result


def make_review_template(audit: AcceptanceResult) -> dict[str, Any]:
    """Generate only pending reviews; never manufacture human approvals."""
    return {
        "schema_version": "sap-manual-reviews-v1",
        "reviews": [
            {"run_id": run["run_id"], "manifest_sha256": run.get("manifest_sha256", ""),
             "verdict": "needs_review", "reviewer": "", "reason": "",
             "checks": {key: False for key in _REVIEW_CHECKS}}
            for run in audit.report["runs"] if "manifest_sha256" in run
        ],
    }


def audit_batch(
    project_root: Path,
    *,
    reviews: dict[str, Any] | None = None,
    min_outputs: int = 20,
    min_unique_sources: int = 5,
) -> AcceptanceResult:
    """Reopen every run; verify SHA, private origin and all manual review gates."""
    if type(min_outputs) is not int or min_outputs < 1:
        raise AcceptanceError("min_outputs must be positive")
    if type(min_unique_sources) is not int or min_unique_sources < 1:
        raise AcceptanceError("min_unique_sources must be positive")
    root = Path(project_root).resolve(strict=True)
    if not (root / "pyproject.toml").is_file():
        raise AcceptanceError("project_root must be a SAP checkout")
    runs_folder = root / "outputs/sap_runs"
    if runs_folder.is_symlink():
        raise AcceptanceError("runs folder must not be a symlink")
    entries = sorted(runs_folder.iterdir(), key=lambda p: p.name) if runs_folder.is_dir() else []
    # A regular file (e.g. a ZIP backup) is not an exported run.
    # Keep symlinks and ALL directories: _run_check must reject unsafe/incomplete runs.
    entries = [p for p in entries if p.is_symlink() or p.is_dir()]
    if entries:
        index = read_private_index(root / "data/work/index/dataset_index.json")
    else:
        index = {"items": []}
    parsed = parse_reviews(reviews) if reviews is not None else {}
    run_reports = []
    used_sources: dict[str, set[str]] = {}
    for entry in entries:
        run = _run_check(entry, root, index)
        review = parsed.get(entry.name)
        reasons = run["issues"]
        manifest_sha = run.get("manifest_sha256")
        valid_review = bool(review and manifest_sha
                            and review.get("manifest_sha256") == manifest_sha)
        if review and not valid_review:
            reasons.append("stale_manual_review")
        verdict = review.get("verdict") if valid_review else "needs_review"
        if verdict == "approved":
            checks = review.get("checks")
            valid_approval = (isinstance(review.get("reviewer"), str)
                              and bool(review["reviewer"].strip())
                              and isinstance(checks, dict)
                              and all(checks.get(check) is True for check in _REVIEW_CHECKS))
            if not valid_approval:
                reasons.append("incomplete_manual_approval")
        elif verdict == "rejected":
            reasons.append("manually_rejected")
        run["manual_verdict"] = verdict
        run["manual_approved"] = verdict == "approved" and not reasons
        source_sha = run.get("source_sha256")
        split = run.get("source_split")
        if isinstance(source_sha, str) and _SHA.fullmatch(source_sha) and split in ("learning", "verification"):
            used_sources.setdefault(source_sha, set()).add(split)
        run_reports.append(run)
    cross_split = sorted(sha for sha, splits in used_sources.items() if len(splits) > 1)
    source_count = len(used_sources)
    invalid_count = sum(bool(run["issues"]) for run in run_reports)
    approved_count = sum(run["manual_approved"] for run in run_reports)
    rejected_count = sum(run["manual_verdict"] == "rejected" for run in run_reports)
    gates = {
        "minimum_outputs": len(run_reports) >= min_outputs,
        "minimum_unique_sources": source_count >= min_unique_sources,
        "all_artifacts_valid": invalid_count == 0,
        "all_manually_approved": bool(run_reports) and approved_count == len(run_reports),
        "no_cross_split_leakage": not cross_split,
        "no_orphan_reviews": not (set(parsed) - {r["run_id"] for r in run_reports}),
    }
    hard_failure = invalid_count > 0 or rejected_count > 0 or bool(cross_split) or not gates["no_orphan_reviews"]
    status = "FAILED" if hard_failure else ("ACCEPTED" if all(gates.values()) else "NOT_READY")
    return AcceptanceResult({
        "schema_version": "sap-batch-acceptance-v1", "status": status,
        "stage8_accepted": status == "ACCEPTED", "training_approved": False,
        "manual_review_required": True,
        "requirements": {"min_outputs": min_outputs, "min_unique_sources": min_unique_sources},
        "totals": {"outputs": len(run_reports), "unique_sources": source_count,
                   "valid": len(run_reports) - invalid_count, "invalid": invalid_count,
                   "manually_approved": approved_count, "manually_rejected": rejected_count,
                   "awaiting_review": len(run_reports) - approved_count - rejected_count,
                   "cross_split_duplicate_sources": len(cross_split)},
        "gates": gates, "runs": run_reports,
    })
