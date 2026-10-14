"""Stage-8 audit uses generated synthetic ZIP/images and private temp directories only."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image
from test_artifact_exporter import prepared

from sapipeline.dataset import build_index, load_indexed_pair
from sapipeline.export import export_artifact
from sapipeline.qa import validate_artifact
from sapipeline.qa.acceptance import (
    AcceptanceError,
    audit_batch,
    make_review_template,
    parse_reviews,
)
from sapipeline.selection import select_rois


def one_bundle(tmp_path: Path) -> Path:
    args = prepared(tmp_path)
    return export_artifact(**args).directory


def approvals(root: Path, *, min_outputs: int = 1, min_unique_sources: int = 1) -> dict:
    result = audit_batch(root, min_outputs=min_outputs,
                         min_unique_sources=min_unique_sources)
    template = make_review_template(result)
    for entry in template["reviews"]:
        entry["verdict"] = "approved"
        entry["reviewer"] = "human-reviewer-01"
        entry["checks"] = {key: True for key in entry["checks"]}
    return template


def test_real_stage7_synthetic_bundle_is_valid_but_pending(tmp_path: Path) -> None:
    one_bundle(tmp_path)
    audit = audit_batch(tmp_path)
    assert audit.report["status"] == "NOT_READY"
    assert audit.report["totals"]["outputs"] == 1
    assert audit.report["totals"]["valid"] == 1
    assert audit.report["totals"]["unique_sources"] == 1
    assert audit.report["totals"]["manually_approved"] == 0
    assert audit.report["runs"][0]["checks"]["source_seg_same"]
    assert not audit.report["training_approved"]
    template = make_review_template(audit)
    assert template["reviews"][0]["verdict"] == "needs_review"
    assert set(template["reviews"][0]["checks"].values()) == {False}


def test_manual_approval_is_bound_to_manifest(tmp_path: Path) -> None:
    run = one_bundle(tmp_path)
    reviews = approvals(tmp_path)
    audit = audit_batch(tmp_path, reviews=reviews, min_outputs=1,
                        min_unique_sources=1)
    assert audit.accepted
    assert audit.report["status"] == "ACCEPTED"
    assert audit.report["stage8_accepted"]
    assert audit.report["training_approved"] is False
    reviews["reviews"][0]["manifest_sha256"] = "0" * 64
    bad = audit_batch(tmp_path, reviews=reviews, min_outputs=1,
                      min_unique_sources=1)
    assert bad.report["status"] == "FAILED"
    assert "stale_manual_review" in bad.report["runs"][0]["issues"]
    # The Stage-7 bundle remains immutable even when reviews are edited.
    assert (run / "manifest.json").exists()


def test_incomplete_approval_does_not_pass(tmp_path: Path) -> None:
    one_bundle(tmp_path)
    reviews = approvals(tmp_path)
    reviews["reviews"][0]["checks"]["glare_naturalness"] = False
    result = audit_batch(tmp_path, reviews=reviews, min_outputs=1,
                         min_unique_sources=1)
    assert result.report["status"] == "FAILED"
    assert "incomplete_manual_approval" in result.report["runs"][0]["issues"]


@pytest.mark.parametrize("invalid", ["missing", "tampered", "mask_leak", "wrong_target"])
def test_corrupt_bundle_is_fail_closed(tmp_path: Path, invalid: str) -> None:
    run = one_bundle(tmp_path)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if invalid == "missing":
        (run / manifest["output"]["annotation"]).unlink()
    elif invalid == "tampered":
        (run / manifest["output"]["image"]).write_bytes(b"changed")
    elif invalid == "mask_leak":
        # SHA is recomputed by attacker; audit still enforces pixel invariants.
        image_path = run / manifest["output"]["image"]
        with Image.open(image_path) as im:
            rgb = np.asarray(im).copy()
        rgb[0, 0, 0] ^= 2
        Image.fromarray(rgb).save(image_path)
        manifest["output_sha256"]["image"] = hashlib.sha256(image_path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
    elif invalid == "wrong_target":
        manifest["targets"][0]["bbox_xyxy_px"] = [0, 0, 1, 1]
        manifest_path.write_text(json.dumps(manifest))
    report = audit_batch(tmp_path, min_outputs=1, min_unique_sources=1).report
    assert report["status"] == "FAILED"
    assert report["totals"]["invalid"] == 1


def test_twenty_outputs_do_not_substitute_five_distinct_sources(tmp_path: Path) -> None:
    args = prepared(tmp_path)
    for i in range(20):
        args["run_id"] = f"test_{i:03d}"
        export_artifact(**args)
    reviews = approvals(tmp_path, min_outputs=20)
    report = audit_batch(tmp_path, reviews=reviews).report
    assert report["totals"]["outputs"] == 20
    assert report["totals"]["unique_sources"] == 1
    assert report["gates"]["minimum_outputs"]
    assert not report["gates"]["minimum_unique_sources"]
    assert report["status"] == "NOT_READY"


def test_rejected_and_orphan_reviews_fail(tmp_path: Path) -> None:
    one_bundle(tmp_path)
    reviews = approvals(tmp_path)
    reviews["reviews"][0]["verdict"] = "rejected"
    reviews["reviews"][0]["reason"] = "Unnatural glare"
    rejected = audit_batch(tmp_path, reviews=reviews, min_outputs=1,
                           min_unique_sources=1)
    assert rejected.report["status"] == "FAILED"
    reviews["reviews"][0]["verdict"] = "approved"
    reviews["reviews"].append({"run_id": "unknown", "verdict": "needs_review"})
    orphan = audit_batch(tmp_path, reviews=reviews, min_outputs=1,
                         min_unique_sources=1)
    assert not orphan.report["gates"]["no_orphan_reviews"]
    assert orphan.report["status"] == "FAILED"


def test_duplicate_manual_reviews_are_refused() -> None:
    reviews = {"schema_version": "sap-manual-reviews-v1", "reviews": [
        {"run_id": "x", "verdict": "needs_review"},
        {"run_id": "x", "verdict": "approved"},
    ]}
    with pytest.raises(AcceptanceError, match="duplicate"):
        parse_reviews(reviews)


def test_symlinked_bundle_is_not_followed(tmp_path: Path) -> None:
    run = one_bundle(tmp_path)
    try:
        (run.parent / "imposter").symlink_to(run, target_is_directory=True)
    except OSError:
        pytest.skip("OS does not permit symlinks")
    report = audit_batch(tmp_path, min_outputs=1, min_unique_sources=1).report
    assert report["status"] == "FAILED"
    assert report["totals"]["invalid"] == 1


def test_review_template_contains_only_manifest_bound_pending(tmp_path: Path) -> None:
    one_bundle(tmp_path)
    data = make_review_template(audit_batch(tmp_path))
    manifest = (tmp_path / "outputs/sap_runs/my_run_001/manifest.json").read_bytes()
    assert data["reviews"][0]["manifest_sha256"] == hashlib.sha256(manifest).hexdigest()
    assert all(not x["checks"]["glare_naturalness"] for x in data["reviews"])


def test_twenty_manually_approved_outputs_pass_when_min_unique_is_one(tmp_path: Path) -> None:
    args = prepared(tmp_path)
    for i in range(20):
        args["run_id"] = f"accepted_{i:03d}"
        export_artifact(**args)
    reviews = approvals(tmp_path, min_outputs=20, min_unique_sources=1)
    result = audit_batch(tmp_path, reviews=reviews, min_outputs=20, min_unique_sources=1)
    assert result.accepted
    assert result.report["totals"]["manually_approved"] == 20


def test_cli_init_audit_and_refuses_overwrite(tmp_path: Path, capsys) -> None:
    from scripts.qa_acceptance import main

    one_bundle(tmp_path)
    assert main(["init", "--project-root", str(tmp_path)]) == 0
    review_path = tmp_path / "reports/qa_acceptance/reviews.json"
    reviews = json.loads(review_path.read_text())
    assert reviews["reviews"][0]["verdict"] == "needs_review"
    assert main(["init", "--project-root", str(tmp_path)]) == 2
    assert main(["audit", "--project-root", str(tmp_path)]) == 1
    assert "NOT_READY" in capsys.readouterr().out
    report = tmp_path / "reports/qa_acceptance/acceptance.json"
    assert report.exists()
    assert main(["audit", "--project-root", str(tmp_path)]) == 2
    assert main(["audit", "--project-root", str(tmp_path), "--overwrite-report"]) == 1
    assert not (tmp_path / "outputs/sap_runs/my_run_001/acceptance.json").exists()
    assert main(["audit", "--project-root", str(tmp_path),
                 "--report", "README.md", "--overwrite-report"]) == 2


def test_bundle_manifest_cannot_escape_bundle(tmp_path: Path) -> None:
    run = one_bundle(tmp_path)
    file = run / "manifest.json"
    manifest = json.loads(file.read_text())
    manifest["output"]["image"] = "../../private.txt"
    file.write_text(json.dumps(manifest))
    result = audit_batch(tmp_path, min_outputs=1, min_unique_sources=1)
    assert result.report["status"] == "FAILED"
    assert result.report["totals"]["invalid"] == 1


def test_twenty_distinct_source_bundle_approvals_pass_stage8(tmp_path: Path) -> None:
    """Positive gate: five different source JPEGs, four variants each, all reviewed."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname='synthetic-checkout'\n")
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    archive = raw / "for_learning.zip"
    with ZipFile(archive, "w") as zipped:
        for index in range(5):
            name = f"scene_{index:02d}"
            buffer = BytesIO()
            Image.new("RGB", (48, 32), (40 + index * 21, 70, 110)).save(buffer, "JPEG")
            zipped.writestr(f"img/{name}.jpg", buffer.getvalue())
            zipped.writestr(f"seg/{name}.seg", json.dumps({
                "filename": f"{name}.jpg",
                "selectedList": [{"name": "fictional_item", "rect": {
                    "x": 7, "y": 5, "width": 16, "height": 14}}],
            }))
    index, _ = build_index({"learning": archive})
    index_file = tmp_path / "data/work/index/dataset_index.json"
    index_file.parent.mkdir(parents=True)
    index_file.write_text(json.dumps(index))
    for image_index, entry in enumerate(index["items"]):
        pair = load_indexed_pair(index, entry, data_dir=raw)
        orig = np.asarray(pair.image).astype(np.float32) / 255
        mask = np.zeros(orig.shape[:2], dtype=np.float32)
        mask[5:19, 7:23] = 0.7
        modified = orig.copy()
        modified[mask > 0] = np.minimum(1.0, modified[mask > 0] + 0.1)
        selection = select_rois(pair.annotations, mode="by_index", instance_index=0)
        verdict = validate_artifact(orig, modified, mask, pair.annotations, selection)
        assert verdict.passed
        effect = {"schema_version": "sap-glare-v1", "kind": "elliptical", "seed": 42017,
                  "angle_deg": -25., "radius_scale": 0.85, "aspect_ratio": 0.35,
                  "falloff": 1.25, "intensity": 0.55, "center_jitter": 0.1,
                  "image_size_px": [48, 32],
                  "changed_pixels": verdict.metrics["changed_pixels"],
                  "effect_mask_active_pixels": verdict.metrics["mask_active_pixels"]}
        for repeat in range(4):
            export_artifact(
                project_root=tmp_path, run_id=f"scene_{image_index:02d}_{repeat:02d}",
                original=orig, modified=modified, mask=mask,
                annotations=pair.annotations, selection=selection,
                qa_pass=True, qa_report=verdict.to_json(),
                source_context={key: entry[key] for key in (
                    "archive", "image_member", "annotation_member", "image_sha256",
                    "annotation_sha256", "split")},
                effect_summary=json.dumps(effect),
                mask_summary=json.dumps({"mask_kind": "soft_rect", "image_size_px": [48, 32],
                                         "selected_indices": [0], "pad_px": 0,
                                         "feather_px": 16.0, "strength": 1.0}),
            )
    reviews = approvals(tmp_path, min_outputs=20, min_unique_sources=5)
    report = audit_batch(tmp_path, reviews=reviews).report
    assert report["status"] == "ACCEPTED"
    assert report["stage8_accepted"] is True
    assert report["training_approved"] is False
    assert report["totals"]["outputs"] == 20
    assert report["totals"]["unique_sources"] == 5
    assert report["totals"]["manually_approved"] == 20
