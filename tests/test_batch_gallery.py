"""Stage 8.2 offline gallery tests use only synthetic, self-created artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sapipeline.review.gallery import BatchGalleryError, build_batch_gallery


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _put(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _create_run(root: Path, name: str, *, split: str = "learning", seed: int = 42017,
                source: str = "a" * 64, kind: str = "projected_strips") -> None:
    run = root / "outputs/sap_runs" / name
    run.mkdir(parents=True)
    raw = np.full((20, 30, 3), 80, dtype=np.uint8)
    original = run / "original.png"
    mod = run / "images" / f"{name}.png"
    for file, data in ((original, raw), (mod, np.clip(raw.astype(int) + 22, 0, 255).astype(np.uint8))):
        file.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(data, mode="RGB").save(file)
    out = {
        "original": "original.png", "image": f"images/{name}.png",
        "annotation": f"annotations/{name}.seg", "effect_mask": f"masks/{name}.png",
        "qa_report": "qa.json",
    }
    mask = run / out["effect_mask"]
    mask.parent.mkdir(parents=True)
    Image.new("I;16", (30, 20), 65535).save(mask)
    _put(run / out["annotation"], b'{"selectedList": []}')
    qa = {"schema_version": "sap-qa-v1", "status": "PASS", "checks": {}}
    _put(run / "qa.json", json.dumps(qa).encode())
    effect = {"schema_version": "sap-glass-reflection-v1", "kind": kind,
              "seed": seed, "intensity": .7}
    manifest = {
        "schema_version": "sap-artifact-v1", "run_id": name,
        "source": {"image_sha256": source, "split": split},
        "output": out, "output_sha256": {k: _sha((run / v).read_bytes()) for k, v in out.items()},
        "qa": {"status": "PASS", "manual_review_required": True, "training_approved": False},
        "image_size_px": [30, 20], "effects": [effect],
    }
    m_path = run / "manifest.json"
    _put(m_path, json.dumps(manifest, ensure_ascii=False).encode())
    review = root / "reports/visual_review" / name
    review.mkdir(parents=True)
    sheet = review / "review_sheet.png"
    Image.new("RGB", (160, 200), "white").save(sheet)
    metrics = {
        "schema_version": "sap-visual-review-v1", "run_id": name,
        "manifest_sha256": _sha(m_path.read_bytes()),
        "review_sheet_sha256": _sha(sheet.read_bytes()),
        "technical_qa_status": "PASS", "source_split": split,
        "source_sha256": source, "effect": effect,
        "metrics": {"mask_coverage_percent": 43, "changed_fraction_percent": 38,
                    "changed_outside_mask": 0, "mean_abs_diff_all_channels_255": 12.4},
    }
    _put(review / "review_metrics.json", json.dumps(metrics).encode())


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")
    _create_run(tmp_path, "run_alpha")
    return tmp_path


def test_gallery_contains_cards_photos_metrics_filters_and_relative_links(project: Path) -> None:
    result = build_batch_gallery(project)
    text = result.html_page.decode("utf-8")
    assert "<!doctype html>" in text
    assert "run_alpha" in text
    assert "thumbs/run_alpha_modified.jpg" in text
    assert "../../outputs/sap_runs/run_alpha/images/run_alpha.png" in text
    assert "../visual_review/run_alpha/review_sheet.png" in text
    assert "All effects" in text and "All reviews" in text and "All splits" in text
    assert "Not training approval".lower() in text.lower()
    assert "38.00%" in text
    assert result.summary["runs"] == 1
    assert result.summary["training_approved"] is False
    assert result.summary["stage8_accepted"] is False
    assert set(result.thumbnails) == {
        "thumbs/run_alpha_original.jpg", "thumbs/run_alpha_modified.jpg"}
    for data in result.thumbnails.values():
        assert data.startswith(b"\xff\xd8")
    assert not (project / "reports/batch_gallery").exists()


def test_exported_sources_are_unchanged_by_gallery(project: Path) -> None:
    run = project / "outputs/sap_runs/run_alpha"
    before = {str(p.relative_to(run)): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    build_batch_gallery(project)
    assert before == {str(p.relative_to(run)): p.read_bytes() for p in run.rglob("*") if p.is_file()}


def test_fails_modified_export_hash(project: Path) -> None:
    file = project / "outputs/sap_runs/run_alpha/images/run_alpha.png"
    file.write_bytes(file.read_bytes() + b"corrupted")
    with pytest.raises(BatchGalleryError, match="SHA256"):
        build_batch_gallery(project)


def test_fails_stale_review_sheet(project: Path) -> None:
    file = project / "reports/visual_review/run_alpha/review_sheet.png"
    file.write_bytes(file.read_bytes() + b"corrupted")
    with pytest.raises(BatchGalleryError, match="Stale or modified visual report"):
        build_batch_gallery(project)


def test_missing_sheet_gives_actionable_failure(project: Path) -> None:
    (project / "reports/visual_review/run_alpha/review_sheet.png").unlink()
    with pytest.raises(BatchGalleryError, match="Missing or unsafe review_sheet"):
        build_batch_gallery(project)


def test_reviewer_verdict_uses_current_reviews_not_cached_sheet(project: Path) -> None:
    manifest = project / "outputs/sap_runs/run_alpha/manifest.json"
    review = {"run_id": "run_alpha", "manifest_sha256": _sha(manifest.read_bytes()),
              "verdict": "approved", "reviewer": "tester",
              "checks": {"bbox_consistency": True, "glare_naturalness": True,
                         "product_identity": True}}
    dst = project / "reports/qa_acceptance/reviews.json"
    _put(dst, json.dumps({"schema_version": "sap-manual-reviews-v1", "reviews": [review]}).encode())
    assert build_batch_gallery(project).summary["records"][0]["manual"] == "approved"
    review["checks"]["glare_naturalness"] = False
    _put(dst, json.dumps({"schema_version": "sap-manual-reviews-v1", "reviews": [review]}).encode())
    assert build_batch_gallery(project).summary["records"][0]["manual"] == "incomplete_approval"
    review["manifest_sha256"] = "0" * 64
    _put(dst, json.dumps({"schema_version": "sap-manual-reviews-v1", "reviews": [review]}).encode())
    assert build_batch_gallery(project).summary["records"][0]["manual"] == "stale_review"


def test_multiple_runs_duplicate_source_split_warning_and_zip_skip(project: Path) -> None:
    _create_run(project, "run_beta", split="verification")
    (project / "outputs/sap_runs/random_archive.zip").write_bytes(b"placeholder")
    result = build_batch_gallery(project)
    assert result.summary["runs"] == 2
    assert result.summary["unique_sources"] == 1
    assert result.summary["cross_split_duplicate_sources"] == 1
    assert len(result.thumbnails) == 4
    assert "Cross-split duplicates" in result.html_page.decode()


def test_reject_unsafe_run_and_symlink(project: Path) -> None:
    bad = project / "outputs/sap_runs/unsafe.name"
    bad.mkdir()
    (bad / "manifest.json").write_text("{}")
    with pytest.raises(BatchGalleryError, match="Invalid run directory"):
        build_batch_gallery(project)
    (bad / "manifest.json").unlink()
    bad.rmdir()
    symlink = project / "outputs/sap_runs/another_run"
    try:
        symlink.symlink_to(project / "outputs/sap_runs/run_alpha", target_is_directory=True)
    except OSError:
        pytest.skip("Symlinks unavailable on this system")
    with pytest.raises(BatchGalleryError, match="symlink"):
        build_batch_gallery(project)


def test_reject_manifest_traversal(project: Path) -> None:
    m_path = project / "outputs/sap_runs/run_alpha/manifest.json"
    m = json.loads(m_path.read_text())
    m["output"]["image"] = "../../outside.png"
    m_path.write_text(json.dumps(m))
    with pytest.raises(BatchGalleryError, match="Unsafe output path"):
        build_batch_gallery(project)


def test_refuses_untrusted_review_schema(project: Path) -> None:
    _put(project / "reports/qa_acceptance/reviews.json", b'{"reviews": []}')
    with pytest.raises(BatchGalleryError, match="Invalid manual reviews schema"):
        build_batch_gallery(project)


def test_cli_atomic_publish_and_overwrite(project: Path) -> None:
    from scripts.build_batch_gallery import publish

    result, output, runs = publish(project)
    assert result == "CREATED" and runs == 1 and output.is_file()
    old_html = output.read_bytes()
    old_run = (project / "outputs/sap_runs/run_alpha/original.png").read_bytes()
    with pytest.raises(BatchGalleryError, match="--overwrite"):
        publish(project)
    result2, output2, count2 = publish(project, overwrite=True)
    assert (result2, output2, count2) == (result, output, runs)
    assert output.read_bytes() == old_html
    assert old_run == (project / "outputs/sap_runs/run_alpha/original.png").read_bytes()
    assert len(list((project / "reports/batch_gallery/thumbs").glob("*.jpg"))) == 2
    assert not list((project / "reports").glob(".gallery_pending_*"))


def test_publish_invalid_input_does_not_replace_previous_gallery(project: Path) -> None:
    from scripts.build_batch_gallery import publish

    _, index, _ = publish(project)
    before = index.read_bytes()
    (project / "reports/visual_review/run_alpha/review_metrics.json").unlink()
    with pytest.raises(BatchGalleryError, match="Missing or unsafe JSON"):
        publish(project, overwrite=True)
    assert index.read_bytes() == before
