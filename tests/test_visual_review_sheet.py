"""Stage 8.1 visual QA: synthetic sources only, never private dataset assets."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from test_artifact_exporter import prepared

from sapipeline.export import export_artifact
from sapipeline.review import ReviewSheetError, build_review_sheet

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/render_review_sheet.py"


def _export(tmp_path: Path, run_id: str = "my_run_001") -> Path:
    values = prepared(tmp_path)
    values["run_id"] = run_id
    export_artifact(**values)
    return tmp_path / "outputs/sap_runs" / run_id


def _reviews(tmp_path: Path, run_id: str, *, sha: str = "", verdict: str = "approved",
             reviewer: str = "local-reviewer", complete: bool = True) -> None:
    path = tmp_path / "reports/qa_acceptance/reviews.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not sha:
        sha = hashlib.sha256((tmp_path / "outputs/sap_runs" / run_id / "manifest.json").read_bytes()).hexdigest()
    path.write_text(json.dumps({
        "schema_version": "sap-manual-reviews-v1",
        "reviews": [{"run_id": run_id, "manifest_sha256": sha, "verdict": verdict,
                     "reviewer": reviewer, "checks": {
                         "bbox_consistency": complete,
                         "glare_naturalness": complete,
                         "product_identity": complete,
                     }}],
    }), encoding="utf-8")


def test_readonly_sheet_has_expected_image_metrics_and_no_approval(tmp_path: Path) -> None:
    run = _export(tmp_path)
    snapshots = {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    result = build_review_sheet(tmp_path, "my_run_001")
    assert result.png.startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(io.BytesIO(result.png)) as img:
        assert img.size == (1880, 3160)
    m = result.metrics
    assert m["schema_version"] == "sap-visual-review-v1"
    assert m["manual_verdict"] == "needs_review"
    assert m["training_approved"] is False
    assert m["effect"]["kind"] == "elliptical"
    assert m["effect"]["seed"] == 42017
    assert m["metrics"]["changed_outside_mask"] == 0
    assert m["metrics"]["changed_pixels"] > 0
    assert m["metrics"]["mask_coverage_percent"] > 0
    assert sum(i["count"] for i in m["diff_histogram"]) == m["metrics"]["mask_active_pixels_16bit"]
    assert snapshots == {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    again = build_review_sheet(tmp_path, "my_run_001")
    assert again.png == result.png
    assert again.metrics == result.metrics


def test_cli_creates_private_sheet_json_but_no_extra_run_files(tmp_path: Path) -> None:
    run = _export(tmp_path)
    before = sorted(p.relative_to(run) for p in run.rglob("*") if p.is_file())
    cmd = [sys.executable, str(SCRIPT), "--project-root", str(tmp_path), "--run-id", "my_run_001"]
    first = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr
    review = tmp_path / "reports/visual_review/my_run_001"
    png = (review / "review_sheet.png").read_bytes()
    report = json.loads((review / "review_metrics.json").read_text())
    assert hashlib.sha256(png).hexdigest() == report["review_sheet_sha256"]
    assert len(list(review.iterdir())) == 2
    assert before == sorted(p.relative_to(run) for p in run.rglob("*") if p.is_file())
    second = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert second.returncode == 0 and "EXISTS" in second.stdout
    assert (review / "review_sheet.png").read_bytes() == png
    _reviews(tmp_path, "my_run_001")
    rerun = subprocess.run([*cmd, "--overwrite"], capture_output=True, text=True, check=False)
    assert rerun.returncode == 0, rerun.stderr
    assert json.loads((review / "review_metrics.json").read_text())["manual_verdict"] == "approved"


def test_review_requires_manifest_match_all_checks_and_reviewer(tmp_path: Path) -> None:
    _export(tmp_path)
    _reviews(tmp_path, "my_run_001", sha="a" * 64)
    assert build_review_sheet(tmp_path, "my_run_001").metrics["manual_verdict"] == "stale_review"
    _reviews(tmp_path, "my_run_001", complete=False)
    assert build_review_sheet(tmp_path, "my_run_001").metrics["manual_verdict"] == "incomplete_approval"
    _reviews(tmp_path, "my_run_001", reviewer=" ")
    assert build_review_sheet(tmp_path, "my_run_001").metrics["manual_verdict"] == "incomplete_approval"
    _reviews(tmp_path, "my_run_001", verdict="rejected")
    assert build_review_sheet(tmp_path, "my_run_001").metrics["manual_verdict"] == "rejected"


def test_reject_modified_image_with_wrong_hash(tmp_path: Path) -> None:
    run = _export(tmp_path)
    path = run / "images/my_run_001.png"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ReviewSheetError, match="SHA256"):
        build_review_sheet(tmp_path, "my_run_001")
    assert not (tmp_path / "reports/visual_review/my_run_001").exists()


def test_reject_outside_mask_edit_even_with_updated_hash(tmp_path: Path) -> None:
    run = _export(tmp_path)
    file = run / "images/my_run_001.png"
    with Image.open(file) as img:
        pic = np.array(img, copy=True)
    pic[0, 0, 0] = (int(pic[0, 0, 0]) + 7) % 255
    Image.fromarray(pic).save(file)
    manifest_file = run / "manifest.json"
    data = json.loads(manifest_file.read_text())
    data["output_sha256"]["image"] = hashlib.sha256(file.read_bytes()).hexdigest()
    manifest_file.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReviewSheetError, match="outside encoded mask"):
        build_review_sheet(tmp_path, "my_run_001")


def test_reject_path_traversal_even_with_valid_hash(tmp_path: Path) -> None:
    run = _export(tmp_path)
    manifest_file = run / "manifest.json"
    m = json.loads(manifest_file.read_text())
    m["output"]["image"] = "../../secrets.png"
    manifest_file.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ReviewSheetError, match="Unsafe"):
        build_review_sheet(tmp_path, "my_run_001")


def test_reject_manifest_target_bbox_mismatch(tmp_path: Path) -> None:
    run = _export(tmp_path)
    manifest_file = run / "manifest.json"
    m = json.loads(manifest_file.read_text())
    m["targets"][0]["bbox_xyxy_px"][0] += 1
    manifest_file.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ReviewSheetError, match="bbox / manifest"):
        build_review_sheet(tmp_path, "my_run_001")


def test_unknown_run_and_symlinked_report_blocked(tmp_path: Path) -> None:
    _export(tmp_path)
    with pytest.raises(ReviewSheetError, match="Unsafe"):
        build_review_sheet(tmp_path, "../invalid")
    target = tmp_path / "unsafe_reports"
    target.mkdir()
    folder = tmp_path / "reports"
    try:
        folder.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    proc = subprocess.run([sys.executable, str(SCRIPT), "--project-root", str(tmp_path),
                           "--run-id", "my_run_001"], capture_output=True, text=True, check=False)
    assert proc.returncode == 2
    assert list(target.iterdir()) == []


def test_all_renders_multiple_existing_runs(tmp_path: Path) -> None:
    args = prepared(tmp_path)
    args["run_id"] = "run_one"
    export_artifact(**args)
    args["run_id"] = "run_two"
    export_artifact(**args)
    proc = subprocess.run([sys.executable, str(SCRIPT), "--project-root", str(tmp_path),
                           "--all"], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    for name in ("run_one", "run_two"):
        path = tmp_path / "reports/visual_review" / name
        assert (path / "review_sheet.png").is_file()
        assert (path / "review_metrics.json").is_file()


def test_all_skips_unrelated_files_and_non_run_folders(tmp_path: Path) -> None:
    _export(tmp_path)
    runs = tmp_path / "outputs/sap_runs"
    (runs / "zzz_notes.txt").write_text("not an artifact")
    (runs / "zzz.backup").mkdir()
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--project-root", str(tmp_path), "--all"],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "SKIPPED non-directory entry: zzz_notes.txt" in proc.stdout
    assert "SKIPPED non-run directory: zzz.backup" in proc.stdout
    assert (tmp_path / "reports/visual_review/my_run_001/review_sheet.png").is_file()
    assert (runs / "zzz_notes.txt").read_text() == "not an artifact"
    assert (runs / "zzz.backup").is_dir()


def test_all_rejects_incomplete_or_invalid_export_folders(tmp_path: Path) -> None:
    _export(tmp_path)
    runs = tmp_path / "outputs/sap_runs"
    broken = runs / "zzz_incomplete"
    broken.mkdir()
    cmd = [sys.executable, str(SCRIPT), "--project-root", str(tmp_path), "--all"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 2
    assert "Incomplete exported run" in proc.stderr
    assert "zzz_incomplete" in proc.stderr
    assert not (tmp_path / "reports/visual_review/my_run_001").exists()
    broken.rmdir()

    suspicious = runs / "zzz.invalid"
    suspicious.mkdir()
    (suspicious / "manifest.json").write_text("{}")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 2
    assert "Invalid run ID with manifest: zzz.invalid" in proc.stderr
    assert not (tmp_path / "reports/visual_review/my_run_001").exists()
