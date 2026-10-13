"""Synthetic CLI integration checks; reports stay in Git-ignored private area."""

from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from scripts import validate_annotations


def make_zip(path: Path, *, invalid: bool = False) -> Path:
    buffer = BytesIO()
    Image.new("RGB", (64, 48), (25, 45, 65)).save(buffer, "JPEG")
    rect = {"x": 5, "y": 7, "width": 15, "height": 12}
    if invalid:
        rect["width"] = 70
    annotation = {
        "filename": "dummy.jpg",
        "selectedList": [{"name": "fictional_item", "gender": False, "rect": rect}],
    }
    with ZipFile(path, "w") as zipped:
        zipped.writestr("img/dummy.jpg", buffer.getvalue())
        zipped.writestr("seg/dummy.seg", json.dumps(annotation))
    return path


def test_cli_generates_private_report_and_preview(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(validate_annotations, "ROOT", tmp_path)
    a, b = make_zip(tmp_path / "learning.zip"), make_zip(tmp_path / "verification.zip")
    report = tmp_path / "reports/check.json"
    args = ["--learning", str(a), "--verification", str(b), "--report", str(report),
            "--preview-count", "1"]
    assert validate_annotations.main(args) == 0
    summary = json.loads(report.read_text(encoding="utf-8"))
    assert summary["valid_pairs"] == 2
    assert summary["valid_bbox_entries"] == 2
    assert summary["preview_count"] == 1
    assert summary["preview_examples"][0]["objects"][0]["bbox_xyxy_px"] == [5, 7, 20, 19]
    preview = tmp_path / "reports/annotation_previews/learning_001.png"
    with Image.open(preview) as image:
        assert image.size == (64, 48)
    assert validate_annotations.main(args) == 2  # refuse existing report without --force
    assert validate_annotations.main([*args, "--force"]) == 0


def test_cli_fails_closed_on_bad_geometry_without_writing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(validate_annotations, "ROOT", tmp_path)
    a = make_zip(tmp_path / "learning.zip", invalid=True)
    b = make_zip(tmp_path / "verification.zip")
    report = tmp_path / "reports/check.json"
    result = validate_annotations.main(["--learning", str(a), "--verification", str(b),
                                        "--report", str(report), "--preview-count", "1"])
    assert result == 2
    assert not report.exists()
    assert not (tmp_path / "reports/annotation_previews").exists()


def test_cli_rejects_public_output_location(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(validate_annotations, "ROOT", tmp_path)
    assert validate_annotations.main(["--report", str(tmp_path / "README.md")]) == 2
