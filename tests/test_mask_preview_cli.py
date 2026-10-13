"""Private end-to-end CLI mask preview checks on artificial ZIP bytes only."""

from __future__ import annotations

import importlib.util
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest
from PIL import Image

from sapipeline.dataset import build_index

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/preview_mask.py"


def _module():
    spec = importlib.util.spec_from_file_location("preview_mask_local", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def prepared(tmp_path: Path):
    source_dir = tmp_path / "data"
    source_dir.mkdir()
    source = source_dir / "learning.zip"
    im = Image.new("RGB", (100, 80), (140, 130, 125))
    raw = BytesIO()
    im.save(raw, format="JPEG")
    annotations = {"filename": "synthetic.jpg", "selectedList": [
        {"name": "test_package", "rect": {"x": 10, "y": 12, "width": 20, "height": 30}},
        {"name": "another_package", "rect": {"x": 45, "y": 25, "width": 18, "height": 35}},
        {"name": "glint", "rect": {"x": 75, "y": 10, "width": 15, "height": 20}},
    ]}
    with ZipFile(source, "w", ZIP_DEFLATED) as zipped:
        zipped.writestr("img/synthetic.jpg", raw.getvalue())
        zipped.writestr("seg/synthetic.seg", json.dumps(annotations).encode("utf-8"))
    index, _ = build_index({"learning": source})
    index_file = tmp_path / "index.json"
    index_file.write_text(json.dumps(index), encoding="utf-8")
    return source_dir, source, index_file


def test_end_to_end_private_mask_overlay_report_and_refusal_to_overwrite(
    tmp_path: Path, prepared, monkeypatch
) -> None:
    module = _module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    data, zip_path, index_file = prepared
    out = tmp_path / "reports/mask_previews"
    argv = ["--index", str(index_file), "--data-dir", str(data),
            "--out-dir", str(out), "--mode", "random_n", "--count", "2",
            "--seed", "42017", "--mask-kind", "soft_rect", "--feather-px", "5"]
    before_zip = zip_path.read_bytes()
    assert module.main(argv) == 0
    pngs = list(out.glob("*.png"))
    reports = list(out.glob("*.json"))
    assert len(pngs) == 2 and len(reports) == 1
    mask_path = next(out.glob("*_mask.png"))
    preview_path = next(out.glob("*_overlay.png"))
    with Image.open(mask_path) as mask:
        values = np.asarray(mask)
        assert mask.size == (100, 80)
        assert mask.mode == "L"
        assert values[0, 0] == 0
        assert 0 < values[12, 10] < 255
        assert values[25, 20] == 255
        assert values[45, 52] == 255
        assert values[15, 80] == 0
    with Image.open(preview_path) as image:
        assert image.size == (100, 80) and image.mode == "RGB"
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["mask_definition"] == "bbox-effect-weights-not-segmentation"
    assert report["parameters"]["indices"] == [0, 1]
    assert report["qa"]["status"] == "pending_manual_review"
    assert zip_path.read_bytes() == before_zip
    original_bytes = mask_path.read_bytes()
    assert module.main(argv) == 2
    assert mask_path.read_bytes() == original_bytes
    assert module.main(argv + ["--force"]) == 0
    assert mask_path.read_bytes() == original_bytes


def test_explicit_rect_only_target(tmp_path: Path, prepared, monkeypatch) -> None:
    module = _module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    data, _, index_file = prepared
    out = tmp_path / "reports/masks"
    args = ["--index", str(index_file), "--data-dir", str(data),
            "--out-dir", str(out), "--mode", "by_index", "--instance-index", "1",
            "--mask-kind", "rect", "--strength", "0.5", "--pad-px", "1"]
    assert module.main(args) == 0
    with Image.open(next(out.glob("*_mask.png"))) as im:
        pixels = np.asarray(im)
        assert pixels[25, 45] == 128
        assert pixels[24, 44] == 128
        assert pixels[12, 10] == 0
        assert pixels[24, 44] != 0


@pytest.mark.parametrize("flags", [
    ["--mask-kind", "rect", "--strength", "0"],
    ["--mask-kind", "soft_rect", "--feather-px", "-1"],
    ["--mask-kind", "soft_rect", "--feather-px", "nan"],
    ["--mask-kind", "rect", "--pad-px", "-1"],
    ["--mode", "by_index"],
    ["--mode", "random_n", "--count", "3"],
    ["--mode", "by_class", "--class-name", "glint"],
    ["--mode", "random_one", "--allow-service-targets"],
])
def test_cli_bad_params_never_create_outputs(tmp_path: Path, prepared, monkeypatch, flags) -> None:
    module = _module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    data, _, index_file = prepared
    out = tmp_path / "reports/mask_previews"
    argv = ["--index", str(index_file), "--data-dir", str(data),
            "--out-dir", str(out)] + flags
    assert module.main(argv) == 2
    assert not out.exists()


def test_outside_private_reports_is_forbidden(tmp_path: Path, prepared, monkeypatch) -> None:
    module = _module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    data, _, index_file = prepared
    bad = tmp_path / "public_out"
    assert module.main(["--index", str(index_file), "--data-dir", str(data),
                        "--out-dir", str(bad)]) == 2
    assert not bad.exists()
