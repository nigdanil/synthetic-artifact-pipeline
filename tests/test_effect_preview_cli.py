"""Private end-to-end CLI effect preview checks on artificial ZIP bytes only."""

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

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/preview_effect.py"


def _module():
    spec = importlib.util.spec_from_file_location("preview_effect_local", SCRIPT)
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


def test_end_to_end_private_effect_preview(tmp_path: Path, prepared, monkeypatch) -> None:
    module = _module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    data, zip_path, index_file = prepared
    out = tmp_path / "reports/effect_previews"
    argv = ["--index", str(index_file), "--data-dir", str(data),
            "--out-dir", str(out), "--mode", "random_n", "--count", "2",
            "--seed", "42017", "--mask-kind", "soft_rect", "--effect-kind", "soft_glow"]
    before_zip = zip_path.read_bytes()
    assert module.main(argv) == 0
    assert len(list(out.glob("*_effect.png"))) == 1
    assert len(list(out.glob("*_mask.png"))) == 1
    assert len(list(out.glob("*_overlay.png"))) == 1
    assert len(list(out.glob("*.json"))) == 1
    with Image.open(next(out.glob("*_effect.png"))) as effect:
        assert effect.size == (100, 80) and effect.mode == "RGB"
        values = np.asarray(effect)
        # top-left outside ROI should remain unchanged from the decoded JPEG,
        # while a selected region becomes brighter.
        baseline = values[0, 0].astype(int)
        assert int(values[20, 20, 0]) >= int(baseline[0])
        assert int(values[20, 20, 1]) >= int(baseline[1])
    report = json.loads(next(out.glob("*.json")).read_text(encoding="utf-8"))
    assert report["parameters"]["effect_kind"] == "soft_glow"
    assert report["qa"]["status"] == "pending_manual_review"
    assert zip_path.read_bytes() == before_zip


def test_refuses_overwrite_without_force(tmp_path: Path, prepared, monkeypatch) -> None:
    module = _module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    data, _, index_file = prepared
    out = tmp_path / "reports/effect_previews"
    argv = ["--index", str(index_file), "--data-dir", str(data), "--out-dir", str(out)]
    assert module.main(argv) == 0
    assert module.main(argv) == 2
    assert module.main(argv + ["--force"]) == 0


def test_outside_private_reports_is_forbidden(tmp_path: Path, prepared, monkeypatch) -> None:
    module = _module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    data, _, index_file = prepared
    bad = tmp_path / "public_out"
    assert module.main(["--index", str(index_file), "--data-dir", str(data),
                        "--out-dir", str(bad)]) == 2
    assert not bad.exists()
