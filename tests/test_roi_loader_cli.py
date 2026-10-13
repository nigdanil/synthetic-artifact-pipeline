"""Read-only pair loading and CLI selection tests with synthetic ZIPs."""

from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from PIL import Image

from sapipeline.dataset import (
    DatasetIndexError,
    build_index,
    choose_index_entry,
    load_indexed_pair,
    read_private_index,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/preview_roi.py"


def create_zip(path: Path, *, stem: str = "test001", image: bytes | None = None,
               annotation: bytes | None = None, orientation: int | None = None) -> None:
    if image is None:
        buffer = BytesIO()
        pixels = Image.new("RGB", (100, 80), (80, 120, 140))
        if orientation is not None:
            exif = Image.Exif()
            exif[274] = orientation
            pixels.save(buffer, format="JPEG", exif=exif)
        else:
            pixels.save(buffer, format="JPEG")
        image = buffer.getvalue()
    if annotation is None:
        annotation = json.dumps({"filename": f"{stem}.jpg", "selectedList": [
            {"name": "sample_ice_cream", "gender": False,
             "rect": {"x": 10, "y": 5, "width": 22, "height": 30}},
            {"name": "sample_ice_cream", "gender": False,
             "rect": {"x": 35, "y": 10, "width": 15, "height": 28}},
            {"name": "glint", "rect": {"x": 65, "y": 15, "width": 12, "height": 18}},
        ]}).encode("utf-8")
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as source:
        source.writestr(f"img/{stem}.jpg", image)
        source.writestr(f"seg/{stem}.seg", annotation)


@pytest.fixture
def prepared(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    create_zip(data / "learning.zip")
    index, _ = build_index({"learning": data / "learning.zip"})
    index_path = tmp_path / "private-index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    return tmp_path, data, index, index_path


def test_load_encoded_rgb_and_annotation_without_mutating_zip(prepared):
    _, data, index, _ = prepared
    path = data / "learning.zip"
    before = path.read_bytes()
    entry = choose_index_entry(index, split="learning")
    pair = load_indexed_pair(index, entry, data_dir=data)
    assert (pair.image.width, pair.image.height) == (100, 80)
    assert pair.image.mode == "RGB"
    assert [x.instance_index for x in pair.annotations.objects] == [0, 1, 2]
    assert pair.annotations.objects[0].bbox_xyxy_px == (10, 5, 32, 35)
    assert pair.annotations.objects[0].raw_fields == {"gender": False}
    assert path.read_bytes() == before


def test_index_reader_fails_closed_on_missing_or_invalid(tmp_path: Path, prepared):
    with pytest.raises(DatasetIndexError, match="Cannot read"):
        read_private_index(tmp_path / "missing.json")
    broken = tmp_path / "invalid.json"
    broken.write_text('{"items": []}')
    with pytest.raises(DatasetIndexError, match="Unsupported"):
        read_private_index(broken)
    assert len(read_private_index(prepared[3])["items"]) == 1


def test_choose_explicit_or_default_and_reject_missing(prepared):
    _, _, index, _ = prepared
    exact = choose_index_entry(index, split="learning", image_member="img/test001.jpg")
    assert exact["image_member"] == "img/test001.jpg"
    assert choose_index_entry(index, split="learning") == exact
    assert choose_index_entry(index, split="learning", sample_index=0) == exact
    with pytest.raises(DatasetIndexError, match="out of range"):
        choose_index_entry(index, split="learning", sample_index=1)
    with pytest.raises(DatasetIndexError, match="non-negative"):
        choose_index_entry(index, split="learning", sample_index=-1)
    with pytest.raises(DatasetIndexError, match="combined"):
        choose_index_entry(index, split="learning", image_member="img/test001.jpg",
                           sample_index=1)
    with pytest.raises(DatasetIndexError, match="exactly one"):
        choose_index_entry(index, split="verification", image_member="img/test001.jpg")
    with pytest.raises(DatasetIndexError, match="Unsafe"):
        choose_index_entry(index, split="learning", image_member="img/../sample.jpg")
    with pytest.raises(DatasetIndexError, match="No unique"):
        choose_index_entry(index, split="verification")


def test_auto_choice_never_uses_duplicate_even_within_split(prepared):
    _, _, index, _ = prepared
    other = deepcopy(index["items"][0])
    other["image_member"] = "img/duplicate.jpg"
    other["annotation_member"] = "seg/duplicate.seg"
    index["items"].append(other)
    with pytest.raises(DatasetIndexError, match="No unique"):
        choose_index_entry(index, split="learning")
    assert choose_index_entry(index, split="learning", image_member="img/test001.jpg")


@pytest.mark.parametrize("mutation", [
    lambda row: row.update({"image_sha256": "0" * 64}),
    lambda row: row.update({"annotation_sha256": "0" * 64}),
    lambda row: row.update({"width": 200}),
    lambda row: row.update({"bbox_count_declared": 99}),
    lambda row: row.update({"archive": "../learning.zip"}),
    lambda row: row.update({"image_member": "img/../test001.jpg"}),
    lambda row: row.update({"annotation_member": "seg/mismatch.seg"}),
    lambda row: row.update({"exif_orientation": 65537}),
])
def test_index_tampering_cannot_produce_valid_pair(prepared, mutation):
    _, data, index, _ = prepared
    modified = deepcopy(index["items"][0])
    mutation(modified)
    with pytest.raises(DatasetIndexError):
        load_indexed_pair(index, modified, data_dir=data)


def test_real_nondefault_exif_is_refused(tmp_path: Path):
    source = tmp_path / "orient.zip"
    create_zip(source, orientation=6)
    index, _ = build_index({"learning": source})
    with pytest.raises(DatasetIndexError, match="Non-default EXIF"):
        load_indexed_pair(index, index["items"][0], data_dir=tmp_path)


def _cli_module():
    spec = importlib.util.spec_from_file_location("preview_roi_local", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cli_preview_and_report_is_local_deterministic(prepared, monkeypatch):
    tmp_path, data, _, index_path = prepared
    module = _cli_module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    out = tmp_path / "reports/roi_previews"
    args = ["--index", str(index_path), "--data-dir", str(data), "--out-dir", str(out),
            "--mode", "random_n", "--count", "2", "--seed", "42017"]
    assert module.main(args) == 0
    png = list(out.glob("*.png"))
    reports = list(out.glob("*.json"))
    assert len(png) == len(reports) == 1
    with Image.open(png[0]) as preview:
        assert preview.size == (100, 80)
        assert preview.mode == "RGB"
    selection = json.loads(reports[0].read_text(encoding="utf-8"))
    assert selection["selection"]["instance_indices"] == [0, 1]
    assert len([obj for obj in selection["objects"] if obj["selected"]]) == 2
    assert selection["qa"]["status"] == "pending_manual_review"
    assert module.main(args) == 2  # No unintentional overwrite.
    before = png[0].read_bytes()
    assert module.main(args + ["--force"]) == 0
    assert before == png[0].read_bytes()


@pytest.mark.parametrize("flags", [
    ["--mode", "by_index"],
    ["--mode", "by_class"],
    ["--mode", "random_n", "--count", "3"],
    ["--mode", "random_one", "--instance-index", "0"],
    ["--mode", "random_n", "--count", "-2"],
    ["--mode", "random_one", "--seed", "-1"],
    ["--mode", "by_index", "--instance-index", "2"],
    ["--mode", "by_class", "--class-name", "glint"],
    ["--mode", "random_n", "--allow-service-targets"],
])
def test_cli_rejects_bad_selection_without_writes(prepared, monkeypatch, flags):
    tmp_path, data, _, index_path = prepared
    module = _cli_module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    out = tmp_path / "reports/roi_previews"
    assert module.main(["--index", str(index_path), "--data-dir", str(data),
                        "--out-dir", str(out)] + flags) == 2
    assert not out.exists()


def test_cli_explicit_index_and_service_optin(prepared, monkeypatch):
    tmp_path, data, _, index_path = prepared
    module = _cli_module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    out = tmp_path / "reports/roi_previews"
    assert module.main(["--index", str(index_path), "--data-dir", str(data),
                        "--out-dir", str(out), "--mode", "by_index", "--instance-index", "2",
                        "--allow-service-targets"]) == 0
    obj = json.loads(next(out.glob("*.json")).read_text(encoding="utf-8"))
    assert obj["selection"]["instance_indices"] == [2]


def test_cli_refuses_outputs_outside_private_reports(prepared, monkeypatch):
    tmp_path, data, _, index_path = prepared
    module = _cli_module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    assert module.main(["--index", str(index_path), "--data-dir", str(data),
                        "--out-dir", str(tmp_path / "public")]) == 2
    assert not (tmp_path / "public").exists()
