"""Private, synthetic-only integration checks for atomic Stage-7 export."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image

from sapipeline.annotations import parse_seg
from sapipeline.dataset import build_index, load_indexed_pair
from sapipeline.export import ArtifactExportError, export_artifact
from sapipeline.qa import validate_artifact
from sapipeline.selection import select_rois


def prepared(tmp_path: Path) -> dict:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fake-project'\n")
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    buffer = BytesIO()
    image = Image.new("RGB", (60, 40), (100, 120, 140))
    image.save(buffer, format="JPEG")
    source_seg = {
        "filename": "test.jpg", "unknown_root": {"keep": [1, "hello"]},
        "selectedList": [{"name": "fictional_product", "gender": False,
                          "rect": {"x": 10, "y": 5, "width": 20, "height": 20},
                          "unknown": {"preserve": True}}],
    }
    archive = raw / "for_learning.zip"
    with ZipFile(archive, "w") as zipped:
        zipped.writestr("img/test.jpg", buffer.getvalue())
        zipped.writestr("seg/test.seg", json.dumps(source_seg).encode())
    index, _ = build_index({"learning": archive})
    index_path = tmp_path / "data/work/index/dataset_index.json"
    index_path.parent.mkdir(parents=True)
    index_path.write_text(json.dumps(index))
    entry = index["items"][0]
    pair = load_indexed_pair(index, entry, data_dir=raw)
    orig = np.asarray(pair.image).astype(np.float32) / 255
    mask = np.zeros(orig.shape[:2], dtype=np.float32)
    mask[5:25, 10:30] = 0.7
    modified = orig.copy()
    modified[mask > 0] = np.minimum(1, modified[mask > 0] + 0.1)
    selection = select_rois(pair.annotations, mode="by_index", instance_index=0)
    verdict = validate_artifact(orig, modified, mask, pair.annotations, selection)
    assert verdict.passed
    effect = {
        "schema_version": "sap-glare-v1", "kind": "elliptical", "seed": 42017,
        "angle_deg": -25., "radius_scale": 0.85, "aspect_ratio": 0.35,
        "falloff": 1.25, "intensity": 0.55, "center_jitter": 0.1,
        "image_size_px": [60, 40],
        "changed_pixels": verdict.metrics["changed_pixels"],
        "effect_mask_active_pixels": verdict.metrics["mask_active_pixels"],
    }
    return {
        "project_root": tmp_path, "run_id": "my_run_001", "original": orig,
        "modified": modified, "mask": mask, "annotations": pair.annotations,
        "selection": selection, "qa_pass": True, "qa_report": verdict.to_json(),
        "source_context": {key: entry[key] for key in (
            "archive", "image_member", "annotation_member", "image_sha256",
            "annotation_sha256", "split",
        )},
        "effect_summary": json.dumps(effect),
        "mask_summary": json.dumps({"mask_kind":"soft_rect", "image_size_px":[60,40],
                                     "selected_indices":[0], "pad_px":0,
                                     "feather_px":16.0, "strength":1.0}),
    }


def test_complete_round_trip_and_preserves_source(tmp_path: Path) -> None:
    args = prepared(tmp_path)
    snapshot = [array.copy() for array in (args["original"], args["modified"], args["mask"])]
    archive = tmp_path / "data/raw/for_learning.zip"
    before = archive.read_bytes()
    result = export_artifact(**args)
    run = tmp_path / "outputs/sap_runs/my_run_001"
    assert result.directory == run
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["schema_version"] == "sap-artifact-v1"
    assert manifest["source"]["image_sha256"] == args["source_context"]["image_sha256"]
    assert manifest["seed"] == 42017
    assert manifest["selection"]["seed"] == 0
    assert manifest["mask_builder"]["mask_kind"] == "soft_rect"
    assert manifest["image_size_px"] == [60, 40]
    assert manifest["geometry_changed"] is False
    assert manifest["qa"]["status"] == "PASS"
    assert manifest["qa"]["training_approved"] is False
    assert manifest["yolo_exported"] is False
    for key, name in manifest["output"].items():
        file = run / name
        assert file.is_file(), key
        assert hashlib.sha256(file.read_bytes()).hexdigest() == manifest["output_sha256"][key]
    with Image.open(run / manifest["output"]["image"]) as img:
        assert np.array_equal(np.asarray(img.convert("RGB")), np.rint(args["modified"]*255).astype(np.uint8))
    with Image.open(run / manifest["output"]["effect_mask"]) as mask:
        assert np.array_equal(np.asarray(mask), np.rint(args["mask"]*65535).astype(np.uint16))
    exported_seg = (run / manifest["output"]["annotation"]).read_bytes()
    seg = parse_seg(exported_seg, image_width=60, image_height=40, image_filename="my_run_001.png")
    assert seg.objects == args["annotations"].objects
    assert seg.source_seg["selectedList"][0]["unknown"] == {"preserve": True}
    assert seg.source_seg["unknown_root"] == {"keep": [1, "hello"]}
    assert archive.read_bytes() == before
    for current, original in zip((args["original"], args["modified"], args["mask"]),snapshot,strict=True):
        assert np.array_equal(current, original)


def test_duplicate_never_overwrites_existing_run(tmp_path: Path) -> None:
    args = prepared(tmp_path)
    first = export_artifact(**args)
    existing = (first.directory / "manifest.json").read_bytes()
    with pytest.raises(ArtifactExportError, match="duplicate_output"):
        export_artifact(**args)
    assert (first.directory / "manifest.json").read_bytes() == existing


@pytest.mark.parametrize("name", ["../test", "a/b", "a\\b", ".hidden", "", "über", "A" * 65])
def test_run_id_rejected_before_creating_outputs(tmp_path: Path, name: str) -> None:
    args = prepared(tmp_path)
    args["run_id"] = name
    with pytest.raises(ArtifactExportError, match="run_id"):
        export_artifact(**args)
    assert not (tmp_path / "outputs").exists()


def test_refuses_false_qa_without_writes(tmp_path: Path) -> None:
    args = prepared(tmp_path)
    args["qa_pass"] = False
    with pytest.raises(ArtifactExportError, match="QA boolean"):
        export_artifact(**args)
    assert not (tmp_path / "outputs").exists()


def test_rejects_modified_qa_report(tmp_path: Path) -> None:
    args = prepared(tmp_path)
    data = json.loads(args["qa_report"])
    data["metrics"]["changed_pixels"] = 0
    args["qa_report"] = json.dumps(data)
    with pytest.raises(ArtifactExportError, match="stale"):
        export_artifact(**args)
    assert not (tmp_path / "outputs").exists()


def test_rejects_failed_check_in_recomputed_qa(tmp_path: Path) -> None:
    args = prepared(tmp_path)
    args["modified"] = args["modified"].copy()
    args["modified"][0, 0] += 0.1
    with pytest.raises(ArtifactExportError, match="Recomputed QA FAIL"):
        export_artifact(**args)
    assert not (tmp_path / "outputs").exists()


def test_refuses_unverified_source_pixels(tmp_path: Path) -> None:
    args = prepared(tmp_path)
    args["original"] = args["original"].copy()
    args["original"][0, 0] += .01
    # Pretend QA accepted it: input payload must still match the real ZIP.
    verdict = validate_artifact(args["original"], args["modified"], args["mask"],
                                args["annotations"], args["selection"])
    assert not verdict.passed
    with pytest.raises(ArtifactExportError, match="QA FAIL"):
        export_artifact(**args)


def test_rejects_forged_index_hash(tmp_path: Path) -> None:
    args = prepared(tmp_path)
    args["source_context"]["image_sha256"] = "0" * 64
    with pytest.raises(ArtifactExportError, match="Source index mismatch"):
        export_artifact(**args)
    assert not (tmp_path / "outputs").exists()


def test_rejects_unsupported_effect_without_writes(tmp_path: Path) -> None:
    args = prepared(tmp_path)
    effect = json.loads(args["effect_summary"])
    effect["kind"] = "invalid"
    args["effect_summary"] = json.dumps(effect)
    with pytest.raises(ArtifactExportError, match="effect kind"):
        export_artifact(**args)


def test_refuses_symlinked_output_directory(tmp_path: Path) -> None:
    args = prepared(tmp_path)
    target = tmp_path / "private_target"
    target.mkdir()
    try:
        (tmp_path / "outputs").symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation not permitted on this OS")
    with pytest.raises(ArtifactExportError, match="symlink"):
        export_artifact(**args)
    assert list(target.iterdir()) == []


def test_incomplete_write_cleans_staging_directory(tmp_path: Path, monkeypatch) -> None:
    args = prepared(tmp_path)
    original_save = Image.Image.save
    calls = 0

    def crash(self, *a, **kw):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated image writing failure")
        return original_save(self, *a, **kw)

    monkeypatch.setattr(Image.Image, "save", crash)
    with pytest.raises(OSError, match="simulated"):
        export_artifact(**args)
    runs = tmp_path / "outputs/sap_runs"
    assert runs.is_dir()
    assert list(runs.iterdir()) == []
