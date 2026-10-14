"""Second glass pilot: planning only, no private sample or output mutations."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest
from PIL import Image

from sapipeline.batch.glass_pilot import (
    build_glass_plan,
    load_glass_plan,
    save_glass_plan,
    validate_glass_plan,
)
from sapipeline.batch.runner import BatchRunError
from sapipeline.dataset import build_index


def _project(root: Path, *, scenes: int = 61) -> Path:
    (root / "pyproject.toml").write_text("[project]\nname='fake-sap'\n", encoding="utf-8")
    raw = root / "data/raw"
    raw.mkdir(parents=True)
    archive = raw / "for_learning.zip"
    with ZipFile(archive, "w") as zipped:
        for n in range(scenes):
            data = BytesIO()
            Image.new("RGB", (120, 90), (35 + n * 3, 75, 125)).save(data, "JPEG")
            name = f"sample_{n:03d}"
            zipped.writestr(f"img/{name}.jpg", data.getvalue())
            zipped.writestr(f"seg/{name}.seg", json.dumps({
                "filename": f"{name}.jpg",
                "selectedList": [{"name": "fictional_item", "rect": {
                    "x": 15, "y": 12, "width": 85, "height": 65,
                }}],
            }))
    index, _ = build_index({"learning": archive})
    dest = root / "data/work/index/dataset_index.json"
    dest.parent.mkdir(parents=True)
    dest.write_text(json.dumps(index), encoding="utf-8")
    return root


def test_plan_is_reproducible_and_does_not_generate_any_artifacts(tmp_path: Path) -> None:
    _project(tmp_path)
    archive = tmp_path / "data/raw/for_learning.zip"
    before = hashlib.sha256(archive.read_bytes()).hexdigest()
    plan = build_glass_plan(tmp_path)
    assert plan == build_glass_plan(tmp_path)
    assert len(plan["sources"]) == 60
    assert len(plan["runs"]) == 120
    assert len({row["source_sha256"] for row in plan["runs"]}) == 60
    assert plan["target"] == 100
    assert plan["manual_review_required"] is True
    assert plan["training_approved"] is False
    assert all(source["glass_quad_norm"] is None for source in plan["sources"])
    assert [row["variant"] for row in plan["runs"][:2]] == [
        "glass_diffuse", "glass_fluorescent"]
    assert plan["runs"][0]["effect"]["band_count"] == 1
    assert plan["runs"][1]["effect"]["band_count"] == 3
    assert not (tmp_path / "outputs").exists()
    assert not (tmp_path / "reports").exists()
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == before


def test_plan_skips_sources_already_exported(tmp_path: Path) -> None:
    _project(tmp_path)
    preliminary = build_glass_plan(tmp_path)
    already = preliminary["sources"][0]["image_sha256"]
    folder = tmp_path / "outputs/sap_runs/previous"
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps({
        "schema_version": "sap-artifact-v1", "run_id": "previous",
        "source": {"image_sha256": already},
    }), encoding="utf-8")
    fresh = build_glass_plan(tmp_path)
    assert already not in {entry["image_sha256"] for entry in fresh["sources"]}
    assert len(fresh["sources"]) == 60


def test_plan_is_private_immutable_and_verifies_manual_quad(tmp_path: Path) -> None:
    _project(tmp_path)
    plan = build_glass_plan(tmp_path, scenes=50)
    path = save_glass_plan(tmp_path, plan)
    assert path == tmp_path / "reports/glass_pilot100/plan.json"
    with pytest.raises(FileExistsError):
        save_glass_plan(tmp_path, plan)
    assert load_glass_plan(tmp_path)[1] == 50
    plan["sources"][0]["glass_quad_norm"] = [
        [0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]
    path.write_text(json.dumps(plan), encoding="utf-8")
    assert load_glass_plan(tmp_path)[1] == 49
    plan["sources"][0]["glass_quad_norm"] = [
        [0.1, 0.1], [0.9, 0.9], [0.9, 0.1], [0.1, 0.9]]
    with pytest.raises(BatchRunError, match="Invalid glass_quad_norm"):
        validate_glass_plan(tmp_path, plan)
    plan["sources"][0]["glass_quad_norm"] = None
    plan["runs"][0]["effect"]["intensity"] = 1.0
    with pytest.raises(BatchRunError, match="was modified"):
        validate_glass_plan(tmp_path, plan)


def test_invalid_inputs_and_changes_to_ids_fail_closed(tmp_path: Path) -> None:
    _project(tmp_path)
    with pytest.raises(BatchRunError, match="scenes"):
        build_glass_plan(tmp_path, scenes=49)
    with pytest.raises(BatchRunError, match="seed"):
        build_glass_plan(tmp_path, seed=-1)
    with pytest.raises(BatchRunError, match="eligible unused"):
        build_glass_plan(tmp_path, scenes=62)
    plan = build_glass_plan(tmp_path, scenes=50)
    plan["runs"][0]["run_id"] = "p9_previous_pilot"
    with pytest.raises(BatchRunError, match="was modified"):
        validate_glass_plan(tmp_path, plan)
    assert not (tmp_path / "outputs").exists()
    assert not (tmp_path / "reports").exists()
