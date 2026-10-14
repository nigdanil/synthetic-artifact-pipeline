"""Stage-8 batch runner: synthetic data only, no private dataset references."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest
from PIL import Image

from sapipeline.batch import BatchRunError, build_plan, execute_plan, save_plan
from sapipeline.dataset import build_index


def _project(root: Path, *, scenes: int = 5) -> Path:
    (root / "pyproject.toml").write_text("[project]\nname='fake-sap'\n", encoding="utf-8")
    raw = root / "data/raw"
    raw.mkdir(parents=True)
    archive = raw / "for_learning.zip"
    with ZipFile(archive, "w") as zipped:
        for n in range(scenes):
            data = BytesIO()
            Image.new("RGB", (120, 90), (55 + n * 12, 75, 110)).save(data, "JPEG")
            name = f"sample_{n}"
            zipped.writestr(f"img/{name}.jpg", data.getvalue())
            zipped.writestr(f"seg/{name}.seg", json.dumps({
                "filename": f"{name}.jpg",
                "selectedList": [{"name": "fictional_item", "rect": {
                    "x": 15, "y": 12, "width": 85, "height": 65}},
                ],
            }))
    index, _ = build_index({"learning": archive})
    dst = root / "data/work/index/dataset_index.json"
    dst.parent.mkdir(parents=True)
    dst.write_text(json.dumps(index), encoding="utf-8")
    return root


def test_plan_selects_four_new_unique_scenes_and_stable_run_ids(tmp_path: Path) -> None:
    _project(tmp_path)
    plan = build_plan(tmp_path, scenes=4, seed=100)
    assert len(plan["sources"]) == 4
    assert len({s["image_sha256"] for s in plan["sources"]}) == 4
    assert [r["variant"] for r in plan["runs"][:4]] == [
        "glare_directional", "glare_elliptical", "glass_soft", "glass_strong"]
    assert [r["seed"] for r in plan["runs"][:4]] == [100, 101, 102, 103]
    assert len(plan["runs"]) == 16
    assert build_plan(tmp_path, scenes=4, seed=100) == plan
    assert not (tmp_path / "reports").exists()


def test_plan_rejects_existing_source_and_does_not_overwrite_plan(tmp_path: Path) -> None:
    _project(tmp_path)
    plan = build_plan(tmp_path, scenes=4)
    path = save_plan(tmp_path, plan)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_plan(tmp_path, plan)
    assert path.read_bytes() == original
    first = plan["sources"][0]
    exported = tmp_path / "outputs/sap_runs/previous"
    exported.mkdir(parents=True)
    (exported / "manifest.json").write_text(json.dumps({
        "schema_version": "sap-artifact-v1", "run_id": "previous",
        "source": {"image_sha256": first["image_sha256"]},
    }), encoding="utf-8")
    next_plan = build_plan(tmp_path, scenes=4)
    assert first["image_sha256"] not in {s["image_sha256"] for s in next_plan["sources"]}


def test_one_glare_export_resume_and_existing_sha_validation(tmp_path: Path) -> None:
    _project(tmp_path)
    plan = build_plan(tmp_path, scenes=1, seed=430)
    save_plan(tmp_path, plan)
    created, skipped = execute_plan(tmp_path, max_new=1)
    assert created == [plan["runs"][0]["run_id"]]
    assert not skipped
    run_dir = tmp_path / "outputs/sap_runs" / created[0]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["qa"]["status"] == "PASS"
    assert manifest["qa"]["training_approved"] is False
    assert manifest["source"]["image_sha256"] == plan["sources"][0]["image_sha256"]
    assert (run_dir / "annotations" / (created[0] + ".seg")).exists()
    created2, skipped2 = execute_plan(tmp_path, max_new=1)
    assert created2 == [plan["runs"][1]["run_id"]]
    assert skipped2 == created
    (run_dir / "images" / (created[0] + ".png")).write_bytes(b"tampered")
    with pytest.raises(BatchRunError, match="hash"):
        execute_plan(tmp_path, max_new=1)


def test_missing_glass_quad_fail_closed_no_third_output(tmp_path: Path) -> None:
    _project(tmp_path)
    plan = build_plan(tmp_path, scenes=1)
    save_plan(tmp_path, plan)
    created, skipped = execute_plan(tmp_path, max_new=2)
    assert len(created) == 2 and skipped == []
    with pytest.raises(BatchRunError, match="Glass quad is missing"):
        execute_plan(tmp_path, max_new=1)
    assert len(list((tmp_path / "outputs/sap_runs").iterdir())) == 2


def test_plan_changes_to_seeds_runids_and_unsafe_quad_blocked(tmp_path: Path) -> None:
    _project(tmp_path)
    plan = build_plan(tmp_path, scenes=1)
    path = save_plan(tmp_path, plan)
    plan["runs"][0]["seed"] += 100
    path.write_text(json.dumps(plan))
    with pytest.raises(BatchRunError, match="was changed"):
        execute_plan(tmp_path)
    plan["runs"][0]["seed"] -= 100
    plan["sources"][0]["glass_quad_norm"] = "../arbitrary"
    path.write_text(json.dumps(plan))
    with pytest.raises(BatchRunError, match="invalid scene/quad"):
        execute_plan(tmp_path)
    assert not (tmp_path / "outputs").exists()


def test_plan_only_accepts_learning_and_rejects_wrong_counts(tmp_path: Path) -> None:
    _project(tmp_path, scenes=1)
    with pytest.raises(BatchRunError, match="Not enough unused"):
        build_plan(tmp_path, scenes=2)
    with pytest.raises(BatchRunError, match="scenes"):
        build_plan(tmp_path, scenes=0)
    with pytest.raises(BatchRunError, match="seed"):
        build_plan(tmp_path, scenes=1, seed=-1)


def test_can_export_configured_glass_though_still_manual(tmp_path: Path) -> None:
    _project(tmp_path, scenes=1)
    plan = build_plan(tmp_path, scenes=1, seed=500)
    plan["sources"][0]["glass_quad_norm"] = [
        [0.10, 0.10], [0.92, 0.10], [0.92, 0.90], [0.10, 0.90]]
    save_plan(tmp_path, plan)
    created, _ = execute_plan(tmp_path, max_new=4)
    assert len(created) == 4
    last = tmp_path / "outputs/sap_runs" / created[-1]
    manifest = json.loads((last / "manifest.json").read_text())
    assert manifest["effects"][0]["schema_version"] == "sap-glass-reflection-v1"
    assert manifest["qa"]["manual_review_required"] is True
    assert manifest["qa"]["training_approved"] is False


def test_glass_quad_change_does_not_reuse_old_export(tmp_path: Path) -> None:
    _project(tmp_path, scenes=1)
    plan = build_plan(tmp_path, scenes=1)
    plan["sources"][0]["glass_quad_norm"] = [
        [0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]
    path = save_plan(tmp_path, plan)
    execute_plan(tmp_path, max_new=4)
    plan["sources"][0]["glass_quad_norm"] = [
        [0.15, 0.1], [0.9, 0.1], [0.9, 0.9], [0.15, 0.9]]
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(BatchRunError, match="settings/quad differ"):
        execute_plan(tmp_path)


def test_existing_run_collision_rejected_before_overwrite(tmp_path: Path) -> None:
    _project(tmp_path, scenes=1)
    plan = build_plan(tmp_path, scenes=1)
    save_plan(tmp_path, plan)
    path = tmp_path / "outputs/sap_runs" / plan["runs"][0]["run_id"]
    path.mkdir(parents=True)
    (path / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(BatchRunError, match="incomplete"):
        execute_plan(tmp_path)
    assert (path / "manifest.json").read_text() == "{}"
