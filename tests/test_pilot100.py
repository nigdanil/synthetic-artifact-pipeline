"""Synthetic-only tests for resumable private technical 100-image pilot."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest
from PIL import Image

from sapipeline.batch.pilot import (
    build_pilot_plan,
    pack_pilot,
    run_pilot,
    save_pilot_plan,
)
from sapipeline.batch.runner import BatchRunError
from sapipeline.dataset import build_index


def _fixture_project(root: Path, scenes: int = 61) -> Path:
    (root / "pyproject.toml").write_text("[project]\nname='fake-sap'\n", encoding="utf-8")
    raw = root / "data/raw"
    raw.mkdir(parents=True)
    archive = raw / "for_learning.zip"
    with ZipFile(archive, "w") as z:
        for n in range(scenes):
            buf = BytesIO()
            Image.new("RGB", (120, 90), (35 + n * 3, 75, 110)).save(buf, "JPEG")
            name = f"sample_{n:03d}"
            z.writestr(f"img/{name}.jpg", buf.getvalue())
            z.writestr(f"seg/{name}.seg", json.dumps({
                "filename": f"{name}.jpg", "selectedList": [
                    {"name": "fictional_item", "rect": {
                        "x": 15, "y": 12, "width": 85, "height": 65,
                    }},
                ],
            }))
    idx, _ = build_index({"learning": archive})
    file = root / "data/work/index/dataset_index.json"
    file.parent.mkdir(parents=True)
    file.write_text(json.dumps(idx), encoding="utf-8")
    return root


def test_plan_chooses_60_distinct_learning_scenes_and_has_no_glass(tmp_path: Path) -> None:
    _fixture_project(tmp_path)
    plan = build_pilot_plan(tmp_path)
    assert len(plan["sources"]) == 60
    assert len({r["source_sha256"] for r in plan["runs"]}) == 60
    assert len(plan["runs"]) == 120
    assert plan["target"] == 100
    assert [row["variant"] for row in plan["runs"][:2]] == [
        "glare_directional", "glare_elliptical"]
    assert not (tmp_path / "reports").exists()
    assert build_pilot_plan(tmp_path) == plan


def test_plan_refuses_overwrite_invalid_count_and_tampering(tmp_path: Path) -> None:
    _fixture_project(tmp_path)
    with pytest.raises(BatchRunError, match="scenes"):
        build_pilot_plan(tmp_path, scenes=49)
    plan = build_pilot_plan(tmp_path, scenes=50)
    dst = save_pilot_plan(tmp_path, plan)
    with pytest.raises(FileExistsError):
        save_pilot_plan(tmp_path, plan)
    plan["runs"][0]["seed"] += 1
    dst.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(BatchRunError, match="modified"):
        run_pilot(tmp_path, max_new=1)
    assert not (tmp_path / "outputs").exists()


def test_pilot_smoke_resumes_and_never_overwrites_existing(tmp_path: Path) -> None:
    _fixture_project(tmp_path)
    doc = build_pilot_plan(tmp_path, scenes=50)
    save_pilot_plan(tmp_path, doc)
    first, previous, rejected = run_pilot(tmp_path, max_new=1)
    assert (first, previous, rejected) == ([doc["runs"][0]["run_id"]], 0, [])
    folder = tmp_path / "outputs/sap_runs" / first[0]
    before = (folder / "manifest.json").read_bytes()
    second, previous2, rejected2 = run_pilot(tmp_path, max_new=1)
    assert (second, previous2, rejected2) == ([doc["runs"][1]["run_id"]], 1, [])
    assert (folder / "manifest.json").read_bytes() == before
    png = folder / "images" / (first[0] + ".png")
    png.write_bytes(png.read_bytes() + b"corruption")
    with pytest.raises(BatchRunError, match="hash"):
        run_pilot(tmp_path, max_new=1)


def test_full_100_candidates_and_private_preview_zip(tmp_path: Path) -> None:
    _fixture_project(tmp_path)
    doc = build_pilot_plan(tmp_path, scenes=60)
    save_pilot_plan(tmp_path, doc)
    with pytest.raises(BatchRunError, match="Only 0/100"):
        pack_pilot(tmp_path)
    made, previous, rejected = run_pilot(tmp_path, max_new=100)
    assert len(made) == 100
    assert previous == 0 and rejected == []
    assert len(list((tmp_path / "outputs/sap_runs").iterdir())) == 100
    assert len({json.loads((tmp_path / "outputs/sap_runs" / rid / "manifest.json")
                           .read_text())["source"]["image_sha256"] for rid in made}) == 50
    another, previous2, rejected2 = run_pilot(tmp_path, max_new=100)
    assert another == [] and previous2 == 100 and rejected2 == []
    output, count = pack_pilot(tmp_path)
    assert count == 100 and output.is_file()
    with ZipFile(output) as archive:
        names = archive.namelist()
        assert len(names) == 201
        assert len([n for n in names if n.startswith("images/")]) == 100
        assert len([n for n in names if n.startswith("annotations/")]) == 100
        summary = json.loads(archive.read("qa_preview_manifest.json"))
        assert summary["total"] == 100
        assert summary["unique_source_scenes"] == 50
        assert summary["training_approved"] is False
    with pytest.raises(BatchRunError, match="exists"):
        pack_pilot(tmp_path)


def test_skipped_technical_qa_fills_100_from_reserved_scenes(tmp_path: Path, monkeypatch) -> None:
    import sapipeline.batch.pilot as module

    _fixture_project(tmp_path)
    plan = build_pilot_plan(tmp_path, scenes=60)
    save_pilot_plan(tmp_path, plan)
    render = module._render
    first_id = plan["runs"][0]["run_id"]

    def one_qa_failure(root, entry, row, quad, index):
        if row["run_id"] == first_id:
            raise BatchRunError(f"Technical QA FAIL for {first_id}: new_clipping_below_limit")
        return render(root, entry, row, quad, index)

    monkeypatch.setattr(module, "_render", one_qa_failure)
    made, previous, qa_failed = run_pilot(tmp_path, max_new=100)
    assert (len(made), previous, qa_failed) == (100, 0, [first_id])
    assert first_id not in made
    output, count = pack_pilot(tmp_path)
    assert count == 100 and output.is_file()
