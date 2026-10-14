"""Private glass calibration/preview tests using fictional synthetic sources only."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest
from PIL import Image

from sapipeline.batch.glass_pilot import build_glass_plan, save_glass_plan
from sapipeline.batch.glass_preview import (
    create_geometry_guides,
    generate_glass_previews,
    set_scene_quad,
)
from sapipeline.batch.runner import BatchRunError
from sapipeline.dataset import build_index

QUAD = [[0.12, 0.18], [0.88, 0.18], [0.88, 0.84], [0.12, 0.84]]


def _project(root: Path) -> Path:
    (root / "pyproject.toml").write_text("[project]\nname='fake-sap'\n", encoding="utf-8")
    raw = root / "data/raw"
    raw.mkdir(parents=True)
    archive = raw / "for_learning.zip"
    with ZipFile(archive, "w") as zipped:
        for n in range(50):
            buf = BytesIO()
            Image.new("RGB", (120, 90), (35 + 3 * n, 75, 110)).save(buf, "JPEG")
            name = f"scene_{n:03d}"
            zipped.writestr(f"img/{name}.jpg", buf.getvalue())
            zipped.writestr(f"seg/{name}.seg", json.dumps({
                "filename": f"{name}.jpg", "selectedList": [{
                    "name": "fictional_item", "rect": {
                        "x": 12, "y": 10, "width": 96, "height": 70,
                    },
                }],
            }))
    index, _ = build_index({"learning": archive})
    dest = root / "data/work/index/dataset_index.json"
    dest.parent.mkdir(parents=True)
    dest.write_text(json.dumps(index), encoding="utf-8")
    save_glass_plan(root, build_glass_plan(root, scenes=50))
    return root


def test_guides_use_verified_sources_and_never_edit_plan_or_zip(tmp_path: Path) -> None:
    root = _project(tmp_path)
    archive = root / "data/raw/for_learning.zip"
    plan = root / "reports/glass_pilot100/plan.json"
    original_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    original_plan = plan.read_bytes()
    paths = create_geometry_guides(root, count=3)
    assert len(paths) == 3
    for i, path in enumerate(paths):
        assert path.name == "select_corners.html"
        assert f"Scene {i:02d}" in path.read_text(encoding="utf-8")
        assert "top-left → top-right → bottom-right → bottom-left" in path.read_text(
            encoding="utf-8"
        )
        with Image.open(path.parent / "source.png") as source:
            assert source.size == (120, 90)
    assert plan.read_bytes() == original_plan
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == original_hash
    assert not (root / "outputs").exists()
    with pytest.raises(BatchRunError, match="already exists"):
        create_geometry_guides(root, count=3)


def test_manual_quad_is_validated_atomically_and_backed_up(tmp_path: Path) -> None:
    root = _project(tmp_path)
    path = root / "reports/glass_pilot100/plan.json"
    old = path.read_bytes()
    bad = [[0.1, 0.1], [0.9, 0.9], [0.9, 0.1], [0.1, 0.9]]
    with pytest.raises(BatchRunError, match="Invalid glass polygon"):
        set_scene_quad(root, scene=0, quad=bad)
    assert path.read_bytes() == old
    set_scene_quad(root, scene=0, quad=QUAD)
    updated = json.loads(path.read_text())
    assert updated["sources"][0]["glass_quad_norm"] == QUAD
    assert updated["sources"][1]["glass_quad_norm"] is None
    assert updated["training_approved"] is False
    assert updated["runs"] == json.loads(old)["runs"]
    assert (root / "reports/glass_pilot100/plan_backups" /
            f"plan_{hashlib.sha256(old).hexdigest()}.json").read_bytes() == old
    with pytest.raises(BatchRunError, match="--replace"):
        set_scene_quad(root, scene=0, quad=[[0.11, 0.18], *QUAD[1:]])
    revised = [[0.11, 0.18], *QUAD[1:]]
    set_scene_quad(root, scene=0, quad=revised, replace=True)
    assert json.loads(path.read_text())["sources"][0]["glass_quad_norm"] == revised
    assert len(list((root / "reports/glass_pilot100/plan_backups").glob("*.json"))) == 2


def test_previews_require_quads_then_render_only_private_candidates(tmp_path: Path) -> None:
    root = _project(tmp_path)
    with pytest.raises(BatchRunError, match="missing for scenes"):
        generate_glass_previews(root, count=2)
    assert not (root / "reports/glass_pilot100/effect_previews").exists()
    for n in range(2):
        set_scene_quad(root, scene=n, quad=QUAD)
    outputs = generate_glass_previews(root, count=2)
    assert len(outputs) == 2
    for folder, report in outputs:
        assert report["stage7_export"] is False
        assert report["training_approved"] is False
        assert report["manual_review_required"] is True
        assert [x["variant"] for x in report["variants"]] == [
            "glass_diffuse", "glass_fluorescent",
        ]
        assert all(x["technical_qa"]["status"] in ("PASS", "FAIL")
                   for x in report["variants"])
        assert {p.name for p in folder.iterdir()} == {
            "original.png", "glass_plane.png", "glass_diffuse.png",
            "glass_fluorescent.png", "glass_diffuse_mask_visual.png",
            "glass_fluorescent_mask_visual.png", "preview_report.json",
        }
        with Image.open(folder / "glass_diffuse.png") as im:
            assert im.size == (120, 90)
        assert json.loads((folder / "preview_report.json").read_text()) == report
    assert not (root / "outputs").exists()
    with pytest.raises(BatchRunError, match="already exists"):
        generate_glass_previews(root, count=2)


def test_changed_quad_creates_new_preview_without_overwriting_old(tmp_path: Path) -> None:
    root = _project(tmp_path)
    set_scene_quad(root, scene=0, quad=QUAD)
    first = generate_glass_previews(root, count=1)[0][0]
    original_report = (first / "preview_report.json").read_bytes()
    set_scene_quad(root, scene=0, quad=[[0.13, 0.18], *QUAD[1:]], replace=True)
    second = generate_glass_previews(root, count=1)[0][0]
    assert first != second
    assert (first / "preview_report.json").read_bytes() == original_report
    assert second.exists()


def test_bounds_and_unknown_geometry_rejected_before_preview(tmp_path: Path) -> None:
    root = _project(tmp_path)
    with pytest.raises(BatchRunError, match="within the glass plan"):
        create_geometry_guides(root, start=49, count=2)
    with pytest.raises(BatchRunError, match="within the glass plan"):
        generate_glass_previews(root, start=-1)
    with pytest.raises(BatchRunError, match="within the glass plan"):
        set_scene_quad(root, scene=60, quad=QUAD)
    assert not (root / "outputs").exists()
