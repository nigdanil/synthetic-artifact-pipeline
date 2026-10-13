"""Task 6.4: CPU-only ComfyUI mask adapter tests without torch installation."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sapipeline.annotations import parse_seg
from sapipeline.selection import select_rois

ROOT = Path(__file__).resolve().parents[1]
NODE_PATH = ROOT / "custom_nodes/sap_dataset_loader/mask.py"
WORKFLOW_PATH = ROOT / "workflows/templates/sap_mask_builder_v1.json"


class FakeTensor:
    def __init__(self, array: np.ndarray):
        self.array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def unsqueeze(self, dim: int):
        self.array = np.expand_dims(self.array, dim)
        return self


@pytest.fixture(autouse=True)
def fake_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(from_numpy=FakeTensor))


def node():
    spec = importlib.util.spec_from_file_location("sap_mask_local", NODE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SAPMaskBuilder()


@pytest.fixture
def example():
    doc = parse_seg(
        json.dumps(
            {
                "filename": "toy.jpg",
                "selectedList": [
                    {"name": "sku_one", "rect": {"x": 2, "y": 3, "width": 9, "height": 9}},
                    {"name": "sku_two", "rect": {"x": 15, "y": 2, "width": 8, "height": 12}},
                    {"name": "sku_three", "rect": {"x": 5, "y": 15, "width": 8, "height": 7}},
                ],
            }
        ),
        image_width=32,
        image_height=24,
        image_filename="toy.jpg",
    )
    selection = select_rois(doc, mode="by_index", instance_index=0)
    source = FakeTensor(np.full((1, 24, 32, 3), 0.25, dtype=np.float32))
    return selection, source


def test_node_contract_and_registration():
    cls = type(node())
    assert cls.RETURN_TYPES == ("MASK", "IMAGE", "STRING")
    assert cls.RETURN_NAMES == ("mask", "mask_overlay", "mask_summary")
    assert cls.FUNCTION == "build"
    assert "mask_kind" in cls.INPUT_TYPES()["required"]
    assert np.isnan(cls.IS_CHANGED())
    assert "import torch" not in NODE_PATH.read_text().split("class SAPMaskBuilder:")[0]

    proc = subprocess.run(
        [sys.executable, "-c", (
            "import sys; sys.path.insert(0, r'" + str(ROOT / "custom_nodes") + "'); "
            "import sap_dataset_loader; print(sorted(sap_dataset_loader.NODE_CLASS_MAPPINGS))"
        )],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "SAPMaskBuilder" in proc.stdout
    assert "SAPROISelector" in proc.stdout
    assert "SAPDatasetLoader" in proc.stdout


@pytest.mark.parametrize("kind", ["rect", "soft_rect"])
def test_full_resolution_mask_and_preview(example, kind):
    selection, source = example
    unchanged = source.array.copy()
    mask, preview, summary = node().build(selection, source, mask_kind=kind)
    assert mask.array.shape == (1, 24, 32)
    assert mask.array.dtype == np.float32
    assert preview.array.shape == (1, 24, 32, 3)
    assert preview.array.dtype == np.float32
    assert np.array_equal(source.array, unchanged)
    assert 0 <= mask.array.min() and mask.array.max() <= 1.0
    assert mask.array[0, 0, 0] == 0
    assert mask.array[0, 7, 6] > 0
    assert np.array_equal(preview.array[0, 0, 0], unchanged[0, 0, 0])
    assert not np.array_equal(preview.array[0, 7, 6], unchanged[0, 7, 6])
    data = json.loads(summary)
    assert data["selected_indices"] == [0]
    assert data["target_count"] == 1
    assert data["image_size_px"] == [32, 24]
    assert "sku_one" not in summary  # avoid publishing vendor identifiers
    assert data["mask_type"] == "bbox-effect-mask-not-segmentation"


def test_strength_parameter(example):
    selection, image = example
    strong, _, _ = node().build(selection, image, mask_kind="rect", strength=1)
    weak, _, _ = node().build(selection, image, mask_kind="rect", strength=0.3)
    assert strong.array.max() == pytest.approx(1)
    assert weak.array.max() == pytest.approx(0.3)


def test_feather_softens_inside_box(example):
    selection, image = example
    soft, _, _ = node().build(selection, image, mask_kind="soft_rect", feather_px=3)
    hard, _, _ = node().build(selection, image, mask_kind="rect", feather_px=3)
    assert 0 < soft.array[0, 3, 2] < 1
    assert hard.array[0, 3, 2] == 1
    assert soft.array[0, 7, 6] == pytest.approx(1)


def test_multi_box_union_is_max_not_sum(example):
    _, image = example
    doc = parse_seg(
        json.dumps({"filename": "toy.jpg", "selectedList": [
            {"name": "one", "rect": {"x": 2, "y": 3, "width": 10, "height": 10}},
            {"name": "two", "rect": {"x": 6, "y": 6, "width": 10, "height": 10}},
        ]}),
        image_width=32, image_height=24, image_filename="toy.jpg",
    )
    selection = select_rois(doc, mode="by_class", class_name="one")
    selection2 = select_rois(doc, mode="by_class", class_name="two")
    from sapipeline.selection import SelectionResult

    both = SelectionResult("random_n", 1, selection.targets + selection2.targets)
    merged, _, report = node().build(both, image, mask_kind="rect", strength=0.5)
    assert merged.array[0, 8, 8] == pytest.approx(0.5)
    assert json.loads(report)["selected_indices"] == [0, 1]


@pytest.mark.parametrize("kwargs, error", [
    ({"mask_kind": "circle"}, "kind"),
    ({"pad_px": -1}, "pad_px"),
    ({"feather_px": -1}, "feather_px"),
    ({"strength": 0}, "strength"),
    ({"strength": 2}, "strength"),
])
def test_invalid_settings_rejected(example, kwargs, error):
    selection, image = example
    with pytest.raises(ValueError, match=error):
        node().build(selection, image, **kwargs)


@pytest.mark.parametrize("bad_image, error", [
    (np.zeros((2, 24, 32, 3), dtype=np.float32), "one RGB"),
    (np.zeros((1, 24, 32, 4), dtype=np.float32), "one RGB"),
    (np.full((1, 24, 32, 3), np.nan, dtype=np.float32), "finite"),
    (np.full((1, 24, 32, 3), 2.0, dtype=np.float32), "finite in"),
])
def test_invalid_image_rejected(example, bad_image, error):
    selection, _ = example
    with pytest.raises(ValueError, match=error):
        node().build(selection, FakeTensor(bad_image))


def test_non_selection_or_non_tensor_rejected(example):
    selection, image = example
    with pytest.raises(TypeError, match="SAP ROI Selector"):
        node().build({}, image)
    with pytest.raises(TypeError, match="IMAGE"):
        node().build(selection, "not-an-image")


def test_selected_annotations_not_mutated(example):
    selection, source = example
    old = deepcopy(selection)
    node().build(selection, source, mask_kind="soft_rect", pad_px=1)
    assert selection == old


def test_public_workflow_is_connected_and_contains_no_private_paths():
    contents = WORKFLOW_PATH.read_text(encoding="utf-8")
    graph = json.loads(contents)
    assert "D:/my_work" not in contents
    assert len(graph["nodes"]) == 9
    assert len(graph["links"]) == 10
    assert graph["last_node_id"] == 9
    assert graph["last_link_id"] == 10
    by_id = {n["id"]: n for n in graph["nodes"]}
    assert len(by_id) == 9
    assert by_id[1]["widgets_values"][0] == ""
    assert by_id[4]["widgets_values"] == [
        "random_n", 42017, "fixed", 3, 0, "", False, False
    ]
    assert by_id[6]["widgets_values"] == ["soft_rect", 0, 16.0, 1.0]
    assert [6, 4, 0, 6, 0, "SAP_ROI_SELECTION"] in graph["links"]
    assert [7, 1, 0, 6, 1, "IMAGE"] in graph["links"]
    assert [8, 6, 0, 7, 0, "MASK"] in graph["links"]
    assert [9, 7, 0, 8, 0, "IMAGE"] in graph["links"]
    assert [10, 6, 1, 9, 0, "IMAGE"] in graph["links"]
    for link_id, src, src_idx, dst, dst_idx, kind in graph["links"]:
        assert src in by_id and dst in by_id and isinstance(link_id, int)
        source_slot = by_id[src]["outputs"][src_idx]
        sink_slot = by_id[dst]["inputs"][dst_idx]
        assert source_slot["type"] == sink_slot["type"] == kind
        assert link_id in source_slot["links"]
        assert sink_slot["link"] == link_id
