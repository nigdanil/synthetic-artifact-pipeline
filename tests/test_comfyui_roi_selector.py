"""CPU-only tests. ComfyUI's real torch install is not required in SAP .venv."""

from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sapipeline.annotations import parse_seg
from sapipeline.selection import SERVICE_CLASSES, SelectionError

ROOT = Path(__file__).resolve().parents[1]
NODE_FILE = ROOT / "custom_nodes/sap_dataset_loader/roi.py"
WORKFLOW = ROOT / "workflows/templates/sap_roi_selector_v1.json"


class StubTensor:
    def __init__(self, arr: np.ndarray):
        self.array = arr

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def to(self, dtype):
        self.array = self.array.astype(dtype)
        return self

    def div_(self, n):
        self.array /= n
        return self

    def unsqueeze(self, n):
        self.array = np.expand_dims(self.array, n)
        return self


def node():
    spec = importlib.util.spec_from_file_location("sap_roi_local", NODE_FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SAPROISelector()


@pytest.fixture(autouse=True)
def fake_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(from_numpy=StubTensor,
                                                               float32=np.float32))


@pytest.fixture
def example():
    data = {
        "filename": "toy.jpg",
        "selectedList": [
            {"name": "sku_red", "rect": {"x": 1, "y": 1, "width": 6, "height": 8}},
            {"name": "sku_blue", "rect": {"x": 10, "y": 1, "width": 7, "height": 8}},
            {"name": "sku_red", "rect": {"x": 19, "y": 1, "width": 8, "height": 8}},
            {"name": "glint", "rect": {"x": 1, "y": 11, "width": 8, "height": 6}},
            {"name": "product", "rect": {"x": 11, "y": 11, "width": 8, "height": 6}},
        ],
    }
    doc = parse_seg(json.dumps(data), image_width=32, image_height=24,
                    image_filename="toy.jpg")
    tensor = StubTensor(np.full((1, 24, 32, 3), 0.3, dtype=np.float32))
    return doc, tensor


def test_contract_and_comfy_discovery_import():
    cls = type(node())
    assert cls.RETURN_TYPES == ("SAP_ROI_SELECTION", "IMAGE", "STRING")
    assert cls.RETURN_NAMES == ("selection", "roi_preview", "selection_summary")
    assert cls.CATEGORY.startswith("SAP/")
    assert "annotations" in cls.INPUT_TYPES()["required"]
    assert "image" in cls.INPUT_TYPES()["required"]
    assert np.isnan(cls.IS_CHANGED())
    assert "torch" not in NODE_FILE.read_text().split("class SAPROISelector:")[0]


def test_by_index_produces_single_cyan_box_without_altering_original(example):
    doc, image = example
    before = image.array.copy()
    sel, preview, report = node().select(doc, image, "by_index", instance_index=1)
    assert [a.instance_index for a in sel.targets] == [1]
    assert preview.array.shape == image.array.shape
    assert np.array_equal(before, image.array)
    assert not np.array_equal(before, preview.array)
    assert len(sel.targets) == 1
    assert "sku_blue" not in report
    assert json.loads(report)["selected_instance_indices"] == [1]
    assert np.any(
        np.all(np.rint(preview.array[0] * 255).astype(np.uint8) == [0, 255, 255], axis=-1)
    )
    assert np.allclose(preview.array[0, 21, 30], before[0, 21, 30], atol=1/255)


def test_random_n_is_deterministic_and_blocks_service_classes(example):
    doc, image = example
    a, _, _ = node().select(doc, image, "random_n", 42017, 2)
    b, _, _ = node().select(doc, image, "random_n", 42017, 2)
    assert [i.instance_index for i in a.targets] == [i.instance_index for i in b.targets]
    assert all(obj.class_name not in SERVICE_CLASSES for obj in a.targets)
    assert all(obj.class_name != "product" for obj in a.targets)


def test_include_generic_product_allows_fourth_candidate(example):
    doc, image = example
    selection, _, _ = node().select(doc, image, "random_n", 17, 4,
                                   include_generic_product=True)
    assert len(selection.targets) == 4
    assert any(a.class_name == "product" for a in selection.targets)


def test_by_class_returns_all_matching_instances(example):
    doc, image = example
    selection, preview, _ = node().select(doc, image, "by_class", class_name="sku_red")
    assert [o.instance_index for o in selection.targets] == [0, 2]
    assert preview.array.shape == (1, 24, 32, 3)


def test_service_class_requires_explicit_consent(example):
    doc, image = example
    with pytest.raises(SelectionError, match="Service ROI"):
        node().select(doc, image, "by_index", instance_index=3)
    selected, _, _ = node().select(doc, image, "by_index", instance_index=3,
                                    allow_service_targets=True)
    assert selected.targets[0].class_name == "glint"
    with pytest.raises(ValueError, match="requires explicit"):
        node().select(doc, image, "random_n", allow_service_targets=True)


@pytest.mark.parametrize("bad, message", [
    ({"mode": "not_a_mode"}, "Unsupported selection"),
    ({"mode": "by_class", "class_name": ""}, "requires class_name"),
    ({"mode": "random_one", "class_name": "sku_red"}, "class_name is only valid"),
    ({"mode": "by_class", "include_generic_product": True, "class_name": "sku_red"},
     "include_generic_product"),
    ({"mode": "random_n", "count": 5}, "only 3 eligible"),
    ({"mode": "by_index", "instance_index": 100}, "instance_index=100"),
    ({"mode": "random_n", "instance_index": 1}, "only valid in by_index"),
    ({"mode": "random_n", "seed": -1}, "seed"),
    ({"mode": "by_index", "instance_index": 3}, "Service ROI"),
])
def test_errors_are_explicit(example, bad, message):
    doc, img = example
    with pytest.raises(ValueError, match=message):
        node().select(doc, img, **bad)


def test_non_documents_rejected(example):
    _, img = example
    with pytest.raises(TypeError, match="SAP Dataset Loader"):
        node().select({}, img)


@pytest.mark.parametrize("bad_img, error", [
    (np.zeros((2, 24, 32, 3)), "single RGB IMAGE"),
    (np.zeros((1, 24, 32, 4)), "single RGB IMAGE"),
    (np.zeros((1, 23, 32, 3)), "does not match"),
    (np.full((1, 24, 32, 3), -0.1), "normalized"),
    (np.full((1, 24, 32, 3), np.nan), "finite"),
])
def test_image_validation(example, bad_img, error):
    doc, _ = example
    with pytest.raises(ValueError, match=error):
        node().select(doc, StubTensor(bad_img))


def test_original_document_is_preserved(example):
    doc, img = example
    copy = deepcopy(doc)
    node().select(doc, img, "random_n", seed=41, count=2)
    assert doc == copy


def test_workflow_wires_loader_annotations_and_image():
    graph = json.loads(WORKFLOW.read_text())
    assert len(graph["nodes"]) == 5
    assert len(graph["links"]) == 5
    kinds = {n["id"]: n["type"] for n in graph["nodes"]}
    assert list(kinds.values()).count("SAPDatasetLoader") == 1
    assert list(kinds.values()).count("SAPROISelector") == 1
    assert [x for x in graph["links"] if x[5] == "SAP_ANNOTATIONS"] == [
        [3, 1, 1, 4, 0, "SAP_ANNOTATIONS"]
    ]
    assert [4, 1, 0, 4, 1, "IMAGE"] in graph["links"]
    assert [5, 4, 1, 5, 0, "IMAGE"] in graph["links"]
    assert all(x[1] in kinds and x[3] in kinds for x in graph["links"])
    assert graph["nodes"][0]["widgets_values"][0] == ""
    assert "D:/" not in WORKFLOW.read_text()
    selector = next(n for n in graph["nodes"] if n["type"] == "SAPROISelector")
    # ComfyUI frontend inserts an extra control_after_generate widget after INT seed.
    # Without its persisted value, count/instance_index/class_name shift one slot.
    assert selector["widgets_values"] == [
        "random_n", 42017, "fixed", 3, 0, "", False, False
    ]


def test_comfyui_mapping_declares_both_nodes():
    # A subprocess tests classic ComfyUI discovery and remains torch-free.
    import subprocess
    result = subprocess.run(
        [sys.executable, "-c", (
            "import sys; sys.path.insert(0, r'" + str(ROOT / "custom_nodes") + "'); "
            "import sap_dataset_loader; "
            "print(sorted(sap_dataset_loader.NODE_CLASS_MAPPINGS))"
        )], capture_output=True, text=True, check=True,
    )
    assert "SAPROISelector" in result.stdout
    assert "SAPDatasetLoader" in result.stdout
