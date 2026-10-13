"""Classic SAP Realistic Glare node with fake torch; no ComfyUI or real data."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sapipeline.annotations import parse_seg
from sapipeline.qa import validate_artifact
from sapipeline.selection import select_rois

ROOT = Path(__file__).resolve().parents[1]
NODE = ROOT / "custom_nodes/sap_dataset_loader/glare.py"
WORKFLOW = ROOT / "workflows/templates/sap_realistic_glare_v1.json"


class FakeTensor:
    def __init__(self, array, device=None):
        self.array = array
        self.device = device

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def unsqueeze(self, axis):
        return FakeTensor(np.expand_dims(self.array, axis), self.device)

    def to(self, *, device=None):
        self.device = device
        return self


@pytest.fixture(autouse=True)
def fake_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(from_numpy=FakeTensor))


def node():
    spec = importlib.util.spec_from_file_location("sap_local_glare", NODE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SAPRealisticGlare()


@pytest.fixture
def example():
    doc = parse_seg(json.dumps({"filename": "test.jpg", "selectedList": [
        {"name": "sample_a", "rect": {"x": 6, "y": 8, "width": 25, "height": 24}},
        {"name": "sample_b", "rect": {"x": 45, "y": 10, "width": 27, "height": 23}},
    ]}), image_width=80, image_height=50)
    selected = select_rois(doc, mode="random_n", count=2, seed=42)
    rgb = np.full((1, 50, 80, 3), 0.20, dtype=np.float32)
    mask = np.zeros((1, 50, 80), dtype=np.float32)
    mask[0, 8:32, 6:31] = 1
    mask[0, 10:33, 45:72] = 1
    return FakeTensor(rgb, "cuda:0"), FakeTensor(mask, "cpu"), doc, selected


def test_registration_and_contract():
    cls = type(node())
    assert cls.RETURN_TYPES == ("IMAGE", "MASK", "STRING")
    assert cls.FUNCTION == "apply"
    assert "selection" in cls.INPUT_TYPES()["required"]
    assert np.isnan(cls.IS_CHANGED())
    proc = subprocess.run([sys.executable, "-c", (
        "import sys;sys.path.insert(0,r'" + str(ROOT / "custom_nodes") + "');"
        "import sap_dataset_loader;print(sorted(sap_dataset_loader.NODE_CLASS_MAPPINGS))"
    )], capture_output=True, text=True, check=True)
    assert "SAPRealisticGlare" in proc.stdout
    assert "SAPQAValidator" in proc.stdout


@pytest.mark.parametrize("kind", ["elliptical", "directional"])
def test_end_to_end_qa_pass_unchanged_inputs_and_determinism(example, kind):
    image, mask, doc, selected = example
    original, weights = image.array.copy(), mask.array.copy()
    output, effect_mask, summary = node().apply(image, mask, selected, kind=kind)
    second, second_mask, second_summary = node().apply(image, mask, selected, kind=kind)
    assert output.device == image.device
    assert effect_mask.device == mask.device
    assert output.array.shape == original.shape
    assert effect_mask.array.shape == weights.shape
    assert np.array_equal(output.array, second.array)
    assert np.array_equal(effect_mask.array, second_mask.array)
    assert summary == second_summary
    assert np.array_equal(output.array[0][effect_mask.array[0] == 0],
                          original[0][effect_mask.array[0] == 0])
    assert np.array_equal(image.array, original)
    assert np.array_equal(mask.array, weights)
    details = json.loads(summary)
    assert details["training_approved"] is False
    assert details["scope"] == "selected_bbox_local_not_glass_plane"
    qa = validate_artifact(original[0], output.array[0], effect_mask.array[0], doc, selected)
    assert qa.passed, qa.failures


def test_invalid_comfy_image_or_mask_is_rejected(example):
    image, mask, _doc, selected = example
    with pytest.raises(TypeError, match="ComfyUI tensors"):
        node().apply(object(), mask, selected)
    with pytest.raises(ValueError, match="MASK"):
        node().apply(image, FakeTensor(mask.array[:, :-1]), selected)
    corrupted = image.array.copy()
    corrupted[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        node().apply(FakeTensor(corrupted), mask, selected)


def test_public_workflow_is_connected_and_portable():
    graph = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    index = {item["id"]: item for item in graph["nodes"]}
    assert len(index) == 16
    assert index[1]["widgets_values"][0] == ""
    glare = next(n for n in graph["nodes"] if n["type"] == "SAPRealisticGlare")
    qa = next(n for n in graph["nodes"] if n["type"] == "SAPQAValidator")
    assert glare["widgets_values"][0] == "directional"
    assert glare["widgets_values"][-2] == "fixed"  # ComfyUI seed widget mode.
    links = {(source, output, target, position): typ
             for _id, source, output, target, position, typ in graph["links"]}
    assert links[(glare["id"], 0, qa["id"], 1)] == "IMAGE"
    assert links[(glare["id"], 1, qa["id"], 2)] == "MASK"
    assert links[(4, 0, glare["id"], 2)] == "SAP_ROI_SELECTION"
    assert any(l[1] == qa["id"] and l[2] == 2 for l in graph["links"])
    assert graph["last_link_id"] == max(link[0] for link in graph["links"])
