"""Stage 6.6: ComfyUI QA node requires no real ComfyUI installation in pytest."""

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
from sapipeline.selection import select_rois

ROOT = Path(__file__).resolve().parents[1]
NODE = ROOT / "custom_nodes/sap_dataset_loader/qa.py"
GRAPH = ROOT / "workflows/templates/sap_qa_validator_v1.json"


class FakeTensor:
    def __init__(self, array: np.ndarray, device=None):
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
    spec = importlib.util.spec_from_file_location("sap_qa_local", NODE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SAPQAValidator()


@pytest.fixture
def sample():
    doc = parse_seg(json.dumps({"filename": "fake.jpg", "selectedList": [
        {"name": "fake_a", "rect": {"x": 3, "y": 4, "width": 6, "height": 7}},
    ]}), image_width=24, image_height=16)
    selection = select_rois(doc, mode="by_index", instance_index=0)
    before = np.full((1, 16, 24, 3), 0.2, dtype=np.float32)
    after = before.copy()
    mask = np.zeros((1, 16, 24), dtype=np.float32)
    mask[0, 4:11, 3:9] = 1.0
    after[0, 4:11, 3:9] = 0.5
    return FakeTensor(before), FakeTensor(after), FakeTensor(mask), doc, selection


def test_node_registration_and_signature():
    cls = type(node())
    assert cls.RETURN_TYPES == ("BOOLEAN", "STRING", "IMAGE")
    assert set(cls.INPUT_TYPES()["required"]) == {
        "original", "modified", "mask", "annotations", "selection",
        "max_new_clipping_fraction", "max_mask_coverage", "outside_tolerance",
    }
    assert np.isnan(cls.IS_CHANGED())
    result = subprocess.run(
        [sys.executable, "-c", (
            "import sys;sys.path.insert(0,r'" + str(ROOT / "custom_nodes") + "');"
            "import sap_dataset_loader;print(sorted(sap_dataset_loader.NODE_CLASS_MAPPINGS))"
        )], check=True, capture_output=True, text=True,
    )
    assert "SAPQAValidator" in result.stdout


def test_pass_green_and_no_source_mutation(sample):
    before, after, mask = sample[:3]
    snapshots = [item.array.copy() for item in (before, after, mask)]
    passed, report, preview = node().validate(*sample)
    assert passed is True
    assert json.loads(report)["status"] == "PASS"
    assert preview.array.shape == before.array.shape
    assert np.allclose(preview.array[0, 0, 0], [0.07, 0.9, 0.2])
    for tensor, original in zip((before, after, mask), snapshots, strict=True):
        assert np.array_equal(tensor.array, original)


def test_fail_red_on_spill(sample):
    before, after, mask, doc, selection = sample
    after.array[0, 0, 12] = 0.7
    passed, report, preview = node().validate(before, after, mask, doc, selection)
    assert passed is False
    assert "no_changes_outside_mask" in json.loads(report)["failures"]
    assert np.allclose(preview.array[0, 0, 0], [1.0, 0.1, 0.1])


def test_gpu_device_retained(sample):
    sample[1].device = "cuda:0"
    _, _, preview = node().validate(*sample)
    assert preview.device == "cuda:0"


def test_invalid_tensors_fail_closed(sample):
    with pytest.raises(TypeError, match="ComfyUI tensors"):
        node().validate(object(), *sample[1:])
    wrong = FakeTensor(np.zeros((1, 16, 24, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="one RGB IMAGE"):
        node().validate(wrong, *sample[1:])
    with pytest.raises(ValueError, match="MASK must"):
        node().validate(sample[0], sample[1], FakeTensor(np.zeros((16, 24))), *sample[3:])


def test_public_workflow_structure():
    graph = json.loads(GRAPH.read_text(encoding="utf-8"))
    assert len(graph["nodes"]) == 14
    assert len(graph["links"]) == 20
    types = [n["type"] for n in graph["nodes"]]
    assert types.count("SAPQAValidator") == 1
    assert types.count("SAPHighlightEffect") == 1
    assert types.count("SaveImage") == 2  # local visual previews, not training export
    assert "D:/" not in GRAPH.read_text(encoding="utf-8")
    qa = next(n for n in graph["nodes"] if n["type"] == "SAPQAValidator")
    loader = next(n for n in graph["nodes"] if n["type"] == "SAPDatasetLoader")
    assert loader["widgets_values"][0] == ""
    assert qa["widgets_values"] == [0.25, 0.6, 0.0]
    edges = {link[0]: link for link in graph["links"]}
    assert len(edges) == len(graph["links"])
    assert {link[4] for link in edges.values() if link[3] == qa["id"]} == {0, 1, 2, 3, 4}
    node_ids = {n["id"] for n in graph["nodes"]}
    assert all(link[1] in node_ids and link[3] in node_ids for link in edges.values())
    assert any(link[1] == qa["id"] and link[5] == "IMAGE" for link in edges.values())
