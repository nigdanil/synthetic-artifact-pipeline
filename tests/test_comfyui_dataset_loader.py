"""CPU-only tests for the stand-alone classic ComfyUI custom node."""

from __future__ import annotations

import importlib.util
import json
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image

from sapipeline.dataset import build_index

ROOT = Path(__file__).resolve().parents[1]
NODE_FILE = ROOT / "custom_nodes/sap_dataset_loader/node.py"
WORKFLOW = ROOT / "workflows/templates/sap_dataset_loader_v1.json"


def _get_node():
    spec = importlib.util.spec_from_file_location("sap_loader_local", NODE_FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SAPDatasetLoader


@pytest.fixture
def project(tmp_path: Path) -> Path:
    src = tmp_path / "src" / "sapipeline" / "dataset"
    src.mkdir(parents=True)
    (src / "loader.py").write_text("# test-marker", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='synthetic-artifact-pipeline'\n")
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    image_buffer = BytesIO()
    Image.new("RGB", (12, 8), (95, 105, 115)).save(image_buffer, format="JPEG")
    annotation = json.dumps({
        "filename": "demo.jpg", "selectedList": [
            {"name": "fictional_product", "rect": {"x": 2, "y": 1, "width": 4, "height": 3}}
        ]
    }).encode("utf-8")
    archive = raw / "for_learning.zip"
    with ZipFile(archive, "w") as zipped:
        zipped.writestr("img/demo.jpg", image_buffer.getvalue())
        zipped.writestr("seg/demo.seg", annotation)
    index, _ = build_index({"learning": archive})
    index_file = tmp_path / "data/work/index/dataset_index.json"
    index_file.parent.mkdir(parents=True)
    index_file.write_text(json.dumps(index), encoding="utf-8")
    return tmp_path


def test_node_metadata_and_lazy_import() -> None:
    node = _get_node()
    assert node.RETURN_TYPES == ("IMAGE", "SAP_ANNOTATIONS", "STRING")
    assert node.FUNCTION == "load"
    assert node.CATEGORY == "SAP/Dataset"
    assert "project_root" in node.INPUT_TYPES()["required"]
    assert np.isnan(node.IS_CHANGED())


def test_real_zip_loader_tensor_and_annotations(project: Path, monkeypatch) -> None:
    # SAP dev environment must NOT need torch; the real ComfyUI supplies it.
    # This fake verifies the converter's shape and normalization contract.
    class StubTensor:
        def __init__(self, array):
            self.array = array

        def to(self, dtype):
            self.array = self.array.astype(dtype)
            return self

        def div_(self, divisor):
            self.array /= divisor
            return self

        def unsqueeze(self, axis):
            self.array = np.expand_dims(self.array, axis)
            return self

    fake_torch = SimpleNamespace(from_numpy=StubTensor, float32=np.float32)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    node = _get_node()
    # Test fixture's fake checkout has only the marker file. Use the public
    # package already loaded by pytest but keep the archive/index fixture.
    monkeypatch.setattr(node, "_core", staticmethod(lambda _root: (
        __import__("sapipeline.dataset", fromlist=["read_private_index"]), project
    )))
    image, annotations, summary = node().load(str(project), "learning", 0, "")
    assert isinstance(image, StubTensor)
    assert image.array.dtype == np.float32
    assert image.array.shape == (1, 8, 12, 3)
    assert 0 <= image.array.min() <= image.array.max() <= 1
    assert len(annotations.objects) == 1
    assert annotations.objects[0].bbox_xyxy_px == (2, 1, 6, 4)
    details = json.loads(summary)
    assert details["bbox_count"] == 1
    assert details["size_px"] == [12, 8]
    assert "fictional_product" not in summary
    assert "demo.jpg" not in summary


@pytest.mark.parametrize("kwargs, expected", [
    ({"project_root": "", "split": "learning", "sample_index": 0, "image_member": ""}, "project_root"),
    ({"project_root": "/tmp/x", "split": "other", "sample_index": 0, "image_member": ""}, "split"),
    ({"project_root": "/tmp/x", "split": "learning", "sample_index": -1, "image_member": ""}, "sample_index"),
    ({"project_root": "/tmp/x", "split": "learning", "sample_index": 1, "image_member": "img/a.jpg"}, "sample_index"),
])
def test_invalid_inputs_are_rejected(kwargs, expected) -> None:
    with pytest.raises(ValueError, match=expected):
        _get_node()().load(**kwargs)


def test_refuse_nonproject_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="project_root"):
        _get_node()._core(str(tmp_path))
    with pytest.raises(ValueError, match="absolute"):
        _get_node()._core("../synthetic-artifact-pipeline-public")


def test_real_project_module_resolution() -> None:
    dataset, resolved = _get_node()._core(str(ROOT))
    assert resolved == ROOT
    assert callable(dataset.load_indexed_pair)


def test_workflow_template_consistent_and_has_no_private_path() -> None:
    workflow = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    assert len(workflow["nodes"]) == 3
    assert len(workflow["links"]) == 2
    assert {node["type"] for node in workflow["nodes"]} == {
        "SAPDatasetLoader", "PreviewImage", "SaveImage"
    }
    loader = workflow["nodes"][0]
    assert loader["widgets_values"] == ["", "learning", 0, ""]
    ids = {node["id"] for node in workflow["nodes"]}
    for link in workflow["links"]:
        assert link[1] in ids and link[3] in ids
        assert link[5] == "IMAGE"
