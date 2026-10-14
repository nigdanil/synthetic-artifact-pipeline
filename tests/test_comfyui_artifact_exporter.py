"""Portable workflow and ComfyUI adapter checks; no source data in repository."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from custom_nodes.sap_dataset_loader import NODE_CLASS_MAPPINGS
from custom_nodes.sap_dataset_loader.export import SAPArtifactExporter
from custom_nodes.sap_dataset_loader.node import SAPDatasetLoader


class FakeTensor:
    """Only the adapter's minimal tensor protocol, no torch in the dev venv."""

    def __init__(self, data: np.ndarray):
        self.data = data

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "workflows/templates/sap_artifact_exporter_v1.json"


def test_exporter_metadata_and_required_typed_connections() -> None:
    assert NODE_CLASS_MAPPINGS["SAPArtifactExporter"] is SAPArtifactExporter
    assert SAPArtifactExporter.OUTPUT_NODE is True
    assert SAPArtifactExporter.CATEGORY == "SAP/Export"
    assert SAPArtifactExporter.RETURN_NAMES == ("export_directory", "export_summary")
    required = SAPArtifactExporter.INPUT_TYPES()["required"]
    assert required["source_context"][0] == "SAP_SOURCE_CONTEXT"
    assert required["qa_pass"][0] == "BOOLEAN"
    assert required["qa_report"][0] == "STRING"
    assert required["selection"][0] == "SAP_ROI_SELECTION"
    assert SAPDatasetLoader.RETURN_NAMES[-1] == "source_context"
    assert np.isnan(SAPArtifactExporter.IS_CHANGED())


def test_adapter_passes_detached_unmutated_tensors(monkeypatch, tmp_path: Path) -> None:
    image = FakeTensor(np.random.default_rng(17).random((1, 6, 8, 3), dtype=np.float32))
    modified = FakeTensor(image.data.copy())
    modified.data[:, 2:4, 2:5] *= 0.85
    mask = FakeTensor(np.zeros((1, 6, 8), dtype=np.float32))
    mask.data[:, 2:4, 2:5] = 1
    snapshot = [v.data.copy() for v in (image, modified, mask)]
    received: dict = {}

    def save(**kwargs):
        received.update(kwargs)
        return SimpleNamespace(
            directory=tmp_path / "outputs/sap_runs/hello",
            manifest={"run_id": "hello", "output_sha256": {"image": "sha"}},
        )

    monkeypatch.setattr(SAPDatasetLoader, "_core", staticmethod(lambda _: (None, tmp_path)))
    original_import = __import__("importlib").import_module
    monkeypatch.setattr(
        "custom_nodes.sap_dataset_loader.export.importlib.import_module",
        lambda name: SimpleNamespace(export_artifact=save)
        if name == "sapipeline.export" else original_import(name),
    )
    path, summary = SAPArtifactExporter().export(
        str(tmp_path), "hello", image, modified, mask,
        object(), object(), True, "{}", {"split": "learning"}, "{}", "{}",
    )
    assert Path(path).parts[-3:] == ("outputs", "sap_runs", "hello")
    assert json.loads(summary)["training_approved"] is False
    assert received["original"].shape == (6, 8, 3)
    assert received["mask"].shape == (6, 8)
    assert received["project_root"] == tmp_path
    for tensor, copy in zip((image, modified, mask), snapshot, strict=True):
        assert np.array_equal(tensor.data, copy)


def test_adapter_rejects_empty_project_path_without_save() -> None:
    with pytest.raises(ValueError, match="project_root"):
        SAPArtifactExporter().export(
            "", "x", FakeTensor(np.zeros((1, 3, 4, 3), dtype=np.float32)),
            FakeTensor(np.zeros((1, 3, 4, 3), dtype=np.float32)),
            FakeTensor(np.zeros((1, 3, 4), dtype=np.float32)),
            object(), object(), True, "{}", {}, "{}", "{}",
        )


def test_public_template_wiring_and_absence_of_private_paths() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "D:/" not in text and "C:/" not in text
    assert "for_learning.zip" not in text and "/Users/" not in text
    doc = json.loads(text)
    nodes = {n["id"]: n for n in doc["nodes"]}
    assert len(nodes) == 15
    assert len(doc["links"]) == 31
    assert {n["type"] for n in doc["nodes"]} >= {
        "SAPDatasetLoader", "SAPRealisticGlare", "SAPQAValidator", "SAPArtifactExporter"
    }
    assert all(n["type"] != "SaveImage" for n in nodes.values())
    assert nodes[1]["widgets_values"][0] == ""
    assert nodes[17]["widgets_values"] == ["", "glare_0001"]
    assert nodes[1]["outputs"][3]["name"] == "source_context"
    linked = {link[4]: link for link in doc["links"] if link[3] == 17}
    assert {v["name"] for v in nodes[17]["inputs"][2:]} == {
        "original", "modified", "mask", "annotations", "selection", "qa_pass",
        "qa_report", "source_context", "effect_summary", "mask_summary",
    }
    assert linked[7][1:3] == [13, 0]
    assert linked[8][1:3] == [13, 1]
    assert linked[9][1:3] == [1, 3]
    assert linked[4][1:3] == [10, 1]
    for link_id, src, slot, dest, dstslot, _kind in doc["links"]:
        assert src in nodes and dest in nodes
        assert nodes[dest]["inputs"][dstslot]["link"] == link_id
        assert link_id in (nodes[src]["outputs"][slot].get("links") or [])
