"""Task 6.5: IMAGE+MASK ComfyUI effect adapter without a torch dependency."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NODE_FILE = ROOT / "custom_nodes/sap_dataset_loader/highlight.py"
WORKFLOW_FILE = ROOT / "workflows/templates/sap_highlight_effect_v1.json"


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

    def unsqueeze(self, axis: int):
        return FakeTensor(np.expand_dims(self.array, axis), self.device)

    def to(self, *, device=None):
        self.device = device
        return self


@pytest.fixture(autouse=True)
def fake_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(from_numpy=FakeTensor))


def node():
    spec = importlib.util.spec_from_file_location("sap_highlight_local", NODE_FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SAPHighlightEffect()


@pytest.fixture
def pair():
    rgb = np.full((1, 24, 32, 3), 0.20, dtype=np.float32)
    rgb[0, 8:12, 7:15, :] = 0.31
    mask = np.zeros((1, 24, 32), dtype=np.float32)
    mask[0, 5:17, 5:21] = 1.0
    return FakeTensor(rgb), FakeTensor(mask)


def test_registration_and_contract():
    cls = type(node())
    assert cls.RETURN_TYPES == ("IMAGE", "STRING")
    assert cls.FUNCTION == "apply"
    assert "mask" in cls.INPUT_TYPES()["required"]
    assert np.isnan(cls.IS_CHANGED())
    imported = subprocess.run(
        [sys.executable, "-c", (
            "import sys;sys.path.insert(0,r'" + str(ROOT / "custom_nodes") + "');"
            "import sap_dataset_loader;print(sorted(sap_dataset_loader.NODE_CLASS_MAPPINGS))"
        )], check=True, capture_output=True, text=True,
    )
    assert "SAPHighlightEffect" in imported.stdout
    assert "SAPMaskBuilder" in imported.stdout


@pytest.mark.parametrize("kind", ["soft_glow", "hotspot"])
def test_apply_changes_only_mask_and_does_not_mutate_inputs(pair, kind):
    image, mask = pair
    old_image, old_mask = image.array.copy(), mask.array.copy()
    effect, report = node().apply(image, mask, effect_kind=kind)
    out = effect.array
    assert out.shape == old_image.shape
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))
    assert np.any(out[0][mask.array[0] > 0] != old_image[0][mask.array[0] > 0])
    assert np.array_equal(out[0][mask.array[0] == 0], old_image[0][mask.array[0] == 0])
    assert np.array_equal(image.array, old_image)
    assert np.array_equal(mask.array, old_mask)
    details = json.loads(report)
    assert details["effect_kind"] == kind
    assert details["image_size_px"] == [32, 24]
    assert details["training_approval"] == "pending_manual_review"


def test_zero_mask_is_identity_even_for_non_quantized_float_input():
    image = FakeTensor(np.full((1, 4, 6, 3), 0.33333334, dtype=np.float32))
    mask = FakeTensor(np.zeros((1, 4, 6), dtype=np.float32))
    out, report = node().apply(image, mask)
    assert np.array_equal(out.array, image.array)
    assert json.loads(report)["changed_pixels"] == 0



def test_zero_intensity_is_exact_identity(pair):
    image, mask = pair
    image.array[0, 0, 0] = np.float32(0.33333334)
    before = image.array.copy()
    result, _ = node().apply(image, mask, intensity=0.0)
    assert np.array_equal(result.array, before)

def test_effect_is_deterministic(pair):
    image, mask = pair
    first, summary_a = node().apply(image, mask)
    second, summary_b = node().apply(image, mask)
    assert np.array_equal(first.array, second.array)
    assert summary_a == summary_b


def test_device_is_preserved(pair):
    image, mask = pair
    image.device = "cuda:0"
    out, _ = node().apply(image, mask)
    assert out.device == "cuda:0"


@pytest.mark.parametrize("wrong", ["image", "mask"])
def test_wrong_tensor_type_rejected(pair, wrong):
    image, mask = pair
    kwargs = {"image": image, "mask": mask}
    kwargs[wrong] = object()
    with pytest.raises(TypeError, match="torch tensors"):
        node().apply(**kwargs)


@pytest.mark.parametrize("bad_mask", [
    np.zeros((24, 32), dtype=np.float32),
    np.zeros((1, 23, 32), dtype=np.float32),
    np.zeros((2, 24, 32), dtype=np.float32),
])
def test_mask_shape_rejected(pair, bad_mask):
    with pytest.raises(ValueError, match="MASK must"):
        node().apply(pair[0], FakeTensor(bad_mask))


@pytest.mark.parametrize("data, match", [
    (np.array([np.nan]), "MASK values"),
    (np.array([-0.01]), "MASK values"),
    (np.array([1.01]), "MASK values"),
])
def test_invalid_mask_values_rejected(pair, data, match):
    mask = np.zeros_like(pair[1].array)
    mask[0, 0, 0] = data.item()
    with pytest.raises(ValueError, match=match):
        node().apply(pair[0], FakeTensor(mask))


@pytest.mark.parametrize("params", [
    {"effect_kind": "bogus"},
    {"intensity": 1.1},
    {"white_boost": -0.01},
    {"gamma_lift": float("nan")},
])
def test_invalid_parameters_rejected(pair, params):
    with pytest.raises(ValueError):
        node().apply(*pair, **params)


def test_bad_image_shape_and_values_rejected(pair):
    with pytest.raises(ValueError, match="RGB IMAGE"):
        node().apply(FakeTensor(np.zeros((2, 24, 32, 3), dtype=np.float32)), pair[1])
    bad = pair[0].array.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="IMAGE values"):
        node().apply(FakeTensor(bad), pair[1])


def test_workflow_contract_and_no_local_paths():
    graph = json.loads(WORKFLOW_FILE.read_text(encoding="utf-8"))
    assert len(graph["nodes"]) == 12
    assert len(graph["links"]) == 14
    types = [item["type"] for item in graph["nodes"]]
    assert "SAPDatasetLoader" in types
    assert "SAPROISelector" in types
    assert "SAPMaskBuilder" in types
    assert "SAPHighlightEffect" in types
    assert types.count("PreviewImage") == 5
    assert types.count("SaveImage") == 2
    assert "D:/" not in WORKFLOW_FILE.read_text(encoding="utf-8")
    loader = next(n for n in graph["nodes"] if n["type"] == "SAPDatasetLoader")
    assert loader["widgets_values"][0] == ""
    effect = next(n for n in graph["nodes"] if n["type"] == "SAPHighlightEffect")
    assert effect["widgets_values"] == ["soft_glow", 0.65, 0.55, 0.18]
    links = {item[0]: item for item in graph["links"]}
    node_by_id = {n["id"]: n for n in graph["nodes"]}
    for link in links.values():
        assert link[1] in node_by_id and link[3] in node_by_id
    assert any(l[3] == effect["id"] and l[5] == "MASK" for l in links.values())
    assert any(l[3] == effect["id"] and l[5] == "IMAGE" for l in links.values())
