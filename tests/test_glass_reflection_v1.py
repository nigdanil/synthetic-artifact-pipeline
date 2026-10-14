"""Glass Reflection V1 unit and export tests use synthetic images only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from test_artifact_exporter import prepared

from sapipeline.effects.glass_reflection import (
    DEFAULT_GLASS_QUAD,
    GlassReflectionError,
    apply_glass_reflection,
)
from sapipeline.export import ArtifactExportError, export_artifact
from sapipeline.qa import validate_artifact


def sample() -> Image.Image:
    pixels = np.full((180, 240, 3), (50, 90, 135), dtype=np.uint8)
    pixels[35:130, 20:220] = (110, 120, 140)
    return Image.fromarray(pixels, mode="RGB")


def test_seed_reproducibility_and_nonzero_effect() -> None:
    img = sample()
    before = np.asarray(img).copy()
    a = apply_glass_reflection(img, seed=17)
    b = apply_glass_reflection(img, seed=17)
    c = apply_glass_reflection(img, seed=18)
    assert np.array_equal(a.effect_mask, b.effect_mask)
    assert np.array_equal(np.asarray(a.image), np.asarray(b.image))
    assert not np.array_equal(a.effect_mask, c.effect_mask)
    assert np.any(np.asarray(a.image) != before)
    assert np.array_equal(np.asarray(img), before)


def test_outside_glass_polygon_is_unchanged() -> None:
    img = sample()
    quad = ((0.2, 0.25), (0.8, 0.25), (0.8, 0.75), (0.2, 0.75))
    result = apply_glass_reflection(img, quad_norm=quad)
    mask = result.effect_mask
    diff = np.any(np.asarray(img) != np.asarray(result.image), axis=-1)
    assert not np.any(diff & (mask == 0))
    assert np.all(mask[:38] == 0) and np.all(mask[145:] == 0)
    assert np.all(mask[:, :40] == 0) and np.all(mask[:, 200:] == 0)
    assert np.any(diff)
    assert mask.dtype == np.float32


@pytest.mark.parametrize("quad", [
    (), [[0., 0.], [1., 1.], [1., 0.], [0., 1.]],  # self intersecting
    [[0.1, 0.1], [0.2, 0.1], [0.2, 0.11], [0.1, 0.11]],  # too small
    [[-0.1, 0.1], [1., 0.1], [1., .9], [0., .9]],
    [[0.1, 0.1], [0.9, 0.1], [0.9, 0.8], [0.8, 0.2]],  # concave
])
def test_invalid_geometry_is_rejected(quad) -> None:
    with pytest.raises(GlassReflectionError, match="quad_norm"):
        apply_glass_reflection(sample(), quad_norm=quad)


@pytest.mark.parametrize("kwargs", [
    {"intensity": float("nan")}, {"band_width": 0.0}, {"angle_deg": 80},
    {"seed": -1}, {"seed": True}, {"band_count": 4}, {"band_count": 1.0},
    {"edge_feather": float("inf")},
])
def test_invalid_effect_controls_rejected(kwargs) -> None:
    with pytest.raises(GlassReflectionError):
        apply_glass_reflection(sample(), **kwargs)


def _glass_args(tmp_path: Path):
    args = prepared(tmp_path)
    original = np.rint(args["original"]*255).astype(np.uint8)
    config = {
        "quad_norm": [[0.08, 0.12], [0.92, 0.12], [0.92, 0.88], [0.08, 0.88]],
        "seed": 17, "intensity": .70, "angle_deg": -12., "band_width": .060,
        "band_count": 2, "band_separation": .2, "edge_feather": .03,
    }
    generated = apply_glass_reflection(Image.fromarray(original, mode="RGB"), **config)
    modified = np.asarray(generated.image).astype(np.float32)/255
    mask = generated.effect_mask
    verdict = validate_artifact(args["original"], modified, mask,
                                args["annotations"], args["selection"])
    assert verdict.passed, verdict.failures
    effect = {
        "schema_version": "sap-glass-reflection-v1", "kind": "projected_strips",
        **config,
        "image_size_px": [original.shape[1], original.shape[0]],
        "changed_pixels": verdict.metrics["changed_pixels"],
        "effect_mask_active_pixels": verdict.metrics["mask_active_pixels"],
        "manual_review_required": True, "training_approved": False,
    }
    args.update(modified=modified, mask=mask, qa_report=verdict.to_json(),
                effect_summary=json.dumps(effect))
    return args


def test_exporter_accepts_real_reproducible_glass_and_preserves_seg(tmp_path: Path) -> None:
    args = _glass_args(tmp_path)
    out = export_artifact(**args)
    manifest = json.loads((out.directory / "manifest.json").read_text())
    assert manifest["effects"][0]["schema_version"] == "sap-glass-reflection-v1"
    assert manifest["qa"]["status"] == "PASS"
    assert manifest["qa"]["training_approved"] is False
    assert (out.directory / "annotations/my_run_001.seg").exists()


def test_exporter_rejects_glass_metadata_forgery_without_writing(tmp_path: Path) -> None:
    args = _glass_args(tmp_path)
    data = json.loads(args["effect_summary"])
    data["seed"] = 999  # Same image/mask but dishonest reproducibility metadata.
    args["effect_summary"] = json.dumps(data)
    with pytest.raises(ArtifactExportError, match="differ from declared parameters"):
        export_artifact(**args)
    assert not (tmp_path / "outputs").exists()


def test_exporter_rejects_bad_quad_in_effect_metadata(tmp_path: Path) -> None:
    args = _glass_args(tmp_path)
    data = json.loads(args["effect_summary"])
    data["quad_norm"] = [[0, 0], [0, 0], [0, 0], [0, 0]]
    args["effect_summary"] = json.dumps(data)
    with pytest.raises(ArtifactExportError, match="Invalid glass effect metadata"):
        export_artifact(**args)


def test_stage8_audit_reads_glass_bundle_as_valid_but_not_approved(tmp_path: Path) -> None:
    from sapipeline.qa.acceptance import audit_batch

    args = _glass_args(tmp_path)
    export_artifact(**args)
    result = audit_batch(tmp_path)
    assert result.report["status"] == "NOT_READY"
    assert result.report["totals"]["valid"] == 1
    assert result.report["totals"]["manually_approved"] == 0
    assert result.report["training_approved"] is False


def test_template_is_portable_and_wired_to_actual_mask() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "workflows/templates/sap_glass_reflection_v1.json").read_text()
    assert "D:/" not in source and "C:/" not in source
    assert "LOCAL_READY" not in source
    graph = json.loads(source)
    nodes = {node["id"]: node for node in graph["nodes"]}
    assert nodes[10]["type"] == "SAPGlassReflection"
    assert nodes[1]["widgets_values"][0] == ""
    assert nodes[17]["widgets_values"][0] == ""
    assert nodes[17]["widgets_values"][1] == "glass_reflection_0001"
    links = {v[0]: v for v in graph["links"]}
    assert links[17][1:5] == [10, 1, 13, 2]  # real effect MASK -> QA
    assert links[26][1:5] == [10, 1, 17, 4]  # same MASK -> exporter
    assert links[25][1:5] == [10, 0, 17, 3]  # actual modified pixels
    assert not any(node["type"] == "SaveImage" for node in nodes.values())


def test_default_quad_is_valid() -> None:
    result = apply_glass_reflection(sample(), quad_norm=DEFAULT_GLASS_QUAD)
    assert result.effect_mask.shape == (180, 240)


def test_comfy_adapter_uses_spatial_mask_and_does_not_mutate_tensor() -> None:
    import importlib.util

    torch = pytest.importorskip("torch")

    module_path = Path(__file__).resolve().parents[1] / "custom_nodes/sap_dataset_loader/glass.py"
    spec = importlib.util.spec_from_file_location("glass_adapter_v1", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rgb = np.asarray(sample()).astype(np.float32)/255.0
    tensor = torch.from_numpy(rgb.copy()).unsqueeze(0)
    snapshot = tensor.clone()
    image, mask, summary = module.SAPGlassReflection().apply(
        tensor, json.dumps(DEFAULT_GLASS_QUAD), seed=42017,
    )
    report = json.loads(summary)
    assert image.shape == tensor.shape
    assert mask.shape == (1, 180, 240)
    assert report["schema_version"] == "sap-glass-reflection-v1"
    assert report["kind"] == "projected_strips"
    assert report["training_approved"] is False
    assert torch.equal(tensor, snapshot)
    altered = image.numpy()[0]
    mask_grid = mask.numpy()[0]
    assert np.array_equal(altered[mask_grid == 0], rgb[mask_grid == 0])
    assert np.any(altered[mask_grid > 0] != rgb[mask_grid > 0])


def test_comfy_adapter_rejects_invalid_json() -> None:
    import importlib.util

    torch = pytest.importorskip("torch")

    module_path = Path(__file__).resolve().parents[1] / "custom_nodes/sap_dataset_loader/glass.py"
    spec = importlib.util.spec_from_file_location("glass_adapter_invalid", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="quad_json"):
        module.SAPGlassReflection().apply(torch.zeros((1, 48, 64, 3)), "nope")
