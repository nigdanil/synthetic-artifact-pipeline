"""Public synthetic-only tests for seeded nonrectangular ROI glare (stage 6.7)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from sapipeline.annotations import parse_seg
from sapipeline.effects import apply_realistic_glare, build_glare_mask
from sapipeline.effects.highlight import EffectError
from sapipeline.qa import validate_artifact
from sapipeline.selection import select_rois


@pytest.fixture
def scene():
    doc = parse_seg(json.dumps({"filename": "fixture.jpg", "selectedList": [
        {"name": "sample_a", "rect": {"x": 12, "y": 15, "width": 50, "height": 40}},
        {"name": "sample_b", "rect": {"x": 85, "y": 25, "width": 45, "height": 50}},
        {"name": "sample_c", "rect": {"x": 35, "y": 76, "width": 45, "height": 30}},
    ]}), image_width=160, image_height=120)
    chosen = select_rois(doc, mode="random_n", seed=42017, count=3)
    mask = np.zeros((120, 160), dtype=np.float32)
    for target in chosen.targets:
        x1, y1, x2, y2 = map(int, target.bbox_xyxy_px)
        mask[y1:y2, x1:x2] = 1.0
    pixels = np.full((120, 160, 3), 80, dtype=np.uint8)
    return Image.fromarray(pixels, "RGB"), mask, doc, chosen


@pytest.mark.parametrize("kind", ["elliptical", "directional"])
def test_mask_is_nonrectangular_seeded_and_qa_pass(scene, kind):
    image, mask, doc, chosen = scene
    original = np.asarray(image).copy()
    raw_mask = mask.copy()
    first = apply_realistic_glare(image, mask, chosen, kind=kind)
    second = apply_realistic_glare(image, mask, chosen, kind=kind)
    pixels = np.asarray(first.image)
    assert np.array_equal(pixels, np.asarray(second.image))
    assert np.array_equal(first.effect_mask, second.effect_mask)
    assert first.effect_mask.dtype == np.float32
    assert first.effect_mask.shape == mask.shape
    assert np.all(first.effect_mask[mask == 0] == 0)
    assert np.all(pixels[mask == 0] == original[mask == 0])
    assert np.count_nonzero(pixels != original) > 0
    # Taper makes glare near bbox border weaker than inside the bbox.
    first_box = chosen.targets[0].bbox_xyxy_px
    x1, y1, x2, y2 = map(int, first_box)
    region = first.effect_mask[y1:y2, x1:x2]
    assert region.max() > region[0].max() + 0.1
    assert region.max() > region[:, 0].max() + 0.1
    assert np.array_equal(np.asarray(image), original)
    assert np.array_equal(mask, raw_mask)
    qa = validate_artifact(
        original.astype(np.float32) / 255.0,
        pixels.astype(np.float32) / 255.0,
        first.effect_mask, doc, chosen,
    )
    assert qa.passed, qa.failures
    assert qa.metrics["outside_mask_changed_pixels"] == 0
    assert qa.metrics["newly_clipped_pixels"] == 0


def test_seed_and_angles_change_masks_reproducibly(scene):
    _image, mask, _doc, chosen = scene
    a = build_glare_mask(mask, chosen, seed=123)
    b = build_glare_mask(mask, chosen, seed=123)
    c = build_glare_mask(mask, chosen, seed=124)
    d = build_glare_mask(mask, chosen, seed=123, angle_deg=65)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert not np.array_equal(a, d)


def test_ellipse_vs_directional_has_distinct_shape(scene):
    _image, mask, _doc, chosen = scene
    a = build_glare_mask(mask, chosen, kind="elliptical")
    b = build_glare_mask(mask, chosen, kind="directional")
    assert not np.array_equal(a, b)


def test_zero_mask_or_intensity_keeps_image_identical(scene):
    image, mask, _doc, chosen = scene
    zero = np.zeros_like(mask)
    result = apply_realistic_glare(image, zero, chosen)
    assert np.array_equal(np.asarray(result.image), np.asarray(image))
    assert np.count_nonzero(result.effect_mask) == 0
    result = apply_realistic_glare(image, mask, chosen, intensity=0)
    assert np.array_equal(np.asarray(result.image), np.asarray(image))


@pytest.mark.parametrize("argument,value", [
    ("kind", "rectangle"), ("angle_deg", 181), ("radius_scale", 0),
    ("aspect_ratio", 1.1), ("falloff", float("nan")),
    ("seed", -1), ("seed", True), ("center_jitter", 0.5),
])
def test_invalid_parameters_fail(scene, argument, value):
    image, mask, _doc, chosen = scene
    with pytest.raises((EffectError, TypeError)):
        apply_realistic_glare(image, mask, chosen, **{argument: value})


@pytest.mark.parametrize("mask_change", ["nonfinite", "negative", "over_one", "wrong_shape"])
def test_invalid_mask_fails(scene, mask_change):
    _image, mask, _doc, chosen = scene
    if mask_change == "wrong_shape":
        mask = mask[:, :-1]
    else:
        mask = mask.copy()
        mask[0, 0] = {"nonfinite": np.nan, "negative": -0.1, "over_one": 1.01}[mask_change]
    with pytest.raises(EffectError):
        if mask_change == "wrong_shape":
            apply_realistic_glare(_image, mask, chosen)
        else:
            build_glare_mask(mask, chosen)


def test_input_is_required_to_be_rgb(scene):
    image, mask, _doc, chosen = scene
    with pytest.raises(EffectError, match="RGB"):
        apply_realistic_glare(image.convert("L"), mask, chosen)
