from __future__ import annotations

import numpy as np
from PIL import Image
import pytest

from sapipeline.effects import EffectError, apply_highlight


def _sample_image() -> Image.Image:
    data = np.array(
        [
            [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
            [[100, 110, 120], [130, 140, 150], [160, 170, 180]],
        ],
        dtype=np.uint8,
    )
    return Image.fromarray(data, mode="RGB")


def test_zero_mask_preserves_all_pixels() -> None:
    image = _sample_image()
    mask = np.zeros((2, 3), dtype=np.float32)
    result = apply_highlight(image, mask)
    assert np.array_equal(np.asarray(result.image), np.asarray(image))


@pytest.mark.parametrize("kind", ["soft_glow", "hotspot"])
def test_positive_mask_changes_target_pixels(kind: str) -> None:
    image = _sample_image()
    mask = np.zeros((2, 3), dtype=np.float32)
    mask[0, 1] = 1.0
    result = apply_highlight(image, mask, kind=kind, intensity=1.0)
    out = np.asarray(result.image)
    src = np.asarray(image)
    assert not np.array_equal(out[0, 1], src[0, 1])
    assert np.array_equal(out[1, 0], src[1, 0])


@pytest.mark.parametrize(
    ("mask", "message"),
    [
        (np.zeros((2, 2, 1), dtype=np.float32), "2-dimensional"),
        (np.array([[0, np.nan, 1], [0, 0, 1]], dtype=np.float32), "finite"),
    ],
)
def test_invalid_mask_rejected(mask: np.ndarray, message: str) -> None:
    image = _sample_image()
    with pytest.raises(EffectError, match=message):
        apply_highlight(image, mask)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"kind": "bad"}, "kind"),
        ({"intensity": -0.1}, "intensity"),
        ({"white_boost": 2}, "white_boost"),
        ({"gamma_lift": 2}, "gamma_lift"),
    ],
)
def test_invalid_parameters_rejected(kwargs, message: str) -> None:
    image = _sample_image()
    mask = np.ones((2, 3), dtype=np.float32)
    with pytest.raises(EffectError, match=message):
        apply_highlight(image, mask, **kwargs)


def test_output_shape_and_mode_preserved() -> None:
    image = _sample_image()
    mask = np.ones((2, 3), dtype=np.float32)
    result = apply_highlight(image, mask, kind="soft_glow")
    assert result.image.size == image.size
    assert result.image.mode == "RGB"
