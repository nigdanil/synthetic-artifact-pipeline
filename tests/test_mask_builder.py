"""Effect mask geometry: no pixels outside the padded selected ROIs."""

from __future__ import annotations

import numpy as np
import pytest

from sapipeline.annotations import ObjectAnnotation
from sapipeline.masks import MaskError, build_mask


def obj(index: int, box=(2, 3, 8, 9)) -> ObjectAnnotation:
    return ObjectAnnotation(
        instance_index=index, class_name="test_product", class_id=None,
        sku_id=None, bbox_xyxy_px=box, raw_fields={},
    )


def test_rect_mask_is_float32_half_open_and_exactly_zero_outside() -> None:
    result = build_mask(12, 11, [obj(0)], kind="rect", feather_px=3, strength=0.4)
    assert result.mask.shape == (11, 12)
    assert result.mask.dtype == np.float32
    assert np.all(result.mask[3:9, 2:8] == np.float32(0.4))
    outside = result.mask.copy()
    outside[3:9, 2:8] = 0
    assert np.count_nonzero(outside) == 0
    assert result.padded_boxes_xyxy_px == ((2, 3, 8, 9),)


def test_soft_rect_has_smooth_edges_and_fully_weighted_center() -> None:
    result = build_mask(24, 24, [obj(2, (2, 2, 18, 18))],
                        kind="soft_rect", feather_px=4)
    mask = result.mask
    assert 0 < mask[2, 2] < mask[3, 3] < mask[6, 6] == 1
    assert mask[10, 10] == 1
    assert np.count_nonzero(mask[:2, :]) == 0
    assert np.count_nonzero(mask[:, 18:]) == 0
    assert np.all(mask >= 0) and np.all(mask <= 1)


def test_feather_zero_equals_rect() -> None:
    rect = build_mask(12, 11, [obj(0)], kind="rect").mask
    soft = build_mask(12, 11, [obj(0)], kind="soft_rect", feather_px=0).mask
    assert np.array_equal(rect, soft)


def test_thin_box_and_large_feather_keep_center_nonzero() -> None:
    result = build_mask(8, 8, [obj(0, (4, 4, 5, 5))], feather_px=999)
    assert result.mask[4, 4] == 1
    assert int(np.count_nonzero(result.mask)) == 1


def test_padding_expands_and_clips_only_at_image_bounds() -> None:
    result = build_mask(10, 10, [obj(0, (1, 1, 4, 4))],
                        kind="rect", pad_px=2)
    assert result.padded_boxes_xyxy_px == ((0, 0, 6, 6),)
    assert np.count_nonzero(result.mask) == 36
    assert result.mask[9, 9] == 0


def test_float_bbox_pixel_centers() -> None:
    result = build_mask(6, 6, [obj(0, (1.25, 0.75, 3.25, 2.75))], kind="rect")
    assert np.argwhere(result.mask > 0).tolist() == [[1, 1], [1, 2], [2, 1], [2, 2]]


def test_union_max_not_sum_with_overlapping_boxes() -> None:
    one = obj(0, (1, 1, 7, 7))
    two = obj(1, (4, 4, 9, 9))
    result = build_mask(12, 12, [one, two], kind="rect", strength=0.7)
    assert result.mask[5, 5] == np.float32(0.7)
    assert result.mask.max() == np.float32(0.7)
    a = build_mask(12, 12, [one], kind="soft_rect", feather_px=2).mask
    b = build_mask(12, 12, [two], kind="soft_rect", feather_px=2).mask
    both = build_mask(12, 12, [one, two], kind="soft_rect", feather_px=2).mask
    assert np.array_equal(both, np.maximum(a, b))


def test_determinism_does_not_depend_on_input_order() -> None:
    a = obj(0, (1, 1, 5, 6))
    b = obj(1, (6, 3, 10, 8))
    first = build_mask(12, 12, [a, b], pad_px=1, feather_px=3).mask
    second = build_mask(12, 12, [b, a], pad_px=1, feather_px=3).mask
    assert np.array_equal(first, second)


@pytest.mark.parametrize("kind,kwargs", [
    ("wrong", {}),
    ("rect", {"pad_px": -1}),
    ("rect", {"pad_px": 0.5}),
    ("rect", {"pad_px": True}),
    ("soft_rect", {"feather_px": -1}),
    ("soft_rect", {"feather_px": float("inf")}),
    ("soft_rect", {"feather_px": float("nan")}),
    ("rect", {"strength": 0}),
    ("rect", {"strength": 1.001}),
    ("rect", {"strength": float("nan")}),
    ("rect", {"strength": True}),
])
def test_invalid_parameters_fail_closed(kind, kwargs) -> None:
    with pytest.raises(MaskError):
        build_mask(12, 12, [obj(0)], kind=kind, **kwargs)


@pytest.mark.parametrize("image_width,image_height", [(0, 10), (1.1, 10), (10, False)])
def test_invalid_dimensions_fail(image_width, image_height) -> None:
    with pytest.raises(MaskError):
        build_mask(image_width, image_height, [obj(0)])


@pytest.mark.parametrize("box", [
    (-1, 1, 3, 3), (1, 1, 13, 3), (1, 1, 2, 13),
    (4, 4, 4, 8), (5, 5, 2, 9),
    (1, float("nan"), 4, 5), (1, float("inf"), 4, 5),
    (1, True, 4, 5),
])
def test_invalid_bbox_fails_without_clipping_source(box) -> None:
    with pytest.raises(MaskError):
        build_mask(12, 12, [obj(0, box)])


def test_empty_or_repeated_instances_fail() -> None:
    with pytest.raises(MaskError):
        build_mask(12, 12, [])
    with pytest.raises(MaskError):
        build_mask(12, 12, [obj(0), obj(0, (7, 7, 9, 9))])


def test_mask_not_segmentation_ground_truth() -> None:
    mask = build_mask(12, 12, [obj(0)]).mask
    assert mask.ndim == 2
    assert mask.dtype == np.float32
