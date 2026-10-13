"""Stage 6.6: technical QA gates on entirely synthetic RGB/bbox fixtures."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from sapipeline.annotations import parse_seg
from sapipeline.qa import validate_artifact
from sapipeline.selection import select_rois


@pytest.fixture
def case():
    width, height = 24, 16
    doc = parse_seg(
        json.dumps({
            "filename": "fake.jpg",
            "selectedList": [
                {"name": "fake_a", "rect": {"x": 3, "y": 4, "width": 6, "height": 7}},
                {"name": "fake_b", "rect": {"x": 14, "y": 3, "width": 7, "height": 7}},
            ],
        }), image_width=width, image_height=height,
    )
    selected = select_rois(doc, mode="random_n", count=2, seed=42017)
    original = np.full((height, width, 3), 0.25, dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.float32)
    mask[4:11, 3:9] = 1
    mask[3:10, 14:21] = 1
    modified = original.copy()
    modified[mask > 0] += 0.3
    return original, modified, mask, doc, selected


def test_realistic_pass_is_not_training_approval(case):
    before, after, mask, doc, selected = case
    result = validate_artifact(before, after, mask, doc, selected)
    assert result.passed
    assert all(result.checks.values())
    output = json.loads(result.to_json())
    assert output["status"] == "PASS"
    assert output["manual_review_required"] is True
    assert output["training_approved"] is False
    assert output["metrics"]["bbox_count"] == 2
    assert output["metrics"]["selected_indices"] == [0, 1]
    assert result.to_json() == validate_artifact(before, after, mask, doc, selected).to_json()


def test_inputs_are_never_mutated(case):
    before, after, mask, doc, selected = case
    snapshots = [array.copy() for array in (before, after, mask)]
    validate_artifact(before, after, mask, doc, selected)
    for array, snapshot in zip((before, after, mask), snapshots, strict=True):
        assert np.array_equal(array, snapshot)


def test_single_pixel_leak_is_detected(case):
    before, after, mask, doc, selected = case
    after[0, 0] += 1 / 255
    result = validate_artifact(before, after, mask, doc, selected)
    assert not result.passed
    assert result.failures == ("no_changes_outside_mask",)
    assert result.metrics["outside_mask_changed_pixels"] == 1


def test_inside_effect_but_one_selected_roi_unaffected_is_failure(case):
    before, after, mask, doc, selected = case
    after[3:10, 14:21] = before[3:10, 14:21]
    qa = validate_artifact(before, after, mask, doc, selected)
    assert "all_selected_rois_affected" in qa.failures


def test_no_effect_is_failure(case):
    before, _after, mask, doc, selected = case
    qa = validate_artifact(before, before.copy(), mask, doc, selected)
    assert "visible_effect" in qa.failures
    assert "all_selected_rois_affected" in qa.failures


def test_empty_mask_is_failure(case):
    before, after, mask, doc, selected = case
    mask[:] = 0
    after[:] = before
    qa = validate_artifact(before, after, mask, doc, selected)
    assert "mask_nonempty" in qa.failures


def test_wrong_shape_reports_fail_instead_of_crashing(case):
    before, after, mask, doc, selected = case
    qa = validate_artifact(before, after[:, :-1], mask, doc, selected)
    assert qa.failures == ("image_geometry",)
    qa2 = validate_artifact(before, after, mask[:, :-1], doc, selected)
    assert "mask_geometry" in qa2.failures


@pytest.mark.parametrize("invalid", [float("nan"), -0.1, 1.1])
def test_bad_mask_values_fail_closed(case, invalid):
    before, after, mask, doc, selected = case
    mask[0, 0] = invalid
    result = validate_artifact(before, after, mask, doc, selected)
    assert result.failures == ("mask_values",)
    assert "NaN" not in result.to_json()


def test_out_of_range_image_is_rejected(case):
    before, after, mask, doc, selected = case
    after[0, 0] = np.nan
    qa = validate_artifact(before, after, mask, doc, selected)
    assert qa.failures == ("image_values",)


def test_bbox_geometry_is_strict(case):
    before, after, mask, doc, selected = case
    bad = replace(doc.objects[0], bbox_xyxy_px=(0, 0, 99, 20))
    invalid = replace(doc, objects=(bad, doc.objects[1]))
    qa = validate_artifact(before, after, mask, invalid, selected)
    assert "annotation_bboxes" in qa.failures
    assert "selection_matches_annotations" in qa.failures
    invalid_dimensions = replace(doc, image_width=900)
    qa = validate_artifact(before, after, mask, invalid_dimensions, selected)
    assert "annotation_dimensions" in qa.failures


def test_changed_selection_metadata_fails_even_if_same_rect(case):
    before, after, mask, doc, selected = case
    mismatched = replace(selected.targets[0], class_name="altered_label")
    invalid_selection = replace(selected, targets=(mismatched, selected.targets[1]))
    qa = validate_artifact(before, after, mask, doc, invalid_selection)
    assert "selection_matches_annotations" in qa.failures
    duplicated = replace(selected, targets=(selected.targets[0], selected.targets[0]))
    qa = validate_artifact(before, after, mask, doc, duplicated)
    assert "selection_matches_annotations" in qa.failures


def test_new_clipping_detection_only_counts_new_saturation(case):
    before, after, mask, doc, selected = case
    # Previously clipped pixels should not count as newly clipped.
    before[4, 4, 0] = 1.0
    after[4, 4, 0] = 1.0
    after[mask > 0] = 1.0
    qa = validate_artifact(before, after, mask, doc, selected, max_new_clipping_fraction=0.2)
    assert "new_clipping_below_limit" in qa.failures
    assert qa.metrics["newly_clipped_pixels"] > 0
    assert qa.metrics["new_clipping_fraction"] > 0.2


def test_coverage_threshold(case):
    before, after, mask, doc, selected = case
    qa = validate_artifact(before, after, mask, doc, selected, max_mask_coverage=0.05)
    assert "mask_coverage" in qa.failures


@pytest.mark.parametrize("options", [
    {"max_mask_coverage": -0.1},
    {"max_new_clipping_fraction": float("nan")},
    {"outside_tolerance": 1.1},
])
def test_invalid_threshold_rejected(case, options):
    with pytest.raises(ValueError, match="finite number"):
        validate_artifact(*case, **options)


def test_wrong_annotation_selection_types_rejected(case):
    before, after, mask, doc, selected = case
    with pytest.raises(TypeError, match="AnnotationDocument"):
        validate_artifact(before, after, mask, None, selected)
    with pytest.raises(TypeError, match="SelectionResult"):
        validate_artifact(before, after, mask, doc, None)
