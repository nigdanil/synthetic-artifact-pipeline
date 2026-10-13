"""Fail-closed, read-only technical QA for photometric bbox-mask augmentation.

Coordinates remain in the original JPEG pixel grid; these tests do not certify
photorealism, true object segmentation, or suitability for YOLO training.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

from sapipeline.annotations import AnnotationDocument
from sapipeline.selection import SelectionResult


@dataclass(frozen=True, slots=True)
class QAResult:
    passed: bool
    checks: dict[str, bool]
    metrics: dict[str, int | float | list[int]]
    failures: tuple[str, ...]

    def to_json(self) -> str:
        return json.dumps(
            {
                "schema_version": "sap-qa-v1",
                "status": "PASS" if self.passed else "FAIL",
                "checks": self.checks,
                "metrics": self.metrics,
                "failures": list(self.failures),
                "manual_review_required": True,
                "training_approved": False,
            },
            sort_keys=True,
            allow_nan=False,
        )


def validate_artifact(
    original: np.ndarray,
    modified: np.ndarray,
    mask: np.ndarray,
    annotations: AnnotationDocument,
    selection: SelectionResult,
    *,
    max_new_clipping_fraction: float = 0.25,
    max_mask_coverage: float = 0.60,
    outside_tolerance: float = 0.0,
) -> QAResult:
    """Check one [H,W,3] float image pair and a [H,W] weight mask.

    New clipping is defined as pixels that reach 1.0 in any channel only after
    modification, divided by changed pixels *inside the mask*. This avoids
    penalizing highlights already saturated in the original photograph.
    Exact outside-mask equality is the default. No inputs are changed.
    """
    for label, threshold in (
        ("max_new_clipping_fraction", max_new_clipping_fraction),
        ("max_mask_coverage", max_mask_coverage),
        ("outside_tolerance", outside_tolerance),
    ):
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise TypeError(f"{label} must be a finite number in [0,1]")
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError(f"{label} must be a finite number in [0,1]")
    if not isinstance(annotations, AnnotationDocument):
        raise TypeError("annotations must be an SAP AnnotationDocument")
    if not isinstance(selection, SelectionResult):
        raise TypeError("selection must be an SAP SelectionResult")

    checks: dict[str, bool] = {}
    metrics: dict[str, int | float | list[int]] = {}

    def check(code: str, ok: bool) -> None:
        checks[code] = bool(ok)

    shape_ok = (
        isinstance(original, np.ndarray)
        and isinstance(modified, np.ndarray)
        and original.ndim == modified.ndim == 3
        and original.shape == modified.shape
        and original.shape[-1] == 3
        and original.shape[0] > 0 and original.shape[1] > 0
    )
    check("image_geometry", shape_ok)
    if not shape_ok:
        return _finish(checks, metrics)

    height, width, _ = original.shape
    metrics["image_size_px"] = [width, height]
    metrics["bbox_count"] = len(annotations.objects)
    metrics["selected_count"] = len(selection.targets)
    check("image_values", _valid_unit_image(original) and _valid_unit_image(modified))
    check("mask_geometry", isinstance(mask, np.ndarray) and mask.shape == (height, width))
    if not checks["mask_geometry"]:
        return _finish(checks, metrics)
    check(
        "mask_values",
        mask.dtype.kind == "f" and bool(np.isfinite(mask).all())
        and bool(np.all((mask >= 0) & (mask <= 1))),
    )

    check("annotation_dimensions", (
        annotations.image_width == width and annotations.image_height == height
    ))
    indexed: dict[int, object] = {}
    boxes_valid = True
    for obj in annotations.objects:
        index = obj.instance_index
        if type(index) is not int or index < 0 or index in indexed:
            boxes_valid = False
        else:
            indexed[index] = obj
        try:
            x1, y1, x2, y2 = obj.bbox_xyxy_px
            valid = (len(obj.bbox_xyxy_px) == 4 and
                     all(type(v) in (int, float) and math.isfinite(v)
                         for v in (x1, y1, x2, y2)) and
                     0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height and
                     isinstance(obj.class_name, str) and bool(obj.class_name))
        except (ValueError, TypeError):
            valid = False
        boxes_valid &= valid
    check("annotation_bboxes", boxes_valid)

    selected_indices = [obj.instance_index for obj in selection.targets]
    selected_indices_valid = all(type(idx) is int and idx >= 0 for idx in selected_indices)
    check("selection_matches_annotations", bool(selection.targets) and
          selected_indices_valid and len(selected_indices) == len(set(selected_indices)) and
          all(indexed.get(obj.instance_index) == obj for obj in selection.targets))
    metrics["selected_indices"] = sorted(selected_indices) if selected_indices_valid else []

    if not checks["mask_values"] or not checks["image_values"]:
        return _finish(checks, metrics)

    active = mask > 0
    active_count = int(np.count_nonzero(active))
    coverage = active_count / (height * width)
    metrics["mask_active_pixels"] = active_count
    metrics["mask_coverage_fraction"] = round(coverage, 8)
    check("mask_nonempty", active_count > 0)
    check("mask_coverage", coverage <= max_mask_coverage)

    pixel_delta = np.max(np.abs(modified.astype(np.float64) - original), axis=2)
    changed = pixel_delta > 0
    leaked = (~active) & (pixel_delta > outside_tolerance)
    changed_inside = changed & active
    metrics["changed_pixels"] = int(np.count_nonzero(changed_inside))
    metrics["outside_mask_changed_pixels"] = int(np.count_nonzero(leaked))
    metrics["max_outside_mask_delta"] = (
        float(np.max(pixel_delta[~active])) if (~active).any() else 0.0
    )
    check("no_changes_outside_mask", not bool(np.any(leaked)))
    check("visible_effect", bool(np.any(changed_inside)))

    selected_affected = bool(selection.targets) and boxes_valid
    if selected_affected:
        for obj in selection.targets:
            x1, y1, x2, y2 = obj.bbox_xyxy_px
            # Pixel-grid box overlap; fractional boxes are handled conservatively.
            left, top = math.floor(x1), math.floor(y1)
            right, bottom = math.ceil(x2), math.ceil(y2)
            if not np.any(changed_inside[top:bottom, left:right]):
                selected_affected = False
                break
    check("all_selected_rois_affected", selected_affected)

    orig_saturated = np.any(original >= 1.0, axis=2)
    new_saturated = np.any(modified >= 1.0, axis=2)
    newly_clipped = changed_inside & new_saturated & ~orig_saturated
    clipping_count = int(np.count_nonzero(newly_clipped))
    fraction = clipping_count / max(1, int(np.count_nonzero(changed_inside)))
    metrics["newly_clipped_pixels"] = clipping_count
    metrics["new_clipping_fraction"] = round(fraction, 8)
    check("new_clipping_below_limit", fraction <= max_new_clipping_fraction)
    return _finish(checks, metrics)


def _valid_unit_image(array: np.ndarray) -> bool:
    return bool(array.dtype.kind == "f" and np.isfinite(array).all()
                and np.all((array >= 0) & (array <= 1)))


def _finish(checks: dict[str, bool], metrics: dict) -> QAResult:
    failures = tuple(code for code, passed in checks.items() if not passed)
    return QAResult(not failures, checks, metrics, failures)
