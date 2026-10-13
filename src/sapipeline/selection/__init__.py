"""Deterministic selection of product bounding boxes by class, index or seed."""

from .selector import (
    GENERIC_CLASS,
    SERVICE_CLASSES,
    SelectionError,
    SelectionResult,
    select_rois,
)

__all__ = [
    "GENERIC_CLASS", "SERVICE_CLASSES", "SelectionError", "SelectionResult", "select_rois",
]
