"""Canonical bbox adapters; no implicit transforms or private class tables."""

from .parser import (
    AnnotationDocument,
    AnnotationError,
    ObjectAnnotation,
    load_class_mapping,
    parse_seg,
    parse_yolo,
    to_seg_json,
    to_yolo_txt,
    validate_class_mapping,
)

__all__ = [
    "AnnotationDocument",
    "AnnotationError",
    "ObjectAnnotation",
    "load_class_mapping",
    "parse_seg",
    "parse_yolo",
    "to_seg_json",
    "to_yolo_txt",
    "validate_class_mapping",
]
