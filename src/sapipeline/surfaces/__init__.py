"""Manually verified physical surface annotations."""

from .schema import (
    MATERIALS,
    SCHEMA_VERSION,
    SurfaceError,
    dumps_document,
    loads_document,
    new_document,
    validate_document,
)

__all__ = [
    "MATERIALS",
    "SCHEMA_VERSION",
    "SurfaceError",
    "dumps_document",
    "loads_document",
    "new_document",
    "validate_document",
]
