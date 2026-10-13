"""Local dataset indexing; annotation parsing is a later pipeline stage."""

from .indexer import DatasetIndexError, build_index

__all__ = ["DatasetIndexError", "build_index"]
