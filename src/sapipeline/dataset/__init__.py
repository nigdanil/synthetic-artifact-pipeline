"""Read-only local dataset indexing and per-pair source loading."""

from .indexer import DatasetIndexError, build_index
from .loader import LoadedPair, choose_index_entry, load_indexed_pair, read_private_index

__all__ = [
    "DatasetIndexError", "LoadedPair", "build_index", "choose_index_entry",
    "load_indexed_pair", "read_private_index",
]
