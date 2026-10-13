"""BBox-derived effect mask creation (not pixel-accurate object segmentation)."""

from .builder import MaskBuildResult, MaskError, MaskKind, build_mask

__all__ = ["MaskBuildResult", "MaskError", "MaskKind", "build_mask"]
