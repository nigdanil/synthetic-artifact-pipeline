"""Deterministic local photometric effects."""

from .glare import GlareResult, apply_realistic_glare, build_glare_mask
from .highlight import EffectError, HighlightResult, apply_highlight

__all__ = [
    "EffectError",
    "GlareResult",
    "HighlightResult",
    "apply_highlight",
    "apply_realistic_glare",
    "build_glare_mask",
]
