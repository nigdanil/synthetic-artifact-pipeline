"""ComfyUI classic custom-node registration for SAP's verified pipeline adapters.

No ComfyUI, torch or SAP core imports at extension discovery time.
"""

from .export import SAPArtifactExporter
from .glare import SAPRealisticGlare
from .glass import SAPGlassReflection
from .highlight import SAPHighlightEffect
from .mask import SAPMaskBuilder
from .node import SAPDatasetLoader
from .qa import SAPQAValidator
from .roi import SAPROISelector

NODE_CLASS_MAPPINGS = {
    "SAPDatasetLoader": SAPDatasetLoader,
    "SAPROISelector": SAPROISelector,
    "SAPMaskBuilder": SAPMaskBuilder,
    "SAPHighlightEffect": SAPHighlightEffect,
    "SAPRealisticGlare": SAPRealisticGlare,
    "SAPGlassReflection": SAPGlassReflection,
    "SAPQAValidator": SAPQAValidator,
    "SAPArtifactExporter": SAPArtifactExporter,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SAPDatasetLoader": "SAP Dataset Loader (ZIP + SEG)",
    "SAPROISelector": "SAP ROI Selector (bbox selection)",
    "SAPMaskBuilder": "SAP Mask Builder (ROI to MASK)",
    "SAPHighlightEffect": "SAP Highlight Effect (IMAGE + MASK)",
    "SAPRealisticGlare": "SAP Realistic Glare (ROI-local directional/ellipse)",
    "SAPGlassReflection": "SAP Glass Reflection (projected glass plane)",
    "SAPQAValidator": "SAP QA Validator (technical PASS/FAIL)",
    "SAPArtifactExporter": "SAP Artifact Exporter (private QA-gated bundle)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
