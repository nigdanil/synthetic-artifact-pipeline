"""ComfyUI classic custom-node registration for SAP's verified pipeline adapters.

No ComfyUI, torch or SAP core imports at extension discovery time.
"""

from .highlight import SAPHighlightEffect
from .mask import SAPMaskBuilder
from .node import SAPDatasetLoader
from .roi import SAPROISelector

NODE_CLASS_MAPPINGS = {
    "SAPDatasetLoader": SAPDatasetLoader,
    "SAPROISelector": SAPROISelector,
    "SAPMaskBuilder": SAPMaskBuilder,
    "SAPHighlightEffect": SAPHighlightEffect,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SAPDatasetLoader": "SAP Dataset Loader (ZIP + SEG)",
    "SAPROISelector": "SAP ROI Selector (bbox selection)",
    "SAPMaskBuilder": "SAP Mask Builder (ROI to MASK)",
    "SAPHighlightEffect": "SAP Highlight Effect (IMAGE + MASK)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
