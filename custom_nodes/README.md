# SAP custom nodes for ComfyUI

The `sap_dataset_loader/` extension now provides four classic ComfyUI nodes:

- `SAPDatasetLoader` — loads indexed local ZIP+SEG pairs (read-only).
- `SAPROISelector` — selects annotated bounding-box instances.
- `SAPMaskBuilder` — converts selected ROI into a ComfyUI `MASK` and an overlay preview.
- `SAPHighlightEffect` — applies `soft_glow` or `hotspot` to `IMAGE` within `MASK`.

Setup and behavior: [Dataset Loader](../docs/COMFYUI_DATASET_LOADER.md),
[ROI Selector](../docs/COMFYUI_ROI_SELECTOR.md),
[Mask Builder](../docs/COMFYUI_MASK_BUILDER.md),
[Highlight Effect](../docs/COMFYUI_HIGHLIGHT_EFFECT.md).

This directory holds extension source only; installation into an existing ComfyUI
is manual. Do not publish private dataset archives, output images, class mappings,
or machine-specific workflows. See [roadmap](../docs/ROADMAP.md).
