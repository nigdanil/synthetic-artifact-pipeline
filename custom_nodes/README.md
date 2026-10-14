# SAP custom nodes for ComfyUI

The `sap_dataset_loader/` extension now provides seven classic ComfyUI nodes:

- `SAPDatasetLoader` — loads indexed local ZIP+SEG pairs (read-only).
- `SAPROISelector` — selects annotated bounding-box instances.
- `SAPMaskBuilder` — converts selected ROI into a ComfyUI `MASK` and an overlay preview.
- `SAPHighlightEffect` — applies `soft_glow` or `hotspot` to `IMAGE` within `MASK`.
- `SAPRealisticGlare` — creates directional or elliptical photometric glare.
- `SAPQAValidator` — checks pixel, bbox and annotation invariants.
- `SAPArtifactExporter` — QA-gated, private atomic artifact bundle.

Setup and behavior: [Dataset Loader](../docs/COMFYUI_DATASET_LOADER.md),
[ROI Selector](../docs/COMFYUI_ROI_SELECTOR.md),
[Mask Builder](../docs/COMFYUI_MASK_BUILDER.md),
[Highlight Effect](../docs/COMFYUI_HIGHLIGHT_EFFECT.md),
[Realistic Glare](../docs/COMFYUI_REALISTIC_GLARE.md),
[QA Validator](../docs/COMFYUI_QA_VALIDATOR.md),
[Artifact Exporter](../docs/COMFYUI_ARTIFACT_EXPORTER.md).

This directory holds extension source only; installation into an existing ComfyUI
is manual. Do not publish private dataset archives, output images, class mappings,
or machine-specific workflows. See [roadmap](../docs/ROADMAP.md).
