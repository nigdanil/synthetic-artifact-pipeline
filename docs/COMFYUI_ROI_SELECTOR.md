# ComfyUI SAP ROI Selector — stage 6.3

The new classic ComfyUI node `SAPROISelector` connects to `SAPDatasetLoader`
and uses the existing **pure-Python** `sapipeline.selection.select_rois`.
No model, Torch upgrade, new dependencies or modification of the original
photographs is required.

## Inputs

- `annotations` (`SAP_ANNOTATIONS`): opaque document from `SAPDatasetLoader`.
- `image` (`IMAGE`): the matching source image from that same loader;
  must be a single RGB image with original width and height.
- `mode`: `random_n`, `random_one`, `by_index`, `by_class`.
- `seed`: non-negative random selection seed; `random_n` with the same
  seed and document selects the same unique instances.
- `count`: number of targets for `random_n`; ignored for other modes.
- `instance_index`: exact annotation index for `by_index`.
- `class_name`: original case-sensitive label, for `by_class` only. Source
  labels stay on the local machine; no class name is written into the workflow.
- `include_generic_product`: optionally include generic `product` class in
  random selection; service classes are **always excluded** from random modes.
- `allow_service_targets`: explicit opt-in for `by_index`/`by_class` only.

## Outputs

- `selection` (`SAP_ROI_SELECTION`): opaque `SelectionResult` containing the
  exact selected instances; reserved for the subsequent `SAP Mask Builder`.
- `roi_preview` (`IMAGE`): **visualization-only** copy of the original image
  with selected bboxes cyan and other bboxes red, at original resolution.
- `selection_summary` (`STRING`): selected instance indices/count, without
  vendor class names, source filenames or private paths.

## Install / update (Git Bash)

From your SAP project folder in Git Bash, copy the updated extension to ComfyUI
**after confirming the existing target folder and making a private backup outside custom_nodes**:

```bash
COMFY='/d/ComfyUI/ComfyUI-NSFW/ComfyUI'
test -f "$COMFY/custom_nodes/sap_dataset_loader/node.py" || exit 1
test ! -e workflows/existing/sap_dataset_loader_backup_6_2 || exit 1
cp -a "$COMFY/custom_nodes/sap_dataset_loader" \
  workflows/existing/sap_dataset_loader_backup_6_2
cp -f custom_nodes/sap_dataset_loader/*.py \
  "$COMFY/custom_nodes/sap_dataset_loader/"
```

Restart the ComfyUI *backend process*; frontend reload alone is insufficient.
Import `workflows/templates/sap_roi_selector_v1.json` as a new graph.
Set **only** `SAPDatasetLoader.project_root` to the absolute local SAP project
folder. Keep this absolute local path out of the public workflow template.
Press Run. Expect a photo in both the original and bbox-preview nodes.

**Do not commit:** real dataset ZIP, private indices/manifest, vendor class
mappings or exported images. A green preview is not annotation QA signoff.
