# Task 6.4 — ComfyUI SAP Mask Builder

`SAP Mask Builder` converts the **opaque** `SAP_ROI_SELECTION` produced by `SAP ROI Selector` into a native ComfyUI `MASK` (`float32 [1,H,W]`, 0–1) without modifying the original image or the `.seg` annotations. The node uses the same CPU-only SAP `build_mask` implementation as the CLI preview in Task 4.

## Inputs and outputs

- Inputs: `selection` (`SAP_ROI_SELECTION`), `image` (`IMAGE`), `mask_kind` (`rect` or `soft_rect`), `pad_px` (nonnegative), `feather_px` (nonnegative), `strength` (0.01–1.0).
- `mask`: regular ComfyUI `MASK`; connect to native `MaskToImage` or later to SAP Highlight.
- `mask_overlay`: RGB `IMAGE`, cyan tint scaled by actual mask weight. **Preview only**, not an augmented image.
- `mask_summary`: short `STRING` of technical dimensions, parameters and selected numeric indices. No private file paths or vendor class names.

The mask is a **bounding-box effect mask**, not pixel-accurate product segmentation. `pad_px` may affect neighboring pixels when nonzero. Intersections are merged by taking the maximum, not adding intensities. No diffusion models or new dependencies are needed.

## Git Bash / ComfyUI smoke test

1. After applying the patch and successful tests, **close ComfyUI completely**.
2. From the SAP repository, copy **only the updated source files** to your installed ComfyUI custom-node directory:

```bash
COMFY='/d/ComfyUI/ComfyUI-NSFW/ComfyUI'
DEST="$COMFY/custom_nodes/sap_dataset_loader"
test -f "$DEST/node.py" || { echo 'Install Task 6.2 first'; exit 1; }
mkdir -p workflows/existing/sap_dataset_loader_backup_6_3
cp -a "$DEST/." workflows/existing/sap_dataset_loader_backup_6_3/
cp -v custom_nodes/sap_dataset_loader/{__init__,node,roi,mask}.py "$DEST/"
```

3. Restart ComfyUI and open the **local, private** `sap_mask_builder_v1_local_ready.json` provided alongside the patch. The public template `workflows/templates/sap_mask_builder_v1.json` keeps `project_root` blank and must be configured in the UI.
4. Check the loader `project_root`, `split=learning`, `sample_index=0`; selector `random_n`, `seed=42017`, `count=3`; mask node `soft_rect`, `pad_px=0`, `feather_px=16.0`, `strength=1.0`. Press **Run**.
5. Expected previews: original photo; cyan/red bbox ROI preview; grayscale true MASK; cyan mask overlay on original RGB.

Source images and `.seg` stay unchanged. The workflow and generated preview data are not dataset export artifacts. Do not publish local ready workflow, photos, `.seg`, reports or backup copies.
