# SAP Glass Reflection V1 — experimental Stage-8 realism improvement

This is a **separate**, deterministic photometric effect. It does **not** replace
`SAP Realistic Glare` and does not change JPEG/SEG source files, geometry,
classes, or object identity. It is a visual experiment to review during Stage 8,
not an accepted synthetic training set.

## What it does / does not do

- User provides a 4-point quadrilateral representing the **visible glass plane**:
  normalized `[x,y]` image coordinates of `TL, TR, BR, BL` corners.
- A planar homography maps output pixels to the local `[u,v]` glass plane.
- Seeded, soft elongated light bands cross multiple SKU boxes and the gaps
  between them. Bands are clipped to the plane, with edge feathering.
- Outputs modified `IMAGE`, **actual glass effect `MASK`**, and JSON effect summary.
- No object masks, optical surface detection, actual 3D ray tracing, glass frames,
  occluder detection or perspective estimation is performed. Setting the
  quadrilateral inaccurately can brighten non-glass objects (e.g. cabinet frame).
- Existing original bounding boxes remain the only source labels. **Do not add
  automatic synthetic `glint` labels**. Treat product detection and glint labels
  as separate annotation-policy questions.

## Local ComfyUI install (Windows Git Bash)

Run only after applying the stage patch and **stopping** ComfyUI:

```bash
cd /d/path/to/synthetic-artifact-pipeline-public
COMFY='/d/ComfyUI/ComfyUI-NSFW/ComfyUI'
DEST="$COMFY/custom_nodes/sap_dataset_loader"
BACKUP='workflows/existing/sap_dataset_loader_backup_before_glass_v1'
if [ ! -e "$BACKUP" ]; then cp -a "$DEST" "$BACKUP"; fi
cp -v custom_nodes/sap_dataset_loader/{__init__,glass,export}.py "$DEST/"
```

Restart ComfyUI. Open the **private** ready workflow from
`workflows/existing/sap_glass_reflection_v1_local_ready.json`; alternately open
public `workflows/templates/sap_glass_reflection_v1.json` and set both
`project_root` widgets manually. Only the private graph contains a local path.

## Verify one run

1. In `SAP Dataset Loader`, use a known sample (same as previous Stage-7 run).
2. Adjust `SAP Glass Reflection.quad_json` to exactly surround visible glass.
   Default for the freezer prototype:
   `[[0.09,0.21],[0.98,0.21],[0.99,0.75],[0.03,0.74]]`.
   Values are fractions of whole image; **do not use pixel coordinates**.
3. Start with two bands, `band_width=0.060`, `band_separation=0.23`,
   `intensity=0.65`, `angle_deg=-12`, and `seed=42017`.
4. Inspect **11** (effect preview), **16** (actual mask), **14** (QA status).
   The glass-plane effect mask must feed **both QA and Exporter**; do not feed
   the upstream bbox Mask Builder mask into either of those ports.
5. Export with unique `run_id=glass_reflection_0001` (if free). The exporter
   re-computes this effect from the source JPEG/quad/seed and rejects any
   differences in the IMAGE/MASK as well as duplicate outputs and QA FAIL.
6. Run the Stage-8 audit after a successful export (it remains `NOT_READY`
   until enough distinct source images and explicit manual reviews exist).

`QA PASS` validates technical geometry, clipping and changes outside the effect
mask; **it does not certify photorealism**. In addition, the QA validator
requires all selected SKU ROIs to be affected: if a configured light band misses
selected objects, move the band or select appropriate ROI; never relax QA limits.

The upstream `SAP Mask Builder` still provides its original `mask_summary` to
Exporter for ROI provenance, but its rectangular MASK is **not** applied to
Glass Reflection. Only the spatially projected glass effect MASK is exported.

## Safety and performance

- CPU/Numpy-only and no new models, custom dependencies or GPU setup.
- Source archives/index never written to. `outputs/sap_runs/` and
  `workflows/existing/` are Git-ignored. Do not commit real photos, SEG, review
  reports, or machine-specific workflow JSON.
- If `glare_naturalness` is doubtful, keep manual verdict `needs_review`;
  do not claim training suitability from a technical PASS alone.
- Fail-closed constraints: invalid/concave/self-crossing/out-of-image glass
  quad rejected; zero changes outside actual spatial mask; deterministic seed.
