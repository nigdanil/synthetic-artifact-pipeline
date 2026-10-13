# Stage 6.7 — SAP Realistic Glare (local bbox effect)

**Status:** implemented and automatically tested; **ComfyUI visual acceptance pending**. This does **not** simulate physically correct specular reflection from the freezer glass plane. This node generates smooth **ROI-local** light variation for testing; no autonomous new `glint` bounding boxes are created.

## Contract and topology

`SAP Dataset Loader` → `SAP ROI Selector` → `SAP Mask Builder` → **`SAP Realistic Glare`** → `SAP QA Validator`.

Node inputs: original `IMAGE [1,H,W,3]`, native SAP `MASK [1,H,W]`, and the **same** `SAP_ROI_SELECTION` used by Mask Builder. The output is (1) `effect_image`, (2) **actual** `effect_mask`, (3) `effect_summary` JSON. `effect_mask` is smooth, nonrectangular, and guaranteed zero wherever the **input** MASK is zero. A private preview shows its exact footprint. Source image, bbox selection, and annotations are never modified.

Connect `effect_image` to QA `modified`, **`effect_mask`** to QA `mask`, the original IMAGE to QA `original`, and the source annotation/selection to the existing QA ports. QA compares the actual permitted footprint. An existing `SAP Highlight Effect` remains available separately.

## Parameters (defaults are experimental)

| Widget | Default | Behavior |
|---|---:|---|
| `kind` | `directional` | `directional` = narrow oriented light band; `elliptical` = rotated oval |
| `angle_deg` | -25 | Orientation within original image pixel coordinates, -180…180 degrees |
| `radius_scale` | 0.85 | Extent in each ROI; 0.20…1.50 |
| `aspect_ratio` | 0.35 | Minor-axis / ROI height scaling, 0.10…1.00 |
| `falloff` | 1.25 | Softness of Gaussian envelope, 0.50…4.00 |
| `intensity` | 0.55 | Photometric light strength, 0…1 |
| `seed` | 42017 | Deterministic seed per selected ROI; choose **fixed** for repeatability |
| `center_jitter` | 0.10 | Per-ROI center variation (0…0.30 of bbox size); set 0 for exact center |

The model multiplies a rotated elliptical/strip envelope by a **smooth bbox-edge taper** and the existing SAP MASK. This minimizes hard rectangular borders, but it can **still** illuminate neighboring products *inside* a bbox. It does not understand depth, glass perspective, light sources, or object segmentation.

## Run in the existing ComfyUI, Git Bash

1. From the repository root, apply the stage-6.7 patch with `git apply --check` then `git apply`, and run the Python tests plus Ruff. Do not publish anything yet.
2. Fully stop the ComfyUI backend. Back up the previous installed `custom_nodes/sap_dataset_loader/` extension under ignored `workflows/existing/`.
3. Copy the updated six base files and new `glare.py` into the already-installed extension, then restart the backend. No model downloads or Python package upgrades are needed.
4. Open the **local** `sap_realistic_glare_v1_local_ready.json` from `workflows/existing/` (never commit this). Check that the Dataset Loader contains your own absolute `project_root`, `seed=42017` and seed behavior `fixed`.
5. Press **Run**. Inspect the original preview, new `Realistic glare IMAGE preview`, **`Actual nonrectangular glare mask preview`**, and green/red `QA STATUS`. Compare `directional` and `elliptical` on the same photo using `fixed` seed.
6. Confirm that the effect is visible but does not create square/rectangular borders, product labels are legible and the separate actual mask is nonrectangular. Green technical `PASS` is necessary, not sufficient. Keep outputs local.

Expected automated-test baseline from the preceding stage: **263 passed**; stage 6.7 adds more checks. Ruff and package health **must be verified on Windows** before commit. All public templates leave `project_root` empty. Local-ready templates containing Windows paths belong exclusively in `workflows/existing/`.

## QA and annotation policy

- A change to any pixel **outside the actual effect MASK** is a technical defect; QA should return FAIL.
- No source bbox coordinates/classes, image size, or source JPEG bytes may change.
- The output `effect_mask` is an augmentation footprint, **not** ground-truth instance segmentation.
- The photo and raw labels are not suitable for automatically annotating any new `glint` object. Glass-plane reflections require a separate future policy and perspective model.
- `QA PASS` never implies `training_approved`; until manual visual review, do not add generated results to a YOLO training corpus.

**Next in roadmap:** Stage 7 Artifact Exporter (after visual acceptance and Git commit), Stage 8 QA Acceptance, Stage 9 pilot of 100 samples.
