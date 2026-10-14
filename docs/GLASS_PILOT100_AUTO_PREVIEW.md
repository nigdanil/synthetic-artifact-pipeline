# GLASS_PILOT100_AUTO_PREVIEW

## Purpose

Generate **private preview images only** for the second 100-image glass-reflection pilot
without any manual `glass_quad_norm` markup.

The preview driver reuses the existing private plan at
`reports/glass_pilot100/plan.json`, but it **does not** require per-scene glass
geometry. Instead, it derives a broad reflection region from several nearby
product ROI found in the existing SEG/YOLO annotations.

## Key behavior

- does **not** edit the source ZIP archives;
- does **not** rewrite the plan;
- does **not** export Stage-7 artifacts;
- does **not** approve anything for training;
- generates preview folders only under `reports/glass_pilot100/auto_previews/`.

## Effect strategy

For each scene:

1. load product bbox from the indexed learning sample;
2. ignore service/non-product classes;
3. choose a deterministic upper/middle horizontal cluster of up to 6 nearby products;
4. derive one broadened trapezoid-like reflection region from the union of those boxes;
5. render two variants:
   - `glass_diffuse` — broad soft glare;
   - `glass_fluorescent` — multiple narrow lamp-like reflections.

This is intentionally **bbox-driven**, not glass-plane-accurate geometry.
The goal is to create realistic *glass-like* glare above groups of products,
not to solve full scene geometry.
The cluster selector prefers upper/middle product rows and reduces the chance of
selecting products behind a bottom GPS map/footer overlay. Selection remains
heuristic and requires human visual review.

The second-generation auto-preview uses a separate `_v2` folder suffix to
preserve existing `scene_XX_...` preview folders. Do **not** remove old previews.

## Usage

```bash
python scripts/glass_auto_preview.py preview --start 0 --count 5
```

Each output folder contains:

- `original.png`
- `glass_region.png`
- `roi_mask_visual.png`
- `glass_diffuse.png`
- `glass_fluorescent.png`
- `glass_diffuse_mask_visual.png`
- `glass_fluorescent_mask_visual.png`
- `preview_report.json`

## Acceptance for this step

- works with the current `glass_pilot100` plan even when all `glass_quad_norm` are `null`;
- generates previews for 3–5 scenes;
- preserves source ZIP, SEG and existing plan bytes unchanged;
- fails closed on out-of-range scene requests;
- refuses to overwrite an existing preview folder.
