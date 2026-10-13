# Task 5 — deterministic local highlight effect

This stage applies a **local photometric effect** to the original RGB image using
a prebuilt ROI mask from Task 4. It does **not** alter bounding boxes, create
segmentations, or use diffusion models.

## Effect kinds

- `soft_glow` — smooth lightening with controlled whitening and gamma lift.
- `hotspot` — stronger central highlight suitable for obvious local overexposure.

## Key rules

- Input mask is a float32 array in `[0,1]` on the **encoded image grid**.
- Pixels where the mask is exactly `0` are preserved bit-for-bit.
- Output remains RGB uint8 and the same size as the source image.
- All outputs are local-only under Git-ignored `reports/effect_previews/`.

## Git Bash examples

```bash
source .venv/Scripts/activate
python scripts/preview_effect.py --mode random_n --count 3 --seed 42017 \
  --mask-kind soft_rect --effect-kind soft_glow --feather-px 16
python scripts/preview_effect.py --mode by_index --instance-index 0 \
  --mask-kind rect --effect-kind hotspot
```

This writes:

- `*_effect.png` — the augmented preview image
- `*_mask.png` — grayscale mask used for compositing
- `*_overlay.png` — original image with cyan overlay and ROI outlines
- `*.json` — private run manifest

The outputs are for internal review only and must not be committed.
