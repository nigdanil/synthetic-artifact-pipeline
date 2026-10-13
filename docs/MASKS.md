# Task 4 — Local bbox effect masks (Git Bash)

A **Mask Builder**, not ground-truth segmentation. Reads the private Task-1 index,
loads an original JPEG+SEG pair from the ZIP (Task 3), chooses ROI targets (Task 3)
and creates a single `[height,width]` `float32` effect mask in `[0,1]`.
No change to source JPEGs, annotations, class mappings, data splits or ComfyUI.

## Setup

```bash
cd /d/my_work/synthetic-artifact-pipeline-public
source .venv/Scripts/activate
# Requires data/raw/for_learning.zip, data/raw/for_verification.zip and Task-1 index
python -m pytest -q
python scripts/preview_mask.py --mode random_n --count 3 --seed 42017 \
  --mask-kind soft_rect --feather-px 16 --pad-px 0 --strength 1.0
```

An exact instance can be chosen, without any class mapping:

```bash
python scripts/preview_mask.py --mode by_index --instance-index 0 --mask-kind rect
```

Sample five distinct normal-EXIF, globally unique frames. `--force` overwrites
local artifacts and should only be used intentionally:

```bash
for i in 0 1 2 3 4; do
  python scripts/preview_mask.py --sample-index "$i" --mode random_n \
    --count 3 --seed 42017 --mask-kind soft_rect --force
done
explorer.exe "$(cygpath -w "$PWD/reports/mask_previews")"
```

## Geometry and parameter contract

| Option | Behavior |
|---|---|
| `--mask-kind rect` | Flat value `strength` inside padded bbox; exactly zero outside |
| `--mask-kind soft_rect` | Smoothstep feather **inside** the padded bbox; no spill outside |
| `--pad-px N` | Nonnegative whole-pixel expansion, clipped at image boundary |
| `--feather-px F` | Nonnegative feather in pixels; internally capped to half shorter bbox side |
| `--strength F` | Maximum opacity, strictly greater than zero and up to one |
| selection flags | Same `by_index`, `by_class`, `random_one`, `random_n`, `seed` as Task 3 |

The mask samples each pixel **at its center**, `(x+0.5,y+0.5)`.
Boxes use half-open XYXY pixel coordinates. The selected boxes are fully validated
before padding; malformed, empty and duplicated targets fail closed. Multiple
masks are united with a **per-pixel maximum**, not a sum. Combining boxes can
include other products, as the mask is rectangular. EXIF orientation other than
1 is rejected by the loader. No resize or EXIF transpose occurs.

`feather_px=0` gives the same mask as `rect`; a thin bbox automatically uses a
smaller feather so its center is not erased.

## Outputs (all private, ignored by Git)

Files are saved only in a subdirectory under `reports/`:

- `*_mask.png`: lossless, grayscale 8-bit effect mask. `0` means no effect and
  `255` means full strength. The float32 source is computed in memory; 8-bit
  storage introduces at most 0.5/255 quantization error.
- `*_overlay.png`: original RGB image with a cyan mask overlay and selected bbox
  outlines for manual review. **Not a generated training image.**
- `*.json`: manifest with source archive/member hashes, original dimensions,
  selected instance indices and classes, bbox/padded bbox geometry, exact
  numerical parameters, filenames and `qa.status=pending_manual_review`.

A stable parameter digest is included in the filename so different settings
produce distinct outputs. Existing files are not overwritten unless `--force`
is supplied. Neither raw images, mask PNGs, the index nor reports should ever
be staged or pushed to GitHub. `python scripts/check_public_repo.py` remains
required before any public push.

**Acceptance:** check at least five overlays: position, feather, excess padding,
no leakage outside the mask, no unintended neighbors. This completes Mask
Builder only; the actual photometric glare effect belongs to the next task.
