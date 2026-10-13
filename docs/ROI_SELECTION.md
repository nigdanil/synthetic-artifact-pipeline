# Task 3 — Verified local photo loader and deterministic ROI selection

This step implements **bbox target selection only**, not pixel-level segmentation,
mask generation, glow, image augmentation or ComfyUI nodes. Sources are read from
private ZIP archives without unpacking or rewriting them.

## Git Bash quickstart

From the repository root, activate the project venv, then ensure Task 1 has built a
private index under `data/work/index/dataset_index.json`:

```bash
source .venv/Scripts/activate
python scripts/index_dataset.py  # First time only; if index exists, use the existing one
python scripts/preview_roi.py --mode random_one --seed 42017
python scripts/preview_roi.py --mode random_n --count 3 --seed 42017
# Five different original frames for visual Gate G1:
for i in 0 1 2 3 4; do
  python scripts/preview_roi.py --sample-index "$i" --mode random_n --count 3 --seed 42017
done
```

The script automatically picks the first normally oriented JPEG whose SHA-256
occurs **exactly once across all indexed splits**. `--sample-index N` chooses
the Nth eligible example in a stable sorted order, starting at 0. It never silently picks an
image with train/verification leakage or uncertain EXIF orientation.

Choose a specific original image (copy the exact `image_member` from the **private**
`dataset_index.json`):

```bash
python scripts/preview_roi.py --split learning --image 'img/YOUR_FILE.jpg' \
  --mode by_index --instance-index 0
python scripts/preview_roi.py --split learning --image 'img/YOUR_FILE.jpg' \
  --mode by_class --class-name 'YOUR_CLASS'
```

The example filenames/classes above are **placeholders**, not data from the
private dataset. `by_class` selects *all* instances with that exact name,
including repeated SKU labels; `by_index` selects exactly one instance.

**Selection policy:** random modes exclude `glint`, `obstacle_object`,
`other_shelf`, `other_lary` and also the generic `product` by default. To include
`product` in a random sample, pass `--include-generic-product`. A service
class can be targeted by `by_index` or `by_class` only with the explicit
`--allow-service-targets` flag. **Random selection never includes service classes.**
An empty selection, absent class or insufficient number of candidates is an
error, not a fallback to full-frame effects.

## Output and controls

Full-resolution PNG and a private JSON selection record are written under
`reports/roi_previews/`. All source bboxes are red; selected bboxes are thicker
cyan rectangles with their **instance_index**. The PNG is a *review artifact*,
not an augmented image. JSON includes original image/annotation SHA-256s, class
strings, exact pixel XYXY and selected indices. Treat the report as private.

By default, existing previews are not overwritten; pass `--force` only when
replacement is intentional. You may provide `--out-dir` only inside `reports/`.
Do not add outputs to Git; these files are already Git-ignored.

```bash
python -m pytest -q
python scripts/check_public_repo.py
# On Windows, open previews from Git Bash:
explorer.exe "$(cygpath -w "$PWD/reports/roi_previews")"
```

## Coordinate and safety contract

- Pixel coordinates refer to the **encoded original** width and height. No
  EXIF transpose, resize, crop or annotation transform is performed.
- The loader checks the original ZIP image and SEG SHA-256 against the local
  index, source dimensions, matching file stems, and full annotation validity.
- EXIF orientation other than 1 fails closed until reviewed. Do not silently
  rotate a photo and leave bboxes in a different coordinate system.
- The selected ROI is an existing **bbox**, not a segmentation mask; nearby
  products or boxes may overlap. Manual visual QA is required before effects.
- SHA-256 detects exact duplicates only. Do not interpret these two ZIP splits
  as a clean independent benchmark without the separate data-policy review.

**Gate G1:** review at least five different frames and confirm the cyan boxes
and selected indices match the intended merchandise before proceeding to masks.
