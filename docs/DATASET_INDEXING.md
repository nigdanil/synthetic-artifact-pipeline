# Task 1: Local dataset inventory (Git Bash, Windows)

This is **read-only input indexing**, not the annotation parser, ComfyUI integration or image augmentation. Keep all original images and `.seg` files private. The script reads ZIPs directly and **does not extract or modify them**.

## Setup (Git Bash only)

From the public repository root, with the Python `.venv` already installed:

```bash
# Open Git Bash in the project directory first.
source .venv/Scripts/activate

# Copy your authorized archives locally into data/raw/ (do not commit them):
# data/raw/for_learning.zip
# data/raw/for_verification.zip

python scripts/index_dataset.py
```

You may instead specify external, local ZIP paths:

```bash
python scripts/index_dataset.py \
  --learning '/d/datasets/for_learning.zip' \
  --verification '/d/datasets/for_verification.zip'
```

Replace example paths with actual local paths. ZIP archives must contain `img/<same-stem>.jpg` (or JPEG/PNG) and `seg/<same-stem>.seg` for each pair. Nested relative subdirectories within `img/` and `seg/` are supported when they match.

## Private outputs

The command writes the following **Git-ignored** JSON files only after validation completes:

- `data/work/index/dataset_index.json`: source ZIP hash, JPEG/SEG hashes, canonical SHA-256 image ID, member paths, dimensions and declared annotation-list count per photo.
- `reports/dataset_index_summary.json`: counts by split, exact duplicate groups, cross-split groups and annotation disagreements requiring manual review.

Both files remain **local only** and must never be added to the public repository. The script blocks output locations outside the private `data/work/` and `reports/` directories. It will not overwrite existing index/report files by default; for an intentional rebuild use:

```bash
python scripts/index_dataset.py --force
```

## Checks

```bash
python -m pytest -q
python -m pip check
python scripts/check_public_repo.py
git status --short
```

Compare the reported image/annotation counts and duplicate groups against your **private** dataset audit before continuing. Do not publish raw audit counts or dataset report output. Output SHA-256 is based on raw image/annotation bytes, with no EXIF rotation or coordinate transformations.

**Safety:** A same-hash photograph can have different annotation lists. Such cases are flagged, not automatically resolved. Cross-split duplicate families cannot be used for an independent YOLO test. Exact SHA-256 does **not** catch visually similar shots; perceptual grouping and final train/validation/test policy need subsequent review. `bbox_count_declared` is only `len(selectedList)`: geometry validation and YOLO TXT conversion belong to Task 2.

Do not commit local reports, ZIPs, masks, class mappings or full local workflow JSON. Use Git Bash commands exclusively. If a source ZIP is missing, damaged, ambiguous or has orphan annotations, the command exits with error rather than publishing a success index.
