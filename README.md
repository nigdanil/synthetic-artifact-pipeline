# Synthetic Artifact Pipeline (SAP)

Public Python scaffold for **deterministic image augmentation with bbox-aware effect masks**. Planned integration with an existing **ComfyUI** graph; this repository does not contain a trained model, original photographs or a working effect pipeline yet.

**State (2026-10-13):** Tasks 1–5 implemented: ZIP indexer, SEG/YOLO adapters, verified ZIP loader, ROI selection, bbox-derived effect masks and deterministic local highlight effects. ComfyUI nodes and exporter remain **not implemented**.

## Quick start on Windows (Git Bash)

```bash
cd /d/my_work/synthetic-artifact-pipeline-public
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -e '.[dev,image]'
python -m pytest -q
python scripts/check_environment.py
```

Install Python 3.12 and Git first. ComfyUI stays in its existing installation and is
**not** modified by this setup.

## Task 1: local ZIP indexing (Git Bash)

Keep the original ZIP archives under `data/raw/` (Git-ignored), then run from the active `.venv`:

```bash
source .venv/Scripts/activate
python scripts/index_dataset.py
python -m pytest -q
```

Private index/report outputs are ignored by Git. See [dataset indexing instructions](docs/DATASET_INDEXING.md); never commit datasets, the index or the report.

## Task 2: validate SEG geometry and create private overlays (Git Bash)

```bash
source .venv/Scripts/activate
python scripts/validate_annotations.py
```

The parser uses **encoded image pixels without EXIF rotation**. YOLO TXT requires an
explicit class-ID mapping; no class IDs are guessed. Preview PNGs and report JSON
are written under Git-ignored `reports/`. See [annotation adapters](docs/ANNOTATIONS.md).

## Task 3: verified local image + ROI preview (Git Bash)

```bash
source .venv/Scripts/activate
python scripts/preview_roi.py --mode random_one --seed 42017
python scripts/preview_roi.py --mode random_n --count 3 --seed 42017
for i in 0 1 2 3 4; do python scripts/preview_roi.py --sample-index "$i" --mode random_n --count 3; done
```

Run Task 1 indexing first and keep source ZIP files under ignored `data/raw/`.
Selected bounding boxes are shown in **cyan**, other labels in red. Preview PNG
and JSON stay private under `reports/roi_previews/`. See [ROI selection](docs/ROI_SELECTION.md).
No mask or glow is produced in this step.

## Task 4: build private rectangular effect masks (Git Bash)

```bash
source .venv/Scripts/activate
python scripts/preview_mask.py --mode random_n --count 3 --seed 42017 \
  --mask-kind soft_rect --feather-px 16 --pad-px 0 --strength 1
python scripts/preview_mask.py --mode by_index --instance-index 0 --mask-kind rect
```

Masks are float32 in memory and saved as grayscale PNGs with a separate
**visual-only** cyan overlay and private JSON report under ignored
`reports/mask_previews/`. These are bbox *effect* masks, not segmentations
and not generated photographs. See [Mask Builder](docs/MASKS.md).


## Task 5: private local highlight preview (Git Bash)

```bash
source .venv/Scripts/activate
python scripts/preview_effect.py --mode random_n --count 3 --seed 42017 \
  --mask-kind soft_rect --effect-kind soft_glow --feather-px 16
python scripts/preview_effect.py --mode by_index --instance-index 0 \
  --mask-kind rect --effect-kind hotspot
```

This stage applies a deterministic **photometric** effect using the mask from Task 4
and writes local-only `*_effect.png`, `*_mask.png`, `*_overlay.png` and JSON report
files to ignored `reports/effect_previews/`. See [Effects](docs/EFFECTS.md).

## Source layout

- `src/sapipeline/` — pure Python library namespace for planned parsing, mask generation and compositing
- `custom_nodes/` — future ComfyUI integration, currently a placeholder
- `workflows/existing/` — **local-only** copy of a personal ComfyUI graph
- `data/raw/` — **local-only** input images and annotations; never commit
- `data/work/`, `data/pilot/`, `outputs/`, `reports/` — local-only derivatives
- `configs/class_mappings/` — local-only third-party class-to-ID mappings
- `scripts/index_dataset.py` — read-only local ZIP indexer (Task 1)
- `scripts/validate_annotations.py` — private SEG geometry report and 5 bbox previews (Task 2)
- `scripts/preview_roi.py` — private image + selected ROI preview (Task 3)
- `scripts/preview_mask.py` — private bbox effect mask and overlay preview (Task 4)
- `scripts/preview_effect.py` — private local highlight compositing preview (Task 5)
- `scripts/check_environment.py` — read-only diagnostics
- `scripts/check_public_repo.py` — audit staged/tracked content before every public push
- `docs/` — public setup and integration documentation

## Privacy and publication

**Do not commit images, annotations, class mappings from third parties, dataset reports, private workflow JSON, tokens or internal reference documentation.** They may contain proprietary data or personal computer paths. The `.gitignore` is a first line of defense, **not** an information security guarantee. Run `python scripts/check_public_repo.py` after `git add` and before `git commit` / `git push`.

Internal project specifications and any real datasets must be stored separately from the public Git history. If sensitive material was committed earlier, removing it in a newer commit is **not enough**: treat the history as exposed and rotate leaked secrets.

## Project direction

1. Build a read-only index of local JPEG / annotation pairs, detect duplicates and preserve split provenance.
2. Parse vendor-style bbox JSON and optional standard YOLO TXT into canonical coordinates.
3. Select target object ROIs and build soft masks.
4. Apply deterministic local highlights without changing source geometry.
5. Connect Python logic to the user's existing ComfyUI graph; export image, annotations, mask and manifest.
6. Run acceptance tests and controlled pilot batches before any YOLO benchmark.

See [Windows setup](docs/SETUP_WINDOWS.md), [ComfyUI integration](docs/COMFYUI_INTEGRATION.md), [security guidance](SECURITY.md) and [current status](docs/STATUS.md).

**Licensing note:** public GitHub visibility does not automatically grant a software license. No open-source license is included until the owner chooses one.
