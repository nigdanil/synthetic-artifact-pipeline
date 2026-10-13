# Synthetic Artifact Pipeline (SAP)

Public Python scaffold for **deterministic image augmentation with bbox-aware effect masks**. Planned integration with an existing **ComfyUI** graph; this repository does not contain a trained model, original photographs or a working effect pipeline yet.

**State (2026-10-13):** project skeleton only. Dataset indexer, SEG/YOLO parsers, ROI selection, masks, effects, custom nodes and exporter are planned, **not implemented**.

## Quick start on Windows (PowerShell)

```powershell
# From the repository root
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,image]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\check_environment.py
```

Install Python 3.12 and Git first. ComfyUI stays in its existing installation and is **not** modified by this setup.

## Source layout

- `src/sapipeline/` — pure Python library namespace for planned parsing, mask generation and compositing
- `custom_nodes/` — future ComfyUI integration, currently a placeholder
- `workflows/existing/` — **local-only** copy of a personal ComfyUI graph
- `data/raw/` — **local-only** input images and annotations; never commit
- `data/work/`, `data/pilot/`, `outputs/`, `reports/` — local-only derivatives
- `configs/class_mappings/` — local-only third-party class-to-ID mappings
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
