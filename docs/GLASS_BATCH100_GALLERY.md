# Glass BBox 100: Offline visual QA gallery

## Goal

Review **50 source scenes / 100 generated glare candidates** as side-by-side
`Original | Diffuse | Fluorescent` images. The gallery does **not** regenerate,
edit, approve for training, or export any source/SEG/preview image.

The generator independently revalidates all 100 existing `_v2` previews using
the same strict checks as the resumable batch runner. It refuses to publish an
incomplete, damaged or mismatched gallery. Source ZIP, SEG, plan and first pilot
are untouched.

## Use (Windows Git Bash, from repository root)

```bash
python -m ruff check src/sapipeline/ scripts/ tests/
python -m pytest -q tests/test_glass_gallery.py
python scripts/glass_gallery.py build
```

Then open the **local** file:

`reports/glass_pilot100/gallery/index.html`

In Git Bash, `explorer.exe "$(cygpath -w "$PWD/reports/glass_pilot100/gallery")"`
opens the folder. The HTML references the original PNGs relatively in
`../auto_previews/`; keep the directory structure intact. No network or
local web server is required. All files remain ignored by Git.

## Review workflow

- Each card shows one source and its two synthetics side by side. Click images
  to enlarge.
- Independently mark **KEEP**, **REJECT**, or **NEEDS_REVIEW** for each variant.
  Clicking an active choice again restores **UNREVIEWED**. Add optional notes.
- Use the status filter to revisit undecided/problematic scenes.
- **Save/export** `glass_bbox100_visual_review.json` to keep manual decisions;
  **Import** this JSON later to continue review. Import checks exact run IDs,
  source SHA-256, effect names and review schema. Browser changes are not
  automatically written into files.
- Approval in the gallery means *visual quality only*; `training_approved=false`
  in every exported review JSON. This stage does not add data to YOLO training.

## Safety

The gallery is created exactly once at
`reports/glass_pilot100/gallery/index.html`. Existing gallery files are not
replaced. `pack` is **not** a prerequisite: gallery reads independently
verified private preview folders. Source scene content must never be committed
or uploaded publicly.
