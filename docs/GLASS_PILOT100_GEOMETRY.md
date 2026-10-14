# Glass Pilot 100 — Stage 2: calibrate glass polygons and preview 3–5 scenes

This is a **private visual experiment** after `glass_pilot100.py plan`, not
batch generation, Stage-7 artifact export or acceptance for YOLO training.
The older 100-image `pilot100` and its immutable runs are left untouched.

## 1. Generate five offline geometry guides (Windows Git Bash)

From the repository root with `.venv-1` active:

```bash
python scripts/glass_preview.py guide --start 0 --count 5
```

Five private directories are created under
`reports/glass_pilot100/geometry_guides/scene_XX_<source-sha-prefix>/`.
Each contains `source.png` and `select_corners.html`. Open each HTML file in
your browser (double-click or use `explorer.exe` with its Windows path).
The HTML works entirely offline: no server, network, dependencies, or upload.

On each image, click **TL → TR → BR → BL**, around the *actual visible glass
plane*, excluding metal frames, handles, walls and foreground objects.
The overlay and normalized JSON appear below the image. Click **Copy JSON**
(or manually select the text). Example only (not a universal default):

```json
[[0.12, 0.18], [0.88, 0.18], [0.88, 0.84], [0.12, 0.84]]
```

One convex quadrilateral models **one planar glass surface**, not multiple
independently angled lids. If a scene has separate glass panes or substantial
occlusions, do **not** force one polygon across them; mark it as unsuitable
for V1 or defer it for a multi-plane model. The image coordinates match the
encoded, unrotated JPEG pixel grid used by SEG.

## 2. Save and validate each scene's manually selected glass polygon

Paste the copied JSON into Git Bash, replacing the coordinates below:

```bash
python scripts/glass_preview.py set-quad --scene 0 \
  --quad '[[0.12,0.18],[0.88,0.18],[0.88,0.84],[0.12,0.84]]'
python scripts/glass_pilot100.py check
```

Repeat for scenes 1, 2, 3, 4, **with their own image-specific coordinates**.
Only the `glass_quad_norm` field of that particular scene is changed in the
private `reports/glass_pilot100/plan.json`. The tool validates the entire plan,
creates a SHA-256-named backup in `reports/glass_pilot100/plan_backups/`,
and atomically replaces only the private plan. It does not touch source JPEG,
SEG, index, other pilots, or already exported Stage-7 runs.

To revise a saved polygon after inspection, use `--replace` with new JSON.
The old plan is backed up again. Invalid, crossed, degenerate or out-of-image
polygons are rejected; passing that check **does not** establish correct
physical glass geometry.

## 3. Render both effects on the first five scenes

```bash
python scripts/glass_preview.py preview --start 0 --count 5
```

**All five scene polygons must be set** before this command starts; a missing
polygon stops the command without creating preview images. For one configured
scene, use `preview --start 0 --count 1` instead.

Each scene produces a separate folder under
`reports/glass_pilot100/effect_previews/scene_XX_<hash>_<quad-hash>/`:

- `original.png` — unmodified reference copy for private comparison.
- `glass_plane.png` — cyan glass outline and orange corner markers.
- `glass_diffuse.png` — broad, diffuse light band.
- `glass_fluorescent.png` — three narrower light bands.
- `glass_*_mask_visual.png` — grayscale **8-bit visualizations only**, not
  the 16-bit artifact masks exported in Stage 7.
- `preview_report.json` — source hash, selected polygon, effect parameters,
  technical QA results and explicit `training_approved=false`.

Compare with the supplied chest-freezer glare reference: shape, location,
perspective, band softness, overexposed pixels, logo legibility, shelf/frame
spill and glare across multiple bounding boxes. **QA PASS alone is not
photorealism approval**. A technical QA failure remains visible in the private
preview report; it never counts toward the 100-image target.

A preview directory is never overwritten. Rerunning with identical geometry
fails explicitly; changing the polygon yields a new geometry-hashed directory
and keeps the old preview for comparison. No edits to any six-file Stage-7
bundle, manual review record, existing pilot plan or generated package.

## Limits and handoff

This stage is manual calibration of 3–5 scenes plus visual **previews only**.
No automatic glass detection, glass refraction, camera/lighting model, multi-
plane handling or unattended 100-image export is introduced. The next stage
must explicitly review the resulting images and QA reports before adding a
resumable second batch exporter and private preview ZIP.

Never commit `data/raw/`, `reports/`, `outputs/`, private polygons, raw JPEG/SEG,
preview PNGs or local workflow files. The Python code, tests and this Markdown
are safe for the public repository after the usual `check_public_repo.py` scan.
