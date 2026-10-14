# Stage 8.2 — Local Batch Gallery (offline, read-only)

**Status:** implementation prepared; acceptance requires Windows testing and visual inspection.

Batch Gallery creates one private `index.html`, a `gallery_summary.json`, and lightweight JPEG
thumbnails under `reports/batch_gallery/`. The gallery loads existing artifacts in
`outputs/sap_runs/<run_id>/` and Stage 8.1 comparison sheets from
`reports/visual_review/<run_id>/`. Nothing is uploaded or served over a network.

## Usage (Windows Git Bash)

From the SAP project checkout, with `.venv` activated:

```bash
python scripts/render_review_sheet.py --all
python scripts/build_batch_gallery.py
explorer.exe "$(cygpath -w reports/batch_gallery/index.html)"
```

The first command is only necessary if new runs lack review sheets.
When generating a fresh gallery after adding runs or changing manual reviews:

```bash
python scripts/build_batch_gallery.py --overwrite
```

The first run refuses to overwrite an existing gallery without `--overwrite`.
The output is a read-only **snapshot**. Review statuses come from the current
`reports/qa_acceptance/reviews.json`, validated against the actual manifest SHA-256.
If reviews are changed later, run `--overwrite` to refresh the snapshot.

## Gallery functions

- Responsive Original / Modified image pairs, with links to full exported PNGs.
- One-click full-size `review_sheet.png` and `review_metrics.json`.
- Filters for run ID, effect type, QA status, manual-review verdict, and source split.
- Effect distribution and review-status bars; progress bars for changed pixel share.
- QA PASS counts, source-scene uniqueness, and cross-split source hash overlap.
- Local JPEG thumbnails to avoid loading 100 full-resolution sources when opening the page.

The generation command validates all five export file hashes, the manifest's
own run ID and schema, the QA JSON, the review-sheet SHA-256, and the linkage
between `review_metrics.json`, manifest, source hash, split, and effect parameters.
Unrelated files (such as exported ZIP backups) are skipped. Unsafe run folders,
symlinks, missing/stale reports, mismatched hashes, and invalid reviews fail closed.

The gallery's cross-split warning is **informational**. It is not a replacement
for `scripts/qa_acceptance.py audit`, which remains the source of truth for full
Stage-8 acceptance. The gallery never writes to source ZIPs, exported six-file
bundles, review sheets, or `reviews.json` and never sets `training_approved=true`.

## Known limits and next iteration

- The HTML is a static local file, not a server; it **cannot save manual verdicts**.
- It shows existing review decisions, not interactive editing. Review updates
  require the approved Stage-8 review workflow and then regenerating the gallery.
- Browser support for local `file://` links varies; use the browser that can open
  local files. No CDN, external fonts, remote JavaScript, or network required.
- Original and synthetic image paths are relative to the project; moving the HTML
  alone breaks those links. The whole project stays local/private.
- No publication of images, dataset metadata, class tables, or private workflows.
- Next: validated manual-review UI with explicit save/confirmation, then 20-image
  acceptance (at least five unique original scenes), then Stage 9 pilot of 100.

## Privacy

Only Python code, tests, and this Markdown document belong in GitHub.
`reports/batch_gallery/`, `reports/visual_review/`, `outputs/sap_runs/`, source
ZIPs, `.seg` files, and manual decisions remain untracked and local.
