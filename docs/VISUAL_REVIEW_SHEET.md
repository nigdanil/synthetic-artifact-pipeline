# Stage 8.1 — Visual Review Sheet (private, read-only)

The **Stage-7 six-file exported bundle stays immutable**. All visual material
is created only under the Git-ignored `reports/visual_review/<run_id>/`.
Never place raw images, SEG files, generated reports, or machine-specific
workflows under public Git-tracked directories.

## Usage — Windows Git Bash

```bash
cd /d/my_work/synthetic-artifact-pipeline-public
source .venv/Scripts/activate
python scripts/render_review_sheet.py --run-id glass_reflection_0001
```

Outputs:

```text
reports/visual_review/glass_reflection_0001/
  review_sheet.png       # 1880 x 3160 human-readable report
  review_metrics.json    # machine-readable measurements and PNG SHA-256
```

To process all exported runs (Stage 8 >=20), run:

```bash
python scripts/render_review_sheet.py --all
```

Existing report directories are skipped by default, **not silently overwritten**.
To refresh after changing `reports/qa_acceptance/reviews.json`:

```bash
python scripts/render_review_sheet.py --run-id glass_reflection_0001 --overwrite
```

`--overwrite` touches **derived review files only**, never `outputs/sap_runs/`.
Reviewing a picture does *not* modify `verdict` in `reviews.json`.

## What the sheet displays

1. Original exported RGB photograph (unchanged).
2. Modified RGB photograph.
3. The **actual PNG-16 mask**, visualized as grayscale.
4. Absolute-difference heatmap. **Contrast is deliberately enhanced**, so a
   red area is *not* a claim of saturation or physical temperature.
5. Mask overlay on the original, followed by three paired target ROI crops, original versus modified; boxes originate
   from `manifest.targets` without recomputing bbox geometry.
6. Numeric QA/photometric metrics (global and mask-local) and a histogram of
   max per-pixel channel difference (0–255) *inside the encoded mask*.
7. Effect parameters from `manifest.effects`, technical QA result, and an
   independently manifest-bound **manual review verdict** if available.

The detailed `review_metrics.json` stores source hash/split, run ID, effect
configuration (including seed), SHA-256 of the manifest, histogram counts,
mask coverage, changed pixels, inside/outside-mask pixel changes, brightness
shift, clipped fraction, and PNG SHA-256. It explicitly says
`training_approved: false`.

**Compatibility:** `sap-glare-v1` and `sap-glass-reflection-v1` are displayed
without changing either effect or its six-file export contract. Batch command
is deterministic for a given bundle, review file and image/font environment.

## Integrity and manual acceptance

Before drawing anything, the program checks all manifest-listed SHA-256 hashes,
PNG dimensions/encoding, image/mask alignment, QA PASS, safe relative paths,
no changed pixels outside the stored PNG-16 mask, and missing/unexpected input
keys. A corrupted bundle fails closed with non-zero exit and is *not repaired*.

Human verdict is **read-only** from `reports/qa_acceptance/reviews.json`.
`approved` is displayed only when `manifest_sha256` matches and all three
required checks plus the reviewer are present. Stale or incomplete approvals
are labelled explicitly. This sheet never sets `approved` or
`training_approved`.

This report supports Stage-8 visual inspection; Stage-8 batch audit and the
20-output/5-source/manual-approval gate remain authoritative. For example,
`qa.status=PASS` and `manual_verdict=needs_review` is expected after export.

## Limitations

- Bbox zoom crops are inspection aids; they are **not segmentation masks**.
- A projected glass-plane effect can touch metal frames and adjacent items;
  human review still judges realism and label consistency.
- Mean brightness and diff heatmap cannot prove that a sample is useful to
  YOLO; scientific evidence requires Stage-9/benchmarking against real data.
- `review_metrics.json` is private because source hashes and effect settings
  can reveal dataset information. The project may contain licensed images.
