# Stage 8 — Private batch QA acceptance (implementation; acceptance pending)

Stage 8 is a **read-only audit** over immutable Stage-7 bundles. It does not
augment images, alter source ZIPs, overwrite exported runs or create YOLO training
sets. All reports and human reviews must stay in the Git-ignored `reports/` tree.

## Start in Git Bash

```bash
cd /path/to/checkout
source .venv/Scripts/activate
python scripts/qa_acceptance.py init --project-root .
```

This creates `reports/qa_acceptance/reviews.json`, **only once**, and initially
sets every `verdict` to `needs_review`. It must **never** auto-approve.

For each exported run, inspect `original.png`, `images/<run_id>.png`,
`masks/<run_id>.png` (16-bit effect weights), `annotations/<run_id>.seg`,
`qa.json` and `manifest.json`. Record **one human reviewer** and a verdict:
`approved`, `rejected` or `needs_review`. To approve a sample, all three review
checks must be exactly `true`:

- `product_identity`: packaging/SKU still clearly identifiable; no generative
  modifications to content, no destructive clipping;
- `glare_naturalness`: illumination does not show artificial rectangular edges,
  wrong direction or obvious discontinuities;
- `bbox_consistency`: object bounding boxes and original labels still describe
  the visible scene appropriately, including overlap/occlusion.

A `rejected` review should include a meaningful `reason`. Never edit the
`manifest_sha256` value: it binds review to the exact immutable manifest. If
new runs are added later, generate a fresh review template **at a different
private filename** and merge new pending entries by hand; don't overwrite
existing reviews. Rejected or pending samples never qualify as approved.

Run the acceptance audit:

```bash
python scripts/qa_acceptance.py audit --project-root . \
  --reviews reports/qa_acceptance/reviews.json \
  --report reports/qa_acceptance/acceptance_first.json
```

The command exits **1** for `NOT_READY` / `FAILED` (expected with fewer than 20
reviewed outputs), 0 only for `ACCEPTED`, and 2 for an execution/schema error.
Reports refuse overwrite by default; choose a **new report filename** for each
run, or explicitly use `--overwrite-report`. No dataset is exported automatically
for training.

## Automatic checks (on every bundle)

- 5 exact output file hashes from `manifest.json` and required sixth manifest;
- source ZIP/index SHA-256, unchanged RGB pixels and complete SEG fields;
- 8-bit RGB images and 16-bit MASK round trips, sizes, boundaries;
- original bbox/class preservation and selected target geometry;
- QA PASS and no changes in pixels where the **saved** effect MASK is zero;
- no duplicate raw source image SHA-256 across `learning` and `verification`,
  including entries in the private index;
- stable `run_id`, no missing files, unsafe relative paths or symlinks;
- reviewer verdict bound to `manifest_sha256` and all three manual checks.

**Acceptance gate:** at least 20 valid exported bundles, at least five distinct
source images, zero broken files/split leakage and an explicit manual approval
for **every** audited bundle. A rejected bundle is a hard failure; the audit
will not silently ignore it. Coverage thresholds are V1 review parameters,
not an assertion of a statistically representative training dataset.

Technical `PASS` on a single image is not an acceptance of the Stage-8 batch.
Before any YOLO benchmark, Stage 9 must still pass its 100-output pilot gates
and the class/label policy must be reviewed.
