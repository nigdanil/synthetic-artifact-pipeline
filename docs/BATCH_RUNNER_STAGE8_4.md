# Stage 8.4 — Private serial batch-runner (candidate generation; gate still open)

**Purpose:** generate more reproducible Stage-7 QA-gated candidate bundles from
new indexed learning scenes, without a ComfyUI server, GPU model, or 20 manual
workflow executions. This is not an approval mechanism, training-data exporter,
or Stage-9 100-image pilot. Visual quality must be reviewed by a human.

## First use (Windows Git Bash, Python `.venv-1` with existing dependencies)

```bash
cd /d/my_work/synthetic-artifact-pipeline-public
python scripts/batch_run.py plan --scenes 4
```

Creates a **private** `reports/batch_run/plan.json` once, refusing to overwrite.
Excludes source SHA-256s already found among `outputs/sap_runs/` manifests,
thus four additional source scenes together with the existing one yield at
least five unique original scenes. Each chosen scene is normally oriented,
unique across the *entire* indexed dataset and contains at least one bbox. All
new generated samples are in the `learning` split, not `verification`.

Four variants per new source: `glare_directional`, `glare_elliptical`,
`glass_soft`, `glass_strong`. With four scenes, 16 candidate runs are planned;
plus five previously exported runs, the batch will contain 21 candidates if
all technical QA gates pass. The acceptance gate is **at least 20** valid
runs across **at least five** sources, with individual manual approvals.

**Do not blindly reuse the first freezer's glass plane.** In the *private*
plan, each source has `glass_quad_norm: null`. Before running either `glass_*`
variant for that source, inspect its image and replace null with four normalized
`[x,y]` points in **TL, TR, BR, BL** order. The quadrilateral must describe
actual glass, not merchandise; verify shape visually. For example, only as
an illustration (do not reuse across scenes):

```json
"glass_quad_norm": [[0.10, 0.20], [0.90, 0.20], [0.90, 0.80], [0.10, 0.80]]
```

Only edit `glass_quad_norm`. Changing source identifiers, variants, `run_id`
or `seed` is blocked. The glass effect must still satisfy ROI intersection
and technical QA; a glass plane without an affected product will be rejected.

## One-run smoke test (do not start all 16 yet)

```bash
python scripts/batch_run.py run --max-new 1
```

The command serially verifies the private ZIP/SEG/index and effect, recomputes
technical QA, and calls the existing immutable Artifact Exporter. Any QA failure
stops that run **without exporting**. It never overwrites existing run IDs and
checks existing output SHA hashes before skipping previously generated ones.
A successful export is still **NOT** manually approved.

After checking the first result, generate its review sheet and inspect:

```bash
python scripts/render_review_sheet.py --all
python scripts/build_batch_gallery.py --overwrite
python scripts/manual_review_ui.py --port 8765
```

Future runs: call `python scripts/batch_run.py run --max-new 1` repeatedly
until the planned glass variants require geometry. After geometry is reviewed,
run one at a time. The CLI can process up to 20 newly created candidates per
invocation but **single-run mode is recommended** for first validation.
Avoid simultaneous processes using the same plan.

Re-audit with a **new** private report filename:

```bash
python scripts/qa_acceptance.py audit --project-root . \
  --reviews reports/qa_acceptance/reviews.json \
  --report reports/qa_acceptance/acceptance_after_batch.json
```

Exit `1` and `NOT_READY` is expected until every output is manually reviewed.
The local Manual Review UI can add new review entries, preserving existing ones.
Never rerun `qa_acceptance.py init` on an existing review file.

## Invariants / limitations

- Source ZIP, SEG, index, Stage-7 exports, manual reviews are read-only.
- All output bundles are private under Git-ignored `outputs/sap_runs/`.
  The plan is private under Git-ignored `reports/batch_run/`.
- The generator is CPU-bound and works on one image at a time. Full-resolution
  images/quad homography can use significant RAM; start with `--max-new 1`.
- Stronger variants may fail clipping, coverage or visible-change QA.
  **Do not weaken QA thresholds** to force throughput; inspect parameters.
- Existing run IDs are not overwritten or renamed. Resumption is bounded by
  checksums, source identity, seed, and effect kind. No automatic `approved`.
- Batch plan must not enter GitHub. Do not publish source image names,
  thumbnails, masks, `.seg`, private review sheets or class names.
