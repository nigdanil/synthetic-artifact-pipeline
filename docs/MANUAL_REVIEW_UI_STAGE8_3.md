# Stage 8.3 — Private Manual Review UI

**Scope:** a local Python HTTP interface for recording *explicit human decisions* in
`reports/qa_acceptance/reviews.json`. It is a tool for the Stage-8 acceptance
process, **not** an automatic training-quality classifier, effect generator, or
replacement for `scripts/qa_acceptance.py audit`.

## Requirements and startup (Windows Git Bash)

1. Use the SAP checkout with `.venv` active. Its `outputs/sap_runs/` must
   contain complete Stage-7 bundles and `reports/visual_review/` must contain
   valid Stage-8.1 comparison sheets. Run
   `python scripts/render_review_sheet.py --all` first when new runs exist.
2. Start the local interface:

   ```bash
   cd /d/path/to/synthetic-artifact-pipeline-public
   source .venv/Scripts/activate
   python scripts/manual_review_ui.py --port 8765
   ```
3. Open **http://127.0.0.1:8765/** in the browser, not as `file://`.
   Keep the Git Bash process running; stop with **Ctrl+C**.
4. Select one run. Inspect Original, Modified, the Full Review Sheet, and the
   bounding-box crops. Enter the human reviewer's name and fill all three
   checks (`product_identity`, `glare_naturalness`, `bbox_consistency`). Set
   `approved`, `rejected`, or `needs_review`; a rejection requires a meaningful
   reason (at least 8 characters). Click **Save decision**.
5. The page confirms a completed save. No auto-save, no default approval.
   Repeat for further runs; refresh the page if it reports a changed review or
   manifest. The existing `reviews.json` is **never reset by the UI**.

If `reports/qa_acceptance/reviews.json` has not been initialized, its first
version is created with an empty list, and the first saved review is added.
If it already exists, only the explicitly selected run is updated. Previous
revisions of the review file are backed up under
`reports/qa_acceptance/backups/` using their SHA-256 filenames. These backups
and all human review decisions are private and Git-ignored.

## Safety / invariants

- HTTP binds **only to `127.0.0.1`**, not `0.0.0.0`. Do **not** forward
  the port, proxy it, or expose it to a network. The interface is intended
  for personal trusted local use, not a multiuser production security boundary.
- POST requires matching local `Host`/`Origin`, a server-generated CSRF token
  and a bounded JSON body. Browser assets are served from an allowlisted route.
- Every save reloads the artifact gallery, checks its exported file hashes,
  checks the manifest's SHA-256 against the reviewed version, rejects stale
  concurrent browser reviews using an `reviews_etag` and writes atomically.
- To approve, technical `QA PASS`, all three **explicitly true** manual checks,
  and a nonblank reviewer are required. Rejection requires a reviewer and reason.
  Pending results stay pending; the UI never synthesizes human approval.
- Changes to the manifest invalidate the comparison sheet and previous review.
  Regenerate the sheet and explicitly inspect/approve the changed version.
- No modifications to the six-file export bundle, source ZIP, original SEG,
  masks, or source images. No network uploads, external JS or dependencies.
- Neither this UI nor the existing offline gallery may set
  `training_approved=true`. That gate remains **false** until separately
  reviewed Stage 8 and Stage 9 pilot requirements are met.

## Inspect or audit results

Run the existing batch audit with a **new report path** to avoid overwriting
prior QA reports:

```bash
python scripts/qa_acceptance.py audit \
  --project-root . \
  --reviews reports/qa_acceptance/reviews.json \
  --report reports/qa_acceptance/acceptance_after_manual.json
```

The normal exit code is 1 (`NOT_READY`) until the formal acceptance gate:
at least **20 technically valid outputs**, from **5 distinct source images**,
with *every* run manually approved, no failed results and no cross-split leakage.
An explicit rejection forces `FAILED` (by design). Five variants of the same
original photograph do **not** count as five unique source scenes.

If a ZIP backup such as `glass_reflection_0001.zip` exists directly inside
`outputs/sap_runs/`, move it **out of that folder** into a separate Git-ignored
private backup location before running the acceptance auditor. The current
Stage-8 audit treats extra entries inside `outputs/sap_runs/` as invalid runs.
Do not delete or modify an exported bundle to silence a failure.

After reviewing, regenerate the *read-only snapshot* to reflect decisions:

```bash
python scripts/build_batch_gallery.py --overwrite
```

## Current roadmap

- Stage 7: exported image / mask / SEG / QA / manifest integrity verified.
- Stage 8.1: private `review_sheet.png` and `review_metrics.json`.
- Stage 8.2: offline read-only gallery with filters and distributions.
- **Stage 8.3:** this local manual-review editor; acceptance requires a
  Windows/browser smoke test and inspection of saved `reviews.json`.
- Stage 8.4: actual **20-output / 5-source** human acceptance gate (pending).
- Stage 9: reproducible **100-output** pilot, followed by YOLO comparison
  against a real holdout set (not approved by this editor).

**Public GitHub scope:** only Python modules, unit tests and this Markdown
file. No photo, ZIP, SEG, private workflow, review report, or `reviews.json`.
