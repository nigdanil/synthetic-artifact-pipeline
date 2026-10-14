# SAP Pilot 100 — private technical QA preview for a deadline

**Goal:** Create 100 *new* QA-PASS photometric synthetic candidates from at
least 50 distinct learning scenes, while maintaining the Stage-8 acceptance gate.
This is a technical pilot, **not** a training-approved or photorealistic glass
reflection dataset. It does not use or modify the existing `reports/batch_run/plan.json`.

Only previously validated bbox-local `glare_directional` / `glare_elliptical`
are used. Do **not** describe these as reflections over the freezer glass plane.
Every exported candidate has the immutable six-file Stage-7 bundle, verified
source SHA, original SEG geometry and technical QA report. Manual review is
still `needs_review` by default.

## Windows Git Bash, inside `.venv-1`

```bash
python scripts/pilot100.py plan                   # 60 scenes / 120 candidates / target 100
python scripts/pilot100.py run --max-new 1        # first smoke test
python scripts/pilot100.py run --max-new 100      # after smoke test: resume to total 100
python scripts/pilot100.py pack                   # only after 100 SHA-verified QA-PASS runs
```

The new plan and ZIP are **private**, located in Git-ignored
`reports/pilot100/plan.json` and `reports/pilot100/pilot100_qa_preview.zip`.
A plan already present is never overwritten. Candidate run IDs are deterministic
`p9_<source-hash-prefix>_<variant>`; pre-existing runs are checked and resumed
without mutation. `run --max-new 100` produces at most 100 new bundles and stops
when the pilot has 100 passing candidates. Ten extra scenes are reserved to
absorb a few Technical QA failures. Only explicit per-candidate technical QA
failures are skipped and printed; invalid or tampered bundles abort the run.

**Never delete a failed or an already exported run just to force a rerun.**
If fewer than 100 PASS candidates are available, stop and inspect QA failures.
Rerunning verifies previously exported manifests and file hashes. Do not run
two pilot processes in parallel.

After generation, for private visual spot checks:

```bash
python scripts/render_review_sheet.py --all
python scripts/build_batch_gallery.py --overwrite
python scripts/manual_review_ui.py --port 8765
```

`pack` produces a ZIP containing exactly 100 modified PNG images and their
corresponding `.seg` annotation files, plus `qa_preview_manifest.json` with
source hashes, effect type and **unapproved** status. Packaging re-verifies
export hashes and refuses any existing output ZIP. The contents are suitable
for **internal technical preview only**. Business stakeholders must be told
that the effect is a limited bbox-local pilot and training readiness requires
separate human review and acceptance; do not publish the raw image/label set.

The public Git commit may include **only** this module, CLI, tests, and doc.
Never commit `reports/`, `outputs/`, ZIP files, indexed image member names,
private SKU mappings, or raw source JPEG/SEG.
