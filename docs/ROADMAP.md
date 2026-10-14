# SAP implementation roadmap and acceptance gates (2026-10-13)

This is the public, non-sensitive progress tracker. The private dataset audit,
original archives and local ComfyUI workflows must remain outside Git history.

## Existing implementation (prototype)

- [x] Stage 1 — read-only ZIP indexing, source pairing, SHA-256 and duplicate audit.
- [x] Stage 2 — `.seg` and YOLO annotation adapters; canonical bounding boxes.
- [x] Stage 3 — verified local image loader, ROI choice and previews.
- [x] Stage 4 — rectangular/soft-rect bbox-derived effect masks. **Ellipse and more precise shapes are still pending.**
- [x] Stage 5 — deterministic `soft_glow` and `hotspot` CPU image effects. **Directional glare and realism review are still pending.**
- [x] Stages 6.1–6.5 — ComfyUI preflight, Dataset Loader, ROI Selector, Mask Builder and Highlight Effect, tested end-to-end on one private source image.

These checked items mean the *prototype implementation* was verified, not that
all acceptance criteria for production-quality synthetic data are met.

## Next tasks — one at a time

- [x] **6.6 — Early SAP QA Validator (accepted technical prototype):** accept original IMAGE,
  modified IMAGE, MASK, annotations and ROI selection; check dimensions,
  bbox/annotation invariants, exact unchanged pixels outside the mask,
  clipping/coverage and return a structured QA result plus clear failures.
  **Gate:** 263 automated tests and Ruff passed; real-image ComfyUI smoke test showed green QA PASS. Negative failure cases covered by automated tests; technical PASS is not training approval.
- [x] **6.7 — Realistic glare (accepted technical prototype):**
  added a separate seeded ROI-local elliptical/directional Gaussian light
  envelope with angle, radius, aspect, falloff, intensity and jitter controls.
  Its actual effect MASK is routed through technical QA; geometry and annotations
  remain invariant. This does not yet model the freezer glass plane.
  **Gate:** 286 Windows automated tests/Ruff passed; both ComfyUI modes
  rendered nonrectangular masks and green QA previews. No training approval.
- [x] **7 — Artifact Exporter (accepted):** QA-gated private bundles include
  original/modified PNG, 16-bit effect MASK, original SEG geometry, QA JSON,
  SHA-256 manifest and verified ZIP source provenance. Windows: 306 tests passed,
  1 skipped, Ruff clean; real ComfyUI `glare_0001` export produced all 6 files,
  with independent manifest hash verification and a clean Git commit `6a778f2`.
- [ ] **8 — Full QA acceptance (implementation ready; dataset gate open):**
  audit every Stage-7 bundle and its private ZIP source, detect broken hashes,
  changed geometry, unmasked edits and cross-split leakage; create a separate
  manifest-bound human review file. Require at least 20 reviewed bundles,
  at least 5 unique sources, explicit approval of each, no failed checks.
  **Gate still open:** Windows tests and actual human review of 20 exported
  outputs. CLI and instructions: `docs/QA_ACCEPTANCE_STAGE8.md`.
- [ ] **8.1 — Visual Review Sheet (implementation pending user-side check):**
  offline, private original/modified/mask/diff comparison; target ROI zooms,
  changed pixel and brightness metrics, histogram, effect parameters, QA and
  manifest-bound manual verdict. Stored under ignored `reports/visual_review/`
  without changing any six-file Stage-7 export.
  **Gate:** Windows pytest/Ruff, a real `glass_reflection_0001` report,
  inspect readability and verify no source/manifest bytes change.
  See `docs/VISUAL_REVIEW_SHEET.md`.
- [ ] **9 — Batch pilot:** produce 100 variants from a safe set of unique
  source scenes, with resume, seeds and separate QC results.
  **Gate:** no missing manifests/labels, no cross-split leakage and no
  unreviewed failures before any YOLO training comparison.

The original master-plan stages remain numbered **7 Export, 8 QA, 9 Pilot**.
We deliberately add *early QA at 6.6 before export* to detect defects sooner,
while Stage 8 remains the formal dataset-wide acceptance gate.

## Technical constraints

- Input archives stay read-only; `data/raw/`, local reports, generated images,
  class mappings and `workflows/existing/` are Git-ignored.
- Bbox effect masks are not object segmentation and can include adjacent
  products. Their limitations must be visible in QA.
- Identical seed and parameters must reproduce the same output.
- Do not change source coordinates or classes when applying photometric
  effects; any future annotation policy change requires a separate review.
- Current ComfyUI custom nodes are CPU-first integrations; do not reinstall
  Torch/CUDA or models as part of these tasks.
