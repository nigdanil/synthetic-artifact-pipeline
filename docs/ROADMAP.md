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

- [ ] **6.6 — Early SAP QA Validator (next):** accept original IMAGE,
  modified IMAGE, MASK, annotations and ROI selection; check dimensions,
  bbox/annotation invariants, exact unchanged pixels outside the mask,
  clipping/coverage and return a structured QA result plus clear failures.
  **Gate:** automated tests, one real-image run, negative cases, manual review.
- [ ] **6.7 — More realistic glare:** introduce smooth elliptical / oriented /
  directional illumination with controlled angle, radius, falloff,
  intensity and seed; retain mask constraints and photometric invariants.
  **Gate:** a small, deterministic set of varied examples reviewed for
  unnatural rectangular borders and preserved product identity.
- [ ] **7 — Annotation-preserving Artifact Exporter:** output modified image,
  original geometry/labels, effect mask and reproducibility manifest in a
  new local run directory; no overwrites and no publication of dataset assets.
  **Gate:** re-open every artifact, verify hashes, metadata, seed and bbox;
  verify safe failure on incomplete writes or duplicate output names.
- [ ] **8 — Full QA acceptance:** review at least 20 outputs, including
  image quality, clipping, duplicate leakage and target/class integrity.
  **Gate:** automated reports plus manual approvals; failed results rejected.
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
