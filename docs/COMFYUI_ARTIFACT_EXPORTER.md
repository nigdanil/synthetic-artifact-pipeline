# Stage 7: SAP Artifact Exporter (private QA-gated candidate bundle)

**Status:** implementation candidate; user-side Windows/ComfyUI acceptance pending.
The export is **NOT a YOLO training approval**. See `ROADMAP.md` for the subsequent
Stage 8 manual acceptance and Stage 9 batch pilot.

## Contract and nodes

`SAP Dataset Loader` now has a **fourth**, in-process output called
`SAP_SOURCE_CONTEXT`. Existing `IMAGE`, `SAP_ANNOTATIONS` and `source_summary`
remain in slots 0, 1 and 2. The new opaque output contains relative indexed
ZIP member names and SHA-256 hashes and is **not** embedded into a public
workflow JSON. ComfyUI must be restarted after copying the updated custom nodes.

`SAP Artifact Exporter` (an output node) requires all of:

- trusted `project_root` (absolute path), new unique `run_id`;
- `original` IMAGE, **effect** `modified` IMAGE and the **actual effect** MASK;
- original `SAP_ANNOTATIONS` and `SAP_ROI_SELECTION`;
- `qa_pass` BOOLEAN and `qa_report` STRING from SAP QA Validator;
- `source_context` from SAP Dataset Loader;
- `effect_summary` STRING from SAP Realistic Glare and `mask_summary`
  STRING from SAP Mask Builder.

This V1 supports **SEG source** annotations and the `sap-glare-v1` effect only.
It does not infer SKU IDs, automatically create `glint` labels, or generate YOLO
TXT without an externally defined class-ID mapping.

## Fail-closed behavior

Before disk writes, exporter **recomputes QA** using strict default thresholds,
compares QA checks/metrics with the connected report, verifies seed and effect
statistics, **reopens the indexed source ZIP** and confirms the source image
pixels and entire annotation object. A failed/invalid QA, stale/mismatched
report, wrong source hash or modified bbox prevents export. Even a green QA
preview is **not sufficient** without passing these fresh checks.

No arbitrary output path is accepted: results go to the local project's ignored
`outputs/sap_runs/<run_id>/`, and an existing run directory is **never** replaced.
The exporter writes a temporary sibling directory, reopens each written
artifact, verifies lossless image/mask/SEG round trips and moves the complete
bundle into place. An ordinary write error removes its unfinished staging data.
A process crash could leave a hidden `.pending_*` temporary directory; remove
such abandoned directories only after stopping ComfyUI and checking runs.

## Per-run contents

```text
outputs/sap_runs/glare_0001/
  original.png                   # decoded source JPEG grid, lossless PNG
  images/glare_0001.png          # modified RGB (lossless PNG)
  annotations/glare_0001.seg     # SEG source fields preserved; filename changed
  masks/glare_0001.png           # 16-bit lossless grayscale actual effect mask
  qa.json                        # recomputed technical QA outcome
  manifest.json                  # source provenance, checksums, targets, parameters
```

The manifest has schema `sap-artifact-v1`, records effect parameters,
ROI selection seed, Mask Builder settings and NumPy/Pillow versions.
Source archive/member paths are
relative **inside the private dataset ZIP**; source SHA-256 refers to JPEG/SEG
source bytes verified by the indexer, while `output_sha256` refers to **saved
artifact bytes**. Source original PNG is a lossless rendering of decoded JPEG
pixels, not a copy of JPEG bytes. The 16-bit mask encodes weights as
`round(weight * 65535)`; float32 weights are reproducible from effect settings,
not bit-for-bit preserved by the 16-bit PNG.

Always `qa.manual_review_required=true`, `qa.training_approved=false`.
Geometry is marked unchanged and only SEG `filename` is rewritten.

## Manual local acceptance (Git Bash)

1. Run all pytest/Ruff tests and `scripts/check_public_repo.py` **before** copying
   node changes into the active ComfyUI.
2. Back up the existing `sap_dataset_loader` extension, copy its updated
   `__init__.py`, `node.py`, `roi.py`, `mask.py`, `highlight.py`, `qa.py`,
   `glare.py`, `export.py`; fully restart ComfyUI.
3. Open the local (Git-ignored) ready workflow, check `project_root` in both
   Loader and Exporter. Leave all effects unchanged for the initial test.
4. Set a new `run_id`, e.g. `glare_0001`; run once and confirm QA green, then
   inspect `outputs/sap_runs/glare_0001/` and reopen all six files.
5. Run again with **the same** `run_id`. Expect `duplicate_output`, and verify
   the first bundle has not changed. Change to `glare_0002` for subsequent runs.
6. Deliberately set a failing technical QA threshold / reconnect the wrong
   mask: no **new** `sap_runs/<run_id>` directory may appear.
7. Compare original and exported bbox and unknown SEG fields; validate all
   hashes listed in `manifest.json` before marking Stage 7 accepted.

**Do not commit** local workflows, generated images, source archives,
real annotations, reports or exported manifests into the public repository.
