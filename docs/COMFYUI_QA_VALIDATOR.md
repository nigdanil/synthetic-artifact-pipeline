# SAP QA Validator — Stage 6.6 (technical early gate)

**Status:** technical prototype accepted after local user-side ComfyUI green-PASS smoke test (263 tests passed). This is a
technical gate, **not** a claim that artificial highlights are photorealistic or
approved for YOLO training. The public workflow intentionally has an empty
`project_root`; the owner's filled local workflow must never be committed.

## What enters the QA node

| Input | Producer | Meaning |
|---|---|---|
| `original` IMAGE | SAP Dataset Loader | Original encoded-grid RGB [1,H,W,3] |
| `modified` IMAGE | SAP Highlight Effect | Photometrically modified RGB image |
| `mask` MASK | SAP Mask Builder | Float effect weights [1,H,W], **not** object segmentation |
| `annotations` SAP_ANNOTATIONS | SAP Dataset Loader | Original canonical bounding boxes |
| `selection` SAP_ROI_SELECTION | SAP ROI Selector | Original chosen object instances |

Parameters: `max_new_clipping_fraction=0.25`, `max_mask_coverage=0.60`,
`outside_tolerance=0.0` by default. Thresholds are configurable pilot values,
not calibrated scientific acceptance limits. Start with exact outside-mask equality.

## Outputs

- `qa_pass` BOOLEAN: all technical checks are true.
- `qa_report` STRING: deterministic JSON with `status=PASS|FAIL`, failed rule
  codes, counts/fractions and `training_approved=false`.
- `qa_preview` IMAGE: original modified result **with green border for PASS**,
  **red for FAIL**. This status overlay is inspection-only; **never train on it**.

The validator checks original/output shape, normalized floats, bbox dimensions
and coordinates, selection identity against the input annotation document,
non-empty/rational mask coverage, all selected ROI visibly affected, exact
unchanged pixels outside the mask, and *new* saturation of modified pixels.
Old saturated source pixels are excluded from new-clipping calculations.
It **does not** compare an exported annotation file (exporter is Stage 7), prove
true object contours, detect mislabeled source objects, prove realistic lighting,
or verify cross-split uniqueness. Manual review is still mandatory.

**Important:** the V1 starter workflow retains `SaveImage` for a *local visual
prototype PNG*. It does not have a QA-gated training exporter; a technical FAIL
may still leave a visual preview PNG. Do not use these previews as an approved
training dataset. Later Stage 7 must fail closed on `qa_pass=false`.

## Installation in existing ComfyUI — Git Bash only

Run from the root of your local trusted SAP checkout **after applying the
Stage 6.6 patch and passing tests**. Fully stop the ComfyUI server first.

```bash
COMFY='/d/ComfyUI/ComfyUI-NSFW/ComfyUI'
DEST="$COMFY/custom_nodes/sap_dataset_loader"
BACKUP='workflows/existing/sap_dataset_loader_backup_6_5'

test -f "$DEST/highlight.py" || { echo 'Missing Stage 6.5 extension'; exit 1; }
if [ ! -e "$BACKUP" ]; then cp -a "$DEST" "$BACKUP"; fi
cp -v custom_nodes/sap_dataset_loader/{__init__,node,roi,mask,highlight,qa}.py "$DEST/"
```

Restart ComfyUI completely; load the separately supplied local ready JSON from
`workflows/existing/`, confirm the Loader root path, then click **Run**.

## Acceptance checklist

- [ ] `python -m pytest -q` passes at least **263** tests.
- [ ] `python -m pip check` reports no broken requirements.
- [ ] `python -m ruff check src/sapipeline/ scripts/ custom_nodes/ tests/` passes.
- [ ] Local graph shows the original, three selected bbox and mask previews.
- [ ] `SAP QA Validator` appears; `qa_preview` shows green border on normal run.
- [ ] `qa_report` has `status=PASS`, 3 selected indices and zero pixel leakage.
- [ ] Negative run deliberately changes one pixel outside mask and yields red
  border / `no_changes_outside_mask` FAIL (covered automatically by pytest).
- [ ] Reviewer inspects the synthetic light for non-physical bbox rectangles.
- [ ] `workflows/existing/`, ZIPs, real photos, reports never enter Git.

After acceptance, move to **6.7 realistic glare**; maintain Stages 7 Export,
8 Full QA and 9 Batch Pilot as written in `docs/ROADMAP.md`.
