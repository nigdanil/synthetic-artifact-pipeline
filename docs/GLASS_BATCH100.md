# Glass BBox Pilot: Batch 100 (QA preview only)

## Goal

Generate 100 **additional** bbox-driven synthetic glare images from the
existing private plan `reports/glass_pilot100/plan.json`, using two variants:
`glass_diffuse` and `glass_fluorescent`. This is **not** the existing first
100-image local glare pilot. The plan contains 60 unique source scenes and 120
candidate variants; the batch stops after 100 technical QA PASS candidates.

`glass_quad_norm` may stay `null`. This workflow uses the deterministic
product-bbox cluster selection from `glass_auto_preview.py`; it does **not**
segment the physical glass panel. It does not overwrite source ZIP archives,
SEG annotations, first pilot runs, or existing v2 previews.

## Commands (Git Bash; from repository root)

```bash
python -m ruff check src/sapipeline/ scripts/ tests/
python -m pytest -q tests/test_glass_batch100.py
python scripts/glass_batch100.py run --max-new-scenes 1
python scripts/glass_batch100.py run --max-new-scenes 60
python scripts/glass_batch100.py pack
```

`run` reuses any existing **verified** `_v2` preview scene. It refuses to
silently trust a modified/corrupt report or image: the effects and masks are
recomputed from the source and checked. The default `--max-new-scenes=60`
allows the full pilot. If processing is interrupted, run the same command
again; already generated scenes are validated and skipped.

If the output says `qa_pass=100`, the QA-PASS target was achieved. Individual
scenes can produce QA rejects, so a 100-image result is **not guaranteed**.
Do not run `pack` until the full 100 are verified. If the plan exhausts without
100, review rejected scenes rather than duplicating existing sources.

## Packaging

`pack` writes only when 100 technical QA PASS previews are available:

`reports/glass_pilot100/glass_bbox100_qa_preview.zip`

Inside the private ZIP:

- `images/pg2a_..._1.png` etc. — 100 modified RGB images;
- `annotations/pg2a_..._1.seg` etc. — 100 SEG copies with only the
  `filename` field adjusted for the output PNG; box geometry and unknown fields
  retained;
- `masks_visual/pg2a_..._1.png` — 8-bit visualizations, **not** ground truth segmentation;
- `qa_preview_manifest.json` — exact identifiers, source hashes, actual effect
  parameters/seed, technical PASS status and SHA-256 output hashes.

The independent auto-preview renderer uses seeds derived as
`plan.base_seed + 10*scene_index + variant_index` and **its own** effect
profiles, different from the plan's manual-quad variant parameters. This
package records actual auto-preview parameters and uses distinct `pg2a_` IDs;
it must not be presented as the manual-quad runs in `plan.runs`.

**The ZIP is a private manual QA candidate deliverable only**, not a Stage-7
artifact and not an approved YOLO training set. `training_approved=false`.
Do not add the ZIP or private `reports/` or `data/` to Git.

## Failure conditions and manual QA

- Existing folder mismatch: fail closed, never delete/overwrite.
- Missing annotation/source pair or index hash mismatch: fail closed.
- Wrong image dimensions, changed geometry, changed seed or tampered PNG: fail closed.
- Existing ZIP: refuse overwrite.
- Automatic geometry may brighten the freezer frame / metal divider; technical
  PASS is not evidence of physical plausibility. Inspect varied scenes and
  reject poor-looking reflections before training.
