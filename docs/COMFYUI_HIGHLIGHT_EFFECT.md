# ComfyUI: SAP Highlight Effect (Task 6.5)

This task adds the **fourth** SAP classic custom node to the same extension.
It consumes the original `IMAGE` from SAP Dataset Loader plus `MASK` from
SAP Mask Builder. The existing deterministic `sapipeline.effects.apply_highlight`
is reused; a separate ComfyUI photometric implementation is **not** created.

## Contract

- Inputs: `IMAGE` `[1,H,W,3]`, `MASK` `[1,H,W]`, effect kind and three strengths.
- Output: `effect_image` (`IMAGE`) and an `effect_summary` (`STRING`).
- Supported kinds: `soft_glow`, `hotspot`.
- All input tensors are read-only; source bbox/SEG geometry does not change.
- No automatic training-data approval: this is an experimental preview.
- Source and mask must have matching resolution. Zero-mask pixels are unchanged
  *exactly* on the normalized float image grid.
- CPU implementation, using NumPy and Pillow from the **existing** ComfyUI
  runtime. No diffusion model required.

## Git Bash installation

Use your existing Git Bash shell and project `.venv` for tests. **Shut down
ComfyUI completely** before copying files:

```bash
cd /d/my_work/synthetic-artifact-pipeline-public
source .venv/Scripts/activate
python -m pytest -q
python -m ruff check src/sapipeline/ scripts/ custom_nodes/sap_dataset_loader/ tests/test_comfyui_highlight_effect.py

COMFY='/d/ComfyUI/ComfyUI-NSFW/ComfyUI'
DEST="$COMFY/custom_nodes/sap_dataset_loader"
BACKUP="workflows/existing/sap_dataset_loader_backup_6_4"
test -f "$DEST/mask.py" || { echo 'SAP Mask Builder is not installed'; exit 1; }
if [ ! -e "$BACKUP" ]; then cp -a "$DEST" "$BACKUP"; fi
cp -v custom_nodes/sap_dataset_loader/{__init__,node,roi,mask,highlight}.py "$DEST/"
```

Restart ComfyUI and open `workflows/templates/sap_highlight_effect_v1.json`.
For local convenience, place a copy with your local `project_root` set into the
Git-ignored `workflows/existing/` folder. Never commit that local workflow.

The workflow includes 12 nodes; original / bbox preview / grayscale mask /
cyan overlay remain available for comparison. New connections:

```text
SAP Dataset Loader.IMAGE -----------> SAP Highlight Effect.image
SAP Mask Builder.MASK --------------> SAP Highlight Effect.mask
SAP Highlight Effect.effect_image ---> Preview Image / Save Image
```

## First-run settings

| Field | Value |
|---|---|
| effect_kind | soft_glow |
| intensity | 0.65 |
| white_boost | 0.55 |
| gamma_lift | 0.18 |

Select `random_n`, `seed=42017`, `count=3` upstream, with `control after
generate` set to `fixed`. Confirm that three products change while surrounding
pixels and original source image remain intact.

`hotspot` produces a much harsher and often rectangular-looking effect; it is
**not accepted for YOLO training** without further realism work and QA.
Do not commit the generated images or actual dataset archives to GitHub.

## Acceptance criteria

- ComfyUI recognizes `SAPHighlightEffect` after a full backend restart.
- `Preview Image` displays highlighted selected objects; source preview unchanged.
- `Save Image` emits only the derived preview to ComfyUI's local output folder.
- No shape/range errors on a real indexed JPEG+SEG pair.
- Visually review each effect. The task proves technical integration, **not**
  photorealistic data readiness.
