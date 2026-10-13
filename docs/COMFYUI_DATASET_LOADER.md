# Task 6.2 — SAP Dataset Loader custom node

Only one SAP node is implemented in this task: `SAPDatasetLoader`.
It uses the existing validated ZIP/SEG reader and emits:

- `IMAGE`: ComfyUI-compatible RGB `torch.float32` `[1,H,W,3]`, 0–1.
- `SAP_ANNOTATIONS`: opaque in-process `AnnotationDocument` for a **future** selector node.
- `STRING`: short source summary (split, image dimensions, count, SHA-256); no filename or class list.

All coordinates stay on the **encoded** JPEG pixel grid; there is no EXIF rotation,
resize, or geometric transformation. The source ZIP and `.seg` are never rewritten.

## Safe installation (Git Bash, Windows)

**No ComfyUI pip installs and no updates to torch, NumPy or CUDA.**
The loader adds the trusted source tree from `project_root/src` to Python's search
path when run. It is not installed into ComfyUI's environment as a wheel.

1. From the repository, after successful `git apply --check`, tests and public-file audit:

   ```bash
   cd /d/my_work/synthetic-artifact-pipeline-public
   source .venv/Scripts/activate
   python -m pytest -q
   python -m ruff check src/sapipeline/ scripts/ custom_nodes/sap_dataset_loader/
   ```

2. Copy **only** the new extension into the *existing* ComfyUI, making a backup
   if the target folder already exists. To avoid overwriting user code, this
   example deliberately refuses if that directory already exists:

   ```bash
   COMFY='/d/ComfyUI/ComfyUI-NSFW/ComfyUI'
   test -f "$COMFY/main.py" || { echo 'ComfyUI not found'; exit 1; }
   test ! -e "$COMFY/custom_nodes/sap_dataset_loader" || {
     echo 'Existing sap_dataset_loader detected; do not overwrite'; exit 1;
   }
   mkdir -p "$COMFY/custom_nodes/sap_dataset_loader"
   cp custom_nodes/sap_dataset_loader/*.py "$COMFY/custom_nodes/sap_dataset_loader/"
   ```

3. **Restart ComfyUI** (use its normal UI controls) so it discovers the extension.
   This is the only manual installation step. The project `.venv` is independent
   of ComfyUI's own `.venv`; never mix interpreters.

4. Open `workflows/templates/sap_dataset_loader_v1.json` through ComfyUI
   **Workflow → Open**. Set `project_root` on the `SAP Dataset Loader` node to:

   `D:/my_work/synthetic-artifact-pipeline-public`

   Leave `split=learning`, `sample_index=0`, `image_member` blank. Click **Run**.
   A real source photo should appear in PreviewImage; the same photo is saved by
   SaveImage under ComfyUI's `output/`.

5. If the first example fails due to missing index, rerun in **project Git Bash**:

   ```bash
   python scripts/index_dataset.py
   ```

   Ensure the two source ZIP files remain under `data/raw/`. The node never
   downloads datasets, trains models, or selects ROI; those are later tasks.

## Inputs

| Name | Meaning |
|---|---|
| `project_root` | Absolute path to your own trusted SAP repository |
| `split` | `learning` or `verification` |
| `sample_index` | Zero-based index among eligible **globally unique**, normally oriented samples |
| `image_member` | Optional exact ZIP member `img/example.jpg`, requires `sample_index=0` |

`IS_CHANGED` deliberately invalidates the node's cache each execution so a changed
ZIP or index must pass SHA-256 verification before producing an image. Explicit
selection can inspect otherwise filtered entries, but non-default EXIF orientation
is still rejected until manually resolved.

## Operational cautions

- ComfyUI workflow JSON is not the dataset: don't export or publish a workflow
  after filling it with private filenames or machine-specific paths.
- Local source archives, private index, manifests, and rendered source images
  are **not** for public GitHub. The provided template has empty path fields.
- `SAP_ANNOTATIONS` is a custom in-process data type, **not** segmentation masks.
- This stage is CPU I/O and tensor conversion. No diffusion model and no GPU
  inference are required.
- For manual rollback, stop ComfyUI and remove only the newly copied
  `custom_nodes/sap_dataset_loader/` directory, then restart ComfyUI.
