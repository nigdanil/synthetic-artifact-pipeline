# Task 6.1 — non-invasive ComfyUI environment and workflow preflight

**Purpose:** inspect the actual local ComfyUI installation and the user's exported
workflow *before* proposing custom nodes or changing the working graph.

This step does **not** install libraries, execute a graph, import ComfyUI modules,
modify custom nodes, or require a GPU. It reads directory names and structure
of an optional workflow JSON file. The original workflow and source data must
remain outside Git history.

## Git Bash usage (Windows)

From the SAP project, with `.venv` activated:

```bash
# Replace the sample location with your *actual* ComfyUI directory.
python scripts/inspect_comfyui.py --comfyui-dir 'D:/path/to/ComfyUI'
```

If you already exported a backup workflow into ignored `workflows/existing/`:

```bash
python scripts/inspect_comfyui.py \
  --comfyui-dir 'D:/path/to/ComfyUI' \
  --workflow workflows/existing/my_workflow.json --force
```

When the report already exists, pass `--force` to regenerate it. The report is
saved to **private**, Git-ignored `reports/comfyui_preflight.json`.

## Report scope

- Whether `main.py` and requirement files exist in the selected directory
- Directory names under `custom_nodes/` (only locally)
- Conventional embedded/virtual Python paths **relative** to the selected root
- Exported UI workflow or API prompt: node counts, node type counts, link counts
- Count of UI links whose source or target node does not exist

**Explicit omissions:** widget values, prompt text, paths in workflow inputs,
API tokens, images, and graph execution output. This script never reads
custom-node source code, executes user-supplied code or checks real GPU/CUDA
functionality.

## Next gate

Review the local report with the user. Ask for any environment details needed
to design a minimal `IMAGE`/`MASK` adapter for the *existing* graph. The workflow
JSON itself must never be committed, and ComfyUI itself must not be modified
until the user approves the integration approach.

Even the summarized node names or custom-node directory names can identify
personal software; inspect the report before sharing it publicly.
