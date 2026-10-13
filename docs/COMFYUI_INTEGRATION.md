# Integrating an existing ComfyUI workflow

The owner already has an existing workflow. **Do not recreate or modify it** until its node types, inputs, and dependencies have been reviewed.

- `src/sapipeline/` will hold a reusable Python core (not yet implemented).
- `custom_nodes/` is a placeholder for a future ComfyUI adapter.
- `workflows/existing/` is local-only. Do not upload the owner's JSON: it may contain secrets, paths, downloaded model names or private metadata.

## Integration sequence (later, one task at a time)

1. Export a backup copy of the existing ComfyUI graph to `workflows/existing/`.
2. Inspect node types, Python/ComfyUI versions and installed dependencies locally.
3. Agree on `IMAGE`/`MASK` and annotation metadata interfaces.
4. Implement and test a single input and bbox visualization before adding effects.
5. Change a **copy** of the ComfyUI graph, never the working original.

Do not pip-install or upgrade torch, NumPy, CUDA or other libraries inside the existing ComfyUI environment at scaffold-setup time.

Official references: https://docs.comfy.org/custom-nodes/overview and https://docs.comfy.org/custom-nodes/walkthrough
