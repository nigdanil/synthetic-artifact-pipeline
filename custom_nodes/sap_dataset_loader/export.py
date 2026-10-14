"""ComfyUI output node: fail-closed private, atomic SAP artifact exporter."""

from __future__ import annotations

import importlib
import json
import math
from pathlib import Path

from .node import SAPDatasetLoader


class SAPArtifactExporter:
    """Save a validated image+SEG+mask+manifest bundle under project outputs/."""

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("export_directory", "export_summary")
    FUNCTION = "export"
    CATEGORY = "SAP/Export"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Private export under project_root/outputs/sap_runs/<run_id>. "
        "Requires technical QA PASS, rechecks source ZIP and image, and refuses duplicate run IDs. "
        "Outputs are NOT approved training samples until manual review."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "project_root": ("STRING", {"default": "", "multiline": False}),
            "run_id": ("STRING", {"default": "glare_0001", "multiline": False}),
            "original": ("IMAGE",),
            "modified": ("IMAGE",),
            "mask": ("MASK",),
            "annotations": ("SAP_ANNOTATIONS",),
            "selection": ("SAP_ROI_SELECTION",),
            "qa_pass": ("BOOLEAN",),
            "qa_report": ("STRING",),
            "source_context": ("SAP_SOURCE_CONTEXT",),
            "effect_summary": ("STRING",),
            "mask_summary": ("STRING",),
        }}

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        # Deliberate explicit duplicate run IDs must be rejected even on cached executions.
        return math.nan

    def export(
        self, project_root, run_id, original, modified, mask, annotations,
        selection, qa_pass, qa_report, source_context, effect_summary, mask_summary,
    ):
        if not all(hasattr(value, "detach") for value in (original, modified, mask)):
            raise TypeError("original, modified and mask must be ComfyUI tensors")
        if not isinstance(project_root, str) or not project_root.strip():
            raise ValueError("project_root is required")
        # Reuse the explicitly selected trusted checkout and reject mixed SAP imports.
        _, root = SAPDatasetLoader._core(project_root)
        exporter = importlib.import_module("sapipeline.export")
        images = [original.detach().cpu().numpy(), modified.detach().cpu().numpy()]
        weights = mask.detach().cpu().numpy()
        if any(x.ndim != 4 or x.shape[0] != 1 or x.shape[-1] != 3 for x in images):
            raise ValueError("Expected single RGB ComfyUI images [1,H,W,3]")
        if weights.ndim != 3 or weights.shape[0] != 1:
            raise ValueError("Expected single MASK [1,H,W]")
        result = exporter.export_artifact(
            project_root=Path(root), run_id=run_id,
            original=images[0][0], modified=images[1][0], mask=weights[0],
            annotations=annotations, selection=selection,
            qa_pass=qa_pass, qa_report=qa_report,
            source_context=source_context, effect_summary=effect_summary,
            mask_summary=mask_summary,
        )
        summary = json.dumps({
            "schema_version": "sap-export-node-v1", "status": "EXPORTED",
            "run_id": result.manifest["run_id"],
            "output_sha256": result.manifest["output_sha256"],
            "qa_status": "PASS", "manual_review_required": True,
            "training_approved": False,
        }, sort_keys=True)
        return str(result.directory), summary
