"""Atomic, annotation-preserving export of one reviewed SAP effect candidate.

Exports are always private and must be manually reviewed before training.
No source ZIP or original annotation is modified. A complete bundle is either
committed as a new directory or absent; existing bundles are never overwritten.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sapipeline.annotations import AnnotationDocument, parse_seg, to_seg_json
from sapipeline.dataset import choose_index_entry, load_indexed_pair, read_private_index
from sapipeline.qa import validate_artifact
from sapipeline.selection import SelectionResult

_RUN_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}\Z")
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")


class ArtifactExportError(ValueError):
    """Expected, fail-closed exporter input or filesystem rejection."""


@dataclass(frozen=True, slots=True)
class ArtifactExportResult:
    directory: Path
    manifest: dict[str, Any]


def _json_object(value: str, label: str) -> dict[str, Any]:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a JSON string")
    try:
        result = json.loads(value)
    except (ValueError, TypeError) as exc:
        raise ArtifactExportError(f"{label} must contain valid JSON") from exc
    if not isinstance(result, dict):
        raise ArtifactExportError(f"{label} must be a JSON object")
    return result


def _rgb(value: np.ndarray, label: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{label} must be a NumPy array")
    if value.ndim != 3 or value.shape[2] != 3:
        raise ArtifactExportError(f"{label} must be RGB [H,W,3]")
    if value.dtype.kind != "f" or not np.isfinite(value).all():
        raise ArtifactExportError(f"{label} must contain finite normalized floats")
    if (value < 0).any() or (value > 1).any():
        raise ArtifactExportError(f"{label} must be in [0,1]")
    return np.rint(value * 255).astype(np.uint8)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_provenance(
    project_root: Path, source_info: dict[str, Any], original_rgb: np.ndarray,
    annotations: AnnotationDocument,
) -> dict[str, Any]:
    """Reopen ZIP via the private index; don't trust arbitrary workflow metadata."""
    required = ("archive", "image_member", "annotation_member", "image_sha256",
                "annotation_sha256", "split")
    if any(not isinstance(source_info.get(key), str) or not source_info[key]
           for key in required):
        raise ArtifactExportError("source_context lacks indexed ZIP provenance; update SAP Dataset Loader")
    if source_info["split"] not in ("learning", "verification"):
        raise ArtifactExportError("Unrecognized source split")
    if not all(_SHA256.fullmatch(source_info[key])
               for key in ("image_sha256", "annotation_sha256")):
        raise ArtifactExportError("Invalid source SHA-256")
    index = read_private_index(project_root / "data/work/index/dataset_index.json")
    entry = choose_index_entry(
        index, split=source_info["split"],
        image_member=source_info["image_member"],
    )
    for key in required:
        if entry.get(key) != source_info[key]:
            raise ArtifactExportError(f"Source index mismatch: {key}")
    pair = load_indexed_pair(index, entry, data_dir=project_root / "data/raw")
    if pair.annotations != annotations:
        raise ArtifactExportError("Source annotations differ from private ZIP")
    if not np.array_equal(np.asarray(pair.image.convert("RGB")), original_rgb):
        raise ArtifactExportError("Source IMAGE differs from verified JPEG")
    return {key: entry[key] for key in required}


def _require_clean_root(project_root: Path) -> Path:
    if not isinstance(project_root, Path):
        raise TypeError("project_root must be an absolute Path")
    if not project_root.is_absolute():
        raise ArtifactExportError("project_root must be an absolute Path")
    if project_root.is_symlink() or not (project_root / "pyproject.toml").is_file():
        raise ArtifactExportError("project_root must be an existing project checkout")
    return project_root.resolve(strict=True)


def export_artifact(
    *,
    project_root: Path,
    run_id: str,
    original: np.ndarray,
    modified: np.ndarray,
    mask: np.ndarray,
    annotations: AnnotationDocument,
    selection: SelectionResult,
    qa_pass: bool,
    qa_report: str,
    source_context: dict[str, Any],
    effect_summary: str,
    mask_summary: str,
) -> ArtifactExportResult:
    """Export one immutable, complete, SHA-verified candidate bundle.

    The original and modified ComfyUI tensors must be converted to [H,W,3]
    NumPy floats and the MASK to [H,W] floats by the node adapter.
    """
    root = _require_clean_root(project_root)
    if not isinstance(run_id, str):
        raise TypeError("run_id must be a string")
    if not _RUN_ID.fullmatch(run_id):
        raise ArtifactExportError("run_id must be 1–64 ASCII letters/digits/_/- without paths")
    if not isinstance(qa_pass, bool):
        raise TypeError("QA boolean must be a boolean PASS")
    if not qa_pass:
        raise ArtifactExportError("Export blocked: QA boolean must be PASS")
    if not isinstance(annotations, AnnotationDocument):
        raise TypeError("SAP annotations are required")
    if not isinstance(selection, SelectionResult):
        raise TypeError("SAP ROI selection is required")
    if annotations.source_format != "seg" or annotations.source_seg is None:
        raise ArtifactExportError("V1 exporter requires source SEG; YOLO needs explicit mapping policy")
    if not isinstance(source_context, dict):
        raise TypeError("source_context must come from SAP Dataset Loader")
    source = source_context
    effect = _json_object(effect_summary, "effect_summary")
    mask_info = _json_object(mask_summary, "mask_summary")
    qa = _json_object(qa_report, "qa_report")
    if qa.get("schema_version") != "sap-qa-v1" or qa.get("status") != "PASS":
        raise ArtifactExportError("Export blocked: QA report must be a genuine PASS schema")
    if effect.get("schema_version") != "sap-glare-v1":
        raise ArtifactExportError("V1 export supports SAP Realistic Glare summaries only")
    if effect.get("kind") not in ("directional", "elliptical"):
        raise ArtifactExportError("Unrecognized effect kind")
    seed = effect.get("seed")
    if type(seed) is not int or seed < 0:
        raise ArtifactExportError("effect seed must be a nonnegative integer")
    for name in ("angle_deg", "radius_scale", "aspect_ratio", "falloff", "intensity",
                 "center_jitter"):
        val = effect.get(name)
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(f"effect {name} must be numeric")
        if not math.isfinite(val):
            raise ArtifactExportError(f"effect {name} must be finite numeric")
    if mask_info.get("mask_kind") not in ("rect", "soft_rect"):
        raise ArtifactExportError("unsupported mask builder kind")
    if mask_info.get("image_size_px") != [annotations.image_width, annotations.image_height]:
        raise ArtifactExportError("mask builder image geometry mismatch")
    if mask_info.get("selected_indices") != sorted(x.instance_index for x in selection.targets):
        raise ArtifactExportError("mask builder selection mismatch")
    orig = _rgb(original, "original")
    mod = _rgb(modified, "modified")
    if not isinstance(mask, np.ndarray):
        raise TypeError("MASK must be a NumPy array")
    if mask.ndim != 2 or mask.dtype.kind != "f":
        raise ArtifactExportError("MASK must be float [H,W]")
    if not np.isfinite(mask).all() or np.any((mask < 0) | (mask > 1)):
        raise ArtifactExportError("MASK must be finite and in [0,1]")
    verdict = validate_artifact(original, modified, mask, annotations, selection)
    if not verdict.passed:
        raise ArtifactExportError(f"Recomputed QA FAIL: {', '.join(verdict.failures)}")
    if qa.get("metrics") != verdict.metrics or qa.get("checks") != verdict.checks:
        raise ArtifactExportError("QA report is stale or does not match the requested artifact")
    if effect.get("image_size_px") != [original.shape[1], original.shape[0]]:
        raise ArtifactExportError("Effect metadata has wrong image size")
    if effect.get("changed_pixels") != verdict.metrics.get("changed_pixels"):
        raise ArtifactExportError("Effect metadata differs from modified IMAGE")
    if effect.get("effect_mask_active_pixels") != verdict.metrics.get("mask_active_pixels"):
        raise ArtifactExportError("Effect metadata differs from MASK")
    source_ref = _validate_provenance(root, source, orig, annotations)

    # Never allow an output outside the project's Git-ignored outputs directory.
    outputs = root / "outputs"
    if outputs.is_symlink():
        raise ArtifactExportError("outputs cannot be a symlink")
    outputs.mkdir(exist_ok=True)
    runs = outputs / "sap_runs"
    if runs.is_symlink():
        raise ArtifactExportError("sap_runs cannot be a symlink")
    runs.mkdir(exist_ok=True)
    destination = runs / run_id
    if destination.exists() or destination.is_symlink():
        raise ArtifactExportError(f"duplicate_output: {run_id}")

    temp: Path | None = None
    try:
        temp = Path(tempfile.mkdtemp(prefix=".pending_", dir=runs))
        image_name = f"{run_id}.png"
        paths = {
            "original": "original.png",
            "image": f"images/{image_name}",
            "annotation": f"annotations/{run_id}.seg",
            "effect_mask": f"masks/{run_id}.png",
            "qa_report": "qa.json",
        }
        for path in paths.values():
            (temp / path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(orig).save(temp / paths["original"], format="PNG")
        Image.fromarray(mod).save(temp / paths["image"], format="PNG")
        # 16-bit lossless PNG representation of the float effect-mask weights.
        mask_16 = np.rint(mask * 65535).astype(np.uint16)
        Image.fromarray(mask_16).save(temp / paths["effect_mask"], format="PNG")
        (temp / paths["annotation"]).write_bytes(to_seg_json(annotations, filename=image_name))
        (temp / paths["qa_report"]).write_text(verdict.to_json() + "\n", encoding="utf-8")

        # Round-trip: re-open encoded artifacts and recover original bbox fields.
        with Image.open(temp / paths["image"]) as stored_image:
            if not np.array_equal(np.asarray(stored_image.convert("RGB")), mod):
                raise ArtifactExportError("failed_export: image PNG round trip differs")
        with Image.open(temp / paths["original"]) as stored_original:
            if not np.array_equal(np.asarray(stored_original.convert("RGB")), orig):
                raise ArtifactExportError("failed_export: original PNG round trip differs")
        with Image.open(temp / paths["effect_mask"]) as stored_mask:
            if not np.array_equal(np.asarray(stored_mask, dtype=np.uint16), mask_16):
                raise ArtifactExportError("failed_export: 16-bit MASK round trip differs")
        exported = parse_seg(
            (temp / paths["annotation"]).read_bytes(), image_width=orig.shape[1],
            image_height=orig.shape[0], image_filename=image_name,
        )
        if exported.objects != annotations.objects:
            raise ArtifactExportError("failed_export: bbox/labels changed during SEG round trip")
        if exported.source_seg != {**annotations.source_seg, "filename": image_name}:
            raise ArtifactExportError("failed_export: SEG source fields changed")
        hashes = {name: _sha256(temp / rel) for name, rel in paths.items()}
        manifest = {
            "schema_version": "sap-artifact-v1",
            "run_id": run_id,
            "source": source_ref,
            "output": paths,
            "output_sha256": hashes,
            "image_size_px": [orig.shape[1], orig.shape[0]],
            "geometry_changed": False,
            "seed": seed,
            "selection": {"mode": selection.mode, "seed": selection.seed},
            "mask_builder": mask_info,
            "targets": [
                {"instance_index": obj.instance_index, "class_name": obj.class_name,
                 "bbox_xyxy_px": list(obj.bbox_xyxy_px)} for obj in selection.targets
            ],
            "effects": [effect],
            "mask_encoding": "PNG-16bit-grayscale (round(weight * 65535))",
            "color_encoding": "8bit-sRGB-RGB-PNG-from-float-[0,1]",
            "qa": {"status": "PASS", "technical_metrics": verdict.metrics,
                   "manual_review_required": True, "training_approved": False},
            "label_policy": "source_seg_geometry_unchanged_no_new_glint_labels",
            "yolo_exported": False,
            "software": {"core": "sapipeline", "exporter_schema": "sap-artifact-v1",
                         "numpy_version": np.__version__, "pillow_version": Image.__version__},
        }
        (temp / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        # Fail if another invocation reserved this ID while we were writing.
        if destination.exists() or destination.is_symlink():
            raise ArtifactExportError(f"duplicate_output: {run_id}")
        temp.rename(destination)  # atomic on the same filesystem; refuses existing dir on Windows
        temp = None
    except Exception:
        if temp is not None:
            shutil.rmtree(temp, ignore_errors=True)
        raise
    return ArtifactExportResult(directory=destination, manifest=manifest)
