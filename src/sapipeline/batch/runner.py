"""Fail-closed, single-process private batch runner using existing SAP contracts.

No ComfyUI server or GPU is required. Source ZIP/index and Stage-7 bundles are
never modified; all generated samples remain unapproved until manual review.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from sapipeline.dataset import (
    DatasetIndexError,
    choose_index_entry,
    load_indexed_pair,
    read_private_index,
)
from sapipeline.effects import apply_realistic_glare
from sapipeline.effects.glass_reflection import apply_glass_reflection
from sapipeline.export import export_artifact
from sapipeline.masks import build_mask
from sapipeline.qa import validate_artifact
from sapipeline.selection import SERVICE_CLASSES, select_rois

_SHA = re.compile(r"[0-9a-f]{64}\Z")
_RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_VARIANTS = ("glare_directional", "glare_elliptical", "glass_soft", "glass_strong")
_PROVENANCE = ("archive", "image_member", "annotation_member", "image_sha256",
               "annotation_sha256", "split")


class BatchRunError(ValueError):
    """Rejected batch plan, unsafe location, stale export, or failed technical QA."""


def _root(project_root: Path) -> Path:
    root = Path(project_root)
    if not root.is_absolute() or root.is_symlink() or not (root / "pyproject.toml").is_file():
        raise BatchRunError("project_root must be an absolute SAP checkout")
    return root.resolve(strict=True)


def _paths(root: Path) -> tuple[Path, Path]:
    reports = root / "reports"
    batch = reports / "batch_run"
    plan = batch / "plan.json"
    if any(path.is_symlink() for path in (reports, batch, plan)):
        raise BatchRunError("Symlinked private plan path")
    if any(path.exists() and not path.is_dir() for path in (reports, batch)):
        raise BatchRunError("Private reports path must be a directory")
    return batch, plan


def _existing_sources(root: Path) -> set[str]:
    runs = root / "outputs/sap_runs"
    if runs.is_symlink():
        raise BatchRunError("Symlinked outputs/sap_runs")
    if not runs.exists():
        return set()
    if not runs.is_dir():
        raise BatchRunError("outputs/sap_runs must be a directory")
    used: set[str] = set()
    for path in sorted(runs.iterdir()):
        if path.is_symlink():
            raise BatchRunError("Unsafe symlink in exports")
        if not path.is_dir():
            continue  # Backup archives/notes are not exported runs.
        manifest = path / "manifest.json"
        if manifest.is_symlink() or not manifest.is_file() or not _RUN.fullmatch(path.name):
            raise BatchRunError("Unsafe or incomplete existing run")
        try:
            doc = json.loads(manifest.read_text(encoding="utf-8"))
            sha = doc["source"]["image_sha256"]
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise BatchRunError("Existing run has unreadable provenance") from exc
        if doc.get("schema_version") != "sap-artifact-v1" or doc.get("run_id") != path.name:
            raise BatchRunError("Existing run has inconsistent manifest")
        if not isinstance(sha, str) or not _SHA.fullmatch(sha):
            raise BatchRunError("Existing run has invalid source SHA")
        used.add(sha)
    return used


def _seed(base: int, scene: int, variant: int) -> int:
    value = base + scene * len(_VARIANTS) + variant
    if value > 2147483647:
        raise BatchRunError("Seed exceeds effect-engine limit")
    return value


def _id(source_sha: str, variant: int) -> str:
    return f"s8_{source_sha[:16]}_{variant + 1}"


def build_plan(project_root: Path, *, scenes: int = 4, seed: int = 42017) -> dict[str, Any]:
    """Plan 4 variants each on NEW unique, normally oriented learning images."""
    root = _root(project_root)
    if type(scenes) is not int or not 1 <= scenes <= 20:
        raise BatchRunError("scenes must be an integer in [1,20]")
    if type(seed) is not int or not 0 <= seed <= 2147483647 - scenes * 4:
        raise BatchRunError("seed must fit the 31-bit effect-engine range")
    index = read_private_index(root / "data/work/index/dataset_index.json")
    used = _existing_sources(root)
    entries: list[dict[str, Any]] = []
    for sample in range(len(index["items"])):
        try:
            entry = choose_index_entry(index, split="learning", sample_index=sample)
        except DatasetIndexError as exc:
            # Only exhaustion is expected; all other errors indicate damaged input.
            if "out of range" in str(exc) or "No unique normally oriented" in str(exc):
                break
            raise
        if entry["image_sha256"] in used:
            continue
        entries.append(entry)
        used.add(entry["image_sha256"])
        if len(entries) == scenes:
            break
    if len(entries) != scenes:
        raise BatchRunError(f"Not enough unused eligible learning scenes: {len(entries)}/{scenes}")
    sources = [{"image_member": e["image_member"],
                "image_sha256": e["image_sha256"], "glass_quad_norm": None}
               for e in entries]
    runs = [{"run_id": _id(e["image_sha256"], j),
             "source_sha256": e["image_sha256"], "variant": variant,
             "seed": _seed(seed, i, j)}
            for i, e in enumerate(entries) for j, variant in enumerate(_VARIANTS)]
    return {"schema_version": "sap-batch-plan-v1", "source_split": "learning",
            "base_seed": seed, "sources": sources, "runs": runs,
            "manual_review_required": True, "training_approved": False}


def save_plan(root: Path, doc: dict[str, Any]) -> Path:
    """Create a private plan once, refusing to replace an existing plan."""
    root = _root(root)
    folder, plan = _paths(root)
    folder.mkdir(parents=True, exist_ok=True)
    with plan.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(doc, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return plan


def _quad(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    return all(isinstance(p, list) and len(p) == 2 and
               all(type(v) in (int, float) and np.isfinite(v) and 0 <= v <= 1 for v in p)
               for p in value)


def _validated(root: Path, doc: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Do not trust edits to provenance, run IDs, seed, or effect types in the plan."""
    if (not isinstance(doc, dict) or doc.get("schema_version") != "sap-batch-plan-v1"
            or doc.get("source_split") != "learning"
            or doc.get("training_approved") is not False
            or doc.get("manual_review_required") is not True):
        raise BatchRunError("Invalid or unsafe plan schema")
    sources, runs, base = doc.get("sources"), doc.get("runs"), doc.get("base_seed")
    if (not isinstance(sources, list) or not 1 <= len(sources) <= 20
            or not isinstance(runs, list) or len(runs) != len(sources) * len(_VARIANTS)
            or type(base) is not int or not 0 <= base <= 2147483647 - len(runs)):
        raise BatchRunError("Invalid plan source/run counts or seed")
    index = read_private_index(root / "data/work/index/dataset_index.json")
    counts: dict[str, int] = {}
    for entry in index["items"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("image_sha256"), str):
            raise BatchRunError("Malformed indexed source")
        sha = entry["image_sha256"]
        counts[sha] = counts.get(sha, 0) + 1
    verified: list[dict[str, Any]] = []
    seen: set[str] = set()
    for scene in sources:
        if (not isinstance(scene, dict) or
                set(scene) != {"image_member", "image_sha256", "glass_quad_norm"}):
            raise BatchRunError("Malformed scene configuration")
        sha = scene["image_sha256"]
        quad = scene["glass_quad_norm"]
        if (not isinstance(sha, str) or not _SHA.fullmatch(sha) or sha in seen
                or counts.get(sha) != 1 or (quad is not None and not _quad(quad))):
            raise BatchRunError("Duplicated, unsafe or invalid scene/quad")
        entry = choose_index_entry(index, split="learning", image_member=scene["image_member"])
        if (entry.get("image_sha256") != sha or entry.get("exif_orientation") != 1
                or type(entry.get("bbox_count_declared")) is not int
                or entry["bbox_count_declared"] < 1):
            raise BatchRunError("Stale or ineligible source in plan")
        seen.add(sha)
        verified.append(entry)
    for i, entry in enumerate(verified):
        for j, variant in enumerate(_VARIANTS):
            row = runs[i * 4 + j]
            expected = {"run_id": _id(entry["image_sha256"], j),
                        "source_sha256": entry["image_sha256"],
                        "variant": variant, "seed": _seed(base, i, j)}
            if type(row) is not dict or row != expected:
                raise BatchRunError("Plan run_id, seed, source or variant was changed")
    return index, verified


def _resume_existing(
    root: Path, run: dict[str, Any], entry: dict[str, Any], quad: Any,
) -> bool:
    runs = root / "outputs/sap_runs"
    if runs.is_symlink():
        raise BatchRunError("Symlinked outputs/sap_runs")
    path = runs / run["run_id"]
    if path.is_symlink():
        raise BatchRunError("Symlinked exported run")
    if not path.exists():
        return False
    if not path.is_dir():
        raise BatchRunError("Existing run ID collides with a file")
    manifest_path = path / "manifest.json"
    if manifest_path.is_symlink():
        raise BatchRunError("Symlinked manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source = manifest["source"]
        effect = manifest["effects"][0]
        outputs = manifest["output"]
        hashes = manifest["output_sha256"]
    except (OSError, ValueError, TypeError, IndexError, KeyError) as exc:
        raise BatchRunError("Existing run is incomplete; never overwrite") from exc
    schema, kind = (("sap-glare-v1", "directional") if run["variant"] == "glare_directional"
                    else ("sap-glare-v1", "elliptical") if run["variant"] == "glare_elliptical"
                    else ("sap-glass-reflection-v1", "projected_strips"))
    if (manifest.get("run_id") != run["run_id"] or
            manifest.get("schema_version") != "sap-artifact-v1" or
            manifest.get("qa", {}).get("status") != "PASS" or
            source.get("image_sha256") != entry["image_sha256"] or
            any(source.get(key) != entry[key] for key in _PROVENANCE) or
            effect.get("schema_version") != schema or effect.get("kind") != kind or
            effect.get("seed") != run["seed"] or
            not isinstance(outputs, dict) or not isinstance(hashes, dict) or
            set(outputs) != {"original", "image", "annotation", "effect_mask", "qa_report"}
            or set(outputs) != set(hashes)):
        raise BatchRunError("Existing run differs from batch plan; never overwrite")
    if schema == "sap-glare-v1":
        expected_effect = {"intensity": 0.65 if kind == "directional" else 0.85,
                           "angle_deg": -25.0, "radius_scale": 0.85,
                           "aspect_ratio": 0.35, "falloff": 1.25,
                           "center_jitter": 0.10}
    else:
        expected_effect = {"quad_norm": quad,
                           "intensity": 0.55 if run["variant"] == "glass_soft" else 0.75,
                           "angle_deg": -12.0, "band_width": 0.060, "band_count": 2,
                           "band_separation": 0.23, "edge_feather": 0.035}
    if any(effect.get(key) != value for key, value in expected_effect.items()):
        raise BatchRunError("Existing effect settings/quad differ from batch plan")
    for key, relative in outputs.items():
        # Exporter uses fixed names; refuse non-canonical or traversing paths.
        expected = {"original": "original.png", "image": f"images/{run['run_id']}.png",
                    "annotation": f"annotations/{run['run_id']}.seg",
                    "effect_mask": f"masks/{run['run_id']}.png", "qa_report": "qa.json"}[key]
        file = path / expected
        if (relative != expected or file.is_symlink() or file.parent.is_symlink()
                or not file.is_file()
                or hashlib.sha256(file.read_bytes()).hexdigest() != hashes[key]):
            raise BatchRunError("Existing exported file/hash differs from manifest")
    return True


def _glass_selection(document: Any, mask: np.ndarray, changed: np.ndarray, seed: int) -> Any:
    candidates = []
    for obj in document.objects:
        if obj.class_name in SERVICE_CLASSES:
            continue
        x1, y1, x2, y2 = obj.bbox_xyxy_px
        if np.any(changed[int(y1):int(np.ceil(y2)), int(x1):int(np.ceil(x2))] &
                  (mask[int(y1):int(np.ceil(y2)), int(x1):int(np.ceil(x2))] > 0)):
            candidates.append(obj.instance_index)
    if not candidates:
        raise BatchRunError("No product ROI intersects the actual glass reflection; adjust quad")
    chosen = sorted(candidates)[seed % len(candidates)]
    return select_rois(document, mode="by_index", instance_index=chosen, seed=seed)


def _render(root: Path, entry: dict[str, Any], row: dict[str, Any], quad: Any,
            index: dict[str, Any]) -> Path:
    pair = load_indexed_pair(index, entry, data_dir=root / "data/raw")
    img = pair.image
    width, height = img.size
    original = np.asarray(img, dtype=np.uint8).astype(np.float32) / 255.0
    seed = row["seed"]
    variant = row["variant"]
    if variant.startswith("glare_"):
        selection = select_rois(pair.annotations, mode="random_one", seed=seed,
                                include_generic_product=True)
    else:
        # Glass reflection masks span the glass plane, not selected bbox masks.
        # The QA selector must nevertheless point to a truly affected merchandise ROI.
        selection = None
    if selection is not None:
        base = build_mask(width, height, selection.targets, kind="soft_rect", pad_px=0,
                          feather_px=16.0, strength=1.0)
        kind = "directional" if variant == "glare_directional" else "elliptical"
        intensity = 0.65 if kind == "directional" else 0.85
        result = apply_realistic_glare(img, base.mask, selection, kind=kind, seed=seed,
                                      intensity=intensity)
        effect_mask = result.effect_mask
        modified = np.asarray(result.image, dtype=np.uint8).astype(np.float32) / 255.0
        effect = {"schema_version": "sap-glare-v1", "kind": kind, "seed": seed,
                  "angle_deg": -25.0, "radius_scale": 0.85, "aspect_ratio": 0.35,
                  "falloff": 1.25, "intensity": intensity, "center_jitter": 0.10,
                  "scope": "selected_bbox_local_not_glass_plane"}
    else:
        if quad is None:
            raise BatchRunError(
                "Glass quad is missing; set glass_quad_norm for this scene in private plan"
            )
        intensity = 0.55 if variant == "glass_soft" else 0.75
        result = apply_glass_reflection(img, quad_norm=quad, seed=seed, intensity=intensity)
        effect_mask = result.effect_mask
        modified = np.asarray(result.image, dtype=np.uint8).astype(np.float32) / 255.0
        changed = np.any(modified != original, axis=2)
        selection = _glass_selection(pair.annotations, effect_mask, changed, seed)
        base = build_mask(width, height, selection.targets, kind="soft_rect", pad_px=0,
                          feather_px=16.0, strength=1.0)
        effect = {"schema_version": "sap-glass-reflection-v1", "kind": "projected_strips",
                  "quad_norm": quad, "seed": seed, "angle_deg": -12.0,
                  "band_width": 0.060, "band_count": 2, "band_separation": 0.23,
                  "edge_feather": 0.035, "intensity": intensity,
                  "scope": "normalized_glass_plane_not_bbox_local"}
    qa = validate_artifact(original, modified, effect_mask, pair.annotations, selection)
    if not qa.passed:
        raise BatchRunError(f"Technical QA FAIL for {row['run_id']}: {', '.join(qa.failures)}")
    effect.update({"image_size_px": [width, height],
                   "changed_pixels": qa.metrics["changed_pixels"],
                   "effect_mask_active_pixels": qa.metrics["mask_active_pixels"],
                   "outside_effect_mask_unchanged": True,
                   "training_approved": False, "manual_review_required": True})
    mask_info = {"mask_kind": "soft_rect", "image_size_px": [width, height],
                 "target_count": len(selection.targets),
                 "selected_indices": sorted(obj.instance_index for obj in selection.targets),
                 "pad_px": base.pad_px, "feather_px": base.feather_px,
                 "strength": base.strength, "mask_type": "bbox-effect-mask-not-segmentation"}
    exported = export_artifact(
        project_root=root, run_id=row["run_id"], original=original, modified=modified,
        mask=effect_mask, annotations=pair.annotations, selection=selection,
        qa_pass=qa.passed, qa_report=qa.to_json(),
        source_context={key: entry[key] for key in _PROVENANCE},
        effect_summary=json.dumps(effect, allow_nan=False),
        mask_summary=json.dumps(mask_info, allow_nan=False),
    )
    return exported.directory


def execute_plan(project_root: Path, *, max_new: int = 1) -> tuple[list[str], list[str]]:
    """Resume a private plan serially; never overwrite existing Stage-7 artifacts.

    Returns IDs (created, already_present). Stops on the first failed/unsafe run.
    """
    root = _root(project_root)
    if type(max_new) is not int or not 1 <= max_new <= 20:
        raise BatchRunError("max_new must be an integer in [1,20]")
    _, plan_file = _paths(root)
    if not plan_file.is_file():
        raise BatchRunError("No private plan; run `plan` first")
    try:
        doc = json.loads(plan_file.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise BatchRunError("Invalid JSON batch plan") from exc
    index, entries = _validated(root, doc)
    created: list[str] = []
    skipped: list[str] = []
    for i, entry in enumerate(entries):
        for j in range(4):
            row = doc["runs"][i * 4 + j]
            if _resume_existing(root, row, entry, doc["sources"][i]["glass_quad_norm"]):
                skipped.append(row["run_id"])
                continue
            directory = _render(root, entry, row, doc["sources"][i]["glass_quad_norm"], index)
            created.append(directory.name)
            if len(created) >= max_new:
                return created, skipped
    return created, skipped
