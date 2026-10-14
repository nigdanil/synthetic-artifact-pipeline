"""Plan-only second 100-image pilot with glass-plane reflections.

This module never generates/exports images, changes prior runs, or approves data.
Glass coordinates stay unset until manually established for each source scene.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from sapipeline.dataset import (
    DatasetIndexError,
    choose_index_entry,
    load_indexed_pair,
    read_private_index,
)
from sapipeline.effects.glass_reflection import GlassReflectionError, _quad
from sapipeline.selection import SERVICE_CLASSES

from .runner import BatchRunError, _existing_sources, _root

_TARGET = 100
_VARIANTS = ("glass_diffuse", "glass_fluorescent")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_PROFILE = {
    "glass_diffuse": {
        "angle_deg": -12.0,
        "band_width": 0.13,
        "band_count": 1,
        "band_separation": 0.23,
        "edge_feather": 0.035,
        "intensity": 0.50,
    },
    "glass_fluorescent": {
        "angle_deg": -12.0,
        "band_width": 0.050,
        "band_count": 3,
        "band_separation": 0.20,
        "edge_feather": 0.035,
        "intensity": 0.75,
    },
}


def _paths(root: Path) -> tuple[Path, Path]:
    reports = root / "reports"
    folder = reports / "glass_pilot100"
    path = folder / "plan.json"
    if any(item.is_symlink() for item in (reports, folder, path)):
        raise BatchRunError("Unsafe symlink in glass pilot plan location")
    if any(item.exists() and not item.is_dir() for item in (reports, folder)):
        raise BatchRunError("Invalid private glass pilot reports directory")
    return folder, path


def _seed(base: int, scene_index: int, variant_index: int) -> int:
    return base + len(_VARIANTS) * scene_index + variant_index


def _row(sha: str, scene_index: int, variant_index: int, base: int) -> dict[str, Any]:
    variant = _VARIANTS[variant_index]
    return {
        "run_id": f"pg2_{sha[:16]}_{variant_index + 1}",
        "source_sha256": sha,
        "variant": variant,
        "seed": _seed(base, scene_index, variant_index),
        "effect": dict(_PROFILE[variant]),
    }


def build_glass_plan(
    project_root: Path, *, scenes: int = 60, seed: int = 850001,
) -> dict[str, Any]:
    """Reserve two glass-reflection candidate variants per unused learning photo.

    The default 60 scenes supply 120 possible candidates for 100 successful
    outputs. They are *not* 100 reviewed samples, nor 100 glass masks.
    """
    root = _root(project_root)
    if type(scenes) is not int or not 50 <= scenes <= 100:
        raise BatchRunError("glass pilot scenes must be an integer in [50,100]")
    if type(seed) is not int or not 0 <= seed <= 2147483647 - scenes * 2:
        raise BatchRunError("glass pilot base seed is invalid")
    index = read_private_index(root / "data/work/index/dataset_index.json")
    used = _existing_sources(root)
    # Do not silently select any photograph that appears in multiple splits
    # or has an alternative annotation under another name.
    counts: dict[str, int] = {}
    for item in index["items"]:
        sha = item.get("image_sha256")
        if not isinstance(sha, str) or not _SHA.fullmatch(sha):
            raise BatchRunError("Invalid dataset index source SHA-256")
        counts[sha] = counts.get(sha, 0) + 1
    selected: list[dict[str, Any]] = []
    for sample_index in range(len(index["items"])):
        try:
            entry = choose_index_entry(index, split="learning", sample_index=sample_index)
        except DatasetIndexError as exc:
            if "out of range" in str(exc) or "No unique normally oriented" in str(exc):
                break
            raise
        sha = entry["image_sha256"]
        if sha in used or counts.get(sha) != 1 or entry.get("exif_orientation") != 1:
            continue
        pair = load_indexed_pair(index, entry, data_dir=root / "data/raw")
        if not any(obj.class_name not in SERVICE_CLASSES for obj in pair.annotations.objects):
            continue
        selected.append(entry)
        used.add(sha)
        if len(selected) >= scenes:
            break
    if len(selected) != scenes:
        raise BatchRunError(
            f"Only {len(selected)} eligible unused learning sources; requested {scenes}. "
            "Do not fall back to verification or reused images."
        )
    sources = [
        {"image_member": e["image_member"], "image_sha256": e["image_sha256"],
         "glass_quad_norm": None}
        for e in selected
    ]
    runs = [
        _row(entry["image_sha256"], i, j, seed)
        for i, entry in enumerate(selected)
        for j in range(len(_VARIANTS))
    ]
    return {
        "schema_version": "sap-glass-pilot100-plan-v1",
        "target": _TARGET,
        "source_split": "learning",
        "base_seed": seed,
        "sources": sources,
        "runs": runs,
        "manual_review_required": True,
        "training_approved": False,
    }


def save_glass_plan(project_root: Path, plan: dict[str, Any]) -> Path:
    """Create the private plan exactly once, without touching prior pilots."""
    root = _root(project_root)
    folder, path = _paths(root)
    # Validate before any filesystem side effects.
    validate_glass_plan(root, plan)
    folder.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(plan, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return path


def validate_glass_plan(project_root: Path, doc: dict[str, Any]) -> tuple[int, int]:
    """Fail closed on edits except source-specific glass quadrilaterals.

    Returns (source_count, sources_still_missing_glass_geometry).
    This function checks the plan, *not* the realism of its manually set quads.
    """
    root = _root(project_root)
    required = {
        "schema_version", "target", "source_split", "base_seed", "sources", "runs",
        "manual_review_required", "training_approved",
    }
    if (not isinstance(doc, dict) or set(doc) != required
            or doc["schema_version"] != "sap-glass-pilot100-plan-v1"
            or type(doc["target"]) is not int or doc["target"] != _TARGET
            or doc["source_split"] != "learning"
            or doc["manual_review_required"] is not True
            or doc["training_approved"] is not False):
        raise BatchRunError("Invalid glass pilot plan schema")
    sources, runs, base = doc["sources"], doc["runs"], doc["base_seed"]
    if (not isinstance(sources, list) or not 50 <= len(sources) <= 100
            or not isinstance(runs, list) or len(runs) != 2 * len(sources)
            or type(base) is not int or not 0 <= base <= 2147483647 - len(runs)):
        raise BatchRunError("Invalid glass pilot source count, run count or seed")
    index = read_private_index(root / "data/work/index/dataset_index.json")
    counts: dict[str, int] = {}
    for item in index["items"]:
        sha = item.get("image_sha256")
        if not isinstance(sha, str) or not _SHA.fullmatch(sha):
            raise BatchRunError("Invalid dataset index source SHA-256")
        counts[sha] = counts.get(sha, 0) + 1
    seen: set[str] = set()
    missing = 0
    for i, source in enumerate(sources):
        if (not isinstance(source, dict)
                or set(source) != {"image_member", "image_sha256", "glass_quad_norm"}
                or not isinstance(source["image_member"], str)
                or not isinstance(source["image_sha256"], str)
                or not _SHA.fullmatch(source["image_sha256"])):
            raise BatchRunError("Invalid glass pilot source entry")
        sha = source["image_sha256"]
        if sha in seen or counts.get(sha) != 1:
            raise BatchRunError("Glass pilot source is duplicated or split-overlapping")
        seen.add(sha)
        entry = choose_index_entry(index, split="learning", image_member=source["image_member"])
        if (entry.get("image_sha256") != sha or entry.get("exif_orientation") != 1
                or type(entry.get("bbox_count_declared")) is not int
                or entry["bbox_count_declared"] < 1):
            raise BatchRunError("Glass pilot source changed or is ineligible")
        quad = source["glass_quad_norm"]
        if quad is None:
            missing += 1
        else:
            try:
                _quad(quad)
            except (GlassReflectionError, TypeError, ValueError) as exc:
                raise BatchRunError("Invalid glass_quad_norm; expected convex TL/TR/BR/BL") from exc
        for j in range(2):
            expected = _row(sha, i, j, base)
            if type(runs[2 * i + j]) is not dict or runs[2 * i + j] != expected:
                raise BatchRunError("Glass pilot run ID, effect settings or seed was modified")
    return len(sources), missing


def load_glass_plan(project_root: Path) -> tuple[dict[str, Any], int]:
    root = _root(project_root)
    _, path = _paths(root)
    if not path.is_file():
        raise BatchRunError("Glass pilot plan is missing; run `plan` first")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        raise BatchRunError("Cannot read private glass pilot plan") from exc
    _, missing = validate_glass_plan(root, doc)
    return doc, missing
