"""Private 100-candidate overnight pilot: 2 local glare effects on distinct sources.

Existing Stage-8 four-variant plan is independent and never modified.
Only technical QA PASS candidates become immutable Stage-7 export bundles.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from zipfile import ZIP_STORED, ZipFile

from sapipeline.dataset import (
    DatasetIndexError,
    choose_index_entry,
    load_indexed_pair,
    read_private_index,
)
from sapipeline.selection import SelectionError, select_rois

from .runner import (
    BatchRunError,
    _existing_sources,
    _render,
    _resume_existing,
    _root,
)

_TARGET = 100
_VARIANTS = ("glare_directional", "glare_elliptical")


def _locations(root: Path) -> tuple[Path, Path]:
    parent = root / "reports" / "pilot100"
    plan = parent / "plan.json"
    if any(p.is_symlink() for p in (parent.parent, parent, plan)):
        raise BatchRunError("Unsafe symlink in private pilot report path")
    if parent.exists() and not parent.is_dir():
        raise BatchRunError("Pilot report path is not a directory")
    return parent, plan


def _id(source_sha: str, variant_index: int) -> str:
    return f"p9_{source_sha[:16]}_{variant_index + 1}"


def _seed(base: int, scene_index: int, variant_index: int) -> int:
    seed = base + 2 * scene_index + variant_index
    if seed > 2147483647:
        raise BatchRunError("Pilot seed out of range")
    return seed


def build_pilot_plan(root: Path, *, scenes: int = 60, seed: int = 650001) -> dict[str, Any]:
    """Choose unused, unique, normally oriented annotated learning scenes.

    Reserve 10 extra scenes to absorb individual technical QA failures.
    """
    root = _root(root)
    if type(scenes) is not int or not 50 <= scenes <= 100:
        raise BatchRunError("Pilot scenes must be an integer in [50,100]")
    if type(seed) is not int or not 0 <= seed <= 2147483647 - 2 * scenes:
        raise BatchRunError("Invalid pilot base seed")
    index = read_private_index(root / "data/work/index/dataset_index.json")
    used = _existing_sources(root)
    selected = []
    for sample_index in range(len(index["items"])):
        try:
            entry = choose_index_entry(index, split="learning", sample_index=sample_index)
        except DatasetIndexError as exc:
            if "out of range" in str(exc) or "No unique normally oriented" in str(exc):
                break
            raise
        sha = entry["image_sha256"]
        if sha in used:
            continue
        # Skip service-only scenes, not broken source ZIPs or invalid source labels.
        pair = load_indexed_pair(index, entry, data_dir=root / "data/raw")
        try:
            for j in range(2):
                select_rois(pair.annotations, mode="random_one",
                            seed=_seed(seed, len(selected), j), include_generic_product=True)
        except SelectionError:
            continue
        selected.append(entry)
        used.add(sha)
        if len(selected) == scenes:
            break
    if len(selected) < scenes:
        raise BatchRunError(f"Only {len(selected)} eligible NEW learning scenes; need {scenes}")
    sources = [{"image_member": e["image_member"], "image_sha256": e["image_sha256"]}
               for e in selected]
    runs = [
        {"run_id": _id(e["image_sha256"], j), "source_sha256": e["image_sha256"],
         "variant": variant, "seed": _seed(seed, i, j)}
        for i, e in enumerate(selected) for j, variant in enumerate(_VARIANTS)
    ]
    return {"schema_version": "sap-pilot100-plan-v1", "target": _TARGET,
            "source_split": "learning", "base_seed": seed,
            "sources": sources, "runs": runs, "training_approved": False,
            "manual_review_required": True}


def save_pilot_plan(root: Path, plan: dict[str, Any]) -> Path:
    root = _root(root)
    parent, dst = _locations(root)
    parent.mkdir(parents=True, exist_ok=True)
    with dst.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(plan, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return dst


def _verify(root: Path, doc: dict[str, Any]) -> tuple[dict, list[dict]]:
    if (not isinstance(doc, dict) or set(doc) != {
            "schema_version", "target", "source_split", "base_seed", "sources",
            "runs", "training_approved", "manual_review_required"}
            or doc["schema_version"] != "sap-pilot100-plan-v1"
            or type(doc["target"]) is not int or doc["target"] != _TARGET
            or doc["source_split"] != "learning"
            or doc["training_approved"] is not False
            or doc["manual_review_required"] is not True):
        raise BatchRunError("Invalid pilot plan schema")
    sources, runs, base = doc["sources"], doc["runs"], doc["base_seed"]
    if (not isinstance(sources, list) or not 50 <= len(sources) <= 100
            or not isinstance(runs, list) or len(runs) != 2 * len(sources)
            or type(base) is not int or not 0 <= base <= 2147483647 - len(runs)):
        raise BatchRunError("Unsafe pilot plan scene/run counts or base seed")
    index = read_private_index(root / "data/work/index/dataset_index.json")
    counts: dict[str, int] = {}
    for e in index["items"]:
        sha = e.get("image_sha256")
        if not isinstance(sha, str):
            raise BatchRunError("Invalid indexed source hash")
        counts[sha] = counts.get(sha, 0) + 1
    entries = []
    seen = set()
    for i, source in enumerate(sources):
        if (not isinstance(source, dict)
                or set(source) != {"image_member", "image_sha256"}
                or not isinstance(source["image_sha256"], str)):
            raise BatchRunError("Invalid pilot source")
        sha = source["image_sha256"]
        if sha in seen or counts.get(sha) != 1:
            raise BatchRunError("Pilot source is not globally unique")
        entry = choose_index_entry(index, split="learning", image_member=source["image_member"])
        if (entry.get("image_sha256") != sha or entry.get("exif_orientation") != 1
                or type(entry.get("bbox_count_declared")) is not int
                or entry["bbox_count_declared"] < 1):
            raise BatchRunError("Pilot source changed or is ineligible")
        seen.add(sha)
        entries.append(entry)
        for j, variant in enumerate(_VARIANTS):
            expected = {"run_id": _id(sha, j), "source_sha256": sha,
                        "variant": variant, "seed": _seed(base, i, j)}
            if runs[i * 2 + j] != expected:
                raise BatchRunError("Pilot run/seed/effect was modified")
    return index, entries


def load_pilot(root: Path) -> tuple[dict, dict, list[dict]]:
    root = _root(root)
    _, path = _locations(root)
    if not path.is_file():
        raise BatchRunError("No pilot plan; run `plan` first")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise BatchRunError("Cannot read private pilot plan") from exc
    index, entries = _verify(root, doc)
    return doc, index, entries


def run_pilot(root: Path, *, max_new: int = 100) -> tuple[list[str], int, list[str]]:
    """Create until target=100 or max_new exports; fail-closed on corrupt inputs.

    Only the explicit technical QA FAIL outcome is skipped. Skipped candidates
    are reported and never counted towards the target.
    """
    root = _root(root)
    if type(max_new) is not int or not 1 <= max_new <= 100:
        raise BatchRunError("max_new must be an integer in [1,100]")
    doc, index, entries = load_pilot(root)
    created: list[str] = []
    existing = 0
    qa_failed: list[str] = []
    total_good = 0
    for i, entry in enumerate(entries):
        for j in range(2):
            if total_good >= _TARGET or len(created) >= max_new:
                return created, existing, qa_failed
            row = doc["runs"][2 * i + j]
            if _resume_existing(root, row, entry, None):
                existing += 1
                total_good += 1
                continue
            try:
                _render(root, entry, row, None, index)
            except BatchRunError as exc:
                if str(exc).startswith("Technical QA FAIL for "):
                    qa_failed.append(row["run_id"])
                    print(f"QA REJECTED (no export): {row['run_id']}", flush=True)
                    continue
                raise
            created.append(row["run_id"])
            total_good += 1
            print(f"QA PASS {total_good}/{_TARGET}: {row['run_id']}", flush=True)
    if total_good < _TARGET:
        raise BatchRunError(f"Pilot exhausted: only {total_good}/{_TARGET} QA PASS runs; "
                            f"{len(qa_failed)} rejected (no export)")
    return created, existing, qa_failed


def _valid_bundle(root: Path, row: dict[str, Any], entry: dict[str, Any]) -> tuple[Path, str]:
    if not _resume_existing(root, row, entry, None):
        raise BatchRunError("Cannot package a missing pilot run")
    run = root / "outputs/sap_runs" / row["run_id"]
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("qa", {}).get("training_approved") is not False:
        raise BatchRunError("Incorrect training approval in bundle")
    return run, manifest["source"]["image_sha256"]


def pack_pilot(root: Path) -> tuple[Path, int]:
    """Create a private business QA-preview ZIP from the first 100 passing bundles.

    Exactly 100 modified PNGs + 100 untouched SEG annotations, plus summary.
    Does not contain original photos, manual decisions, or claim YOLO readiness.
    """
    root = _root(root)
    doc, _, entries = load_pilot(root)
    parent, _ = _locations(root)
    dst = parent / "pilot100_qa_preview.zip"
    if dst.exists() or dst.is_symlink():
        raise BatchRunError("Preview ZIP exists; do not overwrite")
    if not parent.is_dir() or parent.is_symlink():
        raise BatchRunError("Unsafe private delivery directory")
    chosen: list[tuple[Path, dict[str, Any], str]] = []
    for i, entry in enumerate(entries):
        for j in range(2):
            row = doc["runs"][2*i + j]
            run = root / "outputs/sap_runs" / row["run_id"]
            if not run.exists():
                continue
            folder, sha = _valid_bundle(root, row, entry)
            chosen.append((folder, row, sha))
            if len(chosen) == _TARGET:
                break
        if len(chosen) == _TARGET:
            break
    if len(chosen) != _TARGET:
        raise BatchRunError(f"Only {len(chosen)}/{_TARGET} QA PASS runs; no preview ZIP created")
    sources = {sha for _, _, sha in chosen}
    if len(sources) < 50:
        raise BatchRunError("Pilot lacks 50 unique source scenes")
    fd, temp_name = tempfile.mkstemp(prefix=".pilot_zip_", suffix=".tmp", dir=parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        with ZipFile(temp, "w", compression=ZIP_STORED, allowZip64=True) as archive:
            info = []
            for folder, row, sha in chosen:
                run_id = row["run_id"]
                manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
                # _resume_existing already validates every declared artifact SHA-256.
                archive.write(folder / manifest["output"]["image"], f"images/{run_id}.png")
                archive.write(folder / manifest["output"]["annotation"], f"annotations/{run_id}.seg")
                info.append({"run_id": run_id, "source_sha256": sha,
                             "effect": row["variant"], "technical_qa": "PASS",
                             "manual_review": "REQUIRED", "training_approved": False})
            summary = {"schema_version": "sap-pilot100-preview-v1", "total": len(info),
                       "unique_source_scenes": len(sources), "training_approved": False,
                       "note": "TECHNICAL QA ONLY; manual review and training suitability NOT approved",
                       "items": info}
            archive.writestr("qa_preview_manifest.json",
                             json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
        if dst.exists() or dst.is_symlink():
            raise BatchRunError("Preview ZIP appeared while packaging")
        temp.rename(dst)
    finally:
        temp.unlink(missing_ok=True)
    return dst, len(chosen)
