"""Resume and package 100 private QA preview images for bbox-driven glass glare.

The glass pilot plan is used to reserve source scenes, *not* to declare that
bbox-derived previews have the homography-profile seeds in its runs. Candidate
IDs, parameters and seeds are taken from the actual auto preview renderer.
Nothing here creates Stage-7 bundles or approves a sample for training.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from zipfile import ZIP_STORED, ZipFile

import numpy as np
from PIL import Image

from sapipeline.annotations import to_seg_json
from sapipeline.dataset import choose_index_entry, load_indexed_pair, read_private_index
from sapipeline.effects.glass_reflection import GlassReflectionError, apply_glass_reflection

from .glass_auto_preview import (
    _VARIANTS,
    _mask_visual,
    _preview_root,
    _product_boxes,
    _quad_from_cluster,
    _quad_overlay,
    _report_name,
    _roi_mask,
    _select_cluster,
    generate_auto_glass_previews,
)
from .glass_pilot import load_glass_plan
from .runner import BatchRunError, _existing_sources, _root

_TARGET = 100
_EXPECTED_FILES = {
    'original.png', 'glass_region.png', 'roi_mask_visual.png',
    'glass_diffuse.png', 'glass_fluorescent.png',
    'glass_diffuse_mask_visual.png', 'glass_fluorescent_mask_visual.png',
    'preview_report.json',
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _prepare(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    plan, _ = load_glass_plan(root)
    used = _existing_sources(root)
    planned = {s['image_sha256'] for s in plan['sources']}
    if used & planned:
        raise BatchRunError('Glass plan overlaps existing exported sources; stop before training/export')
    index = read_private_index(root / 'data/work/index/dataset_index.json')
    return plan, index


def _load_png(folder: Path, name: str, mode: str) -> np.ndarray:
    path = folder / name
    if path.is_symlink() or not path.is_file():
        raise BatchRunError(f'Incomplete or symlinked glass preview: {name}')
    try:
        with Image.open(path) as opened:
            if opened.format != 'PNG' or opened.mode != mode:
                raise BatchRunError(f'Wrong glass preview image encoding: {name}')
            return np.asarray(opened).copy()
    except OSError as exc:
        raise BatchRunError(f'Unreadable glass preview: {name}') from exc


def _verified_scene(
    root: Path, plan: dict[str, Any], index: dict[str, Any], scene: int,
) -> tuple[Path, list[dict[str, Any]], Any]:
    source = plan['sources'][scene]
    folder = _preview_root(root) / _report_name(scene, source['image_sha256'])
    if folder.is_symlink() or not folder.is_dir():
        raise BatchRunError(f'Glass preview scene missing or unsafe: {scene}')
    if any(item.is_symlink() for item in folder.iterdir()):
        raise BatchRunError(f'Symlinked glass preview file for scene {scene}')
    if {p.name for p in folder.iterdir()} != _EXPECTED_FILES:
        raise BatchRunError(f'Glass preview file set differs for scene {scene}')
    try:
        report = json.loads((folder / 'preview_report.json').read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError) as exc:
        raise BatchRunError(f'Unreadable glass preview report for scene {scene}') from exc
    entry = choose_index_entry(index, split='learning', image_member=source['image_member'])
    if entry['image_sha256'] != source['image_sha256']:
        raise BatchRunError(f'Glass preview source hash changed for scene {scene}')
    pair = load_indexed_pair(index, entry, data_dir=root / 'data/raw')
    if (not isinstance(report, dict) or report.get('schema_version') != 'sap-glass-auto-preview-v1'
            or report.get('scene_index') != scene
            or report.get('source_sha256') != source['image_sha256']
            or report.get('image_member') != source['image_member']
            or report.get('manual_review_required') is not True
            or report.get('training_approved') is not False
            or report.get('stage7_export') is not False
            or report.get('selection_mode') != 'bbox_upper_row_v2'):
        raise BatchRunError(f'Glass preview provenance changed for scene {scene}')
    boxes = _product_boxes(pair)
    cluster = _select_cluster(boxes, image_size=pair.image.size, seed=plan['base_seed'] + scene)
    quad = _quad_from_cluster(cluster, image_size=pair.image.size)
    expected_objects = [
        {'class_name': b['class_name'], 'bbox_xyxy_px': b['bbox_xyxy_px']} for b in cluster
    ]
    if report.get('auto_quad_norm') != quad or report.get('selected_objects') != expected_objects:
        raise BatchRunError(f'Glass preview bbox geometry changed for scene {scene}')
    rgb = np.asarray(pair.image)
    roi = _roi_mask(pair.image.size, cluster)
    if not np.array_equal(_load_png(folder, 'original.png', 'RGB'), rgb):
        raise BatchRunError(f'Glass preview original differs for scene {scene}')
    if not np.array_equal(_load_png(folder, 'glass_region.png', 'RGB'), np.asarray(_quad_overlay(pair.image, quad))):
        raise BatchRunError(f'Glass preview region differs for scene {scene}')
    if not np.array_equal(_load_png(folder, 'roi_mask_visual.png', 'L'), roi):
        raise BatchRunError(f'Glass preview selected ROI mask differs for scene {scene}')
    if not isinstance(report.get('variants'), list) or len(report['variants']) != len(_VARIANTS):
        raise BatchRunError(f'Invalid glass preview variant count for scene {scene}')
    verified: list[dict[str, Any]] = []
    for j, (name, params) in enumerate(_VARIANTS):
        seed = plan['base_seed'] + scene * 10 + j
        result = apply_glass_reflection(
            pair.image, quad_norm=quad, seed=seed,
            intensity=float(params['intensity']), angle_deg=float(params['angle_deg']),
            band_width=float(params['band_width']), band_count=int(params['band_count']),
            band_separation=float(params['band_separation']), edge_feather=float(params['edge_feather']),
        )
        modified = np.asarray(result.image)
        mask = np.asarray(_mask_visual(result.effect_mask))
        changed = bool(np.any(modified != rgb))
        intersect = bool(np.any((result.effect_mask > 0.05) & (roi > 0)))
        status = 'PASS' if changed and intersect else 'FAIL'
        row = report['variants'][j]
        if row != {
            'variant': name, 'seed': seed, 'effect': dict(params),
            'technical_qa': {
                'status': status, 'changed_pixels': changed, 'intersects_roi': intersect,
            },
        }:
            raise BatchRunError(f'Glass preview technical QA/parameters changed for scene {scene}/{name}')
        if not np.array_equal(_load_png(folder, f'{name}.png', 'RGB'), modified):
            raise BatchRunError(f'Glass preview pixels differ for scene {scene}/{name}')
        if not np.array_equal(_load_png(folder, f'{name}_mask_visual.png', 'L'), mask):
            raise BatchRunError(f'Glass preview mask differs for scene {scene}/{name}')
        verified.append(row)
    return folder, verified, pair


def run_batch(root: Path, *, max_new_scenes: int = 60) -> dict[str, int]:
    """Process up to 100 PASS candidates; safely reuse *verified* v2 previews."""
    root = _root(root)
    if type(max_new_scenes) is not int or not 1 <= max_new_scenes <= 60:
        raise BatchRunError('max_new_scenes must be an integer in [1,60]')
    plan, index = _prepare(root)
    existing = created = rejected = accepted = 0
    for scene, source in enumerate(plan['sources']):
        if accepted >= _TARGET:
            break
        folder = _preview_root(root) / _report_name(scene, source['image_sha256'])
        if folder.exists() or folder.is_symlink():
            _, rows, _ = _verified_scene(root, plan, index, scene)
            existing += 1
        else:
            if created >= max_new_scenes:
                break
            # Expected geometry limitation: reject this scene and use a reserve.
            # Source/index/IO/corruption failures are not silently ignored.
            try:
                generate_auto_glass_previews(root, start=scene, count=1)
            except GlassReflectionError as exc:
                print(f'QA REJECTED scene={scene}: glass geometry {exc}', flush=True)
                rejected += 2
                continue
            _, rows, _ = _verified_scene(root, plan, index, scene)
            created += 1
        passed = sum(row['technical_qa']['status'] == 'PASS' for row in rows)
        rejected += len(rows) - passed
        accepted += min(passed, _TARGET - accepted)
        print(f'GLASS QA {accepted}/{_TARGET}: scene={scene} PASS={passed} '
              f'CREATED_SCENES={created} REUSED_SCENES={existing}', flush=True)
    return {
        'qa_pass': accepted, 'target': _TARGET, 'created_scenes': created,
        'reused_scenes': existing, 'qa_rejected': rejected,
    }


def pack_batch(root: Path) -> tuple[Path, int]:
    """Package 100 QA-PASS preview PNG/SEG pairs without training approval."""
    root = _root(root)
    plan, index = _prepare(root)
    parent = root / 'reports/glass_pilot100'
    dst = parent / 'glass_bbox100_qa_preview.zip'
    if dst.exists() or dst.is_symlink() or parent.is_symlink():
        raise BatchRunError('Glass QA preview ZIP already exists or its path is unsafe')
    selected: list[tuple[Path, dict[str, Any], Any, str]] = []
    for scene, source in enumerate(plan['sources']):
        if len(selected) >= _TARGET:
            break
        folder = _preview_root(root) / _report_name(scene, source['image_sha256'])
        if not folder.exists() and not folder.is_symlink():
            continue
        checked, rows, pair = _verified_scene(root, plan, index, scene)
        for j, row in enumerate(rows):
            if row['technical_qa']['status'] == 'PASS':
                run_id = f"pg2a_{source['image_sha256'][:16]}_{j+1}"
                selected.append((checked, row, pair, run_id))
            if len(selected) == _TARGET:
                break
    if len(selected) != _TARGET:
        raise BatchRunError(f'Only {len(selected)}/{_TARGET} verified glass previews; no ZIP created')
    sources = {item[2].entry['image_sha256'] for item in selected}
    if len(sources) < 50:
        raise BatchRunError('Fewer than 50 unique glass pilot source scenes')
    fd, tmp_name = tempfile.mkstemp(dir=parent, prefix='.glass_bbox100_', suffix='.tmp')
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with ZipFile(tmp, 'w', compression=ZIP_STORED, allowZip64=True) as archive:
            manifest_rows: list[dict[str, Any]] = []
            for folder, variant, pair, run_id in selected:
                name = variant['variant']
                img_path = folder / f'{name}.png'
                seg = to_seg_json(pair.annotations, filename=f'{run_id}.png')
                image_bytes = img_path.read_bytes()
                archive.writestr(f'images/{run_id}.png', image_bytes)
                archive.writestr(f'annotations/{run_id}.seg', seg)
                # 8-bit visualization, NOT an instance ground truth mask.
                mask_bytes = (folder / f'{name}_mask_visual.png').read_bytes()
                archive.writestr(f'masks_visual/{run_id}.png', mask_bytes)
                manifest_rows.append({
                    'run_id': run_id, 'source_sha256': pair.entry['image_sha256'],
                    'variant': name, 'seed': variant['seed'], 'effect': variant['effect'],
                    'technical_qa': 'PASS', 'manual_review': 'REQUIRED',
                    'training_approved': False, 'image_sha256': _sha(image_bytes),
                    'annotation_sha256': _sha(seg), 'mask_visual_sha256': _sha(mask_bytes),
                })
            manifest = {
                'schema_version': 'sap-glass-bbox100-qa-preview-v1',
                'total': len(manifest_rows), 'unique_source_scenes': len(sources),
                'source_split': 'learning', 'manual_review_required': True,
                'training_approved': False, 'stage7_export': False,
                'note': 'TECHNICAL QA ONLY. Bbox-derived reflection; visual/manual review required.',
                'items': manifest_rows,
            }
            archive.writestr('qa_preview_manifest.json', json.dumps(
                manifest, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False,
            ) + '\n')
        if dst.exists() or dst.is_symlink():
            raise BatchRunError('Glass QA preview ZIP appeared during packaging; no overwrite')
        tmp.rename(dst)
    finally:
        tmp.unlink(missing_ok=True)
    return dst, len(selected)
