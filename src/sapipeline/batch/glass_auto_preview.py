"""Automatic glass-like previews driven by product bbox clusters.

This module derives a broad reflection region from several nearby product ROI.
It never edits source ZIP files, SEG labels, or previous pilot outputs.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from sapipeline.dataset import choose_index_entry, load_indexed_pair, read_private_index
from sapipeline.effects.glass_reflection import _quad, apply_glass_reflection
from sapipeline.selection import SERVICE_CLASSES

from .glass_pilot import load_glass_plan
from .runner import BatchRunError, _root

_VARIANTS: tuple[tuple[str, dict[str, float | int]], ...] = (
    (
        'glass_diffuse',
        {
            'angle_deg': -10.0,
            'band_width': 0.150,
            'band_count': 1,
            'band_separation': 0.18,
            'edge_feather': 0.045,
            'intensity': 0.44,
        },
    ),
    (
        'glass_fluorescent',
        {
            'angle_deg': -12.0,
            'band_width': 0.050,
            'band_count': 3,
            'band_separation': 0.16,
            'edge_feather': 0.040,
            'intensity': 0.72,
        },
    ),
)


def _preview_root(root: Path) -> Path:
    base = root / 'reports' / 'glass_pilot100' / 'auto_previews'
    if any(item.is_symlink() for item in (root / 'reports', root / 'reports' / 'glass_pilot100', base)):
        raise BatchRunError('Unsafe symlink in glass auto preview path')
    return base


def _product_boxes(pair: Any) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    for obj in pair.annotations.objects:
        if obj.class_name in SERVICE_CLASSES:
            continue
        x1, y1, x2, y2 = [int(v) for v in obj.bbox_xyxy_px]
        if x2 <= x1 or y2 <= y1:
            continue
        area = (x2 - x1) * (y2 - y1)
        if area <= 0:
            continue
        boxes.append({
            'class_name': obj.class_name,
            'bbox_xyxy_px': [x1, y1, x2, y2],
            'cx': (x1 + x2) / 2.0,
            'cy': (y1 + y2) / 2.0,
            'area': area,
        })
    if not boxes:
        raise BatchRunError('No eligible product ROI found for automatic glass preview')
    return boxes


def _select_cluster(
    boxes: list[dict[str, Any]], *, image_size: tuple[int, int], seed: int,
) -> list[dict[str, Any]]:
    """Favor a broad upper/middle horizontal product group, not footer ROI.

    This is a *product bbox* heuristic, never physical glass segmentation.
    Deterministic ties are resolved from coordinates, classes and seed.
    """
    width, height = image_size
    pool = boxes
    for upper_limit in (0.58, 0.68, 0.78):
        eligible = [
            box for box in boxes
            if box['cy'] <= upper_limit * height
            and box['bbox_xyxy_px'][3] <= (upper_limit + 0.10) * height
        ]
        if len(eligible) >= 3:
            pool = eligible
            break

    row_tolerance = max(36.0, 0.12 * height)
    group_size = min(6, len(pool))
    minimum_size = min(3, group_size)
    best: tuple[float, ...] | None = None
    chosen: list[dict[str, Any]] = []
    anchors = sorted(pool, key=lambda box: (box['cy'], box['cx'], box['class_name']))
    for i, anchor in enumerate(anchors):
        row = sorted(
            (box for box in pool if abs(box['cy'] - anchor['cy']) <= row_tolerance),
            key=lambda box: (box['cx'], box['cy'], box['class_name']),
        )
        for size in range(minimum_size, min(group_size, len(row)) + 1):
            for left_index in range(len(row) - size + 1):
                group = row[left_index : left_index + size]
                left = min(box['bbox_xyxy_px'][0] for box in group)
                right = max(box['bbox_xyxy_px'][2] for box in group)
                mean_y = sum(box['cy'] for box in group) / size
                y_spread = max(box['cy'] for box in group) - min(box['cy'] for box in group)
                width_fraction = (right - left) / width
                area_fraction = sum(box['area'] for box in group) / (width * height)
                center_bias = abs((left + right) / (2 * width) - 0.5)
                jitter = ((seed + i * 31 + left_index * 17 + size * 7) % 97) / 10000.0
                score = (
                    width_fraction * 3.2 + size * 0.45 + area_fraction * 1.2
                    - (mean_y / height) * 1.5 - center_bias * 0.35
                    - (y_spread / height) * 0.25 - jitter
                )
                priority = (score, width_fraction, -mean_y, -y_spread, float(size))
                if best is None or priority > best:
                    best, chosen = priority, group
    if chosen:
        return chosen
    # No group formed: keep behavior explicit for sparsely annotated scenes.
    return sorted(pool, key=lambda box: (box['cy'], box['cx'], box['class_name']))[:group_size]


def _quad_from_cluster(cluster: list[dict[str, Any]], *, image_size: tuple[int, int]) -> list[list[float]]:
    width, height = image_size
    x1 = min(row['bbox_xyxy_px'][0] for row in cluster)
    y1 = min(row['bbox_xyxy_px'][1] for row in cluster)
    x2 = max(row['bbox_xyxy_px'][2] for row in cluster)
    y2 = max(row['bbox_xyxy_px'][3] for row in cluster)
    span_x = max(1, x2 - x1)
    span_y = max(1, y2 - y1)
    pad_left = max(18.0, 0.22 * span_x)
    pad_right = max(18.0, 0.22 * span_x)
    pad_top = min(max(24.0, 0.40 * span_y), 0.10 * height)
    pad_bottom = max(10.0, 0.14 * span_y)
    left = max(0.0, x1 - pad_left)
    right = min(float(width - 1), x2 + pad_right)
    top = max(0.0, y1 - pad_top)
    bottom = min(float(height - 1), float(height * 0.76), y2 + pad_bottom)
    quad = [
        [left / width, top / height],
        [right / width, max(0.0, top + 0.02 * span_y) / height],
        [min(float(width - 1), right - 0.01 * span_x) / width, bottom / height],
        [max(0.0, left + 0.01 * span_x) / width, min(float(height - 1), bottom + 0.01 * span_y) / height],
    ]
    _quad(quad)
    return quad


def _roi_mask(image_size: tuple[int, int], cluster: list[dict[str, Any]]) -> np.ndarray:
    width, height = image_size
    mask = np.zeros((height, width), dtype=np.uint8)
    for row in cluster:
        x1, y1, x2, y2 = row['bbox_xyxy_px']
        mask[y1:y2, x1:x2] = 255
    return mask


def _mask_visual(mask: np.ndarray) -> Image.Image:
    if mask.dtype != np.uint8:
        arr = np.clip(mask * 255.0, 0.0, 255.0).astype(np.uint8)
    else:
        arr = mask
    return Image.fromarray(arr, mode='L')


def _quad_overlay(image: Image.Image, quad_norm: list[list[float]]) -> Image.Image:
    draw = image.convert('RGB').copy()
    painter = ImageDraw.Draw(draw)
    w, h = draw.size
    pts = [(round(x * w), round(y * h)) for x, y in quad_norm]
    painter.polygon(pts, outline=(255, 255, 0), width=3)
    return draw


def _report_name(scene: int, sha: str) -> str:
    # Keep the original previews immutable; v2 creates a separate private folder.
    return f'scene_{scene:02d}_{sha[:16]}_v2'


def _write_atomic(folder: Path, build) -> None:
    if folder.exists():
        raise BatchRunError(f'Preview folder already exists: {folder}')
    folder.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='sap-glass-auto-', dir=folder.parent) as tmp:
        temp = Path(tmp)
        build(temp)
        temp.replace(folder)


def generate_auto_glass_previews(project_root: Path, *, start: int = 0, count: int = 5) -> list[tuple[Path, dict[str, Any]]]:
    root = _root(project_root)
    if type(start) is not int or type(count) is not int or count < 1:
        raise BatchRunError('start/count must be positive integers')
    plan, _ = load_glass_plan(root)
    sources = plan['sources']
    if start < 0 or start + count > len(sources):
        raise BatchRunError('Requested scenes must stay within the glass plan')
    index = read_private_index(root / 'data' / 'work' / 'index' / 'dataset_index.json')
    output_root = _preview_root(root)
    results_all: list[tuple[Path, dict[str, Any]]] = []
    for scene in range(start, start + count):
        source = sources[scene]
        entry = choose_index_entry(index, split='learning', image_member=source['image_member'])
        pair = load_indexed_pair(index, entry, data_dir=root / 'data' / 'raw')
        boxes = _product_boxes(pair)
        cluster = _select_cluster(boxes, image_size=pair.image.size, seed=plan['base_seed'] + scene)
        quad = _quad_from_cluster(cluster, image_size=pair.image.size)
        roi_mask = _roi_mask(pair.image.size, cluster)
        variants: list[dict[str, Any]] = []
        preview_rows: list[tuple[str, Image.Image, np.ndarray]] = []
        for variant_index, (name, params) in enumerate(_VARIANTS):
            result = apply_glass_reflection(
                pair.image,
                quad_norm=quad,
                seed=plan['base_seed'] + scene * 10 + variant_index,
                intensity=float(params['intensity']),
                angle_deg=float(params['angle_deg']),
                band_width=float(params['band_width']),
                band_count=int(params['band_count']),
                band_separation=float(params['band_separation']),
                edge_feather=float(params['edge_feather']),
            )
            changed = np.any(np.asarray(result.image) != np.asarray(pair.image))
            intersect = bool(np.any((result.effect_mask > 0.05) & (roi_mask > 0)))
            qa = 'PASS' if changed and intersect else 'FAIL'
            variants.append({
                'variant': name,
                'seed': plan['base_seed'] + scene * 10 + variant_index,
                'effect': dict(params),
                'technical_qa': {'status': qa, 'changed_pixels': bool(changed), 'intersects_roi': bool(intersect)},
            })
            preview_rows.append((name, result.image, result.effect_mask))
        report = {
            'schema_version': 'sap-glass-auto-preview-v1',
            'scene_index': scene,
            'source_sha256': source['image_sha256'],
            'image_member': source['image_member'],
            'manual_review_required': True,
            'training_approved': False,
            'stage7_export': False,
            'selection_mode': 'bbox_upper_row_v2',
            'selected_objects': [
                {'class_name': row['class_name'], 'bbox_xyxy_px': row['bbox_xyxy_px']} for row in cluster
            ],
            'auto_quad_norm': quad,
            'variants': variants,
        }
        folder = output_root / _report_name(scene, source['image_sha256'])

        def build(temp: Path, pair=pair, quad=quad, roi_mask=roi_mask, preview_rows=tuple(preview_rows), report=report):
            pair.image.save(temp / 'original.png')
            _quad_overlay(pair.image, quad).save(temp / 'glass_region.png')
            _mask_visual(roi_mask).save(temp / 'roi_mask_visual.png')
            for name, image, mask in preview_rows:
                image.save(temp / f'{name}.png')
                _mask_visual(mask).save(temp / f'{name}_mask_visual.png')
            (temp / 'preview_report.json').write_text(
                json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n',
                encoding='utf-8',
            )

        _write_atomic(folder, build)
        results_all.append((folder, report))
    return results_all
