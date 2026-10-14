from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest
from PIL import Image

from sapipeline.batch.glass_auto_preview import generate_auto_glass_previews
from sapipeline.batch.glass_pilot import build_glass_plan, save_glass_plan
from sapipeline.batch.runner import BatchRunError
from sapipeline.dataset import build_index


def _project(root: Path) -> Path:
    (root / 'pyproject.toml').write_text('[project]\nname="fake-sap"\n', encoding='utf-8')
    raw = root / 'data' / 'raw'
    raw.mkdir(parents=True)
    archive = raw / 'for_learning.zip'
    with ZipFile(archive, 'w') as zipped:
        for n in range(50):
            data = BytesIO()
            image = Image.new('RGB', (160, 120), (35 + n, 90, 120))
            from PIL import ImageDraw
            draw = ImageDraw.Draw(image)
            draw.text((8, 8), str(n), fill=(255, 255, 255))
            draw.rectangle((10 + (n % 7), 95, 30 + (n % 7), 110), fill=(10 * (n % 20), 20, 30))
            image.save(data, 'JPEG')
            name = f'scene_{n:03d}'
            zipped.writestr(f'img/{name}.jpg', data.getvalue())
            anns = {
                'filename': f'{name}.jpg',
                'selectedList': [
                    {'name': 'sku_a', 'rect': {'x': 18, 'y': 20, 'width': 28, 'height': 42}},
                    {'name': 'sku_b', 'rect': {'x': 52, 'y': 18, 'width': 30, 'height': 44}},
                    {'name': 'sku_c', 'rect': {'x': 86, 'y': 22, 'width': 26, 'height': 40}},
                    {'name': 'sku_footer_a', 'rect': {'x': 12, 'y': 82, 'width': 40, 'height': 24}},
                    {'name': 'sku_footer_b', 'rect': {'x': 60, 'y': 84, 'width': 42, 'height': 22}},
                    {'name': 'other_lary', 'rect': {'x': 0, 'y': 90, 'width': 160, 'height': 30}},
                ],
            }
            zipped.writestr(f'seg/{name}.seg', json.dumps(anns))
    index, _ = build_index({'learning': archive})
    dest = root / 'data' / 'work' / 'index' / 'dataset_index.json'
    dest.parent.mkdir(parents=True)
    dest.write_text(json.dumps(index), encoding='utf-8')
    save_glass_plan(root, build_glass_plan(root, scenes=50))
    return root


def test_auto_previews_render_without_manual_quads(tmp_path: Path) -> None:
    root = _project(tmp_path)
    archive = root / 'data/raw/for_learning.zip'
    plan = root / 'reports/glass_pilot100/plan.json'
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    plan_bytes = plan.read_bytes()
    outputs = generate_auto_glass_previews(root, count=3)
    assert len(outputs) == 3
    for folder, report in outputs:
        assert folder.is_dir()
        assert report['stage7_export'] is False
        assert report['training_approved'] is False
        assert len(report['selected_objects']) >= 2
        assert report['selection_mode'] == 'bbox_upper_row_v2'
        assert folder.name.endswith('_v2')
        assert len(report['auto_quad_norm']) == 4
        assert {p.name for p in folder.iterdir()} == {
            'original.png', 'glass_region.png', 'roi_mask_visual.png',
            'glass_diffuse.png', 'glass_fluorescent.png',
            'glass_diffuse_mask_visual.png', 'glass_fluorescent_mask_visual.png',
            'preview_report.json',
        }
        assert [x['technical_qa']['status'] for x in report['variants']] == ['PASS', 'PASS']
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == archive_hash
    assert plan.read_bytes() == plan_bytes
    assert not (root / 'outputs').exists()


def test_auto_previews_are_not_overwritten(tmp_path: Path) -> None:
    root = _project(tmp_path)
    first = generate_auto_glass_previews(root, count=1)[0][0]
    report_before = (first / 'preview_report.json').read_bytes()
    with pytest.raises(BatchRunError, match='already exists'):
        generate_auto_glass_previews(root, count=1)
    assert (first / 'preview_report.json').read_bytes() == report_before


def test_bounds_and_missing_products_fail_closed(tmp_path: Path) -> None:
    root = _project(tmp_path)
    with pytest.raises(BatchRunError, match='within the glass plan'):
        generate_auto_glass_previews(root, start=49, count=2)
    with pytest.raises(BatchRunError, match='positive integers'):
        generate_auto_glass_previews(root, count=0)


def test_prefers_upper_products_when_footer_bbox_also_present(tmp_path: Path) -> None:
    root = _project(tmp_path)
    _, report = generate_auto_glass_previews(root, count=1)[0]
    selected = {row['class_name'] for row in report['selected_objects']}
    assert {'sku_a', 'sku_b', 'sku_c'} <= selected
    assert 'sku_footer_a' not in selected
    assert 'sku_footer_b' not in selected
    assert report['auto_quad_norm'][2][1] <= 0.78


def test_legacy_preview_folder_is_preserved(tmp_path: Path) -> None:
    root = _project(tmp_path)
    plan_file = root / 'reports/glass_pilot100/plan.json'
    source = json.loads(plan_file.read_text(encoding='utf-8'))['sources'][0]
    folder = root / 'reports/glass_pilot100/auto_previews' / (
        'scene_00_' + source['image_sha256'][:16]
    )
    folder.mkdir(parents=True)
    (folder / 'keep.txt').write_text('old preview', encoding='utf-8')
    new_folder, _ = generate_auto_glass_previews(root, count=1)[0]
    assert new_folder != folder
    assert new_folder.name.endswith('_v2')
    assert (folder / 'keep.txt').read_text(encoding='utf-8') == 'old preview'
