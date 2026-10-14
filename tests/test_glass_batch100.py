"""Synthetic-only tests of the resumable 100-image bbox glass preview pilot."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest
from PIL import Image, ImageDraw

from sapipeline.batch.glass_auto_preview import generate_auto_glass_previews
from sapipeline.batch.glass_batch100 import pack_batch, run_batch
from sapipeline.batch.glass_pilot import build_glass_plan, save_glass_plan
from sapipeline.batch.runner import BatchRunError
from sapipeline.dataset import build_index


def _project(root: Path) -> Path:
    (root / 'pyproject.toml').write_text('[project]\nname="fake-sap"\n', encoding='utf-8')
    raw = root / 'data/raw'
    raw.mkdir(parents=True)
    archive = raw / 'for_learning.zip'
    with ZipFile(archive, 'w') as zipped:
        for n in range(50):
            photo = Image.new('RGB', (160, 120), (28 + n, 78, 115))
            draw = ImageDraw.Draw(photo)
            draw.text((8, 8), f'SCENE {n}', fill=(255, 255, 255))
            draw.rectangle((n % 11 + 10, 97, n % 11 + 35, 116), fill=(120, 22, 32))
            buf = BytesIO()
            photo.save(buf, format='JPEG')
            stem = f'private_scene_{n:03d}'
            zipped.writestr(f'img/{stem}.jpg', buf.getvalue())
            zipped.writestr(f'seg/{stem}.seg', json.dumps({
                'filename': f'{stem}.jpg',
                'selectedList': [
                    {'name': 'sku_a', 'gender': False,
                     'rect': {'x': 20, 'y': 18, 'width': 26, 'height': 37}},
                    {'name': 'sku_b', 'gender': False,
                     'rect': {'x': 54, 'y': 17, 'width': 29, 'height': 41}},
                    {'name': 'sku_c', 'gender': False,
                     'rect': {'x': 90, 'y': 20, 'width': 27, 'height': 36}},
                    {'name': 'other_lary',
                     'rect': {'x': 0, 'y': 92, 'width': 160, 'height': 25}},
                ],
            }))
    index, _ = build_index({'learning': archive})
    path = root / 'data/work/index/dataset_index.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(index), encoding='utf-8')
    save_glass_plan(root, build_glass_plan(root, scenes=50))
    return root


def test_batch_resumes_and_packs_exactly_100_without_source_mutation(tmp_path: Path) -> None:
    root = _project(tmp_path)
    archive = root / 'data/raw/for_learning.zip'
    plan_path = root / 'reports/glass_pilot100/plan.json'
    original_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    plan_bytes = plan_path.read_bytes()
    preview = generate_auto_glass_previews(root, count=1)[0][0]
    old_report = (preview / 'preview_report.json').read_bytes()
    status = run_batch(root, max_new_scenes=1)
    assert status == {
        'qa_pass': 4, 'target': 100, 'created_scenes': 1,
        'reused_scenes': 1, 'qa_rejected': 0,
    }
    status = run_batch(root)
    assert status['qa_pass'] == 100
    assert status['created_scenes'] == 48
    assert status['reused_scenes'] == 2
    assert status['qa_rejected'] == 0
    assert (preview / 'preview_report.json').read_bytes() == old_report
    assert plan_path.read_bytes() == plan_bytes
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == original_sha
    assert not (root / 'outputs').exists()
    zip_path, count = pack_batch(root)
    assert count == 100
    with ZipFile(zip_path) as archive_zip:
        names = archive_zip.namelist()
        assert len([name for name in names if name.startswith('images/')]) == 100
        assert len([name for name in names if name.startswith('annotations/')]) == 100
        assert len([name for name in names if name.startswith('masks_visual/')]) == 100
        summary = json.loads(archive_zip.read('qa_preview_manifest.json'))
        assert summary['total'] == 100
        assert summary['unique_source_scenes'] == 50
        assert summary['training_approved'] is False
        assert summary['stage7_export'] is False
        assert all(row['technical_qa'] == 'PASS' for row in summary['items'])
        first = summary['items'][0]
        ann = json.loads(archive_zip.read('annotations/' + first['run_id'] + '.seg'))
        assert ann['filename'] == first['run_id'] + '.png'
        assert ann['selectedList'][0]['gender'] is False
        assert ann['selectedList'][0]['rect'] == {
            'x': 20, 'y': 18, 'width': 26, 'height': 37,
        }
    with pytest.raises(BatchRunError, match='already exists'):
        pack_batch(root)


def test_corrupted_previous_preview_rejected_and_not_overwritten(tmp_path: Path) -> None:
    root = _project(tmp_path)
    preview = generate_auto_glass_previews(root, count=1)[0][0]
    path = preview / 'glass_fluorescent.png'
    path.write_bytes(b'BAD')
    with pytest.raises(BatchRunError, match='Unreadable glass preview'):
        run_batch(root)
    assert path.read_bytes() == b'BAD'
    with pytest.raises(BatchRunError, match='Unreadable glass preview'):
        pack_batch(root)
    assert not (root / 'reports/glass_pilot100/glass_bbox100_qa_preview.zip').exists()


def test_small_incomplete_batch_cannot_be_packed(tmp_path: Path) -> None:
    root = _project(tmp_path)
    status = run_batch(root, max_new_scenes=1)
    assert status['qa_pass'] == 2
    with pytest.raises(BatchRunError, match='Only 2/100'):
        pack_batch(root)
