"""Synthetic only: private 50-scene/100-effect HTML gallery safeguards."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
from test_glass_batch100 import _project

from sapipeline.batch.glass_batch100 import run_batch
from sapipeline.batch.glass_gallery import generate_glass_gallery
from sapipeline.batch.runner import BatchRunError


def _embedded(html: str) -> dict:
    match = re.search(r'<script id="gallery-data" type="application/json">(.*?)</script>',
                      html, flags=re.DOTALL)
    assert match is not None
    return json.loads(match.group(1))


def test_html_gallery_has_exact_50_scenes_and_100_reviewable_variants(tmp_path: Path) -> None:
    root = _project(tmp_path)
    assert run_batch(root)['qa_pass'] == 100
    plan = root / 'reports/glass_pilot100/plan.json'
    original_plan = plan.read_bytes()
    src_zip = root / 'data/raw/for_learning.zip'
    zip_sha = hashlib.sha256(src_zip.read_bytes()).hexdigest()

    path, count = generate_glass_gallery(root)
    assert count == 100
    assert path == root / 'reports/glass_pilot100/gallery/index.html'
    html = path.read_text(encoding='utf-8')
    doc = _embedded(html)
    assert doc['schema_version'] == 'sap-glass-bbox100-gallery-v1'
    assert doc['training_approved'] is False
    assert doc['manual_review_required'] is True
    assert len(doc['scenes']) == 50
    assert sum(len(row['variants']) for row in doc['scenes']) == 100
    assert len({v['run_id'] for row in doc['scenes'] for v in row['variants']}) == 100
    for scene in doc['scenes']:
        assert len(scene['variants']) == 2
        assert scene['original'].startswith('../auto_previews/')
        assert (path.parent / scene['original']).is_file()
        assert [v['variant'] for v in scene['variants']] == ['glass_diffuse', 'glass_fluorescent']
        for variant in scene['variants']:
            assert variant['technical_qa'] == 'PASS'
            assert (path.parent / variant['image']).is_file()
    assert 'Скачать решения JSON' in html
    assert 'Загрузить решения JSON' in html
    assert 'sap-glass-bbox100-visual-review-v1' in html
    assert 'training_approved:false' in html
    assert 'https://' not in html
    assert plan.read_bytes() == original_plan
    assert hashlib.sha256(src_zip.read_bytes()).hexdigest() == zip_sha
    assert not (root / 'outputs').exists()

    first_html_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(BatchRunError, match='already exists'):
        generate_glass_gallery(root)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == first_html_sha


def test_gallery_rejects_incomplete_batch_before_writing(tmp_path: Path) -> None:
    root = _project(tmp_path)
    run_batch(root, max_new_scenes=1)
    with pytest.raises(BatchRunError, match='missing|requires'):
        generate_glass_gallery(root)
    assert not (root / 'reports/glass_pilot100/gallery/index.html').exists()


def test_gallery_fails_closed_on_tampered_preview(tmp_path: Path) -> None:
    root = _project(tmp_path)
    run_batch(root, max_new_scenes=1)
    plan = json.loads((root / 'reports/glass_pilot100/plan.json').read_text(encoding='utf-8'))
    folder = root / 'reports/glass_pilot100/auto_previews' / (
        'scene_00_' + plan['sources'][0]['image_sha256'][:16] + '_v2'
    )
    (folder / 'glass_diffuse.png').write_bytes(b'damaged')
    with pytest.raises(BatchRunError, match='Unreadable glass preview'):
        generate_glass_gallery(root)
    assert not (root / 'reports/glass_pilot100/gallery/index.html').exists()
