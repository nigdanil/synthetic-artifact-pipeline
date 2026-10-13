"""ROI selector policy checks using invented classes, never source dataset names."""

from __future__ import annotations

import json

import pytest

from sapipeline.annotations import parse_seg
from sapipeline.selection import SelectionError, select_rois


def document():
    classes = ["sample_a", "glint", "sample_a", "product", "other_shelf",
               "sample_b", "obstacle_object", "other_lary", "sample_c"]
    payload = {"filename": "sample.jpg", "selectedList": [
        {"name": name, "rect": {"x": i * 10, "y": 5, "width": 8, "height": 16}}
        for i, name in enumerate(classes)
    ]}
    return parse_seg(json.dumps(payload), image_width=100, image_height=60)


def indices(result):
    return [x.instance_index for x in result.targets]


def test_explicit_instance_is_not_class_lookup():
    assert indices(select_rois(document(), mode="by_index", instance_index=2)) == [2]


def test_class_selects_all_repeated_instances():
    assert indices(select_rois(document(), mode="by_class", class_name="sample_a")) == [0, 2]
    assert indices(select_rois(document(), mode="by_class", class_name="product")) == [3]


def test_random_selection_product_policy_and_uniqueness():
    for seed in range(30):
        result = select_rois(document(), mode="random_n", seed=seed, count=4)
        assert len(indices(result)) == len(set(indices(result))) == 4
        assert set(indices(result)) == {0, 2, 5, 8}
        assert indices(result) == sorted(indices(result))
    result = select_rois(document(), mode="random_n", seed=2, count=5,
                         include_generic_product=True)
    assert set(indices(result)) == {0, 2, 3, 5, 8}


def test_seed_reproducibility_and_no_global_random_side_effect():
    import random
    random.seed(100)
    before = random.getstate()
    a = indices(select_rois(document(), mode="random_n", seed=42017, count=3))
    b = indices(select_rois(document(), mode="random_n", seed=42017, count=3))
    assert a == b
    assert random.getstate() == before
    assert len(indices(select_rois(document(), mode="random_one", seed=42017))) == 1


def test_explicit_service_requires_opt_in():
    for mode, kwargs in [("by_index", {"instance_index": 1}),
                         ("by_class", {"class_name": "glint"})]:
        with pytest.raises(SelectionError, match="Service ROI"):
            select_rois(document(), mode=mode, **kwargs)
        assert indices(select_rois(document(), mode=mode,
                                  allow_service_targets=True, **kwargs)) == [1]


@pytest.mark.parametrize("kwargs", [
    {"mode": "by_index", "instance_index": 99},
    {"mode": "by_index", "instance_index": -1},
    {"mode": "by_index", "instance_index": True},
    {"mode": "by_index"},
    {"mode": "by_class", "class_name": "missing"},
    {"mode": "by_class"},
    {"mode": "random_n", "count": 5},
    {"mode": "random_n", "count": 0},
    {"mode": "random_n", "count": True},
    {"mode": "random_one", "seed": -1},
    {"mode": "bad_mode"},
])
def test_invalid_selections_fail_without_fallback(kwargs):
    with pytest.raises(SelectionError):
        select_rois(document(), **kwargs)


def test_empty_or_only_service_boxes_rejected():
    doc = document()
    from dataclasses import replace
    for indices_ in ([], [1, 4, 6, 7]):
        doc2 = replace(doc, objects=tuple(doc.objects[i] for i in indices_))
        with pytest.raises(SelectionError, match="eligible"):
            select_rois(doc2, mode="random_one")
