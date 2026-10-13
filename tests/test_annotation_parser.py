"""Only synthetic classes, rectangles and images: safe for public Git history."""

import json
from pathlib import Path

import pytest

from sapipeline.annotations import (
    AnnotationError,
    load_class_mapping,
    parse_seg,
    parse_yolo,
    to_seg_json,
    to_yolo_txt,
    validate_class_mapping,
)

CLASSES = {0: "sample_one", 5: "sample_two"}


def source_seg() -> bytes:
    return json.dumps({
        "filename": "sample.jpg",
        "owner_field": {"source": "synthetic"},
        "selectedList": [
            {"name": "sample_one", "rect": {"x": 10, "y": 20, "width": 40, "height": 60,
                                            "vendor_padding": 2}, "gender": False,
             "meta": {"anything": [1, 2, 3]}},
            {"name": "sample_one", "rect": {"x": 0, "y": 0, "width": 10, "height": 10}},
            {"name": "sample_two", "rect": {"x": 80, "y": 100, "width": 20, "height": 20}},
        ],
    }, ensure_ascii=False).encode()


def parse(data: bytes | str | None = None):
    return parse_seg(source_seg() if data is None else data, image_width=100,
                     image_height=120, image_filename="sample.jpg")


def test_seg_rectangles_and_class_repetitions() -> None:
    doc = parse()
    assert len(doc.objects) == 3
    assert doc.objects[0].bbox_xyxy_px == (10, 20, 50, 80)
    assert doc.objects[1].bbox_xyxy_px == (0, 0, 10, 10)
    assert doc.objects[2].bbox_xyxy_px == (80, 100, 100, 120)
    assert [obj.instance_index for obj in doc.objects] == [0, 1, 2]
    assert doc.objects[0].class_name == doc.objects[1].class_name
    assert doc.objects[0].class_id is None
    assert doc.objects[0].sku_id is None
    assert doc.objects[0].raw_fields == {"gender": False, "meta": {"anything": [1, 2, 3]}}


def test_seg_round_trip_preserves_every_unknown_field() -> None:
    doc = parse()
    output = json.loads(to_seg_json(doc, filename="synthetic_aug.png"))
    original = json.loads(source_seg())
    assert original["filename"] == "sample.jpg"
    assert output == {**original, "filename": "synthetic_aug.png"}
    reread = parse_seg(json.dumps(output), image_width=100, image_height=120,
                       image_filename="synthetic_aug.png")
    assert [x.bbox_xyxy_px for x in reread.objects] == [x.bbox_xyxy_px for x in doc.objects]


def test_seg_yolo_seg_geometry_round_trip() -> None:
    doc = parse()
    text = to_yolo_txt(doc, CLASSES)
    yolo = parse_yolo(text, image_width=100, image_height=120, class_mapping=CLASSES)
    assert [o.class_id for o in yolo.objects] == [0, 0, 5]
    assert [o.class_name for o in yolo.objects] == [o.class_name for o in doc.objects]
    assert [o.bbox_xyxy_px for o in yolo.objects] == [o.bbox_xyxy_px for o in doc.objects]
    minimal = json.loads(to_seg_json(yolo, filename="sample.png"))
    assert minimal["filename"] == "sample.png"
    assert minimal["selectedList"][0]["rect"] == {"x": 10, "y": 20, "width": 40,
                                                    "height": 60}


def test_utf8_bom_and_nonascii_names() -> None:
    raw = {"filename": "sample.jpg", "selectedList": [{"name": "пример_1",
           "rect": {"x": 1, "y": 2, "width": 3, "height": 4}}]}
    doc = parse_seg(b"\xef\xbb\xbf" + json.dumps(raw, ensure_ascii=False).encode(),
                    image_width=100, image_height=120)
    assert doc.objects[0].class_name == "пример_1"
    assert json.loads(to_seg_json(doc, filename="out.png"))["selectedList"][0]["name"] == "пример_1"


@pytest.mark.parametrize("mutation,code", [
    (lambda x: x.pop("filename"), "schema_mismatch"),
    (lambda x: x.update({"selectedList": {}}), "schema_mismatch"),
    (lambda x: x["selectedList"][0].pop("name"), "schema_mismatch"),
    (lambda x: x["selectedList"][0].update({"name": ""}), "schema_mismatch"),
    (lambda x: x["selectedList"][0].pop("rect"), "schema_mismatch"),
    (lambda x: x["selectedList"][0]["rect"].update({"width": 0}), "invalid_bbox"),
    (lambda x: x["selectedList"][0]["rect"].update({"height": -2}), "invalid_bbox"),
    (lambda x: x["selectedList"][0]["rect"].update({"x": -1}), "invalid_bbox"),
    (lambda x: x["selectedList"][0]["rect"].update({"x": 99}), "invalid_bbox"),
    (lambda x: x["selectedList"][0]["rect"].update({"width": True}), "invalid_bbox"),
    (lambda x: x["selectedList"][0]["rect"].update({"y": float("nan")}), "invalid_bbox"),
])
def test_seg_rejects_corrupt_data(mutation, code: str) -> None:
    obj = json.loads(source_seg())
    mutation(obj)
    with pytest.raises(AnnotationError) as error:
        parse(json.dumps(obj))
    assert error.value.code == code


def test_seg_invalid_json_or_filename_mismatch() -> None:
    with pytest.raises(AnnotationError, match="invalid_json"):
        parse(b"{oops")
    with pytest.raises(AnnotationError, match="schema_mismatch"):
        parse_seg(source_seg(), image_width=100, image_height=120,
                  image_filename="wrong.jpg")


@pytest.mark.parametrize("input_text,code", [
    ("0 0.5 0.5 0.2", "schema_mismatch"),
    ("1 0.5 0.5 0.2 0.2", "unknown_class_mapping"),
    ("0 nan 0.5 0.2 0.2", "invalid_bbox"),
    ("0 inf 0.5 0.2 0.2", "invalid_bbox"),
    ("0 0.5 0.5 0 0.2", "invalid_bbox"),
    ("0 0.5 0.5 -0.2 0.2", "invalid_bbox"),
    ("0 0.02 0.5 0.2 0.2", "invalid_bbox"),
    ("0 0.5 1.1 0.2 0.2", "invalid_bbox"),
    ("garbage 0.5 0.5 0.2 0.2", "schema_mismatch"),
    ("0 0.5 0.5 xx 0.2", "invalid_bbox"),
])
def test_yolo_rejects_invalid_lines(input_text: str, code: str) -> None:
    with pytest.raises(AnnotationError) as error:
        parse_yolo(input_text, image_width=100, image_height=120, class_mapping=CLASSES)
    assert error.value.code == code


def test_yolo_blank_lines_and_fractional_pixels() -> None:
    doc = parse_yolo("\n0 0.5 0.5 0.25 0.25\n\n", image_width=100,
                     image_height=120, class_mapping=CLASSES)
    assert len(doc.objects) == 1
    assert doc.objects[0].bbox_xyxy_px == (37.5, 45, 62.5, 75)


def test_yolo_round_trip_for_fractional_bbox() -> None:
    source = "5 0.32759 0.44152 0.12341 0.15871\n"
    doc = parse_yolo(source, image_width=1920, image_height=1080, class_mapping=CLASSES)
    recovered = parse_yolo(to_yolo_txt(doc, CLASSES), image_width=1920,
                           image_height=1080, class_mapping=CLASSES)
    assert recovered.objects[0].bbox_xyxy_px == pytest.approx(doc.objects[0].bbox_xyxy_px)


def test_mapping_is_authoritative_and_not_alphabetical(tmp_path: Path) -> None:
    source = tmp_path / "classes.json"
    source.write_text('{"5": "sample_two", "0": "sample_one"}', encoding="utf-8")
    assert load_class_mapping(source) == CLASSES
    with pytest.raises(AnnotationError, match="unknown_class_mapping"):
        to_yolo_txt(parse(), {0: "sample_one"})


@pytest.mark.parametrize("mapping", [
    {}, {True: "sample_one"}, {-1: "sample_one"}, {0: ""}, {0: "x", 1: "x"},
])
def test_rejects_invalid_mapping(mapping) -> None:
    with pytest.raises(AnnotationError, match="unknown_class_mapping"):
        validate_class_mapping(mapping)


@pytest.mark.parametrize("text", ['{"00": "sample_one"}', '{"id": "sample_one"}',
                                  '["sample_one"]', '{"0": "a", "1": "a"}'])
def test_load_mapping_rejects_invalid_json_schema(tmp_path: Path, text: str) -> None:
    path = tmp_path / "mapping.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(AnnotationError, match="unknown_class_mapping"):
        load_class_mapping(path)


def test_rejects_nonpositive_image_size_and_unsafe_output_filename() -> None:
    with pytest.raises(AnnotationError, match="schema_mismatch"):
        parse_seg(source_seg(), image_width=0, image_height=120)
    doc = parse()
    for filename in ("../escape.png", "folder/a.png", "", "\\a.png"):
        with pytest.raises(AnnotationError, match="schema_mismatch"):
            to_seg_json(doc, filename=filename)
