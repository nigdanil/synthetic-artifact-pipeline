"""Surface Schema V1 tests using only fictional source identities."""

from copy import deepcopy

import pytest

from sapipeline.surfaces import (
    SurfaceError,
    dumps_document,
    loads_document,
    new_document,
    validate_document,
)

SOURCE_SHA = "a" * 64
QUAD = [[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]]


def example():
    doc = new_document(SOURCE_SHA, 1200, 900)
    doc["surfaces"] = [
        {
            "surface_id": "surface_001",
            "material": "glass",
            "quad_norm": QUAD,
            "reviewed": True,
            "note": "Первая стеклянная панель",
        },
        {
            "surface_id": "surface_002",
            "material": "metal",
            "quad_norm": [
                [0.1, 0.85], [0.9, 0.85],
                [0.9, 0.95], [0.1, 0.95],
            ],
            "reviewed": False,
            "note": "",
        },
    ]
    return doc


def test_empty_document():
    doc = new_document(SOURCE_SHA, 1200, 900)
    assert doc["surfaces"] == []
    assert doc["training_approved"] is False


def test_multiple_surfaces():
    doc = validate_document(example())
    assert len(doc["surfaces"]) == 2
    assert [s["material"] for s in doc["surfaces"]] == [
        "glass", "metal",
    ]


def test_json_roundtrip():
    doc = example()
    encoded = dumps_document(doc)
    decoded = loads_document(encoded.encode("utf-8"))
    assert decoded == validate_document(doc)
    assert dumps_document(decoded) == encoded


def test_source_binding():
    doc = example()
    validate_document(
        doc, expected_sha256=SOURCE_SHA,
        expected_size=(1200, 900),
    )
    with pytest.raises(SurfaceError, match="SHA-256 mismatch"):
        validate_document(doc, expected_sha256="b" * 64)


def test_source_size_mismatch():
    with pytest.raises(SurfaceError, match="dimensions mismatch"):
        validate_document(example(), expected_size=(800, 600))


def test_original_document_not_modified():
    doc = example()
    original = deepcopy(doc)
    validate_document(doc)
    assert doc == original


def test_duplicate_surface_ids():
    doc = example()
    doc["surfaces"][1]["surface_id"] = "surface_001"
    with pytest.raises(SurfaceError, match="duplicate surface_id"):
        validate_document(doc)


@pytest.mark.parametrize("quad", [
    [[0.1, 0.1], [0.9, 0.9], [0.9, 0.1], [0.1, 0.9]],
    [[0.1, 0.2], [1.1, 0.2], [0.9, 0.8], [0.1, 0.8]],
    [[0.1, 0.2], [0.9, 0.2], [0.1, 0.8], [0.9, 0.8]],
    [[0.1, 0.1], [0.10001, 0.1],
     [0.10001, 0.10001], [0.1, 0.10001]],
])
def test_bad_geometry(quad):
    doc = example()
    doc["surfaces"][0]["quad_norm"] = quad
    with pytest.raises(SurfaceError):
        validate_document(doc)


def test_invalid_material():
    doc = example()
    doc["surfaces"][0]["material"] = "oil"
    with pytest.raises(SurfaceError, match="unsupported"):
        validate_document(doc)


def test_unknown_material_not_reviewed():
    doc = example()
    doc["surfaces"][0]["material"] = "unknown"
    with pytest.raises(SurfaceError, match="cannot be confirmed"):
        validate_document(doc)


def test_training_not_approved():
    doc = example()
    doc["training_approved"] = True
    with pytest.raises(SurfaceError, match="cannot approve"):
        validate_document(doc)


def test_unknown_fields_rejected():
    doc = example()
    doc["surfaces"][0]["unexpected"] = "value"
    with pytest.raises(SurfaceError, match="unexpected"):
        validate_document(doc)


def test_duplicate_json_keys_rejected():
    with pytest.raises(SurfaceError, match="duplicate JSON key"):
        loads_document('{"schema_version":1,"schema_version":2}')
