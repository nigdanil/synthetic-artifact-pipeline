"""Synthetic-only fixtures: no proprietary photos or annotations in Git history."""

import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest
from PIL import Image

from sapipeline.dataset import DatasetIndexError, build_index


def jpeg_bytes(color: tuple[int, int, int] = (20, 30, 40)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 6), color).save(buffer, format="JPEG")
    return buffer.getvalue()


def seg_bytes(name: str, objects: list[dict] | None = None) -> bytes:
    return json.dumps(
        {
            "filename": name,
            "selectedList": objects
            if objects is not None
            else [{"name": "fake_item", "rect": {"x": 1, "y": 2, "width": 3, "height": 2}}],
        },
        ensure_ascii=False,
    ).encode("utf-8")


def write_zip(
    path: Path,
    pairs: list[tuple[str, bytes, bytes]],
    extras: dict[str, bytes] | None = None,
) -> Path:
    with ZipFile(path, "w") as zipfile:
        for image_name, image_bytes, annotation_bytes in pairs:
            zipfile.writestr(f"img/{image_name}", image_bytes)
            zipfile.writestr(f"seg/{Path(image_name).stem}.seg", annotation_bytes)
        for name, data in (extras or {}).items():
            zipfile.writestr(name, data)
    return path


def test_exact_duplicates_and_annotation_disagreements(tmp_path: Path) -> None:
    image_a = jpeg_bytes()
    image_b = jpeg_bytes((60, 80, 90))
    first = {"name": "fake_one", "rect": {"x": 1, "y": 2, "width": 3, "height": 2}}
    second = {"name": "fake_two", "rect": {"x": 3, "y": 1, "width": 2, "height": 2}}
    learning = write_zip(tmp_path / "learning.zip", [
        ("alpha.jpg", image_a, seg_bytes("alpha.jpg", [first, second])),
        ("beta.jpg", image_b, seg_bytes("beta.jpg", [first])),
    ])
    verification = write_zip(tmp_path / "verification.zip", [
        ("renamed.jpg", image_a, seg_bytes("renamed.jpg", [second, first])),
    ])
    index, report = build_index({"verification": verification, "learning": learning})
    assert report["pairs_by_split"] == {"learning": 2, "verification": 1}
    assert report["total_pairs"] == 3
    assert report["total_bbox_entries_declared"] == 5
    assert report["unique_images_sha256"] == 2
    assert report["exact_duplicate_groups"] == 1
    assert report["cross_split_duplicate_groups"] == 1
    assert report["annotation_disagreement_groups"] == 0  # ordering/filename ignored
    assert len(report["duplicate_groups"][0]["members"]) == 2
    assert index["items"][0]["width"] == 8
    assert index["items"][0]["height"] == 6
    assert index["items"][0]["canonical_image_id"] == index["items"][2]["canonical_image_id"]
    assert build_index({"verification": verification, "learning": learning}) == (index, report)

    with ZipFile(verification, "w") as archive:
        archive.writestr("img/renamed.jpg", image_a)
        archive.writestr("seg/renamed.seg", seg_bytes("renamed.jpg", [first]))
    _, changed = build_index({"learning": learning, "verification": verification})
    assert changed["annotation_disagreement_groups"] == 1


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ({"img/orphan.jpg": jpeg_bytes()}, "without SEG"),
        ({"seg/orphan.seg": seg_bytes("orphan.jpg")}, "without image"),
        ({"img/item.jpeg": jpeg_bytes()}, "ambiguous"),
        ({"../unsafe.txt": b"x"}, "unsafe ZIP member"),
    ],
)
def test_rejects_incomplete_or_unsafe_archives(
    tmp_path: Path, extra: dict[str, bytes], expected: str
) -> None:
    path = write_zip(
        tmp_path / "bad.zip", [("item.jpg", jpeg_bytes(), seg_bytes("item.jpg"))], extra
    )
    with pytest.raises(DatasetIndexError, match=expected):
        build_index({"learning": path})


@pytest.mark.parametrize(
    ("image", "annotation", "expected"),
    [
        (b"not-a-photo", seg_bytes("item.jpg"), "invalid image"),
        (jpeg_bytes(), b"{bad json", "invalid SEG JSON"),
        (jpeg_bytes(), seg_bytes("wrong.jpg"), "SEG filename differs"),
        (jpeg_bytes(), b'{}', "SEG must have"),
    ],
)
def test_rejects_damaged_or_mismatched_pair(
    tmp_path: Path, image: bytes, annotation: bytes, expected: str
) -> None:
    path = write_zip(tmp_path / "invalid.zip", [("item.jpg", image, annotation)])
    with pytest.raises(DatasetIndexError, match=expected):
        build_index({"learning": path})


def test_different_images_with_same_filename_are_not_merged(tmp_path: Path) -> None:
    a = write_zip(tmp_path / "a.zip", [("same.jpg", jpeg_bytes(), seg_bytes("same.jpg"))])
    b = write_zip(tmp_path / "b.zip", [("same.jpg", jpeg_bytes((1, 2, 3)), seg_bytes("same.jpg"))])
    index, summary = build_index({"learning": a, "verification": b})
    assert summary["unique_images_sha256"] == 2
    assert summary["exact_duplicate_groups"] == 0
    assert index["items"][0]["canonical_image_id"] != index["items"][1]["canonical_image_id"]


def test_does_not_create_files_or_mutate_archives(tmp_path: Path) -> None:
    path = write_zip(
        tmp_path / "read_only.zip", [("item.jpg", jpeg_bytes(), seg_bytes("item.jpg"))]
    )
    before = path.read_bytes()
    build_index({"learning": path})
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["read_only.zip"]
