# Task 2: SEG / YOLO TXT adapters and geometry validation

This stage provides **read-only** parsing, validation, reversible SEG metadata updates,
and normalized YOLO TXT conversion. No image alteration, ComfyUI modification,
mask production or annotation export to a dataset happens in this stage.

## Coordinate contract

- Canonical `bbox_xyxy_px`: `(x1, y1, x2, y2)` on **unrotated, encoded-image pixels**;
  half-open rectangles (`x2=x+width`, `y2=y+height`). `0 <= x1 < x2 <= W` and
  `0 <= y1 < y2 <= H`. No resize, transpose, or silent clipping.
- SEG `selectedList` preserves item order (`instance_index`), exact `name` spelling,
  and other fields (including future unknown attributes). `class_id` and `sku_id`
  are `None` until a trusted external mapping supplies IDs.
- YOLO TXT is standard five-column `class_id cx_norm cy_norm w_norm h_norm`, each
  normalized coordinate within `[0,1]`, with strictly positive width and height.
  Pixel-level floating-point noise is snapped when closer than `1e-8` pixel
  to an integer, allowing SEG -> YOLO -> canonical bbox round-trips.
- Images with non-default/invalid EXIF orientation are **flagged for review**, not
  automatically rotated. If applying rotation later, bbox must be transformed too.

## API

```python
from sapipeline.annotations import parse_seg, parse_yolo, to_seg_json, to_yolo_txt

# `raw_seg_bytes` and `image_size` come from local source files / ZIP entries.
annotation = parse_seg(raw_seg_bytes, image_width=1080, image_height=810,
                       image_filename="sample.jpg")
boxes = [item.bbox_xyxy_px for item in annotation.objects]
new_seg_bytes = to_seg_json(annotation, filename="sample_aug.png")

# Mapping is explicitly provided, stable for *every split*, never inferred.
classes = {0: "example_product", 5: "example_other"}
yolo_text = to_yolo_txt(annotation, classes)  # fails if class is unmapped
parsed = parse_yolo(yolo_text, image_width=1080, image_height=810,
                    class_mapping=classes, image_filename="sample.jpg")
```

These names, coordinates and class IDs are **synthetic examples**, not dataset classes.
Store any true vendor mapping as a local JSON file under
`configs/class_mappings/`, e.g. `{"0":"example_product","5":"example_other"}`.
That directory is Git-ignored. No YAML dependency is required.

The `to_seg_json()` method updates `filename` in a deep copy of the original SEG
structure; it preserves unknown root, object and rect fields and does not change
bbox values. Serialization may change whitespace/key formatting. The exporter
stage will later use this capability to write *new* derivative annotations.

## Full local-dataset check and private previews (Git Bash)

Place permitted source archives under `data/raw/`, activate `.venv`, then:

```bash
source .venv/Scripts/activate
python scripts/validate_annotations.py
```

The script validates all paired SEG files and renders **five sample** bbox overlays
from distinct learning photographs in raw pixel coordinates. Outputs are *local only*:

- `reports/annotation_validation_summary.json` — counts, orientation review flags
  and geometry policy; may contain private source names.
- `reports/annotation_previews/learning_001.png` through `learning_005.png` — local
  preview overlays, **not** intended for publication or model training.

On a second run, opt into replacing existing private files using `--force`:

```bash
python scripts/validate_annotations.py --force
```

Optional `--class-map configs/class_mappings/classes.json` checks that **all** SEG
names have a mapping for YOLO conversion. It does not create or infer a mapping.
Only commit the implementation/tests/docs. Never commit archives, class mappings,
index, reports or previews. Run `python scripts/check_public_repo.py` before push.

## Acceptance gate

1. All source pairs pass geometry check; bbox total matches the local dataset index.
   Non-default EXIF metadata remains a separate review flag.
2. Manually open all five **local** bbox previews, verify overlays on raw photo.
3. Synthetic tests cover invalid geometry, missing schema fields, unknown IDs,
   duplicate mapping names, preserved SEG fields and SEG/YOLO round trips.
4. Do not interpret parsing or previews as approval of the train/test split or
   vendor annotation correctness.
