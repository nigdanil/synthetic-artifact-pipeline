"""Stage 8.1: offline visual review sheet; original SAP runs are never modified.

The chart is for human *inspection*, not proof of realism or YOLO readiness.
All inputs are re-opened from a Stage-7 bundle with SHA-256 verification.
Derived PNG/JSON files are written only under Git-ignored reports/visual_review/.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from sapipeline.annotations import parse_seg

_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_HASH = re.compile(r"[a-f0-9]{64}\Z")
_REQUIRED = {"original", "image", "effect_mask", "annotation", "qa_report"}
_W, _H = 1880, 3160
_BG = "#0f1722"
_CARD = "#1c2735"
_MUTED = "#a9bbc9"
_WHITE = "#f5f8fa"
_ACCENT = "#57c9e7"


class ReviewSheetError(ValueError):
    """Invalid/stale SAP export, unsafe path, or corrupted source payload."""


@dataclass(frozen=True, slots=True)
class ReviewSheet:
    png: bytes
    metrics: dict[str, Any]


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(data: bytes, label: str) -> dict[str, Any]:
    try:
        obj = json.loads(data)
    except (UnicodeError, ValueError) as exc:
        raise ReviewSheetError(f"Invalid {label} JSON") from exc
    if not isinstance(obj, dict):
        raise ReviewSheetError(f"{label} must be a JSON object")
    return obj


def _safe_input(run: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ReviewSheetError("Unsafe manifest artifact path")
    name = Path(relative)
    if name.is_absolute() or any(part in ("..", ".", "") for part in name.parts):
        raise ReviewSheetError("Unsafe manifest artifact path")
    candidate = run / name
    if any(piece.is_symlink() for piece in (candidate, *candidate.parents) if piece != run.parent):
        raise ReviewSheetError("Symlinks are not allowed in artifact paths")
    if not candidate.is_file():
        raise ReviewSheetError("Missing manifest-listed artifact")
    return candidate


def _open_png(data: bytes, label: str, mode: str) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.format != "PNG" or img.mode != mode:
                raise ReviewSheetError(f"{label} must be PNG in mode {mode}")
            return np.array(img, copy=True)
    except (OSError, ValueError) as exc:
        raise ReviewSheetError(f"Cannot decode {label} PNG") from exc


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def _text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], label: str,
          size: int = 22, color: str = _WHITE, bold: bool = False) -> None:
    draw.text(xy, label, font=_font(size, bold=bold), fill=color)


def _panel(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], title: str,
           sub: str = "") -> None:
    draw.rounded_rectangle(rect, radius=16, fill=_CARD)
    left, top, right, _ = rect
    _text(draw, (left + 18, top + 12), title, 26, bold=True)
    if sub:
        box = draw.textbbox((0, 0), sub, font=_font(19))
        _text(draw, (right - (box[2] - box[0]) - 20, top + 18), sub, 19, _MUTED)


def _image_in_frame(canvas: Image.Image, source: Image.Image,
                    area: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = area
    width, height = x1 - x0, y1 - y0
    scaled = ImageOps.contain(source, (width, height), Image.Resampling.LANCZOS)
    bg = Image.new("RGB", (width, height), (9, 14, 23))
    bg.paste(scaled, ((width - scaled.width) // 2, (height - scaled.height) // 2))
    canvas.paste(bg, (x0, y0))


def _heatmap(original: np.ndarray, modified: np.ndarray) -> Image.Image:
    """False-color, gamma-enhanced difference; labels explicitly say 'enhanced'."""
    delta = np.max(np.abs(modified.astype(np.int16) - original.astype(np.int16)), axis=2)
    peak = max(1, int(delta.max()))
    strength = np.power(delta.astype(np.float32) / peak, 0.60)
    rgb = np.zeros((*delta.shape, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.rint(strength * 255).astype(np.uint8)
    rgb[:, :, 1] = np.rint(np.clip(strength * 1.6 - 0.42, 0, 1) * 185).astype(np.uint8)
    rgb[:, :, 2] = np.rint(np.clip(strength - 0.8, 0, 0.2) * 240).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _mask_overlay(original: np.ndarray, mask: np.ndarray) -> Image.Image:
    weight = mask.astype(np.float32) / 65535.0
    alpha = weight[..., None] * .56
    colored = np.array([0, 219, 240], dtype=np.float32)
    blended = np.rint(original * (1 - alpha) + colored * alpha).astype(np.uint8)
    return Image.fromarray(blended, mode="RGB")


def _zoom_roi(original: Image.Image, modified: Image.Image,
              bbox: list[int | float], tile_size: tuple[int, int]) -> Image.Image:
    w, h = original.size
    x1, y1, x2, y2 = (float(v) for v in bbox)
    margin = max(12, int(max(x2 - x1, y2 - y1) * .22))
    left, top = max(0, math.floor(x1 - margin)), max(0, math.floor(y1 - margin))
    right, bottom = min(w, math.ceil(x2 + margin)), min(h, math.ceil(y2 + margin))
    if not (0 <= left < right <= w and 0 <= top < bottom <= h):
        raise ReviewSheetError("Invalid ROI crop coordinates")
    before, after = original.crop((left, top, right, bottom)), modified.crop((left, top, right, bottom))
    target = Image.new("RGB", tile_size, (12, 18, 28))
    half, height = tile_size[0] // 2, tile_size[1]
    _image_in_frame(target, before, (0, 0, half - 5, height))
    _image_in_frame(target, after, (half + 5, 0, tile_size[0], height))
    return target


def _review_status(root: Path, run_id: str, manifest_sha: str) -> tuple[str, str]:
    """Use only manifest-matching local human decisions; never auto-approve."""
    review_file = root / "reports/qa_acceptance/reviews.json"
    if not review_file.exists():
        return "needs_review", ""
    if review_file.is_symlink():
        raise ReviewSheetError("Symlinked manual review file")
    data = _read_json(review_file.read_bytes(), "manual review")
    if data.get("schema_version") != "sap-manual-reviews-v1":
        raise ReviewSheetError("Unknown manual review schema")
    entries = data.get("reviews")
    if not isinstance(entries, list):
        raise ReviewSheetError("Invalid manual review list")
    matching = [entry for entry in entries if isinstance(entry, dict) and entry.get("run_id") == run_id]
    if len(matching) > 1:
        raise ReviewSheetError("Duplicate human review entries")
    if not matching:
        return "needs_review", ""
    review = matching[0]
    if review.get("manifest_sha256") != manifest_sha:
        return "stale_review", ""
    verdict = review.get("verdict")
    if verdict not in ("approved", "rejected", "needs_review"):
        raise ReviewSheetError("Invalid human review verdict")
    reviewer = review.get("reviewer") if isinstance(review.get("reviewer"), str) else ""
    if verdict == "approved":
        checks = review.get("checks")
        if not reviewer.strip() or not isinstance(checks, dict) or not all(
            checks.get(x) is True for x in ("bbox_consistency", "glare_naturalness", "product_identity")
        ):
            return "incomplete_approval", reviewer
    return verdict, reviewer


def _validate_and_load(project_root: Path, run_id: str) -> tuple[
    dict[str, Any], dict[str, Any], np.ndarray, np.ndarray, np.ndarray, str
]:
    if not _RUN_ID.fullmatch(run_id):
        raise ReviewSheetError("Unsafe run_id")
    root = Path(project_root).resolve(strict=True)
    if not (root / "pyproject.toml").is_file():
        raise ReviewSheetError("project_root must be an SAP checkout")
    runs = root / "outputs/sap_runs"
    run = runs / run_id
    if runs.is_symlink() or run.is_symlink() or not run.is_dir():
        raise ReviewSheetError("Run directory missing or symlinked")
    manifest_path = _safe_input(run, "manifest.json")
    manifest_raw = manifest_path.read_bytes()
    manifest = _read_json(manifest_raw, "manifest")
    if manifest.get("schema_version") != "sap-artifact-v1" or manifest.get("run_id") != run_id:
        raise ReviewSheetError("Wrong export schema or run_id")
    output, hashes = manifest.get("output"), manifest.get("output_sha256")
    if not isinstance(output, dict) or not isinstance(hashes, dict):
        raise ReviewSheetError("Missing export files/hashes")
    if set(output) != _REQUIRED or set(hashes) != _REQUIRED:
        raise ReviewSheetError("Wrong exported artifact set")
    artifacts: dict[str, bytes] = {}
    for key, rel in output.items():
        expected = hashes[key]
        if not isinstance(expected, str) or not _HASH.fullmatch(expected):
            raise ReviewSheetError("Invalid expected SHA256")
        data = _safe_input(run, rel).read_bytes()
        if _digest(data) != expected:
            raise ReviewSheetError(f"Corrupted {key}: SHA256 mismatch")
        artifacts[key] = data
    if len(set(output.values())) != len(output):
        raise ReviewSheetError("Repeated artifact file paths")
    qa = _read_json(artifacts["qa_report"], "technical QA")
    if qa.get("schema_version") != "sap-qa-v1" or qa.get("status") != "PASS" or qa.get("failures"):
        raise ReviewSheetError("Invalid or failed technical QA")
    if not isinstance(qa.get("checks"), dict) or not all(x is True for x in qa["checks"].values()):
        raise ReviewSheetError("Failed technical QA checks")
    if not isinstance(manifest.get("qa"), dict) or manifest["qa"].get("status") != "PASS":
        raise ReviewSheetError("Manifest QA status mismatch")
    if manifest["qa"].get("training_approved") is not False:
        raise ReviewSheetError("Training approval must not be asserted")
    original = _open_png(artifacts["original"], "original", "RGB")
    modified = _open_png(artifacts["image"], "modified", "RGB")
    mask = _open_png(artifacts["effect_mask"], "mask", "I;16")
    if original.shape != modified.shape or mask.shape != original.shape[:2]:
        raise ReviewSheetError("Image/mask geometry mismatch")
    if manifest.get("image_size_px") != [int(original.shape[1]), int(original.shape[0])]:
        raise ReviewSheetError("Wrong geometry in manifest")
    try:
        seg = parse_seg(artifacts["annotation"], image_width=int(original.shape[1]),
                        image_height=int(original.shape[0]),
                        image_filename=Path(output["image"]).name)
    except ValueError as exc:
        raise ReviewSheetError("Invalid SEG geometry or filename") from exc
    targets = manifest.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ReviewSheetError("Missing selected bbox targets")
    objects = {item.instance_index: item for item in seg.objects}
    if not all(isinstance(target, dict)
               and type(target.get("instance_index")) is int
               and target["instance_index"] in objects
               and target.get("bbox_xyxy_px") == list(objects[target["instance_index"]].bbox_xyxy_px)
               and target.get("class_name") == objects[target["instance_index"]].class_name
               for target in targets):
        raise ReviewSheetError("SEG bbox / manifest target mismatch")
    changed = np.any(original != modified, axis=2)
    if np.any(changed & (mask == 0)):
        raise ReviewSheetError("Modified pixels outside encoded mask")
    if not np.any(changed):
        raise ReviewSheetError("No observable effect")
    return manifest, qa, original, modified, mask, _digest(manifest_raw)


def build_review_sheet(project_root: Path, run_id: str) -> ReviewSheet:
    """Return PNG bytes plus independent metrics; does not write to any directory."""
    root = Path(project_root).resolve(strict=True)
    manifest, qa, original, modified, mask, manifest_sha = _validate_and_load(root, run_id)
    status, reviewer = _review_status(root, run_id, manifest_sha)
    height, width = mask.shape
    active = mask > 0
    diff = np.abs(modified.astype(np.int16) - original.astype(np.int16))
    per_pixel = diff.max(axis=2)
    changed = per_pixel > 0
    inside = per_pixel[active]
    changed_n = int(changed.sum())
    changed_inside = int((changed & active).sum())
    coverage = float(active.mean() * 100)
    bins = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
    hist = np.histogram(inside, bins=bins)[0]
    brightness_original = float(original[active].mean()) if active.any() else 0.0
    brightness_modified = float(modified[active].mean()) if active.any() else 0.0
    effects = manifest.get("effects")
    if not isinstance(effects, list) or not effects or not isinstance(effects[0], dict):
        raise ReviewSheetError("No effect parameters in manifest")
    effect = effects[0]
    if effect.get("changed_pixels") != qa.get("metrics", {}).get("changed_pixels"):
        raise ReviewSheetError("Effect and QA metadata disagree")
    metrics: dict[str, Any] = {
        "schema_version": "sap-visual-review-v1",
        "run_id": run_id,
        "manifest_sha256": manifest_sha,
        "source_sha256": manifest.get("source", {}).get("image_sha256"),
        "source_split": manifest.get("source", {}).get("split"),
        "image_size_px": [width, height],
        "effect": effect,
        "technical_qa_status": "PASS",
        "manual_verdict": status,
        "reviewer": reviewer,
        "manual_review_required": True,
        "training_approved": False,
        "metrics": {
            "changed_pixels": changed_n,
            "changed_fraction_percent": round(100 * changed_n / (width * height), 5),
            "mask_active_pixels_16bit": int(active.sum()),
            "mask_coverage_percent": round(coverage, 5),
            "changed_inside_mask": changed_inside,
            "changed_outside_mask": int((changed & ~active).sum()),
            "mean_abs_diff_all_channels_255": round(float(diff.mean()), 5),
            "mean_abs_diff_changed_channels_255": round(float(diff[changed].mean()), 5),
            "max_abs_diff_255": int(diff.max()),
            "mean_brightness_original_inside_255": round(brightness_original, 5),
            "mean_brightness_modified_inside_255": round(brightness_modified, 5),
            "brightness_delta_inside_255": round(brightness_modified - brightness_original, 5),
            "new_clipping_fraction": qa.get("metrics", {}).get("new_clipping_fraction"),
            "outside_mask_delta_qa": qa.get("metrics", {}).get("max_outside_mask_delta"),
        },
        "diff_histogram": [
            {"range": f"{lo}-{hi-1}", "count": int(count), "mask_fraction_percent":
             round(float(count * 100 / max(1, len(inside))), 3)}
            for lo, hi, count in zip(bins[:-1], bins[1:], hist, strict=True)
        ],
        "note": "Visualization only. Heatmap contrast is enhanced; QA PASS is not training approval.",
    }
    orig_img = Image.fromarray(original, mode="RGB")
    modified_img = Image.fromarray(modified, mode="RGB")
    canvas = Image.new("RGB", (_W, _H), _BG)
    d = ImageDraw.Draw(canvas)
    _text(d, (44, 28), "SAP  /  VISUAL QUALITY REVIEW", 42, bold=True)
    _text(d, (46, 93), f"RUN {run_id}   |   {effect.get('schema_version', '?')} / {effect.get('kind', '?')}",
          24, _MUTED)
    # Technical and manual review are intentionally separate.
    d.rounded_rectangle((1380, 34, 1820, 114), radius=13, fill="#19392f")
    _text(d, (1401, 44), "TECHNICAL QA: PASS", 23, "#9ae9bd", bold=True)
    _text(d, (1401, 77), f"HUMAN: {status.upper()}", 18, _WHITE)

    lx, rx, cw, panel_h = 42, 966, 872, 700
    for x, y, title, subtitle, photo in (
        (lx, 153, "01  ORIGINAL", "unaltered exported RGB", orig_img),
        (rx, 153, "02  MODIFIED", "photometric result", modified_img),
        (lx, 874, "03  MASK", "encoded PNG-16 / actual weights",
         Image.fromarray(np.rint(mask / 257).astype(np.uint8), mode="L").convert("RGB")),
        (rx, 874, "04  ABS DIFF HEATMAP", "enhanced contrast - not physical exposure",
         _heatmap(original, modified)),
        (lx, 1595, "05  MASK OVERLAY ON ORIGINAL", "cyan indicates mask weight",
         _mask_overlay(original, mask)),
    ):
        _panel(d, (x, y, x + cw, y + panel_h), title, subtitle)
        _image_in_frame(canvas, photo, (x + 16, y + 58, x + cw - 16, y + panel_h - 16))

    _panel(d, (rx, 1595, rx + cw, 2295), "06  NUMERIC METRICS")
    numbers = metrics["metrics"]
    rows = [
        ("Changed pixels", f"{changed_n:,} / {width * height:,}  ({numbers['changed_fraction_percent']:.2f}%)"),
        ("Mask coverage", f"{numbers['mask_coverage_percent']:.2f}%"),
        ("Mean abs diff (0-255)", f"{numbers['mean_abs_diff_all_channels_255']:.2f}"),
        ("Mean diff, changed", f"{numbers['mean_abs_diff_changed_channels_255']:.2f}"),
        ("Max channel diff", f"{numbers['max_abs_diff_255']}"),
        ("Brightness delta in mask", f"{numbers['brightness_delta_inside_255']:+.2f} / 255"),
        ("Changed outside mask", f"{numbers['changed_outside_mask']}"),
        ("New clipping fraction", f"{numbers['new_clipping_fraction']}"),
    ]
    for j, (key, val) in enumerate(rows):
        yy = 1685 + 65 * j
        _text(d, (989, yy), key, 21, _MUTED)
        _text(d, (1490, yy), val, 22, bold=True)

    # Percentage bars retain an absolute 0-100% scale across all runs.
    for j, (label, percent) in enumerate((
        ("MASK", numbers["mask_coverage_percent"]),
        ("CHANGED", numbers["changed_fraction_percent"]),
    )):
        yy = 2208 + 34 * j
        _text(d, (991, yy - 3), label, 17, _MUTED)
        d.rounded_rectangle((1150, yy, 1760, yy + 16), radius=5, fill="#334356")
        bar_end = 1150 + max(1, round(610 * percent / 100))
        d.rounded_rectangle((1150, yy, bar_end, yy + 16), radius=5, fill=_ACCENT)

    _panel(d, (lx, 2316, lx + cw, 2654), "07  DIFF HISTOGRAM", "max channel delta inside MASK")
    chart_left, chart_top, chart_w = 163, 2388, 590
    ymax = max(1, int(hist.max()))
    for i, (bar, item) in enumerate(zip(hist, metrics["diff_histogram"], strict=True)):
        yy = chart_top + i * 28
        _text(d, (72, yy), item["range"].rjust(7), 17, _MUTED)
        width_px = int(chart_w * int(bar) / ymax)
        d.rounded_rectangle((chart_left, yy + 1, chart_left + max(1, width_px), yy + 17),
                            radius=3, fill=_ACCENT)
        _text(d, (chart_left + chart_w + 9, yy), f"{item['mask_fraction_percent']:.1f}%", 17)

    _panel(d, (rx, 2316, rx + cw, 2654), "08  EFFECT PARAMETERS")
    keys = ("kind", "seed", "intensity", "angle_deg", "band_count", "band_width",
            "band_separation", "edge_feather", "radius_scale", "aspect_ratio",
            "falloff", "center_jitter")
    params = [(name, effect[name]) for name in keys if name in effect]
    for j, (name, val) in enumerate(params[:12]):
        col, row = divmod(j, 6)
        xx, yy = rx + 22 + col * 425, 2385 + 40 * row
        label = f"{name}: {str(val)[:18]}"
        _text(d, (xx, yy), label, 20, _WHITE if col else _MUTED)

    _text(d, (45, 2692), "09  TARGET ROI ZOOMS  /  BEFORE  |  AFTER", 29, bold=True)
    targets = manifest.get("targets")
    if not isinstance(targets, list):
        raise ReviewSheetError("Invalid target list")
    target_rows = [item for item in targets if isinstance(item, dict)][:3]
    roi_y = 2748
    tile_width, tile_gap, tile_height = 580, 28, 315
    for i in range(3):
        xx = 42 + i * (tile_width + tile_gap)
        _panel(d, (xx, roi_y, xx + tile_width, roi_y + 345),
               f"ROI {target_rows[i].get('instance_index')}" if i < len(target_rows) else "ROI unavailable")
        if i < len(target_rows):
            crop = _zoom_roi(orig_img, modified_img, target_rows[i].get("bbox_xyxy_px", []),
                             (tile_width - 32, tile_height - 60))
            canvas.paste(crop, (xx + 16, roi_y + 64))
            _text(d, (xx + 25, roi_y + 41), "BEFORE", 15, _MUTED)
            _text(d, (xx + tile_width // 2 + 10, roi_y + 41), "AFTER", 15, _MUTED)

    _text(d, (46, 3120), "PRIVATE  |  QA PASS is technical only; human review is separate. Never auto-approve training.",
          19, _MUTED)
    output = io.BytesIO()
    canvas.save(output, format="PNG", optimize=True)
    return ReviewSheet(output.getvalue(), metrics)
