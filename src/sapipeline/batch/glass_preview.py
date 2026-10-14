"""Private, human-guided glass-plane calibration and preview for Glass Pilot 100.

Never writes Stage-7 bundles, dataset source ZIPs/SEG or training approvals.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from sapipeline.dataset import choose_index_entry, load_indexed_pair, read_private_index
from sapipeline.effects.glass_reflection import GlassReflectionError, _quad, apply_glass_reflection
from sapipeline.qa import validate_artifact

from .glass_pilot import _paths, load_glass_plan, validate_glass_plan
from .runner import BatchRunError, _glass_selection, _root


def _source_indices(doc: dict[str, Any], start: int, count: int) -> range:
    if (type(start) is not int or type(count) is not int or start < 0
            or not 1 <= count <= 5 or start + count > len(doc["sources"])):
        raise BatchRunError("Choose --start >= 0 and --count 1..5 within the glass plan")
    return range(start, start + count)


def _safe_private(root: Path, category: str, folder_name: str) -> Path:
    """Reject symbolic links, traversal and file collisions in derived-only paths."""
    if category not in ("geometry_guides", "effect_previews"):
        raise BatchRunError("Unexpected glass preview category")
    if not folder_name.startswith("scene_") or not all(
        c.isascii() and (c.isalnum() or c == "_") for c in folder_name
    ):
        raise BatchRunError("Unsafe preview folder name")
    parent, _ = _paths(root)
    path = parent / category / folder_name
    if any(p.is_symlink() for p in (parent, path.parent, path)):
        raise BatchRunError("Symbolic link in private preview path")
    if path.exists():
        raise BatchRunError(f"Private preview already exists (not overwritten): {path}")
    for ancestor in (parent, path.parent):
        if ancestor.exists() and not ancestor.is_dir():
            raise BatchRunError("Private preview parent is not a directory")
    return path


def _publish(path: Path, build: Any) -> None:
    """Stage files in a new private directory; publish complete files at once."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".glass_preview_", dir=path.parent))
    try:
        build(temporary)
        if path.exists() or path.is_symlink():
            raise BatchRunError("Preview destination appeared concurrently")
        temporary.rename(path)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _entry(index: dict, doc: dict, number: int) -> dict:
    source = doc["sources"][number]
    entry = choose_index_entry(index, split="learning", image_member=source["image_member"])
    if entry["image_sha256"] != source["image_sha256"]:
        raise BatchRunError("Private dataset index differs from glass plan")
    return entry


def _page(scene: int) -> str:
    """Standalone, offline page. Users click TL/TR/BR/BL and copy JSON."""
    return r'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SAP Glass Quad | Scene __SCENE__</title><style>
body{font:16px system-ui;background:#15212b;color:#ecf3f7;margin:22px auto;max-width:1100px;padding:0 16px}
a{color:#98d8ff}button,textarea{font:inherit}button{padding:9px 15px;margin:6px;background:#34566c;color:white;border:1px solid #7399a9;border-radius:5px;cursor:pointer}
#frame{position:relative;max-width:100%;width:fit-content}#image{display:block;max-width:100%;height:auto;cursor:crosshair}
#overlay{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}textarea{width:100%;box-sizing:border-box;min-height:80px;background:#223c4d;color:white;padding:8px}
small{color:#b7d0e0}code{color:#a9e0ff}
</style></head><body>
<h2>Scene __SCENE__ · select the visible glass plane</h2>
<p>Click four corners in order: <strong>top-left → top-right → bottom-right → bottom-left</strong> of the <em>glass surface</em>.
Exclude the freezer frame, handles and non-glass surfaces. This is a manual visual decision, not auto-segmentation.</p>
<div id="frame"><img id="image" src="source.png" alt="Private source scene">
<svg id="overlay" viewBox="0 0 1000 1000" preserveAspectRatio="none"></svg></div>
<p id="status">Click corner 1: TL</p><button type="button" id="reset">Reset corners</button>
<button type="button" id="copy">Copy JSON</button>
<textarea readonly id="json" aria-label="Glass plane normalized corners"></textarea>
<p><small>Paste the copied JSON as the <code>--quad</code> argument in Git Bash:
<code>python scripts/glass_preview.py set-quad --scene __SCENE_NUM__ --quad '[[...]]'</code>.
You can select/copy the text manually if the clipboard button is blocked for local files.
The tool validates convexity; visual correctness still requires your review.</small></p>
<script>
const img=document.getElementById('image'), svg=document.getElementById('overlay');
const names=['TL','TR','BR','BL'];let corners=[];
function draw(){
 svg.replaceChildren();
 if(corners.length>1){const p=document.createElementNS('http://www.w3.org/2000/svg','polyline');
  p.setAttribute('points',corners.map(v=>`${v[0]*1000},${v[1]*1000}`).join(' '));
  p.setAttribute('fill','none');p.setAttribute('stroke','#15e3fa');p.setAttribute('stroke-width','4');svg.appendChild(p);}
 corners.forEach((v,i)=>{const dot=document.createElementNS('http://www.w3.org/2000/svg','circle');
  dot.setAttribute('cx',v[0]*1000);dot.setAttribute('cy',v[1]*1000);dot.setAttribute('r',8);
  dot.setAttribute('fill','#ffb02b');svg.appendChild(dot);});
 document.getElementById('json').value=corners.length===4?JSON.stringify(corners):'';
 document.getElementById('status').textContent=corners.length===4?
  'Four corners selected. Check the outline and copy the coordinates.':`Click corner ${corners.length+1}: ${names[corners.length]}`;
}
img.addEventListener('click',e=>{if(corners.length===4)return;const r=img.getBoundingClientRect();
 const x=Math.max(0,Math.min(1,(e.clientX-r.left)/r.width));
 const y=Math.max(0,Math.min(1,(e.clientY-r.top)/r.height));
 corners.push([Number(x.toFixed(6)),Number(y.toFixed(6))]);draw();});
document.getElementById('reset').addEventListener('click',()=>{corners=[];draw();});
document.getElementById('copy').addEventListener('click',async()=>{const field=document.getElementById('json');
 if(!field.value)return;try{await navigator.clipboard.writeText(field.value);document.getElementById('status').textContent='Copied JSON';}
 catch(e){field.focus();field.select();document.getElementById('status').textContent='Select and copy the highlighted JSON manually';}});
draw();
</script></body></html>
'''.replace("__SCENE_NUM__", str(scene)).replace("__SCENE__", f"{scene:02d}")


def create_geometry_guides(project_root: Path, *, start: int = 0, count: int = 5) -> list[Path]:
    """Create 1..5 offline clickable geometry helpers, but edit no source/plan."""
    root = _root(project_root)
    doc, _ = load_glass_plan(root)
    numbers = _source_indices(doc, start, count)
    index = read_private_index(root / "data/work/index/dataset_index.json")
    tasks = []
    for i in numbers:
        source = doc["sources"][i]
        name = f"scene_{i:02d}_{source['image_sha256'][:16]}"
        tasks.append((i, _entry(index, doc, i), _safe_private(root, "geometry_guides", name)))
    outputs = []
    for i, entry, dst in tasks:
        pair = load_indexed_pair(index, entry, data_dir=root / "data/raw")

        def build(
            temp: Path, *, source_image: Image.Image = pair.image, scene_index: int = i,
        ) -> None:
            source_image.save(temp / "source.png")
            (temp / "select_corners.html").write_text(
                _page(scene_index), encoding="utf-8"
            )

        _publish(dst, build)
        outputs.append(dst / "select_corners.html")
    return outputs


def set_scene_quad(
    project_root: Path, *, scene: int, quad: Any, replace: bool = False,
) -> Path:
    """Atomically save one manually clicked quad, backing up the previous plan."""
    root = _root(project_root)
    doc, _ = load_glass_plan(root)
    _source_indices(doc, scene, 1)
    try:
        coords = _quad(quad).tolist()
    except (GlassReflectionError, ValueError, TypeError) as exc:
        raise BatchRunError("Invalid glass polygon; expected convex TL/TR/BR/BL") from exc
    old = doc["sources"][scene]["glass_quad_norm"]
    if old is not None and not replace:
        raise BatchRunError("Glass quad exists; pass --replace to change it")
    if old == coords:
        raise BatchRunError("The proposed glass quad is already configured")
    changed = deepcopy(doc)
    changed["sources"][scene]["glass_quad_norm"] = coords
    validate_glass_plan(root, changed)
    _, path = _paths(root)
    raw = path.read_bytes()
    backup_dir = path.parent / "plan_backups"
    backup = backup_dir / f"plan_{hashlib.sha256(raw).hexdigest()}.json"
    if backup_dir.is_symlink() or backup.is_symlink():
        raise BatchRunError("Unsafe plan backup path")
    backup_dir.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        with backup.open("xb") as stream:
            stream.write(raw)
    elif backup.read_bytes() != raw:
        raise BatchRunError("Previous glass plan backup differs unexpectedly")
    payload = (json.dumps(changed, indent=2, sort_keys=True, ensure_ascii=False,
                          allow_nan=False) + "\n").encode("utf-8")
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".plan_quad_", suffix=".tmp", dir=path.parent,
            delete=False,
        ) as stream:
            temp_name = stream.name
            stream.write(payload)
        if path.read_bytes() != raw:
            raise BatchRunError("Glass plan changed concurrently; refusing to replace")
        os.replace(temp_name, path)
        temp_name = None
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)
    return path


def _quad_image(image: Image.Image, coords: list[list[float]]) -> Image.Image:
    copy = image.copy()
    drawing = ImageDraw.Draw(copy)
    w, h = image.size
    pts = [(round(x * w), round(y * h)) for x, y in coords]
    drawing.line(pts + pts[:1], fill=(0, 245, 255), width=max(2, min(w, h)//160))
    for point in pts:
        r = max(3, min(w, h)//100)
        drawing.ellipse((point[0]-r, point[1]-r, point[0]+r, point[1]+r),
                        fill=(255, 181, 38))
    return copy


def generate_glass_previews(
    project_root: Path, *, start: int = 0, count: int = 5,
) -> list[tuple[Path, dict[str, Any]]]:
    """Render 2 non-exported previews per scene; record technical QA, never approve."""
    root = _root(project_root)
    doc, _ = load_glass_plan(root)
    numbers = _source_indices(doc, start, count)
    missing = [i for i in numbers if doc["sources"][i]["glass_quad_norm"] is None]
    if missing:
        raise BatchRunError(f"Glass polygons missing for scenes {missing}; use set-quad first")
    index = read_private_index(root / "data/work/index/dataset_index.json")
    tasks = []
    for i in numbers:
        source = doc["sources"][i]
        quad = source["glass_quad_norm"]
        signature = hashlib.sha256(json.dumps(quad, separators=(",", ":")).encode()).hexdigest()[:12]
        name = f"scene_{i:02d}_{source['image_sha256'][:16]}_{signature}"
        tasks.append((i, _entry(index, doc, i), quad,
                      _safe_private(root, "effect_previews", name)))
    output = []
    for i, entry, quad, dst in tasks:
        pair = load_indexed_pair(index, entry, data_dir=root / "data/raw")
        original = np.asarray(pair.image, dtype=np.float32) / 255.0
        results: list[tuple[str, Image.Image, np.ndarray]] = []
        reviews = []
        for row in doc["runs"][2*i:2*i+2]:
            result = apply_glass_reflection(
                pair.image, quad_norm=quad, seed=row["seed"], **row["effect"]
            )
            modified = np.asarray(result.image, dtype=np.float32) / 255.0
            changed = np.any(modified != original, axis=2)
            try:
                selection = _glass_selection(
                    pair.annotations, result.effect_mask, changed, row["seed"]
                )
                qa = json.loads(validate_artifact(
                    original, modified, result.effect_mask, pair.annotations, selection
                ).to_json())
            except BatchRunError as exc:
                # A miss is itself important visual feedback; still render diagnostic preview.
                if "No product ROI intersects" not in str(exc):
                    raise
                qa = {"status": "FAIL", "failures": ["no_product_roi_affected"],
                      "metrics": {}, "manual_review_required": True,
                      "training_approved": False}
            name = row["variant"]
            results.append((name, result.image, result.effect_mask))
            reviews.append({"variant": name, "seed": row["seed"],
                            "settings": row["effect"], "technical_qa": qa})
        summary = {"schema_version": "sap-glass-preview-v1", "scene_index": i,
                   "source_sha256": entry["image_sha256"], "glass_quad_norm": quad,
                   "variants": reviews, "manual_review_required": True,
                   "training_approved": False, "stage7_export": False}

        def build(
            temp: Path, *, source_image: Image.Image = pair.image,
            glass_quad: list[list[float]] = quad,
            previews: list[tuple[str, Image.Image, np.ndarray]] = results,
            report: dict[str, Any] = summary,
        ) -> None:
            source_image.save(temp / "original.png")
            _quad_image(source_image, glass_quad).save(temp / "glass_plane.png")
            for name, image, mask in previews:
                image.save(temp / f"{name}.png")
                # 8-bit visualization only: NOT the 16-bit immutable Stage-7 mask.
                Image.fromarray(np.rint(mask * 255).astype(np.uint8), mode="L").save(
                    temp / f"{name}_mask_visual.png"
                )
            (temp / "preview_report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False,
                           allow_nan=False) + "\n", encoding="utf-8"
            )

        _publish(dst, build)
        output.append((dst, summary))
    return output
