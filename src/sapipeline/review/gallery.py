"""Local-only, read-only input validation and offline HTML gallery rendering.

The gallery is a *view*, not a QA gate. It never approves training samples,
changes review decisions, or modifies an exported six-file bundle.
"""

from __future__ import annotations

import hashlib
import html
import io
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, ImageOps

_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_SHA = re.compile(r"[a-f0-9]{64}\Z")
_REQUIRED_OUTPUTS = {"original", "image", "annotation", "effect_mask", "qa_report"}


class BatchGalleryError(ValueError):
    """Fail-closed rejection of invalid, stale, or unsafe gallery inputs."""


@dataclass(frozen=True, slots=True)
class GalleryBuild:
    html_page: bytes
    summary: dict[str, Any]
    thumbnails: dict[str, bytes]


def _json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise BatchGalleryError(f"Missing or unsafe JSON: {path.name}")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError, OSError) as exc:
        raise BatchGalleryError(f"Invalid JSON: {path.name}") from exc
    if not isinstance(obj, dict):
        raise BatchGalleryError(f"JSON object required: {path.name}")
    return obj


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_file(root: Path, relative: str) -> Path:
    if not isinstance(relative, str):
        raise BatchGalleryError("Output path must be a string")
    path = PurePosixPath(relative)
    if (path.is_absolute() or path.as_posix() != relative or
            any(p in (".", "..", "") for p in path.parts) or
            "\\" in relative or ":" in relative or not path.parts):
        raise BatchGalleryError("Unsafe output path in manifest")
    candidate = root
    for part in path.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise BatchGalleryError("Symlinked artifact path")
    if not candidate.is_file():
        raise BatchGalleryError(f"Missing artifact: {relative}")
    return candidate


def _read_reviews(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "reports/qa_acceptance/reviews.json"
    if not path.exists() and not path.is_symlink():
        return {}
    doc = _json_object(path)
    if doc.get("schema_version") != "sap-manual-reviews-v1" or not isinstance(doc.get("reviews"), list):
        raise BatchGalleryError("Invalid manual reviews schema")
    reviews = {}
    for item in doc["reviews"]:
        if not isinstance(item, dict) or not isinstance(item.get("run_id"), str):
            raise BatchGalleryError("Malformed manual review")
        run_id = item["run_id"]
        if not _RUN_ID.fullmatch(run_id) or run_id in reviews:
            raise BatchGalleryError("Invalid or duplicated manual review run_id")
        reviews[run_id] = item
    return reviews


def _review_status(review: dict[str, Any] | None, manifest_sha: str) -> str:
    if review is None:
        return "needs_review"
    if review.get("manifest_sha256") != manifest_sha:
        return "stale_review"
    verdict = review.get("verdict")
    if verdict == "rejected":
        return "rejected"
    if verdict == "needs_review":
        return "needs_review"
    if verdict != "approved":
        return "invalid_review"
    checks = review.get("checks")
    required = ("bbox_consistency", "glare_naturalness", "product_identity")
    if (not isinstance(checks, dict) or
            any(checks.get(k) is not True for k in required) or
            not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip()):
        return "incomplete_approval"
    return "approved"


def _thumb(file: Path, *, size: tuple[int, int] = (460, 300)) -> bytes:
    try:
        with Image.open(file) as im:
            if im.format != "PNG":
                raise BatchGalleryError("Exported image must be PNG")
            original = ImageOps.exif_transpose(im.convert("RGB"))
            original.thumbnail(size, Image.Resampling.LANCZOS)
            background = Image.new("RGB", size, (28, 35, 44))
            background.paste(original, ((size[0]-original.width)//2, (size[1]-original.height)//2))
            buffer = io.BytesIO()
            background.save(buffer, format="JPEG", quality=83, optimize=True)
            return buffer.getvalue()
    except (OSError, ValueError) as exc:
        raise BatchGalleryError(f"Cannot decode PNG: {file.name}") from exc


def _as_number(value: Any, label: str, *, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise BatchGalleryError(f"Invalid numeric metric: {label}")
    if value < 0 or (maximum is not None and value > maximum):
        raise BatchGalleryError(f"Metric out of bounds: {label}")
    return float(value)


def _discover(root: Path) -> list[Path]:
    runs = root / "outputs/sap_runs"
    if runs.is_symlink() or not runs.is_dir():
        raise BatchGalleryError("Missing or unsafe outputs/sap_runs directory")
    entries = []
    for path in sorted(runs.iterdir()):
        if path.is_symlink():
            raise BatchGalleryError(f"Unsafe symlink in runs: {path.name}")
        if not path.is_dir():
            continue  # exported ZIP backups are not run directories
        manifest = path / "manifest.json"
        if not _RUN_ID.fullmatch(path.name):
            if manifest.exists() or manifest.is_symlink():
                raise BatchGalleryError("Invalid run directory containing a manifest")
            continue
        if not manifest.is_file() or manifest.is_symlink():
            raise BatchGalleryError(f"Incomplete exported run: {path.name}")
        entries.append(path)
    if not entries:
        raise BatchGalleryError("No exported runs found")
    return entries


def _extract_run(root: Path, run: Path, reviews: dict[str, dict[str, Any]]) -> tuple[dict, dict[str, bytes]]:
    name = run.name
    manifest_path = run / "manifest.json"
    manifest = _json_object(manifest_path)
    if manifest.get("schema_version") != "sap-artifact-v1" or manifest.get("run_id") != name:
        raise BatchGalleryError(f"Manifest schema/run_id mismatch: {name}")
    paths = manifest.get("output")
    hashes = manifest.get("output_sha256")
    if not isinstance(paths, dict) or set(paths) != _REQUIRED_OUTPUTS or not isinstance(hashes, dict) or set(hashes) != _REQUIRED_OUTPUTS:
        raise BatchGalleryError(f"Invalid exported-file map: {name}")
    files = {}
    for key in _REQUIRED_OUTPUTS:
        file = _safe_file(run, paths[key])
        expected = hashes.get(key)
        if not isinstance(expected, str) or not _SHA.fullmatch(expected) or _digest(file.read_bytes()) != expected:
            raise BatchGalleryError(f"SHA256 failed for {name}/{key}")
        files[key] = file
    manifest_sha = _digest(manifest_path.read_bytes())
    qa = _json_object(files["qa_report"])
    if (qa.get("schema_version") != "sap-qa-v1" or
            qa.get("status") != manifest.get("qa", {}).get("status") or
            qa.get("status") not in ("PASS", "FAIL")):
        raise BatchGalleryError(f"QA/manifest mismatch: {name}")
    review_dir = root / "reports/visual_review" / name
    if review_dir.is_symlink() or not review_dir.is_dir():
        raise BatchGalleryError(f"Missing review sheet for {name}; run render_review_sheet.py --all")
    metrics_file = review_dir / "review_metrics.json"
    sheet_file = review_dir / "review_sheet.png"
    metrics = _json_object(metrics_file)
    if sheet_file.is_symlink() or not sheet_file.is_file():
        raise BatchGalleryError(f"Missing or unsafe review_sheet.png: {name}")
    if (metrics.get("schema_version") != "sap-visual-review-v1" or
            metrics.get("run_id") != name or
            metrics.get("manifest_sha256") != manifest_sha or
            metrics.get("technical_qa_status") != qa["status"] or
            not isinstance(metrics.get("review_sheet_sha256"), str) or
            _digest(sheet_file.read_bytes()) != metrics["review_sheet_sha256"]):
        raise BatchGalleryError(f"Stale or modified visual report: {name}")
    effect = metrics.get("effect")
    metrics_values = metrics.get("metrics")
    source = manifest.get("source")
    if (not isinstance(effect, dict) or not isinstance(metrics_values, dict) or
            not isinstance(source, dict) or not isinstance(manifest.get("effects"), list) or
            len(manifest["effects"]) != 1 or effect != manifest["effects"][0] or
            source.get("image_sha256") != metrics.get("source_sha256") or
            source.get("split") != metrics.get("source_split")):
        raise BatchGalleryError(f"Effect/source metadata mismatch: {name}")
    source_sha = source["image_sha256"]
    split = source["split"]
    if not isinstance(source_sha, str) or not _SHA.fullmatch(source_sha) or split not in ("learning", "verification"):
        raise BatchGalleryError(f"Invalid source metadata: {name}")
    kind = effect.get("kind")
    schema = effect.get("schema_version")
    seed = effect.get("seed")
    if not isinstance(kind, str) or not isinstance(schema, str) or type(seed) is not int:
        raise BatchGalleryError(f"Invalid effect metadata: {name}")
    coverage = _as_number(metrics_values.get("mask_coverage_percent"), "coverage", maximum=100)
    changed = _as_number(metrics_values.get("changed_fraction_percent"), "changed", maximum=100)
    outside = _as_number(metrics_values.get("changed_outside_mask"), "outside")
    mean_diff = _as_number(metrics_values.get("mean_abs_diff_all_channels_255"), "mean diff", maximum=255)
    photo_size = manifest.get("image_size_px")
    if not isinstance(photo_size, list) or len(photo_size) != 2 or any(type(x) is not int or x <= 0 for x in photo_size):
        raise BatchGalleryError(f"Invalid dimensions: {name}")
    thumbs = {
        f"thumbs/{name}_original.jpg": _thumb(files["original"]),
        f"thumbs/{name}_modified.jpg": _thumb(files["image"]),
    }
    return ({
        "run_id": name, "effect": f"{schema}:{kind}", "kind": kind,
        "seed": seed, "source_split": split, "source_sha256": source_sha,
        "qa": qa["status"], "manual": _review_status(reviews.get(name), manifest_sha),
        "mask_coverage_percent": round(coverage, 3),
        "changed_fraction_percent": round(changed, 3),
        "changed_outside_mask": int(outside),
        "mean_abs_diff_255": round(mean_diff, 2),
        "original_link": f"../../outputs/sap_runs/{name}/{paths['original']}",
        "modified_link": f"../../outputs/sap_runs/{name}/{paths['image']}",
        "sheet_link": f"../visual_review/{name}/review_sheet.png",
        "metrics_link": f"../visual_review/{name}/review_metrics.json",
    }, thumbs)


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _bar(label: str, count: int, maximum: int) -> str:
    pct = 100 * count / max(1, maximum)
    return (f'<div class="hrow"><span>{_esc(label)}</span>'
            f'<div class="bar"><div style="width:{pct:.2f}%"></div></div>'
            f'<b>{count}</b></div>')


def _html(records: list[dict], summary: dict) -> bytes:
    effects = summary["effects"]
    reviews = summary["reviews"]
    effect_options = ''.join(
        f'<option value="{_esc(k)}">{_esc(k)} ({v})</option>'
        for k, v in sorted(effects.items())
    )
    qa_options = ''.join(f'<option value="{_esc(s)}">{_esc(s)}</option>'
                         for s in sorted({r["qa"] for r in records}))
    review_options = ''.join(f'<option value="{_esc(s)}">{_esc(s)}</option>'
                             for s in sorted(reviews))
    effect_bars = ''.join(_bar(k, v, len(records)) for k, v in sorted(effects.items()))
    review_bars = ''.join(_bar(k, v, len(records)) for k, v in sorted(reviews.items()))
    cards = []
    for r in records:
        name = r["run_id"]
        t1 = f"thumbs/{name}_original.jpg"
        t2 = f"thumbs/{name}_modified.jpg"
        changed = r["changed_fraction_percent"]
        coverage = r["mask_coverage_percent"]
        cards.append(f'''<article class="card" data-run="{_esc(name)}" data-effect="{_esc(r['effect'])}"
 data-qa="{_esc(r['qa'])}" data-review="{_esc(r['manual'])}" data-split="{_esc(r['source_split'])}">
 <div class="card-head"><strong>{_esc(name)}</strong><span class="pill">{_esc(r['qa'])} · {_esc(r['manual'])}</span></div>
 <div class="meta">{_esc(r['effect'])} · {_esc(r['source_split'])} · seed {_esc(r['seed'])}</div>
 <div class="photos"><a href="{_esc(r['original_link'])}" target="_blank" rel="noopener"><img loading="lazy" src="{_esc(t1)}" alt="Original"><small>Original</small></a>
 <a href="{_esc(r['modified_link'])}" target="_blank" rel="noopener"><img loading="lazy" src="{_esc(t2)}" alt="Modified"><small>Modified</small></a></div>
 <div class="statline"><span>Changed {changed:.2f}%</span><span>Mask {coverage:.2f}%</span></div>
 <div class="meters"><div style="width:{changed:.2f}%"></div></div>
 <p class="detail">Mean diff {_esc(r['mean_abs_diff_255'])}/255 · Outside mask {_esc(r['changed_outside_mask'])} px</p>
 <div class="actions"><a href="{_esc(r['sheet_link'])}" target="_blank" rel="noopener">Full review sheet ↗</a>
 <a href="{_esc(r['metrics_link'])}" target="_blank" rel="noopener">Metrics JSON ↗</a></div>
 </article>''')
    page = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SAP Batch Gallery — private visual QA</title><style>
:root{color-scheme:dark;font-family:system-ui,-apple-system,Segoe UI,sans-serif;background:#101820;color:#eaf0f4}
*{box-sizing:border-box}body{margin:0 auto;max-width:1500px;padding:26px}h1{margin:0 0 8px;font-size:28px}h2{font-size:17px}
.sub{color:#a3bac9;margin:0 0 20px}.warning{background:#293039;border-left:3px solid #edc475;padding:13px 16px;border-radius:6px;margin:16px 0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}.kpi{background:#1d2a33;padding:16px;border:1px solid #33444f;border-radius:12px}
.kpi b{font-size:30px;display:block}.kpi small,.meta,.detail{color:#a4b8c6}.charts{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:18px 0}
.panel{background:#1d2a33;border:1px solid #33444f;border-radius:12px;padding:18px}.hrow{display:grid;grid-template-columns:minmax(80px,2fr) 2fr 36px;align-items:center;gap:10px;font-size:12px;margin:9px 0;overflow-wrap:anywhere}
.bar,.meters{height:10px;background:#344754;border-radius:10px;overflow:hidden}.bar div,.meters div{height:100%;background:#62aac9}
.filters{display:flex;flex-wrap:wrap;gap:9px;padding:18px 0;position:sticky;top:0;background:#101820;z-index:2}
input,select,button{background:#263744;color:#ecf5fc;border:1px solid #49606e;border-radius:8px;padding:11px;max-width:100%}
#q{flex:1;min-width:150px}.results{display:grid;grid-template-columns:repeat(auto-fill,minmax(365px,1fr));gap:14px}
.card{background:#1c2933;border:1px solid #354955;border-radius:12px;padding:15px;min-width:0}.card-head{display:flex;justify-content:space-between;align-items:center;gap:5px;overflow-wrap:anywhere}
.pill{font-size:11px;border-radius:99px;background:#344554;padding:6px 9px;white-space:nowrap}.meta{font-size:12px;margin:6px 0 12px;overflow-wrap:anywhere}
.photos{display:grid;grid-template-columns:1fr 1fr;gap:9px}.photos a{background:#13202a;border-radius:8px;overflow:hidden;color:#bddced;text-decoration:none}
.photos img{display:block;width:100%;aspect-ratio:460/300;object-fit:contain}.photos small{display:block;text-align:center;padding:5px}
.statline{display:flex;justify-content:space-between;margin:15px 0 7px;font-size:13px}.detail{font-size:12px;margin:9px 0}
.actions{display:flex;gap:12px;flex-wrap:wrap;margin-top:12px}.actions a{color:#8ccff0;font-size:13px}
footer{color:#9faeb7;margin-top:30px;font-size:12px}@media(max-width:760px){.charts{grid-template-columns:1fr}.filters{position:static}}
</style></head><body>
<h1>SAP · Batch Gallery</h1><p class="sub">Offline visual review | read-only snapshot | no uploads | not training approval</p>
<div class="warning"><b>Technical QA ≠ visual approval.</b> Manual verdicts reflect reviews.json at gallery generation time. Regenerate after decisions change. Stage 8 and training eligibility are evaluated separately.</div>
<section class="kpis">
'''
    for key, val in (("Runs",summary["runs"]),("Unique source scenes",summary["unique_sources"]),
                     ("QA PASS",summary["qa_pass"]),("Manual approved",summary["manual_approved"]),
                     ("Cross-split duplicates",summary["cross_split_duplicate_sources"])):
        page += f'<div class="kpi"><b>{val}</b><small>{_esc(key)}</small></div>'
    page += (f'</section><section class="charts"><div class="panel"><h2>Effects</h2>{effect_bars}</div>'
             f'<div class="panel"><h2>Manual review</h2>{review_bars}</div></section>')
    page += f'''<nav class="filters"><input id="q" type="search" placeholder="Search run ID / effect" aria-label="Search">
<select id="effect" aria-label="Effect"><option value="">All effects</option>{effect_options}</select>
<select id="qa" aria-label="QA"><option value="">All QA</option>{qa_options}</select>
<select id="review" aria-label="Review"><option value="">All reviews</option>{review_options}</select>
<select id="split" aria-label="Split"><option value="">All splits</option><option>learning</option><option>verification</option></select>
<button id="reset" type="button">Reset</button><span id="visible"></span></nav><main class="results">{''.join(cards)}</main>
''' + '''<footer>Local private report. Original exported bundles and manual reviews remain unchanged. Generated images are not automatically approved for YOLO training.</footer>
<script>
const fields=['q','effect','qa','review','split'];const controls=Object.fromEntries(fields.map(k=>[k,document.getElementById(k)]));
const cards=[...document.querySelectorAll('.card')];function update(){const q=controls.q.value.toLowerCase().trim();let shown=0;
for(const card of cards){const data=card.dataset;const show=(!q||(data.run+' '+data.effect).toLowerCase().includes(q))&&
(!controls.effect.value||data.effect===controls.effect.value)&&(!controls.qa.value||data.qa===controls.qa.value)&&
(!controls.review.value||data.review===controls.review.value)&&(!controls.split.value||data.split===controls.split.value);
card.hidden=!show;if(show)shown++;}document.getElementById('visible').textContent=shown+' / '+cards.length+' shown';}
for(const el of Object.values(controls))el.addEventListener('input',update);
document.getElementById('reset').addEventListener('click',()=>{for(const c of Object.values(controls))c.value='';update()});update();
</script></body></html>'''
    return page.encode("utf-8")


def build_batch_gallery(project_root: Path) -> GalleryBuild:
    """Validate existing derived reports and build an offline gallery in memory.

    No filesystem writes are performed; the CLI owns atomic output publishing.
    Gallery checks are not a replacement for the Stage-8 acceptance audit.
    """
    root = Path(project_root)
    if (not root.is_absolute() or root.is_symlink() or
            not (root / "pyproject.toml").is_file()):
        raise BatchGalleryError("project_root must be an existing absolute SAP project")
    if any((root / relative).is_symlink() for relative in
           ("reports", "reports/visual_review", "reports/qa_acceptance", "outputs")):
        raise BatchGalleryError("Unsafe symlinked project data directory")
    root = root.resolve(strict=True)
    reviews = _read_reviews(root)
    records, thumbnails = [], {}
    for run in _discover(root):
        data, thumbs = _extract_run(root, run, reviews)
        records.append(data)
        thumbnails.update(thumbs)
    splits: dict[str, set[str]] = {}
    for r in records:
        splits.setdefault(r["source_sha256"], set()).add(r["source_split"])
    cross_split = sum(len(s) > 1 for s in splits.values())
    effects: dict[str, int] = {}
    verdicts: dict[str, int] = {}
    for r in records:
        effects[r["effect"]] = effects.get(r["effect"], 0) + 1
        verdicts[r["manual"]] = verdicts.get(r["manual"], 0) + 1
    summary = {
        "schema_version": "sap-batch-gallery-v1",
        "runs": len(records), "unique_sources": len(splits),
        "qa_pass": sum(r["qa"] == "PASS" for r in records),
        "manual_approved": verdicts.get("approved", 0),
        "cross_split_duplicate_sources": cross_split,
        "effects": effects, "reviews": verdicts,
        "records": records, "training_approved": False,
        "stage8_accepted": False,
        "note": "Visual snapshot only. Run Stage-8 acceptance audit independently.",
    }
    return GalleryBuild(_html(records, summary), summary, thumbnails)
