"""Localhost-only manual visual QA editor for immutable SAP exported bundles.

Run: python scripts/manual_review_ui.py [--port 8765]
Stop: Ctrl+C. Never expose this development UI on public network interfaces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sapipeline.review.manual import ManualReviewError, read_state, save_review

_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>SAP · Manual Review</title>
<style nonce="__NONCE__">
:root{font-family:system-ui,-apple-system,Segoe UI,sans-serif;color-scheme:dark;background:#111a22;color:#edf5fa}
*{box-sizing:border-box}body{margin:0 auto;max-width:1450px;padding:20px}h1{margin:0 0 7px;font-size:24px}
a{color:#8fd1f5}.muted{color:#aabfcb}.notice{background:#29343d;border-left:3px solid #ebba6a;padding:14px;margin:14px 0}
.grid{display:grid;grid-template-columns:300px 1fr;gap:18px}.panel{background:#1d2c37;border:1px solid #394d5c;border-radius:10px;padding:16px}
input,textarea,select,button{color:#edf7ff;background:#253946;border:1px solid #506979;border-radius:7px;padding:10px;max-width:100%}
input[type=checkbox]{width:19px;height:19px}button{cursor:pointer;background:#31688a}button:disabled{opacity:.4;cursor:not-allowed}
#search{width:100%;margin-bottom:8px}#runs{max-height:75vh;overflow:auto}.run{display:block;padding:12px;margin:7px 0;width:100%;text-align:left;background:#223544}
.run small{display:block;color:#a4bac7}.run.selected{border-color:#9dd4ed;background:#29536b}
.photos{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:12px}.photos img{width:100%;height:auto;max-height:380px;object-fit:contain;background:#14222c}
figcaption{text-align:center;font-size:13px;color:#a9bed0}figure{margin:0}.row{display:flex;gap:16px;flex-wrap:wrap;margin:15px 0}
label{display:block;margin:11px 0}.checks label{display:flex;gap:10px;align-items:center}
#reviewer{width:min(400px,100%)}#reason{width:100%;height:80px;resize:vertical}#message{min-height:25px;white-space:pre-wrap}
.kpis{display:flex;gap:16px;flex-wrap:wrap;margin:16px 0}.kpis span{background:#263a48;padding:10px;border-radius:8px}
@media(max-width:920px){.grid{grid-template-columns:1fr}.photos{grid-template-columns:1fr}#runs{max-height:230px}}
</style></head><body><h1>SAP · Manual Review</h1>
<div class="muted">Localhost only · private reviews.json · no uploads · no automatic training approval</div>
<div class="notice"><strong>Technical QA is not human approval.</strong> Inspect the original, modified image and full review sheet. Save one explicit verdict. Reopen or refresh if files changed.</div>
<div class="kpis"><span id="count">Runs —</span><span id="approved">Approved —</span><span id="current">Selected —</span></div>
<div class="grid"><aside class="panel"><h2>Runs</h2><input id="search" type="search" placeholder="Search run / effect"><div id="runs"></div></aside>
<main class="panel"><h2 id="heading">Choose a run</h2><div id="meta" class="muted"></div>
<div class="photos"><figure><img id="original" alt="Original photograph"><figcaption>Original</figcaption></figure>
<figure><img id="modified" alt="Modified photograph"><figcaption>Modified</figcaption></figure></div>
<div class="row"><a id="sheet" target="_blank" rel="noopener">Open full review sheet ↗</a></div>
<form id="form" autocomplete="off"><label>Reviewer <input id="reviewer" maxlength="80" placeholder="Human reviewer name"></label>
<div class="checks"><label><input id="product_identity" type="checkbox"> Product identity / readable packaging</label>
<label><input id="glare_naturalness" type="checkbox"> Illumination looks physically plausible</label>
<label><input id="bbox_consistency" type="checkbox"> Bounding boxes and labels remain appropriate</label></div>
<label>Verdict <select id="verdict"><option value="needs_review">needs_review</option>
<option value="approved">approved</option><option value="rejected">rejected</option></select></label>
<label>Reason / notes <textarea id="reason" maxlength="1000" placeholder="Required for rejected; optional otherwise"></textarea></label>
<button id="save" type="submit">Save decision for selected run</button>
<p id="message" role="status" aria-live="polite"></p></form></main></div>
<script nonce="__NONCE__">
'use strict';const csrf=__CSRF__;const byId=x=>document.getElementById(x);
let snapshot=null,selected=null;const checks=['product_identity','glare_naturalness','bbox_consistency'];
function message(s){byId('message').textContent=s}
async function refresh(preferred){const res=await fetch('/api/state',{cache:'no-store'});const data=await res.json();
 if(!res.ok)throw Error(data.error||'Cannot load');snapshot=data;renderList();
 let rows=data.runs;selectRun(rows.find(r=>r.run_id===preferred)||rows[0]);}
function renderList(){let q=byId('search').value.toLowerCase();const box=byId('runs');box.replaceChildren();
 byId('count').textContent='Runs: '+snapshot.totals.runs;byId('approved').textContent='Approved: '+snapshot.totals.approved;
 for(const r of snapshot.runs){if(!(r.run_id+' '+r.effect).toLowerCase().includes(q))continue;
 const b=document.createElement('button');b.type='button';b.className='run'+(r.run_id===selected?' selected':'');
 b.textContent=r.run_id;const d=document.createElement('small');d.textContent=r.effect+' · '+r.qa+' · '+r.review_status;
 b.append(d);b.addEventListener('click',()=>selectRun(r));box.append(b)}}
function selectRun(r){if(!r)return;selected=r.run_id;renderList();byId('heading').textContent=r.run_id;
 byId('meta').textContent=r.effect+' · '+r.source_split+' · QA '+r.qa+' · '+r.review_status+
 ' · changed '+r.changed_fraction_percent+'% · mask '+r.mask_coverage_percent+'%';
 byId('current').textContent='Selected: '+r.run_id;
 for(const type of ['original','modified'])byId(type).src='/media/'+encodeURIComponent(r.run_id)+'/'+type;
 byId('sheet').href='/media/'+encodeURIComponent(r.run_id)+'/sheet';
 for(const key of checks)byId(key).checked=r.review.checks?.[key]===true;
 byId('verdict').value=r.review.verdict;byId('reviewer').value=r.review.reviewer||'';
 byId('reason').value=r.review.reason||'';message('No changes saved yet.');}
byId('search').addEventListener('input',renderList);
byId('form').addEventListener('submit',async e=>{e.preventDefault();if(!snapshot||!selected)return;
 const run=snapshot.runs.find(r=>r.run_id===selected);const payload={run_id:selected,manifest_sha256:run.manifest_sha256,
 reviews_etag:snapshot.reviews_etag,verdict:byId('verdict').value,reviewer:byId('reviewer').value,
 reason:byId('reason').value,checks:Object.fromEntries(checks.map(k=>[k,byId(k).checked]))};
 byId('save').disabled=true;message('Saving…');try{
 const res=await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json','X-CSRF-Token':csrf},body:JSON.stringify(payload)});
 const data=await res.json();if(!res.ok)throw Error(data.error||'Save failed');await refresh(selected);
 message('Saved '+data.run_id+': '+data.verdict+' (private reviews.json). Training not approved.');
 }catch(err){message('NOT SAVED: '+err.message+' — refresh page if the review changed.');}
 finally{byId('save').disabled=false}});
refresh(null).catch(e=>message('Cannot load records: '+e.message));
</script></body></html>'''


def make_page(csrf: str, nonce: str) -> bytes:
    """Private UI: dynamic content enters via textContent, never HTML injection."""
    return _HTML.replace("__NONCE__", nonce).replace("__CSRF__", json.dumps(csrf)).encode("utf-8")


def make_handler(project_root: Path, *, csrf: str, nonce: str):
    """Request handler bound to immutable local project root and secret token."""
    root = project_root
    known_runs = {run["run_id"] for run in read_state(root)["runs"]}

    class ManualReviewHandler(BaseHTTPRequestHandler):
        def log_message(self, _fmt, *_args):
            pass  # Do not put private image names or paths in terminal logs.

        def _send(self, code: int, data: bytes, mime: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", (
                "default-src 'none'; img-src 'self'; style-src 'nonce-" + nonce
                + "'; script-src 'nonce-" + nonce
                + "'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'"
            ))
            self.end_headers()
            self.wfile.write(data)

        def _json(self, code: int, data: dict) -> None:
            self._send(code, json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def _host_ok(self) -> bool:
            return self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"

        def do_GET(self) -> None:
            if not self._host_ok():
                self._json(403, {"error": "Invalid local host"})
                return
            route = urlsplit(self.path).path
            try:
                if route == "/":
                    self._send(200, make_page(csrf, nonce), "text/html; charset=utf-8")
                elif route == "/api/state":
                    state = read_state(root)
                    known_runs.clear()
                    known_runs.update(run["run_id"] for run in state["runs"])
                    self._json(200, state)
                elif route.startswith("/media/"):
                    match = re.fullmatch(
                        r"/media/([A-Za-z0-9][A-Za-z0-9_-]{0,63})/(original|modified|sheet)", route,
                    )
                    if not match:
                        raise ManualReviewError("Unsupported media request")
                    run_id, kind = match.groups()
                    if run_id not in known_runs:
                        raise ManualReviewError("Unknown run")
                    run_path = root / "outputs/sap_runs" / run_id
                    manifest_path = run_path / "manifest.json"
                    if run_path.is_symlink() or manifest_path.is_symlink():
                        raise ManualReviewError("Symlinked run path")
                    manifest_bytes = manifest_path.read_bytes()
                    manifest = json.loads(manifest_bytes)
                    if manifest.get("run_id") != run_id or manifest.get("schema_version") != "sap-artifact-v1":
                        raise ManualReviewError("Manifest changed")
                    if kind == "sheet":
                        file = root / "reports/visual_review" / run_id / "review_sheet.png"
                        metrics_path = file.parent / "review_metrics.json"
                        if file.parent.is_symlink() or metrics_path.is_symlink():
                            raise ManualReviewError("Symlinked report path")
                        metrics = json.loads(metrics_path.read_bytes())
                        if metrics.get("manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest():
                            raise ManualReviewError("Stale review sheet")
                    else:
                        key = "original" if kind == "original" else "image"
                        relative = manifest["output"][key]
                        # File links were checked by Batch Gallery, but re-check confinement.
                        base = root / "outputs/sap_runs" / run_id
                        file = base / relative
                        if not file.resolve().is_relative_to(base.resolve()):
                            raise ManualReviewError("Unsafe media path")
                    if file.is_symlink() or not file.is_file():
                        raise ManualReviewError("Missing media")
                    data = file.read_bytes()
                    expected = (metrics.get("review_sheet_sha256") if kind == "sheet"
                                else manifest["output_sha256"][key])
                    if hashlib.sha256(data).hexdigest() != expected:
                        raise ManualReviewError("Media checksum mismatch")
                    self._send(200, data, "image/png")
                else:
                    self._json(404, {"error": "Not found"})
            except (ManualReviewError, OSError, ValueError, KeyError) as exc:
                self._json(409, {"error": str(exc)})

        def do_POST(self) -> None:
            origin = f"http://127.0.0.1:{self.server.server_port}"
            if (not self._host_ok() or self.path != "/api/review"
                    or self.headers.get("Origin") != origin
                    or self.headers.get("X-CSRF-Token") != csrf
                    or self.headers.get("Content-Type") != "application/json"):
                self._json(403, {"error": "Forbidden request"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 8192:
                    raise ManualReviewError("Invalid request size")
                obj = json.loads(self.rfile.read(size))
                self._json(200, save_review(root, obj))
            except (ManualReviewError, OSError, ValueError, TypeError, KeyError) as exc:
                self._json(409, {"error": str(exc)})

    return ManualReviewHandler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 8.3 private, localhost-only review editor")
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be between 1024 and 65535")
    try:
        root = Path(args.project_root).resolve(strict=True)
        state = read_state(root)
        token = secrets.token_urlsafe(32)
        nonce = secrets.token_urlsafe(24)
        handler = make_handler(root, csrf=token, nonce=nonce)
        with HTTPServer(("127.0.0.1", args.port), handler) as server:
            print(f"SAP Manual Review UI: {state['totals']['runs']} runs")
            print(f"Open: http://127.0.0.1:{server.server_port}/")
            print("Private localhost service; Ctrl+C to stop. Never forward its port.")
            try:
                server.serve_forever(poll_interval=0.2)
            except KeyboardInterrupt:
                print("Stopped.")
        return 0
    except (ManualReviewError, OSError, ValueError) as exc:
        print(f"Manual Review UI ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
