"""Offline, private manual-QA gallery for the 100 bbox glass pilot candidates.

Revalidates completed previews; never alters preview images, SEG or pilot plans.
Browser review decisions are explicitly exported/imported by the reviewer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .glass_batch100 import _TARGET, _prepare, _verified_scene
from .runner import BatchRunError, _root

_GALLERY_SCHEMA = "sap-glass-bbox100-gallery-v1"
_HTML_START = '''<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="referrer" content="no-referrer">
<title>Glass Pilot 100 · Visual QA</title>
<style>
:root{color-scheme:dark;font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;
--bg:#10151c;--panel:#1b2530;--card:#202d38;--text:#f4f7fa;--muted:#a3b4c2;
--line:#344654;--primary:#72d4c4;--green:#8de4ac;--red:#ffab9e;--amber:#fbd48e}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text)}
header{position:sticky;top:0;z-index:5;background:#131c27ee;backdrop-filter:blur(10px);
border-bottom:1px solid var(--line);padding:16px max(20px,calc((100vw - 1600px)/2))}
h1{font-size:22px;letter-spacing:-.02em;margin:0 0 4px}p{margin:6px 0}
.subtle{color:var(--muted)}.summary{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0}
.summary span{border:1px solid var(--line);border-radius:8px;padding:8px 12px;background:var(--panel)}
.controls{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin-top:12px}
button,select,input,textarea{font:inherit;color:var(--text);background:#18232d;border:1px solid #455866;border-radius:7px}
button,select{padding:9px 12px}button{cursor:pointer}button:hover{border-color:var(--primary)}
button:focus-visible,select:focus-visible,textarea:focus-visible,input:focus-visible{outline:2px solid var(--primary)}
button.main{background:#21574e;border-color:#378d7d;color:white}button.warn{border-color:#866137}
button.active[data-status="KEEP"]{background:#245b39;border-color:var(--green)}
button.active[data-status="REJECT"]{background:#713a39;border-color:var(--red)}
button.active[data-status="NEEDS_REVIEW"]{background:#735528;border-color:var(--amber)}
main{max-width:1650px;margin:auto;padding:22px 20px 70px}
article{background:var(--panel);border:1px solid var(--line);border-radius:12px;
padding:16px;margin:0 0 18px;box-shadow:0 4px 16px #0002}
.scenehead{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;align-items:baseline;margin-bottom:12px}
.scenehead h2{font-size:16px;margin:0}.trio{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}
.figure{background:var(--card);border-radius:9px;padding:10px;min-width:0}
.figure h3{font-size:14px;margin:0 0 9px;display:flex;justify-content:space-between;gap:4px}
.figure img{width:100%;height:auto;aspect-ratio:4/3;object-fit:contain;background:#0c1219;
border-radius:7px;cursor:zoom-in;display:block}
.figure img:hover{outline:1px solid var(--primary)}.decision{display:flex;gap:5px;flex-wrap:wrap;margin-top:10px}
.decision button{flex:1;padding:7px 6px;white-space:nowrap}
textarea{display:block;width:100%;min-height:53px;margin-top:8px;padding:8px;resize:vertical}
small{color:var(--muted)}.note{padding:10px 12px;background:#152631;border:1px solid #315363;border-radius:8px}
#empty{padding:60px;text-align:center;color:var(--muted)}
#viewer{position:fixed;inset:0;z-index:9;background:#000e;display:none;align-items:center;justify-content:center;padding:30px}
#viewer.open{display:flex}#viewer img{max-width:96vw;max-height:90vh;object-fit:contain}
#viewer button{position:absolute;right:15px;top:15px;font-size:18px}
#fileinput{display:none}
@media(max-width:950px){header{position:static}.trio{grid-template-columns:1fr}article{padding:10px}}
</style></head><body>
<header><h1>Glass Pilot 100 · Visual QA</h1>
<p class="subtle">50 исходных сцен · 100 синтетических изображений · только локальный просмотр</p>
<div class="summary" id="summary" aria-live="polite"></div>
<div class="controls">
<label>Фильтр <select id="filter">
<option value="all">Все сцены</option><option value="UNREVIEWED">Есть непросмотренные</option>
<option value="KEEP">Есть «Оставить»</option><option value="REJECT">Есть «Отклонить»</option>
<option value="NEEDS_REVIEW">Есть «Проверить ещё»</option>
</select></label>
<button id="export" class="main">Скачать решения JSON</button>
<button id="import">Загрузить решения JSON</button>
<input id="fileinput" type="file" accept="application/json,.json">
</div>
<p class="note"><strong>Важно:</strong> технический PASS ≠ визуальное одобрение.
Кнопки отмечают только оценку качества; не допускают фото к обучению автоматически.
Перед закрытием страницы скачай JSON: сохранение решений в браузере не гарантируется.</p>
</header><main><div id="scenes"></div><p id="empty" hidden>Нет сцен для выбранного фильтра.</p></main>
<div id="viewer" role="dialog" aria-label="Увеличенное изображение" aria-modal="true">
<button id="close" aria-label="Закрыть">✕</button><img id="large" alt="Увеличенная фотография"></div>
<script id="gallery-data" type="application/json">'''
_HTML_END = r'''</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('gallery-data').textContent);
const valid=new Set(['UNREVIEWED','KEEP','REJECT','NEEDS_REVIEW']);
const rows=data.scenes.flatMap(scene=>scene.variants.map(v=>({...v,scene_index:scene.scene_index,
  source_sha256:scene.source_sha256})));
const ids=new Map(rows.map(row=>[row.run_id,row]));
const decisions=new Map(rows.map(row=>[row.run_id,{visual_review:'UNREVIEWED',note:''}]));
const container=document.getElementById('scenes');
const imageViewer=document.getElementById('viewer');
const large=document.getElementById('large');
function elem(tag,cls,content){const node=document.createElement(tag);if(cls)node.className=cls;
  if(content!==undefined)node.textContent=content;return node;}
function preview(src,alt){const image=elem('img');image.loading='lazy';image.src=src;image.alt=alt;
  image.addEventListener('click',()=>{large.src=src;large.alt=alt;imageViewer.classList.add('open');});
  return image;}
function statusCounts(){const counts={UNREVIEWED:0,KEEP:0,REJECT:0,NEEDS_REVIEW:0};
  for(const row of decisions.values())counts[row.visual_review]++;return counts;}
function refreshSummary(){const c=statusCounts();const parent=document.getElementById('summary');
  parent.replaceChildren();for(const [label,value] of [['Тех. QA PASS',rows.length],
  ['Не оценено',c.UNREVIEWED],['Оставить',c.KEEP],['Отклонить',c.REJECT],
  ['Проверить ещё',c.NEEDS_REVIEW]])parent.appendChild(elem('span','',label+': '+value));}
function shouldShow(scene){const f=document.getElementById('filter').value;
  return f==='all'||scene.variants.some(row=>decisions.get(row.run_id).visual_review===f);}
function draw(){refreshSummary();container.replaceChildren();let count=0;
  for(const scene of data.scenes){if(!shouldShow(scene))continue;count++;
    const card=elem('article');const head=elem('div','scenehead');
    head.appendChild(elem('h2','',`Сцена ${String(scene.scene_index).padStart(2,'0')}`));
    head.appendChild(elem('small','',`Источник SHA-256: ${scene.source_sha256.slice(0,16)}…`));
    card.appendChild(head);const trio=elem('div','trio');
    const orig=elem('section','figure');orig.appendChild(elem('h3','','Оригинал'));
    orig.appendChild(preview(scene.original,'Исходное фото сцены '+scene.scene_index));
    orig.appendChild(elem('small','','Исходные данные не меняются'));trio.appendChild(orig);
    for(const row of scene.variants){const panel=elem('section','figure');
      const title=elem('h3');title.appendChild(elem('span','',row.variant==='glass_diffuse'?'Diffuse':'Fluorescent'));
      title.appendChild(elem('small','','Технический QA: '+row.technical_qa));panel.appendChild(title);
      panel.appendChild(preview(row.image,'Сцена '+scene.scene_index+', '+row.variant));
      const choice=decisions.get(row.run_id);const controls=elem('div','decision');
      for(const [state,label] of [['KEEP','✓ Оставить'],['REJECT','✕ Отклонить'],
          ['NEEDS_REVIEW','? Позже']]){const button=elem('button','',label);button.dataset.status=state;
        if(choice.visual_review===state)button.classList.add('active');
        button.addEventListener('click',()=>{choice.visual_review=choice.visual_review===state?'UNREVIEWED':state;draw();});
        controls.appendChild(button);}panel.appendChild(controls);
      const note=elem('textarea');note.maxLength=1000;note.placeholder='Причина / замечание (необязательно)';
      note.value=choice.note;note.setAttribute('aria-label','Комментарий '+row.run_id);
      note.addEventListener('input',()=>{choice.note=note.value;});panel.appendChild(note);
      panel.appendChild(elem('small','',row.run_id));trio.appendChild(panel);}
    card.appendChild(trio);container.appendChild(card);}
  document.getElementById('empty').hidden=count!==0;}
function record(){return {schema_version:'sap-glass-bbox100-visual-review-v1',
  source_gallery_schema:data.schema_version,manual_review_required:true,training_approved:false,
  total:rows.length,items:rows.map(row=>({run_id:row.run_id,source_sha256:row.source_sha256,
    variant:row.variant,technical_qa:'PASS',visual_review:decisions.get(row.run_id).visual_review,
    note:decisions.get(row.run_id).note,training_approved:false}))};}
document.getElementById('export').addEventListener('click',()=>{
  const blob=new Blob([JSON.stringify(record(),null,2)+'\n'],{type:'application/json'});
  const url=URL.createObjectURL(blob);const link=document.createElement('a');link.href=url;
  link.download='glass_bbox100_visual_review.json';document.body.appendChild(link);link.click();
  link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);});
document.getElementById('import').addEventListener('click',()=>document.getElementById('fileinput').click());
document.getElementById('fileinput').addEventListener('change',async event=>{
  const file=event.target.files[0];if(!file)return;
  try{const imported=JSON.parse(await file.text());
    if(imported.schema_version!=='sap-glass-bbox100-visual-review-v1'||
       imported.source_gallery_schema!==data.schema_version||
       imported.manual_review_required!==true||
       imported.total!==rows.length||imported.training_approved!==false||
       !Array.isArray(imported.items)||imported.items.length!==rows.length)throw Error('Неверный формат файла');
    const validated=new Map();for(const item of imported.items){const source=ids.get(item.run_id);
      if(!source||validated.has(item.run_id)||source.source_sha256!==item.source_sha256||
         source.variant!==item.variant||item.technical_qa!=='PASS'||
         !valid.has(item.visual_review)||typeof item.note!=='string'||item.note.length>1000||
         item.training_approved!==false)throw Error('Некорректная запись: '+item.run_id);
      validated.set(item.run_id,{visual_review:item.visual_review,note:item.note});}
    for(const [id,value] of validated)decisions.set(id,value);draw();
    alert('Загружены решения: '+validated.size);
  }catch(error){alert('Ошибка импорта: '+String(error.message));}
  finally{event.target.value='';}});
document.getElementById('filter').addEventListener('change',draw);
document.getElementById('close').addEventListener('click',()=>imageViewer.classList.remove('open'));
imageViewer.addEventListener('click',event=>{if(event.target===imageViewer)imageViewer.classList.remove('open');});
document.addEventListener('keydown',event=>{if(event.key==='Escape')imageViewer.classList.remove('open');});
draw();
</script></body></html>
'''


def _gallery_destination(root: Path) -> Path:
    parent = root / 'reports' / 'glass_pilot100' / 'gallery'
    path = parent / 'index.html'
    if any(item.is_symlink() for item in (root / 'reports', parent.parent, parent, path)):
        raise BatchRunError('Unsafe symlink in glass gallery path')
    if parent.exists() and not parent.is_dir():
        raise BatchRunError('Glass gallery destination is not a directory')
    return path


def _gallery_document(root: Path) -> dict[str, Any]:
    plan, index = _prepare(root)
    scenes: list[dict[str, Any]] = []
    count = 0
    for i, source in enumerate(plan['sources']):
        if count >= _TARGET:
            break
        folder, verified, _ = _verified_scene(root, plan, index, i)
        if any(row['technical_qa']['status'] != 'PASS' for row in verified):
            raise BatchRunError(f'Glass gallery cannot include QA FAIL scene {i}')
        # Paths remain local, relative to reports/glass_pilot100/gallery/index.html.
        base = '../auto_previews/' + folder.name + '/'
        variants: list[dict[str, str]] = []
        for j, row in enumerate(verified):
            variants.append({
                'run_id': f"pg2a_{source['image_sha256'][:16]}_{j + 1}",
                'variant': row['variant'],
                'technical_qa': 'PASS',
                'image': base + row['variant'] + '.png',
            })
        scenes.append({
            'scene_index': i, 'source_sha256': source['image_sha256'],
            'original': base + 'original.png', 'variants': variants,
        })
        count += len(variants)
    if count != _TARGET or len(scenes) != 50:
        raise BatchRunError(f'Glass gallery requires 100 validated images / 50 scenes; found {count}')
    return {
        'schema_version': _GALLERY_SCHEMA,
        'total': count,
        'source_scenes': len(scenes),
        'manual_review_required': True,
        'training_approved': False,
        'scenes': scenes,
    }


def generate_glass_gallery(project_root: Path) -> tuple[Path, int]:
    """Write a local file://-compatible HTML, refusing to overwrite it."""
    root = _root(project_root)
    destination = _gallery_destination(root)
    if destination.exists() or destination.is_symlink():
        raise BatchRunError('Glass gallery already exists; never overwrite')
    data = _gallery_document(root)  # Validates all 100 *before* creating anything.
    json_text = json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    # Prevent embedded JSON from closing its <script> tag, even for future metadata.
    json_text = json_text.replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    html = _HTML_START + json_text + _HTML_END
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(html)
    return destination, data['total']
