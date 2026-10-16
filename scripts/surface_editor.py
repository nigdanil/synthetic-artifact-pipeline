"""Local-only, manual Surface Editor V1. No remote services or source writes."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sapipeline.surfaces import SurfaceError, loads_document

MAX_IMAGE = 25_000_000
MAX_JSON = 2_000_000

HTML = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer">
<title>SAP Surface Editor V1</title>
<style>
:root{font:14px system-ui,Arial,sans-serif;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:#111a25;color:#ebf2fa}
header{padding:15px 22px;background:#1c2938;border-bottom:1px solid #34465a}
h1{margin:0 0 5px;font-size:21px}
small,.muted{color:#a9bacb}
main{display:grid;grid-template-columns:minmax(0,1fr) 310px;gap:18px;padding:18px}
.toolbar{display:flex;gap:9px;flex-wrap:wrap;align-items:center;margin-bottom:12px}
button,select,input,textarea{font:inherit;background:#263749;color:white;
border:1px solid #547089;border-radius:6px;padding:8px}
button{cursor:pointer}
button:hover{border-color:#79d5e7}
button:disabled{opacity:.4;cursor:default}
input[type=file]{max-width:255px}
#workspace{background:#0b111b;min-height:400px;padding:12px;overflow:auto;
border:1px solid #34465a;border-radius:10px}
#stage{position:relative;display:none;width:fit-content;max-width:100%}
#photo{display:block;max-width:100%;height:auto}
#overlay{position:absolute;inset:0;width:100%;height:100%;
cursor:crosshair;touch-action:none}
aside{background:#1d2b39;border:1px solid #34465a;border-radius:10px;
padding:15px;align-self:start}
aside label{display:block;margin:14px 0 5px}
aside input[type=text],aside textarea,aside select{width:100%}
aside textarea{min-height:70px;resize:vertical}
#surfaces{margin-top:14px;display:grid;gap:6px}
#surfaces button{text-align:left;width:100%}
#surfaces button.active{border-color:#81dbc3;background:#264d4b}
#status{margin-top:10px;padding:10px;background:#25394c;
border-radius:7px;white-space:pre-wrap;min-height:40px}
.check{display:flex;align-items:center;gap:8px}
.check input{width:auto}
@media(max-width:850px){main{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
<h1>SAP · Surface Editor V1</h1>
<div class="muted">Локальная разметка физических поверхностей · без ML-сегментации</div>
</header>
<main>
<section>
<div class="toolbar">
<label>Фото <input id="imageFile" type="file" accept="image/jpeg,image/png,image/webp"></label>
<button id="add" disabled>+ Поверхность</button>
<button id="undo" disabled>↶ Отменить</button>
<button id="redo" disabled>↷ Повторить</button>
<button id="cancel">Сбросить точки</button>
</div>
<div id="workspace">
<div id="stage">
<img id="photo" alt="Исходное изображение">
<svg id="overlay"></svg>
</div>
<p id="empty">Выберите исходное изображение.</p>
</div>
<div class="toolbar" style="margin-top:12px">
<button id="export" disabled>Скачать разметку JSON</button>
<label>Импорт JSON <input id="jsonFile" type="file" accept=".json,application/json"></label>
</div>
<div id="status" role="status">Ожидание изображения.</div>
<small>Четыре клика: TL → TR → BR → BL. Выберите существующую
поверхность и перетаскивайте её вершины. Разметка не изменяет JPEG и SEG/YOLO.</small>
</section>
<aside>
<strong>Поверхности</strong>
<div id="surfaces"></div>
<div id="details" hidden>
<label for="material">Материал</label>
<select id="material">
<option value="glass">Стекло</option>
<option value="metal">Металл</option>
<option value="plastic">Пластик</option>
<option value="wood">Дерево</option>
<option value="unknown">Неизвестно</option>
</select>
<label class="check">
<input id="reviewed" type="checkbox"> Подтверждено оператором
</label>
<label for="note">Комментарий</label>
<textarea id="note" maxlength="500"></textarea>
<button id="remove">Удалить поверхность</button>
</div>
<p class="muted">Редактор сохраняет только геометрию и тип материала.
Допуск к обучению YOLO здесь не выполняется.</p>
</aside>
</main>
<script>
'use strict';
const $=id=>document.getElementById(id);
const NS='http://www.w3.org/2000/svg';
let source=null, items=[],selected=null,draft=null,drag=null;
let history=[],future=[],imageUrl=null,nextId=1;

function message(text){$('status').textContent=text;}
function clone(value){return JSON.parse(JSON.stringify(value));}
function state(){return {items:clone(items),selected};}
function remember(){
  history.push(state());
  if(history.length>100)history.shift();
  future=[];
}
function restore(value){
  items=clone(value.items);selected=value.selected;draft=null;render();
}
function point(e){
  const r=$('overlay').getBoundingClientRect();
  return [
    Number(Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)).toFixed(6)),
    Number(Math.max(0,Math.min(1,(e.clientY-r.top)/r.height)).toFixed(6))
  ];
}
function current(){return items.find(x=>x.surface_id===selected)||null;}
function documentData(){
  return {
    schema_version:'sap-surfaces-v1',
    source:source,
    surfaces:clone(items),
    manual_review_required:true,
    training_approved:false
  };
}
async function validate(doc){
  const response=await fetch('/validate',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify(doc)
  });
  const result=await response.json();
  if(!response.ok)throw Error(result.error||'Ошибка валидации');
  return result.document;
}
function element(tag,attributes={}){
  const node=document.createElementNS(NS,tag);
  for(const [key,value] of Object.entries(attributes))
    node.setAttribute(key,String(value));
  return node;
}
function render(){
  const svg=$('overlay');
  svg.replaceChildren();
  if(source)svg.setAttribute('viewBox',`0 0 ${source.width} ${source.height}`);
  const w=source?.width||1,h=source?.height||1;
  const colors={glass:'#51e7f4',metal:'#ffba69',
                plastic:'#d59bff',wood:'#9de192',unknown:'#aaaaaa'};
  const list=$('surfaces');
  list.replaceChildren();
  for(const item of items){
    const active=item.surface_id===selected;
    const coords=item.quad_norm.map(p=>`${p[0]*w},${p[1]*h}`).join(' ');
    const polygon=element('polygon',{
      points:coords,fill:colors[item.material],
      'fill-opacity':active?.24:.12,stroke:colors[item.material],
      'stroke-width':active?3:2,'data-id':item.surface_id
    });
    svg.appendChild(polygon);
    if(active){
      item.quad_norm.forEach((p,j)=>{
        const circle=element('circle',{
          cx:p[0]*w,cy:p[1]*h,r:Math.max(5,Math.min(w,h)/85),
          fill:'#ffffff',stroke:'#142b37','stroke-width':2,
          'data-id':item.surface_id,'data-vertex':j
        });
        circle.style.cursor='move';
        svg.appendChild(circle);
      });
    }
    const b=document.createElement('button');
    b.textContent=`${item.surface_id} · ${item.material}${item.reviewed?' ✓':''}`;
    if(active)b.className='active';
    b.onclick=()=>{selected=item.surface_id;draft=null;render();};
    list.appendChild(b);
  }
  if(draft!==null){
    if(draft.length){
      const pts=draft.map(p=>`${p[0]*w},${p[1]*h}`).join(' ');
      svg.appendChild(element('polyline',{
        points:pts,fill:'none',stroke:'#ffffff',
        'stroke-width':3,'stroke-dasharray':'8 5'
      }));
      for(const p of draft){
        svg.appendChild(element('circle',{
          cx:p[0]*w,cy:p[1]*h,r:Math.max(4,Math.min(w,h)/100),
          fill:'#ffffff'
        }));
      }
    }
  }
  const chosen=current();
  $('details').hidden=!chosen;
  if(chosen){
    $('material').value=chosen.material;
    $('reviewed').checked=chosen.reviewed;
    $('reviewed').disabled=chosen.material==='unknown';
    $('note').value=chosen.note;
  }
  $('undo').disabled=history.length===0;
  $('redo').disabled=future.length===0;
}
function nextSurfaceId(){
  while(items.some(x=>x.surface_id===`surface_${String(nextId).padStart(3,'0')}`))
    nextId++;
  return `surface_${String(nextId++).padStart(3,'0')}`;
}
async function openImage(file){
  if(!file)return;
  try{
    const response=await fetch('/inspect',{
      method:'POST',
      headers:{'Content-Type':'application/octet-stream'},
      body:await file.arrayBuffer()
    });
    const data=await response.json();
    if(!response.ok)throw Error(data.error);
    const url=URL.createObjectURL(file);
    const img=$('photo');
    img.src=url;
    await img.decode();
    if(img.naturalWidth!==data.width||img.naturalHeight!==data.height)
      throw Error('Размеры браузера отличаются от исходной геометрии');
    if(imageUrl)URL.revokeObjectURL(imageUrl);
    imageUrl=url;
    source={
      source_id:`src_${data.sha256.slice(0,16)}`,
      image_sha256:data.sha256,
      width:data.width,height:data.height,
      pixel_policy:'encoded-pixels-no-exif-transpose'
    };
    items=[];selected=null;draft=null;history=[];future=[];nextId=1;
    $('stage').style.display='inline-block';
    $('empty').hidden=true;
    $('add').disabled=false;
    $('export').disabled=false;
    render();
    message(`Фото: ${file.name} · ${data.width}×${data.height}\nSHA-256: ${data.sha256}`);
  }catch(error){
    message('Ошибка загрузки: '+error.message);
  }
}
$('imageFile').onchange=e=>openImage(e.target.files[0]);
$('add').onclick=()=>{
  if(!source)return;
  draft=[];selected=null;render();
  message('Укажи четыре вершины новой поверхности: TL → TR → BR → BL.');
};
$('cancel').onclick=()=>{draft=null;render();message('Выбор точек отменён.');};

$('overlay').addEventListener('pointerdown',async e=>{
  if(!source)return;
  const target=e.target;
  if(draft!==null){
    if(draft.length>=4)return;
    draft.push(point(e));
    if(draft.length===4){
      const entry={
        surface_id:nextSurfaceId(),material:'glass',
        quad_norm:clone(draft),reviewed:false,note:''
      };
      try{
        await validate({...documentData(),surfaces:[...items,entry]});
        remember();items.push(entry);selected=entry.surface_id;draft=null;
        message('Поверхность добавлена. Укажи материал и подтверди границы.');
      }catch(error){
        draft.pop();
        message('Неверный четырёхугольник: '+error.message+
                '\nПовтори последнюю точку.');
      }
    }
    render();
    return;
  }
  const id=target.getAttribute('data-id');
  if(!id)return;
  const vertex=target.getAttribute('data-vertex');
  selected=id;
  if(vertex!==null){
    remember();
    drag={id,index:Number(vertex),before:state()};
    $('overlay').setPointerCapture(e.pointerId);
  }
  render();
});
$('overlay').addEventListener('pointermove',e=>{
  if(!drag)return;
  const entry=current();
  if(!entry)return;
  entry.quad_norm[drag.index]=point(e);
  entry.reviewed=false;
  render();
});
$('overlay').addEventListener('pointerup',async()=>{
  if(!drag)return;
  const before=drag.before;drag=null;
  try{
    await validate(documentData());
    message('Геометрия поверхности обновлена.');
  }catch(error){
    restore(before);
    history.pop();
    message('Перемещение отменено: '+error.message);
  }
  render();
});
function modify(callback){
  const item=current();
  if(!item)return;
  remember();callback(item);render();
}
$('material').onchange=e=>modify(item=>{
  item.material=e.target.value;
  item.reviewed=false;
});
$('reviewed').onchange=e=>modify(item=>item.reviewed=e.target.checked);
$('note').onchange=e=>modify(item=>item.note=e.target.value);
$('remove').onclick=()=>{
  if(!current())return;
  remember();
  items=items.filter(x=>x.surface_id!==selected);
  selected=null;render();
  message('Поверхность удалена.');
};
$('undo').onclick=()=>{
  if(!history.length)return;
  future.push(state());restore(history.pop());
};
$('redo').onclick=()=>{
  if(!future.length)return;
  history.push(state());restore(future.pop());
};
$('export').onclick=async()=>{
  if(!source)return;
  try{
    const checked=await validate(documentData());
    const blob=new Blob([JSON.stringify(checked,null,2)+'\n'],
                        {type:'application/json'});
    const url=URL.createObjectURL(blob);
    const link=document.createElement('a');
    link.href=url;
    link.download=`surfaces_${source.image_sha256.slice(0,16)}.json`;
    document.body.appendChild(link);link.click();link.remove();
    setTimeout(()=>URL.revokeObjectURL(url),1000);
    message('JSON проверен и передан браузеру для скачивания.');
  }catch(error){message('Ошибка экспорта: '+error.message);}
};
$('jsonFile').onchange=async e=>{
  const file=e.target.files[0];
  if(!file||!source)return;
  try{
    const checked=await validate(JSON.parse(await file.text()));
    if(checked.source.image_sha256!==source.image_sha256||
       checked.source.width!==source.width||
       checked.source.height!==source.height)
      throw Error('Разметка относится к другому изображению');
    remember();items=clone(checked.surfaces);selected=null;draft=null;
    nextId=1;render();
    message(`Загружено поверхностей: ${items.length}.`);
  }catch(error){message('Ошибка импорта: '+error.message);}
  finally{e.target.value='';}
};
document.addEventListener('keydown',e=>{
  if(e.key==='Escape'&&draft!==null){
    draft=null;render();message('Выбор точек отменён.');
  }
});
render();
</script>
</body></html>"""


class SurfaceHandler(BaseHTTPRequestHandler):
    """Read-only local UI with strict image inspection and schema validation."""

    def log_message(self, format, *args):
        pass

    def reply(self, status: int, payload: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def respond_json(self, status: int, value: dict):
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.reply(status, body, "application/json; charset=utf-8")

    def read_body(self, maximum: int) -> bytes:
        length = self.headers.get("Content-Length", "")
        if not length.isascii() or not length.isdecimal():
            raise ValueError("Missing or invalid Content-Length")
        size = int(length)
        if size < 1 or size > maximum:
            raise ValueError("Request exceeds allowed size")
        content = self.rfile.read(size)
        if len(content) != size:
            raise ValueError("Incomplete request body")
        return content

    def allowed(self) -> bool:
        host = f"127.0.0.1:{self.server.server_port}"
        origin = self.headers.get("Origin")
        return self.headers.get("Host") == host and origin in (
            None, f"http://{host}"
        )

    def do_GET(self):
        if not self.allowed():
            return self.respond_json(403, {"error": "Forbidden origin"})
        if self.path != "/":
            return self.respond_json(404, {"error": "Not found"})
        self.reply(200, HTML.encode("utf-8"), "text/html; charset=utf-8")

    def do_POST(self):
        if not self.allowed():
            return self.respond_json(403, {"error": "Forbidden origin"})
        try:
            if self.path == "/inspect":
                if self.headers.get("Content-Type") != "application/octet-stream":
                    raise ValueError("Expected image bytes")
                raw = self.read_body(MAX_IMAGE)
                # Pillow PNG verify() must run immediately after open.
                # Reopen afterwards because verify() invalidates the parser.
                with Image.open(io.BytesIO(raw)) as image:
                    image.verify()
                with Image.open(io.BytesIO(raw)) as img:
                    width, height = img.size
                    if img.format not in {"JPEG", "PNG", "WEBP"}:
                        raise ValueError("Unsupported image format")
                    if img.getexif().get(274, 1) != 1:
                        raise ValueError("EXIF rotated images are not supported")
                    if width < 16 or height < 16:
                        raise ValueError("Image dimensions too small")
                    img.load()
                return self.respond_json(200, {
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "width": width,
                    "height": height,
                })
            if self.path == "/validate":
                if self.headers.get("Content-Type") != "application/json":
                    raise ValueError("Expected JSON")
                checked = loads_document(self.read_body(MAX_JSON))
                return self.respond_json(200, {"document": checked})
            return self.respond_json(404, {"error": "Not found"})
        except (ValueError, SurfaceError, OSError, UnidentifiedImageError, RuntimeError) as exc:
            return self.respond_json(400, {"error": str(exc)})


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 0), SurfaceHandler)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"SAP Surface Editor V1: {url}")
    print("Only local connections. Ctrl+C to stop.")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
