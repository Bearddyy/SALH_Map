#!/usr/bin/env python3
"""
Tithe-map tracer v2  -  assign plot numbers to automatically found field cells.

    1.  python autotrace.py "map.jpg"            (once; writes auto_cells.json)
    2.  python tracer_app.py                      (then open http://127.0.0.1:8123)

In the browser:
    click a cell            -> the plot number in the sidebar is given to that cell
    shift-click cells       -> build a selection (e.g. a wood split into fragments),
                               then press Enter to give the whole selection to the plot
    "Draw polygon"          -> click the corners yourself where no cell fits (hatched strips...)
    click a finished plot   -> select it; drag its white dots to fix the outline,
                               double-click an edge to add a dot, alt-click a dot to delete it
    "Generate maps"         -> writes interactive.html (hover map) and labelled_map.png

Options:  --image X.jpg  --csv X.csv  --cells auto_cells.json  --scale 0.5  --port 8123
"""
import argparse, base64, csv, glob, json, os, re, threading
import numpy as np
import cv2
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))


def _guess(patterns, skip=()):
    """first group of patterns that matches anything wins; the largest matching file is used"""
    for group in patterns:
        files = [f for p in group for f in glob.glob(os.path.join(HERE, p))
                 if os.path.basename(f) not in skip]
        if files:
            return max(files, key=os.path.getsize)
    return None


ap = argparse.ArgumentParser()
ap.add_argument("--image", default=_guess([["*.jpg", "*.jpeg"], ["*.tif", "*.tiff", "*.png"]],
                                          skip=("labelled_map.png", "auto_preview.jpg")),
                help="map scan (default: the largest .jpg in this folder)")
ap.add_argument("--csv", default=_guess([["*.csv"]]))
ap.add_argument("--cells", default=os.path.join(HERE, "auto_cells.json"))
ap.add_argument("--scale", type=float, default=0.5, help="working resolution; must match the autotrace.py run")
ap.add_argument("--port", type=int, default=8123)
ap.add_argument("--display-max", type=int, default=8000, help="longest side (px) of the image shown in the browser; the full-size scan is used if smaller")
ap.add_argument("--embed-full", action="store_true", help="put the full-size image inside interactive.html (single big file) instead of a separate .jpg next to it")
ARGS = ap.parse_args()
SCALE = ARGS.scale
POLY_FILE = os.path.join(HERE, "tithe_polygons.json")
LOCK = threading.Lock()

# ---------------------------------------------------------------- map image
assert ARGS.image and os.path.exists(ARGS.image), "map image not found - pass --image"
print("map image:", ARGS.image, "| csv:", ARGS.csv)
_full = cv2.imread(ARGS.image)
assert _full is not None, f"cannot read {ARGS.image}"
# Two resolutions: polygons live in WORKING pixels (scan x SCALE, same as autotrace.py);
# the browser shows the FULL-size scan drawn into that same coordinate space.
img = cv2.resize(_full, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA) if SCALE != 1 else _full
H, W = img.shape[:2]
_f = min(1.0, ARGS.display_max / max(_full.shape[:2]))
DISP = cv2.resize(_full, None, fx=_f, fy=_f, interpolation=cv2.INTER_AREA) if _f < 1 else _full
DH, DW = DISP.shape[:2]
K = DW / W                                   # display px per working px
if _f == 1 and ARGS.image.lower().endswith((".jpg", ".jpeg")):
    FULL_JPG = open(ARGS.image, "rb").read()   # serve the original file untouched
else:
    FULL_JPG = cv2.imencode(".jpg", DISP, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
MAP_JPG = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()   # light version for zoomed-out views
BG_B64 = "data:image/jpeg;base64," + base64.b64encode(MAP_JPG).decode()
print(f"working image {W}x{H} (coordinates) | display image {DW}x{DH} ({len(FULL_JPG)/1e6:.1f} MB)")
if DW < 1.5 * W: print("NOTE: display image is about the same size as the working image - use the full-size scan and --scale 0.5 for sharper numbers")

# ---------------------------------------------------------------- automatic cells
CELLS = []
if os.path.exists(ARGS.cells):
    _c = json.load(open(ARGS.cells, encoding="utf-8"))
    CELLS = _c["cells"]
    if _c.get("size") and list(_c["size"]) != [W, H]:
        print(f"WARNING: cells were made for a {_c['size']} image but the working image is {[W, H]}. "
              f"Re-run autotrace.py with the same --scale ({SCALE}).")
else:
    print("No auto_cells.json found - run  python autotrace.py <map.jpg>  first (manual drawing still works).")
CELL_BY_ID = {c["id"]: c for c in CELLS}


# ---------------------------------------------------------------- CSV schedule (one entry per plot key)
def norm_key(s):
    s = re.sub(r"\s+", "", (s or "").strip())
    m = re.match(r"^(\d+)(.*)$", s)
    return (m.group(1) + m.group(2).lower()) if m else None


def sort_key(k):
    m = re.match(r"^(\d+)(.*)$", k)
    return int(m.group(1)), m.group(2)


def clean(s):
    return re.sub(r"\s+", " ", (s or "").strip())


def load_schedule(path):
    rows = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            k = norm_key(r.get("plot no"))
            if k:
                rows.setdefault(k, []).append(r)
    out = []
    for k in sorted(rows, key=sort_key):
        owners, occs, name, cult = [], [], "", ""
        for r in rows[k]:
            name = name or clean(r.get("name / description"))
            cult = cult or clean(r.get("state of cultivation"))
            o = clean(f"{r.get('owner_forename') or ''} {r.get('owner_surname') or ''}")
            c = clean(f"{r.get('occupier_forename') or ''} {r.get('occupier_surname') or ''} {r.get('occupier_suffix') or ''}")
            if o and o not in owners: owners.append(o)
            if c and c not in occs: occs.append(c)
        out.append(dict(plot=k, name=name, cult=cult, owners=owners, occs=occs))
    return out


SCHEDULE = load_schedule(ARGS.csv) if ARGS.csv else []
SCHED_BY = {r["plot"]: r for r in SCHEDULE}
if not SCHEDULE: print("WARNING: no CSV found - pass --csv")


# ---------------------------------------------------------------- polygon geometry helpers
def label_point(pts):
    """most 'inside' point of a polygon (so the number is drawn inside, even for L-shapes)"""
    a = np.array(pts, np.int32)
    x0, y0 = a.min(0) - 2
    x1, y1 = a.max(0) + 2
    m = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    cv2.fillPoly(m, [a - [x0, y0]], 255)
    d = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    yy, xx = np.unravel_index(int(np.argmax(d)), d.shape)
    return [int(xx + x0), int(yy + y0)]


def union_rings(rings, eps=1.2):
    """fill all rings, bridge 1-2 px slivers between neighbouring cells, return outer outlines (largest first)"""
    arrs = [np.array(r, np.int32) for r in rings]
    allp = np.concatenate(arrs)
    x0, y0 = allp.min(0) - 5
    x1, y1 = allp.max(0) + 5
    m = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    for a in arrs:
        cv2.fillPoly(m, [a - [x0, y0]], 255)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cs = sorted(cs, key=cv2.contourArea, reverse=True)
    out = []
    for c in cs:
        if cv2.contourArea(c) < 30 and out: continue
        c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(c) >= 3: out.append((c + [x0, y0]).tolist())
    return out


def load_polys():
    if os.path.exists(POLY_FILE):
        with open(POLY_FILE, encoding="utf-8") as fh:
            d = json.load(fh)
        for p in d.get("polygons", []):
            p["plot"] = str(p["plot"])            # older files stored plain integers
        return d
    return {"polygons": []}


def save_polys(d):
    tmp = POLY_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(d, fh)
    os.replace(tmp, POLY_FILE)


def put_polygon(d, plot, rings, src, cells=None):
    d["polygons"] = [p for p in d["polygons"] if p["plot"] != plot]
    rec = dict(plot=plot, pts=rings[0], src=src, label=label_point(rings[0]))
    if len(rings) > 1: rec["parts"] = rings[1:]
    if cells: rec["cells"] = cells
    d["polygons"].append(rec)


# ---------------------------------------------------------------- output: hover map + labelled png
def features():
    out = []
    for p in load_polys()["polygons"]:
        r = SCHED_BY.get(p["plot"], {})
        out.append(dict(plot=p["plot"], rings=[p["pts"]] + p.get("parts", []),
                        label=p.get("label") or label_point(p["pts"]),
                        name=r.get("name", ""), cult=r.get("cult", ""),
                        owners=r.get("owners", []), occs=r.get("occs", [])))
    return out


def generate_outputs():
    feats = features()
    t = max(1, round(K))
    sc = lambda ring: (np.array(ring, np.float64) * K).astype(np.int32)     # working px -> display px
    canvas = DISP.copy()
    for f in feats:
        for ring in f["rings"]:
            cv2.fillPoly(canvas, [sc(ring)], (0, 200, 255))
    out = cv2.addWeighted(DISP, 0.55, canvas, 0.45, 0)
    fs = 0.45 * K
    for f in feats:
        for ring in f["rings"]:
            cv2.polylines(out, [sc(ring)], True, (0, 80, 160), t, cv2.LINE_AA)
        txt = f["plot"]
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, t)
        cx, cy = int(f["label"][0] * K), int(f["label"][1] * K)
        cv2.rectangle(out, (cx - tw // 2 - 3, cy - th - 3), (cx + tw // 2 + 3, cy + 4), (255, 255, 255), -1)
        cv2.putText(out, txt, (cx - tw // 2, cy), cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 20, 20), t, cv2.LINE_AA)
    png = os.path.join(HERE, "labelled_map.png")
    cv2.imwrite(png, out)
    if ARGS.embed_full:
        full_src = "data:image/jpeg;base64," + base64.b64encode(FULL_JPG).decode()
    else:
        full_src = "interactive_map_full.jpg"          # sits next to interactive.html
        with open(os.path.join(HERE, full_src), "wb") as fh:
            fh.write(FULL_JPG)
    html = (INTERACTIVE_HTML.replace("__MAP_B64__", BG_B64).replace("__FULL_SRC__", full_src)
            .replace("__WW__", str(W)).replace("__WH__", str(H))
            .replace("__DATA__", json.dumps(feats).replace("</", "<\\/")))
    path = os.path.join(HERE, "interactive.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return {"png": os.path.basename(png), "html": os.path.basename(path), "count": len(feats),
            "image": None if ARGS.embed_full else full_src}


# ======================================================================== tracer page
TRACER_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>Tithe map tracer</title><style>
*{box-sizing:border-box}
body{font-family:system-ui,sans-serif;margin:0;display:flex;height:100vh;background:#111;color:#eee}
#left{flex:1;position:relative;overflow:hidden}
canvas#cv{position:absolute;top:0;left:0;width:100%;height:100%}
#side{width:340px;background:#1b1b1b;padding:12px;overflow-y:auto;font-size:13.5px}
.row{margin:8px 0}
button,select,input[type=text]{font:inherit;padding:5px 9px;border-radius:6px;border:1px solid #444;background:#2a2a2a;color:#eee;cursor:pointer}
button:hover{background:#3a3a3a} button.on{background:#a60;border-color:#e80}
#info{background:#222;border:1px solid #444;border-radius:8px;padding:9px;min-height:92px;font-size:13px;line-height:1.4}
#plist{max-height:30vh;overflow-y:auto;border:1px solid #333;border-radius:6px}
.pitem{padding:2px 8px;cursor:pointer;display:flex;justify-content:space-between;white-space:nowrap;overflow:hidden}
.pitem:hover{background:#2e2e2e}.done{color:#7c7}.cur{background:#264}
#tip{position:absolute;pointer-events:none;background:#000d;padding:3px 8px;border-radius:6px;display:none;font-size:12.5px;z-index:10;max-width:420px}
#loupe{width:100%;aspect-ratio:1;border:1px solid #444;border-radius:8px;background:#000;image-rendering:auto}
small{color:#999} label{cursor:pointer} .key{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:3px}
</style></head><body>
<div id="left"><canvas id="cv"></canvas><div id="tip"></div></div>
<div id="side">
<h3 style="margin:2px 0 6px">Tithe plot tracer <small id="prog"></small></h3>
<div id="info"></div>
<div class="row">Plot: <input type="text" id="plot" style="width:90px"> <button id="nextBtn" title="jump to next plot without a shape">Next unassigned</button></div>
<div class="row"><button id="assignBtn" title="Enter">Assign selection (<span id="npend">0</span>)</button>
 <button id="drawBtn">Draw polygon</button> <button id="clearBtn" title="Esc">Clear</button></div>
<div class="row"><label><input type="checkbox" id="addMode"> add to the plot's existing shape</label><br>
 <label><input type="checkbox" id="autoNext" checked> auto-advance to next plot</label><br>
 <label><input type="checkbox" id="showCells" checked> show candidate cells</label></div>
<div class="row"><button id="undo">Undo last</button> <button id="delPlot">Delete this plot</button></div>
<div class="row"><button id="gen" style="background:#0a6;color:#fff;font-weight:600">Generate maps</button> <span id="genmsg"></span></div>
<div class="row"><canvas id="loupe" width="300" height="300"></canvas></div>
<div class="row"><small>
<span class="key" style="background:#1e963c"></span>cell with one number-like mark &nbsp;
<span class="key" style="background:#e67800"></span>several &nbsp;<span class="key" style="background:#888"></span>none<br>
Click cell = assign &middot; shift-click = multi-select, Enter to assign<br>
Right/middle-drag = pan &middot; wheel = zoom &middot; Esc = cancel</small></div>
<div id="plist"></div>
</div>
<script>
const $=id=>document.getElementById(id);
const cv=$('cv'),ctx=cv.getContext('2d'),tip=$('tip'),loupe=$('loupe'),lctx=loupe.getContext('2d');
const WW=__WW__,WH=__WH__;   // working-pixel size = coordinate space of all polygons
const bgS=new Image(),bgF=new Image();bgS.src='/map_small.jpg';bgF.onload=()=>req();bgF.src='/map.jpg';
const fullOK=()=>bgF.complete&&bgF.naturalWidth>0;
function curBg(){return fullOK()&&scale>0.5?bgF:bgS}   // light image when zoomed out, full-size when zoomed in
const SCHED=__SCHED__, BYPLOT=Object.fromEntries(SCHED.map(r=>[r.plot,r]));
let cells=[],polys=[],assigned=new Set(),scale=1,ox=0,oy=0,dpr=window.devicePixelRatio||1;
let pending=new Set(),hover=null,selPlot=null,drawPts=null,vdrag=null,pan=null,lastEdit=0,queued=false,mouse=[0,0];
const VR=6;
function esc(s){return String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function area(p){let a=0;for(let i=0,j=p.length-1;i<p.length;j=i++)a+=(p[j][0]+p[i][0])*(p[j][1]-p[i][1]);return Math.abs(a/2)}
function prep(o){const rings=[o.pts].concat(o.parts||[]);let x0=1e9,y0=1e9,x1=-1e9,y1=-1e9;const P=new Path2D();
 rings.forEach(r=>{r.forEach((q,j)=>{if(q[0]<x0)x0=q[0];if(q[0]>x1)x1=q[0];if(q[1]<y0)y0=q[1];if(q[1]>y1)y1=q[1];j?P.lineTo(q[0],q[1]):P.moveTo(q[0],q[1])});P.closePath()});
 o._b=[x0,y0,x1,y1];o._p=P;o._a=area(o.pts);o._r=rings}
function inPoly(pts,x,y){let s=false;for(let i=0,j=pts.length-1;i<pts.length;j=i++){const[xi,yi]=pts[i],[xj,yj]=pts[j];if(((yi>y)!==(yj>y))&&(x<(xj-xi)*(y-yi)/(yj-yi)+xi))s=!s}return s}
function hit(o,x,y){const b=o._b;if(x<b[0]||x>b[2]||y<b[1]||y>b[3])return false;return o._r.some(r=>inPoly(r,x,y))}
function pick(x,y){let best=null;
 polys.forEach(p=>{if(hit(p,x,y)&&(!best||p._a<best.o._a))best={t:'poly',o:p}});
 cells.forEach(c=>{if(assigned.has(c.id))return;if(hit(c,x,y)&&(!best||c._a<best.o._a))best={t:'cell',o:c}});
 return best}
function toImg(e){const r=cv.getBoundingClientRect();return[(e.clientX-r.left-ox)/scale,(e.clientY-r.top-oy)/scale]}
function req(){if(!queued){queued=true;requestAnimationFrame(draw)}}
function draw(){queued=false;const w=cv.clientWidth,h=cv.clientHeight;
 ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,cv.width,cv.height);
 ctx.setTransform(dpr*scale,0,0,dpr*scale,dpr*ox,dpr*oy);ctx.imageSmoothingEnabled=true;ctx.imageSmoothingQuality='high';ctx.drawImage(curBg(),0,0,WW,WH);
 const lw=1.5/scale;
 if($('showCells').checked){ctx.lineWidth=lw;
  cells.forEach(c=>{if(assigned.has(c.id))return;
   ctx.strokeStyle=c.nseeds===1?'rgba(30,150,60,.9)':c.nseeds>1?'rgba(230,120,0,.9)':'rgba(110,110,110,.8)';ctx.stroke(c._p);
   if(pending.has(c.id)){ctx.fillStyle='rgba(255,220,0,.5)';ctx.fill(c._p)}});}
 if(hover&&hover.t==='cell'){ctx.fillStyle='rgba(0,200,255,.35)';ctx.fill(hover.o._p)}
 polys.forEach(p=>{const sel=p.plot===selPlot,hv=hover&&hover.o===p;
  ctx.fillStyle=sel?'rgba(255,170,0,.45)':hv?'rgba(0,255,120,.5)':'rgba(0,180,255,.28)';ctx.fill(p._p);
  ctx.strokeStyle=sel?'#e60':'#09f';ctx.lineWidth=(sel?2.5:1.5)/scale;ctx.stroke(p._p);
  const b=p._b;if((b[2]-b[0])*scale>16){const c=p.label||[(b[0]+b[2])/2,(b[1]+b[3])/2];
   ctx.font=`bold ${13/scale}px sans-serif`;ctx.textAlign='center';ctx.textBaseline='middle';
   ctx.strokeStyle='#000';ctx.lineWidth=3/scale;ctx.strokeText(p.plot,c[0],c[1]);ctx.fillStyle='#fff';ctx.fillText(p.plot,c[0],c[1])}});
 const sp=polys.find(p=>p.plot===selPlot);
 if(sp){sp.pts.forEach(q=>{ctx.beginPath();ctx.arc(q[0],q[1],VR/scale,0,7);ctx.fillStyle='#fff';ctx.fill();ctx.strokeStyle='#e60';ctx.lineWidth=2/scale;ctx.stroke()})}
 if(drawPts&&drawPts.length){ctx.strokeStyle='#f0f';ctx.lineWidth=2/scale;ctx.beginPath();drawPts.forEach((q,i)=>i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]));
  ctx.lineTo(mouse[0],mouse[1]);ctx.stroke();drawPts.forEach(q=>{ctx.beginPath();ctx.arc(q[0],q[1],4/scale,0,7);ctx.fillStyle='#f0f';ctx.fill()})}
}
function resize(){const r=cv.parentElement.getBoundingClientRect();dpr=window.devicePixelRatio||1;cv.width=r.width*dpr;cv.height=r.height*dpr;req()}
function fit(){const r=cv.parentElement.getBoundingClientRect();scale=Math.min(r.width/WW,r.height/WH);ox=(r.width-WW*scale)/2;oy=0;req()}
function zoomTo(o,minScale){const b=o._b,r=cv.parentElement.getBoundingClientRect();
 const s=Math.min(r.width/((b[2]-b[0])*3),r.height/((b[3]-b[1])*3),6);scale=Math.max(s,minScale||0.8);
 ox=r.width/2-(b[0]+b[2])/2*scale;oy=r.height/2-(b[1]+b[3])/2*scale;req()}
function keepVisible(o){const b=o._b,r=cv.parentElement.getBoundingClientRect();
 const cx=(b[0]+b[2])/2*scale+ox,cy=(b[1]+b[3])/2*scale+oy,m=60;
 if(cx<m||cy<m||cx>r.width-m||cy>r.height-m){ox=r.width/2-(b[0]+b[2])/2*scale;oy=r.height/2-(b[1]+b[3])/2*scale;req()}}
async function post(u,b){return fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})}).then(r=>r.json())}
async function reload(){const d=await fetch('/api/polys').then(r=>r.json());polys=d.polygons;polys.forEach(prep);
 assigned=new Set(polys.flatMap(p=>p.cells||[]));renderList();$('npend').textContent=pending.size;req()}
function showInfo(r){$('info').innerHTML=r?`<b>Plot ${esc(r.plot)}</b> &mdash; ${esc(r.name)||'<i>(no name)</i>'}<br>Cultivation: ${esc(r.cult)||'?'}<br>
 Owner${r.owners.length>1?'s':''}: ${esc(r.owners.join('; '))||'?'}<br>Occupier${r.occs.length>1?'s':''}: ${esc(r.occs.join('; '))||'?'}`:'<i>No schedule entry for this number</i>'}
function renderList(){const done=new Set(polys.map(p=>p.plot)),cur=$('plot').value.trim();const el=$('plist');el.innerHTML='';
 SCHED.forEach(r=>{const d=document.createElement('div');d.className='pitem'+(done.has(r.plot)?' done':'')+(r.plot===cur?' cur':'');
  d.innerHTML=`<span>${esc(r.plot)}. ${esc(r.name.slice(0,30))||'&nbsp;'}</span><span>${done.has(r.plot)?'✓':'·'}</span>`;
  d.onclick=()=>setPlot(r.plot,true);el.appendChild(d)});
 $('prog').textContent=`${done.size}/${SCHED.length} plots`}
function setPlot(k,zoom){$('plot').value=k;showInfo(BYPLOT[k]);const p=polys.find(q=>q.plot===k);selPlot=p?k:null;if(p&&zoom)zoomTo(p,0.8);renderList();req()}
function advance(){const done=new Set(polys.map(p=>p.plot));const cur=$('plot').value.trim();let i=SCHED.findIndex(r=>r.plot===cur);
 const order=SCHED.slice(i+1).concat(SCHED.slice(0,i+1));const n=order.find(r=>!done.has(r.plot));if(n)setPlot(n.plot,false);else renderList()}
async function assign(ids,rings){const plot=$('plot').value.trim();if(!plot){alert('Type or choose a plot number first');return}
 if(!BYPLOT[plot]&&!confirm(`"${plot}" is not in the CSV schedule. Assign anyway?`))return;
 const add=$('addMode').checked;
 const r=rings?await post('/api/manual',{plot,pts:rings,add}):await post('/api/assign',{plot,cells:ids,add});
 if(r.error){alert(r.error);return}
 pending.clear();$('addMode').checked=false;selPlot=plot;await reload();const p=polys.find(q=>q.plot===plot);if(p)keepVisible(p);
 if($('autoNext').checked)advance()}
cv.addEventListener('contextmenu',e=>e.preventDefault());
cv.addEventListener('mousedown',e=>{
 if(e.button===1||e.button===2){pan=[e.clientX,e.clientY,ox,oy];e.preventDefault();return}
 if(e.button===0&&!drawPts){const sp=polys.find(p=>p.plot===selPlot);if(sp){const[x,y]=toImg(e);const tol=(VR+3)/scale;
  const j=sp.pts.findIndex(q=>Math.hypot(q[0]-x,q[1]-y)<tol);
  if(j>=0){if(e.altKey){if(sp.pts.length>3){sp.pts.splice(j,1);prep(sp);lastEdit=Date.now();persist(sp);req()}return}
   vdrag={p:sp,j};e.preventDefault()}}}});
window.addEventListener('mousemove',e=>{
 if(vdrag){const[x,y]=toImg(e);vdrag.p.pts[vdrag.j]=[Math.round(x),Math.round(y)];prep(vdrag.p);req();return}
 if(pan){ox=pan[2]+e.clientX-pan[0];oy=pan[3]+e.clientY-pan[1];req();return}
 if(e.target!==cv){return}
 const[x,y]=toImg(e);mouse=[x,y];hover=drawPts?null:pick(x,y);
 const r=cv.parentElement.getBoundingClientRect();
 if(hover){const o=hover.o;tip.style.display='block';tip.style.left=(e.clientX-r.left+14)+'px';tip.style.top=(e.clientY-r.top+14)+'px';
  tip.textContent=hover.t==='poly'?`${o.plot}: ${(BYPLOT[o.plot]||{}).name||''}`:`unassigned cell (${o.nseeds} number-like mark${o.nseeds==1?'':'s'})`}
 else tip.style.display='none';
 cv.style.cursor=drawPts?'crosshair':hover?'pointer':'default';
 lctx.imageSmoothingEnabled=true;lctx.fillStyle='#000';lctx.fillRect(0,0,300,300);{const b=fullOK()?bgF:bgS,k=b.naturalWidth/WW;lctx.drawImage(b,(x-20)*k,(y-20)*k,40*k,40*k,0,0,300,300)}
 lctx.strokeStyle='#f0f8';lctx.beginPath();lctx.moveTo(150,138);lctx.lineTo(150,162);lctx.moveTo(138,150);lctx.lineTo(162,150);lctx.stroke();
 req()});
window.addEventListener('mouseup',()=>{if(vdrag){const p=vdrag.p;vdrag=null;lastEdit=Date.now();persist(p)}pan=null});
function persist(p){post('/api/setpts',{plot:p.plot,pts:p.pts})}
cv.addEventListener('click',e=>{
 if(e.button!==0||vdrag||Date.now()-lastEdit<300)return;const[x,y]=toImg(e);
 if(drawPts){drawPts.push([Math.round(x),Math.round(y)]);req();return}
 const h=pick(x,y);if(!h)return;
 if(h.t==='poly'){setPlot(h.o.plot,false);return}
 if(e.shiftKey){pending.has(h.o.id)?pending.delete(h.o.id):pending.add(h.o.id);$('npend').textContent=pending.size;req();return}
 pending.add(h.o.id);assign([...pending])});
cv.addEventListener('dblclick',e=>{const[x,y]=toImg(e);
 if(drawPts){drawPts.pop();finishDraw();return}
 const sp=polys.find(p=>p.plot===selPlot);if(!sp)return;const tol=(VR+3)/scale;
 for(let i=0;i<sp.pts.length;i++){const a=sp.pts[i],b=sp.pts[(i+1)%sp.pts.length];
  if(Math.hypot((a[0]+b[0])/2-x,(a[1]+b[1])/2-y)<tol){lastEdit=Date.now();sp.pts.splice(i+1,0,[Math.round(x),Math.round(y)]);prep(sp);persist(sp);req();return}}});
cv.addEventListener('wheel',e=>{e.preventDefault();const[x,y]=toImg(e);const f=e.deltaY<0?1.2:1/1.2;
 scale=Math.min(10,Math.max(0.15,scale*f));const r=cv.getBoundingClientRect();ox=e.clientX-r.left-x*scale;oy=e.clientY-r.top-y*scale;req()},{passive:false});
function finishDraw(){if(drawPts)drawPts=drawPts.filter((q,i)=>!i||q[0]!==drawPts[i-1][0]||q[1]!==drawPts[i-1][1]);
 if(drawPts&&drawPts.length>=3){const pts=drawPts;drawPts=null;$('drawBtn').classList.remove('on');assign(null,pts)}else{alert('Need at least 3 corners')}}
$('drawBtn').onclick=()=>{if(drawPts){drawPts=null;$('drawBtn').classList.remove('on')}else{drawPts=[];$('drawBtn').classList.add('on');pending.clear()}req()};
$('assignBtn').onclick=()=>{if(drawPts)finishDraw();else if(pending.size)assign([...pending])};
$('clearBtn').onclick=()=>{pending.clear();drawPts=null;$('drawBtn').classList.remove('on');$('npend').textContent=0;req()};
window.addEventListener('keydown',e=>{if(['INPUT','SELECT'].includes(e.target.tagName))return;
 if(e.key==='Enter')$('assignBtn').onclick();if(e.key==='Escape')$('clearBtn').onclick()});
$('plot').addEventListener('change',()=>setPlot($('plot').value.trim(),true));
$('nextBtn').onclick=advance;$('showCells').onchange=req;
$('undo').onclick=async()=>{await post('/api/undo');selPlot=null;reload()};
$('delPlot').onclick=async()=>{const k=$('plot').value.trim();if(!k)return;await post('/api/delete',{plot:k});if(selPlot===k)selPlot=null;reload()};
$('gen').onclick=async()=>{$('genmsg').textContent='generating...';const r=await post('/api/generate');$('genmsg').textContent=`${r.count} plots → ${r.html}, ${r.png}`};
window.addEventListener('resize',resize);
(async()=>{cells=await fetch('/api/cells').then(r=>r.json());cells.forEach(prep);
 await new Promise(ok=>{if(bgS.complete&&bgS.naturalWidth)ok();else bgS.onload=ok});
 resize();fit();await reload();if(SCHED.length){const n=SCHED.find(r=>!polys.some(p=>p.plot===r.plot))||SCHED[0];setPlot(n.plot,false)}window.__ready=true})();
</script></body></html>"""

# ======================================================================== hover map page
INTERACTIVE_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>Stanstead Abbots Tithe Map 1840 - interactive</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--panel:#fffffff0;--ink:#2a2418;--mute:#6b6250;--line:#b7ad93}
body{margin:0;font-family:Georgia,'Times New Roman',serif;background:#e9e2d0;color:var(--ink)}
#wrap{position:fixed;inset:0;overflow:hidden}
canvas{position:absolute;top:0;left:0;width:100%;height:100%;cursor:grab;touch-action:none;-webkit-user-select:none;user-select:none;-webkit-tap-highlight-color:transparent}
#panel{position:absolute;top:10px;left:10px;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 14px;width:340px;max-width:calc(100vw - 40px);
 box-shadow:0 2px 12px #0003;font-size:14px;max-height:calc(100vh - 30px);overflow-y:auto}
#panel h1{font-size:16px;margin:0 0 2px}.sub{color:var(--mute);font-size:12.5px}
#detail{margin:8px 0;line-height:1.45}#detail b{font-size:15px}#detail .k{color:var(--mute);font-style:italic}
input,select{font:inherit;font-size:13px;padding:4px 7px;border:1px solid var(--line);border-radius:5px;background:#fff}
#search{width:100%;box-sizing:border-box;margin-top:6px}
#list{max-height:28vh;overflow-y:auto;margin-top:6px;font-size:12.5px}#list div{padding:2px 5px;cursor:pointer;border-radius:4px}#list div:hover{background:#0001}
#legend{font-size:12px;margin-top:6px}#legend span{display:inline-block;margin-right:8px;white-space:nowrap}#legend i{display:inline-block;width:10px;height:10px;margin-right:3px;border-radius:2px;vertical-align:-1px}
#tip{position:absolute;pointer-events:none;background:#000c;color:#fff;padding:4px 9px;border-radius:6px;font-size:12.5px;display:none;z-index:5;font-family:system-ui,sans-serif;max-width:360px}
</style></head><body><div id="wrap"><canvas id="cv"></canvas>
<div id="panel"><h1>Stanstead Abbots Tithe Map, 1840</h1>
<div class="sub">Hover a field for details &middot; drag to pan &middot; scroll to zoom</div>
<div id="detail"><i>Hover or click a plot</i></div>
<div>Colour by <select id="colour"><option value="none">nothing</option><option value="cult" selected>cultivation</option><option value="owner">owner</option></select></div>
<div style="margin-top:4px"><label style="font-size:12.5px"><input type="checkbox" id="lbl" checked> labels</label> <select id="lbltype" style="margin-left:4px"><option value="name" selected>names</option><option value="num">plot numbers</option></select></div>
<div id="legend"></div>
<input id="search" placeholder="Search plot no, field name, owner, occupier..."><div id="list"></div></div>
<div id="tip"></div></div>
<script>
const WW=__WW__,WH=__WH__;
const MAP=new Image();MAP.src="__MAP_B64__";
const MAPF=new Image();MAPF.onload=()=>req();MAPF.src="__FULL_SRC__";   // sharp version, used when zoomed in
function curMap(){return MAPF.complete&&MAPF.naturalWidth>0&&scale>0.5?MAPF:MAP}
const DATA=__DATA__;
const $=id=>document.getElementById(id),cv=$('cv'),ctx=cv.getContext('2d'),tip=$('tip');
let scale=1,ox=0,oy=0,hover=null,sel=null,dpr=window.devicePixelRatio||1,queued=false;
const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function area(p){let a=0;for(let i=0,j=p.length-1;i<p.length;j=i++)a+=(p[j][0]+p[i][0])*(p[j][1]-p[i][1]);return Math.abs(a/2)}
DATA.forEach(f=>{let x0=1e9,y0=1e9,x1=-1e9,y1=-1e9;const P=new Path2D();
 f.rings.forEach(r=>{r.forEach((q,j)=>{x0=Math.min(x0,q[0]);x1=Math.max(x1,q[0]);y0=Math.min(y0,q[1]);y1=Math.max(y1,q[1]);j?P.lineTo(q[0],q[1]):P.moveTo(q[0],q[1])});P.closePath()});
 f._b=[x0,y0,x1,y1];f._p=P;f._a=area(f.rings[0])});
function inPoly(pts,x,y){let s=false;for(let i=0,j=pts.length-1;i<pts.length;j=i++){const[xi,yi]=pts[i],[xj,yj]=pts[j];if(((yi>y)!==(yj>y))&&(x<(xj-xi)*(y-yi)/(yj-yi)+xi))s=!s}return s}
function pick(x,y){let best=null;DATA.forEach(f=>{const b=f._b;if(x<b[0]||x>b[2]||y<b[1]||y>b[3])return;
 if(f.rings.some(r=>inPoly(r,x,y))&&(!best||f._a<best._a))best=f});return best}
const PAL=['#e6194b','#3cb44b','#ffb300','#4363d8','#f58231','#911eb4','#42d4f4','#f032e6','#bfef45','#9a6324','#469990','#800000','#808000','#000075','#a9a9a9'];
let catOf=()=>'',colours={};
function setColour(){const m=$('colour').value;const lg=$('legend');lg.innerHTML='';colours={};
 if(m==='none'){catOf=()=>'';return}
 catOf=m==='cult'?f=>f.cult||'(unrecorded)':f=>f.owners[0]||'(unrecorded)';
 const cnt={};DATA.forEach(f=>{const c=catOf(f);cnt[c]=(cnt[c]||0)+1});
 Object.entries(cnt).sort((a,b)=>b[1]-a[1]).forEach(([c],i)=>{colours[c]=PAL[i%PAL.length]});
 Object.entries(colours).slice(0,10).forEach(([c,col])=>{lg.insertAdjacentHTML('beforeend',`<span><i style="background:${col}"></i>${esc(c)}</span>`)})}
function req(){if(!queued){queued=true;requestAnimationFrame(draw)}}
function draw(){queued=false;ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,cv.width,cv.height);
 ctx.setTransform(dpr*scale,0,0,dpr*scale,dpr*ox,dpr*oy);ctx.imageSmoothingEnabled=true;ctx.imageSmoothingQuality='high';ctx.drawImage(curMap(),0,0,WW,WH);
 const showL=$('lbl').checked;
 DATA.forEach(f=>{const col=colours[catOf(f)]||'#1e6edc';const on=f===hover,s=f===sel;
  ctx.globalAlpha=on||s?0.55:0.28;ctx.fillStyle=s?'#c02828':on?'#28a03c':col;ctx.fill(f._p);ctx.globalAlpha=1;
  ctx.strokeStyle='rgba(60,50,30,.85)';ctx.lineWidth=(on||s?2.2:1)/scale;ctx.stroke(f._p);
  const b=f._b;if(showL&&(b[2]-b[0])*scale>20){ctx.font=`bold ${12/scale}px sans-serif`;ctx.textAlign='center';ctx.textBaseline='middle';
   ctx.strokeStyle='#fffc';ctx.lineWidth=3/scale;const txt=($('lbltype').value=='num'?f.plot:(f.name||f.plot));if(ctx.measureText(txt).width>(b[2]-b[0])*.95)return;ctx.strokeText(txt,f.label[0],f.label[1]);ctx.fillStyle='#222';ctx.fillText(txt,f.label[0],f.label[1])}})}
function toImg(e){const r=cv.getBoundingClientRect();return[(e.clientX-r.left-ox)/scale,(e.clientY-r.top-oy)/scale]}
function detail(f){$('detail').innerHTML=f?`<b>Plot ${esc(f.plot)}</b> &mdash; ${esc(f.name)||'(unnamed)'}<br><span class="k">Cultivation:</span> ${esc(f.cult)||'?'}<br>
 <span class="k">Owner${f.owners.length>1?'s':''}:</span> ${esc(f.owners.join('; '))||'?'}<br><span class="k">Occupier${f.occs.length>1?'s':''}:</span> ${esc(f.occs.join('; '))||'?'}`:'<i>Hover or click a plot</i>'}
let drag=null;
cv.addEventListener('mousedown',e=>{drag=[e.clientX,e.clientY,ox,oy,false];cv.style.cursor='grabbing'});
window.addEventListener('mouseup',e=>{if(drag&&!drag[4]){const[x,y]=toImg(e);sel=pick(x,y);detail(sel||hover);req()}drag=null;cv.style.cursor='grab'});
cv.addEventListener('mousemove',e=>{
 if(drag){if(Math.hypot(e.clientX-drag[0],e.clientY-drag[1])>4)drag[4]=true;if(drag[4]){ox=drag[2]+e.clientX-drag[0];oy=drag[3]+e.clientY-drag[1];tip.style.display='none';req();return}}
 const[x,y]=toImg(e);const h=pick(x,y);if(h!==hover){hover=h;if(!sel)detail(h);req()}
 if(h){tip.style.display='block';tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY+14)+'px';
  tip.textContent=`${h.plot}: ${h.name||'(unnamed)'} — ${h.owners[0]||'owner?'}${h.owners.length>1?` +${h.owners.length-1}`:''}`}else tip.style.display='none'});
cv.addEventListener('wheel',e=>{e.preventDefault();const[x,y]=toImg(e);const f=e.deltaY<0?1.2:1/1.2;scale=Math.min(12,Math.max(0.1,scale*f));
 const r=cv.getBoundingClientRect();ox=e.clientX-r.left-x*scale;oy=e.clientY-r.top-y*scale;req()},{passive:false});
// ---- touch: one finger pans, two fingers pinch-zoom about the midpoint (touch-action:none stops browser page-zoom)
let pinch=null,touchDrag=null;const tpt=(r,t)=>({clientX:t.clientX,clientY:t.clientY});
cv.addEventListener('touchstart',e=>{e.preventDefault();
 if(e.touches.length===2){touchDrag=null;const[a,b]=[tpt(cv,e.touches[0]),tpt(cv,e.touches[1])];
  const r=cv.getBoundingClientRect(),mx=(a.clientX+b.clientX)/2,my=(a.clientY+b.clientY)/2;
  pinch={d:Math.hypot(a.clientX-b.clientX,a.clientY-b.clientY),s:scale,ix:(mx-r.left-ox)/scale,iy:(my-r.top-oy)/scale};}
 else if(e.touches.length===1){const r=cv.getBoundingClientRect(),p=tpt(cv,e.touches[0]);
  touchDrag=[p.clientX,p.clientY,ox,oy,e.touches[0].identifier];cv.style.cursor='grabbing';}},{passive:false});
cv.addEventListener('touchmove',e=>{e.preventDefault();
 if(e.touches.length===2&&pinch){const[a,b]=[tpt(cv,e.touches[0]),tpt(cv,e.touches[1])];
  const r=cv.getBoundingClientRect(),mx=(a.clientX+b.clientX)/2,my=(a.clientY+b.clientY)/2;
  scale=Math.min(12,Math.max(0.1,pinch.s*Math.hypot(a.clientX-b.clientX,a.clientY-b.clientY)/pinch.d));
  ox=mx-r.left-pinch.ix*scale;oy=my-r.top-pinch.iy*scale;tip.style.display='none';req();}
 else if(e.touches.length===1&&touchDrag&&e.touches[0].identifier===touchDrag[4]){const p=tpt(cv,e.touches[0]);
  if(Math.hypot(p.clientX-touchDrag[0],p.clientY-touchDrag[1])>6)touchDrag.push(1);
  if(touchDrag[5]){ox=touchDrag[2]+p.clientX-touchDrag[0];oy=touchDrag[3]+p.clientY-touchDrag[1];tip.style.display='none';req();}}},{passive:false});
cv.addEventListener('touchend',e=>{e.preventDefault();
 if(pinch&&e.touches.length<2)pinch=null;
 if(touchDrag&&e.touches.length===0){const tap=!touchDrag[5];const t=[...e.changedTouches].find(c=>c.identifier===touchDrag[4]);touchDrag=null;cv.style.cursor='grab';
  if(tap&&t){const[x,y]=toImg(t);sel=pick(x,y);detail(sel);req()}}},{passive:false});
cv.addEventListener('touchcancel',()=>{pinch=null;touchDrag=null});
function renderList(){const q=$('search').value.toLowerCase().trim();const L=$('list');L.innerHTML='';
 DATA.filter(f=>!q||[f.plot,f.name,f.cult,f.owners.join(' '),f.occs.join(' ')].join(' ').toLowerCase().includes(q)).slice(0,300).forEach(f=>{
  const d=document.createElement('div');d.textContent=`${f.plot}. ${f.name||'(unnamed)'} — ${f.owners[0]||''}`;
  d.onclick=()=>{sel=f;detail(f);const b=f._b,r=cv.getBoundingClientRect();scale=Math.min(6,Math.max(1.2,Math.min(r.width/((b[2]-b[0])*4),r.height/((b[3]-b[1])*4))));
   ox=r.width/2-(b[0]+b[2])/2*scale;oy=r.height/2-(b[1]+b[3])/2*scale;req()};L.appendChild(d)})}
$('search').addEventListener('input',renderList);$('colour').onchange=()=>{setColour();req()};$('lbl').onchange=req;$('lbltype').onchange=req;
function resize(){const r=cv.getBoundingClientRect();dpr=window.devicePixelRatio||1;cv.width=r.width*dpr;cv.height=r.height*dpr;req()}
function fit(){const r=cv.getBoundingClientRect();scale=Math.min(r.width/WW,r.height/WH);ox=(r.width-WW*scale)/2;oy=0;req()}
window.addEventListener('resize',()=>{resize()});
MAP.onload=()=>{setColour();resize();fit();renderList();window.__ready=true};
</script></body></html>"""


# ======================================================================== HTTP
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)): body = json.dumps(body).encode()
        elif isinstance(body, str): body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html = (TRACER_HTML.replace("__WW__", str(W)).replace("__WH__", str(H))
                    .replace("__SCHED__", json.dumps(SCHEDULE).replace("</", "<\\/")))
            self._send(200, html, "text/html; charset=utf-8")
        elif self.path == "/map.jpg":
            self._send(200, FULL_JPG, "image/jpeg")
        elif self.path == "/map_small.jpg":
            self._send(200, MAP_JPG, "image/jpeg")
        elif self.path == "/api/cells":
            self._send(200, CELLS)
        elif self.path == "/api/polys":
            with LOCK:
                d = load_polys()
                for p in d["polygons"]:
                    if "label" not in p: p["label"] = label_point(p["pts"])
                self._send(200, d)
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        ln = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(ln) or b"{}")
        plot = str(body.get("plot", "")).strip()
        with LOCK:
            d = load_polys()
            if self.path == "/api/assign":
                ids = [i for i in body.get("cells", []) if i in CELL_BY_ID]
                if not ids: return self._send(200, {"error": "No such cell"})
                rings = [CELL_BY_ID[i]["pts"] for i in ids]
                old = next((p for p in d["polygons"] if p["plot"] == plot), None)
                if body.get("add") and old:
                    rings += [old["pts"]] + old.get("parts", [])
                    ids = list(dict.fromkeys((old.get("cells") or []) + ids))
                out = union_rings(rings) if len(rings) > 1 else [rings[0]]
                put_polygon(d, plot, out, "auto", ids)
            elif self.path == "/api/manual":
                pts = [[int(a), int(b)] for a, b in body.get("pts", [])]
                if len(pts) < 3: return self._send(200, {"error": "Need at least 3 corners"})
                old = next((p for p in d["polygons"] if p["plot"] == plot), None)
                if body.get("add") and old:
                    out = union_rings([pts, old["pts"]] + old.get("parts", []))
                    put_polygon(d, plot, out, "manual", old.get("cells"))
                else:
                    put_polygon(d, plot, [pts], "manual")
            elif self.path == "/api/setpts":
                for p in d["polygons"]:
                    if p["plot"] == plot:
                        p["pts"] = [[int(a), int(b)] for a, b in body["pts"]]
                        p["label"] = label_point(p["pts"])
                        p["edited"] = True
            elif self.path == "/api/undo":
                if d["polygons"]: d["polygons"].pop()
            elif self.path == "/api/delete":
                d["polygons"] = [p for p in d["polygons"] if p["plot"] != plot]
            elif self.path == "/api/generate":
                save_polys(d)
                return self._send(200, generate_outputs())
            else:
                return self._send(404, {"error": "not found"})
            save_polys(d)
        self._send(200, {"ok": True})


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler)
    print(f"Tracer running at http://127.0.0.1:{ARGS.port}   "
          f"({W}x{H} working res, {len(CELLS)} candidate cells, {len(SCHEDULE)} plots in schedule)")
    srv.serve_forever()
