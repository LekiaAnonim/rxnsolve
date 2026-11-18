"""Local web app: connect an RMG-models folder, pick a mechanism, identify it.

    python -m rxnsolve serve --models ../RMG-models            # http://127.0.0.1:8770

Every folder that contains an ``import.sh`` (the RMG importer layout) or a
CHEMKIN ``*.inp`` file is listed. Runs go one at a time through a queue; each
writes the static report site into ``--runs`` and the app links to it.
The server listens on 127.0.0.1 only.
"""

from __future__ import annotations

import glob
import json
import os
import queue
import re
import threading
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .report import CSS
from .thermo import find_model_inputs

STATE = {"models_dir": None, "runs_dir": None, "models": {}, "details": {}}
JOBS: dict[str, dict] = {}
QUEUE: "queue.Queue[str]" = queue.Queue()
LOCK = threading.Lock()


# ====================================================================
# Models
# ====================================================================

def discover(models_dir: str) -> dict[str, dict]:
    """Every folder with an import.sh (RMG importer layout) or a CHEMKIN *.inp file."""
    models = {}
    root = os.path.abspath(models_dir)
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and not d.startswith("RMG-Py-"))
        if "import.sh" not in files and not any(f.lower().endswith(".inp") for f in files):
            continue
        inp = find_model_inputs(dirpath)
        if not inp["chem"]:
            continue
        rel = os.path.relpath(dirpath, root)
        models[rel] = {"id": rel, "name": rel, "dir": dirpath, "chem": inp["chem"],
                       "thermo": inp["thermo"], "known": inp["known"]}
    return models


def model_details(mid: str) -> dict:
    if mid in STATE["details"]:
        return STATE["details"][mid]
    from .chemkin import parse_chemkin, read_known_smiles

    m = STATE["models"][mid]
    mech = parse_chemkin(m["chem"], m["thermo"])
    known = read_known_smiles(m["known"]) if m["known"] else {}
    d = {"species": len(mech.species), "reactions": len(mech.reactions),
         "with_thermo": sum(1 for s in mech.species if mech.formula(s)),
         "known": sum(1 for k in known if k in mech.species),
         "files": {k: (os.path.relpath(m[k], m["dir"]) if m[k] else None) for k in ("chem", "thermo", "known")},
         "warnings": mech.warnings[:3]}
    STATE["details"][mid] = d
    return d


def previous_runs(mid: str) -> list[dict]:
    base = os.path.join(STATE["runs_dir"], _slug(mid))
    out = []
    for d in sorted(glob.glob(os.path.join(base, "*")), reverse=True):
        idx = os.path.join(d, "index.html")
        if os.path.exists(idx):
            summ = {}
            jf = os.path.join(d, "run.json")
            if os.path.exists(jf):
                try:
                    summ = json.load(open(jf))
                except ValueError:
                    pass
            out.append({"name": os.path.basename(d), "url": "/runs/" + os.path.relpath(idx, STATE["runs_dir"]),
                        **summ, **run_files(d)})
    return out[:20]


def run_files(d: str) -> dict:
    """Absolute folder of a run and links to the files a user wants from it."""
    rel = os.path.relpath(d, STATE["runs_dir"])
    files = [(name, "/runs/" + rel + "/" + name) for name in ("index.html", "SMILES.txt", "identification.json")
             if os.path.exists(os.path.join(d, name))]
    return {"dir": os.path.abspath(d), "files": files}


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "model"


# ====================================================================
# Jobs
# ====================================================================

def submit(mid: str, opts: dict) -> str:
    jid = f"{int(time.time() * 1000):x}"
    JOBS[jid] = {"id": jid, "model": mid, "opts": opts, "status": "queued", "phase": "queued",
                 "log": [], "created": time.time(), "started": None, "finished": None,
                 "url": None, "summary": None, "error": None, "ident": None, "dir": None, "files": []}
    QUEUE.put(jid)
    return jid


def worker():
    from .pipeline import run_pipeline

    while True:
        jid = QUEUE.get()
        job = JOBS[jid]
        m = STATE["models"][job["model"]]
        o = job["opts"]
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out = os.path.join(STATE["runs_dir"], _slug(m["id"]), f"{stamp}-{o['mode']}")
        job.update(status="running", started=time.time(), phase="starting")
        try:
            known = m["known"] if (o["mode"] == "audit" or o.get("seed")) else None
            res = run_pipeline(
                m["chem"], m["thermo"], known, out_dir=out, mode=o["mode"],
                use_names=o.get("names", True), use_thermo=o.get("thermo", True),
                exclude_models={m["id"]} if o.get("exclude", True) else None,
                on_ident=lambda ident: job.__setitem__("ident", ident))
            summary = {k: res[k] for k in ("counts", "seconds", "flagged", "reactions_solved", "reactions")}
            summary["mode"] = o["mode"]
            with open(os.path.join(out, "run.json"), "w") as fh:
                json.dump(summary, fh)
            job.update(status="done", phase="done", summary=summary,
                       url="/runs/" + os.path.relpath(res["index"], STATE["runs_dir"]), **run_files(out))
        except Exception as e:  # noqa: BLE001 - report any failure to the page
            job.update(status="error", phase="error", error=f"{type(e).__name__}: {e}",
                       log=job["log"] + traceback.format_exc().splitlines()[-6:])
        finally:
            job["finished"] = time.time()
            QUEUE.task_done()


def job_view(job: dict) -> dict:
    ident = job.get("ident")
    log = list(job["log"])
    phase = job["phase"]
    if ident is not None:
        log = list(ident.log) + log
        if job["status"] == "running":
            phase = getattr(ident, "phase", phase)
    t1 = job["finished"] or time.time()
    return {k: job[k] for k in ("id", "model", "opts", "status", "url", "summary", "error", "dir", "files")} | {
        "phase": phase, "log": log[-14:], "elapsed": round(t1 - (job["started"] or t1), 1),
        "queued_ahead": sum(1 for j in JOBS.values() if j["status"] == "queued" and j["created"] < job["created"])}


# ====================================================================
# HTTP
# ====================================================================

class Handler(BaseHTTPRequestHandler):
    server_version = "rxnsolve"

    def log_message(self, fmt, *args):  # quiet console
        pass

    def _send(self, code, body: bytes, ctype="text/html; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = dict(urllib.parse.parse_qsl(u.query))
        if u.path in ("/", "/index.html"):
            return self._send(200, APP_HTML.encode())
        if u.path == "/style.css":
            return self._send(200, (CSS + APP_CSS).encode(), "text/css")
        if u.path == "/api/config":
            return self._json({"models_dir": STATE["models_dir"], "runs_dir": STATE["runs_dir"],
                               "n_models": len(STATE["models"])})
        if u.path == "/api/models":
            return self._json([{"id": m["id"], "has_known": bool(m["known"])}
                               for m in sorted(STATE["models"].values(), key=lambda m: m["id"].lower())])
        if u.path == "/api/model":
            mid = q.get("id")
            if mid not in STATE["models"]:
                return self._json({"error": "unknown model"}, 404)
            try:
                det = model_details(mid)
            except Exception as e:  # noqa: BLE001
                return self._json({"error": f"could not read the mechanism: {e}"}, 500)
            jobs = [job_view(j) for j in JOBS.values() if j["model"] == mid]
            return self._json({"id": mid, **det, "runs": previous_runs(mid),
                               "jobs": sorted(jobs, key=lambda j: j["id"], reverse=True)[:5]})
        if u.path == "/api/job":
            job = JOBS.get(q.get("id", ""))
            return self._json(job_view(job) if job else {"error": "unknown job"}, 200 if job else 404)
        if u.path.startswith("/runs/"):
            return self._static(u.path[len("/runs/"):])
        return self._send(404, b"not found", "text/plain")

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self._json({"error": "bad JSON"}, 400)
        path = urllib.parse.urlparse(self.path).path
        if path == "/api/runs_dir":
            d = os.path.abspath(os.path.expanduser(str(data.get("runs_dir", ""))))
            try:
                os.makedirs(d, exist_ok=True)
            except OSError as e:
                return self._json({"error": f"cannot use {d}: {e}"}, 400)
            STATE["runs_dir"] = d
            return self._json({"runs_dir": d})
        if path == "/api/reveal":
            target = os.path.realpath(str(data.get("dir", "")))
            root = os.path.realpath(STATE["runs_dir"])
            if not (target == root or target.startswith(root + os.sep)) or not os.path.exists(target):
                return self._json({"error": "only run folders can be shown"}, 400)
            import subprocess
            import sys
            cmd = ["open", target] if sys.platform == "darwin" else ["xdg-open", target]
            try:
                subprocess.Popen(cmd)
            except OSError as e:
                return self._json({"error": str(e)}, 500)
            return self._json({"ok": True})
        if path == "/api/connect":
            d = os.path.abspath(os.path.expanduser(str(data.get("models_dir", ""))))
            if not os.path.isdir(d):
                return self._json({"error": f"not a folder: {d}"}, 400)
            connect(d)
            return self._json({"models_dir": d, "n_models": len(STATE["models"])})
        if path == "/api/run":
            mid = data.get("id")
            if mid not in STATE["models"]:
                return self._json({"error": "unknown model"}, 404)
            mode = data.get("mode", "identify")
            if mode not in ("identify", "audit"):
                return self._json({"error": "mode must be identify or audit"}, 400)
            if mode == "audit" and not STATE["models"][mid]["known"]:
                return self._json({"error": "audit needs a SMILES.txt in the model folder"}, 400)
            opts = {"mode": mode, "names": bool(data.get("names", True)), "thermo": bool(data.get("thermo", True)),
                    "exclude": bool(data.get("exclude", True)), "seed": bool(data.get("seed", False))}
            return self._json({"job": submit(mid, opts)})
        return self._send(404, b"not found", "text/plain")

    def _static(self, rel):
        root = os.path.realpath(STATE["runs_dir"])
        full = os.path.realpath(os.path.join(root, urllib.parse.unquote(rel)))
        if not full.startswith(root + os.sep) or not os.path.isfile(full):
            return self._send(404, b"not found", "text/plain")
        ctype = {".html": "text/html; charset=utf-8", ".css": "text/css", ".json": "application/json",
                 ".txt": "text/plain; charset=utf-8", ".svg": "image/svg+xml"}.get(
            os.path.splitext(full)[1], "application/octet-stream")
        with open(full, "rb") as fh:
            self._send(200, fh.read(), ctype)


def connect(models_dir: str) -> None:
    with LOCK:
        STATE["models_dir"] = os.path.abspath(models_dir)
        STATE["models"] = discover(models_dir)
        STATE["details"] = {}


def serve(models_dir: str | None, runs_dir: str, host: str = "127.0.0.1", port: int = 8770) -> None:
    STATE["runs_dir"] = os.path.abspath(runs_dir)
    os.makedirs(STATE["runs_dir"], exist_ok=True)
    if models_dir:
        connect(models_dir)
    threading.Thread(target=worker, daemon=True).start()
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"rxnsolve app on http://{host}:{port}  (models: {STATE['models_dir'] or 'not connected'}, "
          f"{len(STATE['models'])} mechanisms; runs saved in {STATE['runs_dir']})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


# ====================================================================
# Page
# ====================================================================

APP_CSS = """
.app{display:grid;grid-template-columns:300px 1fr;gap:28px;align-items:start}
@media(max-width:860px){.app{grid-template-columns:1fr}}
.connect{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:14px 0 24px}
.connect input{flex:1;min-width:280px}
button{font:inherit;font-size:14px;border-radius:8px;padding:8px 14px;border:1px solid var(--ink);background:var(--ink);color:#fff;cursor:pointer}
button.ghost{background:#fff;color:var(--ink);border-color:var(--line)}
button:disabled{opacity:.5;cursor:default}
.list{border:1px solid var(--line);border-radius:10px;max-height:70vh;overflow:auto}
.list a{display:flex;justify-content:space-between;gap:8px;padding:8px 12px;border-bottom:1px solid var(--line);color:var(--ink);font-size:14px}
.list a:last-child{border-bottom:0}.list a:hover{background:var(--soft);text-decoration:none}
.list a.on{background:var(--givenbg);font-weight:600}
.list .tag{font-size:11px;color:var(--given);white-space:nowrap}
.opts{display:grid;gap:8px;margin:12px 0 16px;font-size:14px}
.opts label{display:flex;gap:8px;align-items:flex-start}.opts small{color:var(--muted);display:block}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}
.seg label{padding:7px 14px;cursor:pointer;font-size:14px;border-right:1px solid var(--line)}.seg label:last-child{border-right:0}
.seg input{display:none}.seg label.on{background:var(--ink);color:#fff}
.log{font:12.5px/1.5 var(--mono);background:var(--soft);border:1px solid var(--line);border-radius:8px;padding:10px 12px;white-space:pre-wrap;max-height:260px;overflow:auto;margin-top:10px}
.phase{display:flex;gap:6px;flex-wrap:wrap;margin:10px 0 0}
.phase span{font-size:12px;padding:3px 9px;border-radius:999px;background:var(--soft);color:var(--faint);border:1px solid var(--line)}
.phase span.done{color:var(--ok);background:var(--okbg);border-color:transparent}
.phase span.now{color:#fff;background:var(--ink);border-color:var(--ink)}
.empty{color:var(--muted);padding:40px 0;text-align:center}
"""

APP_HTML = """<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>rxnsolve workbench</title>
<link rel='stylesheet' href='/style.css'></head><body>
<header class='top'><div class='in'><span class='brand'>rxnsolve</span><span class='sub'>mechanism workbench</span></div></header>
<main>
<h1>Identify species in a CHEMKIN mechanism</h1>
<div class='sub'>Connect a folder of mechanisms (the RMG-models layout: one folder per model with
<span class='mono'>import.sh</span>, mechanism, thermo and optionally <span class='mono'>SMILES.txt</span>), pick one and run.</div>
<div class='connect'><input id='dir' type='text' placeholder='Path to RMG-models'><button id='go'>Connect</button>
<span id='cstat' class='sub'></span></div>
<div class='connect' style='margin-top:-14px'><span class='sub' style='min-width:110px'>Save results in</span>
<input id='rdir' type='text' placeholder='Folder for reports'><button id='rgo' class='ghost'>Use folder</button>
<span id='rstat2' class='sub'></span></div>
<div class='app'>
 <div><input id='filter' type='search' placeholder='Filter mechanisms' style='width:100%;min-width:0;margin-bottom:10px'>
  <div class='list' id='list'></div></div>
 <div id='detail'><div class='empty'>Choose a mechanism on the left.</div></div>
</div></main>
<script>
const $=s=>document.querySelector(s);let models=[],cur=null,poll=null;
const PH=['identify','reactions','report','done'],PHA=['audit','identify','reactions','report','done'];
async function api(p,b){const r=await fetch(p,b?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)}:{});return r.json()}
function esc(s){return String(s??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]))}
async function init(){const c=await api('/api/config');$('#dir').value=c.models_dir||'';$('#rdir').value=c.runs_dir||'';stat(c);await load()}
$('#rgo').onclick=async()=>{const r=await api('/api/runs_dir',{runs_dir:$('#rdir').value});$('#rstat2').textContent=r.error||'reports will be saved here';if(!r.error)$('#rdir').value=r.runs_dir};
async function reveal(d){const r=await api('/api/reveal',{dir:d});if(r.error)alert(r.error)}
function filesHtml(r){return `<div class='small' style='margin-top:4px'><span class='mono muted'>${esc(r.dir)}</span><br>${(r.files||[]).map(([n,u])=>`<a href='${u}' target='_blank'>${esc(n==='index.html'?'report':n)}</a>`).join(' · ')}${r.dir?` · <a href='#' onclick='reveal(${JSON.stringify(r.dir).replace(/'/g,"&#39;")});return false'>Show in Finder</a>`:''}</div>`}
function stat(c){$('#cstat').textContent=c.models_dir?`${c.n_models} mechanisms found`:'not connected'}
async function load(){models=await api('/api/models');draw()}
function draw(){const t=$('#filter').value.toLowerCase();$('#list').innerHTML=models.filter(m=>!t||m.id.toLowerCase().includes(t))
 .map(m=>`<a href='#' data-id='${esc(m.id)}' class='${m.id===cur?'on':''}'><span>${esc(m.id)}</span>${m.has_known?"<span class='tag'>SMILES.txt</span>":''}</a>`).join('')||"<div class='empty'>No mechanisms.</div>";
 document.querySelectorAll('#list a').forEach(a=>a.onclick=e=>{e.preventDefault();pick(a.dataset.id)})}
$('#filter').oninput=draw;
$('#go').onclick=async()=>{const r=await api('/api/connect',{models_dir:$('#dir').value});if(r.error){$('#cstat').textContent=r.error;return}stat(r);cur=null;await load();$('#detail').innerHTML="<div class='empty'>Choose a mechanism on the left.</div>"};
async function pick(id){cur=id;draw();$('#detail').innerHTML="<div class='empty'>Reading the mechanism…</div>";const d=await api('/api/model?id='+encodeURIComponent(id));show(d)}
function show(d){if(d.error){$('#detail').innerHTML=`<div class='flagbox'>${esc(d.error)}</div>`;return}
 const hasK=!!d.files.known;
 $('#detail').innerHTML=`<h2 style='margin-top:0' class='mono'>${esc(d.id)}</h2>
 <div class='stats' style='margin-top:10px'><div class='stat'><b>${d.species}</b><span>species</span></div>
 <div class='stat'><b>${d.reactions}</b><span>reactions</span></div><div class='stat'><b>${hasK?d.known:'–'}</b><span>structures in SMILES.txt</span></div></div>
 ${d.with_thermo<d.species?`<div class='flagbox' style='${d.with_thermo?"border-left-color:var(--warn);background:var(--warnbg)":""}'>${d.with_thermo
   ?`${d.species-d.with_thermo} of ${d.species} species have no readable thermo card and cannot be identified.`
   :`No species has a readable thermo card${d.species?'':' (no species found)'}, so nothing can be identified. ${esc((d.warnings||[]).join(' '))}`}</div>`:''}
 <dl class='kv small' style='margin:8px 0 18px'><dt>mechanism</dt><dd class='mono'>${esc(d.files.chem)}</dd><dt>thermo</dt><dd class='mono'>${esc(d.files.thermo||'inside mechanism')}</dd><dt>SMILES</dt><dd class='mono'>${esc(d.files.known||'none')}</dd></dl>
 <h3>Run</h3><div class='seg' id='mode'><label class='on'><input type='radio' name='m' value='identify' checked>Identify</label>
 <label class='${hasK?'':'muted'}'><input type='radio' name='m' value='audit' ${hasK?'':'disabled'}>Audit SMILES.txt, then identify the rest</label></div>
 <div class='opts'>
 <label><input type='checkbox' id='o_names' checked><span>Use species names as hints<small>Off = blind: structures from formula, thermochemistry and reactions only.</small></span></label>
 <label id='seedrow'><input type='checkbox' id='o_seed' ${hasK?'':'disabled'}><span>Start from the structures in SMILES.txt<small>They are taken as given; only the other species are identified.</small></span></label>
 <label><input type='checkbox' id='o_excl' checked><span>Leave this mechanism out of the thermo library<small>Its own thermo would otherwise match itself; keep on for an honest test.</small></span></label>
 <label><input type='checkbox' id='o_thermo' checked><span>Use thermochemistry</span></label></div>
 <button id='run'>Run</button> <span class='sub' id='rstat'></span>
 <div id='job'></div><h3 style='margin-top:28px'>Reports</h3><div id='runs'></div>`;
 document.querySelectorAll('#mode label').forEach(l=>l.onclick=()=>{if(l.querySelector('input').disabled)return;document.querySelectorAll('#mode label').forEach(x=>x.classList.remove('on'));l.classList.add('on');l.querySelector('input').checked=true;$('#seedrow').style.display=mode()==='audit'?'none':''});
 $('#run').onclick=run;if(!d.with_thermo)$('#run').disabled=true;runs(d.runs);const live=(d.jobs||[]).find(j=>j.status==='running'||j.status==='queued');if(live)watch(live.id)}
function mode(){return document.querySelector('#mode input:checked').value}
function runs(rs){$('#runs').innerHTML=rs.length?`<table class='small'><thead><tr><th>run</th><th>result and files</th></tr></thead><tbody>${rs.map(r=>`<tr><td class='mono'>${esc(r.name)}</td><td>${r.counts?Object.entries(r.counts).map(([k,v])=>`${v} ${k==='known'?'given':k==='unresolved'?'open':k==='no-formula'?'without thermo':k}`).join(', '):''}${r.flagged?`, ${r.flagged} flagged`:''}${r.seconds?` · ${Math.round(r.seconds)} s`:''}${filesHtml(r)}</td></tr>`).join('')}</tbody></table>`:"<p class='sub'>No reports yet.</p>"}
async function run(){const b={id:cur,mode:mode(),names:$('#o_names').checked,seed:$('#o_seed').checked,exclude:$('#o_excl').checked,thermo:$('#o_thermo').checked};
 const r=await api('/api/run',b);if(r.error){$('#rstat').textContent=r.error;return}watch(r.job)}
function watch(id){clearInterval(poll);$('#run').disabled=true;const tick=async()=>{const j=await api('/api/job?id='+id);if(j.model!==cur)return;
 const ph=(j.opts.mode==='audit'?PHA:PH),i=ph.indexOf(j.phase);
 $('#job').innerHTML=`<div class='phase'>${ph.map((p,k)=>`<span class='${j.status==='done'||k<i?'done':(k===i?'now':'')}'>${p}</span>`).join('')}</div>
 <div class='sub' style='margin-top:8px'>${j.status==='queued'?`queued (${j.queued_ahead} ahead)`:j.status==='running'?`running · ${Math.round(j.elapsed)} s`:j.status==='done'?`finished in ${Math.round(j.elapsed)} s · <a href='${j.url}' target='_blank'><b>Open report</b></a>${filesHtml(j)}`:`<span style='color:var(--bad)'>${esc(j.error)}</span>`}</div>
 <div class='log'>${esc(j.log.join('\\n'))}</div>`;
 if(j.status==='done'||j.status==='error'){clearInterval(poll);$('#run').disabled=false;const d=await api('/api/model?id='+encodeURIComponent(cur));runs(d.runs)}};tick();poll=setInterval(tick,1500)}
init();
</script></body></html>"""
