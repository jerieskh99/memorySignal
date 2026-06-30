#!/usr/bin/env python3
"""Interactive substrate-implementation tracker -> docs/substrate_progress.html.

A self-contained HTML page to log Rust-implementation progress family by family,
metric by metric: each metric has Implemented + Tested checkboxes, a notes/log
field, live per-family and overall progress bars, and browser persistence
(localStorage) plus JSON export/import so progress survives.

The metric list is IMPORTED from make_feature_substrate_spec (one source of truth),
so the tracker always matches the spec. Phase 1 = families A-F (trackable);
Phase 2 = G, H (shown deferred, reference-only).

Run: python3 plan07_campaign/make_substrate_progress.py
"""
import json
import re
from pathlib import Path

import make_feature_substrate_spec as spec

DOCS = Path(__file__).resolve().parent.parent / "docs"
PHASE1 = {"A", "B", "C", "D", "E", "F"}


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def build_data():
    fams = []
    for fam in spec.FAMILIES:
        metrics = []
        for row in fam["rows"]:
            if fam["grouped"]:
                group, name, meas, catch = row
            else:
                name, meas, catch = row
                group = ""
            metrics.append({"id": f'{fam["id"]}-{slug(name)}', "group": group,
                            "name": name, "measures": meas, "catches": catch})
        fams.append({"id": fam["id"], "title": fam["title"], "intro": fam["intro"],
                     "grouped": fam["grouped"], "group_label": fam["group_label"],
                     "phase": 1 if fam["id"] in PHASE1 else 2, "metrics": metrics})
    return fams


CSS = """
:root{--ink:#1A1A1A;--muted:#5B5B5B;--line:#DDD9CF;--paper:#fff;--panel:#F5F3EE;
 --gold:#B9822A;--blue:#3F6CA8;--green:#2E7D52;--amber:#C58A1E;--bg:#ECEAE3;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
 font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;line-height:1.5;font-size:15px}
header{position:sticky;top:0;z-index:10;background:var(--paper);border-bottom:1px solid var(--line);
 padding:14px 22px;box-shadow:0 1px 8px rgba(0,0,0,.06)}
h1{font-size:20px;margin:0 0 8px}
.wrap{max-width:1000px;margin:0 auto;padding:0 22px 80px}
.bars{display:flex;gap:18px;flex-wrap:wrap;align-items:center;font-size:13px}
.barwrap{flex:1;min-width:220px}
.bar{height:9px;background:var(--panel);border-radius:5px;overflow:hidden;margin-top:3px}
.bar>span{display:block;height:100%;width:0;transition:width .25s}
.bar.impl>span{background:var(--blue)} .bar.test>span{background:var(--green)}
.ctrls{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px;align-items:center}
button{font:inherit;font-size:13px;padding:5px 11px;border:1px solid var(--line);background:var(--paper);
 border-radius:6px;cursor:pointer}
button:hover{background:var(--panel)}
.filter button.on{background:var(--ink);color:#fff;border-color:var(--ink)}
.fam{background:var(--paper);border:1px solid var(--line);border-radius:10px;margin:18px 0;overflow:hidden}
.fam>.fh{padding:13px 18px;border-bottom:1px solid var(--line);background:var(--panel)}
.fh h2{font-size:17px;margin:0 0 4px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.fh .intro{font-size:12.5px;color:var(--muted);margin:6px 0 0;max-width:80ch}
.badge{font-size:11px;font-weight:700;padding:2px 8px;border-radius:20px;letter-spacing:.02em}
.badge.p1{background:#E7F0E9;color:var(--green)} .badge.p2{background:#F0E7D6;color:var(--gold)}
.fminibars{display:flex;gap:14px;font-size:11.5px;color:var(--muted);margin-top:8px;flex-wrap:wrap}
.fminibars .barwrap{min-width:150px;max-width:240px}
.grp{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--gold);
 padding:12px 18px 2px}
.m{padding:11px 18px;border-top:1px solid #F0EEE8}
.m:first-of-type{border-top:none}
.m .top{display:flex;align-items:flex-start;gap:12px;justify-content:space-between}
.m .name{font-weight:700;font-size:14.5px}
.m .meta{font-size:12px;color:var(--muted);margin:3px 0 0;max-width:78ch}
.m .meta b{color:#444}
.m code{font-family:ui-monospace,Menlo,monospace;font-size:11.5px;background:#F0EEE8;padding:1px 5px;border-radius:4px}
.controls{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-top:8px}
label.cb{display:inline-flex;align-items:center;gap:6px;font-size:13px;cursor:pointer;user-select:none}
label.cb input{width:16px;height:16px;cursor:pointer}
.pill{font-size:11px;font-weight:700;padding:2px 9px;border-radius:20px}
.pill.todo{background:#EEE;color:#777} .pill.prog{background:#FBF0DA;color:var(--amber)}
.pill.done{background:#E1F0E7;color:var(--green)}
.ts{font-size:11px;color:#999;margin-left:auto}
.noteBtn{font-size:12px;color:var(--blue);background:none;border:none;cursor:pointer;padding:2px 4px}
.notes{margin-top:8px;display:none}
.notes textarea{width:100%;min-height:54px;font:inherit;font-size:12.5px;padding:7px 9px;
 border:1px solid var(--line);border-radius:6px;resize:vertical;background:#FCFBF8}
.deferred .m{opacity:.62}
.foot{color:var(--muted);font-size:12px;margin-top:24px;text-align:center}
.m .meta code.term{position:relative;cursor:help;border-bottom:1px dotted var(--gold)}
.m .meta code.term:hover::after{content:attr(data-def);position:absolute;left:0;top:150%;z-index:50;
 width:max-content;max-width:340px;background:#0d0f12;color:#fff;border:1px solid #333;border-radius:6px;
 padding:8px 11px;font-size:12px;font-weight:400;line-height:1.45;white-space:normal;font-family:system-ui,sans-serif;
 box-shadow:0 8px 24px rgba(0,0,0,.5)}
.m .meta .eqlabel{color:var(--muted);font-size:11px;margin-right:5px}
"""

JS = r"""
const KEY='substrate_progress_v1';
let state={}; try{state=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){state={}}
let filter='all';
function save(){try{localStorage.setItem(KEY,JSON.stringify(state))}catch(e){}}
function st(id){return state[id]||{impl:false,test:false,notes:''}}
function stamp(){return new Date().toISOString().slice(0,16).replace('T',' ')}
function set(id,f,v){const s=st(id);s[f]=v;s.ts=stamp();state[id]=s;save();renderProgress();applyFilter()}
function status(s){if(s.impl&&s.test)return'done';if(s.impl||s.test)return'prog';return'todo'}
function phase1Metrics(){return DATA.filter(f=>f.phase===1).flatMap(f=>f.metrics)}

function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;')}

function metricHTML(m,track){
  const s=st(m.id),pst=status(s);
  const cbs= track ? `
    <label class="cb"><input type="checkbox" data-id="${m.id}" data-f="impl" ${s.impl?'checked':''}> Implemented</label>
    <label class="cb"><input type="checkbox" data-id="${m.id}" data-f="test" ${s.test?'checked':''}> Tested</label>
    <span class="pill ${pst}">${pst==='done'?'done':pst==='prog'?'in progress':'to do'}</span>
    ${s.ts?`<span class="ts">updated ${s.ts}</span>`:''}` : `<span class="pill todo">deferred</span>`;
  const notes= track ? `
    <button class="noteBtn" data-note="${m.id}">notes / log &#9662;</button>
    <div class="notes" id="n-${m.id}"><textarea data-id="${m.id}" data-f="notes"
      placeholder="log: what you implemented, the test you ran, the result, gotchas...">${esc(s.notes||'')}</textarea></div>` : '';
  return `<div class="m" data-id="${m.id}" data-status="${pst}">
    <div class="top"><div>
      <div class="name">${esc(m.name)}</div>
      <div class="meta"><span class="eqlabel">eq</span><code class="term" data-def="${esc(m.catches)}">${esc(m.measures)}</code></div>
    </div></div>
    <div class="controls">${cbs}</div>${notes}</div>`;
}

function familyHTML(f){
  const track=f.phase===1;
  let rows='',lastG=null;
  for(const m of f.metrics){
    if(f.grouped && m.group!==lastG){rows+=`<div class="grp">${esc(f.group_label)}: ${esc(m.group)}</div>`;lastG=m.group}
    rows+=metricHTML(m,track);
  }
  const badge=track?'<span class="badge p1">Phase 1</span>':'<span class="badge p2">Phase 2 &middot; deferred</span>';
  const mini=track?`<div class="fminibars">
     <div class="barwrap">impl <span id="fi-${f.id}">0/${f.metrics.length}</span><div class="bar impl"><span id="fib-${f.id}"></span></div></div>
     <div class="barwrap">test <span id="ft-${f.id}">0/${f.metrics.length}</span><div class="bar test"><span id="ftb-${f.id}"></span></div></div>
   </div>`:'';
  return `<section class="fam ${track?'':'deferred'}" data-fam="${f.id}">
    <div class="fh"><h2>Family ${f.id} &mdash; ${esc(f.title)} ${badge}</h2>
      <p class="intro">${esc(f.intro)}</p>${mini}</div>${rows}</section>`;
}

function render(){
  document.getElementById('families').innerHTML=DATA.map(familyHTML).join('');
  renderProgress(); applyFilter();
}
function renderProgress(){
  const ms=phase1Metrics();
  const ni=ms.filter(m=>st(m.id).impl).length, nt=ms.filter(m=>st(m.id).test).length, n=ms.length;
  document.getElementById('oi').textContent=`${ni}/${n}`;
  document.getElementById('ot').textContent=`${nt}/${n}`;
  document.getElementById('oib').style.width=(100*ni/n)+'%';
  document.getElementById('otb').style.width=(100*nt/n)+'%';
  for(const f of DATA){ if(f.phase!==1)continue;
    const fm=f.metrics, fi=fm.filter(m=>st(m.id).impl).length, ft=fm.filter(m=>st(m.id).test).length;
    document.getElementById('fi-'+f.id).textContent=`${fi}/${fm.length}`;
    document.getElementById('ft-'+f.id).textContent=`${ft}/${fm.length}`;
    document.getElementById('fib-'+f.id).style.width=(100*fi/fm.length)+'%';
    document.getElementById('ftb-'+f.id).style.width=(100*ft/fm.length)+'%';
  }
}
function applyFilter(){
  document.querySelectorAll('.m[data-status]').forEach(el=>{
    const ok = filter==='all' || el.dataset.status===filter;
    el.style.display = ok ? '' : 'none';
  });
}

document.addEventListener('change',e=>{
  const t=e.target; if(t.matches('input[type=checkbox][data-id]')) set(t.dataset.id,t.dataset.f,t.checked);
});
document.addEventListener('input',e=>{
  const t=e.target; if(t.matches('textarea[data-id]')) set(t.dataset.id,t.dataset.f,t.value);
});
document.addEventListener('click',e=>{
  const b=e.target.closest('[data-note]'); if(b){const d=document.getElementById('n-'+b.dataset.note);
    d.style.display=d.style.display==='block'?'none':'block';}
  const fb=e.target.closest('.filter button'); if(fb){filter=fb.dataset.flt;
    document.querySelectorAll('.filter button').forEach(x=>x.classList.toggle('on',x===fb));applyFilter();}
});
function expJSON(){const blob=new Blob([JSON.stringify(state,null,2)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);
  a.download='substrate_progress.json';a.click();}
function impJSON(){const i=document.createElement('input');i.type='file';i.accept='.json';
  i.onchange=()=>{const r=new FileReader();r.onload=()=>{try{const o=JSON.parse(r.result);
    state=Object.assign(state,o);save();render();}catch(e){alert('bad json')}};r.readAsText(i.files[0]);};i.click();}
function resetAll(){if(confirm('Clear ALL progress?')){state={};save();render();}}

render();
"""


def render_html(data):
    head = ("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            "<title>Substrate Implementation Tracker</title><style>" + CSS + "</style></head><body>")
    header = """
<header><h1>Substrate Implementation Tracker
 <span style="font-size:13px;font-weight:400;color:var(--muted)">&mdash; Rust, family by family, metric by metric</span></h1>
<div class="bars">
 <div class="barwrap">Implemented <b id="oi">0/0</b><div class="bar impl"><span id="oib"></span></div></div>
 <div class="barwrap">Tested <b id="ot">0/0</b><div class="bar test"><span id="otb"></span></div></div>
</div>
<div class="ctrls">
 <span class="filter" style="display:inline-flex;gap:6px">
   <button data-flt="all" class="on">All</button>
   <button data-flt="todo">To do</button>
   <button data-flt="prog">In progress</button>
   <button data-flt="done">Done</button>
 </span>
 <span style="flex:1"></span>
 <button onclick="expJSON()">Export JSON</button>
 <button onclick="impJSON()">Import JSON</button>
 <button onclick="resetAll()">Reset</button>
</div></header>
<div class="wrap"><div id="families"></div>
<p class="foot">Progress is saved in this browser (localStorage). Use Export JSON to back it up.
Phase 1 = families A&ndash;F (trackable); Phase 2 = G, H (deferred reference).
Generated by plan07_campaign/make_substrate_progress.py.</p></div>
"""
    script = "<script>const DATA=" + json.dumps(data) + ";\n" + JS + "</script>"
    (DOCS / "substrate_progress.html").write_text(head + header + script + "</body></html>")
    print(f"tracker -> {DOCS / 'substrate_progress.html'}")


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    render_html(build_data())
