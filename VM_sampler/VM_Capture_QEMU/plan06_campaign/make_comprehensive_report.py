#!/usr/bin/env python3
"""Comprehensive Plan 05 -> Plan 06 data report: every cell, every family, every
view (CSV + HTML tables + matplotlib figures inline).

Inputs (Plan 06 full 2-channel run):
  sweep_full.csv            APF features per cell (plan03_sweep)
  cells_full/               apf + diskio trajectories per cell
  diskio_lift_full.json     classification results (APF vs +peakvar vs +diskio)

Outputs (docs/):
  plan06_all_data.csv          66-cell table (memory + disk columns)
  plan06_figs/*.png            standalone figures
  plan06_comprehensive.html    tables + figures inline (self-contained)

    python3 plan06_campaign/make_comprehensive_report.py
"""
import base64
import csv
import io
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
P5 = HERE.parent / "plan05_campaign"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(P5))
from extra_features import load_extra            # noqa: E402
from diskio_features import load_diskio          # noqa: E402
from behavior_families import family_of          # noqa: E402

DOCS = HERE.parent / "docs"
FIGDIR = DOCS / "plan06_figs"
FIGDIR.mkdir(parents=True, exist_ok=True)
SWEEP = HERE / "downstream" / "sweep_full.csv"
CELLS = HERE / "downstream" / "cells_full"
LIFT = HERE / "downstream" / "diskio_lift_full.json"

FAM_COLOR = {"ransomware": "#e8553a", "scanner": "#e8a33a", "mem_sweep": "#4f86c6",
             "mem_fault": "#3aa6a6", "app": "#7cb96e"}
THREAT = ("ransom", "scanner")


def is_threat(w): return any(k in w.lower() for k in THREAT)


def short(w):
    return (w.replace("sandbox_", "").replace("_v2", "")
             .replace("_intensive", "").replace("_density", "").replace("_traversal", "")
             .replace("_metadata", "").replace("_sweep", "").replace("_intensity", ""))


def fnum(r, c):
    try:
        return float(r.get(c, "") or "nan")
    except ValueError:
        return float("nan")


# ---- load per-cell ----------------------------------------------------------
rows = [r for r in csv.DictReader(open(SWEEP))
        if r["window"] == "8" and r["hop"] == "4" and r["status"].startswith("ok")]
extra = load_extra(CELLS)
diskio = load_diskio(CELLS)

cells = []
for r in rows:
    cid = r["cell_id"]
    e = extra.get(cid, {}); d = diskio.get(cid, {})
    wl = r["workload"]
    cells.append({
        "workload": wl, "short": short(wl), "family": family_of(wl),
        "kind": "threat" if is_threat(wl) else "benign",
        "duration_s": int(r["duration_s"]), "rep": int(r["replicate"]),
        "apf_mean": fnum(r, "apf_mean"), "apf_std": fnum(r, "apf_std"),
        "apf_cov": e.get("apf_cov", float("nan")), "apf_max": e.get("apf_max", float("nan")),
        "duty_gt05": e.get("duty_gt05", float("nan")), "n_windows": int(float(r["n_windows"])),
        "wr_rate_mbs": d.get("wr_rate_mean_mbs", float("nan")),
        "wr_total_mb": d.get("wr_total_mb", float("nan")),
        "rd_rate_mbs": d.get("rd_rate_mean_mbs", float("nan")),
    })
cells.sort(key=lambda c: (c["family"], c["workload"], c["duration_s"], c["rep"]))

# ---- per-workload aggregates ------------------------------------------------
WL = sorted({c["workload"] for c in cells}, key=lambda w: (family_of(w), w))
def agg(wl, k): return st.mean([c[k] for c in cells if c["workload"] == wl])
perwl = [{"workload": wl, "short": short(wl), "family": family_of(wl),
          "kind": "threat" if is_threat(wl) else "benign",
          "apf_mean": agg(wl, "apf_mean"), "apf_cov": agg(wl, "apf_cov"),
          "wr_rate_mbs": agg(wl, "wr_rate_mbs"), "wr_total_mb": agg(wl, "wr_total_mb"),
          "n_windows": agg(wl, "n_windows")} for wl in WL]

# ---- 66-cell CSV ------------------------------------------------------------
csv_cols = ["family", "workload", "kind", "duration_s", "rep", "apf_mean", "apf_cov",
            "apf_max", "duty_gt05", "n_windows", "wr_rate_mbs", "wr_total_mb", "rd_rate_mbs"]
with (DOCS / "plan06_all_data.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=csv_cols, extrasaction="ignore")
    w.writeheader()
    for c in cells:
        w.writerow(c)

lift = json.load(open(LIFT))["results"]

# ---- glossary (drives the hover tooltips AND the glossary table) -------------
GLOSSARY = {
    "APF": "Active Page Fraction: fraction of 4 KB guest RAM pages that changed "
           "between two consecutive memory snapshots. The core memory signal.",
    "disk I/O": "Bytes the VM reads/writes to its virtual disk, measured host-side. "
                "The Plan 06 second channel.",
    "diskio": "Short name for the disk-I/O channel: per-snapshot rd/wr byte counters "
              "read via domblkstat.",
    "pmemsave": "QEMU monitor command that dumps guest RAM to a file. APF is computed "
                "from two consecutive dumps.",
    "domblkstat": "libvirt command returning a VM disk's cumulative rd_bytes/wr_bytes. "
                  "Disk rate = delta / elapsed time.",
    "peak/var": "Peak/variance shape features of the APF trajectory (apf_max, apf_p95, "
                "apf_cov, peak-to-median, duty cycle), beyond its mean.",
    "CoV": "Coefficient of Variation = std / mean. Unitless measure of how bursty the signal is.",
    "cell": "One capture run = (workload, duration, replicate). The unit of the 66-cell matrix.",
    "replicate": "A repeated run of the same workload and duration, used to measure "
                 "run-to-run variance.",
    "windows": "Count of analysis windows a cell yields (its degrees of freedom). "
               "Too few = starved.",
    "window": "A sliding analysis window over the trajectory, W = 8 snapshots wide.",
    "hop": "Step between consecutive analysis windows, H = 4 snapshots.",
    "DOF": "Degrees of freedom: number of analysis windows a cell produces at the "
           "frozen (W,H) = (8,4).",
    "family": "A group of workloads sharing behavior: ransomware, scanner, mem_sweep, "
              "mem_fault, app.",
    "instance": "An individual workload (11 total). Instance-level ID names the exact workload.",
    "threat": "Malicious-class workload: the 4 ransomware variants + the scanner.",
    "benign": "Non-malicious workload: the mem_* family members + app_hashtable.",
    "masquerade": "A threat whose memory signal mimics a benign one (slowburn looks like "
                  "pagefault in APF).",
    "baseline": "Majority-class accuracy: the score from always guessing the most common "
                "label. Beating it = real signal.",
    "LORO": "Leave-One-Replicate-Out CV: test on a held-out repeat of a workload the model "
            "already saw. Optimistic (memorization-prone).",
    "LOWO": "Leave-One-Workload-Out CV: test on a workload never seen in training. Honest "
            "novel-workload generalization.",
    "LOFO": "Leave-One-Family-Out CV: test on an entire behavior family never seen in "
            "training. The hardest split.",
}
glossary_rows = "\n".join(
    f'<tr><td class="mono">{t}</td><td>{d}</td></tr>' for t, d in GLOSSARY.items())
glossary_section = (
    '<details id="glossary" open>'
    '<summary>Glossary &mdash; jargon used on this page (underlined terms also explain on hover)</summary>'
    '<table><thead><tr><th>term</th><th>meaning</th></tr></thead><tbody>'
    f'{glossary_rows}</tbody></table></details>')

WALKER_JS = r'''(function(){
  var terms=Object.keys(GLOSSARY).sort(function(a,b){return b.length-a.length;});
  function esc(s){return s.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');}
  var re=new RegExp('(?<!\\w)('+terms.map(esc).join('|')+')(?!\\w)','gi');
  var low={}; for(var k in GLOSSARY){low[k.toLowerCase()]=GLOSSARY[k];}
  var SKIPTAG={SCRIPT:1,STYLE:1,CODE:1,SUMMARY:1};
  var SKIPID={glossary:1,celltable:1};
  function walk(node){
    if(node.nodeType===1){
      if(SKIPTAG[node.tagName]) return;
      if(node.id&&SKIPID[node.id]) return;
      if(node.classList&&(node.classList.contains('term')||node.classList.contains('mono'))) return;
      var kids=Array.prototype.slice.call(node.childNodes);
      for(var i=0;i<kids.length;i++) walk(kids[i]);
    } else if(node.nodeType===3){
      var txt=node.nodeValue;
      re.lastIndex=0;
      if(!re.test(txt)) return;
      var span=document.createElement('span');
      span.innerHTML=txt.replace(re,function(m){
        var d=(low[m.toLowerCase()]||'').replace(/"/g,'&quot;');
        return '<span class="term" tabindex="0" data-def="'+d+'">'+m+'</span>';
      });
      var frag=Array.prototype.slice.call(span.childNodes), parent=node.parentNode;
      for(var j=0;j<frag.length;j++) parent.insertBefore(frag[j],node);
      parent.removeChild(node);
    }
  }
  document.addEventListener('DOMContentLoaded',function(){
    var m=document.querySelector('main'); if(m) walk(m);
  });
})();'''
script_html = '<script>\nconst GLOSSARY=' + json.dumps(GLOSSARY, ensure_ascii=False) + ';\n' + WALKER_JS + '\n</script>'


def b64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def savepng(fig, name):
    fig.savefig(FIGDIR / name, format="png", dpi=130, bbox_inches="tight")


figs = {}

# Fig 1: 2-channel scatter (APF vs disk write rate), the money chart
fig, ax = plt.subplots(figsize=(8, 5.5))
for c in cells:
    ax.scatter(max(c["apf_mean"], 1e-3), max(c["wr_rate_mbs"], 1e-3),
               c=FAM_COLOR[c["family"]], marker=("X" if c["kind"] == "threat" else "o"),
               s=70, edgecolors="white", linewidths=0.5, alpha=0.85)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("APF mean  (memory channel, Plan 05)")
ax.set_ylabel("disk write rate MB/s  (Plan 06)")
ax.set_title("Two-channel behavior space: disk axis disambiguates memory-aliased workloads")
ax.grid(True, which="both", ls=":", alpha=0.4)
# annotate the masquerade pair
for c in cells:
    if c["short"] in ("ransom_slowburn", "pagefault") and c["rep"] == 1 and c["duration_s"] == 600:
        ax.annotate(c["short"], (max(c["apf_mean"], 1e-3), max(c["wr_rate_mbs"], 1e-3)),
                    fontsize=8, xytext=(5, 5), textcoords="offset points")
fam_leg = [Line2D([0], [0], marker="s", color="w", markerfacecolor=v, markersize=9, label=k)
           for k, v in FAM_COLOR.items()]
kind_leg = [Line2D([0], [0], marker="X", color="gray", linestyle="", markersize=9, label="threat"),
            Line2D([0], [0], marker="o", color="gray", linestyle="", markersize=9, label="benign")]
ax.legend(handles=fam_leg + kind_leg, fontsize=8, loc="lower right", framealpha=0.9)
savepng(fig, "scatter_2channel.png"); figs["scatter"] = b64(fig)

# Fig 2: per-workload APF mean bar
fig, ax = plt.subplots(figsize=(8, 4.2))
pw = sorted(perwl, key=lambda x: -x["apf_mean"])
ax.bar([p["short"] for p in pw], [p["apf_mean"] for p in pw],
       color=[FAM_COLOR[p["family"]] for p in pw])
ax.set_ylabel("APF mean"); ax.set_title("Memory channel (Plan 05): APF by workload")
ax.tick_params(axis="x", rotation=55, labelsize=8)
for lbl, p in zip(ax.get_xticklabels(), pw):
    lbl.set_ha("right")
savepng(fig, "bar_apf.png"); figs["apf_bar"] = b64(fig)

# Fig 3: per-workload disk write rate bar (log)
fig, ax = plt.subplots(figsize=(8, 4.2))
pw = sorted(perwl, key=lambda x: -x["wr_rate_mbs"])
ax.bar([p["short"] for p in pw], [max(p["wr_rate_mbs"], 1e-3) for p in pw],
       color=[FAM_COLOR[p["family"]] for p in pw])
ax.set_yscale("log"); ax.set_ylabel("disk write rate MB/s (log)")
ax.set_title("Disk channel (Plan 06): write rate by workload  (benign mmap is the heaviest)")
ax.tick_params(axis="x", rotation=55, labelsize=8)
for lbl in ax.get_xticklabels():
    lbl.set_ha("right")
savepng(fig, "bar_disk.png"); figs["disk_bar"] = b64(fig)

# Fig 4: classification lift grouped bars
tasks = [("binary LOWO", lambda r: r["binary_lowo"]["lowo_acc"]),
         ("binary LOFO", lambda r: r["binary_lofo"]["lofo_acc"]),
         ("family LORO", lambda r: r["family5_lowo"]["loro_acc"]),
         ("family LOWO", lambda r: r["family5_lowo"]["lowo_acc"]),
         ("instance LORO", lambda r: r["instance11_loro"]["loro_acc"])]
sets = ["AGNOSTIC", "AGNOSTIC+peakvar", "AGNOSTIC+peakvar+diskio"]
setcol = {"AGNOSTIC": "#9aa0a8", "AGNOSTIC+peakvar": "#4f86c6", "AGNOSTIC+peakvar+diskio": "#e8553a"}
import numpy as np
x = np.arange(len(tasks)); w = 0.26
fig, ax = plt.subplots(figsize=(9, 4.6))
for i, sname in enumerate(sets):
    ax.bar(x + (i - 1) * w, [g(lift[sname]) for _, g in tasks], w,
           label=sname.replace("AGNOSTIC", "APF").replace("+peakvar", "+pv").replace("+diskio", "+disk"),
           color=setcol[sname])
ax.set_xticks(x); ax.set_xticklabels([t for t, _ in tasks], fontsize=9)
ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.05)
ax.set_title("Classification: memory (APF) -> +shape -> +disk  (disk helps ID, hurts threat/benign)")
ax.axhline(0.545, ls="--", c="#888", lw=0.8); ax.text(4.3, 0.55, "binary baseline", fontsize=7, color="#888")
ax.legend(fontsize=8); ax.grid(axis="y", ls=":", alpha=0.4)
savepng(fig, "bar_classification.png"); figs["clf_bar"] = b64(fig)

# Fig 5: per-workload metric heatmap (normalized columns)
metrics = [("apf_mean", "APF mean"), ("apf_cov", "APF CoV"), ("n_windows", "windows"),
           ("wr_rate_mbs", "disk MB/s"), ("wr_total_mb", "disk total MB")]
pw = sorted(perwl, key=lambda x: (x["family"], x["short"]))
mat = []
for mk, _ in metrics:
    col = [p[mk] for p in pw]
    lo, hi = min(col), max(col)
    mat.append([(v - lo) / (hi - lo) if hi > lo else 0.0 for v in col])
mat = np.array(mat)
fig, ax = plt.subplots(figsize=(9, 3.6))
im = ax.imshow(mat, aspect="auto", cmap="magma")
ax.set_xticks(range(len(pw))); ax.set_xticklabels([p["short"] for p in pw], rotation=55, ha="right", fontsize=7.5)
ax.set_yticks(range(len(metrics))); ax.set_yticklabels([m[1] for m in metrics], fontsize=8)
ax.set_title("Per-workload metric heatmap (each row min-max normalized)")
fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
savepng(fig, "heatmap.png"); figs["heatmap"] = b64(fig)

print(f"figures -> {FIGDIR}")
print(f"csv -> {DOCS / 'plan06_all_data.csv'} ({len(cells)} cells)")

# ---- HTML -------------------------------------------------------------------
def img(key, cap):
    return (f'<figure><img src="data:image/png;base64,{figs[key]}" style="width:100%;max-width:820px">'
            f'<figcaption>{cap}</figcaption></figure>')


def cell_rows():
    out = []
    for c in cells:
        out.append(
            f'<tr><td>{c["family"]}</td><td class="mono">{c["short"]}</td>'
            f'<td class="{c["kind"]}">{c["kind"]}</td><td class="r">{c["duration_s"]}</td>'
            f'<td class="r">{c["rep"]}</td><td class="r">{c["apf_mean"]:.4f}</td>'
            f'<td class="r">{c["apf_cov"]:.2f}</td><td class="r">{c["n_windows"]}</td>'
            f'<td class="r">{c["wr_rate_mbs"]:.3f}</td><td class="r">{c["wr_total_mb"]:.1f}</td></tr>')
    return "\n".join(out)


def wl_rows():
    out = []
    for p in sorted(perwl, key=lambda x: (x["family"], x["short"])):
        out.append(
            f'<tr><td>{p["family"]}</td><td class="mono">{p["short"]}</td>'
            f'<td class="{p["kind"]}">{p["kind"]}</td><td class="r">{p["apf_mean"]:.4f}</td>'
            f'<td class="r">{p["apf_cov"]:.2f}</td><td class="r">{p["wr_rate_mbs"]:.3f}</td>'
            f'<td class="r">{p["wr_total_mb"]:.1f}</td><td class="r">{p["n_windows"]:.0f}</td></tr>')
    return "\n".join(out)


def clf_rows():
    out = []
    for tname, g in tasks:
        a, pv, dk = (g(lift["AGNOSTIC"]), g(lift["AGNOSTIC+peakvar"]),
                     g(lift["AGNOSTIC+peakvar+diskio"]))
        arrow = "&#8593;" if dk > pv else ("&#8595;" if dk < pv else "&ndash;")
        out.append(f'<tr><td>{tname}</td><td class="r">{a:.3f}</td><td class="r">{pv:.3f}</td>'
                   f'<td class="r"><b>{dk:.3f}</b> {arrow}</td></tr>')
    return "\n".join(out)


html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Plan 05 -> Plan 06 -- comprehensive data</title>
<style>
 :root{{--bg:#15171a;--card:#1d2024;--text:#e8e8ea;--muted:#9aa0a8;--line:#2c3036;
        --good:#7cb96e;--bad:#e8553a;}}
 body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.6 -apple-system,Segoe UI,Roboto,sans-serif}}
 main{{max-width:900px;margin:0 auto;padding:36px 20px 80px}}
 h1{{font-size:24px;margin:0 0 4px}} h2{{font-size:18px;margin:30px 0 8px;color:#cfe0f0}}
 .sub{{color:var(--muted);margin:0 0 20px}}
 table{{width:100%;border-collapse:collapse;font-size:12.5px;margin:10px 0}}
 th,td{{padding:5px 8px;border-bottom:1px solid var(--line);text-align:left}}
 th{{color:var(--muted);position:sticky;top:0;background:var(--card)}}
 .r{{text-align:right}} .mono{{font-family:ui-monospace,Menlo,monospace}}
 .threat{{color:var(--bad)}} .benign{{color:var(--good)}}
 figure{{margin:14px 0}} figcaption{{color:var(--muted);font-size:12px;margin-top:4px}}
 .wrap{{max-height:560px;overflow:auto;border:1px solid var(--line);border-radius:8px}}
 .note{{background:#22262b;border-left:3px solid #4f86c6;padding:10px 14px;border-radius:6px;margin:12px 0}}
 .term{{position:relative;border-bottom:1px dotted var(--muted);cursor:help}}
 .term:hover::after,.term:focus::after{{content:attr(data-def);position:absolute;left:0;top:142%;z-index:40;
   width:max-content;max-width:300px;background:#0d0f12;color:var(--text);border:1px solid var(--line);
   border-radius:6px;padding:8px 11px;font-size:12px;font-weight:400;line-height:1.45;white-space:normal;
   box-shadow:0 8px 24px rgba(0,0,0,.55)}}
 details#glossary{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px 16px;margin:16px 0}}
 details#glossary summary{{cursor:pointer;color:#cfe0f0;font-size:16px;font-weight:600;padding:6px 0}}
 details#glossary td:first-child{{white-space:nowrap;color:#cfe0f0;font-weight:600}}
</style></head><body><main>
<h1>Plan 05 &rarr; Plan 06 &mdash; comprehensive data</h1>
<p class="sub">All 66 cells, all 11 workloads, both channels. Memory (APF) is the
Plan 05 view; the disk columns are the Plan 06 addition. Hover any
<span class="term" data-def="Like this: hover a dotted-underlined term to see its definition.">underlined term</span>
for its definition, or use the glossary below.</p>

{glossary_section}

<h2>1. Two-channel behavior space (the headline view)</h2>
{img("scatter", "Each point is one cell. X = APF mean (memory). Y = disk write rate (Plan 06). "
     "Workloads that overlap on the APF axis (e.g. slowburn vs pagefault, both low) separate on "
     "the disk axis. Note the heaviest writer is benign (mmap) &mdash; disk is a behavior axis, not a threat axis.")}

<h2>2. Memory vs disk, per workload</h2>
{img("apf_bar", "Memory channel: APF mean per workload (color = family).")}
{img("disk_bar", "Disk channel: write rate per workload, log scale. Benign mmap dominates; "
     "memory-capped threat batched writes almost nothing.")}
{img("heatmap", "Per-workload metric heatmap, each row min-max normalized.")}

<table><thead><tr><th>family</th><th>workload</th><th>kind</th><th class="r">APF mean</th>
<th class="r">APF CoV</th><th class="r">disk MB/s</th><th class="r">disk total MB</th>
<th class="r">windows</th></tr></thead><tbody>
{wl_rows()}
</tbody></table>

<h2>3. Classification: what each channel buys</h2>
{img("clf_bar", "Accuracy by feature set (APF -> +shape -> +disk) across tasks. Disk helps "
     "identification (family/instance) and hurts threat/benign detection.")}
<table><thead><tr><th>task</th><th class="r">APF only</th><th class="r">+peak/var</th>
<th class="r">+disk</th></tr></thead><tbody>
{clf_rows()}
</tbody></table>
<div class="note">Disk I/O <b>helps characterization</b> (family LORO 0.955&rarr;0.985,
family LOWO 0.348&rarr;0.455 crossing the 0.364 baseline, instance 0.924&rarr;0.939) and
<b>hurts detection</b> (binary LOWO 0.742&rarr;0.636) &mdash; a characterization channel, not a detector.</div>

<h2>4. Every cell (66)</h2>
<div class="wrap"><table id="celltable"><thead><tr><th>family</th><th>workload</th><th>kind</th>
<th class="r">dur s</th><th class="r">rep</th><th class="r">APF mean</th><th class="r">APF CoV</th>
<th class="r">windows</th><th class="r">disk MB/s</th><th class="r">disk total MB</th></tr></thead><tbody>
{cell_rows()}
</tbody></table></div>
<p class="sub">Full table also at <code>docs/plan06_all_data.csv</code>; standalone figures in
<code>docs/plan06_figs/</code>.</p>
</main>
{script_html}
</body></html>"""

(DOCS / "plan06_comprehensive.html").write_text(html)
print(f"html -> {DOCS / 'plan06_comprehensive.html'}")
