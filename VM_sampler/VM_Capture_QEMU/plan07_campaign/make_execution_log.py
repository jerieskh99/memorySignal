#!/usr/bin/env python3
"""Plan 07 -- EXECUTION LOG (HTML). One growing document; one section per step actually run.

Each step records: what it is, the motivation, what it unlocks (blocks downstream), the full
results, figures, and the conclusions. Re-run after each step to refresh.

Step 1 reads plan07_campaign/pilot_baseline.json (the peakvar_lift run on the 66 cells).
Later steps append their own sections as their outputs appear.

    python3 plan07_campaign/make_execution_log.py [pilot_baseline.json]  -> docs/plan07_execution_log.html
"""
import base64
import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent / "docs"
DOCS.mkdir(exist_ok=True)

P = {"ink": "#1A1A1A", "muted": "#5B5B5B", "line": "#D8D6CE", "panel": "#F5F3EE",
     "agn": "#9AA0A8", "pv": "#3F6CA8", "gold": "#B9822A", "threat": "#C0392B",
     "benign": "#2E7D52", "good": "#2E7D52", "bad": "#C0392B",
     "ransomware": "#C0392B", "scanner": "#E08A1E", "mem_sweep": "#3F6CA8",
     "mem_fault": "#2C9B8F", "app": "#2E7D52"}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                     "font.size": 10.5, "axes.titlesize": 12, "axes.titleweight": "bold",
                     "figure.facecolor": "white", "savefig.facecolor": "white",
                     "axes.edgecolor": P["muted"], "xtick.color": P["muted"], "ytick.color": P["muted"]})

# workload order + short names, grouped by family (threat first)
ORDER = [("sandbox_ransom_seq", "seq", "ransomware"), ("sandbox_ransom_batched", "batched", "ransomware"),
         ("sandbox_ransom_slowburn", "slowburn", "ransomware"), ("sandbox_ransom_selective", "selective", "ransomware"),
         ("sandbox_scanner_metadata", "scanner", "scanner"), ("mem_rmw_intensity_v2", "rmw", "mem_sweep"),
         ("mem_workingset_sweep_v2", "workingset", "mem_sweep"), ("mem_writemag_sweep_v2", "writemag", "mem_sweep"),
         ("mem_mmap_traversal_v2", "mmap", "mem_fault"), ("mem_pagefault_density_v2", "pagefault", "mem_fault"),
         ("app_hashtable_intensive_v2", "hashtable", "app")]


# ---- glossary: one source drives the hover tooltips AND the glossary table -------------
GLOSSARY = {
    "APF": "Active Page Fraction: the fraction of 4 KB guest memory pages that change between two "
           "consecutive snapshots. The core memory signal.",
    "cell": "One capture run = (workload, duration, replicate). The unit of the dataset; 66 here.",
    "workload": "A single program that drives the VM (e.g. a memory sweep or a safe file-encryptor "
                "simulation). 11 distinct ones in the current data.",
    "family": "A group of workloads with shared behaviour: ransomware, scanner, mem_sweep, "
              "mem_fault, app.",
    "feature set": "The list of numbers extracted per cell that the models actually see. Two are "
                   "compared: AGNOSTIC and +peakvar.",
    "AGNOSTIC": "The 7 leakage-free features: apf_mean, apf_std, two cepstral (periodicity) "
                "features, stat_pass_frac, n_pairs, n_windows.",
    "leakage-free": "A feature that does NOT use the family label in its own computation, so it "
                    "cannot secretly re-encode the answer the model is supposed to predict.",
    "peakvar": "Five shape features of the APF trajectory (apf_max, apf_p95, apf_cov, peak-to-median, "
               "duty cycle) -- its burstiness, not just its average level.",
    "instance identification": "Naming which of the 11 specific workloads a trace is.",
    "family identification": "Naming which of the 5 behaviour families a trace belongs to.",
    "threat/benign detection": "The two-class call: is this a (simulated) threat or benign? "
                               "'Threat' = the safe ransomware/scanner simulations.",
    "one-class novelty detection": "Train ONLY on benign examples, learn a model of 'normal', and "
                                   "flag whatever deviates -- no threat is ever shown in training.",
    "cross-validation": "Splitting the data into train/test repeatedly so the reported score "
                        "reflects unseen data, not memorisation.",
    "LORO": "Leave-One-Replicate-Out: hold out one repeat of a workload the model has otherwise "
            "seen. Optimistic ('seen-before').",
    "LOWO": "Leave-One-Workload-Out: hold out an entire workload never seen in training. The honest "
            "novel-workload test.",
    "LOFO": "Leave-One-Family-Out: hold out a whole family. The hardest test -- generalising to a "
            "new KIND of behaviour.",
    "LOFO-benign": "A one-class variant: hold a whole benign family out of training, then check the "
                   "detector still flags threats without false-alarming on that unseen benign family.",
    "ROC-AUC": "Area under the ROC curve: the probability the detector ranks a random threat above a "
               "random benign. 0.5 = chance, 1.0 = perfect.",
    "LOF": "Local Outlier Factor: a one-class detector that flags points sitting in low-density "
           "regions relative to their neighbours.",
    "OCSVM": "One-Class SVM: a one-class detector that learns a boundary around the benign region.",
    "IsolationForest": "A one-class detector that flags points easily isolated by random splits.",
    "prior": "The majority-class baseline: the score from always guessing the most common label "
             "(0.545 for threat/benign here). Beating it = real signal.",
    "regression anchor": "A fixed reference result computed before any change, so later edits can be "
                         "judged as an improvement or a regression against it.",
    "characterisation": "Identifying WHICH known behaviour a trace is (its fingerprint), when its "
                        "kind has been seen before.",
    "generalisation": "Handling something NOT seen in training -- an unseen workload, or an unseen "
                      "family.",
    "masquerade": "A threat whose memory signal mimics a benign one (slowburn looks like pagefault).",
    "cohesion": "Whether the workloads of a family resemble each other in the signal -- needed for a "
                "single 'family-normal' model to represent the family.",
    "macro_f1": "The F1 score (harmonic mean of precision and recall) averaged equally over classes, "
                "so rare classes count as much as common ones.",
    "weighted Hamming": "Per changed page, the number of bits that flipped: h = popcount(prev XOR "
                        "curr). Weights each page by HOW HARD it changed, unlike APF's yes/no page count.",
    "byte entropy": "Shannon entropy of a page's 256-bin byte histogram, 0-8 bits/byte. ~8 = "
                    "random / encrypted-looking; ~4-5 = structured. The encryption tell.",
    "ent_pooled": "Entropy of ALL changed bytes this snapshot merged into one histogram -- a single "
                  "global number, the entropy twin of ham_sum. NOT the average of per-page entropies.",
    "magnitude distribution": "The per-page weighted Hamming summarised over the changed pages: "
                              "ham_mean, ham_max, ham_p95, ham_std.",
    "popcount": "Population count: the number of 1-bits in a value. popcount(prev XOR curr) = the "
                "bits that flipped between two versions of a page.",
    "Task C": "The pending capture-side code that emits the depth axes (magnitude + entropy) "
              "additively, flag-gated (TIMING_MAGENT), byte-identical when off.",
}
_glossary_rows = "\n".join(
    f'<tr><td class="gt">{t}</td><td>{d}</td></tr>' for t, d in GLOSSARY.items())
GLOSSARY_SECTION = (
    '<details id="glossary"><summary>Glossary &mdash; jargon used in this log '
    '(underlined terms also explain on hover)</summary>'
    '<table><thead><tr><th>term</th><th>meaning</th></tr></thead><tbody>'
    f'{_glossary_rows}</tbody></table></details>')
_WALKER_JS = r'''(function(){
  var terms=Object.keys(GLOSSARY).sort(function(a,b){return b.length-a.length;});
  function esc(s){return s.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');}
  var re=new RegExp('(?<!\\w)('+terms.map(esc).join('|')+')(?!\\w)','g');
  var low={}; for(var k in GLOSSARY){low[k.toLowerCase()]=GLOSSARY[k];}
  var SKIP={SCRIPT:1,STYLE:1,CODE:1,SUMMARY:1,TH:1};
  var wrapped={};   // first occurrence of each term only
  function walk(node){
    if(node.nodeType===1){
      if(SKIP[node.tagName]) return;
      if(node.id==='glossary') return;
      if(node.classList&&node.classList.contains('term')) return;
      var kids=Array.prototype.slice.call(node.childNodes);
      for(var i=0;i<kids.length;i++) walk(kids[i]);
    } else if(node.nodeType===3){
      var txt=node.nodeValue; re.lastIndex=0;
      if(!re.test(txt)) return;
      var span=document.createElement('span');
      span.innerHTML=txt.replace(re,function(m){
        var key=m.toLowerCase();
        if(wrapped[key]) return m;
        wrapped[key]=1;
        var d=(low[key]||'').replace(/"/g,'&quot;');
        return '<span class="term" tabindex="0" data-def="'+d+'">'+m+'</span>';
      });
      var frag=Array.prototype.slice.call(span.childNodes), parent=node.parentNode;
      for(var j=0;j<frag.length;j++) parent.insertBefore(frag[j],node);
      parent.removeChild(node);
    }
  }
  document.addEventListener('DOMContentLoaded',function(){
    var m=document.querySelector('.page'); if(m) walk(m);
  });
})();'''
SCRIPT_HTML = '<script>\nconst GLOSSARY=' + json.dumps(GLOSSARY, ensure_ascii=False) + ';\n' + _WALKER_JS + '\n</script>'


def b64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=140, bbox_inches="tight"); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def fig_headline(A, Pv):
    tasks = [("instance\nLORO", A["instance11"]["loro_acc"], Pv["instance11"]["loro_acc"]),
             ("binary\nLORO", A["binary"]["loro_acc"], Pv["binary"]["loro_acc"]),
             ("binary\nLOWO", A["binary"]["lowo_acc"], Pv["binary"]["lowo_acc"]),
             ("binary\nLOFO", A["binary_lofo"]["lofo_acc"], Pv["binary_lofo"]["lofo_acc"]),
             ("family\nLORO", A["family5"]["loro_acc"], Pv["family5"]["loro_acc"]),
             ("family\nLOWO", A["family5"]["lowo_acc"], Pv["family5"]["lowo_acc"]),
             ("1-class LOF\nhonest", A["anomaly_roc_auc_honest"]["LOF"], Pv["anomaly_roc_auc_honest"]["LOF"]),
             ("1-class LOF\nLOFO-benign", A["anomaly_roc_auc_lofo_benign"]["LOF"], Pv["anomaly_roc_auc_lofo_benign"]["LOF"])]
    x = np.arange(len(tasks)); w = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.bar(x - w / 2, [t[1] for t in tasks], w, color=P["agn"], label="AGNOSTIC (7 feats)")
    ax.bar(x + w / 2, [t[2] for t in tasks], w, color=P["pv"], label="+ peakvar (12 feats)")
    for xi, t in zip(x, tasks):
        ax.text(xi + w / 2, t[2] + 0.015, f"{t[2]:.2f}", ha="center", fontsize=8, fontweight="bold")
    ax.axhline(0.545, ls="--", c="#888", lw=1); ax.text(7.4, 0.56, "binary prior 0.545", fontsize=7.5, color="#888", ha="right")
    ax.set_xticks(x); ax.set_xticklabels([t[0] for t in tasks], fontsize=8.5)
    ax.set_ylim(0, 1.05); ax.set_ylabel("accuracy / ROC-AUC")
    ax.set_title("Step 1 headline: seen-before (LORO) high; novel (LOWO/LOFO) hard; one-class strong")
    ax.legend(fontsize=8.5, loc="lower left"); ax.grid(axis="y", ls=":", alpha=0.4)
    return b64(fig)


def fig_detector(A, Pv):
    dets = ["LOF", "OCSVM", "IsolationForest"]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    x = np.arange(len(dets)); w = 0.38
    ax.bar(x - w / 2, [Pv["anomaly_roc_auc_honest"][d] for d in dets], w, color=P["pv"], label="honest (leave-one-workload-out)")
    ax.bar(x + w / 2, [Pv["anomaly_roc_auc_lofo_benign"][d] for d in dets], w, color=P["gold"], label="novel benign FAMILY (LOFO-benign)")
    for xi, d in zip(x, dets):
        ax.text(xi - w / 2, Pv["anomaly_roc_auc_honest"][d] + 0.012, f"{Pv['anomaly_roc_auc_honest'][d]:.2f}", ha="center", fontsize=8, fontweight="bold")
        ax.text(xi + w / 2, Pv["anomaly_roc_auc_lofo_benign"][d] + 0.012, f"{Pv['anomaly_roc_auc_lofo_benign'][d]:.2f}", ha="center", fontsize=8, fontweight="bold")
    ax.axhline(Pv["binary"]["lowo_acc"], ls="-", c=P["threat"], lw=1.4)
    ax.text(2.4, Pv["binary"]["lowo_acc"] + 0.012, f"supervised binary LOWO {Pv['binary']['lowo_acc']:.2f}", fontsize=8, color=P["threat"], ha="right")
    ax.axhline(0.545, ls="--", c="#888", lw=1); ax.text(2.4, 0.50, "prior 0.545", fontsize=7.5, color="#888", ha="right")
    ax.set_xticks(x); ax.set_xticklabels(dets); ax.set_ylim(0, 1.05); ax.set_ylabel("ROC-AUC")
    ax.set_title("The hidden detector (+peakvar): one-class novelty >> supervised boundary")
    ax.legend(fontsize=8.5, loc="lower center"); ax.grid(axis="y", ls=":", alpha=0.4)
    return b64(fig)


def fig_heatmap(Pv):
    cols = [("binary LOWO", Pv["binary"]["lowo_per_workload"]),
            ("binary LOFO", Pv["binary_lofo"]["per_workload"]),
            ("family LOWO", Pv["family5"]["lowo_per_workload"])]
    M = np.array([[col.get(full, np.nan) for (_, col) in cols] for (full, short, fam) in ORDER])
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[0] for c in cols], fontsize=9)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([f"{s}" for (_, s, _) in ORDER], fontsize=8.5)
    for yi, (_, s, fam) in enumerate(ORDER):
        ax.get_yticklabels()[yi].set_color(P[fam])
    for yi in range(len(ORDER)):
        for xi in range(len(cols)):
            v = M[yi, xi]
            if not np.isnan(v):
                ax.text(xi, yi, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="black" if 0.25 < v < 0.85 else "white")
    # family separators
    bounds = [4, 5, 8, 10]
    for b in bounds:
        ax.axhline(b - 0.5, color="white", lw=2)
    ax.set_title("Where it fails (per-workload, +peakvar)\nred = missed; rows coloured by family")
    fig.text(0.5, 0.005, "slowburn (threat) and pagefault (benign) are the consistent 0.00 -- the masquerade pair",
             ha="center", fontsize=8, color=P["muted"])
    return b64(fig)


def step1(pilot_path):
    d = json.load(open(pilot_path))
    A = d["results"]["AGNOSTIC"]; Pv = d["results"]["AGNOSTIC+peakvar"]
    f1, f2, f3 = fig_headline(A, Pv), fig_detector(A, Pv), fig_heatmap(Pv)

    def img(b, cap):
        return (f"<figure><img src='data:image/png;base64,{b}' style='width:100%;max-width:780px'>"
                f"<figcaption>{cap}</figcaption></figure>")

    # headline metric table (both feature sets)
    rows = [("instance ID (LORO, seen-before)", A["instance11"]["loro_acc"], Pv["instance11"]["loro_acc"]),
            ("binary threat/benign (LORO)", A["binary"]["loro_acc"], Pv["binary"]["loro_acc"]),
            ("binary threat/benign (LOWO, novel workload)", A["binary"]["lowo_acc"], Pv["binary"]["lowo_acc"]),
            ("binary threat/benign (LOFO, novel family)", A["binary_lofo"]["lofo_acc"], Pv["binary_lofo"]["lofo_acc"]),
            ("family ID (LORO, seen-before)", A["family5"]["loro_acc"], Pv["family5"]["loro_acc"]),
            ("family ID (LOWO, novel workload)", A["family5"]["lowo_acc"], Pv["family5"]["lowo_acc"]),
            ("one-class LOF (honest LOWO)", A["anomaly_roc_auc_honest"]["LOF"], Pv["anomaly_roc_auc_honest"]["LOF"]),
            ("one-class LOF (LOFO-benign)", A["anomaly_roc_auc_lofo_benign"]["LOF"], Pv["anomaly_roc_auc_lofo_benign"]["LOF"]),
            ("one-class OCSVM (honest LOWO)", A["anomaly_roc_auc_honest"]["OCSVM"], Pv["anomaly_roc_auc_honest"]["OCSVM"])]
    mtable = "<table><thead><tr><th>metric</th><th>AGNOSTIC</th><th>+ peakvar</th></tr></thead><tbody>" + \
        "".join(f"<tr><td>{n}</td><td class='r'>{a:.4f}</td><td class='r'><b>{p:.4f}</b></td></tr>" for n, a, p in rows) + \
        "</tbody></table>"

    return f"""
<section>
<h2><span class='sn'>Step 1</span> &mdash; offline pilot baseline (the regression anchor)</h2>
<div class='meta'>script: <code>plan05_campaign/peakvar_lift.py</code> &middot; data: 66 cells
(<code>cells_full</code>) &middot; features: 7 AGNOSTIC + 5 peakvar &middot; sklearn {d['sklearn']}</div>

<h3>What this step is</h3>
<p>The first move runs our existing analysis harness (the script <code>peakvar_lift.py</code>) on
the 66 cells we already captured &mdash; no new data. The harness turns each cell's APF trajectory
into a small vector of features, then trains a battery of models and scores them with honest
cross-validation. Nothing about the capture or the workloads changes; this is pure offline
re-analysis of data already in hand.</p>
<p>It runs two feature sets side by side, so we can see exactly what the shape of the signal buys:
the 7 AGNOSTIC features alone, and those plus the 5 peakvar shape features. On each feature set it
performs four distinct tasks:</p>
<ul>
<li>instance identification &mdash; name which of the 11 specific workloads a trace is;</li>
<li>threat/benign detection &mdash; the two-class call, a simulated threat or benign;</li>
<li>family identification &mdash; name which of the 5 behaviour families;</li>
<li>one-class novelty detection &mdash; the open-set detector: train only on benign, learn
'normal', and flag whatever deviates, with no threat ever shown during training.</li>
</ul>
<p>Every task is then scored under three cross-validation regimes that differ only in <i>what is
held out of training</i>, in increasing order of honesty. LORO holds out one replicate of a
workload the model has otherwise seen (optimistic, 'seen-before'). LOWO holds out an entire
workload never seen in training (the honest novel-workload test). LOFO holds out a whole family
(the hardest test, generalising to a new kind of behaviour). The result is
<code>pilot_baseline.json</code>: an accuracy or ROC-AUC for every task, feature set, and regime,
plus a per-workload breakdown of where each model succeeds and fails.</p>

<h3>Why we ran it first</h3>
<p>Two reasons, both structural.</p>
<p><b>(1) It is the regression anchor.</b> The whole plan pivots on one prior finding: that a
one-class novelty detection model, trained on benign only, already separates threats from benigns
far better than the supervised classifier that had made us conclude "detection is impossible".
Before we touch a single feature or model, we recompute that number on the full data, so every
later edit can be judged as an improvement or a regression against a fixed reference. Had this run
<i>not</i> reproduced the detector, the premise of the entire plan would be in doubt &mdash; so it
is the first thing to confirm.</p>
<p><b>(2) It maps the terrain.</b> The same run separates what already works from what is still
open. Identifying a workload or family whose <i>kind</i> has been seen before is characterisation,
and it works well &mdash; the LORO scores are high. Recognising an <i>unseen</i> workload or family
is generalisation, and it is the hard, open part &mdash; the LOWO and LOFO scores drop. That split
is precisely what the per-family recogniser must fix, and it is read straight off this one
baseline. Running it first therefore tells us, for zero cost and with no new capture, both that
there is a real signal to build on and exactly where the remaining work lies.</p>

<h3>The results</h3>
{mtable}
{img(f1, "Headline metrics, AGNOSTIC vs +peakvar. Seen-before tasks (LORO) are high; novel tasks "
    "(LOWO/LOFO) drop sharply for the supervised classifier; the one-class detector stays strong.")}
{img(f2, "The one-class novelty detector (with peakvar) far exceeds the supervised threat/benign "
    "boundary on novel workloads, and even on a never-seen benign family (LOFO-benign).")}
{img(f3, "Per-workload accuracy. The failures concentrate on two workloads: slowburn (a quiet "
    "threat) and pagefault (its benign twin), both at 0.00 -- the masquerade pair.")}

<h3>What it means, in plain terms</h3>
<p>Reading the table and figures against the three difficulty levels &mdash; <b>seen-before</b>
(LORO), <b>new workload</b> (LOWO), and <b>new family</b> (LOFO) &mdash; five things stand out.</p>

<p class='find'><b>1. Identifying something we have seen before works well.</b> When the model has
seen the workload's kind during training, it names the exact program correctly
{Pv['instance11']['loro_acc']*100:.0f}% of the time, and its family
{Pv['family5']['loro_acc']*100:.0f}% of the time. The memory signal clearly carries a usable
fingerprint.</p>

<p class='find'><b>2. The threat detector works &mdash; and the way we hoped.</b> There are two
ways to detect. The old way (show it threats and benigns and let it draw a line) manages only
{Pv['binary']['lowo_acc']*100:.0f}% on a brand-new workload. The new way (show it ONLY benign
behaviour, then flag anything that is not normal) scores
{Pv['anomaly_roc_auc_honest']['LOF']:.2f} on a new workload and
{Pv['anomaly_roc_auc_lofo_benign']['LOF']:.2f} on a whole benign family it never saw. That second
number is what the plan was betting on, and it came back strong &mdash; a little above the
0.925 / 0.933 we had been quoting.</p>

<p class='find'><b>3. The shape features do the heavy lifting.</b> With only the 7 basic features
the same detector scores about {A['anomaly_roc_auc_honest']['LOF']:.2f}; adding the 5 shape
features pushes it to {Pv['anomaly_roc_auc_honest']['LOF']:.2f}. It is the pattern of the
trajectory, not its loudness, that carries the signal.</p>

<p class='find'><b>4. Generalising to something genuinely new is still hard.</b> Forced onto an
unseen workload or family, naming the family collapses to {Pv['family5']['lowo_acc']*100:.0f}%
(barely above the ~36% you get by always guessing the biggest family), and hiding the entire
ransomware family drops ransomware recognition to just 25%. That 25% is an early warning that the
four ransomware simulations may not actually look alike in memory.</p>

<p class='find'><b>5. We can see exactly where it fails.</b> Workload by workload, almost
everything is caught (most at 100%), but two are consistently 0%: <b>slowburn</b> (a deliberately
quiet threat) and <b>pagefault</b> (a benign workload that looks just like it). This is the
masquerade &mdash; the threat that hides by looking benign, and exactly the case the new Hamming
metric targets.</p>

<div class='box'><b>Bottom line.</b> Three things are now established: the memory signal
fingerprints workloads we have seen (characterisation works); the one-class detector is real and
strong (the plan's premise is confirmed on the full data); and the shape of the trajectory, not
its loudness, carries the signal. One thing is the work ahead &mdash; recognising genuinely unseen
workloads and families, the gap the per-family recogniser exists to close.</div>

<h3>What it unlocks (blocks downstream)</h3>
<p>This anchor green-lights the per-family pilot. It directly motivates the next moves:
the <b>separability matrix</b> (Step 2a) &mdash; are slowburn and pagefault truly aliased in
memory, or just mis-learned by a global model? &mdash; and the <b>cohesion measurement</b>,
which turns the ransomware-0.25 hint into a number. The detector here is a single GLOBAL model
over all benign; the recogniser specialises it per family. Nothing downstream could be judged
without this baseline to compare against.</p>
</section>
"""


def metric_design():
    """Design note for the depth axes (magnitude + entropy) -- Task C, pending capture."""
    depth_rows = [
        ("<b>APF</b> &mdash; breadth: how many pages moved (the baseline, kept)",
         "&mdash; (already a per-snapshot fraction)",
         "<b>apf</b> = changed pages / total; <b>n_changed</b>"),
        ("<b>Magnitude</b> &mdash; weighted Hamming: how hard each page changed. h = popcount(prev XOR "
         "curr) = bits flipped.",
         "<b>ham_mean, ham_max, ham_p95, ham_std</b> &mdash; the <i>magnitude distribution</i> over "
         "changed pages",
         "<b>ham_sum</b> (total bits flipped), <b>ham_frac</b> (/ all memory bits)"),
        ("<b>Entropy</b> &mdash; byte randomness: how random the content looks (bits/byte, 0&ndash;8; "
         "~8 = ciphertext-like).",
         "<b>ent_mean, ent_max, ent_p95, ent_std</b> &mdash; the <i>entropy distribution</i> over "
         "changed pages",
         "<b>ent_pooled</b> &mdash; entropy of all changed bytes merged into one 256-bin histogram"),
    ]
    depth_table = ("<table><thead><tr><th>Axis (what it senses)</th>"
                   "<th>Per-page &mdash; distribution over changed pages</th>"
                   "<th>Per-snapshot &mdash; one pooled value</th></tr></thead><tbody>"
                   + "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in depth_rows)
                   + "</tbody></table>")
    qa = [
        ("Do we still compute APF, or only the new metrics?",
         "Both, every snapshot. APF stays as the breadth baseline in the same record, so 'what do "
         "magnitude / entropy add over APF?' is built in. Not one or the other."),
        ("What is ent_pooled, and why no mean / max on it?",
         "It pools every changed byte this snapshot into one histogram and takes entropy once &mdash; a "
         "single number. A blob has nothing to average within a snapshot. The entropy mean / max you "
         "want are ent_mean / ent_max (per-page). ent_pooled is the global twin of ham_sum."),
        ("Where is the 'magnitude distribution'?",
         "ham_mean, ham_max, ham_p95, ham_std &mdash; bits-flipped per changed page, summarised over "
         "pages. ham_sum / ham_frac are the global total beside it."),
        ("Per page or per snapshot?",
         "Both. Per-page = compute per changed page, then summarise over pages (a distribution). "
         "Per-snapshot = pool into one value. Plus, offline, every per-snapshot value is summarised "
         "over the cell's snapshots (a third, over-time view)."),
        ("Why per-page at all &mdash; why not just pool?",
         "Pooling dilutes localised encryption: one random page among many quiet ones barely moves the "
         "pooled value or APF, but spikes the per-page max. The per-page max is the stealth detector."),
        ("Is it captured yet?",
         "No. Task C (TIMING_MAGENT) is the pending capture-side emit; byte-identical when off. Today's "
         "captures carry APF only."),
    ]
    qa_table = ("<table><thead><tr><th>Question</th><th>Answer</th></tr></thead><tbody>"
                + "".join(f"<tr><td>{q}</td><td>{a}</td></tr>" for q, a in qa)
                + "</tbody></table>")
    return f"""
<section>
<h2><span class='sn'>Design note</span> &mdash; the depth axes: magnitude + entropy (Task C, pending)</h2>
<div class='meta'>spec: <code>plan07_campaign/magnitude_entropy_spec.md</code> &middot; status: designed,
not yet emitted in capture</div>

<h3>What this is</h3>
<p>APF measures only <b>breadth</b> &mdash; how many 4 KB pages changed between two snapshots. It is
blind to <i>how hard</i> each page changed and <i>what the new content looks like</i>, which is exactly
the gap a stealth encryptor hides in: few pages touched (low APF, looks quiet) but each one fully
rewritten. Plan 07 adds two <b>depth axes</b> beside APF &mdash; <b>magnitude</b> (weighted Hamming)
and <b>byte entropy</b> &mdash; each at two granularities. APF is kept, in the same per-snapshot
record, as the breadth baseline the depth axes are compared against. Not one or the other; both.</p>

{depth_table}
<div class='box'><b>How to read it.</b> Two senses (magnitude = how hard; entropy = how random) at two
granularities. <b>Per-page</b> computes the metric on each changed page and summarises the spread over
pages (a distribution: mean / max / p95 / std). <b>Per-snapshot</b> pools everything into one value. A
pooled value has no within-snapshot mean / max &mdash; those reappear <b>offline</b>, taken over the
cell's snapshots (a third, over-time granularity). The per-page <b>max</b> is the stealth detector: one
fully-encrypted page among many quiet ones spikes ham_max / ent_max while the pooled value and APF stay
low.</div>

<h3>The equations</h3>
<p>Magnitude, per changed page: <b>h = popcount(prev XOR curr)</b> (bits). Byte entropy, per changed
page, in the count form we compute (avoids per-bin division, reuses a precomputed table
T[k] = k&middot;log2 k): <b>H = log2(B) &minus; (1/B) &sum;<sub>b</sub> c_b&middot;log2(c_b)</b>, with
B = 4096 bytes and c_b the count of byte value b on the page. Pooled entropy uses the same formula on
the summed histogram C_b = &sum;<sub>pages</sub> c_b (total B&middot;K over K changed pages). One
page-diff feeds APF, the popcount, and the histogram together &mdash; the full single-pass algorithm is
in the spec.</p>

<h3>Questions and answers</h3>
{qa_table}

<h3>What it unlocks (blocks downstream)</h3>
<p>This axis is what RQ5 (does magnitude catch stealth?) and the stealth family
(<code>sandbox_stealth_*</code>) are built to test. It is gated by <b>Task C</b>: an additive,
flag-gated capture emit (<code>TIMING_MAGENT</code>) mirroring the Plan 06 disk block, byte-identical
when off. Until Task C ships, captures carry APF only, so the stealth / Hamming experiments wait on it;
everything else in the plan (the offline pilot, separability, cohesion) runs without it.</p>
</section>
"""


def build(pilot_path):
    css = """
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=Crimson+Pro:wght@400;600;700&display=swap');
:root{--ink:#1A1A1A;--muted:#5B5B5B;--line:#D8D6CE;--pv:#3F6CA8;--gold:#B9822A;}
*{box-sizing:border-box}
body{margin:0;background:#E8E6DF;color:var(--ink);font-family:'Atkinson Hyperlegible',system-ui,sans-serif;line-height:1.6;font-size:16px}
.page{max-width:880px;margin:26px auto;background:#fff;box-shadow:0 2px 18px rgba(0,0,0,.12);padding:48px 60px 70px;border-radius:4px}
h1{font-family:'Crimson Pro',Georgia,serif;font-size:30px;margin:0 0 4px}
.sub{color:var(--muted);margin:0 0 18px}
h2{font-family:'Crimson Pro',serif;font-size:22px;margin:38px 0 6px;border-bottom:2px solid var(--line);padding-bottom:6px}
h3{font-family:'Crimson Pro',serif;font-size:17px;margin:22px 0 6px;color:#333}
.sn{color:var(--gold);font-weight:700}
.meta{color:var(--muted);font-size:13px;margin:2px 0 8px}
p{max-width:72ch} ul{max-width:72ch} li{margin:6px 0}
figure{margin:18px 0;text-align:center} figure img{max-width:100%;border:1px solid var(--line);border-radius:6px}
figcaption{color:var(--muted);font-size:13px;margin-top:6px;text-align:left;max-width:72ch;margin-inline:auto;font-style:italic}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13.5px}
th,td{border:1px solid var(--line);padding:7px 10px;text-align:left}
th{background:#F5F3EE} .r{text-align:right} tr:nth-child(even) td{background:#FAF9F5}
.box{background:#EEF3F8;border-left:4px solid var(--pv);padding:12px 16px;border-radius:0 6px 6px 0;margin:16px 0;max-width:72ch}
.find{max-width:72ch;margin:10px 0}
code{font-family:ui-monospace,Menlo,monospace;font-size:.9em;background:#F0EEE8;padding:1px 4px;border-radius:3px}
.foot{margin-top:46px;border-top:1px solid var(--line);padding-top:12px;color:var(--muted);font-size:12.5px}
.term{position:relative;border-bottom:1px dotted var(--muted);cursor:help}
.term:hover::after,.term:focus::after{content:attr(data-def);position:absolute;left:0;top:142%;z-index:40;width:max-content;max-width:300px;background:#0d0f12;color:#fff;border:1px solid #333;border-radius:6px;padding:8px 11px;font-size:12.5px;font-weight:400;line-height:1.45;white-space:normal;box-shadow:0 8px 24px rgba(0,0,0,.55)}
details#glossary{background:#F5F3EE;border:1px solid var(--line);border-radius:6px;padding:6px 16px;margin:14px 0}
details#glossary summary{cursor:pointer;color:#333;font-weight:700;padding:6px 0;font-family:'Crimson Pro',serif}
details#glossary td.gt{white-space:nowrap;color:#333;font-weight:700}
@media print{body{background:#fff}.page{box-shadow:none;margin:0;max-width:none}.term{border:none}}
"""
    steps = step1(pilot_path) + metric_design()
    html = (f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>Plan 07 -- Execution Log</title><style>{css}</style></head><body><div class='page'>"
            f"<p class='sub'>Memory-Signal Behaviour Recogniser</p>"
            f"<h1>Plan 07 &mdash; Execution Log</h1>"
            f"<p class='sub'>A running record of each step actually run: what, why, results, conclusions, "
            f"and what it unlocks. Updated as the plan executes. Hover any underlined term for a one-line "
            f"definition, or open the glossary.</p>"
            f"{GLOSSARY_SECTION}"
            f"{steps}"
            f"<div class='foot'>Generated by <code>plan07_campaign/make_execution_log.py</code>. "
            f"Figures from the committed step outputs.</div>"
            f"</div>{SCRIPT_HTML}</body></html>")
    out = DOCS / "plan07_execution_log.html"
    out.write_text(html)
    print(f"log -> {out}")


if __name__ == "__main__":
    pilot = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "pilot_baseline.json")
    build(pilot)
