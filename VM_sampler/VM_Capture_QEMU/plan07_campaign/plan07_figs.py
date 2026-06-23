#!/usr/bin/env python3
"""Plan 07 handout figures -- one coherent matplotlib style, shared by the HTML
and the PDF so both documents look identical.

build_all() -> {name: (base64_png, abspath)}.  Run directly to just write PNGs.

Data figures use VERIFIED numbers from this project's committed results; diagram
figures are schematics of the experimental design and are labelled as such.
"""
import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

FIGDIR = (Path(__file__).resolve().parent.parent / "docs" / "plan07_figs")
FIGDIR.mkdir(parents=True, exist_ok=True)

# ---- palette (academic + semantic; family colours match the Plan 06 figures) ----
PAL = {
    "ink": "#1A1A1A", "muted": "#5B5B5B", "line": "#D8D6CE", "paper": "#FFFFFF",
    "panel": "#F5F3EE", "panel2": "#ECE9E1",
    "threat": "#C0392B", "benign": "#2E7D52",
    "old": "#3F6CA8", "new": "#B9822A", "gold": "#B9822A",
    "confirm": "#2E7D52", "refute": "#C0392B", "neutral": "#8A8A8A",
    "ransomware": "#C0392B", "scanner": "#E08A1E", "mem_sweep": "#3F6CA8",
    "mem_fault": "#2C9B8F", "app": "#2E7D52",
    "indigo": "#3B3B6D",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10.5,
    "axes.edgecolor": PAL["muted"], "axes.labelcolor": PAL["ink"],
    "text.color": PAL["ink"], "xtick.color": PAL["muted"], "ytick.color": PAL["muted"],
    "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})

FIGS = {}


def _emit(fig, name):
    fig.savefig(FIGDIR / f"{name}.png", format="png", dpi=200, bbox_inches="tight")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    FIGS[name] = (base64.b64encode(buf.getvalue()).decode(), str(FIGDIR / f"{name}.png"))


def _box(ax, cx, cy, w, h, text, fc, tc="white", ec="none", fs=9.5, weight="bold", lw=1.2):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.012,rounding_size=0.06", fc=fc, ec=ec, lw=lw))
    ax.text(cx, cy, text, ha="center", va="center", color=tc, fontsize=fs,
            fontweight=weight, linespacing=1.32, zorder=5)


def _arrow(ax, x1, y1, x2, y2, color=None, lw=1.6, style="-|>"):
    color = color or PAL["muted"]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=3, shrinkB=3))


def _canvas(w=10, h=6):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    return fig, ax


def _tag(ax, text):
    ax.text(0.99, 0.01, text, ha="right", va="bottom", fontsize=7.5, style="italic",
            color=PAL["muted"], transform=ax.transAxes)


# =====================================================================
# 1. Nested data model (tree) + CV mapping
# =====================================================================
def fig_tree():
    fig, ax = _canvas(10.5, 6.4)
    ax.text(50, 96, "The nested data model", ha="center", fontsize=14, fontweight="bold")
    _box(ax, 50, 87, 22, 8, "TESTS\n(the experiment)", PAL["ink"], fs=10)
    fams = [("ransomware", "ransomware\n4 workloads", PAL["ransomware"]),
            ("scanner", "scanner\n1 workload", PAL["scanner"]),
            ("mem_sweep", "mem_sweep\n3 workloads", PAL["mem_sweep"]),
            ("mem_fault", "mem_fault\n2 workloads", PAL["mem_fault"]),
            ("app", "app\n1 workload", PAL["app"])]
    xs = np.linspace(12, 88, 5)
    for x, (_, label, c) in zip(xs, fams):
        _arrow(ax, 50, 83, x, 73.5, PAL["muted"], 1.1)
        _box(ax, x, 69, 15.5, 9.5, label, c, fs=8.4)
    # workloads under ransomware (illustrative expansion)
    wls = ["seq", "batched", "slowburn", "selective"]
    wxs = np.linspace(4, 30, 4)
    for x, w in zip(wxs, wls):
        _arrow(ax, xs[0], 64.3, x, 55, PAL["ransomware"], 0.8)
        _box(ax, x, 51, 7.0, 6.5, w, PAL["panel2"], tc=PAL["ink"], fs=7.3)
    # durations x reps under one workload
    _arrow(ax, wxs[2], 47.7, wxs[2], 40, PAL["muted"], 0.8)
    _box(ax, wxs[2], 36, 26, 8.5,
         "durations {120,300,600 s}: probe RQ8 only\nrepetitions = SAME length, re-run (variance)\ncurrent dataset: 6 cells / workload",
         PAL["panel"], tc=PAL["ink"], fs=6.9)
    ax.text(58, 52, "...each family expands into its distinct workloads;\neach workload into durations x repetitions.",
            fontsize=8.6, color=PAL["muted"], va="center")
    # CV mapping panel
    ax.add_patch(FancyBboxPatch((40, 6), 56, 22, boxstyle="round,pad=0.2,rounding_size=0.4",
                 fc=PAL["panel"], ec=PAL["line"], lw=1))
    ax.text(68, 24.5, "Hold out a level  =  a generalization test", ha="center", fontsize=9.5, fontweight="bold")
    rows = [("a FAMILY", "LOFO", "novel-family / open-set (hardest)", PAL["ransomware"]),
            ("a WORKLOAD", "LOWO", "honest novel-workload", PAL["mem_sweep"]),
            ("a REPLICATE", "LORO", "optimistic; for variance, not generalization", PAL["benign"])]
    for i, (lvl, code, desc, c) in enumerate(rows):
        y = 19.5 - i * 4.6
        _box(ax, 49, y, 13, 3.4, code, c, fs=9)
        ax.text(57.5, y, f"hold out {lvl}  ->  {desc}", va="center", fontsize=8.2, color=PAL["ink"])
    ax.text(8, 20, "Rule the tree\nenforces:", fontsize=8.6, fontweight="bold", color=PAL["ink"], va="center")
    ax.text(8, 11.5, "nothing below a\nworkload counts as\nan independent\nworkload.", fontsize=8, color=PAL["muted"], va="center")
    _emit(fig, "tree")


# =====================================================================
# 2. Plan overview: waves, gate, parallel tracks
# =====================================================================
def fig_overview():
    fig, ax = _canvas(11, 6.2)
    ax.text(50, 96, "Plan overview: two tracks, three waves, one gate", ha="center", fontsize=13.5, fontweight="bold")
    # track labels
    ax.text(2, 78, "OFFLINE\n(laptop,\nexisting data)", fontsize=8.6, fontweight="bold", color=PAL["old"], va="center")
    ax.text(2, 34, "SERVER\n(new captures,\ngated)", fontsize=8.6, fontweight="bold", color=PAL["new"], va="center")
    # Wave 1 offline boxes
    w1 = [("1c separability\nmatrix", 20), ("1d kill-test\n(benign floor)", 38),
          ("1a fused\ndetector", 56), ("1b CIs +\nsignificance", 74)]
    for label, x in w1:
        _box(ax, x, 78, 15, 9, label, PAL["old"], fs=8.2)
    for i in range(len(w1) - 1):
        _arrow(ax, w1[i][1] + 7.5, 78, w1[i + 1][1] - 7.5, 78, PAL["old"])
    ax.text(47, 90, "WAVE 1  (settles the spine)", ha="center", fontsize=9, fontweight="bold", color=PAL["old"])
    # Wave 2 offline
    _box(ax, 89, 78, 16, 9, "WAVE 2\ndecimation -  2-ch\ncoverage -  disk", PAL["indigo"], fs=7.6)
    _arrow(ax, 81.5, 78, 81, 78, PAL["old"])
    # gate diamond
    gx, gy = 56, 56
    ax.add_patch(plt.Polygon([(gx, gy + 7), (gx + 11, gy), (gx, gy - 7), (gx - 11, gy)],
                 closed=True, fc=PAL["gold"], ec="none", zorder=4))
    ax.text(gx, gy, "GATE\npilot passes?", ha="center", va="center", color="white", fontsize=8.3, fontweight="bold", zorder=5)
    _arrow(ax, 56, 73.5, 56, 63, PAL["muted"])
    # Server track
    _box(ax, 56, 30, 19, 9, "capture more workloads\n(fill families to >=3)", PAL["new"], fs=8)
    _box(ax, 84, 30, 17, 9, "graded-Hamming\nre-capture", PAL["new"], fs=8)
    _arrow(ax, gx + 6, gy - 5, 56, 34.5, PAL["confirm"]); ax.text(60, 44, "PASS", fontsize=7.6, color=PAL["confirm"], fontweight="bold")
    _arrow(ax, 65.5, 30, 75.5, 30, PAL["new"])
    # fail branch
    _box(ax, 22, 30, 22, 9, "keep global detector +\nfamily characterization\n(complete, honest thesis)", PAL["benign"], fs=7.6)
    _arrow(ax, gx - 6, gy - 5, 30, 34.5, PAL["refute"]); ax.text(34, 44, "FAIL", fontsize=7.6, color=PAL["refute"], fontweight="bold")
    # parallel-capture note
    _box(ax, 22, 56, 26, 8, "(parallel) capture app x5 existing +\nauthor scanner x2 -- both machines work", PAL["panel2"], tc=PAL["ink"], fs=7.4)
    _arrow(ax, 22, 52, 22, 34.5, PAL["new"], 1.1, style="-|>")
    _tag(ax, "schematic")
    _emit(fig, "overview")


# =====================================================================
# 3. Separability ladder
# =====================================================================
def fig_ladder():
    fig, ax = _canvas(10.5, 5.4)
    ax.text(50, 95, "The separability ladder: from 'are they different' to 'does it generalise'", ha="center", fontsize=12.5, fontweight="bold")
    rungs = [
        ("1 : 1", "PAIRS", "Is behaviour A separable from B?\nThe 11x11 separability matrix.", "LORO-level", PAL["benign"], 16),
        ("N : 1", "WITHIN-FAMILY", "Train on N family members,\nrecognise a held-out member.\n(e.g. 3:1 ransomware)", "LOWO", PAL["mem_sweep"], 47),
        ("N : 1", "FAMILY-LEVEL", "Train on N families,\nflag a held-out whole family\nas novel.", "LOFO / open-set", PAL["ransomware"], 78),
    ]
    for cross, name, desc, cv, c, x in rungs:
        _box(ax, x, 60, 24, 13, f"{cross}", c, fs=18)
        ax.text(x, 49, name, ha="center", fontsize=10, fontweight="bold", color=PAL["ink"])
        ax.text(x, 39, desc, ha="center", fontsize=8.2, color=PAL["muted"], va="top")
        _box(ax, x, 20, 20, 5.5, cv, PAL["panel2"], tc=PAL["ink"], fs=8.5)
    _arrow(ax, 29, 60, 35, 60, PAL["muted"], 2)
    _arrow(ax, 60, 60, 66, 60, PAL["muted"], 2)
    ax.text(50, 8, "harder, more honest  ->", ha="center", fontsize=9, color=PAL["muted"], style="italic")
    _tag(ax, "schematic")
    _emit(fig, "ladder")


# =====================================================================
# 4. Feature taxonomy (old vs new)
# =====================================================================
def fig_features():
    fig, ax = _canvas(11, 6.0)
    ax.text(50, 96, "Feature taxonomy: what we measure (old) and what we add (new)", ha="center", fontsize=12.5, fontweight="bold")
    # OLD group
    ax.add_patch(FancyBboxPatch((3, 40), 60, 50, boxstyle="round,pad=0.3,rounding_size=0.5", fc=PAL["panel"], ec=PAL["old"], lw=1.3))
    ax.text(33, 86, "MEMORY -- existing", ha="center", fontsize=10, fontweight="bold", color=PAL["old"])
    _box(ax, 18, 76, 26, 7, "AGNOSTIC (7, leakage-free)\napf_mean, apf_std, cepstral,\nstat_pass_frac, n_pairs, n_windows", PAL["old"], fs=6.8)
    ax.text(48, 78.5, "magnitude vs shape:", fontsize=8, fontweight="bold", color=PAL["ink"])
    _box(ax, 18, 62, 26, 8, "peakvar -- MAGNITUDE\napf_max, apf_p95\n(level: 'how loud')", PAL["threat"], fs=7.2)
    _box(ax, 48, 62, 26, 8, "peakvar -- SCALE-INVARIANT\napf_cov, peak2med, duty_gt05\n(shape, unitless)", PAL["benign"], fs=7.2)
    _box(ax, 33, 49, 40, 6.5, "disk channel (Plan 06): wr_rate, rd_rate, rd_frac", PAL["muted"], fs=7.6)
    # NEW group
    ax.add_patch(FancyBboxPatch((66, 40), 31, 50, boxstyle="round,pad=0.3,rounding_size=0.5", fc="#FBF6EC", ec=PAL["new"], lw=1.6))
    ax.text(81.5, 86, "NEW -- per-page Hamming", ha="center", fontsize=10, fontweight="bold", color=PAL["new"])
    _box(ax, 81.5, 75, 26, 6.5, "APF = changed-page fraction\n(BASELINE -- kept for comparison)", PAL["muted"], fs=7.2)
    _box(ax, 81.5, 66, 26, 6, "Hamming SUM (total bits flipped)", PAL["new"], fs=7.6)
    _box(ax, 81.5, 57.5, 26, 7, "MEAN Hamming / changed page\n<- the encryption tell", PAL["new"], fs=7.4)
    _box(ax, 81.5, 49, 26, 6, "p95 / max Hamming per page", PAL["new"], fs=7.6)
    # why-it-matters strip
    ax.add_patch(FancyBboxPatch((3, 6), 94, 27, boxstyle="round,pad=0.3,rounding_size=0.4", fc=PAL["panel2"], ec=PAL["line"]))
    ax.text(50, 28, "Why the new metric: a stealth encryptor changes FEW pages (low APF -> looks quiet)",
            ha="center", fontsize=9, color=PAL["ink"], fontweight="bold")
    ax.text(50, 21.5, "but each touched page is fully encrypted -> ~half its bits flip. APF cannot see this; mean-Hamming-per-changed-page can.",
            ha="center", fontsize=8.6, color=PAL["muted"])
    ax.text(50, 13, "Computed additively at capture (XOR + popcount in the same page comparison APF already does) -> a few cheap scalars, no per-page storage.",
            ha="center", fontsize=8.2, color=PAL["muted"], style="italic")
    _emit(fig, "features")


# =====================================================================
# 5. Two modelling paths
# =====================================================================
def fig_two_paths():
    fig, ax = _canvas(10.5, 5.6)
    ax.text(50, 95, "Two parallel modelling paths", ha="center", fontsize=13, fontweight="bold")
    # path A
    ax.add_patch(FancyBboxPatch((4, 12), 43, 74, boxstyle="round,pad=0.3,rounding_size=0.5", fc=PAL["panel"], ec=PAL["mem_sweep"], lw=1.3))
    ax.text(25.5, 81, "A -- PER-WORKLOAD", ha="center", fontsize=10.5, fontweight="bold", color=PAL["mem_sweep"])
    ax.text(25.5, 75, "each workload = its own class", ha="center", fontsize=8.4, color=PAL["muted"])
    for i, w in enumerate(["seq", "slowburn", "mmap", "hashtable"]):
        _box(ax, 16 + (i % 2) * 19, 64 - (i // 2) * 11, 16, 8, w, PAL["mem_sweep"], fs=8)
    ax.text(25.5, 40, "Question: which exact\nprogram is this?\n(instance identification)", ha="center", fontsize=8.6, color=PAL["ink"])
    ax.text(25.5, 22, "the separability matrix\nlives here", ha="center", fontsize=8, style="italic", color=PAL["muted"])
    # path B
    ax.add_patch(FancyBboxPatch((53, 12), 43, 74, boxstyle="round,pad=0.3,rounding_size=0.5", fc="#FBF6EC", ec=PAL["new"], lw=1.5))
    ax.text(74.5, 81, "B -- PER-FAMILY", ha="center", fontsize=10.5, fontweight="bold", color=PAL["new"])
    ax.text(74.5, 75, "pool a family, learn its NORMAL routine", ha="center", fontsize=8.2, color=PAL["muted"])
    _box(ax, 74.5, 64, 30, 9, "ransomware-normal\n(seq + batched + slowburn + selective)", PAL["ransomware"], fs=7.4)
    ax.text(74.5, 49, "Two birds:", ha="center", fontsize=9, fontweight="bold", color=PAL["ink"])
    ax.text(74.5, 41, "(1) characterises the family\n(2) yields the open-set\nnormality block", ha="center", fontsize=8.4, color=PAL["ink"])
    ax.text(74.5, 24, "Cohesion is NOT assumed --\nit is a research question we MEASURE\n(both answers are findings)", ha="center", fontsize=7.8, style="italic", color=PAL["new"])
    _tag(ax, "schematic")
    _emit(fig, "two_paths")


# =====================================================================
# 6. Two detector flavours
# =====================================================================
def fig_flavors():
    fig, ax = _canvas(10.5, 5.0)
    ax.text(50, 95, "Two detector flavours (open-set / novelty)", ha="center", fontsize=12.5, fontweight="bold")
    ax.add_patch(FancyBboxPatch((4, 14), 43, 70, boxstyle="round,pad=0.3,rounding_size=0.5", fc=PAL["panel"], ec=PAL["benign"], lw=1.3))
    ax.text(25.5, 79, "POPULATION-NORMAL", ha="center", fontsize=10, fontweight="bold", color=PAL["benign"])
    ax.text(25.5, 71, "learn 'normal' from MANY benign\ntraces; flag what doesn't fit", ha="center", fontsize=8.3, color=PAL["muted"])
    ax.text(25.5, 52, "+ catches a uniformly quiet\nthreat (slowburn): it differs\nfrom the benign population", ha="center", fontsize=8.4, color=PAL["confirm"])
    ax.text(25.5, 30, "the existing 0.925 detector\nis this flavour", ha="center", fontsize=8, style="italic", color=PAL["muted"])
    ax.add_patch(FancyBboxPatch((53, 14), 43, 70, boxstyle="round,pad=0.3,rounding_size=0.5", fc=PAL["panel"], ec=PAL["scanner"], lw=1.3))
    ax.text(74.5, 79, "WITHIN-TRACE ('normal window')", ha="center", fontsize=9.4, fontweight="bold", color=PAL["scanner"])
    ax.text(74.5, 71, "learn a trace's OWN recent windows;\nflag when the next one shifts", ha="center", fontsize=8.3, color=PAL["muted"])
    ax.text(74.5, 52, "+ truly online / streaming", ha="center", fontsize=8.4, color=PAL["confirm"])
    ax.text(74.5, 38, "- blind to a uniformly quiet\nthreat (no internal shift\nto trip on)", ha="center", fontsize=8.4, color=PAL["refute"])
    _tag(ax, "schematic")
    _emit(fig, "flavors")


# =====================================================================
# 7. Windows vs duration (DATA -- verified ranges)
# =====================================================================
def fig_windows():
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    durs = [120, 300, 600]
    mn = [5, 11, 15]; mx = [48, 122, 244]
    x = np.arange(3)
    for i in range(3):
        ax.vlines(x[i], mn[i], mx[i], color=PAL["old"], lw=10, alpha=0.35)
        ax.plot(x[i], mn[i], "o", color=PAL["old"], ms=7)
        ax.plot(x[i], mx[i], "o", color=PAL["old"], ms=7)
        ax.text(x[i] + 0.12, mn[i], f"min {mn[i]}", va="center", fontsize=8.5, color=PAL["muted"])
        ax.text(x[i] + 0.12, mx[i], f"max {mx[i]}", va="center", fontsize=8.5, color=PAL["muted"])
    ax.axhspan(0, 3, color=PAL["refute"], alpha=0.08)
    ax.text(2.35, 2.2, "DOF-starved floor", fontsize=7.6, color=PAL["refute"], ha="right")
    ax.axvline(1, color=PAL["new"], ls="--", lw=1.4)
    ax.text(1.04, 200, "recommended:\n300 s", color=PAL["new"], fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"{d} s" for d in durs])
    ax.set_ylabel("analysis windows per cell  (W=8, H=4)")
    ax.set_title("Windows per cell vs capture duration  (range across 22 cells each)")
    ax.set_ylim(0, 260); ax.grid(axis="y", ls=":", alpha=0.4)
    _emit(fig, "windows")


# =====================================================================
# 8. The motivating result: the hidden detector (DATA -- verified)
# =====================================================================
def fig_results():
    fig, ax = plt.subplots(figsize=(9, 4.2))
    labels = ["Supervised RF\nbinary, novel workload (LOWO)",
              "One-class novelty\nbinary, novel workload (LOWO)",
              "One-class novelty\nnovel benign FAMILY (LOFO-benign)*"]
    vals = [0.79, 0.925, 0.933]
    cols = [PAL["old"], PAL["new"], PAL["new"]]
    y = np.arange(3)[::-1]
    ax.barh(y, vals, color=cols, height=0.55)
    for yi, v in zip(y, vals):
        ax.text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=10, fontweight="bold")
    ax.axvline(0.545, color=PAL["neutral"], ls="--", lw=1.3)
    ax.text(0.545, 2.62, "majority prior 0.545", fontsize=8, color=PAL["neutral"], ha="center")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 1.02); ax.set_xlabel("ROC-AUC (one-class) / accuracy (supervised)")
    ax.set_title("The hidden detector: reframing supervised -> one-class novelty")
    ax.grid(axis="x", ls=":", alpha=0.4)
    ax.text(0.01, -0.66, "* LOF LOFO-benign 0.933 flagged INVALID (k=10 neighbours > 6-cell app family); re-run with k capped.",
            transform=ax.get_yaxis_transform() if False else ax.transAxes, fontsize=7, style="italic", color=PAL["muted"])
    _emit(fig, "results")


# =====================================================================
# 9. The masquerade pair (DATA -- verified)
# =====================================================================
def fig_masquerade():
    fig, ax = plt.subplots(figsize=(9, 3.1))
    ax.scatter([0.00203], [0], s=160, color=PAL["threat"], zorder=5, label="slowburn (threat)")
    ax.scatter([0.00509], [0], s=160, color=PAL["benign"], zorder=5, label="pagefault (benign)")
    ax.annotate("slowburn\n0.00203", (0.00203, 0), xytext=(0.00203, 0.6), ha="center", fontsize=9, color=PAL["threat"], fontweight="bold")
    ax.annotate("pagefault\n0.00509", (0.00509, 0), xytext=(0.00509, 0.6), ha="center", fontsize=9, color=PAL["benign"], fontweight="bold")
    ax.annotate("", xy=(0.00509, -0.45), xytext=(0.00203, -0.45), arrowprops=dict(arrowstyle="<->", color=PAL["muted"]))
    ax.text(0.00356, -0.85, "non-overlapping in raw apf_mean  (Cohen's d ~ 5.6)", ha="center", fontsize=9, color=PAL["ink"])
    ax.text(0.00356, 1.5, "Separable in the signal -- yet BOTH score 0.0 under supervised LOWO\n(a labelling-coverage artifact, not memory aliasing)",
            ha="center", fontsize=9.2, fontweight="bold", color=PAL["refute"])
    ax.set_xlim(0.0012, 0.0060); ax.set_ylim(-1.3, 2.3)
    ax.set_yticks([]); ax.set_xlabel("apf_mean (memory churn level)")
    ax.set_title("The 'memory twins' are not twins")
    ax.grid(axis="x", ls=":", alpha=0.4)
    _emit(fig, "masquerade")


# =====================================================================
# 10. Separability-matrix schematic (blocked by family)
# =====================================================================
def fig_sepmatrix():
    order = [("seq", "ransomware"), ("batched", "ransomware"), ("slowburn", "ransomware"),
             ("selective", "ransomware"), ("scanner", "scanner"), ("rmw", "mem_sweep"),
             ("workingset", "mem_sweep"), ("writemag", "mem_sweep"), ("mmap", "mem_fault"),
             ("pagefault", "mem_fault"), ("hashtable", "app")]
    n = len(order)
    rng = np.random.default_rng(7)
    M = np.full((n, n), 0.0)
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i, j] = 0
            elif order[i][1] == order[j][1]:
                M[i, j] = 0.18 + rng.uniform(0, 0.12)      # within-family: similar
            else:
                M[i, j] = 0.62 + rng.uniform(0, 0.3)       # between-family: distinct
    # masquerade: slowburn(2) vs pagefault(9) artificially close
    M[2, 9] = M[9, 2] = 0.12
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    im = ax.imshow(M, cmap="magma_r", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    names = [o[0] for o in order]
    ax.set_xticklabels(names, rotation=55, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    # family separators
    bounds = [4, 5, 8, 10]
    for b in bounds:
        ax.axhline(b - 0.5, color="white", lw=2); ax.axvline(b - 0.5, color="white", lw=2)
    # highlight masquerade cell
    for (i, j) in [(2, 9), (9, 2)]:
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=PAL["confirm"], lw=2.5))
    ax.set_title("Separability matrix, blocked by family  (SCHEMATIC)")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("distance  (dark = similar/aliased)", fontsize=8)
    ax.text(9, 2, "", )
    fig.text(0.5, 0.005, "diagonal blocks = within-family cohesion   |   off-diagonal = between-family separation   |   green cell = a masquerade (cross-family alias)",
             ha="center", fontsize=7.6, color=PAL["muted"])
    _emit(fig, "sepmatrix")


def build_all():
    fig_tree(); fig_overview(); fig_ladder(); fig_features(); fig_two_paths()
    fig_flavors(); fig_windows(); fig_results(); fig_masquerade(); fig_sepmatrix()
    return FIGS


if __name__ == "__main__":
    build_all()
    print(f"wrote {len(FIGS)} figures -> {FIGDIR}")
    for k in FIGS:
        print("  ", k)
