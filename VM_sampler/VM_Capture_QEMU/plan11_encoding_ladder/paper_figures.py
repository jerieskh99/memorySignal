#!/usr/bin/env python3
"""paper_figures.py -- move 19 of plan11_encoding_ladder (optional; added 2026-10-13): the encoding paper's Figures 1
and 2, drawn by the toolkit exactly as the two approved preview scripts draw them (apf_paper/previews, 2026-10-12 and
2026-10-13), so that the paper's figures follow the run. Drawn only by this move: neither figure joins figures.py's
default set (move 12's "figures (all)" is unchanged).

  python3 -m plan11_encoding_ladder.paper_figures run --out <run> [--only representation,shape] [--minute-pairs 93] [--seed 42]

FIGURE 1, THE REPRESENTATION (fig_representation.pdf and .png; copied from apf_paper/previews/representation/
make_representation_preview_v2.py). Three panels, r1, r2, r3 on the horizontal axis against J_t on the vertical; one
mark per run, the median over its pairs with whiskers to the 0.25 and 0.75 quantiles; kernels coloured, idle grey;
the vertical axis spaced by log(1 - J) so the band near 1 opens up (the ticks still read J); every kernel named on
the plot at the median of its runs by a greedy placement that overlaps no earlier label and no data mark (set A
bold); no legend. Per pair r1 = r_l0_q50_per, r2 = r_l1l0_q50_per, r3 = r_haml0_q50_per, J = J, the extract's rows
[cut : n_rows - 1] as series.py takes them. The plotted values are written beside the figure
(fig_representation_medians.csv: per run the three quantiles of J and of each ratio, rounded to 4 places).

FIGURE 2, THE SHAPE OVER TIME (fig_shape_per_kernel.pdf and .png; copied from apf_paper/previews/shape_figure/
make_shape_figure_grid7.py). Two rows of seven at the IEEE text width (7.16 in), in the order of the draft's readings
table: row 1 the 2,048-page kernels then the 4,096-page kernels (and the key), row 2 the other kernels then idle;
per panel APF and wAPF in percent of memory over the 93 pairs right after the cut (pairs cut + 1 to cut + 93; extract
row i is pair i + 1) with J_t in its own strip under them; the x axis shared and labelled once; each panel prints only
the top of its own count scale, and the kernels of one footprint share one count scale (group brackets above). The run
drawn is the seed-42 run of each kernel (cells.csv seed 42, rep 0) and idle run 1 (rep 0): rep_dir is 1 for every run,
so it cannot select a run. The plotted values are written beside the figure (fig_shape_per_kernel_values.csv: per
kernel and pair APF, wAPF, J and the panel's count scale, rounded to 6 places).

THE TWO CHANGES FROM THE PREVIEWS, both so the figures follow the run: each cell's cut comes from the run's
inputs/head_drop.csv (series.load_head_drop and head_drop_for, the idle row for the idle cells) instead of a fixed
number; and the cells are read with the toolkit's own loaders and admissibility (series.load_cells and
series.admissible_cells: status ok and all_hard_pass; the run's excluded cells stay out, as the previews exclude them;
the kernels are the readings table's twelve and idle). Every setting is kept: 7.16 in wide, pdf.fonttype 42 (TrueType
in the PDF, no Type 3 fonts), STIXGeneral, the readings table's kernel names, the colours, the sizes.

OUTPUTS, under report/figures/: fig_representation.pdf, .png, fig_representation_medians.csv; fig_shape_per_kernel.pdf,
.png, fig_shape_per_kernel_values.csv; paper_figures.json (the params: the cuts, the cells drawn, the label-placement
notes, the inputs' sha256). matplotlib must be importable (the paper's figures are never replaced by a placeholder:
a missing matplotlib stops the move with exit 2).
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import argparse  # noqa: E402
import csv  # noqa: E402
import os  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

from plan11_encoding_ladder import schema  # noqa: E402
from plan11_encoding_ladder import series as S  # noqa: E402
from plan11_encoding_ladder._report_common import PACKAGE_VERSION, inputs_sha256, now_iso, result_json, write_json  # noqa: E402

ADDED = "added 2026-10-13"
COPIED = ("Figure 1 copied from apf_paper/previews/representation/make_representation_preview_v2.py (2026-10-12); Figure 2 copied from "
          "apf_paper/previews/shape_figure/make_shape_figure_grid7.py (2026-10-13); the cut from inputs/head_drop.csv and the cells from the "
          "toolkit's loaders and admissibility are the only changes")
CITATION = ("apf_paper/p2e_skeleton.tex (fig:p2e_repr, Figure 1; Figure 2, figures/fig_shape_per_kernel.pdf); " + COPIED + "; series.py load_cells, "
            "admissible_cells, load_extract, load_head_drop, head_drop_for (SPEC 3.1.2, 3.3.1); schema N_PAGES, BITS_PER_PAGE")
FIGURES = ("representation", "shape")
FILE_NAMES = {"representation": "fig_representation", "shape": "fig_shape_per_kernel"}
WIDTH_IN = 7.16                                            # the IEEE two-column text width, drawn at print size
# the readings table's kernels, in its order, and its short names (Figure 1's order is set A first, then the rest of the table)
KERNELS = ["floyd", "histogram", "nbody", "stencil_jacobi", "fft", "gemm", "gibbs", "spmm", "fem_assembly", "lexer", "rmat_gen", "bnb_tsp"]
SET_A = ["floyd", "histogram", "nbody"]
SHOW = {"stencil_jacobi": "stencil", "fem_assembly": "fem", "rmat_gen": "rmat", "bnb_tsp": "bnb"}
COL = {"floyd": "#1f5fbf", "histogram": "#d1452b", "nbody": "#2e9e44", "stencil_jacobi": "#8a4fbf", "fft": "#e08a00",
       "gemm": "#0f8f8f", "gibbs": "#b8336a", "spmm": "#6b7a1f", "fem_assembly": "#7a4a2a", "lexer": "#c9a400",
       "rmat_gen": "#3b3b8f", "bnb_tsp": "#e0607e", "idle": "#9a9a9a"}
CH = [("r_l0_q50_per", r"$r_1$, share of a page's bytes that changed"),
      ("r_l1l0_q50_per", r"$r_2$, change per changed byte"),
      ("r_haml0_q50_per", r"$r_3$, bits flipped per changed byte")]
JT = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
# Figure 2's layout
MINUTE, N, BITS = 93, schema.N_PAGES, schema.BITS_PER_PAGE
SEED = 42
ROW1 = ["floyd", "histogram", "nbody", "fft", "stencil_jacobi", "gemm"]
ROW2 = ["gibbs", "spmm", "fem_assembly", "lexer", "rmat_gen", "bnb_tsp", "idle"]
GROUPS = [(0, 0, 2, "footprint 2,048 pages"), (0, 3, 5, "footprint 4,096 pages"), (1, 0, 5, "the other kernels"), (1, 6, 6, "floor")]
SHARED = [["floyd", "histogram", "nbody"], ["fft", "stencil_jacobi", "gemm"]]   # kernels of one footprint share one count scale
C_APF, C_WAPF, C_J = "#1f4e8c", "#9a9a9a", "#d9822b"
EXIT_OK, EXIT_ERROR, EXIT_MISSING = 0, 1, 2


class Stop(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def mpl_setup():
    """matplotlib with the Agg backend and the previews' rc settings; a missing matplotlib stops the move (exit 2)."""
    try:
        import matplotlib
    except Exception as e:                                 # noqa: BLE001
        raise Stop(EXIT_MISSING, f"missing module: matplotlib is not importable ({type(e).__name__}: {e}); the paper's figures are never replaced by a placeholder")
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42               # TrueType in the PDF, not Type 3 (IEEE PDF checks reject Type 3)
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "STIXGeneral"     # a Times-like face, as in the IEEE template
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    import matplotlib.pyplot as plt
    return matplotlib, plt


# ---------------------------------------------------------------------------------------------
# the data: the admissible cells of the readings table's kernels and idle, the cut per cell, the extracts
# ---------------------------------------------------------------------------------------------
def paper_cells(out: Path) -> tuple[list[dict], list[str], dict]:
    """The cells the figures draw: status ok and all_hard_pass (series.admissible_cells, no pair-rung rule), the
    readings table's twelve kernels and idle; each with its figure kernel ("idle" for the idle cells) and its cut."""
    cells = S.load_cells(out / "cells.csv")
    kept, ex_hard, _, pre_present = S.admissible_cells(out, cells, None)
    hd = S.load_head_drop(out / "inputs" / "head_drop.csv")
    res = []
    for c in kept:
        k = "idle" if c.get("role") == "idle" else c["kernel"]
        if k not in KERNELS and k != "idle":
            continue
        res.append({**c, "fig_kernel": k, "cut": S.head_drop_for(hd, c["kernel"], c.get("role"))})
    return res, sorted(ex_hard), {"head_drop": hd, "preconditions_present": pre_present}


def quantiles(x: np.ndarray) -> np.ndarray:
    x = x[~np.isnan(x)]
    return np.quantile(x, [0.25, 0.5, 0.75]) if x.size else np.array([np.nan] * 3)


# ---------------------------------------------------------------------------------------------
# Figure 1, the representation (copied from make_representation_preview_v2.py)
# ---------------------------------------------------------------------------------------------
def fig_representation(out: Path, fig_dir: Path, cells: list[dict], plt) -> dict:
    from matplotlib.scale import FuncScale
    from matplotlib.transforms import Bbox
    import matplotlib.patheffects as pe
    runs = []
    for c in cells:
        ex = S.load_extract(out, c["cell_id"])
        n = int(ex["_n_rows"])
        lo, hi = int(c["cut"]), n - 1                      # rows [cut : n_rows - 1], as series.py takes them
        d = {k: ex[k][lo:hi] for k in ["J"] + [ch for ch, _ in CH]}
        runs.append({"kernel": c["fig_kernel"], "cell": c["cell_id"], "cut": lo, "n_rows": n, "J": quantiles(d["J"]), **{k: quantiles(d[k]) for k, _ in CH}})
    counts = {k: sum(r["kernel"] == k for r in runs) for k in KERNELS + ["idle"]}
    print(f"[paperfig] representation: {len(runs)} runs " + ", ".join(f"{k} {v}" for k, v in counts.items()), flush=True)
    notes = []
    # v2 (2026-10-12): the vertical axis is J, spaced by log(1 - J) so the crowded band near 1 opens up (ticks still read J);
    # every kernel named on the plot at the median of its runs, so no legend is needed; whiskers lighter.
    fwd = lambda J: -np.log10(np.clip(1 - np.asarray(J, dtype=float), 1e-4, None))        # noqa: E731
    inv = lambda y: 1 - 10 ** (-np.asarray(y, dtype=float))                                # noqa: E731
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_IN, 2.75), sharey=True)
    order = ["idle"] + KERNELS
    for ax, (k, lab) in zip(axes, CH):
        ax.set_yscale(FuncScale(ax, (fwd, inv)))
        for kern in order:
            for r in (x for x in runs if x["kernel"] == kern):
                xq, yq = r[k], r["J"]
                col = COL[kern]
                ax.plot([xq[0], xq[2]], [yq[1], yq[1]], color=col, lw=0.5, alpha=0.45, zorder=2, solid_capstyle="butt")
                ax.plot([xq[1], xq[1]], [yq[0], yq[2]], color=col, lw=0.5, alpha=0.45, zorder=2, solid_capstyle="butt")
                ax.scatter([xq[1]], [yq[1]], s=6, color=col, edgecolors="white", linewidths=0.25, zorder=3)
        logx = k != "r_haml0_q50_per"
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(lab, fontsize=8)
        ax.tick_params(labelsize=7, length=2.5, pad=2)
        ax.set_yticks(JT); ax.set_yticklabels([f"{v:g}" for v in JT])
        ax.set_ylim(0.3, 0.995)
        ax.grid(True, which="major", color="0.92", lw=0.6)
        ax.minorticks_off()
        # name every kernel at the median of its runs; place each label where it overlaps no earlier label (greedy, in screen space)
        fig.canvas.draw()
        placed = []
        obst = []                                          # every run's mark and whiskers, so a label never sits on data
        for r in runs:
            xq, yq = r[k], r["J"]
            cx, cy = ax.transData.transform((xq[1], yq[1]))
            obst.append(Bbox([[cx - 2.5, cy - 2.5], [cx + 2.5, cy + 2.5]]))
            a0, a1 = ax.transData.transform((xq[0], yq[1])), ax.transData.transform((xq[2], yq[1]))
            obst.append(Bbox([[min(a0[0], a1[0]), cy - 1.5], [max(a0[0], a1[0]), cy + 1.5]]))
            b0, b1 = ax.transData.transform((xq[1], yq[0])), ax.transData.transform((xq[1], yq[2]))
            obst.append(Bbox([[cx - 1.5, min(b0[1], b1[1])], [cx + 1.5, max(b0[1], b1[1])]]))
        cen = {kern: (float(np.median([r[k][1] for r in runs if r["kernel"] == kern])), float(np.median([r["J"][1] for r in runs if r["kernel"] == kern])))
               for kern in order if any(r["kernel"] == kern for r in runs)}
        trans = ax.transData
        for kern in sorted(cen, key=lambda z: (z not in SET_A, -cen[z][1])):
            x, y = cen[kern]
            px, py = trans.transform((x, y))
            name = SHOW.get(kern, kern)
            best = None
            for dist in (5, 9, 13, 18, 24, 32):
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)):
                    ox, oy = dx * dist, dy * dist * 0.8
                    ha = "left" if dx > 0 else "right" if dx < 0 else "center"
                    va = "bottom" if dy > 0 else "top" if dy < 0 else "center"
                    t = ax.annotate(name, xy=(x, y), xytext=(ox, oy), textcoords="offset points", ha=ha, va=va, fontsize=7,
                                    fontweight="bold" if kern in SET_A else "normal", color=COL[kern] if kern != "idle" else "0.35",
                                    path_effects=[pe.withStroke(linewidth=1.6, foreground="white")])
                    bb = t.get_window_extent(renderer=fig.canvas.get_renderer()).expanded(1.04, 1.1)
                    inside = ax.get_window_extent().contains(bb.x0, bb.y0) and ax.get_window_extent().contains(bb.x1, bb.y1)
                    if inside and not any(bb.overlaps(o) for o in placed) and not any(bb.overlaps(o) for o in obst):
                        best = (t, bb); break
                    t.remove()
                if best: break
            if not best:                                   # second pass: free of labels only (the white halo keeps it legible)
                for dist in (5, 9, 13, 18, 24, 32):
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)):
                        ox, oy = dx * dist, dy * dist * 0.8
                        ha = "left" if dx > 0 else "right" if dx < 0 else "center"
                        va = "bottom" if dy > 0 else "top" if dy < 0 else "center"
                        t = ax.annotate(name, xy=(x, y), xytext=(ox, oy), textcoords="offset points", ha=ha, va=va, fontsize=7,
                                        fontweight="bold" if kern in SET_A else "normal", color=COL[kern] if kern != "idle" else "0.35",
                                        path_effects=[pe.withStroke(linewidth=1.6, foreground="white")])
                        bb = t.get_window_extent(renderer=fig.canvas.get_renderer()).expanded(1.04, 1.1)
                        inside = ax.get_window_extent().contains(bb.x0, bb.y0) and ax.get_window_extent().contains(bb.x1, bb.y1)
                        if inside and not any(bb.overlaps(o) for o in placed):
                            best = (t, bb); notes.append(f"label over data: {kern} in {k}"); break
                        t.remove()
                    if best: break
            if best:
                placed.append(best[1])
                if abs(best[0].xyann[0]) > 8 or abs(best[0].xyann[1]) > 8:
                    best[0].arrow_patch = None
                    ax.annotate("", xy=(x, y), xytext=best[0].xyann, textcoords="offset points",
                                arrowprops=dict(arrowstyle="-", color="0.55", lw=0.5, shrinkA=0, shrinkB=3))
            else:
                notes.append(f"no free spot for {kern} in {k}")
    axes[0].set_ylabel(r"$J_t$, overlap with the next pair", fontsize=8)
    fig.tight_layout(pad=0.3, w_pad=0.6)
    base = fig_dir / FILE_NAMES["representation"]
    fig.savefig(str(base) + ".png", dpi=300)
    fig.savefig(str(base) + ".pdf")
    plt.close(fig)
    with open(str(base) + "_medians.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["cell_id", "kernel", "J_q25", "J_med", "J_q75"] + [f"{k}_{s}" for k, _ in CH for s in ("q25", "med", "q75")])
        for r in runs: w.writerow([r["cell"], r["kernel"], *np.round(r["J"], 4)] + [v for k, _ in CH for v in np.round(r[k], 4)])
    for n in notes:
        print(f"[paperfig] representation: {n}", flush=True)
    return {"pdf": str(base) + ".pdf", "png": str(base) + ".png", "csv": str(base) + "_medians.csv", "n_runs": len(runs), "runs_per_kernel": counts,
            "cuts": sorted({int(r["cut"]) for r in runs}), "label_notes": notes,
            "medians_by_kernel": {kern: {"J": round(float(np.median([r["J"][1] for r in runs if r["kernel"] == kern])), 4),
                                        **{k.split("_")[1]: round(float(np.median([r[k][1] for r in runs if r["kernel"] == kern])), 4) for k, _ in CH}}
                                  for kern in order if any(r["kernel"] == kern for r in runs)}}


# ---------------------------------------------------------------------------------------------
# Figure 2, the shape over time (copied from make_shape_figure_grid7.py)
# ---------------------------------------------------------------------------------------------
def shape_cells(cells: list[dict], seed: int) -> dict:
    """{figure kernel: cell}: the seed-`seed` run of each kernel (rep 0 among them when several carry the seed) and idle run 1 (rep 0)."""
    chosen = {}
    for c in cells:
        k = c["fig_kernel"]
        if k not in ROW1 + ROW2:
            continue
        hit = (c.get("rep") == 0) if k == "idle" else (c.get("seed") == seed)
        if hit and (k not in chosen or int(c.get("rep") or 0) < int(chosen[k].get("rep") or 0)):
            chosen[k] = c
    return chosen


def fig_shape(out: Path, fig_dir: Path, cells: list[dict], plt, *, minute: int = MINUTE, seed: int = SEED) -> dict:
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    chosen = shape_cells(cells, seed)
    missing = [k for k in ROW1 + ROW2 if k not in chosen]
    if missing:
        raise Stop(EXIT_MISSING, f"missing input: no admissible seed-{seed} run (idle: rep 0) for {', '.join(missing)}; Figure 2 needs all 13 panels")

    def series(c):
        ex = S.load_extract(out, c["cell_id"])
        first = int(c["cut"]) + 1                          # pairs cut + 1 to cut + minute: extract row i is pair i + 1
        lo = first - 1
        sl = slice(lo, lo + minute)
        return ex["K"][sl] / N * 100, ex["ham_sum_all"][sl] / (N * BITS) * 100, ex["J"][sl], np.arange(first, first + int(ex["K"][sl].size)), first

    DATA = {k: series(chosen[k]) for k in ROW1 + ROW2}
    short = [k for k in DATA if DATA[k][0].size < minute]
    if short:
        raise Stop(EXIT_ERROR, f"stopped: fewer than {minute} pairs after the cut for {', '.join(short)}")
    firsts = sorted({int(DATA[k][4]) for k in DATA})
    TOP = {k: float(np.nanmax(DATA[k][0])) for k in DATA}
    for grp in SHARED:
        m = max(TOP[k] for k in grp)
        for k in grp:
            TOP[k] = m
    fig = plt.figure(figsize=(WIDTH_IN, 2.95))
    gs = GridSpec(2, 7, figure=fig, left=0.03, right=0.985, top=0.86, bottom=0.09, wspace=0.2, hspace=0.62)
    axes_pos = {}
    for r, row in enumerate((ROW1, ROW2)):
        for col, k in enumerate(row):
            apf, wapf, J, x, _ = DATA[k]
            sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[r, col], height_ratios=[1.55, 0.8], hspace=0.1)
            ax = fig.add_subplot(sub[0]); aj = fig.add_subplot(sub[1], sharex=ax)
            top = max(TOP[k], 1e-9) * 1.15
            ax.plot(x, apf, color=C_APF, lw=0.6); ax.plot(x, wapf, color=C_WAPF, lw=0.6)
            ax.set_ylim(0, top); ax.set_xlim(x[0], x[-1]); ax.set_yticks([])
            ax.set_title(f"{TOP[k]:.2g}%", fontsize=6, pad=1.5, loc="right", color=C_APF)   # the top of this panel's own scale
            aj.plot(x, J, color=C_J, lw=0.55); aj.set_ylim(0, 1.05); aj.set_yticks([0, 1])
            aj.tick_params(axis="y", labelleft=(col == 0), labelsize=5.5, length=1.5, pad=1)
            for a in (ax, aj):
                for sp in ("top", "right"):
                    a.spines[sp].set_visible(False)
                a.spines["left"].set_linewidth(0.5); a.spines["bottom"].set_linewidth(0.5)
                a.tick_params(axis="x", length=1.5, labelsize=5.5, pad=1, width=0.5)
                a.tick_params(axis="y", width=0.5)
            plt.setp(ax.get_xticklabels(), visible=False)
            aj.set_xticks([x[0], x[-1]])
            aj.tick_params(axis="x", labelbottom=False)    # every panel spans the same pairs, named once below
            ax.set_title(SHOW.get(k, k), fontsize=7, pad=1.5, loc="left")
            axes_pos[(r, col)] = ax
    # group brackets above the panel titles
    for r, c0, c1, lab in GROUPS:
        b0 = axes_pos[(r, c0)].get_position(); b1 = axes_pos[(r, c1)].get_position()
        y = b0.y1 + 0.075
        fig.add_artist(plt.Line2D([b0.x0, b1.x1], [y, y], color="0.55", lw=0.5, transform=fig.transFigure))
        fig.text((b0.x0 + b1.x1) / 2, y + 0.008, lab, ha="center", va="bottom", fontsize=6.5, color="0.35")
    # the key in the free slot of row 1
    key = fig.add_subplot(gs[0, 6]); key.axis("off")
    key.plot([], [], color=C_APF, lw=1.1, label="APF")
    key.plot([], [], color=C_WAPF, lw=1.1, label="wAPF")
    key.plot([], [], color=C_J, lw=1.1, label=r"$J_t$")
    key.legend(loc="upper left", fontsize=6.5, frameon=False, handlelength=1.4, borderaxespad=0)
    key.text(0.0, 0.32, "APF and wAPF in %\nof memory, up to the\nblue value; " + r"$J_t$" + " from\n0 to 1", transform=key.transAxes,
             ha="left", va="top", fontsize=6, color="0.3", linespacing=1.25)
    if len(firsts) == 1:
        span = f"pairs {firsts[0]} to {firsts[0] + minute - 1}"
    else:                                                  # the cuts differ by kernel (inputs/head_drop.csv): each panel starts right after its own cut
        span = f"the {minute} pairs after each run's cut (pairs {firsts[0]} to {firsts[-1] + minute - 1})"
    fig.text(0.51, 0.025, f"{span} (about one minute) of the seed-{seed} run of each kernel (idle: run 1)", ha="center", fontsize=7)
    base = fig_dir / FILE_NAMES["shape"]
    fig.savefig(str(base) + ".png", dpi=300); fig.savefig(str(base) + ".pdf")
    plt.close(fig)
    with open(str(base) + "_values.csv", "w", newline="") as fh:              # the plotted values, for checking against the preview
        w = csv.writer(fh); w.writerow(["kernel", "cell_id", "pair", "APF_percent", "wAPF_percent", "J", "count_scale_top_percent"])
        for k in ROW1 + ROW2:
            apf, wapf, J, x, _ = DATA[k]
            for i in range(len(x)):
                w.writerow([k, chosen[k]["cell_id"], int(x[i]), round(float(apf[i]), 6), round(float(wapf[i]), 6), round(float(J[i]), 6), round(TOP[k], 6)])
    print(f"[paperfig] shape: 13 panels, {minute} pairs from " + (f"pair {firsts[0]}" if len(firsts) == 1 else f"pairs {firsts[0]} to {firsts[-1]} (per-kernel cuts)")
          + "; runs " + ", ".join(f"{k} {chosen[k]['cell_id']}" for k in ROW1 + ROW2), flush=True)
    return {"pdf": str(base) + ".pdf", "png": str(base) + ".png", "csv": str(base) + "_values.csv", "cells": {k: chosen[k]["cell_id"] for k in ROW1 + ROW2},
            "first_pair": {k: int(DATA[k][4]) for k in ROW1 + ROW2}, "minute_pairs": minute, "seed": seed, "count_scale_top_percent": {k: round(TOP[k], 6) for k in DATA}}


# ---------------------------------------------------------------------------------------------
# the move
# ---------------------------------------------------------------------------------------------
def run(out: Path, only: list[str] | None = None, *, minute: int = MINUTE, seed: int = SEED) -> dict:
    out = Path(out)
    names = list(only) if only else list(FIGURES)
    unknown = [n for n in names if n not in FIGURES]
    if unknown:
        raise Stop(EXIT_MISSING, f"unknown figure name(s): {unknown}; known: {FIGURES}")
    if not (out / "cells.csv").is_file():
        raise Stop(EXIT_MISSING, f"missing input: {out / 'cells.csv'}")
    _, plt = mpl_setup()
    fig_dir = out / "report" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    cells, ex_hard, meta = paper_cells(out)
    if not cells:
        raise Stop(EXIT_MISSING, "missing input: no admissible cell of the readings table's kernels or idle (cells.csv, gates/preconditions.csv)")
    cuts = sorted({int(c["cut"]) for c in cells})
    print(f"[paperfig] run {out}: {len(cells)} admissible cells of the readings table's kernels and idle ({len(ex_hard)} excluded); the cut per cell from inputs/head_drop.csv: "
          f"{cuts[0]}" + (f" to {cuts[-1]}" if len(cuts) > 1 else "") + " pairs; figures " + ", ".join(names), flush=True)
    written = {}
    for n in names:
        written[n] = fig_representation(out, fig_dir, cells, plt) if n == "representation" else fig_shape(out, fig_dir, cells, plt, minute=minute, seed=seed)
    pj = fig_dir / "paper_figures.json"
    prev = {}
    if pj.is_file():
        try:
            from plan11_encoding_ladder._report_common import read_json
            prev = (read_json(pj) or {}).get("written", {}) or {}
        except Exception:                                  # noqa: BLE001
            prev = {}
    prev.update(written)
    params = {"added": ADDED, "out": str(out), "only": names, "minute_pairs": minute, "seed": seed, "width_in": WIDTH_IN, "pdf_fonttype": 42, "font_family": "STIXGeneral",
              "kernels": KERNELS, "set_a": SET_A, "show_names": SHOW, "cuts": {c["cell_id"]: int(c["cut"]) for c in cells}, "head_drop": meta["head_drop"],
              "preconditions_present": meta["preconditions_present"], "excluded_cells": ex_hard, "n_cells": len(cells), "copied": COPIED, "package_version": PACKAGE_VERSION,
              "written_at": now_iso(),
              "inputs_sha256": inputs_sha256([out / "cells.csv", out / "inputs" / "head_drop.csv", out / "gates" / "preconditions.json"])}
    write_json(pj, result_json("paper_figures", params, CITATION, {"status": "ok", "written": prev}))
    for n, v in written.items():
        print(f"[paperfig] {n}: {v['pdf']} (+ .png, {Path(v['csv']).name})", flush=True)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="plan11_encoding_ladder.paper_figures", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="move 19: the paper's Figure 1 (the representation) and Figure 2 (the shape over time), as the approved previews draw them")
    r.add_argument("--out", required=True)
    r.add_argument("--only", default=None, help="comma-separated: " + ",".join(FIGURES) + " (default both)")
    r.add_argument("--minute-pairs", type=int, default=MINUTE, help="Figure 2's pairs after the cut (default 93, about one minute)")
    r.add_argument("--seed", type=int, default=SEED, help="Figure 2's run of each kernel, by cells.csv seed (default 42)")
    a = ap.parse_args(argv)
    out = Path(os.path.expanduser(a.out))
    only = [s.strip() for s in str(a.only).split(",") if s.strip()] if a.only else None
    try:
        run(out, only, minute=int(a.minute_pairs), seed=int(a.seed))
        return EXIT_OK
    except Stop as exc:
        print(str(exc), file=sys.stderr)
        return exc.code
    except FileNotFoundError as exc:
        print(f"missing input: {exc}", file=sys.stderr)
        return EXIT_MISSING
    except Exception:
        traceback.print_exc()
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
