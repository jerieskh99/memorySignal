#!/usr/bin/env python3
"""figures_detection.py -- the figures of the detection paper (SPEC_DETECTION 5.3), one
function each, PNG and PDF under `report/detection/figures/`.

Builder B (report), 2026-09-17. Matplotlib only; when it is absent the module writes
`report/detection/figures/SKIPPED.txt` naming the missing module and exits 0. The eight members
are drawn with the marker shapes `("o", "s", "^", "v", "D", "P", "X", "*")` indexed by member
(member 1 = `o`), one colour per sub-family letter from the default cycle, and the legend reads
`member m (A)`; no member is named by anything else (K3 Sec. 4 preamble; the name-only rule).
Benign kernels keep plan11's per-kernel colours; idle is grey; harness-idle, when present, is
black hollow. Class membership is read from `gates/detection/cell_classes.csv` and never guessed
from a name. Nothing here computes a verdict.

Figures (SPEC_DETECTION 5.3): `fig2_three_floors`, `fig4_fused_plane_tiers`, `fig5_roc_lowo`,
`fig6_ladder`, `fig_level_map`, `fig_apf_per_tier`, `fig_level2_confusion`.

Corrections from the SPEC_DETECTION reviews implemented here:
  - al-Kindi 7: one plane. Figure 4's identity coordinate is `J - J_null` (`--identity excess`,
    the default; `raw` draws J), the same as the miss table's; the mask is one per-pair rule for
    every class, `K > k_factor * floor_K` with `k_factor` and `floor_K` read from
    `gates/gj.json`, computed here from the extract (no edit to gates_readings.py) and recorded
    as `plane_mask = "gj_mask_K, every class"`; when `gj.json` is absent the plane is unmasked
    and the title says so.
  - al-Kindi 12: one panel for the sandbox class with all members as shapes coloured by letter;
    per-letter panels only under `--plane-per-letter`.
  - al-Kindi 3: the level-2 heat map draws the per-row null verdict beside each letter.

CLI (SPEC 7.1): figures_detection.py --out O [--only NAME,...] [--table10-rung combined]
                [--identity excess|raw] [--plane-per-letter]
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import argparse
import math
import os
import traceback
from collections import defaultdict

import numpy as np

from plan11_encoding_ladder._report_common import (  # noqa: E402
    KERNEL_NAMES, N_PAGES, PACKAGE_VERSION, RUNGS, inputs_sha256, load_cells, not_applicable, not_run, now_iso,
    ok_cells, read_csv, read_json, result_json, to_float, write_json,
)
from plan11_encoding_ladder.figures import Extract, _mpl  # noqa: E402  (imported, never changed)
from plan11_encoding_ladder.tables_detection import (  # noqa: E402
    GK0_AT_FLOOR, GK0_AT_HARNESS_FLOOR, HARNESS_STAGE2_ABSENT, LADDER_FROM_PAIR1_ONLY, RUNG_ROWS, det_grid, grid_source,
    load_scores, split_dir,
)

FIGURE_NAMES = ("fig2_three_floors", "fig4_fused_plane_tiers", "fig5_roc_lowo", "fig6_ladder", "fig_level_map",
                "fig_apf_per_tier", "fig_level2_confusion")
CITATION = "N3 Sec. 1 (Figures 2, 4, 5, 6); K3 moves 4, 5, 6, 11, 17; SPEC_DETECTION 5.3; al-Kindi review 7, 12"
MEMBER_MARKERS = ("o", "s", "^", "v", "D", "P", "X", "*")
IDENTITY_DEFAULT = "excess"          # al-Kindi review 7: J - J_null on both the plane and the miss table
PLANE_MASK_LABEL = "gj_mask_K, every class"
IDLE_CLOUD_RULE = "unmasked (the idle cloud is the floor itself; the K mask would remove it by construction)"
BENIGN_TIERS = ("benign_kernel", "benign_relaunched", "benign_breadth")
TIER_DISPLAY = {"benign_kernel": "kernels", "benign_relaunched": "relaunched", "benign_breadth": "breadth"}


def _fig_dir(out: Path) -> Path:
    d = Path(out) / "report" / "detection" / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(fig, fig_dir: Path, name: str) -> dict:
    import matplotlib.pyplot as plt
    png = fig_dir / f"{name}.png"
    pdf = fig_dir / f"{name}.pdf"
    fig.savefig(png, dpi=110)
    fig.savefig(pdf)
    plt.close(fig)
    return {"png": png, "pdf": pdf}


def _placeholder(fig_dir: Path, name: str, text: str) -> dict:
    """A placeholder figure carrying a `not run:` or `not applicable:` string as text, so the
    figure file exists and says what is missing (the rule of plan11's figures.py)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 2.2))
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=9, wrap=True)
    return _save(fig, fig_dir, name)


def _colors(mpl):
    return mpl.rcParams["axes.prop_cycle"].by_key()["color"]


# ----------------------------------------------------------------------------------------------
# the join and the extracts
# ----------------------------------------------------------------------------------------------
def load_join(out: Path) -> dict:
    """`cell_id -> row` of `gates/detection/cell_classes.csv` (SPEC_DETECTION 2.4); {} when absent."""
    p = Path(out) / "gates" / "detection" / "cell_classes.csv"
    if not p.exists():
        return {}
    return {r["cell_id"]: r for r in read_csv(p)}


def _load_extracts(out: Path, cells: list[dict]) -> dict:
    ex = {}
    for c in ok_cells(cells):
        p = Path(out) / "extract" / c["cell_id"] / "extract.csv"
        if p.exists():
            try:
                ex[c["cell_id"]] = Extract(p)
            except Exception:
                continue
    return ex


def _groups(cells: list[dict], join: dict) -> dict:
    """cells by class (from the join; a cell without a join row is `unassigned` and drawn nowhere)."""
    g = defaultdict(list)
    for c in ok_cells(cells):
        j = join.get(c["cell_id"])
        g[j["class"] if j else "unassigned"].append(c)
    return g


def _member_of(join: dict, cell_id: str) -> tuple[int, str]:
    j = join.get(cell_id) or {}
    try:
        m = int(float(j.get("member_index") or 0))
    except ValueError:
        m = 0
    return m, (j.get("subfamily_letter") or "-")


def _letter_colors(mpl, letters: list[str]) -> dict:
    cols = _colors(mpl)
    return {L: cols[i % len(cols)] for i, L in enumerate(sorted(letters))}


def _marker(m: int) -> str:
    return MEMBER_MARKERS[(m - 1) % len(MEMBER_MARKERS)] if m >= 1 else "o"


def _gj_mask_params(out: Path) -> tuple[float | None, float | None]:
    """(k_factor, floor_K) from gates/gj.json (al-Kindi review 7); (None, None) when absent."""
    p = Path(out) / "gates" / "gj.json"
    if not p.exists():
        return None, None
    try:
        j = read_json(p)
    except Exception:
        return None, None
    return to_float((j.get("params") or {}).get("k_factor")), to_float(j.get("floor_median_K"))


def _boundary_rows(out: Path) -> dict:
    """cell_id -> set of boundary seqs from inputs/iteration_boundaries.csv (stage 2); {} when absent."""
    p = Path(out) / "inputs" / "iteration_boundaries.csv"
    if not p.exists():
        return {}
    res = {}
    for r in read_csv(p):
        seqs = set()
        for s in str(r.get("boundary_seqs", "")).split(";"):
            v = to_float(s)
            if v is not None:
                seqs.add(int(v))
        res[r.get("cell_id")] = seqs
    return res


# ----------------------------------------------------------------------------------------------
# Figure 2, the three floors (N3 Sec. 1 validity block; K3 move 4)
# ----------------------------------------------------------------------------------------------
def fig2_three_floors(out: Path, cells: list[dict], ex: dict, join: dict) -> dict:
    """Three panels (K, l0 per changed page, J): the idle cells' pooled distributions as
    histograms with the five quantiles as vertical lines; the harness-idle distribution overlaid
    when present, else the panel title carries HARNESS_STAGE2_ABSENT; a fourth panel: the idle
    sets by campaign as overlaid K histograms when two exist, else the text `not applicable: one
    idle campaign`. Source: the extracts of the idle and harness-idle cells; gk0.json's band edge."""
    import matplotlib.pyplot as plt
    g = _groups(cells, join)
    idle = [c for c in g.get("idle", []) if c["cell_id"] in ex]
    hidle = [c for c in g.get("harness_idle", []) if c["cell_id"] in ex]
    fd = _fig_dir(out)
    if not idle:
        return _placeholder(fd, "fig2_three_floors", not_run("no idle cell with an extract (class idle in cell_classes.csv)"))
    gk = None
    p = Path(out) / "gates" / "detection" / "gk0.json"
    if p.exists():
        try:
            gk = read_json(p)
        except Exception:
            gk = None
    edge = to_float(gk.get("idle_band_edge")) if gk else None
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    panels = (("K", "K (changed pages)", "K"), ("l0_q50_all", "median l0 per changed page", "l0"), ("J", "J (Jaccard to the next snapshot)", "J"))
    for ax, (col, xlabel, short) in zip(axes[:3], panels):
        vals = np.concatenate([ex[c["cell_id"]].col(col) for c in idle])
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            ax.text(0.5, 0.5, not_run(f"no finite {col} in the idle extracts"), ha="center", va="center", fontsize=8, wrap=True)
            ax.axis("off")
            continue
        ax.hist(vals, bins=40, color="0.6", alpha=0.7, label=f"idle (n = {len(idle)} cells)")
        for q in (0.05, 0.25, 0.50, 0.75, 0.95):
            ax.axvline(float(np.quantile(vals, q)), color="k", lw=0.6, ls=":")
        if short == "K" and edge is not None:
            ax.axvline(edge, color="r", lw=0.8, label="idle band edge")
        title = short
        if hidle:
            hv = np.concatenate([ex[c["cell_id"]].col(col) for c in hidle])
            hv = hv[np.isfinite(hv)]
            if len(hv):
                ax.hist(hv, bins=40, histtype="step", color="k", lw=1.0, label=f"harness-idle (n = {len(hidle)})")
        else:
            title = f"{short}: {HARNESS_STAGE2_ABSENT}"
        ax.set_title(title, fontsize=8)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)
    ax = axes[3]
    camps = defaultdict(list)
    for c in idle:
        camps[(join.get(c["cell_id"]) or {}).get("campaign") or c.get("campaign") or "--"].append(c)
    if len(camps) >= 2:
        cols = _colors(plt.matplotlib)
        for i, (camp, cl) in enumerate(sorted(camps.items())):
            vals = np.concatenate([ex[c["cell_id"]].col("K") for c in cl])
            vals = vals[np.isfinite(vals)]
            ax.hist(vals, bins=40, histtype="step", color=cols[i % len(cols)], lw=1.0, label=f"idle set {camp} (n = {len(cl)})")
        ax.set_title("idle sets by campaign (K)", fontsize=8)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=7)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, not_applicable(f"one idle campaign (n = {len(idle)} cells)"), ha="center", va="center", fontsize=8, wrap=True)
    fig.tight_layout()
    return _save(fig, fd, "fig2_three_floors")


# ----------------------------------------------------------------------------------------------
# Figure 4, the fused plane per tier (K3 move 11; N3 Sec. 1 RQ2)
# ----------------------------------------------------------------------------------------------
def _plane_xy(e: Extract, identity: str) -> tuple[np.ndarray, np.ndarray]:
    x = e.col("r_l0_q50_per")
    y = e.col("J") - e.col("J_null") if identity == "excess" else e.col("J")
    return x, y


def _plane_mask(e: Extract, k_factor: float | None, floor_K: float | None) -> np.ndarray:
    """The one per-pair rule for every class (al-Kindi review 7): K > k_factor * floor_K; all
    True when gj.json is absent."""
    if k_factor is None or floor_K is None:
        return np.ones(e.n, dtype=bool)
    return e.col("K") > k_factor * floor_K


def fig4_fused_plane_tiers(out: Path, cells: list[dict], ex: dict, join: dict, *, identity: str = IDENTITY_DEFAULT,
                           per_letter: bool = False) -> dict:
    """Per snapshot (r_l0_q50_per, J - J_null) with the G-J K mask applied to every class; one
    panel per benign tier present (kernels, and each stage-2 family), one panel for the sandbox
    class with all members as marker shapes coloured by sub-family letter (al-Kindi review 12),
    per-letter panels under `per_letter`; the idle cloud (grey) and the harness-idle cloud (when
    present) in every panel; boundary pairs hollow when inputs/iteration_boundaries.csv exists."""
    import matplotlib.pyplot as plt
    g = _groups(cells, join)
    fd = _fig_dir(out)
    k_factor, floor_K = _gj_mask_params(out)
    masked = k_factor is not None and floor_K is not None
    bounds = _boundary_rows(out)
    tiers = [t for t in BENIGN_TIERS if any(c["cell_id"] in ex for c in g.get(t, []))]
    sandbox = [c for c in g.get("sandbox", []) if c["cell_id"] in ex]
    external = [c for c in g.get("external", []) if c["cell_id"] in ex]
    letters = sorted({_member_of(join, c["cell_id"])[1] for c in sandbox})
    panels = [("tier", t) for t in tiers]
    if sandbox:
        panels.append(("class", "sandbox"))
        if per_letter:
            panels += [("letter", L) for L in letters]
    if external:
        panels.append(("class", "external"))
    if not panels:
        return _placeholder(fd, "fig4_fused_plane_tiers", not_run("no classed cell with an extract"))
    ncol = min(3, len(panels))
    nrow = math.ceil(len(panels) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.4 * nrow), sharex=True, sharey=True, squeeze=False)
    colors = _colors(plt.matplotlib)
    lcol = _letter_colors(plt.matplotlib, letters)
    kcol = {k: colors[i % len(colors)] for i, k in enumerate(KERNEL_NAMES)}
    # the idle and harness-idle clouds are the floor itself: the K mask (K above k_factor times the
    # floor's median K) would remove them by construction, so they are drawn unmasked
    # (IDLE_CLOUD_RULE; listed for the author)
    idle_pts = [(_plane_xy(ex[c["cell_id"]], identity), np.ones(ex[c["cell_id"]].n, dtype=bool)) for c in g.get("idle", []) if c["cell_id"] in ex]
    hidle_pts = [(_plane_xy(ex[c["cell_id"]], identity), np.ones(ex[c["cell_id"]].n, dtype=bool)) for c in g.get("harness_idle", []) if c["cell_id"] in ex]
    n_masked = 0

    def draw_cell(ax, c, col, marker="o", size=4, label=None):
        nonlocal n_masked
        e = ex[c["cell_id"]]
        x, y = _plane_xy(e, identity)
        m = _plane_mask(e, k_factor, floor_K)
        n_masked += int((~m).sum())
        b = bounds.get(c["cell_id"])
        hollow = np.zeros(e.n, dtype=bool)
        if b:
            hollow = np.isin(e.col("seq").astype(int), list(b))
        keep = m & ~hollow
        ax.scatter(x[keep], y[keep], s=size, color=col, alpha=0.5, linewidths=0, marker=marker, label=label)
        if hollow.any():
            ax.scatter(x[m & hollow], y[m & hollow], s=size + 6, facecolors="none", edgecolors=col, linewidths=0.5, marker=marker)

    for i, (kind, key) in enumerate(panels):
        ax = axes[i // ncol][i % ncol]
        for (x, y), m in idle_pts:
            ax.scatter(x[m], y[m], s=3, color="0.6", alpha=0.3, linewidths=0)
        for (x, y), m in hidle_pts:
            ax.scatter(x[m], y[m], s=8, facecolors="none", edgecolors="k", alpha=0.5, linewidths=0.4)
        seen = set()
        if kind == "tier":
            for c in g[key]:
                if c["cell_id"] not in ex:
                    continue
                wk = (join.get(c["cell_id"]) or {}).get("workload_key") or c.get("kernel")
                col = kcol.get(c.get("kernel"), colors[hash(wk) % len(colors)])
                draw_cell(ax, c, col, label=(wk if wk not in seen else None))
                seen.add(wk)
            title = f"tier: {TIER_DISPLAY.get(key, key)}"
        elif kind == "class" and key == "sandbox":
            for c in sandbox:
                m, L = _member_of(join, c["cell_id"])
                lab = f"member {m} ({L})"
                draw_cell(ax, c, lcol.get(L, "k"), marker=_marker(m), size=8, label=(lab if lab not in seen else None))
                seen.add(lab)
            title = "class: the sandbox family (all members)"
        elif kind == "letter":
            for c in sandbox:
                m, L = _member_of(join, c["cell_id"])
                if L != key:
                    continue
                lab = f"member {m} ({L})"
                draw_cell(ax, c, lcol.get(L, "k"), marker=_marker(m), size=8, label=(lab if lab not in seen else None))
                seen.add(lab)
            title = f"sub-family {key}"
        else:
            for c in external:
                m, _ = _member_of(join, c["cell_id"])
                lab = f"external {m}"
                draw_cell(ax, c, "k", marker=_marker(m), size=8, label=(lab if lab not in seen else None))
                seen.add(lab)
            title = "class: external (test only)"
        ax.set_xscale("log")
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, markerscale=1.5)
    for j in range(len(panels), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.supxlabel("median l0 / 4096 over persistent pages (amount)", fontsize=9)
    fig.supylabel("J - J_null (identity, excess over the independence null)" if identity == "excess" else "J (identity, raw)", fontsize=9)
    fig.suptitle((f"plane mask = {PLANE_MASK_LABEL} (K > {k_factor} x {floor_K}); idle cloud {IDLE_CLOUD_RULE.split(' (')[0]}" if masked
                  else "plane unmasked: gates/gj.json absent"), fontsize=8)
    fig.tight_layout()
    paths = _save(fig, fd, "fig4_fused_plane_tiers")
    paths["n_masked_points"] = n_masked
    paths["plane_mask"] = PLANE_MASK_LABEL if masked else "unmasked: gates/gj.json absent"
    paths["identity"] = identity
    paths["idle_cloud"] = IDLE_CLOUD_RULE
    return paths


# ----------------------------------------------------------------------------------------------
# Figure 5, ROC per rung under LOWO (N3 Sec. 1 RQ1)
# ----------------------------------------------------------------------------------------------
def fig5_roc_lowo(out: Path, cells: list[dict], ex: dict, join: dict) -> dict:
    """One ROC curve per rung display name from roc.csv under LOWO (norm; `apf raw` dashed), the
    random scorer's diagonal, the operating point marked at (realized FPR, TPR at 5%)."""
    import matplotlib.pyplot as plt
    fd = _fig_dir(out)
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    colors = _colors(plt.matplotlib)
    drawn = 0
    missing = []
    for i, (display, spec) in enumerate(RUNG_ROWS.items()):
        if spec is None or display in ("comparator", "combined (matched)"):
            continue
        rung, variant, stem = spec
        d = split_dir(out, rung, stem, variant)
        if d is None or not (d / "roc.csv").exists():
            missing.append(display)
            continue
        roc = read_csv(d / "roc.csv")
        fpr = [to_float(r.get("fpr")) for r in roc]
        tpr = [to_float(r.get("tpr")) for r in roc]
        pts = [(a, b) for a, b in zip(fpr, tpr) if a is not None and b is not None]
        if not pts:
            missing.append(display)
            continue
        pts.sort()
        col = colors[i % len(colors)]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=1.2, ls="--" if variant == "raw" else "-", label=display)
        sc = load_scores(out, rung, stem, variant)
        if sc:
            fx, ty = to_float(sc.get("fpr_05_realized")), to_float(sc.get("tpr_05"))
            if fx is not None and ty is not None:
                ax.scatter([fx], [ty], color=col, s=28, marker="o", zorder=5)
    ax.plot([0, 1], [0, 1], color="0.5", lw=0.8, ls=":", label="random scorer (AUC 0.5)")
    ax.axvline(0.05, color="0.7", lw=0.6, ls=":")
    ax.set_xlabel("false-positive rate (realized, out of fold)", fontsize=8)
    ax.set_ylabel("true-positive rate", fontsize=8)
    ax.set_title("ROC under LOWO; the point = the in-fold operating point at 5%" + (f"; missing: {', '.join(missing)}" if missing else ""), fontsize=7)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    paths = _save(fig, fd, "fig5_roc_lowo")
    paths["missing"] = missing
    return paths


# ----------------------------------------------------------------------------------------------
# Figure 6, the time-to-hear ladder (K3 move 17)
# ----------------------------------------------------------------------------------------------
def fig6_ladder(out: Path, cells: list[dict], ex: dict, join: dict) -> dict:
    """TPR at 5% against prefix seconds (log x) per rung, solid for `from_pair1`, dashed for
    `from_boundary` (absent when LADDER_FROM_PAIR1_ONLY), the realized FPR as a thin line on a
    second axis. Source: ladder.csv."""
    import matplotlib.pyplot as plt
    fd = _fig_dir(out)
    p = Path(out) / "gates" / "detection" / "ladder.csv"
    if not p.exists():
        return _placeholder(fd, "fig6_ladder", not_run("gates/detection/ladder.csv missing"))
    rows = read_csv(p)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax2 = ax.twinx()
    colors = _colors(plt.matplotlib)
    drawn = 0
    for i, rung in enumerate(RUNGS):
        for reading, ls in (("from_pair1", "-"), ("from_boundary", "--")):
            sub = [r for r in rows if r.get("rung") == rung and r.get("reading") == reading and str(r.get("note", "")) != LADDER_FROM_PAIR1_ONLY]
            pts = [(to_float(r.get("prefix_s")), to_float(r.get("tpr_05")), to_float(r.get("fpr_05_realized"))) for r in sub]
            pts = [q for q in pts if q[0] is not None and q[1] is not None]
            if not pts:
                continue
            pts.sort()
            ax.plot([q[0] for q in pts], [q[1] for q in pts], color=colors[i % len(colors)], ls=ls, lw=1.2, marker="o", ms=3, label=f"{rung} ({reading})")
            fp = [(q[0], q[2]) for q in pts if q[2] is not None]
            if fp:
                ax2.plot([q[0] for q in fp], [q[1] for q in fp], color=colors[i % len(colors)], ls=ls, lw=0.5, alpha=0.6)
            drawn += 1
    if drawn == 0:
        plt.close(fig)
        return _placeholder(fd, "fig6_ladder", not_run("ladder.csv holds no numeric row"))
    ax.set_xscale("log")
    ax.set_xlabel("prefix length (s of guest time)", fontsize=8)
    ax.set_ylabel("TPR at 5% (in-fold)", fontsize=8)
    ax2.set_ylabel("realized FPR (thin)", fontsize=8)
    ax.set_ylim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=7)
    ax2.tick_params(labelsize=7)
    fig.tight_layout()
    return _save(fig, fd, "fig6_ladder")


# ----------------------------------------------------------------------------------------------
# the level map (K3 move 5)
# ----------------------------------------------------------------------------------------------
def fig_level_map(out: Path, cells: list[dict], ex: dict, join: dict) -> dict:
    """One histogram per tier of the level quantity per cell (K_med from gk0_cells.csv, log x),
    the idle band edge and the idle envelope edge as vertical lines, the members as marker shapes
    at their per-cell levels above the histogram."""
    import matplotlib.pyplot as plt
    fd = _fig_dir(out)
    p = Path(out) / "gates" / "detection" / "gk0_cells.csv"
    if not p.exists():
        return _placeholder(fd, "fig_level_map", not_run("gates/detection/gk0_cells.csv missing"))
    rows = read_csv(p)
    gk = None
    pj = Path(out) / "gates" / "detection" / "gk0.json"
    if pj.exists():
        try:
            gk = read_json(pj)
        except Exception:
            gk = None
    edge = to_float(gk.get("idle_band_edge")) if gk else to_float(rows[0].get("idle_band_edge")) if rows else None
    env = to_float((gk.get("idle_envelope") or {}).get("K_med")) if gk else (to_float(rows[0].get("env_K_med")) if rows else None)
    classes = [c for c in ("benign_kernel", "benign_relaunched", "benign_breadth", "idle", "harness_idle", "external") if any(r.get("class") == c for r in rows)]
    sandbox = [r for r in rows if r.get("class") == "sandbox"]
    fig, axes = plt.subplots(len(classes) + (1 if sandbox else 0), 1, figsize=(6.4, 1.8 * (len(classes) + 1)), sharex=True, squeeze=False)
    allK = [to_float(r.get("K_med")) for r in rows if to_float(r.get("K_med")) not in (None, 0)]
    lo = max(1.0, min(allK) * 0.8) if allK else 1.0
    hi = max(allK) * 1.2 if allK else 10.0
    bins = np.logspace(math.log10(lo), math.log10(hi), 40)
    def letter_of(r):  # the letter from the join (gk0_cells.csv carries the member index only)
        return (join.get(r.get("cell_id")) or {}).get("subfamily_letter") or r.get("subfamily_letter") or "-"
    letters = sorted({letter_of(r) for r in sandbox})
    lcol = _letter_colors(plt.matplotlib, letters)
    for i, cls in enumerate(classes):
        ax = axes[i][0]
        vals = [to_float(r.get("K_med")) for r in rows if r.get("class") == cls]
        vals = [v for v in vals if v is not None and v > 0]
        ax.hist(vals, bins=bins, color="0.6" if cls in ("idle", "harness_idle") else "C0", alpha=0.7)
        ax.set_title(f"{cls}: K_med per cell (n = {len(vals)})", fontsize=8)
        for v, col, lab in ((edge, "r", "idle band edge"), (env, "m", "idle envelope K_med")):
            if v:
                ax.axvline(v, color=col, lw=0.8, ls="--", label=lab)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)
    if sandbox:
        ax = axes[len(classes)][0]
        vals = [to_float(r.get("K_med")) for r in sandbox]
        ax.hist([v for v in vals if v], bins=bins, color="0.85", alpha=0.7)
        seen = set()
        for r in sandbox:
            try:
                m = int(float(r.get("member_index") or 0))
            except ValueError:
                m = 0
            L = letter_of(r)
            v = to_float(r.get("K_med"))
            if not v:
                continue
            lab = f"member {m} ({L})"
            ax.scatter([v], [ax.get_ylim()[1] * 0.9 - 0.05 * ax.get_ylim()[1] * (m % 4)], marker=_marker(m), color=lcol.get(L, "k"), s=30,
                       label=(lab if lab not in seen else None), zorder=5)
            seen.add(lab)
        for v, col, lab in ((edge, "r", "idle band edge"), (env, "m", "idle envelope K_med")):
            if v:
                ax.axvline(v, color=col, lw=0.8, ls="--")
        n_floor = sum(1 for r in sandbox if r.get("verdict") in (GK0_AT_FLOOR, GK0_AT_HARNESS_FLOOR))
        ax.set_title(f"the sandbox family: members at their per-cell levels (n at floor = {n_floor})", fontsize=8)
        ax.legend(fontsize=6, ncol=4)
        ax.tick_params(labelsize=7)
    axes[-1][0].set_xscale("log")
    axes[-1][0].set_xlabel("K_med (median changed pages per pair; log)", fontsize=8)
    fig.tight_layout()
    return _save(fig, fd, "fig_level_map")


# ----------------------------------------------------------------------------------------------
# APF(t) per tier (K3 move 6)
# ----------------------------------------------------------------------------------------------
def fig_apf_per_tier(out: Path, cells: list[dict], ex: dict, join: dict) -> dict:
    """APF(t) = K / N (log y) overlaid: one panel per benign workload of the kernel tier (as
    plan11's fig_apf_per_kernel), one for idle, one per stage-2 family, and one panel per
    sub-family with each member's reps in one colour and the member's marker at the line's end."""
    import matplotlib.pyplot as plt
    g = _groups(cells, join)
    fd = _fig_dir(out)
    panels = []
    byk = defaultdict(list)
    for c in g.get("benign_kernel", []):
        byk[(join.get(c["cell_id"]) or {}).get("workload_key") or c.get("kernel")].append(c)
    for k in [k for k in KERNEL_NAMES if k in byk] + sorted(k for k in byk if k not in KERNEL_NAMES):
        panels.append(("kernel", k, byk[k]))
    if g.get("idle"):
        panels.append(("idle", "idle", g["idle"]))
    for cls in ("benign_relaunched", "benign_breadth", "harness_idle"):
        if g.get(cls):
            panels.append((cls, TIER_DISPLAY.get(cls, cls), g[cls]))
    sandbox = g.get("sandbox", [])
    letters = sorted({_member_of(join, c["cell_id"])[1] for c in sandbox})
    for L in letters:
        panels.append(("letter", L, [c for c in sandbox if _member_of(join, c["cell_id"])[1] == L]))
    if g.get("external"):
        panels.append(("external", "external (test only)", g["external"]))
    if not panels:
        return _placeholder(fd, "fig_apf_per_tier", not_run("no classed cell with an extract"))
    ncol = 4
    nrow = max(1, math.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow), sharey=True, squeeze=False)
    colors = _colors(plt.matplotlib)
    lcol = _letter_colors(plt.matplotlib, letters)
    mcol = {}
    for i, (kind, name, cl) in enumerate(panels):
        ax = axes[i // ncol][i % ncol]
        n_drawn = 0
        seen = set()
        for c in cl:
            e = ex.get(c["cell_id"])
            if e is None:
                continue
            y = e.col("K") / N_PAGES
            y = np.where(y > 0, y, np.nan)
            if kind == "letter":
                m, L = _member_of(join, c["cell_id"])
                col = mcol.setdefault(m, colors[(len(mcol)) % len(colors)])
                ax.plot(e.col("seq"), y, color=col, lw=0.6, alpha=0.6)
                lab = f"member {m} ({L})"
                fin = np.where(np.isfinite(y))[0]
                if len(fin):
                    ax.scatter([e.col("seq")[fin[-1]]], [y[fin[-1]]], marker=_marker(m), color=col, s=26, zorder=5, label=(lab if lab not in seen else None))
                    seen.add(lab)
            else:
                col = "0.5" if kind in ("idle", "harness_idle") else ("k" if kind == "external" else colors[i % len(colors)])
                ax.plot(e.col("seq"), y, color=col, lw=0.6, alpha=0.6)
            n_drawn += 1
        ax.set_yscale("log")
        ax.set_title((f"sub-family {name}" if kind == "letter" else f"{name}") + f" (n = {n_drawn})", fontsize=8)
        ax.tick_params(labelsize=7)
        if kind == "letter":
            ax.legend(fontsize=6)
    for j in range(len(panels), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.supxlabel("seq (pair index)", fontsize=9)
    fig.supylabel("APF = K / N", fontsize=9)
    fig.tight_layout()
    return _save(fig, fd, "fig_apf_per_tier")


# ----------------------------------------------------------------------------------------------
# the level-2 confusion heat map
# ----------------------------------------------------------------------------------------------
def fig_level2_confusion(out: Path, cells: list[dict], ex: dict, join: dict) -> dict:
    """The level-2 confusion counts as a heat map per rung (letters on both axes; a row without a
    held-out test hatched with its `no held-out test` string; the per-row null verdict beside the
    letter, al-Kindi review 3). Source: splits/<rung>/<gid>/level2/confusion.csv."""
    import matplotlib.pyplot as plt
    fd = _fig_dir(out)
    avail = []
    for rung in RUNGS:
        gid = det_grid(out, rung)
        if gid is None:
            continue
        p = Path(out) / "gates" / "detection" / "splits" / rung / gid / "level2" / "confusion.csv"
        if p.exists():
            avail.append((rung, p))
    if not avail:
        return _placeholder(fd, "fig_level2_confusion", not_run("no level2/confusion.csv under gates/detection/splits"))
    fig, axes = plt.subplots(1, len(avail), figsize=(3.4 * len(avail), 3.2), squeeze=False)
    for ax, (rung, p) in zip(axes[0], avail):
        conf = read_csv(p)
        letters = sorted({k[len("pred_"):] for r in conf for k in r.keys() if k.startswith("pred_")})
        rows_L = [r.get("true_subfamily") for r in conf]
        M = np.full((len(conf), len(letters)), np.nan)
        for i, r in enumerate(conf):
            for j, L in enumerate(letters):
                v = to_float(r.get(f"pred_{L}"))
                M[i, j] = v if v is not None else np.nan
        ax.imshow(np.nan_to_num(M, nan=0.0), cmap="Blues", aspect="auto")
        for i, r in enumerate(conf):
            held = str(r.get("status", "")).endswith("no held-out test")
            for j in range(len(letters)):
                if held:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="//", edgecolor="0.5", lw=0))
                else:
                    ax.text(j, i, f"{int(M[i, j])}" if np.isfinite(M[i, j]) else "--", ha="center", va="center", fontsize=8)
            if held:
                ax.text(len(letters) / 2 - 0.5, i, str(r.get("status")), ha="center", va="center", fontsize=6, color="0.3")
        ax.set_xticks(range(len(letters)))
        ax.set_xticklabels([f"pred {L}" for L in letters], fontsize=7)
        ax.set_yticks(range(len(conf)))
        ax.set_yticklabels([f"{L} ({r.get('verdict') or '--'})" for L, r in zip(rows_L, conf)], fontsize=6)
        ax.set_title(f"level 2, {rung}", fontsize=8)
    fig.tight_layout()
    return _save(fig, fd, "fig_level2_confusion")


# ----------------------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------------------
def run(out: Path, only: list[str] | None = None, *, table10_rung: str = "combined", identity: str = IDENTITY_DEFAULT,
        plane_per_letter: bool = False) -> dict:
    out = Path(out)
    names = list(only) if only else list(FIGURE_NAMES)
    unknown = [n for n in names if n not in FIGURE_NAMES]
    if unknown:
        raise ValueError(f"unknown figure name(s): {unknown}; known: {FIGURE_NAMES}")
    fd = _fig_dir(out)
    params = {"out": str(out), "only": names, "table10_rung": table10_rung, "identity": identity, "plane_per_letter": plane_per_letter,
              "plane_mask": PLANE_MASK_LABEL, "idle_cloud": IDLE_CLOUD_RULE, "member_markers": list(MEMBER_MARKERS), "package_version": PACKAGE_VERSION,
              "grid_source": grid_source(out), "written_at": now_iso()}
    mpl = _mpl()
    if mpl is None:
        (fd / "SKIPPED.txt").write_text("figures skipped: module matplotlib is not importable (pip install --user matplotlib); "
                                        "see RUNBOOK_DETECTION.md section 0\n", encoding="utf-8")
        write_json(fd / "figures.json", result_json("detection.figures", params, CITATION,
                                                     {"status": not_run("matplotlib not importable"), "written": {}}))
        return {"skipped": fd / "SKIPPED.txt"}
    cells = load_cells(out)
    join = load_join(out)
    ex = _load_extracts(out, cells)
    written, errors = {}, {}
    fns = {"fig2_three_floors": fig2_three_floors, "fig5_roc_lowo": fig5_roc_lowo, "fig6_ladder": fig6_ladder,
           "fig_level_map": fig_level_map, "fig_apf_per_tier": fig_apf_per_tier, "fig_level2_confusion": fig_level2_confusion}
    for n in names:
        try:
            if n == "fig4_fused_plane_tiers":
                written[n] = fig4_fused_plane_tiers(out, cells, ex, join, identity=identity, per_letter=plane_per_letter)
            else:
                written[n] = fns[n](out, cells, ex, join)
        except Exception as e:  # a figure that cannot be drawn writes a placeholder, never a blank
            errors[n] = f"{type(e).__name__}: {e}"
            try:
                written[n] = _placeholder(fd, n, not_run(f"{n}: {type(e).__name__}: {e}"))
            except Exception:
                pass
    prev = fd / "figures.json"
    prev_written = {}
    if prev.exists():
        try:
            prev_written = read_json(prev).get("written", {})
        except Exception:
            prev_written = {}
    prev_written.update({n: {k: str(v) for k, v in v_.items()} for n, v_ in written.items()})
    write_json(prev, result_json("detection.figures", params, CITATION,
                                 {"status": "ok" if not errors else f"errors: {errors}", "n_extracts": len(ex), "n_join_rows": len(join),
                                  "written": prev_written, "errors": errors,
                                  "inputs_sha256": inputs_sha256([out / "cells.csv", out / "gates" / "detection" / "cell_classes.csv",
                                                                  out / "gates" / "gj.json", out / "gates" / "detection" / "gk0_cells.csv",
                                                                  out / "gates" / "detection" / "ladder.csv"])}))
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="plan11 detection figures (builder B)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="comma-separated: " + ",".join(FIGURE_NAMES))
    ap.add_argument("--table10-rung", default="combined", choices=list(RUNGS))
    ap.add_argument("--identity", default=IDENTITY_DEFAULT, choices=["excess", "raw"], help="the plane's identity coordinate (al-Kindi review 7)")
    ap.add_argument("--plane-per-letter", action="store_true", help="also one fused-plane panel per sub-family letter (al-Kindi review 12)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if not (out / "cells.csv").exists() and not os.environ.get("PLAN11_NO_MPL") and _mpl() is not None:
        print(f"missing input: {out / 'cells.csv'}", file=sys.stderr)
        return 2
    only = [s.strip() for s in a.only.split(",") if s.strip()] if a.only else None
    try:
        written = run(out, only, table10_rung=a.table10_rung, identity=a.identity, plane_per_letter=a.plane_per_letter)
    except FileNotFoundError as e:
        print(f"missing input: {e}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 1
    for n, v in written.items():
        print(f"{n}: {v.get('pdf', v) if isinstance(v, dict) else v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
