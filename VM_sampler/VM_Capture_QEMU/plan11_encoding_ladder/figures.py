#!/usr/bin/env python3
"""figures.py -- the figures of P2 Sec. VII (SPEC 6.7), one function each, PNG and PDF.

Builder 3 (report), 2026-09-16. Matplotlib only, no seaborn, colours from the default cycle,
the reps of one kernel in one colour (SPEC 6.7). When matplotlib is absent the module writes
`report/figures/SKIPPED.txt` naming the missing module and exits 0 (SPEC binding rules).

Every figure reads the per-cell extracts (about 1,000 rows each; loaded freely, SPEC binding
rules) and builder 2's result files; the piano roll is the one figure that re-streams a
trajectory (through `extract.open_text` when importable, else the copy in `_report_common`),
subsampling rows by `--piano-stride`. Nothing here computes a verdict.

Corrections from the SPEC reviews implemented here:
  - al-Kindi 9: the fused plane applies `gj_mask`'s `mask_K` column by default and
    `mask_persist` under `--fused-plane-mask persist` (SPEC section 8 gains `fused_plane_mask`).
  - al-Kindi 1 (G-C's detection ratio): the floyd decay figure aligns passes at boundaries it
    reconstructs from the K jump with the detection ratio `--decay-jump-ratio 1.5` (recorded in
    `params`; a figure-only reconstruction, never a gate).

CLI (SPEC 7.1): figures.py --out O [--only NAME,...] [--piano-cell ID] [--piano-stride 16]
                [--fused-plane-mask K|persist] [--decay-jump-ratio 1.5]
Names: apf_per_kernel, level_matched, fused_plane, j_hist, ratio_hist, floyd_decay,
piano_roll, table5_grid, dhodapkar_sweep (build epoch 2, builder A: the Dhodapkar-Smith threshold sweep).
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import argparse
import csv
import math
import os
import statistics
import traceback
from collections import defaultdict

import numpy as np

from plan11_encoding_ladder._report_common import (  # noqa: E402
    GDEC_DECAY, GRID_IDS, KERNEL_NAMES, LEVEL_MATCHED_SETS, N_PAGES, PACKAGE_VERSION, RUNGS,
    RUNG_DISPLAY, get_open_text, grid_label, inputs_sha256, load_cells, not_run, now_iso,
    ok_cells, read_csv, read_json, result_json, to_float, write_json,
)

FIGURE_NAMES = ("apf_per_kernel", "level_matched", "fused_plane", "j_hist", "ratio_hist",
                "floyd_decay", "piano_roll", "table5_grid", "dhodapkar_sweep")
CITATION = "P2 Sec. VII 'Figures'; SPEC 6.7; K2 moves 5, 8, 9, 10"


def _fig_dir(out: Path) -> Path:
    d = Path(out) / "report" / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _mpl():
    """matplotlib with the Agg backend, or None when it cannot be imported (or when the
    environment variable PLAN11_NO_MPL is set, the test's way of exercising the skip path)."""
    if os.environ.get("PLAN11_NO_MPL"):
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401
        return matplotlib
    except Exception:
        return None


def _save(fig, fig_dir: Path, name: str) -> dict:
    import matplotlib.pyplot as plt
    png = fig_dir / f"fig_{name}.png"
    pdf = fig_dir / f"fig_{name}.pdf"
    fig.savefig(png, dpi=110)
    fig.savefig(pdf)
    plt.close(fig)
    return {"png": png, "pdf": pdf}


def _placeholder(fig_dir: Path, name: str, text: str) -> dict:
    """A placeholder figure carrying a verdict or a `not run:` string as text (SPEC 6.7,
    floyd decay), so the figure file exists and says what is missing."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 2.2))
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=9, wrap=True)
    return _save(fig, fig_dir, name)


# ----------------------------------------------------------------------------------------------
# extract loading
# ----------------------------------------------------------------------------------------------
class Extract:
    """One cell's extract as float arrays keyed by column (NaN for blanks)."""

    def __init__(self, path: Path):
        self.path = path
        cols = defaultdict(list)
        with open(path, "r", newline="", encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            self.columns = list(rd.fieldnames or [])
            for r in rd:
                for c in self.columns:
                    v = to_float(r.get(c))
                    cols[c].append(math.nan if v is None else v)
        self.a = {c: np.asarray(v, dtype=float) for c, v in cols.items()}
        self.n = len(self.a.get("seq", []))

    def col(self, name: str) -> np.ndarray:
        return self.a.get(name, np.full(self.n, np.nan))


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


def _by_kernel(cells: list[dict]) -> tuple[dict, list[dict]]:
    byk = defaultdict(list)
    idle = []
    for c in ok_cells(cells):
        if c.get("role") == "idle":
            idle.append(c)
        else:
            byk[c["kernel"]].append(c)
    return byk, idle


def _kernel_order(byk: dict) -> list[str]:
    return [k for k in KERNEL_NAMES if k in byk] + sorted(k for k in byk if k not in KERNEL_NAMES)


def _colors(mpl):
    return mpl.rcParams["axes.prop_cycle"].by_key()["color"]


# ----------------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------------
def fig_apf_per_kernel(out: Path, cells: list[dict], ex: dict) -> dict:
    """`fig_apf_per_kernel`: APF(t) = K / N per kernel, eight reps overlaid, one panel per
    kernel, shared y in log scale; the idle cells as a 13th panel when present (SPEC 6.7; K2
    move 5). Source: the extracts."""
    import matplotlib.pyplot as plt
    byk, idle = _by_kernel(cells)
    order = _kernel_order(byk)
    panels = order + (["idle"] if idle else [])
    ncol = 4
    nrow = max(1, math.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.3 * nrow), sharex=False, sharey=True, squeeze=False)
    colors = _colors(plt.matplotlib)
    for i, name in enumerate(panels):
        ax = axes[i // ncol][i % ncol]
        cl = idle if name == "idle" else byk[name]
        col = "0.5" if name == "idle" else colors[i % len(colors)]
        n_drawn = 0
        for c in cl:
            e = ex.get(c["cell_id"])
            if e is None:
                continue
            y = e.col("K") / N_PAGES
            y = np.where(y > 0, y, np.nan)
            ax.plot(e.col("seq"), y, color=col, lw=0.6, alpha=0.6)
            n_drawn += 1
        ax.set_yscale("log")
        ax.set_title(f"{name} (n = {n_drawn})", fontsize=9)
        ax.tick_params(labelsize=7)
    for j in range(len(panels), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.supxlabel("seq (pair index)", fontsize=9)
    fig.supylabel("APF = K / N", fontsize=9)
    fig.tight_layout()
    return _save(fig, _fig_dir(out), "apf_per_kernel")


def fig_level_matched(out: Path, cells: list[dict], ex: dict) -> dict:
    """`fig_level_matched`: the level-matched sets side by side, panel A floyd, histogram,
    nbody; panel B fft, gemm; all reps thin, one median line per kernel (the median over reps
    at each seq, over the common length) (SPEC 6.7; P2 Sec. 2, Sec. VI; K2 move 5)."""
    import matplotlib.pyplot as plt
    byk, _ = _by_kernel(cells)
    colors = _colors(plt.matplotlib)
    fig, axes = plt.subplots(1, len(LEVEL_MATCHED_SETS), figsize=(5.0 * len(LEVEL_MATCHED_SETS), 3.2), squeeze=False)
    for pi, kset in enumerate(LEVEL_MATCHED_SETS):
        ax = axes[0][pi]
        for ki, k in enumerate(kset):
            col = colors[(pi * 3 + ki) % len(colors)]
            series = []
            for c in byk.get(k, []):
                e = ex.get(c["cell_id"])
                if e is None:
                    continue
                y = e.col("K") / N_PAGES
                ax.plot(e.col("seq"), np.where(y > 0, y, np.nan), color=col, lw=0.4, alpha=0.35)
                series.append(y)
            if series:
                L = min(len(s) for s in series)
                med = np.nanmedian(np.vstack([s[:L] for s in series]), axis=0)
                ax.plot(np.arange(1, L + 1), np.where(med > 0, med, np.nan), color=col, lw=1.6, label=f"{k} (median of {len(series)})")
            else:
                ax.plot([], [], color=col, label=f"{k} (no extract)")
        ax.set_yscale("log")
        ax.set_title(f"panel {'AB'[pi] if pi < 2 else pi}: " + ", ".join(kset), fontsize=9)
        ax.set_xlabel("seq (pair index)", fontsize=8)
        ax.set_ylabel("APF = K / N", fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    return _save(fig, _fig_dir(out), "level_matched")


def _load_mask(out: Path, cell_id: str, which: str) -> np.ndarray | None:
    """`gates/gj_mask/<cell_id>.npy` (SPEC 3.6.1) with al-Kindi review 9's two boolean columns:
    a 1-D array is `mask_K`; a 2-D `[n, 2]` array holds `mask_K` (column 0) and `mask_persist`
    (column 1); an `.npz` or a structured array is read by field name. None when absent."""
    d = Path(out) / "gates" / "gj_mask"
    for p in (d / f"{cell_id}.npy", d / f"{cell_id}.npz"):
        if not p.exists():
            continue
        try:
            m = np.load(p, allow_pickle=False)
        except Exception:
            return None
        if isinstance(m, np.lib.npyio.NpzFile):
            key = "mask_K" if which == "K" else "mask_persist"
            if key in m.files:
                return np.asarray(m[key], dtype=bool)
            return np.asarray(m[m.files[0]], dtype=bool) if m.files else None
        m = np.asarray(m)
        if m.dtype.names:
            key = "mask_K" if which == "K" else "mask_persist"
            return np.asarray(m[key], dtype=bool) if key in m.dtype.names else np.asarray(m[m.dtype.names[0]], dtype=bool)
        if m.ndim == 2 and m.shape[1] >= 2:
            return np.asarray(m[:, 0 if which == "K" else 1], dtype=bool)
        return np.asarray(m, dtype=bool).ravel()
    return None


def _align_mask(mask: np.ndarray | None, n_rows: int) -> np.ndarray:
    """Align a per-series-row mask to the extract's rows (the last row has no J; SPEC 2.4).
    Length n_rows - 1: rows 0..n-2; length n_rows: as is; shorter: a head drop, the mask covers
    the last len(mask) series rows. Unmasked rows read True (interpretable)."""
    full = np.ones(n_rows, dtype=bool)
    if mask is None:
        return full
    L = len(mask)
    if L == n_rows:
        return mask.astype(bool)
    if L == n_rows - 1:
        full[:L] = mask
        return full
    if 0 < L < n_rows - 1:
        full[n_rows - 1 - L:n_rows - 1] = mask
        return full
    return full


def fig_fused_plane(out: Path, cells: list[dict], ex: dict, *, mask_kind: str = "K") -> dict:
    """`fig_fused_plane`: per snapshot `(r_l0_q50_per, J)`, one panel per kernel, eight reps in
    one colour, idle cells overlaid in grey, the G-J mask applied (masked pairs hollow) (SPEC
    6.7; K2 move 8, the move al-Kindi most wants to watch). `mask_kind` = `K` applies
    `mask_K` (the definition, SPEC 3.6.1) and `persist` applies `mask_persist` (al-Kindi
    review 9)."""
    import matplotlib.pyplot as plt
    byk, idle = _by_kernel(cells)
    order = _kernel_order(byk)
    ncol = 4
    nrow = max(1, math.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow), sharex=True, sharey=True, squeeze=False)
    colors = _colors(plt.matplotlib)
    idle_pts = []
    for c in idle:
        e = ex.get(c["cell_id"])
        if e is not None:
            idle_pts.append((e.col("r_l0_q50_per"), e.col("J")))
    n_masked_total = 0
    for i, k in enumerate(order):
        ax = axes[i // ncol][i % ncol]
        col = colors[i % len(colors)]
        for xs, ys in idle_pts:
            ax.scatter(xs, ys, s=3, color="0.6", alpha=0.3, linewidths=0)
        for c in byk[k]:
            e = ex.get(c["cell_id"])
            if e is None:
                continue
            x, y = e.col("r_l0_q50_per"), e.col("J")
            m = _align_mask(_load_mask(out, c["cell_id"], mask_kind), e.n)
            ax.scatter(x[m], y[m], s=4, color=col, alpha=0.5, linewidths=0)
            ax.scatter(x[~m], y[~m], s=8, facecolors="none", edgecolors=col, alpha=0.5, linewidths=0.4)
            n_masked_total += int((~m).sum())
        ax.set_xscale("log")
        ax.set_title(k, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_ylim(-0.02, 1.02)
    for j in range(len(order), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.supxlabel("median l0 / 4096 over persistent pages", fontsize=9)
    fig.supylabel("J (Jaccard to the next snapshot)", fontsize=9)
    fig.tight_layout()
    paths = _save(fig, _fig_dir(out), "fused_plane")
    paths["n_masked_points"] = n_masked_total
    return paths


def _floor_j_quantiles(out: Path) -> tuple[list[float] | None, str]:
    """The idle cells' J quantiles from `gates/gj.json` (SPEC 3.6.1: the five quantiles of J
    pooled over idle cells; the floor null of `fig_j_hist`, CR 2.2 item 31). Builder 2 writes
    them as `idle_J = {"quantiles": [0.05, 0.25, 0.5, 0.75, 0.95], "J": [q05, ..., q95], ...}`
    and `idle_J = null` when there is no idle cell (gates_readings.py gate_gj; CHECK_1.md B4);
    that shape is read first. Fallback: any key containing `quant` and `J` with five numbers, or
    a dict of q05..q95. Returns `(quantiles, reason)`: the five quantiles and `""`, or `None`
    and the reason they are absent (printed on the figure's x axis)."""
    p = Path(out) / "gates" / "gj.json"
    if not p.exists():
        return None, "gates/gj.json missing"
    j = read_json(p)
    if "idle_J" in j:
        ij = j.get("idle_J")
        if ij is None:
            return None, "no idle cell (gates/gj.json idle_J = null)"
        if isinstance(ij, dict) and isinstance(ij.get("J"), list) and len(ij["J"]) == 5 \
                and all(to_float(x) is not None for x in ij["J"]):
            return [float(x) for x in ij["J"]], ""
        if isinstance(ij, dict) and ij.get("J") is None:
            return None, "idle cells have no valid J pair (gates/gj.json idle_J.J = null)"

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                kl = str(k).lower()
                if "quant" in kl and "j" in kl:
                    if isinstance(v, (list, tuple)) and len(v) >= 3 and all(to_float(x) is not None for x in v):
                        return [float(x) for x in v]
                    if isinstance(v, dict):
                        vals = [to_float(v.get(q)) for q in ("q05", "q25", "q50", "q75", "q95")]
                        if all(x is not None for x in vals):
                            return vals
                r = walk(v)
                if r is not None:
                    return r
        return None
    q = walk(j)
    return (q, "") if q is not None else (None, "gates/gj.json carries no idle J quantiles")


def fig_j_hist(out: Path, cells: list[dict], ex: dict) -> dict:
    """`fig_j_hist`: J(t) histograms per kernel with the independence null (the per-pair
    `J_null` histogram, extract column) and the floor null (the idle cells' J quantiles from
    `gates/gj.json` as vertical lines) (SPEC 6.7; P2 Sec. IV rung 1; K2 move 9)."""
    import matplotlib.pyplot as plt
    byk, _ = _by_kernel(cells)
    order = _kernel_order(byk)
    fq, fq_reason = _floor_j_quantiles(out)
    ncol = 4
    nrow = max(1, math.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow), sharex=True, squeeze=False)
    colors = _colors(plt.matplotlib)
    bins = np.linspace(0, 1, 41)
    for i, k in enumerate(order):
        ax = axes[i // ncol][i % ncol]
        col = colors[i % len(colors)]
        J, Jn = [], []
        for c in byk[k]:
            e = ex.get(c["cell_id"])
            if e is None:
                continue
            J.append(e.col("J"))
            Jn.append(e.col("J_null"))
        if J:
            J = np.concatenate(J)
            Jn = np.concatenate(Jn)
            ax.hist(J[~np.isnan(J)], bins=bins, color=col, alpha=0.7, label="J")
            ax.hist(Jn[~np.isnan(Jn)], bins=bins, histtype="step", color="k", lw=0.8, label="independence null")
        if fq:
            for q in fq:
                ax.axvline(q, color="0.5", lw=0.6, ls="--")
        ax.set_title(k, fontsize=9)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=6)
    for j in range(len(order), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.supxlabel("J" + ("" if fq else f"   (floor null: {fq_reason}, no idle quantiles drawn)"), fontsize=9)
    fig.tight_layout()
    paths = _save(fig, _fig_dir(out), "j_hist")
    paths["floor_quantiles"] = fq
    paths["floor_quantiles_absent_reason"] = fq_reason
    return paths


def fig_ratio_hist(out: Path, cells: list[dict], ex: dict) -> dict:
    """`fig_ratio_hist`: histograms of the three ratios' per-snapshot medians per kernel
    (`r_l0_q50_per`, `r_l1l0_q50_per`, `r_haml0_q50_per`), one panel per ratio, one step
    histogram per kernel with reps pooled (SPEC 6.7; P2 Sec. IV rung 2; K2 move 10)."""
    import matplotlib.pyplot as plt
    byk, _ = _by_kernel(cells)
    order = _kernel_order(byk)
    ratios = [("r_l0_q50_per", "l0 / 4096"), ("r_l1l0_q50_per", "l1 / l0"), ("r_haml0_q50_per", "hamming / l0")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    colors = _colors(plt.matplotlib)
    for pi, (col_name, label) in enumerate(ratios):
        ax = axes[pi]
        allv = []
        per_k = {}
        for k in order:
            v = []
            for c in byk[k]:
                e = ex.get(c["cell_id"])
                if e is not None:
                    v.append(e.col(col_name))
            if v:
                v = np.concatenate(v)
                v = v[np.isfinite(v) & (v > 0)]
                per_k[k] = v
                allv.append(v)
        if allv:
            allv = np.concatenate(allv)
            lo, hi = np.nanmin(allv), np.nanmax(allv)
            if lo <= 0 or not np.isfinite(lo):
                lo = 1e-6
            bins = np.logspace(math.log10(lo), math.log10(max(hi, lo * 1.001)), 40)
            for i, k in enumerate(order):
                if k in per_k and len(per_k[k]):
                    ax.hist(per_k[k], bins=bins, histtype="step", color=colors[i % len(colors)], lw=1.0, label=k)
            ax.set_xscale("log")
        ax.set_title(f"per-snapshot median {label}", fontsize=9)
        ax.tick_params(labelsize=7)
        if pi == 2:
            ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    return _save(fig, _fig_dir(out), "ratio_hist")


def _gdec_verdict(out: Path, kernel: str) -> tuple[str | None, dict | None]:
    p = Path(out) / "gates" / "gdec.csv"
    if not p.exists():
        return None, None
    rows = read_csv(p)
    for r in rows:
        if r.get("kernel") == kernel and str(r.get("cell_id", "")) == "all":
            return r.get("verdict", ""), r
    return None, None


def fig_floyd_decay(out: Path, cells: list[dict], ex: dict, *, kernel: str = "floyd",
                    jump_ratio: float = 1.5) -> dict:
    """`fig_floyd_decay`: floyd's within-pass median `l0` over persistent pages (`l0_q50_per`)
    aligned at the pass boundary, reps overlaid; written only when `gates/gdec.csv` says
    `decay` for the exhibit kernel's `cell_id = all` row, else a placeholder PNG with the G-DEC
    verdict as text (SPEC 6.7; P2 Sec. V 5.2 G-DEC; K2 move 10). The boundaries are
    reconstructed for the drawing from the K jump (`K_t >= jump_ratio x cell median K`,
    al-Kindi review 1's detection ratio); this is a figure-only reconstruction recorded in
    `params`, never a gate."""
    import matplotlib.pyplot as plt
    verdict, row = _gdec_verdict(out, kernel)
    fd = _fig_dir(out)
    if verdict is None:
        return _placeholder(fd, "floyd_decay", not_run(f"gates/gdec.csv missing or has no cell_id = all row for {kernel}"))
    if str(verdict).split(" (")[0] != GDEC_DECAY:      # CHECK_2 M2: `decay (floor unmeasured)` is a decay whose idle clause is not run
        return _placeholder(fd, "floyd_decay", f"G-DEC ({kernel}): {verdict}")
    byk, _ = _by_kernel(cells)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    colors = _colors(plt.matplotlib)
    n_pass = 0
    max_len = 0
    for ci, c in enumerate(byk.get(kernel, [])):
        e = ex.get(c["cell_id"])
        if e is None:
            continue
        K = e.col("K")
        l0 = e.col("l0_q50_per")
        medK = np.nanmedian(K)
        b = np.where(K >= jump_ratio * medK)[0]
        if len(b) < 2:
            continue
        col = colors[ci % len(colors)]
        for s, t in zip(b[:-1], b[1:]):
            if t - s < 2:
                continue
            seg = l0[s:t]
            ax.plot(np.arange(len(seg)), seg, color=col, lw=0.6, alpha=0.5)
            n_pass += 1
            max_len = max(max_len, len(seg))
    ax.set_xlabel("snapshots since the pass boundary", fontsize=8)
    ax.set_ylabel("median l0 over persistent pages", fontsize=8)
    ax.set_title(f"{kernel}: within-pass decay, {n_pass} passes ({verdict})", fontsize=9)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    paths = _save(fig, fd, "floyd_decay")
    paths["n_passes_drawn"] = n_pass
    return paths


def fig_piano_roll(out: Path, cells: list[dict], ex: dict, *, piano_cell: str | None = None,
                   stride: int = 16) -> dict:
    """`fig_piano_roll`: one cell's changed-page raster (`seq` on x, `page_index` on y, one dot
    per row) beside its APF and J; the cell chosen by `--piano-cell` (default the first gemm
    cell in `cells.csv` order); this is the one figure that re-streams a trajectory (through
    `extract.open_text`) and subsamples rows by `--piano-stride` (SPEC 6.7; K2 move 8)."""
    import matplotlib.pyplot as plt
    fd = _fig_dir(out)
    oc = ok_cells(cells)
    cell = None
    if piano_cell:
        cell = next((c for c in oc if c["cell_id"] == piano_cell), None)
        if cell is None:
            return _placeholder(fd, "piano_roll", not_run(f"--piano-cell {piano_cell} not in cells.csv"))
    else:
        cell = next((c for c in oc if c.get("kernel") == "gemm"), None) or (oc[0] if oc else None)
    if cell is None:
        return _placeholder(fd, "piano_roll", not_run("cells.csv has no ok cell"))
    cdir = Path(cell.get("path", ""))
    traj = cdir / cell.get("traj_file", "") if cell.get("traj_file") else None
    if traj is None or not traj.exists():
        cands = sorted(cdir.glob("*substrate_trajectory.csv*")) if cdir.exists() else []
        traj = cands[0] if len(cands) == 1 else None
    if traj is None:
        return _placeholder(fd, "piano_roll", not_run(f"trajectory file for {cell['cell_id']} not found under {cdir}"))
    open_text = get_open_text()
    xs, ys = [], []
    n_rows = 0
    with open_text(str(traj)) as fh:
        rd = csv.reader(fh)
        header = next(rd, None)
        if not header or "seq" not in header or "page_index" not in header:
            return _placeholder(fd, "piano_roll", not_run(f"{traj.name} has no seq/page_index header"))
        i_seq, i_pg = header.index("seq"), header.index("page_index")
        for r in rd:
            n_rows += 1
            if (n_rows - 1) % stride:
                continue
            try:
                xs.append(int(r[i_seq]))
                ys.append(int(r[i_pg]))
            except (ValueError, IndexError):
                continue
    e = ex.get(cell["cell_id"])
    fig, axes = plt.subplots(3, 1, figsize=(8, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]})
    axes[0].scatter(xs, ys, s=0.5, color="k", linewidths=0, rasterized=True)
    axes[0].set_ylabel("page_index", fontsize=8)
    axes[0].set_title(f"{cell['cell_id']}: {traj.name} (stride {stride}, {len(xs)} of {n_rows} rows)", fontsize=8)
    if e is not None:
        axes[1].plot(e.col("seq"), e.col("K") / N_PAGES, color="C0", lw=0.7)
        axes[2].plot(e.col("seq"), e.col("J"), color="C1", lw=0.7)
    else:
        axes[1].text(0.5, 0.5, not_run("extract missing"), transform=axes[1].transAxes, ha="center", fontsize=8)
    axes[1].set_ylabel("APF", fontsize=8)
    axes[2].set_ylabel("J", fontsize=8)
    axes[2].set_xlabel("seq", fontsize=8)
    for ax in axes:
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    paths = _save(fig, fd, "piano_roll")
    paths["cell_id"] = cell["cell_id"]
    paths["n_rows_streamed"] = n_rows
    return paths


_CATEGORY = [  # (predicate, category index, label)
    (lambda s: s == "pass" or s.startswith("selected"), 0, "pass"),
    (lambda s: s == "fail", 1, "fail"),
    (lambda s: s.startswith("not applicable") or s.startswith("not run") or s == "undeclared", 2, "not applicable / not run"),
    (lambda s: s in ("order-blind", "order-blind (by construction)"), 3, "order-blind"),
    (lambda s: s == "resolution", 4, "resolution"),
    (lambda s: s.startswith("refused") or s in ("trend present", "undetermined by the interval calibration"), 5, "refused / named refusal"),
]


def _category(s: str) -> int:
    s = (s or "").strip()
    if s == "":
        return 6
    for pred, idx, _ in _CATEGORY:
        if pred(s):
            return idx
    return 6


def fig_table5_grid(out: Path, cells: list[dict], ex: dict) -> dict:
    """`fig_table5_grid`: the Table 5 verdict grid as a categorical heat map, one panel per rung
    (SPEC 6.7). Source: `report/tables/table5.csv` when it exists, else `gates/table5_grid.csv`."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    fd = _fig_dir(out)
    t5 = Path(out) / "report" / "tables" / "table5.csv"
    src = t5 if t5.exists() else Path(out) / "gates" / "table5_grid.csv"
    if not src.exists():
        return _placeholder(fd, "table5_grid", not_run("report/tables/table5.csv and gates/table5_grid.csv missing"))
    rows = read_csv(src)
    if src == t5:
        gate_cols = ["G1", "G2 (0.500 s)", "G2 (0.644 s)", "G2 (pairs)", "G4", "G5 (n non-overlapping windows)", "G-ORD"]
        key_rung, key_gp = "encoding", "grid_point (W x H)"
        disp = {RUNG_DISPLAY[r]: r for r in RUNGS}
    else:
        gate_cols = ["G1", "G2_0500", "G2_0644", "G2_pairs", "G4", "G5", "GORD"]
        key_rung, key_gp = "rung", "grid_id"
        disp = {r: r for r in RUNGS}
    by = defaultdict(dict)
    for r in rows:
        rung = disp.get(r.get(key_rung), r.get(key_rung))
        gp = r.get(key_gp)
        if src != t5:
            gp = grid_label(gp)
        by[rung][gp] = r
    cmap = ListedColormap(["#4daf4a", "#e41a1c", "#999999", "#377eb8", "#17becf", "#ff7f00", "#ffffff"])
    fig, axes = plt.subplots(1, len(RUNGS), figsize=(3.3 * len(RUNGS), 4.6), squeeze=False)
    gps = [grid_label(g) for g in GRID_IDS]
    for pi, rung in enumerate(RUNGS):
        ax = axes[0][pi]
        M = np.full((len(gps), len(gate_cols)), 6, dtype=int)
        for i, gp in enumerate(gps):
            r = by.get(rung, {}).get(gp)
            if r is None:
                continue
            for j, gc in enumerate(gate_cols):
                v = str(r.get(gc, ""))
                v = v.split(" (")[0] if gc.startswith("G5") or gc.startswith("G2 (pairs") else v
                M[i, j] = _category(v)
        ax.imshow(M, cmap=cmap, vmin=0, vmax=6, aspect="auto")
        ax.set_xticks(range(len(gate_cols)))
        ax.set_xticklabels([g.split(" (")[0] if "windows" in g else g for g in gate_cols], rotation=60, fontsize=6, ha="right")
        ax.set_yticks(range(len(gps)))
        ax.set_yticklabels([gp + ("  *" if str(by.get(rung, {}).get(gp, {}).get("selected", "")).startswith("selected") else "") for gp in gps], fontsize=6)
        ax.set_title(RUNG_DISPLAY[rung], fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(6)]
    fig.legend(handles, [c[2] for c in _CATEGORY], loc="lower center", ncol=6, fontsize=6, frameon=False)
    fig.suptitle("Table 5 verdicts per grid point (* = selected)", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    return _save(fig, fd, "table5_grid")


def fig_dhodapkar_sweep(out: Path, cells: list[dict], ex: dict, *, default=None) -> dict:
    """`fig_dhodapkar_sweep`: the Dhodapkar-Smith threshold sweep (SPEC_epoch2 Part 1.9; C14 cand.
    3: a phase change when delta = 1 - Jaccard exceeds a threshold, stability and average phase
    length; AA 2026-09-17: a declared sweep, every point kept, 0.04 the default). Source:
    `gates/comparators/dhodapkar_sweep.csv` (one row per (cell, delta_th)) and
    `dhodapkar.params.json` (the grid and the default that ran). One panel per kernel in
    `schema.KERNELS` order (a 13th panel `idle` when idle rows exist), x = delta_th (linear), left y
    = stability (solid, 0 to 1), right y = mean phase length in pairs (dashed, log), the reps of one
    kernel in the kernel's one colour (the convention of `fig_apf_per_kernel`), a vertical dotted
    line at the default labelled `declared default delta_th = <v>` (or `default set by
    --delta-th-default` when the value that ran is not the declared one), admissible cells only.
    Axis labels, the legend and the panel titles (the kernel names) are the only text. Without the
    sweep file the placeholder `not run: gates/comparators/dhodapkar_sweep.csv missing (move 14)` is
    written on the normal path, so `figures.json` `status` stays `ok`."""
    import matplotlib.pyplot as plt
    fd = _fig_dir(out)
    src = Path(out) / "gates" / "comparators" / "dhodapkar_sweep.csv"
    if not src.exists():
        w = _placeholder(fd, "dhodapkar_sweep", not_run("gates/comparators/dhodapkar_sweep.csv missing (move 14)"))
        return {**w, "n_panels": 0, "n_cells_drawn": 0, "default": default}
    rows = [r for r in read_csv(src) if str(r.get("admissible", "")).lower() != "false"]
    params = {}
    pp = Path(out) / "gates" / "comparators" / "dhodapkar.params.json"
    if pp.exists():
        try:
            params = read_json(pp).get("params") or {}
        except Exception:
            params = {}
    if default is None:
        default = params.get("default")
        if default is None:
            marked = [to_float(r.get("delta_th")) for r in rows if str(r.get("is_default", "")).lower() == "true"]
            default = marked[0] if marked else None
    declared = bool(params.get("default_is_declared", True))
    byk = defaultdict(lambda: defaultdict(list))            # kernel -> cell_id -> [(th, stability, mpl)]
    for r in rows:
        k = "idle" if r.get("role") == "idle" else r.get("kernel")
        th, st, mpl = to_float(r.get("delta_th")), to_float(r.get("stability")), to_float(r.get("mean_phase_length_pairs"))
        if th is None:
            continue
        byk[k][r.get("cell_id")].append((th, st, mpl))
    order = [k for k in KERNEL_NAMES if k in byk] + sorted(k for k in byk if k not in KERNEL_NAMES and k != "idle")
    panels = order + (["idle"] if "idle" in byk else [])
    if not panels:
        w = _placeholder(fd, "dhodapkar_sweep", not_run("gates/comparators/dhodapkar_sweep.csv has no admissible row"))
        return {**w, "n_panels": 0, "n_cells_drawn": 0, "default": default}
    ncol = 4
    nrow = max(1, math.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.5 * nrow), squeeze=False)
    colors = _colors(plt.matplotlib)
    n_cells = 0
    label_default = None
    if default is not None:
        label_default = (f"declared default delta_th = {default:g}" if declared else f"default set by --delta-th-default = {default:g}")
    for i, name in enumerate(panels):
        ax = axes[i // ncol][i % ncol]
        ax2 = ax.twinx()
        col = "0.5" if name == "idle" else colors[i % len(colors)]
        for cid, pts in byk[name].items():
            pts = sorted(pts)
            x = [t for t, _, _ in pts]
            ax.plot(x, [s if s is not None else np.nan for _, s, _ in pts], color=col, lw=0.8, alpha=0.7, ls="-")
            ax2.plot(x, [m if (m is not None and m > 0) else np.nan for _, _, m in pts], color=col, lw=0.8, alpha=0.7, ls="--")
            n_cells += 1
        if default is not None:
            ax.axvline(float(default), color="k", ls=":", lw=0.8, label=label_default)
        ax.set_ylim(0, 1.02)
        ax2.set_yscale("log")
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
        ax2.tick_params(labelsize=7)
    for j in range(len(panels), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    h_s = plt.Line2D([], [], color="k", ls="-", label="stability (left axis)")
    h_m = plt.Line2D([], [], color="k", ls="--", label="mean phase length, pairs (right axis, log)")
    handles = [h_s, h_m] + ([plt.Line2D([], [], color="k", ls=":", label=label_default)] if label_default else [])
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.supxlabel("delta_th (the declared grid)", fontsize=9)
    fig.supylabel("stability", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    w = _save(fig, fd, "dhodapkar_sweep")
    return {**w, "n_panels": len(panels), "n_cells_drawn": n_cells, "default": default}


# ----------------------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------------------
def run(out: Path, only: list[str] | None = None, *, piano_cell: str | None = None, piano_stride: int = 16,
        fused_plane_mask: str = "K", decay_jump_ratio: float = 1.5) -> dict:
    out = Path(out)
    names = list(only) if only else list(FIGURE_NAMES)
    unknown = [n for n in names if n not in FIGURE_NAMES]
    if unknown:
        raise ValueError(f"unknown figure name(s): {unknown}; known: {FIGURE_NAMES}")
    fd = _fig_dir(out)
    params = {"out": str(out), "only": names, "piano_cell": piano_cell, "piano_stride": piano_stride,
              "fused_plane_mask": fused_plane_mask, "decay_jump_ratio": decay_jump_ratio,
              "package_version": PACKAGE_VERSION, "written_at": now_iso()}
    mpl = _mpl()
    if mpl is None:
        (fd / "SKIPPED.txt").write_text("figures skipped: module matplotlib is not importable "
                                        "(pip install --user matplotlib); see RUNBOOK.md move 0\n", encoding="utf-8")
        write_json(fd / "figures.json", result_json("figures", params, CITATION,
                                                     {"status": not_run("matplotlib not importable"), "written": {}}))
        return {"skipped": fd / "SKIPPED.txt"}
    cells = load_cells(out)
    ex = _load_extracts(out, cells)
    written = {}
    errors = {}
    for n in names:
        try:
            if n == "apf_per_kernel":
                written[n] = fig_apf_per_kernel(out, cells, ex)
            elif n == "level_matched":
                written[n] = fig_level_matched(out, cells, ex)
            elif n == "fused_plane":
                written[n] = fig_fused_plane(out, cells, ex, mask_kind=fused_plane_mask)
            elif n == "j_hist":
                written[n] = fig_j_hist(out, cells, ex)
            elif n == "ratio_hist":
                written[n] = fig_ratio_hist(out, cells, ex)
            elif n == "floyd_decay":
                written[n] = fig_floyd_decay(out, cells, ex, jump_ratio=decay_jump_ratio)
            elif n == "piano_roll":
                written[n] = fig_piano_roll(out, cells, ex, piano_cell=piano_cell, stride=piano_stride)
            elif n == "table5_grid":
                written[n] = fig_table5_grid(out, cells, ex)
            elif n == "dhodapkar_sweep":
                written[n] = fig_dhodapkar_sweep(out, cells, ex)
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
    write_json(prev, result_json("figures", params, CITATION,
                                 {"status": "ok" if not errors else f"errors: {errors}", "n_extracts": len(ex),
                                  "written": prev_written, "errors": errors,
                                  "inputs_sha256": inputs_sha256([out / "cells.csv", out / "gates" / "gdec.csv",
                                                                  out / "gates" / "gj.json"])}))
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="plan11 figures (builder 3)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="comma-separated: " + ",".join(FIGURE_NAMES))
    ap.add_argument("--piano-cell", default=None)
    ap.add_argument("--piano-stride", type=int, default=16)
    ap.add_argument("--fused-plane-mask", default="K", choices=["K", "persist"])
    ap.add_argument("--decay-jump-ratio", type=float, default=1.5)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if not (out / "cells.csv").exists() and not os.environ.get("PLAN11_NO_MPL") and _mpl() is not None:
        print(f"missing input: {out / 'cells.csv'}", file=sys.stderr)
        return 2
    only = [s.strip() for s in a.only.split(",") if s.strip()] if a.only else None
    try:
        written = run(out, only, piano_cell=a.piano_cell, piano_stride=a.piano_stride,
                      fused_plane_mask=a.fused_plane_mask, decay_jump_ratio=a.decay_jump_ratio)
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
