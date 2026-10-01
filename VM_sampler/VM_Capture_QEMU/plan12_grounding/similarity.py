#!/usr/bin/env python3
"""similarity.py -- move 5 of plan12_grounding (SPEC move 5; section 5): how similar the runs are,
alike within a kernel and different between kernels, at both cuts.

  python3 -m plan12_grounding.similarity similarity --out O [--n-shuffles 1000] [--seed 20260930]

Per run and series (N, H, A), the three statistic sets of `stats.py`; then:
  1. per statistic, the share of the variation that lies between kernels: ICC(1), the one-way
     intraclass correlation with seeds nested in kernels (the unbalanced one-way form, Fisher's:
     MSB, MSW, n0), over the kernel runs, with a label-shuffle null (`--n-shuffles` shuffles of the
     kernel labels across runs; p = (1 + #null >= observed) / (1 + shuffles)); the point estimate
     with idle as a group beside it;
  2. a map of all runs from the standardized statistics (PCA, first two components), coloured by
     kernel with idle;
  3. the spectral-shape similarity (the council's test): per series, the correlation of level-free
     log spectra (Welch, 128-pair segments or the shortest run, on the standardized series, the
     log of the spectrum normalised to sum 1), same kernel against different kernels, with the same
     label-shuffle null;
  4. leave-one-seed-out: is each run nearest, in the standardized statistics, to its own kernel's
     other runs' centroid rather than to another group's?
Writes tables (CSV) and figures (SVG with the data as CSV) under `moves/05_similarity/cut<H>/` and a
`similarity.json`. Circular statistics are compared between kernels or against shuffles only, never
against a fixed pass mark (SPEC 5.2).
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import argparse  # noqa: E402
import html  # noqa: E402
import os  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
from scipy import signal  # noqa: E402

from plan12_grounding import __version__, toolkit_fingerprint  # noqa: E402
from plan12_grounding.run_moves import install_sigterm, now_iso, write_json  # noqa: E402
from plan12_grounding.stats import ORDER, SERIES, cut_series, cuts_of, load_runs, run_stats, stat_names  # noqa: E402
from plan12_grounding.figures import FONT, PALETTE, panel, svg_open, write_csv  # noqa: E402

CITATION = ("plan12_grounding/SPEC.md move 5 and section 5; the council's spectral test (grounding_paper/council/05_kindi_artifacts/spectra.py, "
            "04_dsp_artifacts/ensemble.py) and ICC(1) with seeds nested in kernels (04_dsp_engineer_how_to_prove.md P8)")
NPERSEG = 128
GROUP_COLOURS = {g: PALETTE[i % len(PALETTE)] if g != "idle" else "#9a9a9a" for i, g in enumerate(ORDER)}
GROUP_COLOURS.update({"stencil_jacobi": "#1f7a8c", "gemm": "#e07b00", "gibbs": "#5c9e31", "spmm": "#a23b72", "fem_assembly": "#8c564b",
                      "lexer": "#17becf", "rmat_gen": "#bcbd22", "bnb_tsp": "#7f7f7f", "floyd": "#1f77b4", "histogram": "#d62728",
                      "nbody": "#9467bd", "fft": "#ff7f0e"})


# ---------------------------------------------------------------------------------------------
# ICC(1), one way, seeds nested in kernels, unbalanced groups
# ---------------------------------------------------------------------------------------------
def icc1(x: np.ndarray, labels: np.ndarray) -> float:
    """One-way ICC(1) of the values `x` grouped by `labels`: (MSB - MSW) / (MSB + (n0 - 1) MSW) with
    MSB = SSB/(k-1), MSW = SSW/(N-k), n0 = (N - sum n_g^2 / N)/(k-1). NaN with fewer than two groups
    or no within-group degrees of freedom."""
    ok = np.isfinite(x)
    x, labels = x[ok], labels[ok]
    if x.size < 3:
        return float("nan")
    groups, inv, counts = np.unique(labels, return_inverse=True, return_counts=True)
    k, N = groups.size, x.size
    if k < 2 or N - k < 1:
        return float("nan")
    sums = np.bincount(inv, weights=x, minlength=k)
    means = sums / counts
    grand = x.mean()
    ssb = float((counts * (means - grand) ** 2).sum())
    sst = float(((x - grand) ** 2).sum())
    ssw = max(sst - ssb, 0.0)
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - (counts ** 2).sum() / N) / (k - 1)
    den = msb + (n0 - 1.0) * msw
    return float((msb - msw) / den) if den > 0 else float("nan")


def icc_table(X: np.ndarray, names: list[str], labels: np.ndarray, rng: np.random.Generator, n_shuffles: int) -> list[dict]:
    rows = []
    perms = [rng.permutation(labels.size) for _ in range(n_shuffles)]
    for j, name in enumerate(names):
        x = X[:, j]
        obs = icc1(x, labels)
        if not np.isfinite(obs):
            rows.append({"statistic": name, "icc": float("nan"), "n_runs": int(np.isfinite(x).sum()), "n_groups": int(np.unique(labels[np.isfinite(x)]).size),
                         "null_mean": float("nan"), "null_p95": float("nan"), "p_value": float("nan")})
            continue
        null = np.array([icc1(x, labels[p]) for p in perms])
        null = null[np.isfinite(null)]
        rows.append({"statistic": name, "icc": obs, "n_runs": int(np.isfinite(x).sum()), "n_groups": int(np.unique(labels[np.isfinite(x)]).size),
                     "null_mean": float(null.mean()) if null.size else float("nan"),
                     "null_p95": float(np.quantile(null, 0.95)) if null.size else float("nan"),
                     "p_value": float((1 + (null >= obs).sum()) / (1 + null.size)) if null.size else float("nan")})
    return rows


# ---------------------------------------------------------------------------------------------
# the standardized statistics, the PCA map, leave-one-seed-out
# ---------------------------------------------------------------------------------------------
def standardize(X: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str], list[dict]]:
    """z-scores over all runs; a statistic with any non-finite value or no spread is dropped and listed."""
    keep, dropped = [], []
    for j, n in enumerate(names):
        col = X[:, j]
        if not np.all(np.isfinite(col)):
            dropped.append({"statistic": n, "reason": f"{int((~np.isfinite(col)).sum())} non-finite values"})
        elif col.std() == 0:
            dropped.append({"statistic": n, "reason": "no spread across runs"})
        else:
            keep.append(j)
    Z = X[:, keep]
    Z = (Z - Z.mean(0)) / Z.std(0)
    return Z, [names[j] for j in keep], dropped


def pca2(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The first two principal components of the standardized matrix: scores, loadings (2 x p), explained shares."""
    if Z.shape[1] == 0 or Z.shape[0] < 2:
        return np.zeros((Z.shape[0], 2)), np.zeros((2, Z.shape[1])), np.zeros(2)
    U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    k = min(2, S.size)
    scores = np.zeros((Z.shape[0], 2)); scores[:, :k] = (U[:, :k] * S[:k])
    load = np.zeros((2, Z.shape[1])); load[:k] = Vt[:k]
    expl = np.zeros(2); expl[:k] = (S[:k] ** 2) / (S ** 2).sum()
    return scores, load, expl


def loso(Z: np.ndarray, labels: np.ndarray) -> list[dict]:
    """For each run: its group, the group whose centroid (of the other runs) is nearest, and the hit."""
    groups = list(dict.fromkeys(labels.tolist()))
    out = []
    for i in range(Z.shape[0]):
        best, best_d, own_d = None, float("inf"), float("nan")
        for g in groups:
            m = (labels == g)
            m[i] = False
            if not m.any():
                continue
            d = float(np.linalg.norm(Z[i] - Z[m].mean(0)))
            if g == labels[i]:
                own_d = d
            if d < best_d:
                best, best_d = g, d
        out.append({"group": labels[i], "nearest": best, "hit": bool(best == labels[i]), "d_own": own_d, "d_nearest": best_d})
    return out


# ---------------------------------------------------------------------------------------------
# the spectral shape
# ---------------------------------------------------------------------------------------------
def log_spectrum(x: np.ndarray, nperseg: int) -> np.ndarray:
    """The level-free log spectrum: the series standardized, Welch (Hann, half overlap, constant
    detrend), f = 0 dropped, the spectrum normalised to sum 1 and logged (the council's `log(P / P.sum())`)."""
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():                      # an undefined angle (no changed page, move 9) takes the series' finite mean
        fin = np.isfinite(x)
        x = np.where(fin, x, x[fin].mean() if fin.any() else 0.0)
    x = (x - x.mean()) / (x.std() + 1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, p = signal.welch(x, nperseg=nperseg, noverlap=nperseg // 2, window="hann", detrend="constant")
    p = p[1:]
    p = p / (p.sum() + 1e-300)
    return np.log(p + 1e-300)


def spectral_similarity(runs: list[dict], cut: int, series: str, labels: np.ndarray, rng: np.random.Generator, n_shuffles: int) -> dict:
    lens = [int(cut_series(r, cut)["pair"].size) for r in runs]
    nperseg = max(4, min(NPERSEG, min(lens)))
    L = np.array([log_spectrum(cut_series(r, cut)[series], nperseg) for r in runs])
    C = np.corrcoef(L) if L.shape[0] > 1 else np.ones((1, 1))
    C = np.nan_to_num(C, nan=0.0)
    off = ~np.eye(len(runs), dtype=bool)

    def within_between(lab):
        same = lab[:, None] == lab[None, :]
        w = C[same & off]; b = C[~same]
        return (float(w.mean()) if w.size else float("nan")), (float(b.mean()) if b.size else float("nan"))
    w, b = within_between(labels)
    obs = w - b
    null = np.array([np.subtract(*within_between(labels[rng.permutation(labels.size)])) for _ in range(n_shuffles)])
    null = null[np.isfinite(null)]
    per_group = []
    for g in dict.fromkeys(labels.tolist()):
        m = labels == g
        ww = C[np.ix_(m, m)][off[np.ix_(m, m)]]
        bb = C[np.ix_(m, ~m)]
        per_group.append({"group": g, "n_runs": int(m.sum()), "within_corr": float(ww.mean()) if ww.size else float("nan"),
                          "to_other_groups_corr": float(bb.mean()) if bb.size else float("nan")})
    return {"series": series, "nperseg": nperseg, "n_bins": int(L.shape[1]), "within_corr": w, "between_corr": b, "within_minus_between": obs,
            "null_mean": float(null.mean()) if null.size else float("nan"), "null_p95": float(np.quantile(null, 0.95)) if null.size else float("nan"),
            "p_value": float((1 + (null >= obs).sum()) / (1 + null.size)) if null.size else float("nan"),
            "corr": C, "per_group": per_group}


# ---------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------
def icc_bars_svg(rows: list[dict], cut: int) -> str:
    rows = [r for r in rows if np.isfinite(r["icc"])]
    rows.sort(key=lambda r: -r["icc"])
    rh, ml, bw = 12, 190, 520
    out = svg_open(ml + bw + 120, 50 + rh * len(rows) + 30, f"ICC(1) per statistic, seeds nested in kernels, cut of {cut} pairs",
                   "bar: the share of variation between kernels (0 to 1, negative values drawn at 0); tick: the label-shuffle null's 95th percentile; grey: not above the null")
    x0, y0 = ml, 46
    out.append(f'<line x1="{x0}" y1="{y0 - 4}" x2="{x0}" y2="{y0 + rh * len(rows)}" stroke="#999"/><line x1="{x0 + bw}" y1="{y0 - 4}" x2="{x0 + bw}" y2="{y0 + rh * len(rows)}" stroke="#ddd"/>')
    out.append(f'<text x="{x0}" y="{y0 - 8}">0</text><text x="{x0 + bw}" y="{y0 - 8}" text-anchor="end">1</text>')
    for i, r in enumerate(rows):
        y = y0 + i * rh
        v = max(0.0, min(1.0, r["icc"]))
        above = np.isfinite(r["null_p95"]) and r["icc"] > r["null_p95"]
        col = GROUP_COLOURS.get("floyd") if above else "#bbbbbb"
        out.append(f'<text x="{x0 - 4}" y="{y + 9}" text-anchor="end">{html.escape(r["statistic"])}</text>')
        out.append(f'<rect x="{x0}" y="{y + 1}" width="{bw * v:.1f}" height="{rh - 3}" fill="{col}"/>')
        if np.isfinite(r["null_p95"]):
            xt = x0 + bw * max(0.0, min(1.0, r["null_p95"]))
            out.append(f'<line x1="{xt:.1f}" y1="{y}" x2="{xt:.1f}" y2="{y + rh - 1}" stroke="#d9822b" stroke-width="1.5"/>')
        out.append(f'<text x="{x0 + bw + 6}" y="{y + 9}" fill="#555">{r["icc"]:.2f} (p {r["p_value"]:.3f})</text>')
    out.append("</svg>")
    return "\n".join(out)


def pca_svg(scores: np.ndarray, labels: np.ndarray, expl: np.ndarray, cut: int) -> str:
    w, h, ml, mt = 620, 520, 60, 50
    out = svg_open(w + ml + 220, h + mt + 40, f"The map of all runs: the first two principal components of the standardized statistics, cut of {cut} pairs",
                   f"PC1 explains {100 * expl[0]:.1f} percent of the variation, PC2 {100 * expl[1]:.1f}; one point per run, coloured by kernel, idle in grey")
    xlo, xhi = float(scores[:, 0].min()), float(scores[:, 0].max()); ylo, yhi = float(scores[:, 1].min()), float(scores[:, 1].max())
    padx, pady = 0.06 * (xhi - xlo or 1.0), 0.06 * (yhi - ylo or 1.0)
    xlo, xhi, ylo, yhi = xlo - padx, xhi + padx, ylo - pady, yhi + pady
    panel(out, ml, mt, w, h, xlo, xhi, ylo, yhi, [], xlabel="PC1")
    X = lambda x: ml + w * (x - xlo) / (xhi - xlo)          # noqa: E731
    Y = lambda v: mt + h - h * (v - ylo) / (yhi - ylo)      # noqa: E731
    for i in range(scores.shape[0]):
        g = labels[i]
        out.append(f'<circle cx="{X(scores[i, 0]):.1f}" cy="{Y(scores[i, 1]):.1f}" r="4" fill="{GROUP_COLOURS.get(g, "#333")}" fill-opacity="0.85" stroke="white" stroke-width="0.6"><title>{html.escape(g)}</title></circle>')
    for i, g in enumerate([g for g in ORDER if g in set(labels.tolist())]):
        out.append(f'<circle cx="{ml + w + 30}" cy="{mt + 14 + 16 * i}" r="4" fill="{GROUP_COLOURS.get(g, "#333")}"/><text x="{ml + w + 40}" y="{mt + 18 + 16 * i}">{html.escape(g)}</text>')
    out.append(f'<text x="{ml - 8}" y="{mt + h / 2:.0f}" text-anchor="end" fill="#666">PC2</text></svg>')
    return "\n".join(out)


def spectral_svg(res: list[dict], cut: int) -> str:
    out = svg_open(760, 60 + 60 * len(res) + 30, f"Spectral-shape similarity, cut of {cut} pairs",
                   "per series: the mean correlation of level-free log spectra within a kernel (dark) and between kernels (light); the text gives the label-shuffle null of the difference")
    x0, bw = 200, 400
    for i, r in enumerate(res):
        y = 50 + 60 * i
        out.append(f'<text x="10" y="{y + 12}" font-weight="bold">{html.escape(r["series"])}</text>')
        for j, (lab, v, col) in enumerate((("within", r["within_corr"], GROUP_COLOURS["floyd"]), ("between", r["between_corr"], "#bbbbbb"))):
            vv = 0.0 if not np.isfinite(v) else max(-1.0, min(1.0, v))
            out.append(f'<text x="{x0 - 6}" y="{y + 10 + 18 * j}" text-anchor="end">{lab}</text>'
                       f'<rect x="{x0 + bw / 2:.1f}" y="{y + 18 * j}" width="{abs(vv) * bw / 2:.1f}" height="12" fill="{col}" transform="{"scale(-1,1) translate(" + str(-2 * (x0 + bw / 2)) + ",0)" if vv < 0 else ""}"/>'
                       f'<text x="{x0 + bw + 8}" y="{y + 10 + 18 * j}" fill="#555">{v:.3f}</text>')
        out.append(f'<text x="{x0}" y="{y + 50}" fill="#555">within minus between {r["within_minus_between"]:.3f}; null mean {r["null_mean"]:.3f}, null p95 {r["null_p95"]:.3f}, p {r["p_value"]:.3f}; {r["n_bins"]} bins ({r["nperseg"]}-pair segments)</text>')
    out.append(f'<line x1="{x0 + bw / 2}" y1="44" x2="{x0 + bw / 2}" y2="{50 + 60 * len(res)}" stroke="#999"/><text x="{x0}" y="{50 + 60 * len(res) + 14}">-1</text><text x="{x0 + bw}" y="{50 + 60 * len(res) + 14}" text-anchor="end">1</text></svg>')
    return "\n".join(out)


def loso_svg(summary: list[dict], cut: int) -> str:
    x0, bw, rh = 130, 460, 16
    out = svg_open(x0 + bw + 200, 50 + rh * len(summary) + 30, f"Leave-one-seed-out: is each run nearest to its own kernel's other runs, cut of {cut} pairs",
                   "bar: the hit rate on all statistics; marks: on N (blue), H (grey), A (orange) alone")
    for i, r in enumerate(summary):
        y = 46 + i * rh
        out.append(f'<text x="{x0 - 6}" y="{y + 11}" text-anchor="end">{html.escape(r["group"])}</text>'
                   f'<rect x="{x0}" y="{y + 2}" width="{bw * r["rate_all"]:.1f}" height="{rh - 5}" fill="{GROUP_COLOURS.get(r["group"], "#333")}" fill-opacity="0.8"/>')
        for s, col in (("N", "#2b5d8a"), ("H", "#8a8a8a"), ("A", "#d9822b")):
            xt = x0 + bw * r[f"rate_{s}"]
            out.append(f'<line x1="{xt:.1f}" y1="{y + 1}" x2="{xt:.1f}" y2="{y + rh - 2}" stroke="{col}" stroke-width="2"/>')
        out.append(f'<text x="{x0 + bw + 8}" y="{y + 11}" fill="#555">{r["rate_all"]:.2f} of {r["n_runs"]} runs</text>')
    out.append(f'<text x="{x0}" y="{50 + rh * len(summary) + 12}">0</text><text x="{x0 + bw}" y="{50 + rh * len(summary) + 12}" text-anchor="end">1</text></svg>')
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------
# the move
# ---------------------------------------------------------------------------------------------
def one_cut(out: Path, cut_name: str, cut: int, runs: list[dict], n_shuffles: int, seed: int, moves_dir: Path | None = None) -> dict:
    d = (moves_dir or out / "moves") / "05_similarity" / f"cut{cut}"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    names = stat_names()
    labels = np.array([r["group"] for r in runs])
    X = np.array([[run_stats(r, cut)[n] for n in names] for r in runs], dtype=np.float64)
    write_csv(d / "stats_per_run.csv", ["cell_id", "group", "kernel", "role", "seed", "rep", "n_pairs_after_cut"] + names,
              [[r["cell_id"], r["group"], r["kernel"], r["role"], r["seed"], r["rep"], int(cut_series(r, cut)["pair"].size)] + X[i].tolist() for i, r in enumerate(runs)])
    # 1. ICC(1) over the kernel runs, the null; idle as a group beside it
    kern = labels != "idle"
    icc_rows = icc_table(X[kern], names, labels[kern], rng, n_shuffles)
    for j, r in enumerate(icc_rows):
        r["icc_with_idle"] = icc1(X[:, j], labels)
    write_csv(d / "icc.csv", ["statistic", "icc", "n_runs", "n_groups", "null_mean", "null_p95", "p_value", "icc_with_idle"],
              [[r[k] for k in ("statistic", "icc", "n_runs", "n_groups", "null_mean", "null_p95", "p_value", "icc_with_idle")] for r in icc_rows])
    (d / "icc_bars.svg").write_text(icc_bars_svg(icc_rows, cut))
    # 2. the map of all runs
    Z, znames, dropped = standardize(X, names)
    scores, load, expl = pca2(Z)
    write_csv(d / "pca_map.csv", ["cell_id", "group", "pc1", "pc2"], [[r["cell_id"], r["group"], scores[i, 0], scores[i, 1]] for i, r in enumerate(runs)])
    write_csv(d / "pca_loadings.csv", ["statistic", "pc1", "pc2"], [[n, load[0, j], load[1, j]] for j, n in enumerate(znames)])
    write_csv(d / "pca_dropped.csv", ["statistic", "reason"], [[x["statistic"], x["reason"]] for x in dropped])
    (d / "pca_map.svg").write_text(pca_svg(scores, labels, expl, cut))
    # 3. the spectral shape
    spec = [spectral_similarity(runs, cut, s, labels, rng, n_shuffles) for s in SERIES]
    for r in spec:
        write_csv(d / f"spectral_corr_{r['series']}.csv", ["cell_id"] + [x["cell_id"] for x in runs], [[runs[i]["cell_id"]] + r["corr"][i].tolist() for i in range(len(runs))])
        write_csv(d / f"spectral_by_group_{r['series']}.csv", ["group", "n_runs", "within_corr", "to_other_groups_corr"],
                  [[g["group"], g["n_runs"], g["within_corr"], g["to_other_groups_corr"]] for g in r["per_group"]])
    write_csv(d / "spectral_similarity.csv", ["series", "nperseg", "n_bins", "within_corr", "between_corr", "within_minus_between", "null_mean", "null_p95", "p_value"],
              [[r[k] for k in ("series", "nperseg", "n_bins", "within_corr", "between_corr", "within_minus_between", "null_mean", "null_p95", "p_value")] for r in spec])
    (d / "spectral_similarity.svg").write_text(spectral_svg(spec, cut))
    # 4. leave-one-seed-out, on all statistics and per series
    hits = {"all": loso(Z, labels)}
    for s in SERIES:
        cols = [j for j, n in enumerate(znames) if n.startswith(s + ".")]
        hits[s] = loso(Z[:, cols], labels) if cols else [{"group": g, "nearest": None, "hit": False, "d_own": float("nan"), "d_nearest": float("nan")} for g in labels]
    write_csv(d / "loso.csv", ["cell_id", "group"] + [f"{k}_{s}" for s in ("all",) + SERIES for k in ("nearest", "hit")],
              [[r["cell_id"], r["group"]] + [v for s in ("all",) + SERIES for v in (hits[s][i]["nearest"], int(hits[s][i]["hit"]))] for i, r in enumerate(runs)])
    summary = []
    for g in [g for g in ORDER if g in set(labels.tolist())]:
        idx = [i for i in range(len(runs)) if labels[i] == g]
        row = {"group": g, "n_runs": len(idx)}
        for s in ("all",) + SERIES:
            row[f"rate_{s}"] = float(np.mean([hits[s][i]["hit"] for i in idx])) if idx else float("nan")
        summary.append(row)
    overall = {s: float(np.mean([h["hit"] for h in hits[s]])) for s in ("all",) + SERIES}
    write_csv(d / "loso_summary.csv", ["group", "n_runs", "rate_all", "rate_N", "rate_H", "rate_A"], [[r["group"], r["n_runs"], r["rate_all"], r["rate_N"], r["rate_H"], r["rate_A"]] for r in summary])
    (d / "loso.svg").write_text(loso_svg(summary, cut))
    icc_finite = [r for r in icc_rows if np.isfinite(r["icc"])]
    res = {"cut": cut, "cut_name": cut_name, "n_runs": len(runs), "n_kernel_runs": int(kern.sum()), "n_statistics": len(names),
           "icc": {"n_finite": len(icc_finite), "median": float(np.median([r["icc"] for r in icc_finite])) if icc_finite else None,
                   "n_above_null_p95": int(sum(1 for r in icc_finite if np.isfinite(r["null_p95"]) and r["icc"] > r["null_p95"])),
                   "top5": [{"statistic": r["statistic"], "icc": round(r["icc"], 3), "p": r["p_value"]} for r in sorted(icc_finite, key=lambda r: -r["icc"])[:5]]},
           "pca": {"n_statistics_used": len(znames), "n_dropped": len(dropped), "explained": [float(e) for e in expl]},
           "spectral": [{k: r[k] for k in ("series", "nperseg", "n_bins", "within_corr", "between_corr", "within_minus_between", "null_p95", "p_value")} for r in spec],
           "loso_rate": overall, "loso_by_group": summary,
           "files": sorted(p.name for p in d.iterdir())}
    write_json(d / "summary.json", {"schema": "plan12.similarity_cut.v1", "citation": CITATION, **res})
    return res


def run(o: argparse.Namespace) -> int:
    out = Path(os.path.expanduser(o.out))
    cuts = cuts_of(out)
    if o.dry_run:
        print(f"[similarity] dry run: would compute the statistics, ICC(1) with {o.n_shuffles} shuffles, the PCA map, the spectral shape and leave-one-seed-out at cuts {cuts}")
        return 0
    runs = load_runs(out)
    if not runs:
        print("no runs with a complete series (run move 1 first)", file=sys.stderr)
        return 2
    run_similarity(out, out / "moves", runs, int(o.n_shuffles), int(o.seed), sys.argv)
    return 0


def run_similarity(out: Path, moves_dir: Path, runs: list[dict], n_shuffles: int, seed: int, argv: list[str]) -> list[dict]:
    """Move 5 at both cuts under `moves_dir` (the driver's `moves/`, or move 9's `moves/09_removed/`)."""
    cuts = cuts_of(out)
    results = [one_cut(out, name, cut, runs, n_shuffles, seed, moves_dir) for name, cut in cuts.items()]
    mdir = moves_dir / "05_similarity"
    write_json(mdir / "similarity.json", {"schema": "plan12.similarity.v1", "citation": CITATION, "package_version": __version__,
                                          "toolkit_fingerprint": toolkit_fingerprint()["sha256"], "command": argv, "written_at": now_iso(),
                                          "params": {"n_shuffles": n_shuffles, "seed": seed, "cuts": cuts, "nperseg": NPERSEG,
                                                     "circular_note": "circular statistics (A.nov.*) are compared between kernels or against shuffles only, never against a fixed pass mark (SPEC 5.2)"},
                                          "results": results})
    for r in results:
        print(f"[similarity] cut {r['cut']} ({r['cut_name']}): {r['n_runs']} runs, {r['n_statistics']} statistics; ICC median {r['icc']['median']}, "
              f"{r['icc']['n_above_null_p95']} above the null's p95; PCA explains {[round(100 * e, 1) for e in r['pca']['explained']]} percent; "
              f"leave-one-seed-out hit rate {r['loso_rate']['all']:.3f}; spectral within-between " + ", ".join(f"{s['series']} {s['within_minus_between']:.3f} (p {s['p_value']:.3f})" for s in r["spectral"]))
    return results


def main(argv: list[str] | None = None) -> int:
    install_sigterm()
    ap = argparse.ArgumentParser(prog="plan12_grounding.similarity", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("similarity", help="move 5: how similar the runs are")
    p.add_argument("--out", required=True)
    p.add_argument("--n-shuffles", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260930)
    p.add_argument("--dry-run", action="store_true")
    o = ap.parse_args(argv)
    try:
        return run(o)
    except FileNotFoundError as exc:
        print(f"missing input: {exc}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
