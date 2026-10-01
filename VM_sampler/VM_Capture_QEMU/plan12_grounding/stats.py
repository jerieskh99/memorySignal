#!/usr/bin/env python3
"""stats.py -- the statistic sets of plan12_grounding (SPEC move 5 and section 5), per run and per
window, on the per-pair series that move 1 writes (`series/<cell_id>.npz`: N_t, H_t, A_t).

Three sets, named `<series>.<set>.<statistic>` (SPEC 5.1):
  b1    plan08_b1/b1_features.py FEAT, imported: mean, std, cov, median, max, p95, peak2med, duty;
  deep  the console's deeper statistics, plan10_analysis/runner/stages.py deep(): skew, kurtosis,
        entropy, tau, n_boundaries, cepstral_peak_idx, ceps_peak_snr_db, through its own helpers
        (`_tau`, `_entropy`, `_ceps_peak`, `plan04_cusum.detect_boundaries_cusum`), imported;
  nov   the November extractor, utility_validation/featureExtractor.py lines 1 to 200, COPIED here
        with the defects of SPEC 5.3 fixed (each fix marked `FIX`): the magnitude features on H and
        N (mean, std, max, p99, p90, nnz_rate, sparsity, lag1, burstiness, gini, switch_rate, l1, l2)
        and the phase features on the angle A (cos_mean, sin_mean, resultant, one_minus_resultant,
        consistency_adjacent).
b1 and deep are computed on N, H and A; nov magnitude on H and N; nov phase on A.

Every statistic is computed on the series after the cut (`cut_series`: a cut of H pairs drops the
first H pairs in pair-index order, SPEC section 2; both cuts from params.json), and per window for
move 6 (`window_stats`). Circular statistics are never compared with a fixed pass mark (SPEC 5.2).

  python3 -m plan12_grounding.stats hand-check --out O   writes moves/05_similarity/hand_check.txt:
      the November features of one hand-made series next to the values computed by hand
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

from plan12_grounding import __version__  # noqa: E402
from plan12_grounding.run_moves import install_sigterm, now_iso, read_json, write_json  # noqa: E402
from plan10_analysis.runner import stages as _stages  # noqa: E402  (the console's deep() helpers; imported, never edited)
from b1_features import FEAT as B1_FEAT, features as _b1_features  # noqa: E402  (plan08_b1, on sys.path through stages)

CITATION = ("plan12_grounding/SPEC.md move 5 and section 5; plan08_b1/b1_features.py FEAT; plan10_analysis/runner/stages.py deep(); "
            "utility_validation/featureExtractor.py lines 1 to 200 (copied, defects of SPEC 5.3 fixed)")
SERIES = ("N", "H", "A")
DEEP_STATS = ("skew", "kurtosis", "entropy", "tau", "n_boundaries", "cepstral_peak_idx", "ceps_peak_snr_db")
NOV_MAG_STATS = ("mean", "std", "max", "p99", "p90", "nnz_rate", "sparsity", "lag1", "burstiness", "gini", "switch_rate", "l1", "l2")
NOV_PHASE_STATS = ("cos_mean", "sin_mean", "resultant", "one_minus_resultant", "consistency_adjacent")
EPS = 1e-8                      # the November extractor's epsilon
ORDER = ["floyd", "histogram", "nbody", "fft", "stencil_jacobi", "gemm", "gibbs", "spmm",
         "fem_assembly", "lexer", "rmat_gen", "bnb_tsp", "idle"]       # apf_paper/previews/make_shape_gallery.py ORDER


# ---------------------------------------------------------------------------------------------
# the runs: cells.csv + extract.json + series/<cell_id>.npz, and the cut
# ---------------------------------------------------------------------------------------------
def load_runs(out: Path, series_index: Path | None = None) -> list[dict]:
    """Every admissible recording with a complete series, in the gallery's kernel order then by rep:
    {cell_id, kernel, group ('idle' for idle cells), role, archetype (predicted, from cells.csv), campaign,
    rec_rel, seed, rep, arrays{pair, N, H, A, C, S, ...}, meta}. `series_index` names another record of
    series (move 9's `moves/09_removed/extract.json`) in place of move 1's."""
    out = Path(out)
    with open(out / "cells.csv", newline="") as fh:
        cells = {r["cell_id"]: r for r in csv.DictReader(fh)}
    ex = read_json(Path(series_index) if series_index else out / "moves" / "01_extract" / "extract.json")
    runs = []
    for cid, rec in (ex.get("recordings") or {}).items():
        if not str(rec.get("status", "")).startswith(("done", "reused")):
            continue
        c = cells.get(cid)
        if c is None or c.get("admissible", "").lower() != "true":
            continue
        with np.load(rec["series"], allow_pickle=False) as z:
            arrays = {k: np.asarray(z[k]) for k in z.files if k != "meta"}
            meta = json.loads(str(z["meta"]))
        group = "idle" if c["role"] == "idle" else c["kernel"]
        runs.append({"cell_id": cid, "kernel": c["kernel"], "group": group, "role": c["role"],
                     "archetype": c.get("archetype_predicted") or "", "campaign": c.get("campaign") or "", "rec_rel": c.get("rec_rel") or "",
                     "seed": (int(c["seed"]) if c.get("seed") not in (None, "") else None), "rep": int(c["rep"]) if c.get("rep") not in (None, "") else None,
                     "arrays": arrays, "meta": meta})
    order = {k: i for i, k in enumerate(ORDER)}
    runs.sort(key=lambda r: (order.get(r["group"], 99), r["rep"] if r["rep"] is not None else 999, r["cell_id"]))
    return runs


def cuts_of(out: Path) -> dict:
    """{'declared': 16, 'measured': 112} from params.json (SPEC section 2)."""
    p = read_json(Path(out) / "params.json")
    return {"declared": int(p["cuts"]["declared_pairs"]), "measured": int(p["cuts"]["measured_pairs"])}


def cut_series(run: dict, cut: int) -> dict:
    """The series after the cut: the first `cut` pairs AND the last pair dropped, in pair-index order,
    exactly the rows the encoding toolkit's rung_series drops (lo = head_drop, hi = n_rows - 1, so
    n_series = n_pairs - 1 - head_drop); SPEC section 2, the convention move 0 and move 1 record."""
    a = run["arrays"]
    n = int(a["pair"].size)
    k = min(max(int(cut), 0), n)
    hi = max(k, n - 1)
    return {key: v[k:hi] for key, v in a.items() if key != "hist_edges" and isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n}


# ---------------------------------------------------------------------------------------------
# set 1: B1's FEAT (imported)
# ---------------------------------------------------------------------------------------------
def b1_stats(x: np.ndarray) -> dict:
    xs = [float(v) for v in np.asarray(x, dtype=np.float64) if np.isfinite(v)]
    f = _b1_features(xs)
    return {k: float(f[k]) for k in B1_FEAT}


# ---------------------------------------------------------------------------------------------
# set 2: the console's deeper statistics (stages.deep(), through its helpers)
# ---------------------------------------------------------------------------------------------
def deep_stats(x: np.ndarray) -> dict:
    """skew, kurtosis, entropy, tau, n_boundaries, cepstral_peak_idx, ceps_peak_snr_db exactly as
    stages.deep() computes them per tile (its stat_pass_frac, f1_phase, coverage_ratio and cv_workingset
    are tile or family properties and are not statistics of a series; cv_workingset equals b1.cov)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    out = {k: float("nan") for k in DEEP_STATS}
    if x.size < 2:
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sd = x.std()
        out["skew"] = float(((x - x.mean()) ** 3).mean() / sd ** 3) if sd > 0 else 0.0
        out["kurtosis"] = float(((x - x.mean()) ** 4).mean() / sd ** 4 - 3.0) if sd > 0 else 0.0
        out["entropy"] = float(_stages._entropy(x))
        out["tau"] = float(_stages._tau(x))
        try:
            out["n_boundaries"] = float(len(_stages.plan04_cusum.detect_boundaries_cusum(x)))
        except Exception:                                  # noqa: BLE001  a series too short for the detector
            out["n_boundaries"] = float("nan")
        try:
            idx, snr = _stages._ceps_peak(x)
            out["cepstral_peak_idx"], out["ceps_peak_snr_db"] = float(idx), float(snr)
        except Exception:                                  # noqa: BLE001
            pass
    return out


# ---------------------------------------------------------------------------------------------
# set 3: the November extractor, copied for one series (a (T, 1) block of the original) and fixed
# ---------------------------------------------------------------------------------------------
def nov_lag1(x: np.ndarray, eps: float = EPS) -> float:
    """FeatureExtractor._lag1_autocorr: sum_t (x_t - mu)(x_{t+1} - mu) / (sum_t (x_t - mu)^2 + eps)."""
    if x.size < 2:
        return 0.0
    mu = x.mean()
    return float(((x[:-1] - mu) * (x[1:] - mu)).sum() / (((x - mu) ** 2).sum() + eps))


def nov_burstiness(x: np.ndarray, eps: float = EPS) -> float:
    """FeatureExtractor._extract_burstiness: var / (mean + eps) of the inter-arrival times of the
    active steps (x > eps); 0 with fewer than two active steps."""
    idx = np.flatnonzero(x > eps)
    if idx.size < 2:
        return 0.0
    iat = np.diff(idx)
    return float(iat.var() / (iat.mean() + eps))


def nov_gini(x: np.ndarray) -> float:
    """FeatureExtractor._extract_gini_index: ((2 i - T - 1) . sorted x) / (T sum x); 0 when sum x <= 0."""
    T = x.size
    s = float(x.sum())
    if T == 0 or s <= 0:
        return 0.0
    xs = np.sort(x)
    i = np.arange(1, T + 1)
    return float(((2 * i - T - 1) @ xs) / (T * s))


def nov_switch_rate(x: np.ndarray, eps: float = EPS) -> float:
    """FeatureExtractor._extract_flips_rate: the fraction of steps at which activity (x > eps) flips.
    FIX (SPEC 5.3, line 85): the original calls `X.shape()`; `shape` is a tuple, not callable, so
    the original raises on every call. Here the length is read from the array."""
    T = x.size                                             # FIX: was `T, B = X.shape()`
    if T < 2:
        return 0.0
    nz = x > eps
    return float((nz[1:] != nz[:-1]).mean())


def nov_magnitude(x: np.ndarray) -> dict:
    """FeatureExtractor.extract_mag_features on one series (the original's (T, B) block with B = 1)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {k: float("nan") for k in NOV_MAG_STATS}
    T = x.size
    sparsity = 1.0 - (np.count_nonzero(x) / T)
    return {
        "mean": float(np.mean(x)), "std": float(np.std(x)), "max": float(np.max(x)),
        "p99": float(np.percentile(x, 99)), "p90": float(np.percentile(x, 90)),
        "nnz_rate": float(1.0 - sparsity), "sparsity": float(sparsity),
        "lag1": nov_lag1(x), "burstiness": nov_burstiness(x), "gini": nov_gini(x), "switch_rate": nov_switch_rate(x),
        "l1": float(np.sum(x)), "l2": float(np.sqrt(np.sum(x ** 2))),
    }


def nov_phase(theta: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """FeatureExtractor.extract_phase_features on one angle series: the weighted circular mean's cos
    and sin parts, the resultant length R, 1 - R (the original's `phase_var`), and the adjacent
    consistency. `weights` (optional, one per step; normalised here) plays the original's active
    mask; None means uniform weights, the original's `use_mask = False` path.

    FIX (SPEC 5.3, line 182): the original's `diff_angles = np.angle(np.exp(1j * phase_t[1:][:-1]))`
    is not a difference (it drops the first and the last angle); here diff_angles = theta[1:] - theta[:-1].
    FIX (SPEC 5.3, line 188): the original uses `diff_angles` under `use_mask = False` without
    defining it; here it is defined on every path. The adjacent consistency is the mean of
    cos(theta_{t+1} - theta_t) over consecutive steps (SPEC 5.3); with weights, the weights of the
    later step of each pair, normalised (the original normalised by the sum of the angles, line 184)."""
    th = np.asarray(theta, dtype=np.float64)
    ok = np.isfinite(th)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        ok &= np.isfinite(w)
    th = th[ok]
    T = th.size
    if T == 0:
        return {k: float("nan") for k in NOV_PHASE_STATS}
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)[ok]
        w = w / (w.sum() + EPS)
    else:
        w = np.full(T, 1.0 / max(T, 1.0))
    c = float((np.cos(th) * w).sum())
    s = float((np.sin(th) * w).sum())
    R = float(math.sqrt(c * c + s * s))
    if T > 1:
        diff_angles = th[1:] - th[:-1]                     # FIX: a true difference of consecutive angles
        if weights is not None:
            wd = np.asarray(weights, dtype=np.float64)[ok][1:]
            wd = wd / (wd.sum() + EPS)                     # FIX: normalised by the weights' sum, not by the angles'
            consistency = float((np.cos(diff_angles) * wd).sum())
        else:
            consistency = float(np.cos(diff_angles).mean())
    else:
        consistency = 0.0
    return {"cos_mean": c, "sin_mean": s, "resultant": R, "one_minus_resultant": 1.0 - R, "consistency_adjacent": consistency}


# ---------------------------------------------------------------------------------------------
# per run, per window
# ---------------------------------------------------------------------------------------------
def series_stats(name: str, x: np.ndarray, theta_weights: np.ndarray | None = None) -> dict:
    """Every statistic of one series: `<name>.b1.*`, `<name>.deep.*`, and `<name>.nov.*` (the
    magnitude set for N and H, the phase set for A, on the angle in radians). A pair whose angle is
    undefined (no changed page: A = NaN, from move 9's removal) is left out of the statistics."""
    x = np.asarray(x, dtype=np.float64)
    fin = np.isfinite(x)
    if not fin.all():
        if theta_weights is not None:
            theta_weights = np.asarray(theta_weights, dtype=np.float64)[fin]
        x = x[fin]
    out = {}
    for k, v in b1_stats(x).items():
        out[f"{name}.b1.{k}"] = v
    for k, v in deep_stats(x).items():
        out[f"{name}.deep.{k}"] = v
    if name == "A":
        for k, v in nov_phase(x, theta_weights).items():
            out[f"{name}.nov.{k}"] = v
    else:
        for k, v in nov_magnitude(x).items():
            out[f"{name}.nov.{k}"] = v
    return out


def stat_names() -> list[str]:
    names = []
    for s in SERIES:
        names += [f"{s}.b1.{k}" for k in B1_FEAT] + [f"{s}.deep.{k}" for k in DEEP_STATS]
        names += [f"{s}.nov.{k}" for k in (NOV_PHASE_STATS if s == "A" else NOV_MAG_STATS)]
    return names


def run_stats(run: dict, cut: int) -> dict:
    """The statistics of one run after the cut, `<series>.<set>.<stat>` (SPEC 5.1)."""
    a = cut_series(run, cut)
    out = {}
    for s in SERIES:
        out.update(series_stats(s, a[s].astype(np.float64)))
    return out


def window_stats(arrays: dict, W: int, hop: int) -> dict:
    """The per-window entry point for move 6 (slice 3): windows of `W` pairs every `hop` pairs over
    an already-cut series dict (`cut_series`), the same statistics per window. Returns {names,
    rows (n_windows x n_stats), starts (the pair index each window begins at)}."""
    n = int(arrays["pair"].size)
    names = stat_names()
    rows, starts = [], []
    t = 0
    while t + W <= n:
        st = {}
        for s in SERIES:
            st.update(series_stats(s, arrays[s][t:t + W].astype(np.float64)))
        rows.append([st[k] for k in names])
        starts.append(int(arrays["pair"][t]))
        t += hop
    return {"names": names, "rows": np.asarray(rows, dtype=np.float64).reshape(len(rows), len(names)), "starts": np.asarray(starts, dtype=np.int64)}


# ---------------------------------------------------------------------------------------------
# the hand check (SPEC 8.1 / the builder prompt): one hand-made series, the module against arithmetic
# ---------------------------------------------------------------------------------------------
def hand_check_text() -> str:
    x = np.array([0.0, 3.0, 0.0, 5.0, 5.0, 0.0, 2.0])
    theta = np.array([0.0, math.pi / 4, math.pi / 2, math.pi / 4, 0.0, math.pi / 2, math.pi / 4])
    r2 = math.sqrt(2.0) / 2.0
    # by hand (each line is the arithmetic, written out; nothing here calls the module)
    hand_mag = {
        "mean": 15.0 / 7.0,
        "std": math.sqrt((0 + 9 + 0 + 25 + 25 + 0 + 4) / 7.0 - (15.0 / 7.0) ** 2),
        "max": 5.0,
        "p99": 5.0,                              # sorted 0,0,0,2,3,5,5: position 0.99 * 6 = 5.94 lies between 5 and 5
        "p90": 5.0,                              # position 0.9 * 6 = 5.4 lies between 5 and 5
        "nnz_rate": 4.0 / 7.0, "sparsity": 3.0 / 7.0,
        "lag1": (((0 - 15 / 7) * (3 - 15 / 7)) + ((3 - 15 / 7) * (0 - 15 / 7)) + ((0 - 15 / 7) * (5 - 15 / 7)) + ((5 - 15 / 7) * (5 - 15 / 7))
                 + ((5 - 15 / 7) * (0 - 15 / 7)) + ((0 - 15 / 7) * (2 - 15 / 7)))
                / (sum((v - 15 / 7) ** 2 for v in (0, 3, 0, 5, 5, 0, 2)) + EPS),
        "burstiness": (((2 - 5 / 3) ** 2 + (1 - 5 / 3) ** 2 + (2 - 5 / 3) ** 2) / 3.0) / (5.0 / 3.0 + EPS),   # active steps 1,3,4,6: gaps 2,1,2
        "gini": (-6 * 0 + -4 * 0 + -2 * 0 + 0 * 2 + 2 * 3 + 4 * 5 + 6 * 5) / (7.0 * 15.0),               # (2i - 8) . sorted x / (T sum x)
        "switch_rate": 5.0 / 6.0,                # activity F T F T T F T flips at 5 of the 6 steps
        "l1": 15.0, "l2": math.sqrt(0 + 9 + 0 + 25 + 25 + 0 + 4),
    }
    hand_phase = {
        "cos_mean": (1 + r2 + 0 + r2 + 1 + 0 + r2) / 7.0,
        "sin_mean": (0 + r2 + 1 + r2 + 0 + 1 + r2) / 7.0,
    }
    hand_phase["resultant"] = math.sqrt(hand_phase["cos_mean"] ** 2 + hand_phase["sin_mean"] ** 2)
    hand_phase["one_minus_resultant"] = 1.0 - hand_phase["resultant"]
    # consecutive differences pi/4, pi/4, -pi/4, -pi/4, pi/2, -pi/4: cosines r2, r2, r2, r2, 0, r2
    hand_phase["consistency_adjacent"] = (r2 + r2 + r2 + r2 + 0 + r2) / 6.0
    got_mag, got_phase = nov_magnitude(x), nov_phase(theta)
    lines = ["November features on one hand-made series (SPEC 8.1; the fixed copy in plan12_grounding/stats.py against arithmetic)",
             f"series x = {x.tolist()}", f"angles theta = [0, pi/4, pi/2, pi/4, 0, pi/2, pi/4] rad",
             f"{'statistic':<26} {'module':>14} {'by hand':>14}  agree"]
    ok_all = True
    for k in NOV_MAG_STATS:
        ok = abs(got_mag[k] - hand_mag[k]) <= 1e-9 * max(1.0, abs(hand_mag[k]))
        ok_all &= ok
        lines.append(f"{'H.nov.' + k:<26} {got_mag[k]:>14.9g} {hand_mag[k]:>14.9g}  {'yes' if ok else 'NO'}")
    for k in NOV_PHASE_STATS:
        ok = abs(got_phase[k] - hand_phase[k]) <= 1e-9
        ok_all &= ok
        lines.append(f"{'A.nov.' + k:<26} {got_phase[k]:>14.9g} {hand_phase[k]:>14.9g}  {'yes' if ok else 'NO'}")
    lines.append(f"all agree: {'yes' if ok_all else 'NO'}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    install_sigterm()
    ap = argparse.ArgumentParser(prog="plan12_grounding.stats", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("hand-check", help="the November features on a hand-made series, next to the values computed by hand")
    p.add_argument("--out", required=True)
    p.add_argument("--dry-run", action="store_true")
    o = ap.parse_args(argv)
    try:
        text = hand_check_text()
        print(text, end="")
        if o.dry_run:
            return 0
        out = Path(os.path.expanduser(o.out)) / "moves" / "05_similarity"
        out.mkdir(parents=True, exist_ok=True)
        (out / "hand_check.txt").write_text(text)
        write_json(out / "hand_check.json", {"schema": "plan12.hand_check.v1", "citation": CITATION, "package_version": __version__,
                                             "written_at": now_iso(), "agree": text.rstrip().endswith("yes")})
        return 0 if text.rstrip().endswith("yes") else 1
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
