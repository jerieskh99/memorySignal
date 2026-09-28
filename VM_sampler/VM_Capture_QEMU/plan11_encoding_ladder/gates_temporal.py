#!/usr/bin/env python3
"""gates_temporal.py -- G1 to G5 as amended, G3 as a per-kernel flag, G-ORD, the grid roll-up and
the selection rule (SPEC section 3.5).

Citation: P2_STRUCTURE.md section V 5.1 (Plan 03, G1-G5 as amended, G-ORD) and section 6 item 7
(G3 option (a)); CR 2.1 items 3 (G1 null), 4 (G2 in pair units), 5 (G3 flag), 6 (G4), 7 (G5
reported), 8 (Delta-5 not applied); CR 2.2 item 27 (G-ORD); ``plan03_metric_kernel.py``
(``stationarity_per_window``, the cepstral path); ``plan03_aggregate.py`` (``g4_pass``,
``_pick_winner``); SPEC_review_al_kindi.md items 3 (G3's null and the mirror index) and 7 (G2 in
pair units); SPEC_review_al_farabi.md items 2.1 (the roll-up of grid-independent refusals), 2.2
(feature artifacts at every grid point) and 2.11 (b) (G-ORD at the whole-cell point).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plan11_encoding_ladder import series as S
from plan11_encoding_ladder import splits as SP
from plan11_encoding_ladder import nulls as NL
from plan11_encoding_ladder import verdicts as V
from plan11_encoding_ladder.series import schema
from plan11_encoding_ladder.gates_calibration import load_pass_table, PassEntry, DT_BRACKET, DURATION_S

G1_Z = 1.0                       # plan03_metric_kernel.py stationarity z-threshold
G1_FLOOR = 0.80                  # plan03_aggregate.py G1_STATIONARITY_FLOOR
G1_TREND_DRIFT_SD = 1.0          # section 8 item 10
G1_N_SURROGATES = 200
G1_TREND_PRECEDENCE = "trend_first"
G2_COVERAGE_FLOOR = 2.0          # plan03_aggregate.py G2_COVERAGE_FLOOR
G3_MIN_CELLS = 7                 # section 8 item 11
G3_MIN_QUEF_FRAC = 0.125         # plan03_metric_kernel.py: max(1, n // 8)
G3_N_SURROGATES = 200
G3_NULL = "order_shuffle"        # SPEC_review_al_kindi.md item 3
G3_EPS = 1e-10                   # CepstrumStability eps
G5_MIN_WINDOWS = 5               # plan03_aggregate.py G5_N_WINDOWS_MIN
GORD_N_ORDER_PERM = 20           # section 8 item 12
GORD_NULL_PERM = 100
GRID_ROLLUP = "all_kernels"      # section 8 item 13
ROLLUP_KERNEL_REFUSALS = "not_applicable"   # SPEC_review_al_farabi.md item 2.1; alternative "blocks"
G1_NONE_APPLICABLE = "drop"      # SPEC_epoch2 B7 (CHECK_3 M7; CERT 3, 6.6, 7.4; Part 4 item 17): "drop" | "refuse"
DELTA5_NOTE = "not applied: no kernel counterpart (CR 2.1 item 8)"
CIT_GRID = ("P2 Sec. V 5.1 Plan 03 as amended (G1-G5); CR 2.1 items 3, 4, 6, 7, 8; plan03_metric_kernel.py "
            "stationarity_per_window; plan03_aggregate.py g4_pass; SPEC_review_al_kindi.md item 7; "
            "SPEC_review_al_farabi.md item 2.2")
CIT_G3 = ("P2 Sec. V 5.1 (G3 renamed a per-kernel signal flag) and Sec. 6 item 7 option (a); CR 2.1 item 5; "
          "plan03_metric_kernel.py score (cepstral path); SPEC_review_al_kindi.md item 3")
CIT_GORD = "P2 Sec. V 5.1 G-ORD; CR 2.2 item 27; SPEC_review_al_farabi.md item 2.11 (b)"
CIT_SELECT = ("P2 Sec. V (the binding condition: grid declared, every point kept, rule declared, selected point marked); "
              "plan03_aggregate.py _pick_winner re-pointed; SPEC 3.5.7; SPEC_review_al_farabi.md items 2.1, 2.2, 2.11 (c)")

PER_KERNEL_COLUMNS = ("rung", "grid_id", "W", "H", "hop_ratio", "kernel", "n_cells", "n_windows_median", "n_windows_min",
                      "n_windows_nonoverlap_median", "stat_pass_frac_median", "g1_surrogate_p05", "g1_trend_cells", "G1",
                      "coverage_0500", "coverage_0644", "coverage_pairs", "G2_0500", "G2_0644", "G2_pairs", "G2", "G4", "G5",
                      "gord_score_ordered", "gord_score_shuffled_mean", "gord_null_spread", "GORD")
GRID_COLUMNS = ("rung", "axis", "grid_id", "W", "H", "hop_ratio", "n_windows_median", "n_windows_nonoverlap_median", "G1",
                "G2_0500", "G2_0644", "G2", "coverage_pairs", "G2_pairs", "G4", "G5", "GORD", "n_kernels_na_G1", "n_kernels_na_G2", "n_kernels_undeclared_G2",
                "gates_passed", "selected", "selected_by", "refusal", "gc_verdict")   # gf_part1 removed (SPEC_epoch2 B15; CERT 1(b), 7.2): G-F lives in gates/gf.csv


# --------------------------------------------------------------------------- G1 (3.5.1)

def stationarity_per_window(traj_1d, window: int, hop: int):
    # copied from plan03_metric_kernel.py:stationarity_per_window, 2026-09-16 (verbatim)
    """Per-window stationarity pass fraction (1-D APF path).

    A window passes if ``abs(window_mean - global_mean) / global_std < 1.0``.
    If the global std is zero (constant trajectory) every window passes
    trivially. If fewer than one window exists the function returns None
    so the caller can distinguish "no test was run" from "all failed".
    """
    n = len(traj_1d)
    if n < window:
        return None
    n_windows = max(0, (n - window) // hop + 1)
    if n_windows < 1:
        return None
    g_mean = statistics.fmean(traj_1d)
    g_std = statistics.pstdev(traj_1d) if n >= 2 else 0.0
    if g_std == 0.0:
        return 1.0
    passes = 0
    for i in range(n_windows):
        start = i * hop
        chunk = traj_1d[start:start + window]
        if not chunk:
            continue
        w_mean = statistics.fmean(chunk)
        if abs(w_mean - g_mean) / g_std < 1.0:
            passes += 1
    return passes / n_windows


def stationarity_batch(X: np.ndarray, W: int, H: int) -> np.ndarray:
    """``stationarity_per_window`` for every row of ``X`` [n_series, n] at once (the same rule: a window
    passes when |window mean - global mean| / global population std < 1.0; a constant row passes 1.0).
    Used for the surrogate batch; equal to the verbatim copy (tested)."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[1]
    nw = S.n_windows(n, W, H)
    if nw < 1:
        return np.full(X.shape[0], np.nan)
    gm = X.mean(axis=1, keepdims=True)
    gs = X.std(axis=1)
    idx = np.arange(nw)[:, None] * H + np.arange(W)[None, :]
    wm = X[:, idx].mean(axis=2)                              # [rows, nw]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.abs(wm - gm) / gs[:, None]
    frac = (z < G1_Z).sum(axis=1) / nw
    frac[gs == 0.0] = 1.0
    return frac


def trend_drift_sd(x: np.ndarray) -> float:
    """The least-squares slope of x on its index times (n - 1), in units of the population std of x
    (SPEC 3.5.1; section 8 item 10)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return 0.0
    sd = x.std()
    if sd == 0.0:
        return 0.0
    t = np.arange(n, dtype=np.float64)
    slope = np.polyfit(t, x, 1)[0]
    return float(slope * (n - 1) / sd)


def g1_cell(x: np.ndarray, sur: np.ndarray, W: int, H: int, *, trend_sd: float = G1_TREND_DRIFT_SD) -> dict:
    """One cell at one grid point: pf_obs = stationarity_per_window(x, W, H); pf_sur the same
    statistic on the phase-randomized surrogates (nulls.surrogates); trend = |drift| > trend_sd
    (P2 Sec. V 5.1 G1 as amended; CR 2.1 item 3)."""
    x = np.asarray(x, dtype=np.float64)
    pf = stationarity_per_window(x.tolist(), W, H)
    pfs = stationarity_batch(sur, W, H) if len(sur) else np.zeros(0)
    d = trend_drift_sd(x)
    return {"pf_obs": pf, "pf_sur": pfs, "drift_sd": d, "trend": bool(abs(d) > trend_sd)}


def g1_kernel(cell_results: list[dict]) -> tuple:
    """G1 per kernel (P2 Sec. V 5.1 Plan 03 as amended; CR 2.1 item 3; CIT_GRID). Pass iff the
    median over cells of pf_obs >= 0.80 and not below the median over cells of the surrogates'
    5th percentile; TREND_PRESENT when more than half the kernel's cells have a trend (trend
    first; the cell is handed to the whole-cell reading); else fail.
    Returns (verdict, median pf_obs, median surrogate p05, n_trend)."""
    pfs = [r["pf_obs"] for r in cell_results if r["pf_obs"] is not None]
    p05 = [float(np.quantile(r["pf_sur"], 0.05)) for r in cell_results if len(r["pf_sur"]) and not np.all(np.isnan(r["pf_sur"]))]
    n_trend = sum(1 for r in cell_results if r["trend"])
    med = float(np.median(pfs)) if pfs else None
    sp05 = float(np.median(p05)) if p05 else None
    if not cell_results:
        return V.not_run("no cell"), med, sp05, n_trend
    if n_trend * 2 > len(cell_results):
        return V.TREND_PRESENT, med, sp05, n_trend
    if med is None:
        return V.not_run("no window at this point"), med, sp05, n_trend
    ok = med >= G1_FLOOR and (sp05 is None or med >= sp05)
    return (V.PASS if ok else V.FAIL), med, sp05, n_trend


# --------------------------------------------------------------------------- G2 (3.5.2)

def g2_kernel(entry: PassEntry, W_cells: list[int], n_pairs_cells: list[int], *, dt_bracket=DT_BRACKET,
              duration_s_cells: list[float] | None = None) -> dict:
    """G2, spectral coverage (CR 2.1 item 4; P2 Sec. V 5.1). Seconds: coverage = W dt / T_seconds at
    both bracket ends, floor 2.0; no count -> GP_UNDECLARED; T < 2 dt -> G2_ABOVE_NYQUIST at that dt.
    Pairs (SPEC_review_al_kindi.md item 7): per cell coverage_pairs = W * passes / n_pairs, the
    kernel's median, G2_pairs by the same floor (T_pairs < 2 -> above Nyquist). G2 = the common
    seconds verdict when the two agree, else G2_UNDETERMINED. An inferred count carries ' (INFERRED)'.
    ``T_seconds = duration / passes`` with ``duration`` the median over the kernel's cells of
    ``duration_s_cells`` (each cell's declared duration, ``sidecar duration_s_declared``; SPEC_epoch2
    B12; CHECK_3 M12); without it the constant ``schema.DURATION_S`` (600 s), which is what every
    sidecar of the real corpus declares, so no number moves there. The pass table's count is per cell
    duration (``passes_per_600s`` keeps its name; 600 s on the real corpus, the declared
    ``--duration-s`` on a synthetic one)."""
    if entry.passes is None:
        return {"coverage_0500": None, "coverage_0644": None, "coverage_pairs": None, "G2_0500": V.GP_UNDECLARED,
                "G2_0644": V.GP_UNDECLARED, "G2_pairs": V.GP_UNDECLARED, "G2": V.GP_UNDECLARED}
    sfx = " (INFERRED)" if entry.source_kind == "inferred" else ""
    duration = float(np.median(duration_s_cells)) if duration_s_cells else float(DURATION_S)
    T = duration / entry.passes
    Wmed = float(np.median(W_cells))
    res = {}
    for key, dt in (("0500", dt_bracket[0]), ("0644", dt_bracket[1])):
        cov = Wmed * dt / T
        res[f"coverage_{key}"] = cov
        if T < 2 * dt:
            res[f"G2_{key}"] = V.G2_ABOVE_NYQUIST + sfx
        else:
            res[f"G2_{key}"] = (V.PASS if cov >= G2_COVERAGE_FLOOR else V.FAIL) + sfx
    covp = [w * entry.passes / n for w, n in zip(W_cells, n_pairs_cells)]
    tp = [n / entry.passes for n in n_pairs_cells]
    res["coverage_pairs"] = float(np.median(covp)) if covp else None
    if covp and float(np.median(tp)) < 2.0:
        res["G2_pairs"] = V.G2_ABOVE_NYQUIST + sfx
    else:
        res["G2_pairs"] = ((V.PASS if res["coverage_pairs"] >= G2_COVERAGE_FLOOR else V.FAIL) + sfx) if covp else V.not_run("no cell")
    res["G2"] = res["G2_0500"] if res["G2_0500"] == res["G2_0644"] else V.G2_UNDETERMINED + sfx
    return res


# --------------------------------------------------------------------------- G3 (3.5.3)

def cepstrum(x: np.ndarray) -> np.ndarray:
    """``irfft(log(|rfft(x)| + 1e-10))`` as ``CepstrumStability.compute_cepstrum`` (plan03's path)."""
    x = np.asarray(x, dtype=np.float64)
    return np.fft.irfft(np.log(np.abs(np.fft.rfft(x)) + G3_EPS))


def cepstral_peak(x: np.ndarray, min_quef_frac: float = G3_MIN_QUEF_FRAC) -> tuple:
    """(peak_idx, snr_db, min_quef): the peak of |c[q]| for q in [max(1, int(n * min_quef_frac)),
    len(c) // 2] (the search stops at the half so the mirror index n - P is not returned,
    SPEC_review_al_kindi.md item 3 (a)); snr_db = 10 log10(peak / median(|c[min_quef:]|)) with the
    median over the whole tail as plan03_metric_kernel.py:score does."""
    n = len(x)
    c = np.abs(cepstrum(x))
    min_q = max(1, int(n * min_quef_frac))
    half = len(c) // 2
    if min_q >= half or min_q >= len(c):
        return None, None, min_q
    seg = c[min_q:half + 1]
    peak = int(np.argmax(seg)) + min_q
    tail = c[min_q:]
    med = float(np.median(tail)) if tail.size else 0.0
    pv = float(c[peak])
    snr = 10.0 * math.log10(pv / med) if (med > 0.0 and pv > 0.0) else None
    return peak, snr, min_q


def g3_cell(x: np.ndarray, K_median: float, *, n_surrogates: int = G3_N_SURROGATES, seed: int = NL.SEED_ORDER,
            min_quef_frac: float = G3_MIN_QUEF_FRAC, null: str = G3_NULL) -> dict:
    """G3 per cell: the cepstral peak and SNR; the null is ``n_surrogates`` order shuffles of the series
    (``g3_null = "order_shuffle"``: the iid null that keeps the marginal distribution and destroys
    rhythm; a phase-randomized surrogate preserves the amplitude spectrum and so the cepstrum exactly,
    SPEC_review_al_kindi.md item 3); flag_cell = G3_PRESENT if snr_db > null p95 (strict). cv =
    population std / mean of the whole series (plan02 cv_workingset, re-implemented with the
    population std as SPEC 3.5.3 states), beside the shot-noise floor 1 / sqrt(K_median_cell); no CV
    verdict (section 8 item 11)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    n = len(x)
    peak, snr, min_q = cepstral_peak(x, min_quef_frac)
    rng = np.random.default_rng(seed)
    null_snr = []
    if peak is not None and n_surrogates > 0:
        for _ in range(n_surrogates):
            if null == "order_shuffle":
                xs = NL.order_shuffle(x, rng)
            elif null == "phase_randomize":
                xs = NL.phase_randomize(x, rng)
            else:
                raise ValueError(null)
            _, s2, _ = cepstral_peak(xs, min_quef_frac)
            null_snr.append(s2 if s2 is not None else np.nan)
    null_snr = np.array([v for v in null_snr if not np.isnan(v)], dtype=np.float64)
    p95 = float(np.quantile(null_snr, 0.95)) if len(null_snr) else None
    present = bool(snr is not None and p95 is not None and snr > p95)
    mean = float(x.mean()) if n else 0.0
    cv = float(x.std() / mean) if (n >= 2 and mean != 0) else None
    floor = 1.0 / math.sqrt(K_median) if K_median and K_median > 0 else None
    return {"ceps_peak_idx": peak, "ceps_peak_freq_cyc_per_pair": (1.0 / peak) if peak else None,
            "ceps_snr_db": snr, "snr_surrogate_p95": p95, "cv": cv, "cv_shot_floor": floor,
            "cv_ratio": (cv / floor) if (cv is not None and floor) else None,
            "flag_cell": V.G3_PRESENT if present else V.G3_ABSENT, "quefrency_floor": min_q, "n": n}


# --------------------------------------------------------------------------- G4, G5

def g4_pass(window: int, hop: int) -> bool:
    """G4, hop at most half the window (P2 Sec. V 5.1 Plan 03 as amended; CR 2.1 item 6; CIT_GRID)."""
    # copied from plan03_aggregate.py:g4_pass, 2026-09-16
    return hop * 2 <= window


# --------------------------------------------------------------------------- the grid stage (3.5, per point)

def _cells_for(out: Path, rung: str) -> tuple:
    cells = S.load_cells(out / "cells.csv")
    kept, ex_hard, ex_pair, _ = S.admissible_cells(out, cells, rung)
    return kept, ex_hard, ex_pair


def _kernel_order(cells) -> list[str]:
    ks = [k for k, _ in schema.KERNELS if any(c["kernel"] == k for c in cells)]
    ks += sorted({c["kernel"] for c in cells if c["role"] == "kernel"} - set(ks))
    if any(c["role"] == "idle" for c in cells):
        ks.append("idle")
    return ks


def _grid_cell_stats(x: np.ndarray, n_surrogates: int, seed: int, points: list, trend_sd: float) -> dict:
    """One cell's G1 statistics at every grid point (SPEC 3.5.1): the phase-randomized surrogates
    (``nulls.surrogates`` from the cell's own seeded generator, so the draw does not depend on the
    order or the number of workers) and ``g1_cell`` per point. A module-level function so that
    ``gate_grid`` can score the cells through ``joblib.Parallel`` (SPEC_epoch2 B6; CHECK_3 M6):
    results are identical at any ``n_jobs``."""
    sur = NL.surrogates(x, n_surrogates, seed) if n_surrogates > 0 and len(x) >= 3 else np.zeros((0, len(x)))
    ns = len(x)
    res = {}
    for gid, W, H in points:
        w, h = S.resolve_wh(W, H, ns)
        r = g1_cell(x, sur, w, h, trend_sd=trend_sd)
        res[gid] = {"r": r, "nw": S.n_windows(ns, w, h), "nwn": ns // w if w else 0, "w": w}
    return res


def gate_grid(out: Path, rung: str, *, n_surrogates: int = G1_N_SURROGATES, trend_drift_sd: float = G1_TREND_DRIFT_SD,
              n_jobs: int = 1, seed_offset: int = 0, build_features: bool = True,
              wapf_norm: str = S.WAPF_NORM_DEFAULT) -> list[Path]:
    """Every grid point of one rung (SPEC 3.5): per kernel G1 (with its surrogate null), G2 (seconds
    at both dt and in pair units), G4, G5 (reported) and the window counts, written to
    ``gates/grid/<rung>/<grid_id>/temporal_per_kernel.csv`` with ``g1_surrogates.npz`` beside it;
    and the feature matrices of every point (``series.build_all_grid``, raw and norm) so that
    selection is a choice among artifacts on disk (SPEC_review_al_farabi.md item 2.2). Temporal gates
    run on the level-normalized single-channel series (``params.temporal_channel``); GORD stays
    ``pending`` until ``gate_gord`` fills it.

    Build epoch 2: ``n_jobs`` is honoured (SPEC_epoch2 B6; CHECK_3 M6; E1 6.28): the per-cell G1
    surrogate statistics are scored through ``joblib.Parallel`` (``_grid_cell_stats``, one job per
    cell, every surrogate drawn from the cell's own seeded generator), so the output is identical at
    any job count. ``wapf_norm`` (SPEC_epoch2 B19; CERT 5, 6.13, 7.6; SPEC section 8 item 7) is passed
    to ``series.build_all_grid`` and to the temporal series (``median_K``, the count-rung rule, or
    ``median_self``) and recorded in ``temporal.params.json``. G2's seconds coverage reads each
    cell's declared duration from its sidecar (``duration_s_declared``; SPEC_epoch2 B12; CHECK_3 M12):
    600 on the real corpus, the extractor's ``--duration-s`` on a synthetic one; a sidecar without the
    key falls back to ``schema.DURATION_S`` and is counted in ``params.duration_s_fallback_cells``.
    The idle cells' head drop is the ``idle`` row of ``head_drop.csv`` (B9)."""
    out = Path(out)
    cells, ex_hard, ex_pair = _cells_for(out, rung)
    hd = S.load_head_drop(out / "inputs" / "head_drop.csv")
    pt = load_pass_table(out / "inputs" / "pass_table.csv")
    if build_features:
        S.build_all_grid(out, None, rung, hd, wapf_norm=wapf_norm)
    # series once per cell; the surrogates and the per-point G1 statistics in the (parallel) worker
    per_cell, fallback_cells = {}, []
    for c in cells:
        ex = S.load_extract_cached(out, c["cell_id"])
        x = S.temporal_series(ex, rung, S.head_drop_for(hd, c["kernel"], c["role"]), wapf_norm=wapf_norm)
        xs = x.copy()
        if np.any(np.isnan(xs)):
            xs[np.isnan(xs)] = float(np.nanmean(xs)) if np.any(~np.isnan(xs)) else 0.0
        sc = S.load_sidecar(out, c["cell_id"])
        dur = sc.get("duration_s_declared")
        if dur is None:
            dur = DURATION_S; fallback_cells.append(c["cell_id"])
        per_cell[c["cell_id"]] = {"x": xs, "n_pairs": int(sc["n_pairs"]), "duration_s": float(dur), "kernel": c["kernel"], "role": c["role"]}
    points = S.grid_points_ids()
    seed = NL.SEED_SURROGATE + seed_offset
    cids = list(per_cell)
    n_jobs = int(n_jobs) if n_jobs else 1
    if n_jobs > 1 and len(cids) > 1:
        from joblib import Parallel, delayed
        stats_list = Parallel(n_jobs=n_jobs)(delayed(_grid_cell_stats)(per_cell[cid]["x"], n_surrogates, seed, points, trend_drift_sd) for cid in cids)
    else:
        stats_list = [_grid_cell_stats(per_cell[cid]["x"], n_surrogates, seed, points, trend_drift_sd) for cid in cids]
    stats = dict(zip(cids, stats_list))
    kernels = _kernel_order(cells)
    written = []
    for gid, W, H in points:
        d = out / "gates" / "grid" / rung / gid
        d.mkdir(parents=True, exist_ok=True)
        rows = []
        npz = {"cell_id": [], "pf_obs": [], "pf_sur": []}
        dur_per_kernel = {}
        for k in kernels:
            kc = [cid for cid, v in per_cell.items() if (v["kernel"] == k and v["role"] == "kernel") or (k == "idle" and v["role"] == "idle")]
            res, nws, nwn, Ws = [], [], [], []
            for cid in kc:
                st = stats[cid][gid]
                r = st["r"]
                res.append(r)
                nws.append(st["nw"]); nwn.append(st["nwn"]); Ws.append(st["w"])
                npz["cell_id"].append(cid); npz["pf_obs"].append(np.nan if r["pf_obs"] is None else r["pf_obs"])
                npz["pf_sur"].append(r["pf_sur"] if len(r["pf_sur"]) else np.full(n_surrogates, np.nan))
            G1, med, sp05, ntr = g1_kernel(res)
            entry = pt.get(k, PassEntry(None, "undeclared")) if k != "idle" else PassEntry(None, "undeclared")
            durs = [per_cell[c]["duration_s"] for c in kc]
            g2 = g2_kernel(entry, Ws, [per_cell[c]["n_pairs"] for c in kc], duration_s_cells=durs) if kc else g2_kernel(PassEntry(None, "undeclared"), [], [])
            if durs:
                dur_per_kernel[k] = {"median": float(np.median(durs)), "min": float(min(durs)), "max": float(max(durs))}
            w_rep, h_rep = (W, H) if W is not None else (int(np.median(Ws)) if Ws else None, int(np.median(Ws)) if Ws else None)
            rows.append({"rung": rung, "grid_id": gid, "W": w_rep, "H": h_rep, "hop_ratio": S.hop_ratio(W, H) if W else 1.0,
                         "kernel": k, "n_cells": len(kc), "n_windows_median": float(np.median(nws)) if nws else None,
                         "n_windows_min": int(min(nws)) if nws else None,
                         "n_windows_nonoverlap_median": float(np.median(nwn)) if nwn else None,
                         "stat_pass_frac_median": med, "g1_surrogate_p05": sp05, "g1_trend_cells": ntr, "G1": G1, **g2,
                         "G4": V.PASS if (W is not None and g4_pass(W, H)) else V.FAIL,
                         "G5": (V.PASS if (nws and min(nws) >= G5_MIN_WINDOWS) else V.FAIL),
                         "gord_score_ordered": None, "gord_score_shuffled_mean": None, "gord_null_spread": None,
                         "GORD": V.GORD_ORDER_BLIND_BY_CONSTRUCTION if W is None else V.pending("gate_gord")})
        S.write_csv(d / "temporal_per_kernel.csv", PER_KERNEL_COLUMNS, rows)
        np.savez(d / "g1_surrogates.npz", cell_id=np.array(npz["cell_id"]), pf_obs=np.array(npz["pf_obs"], dtype=np.float64),
                 pf_sur=np.array(npz["pf_sur"], dtype=np.float64) if npz["pf_sur"] else np.zeros((0, n_surrogates)))
        params = {"rung": rung, "grid_id": gid, "W": W, "H": H, "n_surrogates": n_surrogates, "seed_surrogate": seed,
                  "n_jobs": n_jobs, "parallel": "joblib over cells (SPEC_epoch2 B6); identical at any n_jobs",
                  "g1_z": G1_Z, "g1_floor": G1_FLOOR, "g1_trend_drift_sd": trend_drift_sd, "g1_trend_precedence": G1_TREND_PRECEDENCE,
                  "g1_surrogate": "phase_randomized", "g2_coverage_floor": G2_COVERAGE_FLOOR, "dt_bracket": list(DT_BRACKET),
                  "g5_min_windows": G5_MIN_WINDOWS, "delta5": DELTA5_NOTE, "temporal_channel": S.TEMPORAL_CHANNEL if rung in ("content", "combined") else S.rung_channels(rung, True)[0],
                  "wapf_norm": wapf_norm,
                  "duration_source": "sidecar duration_s_declared", "duration_s_per_kernel": dur_per_kernel,
                  "duration_s_fallback": DURATION_S, "duration_s_fallback_cells": fallback_cells,
                  "head_drop": hd, "head_drop_idle": S.head_drop_for(hd, "idle", "idle"),
                  "excluded_cells_hard": ex_hard, "excluded_cells_pair_rungs": ex_pair, "gc_verdict": S.gc_verdict(out, rung),
                  "inputs_sha256": S.inputs_sha256([out / "cells.csv", out / "inputs" / "pass_table.csv", out / "inputs" / "head_drop.csv",
                                                    out / "gates" / "preconditions.csv"], out)}
        S.write_json(d / "temporal.params.json", "plan11.temporal_grid.v1", params, CIT_GRID, {"n_kernels": len(kernels)})
        written.append(d / "temporal_per_kernel.csv")
    return written


# --------------------------------------------------------------------------- G3 flags (3.5.3)

G3_COLUMNS = ("rung", "kernel", "cell_id", "ceps_peak_idx", "ceps_peak_freq_cyc_per_pair", "ceps_snr_db", "snr_surrogate_p95",
              "cv", "cv_shot_floor", "cv_ratio", "flag_cell", "flag_kernel")


def gate_g3(out: Path, rung: str, *, min_cells: int = G3_MIN_CELLS, n_surrogates: int = G3_N_SURROGATES,
            min_quef_frac: float = G3_MIN_QUEF_FRAC, seed_offset: int = 0, null: str = G3_NULL) -> Path:
    """G3 as the per-kernel signal flag off the grid decision (P2 Sec. V 5.1, Sec. 6 item 7 option (a);
    CR 2.1 item 5): gates/g3_flags.csv, one row per cell; flag_kernel = G3_PRESENT if at least
    ``min_cells`` of the kernel's cells are present (section 8 item 11). The quefrency floor
    ``max(1, int(n * min_quef_frac))`` (Plan 03's R2 override, n // 8 at the default) excludes any
    rhythm faster than about n / 8 pairs from the search by construction (``params.quefrency_floor``).
    Rows of this rung are replaced on re-run; other rungs' rows are kept."""
    out = Path(out)
    cells, ex_hard, ex_pair = _cells_for(out, rung)
    hd = S.load_head_drop(out / "inputs" / "head_drop.csv")
    rows = []
    by_k = {}
    for c in cells:
        ex = S.load_extract_cached(out, c["cell_id"])
        x = S.temporal_series(ex, rung, S.head_drop_for(hd, c["kernel"], c["role"]))
        kmed = S.k_median_cell(ex, S.head_drop_for(hd, c["kernel"], c["role"]))
        r = g3_cell(x, kmed, n_surrogates=n_surrogates, seed=NL.SEED_ORDER + seed_offset, min_quef_frac=min_quef_frac, null=null)
        k = "idle" if c["role"] == "idle" else c["kernel"]
        row = {"rung": rung, "kernel": k, "cell_id": c["cell_id"], **{kk: r[kk] for kk in G3_COLUMNS if kk in r}}
        rows.append(row); by_k.setdefault(k, []).append(row)
    for k, rs in by_k.items():
        n_present = sum(1 for r in rs if r["flag_cell"] == V.G3_PRESENT)
        fk = V.G3_PRESENT if n_present >= min_cells else V.G3_ABSENT
        for r in rs:
            r["flag_kernel"] = fk
    p = out / "gates" / "g3_flags.csv"
    old = [r for r in S.read_csv(p) if r["rung"] != rung] if p.is_file() else []
    S.write_csv(p, G3_COLUMNS, old + rows)
    qf = {c["cell_id"]: None for c in cells}
    S.write_json(out / "gates" / f"g3.{rung}.params.json", "plan11.g3.v1",
                 {"rung": rung, "g3_min_cells": min_cells, "n_surrogates": n_surrogates, "min_quef_frac": min_quef_frac,
                  "quefrency_floor": "max(1, int(n * min_quef_frac)); search stops at len(c) // 2", "g3_null": null,
                  "seed_order": NL.SEED_ORDER + seed_offset, "cv_std": "population", "cv_verdict": "none (section 8 item 11)",
                  "retired_thresholds": "4.5 dB and 0.30/0.50 CV ceilings not used", "excluded_cells_hard": ex_hard,
                  "excluded_cells_pair_rungs": ex_pair, "inputs_sha256": S.inputs_sha256([out / "cells.csv", out / "inputs" / "head_drop.csv"], out)},
                 CIT_G3, {"n_rows": len(rows), "per_kernel": {k: rs[0]["flag_kernel"] for k, rs in by_k.items()}})
    return p


# --------------------------------------------------------------------------- G-ORD (3.5.6)

def _loko_accuracy(F: np.ndarray, cell_ids, kernels, archetypes, *, seed, n_estimators, labels_override=None) -> float:
    from plan11_encoding_ladder import models as M
    lab = {"n": len(cell_ids), "cell_id": np.asarray(cell_ids).astype(str), "kernel": np.asarray(kernels).astype(str),
           "archetype": np.asarray(archetypes).astype(str), "win_start": np.zeros(len(cell_ids), dtype=int),
           "rep": np.zeros(len(cell_ids), dtype=int), "campaign": np.array([""] * len(cell_ids))}
    cells = list(dict.fromkeys(lab["cell_id"].tolist()))
    y_unit = {c: str(lab["archetype"][lab["cell_id"] == c][0]) for c in cells}
    if labels_override is not None:
        y_unit = dict(labels_override)
    kernel_of = {c: str(lab["kernel"][lab["cell_id"] == c][0]) for c in cells}
    folds = SP.fold_loko(lab)
    preds = M.fit_predict_units(F, np.array([y_unit[c] for c in lab["cell_id"]]), folds, lab["cell_id"], seed=seed, n_estimators=n_estimators)
    sc = M.score_units(y_unit, {c: v["y_pred"] for c, v in preds.items()}, kernel_of, None)
    return sc["accuracy"]


GORD_PARALLEL_BACKEND = "joblib threads"    # SPEC_epoch2 section 5.2: threads, every random draw made before dispatch


def _gord_featurize(series_: dict, cells: list[dict], arche: dict, rung: str, W: int, H: int, perms: list | None = None) -> tuple:
    """Window features of every kernel cell at (W, H) in ``cells`` order; with ``perms`` (one row
    permutation per cell, in the same order) each cell's series is re-ordered first (G-ORD's order
    shuffle, CR 2.2 item 27). Pure: no random draw happens here."""
    Xs, cid, ker, arc = [], [], [], []
    for i, c in enumerate(cells):
        Sm = series_[c["cell_id"]]
        if perms is not None:
            Sm = np.asarray(Sm)[perms[i]]
        F, _, _ = S.window_features(Sm, rung, W, H)
        if len(F) == 0:
            continue
        Xs.append(F); cid += [c["cell_id"]] * len(F); ker += [c["kernel"]] * len(F); arc += [arche[c["cell_id"]]] * len(F)
    return (np.concatenate(Xs) if Xs else np.zeros((0, len(S.feature_names(rung))))), cid, ker, arc


def _gord_shuffled_score(series_: dict, cells: list[dict], arche: dict, perms: list, W: int, H: int, rung: str,
                         seed: int, n_estimators: int) -> float:
    """One order-shuffle repetition of G-ORD: re-window every cell under its pre-drawn permutation and
    score LOKO / archetype with the fixed forest seed (SPEC_epoch2 section 5.2 step 2). Pure."""
    F2, cid2, ker2, arc2 = _gord_featurize(series_, cells, arche, rung, W, H, perms)
    return _loko_accuracy(F2, cid2, ker2, arc2, seed=seed, n_estimators=n_estimators)


def _gord_null_score(F: np.ndarray, cid: list, ker: list, arc: list, labels_override: dict, seed: int, n_estimators: int) -> float:
    """One label-shuffle draw of G-ORD's null at a fixed W: the ordered features under a pre-drawn
    unit-level label permutation (SPEC_epoch2 section 5.2 step 2). Pure."""
    return _loko_accuracy(F, cid, ker, arc, seed=seed, n_estimators=n_estimators, labels_override=labels_override)


def gate_gord(out: Path, rung: str, *, n_order_perm: int = GORD_N_ORDER_PERM, null_perm: int = GORD_NULL_PERM,
              n_jobs: int = 1, seed_offset: int = 0, n_estimators: int = 300) -> list[Path]:
    """G-ORD, the time-shuffle null (P2 Sec. V 5.1; CR 2.2 item 27). Per rung and per W (at H = W // 2,
    the middle hop ratio), split LOKO, label space archetype, the forest of 4.2: score_ordered; then
    ``n_order_perm`` order shuffles per cell (one permutation drawn per cell per repetition),
    re-windowed, the same split, the mean score_shuffled; and a label-shuffle null of ``null_perm``
    permutations at that W for its spread only. GORD = GORD_ORDER_BLIND if |score_ordered -
    score_shuffled| <= spread (p95 - p05 of the null) else GORD_RESOLUTION. Written to ``gord.json``
    under the H = W // 2 grid directory and copied into the GORD column of every grid point that
    shares that W. The whole-cell point is ``order-blind (by construction)``: the eight shape features
    are permutation-invariant within a window and the whole cell is one window (al-Farabi 2.11 (b)).
    Re-labels the W axis of Table 5 and refuses nothing.

    ``n_jobs`` is honoured since build epoch 2 (SPEC_epoch2 section 5.2; E1 sec. 4 M6; CHECK_3 M6):
    every random object is drawn first in the calling thread, in epoch 1's draw order from the same
    seeds (``rng_o = default_rng(SEED_ORDER + seed_offset)``: repetition outer, cell inner, one
    permutation of the cell's row count each, exactly what ``nulls.order_shuffle`` drew;
    ``rng_l = default_rng(SEED_LABEL_NULL + seed_offset)``: one unit-level label vector per
    repetition), and the two loops are then scored through ``joblib.Parallel(n_jobs, prefer="threads")``
    on the pure functions ``_gord_shuffled_score`` and ``_gord_null_score``. The forest seed is fixed
    and no draw happens after dispatch, so the output is identical at any ``n_jobs`` and identical to
    epoch 1's single-process output for the same inputs; ``params`` records ``n_jobs`` and
    ``parallel_backend``."""
    from joblib import Parallel, delayed
    out = Path(out)
    cells, ex_hard, ex_pair = _cells_for(out, rung)
    cells = [c for c in cells if c["role"] == "kernel"]
    hd = S.load_head_drop(out / "inputs" / "head_drop.csv")
    relabel = S.gk0_relabel(out)
    series_ = {}
    for c in cells:
        ex = S.load_extract_cached(out, c["cell_id"])
        series_[c["cell_id"]] = S.rung_series(ex, rung, normalized=True, head_drop=S.head_drop_for(hd, c["kernel"], c["role"]))
    arche = {c["cell_id"]: relabel.get(c["kernel"], c["archetype_predicted"]) for c in cells}
    seed = NL.SEED_FOREST + seed_offset
    n_jobs = int(n_jobs) if n_jobs else 1
    written = []

    for W in [w for w in schema.GRID_WINDOWS if w != "whole"]:
        H = W // 2
        gid = schema.grid_id(W, H)
        d = out / "gates" / "grid" / rung / gid
        d.mkdir(parents=True, exist_ok=True)
        F, cid, ker, arc = _gord_featurize(series_, cells, arche, rung, W, H)
        if len(F) == 0 or len(set(arc)) < 2:
            rec = {"W": W, "H": H, "score_ordered": None, "score_shuffled_mean": None, "null_spread": None,
                   "GORD": V.not_run("fewer than two archetypes with windows")}
        else:
            s_ord = _loko_accuracy(F, cid, ker, arc, seed=seed, n_estimators=n_estimators)
            # step 1: every random object drawn here, in epoch 1's order, before any dispatch
            rng_o = np.random.default_rng(NL.SEED_ORDER + seed_offset)
            order_perms = [[rng_o.permutation(np.asarray(series_[c["cell_id"]]).shape[0]) for c in cells] for _ in range(n_order_perm)]
            rng_l = np.random.default_rng(NL.SEED_LABEL_NULL + seed_offset)
            ucells = list(dict.fromkeys(cid))
            ukern = np.array([ker[cid.index(c)] for c in ucells]); uarc = np.array([arc[cid.index(c)] for c in ucells])
            label_perms = [NL.shuffle_labels_units(np.array(ucells), ukern, uarc, "loko", "archetype", rng_l) for _ in range(null_perm)]
            # step 3: the two loops, scored in parallel over pure functions; result order = input order
            with Parallel(n_jobs=n_jobs, prefer="threads") as par:
                shuf = list(par(delayed(_gord_shuffled_score)(series_, cells, arche, perms, W, H, rung, seed, n_estimators)
                                for perms in order_perms))
                null = list(par(delayed(_gord_null_score)(F, cid, ker, arc, {c: str(pl[i]) for i, c in enumerate(ucells)}, seed, n_estimators)
                                for pl in label_perms))
            shuf = [float(x) for x in shuf]
            summ = NL.null_summary(s_ord, np.array(null, dtype=np.float64))
            s_sh = float(np.mean(shuf)) if shuf else None
            spread = summ["spread"]
            verdict = V.not_run("no null") if (spread is None or s_sh is None) else (
                V.GORD_ORDER_BLIND if abs(s_ord - s_sh) <= spread else V.GORD_RESOLUTION)
            rec = {"W": W, "H": H, "score_ordered": s_ord, "score_shuffled_mean": s_sh, "score_shuffled": shuf,
                   "null_spread": spread, "null_summary": summ, "GORD": verdict}
        params = {"rung": rung, "W": W, "H": H, "n_order_perm": n_order_perm, "null_perm": null_perm, "split": "loko",
                  "labelspace": "archetype", "seed_forest": seed, "seed_order": NL.SEED_ORDER + seed_offset,
                  "seed_label_null": NL.SEED_LABEL_NULL + seed_offset, "n_estimators": n_estimators,
                  "n_jobs": n_jobs, "parallel_backend": GORD_PARALLEL_BACKEND, "draw_order": "all permutations drawn before dispatch (SPEC_epoch2 5.2)",
                  "gk0_applied": bool(relabel), "relabelled_kernels": sorted(relabel), "excluded_cells_pair_rungs": ex_pair,
                  "inputs_sha256": S.inputs_sha256([out / "cells.csv", out / "gates" / "gk0.csv"], out)}
        S.write_json(d / "gord.json", "plan11.gord.v1", params, CIT_GORD, rec)
        written.append(d / "gord.json")
        # copy into the GORD column of every grid point sharing this W
        for gid2, W2, H2 in S.grid_points_ids():
            if W2 != W:
                continue
            p = out / "gates" / "grid" / rung / gid2 / "temporal_per_kernel.csv"
            if not p.is_file():
                continue
            rows = S.read_csv(p)
            for r in rows:
                r["gord_score_ordered"] = rec["score_ordered"]; r["gord_score_shuffled_mean"] = rec["score_shuffled_mean"]
                r["gord_null_spread"] = rec["null_spread"]; r["GORD"] = rec["GORD"]
            S.write_csv(p, PER_KERNEL_COLUMNS, rows)
    return written


# --------------------------------------------------------------------------- roll-up and selection (3.5.7)

def _applicable(r: dict, gate: str, dt_key: str | None, kernel_refusals: str) -> bool:
    """Is this kernel row applicable for the gate at the roll-up (SPEC_review_al_farabi.md item 2.1)?"""
    if r["kernel"] == "idle":
        return False
    if gate == "G1":
        v = r["G1"]
        if v.startswith("not run"):
            return False
        return not (kernel_refusals == "not_applicable" and v == V.TREND_PRESENT)
    if gate == "G2":
        v = r[dt_key]
        if v == V.GP_UNDECLARED or v.startswith("not run"):
            return False
        return not (kernel_refusals == "not_applicable" and v.split(" (")[0] == V.G2_ABOVE_NYQUIST)
    return True


def _rollup(rows: list[dict], gate: str, relabelled: set, rollup: str, kernel_refusals: str, dt_key: str | None = None) -> tuple:
    """(verdict, n_applicable, n_na) for one gate at one grid point; n_na counts the kernels a grid-independent
    refusal (TREND_PRESENT, G2_ABOVE_NYQUIST) made not applicable, not the undeclared ones."""
    app = [r for r in rows if r["kernel"] not in relabelled and _applicable(r, gate, dt_key, kernel_refusals)]
    key = dt_key or gate
    n_na = sum(1 for r in rows if r["kernel"] != "idle" and r["kernel"] not in relabelled and not _applicable(r, gate, dt_key, kernel_refusals)
               and r[key] != V.GP_UNDECLARED and not r[key].startswith("not run"))
    if not app:
        return (V.not_applicable("no kernel with a declared pass period") if gate == "G2" else V.not_applicable("no applicable kernel")), 0, n_na
    key = dt_key or gate
    ok = [r[key].split(" (")[0] == V.PASS for r in app]
    if rollup == "all_kernels":
        v = V.PASS if all(ok) else V.FAIL
    elif rollup == "majority":
        v = V.PASS if sum(ok) * 2 > len(ok) else V.FAIL
    else:
        raise ValueError(rollup)
    return v, len(app), n_na


def select(out: Path, rung: str, *, rollup: str = GRID_ROLLUP, kernel_refusals: str = ROLLUP_KERNEL_REFUSALS,
           g1_none_applicable: str = G1_NONE_APPLICABLE) -> Path:
    """Roll-up and selection (SPEC 3.5.7; plan03_aggregate.py _pick_winner re-pointed). Per grid point
    a gate passes when it passes for every applicable kernel (``grid_rollup = "all_kernels"``; or
    more than half under ``"majority"``): kernels relabelled IDLE by G-K0 and the idle row are out of
    the roll-up; a kernel whose G1 is TREND_PRESENT is not applicable for G1 and a kernel whose G2 at a
    dt is G2_ABOVE_NYQUIST is not applicable for G2 at that dt (``rollup_kernel_refusals =
    "not_applicable"``; ``"blocks"`` counts them as not passing); G2 is rolled up per dt column then
    combined by 3.5.2's rule; GP_UNDECLARED kernels never block G2. ``gates_passed`` is "k of m" over
    the applicable gates among G1, G2, G4 (G5 reported; G-ORD a label). Selection: the smallest integer
    W whose point passes every applicable gate, tie-break hop_ratio nearest 0.5; else the point passing
    the most gates with the same tie-breaks, ``selected_by = "best-feasible"`` and
    ``passes_acceptance = false``. Writes table5_long.csv, table5_grid.csv, selection.json (merged per
    rung) and grid_complete.json (13 CSVs and 13 feature pairs present; nothing is ever deleted).

    Build epoch 2. ``g1_none_applicable`` (SPEC_epoch2 B7; CHECK_3 M7; SPEC 3.5.7 "every applicable
    gate among G1, G2, G4"): when G1's roll-up is ``not applicable: no applicable kernel`` (every
    kernel TREND_PRESENT or not run), ``"drop"`` leaves G1 out of the applicable gates as G2 is left
    out when no kernel has a declared pass period; ``"refuse"`` keeps the epoch-1 count (the point
    cannot pass) and the entry reads ``refusal = "not run: no applicable kernel for G1"``,
    ``passes_acceptance = false``, ``selected_by = "no applicable kernel"``. A missing grid CSV
    (SPEC_epoch2 B16; CERT 1(d), 7.4): the entry gets ``refusal = "not run: grid incomplete (<n> points
    missing)"``, ``passes_acceptance = false`` and ``selected_by`` suffixed `` (grid incomplete)``; the
    choice among the existing points is still recorded. A best-feasible selection whose refusal would
    be empty (SPEC_epoch2 B17; CERT 3) writes ``refusal = "acceptance failed: <gate>: <verdict>, ..."``
    over every applicable gate not at ``pass``; ``GC_DISCONNECTED`` keeps precedence; ``selected_by``
    stays ``best-feasible``. G-F's verdict is no longer copied into ``table5_grid.csv`` (B15): it lives
    in ``gates/gf.csv`` only, where the tables read it."""
    if g1_none_applicable not in ("drop", "refuse"):
        raise ValueError(g1_none_applicable)
    out = Path(out)
    relabelled = set(S.gk0_relabel(out))
    gc_v = S.gc_verdict(out, rung)
    long_rows, grid_rows, complete = [], [], {"csv": {}, "features": {}}
    for gid, W, H in S.grid_points_ids():
        p = out / "gates" / "grid" / rung / gid / "temporal_per_kernel.csv"
        complete["csv"][gid] = p.is_file()
        complete["features"][gid] = {v: S.features_path(out, rung, gid, v == "norm").is_file()
                                     for v in (("norm",) if rung == "combined" else ("raw", "norm"))}
        if not p.is_file():
            continue
        rows = S.read_csv(p)
        long_rows += rows
        g1, n1, na1 = _rollup(rows, "G1", relabelled, rollup, kernel_refusals)
        g2a, n2a, na2a = _rollup(rows, "G2", relabelled, rollup, kernel_refusals, "G2_0500")
        g2b, n2b, na2b = _rollup(rows, "G2", relabelled, rollup, kernel_refusals, "G2_0644")
        g2p, _, _ = _rollup(rows, "G2", relabelled, rollup, kernel_refusals, "G2_pairs")
        if g2a.startswith("not applicable") and g2b.startswith("not applicable"):
            g2 = g2a
        elif g2a == g2b:
            g2 = g2a
        else:
            g2 = V.G2_UNDETERMINED
        kern_rows = [r for r in rows if r["kernel"] != "idle"]
        n_undecl = sum(1 for r in kern_rows if r["kernel"] not in relabelled and r["G2_0500"] == V.GP_UNDECLARED)
        g4 = V.PASS if (W is not None and g4_pass(W, H)) else V.FAIL
        g5 = V.PASS if kern_rows and all(r["G5"] == V.PASS for r in kern_rows) else V.FAIL
        gord = rows[0]["GORD"] if rows else ""
        applicable = {"G4": g4}
        if not (g1_none_applicable == "drop" and g1.startswith("not applicable")):
            applicable["G1"] = g1                      # SPEC_epoch2 B7: under "drop" G1 leaves the roll-up as G2 does
        if not g2.startswith("not applicable"):
            applicable["G2"] = g2
        k = sum(1 for v in applicable.values() if v == V.PASS); m = len(applicable)
        med = lambda key: (float(np.median([S.to_float(r[key]) for r in kern_rows if r.get(key) not in ("", None)])) if any(r.get(key) not in ("", None) for r in kern_rows) else None)
        grid_rows.append({"rung": rung, "axis": S.AXIS_OF_RUNG[rung], "grid_id": gid, "W": W if W is not None else med("W"),
                          "H": H if H is not None else med("H"), "hop_ratio": S.hop_ratio(W, H) if W else 1.0,
                          "n_windows_median": med("n_windows_median"), "n_windows_nonoverlap_median": med("n_windows_nonoverlap_median"),
                          "G1": g1, "G2_0500": g2a, "G2_0644": g2b, "G2": g2, "coverage_pairs": med("coverage_pairs"), "G2_pairs": g2p,
                          "G4": g4, "G5": g5, "GORD": gord, "n_kernels_na_G1": na1, "n_kernels_na_G2": max(na2a, na2b), "n_kernels_undeclared_G2": n_undecl,
                          "gates_passed": f"{k} of {m}", "_k": k, "_m": m, "_all": k == m, "_W": W, "selected": "", "selected_by": "",
                          "refusal": gc_v if gc_v == V.GC_DISCONNECTED else "", "gc_verdict": gc_v or "",  # CR 2.2 item 22; CHECK_1 M7
                          "_applicable": applicable})
    # selection among integer-W points
    cands = [g for g in grid_rows if g["_W"] is not None]
    sort_key = lambda g: (g["_W"], abs(g["hop_ratio"] - 0.5))
    passing = sorted([g for g in cands if g["_all"] and g["_m"] > 0], key=sort_key)
    sel = None
    if passing:
        sel, by, acc = passing[0], "smallest W passing G1, G2, G4; tie-break hop ratio nearest 0.5", True
    elif cands:
        best = max(g["_k"] for g in cands)
        sel = sorted([g for g in cands if g["_k"] == best], key=sort_key)[0]
        by, acc = "best-feasible", False
    if sel is not None:
        sel["selected"] = "selected" if acc else "selected: best-feasible"
        sel["selected_by"] = by
        # SPEC_epoch2 B17 (CERT 3): a best-feasible selection names the gates that failed acceptance
        if not acc and not sel["refusal"]:
            failed = [f"{g}: {v}" for g, v in sel["_applicable"].items() if v != V.PASS]
            sel["refusal"] = "acceptance failed: " + ", ".join(failed) if failed else ""
    refusal = sel["refusal"] if sel else V.not_run("no grid point computed")
    passes = bool(sel and acc)
    selected_by = by if sel else "no grid point"
    # SPEC_epoch2 B7 "refuse": the point was chosen while G1 had no applicable kernel
    if sel is not None and g1_none_applicable == "refuse" and str(sel["G1"]).startswith("not applicable"):
        refusal, passes, selected_by = V.not_run("no applicable kernel for G1"), False, "no applicable kernel"
    # SPEC_epoch2 B16 (CERT 1(d)): a missing grid CSV refuses the selection; the choice among the existing points stays recorded
    n_missing = sum(1 for v in complete["csv"].values() if not v)
    if n_missing:
        refusal, passes = V.not_run(f"grid incomplete ({n_missing} points missing)"), False
        selected_by = f"{selected_by} (grid incomplete)"
    entry = {"grid_id": sel["grid_id"] if sel else None, "W": sel["_W"] if sel else None, "H": sel["H"] if sel else None,
             "passes_acceptance": passes, "selected_by": selected_by, "gates_passed": sel["gates_passed"] if sel else None,
             "refusal": refusal,
             "params": {"grid_rollup": rollup, "rollup_kernel_refusals": kernel_refusals, "g1_none_applicable": g1_none_applicable,
                        "relabelled_kernels": sorted(relabelled), "n_grid_points_missing": n_missing,
                        "gc_verdict": gc_v, "idle_row_in_rollup": False, "gates_in_rule": ["G1", "G2", "G4"], "g5": "reported", "gord": "label"}}
    complete["complete"] = all(complete["csv"].values()) and all(all(v.values()) for v in complete["features"].values())
    # merge into the shared files
    for name, cols, new in (("table5_long.csv", PER_KERNEL_COLUMNS, long_rows), ("table5_grid.csv", GRID_COLUMNS, grid_rows)):
        p = out / "gates" / name
        old = [r for r in S.read_csv(p) if r["rung"] != rung] if p.is_file() else []
        S.write_csv(p, cols, old + [{k: v for k, v in r.items() if not k.startswith("_")} for r in new])
    sp = out / "gates" / "selection.json"
    doc = S.read_json(sp) if sp.is_file() else {"schema": "plan11.selection.v1", "params": {}, "citation": CIT_SELECT}
    doc[rung] = entry
    doc["params"][rung] = entry["params"]
    doc["params"]["inputs_sha256_" + rung] = S.inputs_sha256([out / "gates" / "gk0.csv", out / "gates" / "gc.csv"] +
                                                              [out / "gates" / "grid" / rung / g / "temporal_per_kernel.csv" for g, _, _ in S.grid_points_ids()], out)
    doc["citation"] = CIT_SELECT
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(doc, indent=1, default=S._json_default))
    gp = out / "gates" / "grid_complete.json"
    gdoc = S.read_json(gp) if gp.is_file() else {"schema": "plan11.grid_complete.v1", "params": {}, "citation": CIT_SELECT}
    gdoc[rung] = complete
    gp.write_text(json.dumps(gdoc, indent=1, default=S._json_default))
    if rung == "apf":
        _refresh_c7(out)
    return sp


def _refresh_c7(out: Path) -> None:
    """C7 (Plan 03 winner) is 'pending' when preconditions run at move 2; once the APF selection exists the
    column is refreshed in place (SPEC 3.3.1: pass iff passes_acceptance, else fail)."""
    from plan11_encoding_ladder.gates_precondition import c7_verdict, PRE_COLUMNS
    p = out / "gates" / "preconditions.csv"
    if not p.is_file():
        return
    rows = S.read_csv(p)
    v = c7_verdict(out)
    for r in rows:
        if not r.get("C7", "").startswith("not run"):
            r["C7"] = v
    S.write_csv(p, PRE_COLUMNS, rows)


# --------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="gates_temporal.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grid"); g.add_argument("--out", required=True); g.add_argument("--rung", required=True, choices=S.RUNGS)
    g.add_argument("--n-surrogates", type=int, default=G1_N_SURROGATES); g.add_argument("--trend-drift-sd", type=float, default=G1_TREND_DRIFT_SD)
    g.add_argument("--n-jobs", type=int, default=1); g.add_argument("--no-features", action="store_true")
    g.add_argument("--wapf-norm", default=S.WAPF_NORM_DEFAULT, choices=("median_K", "median_self"),
                   help="wAPF's level normalization for the feature files and the temporal series (SPEC_epoch2 B19; SPEC section 8 item 7)")
    t = sub.add_parser("g3"); t.add_argument("--out", required=True); t.add_argument("--rung", required=True, choices=S.RUNGS)
    t.add_argument("--min-cells", type=int, default=G3_MIN_CELLS); t.add_argument("--n-surrogates", type=int, default=G3_N_SURROGATES)
    t.add_argument("--min-quef-frac", type=float, default=G3_MIN_QUEF_FRAC); t.add_argument("--null", default=G3_NULL, choices=("order_shuffle", "phase_randomize"))
    o = sub.add_parser("gord"); o.add_argument("--out", required=True); o.add_argument("--rung", required=True, choices=S.RUNGS)
    o.add_argument("--n-order-perm", type=int, default=GORD_N_ORDER_PERM); o.add_argument("--null-perm", type=int, default=GORD_NULL_PERM)
    o.add_argument("--n-jobs", type=int, default=1); o.add_argument("--n-estimators", type=int, default=300)
    s = sub.add_parser("select"); s.add_argument("--out", required=True); s.add_argument("--rung", required=True, choices=S.RUNGS)
    s.add_argument("--rollup", default=GRID_ROLLUP, choices=("all_kernels", "majority"))
    s.add_argument("--rollup-kernel-refusals", default=ROLLUP_KERNEL_REFUSALS, choices=("not_applicable", "blocks"))
    s.add_argument("--g1-none-applicable", default=G1_NONE_APPLICABLE, choices=("drop", "refuse"),
                   help="when no kernel is applicable for G1: drop G1 from the roll-up (as G2) or refuse the selection (SPEC_epoch2 B7)")
    for sp in (g, t, o, s):
        sp.add_argument("--seed-offset", type=int, default=0)
    args = ap.parse_args(argv)
    out = Path(args.out)
    if not (out / "cells.csv").is_file():
        print(f"missing input: {out / 'cells.csv'}", file=sys.stderr); return 2
    try:
        if args.cmd == "grid":
            for p in gate_grid(out, args.rung, n_surrogates=args.n_surrogates, trend_drift_sd=args.trend_drift_sd, n_jobs=args.n_jobs,
                               seed_offset=args.seed_offset, build_features=not args.no_features, wapf_norm=args.wapf_norm):
                print(p)
        elif args.cmd == "g3":
            print(gate_g3(out, args.rung, min_cells=args.min_cells, n_surrogates=args.n_surrogates, min_quef_frac=args.min_quef_frac,
                          seed_offset=args.seed_offset, null=args.null))
        elif args.cmd == "gord":
            for p in gate_gord(out, args.rung, n_order_perm=args.n_order_perm, null_perm=args.null_perm, n_jobs=args.n_jobs,
                               seed_offset=args.seed_offset, n_estimators=args.n_estimators):
                print(p)
        elif args.cmd == "select":
            print(select(out, args.rung, rollup=args.rollup, kernel_refusals=args.rollup_kernel_refusals,
                         g1_none_applicable=args.g1_none_applicable))
    except FileNotFoundError as e:
        print(f"missing input: {e}", file=sys.stderr); return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
