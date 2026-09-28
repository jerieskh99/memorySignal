#!/usr/bin/env python3
"""gates_readings.py -- G-J (the persistence null) and G-DEC (decay validity) (SPEC section 3.6).

Citation: P2_STRUCTURE.md section V 5.2 G-J and G-DEC, section 6 item 6 (no floor subtraction);
CR 2.2 items 31 (G-J) and 34 (G-DEC); K2 Sec. 2 rung 1 (a) and rung 2 (e);
SPEC_review_al_kindi.md item 4 (G-DEC's boundary and the control) and item 9 (the two masks).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plan11_encoding_ladder import series as S
from plan11_encoding_ladder import nulls as NL
from plan11_encoding_ladder import verdicts as V
from plan11_encoding_ladder.series import schema
from plan11_encoding_ladder.gates_calibration import (load_pass_table, gp_kernel_verdicts, k_jump_events,
                                                      GC_JUMP_DETECT_RATIO, GC_JUMP_PREDICTED, GC_JUMP_REFERENCE)

GJ_K_FACTOR = 3.0                    # section 8 item 19
GJ_BELOW_NULL_RULE = "J <= J_null"
GJ_CELL_VERDICT_RULE = "majority_of_pairs"
GDEC_KERNEL = "floyd"
GDEC_CONTROL = "gibbs"
GDEC_BOUNDARY_SOURCE = "k_jump"      # section 8 item 20; alternative "period"
GDEC_PHASE_PAIRS = 0                 # SPEC_review_al_kindi.md item 4 (a): 'period' anchors at seq_first + phase_pairs
GDEC_MIN_RUN = 3
GDEC_MIN_REPS = 7
GDEC_N_SURROGATES = 200
GDEC_PASS_FRAC = 0.5                 # the fall must hold in at least this fraction of a cell's passes (a choice; section 8)
GDEC_IDLE_SLOPE_RULE = "any"         # (e): "any" idle cell with a significant same-sign slope refuses (the definition's words); alternative "min_reps"
CIT_GJ = "P2 Sec. V 5.2 G-J and Sec. 6 item 6; CR 2.2 item 31; K2 Sec. 2 rung 1 (a); SPEC_review_al_kindi.md item 9"
CIT_GDEC = "P2 Sec. V 5.2 G-DEC; CR 2.2 item 34; K2 Sec. 2 rung 2 (e); SPEC_review_al_kindi.md item 4"

GJ_COLUMNS = ("kernel", "cell_id", "n_pairs_J", "floor_median_K", "k_threshold", "frac_interpretable", "J_mean",
              "J_q05", "J_q25", "J_q50", "J_q75", "J_q95", "J_null_q50", "frac_below_null", "gp_verdict_pairs", "mask_verdict",
              "floor_median_n_persist", "frac_interpretable_persist")


def gate_gj(out: Path, cells: list[dict] | None = None, *, k_factor: float = GJ_K_FACTOR) -> Path:
    """G-J, the persistence null. Two nulls per pair: the independence null J_null (extract
    column) and the idle cells' own J distribution (empirical floor null: the five quantiles
    of J pooled over idle cells, to gates/gj.json). Mask: a kernel's J is interpretable
    only at pairs where K_t > k_factor * (the floor's median K, pooled over idle cells'
    rows); pairs below carry GJ_FLOOR_OVERLAP. With no idle cell every pair is
    GJ_FLOOR_UNMEASURED and J is reported unmasked and labelled. No floor subtraction
    (decided 2026-09-16, P2 Sec. 6 item 6). Per cell summary: mean J, the five quantiles,
    fraction of pairs interpretable, fraction of interpretable pairs with J <= J_null (the
    'below null' fraction; the definition names 'a declared null-relative threshold' and this
    toolkit declares J <= J_null), G-P verdict beside it. Citation: P2 Sec. V 5.2 G-J;
    CR 2.2 item 31.
    The per-pair mask is saved as gates/gj_mask/<cell_id>.npy, a structured bool array with two
    fields (SPEC_review_al_kindi.md item 9): ``mask_K`` (the definition, applied by default) and
    ``mask_persist`` (n_persist > k_factor * the idle cells' median n_persist). The per-cell
    ``mask_verdict`` is GJ_INTERPRETABLE when more than half the cell's pairs are interpretable
    under mask_K, else GJ_FLOOR_OVERLAP (``gj_cell_verdict_rule = "majority_of_pairs"``)."""
    out = Path(out)
    if cells is None:
        cells = S.load_cells(out / "cells.csv")
    cells, ex_hard, ex_pair, _ = S.admissible_cells(out, cells, "persist")
    idle = [c for c in cells if c["role"] == "idle"]
    kern = [c for c in cells if c["role"] == "kernel"]
    gpv = gp_kernel_verdicts(out)
    hd = S.load_head_drop(out / "inputs" / "head_drop.csv")
    q = list(schema.QUANTILES)
    floor_K = floor_P = None
    idle_J = None
    if idle:
        Ks = np.concatenate([S.load_extract_cached(out, c["cell_id"])["K"] for c in idle])
        Ps = np.concatenate([S.load_extract_cached(out, c["cell_id"])["n_persist"] for c in idle])
        Js = np.concatenate([S.load_extract_cached(out, c["cell_id"])["J"] for c in idle])
        floor_K = float(np.median(Ks)); floor_P = float(np.nanmedian(Ps))
        Jv = Js[~np.isnan(Js)]
        idle_J = {"quantiles": q, "J": np.quantile(Jv, q).tolist() if len(Jv) else None, "n_pairs": int(len(Jv)), "mean": float(Jv.mean()) if len(Jv) else None}
    rows = []
    mask_dir = out / "gates" / "gj_mask"
    mask_dir.mkdir(parents=True, exist_ok=True)
    for c in kern + idle:
        ex = S.load_extract_cached(out, c["cell_id"])
        h = S.head_drop_for(hd, c["kernel"], c["role"])
        n = int(ex["_n_rows"]) - 1
        J = ex["J"][h:n]; Jn = ex["J_null"][h:n]; K = ex["K"][h:n]; P = ex["n_persist"][h:n]
        if floor_K is None:
            mK = np.ones(len(J), dtype=bool); mP = np.ones(len(J), dtype=bool)
            mv = V.GJ_FLOOR_UNMEASURED
        else:
            mK = K > k_factor * floor_K; mP = P > k_factor * floor_P
            mv = V.GJ_INTERPRETABLE if (len(mK) and mK.mean() > 0.5) else V.GJ_FLOOR_OVERLAP
        valid = ~np.isnan(J)
        Ji = J[valid & mK]; Jni = Jn[valid & mK]
        arr = np.zeros(len(J), dtype=[("mask_K", "?"), ("mask_persist", "?")])
        arr["mask_K"] = mK; arr["mask_persist"] = mP
        np.save(mask_dir / f"{c['cell_id']}.npy", arr)
        qs = np.quantile(Ji, q) if len(Ji) else [None] * 5
        rows.append({"kernel": "idle" if c["role"] == "idle" else c["kernel"], "cell_id": c["cell_id"], "n_pairs_J": int(valid.sum()),
                     "floor_median_K": floor_K, "k_threshold": (k_factor * floor_K) if floor_K is not None else None,
                     "frac_interpretable": float(mK[valid].mean()) if valid.any() else None,
                     "J_mean": float(Ji.mean()) if len(Ji) else None,
                     "J_q05": qs[0], "J_q25": qs[1], "J_q50": qs[2], "J_q75": qs[3], "J_q95": qs[4],
                     "J_null_q50": float(np.median(Jni)) if len(Jni) else None,
                     "frac_below_null": float(np.mean(Ji <= Jni)) if len(Ji) else None,
                     "gp_verdict_pairs": (gpv.get(c["kernel"], {}) or {}).get("verdict_pairs", V.GP_UNDECLARED) if c["role"] == "kernel" else "control",
                     "mask_verdict": "control" if c["role"] == "idle" else mv,
                     "floor_median_n_persist": floor_P, "frac_interpretable_persist": float(mP[valid].mean()) if valid.any() else None})
    p = S.write_csv(out / "gates" / "gj.csv", GJ_COLUMNS, rows)
    S.write_json(out / "gates" / "gj.json", "plan11.gj.v1",
                 {"k_factor": k_factor, "below_null_rule": GJ_BELOW_NULL_RULE, "floor_subtraction": "none (P2 Sec. 6 item 6)",
                  "cell_verdict_rule": GJ_CELL_VERDICT_RULE, "mask_default": "mask_K", "mask_alternative": "mask_persist",
                  "n_idle_cells": len(idle), "excluded_cells_pair_rungs": ex_pair, "head_drop": hd,
                  "inputs_sha256": S.inputs_sha256([out / "cells.csv", out / "gates" / "gp.csv", out / "gates" / "preconditions.csv"], out)},
                 CIT_GJ, {"floor_median_K": floor_K, "floor_median_n_persist": floor_P, "idle_J": idle_J,
                          "mask_files": str(mask_dir), "mask_fields": ["mask_K", "mask_persist"]})
    return p


# --------------------------------------------------------------------------- G-DEC (3.6.2)

def boundaries_of(ex: dict, *, source: str = GDEC_BOUNDARY_SOURCE, detect_ratio: float = GC_JUMP_DETECT_RATIO,
                  reference: str = GC_JUMP_REFERENCE, T_pairs: float | None = None, phase_pairs: int = GDEC_PHASE_PAIRS) -> np.ndarray:
    """Pass boundaries as row indices: 'k_jump' = the first row of every run of consecutive K-jump
    events (the G-C detector at the corrected detection ratio); 'period' = every round(T_pairs) rows
    from ``phase_pairs`` (SPEC_review_al_kindi.md item 4 (a))."""
    n = int(ex["_n_rows"])
    if source == "k_jump":
        ev, _ = k_jump_events(ex["K"], detect_ratio=detect_ratio, reference=reference)
        if len(ev) == 0:
            return np.zeros(0, dtype=np.int64)
        starts = [int(ev[0])] + [int(e) for i, e in enumerate(ev[1:], 1) if e != ev[i - 1] + 1]
        return np.array(starts, dtype=np.int64)
    if source == "period":
        if not T_pairs or T_pairs < 1:
            return np.zeros(0, dtype=np.int64)
        step = max(1, int(round(T_pairs)))
        return np.arange(phase_pairs, n, step, dtype=np.int64)
    raise ValueError(source)


def _slope(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    ok = ~np.isnan(y)
    if ok.sum() < 2:
        return float("nan")
    t = np.arange(len(y), dtype=np.float64)[ok]
    return float(np.polyfit(t, y[ok], 1)[0])


def _within_pass_slope(x: np.ndarray, bounds: np.ndarray) -> float:
    sl = [_slope(x[bounds[i]:bounds[i + 1]]) for i in range(len(bounds) - 1) if bounds[i + 1] - bounds[i] >= 2]
    sl = [s for s in sl if not np.isnan(s)]
    return float(np.mean(sl)) if sl else float("nan")


def dec_cell(ex: dict, bounds: np.ndarray, *, min_run: int = GDEC_MIN_RUN, n_surrogates: int = GDEC_N_SURROGATES,
             seed: int = NL.SEED_ORDER, pass_frac: float = GDEC_PASS_FRAC) -> dict:
    """One cell's G-DEC record on l0_q50_per (l1_q50_per second, hamming's slope sign as the direction
    check): per phase offset p, the fraction of passes in which the series falls strictly across
    min_run consecutive snapshots from p and resets upward at the next boundary; the K and l0
    relative drops over that span; the within-pass slope against n_surrogates within-pass order
    permutations (blocks of one pass). Citation: P2 Sec. V 5.2 G-DEC rules (b) to (d); CR 2.2 item 34; SPEC 3.6.2."""
    l0 = ex["l0_q50_per"]; l1 = ex["l1_q50_per"]; ham = ex["ham_q50_per"]; K = ex["K"]
    n_pass = len(bounds) - 1
    rec = {"n_passes": n_pass, "snaps_per_pass_median": float(np.median(np.diff(bounds))) if n_pass >= 1 else None,
           "phase_frac": {}, "k_rel_drop": {}, "l0_rel_drop": {}, "slope_l0": None, "slope_l1": None, "hamming_sign": None,
           "surrogate_p05_slope": None}
    if n_pass < 1:
        return rec
    max_off = int(max(0, np.min(np.diff(bounds)) - min_run))
    for p in range(0, max_off + 1):
        falls, resets, kd, ld = [], [], [], []
        for i in range(n_pass):
            a, b = int(bounds[i]) + p, int(bounds[i + 1])
            if a + min_run > b:
                continue
            seg = l0[a:a + min_run]
            if np.any(np.isnan(seg)):
                continue
            fall = bool(np.all(np.diff(seg) < 0))
            falls.append(fall)
            nxt = l0[b] if b < len(l0) else np.nan
            resets.append(bool(not np.isnan(nxt) and nxt > seg[-1]))
            if fall:
                kd.append((K[a] - K[a + min_run - 1]) / K[a] if K[a] else 0.0)
                ld.append((seg[0] - seg[-1]) / seg[0] if seg[0] else 0.0)
        if falls:
            f = float(np.mean([x and y for x, y in zip(falls, resets)]))
            rec["phase_frac"][p] = f
            rec["k_rel_drop"][p] = float(np.median(kd)) if kd else None
            rec["l0_rel_drop"][p] = float(np.median(ld)) if ld else None
    rec["slope_l0"] = _within_pass_slope(l0, bounds)
    rec["slope_l1"] = _within_pass_slope(l1, bounds)
    hs = _within_pass_slope(ham, bounds)
    rec["hamming_sign"] = None if np.isnan(hs) else int(np.sign(hs))
    rng = np.random.default_rng(seed)
    sur = []
    for _ in range(n_surrogates):
        xs = l0.copy()
        for i in range(n_pass):
            a, b = int(bounds[i]), int(bounds[i + 1])
            xs[a:b] = xs[a:b][rng.permutation(b - a)]
        sur.append(_within_pass_slope(xs, bounds))
    sur = np.array([s for s in sur if not np.isnan(s)])
    rec["surrogate_p05_slope"] = float(np.quantile(sur, 0.05)) if len(sur) else None
    rec["surrogate_ok"] = bool(len(sur) and not np.isnan(rec["slope_l0"]) and rec["slope_l0"] < rec["surrogate_p05_slope"])
    return rec


def whole_cell_slope_test(ex: dict, *, n_surrogates: int = GDEC_N_SURROGATES, seed: int = NL.SEED_SURROGATE) -> dict:
    """The whole-cell slope of l0_q50_per against its own phase-randomized surrogates' 95th percentile in
    magnitude (G-DEC (e) and the control when no boundary exists; SPEC_review_al_kindi.md item 4 (c))."""
    x = ex["l0_q50_per"][:-1]
    x = np.where(np.isnan(x), np.nanmean(x) if np.any(~np.isnan(x)) else 0.0, x)
    s = _slope(x)
    sur = NL.surrogates(x, n_surrogates, seed) if len(x) >= 3 else np.zeros((0, len(x)))
    mags = np.array([abs(_slope(r)) for r in sur]) if len(sur) else np.zeros(0)
    p95 = float(np.quantile(mags, 0.95)) if len(mags) else None
    return {"slope": s, "p95_abs": p95, "significant": bool(p95 is not None and abs(s) > p95), "sign": int(np.sign(s)) if not np.isnan(s) else 0}


GDEC_COLUMNS = ("kernel", "cell_id", "role_in_test", "n_passes", "snaps_per_pass_median", "phase_offset", "run_length", "slope_l0",
                "slope_l1", "hamming_sign", "k_rel_drop", "l0_rel_drop", "surrogate_p05_slope", "verdict")


def gate_gdec(out: Path, cells: list[dict] | None = None, pass_table: dict | None = None, gp_csv: Path | None = None, *,
              kernel: str = GDEC_KERNEL, control_kernel: str = GDEC_CONTROL, boundary_source: str = GDEC_BOUNDARY_SOURCE,
              jump_factor: float = GC_JUMP_PREDICTED, jump_detect_ratio: float = GC_JUMP_DETECT_RATIO, min_run: int = GDEC_MIN_RUN,
              min_reps: int = GDEC_MIN_REPS, n_surrogates: int = GDEC_N_SURROGATES, phase_pairs: int = GDEC_PHASE_PAIRS,
              pass_frac: float = GDEC_PASS_FRAC, idle_slope_rule: str = GDEC_IDLE_SLOPE_RULE, seed_offset: int = 0) -> Path:
    """G-DEC, decay validity, for one kernel (default floyd, the decay exhibit) with gibbs
    as the no-slope control. Admission: the kernel's within_pass_verdict must be
    GP_ADMITTED, else GDEC_NOT_RESOLVED for every rep. Pass boundaries: boundary_source =
    'k_jump' (the first row of every run of snapshots with K_t >= jump_detect_ratio * cell median K,
    the G-C detector with its corrected detection ratio; jump_factor = 2.0 is the recorded
    prediction) or 'period' (every round(T_pairs) rows from seq_first + phase_pairs, phase_pairs = 0,
    SPEC_review_al_kindi.md item 4 (a)). (a) the series is the per-snapshot median l0 over persistent
    pages (l0_q50_per), l1_q50_per second, hamming as a direction check (sign of its slope reported);
    (b) inside a pass the series falls strictly across at least min_run consecutive snapshots
    starting at the same phase (offset from the boundary) and resets upward at the next boundary,
    in at least pass_frac of the cell's passes, in at least min_reps of the reps at one common phase;
    (c) over the same span the relative drop of K is not larger than the relative drop of median l0,
    else GDEC_NO_BEYOND_BREADTH; (d) a time-block-shuffle surrogate: the snapshot order is permuted
    within each pass block (n_surrogates times), the mean within-pass slope recomputed; the observed
    slope must be below the surrogates' 5th percentile, else GDEC_NO_DECAY; (e) the idle cells'
    whole-cell slope of the same series must not have the same sign while exceeding its own
    phase-randomized surrogate 95th percentile in magnitude, else GDEC_NO_BEYOND_FLOOR (with no idle
    cell, (e) is not_run and the verdict carries the suffix ' (floor unmeasured)'). A cell with fewer
    than two detected boundaries is GDEC_NO_BOUNDARY (item 4 (b)); the control kernel, and (e), are
    evaluated as a whole-cell slope against phase-randomized surrogates when no boundary exists
    (item 4 (c)), reported with the expectation 'no slope'. Citation: P2 Sec. V 5.2 G-DEC; CR 2.2 item 34."""
    out = Path(out)
    if cells is None:
        cells = S.load_cells(out / "cells.csv")
    cells, ex_hard, ex_pair, _ = S.admissible_cells(out, cells, "content")
    if pass_table is None:
        pass_table = load_pass_table(out / "inputs" / "pass_table.csv")
    gpv = gp_kernel_verdicts(out) if gp_csv is None else {r["kernel"]: r for r in S.read_csv(gp_csv) if r.get("cell_id") == "all"}
    idle = [c for c in cells if c["role"] == "idle"]
    rows = []
    seed = NL.SEED_ORDER + seed_offset
    # (e) the idle clause
    idle_res = [whole_cell_slope_test(S.load_extract_cached(out, c["cell_id"]), n_surrogates=n_surrogates, seed=NL.SEED_SURROGATE + seed_offset) for c in idle]
    for c, r in zip(idle, idle_res):
        rows.append({"kernel": "idle", "cell_id": c["cell_id"], "role_in_test": "idle", "n_passes": 0, "slope_l0": r["slope"],
                     "surrogate_p05_slope": r["p95_abs"], "hamming_sign": r["sign"],
                     "verdict": "slope significant" if r["significant"] else "no slope"})
    n_idle_neg = sum(1 for r in idle_res if r["significant"] and r["sign"] < 0)
    idle_neg_sig = (n_idle_neg >= 1) if idle_slope_rule == "any" else (n_idle_neg >= min_reps)

    def kernel_block(k: str, role: str) -> str:
        kc = sorted([c for c in cells if c["kernel"] == k], key=lambda c: c["rep"])
        if not kc:
            rows.append({"kernel": k, "cell_id": "all", "role_in_test": role, "verdict": V.not_run(f"no admissible {k} cell")})
            return V.not_run(f"no admissible {k} cell")
        wpv = (gpv.get(k) or {}).get("within_pass_verdict", V.GP_UNDECLARED)
        entry = pass_table.get(k)
        T_pairs = None
        recs = {}
        for c in kc:
            ex = S.load_extract_cached(out, c["cell_id"])
            if entry and entry.passes:
                T_pairs = int(S.load_sidecar(out, c["cell_id"])["n_pairs"]) / entry.passes
            b = boundaries_of(ex, source=boundary_source, detect_ratio=jump_detect_ratio, T_pairs=T_pairs, phase_pairs=phase_pairs)
            recs[c["cell_id"]] = (ex, b)
        admitted = wpv.split(" (")[0] == V.GP_ADMITTED
        if role == "exhibit" and not admitted:
            for c in kc:
                rows.append({"kernel": k, "cell_id": c["cell_id"], "role_in_test": role, "n_passes": max(0, len(recs[c["cell_id"]][1]) - 1),
                             "verdict": V.GDEC_NOT_RESOLVED})
            rows.append({"kernel": k, "cell_id": "all", "role_in_test": role, "verdict": V.GDEC_NOT_RESOLVED})
            return V.GDEC_NOT_RESOLVED
        per = {}
        for c in kc:
            ex, b = recs[c["cell_id"]]
            if len(b) < 2:
                w = whole_cell_slope_test(ex, n_surrogates=n_surrogates, seed=NL.SEED_SURROGATE + seed_offset)
                per[c["cell_id"]] = {"n_passes": 0, "slope_l0": w["slope"], "surrogate_p05_slope": w["p95_abs"], "hamming_sign": w["sign"],
                                     "verdict": (V.GDEC_NO_BOUNDARY if role == "exhibit" else ("slope significant" if w["significant"] else "no slope")),
                                     "whole": w}
            else:
                per[c["cell_id"]] = dec_cell(ex, b, min_run=min_run, n_surrogates=n_surrogates, seed=seed, pass_frac=pass_frac)
        # the common phase across reps
        with_b = [cid for cid, r in per.items() if r.get("n_passes", 0) >= 1]
        best_p, best_n, best_f = None, 0, -1.0
        phases = sorted({p for cid in with_b for p in per[cid]["phase_frac"]})
        for p in phases:
            n_ok = sum(1 for cid in with_b if per[cid]["phase_frac"].get(p, 0.0) >= pass_frac)
            f_mean = float(np.mean([per[cid]["phase_frac"].get(p, 0.0) for cid in with_b])) if with_b else 0.0
            if (n_ok, f_mean) > (best_n, best_f):
                best_p, best_n, best_f = p, n_ok, f_mean
        verdicts = []
        for c in kc:
            r = per[c["cell_id"]]
            if r.get("n_passes", 0) < 1:
                rows.append({"kernel": k, "cell_id": c["cell_id"], "role_in_test": role, "n_passes": 0, "slope_l0": r["slope_l0"],
                             "surrogate_p05_slope": r["surrogate_p05_slope"], "hamming_sign": r["hamming_sign"], "verdict": r["verdict"]})
                verdicts.append(r["verdict"])
                continue
            fall = best_p is not None and r["phase_frac"].get(best_p, 0.0) >= pass_frac
            kd = r["k_rel_drop"].get(best_p) if best_p is not None else None
            ld = r["l0_rel_drop"].get(best_p) if best_p is not None else None
            if not fall or not r.get("surrogate_ok", False):
                v = V.GDEC_NO_DECAY
            elif kd is not None and ld is not None and kd > ld:
                v = V.GDEC_NO_BEYOND_BREADTH
            elif idle_neg_sig:
                v = V.GDEC_NO_BEYOND_FLOOR
            else:
                v = V.GDEC_DECAY
            if not idle:
                v = v + " (floor unmeasured)"
            verdicts.append(v)
            rows.append({"kernel": k, "cell_id": c["cell_id"], "role_in_test": role, "n_passes": r["n_passes"],
                         "snaps_per_pass_median": r["snaps_per_pass_median"], "phase_offset": best_p, "run_length": min_run,
                         "slope_l0": r["slope_l0"], "slope_l1": r["slope_l1"], "hamming_sign": r["hamming_sign"], "k_rel_drop": kd,
                         "l0_rel_drop": ld, "surrogate_p05_slope": r["surrogate_p05_slope"], "verdict": v})
        heads = [v.split(" (")[0] for v in verdicts]
        n_decay = sum(1 for h in heads if h == V.GDEC_DECAY)
        if role == "control":
            n_sig = sum(1 for h in heads if h in ("slope significant", V.GDEC_DECAY))
            kv = f"slope significant ({n_sig} of {len(heads)} reps)" if n_sig >= min_reps else f"no slope ({n_sig} of {len(heads)} reps with a slope)"
        elif n_decay >= min_reps:
            kv = V.GDEC_DECAY
        elif any(h == V.GDEC_NO_BEYOND_FLOOR for h in heads) and sum(1 for h in heads if h in (V.GDEC_DECAY, V.GDEC_NO_BEYOND_FLOOR)) >= min_reps:
            kv = V.GDEC_NO_BEYOND_FLOOR
        elif any(h == V.GDEC_NO_BEYOND_BREADTH for h in heads) and sum(1 for h in heads if h in (V.GDEC_DECAY, V.GDEC_NO_BEYOND_BREADTH)) >= min_reps:
            kv = V.GDEC_NO_BEYOND_BREADTH
        elif all(h == V.GDEC_NO_BOUNDARY for h in heads):
            kv = V.GDEC_NO_BOUNDARY
        else:
            kv = V.GDEC_NO_DECAY
        if not idle and role == "exhibit":
            kv = kv + " (floor unmeasured)"
        rows.append({"kernel": k, "cell_id": "all", "role_in_test": role, "n_passes": best_n, "phase_offset": best_p, "run_length": min_run, "verdict": kv})
        return kv

    v_exhibit = kernel_block(kernel, "exhibit")
    v_control = kernel_block(control_kernel, "control")
    p = S.write_csv(out / "gates" / "gdec.csv", GDEC_COLUMNS, rows)
    S.write_params(p, "plan11.gdec.v1",
                   {"kernel": kernel, "control_kernel": control_kernel, "boundary_source": boundary_source, "jump_predicted": jump_factor,
                    "jump_detect_ratio": jump_detect_ratio, "phase_pairs": phase_pairs, "min_run": min_run, "min_reps": min_reps,
                    "n_surrogates": n_surrogates, "pass_frac": pass_frac, "idle_slope_rule": idle_slope_rule, "n_idle_negative_slopes": n_idle_neg, "series": ["l0_q50_per", "l1_q50_per", "ham_q50_per (sign)"],
                    "n_idle_cells": len(idle), "seed_order": seed, "verdict_exhibit": v_exhibit, "verdict_control": v_control,
                    "excluded_cells_pair_rungs": ex_pair,
                    "inputs_sha256": S.inputs_sha256([out / "cells.csv", out / "inputs" / "pass_table.csv", out / "gates" / "gp.csv"], out)},
                   CIT_GDEC)
    return p


# --------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="gates_readings.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("gj"); j.add_argument("--out", required=True); j.add_argument("--k-factor", type=float, default=GJ_K_FACTOR)
    d = sub.add_parser("gdec"); d.add_argument("--out", required=True)
    d.add_argument("--kernel", default=GDEC_KERNEL); d.add_argument("--control-kernel", default=GDEC_CONTROL)
    d.add_argument("--boundary-source", default=GDEC_BOUNDARY_SOURCE, choices=("k_jump", "period"))
    d.add_argument("--jump-factor", type=float, default=GC_JUMP_PREDICTED); d.add_argument("--jump-detect-ratio", type=float, default=GC_JUMP_DETECT_RATIO)
    d.add_argument("--min-run", type=int, default=GDEC_MIN_RUN); d.add_argument("--min-reps", type=int, default=GDEC_MIN_REPS)
    d.add_argument("--n-surrogates", type=int, default=GDEC_N_SURROGATES); d.add_argument("--phase-pairs", type=int, default=GDEC_PHASE_PAIRS)
    d.add_argument("--idle-slope-rule", default=GDEC_IDLE_SLOPE_RULE, choices=("any", "min_reps"))
    d.add_argument("--pass-frac", type=float, default=GDEC_PASS_FRAC,
                   help="the fall must hold in at least this fraction of a cell's passes (SPEC_epoch2 B11; CHECK_3 M11)")
    for sp in (j, d):
        sp.add_argument("--seed-offset", type=int, default=0)
    args = ap.parse_args(argv)
    out = Path(args.out)
    if not (out / "cells.csv").is_file():
        print(f"missing input: {out / 'cells.csv'}", file=sys.stderr); return 2
    try:
        if args.cmd == "gj":
            print(gate_gj(out, k_factor=args.k_factor))
        else:
            print(gate_gdec(out, kernel=args.kernel, control_kernel=args.control_kernel, boundary_source=args.boundary_source,
                            jump_factor=args.jump_factor, jump_detect_ratio=args.jump_detect_ratio, min_run=args.min_run,
                            min_reps=args.min_reps, n_surrogates=args.n_surrogates, phase_pairs=args.phase_pairs,
                            idle_slope_rule=args.idle_slope_rule, seed_offset=args.seed_offset, pass_frac=args.pass_frac))
    except FileNotFoundError as e:
        print(f"missing input: {e}", file=sys.stderr); return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
