#!/usr/bin/env python3
"""gates_calibration.py -- the pass table, G-P, G-C and the alias falsifier (SPEC section 3.4).

Citation: P2_STRUCTURE.md section V 5.2 (G-C calibration pulse; G-P pass period) and section 2
falsifier (2); CR 2.2 items 22 (G-C) and 23 (G-P, the alias falsifier); K2 move 3;
SPEC_review_al_kindi.md items 1, 2 (G-C's detection threshold and the absence verdict) and 5
(alias falsifier (b)).
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plan11_encoding_ladder import series as S
from plan11_encoding_ladder import verdicts as V
from plan11_encoding_ladder.series import schema

DT_BRACKET = tuple(schema.DT_BRACKET_S)
DURATION_S = schema.DURATION_S
GP_MIN_PASSES_RHYTHM = 5           # section 8 item 17
GP_MIN_SNAPS_WITHIN_PASS = 3
GP_T_RESOLVABLE = 4.0              # T >= 4 dt
GP_T_MARGINAL = 2.0                # 2 dt <= T < 4 dt
GC_PULSE_KERNEL = "gemm"
GC_JUMP_PREDICTED = 2.0            # the source prediction (two to one), recorded
GC_JUMP_DETECT_RATIO = 1.5         # the detection rule: midpoint between 1 and 2 (al-Kindi item 1)
GC_JUMP_REFERENCE = "cell_median_K"
GC_J_DIP_MAX = 0.75
GC_DIP_WINDOW_PAIRS = 1
GC_MIN_EVENTS_PER_REP = 1
GC_CONTENT_KERNELS = ("gibbs", "histogram", "gemm")
GC_PAIRING = "by_rep_index"
GC_CONTENT_PAGE_SET = "persistent"
GC_PULSE_FULL_FOOTPRINT_PAGES = 4096   # P2 Table 3 gemm row (INFERRED); al-Kindi item 2
GC_REGIME_BAND_RATIO = 1.5             # K_median / 4096 in [1/1.5, 1.5]
GC_REGIME_J_MIN = 0.75
ALIAS_R2_THRESHOLD = 0.5               # section 8 item 18
PASS_TABLE_COLUMNS = ("kernel", "passes_per_600s", "source", "notes")

CIT_GP = "P2 Sec. V 5.2 G-P; CR 2.2 item 23; K2 Sec. 2 'readings'"
CIT_GC = ("P2 Sec. V 5.2 G-C; CR 2.2 item 22; K2 move 3; SPEC_review_al_kindi.md items 1 and 2 "
          "(pulse shape two-to-one and one-half INFERRED from kernel_gemm_v2.c lines 142-143, K2 Sec. 2 rung 1 (b))")
CIT_ALIAS = "P2 Sec. 2 falsifier (2); CR 2.2 item 23 last clause; SPEC_review_al_kindi.md item 5"


# --------------------------------------------------------------------------- the pass table (3.4.1)

@dataclass
class PassEntry:
    passes: float | None
    source_kind: str          # "declared" | "inferred" | "undeclared"
    note: str = ""


def write_pass_table_template(path: Path) -> Path:
    """inputs/pass_table.csv: nbody 6147 'declared: kernel_nbody_v2_metadata.json (AA A7)', the other
    eleven blank with source 'undeclared' (SPEC 3.4.1; P2 Table 3). An 'inferred:' row must come from
    source and base parameters (VME 1.5), never from the trajectory or any toolkit output
    (SPEC_review_al_farabi.md section 3 item 4). The column keeps its name ``passes_per_600s``; the
    count is per cell duration, which is 600 s on the real corpus (every sidecar declares it) and the
    extractor's ``--duration-s`` on a synthetic one (SPEC_epoch2 B12; CHECK_3 M12)."""
    rows = []
    for k, _ in schema.KERNELS:
        if k == "nbody":
            rows.append({"kernel": k, "passes_per_600s": 6147, "source": "declared: kernel_nbody_v2_metadata.json (AA A7)",
                         "notes": "6,147 steps per 600 s; about 5 steps per snapshot (P2 Table 3)"})
        else:
            rows.append({"kernel": k, "passes_per_600s": "", "source": "undeclared",
                         "notes": "fill from source and base parameters only ('inferred: <reason>') or a declared counter ('declared: <file>'); never from the trajectory"})
    return S.write_csv(path, PASS_TABLE_COLUMNS, rows)


def load_pass_table(path: Path | None) -> dict:
    """{kernel: PassEntry} (SPEC 3.4.1). Missing file -> every kernel undeclared."""
    table = {k: PassEntry(None, "undeclared", "no pass table") for k, _ in schema.KERNELS}
    if path is None or not Path(path).is_file():
        return table
    for r in S.read_csv(path):
        k = r["kernel"]
        v = r.get("passes_per_600s", "")
        src = (r.get("source") or "undeclared").strip()
        kind = "declared" if src.startswith("declared") else ("inferred" if src.startswith("inferred") else "undeclared")
        passes = None
        if v not in ("", None) and kind != "undeclared":
            try:
                passes = float(v)
            except ValueError:
                passes = None
        if passes is None or passes <= 0:
            kind = "undeclared"
            passes = None
        table[k] = PassEntry(passes, kind, r.get("notes", ""))
    return table


def _suffix(kind: str) -> str:
    return " (INFERRED)" if kind == "inferred" else ""


# --------------------------------------------------------------------------- G-P (3.4.2)

def gp_verdict_T(T: float, dt: float) -> str:
    """DSP's G-A verdicts on T against dt: T >= 4 dt resolvable; 2 dt <= T < 4 dt marginal;
    T < 2 dt aliased by design (CR 2.2 item 23)."""
    if T >= GP_T_RESOLVABLE * dt:
        return V.GP_RESOLVABLE
    if T >= GP_T_MARGINAL * dt:
        return V.GP_MARGINAL
    return V.GP_ALIASED


def gp_cell(n_pairs: int, entry: PassEntry, *, dt_bracket=DT_BRACKET, min_passes_rhythm=GP_MIN_PASSES_RHYTHM,
            min_snaps_within_pass=GP_MIN_SNAPS_WITHIN_PASS, duration_s: float = DURATION_S) -> dict:
    """One cell's G-P record in pair units (T_pairs = n_pairs / passes) and in seconds
    (T_seconds = duration_s / passes; ``duration_s`` the cell's declared duration from its sidecar,
    600 on the real corpus, SPEC_epoch2 B12) at both ends of the dt bracket (SPEC 3.4.2)."""
    if entry.passes is None:
        return {"passes_per_600s": None, "source_kind": entry.source_kind, "T_seconds": None, "T_pairs": None,
                "verdict_pairs": V.GP_UNDECLARED, "rhythm_verdict": V.GP_UNDECLARED, "within_pass_verdict": V.GP_UNDECLARED,
                "verdict_dt_0500": V.GP_UNDECLARED, "verdict_dt_0644": V.GP_UNDECLARED}
    sfx = _suffix(entry.source_kind)
    T_pairs = n_pairs / entry.passes
    T_s = float(duration_s) / entry.passes
    return {"passes_per_600s": entry.passes, "source_kind": entry.source_kind, "T_seconds": T_s, "T_pairs": T_pairs,
            "verdict_pairs": gp_verdict_T(T_pairs, 1.0) + sfx,
            "rhythm_verdict": (V.GP_ADMITTED if entry.passes >= min_passes_rhythm else V.GP_RHYTHM_UNDERSAMPLED) + sfx,
            "within_pass_verdict": (V.GP_ADMITTED if T_pairs >= min_snaps_within_pass else V.GP_PASS_ALIASED) + sfx,
            "verdict_dt_0500": gp_verdict_T(T_s, dt_bracket[0]) + sfx,
            "verdict_dt_0644": gp_verdict_T(T_s, dt_bracket[1]) + sfx}


GP_COLUMNS = ("kernel", "cell_id", "n_pairs", "passes_per_600s", "source_kind", "T_seconds", "T_pairs", "dt_est_s",
              "verdict_pairs", "rhythm_verdict", "within_pass_verdict", "verdict_dt_0500", "verdict_dt_0644")


def gate_gp(out: Path, cells: list[dict] | None = None, pass_table: dict | None = None, *, dt_bracket=DT_BRACKET,
            min_passes_rhythm: int = GP_MIN_PASSES_RHYTHM, min_snaps_within_pass: int = GP_MIN_SNAPS_WITHIN_PASS) -> Path:
    """G-P, pass period, per cell (rolled up per kernel by majority; a kernel whose cells
    disagree is reported with the count). Pair units: T_pairs = n_pairs / passes.
    verdict_pairs: T_pairs >= 4 -> GP_RESOLVABLE; 2 <= T_pairs < 4 -> GP_MARGINAL;
    T_pairs < 2 -> GP_ALIASED; no count -> GP_UNDECLARED. rhythm_verdict: passes >=
    min_passes_rhythm -> GP_ADMITTED else GP_RHYTHM_UNDERSAMPLED. within_pass_verdict:
    T_pairs >= min_snaps_within_pass -> GP_ADMITTED else GP_PASS_ALIASED. Seconds
    representation, for the paper's bracket: verdict_dt_0500 and verdict_dt_0644 from
    T_seconds against 4 dt and 2 dt with the same three names. Undeclared kernels: every
    verdict GP_UNDECLARED; their readings stay in every table labelled so (al-Kindi over
    DSP, CR 2.2 item 23). Citation: P2 Sec. V 5.2 G-P; CR 2.2 item 23. Per-kernel roll-up rows
    have cell_id = "all" and carry the majority verdict with the disagreement count in n_pairs.
    ``T_seconds`` is ``duration_s_declared / passes`` from each cell's sidecar (SPEC_epoch2 B12;
    CHECK_3 M12): 600 on the real corpus, the same number as before; a sidecar without the key
    falls back to ``schema.DURATION_S`` and is listed in ``params.duration_s_fallback_cells``."""
    out = Path(out)
    if cells is None:
        cells = S.load_cells(out / "cells.csv")
    cells, _, _, _ = S.admissible_cells(out, cells, None)
    pt_path = out / "inputs" / "pass_table.csv"
    if pass_table is None:
        pass_table = load_pass_table(pt_path)
    rows = []
    by_k = {}
    fallback_cells, durations = [], []
    for c in cells:
        if c["role"] != "kernel":
            continue
        sc = S.load_sidecar(out, c["cell_id"])
        entry = pass_table.get(c["kernel"], PassEntry(None, "undeclared"))
        dur = sc.get("duration_s_declared")
        if dur is None:
            dur = DURATION_S; fallback_cells.append(c["cell_id"])
        durations.append(float(dur))
        rec = gp_cell(int(sc["n_pairs"]), entry, dt_bracket=dt_bracket, min_passes_rhythm=min_passes_rhythm,
                      min_snaps_within_pass=min_snaps_within_pass, duration_s=float(dur))
        row = {"kernel": c["kernel"], "cell_id": c["cell_id"], "n_pairs": int(sc["n_pairs"]), "dt_est_s": sc.get("dt_est_s"), **rec}
        rows.append(row)
        by_k.setdefault(c["kernel"], []).append(row)
    roll = []
    for k, rs in by_k.items():
        r = {"kernel": k, "cell_id": "all", "n_pairs": len(rs), "passes_per_600s": rs[0]["passes_per_600s"],
             "source_kind": rs[0]["source_kind"], "T_seconds": rs[0]["T_seconds"],
             "T_pairs": float(np.median([x["T_pairs"] for x in rs])) if rs[0]["T_pairs"] is not None else None,
             "dt_est_s": float(np.median([float(x["dt_est_s"]) for x in rs if x["dt_est_s"] is not None])) if rs else None}
        for col in ("verdict_pairs", "rhythm_verdict", "within_pass_verdict", "verdict_dt_0500", "verdict_dt_0644"):
            cnt = Counter(x[col] for x in rs).most_common()
            r[col] = cnt[0][0] if len(cnt) == 1 else f"{cnt[0][0]} ({cnt[0][1]} of {len(rs)} cells)"
        roll.append(r)
    p = S.write_csv(out / "gates" / "gp.csv", GP_COLUMNS, rows + roll)
    S.write_params(p, "plan11.gp.v1", {"dt_bracket": list(dt_bracket), "min_passes_rhythm": min_passes_rhythm,
                                       "min_snaps_within_pass": min_snaps_within_pass,
                                       "duration_source": "sidecar duration_s_declared",      # SPEC_epoch2 B12
                                       "duration_s_min": min(durations) if durations else None,
                                       "duration_s_max": max(durations) if durations else None,
                                       "duration_s_fallback": DURATION_S, "duration_s_fallback_cells": fallback_cells,
                                       "pass_table": str(pt_path) if pt_path.is_file() else None,
                                       "inputs_sha256": S.inputs_sha256([out / "cells.csv", pt_path], out)}, CIT_GP)
    return p


def gp_kernel_verdicts(out: Path) -> dict:
    """{kernel: the roll-up row of gates/gp.csv} (cell_id == 'all'), or {} when G-P has not run."""
    p = Path(out) / "gates" / "gp.csv"
    if not p.is_file():
        return {}
    return {r["kernel"]: r for r in S.read_csv(p) if r.get("cell_id") == "all"}


# --------------------------------------------------------------------------- G-C (3.4.3)

def k_jump_events(K: np.ndarray, *, detect_ratio: float = GC_JUMP_DETECT_RATIO,
                  reference: str = GC_JUMP_REFERENCE) -> tuple:
    """Indices where K_t >= detect_ratio * reference (reference = the cell's median K, or the median
    of the preceding 8 snapshots for 'local_median_8'); returns (event indices, max ratio).
    Citation: P2 Sec. V 5.2 G-C; CR 2.2 item 22 (the two-to-one prediction); SPEC_review_al_kindi.md item 1 (detection at 1.5)."""
    K = np.asarray(K, dtype=np.float64)
    if reference == "cell_median_K":
        ref = np.full(len(K), np.median(K))
    elif reference == "local_median_8":
        ref = np.array([np.median(K[max(0, i - 8):i]) if i > 0 else np.nan for i in range(len(K))])
    elif reference == "cell_q25_K":
        ref = np.full(len(K), np.quantile(K, 0.25))
    else:
        raise ValueError(reference)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = K / ref
    ratio[~np.isfinite(ratio)] = 0.0
    return np.flatnonzero(ratio >= detect_ratio), float(np.nanmax(ratio)) if len(ratio) else 0.0


def gate_gc(out: Path, cells: list[dict] | None = None, rung: str = "apf", *, pulse_kernel: str = GC_PULSE_KERNEL,
            jump_factor: float = GC_JUMP_PREDICTED, jump_detect_ratio: float = GC_JUMP_DETECT_RATIO,
            jump_reference: str = GC_JUMP_REFERENCE, j_dip_max: float = GC_J_DIP_MAX,
            dip_window_pairs: int = GC_DIP_WINDOW_PAIRS, min_events_per_rep: int = GC_MIN_EVENTS_PER_REP,
            content_kernels=GC_CONTENT_KERNELS, pairing: str = GC_PAIRING, content_page_set: str = GC_CONTENT_PAGE_SET,
            pulse_full_footprint_pages: int = GC_PULSE_FULL_FOOTPRINT_PAGES, regime_band_ratio: float = GC_REGIME_BAND_RATIO,
            regime_j_min: float = GC_REGIME_J_MIN) -> Path:
    """G-C, calibration pulse, per rung. Refuses the whole rung (GC_DISCONNECTED) when the
    pulse is missing in any rep; every negative of that rung is void until fixed.
    apf: in every cell of pulse_kernel, at least min_events_per_rep snapshots with
      K_t >= jump_detect_ratio * reference (reference = the cell's median K, or the median K of
      the preceding 8 snapshots when jump_reference='local_median_8'). The source prediction is
      jump_factor = 2.0 (recorded); detection uses jump_detect_ratio = 1.5, the midpoint on the
      ratio scale, because the floor F sits in both levels and (2K + F)/(K + F) < 2 always
      (SPEC_review_al_kindi.md item 1); the observed per-rep maximum ratio is written as stat_a.
    persist: the apf event and, within +/- dip_window_pairs snapshots of it, J <= j_dip_max
      (the dip toward one half at the pass boundary), in every rep.
    wapf: the definition names no pulse for wAPF; the apf rule applied to the wapf series
      is the default (section 8).
    content: the orderings mean_abs(gibbs) < mean_abs(histogram) < mean_abs(gemm) and
      l0(histogram) < l0(gibbs) < l0(gemm), where a cell's statistic is the median over
      seqs of l1_q50_<set> / 4096 (mean_abs = l1 / 4096, positional.rs) and of l0_q50_<set>,
      set = 'per' (persistent pages) or 'all'; pairing='by_rep_index' compares rep r of the
      three kernels for every rep index present in all three and needs all of them; pairing='envelope'
      needs max(gibbs cells) < min(histogram cells) < ... over all cells.
    combined: pass iff the four rungs pass (a not-applicable rung makes combined not applicable;
      any disconnected rung makes it disconnected).
    Absence verdict (SPEC_review_al_kindi.md item 2): when NO rep shows the event, the regime is
      read blind from the extract: if in every rep K_median / pulse_full_footprint_pages lies in
      [1/regime_band_ratio, regime_band_ratio] and J_q50 >= regime_j_min, the verdict is
      GC_ALIASED_BY_DESIGN (the rung is neither passed nor voided); mixed reps stay GC_DISCONNECTED.
    Citation: P2 Sec. V 5.2 G-C; CR 2.2 item 22."""
    out = Path(out)
    if cells is None:
        cells = S.load_cells(out / "cells.csv")
    cells, _, _, _ = S.admissible_cells(out, cells, None)
    kern = [c for c in cells if c["role"] == "kernel"]
    cols = ("rung", "kernel_or_triple", "rep", "n_events", "first_event_seq", "j_at_event", "stat_a", "stat_b", "stat_c", "verdict")
    p = out / "gates" / "gc.csv"
    old = [r for r in S.read_csv(p)] if p.is_file() else []
    old = [r for r in old if r["rung"] != rung and not (rung != "combined" and r["rung"] == "combined")]
    rows = []
    params = {"rung": rung, "pulse_kernel": pulse_kernel, "jump_predicted": jump_factor, "jump_detect_ratio": jump_detect_ratio,
              "jump_reference": jump_reference, "j_dip_max": j_dip_max, "dip_window_pairs": dip_window_pairs,
              "min_events_per_rep": min_events_per_rep, "content_kernels": list(content_kernels), "pairing": pairing,
              "content_page_set": content_page_set, "pulse_full_footprint_pages": pulse_full_footprint_pages,
              "regime_band_ratio": regime_band_ratio, "regime_j_min": regime_j_min,
              "wapf_rule": "the apf rule on the wapf series (ham_sum_all) against its own cell median (section 8 item 16)",
              "inputs_sha256": S.inputs_sha256([out / "cells.csv", out / "gates" / "preconditions.csv"], out)}

    def pulse_rows(rung_: str) -> tuple:
        pcells = sorted([c for c in kern if c["kernel"] == pulse_kernel], key=lambda c: c["rep"])
        if not pcells:
            return [{"rung": rung_, "kernel_or_triple": pulse_kernel, "rep": "all", "verdict": V.not_run(f"no admissible {pulse_kernel} cell")}], V.not_run(f"no admissible {pulse_kernel} cell")
        rs, seen, aliased_ok = [], [], []
        for c in pcells:
            ex = S.load_extract_cached(out, c["cell_id"])
            K = ex["K"]
            series_ = K if rung_ in ("apf", "persist") else ex["ham_sum_all"]
            ev, mx = k_jump_events(series_, detect_ratio=jump_detect_ratio, reference=jump_reference)
            ok = len(ev) >= min_events_per_rep
            j_at = None
            if rung_ == "persist" and len(ev):
                J = ex["J"]
                dips = []
                for e in ev:
                    lo, hi = max(0, e - dip_window_pairs), min(len(J), e + dip_window_pairs + 1)
                    seg = J[lo:hi]
                    seg = seg[~np.isnan(seg)]
                    if len(seg):
                        dips.append(float(np.min(seg)))
                j_at = min(dips) if dips else None
                ok = ok and (j_at is not None) and (j_at <= j_dip_max)
            elif rung_ == "persist":
                ok = False
            kmed = float(np.median(K))
            jq50 = float(np.nanmedian(ex["J"])) if np.any(~np.isnan(ex["J"])) else float("nan")
            band = (1.0 / regime_band_ratio) <= (kmed / pulse_full_footprint_pages) <= regime_band_ratio
            aliased_ok.append(bool(band and jq50 >= regime_j_min))
            seen.append(len(ev) > 0)
            rs.append({"rung": rung_, "kernel_or_triple": pulse_kernel, "rep": c["rep"], "n_events": int(len(ev)),
                       "first_event_seq": int(ex["seq"][ev[0]]) if len(ev) else None, "j_at_event": j_at,
                       "stat_a": mx, "stat_b": kmed / pulse_full_footprint_pages, "stat_c": jq50,
                       "verdict": V.PASS if ok else V.FAIL})
        if all(r["verdict"] == V.PASS for r in rs):
            verdict = V.PASS
        elif not any(seen) and all(aliased_ok):
            verdict = V.GC_ALIASED_BY_DESIGN
        else:
            verdict = V.GC_DISCONNECTED
        rs.append({"rung": rung_, "kernel_or_triple": pulse_kernel, "rep": "all", "n_events": sum(r["n_events"] for r in rs),
                   "verdict": verdict})
        return rs, verdict

    def content_rows() -> tuple:
        sfx = "per" if content_page_set == "persistent" else "all"
        stats = {}
        for k in content_kernels:
            for c in kern:
                if c["kernel"] != k:
                    continue
                ex = S.load_extract_cached(out, c["cell_id"])
                l1 = ex[f"l1_q50_{sfx}"]; l0 = ex[f"l0_q50_{sfx}"]
                stats.setdefault(k, {})[int(c["rep"])] = (float(np.nanmedian(l1)) / 4096.0, float(np.nanmedian(l0)))
        a, b, g = content_kernels
        missing = [k for k in content_kernels if k not in stats]
        if missing:
            v = V.not_run(f"no admissible cell for {','.join(missing)}")
            return [{"rung": "content", "kernel_or_triple": "+".join(content_kernels), "rep": "all", "verdict": v}], v
        rs = []
        if pairing == "by_rep_index":
            reps = sorted(set(stats[a]) & set(stats[b]) & set(stats[g]))
            for r in reps:
                ma, mb, mg = stats[a][r][0], stats[b][r][0], stats[g][r][0]
                la, lb, lg = stats[a][r][1], stats[b][r][1], stats[g][r][1]
                ok = (ma < mb < mg) and (lb < la < lg)
                rs.append({"rung": "content", "kernel_or_triple": "+".join(content_kernels), "rep": r,
                           "stat_a": ma, "stat_b": mb, "stat_c": mg, "n_events": None,
                           "j_at_event": None, "first_event_seq": None, "verdict": V.PASS if ok else V.FAIL})
            n_expected = max(len(stats[a]), len(stats[b]), len(stats[g]))
            all_ok = bool(rs) and all(r["verdict"] == V.PASS for r in rs) and len(reps) == n_expected
        elif pairing == "envelope":
            ma = [v[0] for v in stats[a].values()]; mb = [v[0] for v in stats[b].values()]; mg = [v[0] for v in stats[g].values()]
            la = [v[1] for v in stats[a].values()]; lb = [v[1] for v in stats[b].values()]; lg = [v[1] for v in stats[g].values()]
            all_ok = (max(ma) < min(mb) and max(mb) < min(mg)) and (max(lb) < min(la) and max(la) < min(lg))
            rs.append({"rung": "content", "kernel_or_triple": "+".join(content_kernels), "rep": "envelope",
                       "stat_a": max(ma), "stat_b": min(mb), "stat_c": min(mg), "verdict": V.PASS if all_ok else V.FAIL})
        else:
            raise ValueError(pairing)
        v = V.PASS if all_ok else V.GC_DISCONNECTED
        rs.append({"rung": "content", "kernel_or_triple": "+".join(content_kernels), "rep": "all", "verdict": v})
        return rs, v

    if rung in ("apf", "persist", "wapf"):
        rows, verdict = pulse_rows(rung)
    elif rung == "content":
        rows, verdict = content_rows()
    elif rung == "combined":
        verdicts = {}
        for r in old:
            if r["rep"] == "all" and r["rung"] in ("apf", "wapf", "persist", "content"):
                verdicts[r["rung"]] = r["verdict"]
        missing = [r_ for r_ in ("apf", "wapf", "persist", "content") if r_ not in verdicts]
        if missing:
            verdict = V.not_run(f"G-C not run for {','.join(missing)}")
        elif any(v == V.GC_DISCONNECTED for v in verdicts.values()):
            verdict = V.GC_DISCONNECTED
        elif all(v == V.PASS for v in verdicts.values()):
            verdict = V.PASS
        else:
            verdict = next(v for v in verdicts.values() if v != V.PASS)
        rows = [{"rung": "combined", "kernel_or_triple": "apf+wapf+persist+content", "rep": "all", "verdict": verdict}]
    else:
        raise ValueError(rung)
    S.write_csv(p, cols, old + rows)
    S.write_json(out / "gates" / f"gc.{rung}.params.json", "plan11.gc.v1", params, CIT_GC, {"verdict": verdict, "rows": rows})
    return p


# --------------------------------------------------------------------------- the alias falsifier (3.4.4)

def alias_falsifier(feature_per_cell: dict, dt_per_cell: dict, *, r2_threshold: float = ALIAS_R2_THRESHOLD) -> dict:
    """Regress the per-cell feature on the cell's realized mean interval (dt_est_s) within a
    kernel (8 points). Returns slope, intercept, r2, n, verdict ALIAS_MOVES if r2 >
    r2_threshold else ALIAS_STAYS. The dt spread on this dataset is about 0.635 to 0.674 s
    (890 to 945 pairs); the result is reported with that spread. Citation: P2 Sec. 2
    falsifier (2); CR 2.2 item 23."""
    cells = [c for c in feature_per_cell if c in dt_per_cell and feature_per_cell[c] is not None
             and not (isinstance(feature_per_cell[c], float) and np.isnan(feature_per_cell[c]))]
    x = np.array([float(dt_per_cell[c]) for c in cells]); y = np.array([float(feature_per_cell[c]) for c in cells])
    n = len(x)
    if n < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"slope": None, "intercept": None, "r2": None, "n": n, "dt_min": float(x.min()) if n else None,
                "dt_max": float(x.max()) if n else None, "verdict": V.not_run("fewer than three cells or no spread")}
    A = np.stack([x, np.ones(n)], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    ss_res = float(np.sum((y - yhat) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"slope": float(coef[0]), "intercept": float(coef[1]), "r2": r2, "n": n, "dt_min": float(x.min()),
            "dt_max": float(x.max()), "verdict": V.ALIAS_MOVES if r2 > r2_threshold else V.ALIAS_STAYS}


ALIAS_COLUMNS = ("kind", "rung", "kernel", "pair_or_set", "feature", "n", "slope", "intercept", "r2", "dt_min", "dt_max", "verdict")


def separating_features(out: Path, rung: str = "apf", grid_id: str | None = None) -> list[dict]:
    """Alias falsifier (b), operational definition (SPEC_review_al_kindi.md item 5): at the rung's
    selected point, per cell the window mean of each normalized feature; a feature separates a
    level-matched pair when the two kernels' cell means have disjoint ranges (no threshold).
    Returns [{set, kernel_a, kernel_b, feature, means_a, means_b}]."""
    out = Path(out)
    gid = grid_id or S.selected_grid_id(out, rung, None)[0]
    if gid is None:
        return []
    p = S.features_path(out, rung, gid, True)
    if not p.is_file():
        return []
    feat = S.load_features(p)
    res = []
    for si, lset in enumerate(schema.LEVEL_MATCHED_SETS):
        means = {}
        for k in lset:
            cells = sorted(set(feat["cell_id"][feat["kernel"] == k].tolist()))
            means[k] = {c: np.nanmean(feat["X"][feat["cell_id"] == c], axis=0) for c in cells}
        for i in range(len(lset)):
            for j in range(i + 1, len(lset)):
                ka, kb = lset[i], lset[j]
                if not means[ka] or not means[kb]:
                    continue
                A = np.stack(list(means[ka].values())); B = np.stack(list(means[kb].values()))
                for f, name in enumerate(feat["feature_names"]):
                    a, b = A[:, f], B[:, f]
                    if np.any(np.isnan(a)) or np.any(np.isnan(b)):
                        continue
                    if a.max() < b.min() or b.max() < a.min():
                        res.append({"set": "AB"[si] if si < 2 else str(si), "kernel_a": ka, "kernel_b": kb, "feature": name,
                                    "means_a": {c: float(v[f]) for c, v in means[ka].items()},
                                    "means_b": {c: float(v[f]) for c, v in means[kb].items()}})
    return res


def run_alias(out: Path, *, r2_threshold: float = ALIAS_R2_THRESHOLD, rung_for_table6: str = "apf") -> Path:
    """gates/alias.csv: (a) every G3 cepstral peak frequency per kernel (kind = g3_peak, from
    gates/g3_flags.csv, one row per (rung, kernel)); (b) every feature that separates a level-matched
    pair at APF's selected point (kind = table6_feature). Idempotent: rows of a kind are replaced."""
    out = Path(out)
    dt = {}
    for c in S.load_cells(out / "cells.csv"):
        sp = S.sidecar_path(out, c["cell_id"])
        if sp.is_file():
            dt[c["cell_id"]] = float(S.read_json(sp).get("dt_est_s", float("nan")))
    rows = []
    g3p = out / "gates" / "g3_flags.csv"
    if g3p.is_file():
        by = {}
        for r in S.read_csv(g3p):
            by.setdefault((r["rung"], r["kernel"]), {})[r["cell_id"]] = S.to_float(r.get("ceps_peak_freq_cyc_per_pair"))
        for (rung, k), fpc in by.items():
            res = alias_falsifier(fpc, dt, r2_threshold=r2_threshold)
            rows.append({"kind": "g3_peak", "rung": rung, "kernel": k, "pair_or_set": "", "feature": "ceps_peak_freq_cyc_per_pair", **res})
    for sep in separating_features(out, rung_for_table6):
        for k, means in ((sep["kernel_a"], sep["means_a"]), (sep["kernel_b"], sep["means_b"])):
            res = alias_falsifier(means, dt, r2_threshold=r2_threshold)
            rows.append({"kind": "table6_feature", "rung": rung_for_table6, "kernel": k,
                         "pair_or_set": f"{sep['set']}:{sep['kernel_a']}-{sep['kernel_b']}", "feature": sep["feature"], **res})
    p = S.write_csv(out / "gates" / "alias.csv", ALIAS_COLUMNS, rows)
    S.write_params(p, "plan11.alias.v1", {"r2_threshold": r2_threshold, "rung_for_table6": rung_for_table6,
                                          "separation_rule": "disjoint ranges of the per-cell window means (al-Kindi item 5)",
                                          "dt_source": "sidecar dt_est_s = 600 / n_pairs",
                                          "inputs_sha256": S.inputs_sha256([out / "cells.csv", g3p, out / "gates" / "selection.json"], out)}, CIT_ALIAS)
    return p


# --------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="gates_calibration.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("pass-table"); t.add_argument("--out", required=True)
    t.add_argument("--force", action="store_true", help="overwrite an existing author input (SPEC_epoch2 B20; default: kept)")
    g = sub.add_parser("gc")
    g.add_argument("--out", required=True)
    g.add_argument("--rung", required=True, choices=S.RUNGS)
    g.add_argument("--pulse-kernel", default=GC_PULSE_KERNEL)
    g.add_argument("--jump-factor", type=float, default=GC_JUMP_PREDICTED)
    g.add_argument("--jump-detect-ratio", type=float, default=GC_JUMP_DETECT_RATIO)
    g.add_argument("--jump-reference", default=GC_JUMP_REFERENCE, choices=("cell_median_K", "local_median_8", "cell_q25_K"))
    g.add_argument("--j-dip-max", type=float, default=GC_J_DIP_MAX)
    g.add_argument("--dip-window-pairs", type=int, default=GC_DIP_WINDOW_PAIRS)
    g.add_argument("--min-events-per-rep", type=int, default=GC_MIN_EVENTS_PER_REP)
    g.add_argument("--pairing", default=GC_PAIRING, choices=("by_rep_index", "envelope"))
    g.add_argument("--content-page-set", default=GC_CONTENT_PAGE_SET, choices=("persistent", "all"))
    g.add_argument("--pulse-full-footprint-pages", type=int, default=GC_PULSE_FULL_FOOTPRINT_PAGES)
    p_ = sub.add_parser("gp")
    p_.add_argument("--out", required=True)
    p_.add_argument("--min-passes-rhythm", type=int, default=GP_MIN_PASSES_RHYTHM)
    p_.add_argument("--min-snaps-within-pass", type=int, default=GP_MIN_SNAPS_WITHIN_PASS)
    a = sub.add_parser("alias")
    a.add_argument("--out", required=True)
    a.add_argument("--r2-threshold", type=float, default=ALIAS_R2_THRESHOLD)
    for sp in (g, p_, a, t):
        sp.add_argument("--seed-offset", type=int, default=0)
    args = ap.parse_args(argv)
    out = Path(args.out)
    if args.cmd == "pass-table":
        p = out / "inputs" / "pass_table.csv"
        if p.is_file() and not args.force:
            print(f"kept: author input exists: {p}"); return 0
        print(write_pass_table_template(p)); return 0
    if not (out / "cells.csv").is_file():
        print(f"missing input: {out / 'cells.csv'}", file=sys.stderr); return 2
    try:
        if args.cmd == "gc":
            print(gate_gc(out, rung=args.rung, pulse_kernel=args.pulse_kernel, jump_factor=args.jump_factor,
                          jump_detect_ratio=args.jump_detect_ratio, jump_reference=args.jump_reference,
                          j_dip_max=args.j_dip_max, dip_window_pairs=args.dip_window_pairs,
                          min_events_per_rep=args.min_events_per_rep, pairing=args.pairing,
                          content_page_set=args.content_page_set, pulse_full_footprint_pages=args.pulse_full_footprint_pages))
        elif args.cmd == "gp":
            print(gate_gp(out, min_passes_rhythm=args.min_passes_rhythm, min_snaps_within_pass=args.min_snaps_within_pass))
        elif args.cmd == "alias":
            print(run_alias(out, r2_threshold=args.r2_threshold))
    except FileNotFoundError as e:
        print(f"missing input: {e}", file=sys.stderr); return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
