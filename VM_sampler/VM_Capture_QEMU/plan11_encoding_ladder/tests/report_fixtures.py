#!/usr/bin/env python3
"""report_fixtures.py -- a synthetic `<out>` tree in SPEC's declared record shapes for builder
3's tests (tables, figures, skeleton, driver).

Builder 3 (report), 2026-09-16. The three builders work in parallel, so builder 3 cannot run
builder 1's extractor and builder 2's gates end to end at build time; this module writes what
they would write, column for column as SPEC sections 2.2, 2.3, 2.7, 3.3 to 3.8 and 4.5 declare
(with the review corrections: `failed_source`, `G2_pairs`, `coverage_pairs`,
`n_kernels_na_G1/G2`, `mask_K`/`mask_persist`). The numbers are synthetic; the verdict strings
are SPEC 3.0's. Every knob below exists to make one table cell refuse:

  gc_disconnected_rung   the rung whose gates/gc.csv `rep = all` row reads `disconnected lead`
  gf_void_rung           the rung whose G-F (i) row reads the void string
  unfalsifiable_split    (rung, split, labelspace) whose scores.json b1_g1 reads near_unfalsifiable
  smoke_perm             n_perm below 500 -> b1_g1 reads `not run: N permutations < 500`
  no_selection_rung      a rung absent from gates/selection.json
  best_feasible_rung     a rung whose selection is `best-feasible`
  gdec_verdict           the floyd `cell_id = all` verdict in gates/gdec.csv
  gx_total_leak          gates/gx.csv apf row reads leak + total confound
  relabel_lexer          gates/gk0.csv relabels the lexer `IDLE, measured`
  matched_all_splits     (epoch 2) gates/splits_matched/combined/<gid>/<split>__<ls>/ for the five Table 7
                         combinations, d = 60 reduced to feature_count_used = 36 (SPEC_epoch2 3.5.1, 6.3 item 6)
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PKG = _HERE.parent
if str(_PKG.parent) not in sys.path:
    sys.path.insert(0, str(_PKG.parent))

import csv
import json
from collections import Counter

import numpy as np

from plan11_encoding_ladder._report_common import (  # noqa: E402
    ARCHETYPES, ARCHETYPE_OF, AXIS_OF_RUNG, EXTRACT_COLUMNS, GRID_IDS, KERNELS, N_PAGES, RUNGS,
    write_csv, write_json,
)

FEATURE_NAMES = {
    "apf": [f"apf.k_over_med.{f}" for f in ("mean", "std", "cov", "median", "max", "p95", "peak2med", "duty")],
    "wapf": [f"wapf.wapf_norm.{f}" for f in ("mean", "std", "cov", "median", "max", "p95", "peak2med", "duty")],
    "persist": [f"persist.j_excess.{f}" for f in ("mean", "std", "cov", "median", "max", "p95", "peak2med", "duty")],
}
FEATURE_NAMES["content"] = ([f"content.{ch}.{f}" for ch in ("r_l0_q50_per", "r_l1l0_q50_per", "r_haml0_q50_per")
                             for f in ("mean", "std", "cov", "median", "max", "p95", "peak2med", "duty")]
                            + [f"content.r_{p}_{q}_per.wmean" for p in ("l0", "l1l0", "haml0") for q in ("q05", "q25", "q75", "q95")])
FEATURE_NAMES["combined"] = FEATURE_NAMES["apf"] + FEATURE_NAMES["wapf"] + FEATURE_NAMES["persist"] + FEATURE_NAMES["content"]

PRESET = {  # kernel: (K0, content, pulse_period, pulse_extra)
    "gemm": (4096, "double", 24, 4096), "floyd": (2048, "decay", 10, 2048), "gibbs": (256, "spin", None, 0),
    "nbody": (2048, "double", None, 0), "spmm": (1024, "double", None, 0), "stencil_jacobi": (3072, "double", None, 0),
    "fft": (4096, "double", None, 0), "histogram": (2048, "counter", None, 0), "fem_assembly": (8256, "double", None, 0),
    "lexer": (0, "idle", None, 0), "rmat_gen": (512, "double", None, 0), "bnb_tsp": (4500, "double", None, 0),
}
CONTENT_RATIOS = {"double": (512 / 4096, 86.0, 4.0), "decay": (512 / 4096, 86.0, 4.0), "counter": (1.5 / 4096, 30.0, 2.5),
                  "spin": (5 / 4096, 1.0, 2.0), "idle": (2 / 4096, 8.0, 2.0)}
TRAJ_HEADER = ["seq", "page_index", "hamming", "cosine", "l0", "l1", "l2", "linf", "mean_abs"] + [f"m{i}" for i in range(57)]


def _cells(reps: int, idle: int) -> list[dict]:
    rows = []
    for ki, (k, a) in enumerate(KERNELS):
        for r in range(reps):
            seed = 42 if r == 0 else 1000 * r + 500 + ki
            camp = "01c1" if ki % 3 else ("01c" if ki % 2 else "dwarfs1")
            rows.append({"cell_id": f"{k}__rep{r:02d}__{camp}", "kernel": k, "role": "kernel", "archetype_predicted": a,
                         "seed": seed, "rep": r, "rep_dir": r + 1, "label": "synth", "campaign": camp,
                         "path": "", "traj_file": "", "status": "ok"})
    for r in range(idle):
        rows.append({"cell_id": f"idle__rep{r:02d}__01c", "kernel": "idle", "role": "idle", "archetype_predicted": "control",
                     "seed": "", "rep": r, "rep_dir": r + 1, "label": "idle", "campaign": "01c",
                     "path": "", "traj_file": "", "status": "ok"})
    return rows


def _extract_rows(rng, kernel: str, n_pairs: int, role: str) -> list[dict]:
    K0, content, period, extra = PRESET.get(kernel, (0, "idle", None, 0)) if role == "kernel" else (0, "idle", None, 0)
    F = 150
    r_l0, r_l1l0, r_ham = CONTENT_RATIOS[content]
    rows = []
    Ks = []
    for t in range(n_pairs):
        K = int(K0 * (1 + rng.uniform(-0.02, 0.02))) + F
        if period and t > 0 and t % period == 0:
            K += extra
        Ks.append(K)
    for t in range(n_pairs):
        seq = t + 1
        K = Ks[t]
        last = t == n_pairs - 1
        row = {c: "" for c in EXTRACT_COLUMNS}
        row["seq"] = seq
        row["K"] = K
        l0 = max(1.0, r_l0 * 4096 * (1 + rng.uniform(-0.1, 0.1)))
        if content == "decay" and period:
            l0 = l0 * (0.7 ** (t % period))
        l1 = l0 * r_l1l0
        ham = l0 * r_ham
        for ch, v in (("ham", ham), ("l0", l0), ("l1", l1)):
            row[f"{ch}_sum_all"] = int(v * K)
            for q, f in (("q05", 0.6), ("q25", 0.85), ("q50", 1.0), ("q75", 1.15), ("q95", 1.4)):
                row[f"{ch}_{q}_all"] = f"{v * f:.10g}"
        if not last:
            Kn = Ks[t + 1]
            inter = int(min(K, Kn) * 0.9)
            J = inter / (K + Kn - inter)
            if period and (t + 1) % period == 0 and extra:
                J = K / (K + extra) * 0.98
            row["n_persist"] = inter
            row["n_union"] = K + Kn - inter
            row["J"] = f"{J:.10g}"
            jni = K * Kn / N_PAGES
            row["J_null_inter"] = f"{jni:.10g}"
            row["J_null"] = f"{jni / (K + Kn - jni):.10g}"
            for ch, v in (("ham", ham), ("l0", l0), ("l1", l1)):
                row[f"{ch}_sum_per"] = int(v * inter)
                for q, f in (("q05", 0.6), ("q25", 0.85), ("q50", 1.0), ("q75", 1.15), ("q95", 1.4)):
                    row[f"{ch}_{q}_per"] = f"{v * f:.10g}"
            for p, v in (("r_l0", l0 / 4096), ("r_l1l0", r_l1l0), ("r_haml0", r_ham)):
                for q, f in (("q05", 0.6), ("q25", 0.85), ("q50", 1.0), ("q75", 1.15), ("q95", 1.4)):
                    row[f"{p}_{q}_per"] = f"{v * f:.10g}"
        rows.append(row)
    return rows


def _write_traj(path: Path, n_pairs: int, rng) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(TRAJ_HEADER)
        for t in range(1, n_pairs + 1):
            for pg in rng.integers(0, N_PAGES, size=40):
                w.writerow([t, int(pg), 4, 0.1, 8, 700, 300, 200, f"{700 / 4096:.6f}"] + [0] * 57)


def _scores(rng, cells, split, ls, *, n_perm=500, unfalsifiable=False, d=8, quality=0.9, na=None):
    kernels = sorted({c["kernel"] for c in cells if c["role"] == "kernel"})
    if na:
        return {"schema": "plan11.scores.v1", "params": {"split": split, "labelspace": ls, "n_perm": n_perm},
                "citation": "SPEC 4.5", "status": na, "accuracy": na, "macro_recall": None, "recall_per_class": {},
                "recall_per_kernel": {}, "majority": None, "b1_g1": na, "b1_g1_rank": None, "null_p95": None,
                "with_quarantine": None, "feature_count": d, "dim_status": None, "seed": 20260919, "n_perm": n_perm}, []
    rpk = {k: float(np.clip(rng.normal(quality, 0.1), 0, 1)) for k in kernels}
    preds = []
    for c in cells:
        if c["role"] != "kernel":
            continue
        y = c["archetype_predicted"] if ls == "archetype" else c["kernel"]
        ok = rng.uniform() < rpk[c["kernel"]]
        wrong = [a for a in ARCHETYPES if a not in ("IDLE", y)] if ls == "archetype" else [k for k in kernels if k != y]
        preds.append({"cell_id": c["cell_id"], "kernel": c["kernel"], "archetype": c["archetype_predicted"],
                      "campaign": c["campaign"], "rep": c["rep"], "fold": c["kernel"] if split == "loko" else c["cell_id"],
                      "y_true": y, "y_pred": y if ok else wrong[int(rng.integers(len(wrong)))], "n_windows": 20,
                      "vote_fraction": 0.8, "held_out_campaign": c["campaign"] if split == "loro" else ""})
    acc = sum(p["y_true"] == p["y_pred"] for p in preds) / len(preds)
    classes = sorted({p["y_true"] for p in preds})
    rpc = {cl: sum(p["y_pred"] == cl for p in preds if p["y_true"] == cl) / max(1, sum(p["y_true"] == cl for p in preds)) for cl in classes}
    headline = [cl for cl in classes if ls != "archetype" or cl in ("WORKING-SET", "SCATTER")]
    macro = float(np.mean([rpc[cl] for cl in headline])) if headline else None
    rank = 1 if unfalsifiable else n_perm
    verdict = "near_unfalsifiable" if unfalsifiable else ("pass" if n_perm >= 500 else f"not run: {n_perm} permutations < 500")
    sc = {"schema": "plan11.scores.v1", "params": {"split": split, "labelspace": ls, "n_perm": n_perm, "normalized": True},
          "citation": "SPEC 4.5; CR 2.1 item 9", "status": "ok", "accuracy": acc, "macro_recall": macro,
          "recall_per_class": rpc, "recall_per_kernel": rpk, "majority": 0.5 if ls == "archetype" else 1 / max(1, len(kernels)),
          "b1_g1": verdict, "b1_g1_rank": rank, "null_p95": 0.62 if ls == "archetype" else 0.15,
          "with_quarantine": {"quarantined_features": []}, "feature_count": d, "dim_status": "full vector",
          "seed": 20260919, "n_perm": n_perm}
    return sc, preds


def make_out(out: Path, *, reps: int = 8, idle: int = 8, n_pairs: int = 60, seed: int = 20260916,
             gc_disconnected_rung: str | None = None, gf_void_rung: str | None = None,
             unfalsifiable_split: tuple | None = None, smoke_perm: int = 500, no_selection_rung: str | None = None,
             best_feasible_rung: str | None = None, gdec_verdict: str = "decay", gx_total_leak: bool = False,
             relabel_lexer: bool = True, with_traj: bool = True, with_gj_mask: bool = True,
             with_gates: bool = True, matched_all_splits: bool = True) -> Path:
    """Write the synthetic `<out>` tree; returns `out`."""
    out = Path(out)
    rng = np.random.default_rng(seed)
    cells = _cells(reps, idle)
    # ---- cells.csv, extracts, sidecars, one trajectory for the piano roll
    root = out / "synth_root"
    for c in cells:
        cdir = root / "kernel" / f"kernel_{c['kernel']}_v2" / f"--seed_{c['seed'] or 0}" / f"rep{c['rep_dir']:03d}__{c['label']}"
        cdir.mkdir(parents=True, exist_ok=True)
        traj = f"run_matrix_test1_kernel_{c['kernel']}_v2.npy.substrate_trajectory.csv"
        c["path"] = str(cdir)
        c["traj_file"] = traj
        if with_traj and c["kernel"] == "gemm" and c["rep"] == 0:
            _write_traj(cdir / traj, n_pairs, rng)
        rows = _extract_rows(rng, c["kernel"], n_pairs, c["role"])
        write_csv(out / "extract" / c["cell_id"] / "extract.csv", rows, list(EXTRACT_COLUMNS))
        Ks = [r["K"] for r in rows]
        write_json(out / "extract" / c["cell_id"] / "sidecar.json", {
            "schema": "plan11.extract.v1", "extractor_version": "0.1.0", "cell_id": c["cell_id"], "kernel": c["kernel"],
            "role": c["role"], "archetype_predicted": c["archetype_predicted"], "seed": c["seed"] or None, "rep": c["rep"],
            "rep_dir": c["rep_dir"], "label": c["label"], "campaign": c["campaign"], "path": c["path"], "traj_file": traj,
            "source_bytes": 0, "N": N_PAGES, "page_size": 4096, "bits_per_page": 32768, "duration_s_declared": 600,
            "quantiles": [0.05, 0.25, 0.5, 0.75, 0.95], "persist_side": "t", "header_sha256": "0" * 64, "header_ncols": 66,
            "columns_used": {"seq": 0, "page_index": 1, "hamming": 2, "l0": 4, "l1": 5}, "n_rows_in": 0, "n_rows_skipped": 0,
            "n_rows_dup_page": 0, "n_rows_zero_hamming": 0, "n_rows_zero_l0": 0, "seq_first": 1, "seq_last": n_pairs,
            "n_seq_present": n_pairs, "n_pairs": n_pairs, "n_seq_gaps": 0, "gap_seqs": [], "dt_est_s": 600 / n_pairs,
            "dt_bracket_s": [0.5, 0.644], "K_median": float(np.median(Ks)), "K_max": int(max(Ks)), "apf_max": max(Ks) / N_PAGES,
            "failed_count": 0, "failed_count_source": "declared zero", "status": "ok",
            "started_at": "2026-09-16T00:00:00+00:00", "finished_at": "2026-09-16T00:00:01+00:00", "elapsed_s": 1.0})
    write_csv(out / "cells.csv", cells, ["cell_id", "kernel", "role", "archetype_predicted", "seed", "rep", "rep_dir",
                                         "label", "campaign", "path", "traj_file", "status"])
    (out / "inputs").mkdir(exist_ok=True)
    write_csv(out / "inputs" / "pass_table.csv", [{"kernel": k, "passes_per_600s": 6147 if k == "nbody" else "",
                                                   "source": "declared: kernel_nbody_v2_metadata.json (AA A7)" if k == "nbody" else "undeclared",
                                                   "notes": ""} for k, _ in KERNELS], ["kernel", "passes_per_600s", "source", "notes"])
    if not with_gates:
        return out
    G = out / "gates"
    G.mkdir(exist_ok=True)
    kcells = [c for c in cells if c["role"] == "kernel"]
    # ---- preconditions
    pre = []
    for c in cells:
        pre.append({"cell_id": c["cell_id"], "role": c["role"], "C1": "pass" if c["role"] == "kernel" else "not applicable: control (C1 re-mapped)",
                    "C1_apf_max": 0.03, "C2": "pass", "C2_n_pairs": n_pairs, "C3": "pass", "C3_n_windows_8_4": (n_pairs - 1 - 8) // 4 + 1,
                    "C4": "not applicable: no settle record in the retention layout",
                    "C5": "not run: producer.log is not in the trajectory; the author reads the campaign log",
                    "C6": "pass", "C6_reason": "n_seq_gaps=0", "C7": "pending: filled by the temporal gates (move 6)",
                    "C8": "not applicable: change-point view not in this paper (decided 2026-09-16)",
                    "failed_count": 0, "failed_verdict": "pass", "failed_source": "declared zero: AA A5", "all_hard_pass": "true"})
    write_csv(G / "preconditions.csv", pre, list(pre[0].keys()))
    write_json(G / "preconditions.json", {"schema": "plan11.preconditions.v1", "params": {"C1_ACTIVITY_MIN": 0.02}, "citation": "SPEC 3.3",
                                          "excluded_cells": [], "excluded_cells_pair_rungs": []})
    # ---- gk0
    gk0 = []
    for k, a in KERNELS:
        idle_meas = relabel_lexer and k == "lexer"
        gk0.append({"kernel": k, "source_statement": "yes", "n_cells": reps, "tail_median_K_median": PRESET[k][0] + 150,
                    "tail_median_K_min": PRESET[k][0] + 140, "tail_median_K_max": PRESET[k][0] + 160, "idle_band_edge": 165,
                    "verdict": "IDLE, measured" if idle_meas else "above floor", "archetype_measured": "IDLE" if idle_meas else a})
    gk0.append({"kernel": "idle", "source_statement": "no", "n_cells": idle, "tail_median_K_median": 150, "tail_median_K_min": 145,
                "tail_median_K_max": 155, "idle_band_edge": 165, "verdict": "control", "archetype_measured": "IDLE"})
    write_csv(G / "gk0.csv", gk0, list(gk0[0].keys()))
    # ---- selection (+ grid), G-C, G-F
    sel = {"schema": "plan11.selection.v1", "params": {"grid_rollup": "all_kernels", "rollup_kernel_refusals": "not_applicable"},
           "citation": "SPEC 3.5.7"}
    grid_rows, long_rows = [], []
    for rung in RUNGS:
        chosen = "W16_H8" if rung != "content" else "W8_H4"
        bf = rung == best_feasible_rung
        if rung != no_selection_rung:
            sel[rung] = {"grid_id": chosen, "W": int(chosen[1:].split("_")[0]), "H": int(chosen.split("_H")[1]),
                         "passes_acceptance": not bf, "selected_by": "best-feasible" if bf else "plan03_rule",
                         "gates_passed": "2 of 3" if bf else "3 of 3", "refusal": ""}
        for gid in GRID_IDS:
            W = "whole" if gid == "Wall_Hall" else int(gid[1:].split("_")[0])
            H = "whole" if gid == "Wall_Hall" else int(gid.split("_H")[1])
            g1 = "pass" if gid != "W8_H8" else "fail"
            g4 = "fail" if gid in ("W8_H8", "W16_H16", "W32_H32", "W64_H64", "Wall_Hall") else "pass"
            grid_rows.append({"rung": rung, "axis": AXIS_OF_RUNG[rung], "grid_id": gid, "W": W, "H": H,
                              "hop_ratio": "" if gid == "Wall_Hall" else round(H / W, 2), "n_windows_median": 1 if gid == "Wall_Hall" else (n_pairs - 1 - W) // H + 1,
                              "n_windows_nonoverlap_median": 1 if gid == "Wall_Hall" else (n_pairs - 1) // W,
                              "coverage_pairs": 0.5, "G1": g1, "G2_0500": "not applicable: no kernel with a declared pass period",
                              "G2_0644": "not applicable: no kernel with a declared pass period", "G2_pairs": "not applicable: no kernel with a declared pass period",
                              "G2": "not applicable: no kernel with a declared pass period", "G4": g4, "G5": "pass",
                              "GORD": "order-blind (by construction)" if gid == "Wall_Hall" else ("resolution" if W >= 16 else "order-blind"),
                              "n_kernels_na_G1": 0, "n_kernels_na_G2": 1, "gates_passed": "2 of 2" if (g1 == "pass" and g4 == "pass") else "1 of 2",
                              "selected": "selected" if (gid == chosen and rung != no_selection_rung) else "",
                              "selected_by": ("best-feasible" if bf else "plan03_rule") if gid == chosen else "", "refusal": ""})
            for k, _ in KERNELS:
                long_rows.append({"rung": rung, "grid_id": gid, "kernel": k, "G1": g1, "G4": g4})
    write_csv(G / "table5_grid.csv", grid_rows, list(grid_rows[0].keys()))
    write_csv(G / "table5_long.csv", long_rows, list(long_rows[0].keys()))
    write_json(G / "selection.json", sel)
    gc = []
    for rung in ("apf", "persist", "wapf", "content", "combined"):
        for r in range(reps):
            gc.append({"rung": rung, "kernel_or_triple": "gemm" if rung != "content" else "gibbs<histogram<gemm", "rep": r, "n_events": 2,
                       "first_event_seq": 25, "j_at_event": 0.5, "stat_a": 1.9, "stat_b": "", "stat_c": "", "verdict": "pass"})
        gc.append({"rung": rung, "kernel_or_triple": "all", "rep": "all", "n_events": "", "first_event_seq": "", "j_at_event": "",
                   "stat_a": "", "stat_b": "", "stat_c": "", "verdict": "disconnected lead" if rung == gc_disconnected_rung else "pass"})
    write_csv(G / "gc.csv", gc, list(gc[0].keys()))
    gf = []
    for rung in RUNGS:
        for gid in ("W8_H4", sel.get(rung, {}).get("grid_id", "W8_H4")):
            gf.append({"rung": rung, "grid_id": gid, "part": "i", "kernel": "idle", "n_cells": idle, "score": 0.12, "null_p95": 0.2,
                       "n_inside_envelope": "", "envelope_lo": "", "envelope_hi": "",
                       "verdict": "void: idle reps separable under this rung" if rung == gf_void_rung else "inseparable at floor"})
            for k, _ in KERNELS:
                gf.append({"rung": rung, "grid_id": gid, "part": "ii", "kernel": k, "n_cells": reps, "score": "", "null_p95": "",
                           "n_inside_envelope": reps if k == "lexer" else 0, "envelope_lo": 0.0005, "envelope_hi": 0.0007,
                           "verdict": "at floor in this lead" if k == "lexer" else "pass"})
    write_csv(G / "gf.csv", gf, list(gf[0].keys()))
    # ---- gp, g3, gj, gdec
    gp = []
    for c in kcells:
        decl = c["kernel"] == "nbody"
        gp.append({"kernel": c["kernel"], "cell_id": c["cell_id"], "n_pairs": n_pairs, "passes_per_600s": 6147 if decl else "",
                   "source_kind": "declared" if decl else "undeclared", "T_seconds": 600 / 6147 if decl else "", "T_pairs": n_pairs / 6147 if decl else "",
                   "dt_est_s": 600 / n_pairs, "verdict_pairs": "aliased by design at this size" if decl else "undeclared",
                   "rhythm_verdict": "admitted" if decl else "undeclared", "within_pass_verdict": "pass aliased" if decl else "undeclared",
                   "verdict_dt_0500": "aliased by design at this size" if decl else "undeclared", "verdict_dt_0644": "aliased by design at this size" if decl else "undeclared"})
    write_csv(G / "gp.csv", gp, list(gp[0].keys()))
    g3 = []
    for rung in RUNGS:
        for c in kcells:
            present = c["kernel"] == "gemm"
            g3.append({"rung": rung, "kernel": c["kernel"], "cell_id": c["cell_id"], "ceps_peak_idx": 24, "ceps_peak_freq_cyc_per_pair": 1 / 24,
                       "ceps_snr_db": 9.0 if present else 2.0, "snr_surrogate_p95": 5.0, "cv": 0.05, "cv_shot_floor": 0.02, "cv_ratio": 2.5,
                       "flag_cell": "rhythm flag: present" if present else "rhythm flag: absent",
                       "flag_kernel": "rhythm flag: present" if present else "rhythm flag: absent"})
    write_csv(G / "g3_flags.csv", g3, list(g3[0].keys()))
    gj = []
    for c in kcells:
        gj.append({"kernel": c["kernel"], "cell_id": c["cell_id"], "n_pairs_J": n_pairs - 1, "floor_median_K": 150, "k_threshold": 450,
                   "frac_interpretable": 0.0 if c["kernel"] == "lexer" else 1.0, "J_mean": 0.8, "J_q05": 0.5, "J_q25": 0.75, "J_q50": 0.82,
                   "J_q75": 0.85, "J_q95": 0.9, "J_null_q50": 0.01, "frac_below_null": 0.0, "gp_verdict_pairs": "undeclared",
                   "mask_verdict": "floor overlap" if c["kernel"] == "lexer" else "interpretable"})
    write_csv(G / "gj.csv", gj, list(gj[0].keys()))
    # builder 2's exact shape (gates_readings.py gate_gj; CHECK_1.md B4): idle_J = null with no idle cell
    write_json(G / "gj.json", {"schema": "plan11.gj.v1", "params": {"k_factor": 3.0, "n_idle_cells": idle}, "citation": "SPEC 3.6.1",
                               "floor_median_K": 150 if idle else None, "floor_median_n_persist": 120 if idle else None,
                               "idle_J": ({"quantiles": [0.05, 0.25, 0.5, 0.75, 0.95], "J": [0.6, 0.8, 0.9, 0.95, 0.99],
                                           "n_pairs": idle * (n_pairs - 1), "mean": 0.88} if idle else None),
                               "mask_files": str(G / "gj_mask"), "mask_fields": ["mask_K", "mask_persist"]})
    if with_gj_mask:
        (G / "gj_mask").mkdir(exist_ok=True)
        for c in kcells:
            m = np.ones((n_pairs - 1, 2), dtype=bool)
            if c["kernel"] == "lexer":
                m[:, 0] = False
            m[::7, 1] = False
            np.save(G / "gj_mask" / f"{c['cell_id']}.npy", m)
    gdec = []
    for c in kcells:
        if c["kernel"] in ("floyd", "gibbs"):
            gdec.append({"kernel": c["kernel"], "cell_id": c["cell_id"], "role_in_test": "exhibit" if c["kernel"] == "floyd" else "control",
                         "n_passes": 5, "snaps_per_pass_median": 10, "phase_offset": 1, "run_length": 6, "slope_l0": -30.0, "slope_l1": -2000.0,
                         "hamming_sign": "-", "k_rel_drop": 0.01, "l0_rel_drop": 0.7, "surrogate_p05_slope": -5.0,
                         "verdict": gdec_verdict if c["kernel"] == "floyd" else "no decay"})
    gdec.append({"kernel": "floyd", "cell_id": "all", "role_in_test": "exhibit", "n_passes": 5, "snaps_per_pass_median": 10, "phase_offset": 1,
                 "run_length": 6, "slope_l0": -30.0, "slope_l1": -2000.0, "hamming_sign": "-", "k_rel_drop": 0.01, "l0_rel_drop": 0.7,
                 "surrogate_p05_slope": -5.0, "verdict": gdec_verdict})
    gdec.append({**gdec[-1], "kernel": "gibbs", "role_in_test": "control", "verdict": "no decay"})
    write_csv(G / "gdec.csv", gdec, list(gdec[0].keys()))
    # ---- splits
    for rung in RUNGS:
        gid = sel.get(rung, {}).get("grid_id")
        if gid is None:
            continue
        d = FEATURE_NAMES[rung].__len__()
        variants = [("", True), ("__raw", False)] if rung == "apf" else [("", True)]
        for suf, norm in variants:
            for split in ("within_trace", "loro", "loko"):
                for ls in (("archetype",) if split == "loko" else ("kernel", "archetype")):
                    unf = unfalsifiable_split == (rung, split, ls)
                    sc, preds = _scores(rng, cells, split, ls, n_perm=smoke_perm, unfalsifiable=unf, d=d,
                                        quality=0.95 if norm else 0.8)
                    sc["params"]["normalized"] = norm
                    sd = G / "splits" / rung / gid / f"{split}__{ls}{suf}"
                    sd.mkdir(parents=True, exist_ok=True)
                    write_json(sd / "scores.json", sc)
                    write_csv(sd / "predictions.csv", preds, ["cell_id", "kernel", "archetype", "campaign", "rep", "fold", "y_true", "y_pred",
                                                             "n_windows", "vote_fraction", "held_out_campaign"])
                    write_json(sd / "null.json", {"schema": "plan11.null.v1", "params": {}, "citation": "SPEC 3.7.1", "scores": []})
                    write_json(sd / "l1_quarantine.json", {"schema": "plan11.l1_quarantine.v1", "params": {}, "citation": "SPEC 3.7.2", "quarantined": []})
        if rung == "combined":
            for split in ("within_trace", "loro", "loko"):
                for ls in (("archetype",) if split == "loko" else ("kernel", "archetype")):
                    sc, preds = _scores(rng, cells, split, ls, n_perm=smoke_perm, d=8, quality=0.85)
                    sd = G / "splits" / "combined_matched" / gid / f"{split}__{ls}"
                    sd.mkdir(parents=True, exist_ok=True)
                    write_json(sd / "scores.json", sc)
                    write_csv(sd / "predictions.csv", preds, list(preds[0].keys()))
            # build epoch 2 (SPEC_epoch2 3.5.1, 6.3 item 6): the matched run gate_gdim writes for every Table 7
            # (split, label space) under gates/splits_matched/combined/<gid>/, d = 60 reduced to 36 per fold
            if matched_all_splits:
                for split in ("loko", "loro", "within_trace"):
                    for ls in (("archetype",) if split == "loko" else ("kernel", "archetype")):
                        sc, preds = _scores(rng, cells, split, ls, n_perm=smoke_perm, d=60, quality=0.85)
                        sc["dim_status"] = "declared reduction"
                        sc["feature_count_used"] = 36
                        sc["feature_count_used_per_fold"] = {p_["fold"]: 36 for p_ in preds}
                        sc["params"].update({"reduce_to": 36, "reduce_method": "train_importance", "base_dir": "splits_matched"})
                        sd = G / "splits_matched" / "combined" / gid / f"{split}__{ls}"
                        sd.mkdir(parents=True, exist_ok=True)
                        write_json(sd / "scores.json", sc)
                        write_csv(sd / "predictions.csv", preds, list(preds[0].keys()))
    # ---- comparison gates
    gl = [{"rung": r, "part": "i", "score_norm": 0.8, "null_p95": 0.62, "r2": "", "slope": "", "n_points": "",
           "verdict": "level only" if r == "wapf" else "pass"} for r in RUNGS]
    gl.append({"rung": "apf", "part": "ii", "score_norm": "", "null_p95": "", "r2": 0.2, "slope": 0.1, "n_points": 12, "verdict": "pass"})
    write_csv(G / "gl.csv", gl, list(gl[0].keys()))
    cnt = Counter(ARCHETYPE_OF[k] if not (relabel_lexer and k == "lexer") else "IDLE" for k, _ in KERNELS)
    gn = []
    for a in ARCHETYPES:
        n = cnt.get(a, 0)
        st = "headline" if n >= 3 else ("one training kernel per fold" if n == 2 else ("structural novelty" if n == 1 else "no kernel row"))
        gn.append({"archetype": a, "n_kernels": n, "kernels": " ".join(k for k, _ in KERNELS if (ARCHETYPE_OF[k] if not (relabel_lexer and k == "lexer") else "IDLE") == a), "status": st})
    write_csv(G / "gn.csv", gn, list(gn[0].keys()))
    gx = [{"rung": r, "score": 0.9 if (gx_total_leak and r == "apf") else 0.3, "null_p95": 0.45, "rank": 500 if (gx_total_leak and r == "apf") else 100,
           "leak_verdict": "campaign predictable" if (gx_total_leak and r == "apf") else "pooling stands",
           "confound_verdict": "confound: total" if (gx_total_leak and r == "apf") else "confound: partial"} for r in RUNGS]
    write_csv(G / "gx.csv", gx, list(gx[0].keys()))
    write_json(G / "gx.json", {"schema": "plan11.gx.v1", "params": {}, "citation": "SPEC 3.7.6", "campaign_sets": {}})
    gdim = [{"rung": r, "d": len(FEATURE_NAMES[r]), "d_matched": "", "method": "", "status": "full vector"} for r in RUNGS]
    gdim.append({"rung": "combined (matched)", "d": 60, "d_matched": 36, "method": "train_importance", "status": "declared reduction"})
    write_csv(G / "gdim.csv", gdim, list(gdim[0].keys()))
    gm = []
    for split in ("loko", "loro", "within_trace"):
        for ra in RUNGS + ("combined (matched)",):
            for rb in RUNGS:
                if ra == rb:
                    continue
                beats = ra in ("content", "combined") and rb == "apf"
                gm.append({"split": split, "rung_a": ra, "rung_b": rb, "score_a": 0.9 if beats else 0.7, "score_b": 0.7, "diff": 0.2 if beats else 0.0,
                           "spread": 0.04, "improving": 8 if beats else 5, "worsening": 0 if beats else 3, "ties": 4 if beats else 4,
                           "verdict": "beats" if beats else "difference with margin"})
    write_csv(G / "gm.csv", gm, list(gm[0].keys()))
    gv, gvs = [], []
    for r in RUNGS:
        n_gt = 0
        for f in FEATURE_NAMES[r]:
            L0, L3 = rng.uniform(0.01, 0.1), rng.uniform(0.05, 0.5)
            if r == "wapf":
                L0, L3 = L3, L0
            n_gt += L0 > L3
            gv.append({"rung": r, "feature": f, "L0": L0, "L2": rng.uniform(0.01, 0.2), "L3": L3, "L0_over_L3": L0 / L3})
        gvs.append({"rung": r, "n_features": len(FEATURE_NAMES[r]), "n_features_L0_gt_L3": n_gt,
                    "verdict": "LOKO not estimable" if n_gt == len(FEATURE_NAMES[r]) else "estimable"})
    write_csv(G / "gv.csv", gv, list(gv[0].keys()))
    write_csv(G / "gv_summary.csv", gvs, list(gvs[0].keys()))
    # builder 2's exact layout (models.py run_clustering; CHECK_1.md B5): a flat `cells` list and
    # per algorithm a `labels` list aligned with it plus `cluster_by_predicted_archetype`
    cl_cells = [c["cell_id"] for c in kcells]
    cl_labels = [int(list(ARCHETYPES).index(c["archetype_predicted"]) % 4) for c in kcells]
    cl_counts: dict = {}
    for c, lab in zip(kcells, cl_labels):
        cl_counts.setdefault(c["archetype_predicted"], {})
        cl_counts[c["archetype_predicted"]][f"c{lab}"] = cl_counts[c["archetype_predicted"]].get(f"c{lab}", 0) + 1
    null_s = {"observed": 0.8, "p95": 0.2, "rank": 500, "n": 500, "exceeds": True}
    write_csv(G / "clustering.csv", [{"rung": "combined", "algo": a, "k": 4, "ari": 0.8, "ari_null_p95": 0.2, "ari_rank": 500, "nmi": 0.8,
                                     "nmi_null_p95": 0.2, "nmi_rank": 500, "exceeds_ari": "true", "exceeds_nmi": "true",
                                     "primary": "true" if a == "kmeans" else "false", "grid_id": "W8_H4", "status": "ok"}
                                    for a in ("kmeans", "gmm", "agglomerative")],
              ["rung", "algo", "k", "ari", "ari_null_p95", "ari_rank", "nmi", "nmi_null_p95", "nmi_rank", "exceeds_ari", "exceeds_nmi",
               "primary", "grid_id", "status"])
    write_json(G / "clustering.json", {"schema": "plan11.clustering.v1", "params": {"rung": "combined", "grid_id": "W8_H4", "n_perm": 500},
                                       "citation": "SPEC 4.4", "k": 4, "cells": cl_cells,
                                       "kernels": [c["kernel"] for c in kcells], "archetype": [c["archetype_predicted"] for c in kcells],
                                       "per_algo": {a: {"labels": cl_labels, "ari": null_s, "nmi": null_s,
                                                        "cluster_by_predicted_archetype": cl_counts}
                                                    for a in ("kmeans", "gmm", "agglomerative")}})
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(make_out(Path(a.out)))
