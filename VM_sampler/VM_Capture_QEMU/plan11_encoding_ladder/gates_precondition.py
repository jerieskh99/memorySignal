#!/usr/bin/env python3
"""gates_precondition.py -- C1 to C8 as re-mapped, the ``failed/`` count, G-K0 and G-F
(SPEC section 3.3).

Citation: P2_STRUCTURE.md section V 5.1 (Plan 02, C1-C8 with C1 re-mapped) and 5.2
'Preconditions of valid observation' (the failed count, G-K0, G-F, the tripwire clause);
CR 2.1 items 1 and 2; CR 2.2 items 20 and 21; K2 moves 2 and 4 and move 14;
``plan05_campaign/validate_campaign.py`` lines 17 to 28 and 39 (the apf_queue re-map);
``plan02_validate_session.py`` D-25 (C3 informational).

Build epoch 2 (SPEC_epoch2 section 4; AD 2026-09-17 "C1 for kernel cells"; E1 sec. 4 M1): C1 for
kernel cells reads the sidecar's ``K_max`` against the idle floor's 95th percentile of K once idle
cells exist (``idle_band_edge``, the function G-K0's band also reads), else against the declared
absolute of 0.1 percent of memory (262 pages); the inherited ``apf_max >= 0.02`` is the documented
alternative ``legacy_apf_max`` and this module's function-level default (see ``gate_preconditions``).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plan11_encoding_ladder import series as S
from plan11_encoding_ladder import splits as SP
from plan11_encoding_ladder import nulls as NL
from plan11_encoding_ladder import verdicts as V
from plan11_encoding_ladder.series import schema

C1_ACTIVITY_MIN = 0.02          # validate_campaign.py line 39; section 8 item 36; kept as the documented alternative
                                # ``legacy_apf_max`` (AD 2026-09-17: "the inherited 0.02 stays only as a documented alternative")
# ---- C1 re-mapped for kernel cells (build epoch 2; AD 2026-09-17 "C1 for kernel cells"; E1 sec. 4 M1; E1 sec. 6 item 7)
C1_RULE_DEFAULT = "auto"                 # the CLI's and the driver's default: "auto" | "idle_floor" | "absolute" | "legacy_apf_max"
C1_RULE_FUNCTION_DEFAULT = "legacy_apf_max"   # gate_preconditions' own default (SPEC_epoch2_review_al_farabi.md 5.1: the
                                              # function keeps the inherited rule so every existing direct call keeps its contract)
C1_RULES = ("auto", "idle_floor", "absolute", "legacy_apf_max")
C1_ABS_FRACTION = 0.001                  # AD 2026-09-17: "a declared absolute of 0.1 percent of memory (262 pages)"
C1_ABS_PAGES = int(C1_ABS_FRACTION * schema.N_PAGES)      # 262
C1_ACTIVITY_MIN_PAGES_T1 = 200           # AA T1's own interim ("K_max >= 200 pages"; SPEC_epoch2 B1, Part 4 item 14): reachable
                                         # through --c1-activity-min-pages 200; the 262 of the AD bullet stays the default here
                                         # (SPEC_epoch2_review_al_farabi.md section 6 item 1: the two records are the author's to reconcile)
C1_IDLE_PERCENTILE = 95.0                # AD 2026-09-17 "the idle floor's 95th percentile"; = GK0_IDLE_PERCENTILE (one edge, one function)
C1_IDLE_POOL = "pooled_snapshots"        # = GK0_IDLE_POOL
C1_MIN_IDLE_CELLS = 1                    # ``auto`` takes the floor once this many idle cells enter it
C1_FLOOR_REFUSAL = V.refused("C1 idle floor requested, no idle cell enters the floor")
C1_CHANGE_RECORD = ("pre-registered gate C1 re-mapped for kernel cells: AD 2026-09-17 (P2_AUTHOR_ANSWERS.md, Decisions of "
                    "2026-09-17; AA T1 for the floor's definition); the epoch-1 rule (apf_max >= 0.02) kept as legacy_apf_max")
C2_MIN_PAIRS = 8                # one window at (8, 4)
C3_MIN_WINDOWS = 3              # informational, '> 3' (plan02_validate_session.py D-25)
C3_W, C3_H = 8, 4
HEADER_NCOLS = 66
C4_STRING = V.not_applicable("no settle record in the retention layout")
C5_STRING = V.not_run("producer.log is not in the trajectory; the author reads the campaign log")
C8_STRING = V.not_applicable("change-point view not in this paper (decided 2026-09-16)")
C7_PENDING = V.pending("filled by the temporal gates (move 6)")
C1_IDLE_STRING = V.not_applicable("control (C1 re-mapped)")
GK0_TAIL_FRACTION = 0.80        # section 8 item 14
GK0_IDLE_POOL = "pooled_snapshots"
GK0_IDLE_PERCENTILE = 95.0
GF_PART1_DESIGN = "within_trace_window"     # section 8 item 15
GF_PART2_RULE = "all_cells_outside"
GF_TEST_FRAC = 0.2
GF_N_PERM = 500
GF_PART1_CONSEQUENCE = "void"   # SPEC_review_al_farabi.md section 3 item 2: "void" | "report"
GF_PART1_FEATURES = "norm"      # the rung as the splits use it (level-normalized); "raw" hears a level step between idle reps
GF_SEPARABLE_REPORTED = "separable within-trace at the window level (reported)"
GF_PERM_FLOOR_CLI = 500         # SPEC_epoch2 B3 (CHECK_3 M3; CERT 3, 6.7): B1-G1's floor (models.B1G1_MIN_PERM) on the gf CLI;
                                # the function default of gate_gf / _gf_part1 is 0 so the existing direct calls keep their contract
ADMISSIBILITY_KEYS = ("same_ssh_path", "rebooted_per_cell", "interval_ms", "differ_speed",
                      "image_state_note", "duration_s", "capture_loop_note")

CIT_PRE = ("P2 Sec. V 5.1 Plan 02 (C1-C8, C1 re-mapped) and 5.2 preconditions (the failed/ count); "
           "CR 2.1 items 1, 2; validate_campaign.py lines 17-28, 39; plan02_validate_session.py D-25; K2 move 2")
CIT_GK0 = "P2 Sec. V 5.2 G-K0; CR 2.2 item 20; K2 move 4"
CIT_GF = "P2 Sec. V 5.2 G-F and the tripwire clause; CR 2.2 item 21, 2.1 item 15; K2 moves 4 and 14"

PRE_COLUMNS = ("cell_id", "role", "C1", "C1_apf_max", "C2", "C2_n_pairs", "C3", "C3_n_windows_8_4",
               "C4", "C5", "C6", "C6_reason", "C7", "C8", "failed_count", "failed_verdict",
               "failed_source", "all_hard_pass",
               # epoch 2 (SPEC_epoch2 section 4): the C1 operand, the threshold in pages and the rule in force, per row
               "C1_K_max", "C1_threshold_pages", "C1_rule")


# --------------------------------------------------------------------------- C1..C8 and the failed count (3.3.1, 3.3.2)

def load_failed_counts(path: Path | None) -> dict:
    """inputs/failed_counts.csv (cell_id, failed_count, source) overrides the sidecar (SPEC 3.3.2)."""
    if path is None or not Path(path).is_file():
        return {}
    res = {}
    for r in S.read_csv(path):
        v = r.get("failed_count", "")
        res[r["cell_id"]] = (None if v in ("", "null", "None") else int(float(v)), r.get("source", "inputs/failed_counts.csv"))
    return res


def failed_verdict(failed_count, *, assume_failed_zero: bool = False, assume_reason: str = "") -> tuple:
    """The per-cell failed/ count verdict (P2 Sec. V 5.2 preconditions, *min*; CR 2.1 item 2; K2
    move 2; AA A5): PASS if the count is 0; ``refused: failed count <n> > 0, seq axis uncorrected``
    if positive (the cell's J after the first failure is unusable); if the count is null,
    ``refused: failed count not recorded`` unless the driver was given ``--assume-failed-zero
    --assume-reason "<text>"``, in which case PASS with ``failed_source = "declared zero: <text>"``
    (SPEC_review_al_farabi.md item 2.9 (a)). Returns (verdict, source)."""
    if failed_count is None:
        if assume_failed_zero and assume_reason:
            return V.PASS, f"declared zero: {assume_reason}"
        return V.refused("failed count not recorded"), "not recorded"
    n = int(failed_count)
    if n == 0:
        return V.PASS, "recorded"
    return V.refused(f"failed count {n} > 0, seq axis uncorrected"), "recorded"


def c7_verdict(out: Path) -> str:
    """C7 (Plan 03 winner): pending until gates/selection.json exists, then pass iff the APF
    selection has passes_acceptance == true (SPEC 3.3.1)."""
    p = Path(out) / "gates" / "selection.json"
    if not p.is_file():
        return C7_PENDING
    sel = S.read_json(p)
    entry = sel.get("apf") or (sel.get("selection") or {}).get("apf")
    if not entry:
        return C7_PENDING
    return V.PASS if entry.get("passes_acceptance") else V.FAIL


def idle_band_edge(out: Path, idle_cells: list[dict], *, idle_pool: str = GK0_IDLE_POOL,
                   idle_percentile: float = GK0_IDLE_PERCENTILE) -> float | None:
    """The idle band's upper edge, one number for G-K0's band and C1's floor (SPEC 3.3.3; P2 Sec. V
    5.2 G-K0; AA T1 "the same floor G-K0 uses"; AD 2026-09-17 "the idle floor's 95th percentile";
    SPEC_epoch2 section 4). ``pooled_snapshots``: the ``idle_percentile``-th percentile of the
    undropped ``K`` pooled over every given idle cell's ``extract/<cell_id>/extract.csv`` rows;
    ``cell_medians``: the same percentile of the idle cells' per-cell medians of ``K``. Returns None
    when no idle cell is given. This is the code that was inline in ``gate_gk0`` in epoch 1,
    extracted so that C1 and G-K0 read one number from one function and one file."""
    if not idle_cells:
        return None
    if idle_pool == "pooled_snapshots":
        pooled = np.concatenate([S.load_extract_cached(out, c["cell_id"])["K"] for c in idle_cells])
        return float(np.percentile(pooled, idle_percentile))
    if idle_pool == "cell_medians":
        meds = [float(np.median(S.load_extract_cached(out, c["cell_id"])["K"])) for c in idle_cells]
        return float(np.percentile(meds, idle_percentile))
    raise ValueError(idle_pool)


def _fmt_param(v) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def gate_preconditions(out: Path, *, cells_csv: Path | None = None, assume_failed_zero: bool = False,
                       assume_reason: str = "", failed_counts_csv: Path | None = None,
                       c1_activity_min: float = C1_ACTIVITY_MIN, c1_rule: str = C1_RULE_FUNCTION_DEFAULT,
                       c1_abs_fraction: float = C1_ABS_FRACTION, c1_idle_percentile: float = C1_IDLE_PERCENTILE,
                       c1_idle_pool: str = C1_IDLE_POOL, c1_min_idle_cells: int = C1_MIN_IDLE_CELLS,
                       c1_activity_min_pages: int | None = None) -> Path:
    """C1 to C8 re-mapped onto the extract, per cell (SPEC 3.3.1 table; P2 Sec. V 5.1 Plan 02;
    CR 2.1 item 1). C2: ``n_pairs >= 8``. C3: ``n_windows(n_pairs - 1, 8, 4) > 3``, informational.
    C4, C5, C8: fixed strings. C6: sidecar status ok, header_ncols == 66, header_sha256 equal to the
    MODAL header over all cells (a tie is ``refused: header mismatch, no majority`` on every cell;
    SPEC_review_al_farabi.md item 2.11 (a)), n_rows_skipped == 0; n_seq_gaps reported, not failing.
    C7: pending until selection.json. ``all_hard_pass = C1 in (pass, not applicable) and C2 == pass
    and C6 == pass``; excluded cells are listed in preconditions.json ``excluded_cells``; cells whose
    failed verdict is a refusal are listed in ``excluded_cells_pair_rungs`` and are kept out of the
    persist, content and combined series (SPEC_review_al_farabi.md item 2.4).

    C1 for kernel cells, re-mapped in build epoch 2. Citation: AD 2026-09-17 "C1 for kernel cells"
    (P2_AUTHOR_ANSWERS.md, Decisions of 2026-09-17): a kernel cell is active if its maximum
    changed-page count is above the idle floor's 95th percentile once the idle cells exist; until
    then a declared absolute of 0.1 percent of memory (262 pages), recorded in the run's params; the
    inherited 0.02 stays only as a documented alternative. AA T1 (the same file, "Build epoch 1: the
    author's decisions", T1) defines the floor as the idle cells' 95th percentile of K, the same
    floor G-K0 uses, and keeps C1 a hard gate; E1 sec. 4 M1 and E1 sec. 6 item 7 are the open
    finding this closes. Idle cells keep the epoch-1 re-map (``C1_IDLE_STRING``, never refused;
    CR 2.1 item 1). The operand is the sidecar's integer ``K_max`` (the undropped maximum the
    extractor records, SPEC 2.3), never the float ``apf_max``, except under ``legacy_apf_max``.
    The rules (``c1_rule``):

    - ``idle_floor``: pass iff ``K_max > edge`` (strict, "exceeds"), ``edge = idle_band_edge(...)``
      at ``c1_idle_percentile`` over the idle cells that enter the floor: ``cells.csv`` status ok, a
      sidecar, and their own C2 and C6 ``pass`` (C1 is not applicable to them; computed in a first
      pass over every cell before any C1 is decided). ``inputs/idle_admissibility.json`` is not
      required, as G-K0 does not require it (E1 sec. 6 item 18). With no idle cell entering the
      floor every kernel row reads ``C1_FLOOR_REFUSAL`` and fails ``all_hard_pass`` (a refusal,
      never a silent fallback).
    - ``absolute``: pass iff ``K_max >= int(c1_abs_fraction * N)`` (262 pages at the default), or
      ``K_max >= c1_activity_min_pages`` when that page count is given (AA T1's own interim is
      ``K_max >= 200`` pages, SPEC_epoch2 B1 and Part 4 item 14; the AD bullet of 2026-09-17 says
      262; ``--c1-activity-min-pages 200`` selects T1's number and the record says which was applied,
      SPEC_epoch2_review_al_farabi.md section 6 item 1).
    - ``legacy_apf_max``: pass iff ``apf_max >= c1_activity_min`` (the epoch-1 rule, unchanged;
      validate_campaign.py line 39; SPEC section 8 item 36).
    - ``auto``: ``idle_floor`` when at least ``c1_min_idle_cells`` idle cells enter the floor, else
      ``absolute``.

    Two defaults, on purpose (SPEC_epoch2_review_al_farabi.md section 5 item 1, the pattern of
    ``perm_floor``): this function's default is ``C1_RULE_FUNCTION_DEFAULT = "legacy_apf_max"`` so
    that every existing direct call keeps its contract; the CLI ``preconditions --c1-rule`` and the
    driver default to ``C1_RULE_DEFAULT = "auto"``, so the paper's path runs the re-mapped rule.
    ``preconditions.csv`` carries ``C1_K_max``, ``C1_threshold_pages`` and ``C1_rule`` per row and
    ``preconditions.json`` ``params`` records the rule requested, the rule applied, the edge, the idle
    cells pooled and the change record (``C1_CHANGE_RECORD``), so the runbook's promise (AA T1: "the
    runbook records which rule was in force for each run") is kept by the artifact itself."""
    if c1_rule not in C1_RULES:
        raise ValueError(f"c1_rule {c1_rule!r} not in {C1_RULES}")
    out = Path(out)
    cells_path = S.cells_csv_path(out, cells_csv)
    cells = S.load_cells(cells_path, only_ok=False)
    fc_path = Path(failed_counts_csv) if failed_counts_csv else out / "inputs" / "failed_counts.csv"
    overrides = load_failed_counts(fc_path)
    sidecars = {}
    for c in cells:
        p = S.sidecar_path(out, c["cell_id"])
        if p.is_file():
            sidecars[c["cell_id"]] = S.read_json(p)
    hashes = Counter(sc.get("header_sha256", "") for sc in sidecars.values())
    modal, no_majority = None, False
    if hashes:
        top = hashes.most_common()
        modal = top[0][0]
        no_majority = len(top) > 1 and top[0][1] == top[1][1]
    rows, excluded, excluded_pair, inputs = [], [], [], [cells_path, fc_path]
    pending = []          # (row, cell, sidecar) of the cells whose C1 is decided in the second pass
    # ---- first pass: every column but C1 (so the idle cells' C2 and C6 are known before any C1)
    for c in cells:
        cid = c["cell_id"]
        sc = sidecars.get(cid)
        row = {"cell_id": cid, "role": c["role"], "C4": C4_STRING, "C5": C5_STRING, "C8": C8_STRING, "C7": c7_verdict(out),
               "C1_K_max": None, "C1_threshold_pages": None, "C1_rule": ""}
        if c.get("status", "ok") != "ok" or sc is None:
            reason = f"cells.csv status {c.get('status')}" if c.get("status", "ok") != "ok" else "sidecar.json missing"
            for k in ("C1", "C2", "C3", "C6"):
                row[k] = V.not_run(reason)
            row.update({"C1_apf_max": None, "C2_n_pairs": None, "C3_n_windows_8_4": None, "C6_reason": reason,
                        "failed_count": None, "failed_verdict": V.not_run(reason), "failed_source": "", "all_hard_pass": False})
            rows.append(row); excluded.append(cid)
            continue
        inputs.append(S.sidecar_path(out, cid))
        apf_max = float(sc.get("apf_max", float("nan")))
        n_pairs = int(sc.get("n_pairs", 0))
        row["C1_apf_max"] = apf_max
        row["C1_K_max"] = None if sc.get("K_max") is None else int(sc["K_max"])
        row["C2_n_pairs"] = n_pairs
        row["C2"] = V.PASS if n_pairs >= C2_MIN_PAIRS else V.FAIL
        nw = S.n_windows(n_pairs - 1, C3_W, C3_H)
        row["C3_n_windows_8_4"] = nw
        row["C3"] = V.PASS if nw > C3_MIN_WINDOWS else V.FAIL
        reasons = []
        status = sc.get("status", "")
        if status != "ok":
            reasons.append(f"extractor status: {status}")
        if int(sc.get("header_ncols", 0)) != HEADER_NCOLS:
            reasons.append(f"header_ncols {sc.get('header_ncols')} != {HEADER_NCOLS}")
        if no_majority:
            reasons.append("header mismatch, no majority")
        elif sc.get("header_sha256") != modal:
            reasons.append("header_sha256 differs from the modal header")
        if int(sc.get("n_rows_skipped", 0)) != 0:
            reasons.append(f"n_rows_skipped {sc.get('n_rows_skipped')}")
        gaps = int(sc.get("n_seq_gaps", 0))
        if no_majority:
            row["C6"] = V.refused("header mismatch, no majority")
        else:
            row["C6"] = V.PASS if not reasons else V.FAIL
        row["C6_reason"] = "; ".join(reasons + [f"n_seq_gaps {gaps}"])
        if cid in overrides:
            fcount, fsrc = overrides[cid]
        else:
            fcount, fsrc = sc.get("failed_count"), sc.get("failed_count_source", "")
        fv, fsource = failed_verdict(fcount, assume_failed_zero=assume_failed_zero, assume_reason=assume_reason)
        row["failed_count"] = fcount
        row["failed_verdict"] = fv
        row["failed_source"] = fsource if fsource != "recorded" else f"recorded ({fsrc})"
        rows.append(row)
        pending.append((row, c, sc))
    # ---- the idle cells that enter the floor: status ok, a sidecar, their own C2 and C6 pass (C1 is not applicable to them)
    idle_in_floor = [c for row, c, sc in pending if c["role"] == "idle" and row["C2"] == V.PASS and row["C6"] == V.PASS]
    edge = None
    if idle_in_floor:
        # the edge is measured and recorded whenever idle cells enter the floor (SPEC_epoch2 section 4, test 6.3.2:
        # "records C1_idle_band_edge anyway"); the rule decides whether it is applied
        edge = idle_band_edge(out, idle_in_floor, idle_pool=c1_idle_pool, idle_percentile=c1_idle_percentile)
        inputs += [S.extract_path(out, c["cell_id"]) for c in idle_in_floor]     # SPEC_epoch2_review_al_farabi.md 5.7
    abs_pages = int(c1_activity_min_pages) if c1_activity_min_pages is not None else int(c1_abs_fraction * schema.N_PAGES)
    if c1_rule == "auto":
        rule_applied = "idle_floor" if len(idle_in_floor) >= c1_min_idle_cells else "absolute"
    else:
        rule_applied = c1_rule
    if rule_applied == "idle_floor":
        rule_label, threshold = f"idle_floor_p{_fmt_param(c1_idle_percentile)}", edge
    elif rule_applied == "absolute":
        rule_label = (f"absolute_{abs_pages}pages" if c1_activity_min_pages is not None else f"absolute_{_fmt_param(c1_abs_fraction)}")
        threshold = abs_pages
    else:
        rule_label, threshold = f"legacy_apf_max_{_fmt_param(c1_activity_min)}", int(np.ceil(c1_activity_min * schema.N_PAGES))
    floor_refused = rule_applied == "idle_floor" and edge is None
    # ---- second pass: C1 and all_hard_pass
    for row, c, sc in pending:
        cid = c["cell_id"]
        if c["role"] == "idle":
            row["C1"] = C1_IDLE_STRING
        elif floor_refused:
            row["C1"] = C1_FLOOR_REFUSAL
            row["C1_rule"] = rule_label
        elif rule_applied == "legacy_apf_max":
            row["C1"] = V.PASS if row["C1_apf_max"] >= c1_activity_min else V.FAIL
            row["C1_rule"], row["C1_threshold_pages"] = rule_label, threshold
        elif row["C1_K_max"] is None:
            row["C1"] = V.not_run("sidecar has no K_max")
            row["C1_rule"], row["C1_threshold_pages"] = rule_label, threshold
        elif rule_applied == "idle_floor":
            row["C1"] = V.PASS if row["C1_K_max"] > edge else V.FAIL
            row["C1_rule"], row["C1_threshold_pages"] = rule_label, threshold
        else:
            row["C1"] = V.PASS if row["C1_K_max"] >= abs_pages else V.FAIL
            row["C1_rule"], row["C1_threshold_pages"] = rule_label, threshold
        hard = (row["C1"] in (V.PASS, C1_IDLE_STRING)) and row["C2"] == V.PASS and row["C6"] == V.PASS
        row["all_hard_pass"] = bool(hard)
        if not hard:
            excluded.append(cid)
        elif V.is_refusal(row["failed_verdict"]):
            excluded_pair.append(cid)
    S.write_csv(out / "gates" / "preconditions.csv", PRE_COLUMNS, rows)
    fc_rows = [{"cell_id": r["cell_id"], "failed_count": r["failed_count"], "failed_verdict": r["failed_verdict"],
                "failed_source": r["failed_source"]} for r in rows]
    S.write_csv(out / "gates" / "failed_counts.csv", ("cell_id", "failed_count", "failed_verdict", "failed_source"), fc_rows)
    params = {"C1_ACTIVITY_MIN": c1_activity_min, "C1_ACTIVITY_MIN_spec_default": C1_ACTIVITY_MIN, "C2_MIN_PAIRS": C2_MIN_PAIRS, "C3_MIN_WINDOWS": C3_MIN_WINDOWS,
              "C3_window_hop": [C3_W, C3_H], "HEADER_NCOLS": HEADER_NCOLS, "header_reference": "modal header_sha256",
              "modal_header_sha256": modal, "assume_failed_zero": assume_failed_zero, "assume_reason": assume_reason,
              "failed_counts_csv": str(fc_path) if fc_path.is_file() else None,
              "gap_rule": "a missing seq is a K = 0 snapshot; reported in C6_reason, never fails C6 (SPEC 2.1, 8.3)",
              # epoch 2: the C1 record (SPEC_epoch2 section 4; AD 2026-09-17)
              "epoch": 2,
              "C1_rule_requested": c1_rule, "C1_rule_applied": rule_applied, "C1_rule_in_force": rule_label,
              "C1_rule_default_cli": C1_RULE_DEFAULT, "C1_rule_default_function": C1_RULE_FUNCTION_DEFAULT,
              "C1_operand": "apf_max (float)" if rule_applied == "legacy_apf_max" else "K_max (sidecar, undropped, int)",
              "C1_threshold_pages": threshold, "C1_idle_band_edge": edge,
              "C1_idle_cells_in_floor": [c["cell_id"] for c in idle_in_floor],
              "C1_idle_percentile": c1_idle_percentile, "C1_idle_pool": c1_idle_pool, "C1_min_idle_cells": c1_min_idle_cells,
              "C1_abs_fraction": c1_abs_fraction, "C1_abs_pages": abs_pages,
              "C1_activity_min_pages": c1_activity_min_pages,      # SPEC_epoch2 B1: AA T1's page count when given (None = the fraction rule)
              "C1_activity_min_pages_T1": C1_ACTIVITY_MIN_PAGES_T1,
              "C1_legacy_apf_max": c1_activity_min if rule_applied == "legacy_apf_max" else f"{_fmt_param(c1_activity_min)} (alternative, not applied)",
              "C1_floor_refused": floor_refused,
              "C1_change_record": C1_CHANGE_RECORD,
              "inputs_sha256": S.inputs_sha256(inputs, out)}
    S.write_json(out / "gates" / "preconditions.json", "plan11.preconditions.v1", params, CIT_PRE,
                 {"n_cells": len(rows), "n_all_hard_pass": sum(1 for r in rows if r["all_hard_pass"]),
                  "excluded_cells": excluded, "excluded_cells_pair_rungs": excluded_pair})
    return out / "gates" / "preconditions.csv"


# --------------------------------------------------------------------------- G-K0 (3.3.3)

GK0_SOURCE_TABLE3 = {
    "gemm": "yes", "floyd": "yes", "gibbs": "yes", "nbody": "yes", "spmm": "unstated",
    "stencil_jacobi": "yes", "fft": "yes", "histogram": "only through pass phase",
    "fem_assembly": "yes", "lexer": "no (after the first pass)", "rmat_gen": "unstated", "bnb_tsp": "yes",
}


def write_gk0_template(path: Path) -> Path:
    """inputs/gk0_source.csv pre-filled from P2 Table 3 column 5; the author edits (SPEC 3.3.3)."""
    rows = [{"kernel": k, "steady_state_changes_content": GK0_SOURCE_TABLE3.get(k, "unstated"),
             "source": "P2_STRUCTURE.md Table 3 column 5 (template; the author edits)"} for k, _ in schema.KERNELS]
    return S.write_csv(path, ("kernel", "steady_state_changes_content", "source"), rows)


def load_gk0_source(path: Path) -> dict:
    if not Path(path).is_file():
        return {}
    return {r["kernel"]: r.get("steady_state_changes_content", "unstated") for r in S.read_csv(path)}


def tail_median_K(ex: dict, tail_fraction: float = GK0_TAIL_FRACTION) -> float:
    """Median K over the last ``tail_fraction`` of the rows (by seq, undropped series; SPEC 3.1.2)."""
    K = ex["K"]
    n = len(K)
    start = int(np.floor(n * (1.0 - tail_fraction)))
    return float(np.median(K[start:])) if n else float("nan")


def gate_gk0(out: Path, cells: list[dict] | None = None, *, tail_fraction: float = GK0_TAIL_FRACTION,
             idle_pool: str = GK0_IDLE_POOL, idle_percentile: float = GK0_IDLE_PERCENTILE) -> Path:
    """G-K0, state-change disclosure. Source part: inputs/gk0_source.csv (kernel,
    steady_state_changes_content, source), template pre-filled from P2 Table 3 column 5
    ('yes' | 'no (after the first pass)' | 'only through pass phase' | 'unstated'), the
    author edits. Measured part: per cell, median K over the last `tail_fraction` of the
    rows (by seq, undropped series) against the idle band's upper edge: the
    `idle_percentile`-th percentile of K pooled over every idle cell's rows
    (idle_pool="pooled_snapshots") or of the idle cells' per-cell medians
    (idle_pool="cell_medians"). Per kernel: relabelled GK0_IDLE_MEASURED when the median of
    its cells' tail medians is inside the band (<= edge), else GK0_ABOVE_FLOOR. With no idle
    cell: not_run('no admissible idle cell'). Citation: P2 Sec. V 5.2 G-K0; CR 2.2 item 20.
    Only cells with all_hard_pass enter (preconditions.csv when present)."""
    out = Path(out)
    if cells is None:
        cells = S.load_cells(out / "cells.csv")
    cells, _, _, _ = S.admissible_cells(out, cells, None)
    src_path = out / "inputs" / "gk0_source.csv"
    src = load_gk0_source(src_path)
    idle = [c for c in cells if c["role"] == "idle"]
    kern = [c for c in cells if c["role"] == "kernel"]
    # the band's edge: one function shared with C1's floor (epoch 2, SPEC_epoch2 section 4)
    edge = idle_band_edge(out, idle, idle_pool=idle_pool, idle_percentile=idle_percentile)
    rows = []
    by_kernel = {}
    for c in kern:
        by_kernel.setdefault(c["kernel"], []).append(c)
    for k in [kk for kk, _ in schema.KERNELS if kk in by_kernel] + sorted(kk for kk in by_kernel if kk not in dict(schema.KERNELS)):
        tm = [tail_median_K(S.load_extract_cached(out, c["cell_id"]), tail_fraction) for c in by_kernel[k]]
        med = float(np.median(tm))
        if edge is None:
            verdict, measured = V.not_run("no admissible idle cell"), by_kernel[k][0]["archetype_predicted"]
        elif med <= edge:
            verdict, measured = V.GK0_IDLE_MEASURED, "IDLE"
        else:
            verdict, measured = V.GK0_ABOVE_FLOOR, by_kernel[k][0]["archetype_predicted"]
        rows.append({"kernel": k, "source_statement": src.get(k, "unstated"), "n_cells": len(tm),
                     "tail_median_K_median": med, "tail_median_K_min": float(min(tm)), "tail_median_K_max": float(max(tm)),
                     "idle_band_edge": edge, "verdict": verdict, "archetype_measured": measured})
    if idle:
        tm = [tail_median_K(S.load_extract_cached(out, c["cell_id"]), tail_fraction) for c in idle]
        rows.append({"kernel": "idle", "source_statement": "control", "n_cells": len(tm),
                     "tail_median_K_median": float(np.median(tm)), "tail_median_K_min": float(min(tm)),
                     "tail_median_K_max": float(max(tm)), "idle_band_edge": edge, "verdict": "control", "archetype_measured": "IDLE"})
    cols = ("kernel", "source_statement", "n_cells", "tail_median_K_median", "tail_median_K_min", "tail_median_K_max",
            "idle_band_edge", "verdict", "archetype_measured")
    p = S.write_csv(out / "gates" / "gk0.csv", cols, rows)
    params = {"tail_fraction": tail_fraction, "idle_pool": idle_pool, "idle_percentile": idle_percentile,
              "n_idle_cells": len(idle), "idle_band_edge": edge, "gk0_source_csv": str(src_path) if src_path.is_file() else None,
              "inputs_sha256": S.inputs_sha256([out / "cells.csv", src_path, out / "gates" / "preconditions.csv"], out)}
    S.write_params(p, "plan11.gk0.v1", params, CIT_GK0)
    return p


# --------------------------------------------------------------------------- G-F (3.3.4)

def write_idle_admissibility_template(path: Path) -> Path:
    """inputs/idle_admissibility.json template; the author fills it (SPEC 3.3.4; CR 2.2 item 21)."""
    doc = {k: None for k in ADMISSIBILITY_KEYS}
    doc["_note"] = ("Fill every key from the run record before G-F runs; the idle control must be the same "
                    "instrument state minus the kernel (AA A3). Leave the file absent to keep G-F 'not run'.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1))
    return path


def load_idle_admissibility(path: Path) -> dict | None:
    if not Path(path).is_file():
        return None
    doc = S.read_json(path)
    if any(doc.get(k) is None for k in ADMISSIBILITY_KEYS):
        return None
    return doc


def _gf_part1(out: Path, rung: str, grid_id: str, idle: list[dict], *, n_perm: int, test_frac: float,
              n_estimators: int, seed_offset: int, n_jobs: int, features: str = GF_PART1_FEATURES) -> dict:
    from plan11_encoding_ladder import models as M
    norm = features == "norm" or rung == "combined"
    p = S.features_path(out, rung, grid_id, norm)
    if not p.is_file():
        W, H = S.parse_grid_id(grid_id)
        hd = S.load_head_drop(out / "inputs" / "head_drop.csv")
        S.build_features(out, None, rung, W, H, norm, hd)
    feat = S.load_features(p)
    ids = {c["cell_id"] for c in idle}
    mask = np.array([cid in ids for cid in feat["cell_id"]])
    lab = SP.make_labels(feat, mask)
    X = feat["X"][lab["_rows"]]
    y = np.array([f"rep{r:02d}" for r in lab["rep"]])
    if lab["n"] < 2 or len(set(y.tolist())) < 2:
        return {"score": None, "null_p95": None, "verdict": V.not_run("fewer than two idle reps with windows"), "n": lab["n"]}
    folds = SP.fold_within_trace(lab, test_frac)
    tr, te = folds[0]["train"], folds[0]["test"]
    if len(te) == 0 or len(tr) == 0:
        return {"score": None, "null_p95": None, "verdict": V.not_applicable("one window per cell"), "n": lab["n"]}

    def score(yy):
        clf = M.make_forest(NL.SEED_FOREST + seed_offset, 1, n_estimators).fit(X[tr], yy[tr])
        return float(np.mean(clf.predict(X[te]) == yy[te]))
    obs = score(y)
    rng = np.random.default_rng(NL.SEED_LABEL_NULL + seed_offset)
    perms = [y[rng.permutation(len(y))] for _ in range(n_perm)]
    if n_jobs > 1 and n_perm > 1:
        from joblib import Parallel, delayed
        null = Parallel(n_jobs=n_jobs)(delayed(score)(pp) for pp in perms)
    else:
        null = [score(pp) for pp in perms]
    summ = NL.null_summary(obs, np.array(null))
    return {"score": obs, "null_p95": summ["p95"], "exceeds": summ["exceeds"], "n": lab["n"], "summary": summ}


def gate_gf(out: Path, cells: list[dict] | None = None, rung: str = "apf", grid_id: str | None = None, *,
            n_perm: int = GF_N_PERM, part1_design: str = GF_PART1_DESIGN, test_frac: float = GF_TEST_FRAC,
            part2_rule: str = GF_PART2_RULE, part1_consequence: str = GF_PART1_CONSEQUENCE,
            part1_features: str = GF_PART1_FEATURES, n_estimators: int = 300, seed_offset: int = 0, n_jobs: int = 1,
            perm_floor: int = 0) -> Path:
    """G-F, floor. grid_id: the rung's selected point from selection.json when it exists,
    else gf_default_grid = "W8_H4" (move 4 runs before move 6; move 13 re-runs at the
    selected point). Admissibility record first: inputs/idle_admissibility.json, written by
    the author (keys: same_ssh_path, rebooted_per_cell, interval_ms, differ_speed,
    image_state_note, duration_s, capture_loop_note); absent -> every G-F row is
    not_run('no admissible idle cell; admissibility record missing') and the tripwire row
    reads 'not run'. Part (i): the idle cells must be mutually inseparable under the rung.
    The definition says leave-one-rep-out on idle cells alone; with one cell per rep label a
    held-out rep can never be predicted, so the executable form is part1_design =
    'within_trace_window': label = rep, the last test_frac of each idle cell's windows is
    test, the forest of 4.2, score = window accuracy, null = 500 window-level label shuffles
    within the idle set; verdict GF_INSEPARABLE if the observed score does not strictly
    exceed the null's p95, else GF_VOID (the rung's result is void; with part1_consequence =
    'report' the row reads GF_SEPARABLE_REPORTED instead and voids nothing,
    SPEC_review_al_farabi.md section 3 item 2). Part (i) runs on the rung's level-normalized
    features, the ones the splits use (``part1_features = "norm"``); ``"raw"`` hears a level step
    between idle reps that normalization removes. Part (ii), per kernel: the kernel's per-cell
    medians of the rung's headline reading (3.1.5) against the envelope [min, max] of the idle
    cells' per-cell medians; part2_rule='all_cells_outside': every cell median outside -> pass,
    else GF_AT_FLOOR; 'median_of_cells_outside': the median of the cell medians. Reported as a
    finding, never as an encoding failure. The three floors (idle K, l0 per changed page as the
    per-snapshot l0_q50_all, J) go to gates/gf_floors.json as five quantiles each, pooled over
    idle cells. Citation: P2 Sec. V 5.2 G-F and the tripwire clause; CR 2.2 item 21, 2.1 item 15.
    ``perm_floor`` (SPEC_epoch2 B3; CHECK_3 M3; P2 Sec. V 5.1 Plan 08 "at least 500 permutations"): when
    ``0 < n < perm_floor`` (``n`` the null permutations scored, as ``models.b1_g1_verdict`` counts them)
    part (i) reads ``not run: <n> permutations < <perm_floor>`` and the score and null p95 stay in
    their columns. Two defaults on purpose: the function default is 0 (judge on any count, the
    epoch-1 contract of every direct call) and the ``gf`` CLI default is ``GF_PERM_FLOOR_CLI = 500``
    (``models.B1G1_MIN_PERM``), so the driver's path refuses an under-powered null. The idle cells'
    head drop is the ``idle`` row of ``inputs/head_drop.csv`` (B9; recorded as ``head_drop_idle``)."""
    out = Path(out)
    if cells is None:
        cells = S.load_cells(out / "cells.csv")
    cells, _, _, _ = S.admissible_cells(out, cells, None)
    if grid_id is None:
        grid_id, gsrc = S.selected_grid_id(out, rung, S.GF_DEFAULT_GRID)
    else:
        gsrc = "argument"
    adm_path = out / "inputs" / "idle_admissibility.json"
    adm = load_idle_admissibility(adm_path)
    idle = [c for c in cells if c["role"] == "idle"]
    kern = [c for c in cells if c["role"] == "kernel"]
    hd = S.load_head_drop(out / "inputs" / "head_drop.csv")
    rows = []
    cols = ("rung", "grid_id", "part", "kernel", "n_cells", "score", "null_p95", "n_inside_envelope",
            "envelope_lo", "envelope_hi", "verdict")
    kernels = [k for k, _ in schema.KERNELS if any(c["kernel"] == k for c in kern)] + \
              sorted({c["kernel"] for c in kern} - set(dict(schema.KERNELS)))
    if adm is None or not idle:
        reason = "no admissible idle cell; admissibility record missing" if adm is None else "no admissible idle cell"
        rows.append({"rung": rung, "grid_id": grid_id, "part": "i", "kernel": "idle", "n_cells": len(idle), "verdict": V.not_run(reason)})
        for k in kernels:
            rows.append({"rung": rung, "grid_id": grid_id, "part": "ii", "kernel": k,
                         "n_cells": sum(1 for c in kern if c["kernel"] == k), "verdict": V.not_run(reason)})
        part1 = {"verdict": V.not_run(reason)}
    else:
        part1 = _gf_part1(out, rung, grid_id, idle, n_perm=n_perm, test_frac=test_frac, n_estimators=n_estimators,
                          seed_offset=seed_offset, n_jobs=n_jobs, features=part1_features)
        if "exceeds" in part1:
            n_null = int((part1.get("summary") or {}).get("n") or 0)
            if perm_floor and 0 < n_null < perm_floor:
                part1["verdict"] = V.not_run(f"{n_null} permutations < {perm_floor}")     # SPEC_epoch2 B3; the numbers stay
            elif not part1["exceeds"]:
                part1["verdict"] = V.GF_INSEPARABLE
            else:
                part1["verdict"] = V.GF_VOID if part1_consequence == "void" else GF_SEPARABLE_REPORTED
        rows.append({"rung": rung, "grid_id": grid_id, "part": "i", "kernel": "idle", "n_cells": len(idle),
                     "score": part1.get("score"), "null_p95": part1.get("null_p95"), "verdict": part1["verdict"]})
        if rung != "combined":
            idle_meds = [S.cell_headline(S.load_extract_cached(out, c["cell_id"]), rung, S.head_drop_for(hd, c["kernel"], c["role"])) for c in idle]
            lo, hi = float(np.nanmin(idle_meds)), float(np.nanmax(idle_meds))
            for k in kernels:
                meds = [S.cell_headline(S.load_extract_cached(out, c["cell_id"]), rung, S.head_drop_for(hd, c["kernel"], c["role"])) for c in kern if c["kernel"] == k]
                inside = [lo <= m <= hi for m in meds]
                if part2_rule == "all_cells_outside":
                    ok = not any(inside)
                elif part2_rule == "median_of_cells_outside":
                    ok = not (lo <= float(np.nanmedian(meds)) <= hi)
                else:
                    raise ValueError(part2_rule)
                rows.append({"rung": rung, "grid_id": grid_id, "part": "ii", "kernel": k, "n_cells": len(meds),
                             "score": float(np.nanmedian(meds)), "n_inside_envelope": int(sum(inside)),
                             "envelope_lo": lo, "envelope_hi": hi, "verdict": V.PASS if ok else V.GF_AT_FLOOR})
        else:
            for k in kernels:
                rows.append({"rung": rung, "grid_id": grid_id, "part": "ii", "kernel": k,
                             "n_cells": sum(1 for c in kern if c["kernel"] == k),
                             "verdict": V.not_applicable("combined has no headline reading (SPEC 3.1.5)")})
        # the three floors
        Ks = np.concatenate([S.load_extract_cached(out, c["cell_id"])["K"] for c in idle])
        l0s = np.concatenate([S.load_extract_cached(out, c["cell_id"])["l0_q50_all"] for c in idle])
        Js = np.concatenate([S.load_extract_cached(out, c["cell_id"])["J"] for c in idle])
        q = list(schema.QUANTILES)
        floors = {"K": np.quantile(Ks[~np.isnan(Ks)], q).tolist() if np.any(~np.isnan(Ks)) else None,
                  "l0_per_changed_page_q50_all": np.quantile(l0s[~np.isnan(l0s)], q).tolist() if np.any(~np.isnan(l0s)) else None,
                  "J": np.quantile(Js[~np.isnan(Js)], q).tolist() if np.any(~np.isnan(Js)) else None,
                  "quantiles": q, "n_idle_cells": len(idle),
                  "note": "l0 floor pooled from per-snapshot medians (a quantile of quantiles; SPEC_review_al_kindi.md section 3 item 4)"}
        S.write_json(out / "gates" / "gf_floors.json", "plan11.gf_floors.v1",
                     {"idle_cells": [c["cell_id"] for c in idle]}, CIT_GF, floors)
    p = out / "gates" / "gf.csv"
    old = [r for r in S.read_csv(p)] if p.is_file() else []
    old = [r for r in old if not (r["rung"] == rung and r["grid_id"] == grid_id)]
    S.write_csv(p, cols, old + rows)
    params = {"rung": rung, "grid_id": grid_id, "grid_source": gsrc, "n_perm": n_perm, "part1_design": part1_design,
              "test_frac": test_frac, "part2_rule": part2_rule, "part1_consequence": part1_consequence, "part1_features": part1_features,
              "n_estimators": n_estimators, "seed_offset": seed_offset, "admissibility_record": adm,
              "perm_floor": perm_floor, "head_drop_idle": S.head_drop_for(hd, "idle", "idle"),
              "n_idle_cells": len(idle), "part1_summary": part1.get("summary"),
              "inputs_sha256": S.inputs_sha256([out / "cells.csv", adm_path, out / "gates" / "preconditions.csv",
                                                out / "gates" / "selection.json"], out)}
    pj = out / "gates" / f"gf.{rung}.{grid_id}.params.json"
    S.write_json(pj, "plan11.gf.v1", params, CIT_GF, {"rows": rows})
    return p


# --------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="gates_precondition.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("preconditions")
    a.add_argument("--out", required=True)
    a.add_argument("--cells-csv", default=None)
    a.add_argument("--assume-failed-zero", action="store_true")
    a.add_argument("--assume-reason", default="")
    a.add_argument("--failed-counts", default=None)
    a.add_argument("--c1-rule", default=C1_RULE_DEFAULT, choices=C1_RULES,
                   help="C1 for kernel cells (AD 2026-09-17; SPEC_epoch2 section 4): auto = idle_floor once an idle cell enters the floor, "
                        "else absolute; the function default is legacy_apf_max, the CLI default is auto")
    a.add_argument("--c1-abs-fraction", type=float, default=C1_ABS_FRACTION,
                   help="the absolute rule's fraction of memory (K_max >= int(fraction * N); 0.001 = 262 pages)")
    a.add_argument("--c1-idle-percentile", type=float, default=C1_IDLE_PERCENTILE, help="the idle floor's percentile of K (G-K0's edge)")
    a.add_argument("--c1-idle-pool", default=C1_IDLE_POOL, choices=("pooled_snapshots", "cell_medians"))
    a.add_argument("--c1-min-idle-cells", type=int, default=C1_MIN_IDLE_CELLS, help="auto takes the floor once this many idle cells enter it")
    a.add_argument("--c1-activity-min", type=float, default=C1_ACTIVITY_MIN,
                   help="the inherited apf_queue re-map threshold (SPEC 8.36; 0.02 = 5,243 pages), applied only under --c1-rule legacy_apf_max")
    a.add_argument("--c1-activity-min-pages", type=int, default=None,
                   help="the absolute rule's threshold as a page count (AA T1: 200; SPEC_epoch2 B1); overrides --c1-abs-fraction when given")
    for name in ("gk0-template", "idle-admissibility-template"):
        t = sub.add_parser(name)
        t.add_argument("--out", required=True)
        t.add_argument("--force", action="store_true", help="overwrite an existing author input (SPEC_epoch2 B20; default: kept)")
    g = sub.add_parser("gk0")
    g.add_argument("--out", required=True)
    g.add_argument("--tail-fraction", type=float, default=GK0_TAIL_FRACTION)
    g.add_argument("--idle-pool", default=GK0_IDLE_POOL, choices=("pooled_snapshots", "cell_medians"))
    g.add_argument("--idle-percentile", type=float, default=GK0_IDLE_PERCENTILE)
    f = sub.add_parser("gf")
    f.add_argument("--out", required=True)
    r = f.add_mutually_exclusive_group(required=True)
    r.add_argument("--rung", choices=S.RUNGS)
    r.add_argument("--all-rungs", action="store_true")
    f.add_argument("--grid-id", default=None)
    f.add_argument("--n-perm", type=int, default=GF_N_PERM)
    f.add_argument("--part1-design", default=GF_PART1_DESIGN, choices=("within_trace_window",))
    f.add_argument("--test-frac", type=float, default=GF_TEST_FRAC)
    f.add_argument("--part2-rule", default=GF_PART2_RULE, choices=("all_cells_outside", "median_of_cells_outside"))
    f.add_argument("--part1-consequence", default=GF_PART1_CONSEQUENCE, choices=("void", "report"))
    f.add_argument("--part1-features", default=GF_PART1_FEATURES, choices=("norm", "raw"))
    f.add_argument("--n-estimators", type=int, default=300)
    f.add_argument("--n-jobs", type=int, default=1)
    f.add_argument("--seed-offset", type=int, default=0)
    f.add_argument("--perm-floor", type=int, default=GF_PERM_FLOOR_CLI,
                   help="part (i) reads `not run: N permutations < floor` when 0 < n_perm < floor (B1-G1's floor, SPEC_epoch2 B3); "
                        "the function default is 0 (judge on any count)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if args.cmd == "gk0-template":
        p = out / "inputs" / "gk0_source.csv"
        if p.is_file() and not args.force:
            print(f"kept: author input exists: {p}"); return 0
        print(write_gk0_template(p)); return 0
    if args.cmd == "idle-admissibility-template":
        p = out / "inputs" / "idle_admissibility.json"
        if p.is_file() and not args.force:
            print(f"kept: author input exists: {p}"); return 0
        print(write_idle_admissibility_template(p)); return 0
    if not S.cells_csv_path(out, getattr(args, "cells_csv", None)).is_file():
        print(f"missing input: {S.cells_csv_path(out, getattr(args, 'cells_csv', None))}", file=sys.stderr)
        return 2
    try:
        if args.cmd == "preconditions":
            if args.assume_failed_zero and not args.assume_reason:
                print("--assume-failed-zero needs --assume-reason TEXT", file=sys.stderr); return 2
            print(gate_preconditions(out, cells_csv=args.cells_csv, assume_failed_zero=args.assume_failed_zero,
                                     assume_reason=args.assume_reason, failed_counts_csv=args.failed_counts,
                                     c1_activity_min=args.c1_activity_min, c1_rule=args.c1_rule,
                                     c1_abs_fraction=args.c1_abs_fraction, c1_idle_percentile=args.c1_idle_percentile,
                                     c1_idle_pool=args.c1_idle_pool, c1_min_idle_cells=args.c1_min_idle_cells,
                                     c1_activity_min_pages=args.c1_activity_min_pages))
        elif args.cmd == "gk0":
            print(gate_gk0(out, tail_fraction=args.tail_fraction, idle_pool=args.idle_pool, idle_percentile=args.idle_percentile))
        elif args.cmd == "gf":
            rungs = S.RUNGS if args.all_rungs else (args.rung,)
            for r_ in rungs:
                print(gate_gf(out, rung=r_, grid_id=args.grid_id, n_perm=args.n_perm, part1_design=args.part1_design,
                              test_frac=args.test_frac, part2_rule=args.part2_rule, part1_consequence=args.part1_consequence,
                              part1_features=args.part1_features, n_estimators=args.n_estimators, seed_offset=args.seed_offset, n_jobs=args.n_jobs,
                              perm_floor=args.perm_floor))
    except FileNotFoundError as e:
        print(f"missing input: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
