#!/usr/bin/env python3
"""gates_detection.py -- the gates of the detection layer (SPEC_DETECTION.md section 3.5, builder A):
detection admissibility, G-K0 for every cell (three quantities, three verdicts), G-N two-class,
G-L (i) two-class, G-OP, G-LM, G-ANCHOR (kernels; idle sets; early-against-late idle), the order
test with the drift regression, G-SIG, G-FP, G-1C, the harness clause, G-CAL, G-M two-class,
G-DIM, the alias falsifier, G-V two-class, the miss table, the leak probes and the head-drop
template.

Every function writes one CSV under gates/detection/ plus a .params.json beside it
(series.write_params), replaces its own rows on a re-run (keyed by rung and the row key), and
never edits another gate's file. Every function reads the split results of detection_metrics by
file and never re-fits a model except where the definition says "recomputed" (G-LM, G-FP), and
then it calls run_detection_split with subset_workloads or exclude_families.

Corrections from the reviews of SPEC_DETECTION.md applied here (each marked "must change"):
  al-Kindi 2.5, al-Farabi (for the author 3)   det_c1_rule applies to every class alike;
  al-Kindi 2.7, 2.8   the miss table's identity is J - J_null (``--identity excess|raw``), the G-J mask
                      is one per-pair rule for every class, ``axis`` is the resemblance axis and
                      ``axis_of_largest`` the residual axis, with d_amount and d_identity printed;
  al-Kindi 2.9        the alias regression uses workload fixed effects; the ``cadence_as_class`` row;
  al-Kindi 2.10       the drift regression uses workload fixed effects; the plain row is labelled;
  al-Kindi 2.11       the order test's half is within each workload by default; the within-class row beside it;
  ML 2.3              G-OP counts the threshold setters per fold and rolls up as the worst fold;
  ML 2.6              the leak probes (n_pairs, dt_est_s, frac_above_band, K_med) under LOWO with the null;
  ML 2.8              gv_two_class writes L0_member and L0_ratio per member (REPS_IDENTICAL_RATIO);
  al-Farabi M5        ``head-drop-template`` writes one zero row per workload key of every class;
  al-Farabi M7        GCAL_SPREAD_RULE, DRIFT_NULL_PERCENTILE, GV_NOTE_FRACTION, LEVEL_WORKLOAD_AGG,
                      GOP_SETTER_RULE are module constants written into params;
  al-Farabi M8        the idle band edge is recomputed by gates/gk0.json's own rule; a mismatch is a written refusal;
  al-Farabi M9        unassigned cells are listed in admissibility.csv with their reason;
  al-Farabi M12       ORDER_TEST_SCOPE is a constant and a CLI flag.

No server path appears here. No sandbox workload is named here.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plan11_encoding_ladder import schema  # noqa: E402
from plan11_encoding_ladder import series as S  # noqa: E402
from plan11_encoding_ladder import verdicts as V  # noqa: E402
from plan11_encoding_ladder import classes as C  # noqa: E402
from plan11_encoding_ladder import detection_splits as DS  # noqa: E402
from plan11_encoding_ladder import detection_metrics as DM  # noqa: E402
from plan11_encoding_ladder import gates_precondition as GP  # noqa: E402
from plan11_encoding_ladder.nulls import SEED_FOREST, SEED_LABEL_NULL  # noqa: E402

from sklearn.metrics import roc_auc_score  # noqa: E402

# --------------------------------------------------------------------------- constants
DET_C1_RULE = "report"               # section 7 item 7; al-Kindi 2.5: the same rule for every class
GK0_IDLE_PERCENTILE = 95.0           # the idle band edge: plan11's rule (SPEC 3.3.3), the same number
GK0_ENVELOPE_PERCENTILE = 95.0       # OPEN in CR3 2.8; section 7 item 14
GK0_VERDICT_QUANTITIES = ("K_med", "K_q90", "frac_above_band")    # section 7 item 14
GK0_QUANTITIES = ("K_med", "K_q90", "frac_above_band", "l0_med_above", "J_consec_above")
GN_SANDBOX_HEADLINE_MIN = 3
GOP_MIN_CELLS = 5; GOP_MIN_WORKLOADS = 3
GOP_CELLS = "threshold_setters"      # section 7 item 18; alternative "realized_fps"
GOP_SETTER_RULE = "ge"               # al-Farabi M7: a setter is a benign training cell whose in-fold score is >= the threshold
GOP_ROLLUP = "worst_fold"            # ML review 2.3
LEVEL_QUANTITY = "median_K"          # stage 1; "per_iteration_K_sum" when inputs/iteration_boundaries.csv exists; section 7 item 16
LEVEL_WORKLOAD_AGG = "median"        # al-Farabi M7
GLM_BAND_FACTOR = 2.0                # ML question 13: a factor of two
GLM_MODEL = "retrain"                # section 7 item 16
GLM_VANISH_RULE = "tpr_le_fpr"       # section 7 item 16
ORDER_TEST_CONSEQUENCE = "size"      # stage 1 (P3 0a); "void" for an interleaved campaign (K3 F5); section 7 item 19
ORDER_TEST_SCOPE = "within_class"    # al-Farabi M12; alternative "campaign"
ORDER_HALF_RULES = ("within_workload", "within_class")   # al-Kindi 2.11: both rows written, the first is the default
ORDER_NULL_PERM = 500
DRIFT_NULL_PERCENTILE = 95.0         # al-Farabi M7
GFP_FLAG_FRACTION = 0.5              # CR3 2.15
HARNESS_COMPARABLE_TOL = 0.10        # OPEN in CR3 2.21; section 7 item 22
HARNESS_RELAUNCH_RULE = "ge_median_member_recall"
GCAL_SPREAD_RULE = DM.GCAL_SPREAD_RULE
GM_ALPHA = 0.05; GM_N_SEEDS = 5
ALIAS_TOP_K = 10; ALIAS_R2 = 0.5
ALIAS_UNIT = "within_workload"       # al-Kindi 2.9
GV_NOTE_FRACTION = 0.5               # al-Farabi M7
REPS_IDENTICAL_RATIO = 0.1           # ML review 2.8 (proposed default; the author's)
MISS_AXES = ("amount", "identity")
MISS_DISTANCE = "standardized_euclidean_to_centroid"   # section 7 item 25; alternative "nearest_cell"
MISS_IDENTITY = "excess"             # al-Kindi 2.7: J - J_null; alternative "raw"
LEAK_QUANTITIES = ("n_pairs", "dt_est_s", "frac_above_band", "K_med")   # ML review 2.6
GM_ROWS = (("apf raw", "apf", False, "lowo"), ("apf", "apf", True, "lowo"), ("wapf", "wapf", True, "lowo"), ("persist", "persist", True, "lowo"),
           ("content", "content", True, "lowo"), ("combined", "combined", True, "lowo"), ("combined (matched)", "combined", True, "lowo_matched"))

CIT_ADM = "CR3 2.23 (the validity check per cell, extended; refusals written, never absorbed); K3 move 2; SPEC 3.3.1; al-Kindi review 2.5"
CIT_GK0 = "CR3 2.8 (three quantities, three verdicts, the source part by the author); K3 F3; ML 3.6 ('at floor: undetectable by construction')"
CIT_GN = "CR3 2.5; ML 2.6"
CIT_GL = "CR3 2.4 (G-L part (i) two-class); ML 3.6, 4.2; K3 F1"
CIT_GOP = "CR3 2.13; ML 2.3 (the operating point's support; per fold, worst fold: ML review 2.3)"
CIT_GLM = "CR3 2.17 (the operating point recomputed against the benign workloads whose level lies within a declared band); ML 4.3; VME section 4 item 7 through K3 F1"
CIT_ANCHOR = "CR3 2.14; ML 3.1; K3 move 4 and 21 (read as a size, never a switch between two headlines: K3 section 5 point 3)"
CIT_ORDER = "ML 3.2; CR3 2.20; P3 0a (the order test is a required row of RQ3); K3 F5; al-Kindi review 2.11; al-Farabi M12"
CIT_DRIFT = "CR3 2.20 ('a regression of the per-cell floor statistics on the cell index'); al-Kindi review 2.10 (workload fixed effects)"
CIT_GSIG = "CR3 2.16; ML 4.3; K3 F2"
CIT_GFP = "CR3 2.15; ML 4.3; K3 F8"
CIT_G1C = "CR3 2.19; K3 section 5 point 1"
CIT_HARNESS = "CR3 2.21; K3 F4 and move 8; CR3 2.12 (the harness clause of the tripwire)"
CIT_GCAL = "CR3 2.18; ML 1.6 item 2"
CIT_GM = "CR3 2.7; ML 2.5 (the exact one-sided binomial over non-tied units)"
CIT_GDIM = "CR3 2.2 (G-DIM unchanged); SPEC 3.7.7"
CIT_ALIAS = "CR3 2.26; SPEC 3.4.4 (the epoch-1 form); al-Kindi review 2.9 (workload fixed effects; the cadence row); ML 3.4"
CIT_GV = "CR3 2.11 (G-V two-class); K3 move 20; ML review 2.8 (the rep-identity disclosure)"
CIT_MISS = "K3 move 18; N3 Sec. 1 RQ2; al-Kindi review 2.7, 2.8"
CIT_LEAK = "ML 3.4, 3.5 (the cadence and active-fraction probes); ML review 2.6"
CIT_HEAD = "al-Farabi M5; al-Kindi review 2.6 (the head drop is a declared constant per workload key)"


# --------------------------------------------------------------------------- helpers

def _out(out) -> Path:
    return Path(out)


def _det(out) -> Path:
    return C.det_dir(out)


def _grid(out: Path, rung: str, grid_id: str | None) -> tuple:
    if grid_id:
        return grid_id, C.grid_source(out)
    gid, _ = S.selected_grid_id(out, rung, None)
    return gid, (C.grid_source(out) if gid else "no selection")


def _no_selection(rung: str) -> str:
    return V.not_run(f"no selection for {rung} (run classes inherit-selection)")


def _replace_rows(path: Path, columns, new_rows: list[dict], key) -> Path:
    """Write ``new_rows`` into ``path``, replacing the existing rows with the same key (a function of a row)."""
    new_keys = {key(r) for r in new_rows}
    old = [r for r in S.read_csv(path) if key(r) not in new_keys] if path.is_file() else []
    return S.write_csv(path, columns, old + new_rows)


def _params(out: Path, extra: dict, inputs: list) -> dict:
    p = dict(extra)
    p["grid_source_file"] = C.grid_source(out)
    p["inputs_sha256"] = S.inputs_sha256(inputs, out)
    return p


def _scores(out: Path, rung: str, gid: str, name: str, normalized: bool = True) -> dict | None:
    return DM.read_scores(out, rung, gid, name, normalized)


def _preds(out: Path, rung: str, gid: str, name: str, normalized: bool = True) -> list[dict]:
    p = DM.split_dir(out, rung, gid, name, normalized) / "predictions.csv"
    return S.read_csv(p) if p.is_file() else []


def _join_ok(out: Path) -> list[dict]:
    return C.load_join(out)


def _adm_set(out: Path, pair: bool = False) -> set:
    p = _det(out) / "admissibility.csv"
    if not p.is_file():
        return set()
    col = "admissible_pair_rungs" if pair else "admissible"
    return {r["cell_id"] for r in S.read_csv(p) if str(r.get(col, "")).lower() == "true"}


def _floor(out: Path) -> dict:
    return DM.floor_verdicts(out) or {}


def _num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _is_num(x) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool) and np.isfinite(x)


# --------------------------------------------------------------------------- 3.5.1 admissibility

ADM_COLUMNS = ("cell_id", "class", "status_cells_csv", "all_hard_pass", "C1", "C2", "C6", "failed_verdict", "c1_rule_applied",
               "admissible", "admissible_pair_rungs", "reason")


def admissibility(out: Path, *, det_c1_rule: str = DET_C1_RULE) -> Path:
    """gates/detection/admissibility.csv: cell_id, class, status_cells_csv, all_hard_pass, C1, C2, C6,
    failed_verdict, c1_rule_applied, admissible, admissible_pair_rungs, reason. Under det_c1_rule
    "report" (the default) a cell of ANY class that fails C1 stays admissible with its floor verdict
    coming from G-K0 (al-Kindi review 2.5: the inclusion rule reads no class label; K3 F3; CR3 2.8);
    under "exclude" plan11's rule applies to every class. admissible = status ok and C2 == pass and
    C6 == pass and (C1 in (pass, not applicable) or (C1 == fail and det_c1_rule == "report")); a cell
    admitted through the report rule has reason 'C1 fail: reported, floor verdict by G-K0 (K3 F3)'.
    admissible_pair_rungs additionally needs failed_verdict not a refusal (SPEC 3.3.2). A cell with
    no class row is listed with admissible = false and reason 'unassigned: no class row in
    inputs/classes.csv' (al-Farabi M9). With gates/preconditions.csv absent every row reads
    admissible = false, reason = not_run('gates/preconditions.csv missing'). The join's S and B are
    refreshed afterwards (SPEC_DETECTION 2.4). Citation: CIT_ADM."""
    out = _out(out)
    if det_c1_rule not in ("report", "exclude"):
        raise ValueError(det_c1_rule)
    cells = S.read_csv(out / "cells.csv")
    join = {j["cell_id"]: j for j in C.load_join(out, include_unassigned=True)}
    pm = S.preconditions_map(out)
    rows, counts = [], {}
    for c in cells:
        cid = c["cell_id"]
        j = join.get(cid)
        cls = j["class"] if j else C.UNASSIGNED
        st = c.get("status", "ok")
        row = {"cell_id": cid, "class": cls, "status_cells_csv": st, "all_hard_pass": "", "C1": "", "C2": "", "C6": "", "failed_verdict": "",
               "c1_rule_applied": det_c1_rule, "admissible": False, "admissible_pair_rungs": False, "reason": ""}
        pr = pm.get(cid)
        if pr is not None:
            row.update({"all_hard_pass": str(pr.get("all_hard_pass", "")).lower() == "true", "C1": pr.get("C1", ""), "C2": pr.get("C2", ""),
                        "C6": pr.get("C6", ""), "failed_verdict": pr.get("failed_verdict", "")})
        if cls == C.UNASSIGNED:
            row["reason"] = "unassigned: no class row in inputs/classes.csv"
        elif not pm:
            row["reason"] = V.not_run("gates/preconditions.csv missing")
        elif st != schema.STATUS_OK:
            row["reason"] = f"cells.csv status {st}"
        elif pr is None:
            row["reason"] = V.not_run("cell not in gates/preconditions.csv")
        else:
            c1, c2, c6 = row["C1"], row["C2"], row["C6"]
            c1_ok = c1 == V.PASS or str(c1).startswith("not applicable")
            c1_rep = (c1 == V.FAIL and det_c1_rule == "report")
            adm = (c2 == V.PASS and c6 == V.PASS and (c1_ok or c1_rep))
            row["admissible"] = bool(adm)
            if adm:
                row["admissible_pair_rungs"] = not V.is_refusal(row["failed_verdict"] or V.PASS)
                row["reason"] = "C1 fail: reported, floor verdict by G-K0 (K3 F3)" if c1_rep else "ok"
                if adm and not row["admissible_pair_rungs"]:
                    row["reason"] += "; pair rungs excluded: failed verdict is a refusal (SPEC 3.3.2)"
            else:
                why = []
                if c2 != V.PASS: why.append("C2 fail")
                if c6 != V.PASS: why.append("C6 fail")
                if not (c1_ok or c1_rep): why.append(f"C1 fail under det_c1_rule = {det_c1_rule}")
                row["reason"] = "; ".join(why) or "not all_hard_pass"
        d = counts.setdefault(cls, {"n_cells": 0, "n_admissible": 0, "n_admissible_pair_rungs": 0, "n_c1_fail_reported": 0, "n_excluded": 0})
        d["n_cells"] += 1; d["n_admissible"] += int(row["admissible"]); d["n_admissible_pair_rungs"] += int(row["admissible_pair_rungs"])
        d["n_c1_fail_reported"] += int(row["reason"].startswith("C1 fail: reported")); d["n_excluded"] += int(not row["admissible"])
        rows.append(row)
    p = S.write_csv(_det(out) / "admissibility.csv", ADM_COLUMNS, rows)
    params = _params(out, {"det_c1_rule": det_c1_rule, "preconditions_present": bool(pm)}, [out / "cells.csv", out / "gates" / "preconditions.csv", C.join_path(out)])
    S.write_json(_det(out) / "admissibility.json", "plan11.detection.admissibility.v1", params, CIT_ADM,
                 {"n_cells": len(rows), "n_admissible": sum(1 for r in rows if r["admissible"]), "counts_per_class": counts,
                  "unassigned_cells": [r["cell_id"] for r in rows if r["class"] == C.UNASSIGNED]})
    # refresh S and B in the join (SPEC_DETECTION 2.4: S is filled after admissibility)
    jp = _det(out) / "cell_classes.json"
    if jp.is_file() and C.classes_path(out).is_file():
        prm = (S.read_json(jp).get("params") or {})
        C.build_join(out, kernel_family_rule=prm.get("kernel_family_rule", C.KERNEL_FAMILY_RULE),
                     relaunched_grouping=prm.get("relaunched_grouping", C.RELAUNCHED_GROUPING), campaign_label=prm.get("campaign_label"))
    return p


# --------------------------------------------------------------------------- 3.5.2 G-K0 two-class

GK0_CELL_COLUMNS = ("cell_id", "class", "workload_key", "member_index", "K_med", "K_q90", "frac_above_band", "l0_med_above", "J_consec_above",
                    "idle_band_edge", "env_K_med", "env_K_q90", "env_frac", "harness_env_K_med", "harness_env_K_q90", "harness_env_frac", "verdict")
GK0_MEMBER_COLUMNS = ("member_index", "subfamily_letter", "source_statement", "n_cells", "n_at_floor", "n_at_harness_floor", "n_above_floor", "verdict")


def gk0_sandbox_template(out: Path) -> Path:
    """inputs/gk0_source_sandbox.csv: one numbered row per member (member_index,
    steady_state_changes_content = 'unstated', source); the author fills it; agents write nothing
    about the members (CR3 2.8; ML 4.2; section 7 item 15). An existing file is kept."""
    out = _out(out)
    p = out / "inputs" / "gk0_source_sandbox.csv"
    if p.is_file():
        return p
    members = sorted({int(j["member_index"]) for j in C.load_join(out) if j["class"] == "sandbox"})
    rows = [{"member_index": m, "steady_state_changes_content": "unstated", "source": "the author fills it (CR3 2.8); never a name"} for m in members]
    return S.write_csv(p, ("member_index", "steady_state_changes_content", "source"), rows)


def _idle_edge(out: Path, idle_ids: list[str], idle_pool: str, idle_percentile: float) -> float | None:
    if not idle_ids:
        return None
    if idle_pool == "pooled_snapshots":
        pooled = np.concatenate([S.load_extract_cached(out, c)["K"] for c in idle_ids])
        return float(np.percentile(pooled, idle_percentile))
    if idle_pool == "cell_medians":
        meds = [float(np.median(S.load_extract_cached(out, c)["K"])) for c in idle_ids]
        return float(np.percentile(meds, idle_percentile))
    raise ValueError(idle_pool)


def gk0_cell_quantities(ex: dict, edge: float | None) -> dict:
    """The measured part per cell (CR3 2.8): K_med (median K over all rows), K_q90 (90th percentile
    of K), frac_above_band (fraction of rows with K > the idle band edge), l0_med_above (median of
    l0_q50_all over the rows above the band; None when none), J_consec_above (median J over the
    pairs above the band; None when none)."""
    K = np.asarray(ex["K"], dtype=np.float64)
    q = {"K_med": float(np.median(K)) if len(K) else None, "K_q90": float(np.percentile(K, 90)) if len(K) else None,
         "frac_above_band": None, "l0_med_above": None, "J_consec_above": None}
    if edge is None or not len(K):
        return q
    above = K > edge
    q["frac_above_band"] = float(np.mean(above))
    if above.any():
        l0 = np.asarray(ex.get("l0_q50_all"), dtype=np.float64)[above]
        l0 = l0[np.isfinite(l0)]
        q["l0_med_above"] = float(np.median(l0)) if len(l0) else None
        J = np.asarray(ex.get("J"), dtype=np.float64)[above]
        J = J[np.isfinite(J)]
        q["J_consec_above"] = float(np.median(J)) if len(J) else None
    return q


def gk0_cells(out: Path, *, idle_percentile: float = GK0_IDLE_PERCENTILE, envelope_percentile: float = GK0_ENVELOPE_PERCENTILE,
              verdict_quantities=GK0_VERDICT_QUANTITIES) -> Path:
    """G-K0 for every cell, three quantities, three verdicts (CR3 2.8; K3 F3; K3 move 5). Source part:
    inputs/gk0_source_sandbox.csv (the author's). Measured part, per admissible cell of every class,
    from extract.csv (gk0_cell_quantities) against the idle band edge: the idle_percentile-th
    percentile of K pooled over every admissible idle cell's rows, by the rule of gates/gk0.json's
    params (idle_pool, idle_percentile) when that file exists, so the edge is plan11's; a mismatch
    with gates/gk0.csv's idle_band_edge is the written refusal 'refused: idle band edge differs from
    gates/gk0.csv (<a> against <b>)' on every cell (al-Farabi M8). Envelopes: for each quantity the
    envelope_percentile-th percentile over the idle cells' per-cell values (the idle envelope) and,
    when harness_idle cells exist, over the harness-idle cells' (the harness envelope). Verdict per
    cell, in order: GK0_AT_FLOOR when every quantity in verdict_quantities is <= its idle envelope
    edge; GK0_AT_HARNESS_FLOOR when not at floor, harness cells exist, and every quantity is <= its
    harness envelope edge; else GK0_ABOVE_FLOOR. Idle and harness-idle cells carry 'control'. Without
    an admissible idle cell every verdict is not_run('no admissible idle cell'). Writes gk0_cells.csv,
    gk0_members.csv (verdict = the common verdict or GK0_MIXED), gk0.json. Citation: CIT_GK0."""
    out = _out(out)
    join = [j for j in C.load_join(out) if j["cell_id"] in _adm_set(out)]
    idle = [j["cell_id"] for j in join if j["class"] == "idle"]
    harness = [j["cell_id"] for j in join if j["class"] == "harness_idle"]
    gk0_json = out / "gates" / "gk0.params.json"      # written by gates_precondition.gate_gk0 through series.write_params
    idle_pool = GP.GK0_IDLE_POOL; pct = idle_percentile; plan11_edge = None
    if gk0_json.is_file():
        prm = S.read_json(gk0_json).get("params") or {}
        idle_pool = prm.get("idle_pool", idle_pool); pct = float(prm.get("idle_percentile", pct))
        plan11_edge = _num(prm.get("idle_band_edge"))
    edge = _idle_edge(out, idle, idle_pool, pct)
    refusal = None
    if edge is not None and plan11_edge is not None and abs(edge - plan11_edge) > 1e-9:
        refusal = V.refused(f"idle band edge differs from gates/gk0.csv ({S.fmt_num(edge)} against {S.fmt_num(plan11_edge)})")
    quants = {j["cell_id"]: gk0_cell_quantities(S.load_extract_cached(out, j["cell_id"]), edge) for j in join}

    def envelope(ids):
        env = {}
        for q in GK0_QUANTITIES:
            v = [quants[c][q] for c in ids if quants[c][q] is not None]
            env[q] = float(np.percentile(v, envelope_percentile)) if v else None
        return env
    env_idle = envelope(idle) if idle else {q: None for q in GK0_QUANTITIES}
    env_h = envelope(harness) if harness else {q: None for q in GK0_QUANTITIES}
    src_p = out / "inputs" / "gk0_source_sandbox.csv"
    src = {str(int(float(r["member_index"]))): r.get("steady_state_changes_content", "unstated") for r in S.read_csv(src_p)} if src_p.is_file() else {}
    rows, per_member = [], {}
    for j in join:
        c = j["cell_id"]; q = quants[c]
        if refusal:
            verdict = refusal
        elif j["class"] in ("idle", "harness_idle"):
            verdict = "control"
        elif edge is None:
            verdict = V.not_run("no admissible idle cell")
        else:
            def inside(env):
                for k in verdict_quantities:
                    if q[k] is None or env.get(k) is None or not (q[k] <= env[k]):
                        return False
                return True
            if inside(env_idle):
                verdict = V.GK0_AT_FLOOR
            elif harness and inside(env_h):
                verdict = V.GK0_AT_HARNESS_FLOOR
            else:
                verdict = V.GK0_ABOVE_FLOOR
        rows.append({"cell_id": c, "class": j["class"], "workload_key": j["workload_key"], "member_index": j["member_index"] or "",
                     **{k: q[k] for k in GK0_QUANTITIES}, "idle_band_edge": edge, "env_K_med": env_idle["K_med"], "env_K_q90": env_idle["K_q90"],
                     "env_frac": env_idle["frac_above_band"], "harness_env_K_med": env_h["K_med"], "harness_env_K_q90": env_h["K_q90"],
                     "harness_env_frac": env_h["frac_above_band"], "verdict": verdict})
        if j["class"] in C.POSITIVE_CLASSES:
            d = per_member.setdefault(int(j["member_index"]), {"letter": j["subfamily_letter"], "verdicts": []})
            d["verdicts"].append(verdict)
    mrows = []
    for m in sorted(per_member):
        vs = per_member[m]["verdicts"]
        common = vs[0] if len(set(vs)) == 1 else V.GK0_MIXED
        mrows.append({"member_index": m, "subfamily_letter": per_member[m]["letter"], "source_statement": src.get(str(m), "unstated"), "n_cells": len(vs),
                      "n_at_floor": vs.count(V.GK0_AT_FLOOR), "n_at_harness_floor": vs.count(V.GK0_AT_HARNESS_FLOOR),
                      "n_above_floor": vs.count(V.GK0_ABOVE_FLOOR), "verdict": common})
    p = S.write_csv(_det(out) / "gk0_cells.csv", GK0_CELL_COLUMNS, rows)
    S.write_csv(_det(out) / "gk0_members.csv", GK0_MEMBER_COLUMNS, mrows)
    params = _params(out, {"idle_percentile": pct, "idle_pool": idle_pool, "envelope_percentile": envelope_percentile, "verdict_quantities": list(verdict_quantities),
                           "quantities": list(GK0_QUANTITIES), "n_idle_cells": len(idle), "n_harness_cells": len(harness), "gk0_source_sandbox_csv": str(src_p) if src_p.is_file() else None,
                           "plan11_edge_source": str(gk0_json) if gk0_json.is_file() else None},
                     [out / "cells.csv", C.join_path(out), _det(out) / "admissibility.csv", gk0_json, out / "gates" / "gk0.csv", src_p])
    S.write_json(_det(out) / "gk0.json", "plan11.detection.gk0.v1", params, CIT_GK0,
                 {"idle_band_edge": edge, "plan11_idle_band_edge": plan11_edge, "refusal": refusal, "idle_envelope": env_idle, "harness_envelope": env_h,
                  "members": mrows, "n_cells": len(rows), "verdict_counts": {v: sum(1 for r in rows if r["verdict"] == v) for v in sorted({r["verdict"] for r in rows})}})
    S.write_params(p, "plan11.detection.gk0.v1", params, CIT_GK0)
    return p


# --------------------------------------------------------------------------- 3.5.3 G-N two-class

GN_COLUMNS = ("row", "class", "n_workloads", "n_cells", "workloads", "status")


def gn_two_class(out: Path) -> Path:
    """gn.csv: one row for the sandbox class (row = 'sandbox', n_workloads = the members with at
    least one admissible above-floor cell, status GN_HEADLINE when >= 3, GN_ONE_TRAIN_WORKLOAD when
    2, GN_NO_SUPERVISED when 1 or 0) and one row per benign family (row = the family, n_workloads,
    status GN_HEADLINE when >= 2 else GN_SINGLE_WORKLOAD); columns row, class, n_workloads, n_cells,
    workloads (public keys), status. Refusal carried into the tables: a family-level sentence about
    the sandbox family is not written below three members. Citation: CIT_GN."""
    out = _out(out)
    adm = _adm_set(out); fb = _floor(out)
    join = [j for j in C.load_join(out) if j["cell_id"] in adm]
    rows = []
    sb = [j for j in join if j["class"] == "sandbox"]
    above = [j for j in sb if fb.get(j["cell_id"]) == V.GK0_ABOVE_FLOOR] if fb else sb
    wk = sorted({j["workload_key"] for j in above})
    n = len(wk)
    status = V.GN_HEADLINE if n >= GN_SANDBOX_HEADLINE_MIN else (V.GN_ONE_TRAIN_WORKLOAD if n == 2 else V.GN_NO_SUPERVISED)
    rows.append({"row": "sandbox", "class": "sandbox", "n_workloads": n, "n_cells": len(above), "workloads": ";".join(wk), "status": status})
    fams: dict = {}
    for j in join:
        if j["y"] == "benign":
            fams.setdefault(j["family"], []).append(j)
    for fam in sorted(fams):
        w = sorted({j["workload_key"] for j in fams[fam]})
        rows.append({"row": fam, "class": ";".join(sorted({j["class"] for j in fams[fam]})), "n_workloads": len(w), "n_cells": len(fams[fam]),
                     "workloads": ";".join(w), "status": V.GN_HEADLINE if len(w) >= 2 else V.GN_SINGLE_WORKLOAD})
    p = S.write_csv(_det(out) / "gn.csv", GN_COLUMNS, rows)
    S.write_params(p, "plan11.detection.gn.v1", _params(out, {"gn_sandbox_headline_min": GN_SANDBOX_HEADLINE_MIN, "floor_applied": bool(fb)},
                                                        [C.join_path(out), _det(out) / "admissibility.csv", _det(out) / "gk0_cells.csv"]), CIT_GN)
    return p


# --------------------------------------------------------------------------- 3.5.4 G-L (i) two-class

GL_COLUMNS = ("rung", "grid_id", "tpr05_norm", "null_p95_norm", "rank_norm", "tpr05_raw", "verdict")


def gl_two_class(out: Path, rung: str, grid_id: str | None = None) -> Path:
    """gl.csv row per rung: rung, grid_id, tpr05_norm, null_p95_norm, rank_norm, tpr05_raw (apf only;
    else ''), verdict: PASS when the normalized LOWO tpr05 null verdict is PASS; GL_LEVEL_ONLY when
    it is NULL_INSIDE (the rung's row is marked and cannot be cited as detecting behaviour; the raw
    row stays as the level-inclusive ceiling); the null's own refusal (NULL_NOT_ESTIMABLE, not run) is
    copied when the null did not decide. Citation: CIT_GL."""
    out = _out(out)
    gid, _ = _grid(out, rung, grid_id)
    p = _det(out) / "gl.csv"
    if gid is None:
        row = {"rung": rung, "grid_id": "", "verdict": _no_selection(rung)}
    else:
        sc = _scores(out, rung, gid, "lowo", True)
        row = {"rung": rung, "grid_id": gid, "tpr05_norm": "", "null_p95_norm": "", "rank_norm": "", "tpr05_raw": "", "verdict": ""}
        if sc is None:
            row["verdict"] = V.not_run("lowo norm split not run")
        elif sc.get("status") != "ok":
            row["verdict"] = sc["status"]
        else:
            row["tpr05_norm"] = sc.get("tpr_05")
            nb = (sc.get("null") or {}).get(DM.NULL_VERDICT_STATISTIC) or {}
            row["null_p95_norm"] = nb.get("p95"); row["rank_norm"] = nb.get("rank_text", "")
            nv = nb.get("verdict") or (sc.get("null") or {}).get("status") or V.not_run("null not run")
            row["verdict"] = V.PASS if nv == V.PASS else (V.GL_LEVEL_ONLY if nv == V.NULL_INSIDE else nv)
        if rung == "apf":
            sr = _scores(out, rung, gid, "lowo", False)
            row["tpr05_raw"] = sr.get("tpr_05") if sr and sr.get("status") == "ok" else (sr.get("status") if sr else V.not_run("lowo raw split not run"))
    _replace_rows(p, GL_COLUMNS, [row], lambda r: r["rung"])
    S.write_params(p, "plan11.detection.gl.v1", _params(out, {"null_verdict_statistic": DM.NULL_VERDICT_STATISTIC}, [out / "gates" / "selection.json"]), CIT_GL)
    return p


# --------------------------------------------------------------------------- 3.5.5 G-OP

GOP_COLUMNS = ("rung", "grid_id", "split", "variant", "fpr_declared", "threshold_in_fold", "cells_rule", "n_setter_cells", "n_setter_workloads",
               "n_folds_unsupported", "n_folds", "n_union_setter_cells", "n_union_setter_workloads", "setter_families", "n_realized_fp_cells",
               "n_realized_fp_workloads", "fp_families", "verdict")


def gop_verdict(n_cells: int, n_workloads: int, *, min_cells: int = GOP_MIN_CELLS, min_workloads: int = GOP_MIN_WORKLOADS) -> str:
    """GOP_SUPPORTED when n_cells >= min_cells and n_workloads >= min_workloads, else GOP_SET_BY_FEW (CR3 2.13; ML 2.3)."""
    return V.GOP_SUPPORTED if (int(n_cells) >= int(min_cells) and int(n_workloads) >= int(min_workloads)) else V.GOP_SET_BY_FEW


def gop(out: Path, rung: str, grid_id: str | None = None, *, split: str = "lowo", variant: str = "norm", min_cells: int = GOP_MIN_CELLS,
        min_workloads: int = GOP_MIN_WORKLOADS, cells_rule: str = GOP_CELLS) -> Path:
    """gop.csv row per (rung, split, variant): fpr_declared, threshold_in_fold (true), the setter
    counts, the realized false-positive counts, verdict. Under 'threshold_setters' the cells behind
    the declared rate are counted PER FOLD (the benign training cells whose in-fold score is >= the
    fold's threshold_05, GOP_SETTER_RULE) and rolled up as the worst fold (ML review 2.3: GOP_SUPPORTED
    only when every fold's setters number at least min_cells from at least min_workloads workloads;
    n_folds_unsupported and the union written as a disclosure); under 'realized_fps' the pooled
    out-of-fold benign cells flagged at threshold_05 are counted. verdict = GOP_SUPPORTED or
    GOP_SET_BY_FEW with the families named in setter_families; the true-positive rate at that
    operating point then carries no verb. Citation: CIT_GOP."""
    out = _out(out)
    gid, _ = _grid(out, rung, grid_id)
    p = _det(out) / "gop.csv"
    norm = variant == "norm"
    row = {"rung": rung, "grid_id": gid or "", "split": split, "variant": variant, "fpr_declared": DM.FPR_DECLARED, "threshold_in_fold": True, "cells_rule": cells_rule}
    sc = _scores(out, rung, gid, split, norm) if gid else None
    if gid is None:
        row["verdict"] = _no_selection(rung)
    elif sc is None or sc.get("status") != "ok":
        row["verdict"] = sc["status"] if sc else V.not_run(f"{split} {variant} split not run")
    else:
        g = sc.get("gop") or {}
        pf = g.get("per_fold") or []
        unsupported = [f for f in pf if gop_verdict(f["n_setter_cells"], f["n_setter_workloads"], min_cells=min_cells, min_workloads=min_workloads) != V.GOP_SUPPORTED]
        row.update({"n_setter_cells": g.get("worst_fold_n_setter_cells"), "n_setter_workloads": g.get("worst_fold_n_setter_workloads"),
                    "n_folds_unsupported": len(unsupported), "n_folds": len(pf), "n_union_setter_cells": g.get("n_union_setter_cells"),
                    "n_union_setter_workloads": g.get("n_union_setter_workloads"), "setter_families": ";".join(g.get("setter_families") or []),
                    "n_realized_fp_cells": g.get("n_realized_fp_cells"), "n_realized_fp_workloads": g.get("n_realized_fp_workloads"),
                    "fp_families": ";".join(g.get("fp_families") or [])})
        if cells_rule == "threshold_setters":
            row["verdict"] = V.GOP_SET_BY_FEW if (unsupported or not pf) else V.GOP_SUPPORTED
        elif cells_rule == "realized_fps":
            row["verdict"] = gop_verdict(g.get("n_realized_fp_cells") or 0, g.get("n_realized_fp_workloads") or 0, min_cells=min_cells, min_workloads=min_workloads)
        else:
            raise ValueError(cells_rule)
    _replace_rows(p, GOP_COLUMNS, [row], lambda r: (r["rung"], r["split"], r["variant"]))
    S.write_params(p, "plan11.detection.gop.v1", _params(out, {"min_cells": min_cells, "min_workloads": min_workloads, "cells_rule": cells_rule,
                                                                "setter_rule": GOP_SETTER_RULE, "rollup": GOP_ROLLUP}, [out / "gates" / "selection.json"]), CIT_GOP)
    return p


# --------------------------------------------------------------------------- 3.5.6 G-LM

GLM_COLUMNS = ("rung", "grid_id", "variant", "member_index", "subfamily_letter", "level_quantity", "level_label", "level", "band_lo", "band_hi",
               "band_workloads", "n_band_cells", "n_band_workloads", "recall_unrestricted", "fpr_unrestricted", "recall_lm", "fpr_lm",
               "threshold_lm_median", "verdict")


def level_of_workload(out: Path, join: list[dict], *, quantity: str = LEVEL_QUANTITY, iteration_quantile: float = 0.5,
                      agg: str = LEVEL_WORKLOAD_AGG) -> tuple[dict, dict]:
    """The level per workload key: the LEVEL_WORKLOAD_AGG ('median') over its admissible cells of the
    cell's level. median_K: series.k_median_cell(extract, head_drop) (the same statistic as plan11's
    normalization), labelled GLM_LABEL_MEDIAN_K. per_iteration_K_sum: from
    inputs/iteration_boundaries.csv, the sum of K over each iteration (between consecutive boundary
    seqs; the first iteration dropped, CR3 2.29) and the iteration_quantile of those sums per cell;
    a cell without boundaries reads not_run('no iteration boundary for <cell>') and the workload's
    level is None. Returns ({workload_key: level}, {cell_id: level or refusal string})."""
    out = _out(out)
    hd = S.load_head_drop(out / "inputs" / "head_drop.csv")
    bounds = DM.load_boundaries(out / "inputs" / "iteration_boundaries.csv") if quantity == "per_iteration_K_sum" else {}
    per_cell: dict = {}
    for j in join:
        c = j["cell_id"]
        ex = S.load_extract_cached(out, c)
        if quantity == "median_K":
            per_cell[c] = S.k_median_cell(ex, S.head_drop_for(hd, j["workload_key"]))
        elif quantity == "per_iteration_K_sum":
            b = bounds.get(c)
            if not b or len(b) < 2:
                per_cell[c] = V.not_run(f"no iteration boundary for {c}")
                continue
            seq = np.asarray(ex["seq"]); K = np.asarray(ex["K"], dtype=np.float64)
            sums = []
            for a, e in zip(b[:-1], b[1:]):
                m = (seq >= a) & (seq < e)
                if m.any():
                    sums.append(float(K[m].sum()))
            per_cell[c] = float(np.quantile(sums, iteration_quantile)) if sums else V.not_run(f"no iteration boundary for {c}")
        else:
            raise ValueError(quantity)
    per_wk: dict = {}
    for j in join:
        v = per_cell[j["cell_id"]]
        if isinstance(v, (int, float)) and np.isfinite(v):
            per_wk.setdefault(j["workload_key"], []).append(float(v))
    level = {}
    for wk, vals in per_wk.items():
        level[wk] = float(np.median(vals)) if agg == "median" else float(np.mean(vals))
    return level, per_cell


def glm_verdict(recall_unrestricted, fpr_unrestricted, recall_lm, fpr_lm, n_band_cells, n_band_workloads, *, rule: str = GLM_VANISH_RULE,
                min_cells: int = GOP_MIN_CELLS, min_workloads: int = GOP_MIN_WORKLOADS) -> str:
    """A pure function. GLM_EMPTY_BAND when n_band_workloads == 0; GLM_NOT_DETECTED when
    recall_unrestricted <= fpr_unrestricted (there is no detection to vanish; the miss table names
    what the member resembles); GOP_SET_BY_FEW + ' (level band)' when the band holds fewer than
    min_cells cells or min_workloads workloads (the level-matched operating point has no support; the
    level-matched recall is written but carries no verb); else GLM_LEVEL_ONLY when detection vanishes
    (rule 'tpr_le_fpr': recall_lm <= fpr_lm, the random scorer's line, ML 1.5; rule
    'below_half_unrestricted': recall_lm <= 0.5 * recall_unrestricted), else GLM_SURVIVES. Citation: CIT_GLM."""
    if int(n_band_workloads) == 0:
        return V.GLM_EMPTY_BAND
    if recall_unrestricted is None or fpr_unrestricted is None or float(recall_unrestricted) <= float(fpr_unrestricted):
        return V.GLM_NOT_DETECTED
    if int(n_band_cells) < int(min_cells) or int(n_band_workloads) < int(min_workloads):
        return V.GOP_SET_BY_FEW + " (level band)"
    if recall_lm is None or fpr_lm is None:
        return V.not_run("level-matched operating point missing")
    if rule == "tpr_le_fpr":
        vanished = float(recall_lm) <= float(fpr_lm)
    elif rule == "below_half_unrestricted":
        vanished = float(recall_lm) <= 0.5 * float(recall_unrestricted)
    else:
        raise ValueError(rule)
    return V.GLM_LEVEL_ONLY if vanished else V.GLM_SURVIVES


def glm(out: Path, rung: str, grid_id: str | None = None, *, quantity: str = LEVEL_QUANTITY, band_factor: float = GLM_BAND_FACTOR, model: str = GLM_MODEL,
        vanish_rule: str = GLM_VANISH_RULE, n_jobs: int = 1, seed_offset: int = 0, n_estimators: int = DM.N_ESTIMATORS,
        threshold_source: str = DM.THRESHOLD_SOURCE, iteration_quantile: float = 0.5) -> Path:
    """G-LM, the level-matched control, G-L part (iii) (CR3 2.17; ML 4.3; K3 F1). Per (rung, member m)
    on the norm features (and raw for apf); a member whose every cell is at floor gets the row
    verdict not_applicable('at floor (G-K0)') and no run. L = the member's level (level_of_workload);
    the band [L / band_factor, L * band_factor]; the level-matched benign = every benign workload
    whose level lies in the band. model 'retrain': run_detection_split(split='lowo', subset_workloads
    = every sandbox member + the band's benign workloads, dir_name 'glm_<m>'): the member's recall at
    the in-fold threshold set on the band benign, and the realized FPR on the band benign folds.
    model 'headline_oof': no retrain; the headline LOWO's out-of-fold scores, the threshold recomputed
    as the (1 - fpr) quantile of the band benign's out-of-fold scores (POST_HOC_LABEL). Every row's
    level_label is GLM_LABEL_MEDIAN_K in stage 1. Citation: CIT_GLM."""
    out = _out(out)
    gid, _ = _grid(out, rung, grid_id)
    p = _det(out) / "glm.csv"
    rows = []
    if gid is None:
        rows.append({"rung": rung, "grid_id": "", "variant": "norm", "member_index": "", "verdict": _no_selection(rung)})
    else:
        adm = _adm_set(out, pair=rung in S.PAIR_RUNGS); fb = _floor(out)
        join = [j for j in C.load_join(out) if j["cell_id"] in adm]
        level, per_cell = level_of_workload(out, join, quantity=quantity, iteration_quantile=iteration_quantile)
        label = V.GLM_LABEL_MEDIAN_K if quantity == "median_K" else f"per-iteration K sum, quantile {iteration_quantile}"
        members = sorted({int(j["member_index"]) for j in join if j["class"] == "sandbox"})
        benign_wk = sorted({j["workload_key"] for j in join if j["y"] == "benign"})
        for variant in (("norm", "raw") if rung == "apf" else ("norm",)):
            norm = variant == "norm"
            head = _scores(out, rung, gid, "lowo", norm)
            for m in members:
                wk = f"sandbox_member_{m}"
                letter = next(j["subfamily_letter"] for j in join if j["workload_key"] == wk)
                row = {"rung": rung, "grid_id": gid, "variant": variant, "member_index": m, "subfamily_letter": letter, "level_quantity": quantity, "level_label": label,
                       "level": level.get(wk), "band_lo": "", "band_hi": "", "band_workloads": "", "n_band_cells": "", "n_band_workloads": "",
                       "recall_unrestricted": "", "fpr_unrestricted": "", "recall_lm": "", "fpr_lm": "", "threshold_lm_median": "", "verdict": ""}
                cells_m = [j["cell_id"] for j in join if j["workload_key"] == wk]
                if fb and all(fb.get(c) in DM.FLOOR_VERDICTS_OUT for c in cells_m):
                    row["verdict"] = V.not_applicable("at floor (G-K0)"); rows.append(row); continue
                if level.get(wk) is None:
                    row["verdict"] = V.not_run(f"level undefined for member {m} under {quantity}"); rows.append(row); continue
                L = level[wk]; lo, hi = L / float(band_factor), L * float(band_factor)
                band = [w for w in benign_wk if level.get(w) is not None and lo <= level[w] <= hi]
                n_band_cells = sum(1 for j in join if j["workload_key"] in band)
                row.update({"band_lo": lo, "band_hi": hi, "band_workloads": ";".join(band), "n_band_cells": n_band_cells, "n_band_workloads": len(band)})
                ru = fu = None
                if head and head.get("status") == "ok":
                    pm = (head.get("per_member") or {}).get(str(m)) or {}
                    ru = pm.get("recall") if _is_num(pm.get("recall")) else None
                    fu = head.get("fpr_05_realized") if _is_num(head.get("fpr_05_realized")) else None
                row["recall_unrestricted"], row["fpr_unrestricted"] = ru, fu
                if not band:
                    row["verdict"] = glm_verdict(ru, fu, None, None, 0, 0, rule=vanish_rule); rows.append(row); continue
                if head is None or head.get("status") != "ok":
                    row["verdict"] = V.not_run("lowo split not run"); rows.append(row); continue
                rl = fl = tl = None
                if model == "retrain":
                    d = DM.run_detection_split(out, rung, gid, "lowo", normalized=norm, n_perm=0, run_null=False, n_jobs=n_jobs, seed_offset=seed_offset,
                                               n_estimators=n_estimators, threshold_source=threshold_source, quarantine=False,
                                               subset_workloads=tuple(f"sandbox_member_{k}" for k in members) + tuple(band), dir_name=f"glm_{m}",
                                               extra_params={"glm_member": m, "glm_band": band})
                    sc = S.read_json(d / "scores.json")
                    if sc.get("status") == "ok":
                        pm = (sc.get("per_member") or {}).get(str(m)) or {}
                        rl = pm.get("recall") if _is_num(pm.get("recall")) else None
                        fl = sc.get("fpr_05_realized") if _is_num(sc.get("fpr_05_realized")) else None
                        fj = S.read_json(d / "folds.json").get("folds") or []
                        ts = [f["threshold_05"] for f in fj if _is_num(f.get("threshold_05"))]
                        tl = float(np.median(ts)) if ts else None
                elif model == "headline_oof":
                    preds = _preds(out, rung, gid, "lowo", norm)
                    bs = [_num(r["score"]) for r in preds if r["workload_key"] in band and _num(r["score"]) is not None]
                    if bs:
                        t = float(np.quantile(bs, 1.0 - DM.FPR_DECLARED, method=DM.THRESHOLD_QUANTILE_METHOD))
                        ms = [_num(r["score"]) for r in preds if r["workload_key"] == wk and str(r.get("in_denominator")).lower() == "true" and _num(r["score"]) is not None]
                        rl = float(np.mean([s > t for s in ms])) if ms else None
                        fl = float(np.mean([s > t for s in bs])); tl = t
                        row["level_label"] = f"{label}; {V.POST_HOC_LABEL}"
                else:
                    raise ValueError(model)
                row.update({"recall_lm": rl, "fpr_lm": fl, "threshold_lm_median": tl})
                row["verdict"] = glm_verdict(ru, fu, rl, fl, n_band_cells, len(band), rule=vanish_rule)
                rows.append(row)
    _replace_rows(p, GLM_COLUMNS, rows, lambda r: (r["rung"], r.get("variant", "norm"), str(r.get("member_index", ""))))
    S.write_params(p, "plan11.detection.glm.v1", _params(out, {"level_quantity": quantity, "level_workload_agg": LEVEL_WORKLOAD_AGG, "band_factor": band_factor, "model": model,
                                                                "vanish_rule": vanish_rule, "min_cells": GOP_MIN_CELLS, "min_workloads": GOP_MIN_WORKLOADS,
                                                                "iteration_quantile": iteration_quantile, "seed_offset": seed_offset},
                                                          [out / "gates" / "selection.json", out / "inputs" / "head_drop.csv", out / "inputs" / "iteration_boundaries.csv"]), CIT_GLM)
    return p


# --------------------------------------------------------------------------- 3.5.7 G-ANCHOR, the order test, the drift regression

ANCHOR_COLUMNS = ("rung", "part", "label_space", "n_cells", "n_labels", "score", "null_p95", "rank", "verdict")
ORDER_COLUMNS = ("rung", "class", "scope", "half_rule", "n_cells", "n_workloads", "order_null_unit", "score", "null_p95", "rank", "n_assignments", "verdict", "consequence")
DRIFT_COLUMNS = ("class", "unit", "n", "n_workloads", "slope", "intercept", "r2", "null_p95_abs_slope", "rank", "verdict", "label")


def _rungs_with_selection(out: Path, rung: str | None = None) -> list[tuple]:
    rungs = [rung] if rung else list(S.RUNGS)
    return [(r, S.selected_grid_id(out, r, None)[0]) for r in rungs]


def anchor(out: Path, *, part: str = "all", n_perm: int = ORDER_NULL_PERM, n_jobs: int = 1, seed_offset: int = 0, rung: str | None = None,
           n_estimators: int = DM.N_ESTIMATORS, n_perm_required: int = ORDER_NULL_PERM) -> Path:
    """ganchor.csv rows. Part (i), 'kernels': per rung, campaign predictability from level-normalized
    features on the benign kernels under LOKO with label = campaign, read from gates/gx.csv (the
    same forest, the same features, 500 campaign-label shuffles): verdict GANCHOR_AUDIBLE when
    leak_verdict == GX_LEAK, GANCHOR_NOT_AUDIBLE when GX_POOLING_STANDS, the string itself when gx
    wrote not applicable; not_run('gates/gx.csv missing (run gates_comparison gx)') when absent.
    Read as a size, never as a switch between two headlines (K3 section 5 point 3). Part (ii),
    'idle_sets': per rung, the idle cells (class idle; harness_idle as a second block when present)
    under leave-one-cell-out with label = campaign (run_detection_split split 'anchor_idle'; score =
    AUC), null = campaign-label shuffles across the idle cells (cell-level); not_applicable('one idle
    campaign (n = <k> cells)') when the idle cells carry one label. 'idle_early_late': per rung, the
    idle cells with label = first half against second half by order_index within the class,
    leave-one-cell-out, AUC, the cell-level shuffle null; not_run('order_index missing for idle') when
    absent. Both are the drift clause of the tripwire (CR3 2.12). Citation: CIT_ANCHOR."""
    out = _out(out)
    p = _det(out) / "ganchor.csv"
    rows = []
    parts = ("kernels", "idle_sets", "idle_early_late") if part == "all" else (part,)
    gx = {r["rung"]: r for r in S.read_csv(out / "gates" / "gx.csv")} if (out / "gates" / "gx.csv").is_file() else None
    join = C.load_join(out)
    has_harness = any(j["class"] == "harness_idle" for j in join)
    for r, gid in _rungs_with_selection(out, rung):
        if "kernels" in parts:
            row = {"rung": r, "part": "kernels", "label_space": "campaign", "n_cells": "", "n_labels": "", "score": "", "null_p95": "", "rank": "", "verdict": ""}
            if gx is None:
                row["verdict"] = V.not_run("gates/gx.csv missing (run gates_comparison gx)")
            elif r not in gx:
                row["verdict"] = V.not_run(f"gates/gx.csv has no row for {r} (run gates_comparison gx --rung {r})")
            else:
                g = gx[r]
                lv = g.get("leak_verdict", "")
                row.update({"n_labels": g.get("n_labels", ""), "score": g.get("score", ""), "null_p95": g.get("null_p95", ""), "rank": g.get("rank", "")})
                row["verdict"] = V.GANCHOR_AUDIBLE if lv == V.GX_LEAK else (V.GANCHOR_NOT_AUDIBLE if lv == V.GX_POOLING_STANDS else lv)
            rows.append(row)
        for prt in ("idle_sets", "idle_early_late"):
            if prt not in parts:
                continue
            for cls in (("idle", "harness_idle") if has_harness else ("idle",)):
                label_space = "campaign" if prt == "idle_sets" else "half (early / late)"
                row = {"rung": r, "part": prt if cls == "idle" else f"{prt} ({cls})", "label_space": label_space, "n_cells": "", "n_labels": "", "score": "", "null_p95": "", "rank": "", "verdict": ""}
                if gid is None:
                    row["verdict"] = _no_selection(r); rows.append(row); continue
                split = "anchor_idle" if prt == "idle_sets" else "idle_early_late"
                d = DM.run_detection_split(out, r, gid, split, normalized=True, n_perm=n_perm, run_null=n_perm > 0, n_jobs=n_jobs, seed_offset=seed_offset,
                                           n_estimators=n_estimators, order_class=cls, dir_name=(split if cls == "idle" else f"{split}_{cls}"), n_perm_required=n_perm_required)
                sc = S.read_json(d / "scores.json")
                n_idle = sum(1 for j in join if j["class"] == cls)
                row["n_cells"] = sc.get("n_positive_scored", 0) + sc.get("n_negative_scored", 0) if sc.get("status") == "ok" else n_idle
                if sc.get("status") != "ok":
                    row["verdict"] = sc["status"]
                else:
                    nb = (sc.get("null") or {}).get("auc") or {}
                    row.update({"n_labels": len(sc.get("params", {}).get("campaign_labels") or []) or 2, "score": sc.get("auc"), "null_p95": nb.get("p95"), "rank": nb.get("rank_text", "")})
                    nv = nb.get("verdict") or (sc.get("null") or {}).get("status") or V.not_run("null not run")
                    if nv == V.PASS:
                        row["verdict"] = V.GANCHOR_AUDIBLE if prt == "idle_sets" else V.ORDER_AUDIBLE
                    elif nv == V.NULL_INSIDE:
                        row["verdict"] = V.GANCHOR_NOT_AUDIBLE if prt == "idle_sets" else V.ORDER_NOT_AUDIBLE
                    else:
                        row["verdict"] = nv
                rows.append(row)
    _replace_rows(p, ANCHOR_COLUMNS, rows, lambda r: (r["rung"], r["part"]))
    S.write_params(p, "plan11.detection.ganchor.v1", _params(out, {"part": part, "n_perm": n_perm, "seed_offset": seed_offset, "n_perm_required": n_perm_required},
                                                             [out / "gates" / "gx.csv", out / "gates" / "selection.json", C.join_path(out)]), CIT_ANCHOR)
    return p


def order_test(out: Path, *, consequence: str = ORDER_TEST_CONSEQUENCE, scope: str = ORDER_TEST_SCOPE, n_perm: int = ORDER_NULL_PERM, n_jobs: int = 1,
               seed_offset: int = 0, rung: str | None = None, n_estimators: int = DM.N_ESTIMATORS, half_rules=ORDER_HALF_RULES,
               n_perm_required: int = ORDER_NULL_PERM) -> Path:
    """order.csv: per rung and per class with order_index on every admissible cell, one row per
    half rule (ORDER_HALF_RULES; al-Kindi review 2.11): 'within_workload' (the default: the first
    floor(k / 2) cells of each workload by realized order against the rest, folds LOWO across
    workloads, null = the half labels permuted within each workload) and 'within_class' (the
    ML-literal row: the first floor(n / 2) cells of the class; null = the workload-level permutation
    of the half labels when every workload lies inside one half, else the per-cell half vector
    within the class, recorded as order_null_unit). For a class with one workload (idle) the folds
    are leave-one-cell-out. score = AUC (positive = second); verdict ORDER_AUDIBLE on strict
    exceedance of the null's p95 else ORDER_NOT_AUDIBLE; consequence 'size': the verdict is a row of
    Table 9; 'void': ORDER_VOID replaces ORDER_AUDIBLE (K3 F5). scope 'within_class' (al-Farabi M12;
    the stage-1 reading) or 'campaign' (one extra row per rung over every classed cell, class = all).
    Stage-1 note in params: the sandbox cells ran in order by member, so member identity and
    position coincide by construction (P3 0a) and the row is a size. Citation: CIT_ORDER."""
    out = _out(out)
    p = _det(out) / "order.csv"
    join = C.load_join(out); adm = _adm_set(out)
    rows = []
    classes = sorted({j["class"] for j in join if j["cell_id"] in adm and j["class"] != C.UNASSIGNED})
    targets = [(c, "within_class") for c in classes] + ([("all", "campaign")] if scope == "campaign" else [])
    for r, gid in _rungs_with_selection(out, rung):
        for cls, sc_ in targets:
            cells = [j for j in join if j["cell_id"] in adm and (cls == "all" or j["class"] == cls)]
            n_wk = len({j["workload_key"] for j in cells})
            for hr in half_rules:
                row = {"rung": r, "class": cls, "scope": sc_, "half_rule": hr, "n_cells": len(cells), "n_workloads": n_wk, "order_null_unit": "", "score": "", "null_p95": "",
                       "rank": "", "n_assignments": "", "verdict": "", "consequence": consequence}
                if gid is None:
                    row["verdict"] = _no_selection(r); rows.append(row); continue
                if any(int(j["order_index"]) < 1 for j in cells):
                    row["verdict"] = V.not_run(f"order_index missing for {cls}"); rows.append(row); continue
                if len(cells) < 2:
                    row["verdict"] = V.not_applicable(f"{cls}: fewer than two cells"); rows.append(row); continue
                d = DM.run_detection_split(out, r, gid, "order", normalized=True, n_perm=n_perm, run_null=n_perm > 0, n_jobs=n_jobs, seed_offset=seed_offset,
                                           n_estimators=n_estimators, order_class=cls, half_rule=hr, dir_name=f"order_{cls}_{hr}", n_perm_required=n_perm_required)
                sc = S.read_json(d / "scores.json")
                if sc.get("status") != "ok":
                    row["verdict"] = sc["status"]; rows.append(row); continue
                nb = (sc.get("null") or {}).get("auc") or {}
                nn = sc.get("null") or {}
                row.update({"order_null_unit": nn.get("null_unit", ""), "score": sc.get("auc"), "null_p95": nb.get("p95"), "rank": nb.get("rank_text", ""), "n_assignments": nn.get("n_assignments", "")})
                nv = nb.get("verdict") or nn.get("status") or V.not_run("null not run")
                if nv == V.PASS:
                    row["verdict"] = V.ORDER_VOID if consequence == "void" else V.ORDER_AUDIBLE
                elif nv == V.NULL_INSIDE:
                    row["verdict"] = V.ORDER_NOT_AUDIBLE
                else:
                    row["verdict"] = nv
                rows.append(row)
    _replace_rows(p, ORDER_COLUMNS, rows, lambda r: (r["rung"], r["class"], r["scope"], r["half_rule"]))
    S.write_params(p, "plan11.detection.order.v1", _params(out, {"consequence": consequence, "scope": scope, "half_rules": list(half_rules), "n_perm": n_perm, "seed_offset": seed_offset,
                                                                  "n_perm_required": n_perm_required,
                                                                  "stage1_note": "the sandbox cells ran in order by member, so member identity and position coincide by construction (P3 0a); the row is a size",
                                                                  "campaign_scope_note": "the campaign-wide reading is void by construction on stage 1 (the halves coincide with the classes)"},
                                                            [out / "gates" / "selection.json", C.join_path(out)]), CIT_ORDER)
    return p


def _ols(x: np.ndarray, y: np.ndarray) -> tuple:
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    if len(x) < 3 or np.ptp(x) == 0:
        return float("nan"), float("nan"), float("nan")
    A = np.stack([x, np.ones_like(x)], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((y - pred) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(coef[0]), float(coef[1]), float(r2)


def drift_regression(out: Path, *, n_perm: int = ORDER_NULL_PERM, seed_offset: int = 0, null_percentile: float = DRIFT_NULL_PERCENTILE) -> Path:
    """drift.csv: per class with order_index, the per-cell floor statistic (K_med from gk0_cells.csv)
    regressed on order_index by ordinary least squares, two rows per class (al-Kindi review 2.10):
    unit 'within_workload' (K_med residual after the workload mean, on order_index; null = n_perm
    shuffles of order_index within each workload; the disclosure row) and unit 'plain' (the plain
    regression, labelled 'confounded with workload order (blocked)'; null = shuffles of order_index
    within the class). Verdict DRIFT_SLOPE when |slope| strictly exceeds the null_percentile-th
    percentile of the permuted |slope|, else DRIFT_NONE. Citation: CIT_DRIFT."""
    out = _out(out)
    p = _det(out) / "drift.csv"
    join = C.load_join(out); adm = _adm_set(out); fb_p = _det(out) / "gk0_cells.csv"
    kmed = {r["cell_id"]: _num(r["K_med"]) for r in S.read_csv(fb_p)} if fb_p.is_file() else {}
    rng = np.random.default_rng(SEED_LABEL_NULL + int(seed_offset))
    rows = []
    for cls in sorted({j["class"] for j in join if j["cell_id"] in adm and j["class"] != C.UNASSIGNED}):
        cells = [j for j in join if j["cell_id"] in adm and j["class"] == cls]
        for unit in ("within_workload", "plain"):
            label = "confounded with workload order (blocked)" if unit == "plain" else "disclosure"
            row = {"class": cls, "unit": unit, "n": len(cells), "n_workloads": len({j["workload_key"] for j in cells}), "slope": "", "intercept": "", "r2": "",
                   "null_p95_abs_slope": "", "rank": "", "verdict": "", "label": label}
            if not kmed:
                row["verdict"] = V.not_run("gates/detection/gk0_cells.csv missing (run gk0-cells)"); rows.append(row); continue
            if any(int(j["order_index"]) < 1 for j in cells):
                row["verdict"] = V.not_run(f"order_index missing for {cls}"); rows.append(row); continue
            x = np.array([float(j["order_index"]) for j in cells]); y = np.array([kmed.get(j["cell_id"], np.nan) for j in cells], dtype=np.float64)
            wk = np.array([j["workload_key"] for j in cells])
            ok = np.isfinite(y)
            x, y, wk = x[ok], y[ok], wk[ok]
            if len(x) < 3:
                row["verdict"] = V.not_applicable("fewer than three cells"); rows.append(row); continue
            if unit == "within_workload":
                yr = y.copy(); xr = x.copy()
                for w in set(wk.tolist()):
                    m = wk == w
                    yr[m] = y[m] - y[m].mean(); xr[m] = x[m] - x[m].mean()
                s, b, r2 = _ols(xr, yr)
            else:
                s, b, r2 = _ols(x, y)
            if not np.isfinite(s):
                row["verdict"] = V.not_applicable("slope undefined (no spread in order_index)"); rows.append(row); continue
            null = []
            for _ in range(int(n_perm)):
                xp = x.copy()
                if unit == "within_workload":
                    for w in set(wk.tolist()):
                        m = np.flatnonzero(wk == w)
                        xp[m] = x[m][rng.permutation(len(m))]
                    yr = y.copy(); xr = xp.copy()
                    for w in set(wk.tolist()):
                        m = wk == w
                        yr[m] = y[m] - y[m].mean(); xr[m] = xp[m] - xp[m].mean()
                    sp, _, _ = _ols(xr, yr)
                else:
                    xp = x[rng.permutation(len(x))]
                    sp, _, _ = _ols(xp, y)
                null.append(abs(sp))
            null = np.asarray(null, dtype=np.float64); null = null[np.isfinite(null)]
            p95 = float(np.percentile(null, null_percentile)) if len(null) else None
            row.update({"slope": s, "intercept": b, "r2": r2, "null_p95_abs_slope": p95, "rank": int(np.sum(null < abs(s))) if len(null) else ""})
            row["verdict"] = (V.DRIFT_SLOPE if (p95 is not None and abs(s) > p95) else V.DRIFT_NONE) if len(null) else V.not_run("no permutation")
            rows.append(row)
    S.write_csv(p, DRIFT_COLUMNS, rows)
    S.write_params(p, "plan11.detection.drift.v1", _params(out, {"n_perm": n_perm, "seed_offset": seed_offset, "null_percentile": null_percentile, "statistic": "K_med", "unit_default": "within_workload"},
                                                           [fb_p, C.join_path(out)]), CIT_DRIFT)
    return p


# --------------------------------------------------------------------------- 3.5.8 G-SIG

GSIG_COLUMNS = ("rung", "grid_id", "tpr05_loco", "tpr05_lowo", "gap_tpr05", "auc_loco", "auc_lowo", "gap_auc", "loco_null_verdict", "lowo_null_verdict", "verdict")


def gsig(out: Path, rung: str, grid_id: str | None = None) -> Path:
    """gsig.csv row per rung: tpr05_loco, tpr05_lowo, gap_tpr05 (= loco - lowo), auc_loco, auc_lowo,
    gap_auc, loco_null_verdict, lowo_null_verdict, verdict: GSIG_IDENTITY when the LOCO tpr05 null
    verdict is PASS and the LOWO one is NULL_INSIDE (the rung cannot be cited for detection, only
    for signature matching with that sentence); GSIG_REPORTED otherwise; not_run('<split> null not
    run') when either null is absent (the gap number still written). Citation: CIT_GSIG."""
    out = _out(out)
    gid, _ = _grid(out, rung, grid_id)
    p = _det(out) / "gsig.csv"
    row = {"rung": rung, "grid_id": gid or ""}
    if gid is None:
        row["verdict"] = _no_selection(rung)
    else:
        lo, lw = _scores(out, rung, gid, "loco", True), _scores(out, rung, gid, "lowo", True)
        def stat(sc, k):
            return sc.get(k) if (sc and sc.get("status") == "ok" and _is_num(sc.get(k))) else None
        def nv(sc):
            if not sc or sc.get("status") != "ok":
                return None
            nb = (sc.get("null") or {}).get(DM.NULL_VERDICT_STATISTIC) or {}
            return nb.get("verdict") or ((sc.get("null") or {}).get("status"))
        row.update({"tpr05_loco": stat(lo, "tpr_05"), "tpr05_lowo": stat(lw, "tpr_05"), "auc_loco": stat(lo, "auc"), "auc_lowo": stat(lw, "auc")})
        row["gap_tpr05"] = (row["tpr05_loco"] - row["tpr05_lowo"]) if (row["tpr05_loco"] is not None and row["tpr05_lowo"] is not None) else None
        row["gap_auc"] = (row["auc_loco"] - row["auc_lowo"]) if (row["auc_loco"] is not None and row["auc_lowo"] is not None) else None
        vlo, vlw = nv(lo), nv(lw)
        row["loco_null_verdict"] = vlo or V.not_run("loco split not run"); row["lowo_null_verdict"] = vlw or V.not_run("lowo split not run")
        if vlo is None or not (vlo in (V.PASS, V.NULL_INSIDE)):
            row["verdict"] = V.not_run("loco null not run")
        elif vlw is None or not (vlw in (V.PASS, V.NULL_INSIDE)):
            row["verdict"] = V.not_run("lowo null not run")
        else:
            row["verdict"] = V.GSIG_IDENTITY if (vlo == V.PASS and vlw == V.NULL_INSIDE) else V.GSIG_REPORTED
    _replace_rows(p, GSIG_COLUMNS, [row], lambda r: r["rung"])
    S.write_params(p, "plan11.detection.gsig.v1", _params(out, {"null_verdict_statistic": DM.NULL_VERDICT_STATISTIC}, [out / "gates" / "selection.json"]), CIT_GSIG)
    return p


# --------------------------------------------------------------------------- 3.5.9 G-FP

GFP_COLUMNS = ("rung", "grid_id", "family", "n_cells", "n_flagged_05", "fraction", "verdict", "tpr05_with", "fpr05_with", "tpr05_without", "fpr05_without")


def gfp(out: Path, rung: str, grid_id: str | None = None, *, flag_fraction: float = GFP_FLAG_FRACTION, n_jobs: int = 1, seed_offset: int = 0,
        n_estimators: int = DM.N_ESTIMATORS, threshold_source: str = DM.THRESHOLD_SOURCE, predictions: list[dict] | None = None) -> Path:
    """gfp.csv row per (rung, benign family) from the LOWO norm predictions: n_cells, n_flagged_05,
    fraction, verdict = GFP_INSEPARABLE when fraction >= flag_fraction else GFP_ATTRIBUTED; for an
    inseparable family the operating point recomputed without it (run_detection_split with
    exclude_families=(family,), dir_name 'lowo__without_<family>'): tpr05_without, fpr05_without, and
    the with-family numbers beside. Refuses nothing; it prevents a family from being averaged away.
    Citation: CIT_GFP."""
    out = _out(out)
    gid, _ = _grid(out, rung, grid_id)
    p = _det(out) / "gfp.csv"
    rows = []
    if gid is None:
        rows.append({"rung": rung, "grid_id": "", "family": "", "verdict": _no_selection(rung)})
    else:
        preds = predictions if predictions is not None else _preds(out, rung, gid, "lowo", True)
        sc = _scores(out, rung, gid, "lowo", True)
        if not preds:
            rows.append({"rung": rung, "grid_id": gid, "family": "", "verdict": V.not_run("lowo norm split not run")})
        else:
            fams: dict = {}
            for r in preds:
                if r.get("y") == "benign" and r.get("score") not in ("", None):
                    d = fams.setdefault(r["family"], {"n": 0, "f": 0})
                    d["n"] += 1; d["f"] += int(str(r.get("flag_05")).lower() == "true")
            for fam in sorted(fams):
                n, f = fams[fam]["n"], fams[fam]["f"]
                frac = f / n if n else None
                row = {"rung": rung, "grid_id": gid, "family": fam, "n_cells": n, "n_flagged_05": f, "fraction": frac,
                       "verdict": V.GFP_INSEPARABLE if (frac is not None and frac >= flag_fraction) else V.GFP_ATTRIBUTED,
                       "tpr05_with": sc.get("tpr_05") if sc and sc.get("status") == "ok" else "", "fpr05_with": sc.get("fpr_05_realized") if sc and sc.get("status") == "ok" else "",
                       "tpr05_without": "", "fpr05_without": ""}
                if row["verdict"] == V.GFP_INSEPARABLE:
                    d = DM.run_detection_split(out, rung, gid, "lowo", normalized=True, n_perm=0, run_null=False, n_jobs=n_jobs, seed_offset=seed_offset, n_estimators=n_estimators,
                                               threshold_source=threshold_source, quarantine=False, exclude_families=(fam,), dir_name=f"lowo__without_{fam}")
                    sw = S.read_json(d / "scores.json")
                    if sw.get("status") == "ok":
                        row["tpr05_without"], row["fpr05_without"] = sw.get("tpr_05"), sw.get("fpr_05_realized")
                    else:
                        row["tpr05_without"] = sw.get("status")
                rows.append(row)
    _replace_rows(p, GFP_COLUMNS, rows, lambda r: (r["rung"], r["family"]))
    S.write_params(p, "plan11.detection.gfp.v1", _params(out, {"flag_fraction": flag_fraction, "seed_offset": seed_offset}, [out / "gates" / "selection.json"]), CIT_GFP)
    return p


# --------------------------------------------------------------------------- 3.5.10 G-1C

G1C_COLUMNS = ("rung", "grid_id", "directory", "model", "primary", "tpr05", "fpr05_realized", "tpr01", "auc", "threshold_source", "null_verdict", "label", "verdict")


def g1c(out: Path, rung: str, grid_id: str | None = None) -> Path:
    """g1c.csv row per (rung, one-class model directory): model, primary, tpr05, fpr05_realized, tpr01,
    auc, threshold_source, null_verdict, label = G1C_PRIMARY for the declared primary, G1C_SECONDARY
    for every other; verdict G1C_SEARCH on the rung when more than one directory exists and none is
    primary (cannot arise through the CLI, which requires --secondary for a non-primary; written if
    the author copies directories by hand). Citation: CIT_G1C."""
    out = _out(out)
    gid, _ = _grid(out, rung, grid_id)
    p = _det(out) / "g1c.csv"
    rows = []
    if gid is None:
        rows.append({"rung": rung, "grid_id": "", "directory": "", "verdict": _no_selection(rung)})
    else:
        base = _det(out) / "splits" / rung / gid
        dirs = sorted(d for d in base.iterdir() if d.is_dir() and d.name.startswith("one_class")) if base.is_dir() else []
        if not dirs:
            rows.append({"rung": rung, "grid_id": gid, "directory": "", "verdict": V.not_run("one-class split not run")})
        else:
            recs = []
            for d in dirs:
                sp = d / "scores.json"
                sc = S.read_json(sp) if sp.is_file() else {"status": V.not_run("scores.json missing")}
                prim = bool(sc.get("primary")) if sc.get("status") == "ok" else False
                nb = (sc.get("null") or {}).get(DM.NULL_VERDICT_STATISTIC) or {}
                recs.append({"rung": rung, "grid_id": gid, "directory": d.name, "model": sc.get("model", ""), "primary": prim,
                             "tpr05": sc.get("tpr_05") if sc.get("status") == "ok" else sc.get("status"), "fpr05_realized": sc.get("fpr_05_realized", ""),
                             "tpr01": sc.get("tpr_01", ""), "auc": sc.get("auc", ""), "threshold_source": sc.get("threshold_source", ""),
                             "null_verdict": nb.get("verdict") or (sc.get("null") or {}).get("status", ""),
                             "label": (V.G1C_PRIMARY if prim else V.G1C_SECONDARY) if sc.get("status") == "ok" else sc.get("status"), "verdict": ""})
            search = len(recs) > 1 and not any(r["primary"] for r in recs)
            for r in recs:
                r["verdict"] = V.G1C_SEARCH if search else (V.PASS if r["primary"] else r["label"])
            rows += recs
    _replace_rows(p, G1C_COLUMNS, rows, lambda r: (r["rung"], r["directory"]))
    S.write_params(p, "plan11.detection.g1c.v1", _params(out, {"one_class_model_declared": DM.ONE_CLASS_MODEL}, [out / "gates" / "selection.json"]), CIT_G1C)
    return p


# --------------------------------------------------------------------------- 3.5.11 the harness clause

HARNESS_COLUMNS = ("rung", "grid_id", "block", "feature", "margin_sb", "margin_hi", "margin_rp", "relaunched_flagged_fraction", "median_member_recall", "verdict")


def _margin(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> float | None:
    """2 |AUC - 0.5| of a feature between two groups (row masks a and b)."""
    xa, xb = x[a], x[b]
    xa, xb = xa[np.isfinite(xa)], xb[np.isfinite(xb)]
    if len(xa) == 0 or len(xb) == 0:
        return None
    y = np.array([1] * len(xa) + [0] * len(xb)); s = np.concatenate([xa, xb])
    if np.ptp(s) == 0:
        return 0.0
    return float(2.0 * abs(roc_auc_score(y, s) - 0.5))


def harness(out: Path, rung: str, grid_id: str | None = None, *, comparable_tol: float = HARNESS_COMPARABLE_TOL, relaunch_rule: str = HARNESS_RELAUNCH_RULE) -> Path:
    """When the classes benign_relaunched and harness_idle are both absent from cell_classes.csv,
    harness.csv holds one row per rung with verdict HARNESS_STAGE2_ABSENT and empty numbers.
    Otherwise, per (rung, feature) on the norm features: the per-cell feature (the cell row) and
    three margins, each 2 * |AUC - 0.5| of the feature between two groups: margin_sb (sandbox
    against every benign cell), margin_hi (harness_idle against idle), margin_rp (benign_relaunched
    against its parent kernel's cells, pooled over the re-launched pairs); label HARNESS_COMPARABLE
    when |margin_sb - margin_hi| <= tol and |margin_sb - margin_rp| <= tol (the feature is the
    harness's and leaves the evidence), else HARNESS_CLASS_EXCEEDS. A second block, one row per
    rung: relaunched_flagged_fraction at the LOWO operating point beside median_member_recall;
    HARNESS_RELAUNCH_NOT_CLASS when the control's flagged fraction >= the median member recall.
    A margin whose group is absent is None and that comparison is skipped. Citation: CIT_HARNESS."""
    out = _out(out)
    gid, _ = _grid(out, rung, grid_id)
    p = _det(out) / "harness.csv"
    join = C.load_join(out)
    has_r = any(j["class"] == "benign_relaunched" for j in join); has_h = any(j["class"] == "harness_idle" for j in join)
    rows = []
    if not has_r and not has_h:
        rows.append({"rung": rung, "grid_id": gid or "", "block": "features", "feature": "", "verdict": V.HARNESS_STAGE2_ABSENT})
    elif gid is None:
        rows.append({"rung": rung, "grid_id": "", "block": "features", "feature": "", "verdict": _no_selection(rung)})
    else:
        try:
            data = DM.load_detection_data(out, rung, gid, normalized=True)
        except FileNotFoundError as e:
            data = None; rows.append({"rung": rung, "grid_id": gid, "block": "features", "feature": "", "verdict": V.not_run(f"input missing: {Path(str(e)).name}")})
        if data is not None:
            lab, X, names = data["lab"], data["X"], data["names"]
            cls = lab["cls"]; y = lab["y"]; wk = lab["workload_key"]
            sb, ben = y == "sandbox", y == "benign"
            hi, idle = cls == "harness_idle", cls == "idle"
            rel = cls == "benign_relaunched"
            parents = set(wk[rel].tolist())
            par = (cls == "benign_kernel") & np.isin(wk, list(parents))
            for j, name in enumerate(names):
                x = X[:, j]
                m_sb = _margin(x, sb, ben); m_hi = _margin(x, hi, idle) if has_h else None; m_rp = _margin(x, rel, par) if has_r else None
                comp = m_sb is not None and all((m is None) or (abs(m_sb - m) <= comparable_tol) for m in (m_hi, m_rp)) and any(m is not None for m in (m_hi, m_rp))
                rows.append({"rung": rung, "grid_id": gid, "block": "features", "feature": name, "margin_sb": m_sb, "margin_hi": m_hi, "margin_rp": m_rp,
                             "relaunched_flagged_fraction": "", "median_member_recall": "", "verdict": V.HARNESS_COMPARABLE if comp else V.HARNESS_CLASS_EXCEEDS})
            row = {"rung": rung, "grid_id": gid, "block": "relaunch", "feature": "", "margin_sb": "", "margin_hi": "", "margin_rp": "", "relaunched_flagged_fraction": "", "median_member_recall": "", "verdict": ""}
            sc = _scores(out, rung, gid, "lowo", True); preds = _preds(out, rung, gid, "lowo", True)
            if not has_r:
                row["verdict"] = V.not_applicable("no benign_relaunched cells")
            elif not sc or sc.get("status") != "ok":
                row["verdict"] = V.not_run("lowo norm split not run")
            else:
                rr = [r for r in preds if r.get("class") == "benign_relaunched" and r.get("score") not in ("", None)]
                frac = float(np.mean([str(r.get("flag_05")).lower() == "true" for r in rr])) if rr else None
                recs = [v.get("recall") for v in (sc.get("per_member") or {}).values() if _is_num(v.get("recall"))]
                med = float(np.median(recs)) if recs else None
                row.update({"relaunched_flagged_fraction": frac, "median_member_recall": med})
                if frac is None or med is None:
                    row["verdict"] = V.not_applicable("no scored re-launched cell or no member recall")
                else:
                    row["verdict"] = V.HARNESS_RELAUNCH_NOT_CLASS if frac >= med else V.PASS
            rows.append(row)
    _replace_rows(p, HARNESS_COLUMNS, rows, lambda r: (r["rung"], r["block"], r["feature"]))
    S.write_params(p, "plan11.detection.harness.v1", _params(out, {"comparable_tol": comparable_tol, "relaunch_rule": relaunch_rule, "margin": "2 * |AUC - 0.5|"},
                                                             [out / "gates" / "selection.json", C.join_path(out)]), CIT_HARNESS)
    return p


# --------------------------------------------------------------------------- 3.5.12 G-CAL

GCAL_COLUMNS = ("rung", "grid_id", "per_fold_tpr05", "pooled_tpr_at_fpr05", "difference", "null_spread", "verdict")


def gcal(out: Path, rung: str, grid_id: str | None = None) -> Path:
    """gcal.csv: the per-rung G-CAL values copied from the LOWO norm scores.json (computed inside the
    split, SPEC_DETECTION 3.3.8): per_fold_tpr05, pooled_tpr_at_fpr05, difference, null_spread
    (GCAL_SPREAD_RULE), verdict GCAL_AGREE | GCAL_PERFOLD | not run. Citation: CIT_GCAL."""
    out = _out(out)
    gid, _ = _grid(out, rung, grid_id)
    p = _det(out) / "gcal.csv"
    row = {"rung": rung, "grid_id": gid or ""}
    sc = _scores(out, rung, gid, "lowo", True) if gid else None
    if gid is None:
        row["verdict"] = _no_selection(rung)
    elif not sc or sc.get("status") != "ok":
        row["verdict"] = sc["status"] if sc else V.not_run("lowo norm split not run")
    else:
        g = sc.get("gcal") or {}
        row.update({"per_fold_tpr05": g.get("per_fold_tpr05"), "pooled_tpr_at_fpr05": g.get("pooled_tpr_at_fpr05"), "difference": g.get("difference"),
                    "null_spread": g.get("null_spread"), "verdict": g.get("verdict") or V.not_run("gcal missing from scores.json")})
    _replace_rows(p, GCAL_COLUMNS, [row], lambda r: r["rung"])
    S.write_params(p, "plan11.detection.gcal.v1", _params(out, {"spread_rule": GCAL_SPREAD_RULE}, [out / "gates" / "selection.json"]), CIT_GCAL)
    return p


# --------------------------------------------------------------------------- 3.5.13 G-M and G-DIM

GM_COLUMNS = ("split", "row_a", "row_b", "tpr05_a", "tpr05_b", "diff", "spread", "improving", "worsening", "ties", "p_exact", "verdict")
GDIM_COLUMNS = ("rung", "row", "grid_id", "d", "d_used", "d_matched", "method", "status")


def exact_sign_test(improving: int, worsening: int) -> float:
    """P(Bin(improving + worsening, 1/2) <= worsening), the exact one-sided binomial over the
    non-tied units (ML 2.5: five up and none down is 0.031; seven up and one down of eight is 0.035)."""
    n = int(improving) + int(worsening); k = int(worsening)
    if n == 0:
        return 1.0
    return float(sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n))


def _gm_rows_available(out: Path) -> list[tuple]:
    rows = []
    for label, rung, norm, name in GM_ROWS:
        gid = S.selected_grid_id(out, rung, None)[0]
        if gid is None:
            continue
        sc = _scores(out, rung, gid, name, norm)
        if sc and sc.get("status") == "ok":
            rows.append((label, rung, gid, name, norm, sc))
    comp = _det(out) / "splits" / "comparator"
    if comp.is_dir():
        for d in sorted(comp.rglob("scores.json")):
            sc = S.read_json(d)
            if sc.get("status") == "ok" and "per_workload_outcome" in sc:
                rows.append(("comparator", "comparator", d.parent.parent.name, d.parent.name, True, sc)); break
    return rows


def gm(out: Path, *, n_seeds: int = GM_N_SEEDS, alpha: float = GM_ALPHA, split: str = "lowo", n_jobs: int = 1, seed_offset: int = 0,
       n_estimators: int = DM.N_ESTIMATORS, threshold_source: str = DM.THRESHOLD_SOURCE) -> Path:
    """gm.csv: every ordered pair of rows among (apf raw, apf, wapf, persist, content, combined,
    combined (matched), and 'comparator' if splits/comparator/ exists in this layer's schema):
    per_workload_outcome of A minus of B per workload (recall for a sandbox workload, one minus its
    flagged fraction for a benign workload; a tie is neither); improving, worsening, ties, p_exact;
    spread = max - min of the apf LOWO tpr05 over n_seeds forest seeds (SEED_FOREST + i, written
    under splits/apf/<grid>/lowo_seed<i>__norm/ through dir_name); verdict GM_BEATS when tpr05_A -
    tpr05_B > spread and p_exact <= alpha, else GM_DIFFERENCE. Citation: CIT_GM."""
    out = _out(out)
    p = _det(out) / "gm.csv"
    avail = _gm_rows_available(out)
    gid_apf = S.selected_grid_id(out, "apf", None)[0]
    spread = None; seeds_tpr = []
    if gid_apf is not None and (S.features_path(out, "apf", gid_apf, True)).is_file():
        for i in range(int(n_seeds)):
            d = DM.run_detection_split(out, "apf", gid_apf, split, normalized=True, n_perm=0, run_null=False, n_jobs=n_jobs, seed=SEED_FOREST + i, seed_offset=seed_offset,
                                       n_estimators=n_estimators, threshold_source=threshold_source, quarantine=False, dir_name=f"lowo_seed{i}")
            sc = S.read_json(d / "scores.json")
            if sc.get("status") == "ok" and _is_num(sc.get("tpr_05")):
                seeds_tpr.append(float(sc["tpr_05"]))
        spread = (max(seeds_tpr) - min(seeds_tpr)) if seeds_tpr else None
    rows = []
    for a in avail:
        for b in avail:
            if a[0] == b[0]:
                continue
            oa, ob = a[5].get("per_workload_outcome") or {}, b[5].get("per_workload_outcome") or {}
            imp = wor = ties = 0
            for w in sorted(set(oa) & set(ob)):
                va, vb = oa[w], ob[w]
                if not (_is_num(va) and _is_num(vb)):
                    continue
                if va > vb: imp += 1
                elif va < vb: wor += 1
                else: ties += 1
            pe = exact_sign_test(imp, wor)
            ta, tb = a[5].get("tpr_05"), b[5].get("tpr_05")
            diff = (ta - tb) if (_is_num(ta) and _is_num(tb)) else None
            if spread is None or diff is None:
                verdict = V.not_run("seed spread unavailable" if spread is None else "tpr05 undefined")
            else:
                verdict = V.GM_BEATS if (diff > spread and pe <= alpha) else V.GM_DIFFERENCE
            rows.append({"split": split, "row_a": a[0], "row_b": b[0], "tpr05_a": ta, "tpr05_b": tb, "diff": diff, "spread": spread, "improving": imp, "worsening": wor,
                         "ties": ties, "p_exact": pe, "verdict": verdict})
    if not rows:
        rows.append({"split": split, "row_a": "", "row_b": "", "verdict": V.not_run("fewer than two rows with LOWO scores")})
    S.write_csv(p, GM_COLUMNS, rows)
    S.write_params(p, "plan11.detection.gm.v1", _params(out, {"n_seeds": n_seeds, "alpha": alpha, "split": split, "seed_spread_tpr05": seeds_tpr, "spread": spread,
                                                               "rows": [r[0] for r in avail], "comparator": V.COMPARATOR_ELSEWHERE if not any(r[0] == "comparator" for r in avail) else "present"},
                                                         [out / "gates" / "selection.json"]), CIT_GM)
    return p


def gdim(out: Path) -> Path:
    """gdim.csv: rung, row, grid_id, d, d_used, d_matched (combined (matched) only), method, status
    (GDIM_FULL | GDIM_REDUCED) from every LOWO scores.json. Citation: CIT_GDIM."""
    out = _out(out)
    p = _det(out) / "gdim.csv"
    rows = []
    for label, rung, norm, name in GM_ROWS:
        gid = S.selected_grid_id(out, rung, None)[0]
        if gid is None:
            rows.append({"rung": rung, "row": label, "grid_id": "", "status": _no_selection(rung)}); continue
        sc = _scores(out, rung, gid, name, norm)
        if not sc:
            rows.append({"rung": rung, "row": label, "grid_id": gid, "status": V.not_run(f"{name} split not run")}); continue
        if sc.get("status") != "ok":
            rows.append({"rung": rung, "row": label, "grid_id": gid, "status": sc["status"]}); continue
        rows.append({"rung": rung, "row": label, "grid_id": gid, "d": sc.get("feature_count"), "d_used": sc.get("feature_count_used"),
                     "d_matched": sc.get("params", {}).get("reduce_to") if name == "lowo_matched" else "", "method": sc.get("params", {}).get("reduce_method", ""),
                     "status": sc.get("dim_status")})
    S.write_csv(p, GDIM_COLUMNS, rows)
    S.write_params(p, "plan11.detection.gdim.v1", _params(out, {}, [out / "gates" / "selection.json"]), CIT_GDIM)
    return p


# --------------------------------------------------------------------------- 3.5.14 the alias falsifier and the leak probes

ALIAS_COLUMNS = ("rung", "grid_id", "feature", "regressor", "unit", "slope", "intercept", "r2", "n", "dt_spread_s", "verdict")
LEAK_COLUMNS = ("rung", "quantity", "n_cells", "n_workloads", "auc", "tpr05", "null_p95", "rank", "n_assignments", "verdict", "note")


def _leak_lab(out: Path) -> tuple:
    """A label dict over the admissible classed cells without a feature file (one row per cell)."""
    adm = _adm_set(out); fb = _floor(out)
    join = [j for j in C.load_join(out) if j["cell_id"] in adm]
    feat = {"X": np.zeros((len(join), 1)), "cell_id": np.array([j["cell_id"] for j in join]), "win_start": np.zeros(len(join), dtype=int)}
    lab = DS.make_detection_labels(feat, join, adm, row_unit="cell", floor_by_cell=fb)
    return lab, fb


def leak_probe_quantities(out: Path, cell_ids: list[str]) -> dict:
    """{quantity: {cell_id: value}} for n_pairs and dt_est_s (the sidecars) and frac_above_band and
    K_med (gk0_cells.csv); missing values are NaN."""
    q = {k: {} for k in LEAK_QUANTITIES}
    gk = {r["cell_id"]: r for r in S.read_csv(_det(out) / "gk0_cells.csv")} if (_det(out) / "gk0_cells.csv").is_file() else {}
    for c in cell_ids:
        sp = S.sidecar_path(out, c)
        sc = S.read_json(sp) if sp.is_file() else {}
        q["n_pairs"][c] = float(sc.get("n_pairs") or np.nan); q["dt_est_s"][c] = float(sc.get("dt_est_s") or np.nan)
        g = gk.get(c, {})
        q["frac_above_band"][c] = _num(g.get("frac_above_band")) if g else np.nan
        q["K_med"][c] = _num(g.get("K_med")) if g else np.nan
        for k in ("frac_above_band", "K_med"):
            if q[k][c] is None:
                q[k][c] = np.nan
    return q


def one_quantity_probe(lab: dict, fb: dict, values: np.ndarray, *, n_perm: int, seed_null: int, n_perm_required: int = ORDER_NULL_PERM) -> dict:
    """The one-feature model on one quantity under LOWO with the workload-level null (ML 3.4, 3.5;
    ML review 2.6): the observed AUC and tpr05 (detection_metrics.l1_feature_run pooled by the
    denominator rule) and the null of the AUC over workload_label_permutations; verdict PASS on
    strict exceedance, NULL_INSIDE otherwise, the null's own refusals copied."""
    X = np.asarray(values, dtype=np.float64)[:, None]
    folds = DS.fold_lowo(lab)
    obs = DM._pool_l1(DM.l1_feature_run(X, lab, folds, j=0)["records"], fb, DM.POSITIVE)
    S_n, B_n, pool = DM._s_b(lab)
    rng = np.random.default_rng(int(seed_null))
    subsets, exhaustive = DM.workload_label_permutations(pool, S_n, int(n_perm), rng)
    n_assign = DM.count_assignments(S_n, B_n)
    null = []
    for sub in subsets:
        y = DM._y_from_subset(lab, sub)
        r = DM._pool_l1(DM.l1_feature_run(X, lab, folds, j=0, y_override=y)["records"], fb, DM.POSITIVE)
        null.append(r["auc"] if r["auc"] is not None else np.nan)
    verdict, summ = DM.null_verdict(obs["auc"], np.asarray(null, dtype=np.float64), n_assign, exhaustive=exhaustive, n_perm_required=n_perm_required)
    return {"auc": obs["auc"], "tpr05": obs["tpr_05"], "null_p95": summ.get("p95"), "rank": summ.get("rank_text", ""), "n_assignments": n_assign, "verdict": verdict,
            "n_cells": int(lab["n"]), "n_workloads": S_n + B_n}


def leak_probe(out: Path, *, n_perm: int = ORDER_NULL_PERM, seed_offset: int = 0, n_perm_required: int = ORDER_NULL_PERM) -> Path:
    """leak_probe.csv (ML review 2.6; ML 3.4, 3.5): for each of n_pairs, dt_est_s (the sidecars),
    frac_above_band and K_med (gk0_cells.csv), the one-feature model under LOWO with the
    workload-level null; rows rung-free: quantity, n_cells, n_workloads, auc, tpr05, null_p95, rank,
    n_assignments, verdict LEAK_AUDIBLE / LEAK_NOT_AUDIBLE, read as a size in Table 9 (rows cadence
    and active fraction); K_med audible is not a leak but is printed beside G-L (note). The
    per-class median of dt_est_s and n_pairs and the between-class difference are recorded in
    params. Both quantities remain logs, never features. Citation: CIT_LEAK."""
    out = _out(out)
    p = _det(out) / "leak_probe.csv"
    lab, fb = _leak_lab(out)
    rows = []
    stats = {}
    if lab["n"] == 0:
        rows.append({"rung": "rung-free", "quantity": "", "verdict": V.not_run("no admissible classed cell")})
    else:
        q = leak_probe_quantities(out, lab["cell_id"].tolist())
        for k in LEAK_QUANTITIES:
            vals = np.array([q[k].get(c, np.nan) for c in lab["cell_id"]], dtype=np.float64)
            note = "level, not a leak (printed beside G-L)" if k == "K_med" else ""
            row = {"rung": "rung-free", "quantity": k, "note": note}
            if not np.isfinite(vals).any():
                row["verdict"] = V.not_run(f"{k} unavailable (sidecars or gk0_cells.csv missing)"); rows.append(row); continue
            r = one_quantity_probe(lab, fb, vals, n_perm=n_perm, seed_null=SEED_LABEL_NULL + int(seed_offset), n_perm_required=n_perm_required)
            v = r["verdict"]
            r["verdict"] = V.LEAK_AUDIBLE if v == V.PASS else (V.LEAK_NOT_AUDIBLE if v == V.NULL_INSIDE else v)
            row.update(r); rows.append(row)
            sb = vals[lab["y"] == "sandbox"]; ben = vals[lab["y"] == "benign"]
            stats[k] = {"median_sandbox": float(np.nanmedian(sb)) if np.isfinite(sb).any() else None, "median_benign": float(np.nanmedian(ben)) if np.isfinite(ben).any() else None}
            if stats[k]["median_sandbox"] is not None and stats[k]["median_benign"] is not None:
                stats[k]["difference"] = stats[k]["median_sandbox"] - stats[k]["median_benign"]
    S.write_csv(p, LEAK_COLUMNS, rows)
    S.write_params(p, "plan11.detection.leak_probe.v1", _params(out, {"n_perm": n_perm, "seed_offset": seed_offset, "quantities": list(LEAK_QUANTITIES), "per_class": stats,
                                                                       "statistic": "auc", "excluded_by_declaration": list(DM.EXCLUDED_BY_DECLARATION)},
                                                                 [C.join_path(out), _det(out) / "admissibility.csv", _det(out) / "gk0_cells.csv"]), CIT_LEAK)
    return p


def alias_detection(out: Path, rung: str, grid_id: str | None = None, *, top_k: int = ALIAS_TOP_K, r2_threshold: float = ALIAS_R2, unit: str = ALIAS_UNIT,
                    n_perm: int = ORDER_NULL_PERM, seed_offset: int = 0, n_perm_required: int = ORDER_NULL_PERM) -> Path:
    """alias.csv: for the top_k features by importance_mean of the LOWO norm run (every feature when
    d <= top_k): the per-cell feature (the cell row) regressed across all admissible cells on
    dt_est_s from the sidecars with workload fixed effects (feature and dt_est_s centred per
    workload_key, ``unit = "within_workload"``, al-Kindi review 2.9; ``"pooled"`` is the epoch-1 form):
    slope, r2, n, verdict ALIAS_MOVES when r2 > r2_threshold else ALIAS_STAYS; and, when
    inputs/iteration_counts.csv exists (cell_id, iteration_count), on the iteration count likewise,
    else the second block reads not_run('no iteration count (stage 1)'). The row ``cadence_as_class``
    (ML 3.4, literally): the one-feature model on dt_est_s alone under LOWO with the workload-level
    null, verdict CADENCE_AUDIBLE on strict exceedance (one_quantity_probe). Both regressors are
    logs, never features; the sidecar's ``path`` and ``traj_file`` are never read. Citation: CIT_ALIAS."""
    out = _out(out)
    gid, _ = _grid(out, rung, grid_id)
    p = _det(out) / "alias.csv"
    rows = []
    if gid is None:
        rows.append({"rung": rung, "grid_id": "", "feature": "", "regressor": "", "verdict": _no_selection(rung)})
    else:
        sc = _scores(out, rung, gid, "lowo", True)
        try:
            data = DM.load_detection_data(out, rung, gid, normalized=True)
        except FileNotFoundError as e:
            data = None; rows.append({"rung": rung, "grid_id": gid, "feature": "", "regressor": "", "verdict": V.not_run(f"input missing: {Path(str(e)).name}")})
        if data is not None:
            lab, X, names = data["lab"], data["X"], data["names"]
            imp = (sc or {}).get("importance_mean") or {}
            order = sorted(range(len(names)), key=lambda j: -float(imp.get(names[j], 0.0))) if imp else list(range(len(names)))
            top = order[: int(top_k)] if len(names) > int(top_k) else order
            dt = np.array([float((S.read_json(S.sidecar_path(out, c)) or {}).get("dt_est_s") or np.nan) if S.sidecar_path(out, c).is_file() else np.nan for c in lab["cell_id"]])
            ic_p = out / "inputs" / "iteration_counts.csv"
            ic = {r["cell_id"]: _num(r.get("iteration_count")) for r in S.read_csv(ic_p)} if ic_p.is_file() else None
            regressors = [("dt_est_s", dt)]
            if ic is not None:
                regressors.append(("iteration_count", np.array([ic.get(c, np.nan) if ic.get(c) is not None else np.nan for c in lab["cell_id"]], dtype=np.float64)))
            wk = lab["workload_key"]
            for reg_name, reg in regressors:
                spread = float(np.nanmax(reg) - np.nanmin(reg)) if np.isfinite(reg).any() else None
                for j in top:
                    x = X[:, j].astype(np.float64); ok = np.isfinite(x) & np.isfinite(reg)
                    xr, rr = x[ok].copy(), reg[ok].copy()
                    if unit == "within_workload":
                        for w in set(wk[ok].tolist()):
                            m = wk[ok] == w
                            xr[m] -= xr[m].mean(); rr[m] -= rr[m].mean()
                    s, b, r2 = _ols(rr, xr)
                    row = {"rung": rung, "grid_id": gid, "feature": names[j], "regressor": reg_name, "unit": unit, "slope": s if np.isfinite(s) else "", "intercept": b if np.isfinite(b) else "",
                           "r2": r2 if np.isfinite(r2) else "", "n": int(ok.sum()), "dt_spread_s": spread if reg_name == "dt_est_s" else ""}
                    row["verdict"] = V.not_applicable("regressor has no spread") if not np.isfinite(s) else (V.ALIAS_MOVES if r2 > r2_threshold else V.ALIAS_STAYS)
                    rows.append(row)
            if ic is None:
                rows.append({"rung": rung, "grid_id": gid, "feature": "", "regressor": "iteration_count", "unit": unit, "verdict": V.not_run("no iteration count (stage 1)")})
            # the cadence row (ML 3.4 literally)
            if np.isfinite(dt).any():
                r = one_quantity_probe(lab, data["floor_by_cell"] or {}, dt, n_perm=n_perm, seed_null=SEED_LABEL_NULL + int(seed_offset), n_perm_required=n_perm_required)
                v = r["verdict"]
                rows.append({"rung": rung, "grid_id": gid, "feature": "cadence_as_class", "regressor": "dt_est_s", "unit": "lowo", "slope": r["auc"], "intercept": r["null_p95"],
                             "r2": "", "n": r["n_cells"], "dt_spread_s": float(np.nanmax(dt) - np.nanmin(dt)),
                             "verdict": V.CADENCE_AUDIBLE if v == V.PASS else (V.LEAK_NOT_AUDIBLE if v == V.NULL_INSIDE else v)})
    _replace_rows(p, ALIAS_COLUMNS, rows, lambda r: (r["rung"], r["feature"], r["regressor"]))
    S.write_params(p, "plan11.detection.alias.v1", _params(out, {"top_k": top_k, "r2_threshold": r2_threshold, "unit": unit, "n_perm": n_perm, "seed_offset": seed_offset,
                                                                  "cadence_row": "slope = AUC, intercept = null p95 (the cadence_as_class row)",
                                                                  "never_read": ["path", "traj_file"]}, [out / "gates" / "selection.json", out / "inputs" / "iteration_counts.csv"]), CIT_ALIAS)
    return p


# --------------------------------------------------------------------------- 3.5.15 G-V two-class

GV_COLUMNS = ("rung", "grid_id", "feature", "L0", "L2", "L3", "L0_b", "L2_b", "L3_families", "L3_over_L3_families")
GV_SUMMARY_COLUMNS = ("rung", "grid_id", "n_features", "n_features_L3_le_L3_families", "note")
GV_MEMBER_COLUMNS = ("rung", "grid_id", "member_index", "subfamily_letter", "n_cells", "L0_member", "L0_ratio", "note")


def gv_two_class(out: Path, rung: str, grid_id: str | None = None, *, reps_identical_ratio: float = REPS_IDENTICAL_RATIO, note_fraction: float = GV_NOTE_FRACTION) -> Path:
    """On the norm features at the grid point, the per-cell vector = the cell row. Per feature: L0 =
    mean over sandbox members of the population variance across the member's cells; L2 = the
    population variance across member means; L3 = the population variance across the two class
    means; and for the benign side L0_b (mean over benign workloads of the within-workload
    variance), L2_b (variance across benign workload means within a family, averaged over families
    with >= 2 workloads), L3_families (the population variance across benign family means).
    gv_two_class.csv; gv_two_class_summary.csv: n_features, n_features_L3_le_L3_families, note = 'a
    class whose L3 is inside the benign families' mutual spread has no more form than any two
    families have between them' when the count is at least GV_NOTE_FRACTION of the features, else
    ''; gv_two_class_members.csv (ML review 2.8): per member L0_member (the median over features of
    the within-member variance) and L0_ratio (the median over features of L0_member_f / the median
    over benign workloads of the within-workload variance), note REPS_NEAR_IDENTICAL when L0_ratio <
    reps_identical_ratio. No refusal; a report. Citation: CIT_GV."""
    out = _out(out)
    gid, _ = _grid(out, rung, grid_id)
    p = _det(out) / "gv_two_class.csv"; ps = _det(out) / "gv_two_class_summary.csv"; pm = _det(out) / "gv_two_class_members.csv"
    rows, srow, mrows = [], {"rung": rung, "grid_id": gid or "", "n_features": "", "n_features_L3_le_L3_families": "", "note": ""}, []
    if gid is None:
        srow["note"] = _no_selection(rung)
    else:
        try:
            data = DM.load_detection_data(out, rung, gid, normalized=True)
        except FileNotFoundError as e:
            data = None; srow["note"] = V.not_run(f"input missing: {Path(str(e)).name}")
        if data is not None:
            lab, X, names = data["lab"], data["X"], data["names"]
            y, wk, fam, mem = lab["y"], lab["workload_key"], lab["family"], lab["member_index"]
            sb = y == "sandbox"; ben = y == "benign"
            members = sorted(set(mem[sb].tolist())); bwk = sorted(set(wk[ben].tolist()))
            n_le = 0
            with np.errstate(all="ignore"):
                for j, name in enumerate(names):
                    x = X[:, j].astype(np.float64)
                    l0 = [float(np.nanvar(x[sb & (mem == m)])) for m in members if np.sum(sb & (mem == m)) >= 1]
                    means_m = [float(np.nanmean(x[sb & (mem == m)])) for m in members]
                    L0 = float(np.mean(l0)) if l0 else None; L2 = float(np.var(means_m)) if means_m else None
                    L3 = float(np.var([np.nanmean(x[sb]), np.nanmean(x[ben])])) if (sb.any() and ben.any()) else None
                    l0b = [float(np.nanvar(x[ben & (wk == w)])) for w in bwk]
                    L0_b = float(np.mean(l0b)) if l0b else None
                    l2b = []
                    for f in sorted(set(fam[ben].tolist())):
                        ws = sorted(set(wk[ben & (fam == f)].tolist()))
                        if len(ws) >= 2:
                            l2b.append(float(np.var([np.nanmean(x[ben & (wk == w)]) for w in ws])))
                    L2_b = float(np.mean(l2b)) if l2b else None
                    fm = [np.nanmean(x[ben & (fam == f)]) for f in sorted(set(fam[ben].tolist()))]
                    L3f = float(np.var(fm)) if len(fm) >= 2 else None
                    ratio = (L3 / L3f) if (L3 is not None and L3f not in (None, 0.0)) else None
                    if L3 is not None and L3f is not None and L3 <= L3f:
                        n_le += 1
                    rows.append({"rung": rung, "grid_id": gid, "feature": name, "L0": L0, "L2": L2, "L3": L3, "L0_b": L0_b, "L2_b": L2_b, "L3_families": L3f, "L3_over_L3_families": ratio})
                srow.update({"n_features": len(names), "n_features_L3_le_L3_families": n_le,
                             "note": "a class whose L3 is inside the benign families' mutual spread has no more form than any two families have between them" if (names and n_le >= note_fraction * len(names)) else ""})
                for m in members:
                    mm = sb & (mem == m)
                    letter = str(lab["subfamily_letter"][mm][0])
                    v_m = np.array([np.nanvar(X[mm, j]) for j in range(len(names))], dtype=np.float64)
                    v_b = np.array([np.median([np.nanvar(X[ben & (wk == w), j]) for w in bwk]) if bwk else np.nan for j in range(len(names))], dtype=np.float64)
                    ratios = v_m / v_b
                    ratios = ratios[np.isfinite(ratios)]
                    L0m = float(np.nanmedian(v_m)) if len(v_m) else None
                    L0r = float(np.median(ratios)) if len(ratios) else None
                    mrows.append({"rung": rung, "grid_id": gid, "member_index": m, "subfamily_letter": letter, "n_cells": int(mm.sum()), "L0_member": L0m, "L0_ratio": L0r,
                                  "note": V.REPS_NEAR_IDENTICAL if (L0r is not None and L0r < reps_identical_ratio) else ""})
    _replace_rows(p, GV_COLUMNS, rows, lambda r: (r["rung"], r["feature"]))
    _replace_rows(ps, GV_SUMMARY_COLUMNS, [srow], lambda r: r["rung"])
    _replace_rows(pm, GV_MEMBER_COLUMNS, mrows, lambda r: (r["rung"], str(r["member_index"])))
    prm = _params(out, {"reps_identical_ratio": reps_identical_ratio, "note_fraction": note_fraction, "variance": "population"}, [out / "gates" / "selection.json"])
    S.write_params(p, "plan11.detection.gv_two_class.v1", prm, CIT_GV); S.write_params(ps, "plan11.detection.gv_two_class.v1", prm, CIT_GV)
    S.write_params(pm, "plan11.detection.gv_two_class.v1", prm, CIT_GV)
    return p


# --------------------------------------------------------------------------- 3.5.16 the miss table

MISS_COLUMNS = ("cell_id", "member_index", "subfamily_letter", "rung", "score", "threshold_05", "nearest_workload", "nearest_family", "distance", "axis",
                "axis_of_largest", "d_amount", "d_identity", "amount_cell", "identity_cell", "amount_centroid", "identity_centroid", "status")
FP_COLUMNS = ("cell_id", "family", "workload_key", "rung", "score", "threshold_05", "nearest_member_index", "nearest_subfamily_letter", "distance", "axis",
              "axis_of_largest", "d_amount", "d_identity", "amount_cell", "identity_cell")


def plane_coordinates(out: Path, cell_ids: list[str], *, identity: str = MISS_IDENTITY) -> tuple[dict, dict]:
    """The plane per cell (K3 move 11, 18): amount = the median over its pairs of r_l0_q50_per,
    identity = the median of J - J_null ('excess', al-Kindi review 2.7) or of J ('raw'); one per-pair
    mask for every class, K > k_factor * floor_K with k_factor and floor_K from gates/gj.json
    (plane_mask = 'gj_mask_K, every class'); unmasked, and said so, when gj.json is absent. Returns
    ({cell_id: (amount, identity)}, mask_params)."""
    gj = out / "gates" / "gj.json"
    kf = fK = None
    if gj.is_file():
        d = S.read_json(gj); kf = _num((d.get("params") or {}).get("k_factor")); fK = _num(d.get("floor_median_K"))
    masked = kf is not None and fK is not None
    coords = {}
    for c in cell_ids:
        ex = S.load_extract_cached(out, c)
        K = np.asarray(ex["K"], dtype=np.float64)
        m = (K > kf * fK) if masked else np.ones(len(K), dtype=bool)
        a = np.asarray(ex["r_l0_q50_per"], dtype=np.float64)[m]
        idv = (np.asarray(ex["J"], dtype=np.float64) - np.asarray(ex["J_null"], dtype=np.float64)) if identity == "excess" else np.asarray(ex["J"], dtype=np.float64)
        idv = idv[m]
        a = a[np.isfinite(a)]; idv = idv[np.isfinite(idv)]
        coords[c] = (float(np.median(a)) if len(a) else np.nan, float(np.median(idv)) if len(idv) else np.nan)
    return coords, {"plane_mask": "gj_mask_K, every class" if masked else "unmasked (gates/gj.json absent)", "k_factor": kf, "floor_K": fK, "identity": identity}


def miss_table(out: Path, rung: str, grid_id: str | None = None, *, distance: str = MISS_DISTANCE, identity: str = MISS_IDENTITY) -> Path:
    """The miss table (K3 move 18; N3 Sec. 1 RQ2, Table 10). The plane per cell (plane_coordinates),
    each axis standardized by its population std over every admissible cell. Per benign workload the
    cloud = its cells' points and its centroid. For every sandbox cell missed under LOWO at the
    operating point (flag_05 false, in_denominator true): the nearest benign workload by the
    standardized Euclidean distance to its centroid ('nearest_cell': to its nearest cell), its
    family, the distance, axis = the resemblance axis (the axis whose absolute standardized
    difference to the centroid is smaller), axis_of_largest = where the residual difference lies,
    d_amount and d_identity (the signed standardized differences), the cell's and the centroid's
    coordinates. A sandbox cell at floor is listed with status AT_FLOOR_NOT_A_MISS and no neighbour.
    fp_table.csv: every benign false positive at the operating point against the members' centroids.
    Read as assignments with counts, never as a confusion matrix. The physical reason (M1 to M6) is
    the author's column and is left empty. Citation: CIT_MISS."""
    out = _out(out)
    gid, _ = _grid(out, rung, grid_id)
    p = _det(out) / "miss_table.csv"; pf = _det(out) / "fp_table.csv"
    rows, frows = [], []
    if gid is None:
        rows.append({"cell_id": "", "rung": rung, "status": _no_selection(rung)})
        mask_params = {}
    else:
        preds = _preds(out, rung, gid, "lowo", True)
        adm = _adm_set(out, pair=True); fb = _floor(out)
        join = [j for j in C.load_join(out) if j["cell_id"] in adm]
        if not preds:
            rows.append({"cell_id": "", "rung": rung, "status": V.not_run("lowo norm split not run")}); mask_params = {}
        else:
            coords, mask_params = plane_coordinates(out, [j["cell_id"] for j in join], identity=identity)
            A = np.array([coords[j["cell_id"]][0] for j in join]); I = np.array([coords[j["cell_id"]][1] for j in join])
            sa = float(np.nanstd(A)) or 1.0; si = float(np.nanstd(I)) or 1.0
            z = {j["cell_id"]: ((coords[j["cell_id"]][0]) / sa, (coords[j["cell_id"]][1]) / si) for j in join}
            jm = {j["cell_id"]: j for j in join}
            clouds: dict = {}
            for j in join:
                clouds.setdefault(("benign" if j["y"] == "benign" else "sandbox", j["workload_key"]), []).append(j["cell_id"])
            def centroid(ids):
                zz = np.array([z[c] for c in ids]); return (float(np.nanmean(zz[:, 0])), float(np.nanmean(zz[:, 1])))
            def nearest(cid, side):
                best = None
                for (sd, wk), ids in clouds.items():
                    if sd != side:
                        continue
                    if distance == "nearest_cell":
                        cands = [(math.hypot(z[cid][0] - z[o][0], z[cid][1] - z[o][1]), o) for o in ids]
                        dd, o = min(cands); cen = z[o]
                    else:
                        cen = centroid(ids); dd = math.hypot(z[cid][0] - cen[0], z[cid][1] - cen[1])
                    if not np.isfinite(dd):
                        continue
                    if best is None or dd < best[0]:
                        best = (dd, wk, cen)
                return best
            for r in preds:
                c = r["cell_id"]
                if c not in jm:
                    continue
                if r["y"] == "sandbox":
                    j = jm[c]
                    base = {"cell_id": c, "member_index": j["member_index"], "subfamily_letter": j["subfamily_letter"], "rung": rung, "score": _num(r["score"]), "threshold_05": _num(r["threshold_05"])}
                    if fb.get(c) in DM.FLOOR_VERDICTS_OUT:
                        rows.append({**base, "status": V.AT_FLOOR_NOT_A_MISS}); continue
                    if str(r.get("in_denominator")).lower() != "true" or str(r.get("flag_05")).lower() == "true":
                        continue
                    nb = nearest(c, "benign")
                    if nb is None:
                        rows.append({**base, "status": V.not_applicable("no benign cloud with finite coordinates")}); continue
                    dd, wk, cen = nb
                    da, di = z[c][0] - cen[0], z[c][1] - cen[1]
                    fam = next(jj["family"] for jj in join if jj["workload_key"] == wk)
                    rows.append({**base, "nearest_workload": wk, "nearest_family": fam, "distance": dd, "axis": "amount" if abs(da) <= abs(di) else "identity",
                                 "axis_of_largest": "identity" if abs(da) <= abs(di) else "amount", "d_amount": da, "d_identity": di,
                                 "amount_cell": coords[c][0], "identity_cell": coords[c][1], "amount_centroid": cen[0] * sa, "identity_centroid": cen[1] * si, "status": "miss"})
                elif str(r.get("flag_05")).lower() == "true":
                    j = jm[c]
                    nb = nearest(c, "sandbox")
                    if nb is None:
                        continue
                    dd, wk, cen = nb
                    da, di = z[c][0] - cen[0], z[c][1] - cen[1]
                    mj = next(jj for jj in join if jj["workload_key"] == wk)
                    frows.append({"cell_id": c, "family": j["family"], "workload_key": j["workload_key"], "rung": rung, "score": _num(r["score"]), "threshold_05": _num(r["threshold_05"]),
                                  "nearest_member_index": mj["member_index"], "nearest_subfamily_letter": mj["subfamily_letter"], "distance": dd,
                                  "axis": "amount" if abs(da) <= abs(di) else "identity", "axis_of_largest": "identity" if abs(da) <= abs(di) else "amount",
                                  "d_amount": da, "d_identity": di, "amount_cell": coords[c][0], "identity_cell": coords[c][1]})
            if not rows:
                rows.append({"cell_id": "", "rung": rung, "status": "no miss at the operating point"})
    _replace_rows(p, MISS_COLUMNS, rows, lambda r: (r["rung"], r["cell_id"]))
    _replace_rows(pf, FP_COLUMNS, frows if frows else [{"cell_id": "", "rung": rung, "family": "", "workload_key": ""}], lambda r: (r["rung"], r["cell_id"]))
    prm = _params(out, {"distance": distance, "identity": identity, "axes": list(MISS_AXES), "standardization": "population std over every admissible cell", **mask_params,
                        "physical_reason": "the author's column (M1 to M6), left empty"}, [out / "gates" / "selection.json", out / "gates" / "gj.json"])
    S.write_params(p, "plan11.detection.miss_table.v1", prm, CIT_MISS); S.write_params(pf, "plan11.detection.miss_table.v1", prm, CIT_MISS)
    return p


# --------------------------------------------------------------------------- the head-drop template (al-Farabi M5)

def head_drop_template(out: Path) -> Path:
    """inputs/head_drop.csv with one row per workload key of cell_classes.csv (the public keys),
    head_drop_pairs = 0 and the reason 'default 0; declared at D2'; existing rows are kept. A
    member's head drop is the author's number from the program's specification, fixed at D2 and
    never read off the APF(t) figure of D5 (al-Farabi M5; al-Kindi review 2.6). Citation: CIT_HEAD."""
    out = _out(out)
    p = out / "inputs" / "head_drop.csv"
    rows = S.read_csv(p) if p.is_file() else []
    have = {r.get("kernel") for r in rows}
    for j in C.load_join(out):
        k = j["workload_key"]
        if k and k not in have:
            rows.append({"kernel": k, "head_drop_pairs": 0, "reason": "default 0; declared at D2"}); have.add(k)
    return S.write_csv(p, S.HEAD_DROP_COLUMNS, rows)


# --------------------------------------------------------------------------- CLI (3.5.17)

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="gates_detection.py")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, rung=False, grid=False):
        s = sub.add_parser(name); s.add_argument("--out", required=True)
        if rung: s.add_argument("--rung", required=True, choices=S.RUNGS)
        if grid: s.add_argument("--grid-id", default=None)
        return s
    a = add("admissibility"); a.add_argument("--c1-rule", default=DET_C1_RULE, choices=("report", "exclude"))
    add("gk0-sandbox-template"); add("head-drop-template")
    g = add("gk0-cells"); g.add_argument("--idle-percentile", type=float, default=GK0_IDLE_PERCENTILE); g.add_argument("--envelope-percentile", type=float, default=GK0_ENVELOPE_PERCENTILE)
    add("gn"); add("gl", True, True)
    o = add("gop", True, True); o.add_argument("--split", default="lowo", choices=("lowo", "loco", "lofo", "all")); o.add_argument("--variant", default="norm", choices=("norm", "raw"))
    o.add_argument("--cells-rule", default=GOP_CELLS, choices=("threshold_setters", "realized_fps"))
    l = add("glm", True, True); l.add_argument("--level-quantity", default=LEVEL_QUANTITY, choices=("median_K", "per_iteration_K_sum")); l.add_argument("--band-factor", type=float, default=GLM_BAND_FACTOR)
    l.add_argument("--model", default=GLM_MODEL, choices=("retrain", "headline_oof")); l.add_argument("--vanish-rule", default=GLM_VANISH_RULE, choices=("tpr_le_fpr", "below_half_unrestricted"))
    l.add_argument("--n-jobs", type=int, default=1); l.add_argument("--seed-offset", type=int, default=0); l.add_argument("--n-estimators", type=int, default=DM.N_ESTIMATORS)
    l.add_argument("--threshold-source", default=DM.THRESHOLD_SOURCE, choices=DM.THRESHOLD_SOURCES)
    n = add("anchor"); n.add_argument("--part", default="all", choices=("all", "kernels", "idle_sets", "idle_early_late")); n.add_argument("--rung", default=None, choices=S.RUNGS)
    n.add_argument("--null-perm", type=int, default=ORDER_NULL_PERM); n.add_argument("--n-jobs", type=int, default=1); n.add_argument("--seed-offset", type=int, default=0)
    n.add_argument("--n-estimators", type=int, default=DM.N_ESTIMATORS); n.add_argument("--n-perm-required", type=int, default=ORDER_NULL_PERM)
    r = add("order"); r.add_argument("--consequence", default=ORDER_TEST_CONSEQUENCE, choices=("size", "void")); r.add_argument("--scope", default=ORDER_TEST_SCOPE, choices=("within_class", "campaign"))
    r.add_argument("--rung", default=None, choices=S.RUNGS); r.add_argument("--null-perm", type=int, default=ORDER_NULL_PERM); r.add_argument("--n-jobs", type=int, default=1)
    r.add_argument("--seed-offset", type=int, default=0); r.add_argument("--n-estimators", type=int, default=DM.N_ESTIMATORS); r.add_argument("--n-perm-required", type=int, default=ORDER_NULL_PERM)
    d = add("drift"); d.add_argument("--null-perm", type=int, default=ORDER_NULL_PERM); d.add_argument("--seed-offset", type=int, default=0)
    add("gsig", True, True)
    f = add("gfp", True, True); f.add_argument("--flag-fraction", type=float, default=GFP_FLAG_FRACTION); f.add_argument("--n-jobs", type=int, default=1); f.add_argument("--seed-offset", type=int, default=0)
    f.add_argument("--n-estimators", type=int, default=DM.N_ESTIMATORS); f.add_argument("--threshold-source", default=DM.THRESHOLD_SOURCE, choices=DM.THRESHOLD_SOURCES)
    add("g1c", True, True)
    h = add("harness", True, True); h.add_argument("--comparable-tol", type=float, default=HARNESS_COMPARABLE_TOL)
    add("gcal", True, True)
    m = add("gm"); m.add_argument("--n-seeds", type=int, default=GM_N_SEEDS); m.add_argument("--n-jobs", type=int, default=1); m.add_argument("--seed-offset", type=int, default=0)
    m.add_argument("--n-estimators", type=int, default=DM.N_ESTIMATORS); m.add_argument("--threshold-source", default=DM.THRESHOLD_SOURCE, choices=DM.THRESHOLD_SOURCES)
    add("gdim")
    al = add("alias", True, True); al.add_argument("--top-k", type=int, default=ALIAS_TOP_K); al.add_argument("--r2-threshold", type=float, default=ALIAS_R2)
    al.add_argument("--unit", default=ALIAS_UNIT, choices=("within_workload", "pooled")); al.add_argument("--null-perm", type=int, default=ORDER_NULL_PERM); al.add_argument("--seed-offset", type=int, default=0)
    gv = add("gv", True, True); gv.add_argument("--reps-identical-ratio", type=float, default=REPS_IDENTICAL_RATIO)
    mt = add("miss-table", True, True); mt.add_argument("--distance", default=MISS_DISTANCE, choices=("standardized_euclidean_to_centroid", "nearest_cell")); mt.add_argument("--identity", default=MISS_IDENTITY, choices=("excess", "raw"))
    lk = add("leak-probe"); lk.add_argument("--null-perm", type=int, default=ORDER_NULL_PERM); lk.add_argument("--seed-offset", type=int, default=0)
    args = ap.parse_args(argv)
    out = Path(args.out)
    if not (out / "cells.csv").is_file():
        print(f"missing input: {out / 'cells.csv'}", file=sys.stderr); return 2
    try:
        cmd = args.cmd
        if cmd == "admissibility": print(admissibility(out, det_c1_rule=args.c1_rule))
        elif cmd == "gk0-sandbox-template": print(gk0_sandbox_template(out))
        elif cmd == "head-drop-template": print(head_drop_template(out))
        elif cmd == "gk0-cells": print(gk0_cells(out, idle_percentile=args.idle_percentile, envelope_percentile=args.envelope_percentile))
        elif cmd == "gn": print(gn_two_class(out))
        elif cmd == "gl": print(gl_two_class(out, args.rung, args.grid_id))
        elif cmd == "gop":
            for sp in (("lowo", "loco", "lofo") if args.split == "all" else (args.split,)):
                print(gop(out, args.rung, args.grid_id, split=sp, variant=args.variant, cells_rule=args.cells_rule))
        elif cmd == "glm":
            print(glm(out, args.rung, args.grid_id, quantity=args.level_quantity, band_factor=args.band_factor, model=args.model, vanish_rule=args.vanish_rule, n_jobs=args.n_jobs,
                      seed_offset=args.seed_offset, n_estimators=args.n_estimators, threshold_source=args.threshold_source))
        elif cmd == "anchor":
            print(anchor(out, part=args.part, n_perm=args.null_perm, n_jobs=args.n_jobs, seed_offset=args.seed_offset, rung=args.rung, n_estimators=args.n_estimators, n_perm_required=args.n_perm_required))
        elif cmd == "order":
            print(order_test(out, consequence=args.consequence, scope=args.scope, n_perm=args.null_perm, n_jobs=args.n_jobs, seed_offset=args.seed_offset, rung=args.rung,
                             n_estimators=args.n_estimators, n_perm_required=args.n_perm_required))
        elif cmd == "drift": print(drift_regression(out, n_perm=args.null_perm, seed_offset=args.seed_offset))
        elif cmd == "gsig": print(gsig(out, args.rung, args.grid_id))
        elif cmd == "gfp":
            print(gfp(out, args.rung, args.grid_id, flag_fraction=args.flag_fraction, n_jobs=args.n_jobs, seed_offset=args.seed_offset, n_estimators=args.n_estimators, threshold_source=args.threshold_source))
        elif cmd == "g1c": print(g1c(out, args.rung, args.grid_id))
        elif cmd == "harness": print(harness(out, args.rung, args.grid_id, comparable_tol=args.comparable_tol))
        elif cmd == "gcal": print(gcal(out, args.rung, args.grid_id))
        elif cmd == "gm": print(gm(out, n_seeds=args.n_seeds, n_jobs=args.n_jobs, seed_offset=args.seed_offset, n_estimators=args.n_estimators, threshold_source=args.threshold_source))
        elif cmd == "gdim": print(gdim(out))
        elif cmd == "alias": print(alias_detection(out, args.rung, args.grid_id, top_k=args.top_k, r2_threshold=args.r2_threshold, unit=args.unit, n_perm=args.null_perm, seed_offset=args.seed_offset))
        elif cmd == "gv": print(gv_two_class(out, args.rung, args.grid_id, reps_identical_ratio=args.reps_identical_ratio))
        elif cmd == "miss-table": print(miss_table(out, args.rung, args.grid_id, distance=args.distance, identity=args.identity))
        elif cmd == "leak-probe": print(leak_probe(out, n_perm=args.null_perm, seed_offset=args.seed_offset))
    except FileNotFoundError as e:
        print(f"missing input: {e}", file=sys.stderr); return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
