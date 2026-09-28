#!/usr/bin/env python3
"""detection_fixtures.py -- a synthetic `<out>` tree in SPEC_DETECTION's declared record shapes
for builder B's tests (tables_detection, figures_detection, latex_skeleton_p3, run_detection).

Builder B (report), 2026-09-17. The two builders of the detection layer work in parallel, so
builder B cannot run builder A's classes, splits, metrics, levels and gates at build time; this
module writes what they would write, column for column as SPEC_DETECTION sections 2.3, 2.4,
2.5, 3.3.7, 3.4 and 3.5 declare, with the corrections of the three reviews that touch the
record shapes:

  - al-Kindi 3, 4: level-2 `confusion.csv` carries `null_p95, rank, verdict` per row and
    `n_at_floor`; level 3 records `null_unit = "cell"`.
  - al-Kindi 8: `miss_table.csv` and `fp_table.csv` carry `axis_of_largest`, `d_amount`,
    `d_identity` beside `axis`.
  - al-Kindi 10, 11: `drift.csv` carries `drift_unit`; `order.csv` carries `half_rule` with the
    values `within_workload` and `within_class`.
  - ML 2.3: `gop.csv` carries `n_folds_unsupported` and rows for `loco` and `lofo`.
  - ML 2.4: `scores.json["l1"]` (the best single feature under the same LOWO) and
    `score_source = "full"`.
  - ML 2.6: `leak_probe.csv` (`quantity, auc, null_p95, rank, verdict`).
  - ML 2.7 / al-Kindi 2: the one-class `scores.json` carries `null`; `g1c.csv` carries
    `null_verdict`.
  - ML 2.8: `gv_two_class_members.csv` (`rung, member_index, L0_member, L0_ratio, note`).
  - al-Farabi M3, M4: `n_without_score`, `excluded_no_score`, `score_status`, and
    `params.null_denominator_rule`; M9: the `unassigned` class in the join and in
    `admissibility.csv`.

The corpus is 24 cells: two kernels (gemm, floyd) at four reps, four idle cells, and three
sandbox members at four reps (members 1 and 2 in sub-family A, member 3 in sub-family B),
all named by index and letter only (`sandbox_member_<m>`). The numbers are synthetic; the
verdict strings are SPEC_DETECTION 3.0's. Every knob of `make_out` exists to make one table cell
print a refusal string:

  order_void_rung        order.csv carries consequence `void` and ORDER_VOID for that rung
  gf_void_rung           gates/gf.csv part (i) reads GF_VOID for that rung
  gc_disconnected_rung   gates/gc.csv `rep = all` reads `disconnected lead` for that rung
  null_inside_rung       the LOWO tpr05 null verdict reads NULL_INSIDE (G-L (i) -> level only;
                         G-SIG -> identity)
  null_not_estimable_rung  the null block reads NULL_NOT_ESTIMABLE (n_assignments below 20)
  smoke_perm             n_perm below 500 -> `not run: N permutations < 500`
  at_floor_member        every cell of that member is `at floor` (leaves the denominator)
  no_score_cell          one sandbox cell with `score_status = not applicable: no finite window`
  missing_split          a (rung, split) directory that is not written
  no_selection_rung      a rung absent from gates/selection.json
  one_class_inside_null  the one-class tpr05 null reads NULL_INSIDE while LOWO's reads PASS
  gop_set_by_few_rung    gop.csv reads GOP_SET_BY_FEW for that rung's lowo row
  stage2                 harness rows with margins (else HARNESS_STAGE2_ABSENT)
  n_unassigned           cells with `class = unassigned` in the join
  with_external          four `external` cells (test_only) and the `lowo/final` block
  no_order_index         no order_index anywhere; the letter sequence holds its `not run` line
  two_idle_campaigns     the idle cells split over two campaigns (G-ANCHOR (ii) audible)
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PKG = _HERE.parent
if str(_PKG.parent) not in sys.path:
    sys.path.insert(0, str(_PKG.parent))

import math
from collections import Counter

import numpy as np

from plan11_encoding_ladder._report_common import (  # noqa: E402
    EXTRACT_COLUMNS, N_PAGES, RUNGS, write_csv, write_json,
)

GRID = "W8_H4"
FEAT = ("mean", "std", "cov", "median", "max", "p95", "peak2med", "duty")
FEATURE_NAMES = {
    "apf": [f"apf.k_over_med.{f}" for f in FEAT],
    "wapf": [f"wapf.wapf_norm.{f}" for f in FEAT],
    "persist": [f"persist.j_excess.{f}" for f in FEAT],
}
FEATURE_NAMES["content"] = ([f"content.{ch}.{f}" for ch in ("r_l0_q50_per", "r_l1l0_q50_per", "r_haml0_q50_per") for f in FEAT]
                            + [f"content.r_{p}_{q}_per.wmean" for p in ("l0", "l1l0", "haml0") for q in ("q05", "q25", "q75", "q95")])
FEATURE_NAMES["combined"] = FEATURE_NAMES["apf"] + FEATURE_NAMES["wapf"] + FEATURE_NAMES["persist"] + FEATURE_NAMES["content"]
RAW_NAMES = {"apf": [f"apf.apf.{f}" for f in FEAT]}

# the vocabulary of SPEC_DETECTION 3.0 (literals; builder A's verdicts.py may not carry them yet)
PASS, FAIL = "pass", "fail"
NULL_NOT_ESTIMABLE = "null_not_estimable"
NULL_INSIDE = "inside the workload-level null"
GOP_SUPPORTED = "operating point supported"
GOP_SET_BY_FEW = "set by one workload, family named"
GLM_SURVIVES = "detection survives level matching"
GLM_LEVEL_ONLY = "detected by level in this lead"
GLM_NOT_DETECTED = "not applicable: not detected at the operating point"
GLM_EMPTY_BAND = "not applicable: no benign workload within the level band"
GLM_LABEL_MEDIAN_K = "median K, stage 1"
GANCHOR_AUDIBLE = "campaign audible"
GANCHOR_NOT_AUDIBLE = "campaign not audible above the null"
ORDER_AUDIBLE = "position audible"
ORDER_NOT_AUDIBLE = "position not audible above the null"
ORDER_VOID = "void: position audible inside the interleaved campaign"
DRIFT_SLOPE = "drift: slope above the shuffle null"
DRIFT_NONE = "no drift above the shuffle null"
GSIG_REPORTED = "gap reported"
GSIG_IDENTITY = "recognizes workload identity, not family behaviour"
GFP_ATTRIBUTED = "attributed"
GFP_INSEPARABLE = "inseparable from sandbox under this rung"
G1C_PRIMARY = "primary"
G1C_SECONDARY = "secondary, not citable"
HARNESS_STAGE2_ABSENT = "not run: stage 2 absent"
HARNESS_COMPARABLE = "harness's (comparable margins)"
HARNESS_CLASS_EXCEEDS = "class margin exceeds the harness margin"
GCAL_AGREE = "per-fold and pooled agree within the null spread"
GCAL_PERFOLD = "per-fold reported; pooled curve threshold set post hoc on the held-out scores"
POST_HOC_LABEL = "threshold set post hoc on the held-out scores"
GK0_AT_FLOOR = "at floor"
GK0_ABOVE_FLOOR = "above floor"
GN_HEADLINE = "headline"
GN_ONE_TRAIN_WORKLOAD = "one training workload per fold"
GN_SINGLE_WORKLOAD = "single workload"
L2_ONE_TRAIN = "one training member per fold"
SIGNATURE_CEILING = "signature ceiling"
LADDER_FROM_PAIR1_ONLY = "from pair 1 only"
AT_FLOOR_NOT_A_MISS = "at floor, not a miss"
LEAK_AUDIBLE = "leak audible"
LEAK_NOT_AUDIBLE = "leak not audible above the null"
GDIM_FULL = "full vector"
GDIM_REDUCED = "declared reduction"
GM_BEATS = "beats"
GM_DIFFERENCE = "difference with margin"
ALIAS_MOVES = "moves with the interval"
ALIAS_STAYS = "does not move with the interval"
GF_INSEPARABLE = "inseparable at floor"
GF_VOID = "void: idle reps separable under this rung"
GC_DISCONNECTED = "disconnected lead"
GX_POOLING_STANDS = "pooling stands"
GX_LEAK = "campaign predictable"

CONTENT = {"double": (512 / 4096, 86.0, 4.0), "decay": (512 / 4096, 86.0, 4.0), "spin": (72 / 4096, 1.0, 2.0),
           "idle": (2 / 4096, 8.0, 2.0)}
# workload_key: (K0, content, churn, family, class)
WORKLOADS = {
    "gemm": (4096, "double", 0.10, "kernels", "benign_kernel"),
    "floyd": (2048, "decay", 0.10, "kernels", "benign_kernel"),
    "idle": (0, "idle", 0.02, "idle", "idle"),
    "sandbox_member_1": (2048, "spin", 0.02, "sandbox", "sandbox"),
    "sandbox_member_2": (4096, "spin", 0.02, "sandbox", "sandbox"),
    "sandbox_member_3": (2048, "double", 0.60, "sandbox", "sandbox"),
}
MEMBER_LETTER = {1: "A", 2: "A", 3: "B"}
CLASS_LETTER = {"sandbox": "S", "benign_kernel": "B", "benign_breadth": "B", "idle": "I",
                "harness_idle": "H", "benign_relaunched": "R", "external": "X"}


def _extract_rows(rng, K0: int, content: str, churn: float, n_pairs: int, floor_F: int = 150) -> list[dict]:
    """Rows of one extract in SPEC 2.2's columns (the shape of tests/report_fixtures.py, with
    the churn parameter driving J so that the identity axis differs per workload)."""
    r_l0, r_l1l0, r_ham = CONTENT[content]
    Ks = [int(K0 * (1 + rng.uniform(-0.02, 0.02))) + floor_F for _ in range(n_pairs)]
    rows = []
    for t in range(n_pairs):
        K = Ks[t]
        last = t == n_pairs - 1
        row = {c: "" for c in EXTRACT_COLUMNS}
        row["seq"] = t + 1
        row["K"] = K
        l0 = max(1.0, r_l0 * 4096 * (1 + rng.uniform(-0.1, 0.1)))
        l1 = l0 * r_l1l0
        ham = l0 * r_ham
        for ch, v in (("ham", ham), ("l0", l0), ("l1", l1)):
            row[f"{ch}_sum_all"] = int(v * K)
            for q, f in (("q05", 0.6), ("q25", 0.85), ("q50", 1.0), ("q75", 1.15), ("q95", 1.4)):
                row[f"{ch}_{q}_all"] = f"{v * f:.10g}"
        if not last:
            Kn = Ks[t + 1]
            inter = int(min(K, Kn) * (1 - churn))
            J = inter / (K + Kn - inter)
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


def _cells(reps: int, idle: int, *, with_external: bool, two_idle_campaigns: bool, n_unassigned: int) -> list[dict]:
    """cells.csv after `classes apply` (SPEC_DETECTION 2.3) plus the join columns of 2.4 kept on
    the same dict under `_class`, `_y`, `_workload_key`, `_family`, `_member`, `_letter`,
    `_split_role`, `_order_index`."""
    rows = []
    order = 1
    for k, a in (("gemm", "WORKING-SET"), ("floyd", "WORKING-SET")):
        for r in range(reps):
            camp = "01c" if k == "gemm" else "01c1"
            rows.append({"cell_id": f"{k}__rep{r:02d}__{camp}", "kernel": k, "role": "kernel", "archetype_predicted": a,
                         "seed": 42 if r == 0 else 1000 * r + 7, "rep": r, "rep_dir": r + 1, "label": "synth", "campaign": camp,
                         "path": "", "traj_file": "", "status": "ok",
                         "_class": "benign_kernel", "_y": "benign", "_workload_key": k, "_family": "kernels", "_member": 0,
                         "_letter": "-", "_split_role": "train_test", "_order_index": order})
            order += 1
    for m in (1, 2, 3):
        for r in range(reps):
            rows.append({"cell_id": f"sandbox_member_{m}__rep{r:02d}__stage1", "kernel": f"sandbox_member_{m}", "role": "sandbox",
                         "archetype_predicted": "sandbox", "seed": 42 if r == 0 else 1000 * r + 12 + m, "rep": r, "rep_dir": r + 1,
                         "label": "synth", "campaign": "stage1", "path": "", "traj_file": "", "status": "ok",
                         "_class": "sandbox", "_y": "sandbox", "_workload_key": f"sandbox_member_{m}", "_family": "sandbox",
                         "_member": m, "_letter": MEMBER_LETTER[m], "_split_role": "train_test", "_order_index": order})
            order += 1
    for r in range(idle):
        camp = ("01c" if r < idle // 2 else "dwarfs1") if two_idle_campaigns else "01c"
        rows.append({"cell_id": f"idle__rep{r:02d}__{camp}", "kernel": "idle", "role": "idle", "archetype_predicted": "control",
                     "seed": "", "rep": r, "rep_dir": r + 1, "label": "idle", "campaign": camp, "path": "", "traj_file": "",
                     "status": "ok", "_class": "idle", "_y": "benign", "_workload_key": "idle", "_family": "idle", "_member": 0,
                     "_letter": "-", "_split_role": "train_test", "_order_index": order})
        order += 1
    if with_external:
        for r in range(4):
            rows.append({"cell_id": f"external_member_1__rep{r:02d}__stage3", "kernel": "external_member_1", "role": "external",
                         "archetype_predicted": "external", "seed": 42 if r == 0 else 1000 * r + 1, "rep": r, "rep_dir": r + 1,
                         "label": "synth", "campaign": "stage3", "path": "", "traj_file": "", "status": "ok",
                         "_class": "external", "_y": "sandbox", "_workload_key": "external_member_1", "_family": "external",
                         "_member": 1, "_letter": "-", "_split_role": "test_only", "_order_index": order})
            order += 1
    for i in range(n_unassigned):
        rows.append({"cell_id": f"unassigned_{i}__rep00__01c", "kernel": f"unassigned_{i}", "role": "unknown",
                     "archetype_predicted": "unknown", "seed": 42, "rep": 0, "rep_dir": 1, "label": "synth", "campaign": "01c",
                     "path": "", "traj_file": "", "status": "refused: unknown kernel",
                     "_class": "unassigned", "_y": "", "_workload_key": "", "_family": "", "_member": "", "_letter": "",
                     "_split_role": "", "_order_index": ""})
    return rows


def _token(c: dict) -> str:
    L = CLASS_LETTER.get(c["_class"], "?")
    m = f"{c['_member']}" if L in ("S", "X") else ""
    return f"{L}{m}r{c['rep']}"


def _member_cells(cells):
    return [c for c in cells if c["_class"] in ("sandbox",)]


def _null_block(rng, observed: float, *, n_perm: int, n_assignments: int, verdict_override: str | None = None,
                inside: bool = False, statistic_name: str = "tpr05") -> dict:
    """The `null` payload of scores.json (SPEC_DETECTION 3.3.7): per statistic p95, p05, spread,
    rank, rank_text, n, exceeds, verdict."""
    vals = rng.uniform(0.0, 0.35, size=max(n_perm, 1))
    if inside:
        vals = np.minimum(1.0, rng.uniform(0.6, 1.0, size=max(n_perm, 1)) + max(0.0, observed - 0.7))
    p95, p05 = float(np.percentile(vals, 95)), float(np.percentile(vals, 5))
    exceeds = bool(observed > p95) and not inside
    rank = int((vals >= observed).sum()) + 1
    if verdict_override is not None:
        verdict = verdict_override
    elif inside and n_assignments >= 20 and n_perm >= 500:
        verdict = NULL_INSIDE
    elif n_assignments < 20:
        verdict = NULL_NOT_ESTIMABLE
    elif n_perm < 500:
        verdict = f"not run: {n_perm} permutations < 500"
    else:
        verdict = PASS if exceeds else NULL_INSIDE
    return {"p95": p95, "p05": p05, "spread": p95 - p05, "rank": rank, "rank_text": f"rank {rank} of {n_perm}", "n": n_perm,
            "exceeds": exceeds, "verdict": verdict, "statistic": statistic_name}


def _write_split(rng, d: Path, rung: str, split: str, variant: str, cells: list[dict], floor: dict, *, n_perm: int,
                 null_inside: bool = False, null_not_estimable: bool = False, no_score_cell: str | None = None,
                 quality: float = 0.95, one_class: bool = False, oc_inside: bool = False, gop_few: bool = False,
                 quarantine: bool = False, reduced: bool = False, with_external: bool = False) -> dict:
    """One split directory of SPEC_DETECTION 3.3.7: predictions.csv, scores.json, null.json,
    roc.csv, folds.json (and l1_quarantine.json)."""
    d.mkdir(parents=True, exist_ok=True)
    names = RAW_NAMES["apf"] if (rung == "apf" and variant == "raw") else FEATURE_NAMES[rung]
    dfull = len(names)
    dused = 8 if reduced else dfull
    preds = []
    pos_scored = ben_scored = 0
    S_keys = sorted({c["_workload_key"] for c in cells if c["_y"] == "sandbox" and c["_split_role"] == "train_test"})
    B_keys = sorted({c["_workload_key"] for c in cells if c["_y"] == "benign" and c["_split_role"] == "train_test"})
    for c in cells:
        if c["_class"] == "unassigned":
            continue
        if split == "lofo" and c["_y"] == "sandbox":
            continue  # sandbox never held out under LOFO
        if c["_split_role"] == "test_only" and not with_external:
            continue
        fv = floor.get(c["cell_id"], GK0_ABOVE_FLOOR)
        if c["_class"] in ("idle", "harness_idle"):
            fv = "control"
        thr05, thr01 = 0.55, 0.72
        if c["_y"] == "sandbox":
            base = quality if c["_member"] in (1, 2) else 0.2
        else:
            base = 0.15
        score = float(np.clip(rng.normal(base, 0.08), 0, 1))
        in_den = (c["_y"] == "sandbox" and fv == GK0_ABOVE_FLOOR) or (c["_y"] == "benign")
        status = ""
        if no_score_cell and c["cell_id"] == no_score_cell:
            score, status, in_den = None, "not applicable: no finite window", False
        fold = (f"{split}/{c['_workload_key']}" if split in ("lowo", "lofo", "one_class") else f"{split}/{c['cell_id']}")
        if c["_split_role"] == "test_only":
            fold = f"{split}/final"
        if split == "lofo":
            fold = f"lofo/{c['_family']}"
        preds.append({"cell_id": c["cell_id"], "class": c["_class"], "y": c["_y"], "workload_key": c["_workload_key"],
                      "family": c["_family"], "member_index": c["_member"], "subfamily_letter": c["_letter"], "rep": c["rep"],
                      "campaign": c["campaign"], "fold": fold, "score": "" if score is None else f"{score:.6f}",
                      "threshold_05": thr05, "threshold_01": thr01,
                      "flag_05": "" if score is None else str(score > thr05).lower(),
                      "flag_01": "" if score is None else str(score > thr01).lower(),
                      "n_windows": 1, "floor_verdict": fv, "in_denominator": str(in_den).lower(), "score_status": status})
        if c["_y"] == "sandbox" and c["_split_role"] == "train_test":
            pos_scored += 1
        elif c["_y"] == "benign":
            ben_scored += 1
    cols = ["cell_id", "class", "y", "workload_key", "family", "member_index", "subfamily_letter", "rep", "campaign", "fold",
            "score", "threshold_05", "threshold_01", "flag_05", "flag_01", "n_windows", "floor_verdict", "in_denominator",
            "score_status"]
    write_csv(d / "predictions.csv", preds, cols)
    den = [p for p in preds if p["y"] == "sandbox" and p["in_denominator"] == "true" and p["class"] != "external"]
    ben = [p for p in preds if p["y"] == "benign"]
    at_floor = [p["cell_id"] for p in preds if p["y"] == "sandbox" and p["floor_verdict"] == GK0_AT_FLOOR]
    no_score = [p["cell_id"] for p in preds if p["score_status"]]
    tpr05 = (sum(p["flag_05"] == "true" for p in den) / len(den)) if den else None
    tpr01 = (sum(p["flag_01"] == "true" for p in den) / len(den)) if den else None
    fpr05 = (sum(p["flag_05"] == "true" for p in ben) / len(ben)) if ben else None
    fpr01 = (sum(p["flag_01"] == "true" for p in ben) / len(ben)) if ben else None
    auc = None
    if den and ben:
        ps = [float(p["score"]) for p in den if p["score"]]
        ns = [float(p["score"]) for p in ben if p["score"]]
        auc = float(np.mean([[1.0 if a > b else (0.5 if a == b else 0.0) for b in ns] for a in ps])) if ps and ns else None
    per_member = {}
    for m in sorted({p["member_index"] for p in preds if p["y"] == "sandbox" and p["class"] != "external"}):
        mm = [p for p in preds if p["member_index"] == m and p["y"] == "sandbox" and p["class"] != "external"]
        dn = [p for p in mm if p["in_denominator"] == "true"]
        hits = sum(p["flag_05"] == "true" for p in dn)
        per_member[str(m)] = {"hits": hits, "denominator": len(dn), "at_floor": sum(p["floor_verdict"] == GK0_AT_FLOOR for p in mm),
                              "no_score": sum(bool(p["score_status"]) for p in mm),
                              "eighths": f"{hits}/{len(dn)}" if dn else f"at floor ({len(mm)})"}
    per_family_fpr = {}
    for fam in sorted({p["family"] for p in ben}):
        ff = [p for p in ben if p["family"] == fam]
        per_family_fpr[fam] = {"n": len(ff), "flagged": sum(p["flag_05"] == "true" for p in ff),
                               "fpr": sum(p["flag_05"] == "true" for p in ff) / len(ff)}
    per_workload = {}
    for wk in sorted({p["workload_key"] for p in preds}):
        ww = [p for p in preds if p["workload_key"] == wk]
        if ww[0]["y"] == "sandbox":
            dn = [p for p in ww if p["in_denominator"] == "true"]
            per_workload[wk] = (sum(p["flag_05"] == "true" for p in dn) / len(dn)) if dn else None
        else:
            per_workload[wk] = 1 - sum(p["flag_05"] == "true" for p in ww) / len(ww)
    n_assign = 5 if null_not_estimable else math.comb(len(S_keys) + len(B_keys), len(S_keys))
    null = None
    if split in ("lowo", "loco", "one_class") and tpr05 is not None:
        null = {"status": "ok", "tpr05": _null_block(rng, tpr05, n_perm=n_perm, n_assignments=n_assign, inside=(null_inside or oc_inside)),
                "auc": _null_block(rng, auc if auc is not None else 0.5, n_perm=n_perm, n_assignments=n_assign,
                                   inside=(null_inside or oc_inside), statistic_name="auc"),
                "n_assignments": n_assign, "exhaustive": n_assign < 500, "n_distinct_drawn": min(n_perm, n_assign),
                "S": len(S_keys), "B": len(B_keys), "unit": "cell" if split == "loco" else "workload"}
    if null is None and split in ("lowo", "loco", "one_class"):
        null = {"status": "not run: null not requested", "n_assignments": n_assign, "exhaustive": None, "n_distinct_drawn": None}
    if null and isinstance(null.get("tpr05"), dict) and isinstance(null["tpr05"].get("spread"), float) and tpr05 is not None:
        gcal_v = GCAL_AGREE if abs(tpr05 - (tpr05 or 0)) <= null["tpr05"]["spread"] else GCAL_PERFOLD
    else:
        gcal_v = "not run: null spread unavailable"
    setters = ["gemm__rep00__01c"] if gop_few else ["gemm__rep00__01c", "gemm__rep01__01c", "floyd__rep00__01c1", "floyd__rep02__01c1",
                                                    "idle__rep00__01c", "idle__rep01__01c"]
    setter_wk = sorted({s.split("__")[0] for s in setters})
    fp_cells = [p["cell_id"] for p in ben if p["flag_05"] == "true"]
    folds = []
    fold_names = sorted({p["fold"] for p in preds})
    for fn in fold_names:
        folds.append({"name": fn, "held_out": fn.split("/", 1)[1], "n_train_cells": 20, "n_benign_train_cells": 10,
                      "threshold_05": 0.55, "threshold_01": 0.72, "setters_05": setters, "d_used": dused,
                      "n_setter_cells": len(setters), "n_setter_workloads": len(setter_wk),
                      "importance": {n: 1.0 / dfull for n in names}})
    importance = {n: (0.5 if i == 0 else 0.5 / max(1, dfull - 1)) for i, n in enumerate(names)}
    l1 = {"features_chosen": [names[0]], "best_feature_by_fold": {fn: names[0] for fn in sorted({p["fold"] for p in preds})},
          "tpr_05": (tpr05 or 0) * 0.9 if tpr05 is not None else None, "tpr_01": (tpr01 or 0) * 0.9 if tpr01 is not None else None,
          "fpr_05_realized": fpr05, "auc": (auc or 0.5) * 0.95 if auc is not None else None, "rule": "train_auc", "direction_rule": "median_sign",
          "null": {"tpr05": _null_block(rng, (tpr05 or 0) * 0.9, n_perm=n_perm, n_assignments=n_assign)} if (null and null.get("status") == "ok") else None}
    q = {"quarantined_features": [names[0]] if quarantine else [], "n_disagree_by_feature": {names[0]: 0 if quarantine else 3}}
    sc = {"schema": "plan11.detection.scores.v1",
          "params": {"rung": rung, "split": split, "variant": variant, "normalized": variant == "norm", "grid_id": GRID,
                     "grid_source": "default: W8_H4 (no inherited selection)", "n_perm": n_perm, "seed": 20260919,
                     "fpr_declared": 0.05, "fpr_limit": 0.01, "threshold_source": "inner_lowo",
                     "threshold_quantile_method": "linear", "row_unit": "cell",
                     "score_aggregation": "not applicable: row unit is the cell", "n_estimators": 300, "min_samples_leaf": 1,
                     "class_weight": None, "train_on_at_floor": False, "null_denominator_rule": "same_as_observed",
                     "det_c1_rule": "report", "floor_source": "gates/detection/gk0_cells.csv",
                     "excluded_by_declaration": ["n_pairs", "n_windows", "dt_est_s", "iteration_count", "order_index",
                                                 "campaign", "label", "path"],
                     "S": len(S_keys), "B": len(B_keys), "class_counts": dict(Counter(c["_class"] for c in cells)),
                     "inputs_sha256": {}},
          "citation": "P3 D5; CR3 1.5; ML 1.6, 2.3, 2.4, 3.7",
          "status": "ok", "split": split, "n_folds": len(fold_names),
          "tpr_05": tpr05 if split != "lofo" else "not applicable: sandbox never held out under LOFO",
          "fpr_05_realized": fpr05,
          "tpr_01": tpr01 if split != "lofo" else "not applicable: sandbox never held out under LOFO",
          "fpr_01_realized": fpr01, "auc": auc, "tpr_at_fpr05_pooled": (tpr05 if tpr05 is not None else None),
          "pooled_label": POST_HOC_LABEL, "n_positive_scored": pos_scored, "n_positive_in_denominator": len(den),
          "n_positive_at_floor": len(at_floor), "excluded_at_floor": at_floor, "excluded_no_score": no_score,
          "n_without_score": len(no_score), "n_benign_scored": ben_scored, "per_member": per_member,
          "per_family_fpr": per_family_fpr, "per_workload_outcome": per_workload,
          "majority_accuracy": (ben_scored / (ben_scored + len(den))) if (ben_scored + len(den)) else None,
          "random_scorer": {"auc": 0.5, "tpr_equals_fpr": True}, "null": null,
          "gop": {"setter_cells": setters, "n_setter_cells": len(setters), "n_setter_workloads": len(setter_wk),
                  "setter_families": ["kernels"] if gop_few else ["kernels", "idle"], "realized_fp_cells": fp_cells,
                  "n_realized_fp_workloads": len({c.split("__")[0] for c in fp_cells})},
          "gcal": {"per_fold_tpr05": tpr05, "pooled_tpr_at_fpr05": tpr05, "difference": 0.0,
                   "null_spread": null["tpr05"]["spread"] if (null and isinstance(null.get("tpr05"), dict)) else None, "verdict": gcal_v},
          "feature_count": dfull, "feature_count_used": dused, "dim_status": GDIM_REDUCED if reduced else GDIM_FULL,
          "importance_mean": importance, "quarantine": q,
          "with_quarantine": ({"tpr_05": (tpr05 or 0) * 0.8, "fpr_05_realized": fpr05, "auc": auc,
                               "quarantined_features": [names[0]]} if quarantine else None),
          "l1": l1, "score_source": "full", "grid_source": "default: W8_H4 (no inherited selection)",
          "seed": 20260919, "n_perm": n_perm}
    if with_external:
        ext = [p for p in preds if p["class"] == "external"]
        sc["external"] = {"n_scored": len(ext), "tpr_05": sum(p["flag_05"] == "true" for p in ext) / len(ext) if ext else None,
                          "tpr_01": sum(p["flag_01"] == "true" for p in ext) / len(ext) if ext else None,
                          "per_member": {"1": {"hits": sum(p["flag_05"] == "true" for p in ext), "denominator": len(ext), "at_floor": 0,
                                               "eighths": f"{sum(p['flag_05'] == 'true' for p in ext)}/{len(ext)}"}},
                          "fold": "lowo/final"}
    if one_class:
        sc["model"] = "isolation_forest"
        sc["primary"] = True
        sc["g1c_label"] = G1C_PRIMARY
        sc["params"]["one_class_threshold_source"] = "inner_lowo"
        sc["threshold_source"] = "inner_lowo"
    write_json(d / "scores.json", sc)
    write_json(d / "null.json", {"schema": "plan11.detection.null.v1", "params": sc["params"], "citation": "CR3 2.1; ML 1.4, 2.2",
                                 "tpr05": (null.get("tpr05") if null else None), "auc": (null.get("auc") if null else None),
                                 "permuted_tpr05": list(np.round(rng.uniform(0, 0.3, size=min(n_perm, 20)), 4)),
                                 "permuted_auc": list(np.round(rng.uniform(0.3, 0.7, size=min(n_perm, 20)), 4))})
    roc = [{"fpr": 0.0, "tpr": 0.0, "threshold": 1.0}, {"fpr": 0.05, "tpr": tpr05 if tpr05 is not None else 0.0, "threshold": 0.55},
           {"fpr": 0.2, "tpr": 0.9, "threshold": 0.4}, {"fpr": 1.0, "tpr": 1.0, "threshold": 0.0}]
    write_csv(d / "roc.csv", roc, ["fpr", "tpr", "threshold"])
    write_json(d / "folds.json", {"schema": "plan11.detection.folds.v1", "params": sc["params"], "citation": "ML 3.7", "folds": folds})
    write_json(d / "l1_quarantine.json", {"schema": "plan11.detection.l1_quarantine.v1", "params": sc["params"],
                                          "citation": "CR3 2.2; ML 3.6",
                                          "features": [{"feature": names[0], "n_disagree_workloads": 0 if quarantine else 3,
                                                        "disagreeing_workloads": [] if quarantine else ["gemm", "floyd", "idle"],
                                                        "n_workloads": len(S_keys) + len(B_keys)}]})
    if quarantine:
        write_csv(d / "predictions_with_quarantine.csv", preds, cols)
    return sc


def _write_levels(rng, base: Path, rung: str, cells: list[dict], floor: dict, *, n_perm: int) -> None:
    """splits/<rung>/<gid>/level2 and level3 (SPEC_DETECTION 3.4 with al-Kindi 3 and 4)."""
    members = sorted({c["_member"] for c in cells if c["_class"] == "sandbox"})
    letters_of = {m: MEMBER_LETTER[m] for m in members}
    by_letter = {}
    for m in members:
        by_letter.setdefault(letters_of[m], []).append(m)
    letters = sorted(by_letter)
    d2 = base / "level2"
    d2.mkdir(parents=True, exist_ok=True)
    conf, mem, preds = [], [], []
    for L in letters:
        ms = by_letter[L]
        mcells = [c for c in cells if c["_member"] in ms and c["_class"] == "sandbox"]
        n_floor = sum(floor.get(c["cell_id"]) == GK0_AT_FLOOR for c in mcells)
        n_test = len(ms)
        row = {"rung": rung, "true_subfamily": L, "n_members": n_test, "n_cells": len(mcells), "n_at_floor": n_floor}
        if n_test >= 2:
            status = GN_HEADLINE if n_test >= 3 else L2_ONE_TRAIN
            den = len(mcells) - n_floor
            hits = int(round(den * 0.75))
            for L2 in letters:
                row[f"pred_{L2}"] = hits if L2 == L else (den - hits if L2 == letters[(letters.index(L) + 1) % len(letters)] else 0)
            row["recall"] = hits / den if den else ""
            nb = _null_block(rng, row["recall"] if den else 0.0, n_perm=min(n_perm, 280), n_assignments=280)
            row.update({"status": status, "null_p95": nb["p95"], "rank": nb["rank_text"], "verdict": nb["verdict"]})
        else:
            for L2 in letters:
                row[f"pred_{L2}"] = ""
            row.update({"recall": "", "status": f"{n_test} member, no held-out test" if n_test == 1 else f"{n_test} members, no held-out test",
                        "null_p95": "", "rank": "", "verdict": ""})
        conf.append(row)
        for m in ms:
            cm = [c for c in mcells if c["_member"] == m]
            nf = sum(floor.get(c["cell_id"]) == GK0_AT_FLOOR for c in cm)
            den = len(cm) - nf
            hits = int(round(den * 0.75)) if n_test >= 2 else ""
            mem.append({"rung": rung, "member_index": m, "subfamily_letter": L, "hits": hits, "denominator": den if n_test >= 2 else "",
                        "n_at_floor": nf, "eighths": (f"{hits}/{den}" if (n_test >= 2 and den) else (f"at floor ({len(cm)})" if den == 0 else ""))})
            for c in cm:
                preds.append({"cell_id": c["cell_id"], "member_index": m, "true_subfamily": L, "pred_subfamily": L, "fold": f"level2/{m}",
                              "floor_verdict": floor.get(c["cell_id"], GK0_ABOVE_FLOOR),
                              "in_denominator": str(floor.get(c["cell_id"]) != GK0_AT_FLOOR).lower()})
    cols = ["rung", "true_subfamily", "n_members", "n_cells", "n_at_floor"] + [f"pred_{L}" for L in letters] + ["recall", "status", "null_p95", "rank", "verdict"]
    write_csv(d2 / "confusion.csv", conf, cols)
    write_csv(d2 / "members.csv", mem, ["rung", "member_index", "subfamily_letter", "hits", "denominator", "n_at_floor", "eighths"])
    write_csv(d2 / "predictions.csv", preds, list(preds[0].keys()) if preds else ["cell_id"])
    head = [r for r in conf if r["status"] == GN_HEADLINE]
    write_json(d2 / "scores.json", {"schema": "plan11.detection.level2.v1", "params": {"rung": rung, "grid_id": GRID, "n_perm": n_perm,
                                                                                       "min_members_headline": 3, "min_members_test": 2,
                                                                                       "train_on_at_floor": False},
                                    "citation": "P3 0a; CR3 2.5", "macro_recall": (float(np.mean([r["recall"] for r in head])) if head else
                                                                                   "not applicable: no headline sub-family"),
                                    "n_folds": sum(r["n_members"] for r in conf if r["status"] in (GN_HEADLINE, L2_ONE_TRAIN)),
                                    "null": {L: {"n": min(n_perm, 280), "exhaustive": True} for L in letters}})
    write_json(d2 / "null.json", {"schema": "plan11.detection.level2_null.v1", "params": {"rung": rung}, "citation": "P3 0a",
                                  "per_letter": {L: list(np.round(rng.uniform(0, 0.6, size=10), 3)) for L in letters}})
    d3 = base / "level3"
    d3.mkdir(parents=True, exist_ok=True)
    conf3, mem3 = [], []
    for m in members:
        cm = [c for c in cells if c["_member"] == m and c["_class"] == "sandbox"]
        nf = sum(floor.get(c["cell_id"]) == GK0_AT_FLOOR for c in cm)
        den = len(cm) - nf
        hits = int(round(den * 0.5))
        row = {"rung": rung, "true_member": m, "n_at_floor": nf}
        for m2 in members:
            row[f"pred_{m2}"] = hits if m2 == m else (den - hits if m2 == members[(members.index(m) + 1) % len(members)] else 0)
        row.update({"recall": (hits / den) if den else "", "label": SIGNATURE_CEILING})
        conf3.append(row)
        mem3.append({"rung": rung, "member_index": m, "subfamily_letter": letters_of[m], "hits": hits, "denominator": den,
                     "n_at_floor": nf, "eighths": f"{hits}/{den}" if den else f"at floor ({len(cm)})"})
    cols3 = ["rung", "true_member", "n_at_floor"] + [f"pred_{m}" for m in members] + ["recall", "label"]
    write_csv(d3 / "confusion.csv", conf3, cols3)
    write_csv(d3 / "members.csv", mem3, ["rung", "member_index", "subfamily_letter", "hits", "denominator", "n_at_floor", "eighths"])
    acc = sum(r["hits"] for r in mem3) / max(1, sum(r["denominator"] for r in mem3))
    nb = _null_block(rng, acc, n_perm=n_perm, n_assignments=10 ** 6, statistic_name="accuracy")
    write_json(d3 / "scores.json", {"schema": "plan11.detection.level3.v1",
                                    "params": {"rung": rung, "grid_id": GRID, "n_perm": n_perm, "split": "rep_index", "null_unit": "cell",
                                               "null_unit_reason": "the label is the workload itself; a workload-level permutation is a relabel",
                                               "train_on_at_floor": False},
                                    "citation": "P3 0a", "accuracy": acc, "per_member_recall": {str(r["member_index"]): (r["hits"] / r["denominator"] if r["denominator"] else None) for r in mem3},
                                    "label": SIGNATURE_CEILING, "null": {"statistic": "accuracy", "null_unit": "cell", "n_perm": n_perm, "n_assignments": 10 ** 6, **nb},
                                    "null_unit": "cell", "n_folds": 4})
    write_json(d3 / "null.json", {"schema": "plan11.detection.level3_null.v1", "params": {"rung": rung}, "citation": "P3 0a",
                                  "permuted_accuracy": list(np.round(rng.uniform(0, 0.5, size=10), 3))})
    write_csv(d3 / "predictions.csv", [{"cell_id": c["cell_id"], "member_index": c["_member"], "pred_member": c["_member"], "fold": "level3/rep0",
                                        "floor_verdict": floor.get(c["cell_id"], GK0_ABOVE_FLOOR),
                                        "in_denominator": str(floor.get(c["cell_id"]) != GK0_AT_FLOOR).lower()}
                                       for c in cells if c["_class"] == "sandbox"],
              ["cell_id", "member_index", "pred_member", "fold", "floor_verdict", "in_denominator"])


def make_out(out: Path, *, reps: int = 4, idle: int = 4, n_pairs: int = 60, seed: int = 20260917,
             order_void_rung: str | None = None, gf_void_rung: str | None = None, gc_disconnected_rung: str | None = None,
             null_inside_rung: str | None = None, null_not_estimable_rung: str | None = None, smoke_perm: int = 500,
             at_floor_member: int | None = 3, no_score_cell: str | None = None, missing_split: tuple | None = None,
             no_selection_rung: str | None = None, one_class_inside_null: bool = False, gop_set_by_few_rung: str | None = None,
             stage2: bool = False, n_unassigned: int = 0, with_external: bool = False, no_order_index: bool = False,
             two_idle_campaigns: bool = False, quarantine_rung: str | None = None, with_leak_probe: bool = True,
             with_gv_members: bool = True, with_gates: bool = True, with_extracts: bool = True) -> Path:
    """Write the synthetic `<out>` tree; returns `out`."""
    out = Path(out)
    rng = np.random.default_rng(seed)
    cells = _cells(reps, idle, with_external=with_external, two_idle_campaigns=two_idle_campaigns, n_unassigned=n_unassigned)
    if no_order_index:
        for c in cells:
            c["_order_index"] = ""
    # ---- cells.csv, cells.pre_classes.csv, cells.index.json, inputs/classes.csv, extracts, sidecars
    root = out / "synth_root"
    for c in cells:
        fam = "kernel" if c["role"] == "kernel" else ("idle" if c["role"] == "idle" else "synthfam")
        cdir = root / fam / f"{fam}_{c['kernel']}_v2" / f"--seed_{c['seed'] or 0}" / f"rep{c['rep_dir']:03d}__{c['label']}"
        c["path"] = str(cdir)
        c["traj_file"] = f"run_matrix_test1_kernel_{c['rep']}_v2.npy.substrate_trajectory.csv"
        if not with_extracts or c["_class"] == "unassigned":
            continue
        wk = c["_workload_key"] if c["_workload_key"] in WORKLOADS else ("sandbox_member_1" if c["_class"] == "external" else "idle")
        K0, content, churn, _, _ = WORKLOADS[wk]
        if at_floor_member and c["_member"] == at_floor_member and c["_class"] == "sandbox":
            K0, content, churn = 0, "idle", 0.02
        rows = _extract_rows(rng, K0, content, churn, n_pairs)
        write_csv(out / "extract" / c["cell_id"] / "extract.csv", rows, list(EXTRACT_COLUMNS))
        Ks = [r["K"] for r in rows]
        write_json(out / "extract" / c["cell_id"] / "sidecar.json", {
            "schema": "plan11.extract.v1", "extractor_version": "0.1.0", "cell_id": c["cell_id"], "kernel": c["kernel"],
            "role": c["role"], "archetype_predicted": c["archetype_predicted"], "seed": c["seed"] or None, "rep": c["rep"],
            "rep_dir": c["rep_dir"], "label": c["label"], "campaign": c["campaign"], "path": c["path"], "traj_file": c["traj_file"],
            "source_bytes": 0, "N": N_PAGES, "page_size": 4096, "bits_per_page": 32768, "duration_s_declared": 600,
            "quantiles": [0.05, 0.25, 0.5, 0.75, 0.95], "persist_side": "t", "header_sha256": "0" * 64, "header_ncols": 66,
            "columns_used": {"seq": 0, "page_index": 1, "hamming": 2, "l0": 4, "l1": 5}, "n_rows_in": 0, "n_rows_skipped": 0,
            "n_rows_dup_page": 0, "n_rows_zero_hamming": 0, "n_rows_zero_l0": 0, "seq_first": 1, "seq_last": n_pairs,
            "n_seq_present": n_pairs, "n_pairs": n_pairs, "n_seq_gaps": 0, "gap_seqs": [], "dt_est_s": 600 / n_pairs,
            "dt_bracket_s": [0.5, 0.644], "K_median": float(np.median(Ks)), "K_max": int(max(Ks)), "apf_max": max(Ks) / N_PAGES,
            "failed_count": 0, "failed_count_source": "declared zero", "status": "ok",
            "started_at": "2026-09-17T00:00:00+00:00", "finished_at": "2026-09-17T00:00:01+00:00", "elapsed_s": 1.0})
    cells_cols = ["cell_id", "kernel", "role", "archetype_predicted", "seed", "rep", "rep_dir", "label", "campaign", "path", "traj_file", "status"]
    write_csv(out / "cells.csv", cells, cells_cols)
    write_csv(out / "cells.pre_classes.csv", cells, cells_cols)
    counts = Counter(c["_class"] for c in cells)
    write_json(out / "cells.index.json", {"schema": "plan11.cells.v1", "params": {}, "citation": "SPEC 2.7", "n_cells": len(cells),
                                          "classes_applied": {"classes_csv_sha256": "0" * 64, "counts": dict(counts)}})
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    cls_rows = [{"path_prefix": "kernel", "class": "benign_kernel", "member_index": "", "subfamily_letter": "", "rep": "", "order_index": "",
                 "family": "", "workload_key": ""},
                {"path_prefix": "idle/idle_idle_v2", "class": "idle", "member_index": "", "subfamily_letter": "", "rep": "", "order_index": "",
                 "family": "", "workload_key": ""}]
    for m in (1, 2, 3):
        cls_rows.append({"path_prefix": f"synthfam/synthfam_member_{m}", "class": "sandbox", "member_index": m, "subfamily_letter": MEMBER_LETTER[m],
                         "rep": "", "order_index": "", "family": "", "workload_key": ""})
    write_csv(out / "inputs" / "classes.csv", cls_rows, ["path_prefix", "class", "member_index", "subfamily_letter", "rep", "order_index", "family", "workload_key"])
    write_csv(out / "inputs" / "gk0_source_sandbox.csv", [{"member_index": m, "steady_state_changes_content": "unstated", "source": "unstated"} for m in (1, 2, 3)],
              ["member_index", "steady_state_changes_content", "source"])
    write_csv(out / "inputs" / "head_drop.csv", [{"kernel": wk, "head_drop_pairs": 0, "reason": "default 0; declared at D2"} for wk in WORKLOADS],
              ["kernel", "head_drop_pairs", "reason"])
    D = out / "gates" / "detection"
    D.mkdir(parents=True, exist_ok=True)
    # ---- the join, the letter sequence, the validation file
    join = []
    for c in cells:
        join.append({"cell_id": c["cell_id"], "class": c["_class"], "y": c["_y"], "workload_key": c["_workload_key"], "family": c["_family"],
                     "member_index": c["_member"], "subfamily_letter": c["_letter"], "rep": c["rep"], "campaign": c["campaign"],
                     "order_index": c["_order_index"], "order_token": _token(c) if (c["_order_index"] != "" and c["_class"] != "unassigned") else "",
                     "split_role": c["_split_role"]})
    write_csv(D / "cell_classes.csv", join, ["cell_id", "class", "y", "workload_key", "family", "member_index", "subfamily_letter", "rep",
                                              "campaign", "order_index", "order_token", "split_role"])
    S_keys = sorted({c["_workload_key"] for c in cells if c["_class"] == "sandbox"})
    B_keys = sorted({c["_workload_key"] for c in cells if c["_y"] == "benign"})
    write_json(D / "cell_classes.json", {"schema": "plan11.detection.cell_classes.v1",
                                         "params": {"classes_csv_sha256": "0" * 64, "kernel_family_rule": "tier", "relaunched_grouping": "parent",
                                                    "campaign_label": None, "campaign_token_default": "stage1"},
                                         "citation": "P3 0a; ML 1.2; CR3 1.8, 2.31",
                                         "counts_per_class": dict(counts), "counts_per_family": dict(Counter(c["_family"] for c in cells if c["_family"])),
                                         "counts_per_member": {str(m): sum(1 for c in cells if c["_member"] == m and c["_class"] == "sandbox") for m in (1, 2, 3)},
                                         "counts_per_subfamily": {"A": 2, "B": 1}, "S": len(S_keys), "B": len(B_keys),
                                         "n_assignments": math.comb(len(S_keys) + len(B_keys), len(S_keys))})
    classed = [c for c in cells if c["_class"] != "unassigned"]
    if no_order_index:
        line = f"not run: order_index missing for {len(classed)} cells"
        (D / "letter_sequence.txt").write_text(line + "\n", encoding="utf-8")
        (D / "letter_sequence.csv").write_text(line + "\n", encoding="utf-8")
    else:
        ordered = sorted(classed, key=lambda c: int(c["_order_index"]))
        (D / "letter_sequence.txt").write_text(" ".join(_token(c) for c in ordered) + "\n", encoding="utf-8")
        write_csv(D / "letter_sequence.csv", [{"order_index": c["_order_index"], "token": _token(c), "class": c["_class"]} for c in ordered],
                  ["order_index", "token", "class"])
    write_json(D / "classes_validation.json", {"schema": "plan11.detection.classes_validation.v1", "params": {"relaunched_grouping": "parent"},
                                               "citation": "P3 0a; P3 D4; ML 3.2; CR3 1.8, 2.31", "status": "ok", "refusals": [],
                                               "unmatched_rows": 0, "unassigned_cells": n_unassigned, "counts": dict(counts),
                                               "members": {"1": "A", "2": "A", "3": "B"}, "subfamilies": {"A": [1, 2], "B": [3]},
                                               "warnings": ["campaign token defaulted to stage1 for 12 rows"]})
    write_json(D / "inherit_selection.json", {"schema": "plan11.detection.inherit_selection.v1", "params": {"grid_source": "default: W8_H4 (no inherited selection)"},
                                              "citation": "SPEC_DETECTION 2.7; al-Farabi M10", "status": "ok", "refusal": ""})
    if not with_gates:
        return out
    G = out / "gates"
    # ---- the plan11 gates run unchanged inside <out>
    pre = []
    for c in cells:
        if c["_class"] == "unassigned":
            continue
        c1 = "pass" if c["role"] == "kernel" else ("not applicable: control (C1 re-mapped)" if c["role"] == "idle" else
                                                   ("fail" if (at_floor_member and c["_member"] == at_floor_member and c["_class"] == "sandbox") else "pass"))
        pre.append({"cell_id": c["cell_id"], "role": c["role"], "C1": c1, "C1_apf_max": 0.03, "C2": "pass", "C2_n_pairs": n_pairs, "C3": "pass",
                    "C3_n_windows_8_4": (n_pairs - 1 - 8) // 4 + 1, "C4": "not applicable: no settle record in the retention layout",
                    "C5": "not run: producer.log is not in the trajectory; the author reads the campaign log", "C6": "pass", "C6_reason": "n_seq_gaps=0",
                    "C7": "pending: filled by the temporal gates (move 6)", "C8": "not applicable: change-point view not in this paper (decided 2026-09-16)",
                    "failed_count": 0, "failed_verdict": "pass", "failed_source": "declared zero: AA A5",
                    "all_hard_pass": "false" if c1 == "fail" else "true"})
    write_csv(G / "preconditions.csv", pre, list(pre[0].keys()))
    write_json(G / "preconditions.json", {"schema": "plan11.preconditions.v1", "params": {"C1_ACTIVITY_MIN": 0.02}, "citation": "SPEC 3.3",
                                          "excluded_cells": [], "excluded_cells_pair_rungs": []})
    sel = {"schema": "plan11.selection.v1", "params": {"grid_source": "default: W8_H4 (no inherited selection)", "inherited_from": None},
           "citation": "SPEC_DETECTION 2.7"}
    for rung in RUNGS:
        if rung != no_selection_rung:
            sel[rung] = {"grid_id": GRID, "W": 8, "H": 4, "passes_acceptance": False, "selected_by": "default: no inherited selection",
                         "gates_passed": [], "refusal": ""}
    write_json(G / "selection.json", sel)
    gc = []
    for rung in ("apf", "persist", "wapf", "content", "combined"):
        for r in range(reps):
            gc.append({"rung": rung, "kernel_or_triple": "gemm", "rep": r, "n_events": 2, "first_event_seq": 25, "j_at_event": 0.5, "stat_a": 1.9,
                       "stat_b": "", "stat_c": "", "verdict": "pass"})
        gc.append({"rung": rung, "kernel_or_triple": "all", "rep": "all", "n_events": "", "first_event_seq": "", "j_at_event": "", "stat_a": "",
                   "stat_b": "", "stat_c": "", "verdict": GC_DISCONNECTED if rung == gc_disconnected_rung else "pass"})
    write_csv(G / "gc.csv", gc, list(gc[0].keys()))
    gf = []
    for rung in RUNGS:
        gf.append({"rung": rung, "grid_id": GRID, "part": "i", "kernel": "idle", "n_cells": idle, "score": 0.12, "null_p95": 0.2, "n_inside_envelope": "",
                   "envelope_lo": "", "envelope_hi": "", "verdict": GF_VOID if rung == gf_void_rung else GF_INSEPARABLE})
        for k in ("gemm", "floyd"):
            gf.append({"rung": rung, "grid_id": GRID, "part": "ii", "kernel": k, "n_cells": reps, "score": "", "null_p95": "", "n_inside_envelope": 0,
                       "envelope_lo": 0.0005, "envelope_hi": 0.0007, "verdict": "pass"})
    write_csv(G / "gf.csv", gf, list(gf[0].keys()))
    gk0 = [{"kernel": k, "source_statement": "yes", "n_cells": reps, "tail_median_K_median": WORKLOADS[k][0] + 150, "tail_median_K_min": WORKLOADS[k][0] + 140,
            "tail_median_K_max": WORKLOADS[k][0] + 160, "idle_band_edge": 165, "verdict": GK0_ABOVE_FLOOR, "archetype_measured": "WORKING-SET"} for k in ("gemm", "floyd")]
    gk0.append({"kernel": "idle", "source_statement": "no", "n_cells": idle, "tail_median_K_median": 150, "tail_median_K_min": 145, "tail_median_K_max": 155,
                "idle_band_edge": 165, "verdict": "control", "archetype_measured": "IDLE"})
    write_csv(G / "gk0.csv", gk0, list(gk0[0].keys()))
    write_json(G / "gk0.json", {"schema": "plan11.gk0.v1", "params": {"idle_pool": "pooled_snapshots", "idle_percentile": 95.0, "tail_fraction": 0.8},
                                "citation": "SPEC 3.3.3", "idle_band_edge": 165.0})
    write_csv(G / "gp.csv", [{"kernel": k, "cell_id": f"{k}__rep00__01c", "n_pairs": n_pairs, "passes_per_600s": "", "source_kind": "undeclared",
                              "verdict_pairs": "undeclared", "rhythm_verdict": "undeclared"} for k in ("gemm", "floyd")],
              ["kernel", "cell_id", "n_pairs", "passes_per_600s", "source_kind", "verdict_pairs", "rhythm_verdict"])
    write_csv(G / "gj.csv", [{"kernel": k, "cell_id": f"{k}__rep00__01c", "n_pairs_J": n_pairs - 1, "floor_median_K": 150, "k_threshold": 450,
                              "frac_interpretable": 1.0, "mask_verdict": "interpretable"} for k in ("gemm", "floyd")],
              ["kernel", "cell_id", "n_pairs_J", "floor_median_K", "k_threshold", "frac_interpretable", "mask_verdict"])
    write_json(G / "gj.json", {"schema": "plan11.gj.v1", "params": {"k_factor": 3.0, "n_idle_cells": idle}, "citation": "SPEC 3.6.1",
                               "floor_median_K": 150, "floor_median_n_persist": 120,
                               "idle_J": {"quantiles": [0.05, 0.25, 0.5, 0.75, 0.95], "J": [0.6, 0.8, 0.9, 0.95, 0.99], "n_pairs": idle * (n_pairs - 1), "mean": 0.88},
                               "mask_files": str(G / "gj_mask"), "mask_fields": ["mask_K", "mask_persist"]})
    write_csv(G / "gx.csv", [{"rung": r, "grid_id": GRID, "score": 0.4, "null_p95": 0.55, "rank": "rank 300 of 500", "leak_verdict": GX_POOLING_STANDS,
                              "confound_verdict": "confound: none", "headline_mark": "", "n_labels": 2} for r in RUNGS],
              ["rung", "grid_id", "score", "null_p95", "rank", "leak_verdict", "confound_verdict", "headline_mark", "n_labels"])
    # ---- admissibility, G-K0 two-class, G-N
    adm = []
    for c in cells:
        if c["_class"] == "unassigned":
            adm.append({"cell_id": c["cell_id"], "class": "unassigned", "status_cells_csv": c["status"], "all_hard_pass": "", "C1": "", "C2": "", "C6": "",
                        "failed_verdict": "", "c1_rule_applied": "report", "admissible": "false", "admissible_pair_rungs": "false",
                        "reason": "unassigned: no class row in inputs/classes.csv"})
            continue
        prow = [p for p in pre if p["cell_id"] == c["cell_id"]][0]
        reported = prow["C1"] == "fail"
        adm.append({"cell_id": c["cell_id"], "class": c["_class"], "status_cells_csv": "ok", "all_hard_pass": prow["all_hard_pass"], "C1": prow["C1"],
                    "C2": "pass", "C6": "pass", "failed_verdict": "pass", "c1_rule_applied": "report", "admissible": "true", "admissible_pair_rungs": "true",
                    "reason": "C1 fail reported for every class: floor verdict by G-K0 (K3 F3; al-Kindi review 5)" if reported else ""})
    write_csv(D / "admissibility.csv", adm, list(adm[0].keys()))
    write_json(D / "admissibility.json", {"schema": "plan11.detection.admissibility.v1", "params": {"det_c1_rule": "report"}, "citation": "CR3 2.23; K3 move 2; SPEC 3.3.1",
                                          "n_admissible": sum(a["admissible"] == "true" for a in adm), "n_unassigned": n_unassigned})
    floor = {}
    gk0c = []
    for c in cells:
        if c["_class"] == "unassigned":
            continue
        is_floor = bool(at_floor_member and c["_member"] == at_floor_member and c["_class"] == "sandbox")
        v = "control" if c["_class"] in ("idle", "harness_idle") else (GK0_AT_FLOOR if is_floor else GK0_ABOVE_FLOOR)
        floor[c["cell_id"]] = v
        K0 = 150 if (is_floor or c["_class"] == "idle") else WORKLOADS.get(c["_workload_key"], WORKLOADS["sandbox_member_1"])[0] + 150
        gk0c.append({"cell_id": c["cell_id"], "class": c["_class"], "workload_key": c["_workload_key"], "member_index": c["_member"], "K_med": K0,
                     "K_q90": int(K0 * 1.05), "frac_above_band": 0.0 if K0 <= 165 else 1.0, "l0_med_above": "" if K0 <= 165 else 300.0,
                     "J_consec_above": "" if K0 <= 165 else 0.8, "idle_band_edge": 165.0, "env_K_med": 160.0, "env_K_q90": 168.0, "env_frac": 0.05,
                     "harness_env_K_med": "", "harness_env_K_q90": "", "harness_env_frac": "", "verdict": v})
    write_csv(D / "gk0_cells.csv", gk0c, list(gk0c[0].keys()))
    gk0m = []
    for m in (1, 2, 3):
        mc = [g for g in gk0c if g["member_index"] == m and g["class"] == "sandbox"]
        nf = sum(g["verdict"] == GK0_AT_FLOOR for g in mc)
        gk0m.append({"member_index": m, "subfamily_letter": MEMBER_LETTER[m], "source_statement": "unstated", "n_cells": len(mc), "n_at_floor": nf,
                     "n_at_harness_floor": 0, "n_above_floor": len(mc) - nf, "verdict": GK0_AT_FLOOR if nf == len(mc) else (GK0_ABOVE_FLOOR if nf == 0 else "cells in more than one verdict")})
    write_csv(D / "gk0_members.csv", gk0m, list(gk0m[0].keys()))
    write_json(D / "gk0.json", {"schema": "plan11.detection.gk0.v1", "params": {"idle_percentile": 95.0, "envelope_percentile": 95.0,
                                                                                "verdict_quantities": ["K_med", "K_q90", "frac_above_band"]},
                                "citation": "CR3 2.8; K3 F3; ML 3.6", "idle_band_edge": 165.0,
                                "idle_envelope": {"K_med": 160.0, "K_q90": 168.0, "frac_above_band": 0.05}, "harness_envelope": None, "status": "ok"})
    n_above = sum(1 for g in gk0m if g["n_above_floor"] > 0)
    gn = [{"row": "sandbox", "class": "sandbox", "n_workloads": n_above, "n_cells": sum(g["n_cells"] for g in gk0m),
           "workloads": ";".join(f"sandbox_member_{g['member_index']}" for g in gk0m if g["n_above_floor"] > 0),
           "status": GN_HEADLINE if n_above >= 3 else (GN_ONE_TRAIN_WORKLOAD if n_above == 2 else "no supervised headline")},
          {"row": "kernels", "class": "benign_kernel", "n_workloads": 2, "n_cells": 2 * reps, "workloads": "gemm;floyd", "status": GN_HEADLINE},
          {"row": "idle", "class": "idle", "n_workloads": 1, "n_cells": idle, "workloads": "idle", "status": GN_SINGLE_WORKLOAD}]
    write_csv(D / "gn.csv", gn, list(gn[0].keys()))
    # ---- the splits per rung
    sc_by = {}
    for rung in RUNGS:
        base = D / "splits" / rung / GRID
        variants = ("raw", "norm") if rung == "apf" else ("norm",)
        for variant in variants:
            for split in ("lowo", "loco", "lofo"):
                if missing_split and (rung, split) == tuple(missing_split):
                    continue
                sc_by[(rung, split, variant)] = _write_split(
                    rng, base / f"{split}__{variant}", rung, split, variant, cells, floor, n_perm=smoke_perm,
                    null_inside=(rung == null_inside_rung and split == "lowo"), null_not_estimable=(rung == null_not_estimable_rung),
                    no_score_cell=no_score_cell if split == "lowo" else None, gop_few=(rung == gop_set_by_few_rung),
                    quarantine=(rung == quarantine_rung and split == "lowo" and variant == "norm"), with_external=with_external)
        if not (missing_split and (rung, "one_class") == tuple(missing_split)):
            sc_by[(rung, "one_class", "norm")] = _write_split(rng, base / "one_class__norm", rung, "one_class", "norm", cells, floor, n_perm=smoke_perm,
                                                              one_class=True, oc_inside=one_class_inside_null, with_external=with_external)
        if rung == "combined":
            sc_by[(rung, "lowo_matched", "norm")] = _write_split(rng, base / "lowo_matched__norm", rung, "lowo", "norm", cells, floor, n_perm=smoke_perm, reduced=True)
        if rung == "apf":
            for i in range(5):
                _write_split(rng, base / f"lowo_seed{i}__norm", rung, "lowo", "norm", cells, floor, n_perm=0, quality=0.9 + 0.01 * i)
        _write_levels(rng, base, rung, cells, floor, n_perm=smoke_perm)
    # ---- the ladder
    lad = []
    for rung in RUNGS:
        for reading in ("from_pair1", "from_boundary"):
            for T in (30, 60, 120, 300, 600):
                n_rows = min(n_pairs - 0, int(round(T / 0.644)))
                if reading == "from_boundary":
                    lad.append({"rung": rung, "grid_id": GRID, "reading": reading, "prefix_s": T, "dt": 0.644, "n_rows_median": "", "n_windows_median": "",
                                "n_cells_with_window": "", "tpr_05": "", "fpr_05_realized": "", "auc": "", "null_verdict": "", "note": LADDER_FROM_PAIR1_ONLY})
                    continue
                nw = max(0, (n_rows - 8) // 4 + 1)
                if nw == 0:
                    lad.append({"rung": rung, "grid_id": GRID, "reading": reading, "prefix_s": T, "dt": 0.644, "n_rows_median": n_rows, "n_windows_median": 0,
                                "n_cells_with_window": 0, "tpr_05": f"not applicable: prefix shorter than one window (n = {n_rows} < W = 8)",
                                "fpr_05_realized": "", "auc": "", "null_verdict": "", "note": ""})
                    continue
                sc = sc_by.get((rung, "lowo", "norm"))
                tp = sc["tpr_05"] if (sc and T == 600 and isinstance(sc["tpr_05"], float)) else round(float(min(1.0, 0.3 + T / 800)), 3)
                lad.append({"rung": rung, "grid_id": GRID, "reading": reading, "prefix_s": T, "dt": 0.644, "n_rows_median": n_rows, "n_windows_median": nw,
                            "n_cells_with_window": len([c for c in cells if c["_class"] != "unassigned"]), "tpr_05": tp, "fpr_05_realized": 0.05, "auc": 0.9,
                            "null_verdict": "not run: ladder null not requested (--ladder-null-perm 0)", "note": ""})
    write_csv(D / "ladder.csv", lad, ["rung", "grid_id", "reading", "prefix_s", "dt", "n_rows_median", "n_windows_median", "n_cells_with_window", "tpr_05",
                                       "fpr_05_realized", "auc", "null_verdict", "note"])
    write_json(D / "ladder.json", {"schema": "plan11.detection.ladder.v1",
                                   "params": {"prefixes_s": [30, 60, 120, 300, 600], "dt": 0.644, "readings": ["from_pair1", "from_boundary"], "norm": "prefix",
                                              "n_perm": 0, "ladder_head_drop_rule": "from_pair1_only"},
                                   "citation": "K3 move 17; CR3 2.29; N3 Sec. 1 RQ1",
                                   "per_cell": {c["cell_id"]: {"pairs_at_0500": 60, "pairs_at_0644": 47, "pairs_per_cell": 3} for c in cells if c["_class"] != "unassigned"}})
    # ---- the gate CSVs of 3.5
    gl = []
    for rung in RUNGS:
        sc = sc_by.get((rung, "lowo", "norm"))
        raw = sc_by.get((rung, "lowo", "raw"))
        if sc is None:
            gl.append({"rung": rung, "grid_id": GRID, "tpr05_norm": "", "null_p95_norm": "", "rank_norm": "", "tpr05_raw": "", "verdict": "not run: lowo scores.json missing"})
            continue
        nv = sc["null"]["tpr05"]["verdict"] if sc.get("null") else "not run: null not run"
        v = PASS if nv == PASS else ("level only" if nv == NULL_INSIDE else nv)
        gl.append({"rung": rung, "grid_id": GRID, "tpr05_norm": sc["tpr_05"], "null_p95_norm": sc["null"]["tpr05"]["p95"] if sc.get("null") else "",
                   "rank_norm": sc["null"]["tpr05"]["rank_text"] if sc.get("null") else "", "tpr05_raw": raw["tpr_05"] if raw else "", "verdict": v})
    write_csv(D / "gl.csv", gl, ["rung", "grid_id", "tpr05_norm", "null_p95_norm", "rank_norm", "tpr05_raw", "verdict"])
    gop = []
    for rung in RUNGS:
        for split, variant in (("lowo", "norm"), ("loco", "norm"), ("lofo", "norm"), ("lowo", "raw")):
            sc = sc_by.get((rung, split, variant))
            if sc is None:
                continue
            g = sc["gop"]
            few = (rung == gop_set_by_few_rung and split == "lowo") or split == "lofo"
            gop.append({"rung": rung, "split": split, "variant": variant, "fpr_declared": 0.05, "threshold_in_fold": "true", "n_setter_cells": g["n_setter_cells"],
                        "n_setter_workloads": g["n_setter_workloads"], "setter_families": ";".join(g["setter_families"]),
                        "n_realized_fp_cells": len(g["realized_fp_cells"]), "n_realized_fp_workloads": g["n_realized_fp_workloads"], "fp_families": "kernels",
                        "n_folds_unsupported": sc["n_folds"] if few else 0, "cells_rule": "threshold_setters",
                        "verdict": GOP_SET_BY_FEW if few else GOP_SUPPORTED})
    write_csv(D / "gop.csv", gop, list(gop[0].keys()))
    glm = []
    for rung in RUNGS:
        for m in (1, 2, 3):
            L = WORKLOADS[f"sandbox_member_{m}"][0] + 150
            at_floor = bool(at_floor_member == m)
            v = ("not applicable: at floor (G-K0)" if at_floor else (GLM_SURVIVES if m in (1, 2) else GLM_NOT_DETECTED))
            band = [k for k in ("gemm", "floyd") if L / 2 <= WORKLOADS[k][0] + 150 <= L * 2]
            glm.append({"rung": rung, "grid_id": GRID, "member_index": m, "subfamily_letter": MEMBER_LETTER[m], "level_quantity": "median_K",
                        "level_label": GLM_LABEL_MEDIAN_K, "level": "" if at_floor else L, "band_lo": "" if at_floor else L / 2, "band_hi": "" if at_floor else L * 2,
                        "band_workloads": "" if at_floor else ";".join(band), "n_band_cells": "" if at_floor else reps * len(band),
                        "n_band_workloads": "" if at_floor else len(band), "recall_unrestricted": "" if at_floor else (1.0 if m in (1, 2) else 0.0),
                        "fpr_unrestricted": "" if at_floor else 0.05, "recall_lm": "" if at_floor else (0.9 if m in (1, 2) else 0.0),
                        "fpr_lm": "" if at_floor else 0.06, "threshold_lm_median": "" if at_floor else 0.6, "verdict": v})
    write_csv(D / "glm.csv", glm, list(glm[0].keys()))
    ganchor = []
    for rung in RUNGS:
        ganchor.append({"rung": rung, "part": "kernels", "label_space": "campaign", "n_cells": 2 * reps, "n_labels": 2, "score": 0.4, "null_p95": 0.55,
                        "rank": "rank 300 of 500", "verdict": GANCHOR_NOT_AUDIBLE})
        if two_idle_campaigns:
            ganchor.append({"rung": rung, "part": "idle_sets", "label_space": "campaign", "n_cells": idle, "n_labels": 2, "score": 0.95, "null_p95": 0.7,
                            "rank": "rank 1 of 500", "verdict": GANCHOR_AUDIBLE})
        else:
            ganchor.append({"rung": rung, "part": "idle_sets", "label_space": "campaign", "n_cells": idle, "n_labels": 1, "score": "", "null_p95": "", "rank": "",
                            "verdict": f"not applicable: one idle campaign (n = {idle} cells)"})
        ganchor.append({"rung": rung, "part": "idle_early_late", "label_space": "half", "n_cells": idle, "n_labels": 2,
                        "score": "" if no_order_index else 0.5, "null_p95": "" if no_order_index else 0.8, "rank": "" if no_order_index else "rank 250 of 500",
                        "verdict": "not run: order_index missing for idle" if no_order_index else ORDER_NOT_AUDIBLE})
    write_csv(D / "ganchor.csv", ganchor, list(ganchor[0].keys()))
    order = []
    for rung in RUNGS:
        for cls, nw in (("sandbox", 3), ("benign_kernel", 2)):
            for half_rule in ("within_workload", "within_class"):
                if no_order_index:
                    order.append({"rung": rung, "class": cls, "n_cells": "", "n_workloads": "", "half_rule": half_rule, "order_null_unit": "", "score": "",
                                  "null_p95": "", "rank": "", "n_assignments": "", "verdict": "not run: order_index missing", "consequence": "size"})
                    continue
                void = rung == order_void_rung
                audible = void or (half_rule == "within_class" and cls == "sandbox")
                order.append({"rung": rung, "class": cls, "n_cells": nw * reps, "n_workloads": nw, "half_rule": half_rule,
                              "order_null_unit": "workload" if half_rule == "within_class" else "cell", "score": 0.9 if audible else 0.5,
                              "null_p95": 0.7, "rank": "rank 1 of 500" if audible else "rank 260 of 500", "n_assignments": 20,
                              "verdict": ORDER_VOID if void else (ORDER_AUDIBLE if audible else ORDER_NOT_AUDIBLE),
                              "consequence": "void" if void else "size"})
    write_csv(D / "order.csv", order, list(order[0].keys()))
    drift = []
    for cls in ("sandbox", "benign_kernel", "idle"):
        for unit in ("within_workload", "confounded with workload order (blocked)"):
            label = "" if unit == "within_workload" else "confounded with workload order (blocked)"
            if no_order_index:
                drift.append({"class": cls, "unit": unit, "n": "", "n_workloads": "", "slope": "", "intercept": "", "r2": "", "null_p95_abs_slope": "", "rank": "",
                              "verdict": "not run: order_index missing", "label": label})
            else:
                drift.append({"class": cls, "unit": unit, "n": 12, "n_workloads": 3, "slope": 0.01, "intercept": 100.0, "r2": 0.02, "null_p95_abs_slope": 2.0,
                              "rank": "rank 300 of 500", "verdict": DRIFT_NONE, "label": label})
    write_csv(D / "drift.csv", drift, list(drift[0].keys()))
    gsig = []
    for rung in RUNGS:
        lo, lw = sc_by.get((rung, "loco", "norm")), sc_by.get((rung, "lowo", "norm"))
        if lo is None or lw is None:
            gsig.append({"rung": rung, "tpr05_loco": "", "tpr05_lowo": "", "gap_tpr05": "", "auc_loco": "", "auc_lowo": "", "gap_auc": "", "loco_null_verdict": "",
                         "lowo_null_verdict": "", "verdict": "not run: loco or lowo scores.json missing"})
            continue
        lv, wv = lo["null"]["tpr05"]["verdict"], lw["null"]["tpr05"]["verdict"]
        v = GSIG_IDENTITY if (lv == PASS and wv == NULL_INSIDE) else GSIG_REPORTED
        gsig.append({"rung": rung, "tpr05_loco": lo["tpr_05"], "tpr05_lowo": lw["tpr_05"], "gap_tpr05": (lo["tpr_05"] or 0) - (lw["tpr_05"] or 0),
                     "auc_loco": lo["auc"], "auc_lowo": lw["auc"], "gap_auc": (lo["auc"] or 0) - (lw["auc"] or 0), "loco_null_verdict": lv, "lowo_null_verdict": wv,
                     "verdict": v})
    write_csv(D / "gsig.csv", gsig, list(gsig[0].keys()))
    gfp = []
    for rung in RUNGS:
        sc = sc_by.get((rung, "lowo", "norm"))
        for fam, st in (sc["per_family_fpr"].items() if sc else ()):
            insep = st["fpr"] >= 0.5
            gfp.append({"rung": rung, "family": fam, "n_cells": st["n"], "n_flagged_05": st["flagged"], "fraction": st["fpr"],
                        "verdict": GFP_INSEPARABLE if insep else GFP_ATTRIBUTED, "tpr05_with": sc["tpr_05"], "fpr05_with": sc["fpr_05_realized"],
                        "tpr05_without": sc["tpr_05"] if insep else "", "fpr05_without": 0.02 if insep else ""})
    write_csv(D / "gfp.csv", gfp, ["rung", "family", "n_cells", "n_flagged_05", "fraction", "verdict", "tpr05_with", "fpr05_with", "tpr05_without", "fpr05_without"])
    g1c = []
    for rung in RUNGS:
        sc = sc_by.get((rung, "one_class", "norm"))
        if sc is None:
            g1c.append({"rung": rung, "model": "", "primary": "", "tpr05": "", "fpr05_realized": "", "tpr01": "", "auc": "", "threshold_source": "", "label": "",
                        "null_verdict": "", "verdict": "not run: one_class scores.json missing"})
            continue
        g1c.append({"rung": rung, "model": "isolation_forest", "primary": "true", "tpr05": sc["tpr_05"], "fpr05_realized": sc["fpr_05_realized"], "tpr01": sc["tpr_01"],
                    "auc": sc["auc"], "threshold_source": "inner_lowo", "label": G1C_PRIMARY, "null_verdict": sc["null"]["tpr05"]["verdict"] if sc.get("null") else "",
                    "verdict": G1C_PRIMARY})
    write_csv(D / "g1c.csv", g1c, list(g1c[0].keys()))
    harness = []
    for rung in RUNGS:
        if not stage2:
            harness.append({"rung": rung, "block": "features", "feature": "", "margin_sb": "", "margin_hi": "", "margin_rp": "", "relaunched_flagged_fraction": "",
                            "median_member_recall": "", "verdict": HARNESS_STAGE2_ABSENT})
            continue
        for f in FEATURE_NAMES[rung][:3]:
            harness.append({"rung": rung, "block": "features", "feature": f, "margin_sb": 0.8, "margin_hi": 0.1, "margin_rp": 0.1, "relaunched_flagged_fraction": "",
                            "median_member_recall": "", "verdict": HARNESS_CLASS_EXCEEDS})
        harness.append({"rung": rung, "block": "relaunch", "feature": "", "margin_sb": "", "margin_hi": "", "margin_rp": "", "relaunched_flagged_fraction": 0.1,
                        "median_member_recall": 0.9, "verdict": "pass"})
    write_csv(D / "harness.csv", harness, list(harness[0].keys()))
    gcal = []
    for rung in RUNGS:
        sc = sc_by.get((rung, "lowo", "norm"))
        g = sc["gcal"] if sc else {"per_fold_tpr05": "", "pooled_tpr_at_fpr05": "", "difference": "", "null_spread": "", "verdict": "not run: lowo scores.json missing"}
        gcal.append({"rung": rung, **g})
    write_csv(D / "gcal.csv", gcal, ["rung", "per_fold_tpr05", "pooled_tpr_at_fpr05", "difference", "null_spread", "verdict"])
    disp = ["apf raw", "apf", "wapf", "persist", "content", "combined", "combined (matched)"]
    gm = []
    for a in disp:
        for b in disp:
            if a == b:
                continue
            gm.append({"rung_a": a, "rung_b": b, "improving": 2, "worsening": 1, "ties": 3, "p_exact": 0.5, "tpr05_a": 0.9, "tpr05_b": 0.8, "diff": 0.1, "spread": 0.05,
                       "verdict": GM_DIFFERENCE})
    gm.append({"rung_a": "comparator", "rung_b": "combined", "improving": "", "worsening": "", "ties": "", "p_exact": "", "tpr05_a": "", "tpr05_b": "", "diff": "", "spread": "",
               "verdict": "not run: comparator row from another session (RQ5)"})
    write_csv(D / "gm.csv", gm, list(gm[0].keys()))
    gdim = [{"rung": rung, "row": rung, "grid_id": GRID, "d": len(FEATURE_NAMES[rung]), "d_used": len(FEATURE_NAMES[rung]), "d_matched": "", "method": "train_importance", "status": GDIM_FULL} for rung in RUNGS]
    gdim.insert(0, {"rung": "apf", "row": "apf raw", "grid_id": GRID, "d": 8, "d_used": 8, "d_matched": "", "method": "train_importance", "status": GDIM_FULL})
    gdim.append({"rung": "combined", "row": "combined (matched)", "grid_id": GRID, "d": 60, "d_used": 8, "d_matched": 8, "method": "train_importance", "status": GDIM_REDUCED})
    write_csv(D / "gdim.csv", gdim, list(gdim[0].keys()))
    alias = []
    for rung in RUNGS:
        for f in FEATURE_NAMES[rung][:2]:
            alias.append({"rung": rung, "feature": f, "regressor": "dt_est_s", "alias_unit": "within_workload", "slope": 0.0, "intercept": 0.1, "r2": 0.01, "n": 20,
                          "dt_spread_s": 0.0, "verdict": ALIAS_STAYS})
            alias.append({"rung": rung, "feature": f, "regressor": "iteration_count", "alias_unit": "within_workload", "slope": "", "intercept": "", "r2": "", "n": "",
                          "dt_spread_s": "", "verdict": "not run: no iteration count (stage 1)"})
        alias.append({"rung": rung, "feature": "cadence_as_class", "regressor": "dt_est_s", "alias_unit": "lowo", "slope": "", "intercept": "", "r2": "", "n": 20,
                      "dt_spread_s": 0.0, "verdict": LEAK_NOT_AUDIBLE})
    write_csv(D / "alias.csv", alias, list(alias[0].keys()))
    gv, gvs, gvm = [], [], []
    for rung in RUNGS:
        for i, f in enumerate(FEATURE_NAMES[rung]):
            gv.append({"rung": rung, "feature": f, "L0": 0.01, "L2": 0.05, "L3": 0.5 if i % 2 else 0.01, "L0_b": 0.01, "L2_b": 0.05, "L3_families": 0.1,
                       "L3_over_L3_families": 5.0 if i % 2 else 0.1})
        n = len(FEATURE_NAMES[rung])
        k = sum(1 for r in gv if r["rung"] == rung and r["L3"] <= r["L3_families"])
        gvs.append({"rung": rung, "n_features": n, "n_features_L3_le_L3_families": k,
                    "note": "a class whose L3 is inside the benign families' mutual spread has no more form than any two families have between them" if k >= n / 2 else ""})
        for m in (1, 2, 3):
            ratio = 0.05 if m == 3 else 1.2
            gvm.append({"rung": rung, "member_index": m, "L0_member": 0.01 * ratio, "L0_ratio": ratio,
                        "note": "reps near identical: LOCO reads as within-trace" if ratio < 0.1 else ""})
    write_csv(D / "gv_two_class.csv", gv, list(gv[0].keys()))
    write_csv(D / "gv_two_class_summary.csv", gvs, list(gvs[0].keys()))
    if with_gv_members:
        write_csv(D / "gv_two_class_members.csv", gvm, list(gvm[0].keys()))
    miss, fp = [], []
    for rung in RUNGS:
        sc = sc_by.get((rung, "lowo", "norm"))
        preds = []
        p = D / "splits" / rung / GRID / "lowo__norm" / "predictions.csv"
        if p.exists():
            import csv as _csv
            with open(p, newline="") as fh:
                preds = list(_csv.DictReader(fh))
        for r in preds:
            if r["y"] == "sandbox" and r["class"] != "external":
                if r["floor_verdict"] == GK0_AT_FLOOR:
                    miss.append({"cell_id": r["cell_id"], "member_index": r["member_index"], "subfamily_letter": r["subfamily_letter"], "rung": rung, "score": r["score"],
                                 "threshold_05": r["threshold_05"], "nearest_workload": "", "nearest_family": "", "distance": "", "axis": "", "axis_of_largest": "",
                                 "d_amount": "", "d_identity": "", "amount_cell": "", "identity_cell": "", "amount_centroid": "", "identity_centroid": "",
                                 "status": AT_FLOOR_NOT_A_MISS})
                elif r["flag_05"] == "false" and r["in_denominator"] == "true":
                    miss.append({"cell_id": r["cell_id"], "member_index": r["member_index"], "subfamily_letter": r["subfamily_letter"], "rung": rung, "score": r["score"],
                                 "threshold_05": r["threshold_05"], "nearest_workload": "gemm", "nearest_family": "kernels", "distance": 0.4, "axis": "amount",
                                 "axis_of_largest": "identity", "d_amount": 0.1, "d_identity": 0.39, "amount_cell": 0.12, "identity_cell": 0.2, "amount_centroid": 0.125,
                                 "identity_centroid": 0.55, "status": "missed"})
            elif r["y"] == "benign" and r["flag_05"] == "true":
                fp.append({"cell_id": r["cell_id"], "family": r["family"], "workload_key": r["workload_key"], "rung": rung, "score": r["score"], "threshold_05": r["threshold_05"],
                           "nearest_member_index": 1, "nearest_subfamily_letter": "A", "distance": 0.3, "axis": "amount", "axis_of_largest": "identity", "d_amount": 0.05,
                           "d_identity": 0.3, "amount_cell": 0.02, "identity_cell": 0.5})
    miss_cols = ["cell_id", "member_index", "subfamily_letter", "rung", "score", "threshold_05", "nearest_workload", "nearest_family", "distance", "axis", "axis_of_largest",
                 "d_amount", "d_identity", "amount_cell", "identity_cell", "amount_centroid", "identity_centroid", "status"]
    fp_cols = ["cell_id", "family", "workload_key", "rung", "score", "threshold_05", "nearest_member_index", "nearest_subfamily_letter", "distance", "axis", "axis_of_largest",
               "d_amount", "d_identity", "amount_cell", "identity_cell"]
    write_csv(D / "miss_table.csv", miss, miss_cols)
    write_csv(D / "fp_table.csv", fp, fp_cols)
    if with_leak_probe:
        write_csv(D / "leak_probe.csv", [{"quantity": q, "auc": 0.5, "null_p95": 0.75, "rank": "rank 240 of 500", "verdict": LEAK_NOT_AUDIBLE}
                                         for q in ("n_pairs", "dt_est_s", "frac_above_band", "K_med")],
                  ["quantity", "auc", "null_p95", "rank", "verdict"])
    return out


if __name__ == "__main__":  # a by-hand look at the tree
    import tempfile
    p = make_out(Path(tempfile.mkdtemp()) / "out")
    print(p)
