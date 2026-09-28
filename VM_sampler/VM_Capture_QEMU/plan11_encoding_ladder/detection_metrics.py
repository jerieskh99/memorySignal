#!/usr/bin/env python3
"""detection_metrics.py -- the two-class forest run with in-fold thresholds, the operating point,
the workload-level label-shuffle null, the one-feature baseline and B1-G3 for two classes, the
one-class run, the split stage of the detection layer and the time-to-detect ladder
(SPEC_DETECTION.md section 3.3, builder A).

Corrections from the reviews of SPEC_DETECTION.md applied here (each marked "must change"):
  ML 2.1          ROW_UNIT = "cell": the learner sees one row per cell (the nanmean of its window
                  features); under window rows the out-of-bag threshold is refused;
  al-Kindi 2.1    THRESHOLD_SOURCE = "inner_lowo" (in-fold, out-of-workload benign scores);
                  "inner_group_kfold" with INNER_K = 4 as the cheap variant; "oob" kept and labelled
                  optimistic;
  ML 2.2, al-Farabi M3   the denominator rule is uniform: `control` counts as at floor in the observed
                  run and under every permutation (NULL_DENOMINATOR_RULE); a pending G-K0 makes every
                  true-positive field `not run: gk0 not run`; a missing gk0_cells.csv is a written refusal;
  ML 2.5, al-Farabi M2   TRAIN_ON_AT_FLOOR = False: positive cells at floor, at harness floor or
                  control leave every training set (observed run and null alike);
  ML 2.4          the one-feature model at the operating point, the L1 row (best single feature under
                  LOWO with its own null), the full-feature forest as the headline (score_source
                  "full"), the re-run beside it;
  ML 2.7, al-Kindi 2.2   the one-class run has the same workload-level null;
  al-Farabi M4    a cell without a finite score is `not applicable: no finite window`, never a miss;
  al-Farabi M6    the ladder caps a prefix at the extract's `_n_rows`; LADDER_HEAD_DROP_RULE declared;
  al-Farabi M7    GCAL_SPREAD_RULE is a constant.

Citation: P3 D5; CR3 1.5, 2.1, 2.2, 2.3, 2.18, 2.19, 2.29; ML 1.4, 1.5, 1.6, 2.2, 2.3, 2.4, 3.6, 3.7;
K3 2.2, move 17; SPEC 4.2 (the forest), 3.7.7 (the reduction).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plan11_encoding_ladder import schema  # noqa: E402
from plan11_encoding_ladder import series as S  # noqa: E402
from plan11_encoding_ladder import models as M  # noqa: E402
from plan11_encoding_ladder import nulls as NL  # noqa: E402
from plan11_encoding_ladder import verdicts as V  # noqa: E402
from plan11_encoding_ladder import classes as C  # noqa: E402
from plan11_encoding_ladder import detection_splits as DS  # noqa: E402
from plan11_encoding_ladder.nulls import SEED_FOREST, SEED_LABEL_NULL  # noqa: E402

from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

# --------------------------------------------------------------------------- constants (3.3.1)
ROW_UNIT = DS.ROW_UNIT
FPR_DECLARED = 0.05                 # P3 D5; CR3 2.13; ML 2.3 (declared before the data)
FPR_RESOLUTION_LIMIT = 0.01         # labelled "resolution limit" (ML 1.6 item 3)
THRESHOLD_SOURCE = "inner_lowo"     # al-Kindi 2.1 (ML 1.6 item 2 literally is "oob", optimistic through sibling reps)
INNER_K = 4                         # al-Kindi 2.1: "inner_group_kfold"
THRESHOLD_SOURCES = ("inner_lowo", "inner_group_kfold", "oob")
THRESHOLD_SOURCE_LABELS = {"inner_lowo": "in-fold, out-of-workload benign scores (leave-one-benign-workload-out inside the training fold)",
                           "inner_group_kfold": f"in-fold, out-of-group benign scores (benign training workloads in {INNER_K} groups)",
                           "oob": "optimistic: sibling reps in bag (out-of-bag scores of the benign training cells)"}
THRESHOLD_QUANTILE_METHOD = "linear"   # numpy.quantile's default; section 7 item 9
SCORE_AGGREGATION = "window_mean_proba"   # section 7 item 8; under ROW_UNIT = "cell" written as not applicable
N_PERM = 500                        # CR3 2.1
NULL_EXHAUSTIVE_BELOW = 500         # exhaustive enumeration when comb(S + B, S) < 500
NULL_MIN_ASSIGNMENTS = 20           # below it: NULL_NOT_ESTIMABLE (ML 1.4, 2.2)
NULL_STATISTICS = ("tpr05", "auc")  # both recorded per permutation
NULL_VERDICT_STATISTIC = "tpr05"    # P3 D5; section 7 item 10
NULL_DENOMINATOR_RULE = "same_as_observed"   # al-Farabi M3: a permuted-positive cell is in the denominator when its verdict is above floor; control counts as at floor
TRAIN_ON_AT_FLOOR = False           # ML 2.5; al-Farabi M2 (section 7): at-floor positives leave every training set
AT_FLOOR_POSITIVES_IN_TRAINING = TRAIN_ON_AT_FLOOR
FLOOR_VERDICTS_OUT = (V.GK0_AT_FLOOR, V.GK0_AT_HARNESS_FLOOR, "control")
N_ESTIMATORS = 300; MIN_SAMPLES_LEAF = 1; CLASS_WEIGHT = None   # ML 3.7 question 12; SPEC 4.2; section 7 item 6
LADDER_PREFIXES_S = (30, 60, 120, 300, 600)   # K3 move 17
LADDER_DT_S = schema.DT_BRACKET_S[1]           # 0.644 s, the derived guest spacing; 0.500 the configured interval; section 7 item 20
LADDER_READINGS = ("from_pair1", "from_boundary")
LADDER_NORM = "prefix"
LADDER_HEAD_DROP_RULE = "from_pair1_only"      # al-Farabi M6: the boundary start replaces the head drop under from_boundary
ONE_CLASS_MODEL = "isolation_forest"           # G-1C's primary; section 7 item 12
ONE_CLASS_MODELS = ("isolation_forest", "gmm", "ocsvm")
ONE_CLASS_THRESHOLD_SOURCE = "inner_lowo"      # section 7 item 13
B1G3_MAX_DISAGREE_WORKLOADS = 1                # CR3 2.2
GCAL_SPREAD_RULE = "p95_minus_p05"             # al-Farabi M7
L1_DIRECTION_RULE = "sign_of_median_difference"   # ML 2.4
L1_BEST_RULE = "training_auc"                  # ML 2.4: the best feature chosen per fold on the training fold
EXCLUDED_BY_DECLARATION = ("n_pairs", "n_windows", "dt_est_s", "iteration_count", "order_index", "campaign", "label", "path")
SCORE_SOURCE_FULL = "full"                     # ML 2.4 (b): the full-feature forest is the headline
POSITIVE = "sandbox"
CITATION_OP = "P3 D5; CR3 1.5; ML 1.6 (metrics), 2.3 (resolution), 2.4 (per-workload recall), 3.7 (in-fold thresholds)"
CITATION_NULL = "CR3 2.1; ML 1.4, 2.2 (the workload-level label-shuffle null)"
CITATION_L1 = "ML 1.5 (L1, the one-feature threshold), 2.4 of the ML review (the one-feature model at the operating point); CR3 2.2 (B1-G3)"
CITATION_1C = "CR3 2.19; ML 1.6 (a single density model declared as primary before the data); K3 section 5 point 1"
CITATION_LADDER = "K3 move 17; CR3 2.29; N3 Sec. 1 RQ1, Figure 6"
NO_SCORE = V.not_applicable("no finite window")
PRED_COLUMNS = ("cell_id", "class", "y", "workload_key", "family", "member_index", "subfamily_letter", "rep", "campaign",
                "fold", "score", "threshold_05", "threshold_01", "flag_05", "flag_01", "n_windows", "floor_verdict",
                "in_denominator", "score_status")


# --------------------------------------------------------------------------- the forest and the cell score (3.3.2)

def make_detection_forest(seed: int, n_jobs: int = 1, *, n_estimators: int = N_ESTIMATORS,
                          min_samples_leaf: int = MIN_SAMPLES_LEAF, class_weight=CLASS_WEIGHT, oob_score: bool = True):
    """models.make_forest(seed, n_jobs, n_estimators, oob_score=..., min_samples_leaf=..., class_weight=...):
    the same imputer, scaler and RandomForestClassifier as the encoding paper (SPEC 4.2), with
    out-of-bag scoring on so the threshold source "oob" can read the benign training cells'
    out-of-bag scores (ML 1.6 item 2). Citation: P3 D5 'a random forest with hyperparameters
    declared before the data'; ML 3.7."""
    return M.make_forest(seed, n_jobs, n_estimators, oob_score=oob_score, min_samples_leaf=min_samples_leaf, class_weight=class_weight)


def cell_scores(cell_ids, proba_pos, rule: str = SCORE_AGGREGATION) -> dict:
    """The cell's score from its rows: 'window_mean_proba' the mean over its rows of P(sandbox);
    'window_median_proba' the median; 'vote_fraction' the fraction of rows with P > 0.5. NaN rows
    are skipped; a cell with no finite row is NaN (and is then reported as ``not applicable: no
    finite window``, al-Farabi M4, never counted as a miss). Under ROW_UNIT = "cell" every cell has
    one row and the rule is the identity (ML review 2.1). Citation: ML 1.6; section 7 item 8."""
    cell_ids = np.asarray(cell_ids).astype(str)
    p = np.asarray(proba_pos, dtype=np.float64)
    out = {}
    for c in dict.fromkeys(cell_ids.tolist()):
        v = p[cell_ids == c]
        v = v[np.isfinite(v)]
        if len(v) == 0:
            out[c] = float("nan")
        elif rule == "window_mean_proba":
            out[c] = float(v.mean())
        elif rule == "window_median_proba":
            out[c] = float(np.median(v))
        elif rule == "vote_fraction":
            out[c] = float(np.mean(v > 0.5))
        else:
            raise ValueError(rule)
    return out


def _pos_column(clf, positive: str) -> int:
    classes = list(clf.named_steps["rf"].classes_) if hasattr(clf, "named_steps") else list(clf.classes_)
    return classes.index(positive)


def _proba_pos(clf, X, positive: str) -> np.ndarray:
    classes = list(clf.named_steps["rf"].classes_)
    if positive not in classes:
        return np.zeros(len(X))
    return clf.predict_proba(X)[:, classes.index(positive)]


def oob_cell_scores(pipeline, X_train, y_train, cell_ids_train, positive: str = POSITIVE, rule: str = SCORE_AGGREGATION) -> dict:
    """The training cells' out-of-bag scores (ML 1.6 item 2): the fitted forest's
    oob_decision_function_ column for the positive class, aggregated per cell by cell_scores; a row
    never out of bag (NaN) is skipped. The imputer and scaler of the pipeline are fitted on the
    training fold, so the out-of-bag rows were transformed by them (params.oob_note). Under cell
    rows the score is optimistic only through sibling reps in bag (al-Kindi 2.1); under window rows
    it is refused (ML review 2.1)."""
    rf = pipeline.named_steps["rf"]
    if not getattr(rf, "oob_score", False) or not hasattr(rf, "oob_decision_function_"):
        raise ValueError("the forest was not fitted with oob_score=True")
    classes = list(rf.classes_)
    if positive not in classes:
        return {c: float("nan") for c in dict.fromkeys(np.asarray(cell_ids_train).astype(str).tolist())}
    oob = rf.oob_decision_function_[:, classes.index(positive)]
    return cell_scores(cell_ids_train, oob, rule)


def in_fold_threshold(scores_benign: dict, fpr: float, method: str = THRESHOLD_QUANTILE_METHOD) -> float:
    """numpy.quantile(values, 1 - fpr, method=method) over the benign training cells' in-fold
    scores (the 95th percentile at fpr = 0.05; the 99th at 0.01). A cell is flagged when its score
    is strictly greater than the threshold (strict, as every exceedance in this toolkit). NaN when
    no benign score exists. Citation: ML 1.6 item 2; CR3 1.5."""
    v = np.asarray([x for x in dict(scores_benign).values() if x is not None and np.isfinite(x)], dtype=np.float64)
    if len(v) == 0:
        return float("nan")
    return float(np.quantile(v, 1.0 - float(fpr), method=method))


def _fit_forest(Xtr, ytr, *, seed, n_jobs, n_estimators, min_samples_leaf, class_weight, oob):
    clf = make_detection_forest(seed, n_jobs, n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                class_weight=class_weight, oob_score=oob)
    return clf.fit(Xtr, ytr)


def _drop_at_floor(lab: dict, tr: np.ndarray, y_rows: np.ndarray, positive: str, train_on_at_floor: bool) -> tuple:
    """Training rows without the positive cells whose floor verdict is at floor, at harness floor or
    control (TRAIN_ON_AT_FLOOR = False; ML 2.5). Returns (rows, n_dropped_cells)."""
    if train_on_at_floor:
        return tr, 0
    fv = lab["floor_verdict"][tr]
    drop = (y_rows[tr] == positive) & np.isin(fv, FLOOR_VERDICTS_OUT)
    n_cells = len(set(lab["cell_id"][tr][drop].tolist()))
    return tr[~drop], n_cells


def inner_benign_scores(X, lab, tr: np.ndarray, y_rows: np.ndarray, *, source: str, positive: str, seed: int, n_jobs: int,
                        n_estimators: int, min_samples_leaf: int, class_weight, score_rule: str, inner_k: int = INNER_K,
                        fitted=None) -> dict:
    """The benign training cells' in-fold scores under the threshold source (al-Kindi 2.1):
    'inner_lowo' fits, inside the training fold, one forest per benign training workload with that
    workload held out and scores it (B extra fits per fold); 'inner_group_kfold' groups the benign
    training workloads into ``inner_k`` groups (a seeded permutation) and does the same per group;
    'oob' reads the fitted forest's out-of-bag scores (``fitted``). Returns {cell_id: score}."""
    ben = tr[y_rows[tr] == "benign"] if positive == POSITIVE else tr[y_rows[tr] != positive]
    if len(ben) == 0:
        return {}
    if source == "oob":
        if fitted is None:
            fitted = _fit_forest(X[tr], y_rows[tr], seed=seed, n_jobs=n_jobs, n_estimators=n_estimators,
                                 min_samples_leaf=min_samples_leaf, class_weight=class_weight, oob=True)
        sc = oob_cell_scores(fitted, X[tr], y_rows[tr], lab["cell_id"][tr], positive, score_rule)
        ben_cells = set(lab["cell_id"][ben].tolist())
        return {c: v for c, v in sc.items() if c in ben_cells}
    wks = list(dict.fromkeys(lab["workload_key"][ben].tolist()))
    if source == "inner_lowo":
        groups = [[w] for w in wks]
    elif source == "inner_group_kfold":
        rng = np.random.default_rng(int(seed))
        perm = [wks[i] for i in rng.permutation(len(wks))]
        k = max(1, min(int(inner_k), len(perm)))
        groups = [perm[i::k] for i in range(k)]
    else:
        raise ValueError(source)
    out: dict = {}
    for g in groups:
        held = np.isin(lab["workload_key"][tr], g) & (y_rows[tr] != positive)
        tr_in = tr[~held]
        te_in = tr[held]
        if len(set(y_rows[tr_in].tolist())) < 2 or len(te_in) == 0:
            continue
        clf = _fit_forest(X[tr_in], y_rows[tr_in], seed=seed, n_jobs=n_jobs, n_estimators=n_estimators,
                          min_samples_leaf=min_samples_leaf, class_weight=class_weight, oob=False)
        out.update(cell_scores(lab["cell_id"][te_in], _proba_pos(clf, X[te_in], positive), score_rule))
    return out


# --------------------------------------------------------------------------- the operating point (3.3.3)

def run_operating_point(X, lab, folds, *, seed=SEED_FOREST, n_jobs=1, n_estimators=N_ESTIMATORS, min_samples_leaf=MIN_SAMPLES_LEAF,
                        class_weight=CLASS_WEIGHT, fpr_declared=FPR_DECLARED, fpr_limit=FPR_RESOLUTION_LIMIT,
                        threshold_source=THRESHOLD_SOURCE, method=THRESHOLD_QUANTILE_METHOD, score_rule=SCORE_AGGREGATION,
                        positive=POSITIVE, label_key="y", y_override=None, reduce_to=None, reduce_method="train_importance",
                        train_on_at_floor=TRAIN_ON_AT_FLOOR, inner_k=INNER_K, set_thresholds=None) -> dict:
    """Per fold: fit the forest on the training rows (positive cells at floor dropped unless
    train_on_at_floor); threshold_05 and threshold_01 from the benign training cells' in-fold
    scores under threshold_source (inner_benign_scores); score every test cell (cell_scores over
    its rows); flag at both thresholds. Returns {'records': [per test cell: cell_id, fold, y, cls,
    workload_key, family, member_index, subfamily_letter, rep, campaign, score, threshold_05,
    threshold_01, flag_05, flag_01, n_windows, floor_verdict, score_status], 'folds': [per fold:
    name, held_out, n_train_cells, n_benign_train_cells, n_train_dropped_at_floor, threshold_05,
    threshold_01, setters_05 (the benign training cells whose in-fold score is >= threshold_05,
    GOP_SETTER_RULE "ge"), setter_workloads_05, d_used, importance], 'importance_mean'}.
    `positive` and `label_key` (or `y_override`, a per-row label array) let the same runner score
    the campaign, half and other binary labels: when the label is not the class label no threshold
    is set (the threshold and flag fields are empty) and the run's statistic is the AUC alone; with
    `reduce_to` the features are reduced per fold on the training fold only through
    models.reduce_fit (SPEC 3.7.7). Citation: CITATION_OP."""
    X = np.asarray(X, dtype=np.float64)
    y_rows = np.asarray(y_override if y_override is not None else lab[label_key]).astype(str)
    class_label = (label_key == "y") if set_thresholds is None else bool(set_thresholds)
    records, fold_recs, imps = [], [], []
    for f in folds:
        tr, te = np.asarray(f["train"], dtype=np.int64), np.asarray(f["test"], dtype=np.int64)
        tr, n_dropped = _drop_at_floor(lab, tr, y_rows, positive, train_on_at_floor) if class_label else (tr, 0)
        frec = {"name": f["name"], "held_out": f.get("held_out"), "n_train_cells": len(set(lab["cell_id"][tr].tolist())),
                "n_benign_train_cells": int(len(set(lab["cell_id"][tr][y_rows[tr] != positive].tolist()))),
                "n_train_dropped_at_floor": int(n_dropped), "threshold_05": None, "threshold_01": None,
                "setters_05": [], "setter_workloads_05": [], "d_used": int(X.shape[1]), "importance": None, "status": "ok"}
        if len(tr) == 0 or len(te) == 0 or len(set(y_rows[tr].tolist())) < 2:
            frec["status"] = V.not_applicable("one class in training" if len(tr) else "empty training fold")
            for c in dict.fromkeys(lab["cell_id"][te].tolist()):
                i = te[lab["cell_id"][te] == c][0]
                records.append(_record(lab, i, f["name"], y_rows[i], float("nan"), None, None, frec["status"]))
            fold_recs.append(frec)
            continue
        Xtr, Xte = X[tr], X[te]
        if reduce_to is not None and int(reduce_to) < Xtr.shape[1]:
            T = M.reduce_fit(Xtr, y_rows[tr], int(reduce_to), reduce_method, seed, n_estimators)
            Xtr, Xte = T(Xtr), T(Xte)
            frec["d_used"] = int(Xtr.shape[1])
        clf = _fit_forest(Xtr, y_rows[tr], seed=seed, n_jobs=n_jobs, n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                          class_weight=class_weight, oob=(threshold_source == "oob" and class_label))
        imp = clf.named_steps["rf"].feature_importances_
        frec["importance"] = imp.tolist()
        if frec["d_used"] == X.shape[1]:
            imps.append(imp)
        t05 = t01 = None
        if class_label:
            if frec["d_used"] == X.shape[1]:
                Xw = X
            else:   # reduced per fold: the inner scores are computed on the reduced matrix of this fold
                Xw = np.zeros((X.shape[0], Xtr.shape[1])); Xw[tr] = Xtr; Xw[te] = Xte
            bs = inner_benign_scores(Xw, lab, tr, y_rows, source=threshold_source, positive=positive, seed=seed, n_jobs=n_jobs,
                                     n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, class_weight=class_weight,
                                     score_rule=score_rule, inner_k=inner_k, fitted=clf)
            t05 = in_fold_threshold(bs, fpr_declared, method)
            t01 = in_fold_threshold(bs, fpr_limit, method)
            frec["threshold_05"], frec["threshold_01"] = t05, t01
            wk_of = {c: str(lab["workload_key"][tr][lab["cell_id"][tr] == c][0]) for c in bs}
            setters = sorted(c for c, v in bs.items() if np.isfinite(v) and np.isfinite(t05) and v >= t05)
            frec["setters_05"] = setters
            frec["setter_workloads_05"] = sorted({wk_of[c] for c in setters})
        sc = cell_scores(lab["cell_id"][te], _proba_pos(clf, Xte, positive), score_rule)
        for c, s in sc.items():
            i = te[lab["cell_id"][te] == c][0]
            records.append(_record(lab, i, f["name"], y_rows[i], s, t05, t01, "ok" if np.isfinite(s) else NO_SCORE))
        fold_recs.append(frec)
    imp_mean = np.mean(np.stack(imps), axis=0).tolist() if imps else None
    return {"records": records, "folds": fold_recs, "importance_mean": imp_mean}


def _record(lab, i, fold, y, score, t05, t01, status) -> dict:
    ok = np.isfinite(score)
    return {"cell_id": str(lab["cell_id"][i]), "class": str(lab["cls"][i]), "y": str(y), "workload_key": str(lab["workload_key"][i]),
            "family": str(lab["family"][i]), "member_index": int(lab["member_index"][i]), "subfamily_letter": str(lab["subfamily_letter"][i]),
            "rep": int(lab["rep"][i]), "campaign": str(lab["campaign"][i]), "fold": fold, "score": float(score) if ok else None,
            "threshold_05": t05, "threshold_01": t01,
            "flag_05": (bool(score > t05) if (ok and t05 is not None and np.isfinite(t05)) else None),
            "flag_01": (bool(score > t01) if (ok and t01 is not None and np.isfinite(t01)) else None),
            "n_windows": int(lab["n_windows"][i]), "floor_verdict": str(lab["floor_verdict"][i]), "score_status": status}


def _eighths(k: int, n: int) -> str:
    return f"{int(k)}/{int(n)}"


def _pooled_tpr_at(fpr_pts, tpr_pts, fpr: float) -> float | None:
    if fpr_pts is None or len(fpr_pts) == 0:
        return None
    f = np.asarray(fpr_pts, dtype=np.float64); t = np.asarray(tpr_pts, dtype=np.float64)
    order = np.argsort(f, kind="stable")
    return float(np.interp(fpr, f[order], t[order]))


def summarize_operating_point(records: list[dict], folds: list[dict], floor_by_cell: dict | None = None, *,
                              fpr_declared: float = FPR_DECLARED, fpr_limit: float = FPR_RESOLUTION_LIMIT, positive: str = POSITIVE,
                              class_label: bool = True) -> dict:
    """The pooled reading (ML 1.6; CR3 1.5): the denominator of every true-positive rate is the
    set of positive cells whose floor verdict is 'above floor' (K3 F3: at-floor, at-harness-floor
    and control cells leave the denominator with their verdict, listed in 'excluded_at_floor';
    the rule is the same in the observed run and under every permutation, NULL_DENOMINATOR_RULE);
    when any positive cell's verdict is 'pending: gk0 not run' every tpr field is
    'not run: gk0 not run' (ML review 2.2). Cells without a finite score are listed in
    'excluded_no_score' and enter no rate (al-Farabi M4). tpr_05 = flagged / denominator;
    fpr_05_realized = flagged benign / benign scored (the realized out-of-fold rate); tpr_01,
    fpr_01_realized likewise; auc = roc_auc_score over the pooled out-of-fold scores (positives in
    the denominator only); roc = roc_curve points; tpr_at_fpr05_pooled = the TPR read from the
    pooled ROC at FPR exactly 0.05 (interpolated), labelled POST_HOC_LABEL; per_member = {m: {hits,
    denominator, at_floor, eighths}} (never a mean; ML 2.4); per_family_fpr = {family: {n, flagged,
    fpr}}; per_workload_outcome = {workload_key: recall (positive) or 1 - flagged fraction (benign)}
    (G-M's paired unit, ML 2.5); majority_accuracy = n_benign / (n_benign + n_positive_in_denominator)
    (B1-G6, CR3 2.3); random_scorer; gop = the per-fold setter counts with the worst fold (ML
    review 2.3) and the realized false positives; gcal = {per_fold_tpr05, pooled_tpr_at_fpr05}."""
    fb = floor_by_cell or {}
    recs = [dict(r) for r in records]
    for r in recs:
        r["floor_verdict"] = fb.get(r["cell_id"], r.get("floor_verdict", DS.PENDING_GK0))
    scored = [r for r in recs if r.get("score") is not None and np.isfinite(r["score"])]
    no_score = sorted({r["cell_id"] for r in recs if r not in scored})
    pos_all = [r for r in recs if r["y"] == positive]
    if not class_label:
        # a binary label that is not the class label: AUC alone
        ys = np.array([r["y"] == positive for r in scored]); ss = np.array([r["score"] for r in scored])
        auc = float(roc_auc_score(ys, ss)) if len(scored) and 0 < ys.sum() < len(ys) else None
        for r in recs:
            r["in_denominator"] = bool(r in scored)
        return {"auc": auc, "n_positive_scored": int(ys.sum()) if len(scored) else 0, "n_negative_scored": int((~ys).sum()) if len(scored) else 0,
                "excluded_no_score": no_score, "n_without_score": len(no_score), "random_scorer": {"auc": 0.5, "tpr_equals_fpr": True}, "records": recs}
    pending = any(r["floor_verdict"].startswith("pending") for r in pos_all)
    for r in recs:
        if r["y"] == positive:
            r["in_denominator"] = bool(r["floor_verdict"] == V.GK0_ABOVE_FLOOR and r in scored)
        else:
            r["in_denominator"] = bool(r in scored)
    pos = [r for r in scored if r["y"] == positive]
    den = [r for r in pos if r["in_denominator"]]
    at_floor = sorted({r["cell_id"] for r in pos_all if r["floor_verdict"] in FLOOR_VERDICTS_OUT})
    ben = [r for r in scored if r["y"] != positive]
    nr = V.not_run("gk0 not run")

    def rate(rows, key):
        v = [r[key] for r in rows if r.get(key) is not None]
        return (float(np.mean(v)) if v else None)
    tpr05 = nr if pending else rate(den, "flag_05")
    tpr01 = nr if pending else rate(den, "flag_01")
    fpr05 = rate(ben, "flag_05")
    fpr01 = rate(ben, "flag_01")
    ys = np.array([1] * len(den) + [0] * len(ben)); ss = np.array([r["score"] for r in den] + [r["score"] for r in ben])
    auc = roc = None
    if len(den) and len(ben):
        auc = float(roc_auc_score(ys, ss))
        fp, tp, th = roc_curve(ys, ss)
        roc = {"fpr": fp.tolist(), "tpr": tp.tolist(), "threshold": [None if not np.isfinite(t) else float(t) for t in th]}
    pooled05 = _pooled_tpr_at(roc["fpr"], roc["tpr"], fpr_declared) if roc else None
    per_member: dict = {}
    for r in pos_all:
        m = str(r["member_index"])
        d = per_member.setdefault(m, {"hits": 0, "denominator": 0, "at_floor": 0, "n_cells": 0, "no_score": 0})
        d["n_cells"] += 1
        if r["floor_verdict"] in FLOOR_VERDICTS_OUT:
            d["at_floor"] += 1
        elif r in scored and r["in_denominator"]:
            d["denominator"] += 1
            d["hits"] += int(bool(r.get("flag_05")))
        elif r not in scored:
            d["no_score"] += 1
    for m, d in per_member.items():
        d["eighths"] = nr if pending else _eighths(d["hits"], d["denominator"])
        d["recall"] = (nr if pending else (d["hits"] / d["denominator"] if d["denominator"] else None))
    per_family: dict = {}
    for r in ben:
        d = per_family.setdefault(r["family"], {"n": 0, "flagged": 0})
        d["n"] += 1; d["flagged"] += int(bool(r.get("flag_05")))
    for fam, d in per_family.items():
        d["fpr"] = d["flagged"] / d["n"] if d["n"] else None
        d["recall_benign"] = (1.0 - d["fpr"]) if d["fpr"] is not None else None
    per_wk: dict = {}
    for r in scored:
        per_wk.setdefault(r["workload_key"], []).append(r)
    per_workload_outcome = {}
    for wk, rows in per_wk.items():
        if rows[0]["y"] == positive:
            dd = [r for r in rows if r["in_denominator"]]
            per_workload_outcome[wk] = (nr if pending else (float(np.mean([bool(r.get("flag_05")) for r in dd])) if dd else None))
        else:
            per_workload_outcome[wk] = 1.0 - float(np.mean([bool(r.get("flag_05")) for r in rows]))
    n_ben = len(ben); n_pos_den = len(den)
    maj = n_ben / (n_ben + n_pos_den) if (n_ben + n_pos_den) else None
    # G-OP counts (ML review 2.3): per fold, then the worst fold; the union as a disclosure
    per_fold = []
    for f in folds:
        if f.get("threshold_05") is None:
            continue
        per_fold.append({"fold": f["name"], "n_setter_cells": len(f.get("setters_05") or []),
                         "n_setter_workloads": len(f.get("setter_workloads_05") or []), "setter_workloads": f.get("setter_workloads_05") or []})
    union_cells = sorted({c for f in folds for c in (f.get("setters_05") or [])})
    union_wk = sorted({w for f in folds for w in (f.get("setter_workloads_05") or [])})
    fps = [r for r in ben if r.get("flag_05")]
    gop = {"per_fold": per_fold, "worst_fold_n_setter_cells": (min(p["n_setter_cells"] for p in per_fold) if per_fold else 0),
           "worst_fold_n_setter_workloads": (min(p["n_setter_workloads"] for p in per_fold) if per_fold else 0),
           "union_setter_cells": union_cells, "n_union_setter_cells": len(union_cells), "n_union_setter_workloads": len(union_wk),
           "setter_families": sorted({r["family"] for r in recs if r["cell_id"] in set(union_cells)} | {w for w in union_wk if w in ("idle", "harness_idle")}),
           "realized_fp_cells": sorted(r["cell_id"] for r in fps), "n_realized_fp_cells": len(fps),
           "n_realized_fp_workloads": len({r["workload_key"] for r in fps}), "fp_families": sorted({r["family"] for r in fps})}
    return {"tpr_05": tpr05, "fpr_05_realized": fpr05, "tpr_01": tpr01, "fpr_01_realized": fpr01, "auc": auc, "roc": roc,
            "tpr_at_fpr05_pooled": pooled05, "pooled_label": V.POST_HOC_LABEL,
            "n_positive_scored": len(pos), "n_positive_in_denominator": n_pos_den, "n_positive_at_floor": len(at_floor),
            "excluded_at_floor": at_floor, "excluded_no_score": no_score, "n_without_score": len(no_score), "n_benign_scored": n_ben,
            "gk0_pending": bool(pending), "per_member": per_member, "per_family_fpr": per_family,
            "per_workload_outcome": per_workload_outcome, "majority_accuracy": maj,
            "random_scorer": {"auc": 0.5, "tpr_equals_fpr": True}, "gop": gop,
            "gcal": {"per_fold_tpr05": tpr05, "pooled_tpr_at_fpr05": pooled05}, "records": recs}


# --------------------------------------------------------------------------- the null (3.3.4)

def count_assignments(S_: int, B_: int) -> int:
    """math.comb(S + B, S): the number of workload-level label assignments (ML 2.2; CR3 2.1)."""
    return math.comb(int(S_) + int(B_), int(S_))


def workload_label_permutations(workload_keys: list[str], S_: int, n_perm: int, rng) -> tuple[list, bool]:
    """The S sandbox labels reassigned to a random S-subset of the S + B workloads, every cell
    inheriting its workload's label (ML 1.4; CR3 2.1; K3 2.2). Exhaustive (itertools.combinations)
    when comb(S + B, S) < NULL_EXHAUSTIVE_BELOW and n_perm covers it (a smoke run with n_perm below
    the count draws n_perm and is non-exhaustive, so its verdict reads 'not run: N permutations <
    500', section 7 item 32), else n_perm draws from rng.choice(S + B, S, replace=False) (duplicates
    allowed and counted by the caller). Returns (subsets, exhaustive). test_only workloads are never
    in the pool (they are never trained on)."""
    keys = list(dict.fromkeys(workload_keys))
    n = len(keys); S_ = int(S_)
    if S_ <= 0 or S_ > n:
        return [], True
    total = math.comb(n, S_)
    if total < NULL_EXHAUSTIVE_BELOW and total <= int(n_perm):
        return [frozenset(c) for c in itertools.combinations(keys, S_)], True
    out = []
    for _ in range(int(n_perm)):
        idx = rng.choice(n, S_, replace=False)
        out.append(frozenset(keys[i] for i in idx))
    return out, False


def null_verdict(observed, null_values, n_assignments: int, *, exhaustive: bool = False, n_perm_required: int = N_PERM) -> tuple[str, dict]:
    """nulls.null_summary plus the verdict: NULL_NOT_ESTIMABLE when n_assignments <
    NULL_MIN_ASSIGNMENTS (the row carries no verb); else PASS when observed strictly exceeds p95
    (ties fail), else NULL_INSIDE; the rank is reported as 'rank r of n'. A run with fewer than 500
    permutations and a non-exhaustive pool writes not_run('N permutations < 500') as the verdict (a
    smoke run; the numbers stay). Citation: CR3 2.1; ML 1.4, 2.2."""
    vals = np.asarray([v for v in np.asarray(null_values, dtype=np.float64) if np.isfinite(v)], dtype=np.float64)
    obs = float(observed) if (observed is not None and isinstance(observed, (int, float, np.floating, np.integer)) and np.isfinite(observed)) else float("nan")
    summ = NL.null_summary(obs, vals)
    summ["rank_text"] = f"rank {summ['rank']} of {summ['n']}" if summ["rank"] is not None else ""
    summ["n_assignments"] = int(n_assignments)
    summ["exhaustive"] = bool(exhaustive)
    if int(n_assignments) < NULL_MIN_ASSIGNMENTS:
        summ["verdict"] = V.NULL_NOT_ESTIMABLE
    elif summ["n"] == 0 or np.isnan(obs):
        summ["verdict"] = V.not_run("observed statistic undefined" if np.isnan(obs) else "no permutation")
    elif not exhaustive and summ["n"] < int(n_perm_required):
        summ["verdict"] = V.not_run(f"{summ['n']} permutations < {int(n_perm_required)}")
    else:
        summ["verdict"] = V.PASS if summ["exceeds"] else V.NULL_INSIDE
    return summ["verdict"], summ


def _y_from_subset(lab: dict, subset: frozenset, positive: str = POSITIVE) -> np.ndarray:
    return np.where(np.isin(lab["workload_key"], list(subset)), positive, "benign").astype(str)


def null_distribution(X, lab, folds, subsets, *, floor_by_cell=None, statistics=NULL_STATISTICS, null_jobs=1, l1=False, **op_kw) -> dict:
    """For every subset: relabel y at the workload, rebuild nothing (LOWO and LOCO folds are
    label-free), re-run run_operating_point and summarize_operating_point, record tpr05 and auc
    (and, with l1, the best-single-feature tpr05 and auc under the same permutation, ML review
    2.4). The same seed for every permutation; permutations are independent and are distributed
    over n_jobs processes. Citation: CITATION_NULL."""
    positive = op_kw.get("positive", POSITIVE)
    fpr_declared = op_kw.get("fpr_declared", FPR_DECLARED); fpr_limit = op_kw.get("fpr_limit", FPR_RESOLUTION_LIMIT)
    op_kw = dict(op_kw)
    if int(null_jobs) > 1:
        op_kw["n_jobs"] = 1      # permutations are the parallel unit; each forest runs on one core

    def one(sub):
        y = _y_from_subset(lab, sub, positive)
        r = run_operating_point(X, lab, folds, y_override=y, **op_kw)
        s = summarize_operating_point(r["records"], r["folds"], floor_by_cell, fpr_declared=fpr_declared, fpr_limit=fpr_limit, positive=positive)
        out = {"tpr05": s["tpr_05"] if isinstance(s["tpr_05"], float) else np.nan, "auc": s["auc"] if s["auc"] is not None else np.nan}
        if l1:
            l = l1_row(X, lab, folds, y_override=y, positive=positive, fpr=fpr_declared, fpr_limit=fpr_limit, floor_by_cell=floor_by_cell,
                       method=op_kw.get("method", THRESHOLD_QUANTILE_METHOD), train_on_at_floor=op_kw.get("train_on_at_floor", TRAIN_ON_AT_FLOOR))
            out["l1_tpr05"] = l["tpr_05"] if isinstance(l["tpr_05"], float) else np.nan
            out["l1_auc"] = l["auc"] if l["auc"] is not None else np.nan
        return out
    if null_jobs and int(null_jobs) > 1 and len(subsets) > 1:
        from joblib import Parallel, delayed
        vals = Parallel(n_jobs=int(null_jobs))(delayed(one)(s) for s in subsets)
    else:
        vals = [one(s) for s in subsets]
    keys = list(statistics) + (["l1_tpr05", "l1_auc"] if l1 else [])
    return {k: np.asarray([v.get(k, np.nan) for v in vals], dtype=np.float64) for k in keys}


def binary_label_permutations(lab: dict, y_obs: np.ndarray, split: str, half_rule: str, n_perm: int, rng) -> tuple:
    """The label null of the binary-label splits (ML 3.1, 3.2; CR3 2.14, 2.20; al-Kindi review 2.11).
    ``order`` under ``within_workload``: the half labels permuted within each workload (the counts
    kept); under ``within_class``: when every workload lies inside one half, the workload-level
    permutation of the half labels with the count of first-half workloads kept (the label is a
    per-workload constant, ML 1.4), else the per-cell half vector permuted within the class.
    ``anchor_idle``: the campaign labels permuted across the idle cells; ``idle_early_late``: the
    half labels permuted across the idle cells (cell-level, as plan11's G-X). Returns (label arrays,
    null_unit, reason, n_assignments)."""
    y = np.asarray(y_obs).astype(str)
    lab_rows = np.flatnonzero(y != "")
    wk = lab["workload_key"]
    perms = []
    if split == "order" and half_rule == "within_workload":
        groups = [lab_rows[wk[lab_rows] == w] for w in dict.fromkeys(wk[lab_rows].tolist())]
        n_assign = 1
        for g in groups:
            n_assign *= math.comb(len(g), int(np.sum(y[g] == "first")))
        for _ in range(n_perm):
            yp = y.copy()
            for g in groups:
                yp[g] = y[g][rng.permutation(len(g))]
            perms.append(yp)
        return perms, "cell within workload", "the half labels permuted within each workload (al-Kindi review 2.11)", n_assign
    if split == "order":
        wks = list(dict.fromkeys(wk[lab_rows].tolist()))
        half_of = {w: set(y[lab_rows][wk[lab_rows] == w].tolist()) for w in wks}
        if all(len(h) == 1 for h in half_of.values()) and len(wks) >= 2:
            firsts = [w for w in wks if half_of[w] == {"first"}]
            k = len(firsts)
            n_assign = math.comb(len(wks), k)
            for _ in range(n_perm):
                chosen = set(np.array(wks)[rng.choice(len(wks), k, replace=False)].tolist()) if k else set()
                yp = y.copy()
                for i in lab_rows:
                    yp[i] = "first" if wk[i] in chosen else "second"
                perms.append(yp)
            return perms, "workload", "every workload lies inside one half: the label is a per-workload constant (ML 1.4)", n_assign
        n_assign = math.comb(len(lab_rows), int(np.sum(y[lab_rows] == "first")))
        for _ in range(n_perm):
            yp = y.copy(); yp[lab_rows] = y[lab_rows][rng.permutation(len(lab_rows))]; perms.append(yp)
        return perms, "cell", "a workload straddles the halves: the per-cell half vector permuted within the class", n_assign
    # anchor_idle and idle_early_late: cell-level shuffles across the idle cells
    labels = y[lab_rows]
    vals, counts = np.unique(labels, return_counts=True)
    n_assign = math.factorial(len(lab_rows))
    for c in counts:
        n_assign //= math.factorial(int(c))
    for _ in range(n_perm):
        yp = y.copy(); yp[lab_rows] = labels[rng.permutation(len(lab_rows))]; perms.append(yp)
    return perms, "cell", "the labels permuted across the idle cells (cell-level, as plan11's G-X)", n_assign


# --------------------------------------------------------------------------- L1 and B1-G3 for two classes (3.3.5; ML review 2.4)

def l1_feature_run(X, lab, folds, *, j: int, positive=POSITIVE, y_override=None, label_key="y", fpr=FPR_DECLARED,
                   fpr_limit=FPR_RESOLUTION_LIMIT, method=THRESHOLD_QUANTILE_METHOD, train_on_at_floor=TRAIN_ON_AT_FLOOR) -> dict:
    """The one-feature model at the operating point (ML review 2.4; ML 1.5 L1): per fold, the
    direction is the sign of the difference between the positive and benign training medians of
    feature j (L1_DIRECTION_RULE), the oriented value v = direction * x, the threshold is the
    (1 - fpr) quantile of the benign training cells' oriented values (in-fold, no fitting on
    held-out cells); a test cell is flagged when v > threshold; its score is v. Returns {'records':
    [cell_id, fold, y, score, flag_05, flag_01], 'train_auc': {fold: AUC on the training rows}}."""
    y_rows = np.asarray(y_override if y_override is not None else lab[label_key]).astype(str)
    x = np.asarray(X[:, j], dtype=np.float64)
    recs, train_auc = [], {}
    for f in folds:
        tr, te = np.asarray(f["train"], dtype=np.int64), np.asarray(f["test"], dtype=np.int64)
        tr, _ = _drop_at_floor(lab, tr, y_rows, positive, train_on_at_floor)
        pos_tr = tr[y_rows[tr] == positive]; ben_tr = tr[y_rows[tr] != positive]
        if len(pos_tr) == 0 or len(ben_tr) == 0 or len(te) == 0:
            train_auc[f["name"]] = np.nan
            for i in te:
                recs.append({"cell_id": str(lab["cell_id"][i]), "fold": f["name"], "y": str(y_rows[i]), "workload_key": str(lab["workload_key"][i]),
                             "score": None, "flag_05": None, "flag_01": None, "floor_verdict": str(lab["floor_verdict"][i])})
            continue
        d = np.nanmedian(x[pos_tr]) - np.nanmedian(x[ben_tr])
        sign = 1.0 if (np.isfinite(d) and d >= 0) else -1.0
        v = sign * x
        vb = v[ben_tr]; vb = vb[np.isfinite(vb)]
        t05 = float(np.quantile(vb, 1.0 - fpr, method=method)) if len(vb) else float("nan")
        t01 = float(np.quantile(vb, 1.0 - fpr_limit, method=method)) if len(vb) else float("nan")
        vt = v[tr]; yt = (y_rows[tr] == positive)
        ok = np.isfinite(vt)
        train_auc[f["name"]] = float(roc_auc_score(yt[ok], vt[ok])) if ok.sum() and 0 < yt[ok].sum() < ok.sum() else np.nan
        for i in te:
            s = float(v[i])
            fin = np.isfinite(s)
            recs.append({"cell_id": str(lab["cell_id"][i]), "fold": f["name"], "y": str(y_rows[i]), "workload_key": str(lab["workload_key"][i]),
                         "score": s if fin else None, "flag_05": bool(s > t05) if (fin and np.isfinite(t05)) else None,
                         "flag_01": bool(s > t01) if (fin and np.isfinite(t01)) else None, "floor_verdict": str(lab["floor_verdict"][i])})
    return {"records": recs, "train_auc": train_auc}


def _pool_l1(recs: list[dict], floor_by_cell, positive, pending_rule=True) -> dict:
    fb = floor_by_cell or {}
    scored = [r for r in recs if r["score"] is not None]
    pos_all = [r for r in recs if r["y"] == positive]
    pending = any(fb.get(r["cell_id"], r.get("floor_verdict", "")).startswith("pending") for r in pos_all)
    den = [r for r in scored if r["y"] == positive and fb.get(r["cell_id"], r.get("floor_verdict")) == V.GK0_ABOVE_FLOOR]
    ben = [r for r in scored if r["y"] != positive]
    nr = V.not_run("gk0 not run")
    tpr = nr if pending else (float(np.mean([bool(r["flag_05"]) for r in den])) if den else None)
    tpr01 = nr if pending else (float(np.mean([bool(r["flag_01"]) for r in den])) if den else None)
    fpr = float(np.mean([bool(r["flag_05"]) for r in ben])) if ben else None
    auc = None
    if den and ben:
        ys = np.array([1] * len(den) + [0] * len(ben)); ss = np.array([r["score"] for r in den] + [r["score"] for r in ben])
        auc = float(roc_auc_score(ys, ss))
    return {"tpr_05": tpr, "tpr_01": tpr01, "fpr_05_realized": fpr, "auc": auc, "n_positive_in_denominator": len(den), "n_benign_scored": len(ben)}


def l1_row(X, lab, folds, *, names=None, positive=POSITIVE, y_override=None, fpr=FPR_DECLARED, fpr_limit=FPR_RESOLUTION_LIMIT,
           method=THRESHOLD_QUANTILE_METHOD, floor_by_cell=None, train_on_at_floor=TRAIN_ON_AT_FLOOR, per_feature=False) -> dict:
    """The L1 baseline row (ML 1.5; ML review 2.4 (c)): for every feature its one-feature model
    under the same folds; per fold the best feature is chosen on the training fold by training AUC
    (L1_BEST_RULE) and its held-out flags and scores form the row. Returns {'tpr_05',
    'fpr_05_realized', 'tpr_01', 'auc', 'best_feature_by_fold', 'features_chosen', 'per_feature'
    (when asked: {feature: pooled tpr_05, fpr_05_realized, auc}), 'runs' (the per-feature runs,
    for B1-G3)}. Citation: CITATION_L1."""
    d = X.shape[1]
    names = list(names) if names is not None else [f"f{j}" for j in range(d)]
    runs = [l1_feature_run(X, lab, folds, j=j, positive=positive, y_override=y_override, fpr=fpr, fpr_limit=fpr_limit, method=method,
                           train_on_at_floor=train_on_at_floor) for j in range(d)]
    best_by_fold, recs = {}, []
    for f in folds:
        aucs = [runs[j]["train_auc"].get(f["name"], np.nan) for j in range(d)]
        aucs = [(-1.0 if not np.isfinite(a) else a) for a in aucs]
        jb = int(np.argmax(aucs)) if d else 0
        best_by_fold[f["name"]] = names[jb]
        recs += [r for r in runs[jb]["records"] if r["fold"] == f["name"]]
    out = _pool_l1(recs, floor_by_cell, positive)
    out.update({"best_feature_by_fold": best_by_fold, "features_chosen": sorted(set(best_by_fold.values())), "rule": L1_BEST_RULE,
                "direction_rule": L1_DIRECTION_RULE, "runs": runs, "records": recs})
    if per_feature:
        out["per_feature"] = {names[j]: {k: v for k, v in _pool_l1(runs[j]["records"], floor_by_cell, positive).items()
                                         if k in ("tpr_05", "fpr_05_realized", "auc")} for j in range(d)}
    return out


def quarantine_l1_two_class(l1_runs: list[dict], names: list[str], forest_records: list[dict], *,
                            max_disagree_workloads: int = B1G3_MAX_DISAGREE_WORKLOADS) -> list[dict]:
    """B1-G3 restated for two classes (CR3 2.2; ML 3.6; ML review 2.4 (a)): a feature whose
    one-feature model at the operating point (l1_feature_run) agrees with the forest's flag_05
    decisions on the cells of all but at most max_disagree_workloads held-out workloads is
    quarantined; returns [{feature, n_disagree_workloads, disagreeing_workloads, n_workloads}].
    Cells without a finite forest score or flag are not compared."""
    fflag = {r["cell_id"]: r.get("flag_05") for r in forest_records if r.get("flag_05") is not None}
    fwk = {r["cell_id"]: r["workload_key"] for r in forest_records}
    out = []
    for j, run in enumerate(l1_runs):
        dis, wks = set(), set()
        for r in run["records"]:
            c = r["cell_id"]
            if c not in fflag or r["flag_05"] is None:
                continue
            wks.add(fwk[c])
            if bool(r["flag_05"]) != bool(fflag[c]):
                dis.add(fwk[c])
        if wks and len(dis) <= int(max_disagree_workloads):
            out.append({"feature": names[j], "n_disagree_workloads": len(dis), "disagreeing_workloads": sorted(dis), "n_workloads": len(wks)})
    return out


# --------------------------------------------------------------------------- the split stage (3.3.7)

def split_dir(out: Path, rung: str, grid_id: str, name: str, normalized: bool = True) -> Path:
    return C.det_dir(out) / "splits" / rung / grid_id / f"{name}__{'norm' if normalized else 'raw'}"


def floor_verdicts(out: Path) -> dict | None:
    """{cell_id: verdict} from gates/detection/gk0_cells.csv, or None when it does not exist."""
    p = C.det_dir(out) / "gk0_cells.csv"
    if not p.is_file():
        return None
    return {r["cell_id"]: r["verdict"] for r in S.read_csv(p)}


def _grid(out: Path, rung: str, grid_id: str | None) -> tuple:
    if grid_id:
        return grid_id, C.grid_source(out)
    gid, src = S.selected_grid_id(out, rung, None)
    return gid, (C.grid_source(out) if gid else "no selection")


def load_detection_data(out: Path, rung: str, grid_id: str, *, normalized: bool = True, mask_classes=None, feature_drop=(),
                        row_unit: str = ROW_UNIT, feat_path: Path | None = None) -> dict:
    """The feature file, the join, the admissibility set and the floor verdicts as one label dict
    (detection_splits.make_detection_labels) plus the feature names and the input paths."""
    out = Path(out)
    fp = Path(feat_path) if feat_path else S.features_path(out, rung, grid_id, normalized)
    if not fp.is_file():
        raise FileNotFoundError(str(fp))
    feat = S.load_features(fp)
    join = C.load_join(out)
    adm = C.admissible_ids(out)
    if adm is None:
        raise FileNotFoundError(str(C.det_dir(out) / "admissibility.csv"))
    if rung in S.PAIR_RUNGS:
        ap = C.det_dir(out) / "admissibility.csv"
        adm = {r["cell_id"] for r in S.read_csv(ap) if str(r.get("admissible_pair_rungs", r.get("admissible"))).lower() == "true"}
    fb = floor_verdicts(out)
    names = list(feat["feature_names"])
    keep = [j for j, n in enumerate(names) if n.split(".")[-1] not in set(feature_drop) and n not in set(feature_drop)]
    if len(keep) < len(names):
        feat = dict(feat); feat["X"] = feat["X"][:, keep]; names = [names[j] for j in keep]
    lab = DS.make_detection_labels(feat, join, adm, mask_classes=mask_classes, row_unit=row_unit, floor_by_cell=fb)
    return {"lab": lab, "X": lab["X"], "names": names, "feat_path": fp, "floor_by_cell": fb, "join": join, "admissible": adm,
            "inputs": [out / "cells.csv", fp, C.join_path(out), C.det_dir(out) / "admissibility.csv", C.det_dir(out) / "gk0_cells.csv",
                       out / "gates" / "selection.json"], "head_drop_json": str(feat.get("head_drop_json", ""))}


def _class_counts(lab: dict) -> dict:
    cells = {}
    for c, k in zip(lab["cell_id"], lab["cls"]):
        cells[str(c)] = str(k)
    out: dict = {}
    for k in cells.values():
        out[k] = out.get(k, 0) + 1
    return out


def _s_b(lab: dict) -> tuple[int, int, list[str]]:
    tt = lab["split_role"] == "train_test"
    pos = sorted({str(w) for w, y in zip(lab["workload_key"][tt], lab["y"][tt]) if y == POSITIVE})
    ben = sorted({str(w) for w, y in zip(lab["workload_key"][tt], lab["y"][tt]) if y != POSITIVE})
    return len(pos), len(ben), pos + ben


def _write_na(d: Path, reason: str, params: dict, split: str, citation: str = CITATION_OP) -> Path:
    S.write_json(d / "scores.json", "plan11.detection.scores.v1", params, citation, {"status": reason, "split": split})
    return d


def _pred_rows(recs: list[dict]) -> list[dict]:
    rows = []
    for r in recs:
        rows.append({**{k: r.get(k) for k in PRED_COLUMNS}, "in_denominator": r.get("in_denominator", False)})
    return rows


def _null_block(stat_obs: dict, null_arrays: dict, n_assign: int, exhaustive: bool, n_perm_required: int) -> dict:
    block = {}
    for k in ("tpr05", "auc"):
        if k not in null_arrays:
            continue
        v, summ = null_verdict(stat_obs.get(k), null_arrays[k], n_assign, exhaustive=exhaustive, n_perm_required=n_perm_required)
        block[k] = {kk: summ[kk] for kk in ("p95", "p05", "spread", "rank", "rank_text", "n", "exceeds", "verdict", "mean", "std")}
    return block


def run_detection_split(out: Path, rung: str, grid_id: str | None = None, split: str = "lowo", *, normalized: bool = True,
                        n_perm: int = N_PERM, run_null: bool = True, seed: int = SEED_FOREST, seed_offset: int = 0, n_jobs: int = 1,
                        n_estimators: int = N_ESTIMATORS, min_samples_leaf: int = MIN_SAMPLES_LEAF, class_weight=CLASS_WEIGHT,
                        fpr_declared: float = FPR_DECLARED, fpr_limit: float = FPR_RESOLUTION_LIMIT, threshold_source: str = THRESHOLD_SOURCE,
                        threshold_quantile_method: str = THRESHOLD_QUANTILE_METHOD, score_aggregation: str = SCORE_AGGREGATION,
                        loco_mode: str = DS.LOCO_MODE, reduce_to: int | None = None, reduce_method: str = "train_importance",
                        quarantine: bool = True, feature_drop: tuple = (), subset_workloads=None, exclude_families: tuple = (),
                        label_key: str = "y", positive: str = POSITIVE, dir_name: str | None = None, row_unit: str = ROW_UNIT,
                        train_on_at_floor: bool = TRAIN_ON_AT_FLOOR, inner_k: int = INNER_K, order_class: str | None = None,
                        half_rule: str = DS.ORDER_HALF_RULE, n_perm_required: int = N_PERM, feat_path: Path | None = None,
                        extra_params: dict | None = None, det_c1_rule: str | None = None) -> Path:
    """The split stage of the detection layer (SPEC_DETECTION 3.3.7). Writes
    gates/detection/splits/<rung>/<grid_id>/<dir_name or split>__<raw|norm>/ with predictions.csv,
    scores.json, null.json, roc.csv, folds.json, l1_quarantine.json (and
    predictions_with_quarantine.csv when a feature was quarantined). Splits: lowo, loco, lofo
    (class label; thresholds set in-fold), order (label = the half by realized order within
    ``order_class``; AUC alone), anchor_idle (label = campaign on the idle cells; AUC alone),
    idle_early_late (label = the half within the idle class; AUC alone). ``subset_workloads``
    restricts the rows to those workload keys (G-LM's level band); ``exclude_families`` drops
    benign families (G-FP's recomputation); ``dir_name`` names the directory for those runs. The
    headline is the full-feature forest (score_source "full", ML review 2.4 (b)); the quarantined
    features and the re-run without them stand beside it. Written refusals: no selection for the
    rung; the feature file missing; gates/detection/gk0_cells.csv missing (al-Farabi M3); the raw
    variant of combined; no benign family under lofo; out-of-bag thresholds under window rows (ML
    review 2.1). Citation: CITATION_OP; CITATION_NULL; CITATION_L1."""
    out = Path(out)
    seed = int(seed) + int(seed_offset)
    seed_null = SEED_LABEL_NULL + int(seed_offset)
    gid, gsrc = _grid(out, rung, grid_id)
    name = dir_name or split
    class_label = label_key == "y" and split in ("lowo", "loco", "lofo")
    params = {"rung": rung, "grid_id": gid, "grid_source": gsrc, "split": split, "dir_name": name, "normalized": bool(normalized),
              "row_unit": row_unit, "cell_vector": DS.CELL_VECTOR, "n_perm": int(n_perm), "run_null": bool(run_null), "seed": seed,
              "seed_null": seed_null, "seed_offset": int(seed_offset), "n_jobs": int(n_jobs), "n_estimators": int(n_estimators),
              "min_samples_leaf": int(min_samples_leaf), "class_weight": class_weight, "max_features": "sqrt", "fpr_declared": fpr_declared,
              "fpr_limit": fpr_limit, "threshold_source": threshold_source, "threshold_source_label": THRESHOLD_SOURCE_LABELS.get(threshold_source, threshold_source),
              "inner_k": int(inner_k), "threshold_quantile_method": threshold_quantile_method,
              "score_aggregation": (V.not_applicable("row unit is the cell") if row_unit == "cell" else score_aggregation),
              "loco_mode": loco_mode, "reduce_to": reduce_to, "reduce_method": reduce_method, "quarantine": bool(quarantine),
              "b1g3_max_disagree_workloads": B1G3_MAX_DISAGREE_WORKLOADS, "feature_drop": list(feature_drop), "subset_workloads": list(subset_workloads or []),
              "exclude_families": list(exclude_families), "label_key": label_key, "positive": positive, "order_class": order_class,
              "half_rule": half_rule, "train_on_at_floor": bool(train_on_at_floor), "at_floor_positives_in_training": bool(train_on_at_floor),
              "null_denominator_rule": NULL_DENOMINATOR_RULE, "null_statistics": list(NULL_STATISTICS), "null_verdict_statistic": NULL_VERDICT_STATISTIC,
              "null_exhaustive_below": NULL_EXHAUSTIVE_BELOW, "null_min_assignments": NULL_MIN_ASSIGNMENTS, "n_perm_required": int(n_perm_required),
              "excluded_by_declaration": list(EXCLUDED_BY_DECLARATION), "gcal_spread_rule": GCAL_SPREAD_RULE, "l1_direction_rule": L1_DIRECTION_RULE,
              "l1_best_rule": L1_BEST_RULE, "det_c1_rule": det_c1_rule, "floor_source": "gates/detection/gk0_cells.csv",
              "oob_note": "the imputer and scaler are fitted on the training fold; out-of-bag rows were transformed by them"}
    params.update(extra_params or {})
    if gid is None:
        d = split_dir(out, rung, "no_selection", name, normalized); d.mkdir(parents=True, exist_ok=True)
        return _write_na(d, V.not_run(f"no selection for {rung} (run classes inherit-selection)"), params, split)
    d = split_dir(out, rung, gid, name, normalized)
    d.mkdir(parents=True, exist_ok=True)
    if rung == "combined" and not normalized:
        return _write_na(d, V.not_applicable("raw variant of combined not defined (SPEC 3.1.1)"), params, split)
    if row_unit == "window" and threshold_source == "oob" and class_label:
        return _write_na(d, V.not_applicable("out-of-bag threshold under window rows (sibling windows in bag)"), params, split)
    try:
        data = load_detection_data(out, rung, gid, normalized=normalized, feature_drop=feature_drop, row_unit=row_unit, feat_path=feat_path)
    except FileNotFoundError as e:
        return _write_na(d, V.not_run(f"input missing: {Path(str(e)).name}"), params, split)
    if data["floor_by_cell"] is None and class_label:
        return _write_na(d, V.not_run("gates/detection/gk0_cells.csv missing (run gk0-cells)"), params, split)
    lab = data["lab"]; X = data["X"]; names = data["names"]; fb = data["floor_by_cell"] or {}
    params["inputs_sha256"] = S.inputs_sha256(data["inputs"], out)
    params["head_drop_json"] = data["head_drop_json"]
    if subset_workloads:
        lab = DS.sublab(lab, np.isin(lab["workload_key"], list(subset_workloads))); X = lab["X"]
    if exclude_families:
        lab = DS.sublab(lab, ~(np.isin(lab["family"], list(exclude_families)) & (lab["y"] != positive))); X = lab["X"]
    params["class_counts"] = _class_counts(lab)
    S_n, B_n, pool = _s_b(lab)
    params.update({"S": S_n, "B": B_n, "n_assignments": count_assignments(S_n, B_n) if class_label else None, "n_cells": int(lab["n"])})
    if lab["n"] == 0:
        return _write_na(d, V.not_applicable("no admissible classed cell"), params, split)
    y_override = None
    if split in ("lowo", "loco", "lofo"):
        if split == "lowo":
            folds = DS.fold_lowo(lab)
        elif split == "loco":
            folds = DS.fold_loco(lab, loco_mode)
        else:
            folds = DS.fold_lofo(lab)
            if not folds:
                return _write_na(d, V.not_applicable("no benign family"), params, split)
    elif split == "order":
        cls = order_class or POSITIVE
        y_override = DS.order_half_labels(lab, cls, half_rule)
        folds = DS.fold_order(lab, cls, half_rule)
        positive = "second"
        if not folds:
            in_cls = np.ones(lab["n"], dtype=bool) if cls == "all" else (lab["cls"] == cls)
            return _write_na(d, V.not_run(f"order_index missing for {cls}") if not np.any(in_cls & (lab["order_index"] >= 1)) else V.not_applicable(f"{cls}: fewer than two cells with order_index"), params, split)
    elif split in ("anchor_idle", "idle_early_late"):
        cls = order_class or "idle"
        folds = DS.fold_anchor_idle(lab, cls)
        if split == "anchor_idle":
            y_override = np.where(lab["cls"] == cls, lab["campaign"].astype(str), "").astype(str)
            labels = sorted({str(c) for c in lab["campaign"][lab["cls"] == cls]})
            if len(labels) < 2:
                n_idle = int(np.sum(lab["cls"] == cls))
                return _write_na(d, V.not_applicable(f"one idle campaign (n = {n_idle} cells)"), params, split)
            positive = labels[-1]
            params["campaign_labels"] = labels
        else:
            y_override = DS.order_half_labels(lab, cls, "within_class")
            positive = "second"
            if not np.any((lab["cls"] == cls) & (lab["order_index"] >= 1)):
                return _write_na(d, V.not_run(f"order_index missing for {cls}"), params, split)
            folds = DS.fold_order(lab, cls, "within_class")
        if not folds:
            return _write_na(d, V.not_applicable(f"fewer than two {cls} cells"), params, split)
    else:
        raise ValueError(split)
    params["positive"] = positive
    op_kw = dict(seed=seed, n_jobs=n_jobs, n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, class_weight=class_weight,
                 fpr_declared=fpr_declared, fpr_limit=fpr_limit, threshold_source=threshold_source, method=threshold_quantile_method,
                 score_rule=score_aggregation, positive=positive, label_key=label_key, reduce_to=reduce_to, reduce_method=reduce_method,
                 train_on_at_floor=train_on_at_floor, inner_k=inner_k, set_thresholds=class_label)
    res = run_operating_point(X, lab, folds, y_override=y_override, **op_kw)
    summ = summarize_operating_point(res["records"], res["folds"], fb, fpr_declared=fpr_declared, fpr_limit=fpr_limit, positive=positive, class_label=class_label)
    payload: dict = {"status": "ok", "split": split, "dir_name": name, "n_folds": len(folds), "grid_source": gsrc, "seed": seed, "n_perm": int(n_perm),
                     "feature_count": int(X.shape[1]), "feature_count_used": int(min(f["d_used"] for f in res["folds"])) if res["folds"] else int(X.shape[1]),
                     "feature_names": names, "importance_mean": (dict(zip(names, res["importance_mean"])) if res["importance_mean"] and len(res["importance_mean"]) == len(names) else None),
                     "score_source": SCORE_SOURCE_FULL, "row_unit": row_unit, "S": S_n, "B": B_n, "positive": positive}
    payload["dim_status"] = V.GDIM_REDUCED if payload["feature_count_used"] < payload["feature_count"] else V.GDIM_FULL
    recs = summ.pop("records")
    payload.update(summ)
    if split == "lofo":
        for k in ("tpr_05", "tpr_01", "tpr_at_fpr05_pooled"):
            payload[k] = V.not_applicable("sandbox never held out under LOFO")
        payload["per_member"] = {}
    # the L1 row and B1-G3
    l1 = None; quarantined = []
    if class_label:
        l1 = l1_row(X, lab, folds, names=names, positive=positive, fpr=fpr_declared, fpr_limit=fpr_limit, method=threshold_quantile_method,
                    floor_by_cell=fb, train_on_at_floor=train_on_at_floor, per_feature=True)
        if quarantine:
            quarantined = quarantine_l1_two_class(l1["runs"], names, recs, max_disagree_workloads=B1G3_MAX_DISAGREE_WORKLOADS)
        payload["l1"] = {k: l1[k] for k in ("tpr_05", "fpr_05_realized", "tpr_01", "auc", "best_feature_by_fold", "features_chosen", "rule", "direction_rule", "per_feature")}
        if split == "lofo":
            payload["l1"]["tpr_05"] = payload["l1"]["tpr_01"] = V.not_applicable("sandbox never held out under LOFO")
    payload["quarantine"] = {"quarantined_features": [q["feature"] for q in quarantined], "n_disagree_by_feature": {q["feature"]: q["n_disagree_workloads"] for q in quarantined}}
    S.write_json(d / "l1_quarantine.json", "plan11.detection.l1_quarantine.v1", params, CITATION_L1,
                 {"quarantined": quarantined, "max_disagree_workloads": B1G3_MAX_DISAGREE_WORKLOADS, "n_features": len(names),
                  "excluded_by_declaration": list(EXCLUDED_BY_DECLARATION)})
    # the null
    null_payload = {"status": V.not_run("null not requested"), "n_assignments": params.get("n_assignments"), "exhaustive": None, "n_distinct_drawn": None}
    null_arrays: dict = {}
    if run_null and int(n_perm) > 0 and class_label and split in ("lowo", "loco"):
        rng = np.random.default_rng(seed_null)
        subsets, exhaustive = workload_label_permutations(pool, S_n, int(n_perm), rng)
        n_assign = count_assignments(S_n, B_n)
        if subsets:
            null_arrays = null_distribution(X, lab, folds, subsets, floor_by_cell=fb, null_jobs=n_jobs, l1=True, **op_kw)
            obs = {"tpr05": payload["tpr_05"] if isinstance(payload["tpr_05"], float) else None, "auc": payload["auc"]}
            block = _null_block(obs, null_arrays, n_assign, exhaustive, n_perm_required)
            l1obs = {"tpr05": payload["l1"]["tpr_05"] if isinstance(payload["l1"]["tpr_05"], float) else None, "auc": payload["l1"]["auc"]}
            l1block = _null_block(l1obs, {"tpr05": null_arrays["l1_tpr05"], "auc": null_arrays["l1_auc"]}, n_assign, exhaustive, n_perm_required)
            null_payload = {"status": "ok", **block, "n_assignments": n_assign, "exhaustive": exhaustive, "n_distinct_drawn": len(set(subsets)),
                            "n_perm": len(subsets), "S": S_n, "B": B_n, "verdict": block.get(NULL_VERDICT_STATISTIC, {}).get("verdict"), "l1": l1block}
            payload["l1"]["null"] = l1block
        else:
            null_payload = {"status": V.not_applicable("no workload to permute"), "n_assignments": n_assign, "exhaustive": True, "n_distinct_drawn": 0}
    elif class_label and split in ("lowo", "loco"):
        null_payload["status"] = V.not_run("null not requested")
    elif split in ("order", "anchor_idle", "idle_early_late") and run_null and int(n_perm) > 0:
        rng = np.random.default_rng(seed_null)
        perms, unit, reason, n_assign = binary_label_permutations(lab, y_override, split, half_rule, int(n_perm), rng)
        vals = []
        for yp in perms:
            rp = run_operating_point(X, lab, folds, y_override=yp, **op_kw)
            sp = summarize_operating_point(rp["records"], rp["folds"], fb, fpr_declared=fpr_declared, fpr_limit=fpr_limit, positive=positive, class_label=False)
            vals.append(sp["auc"] if sp["auc"] is not None else np.nan)
        null_arrays = {"auc": np.asarray(vals, dtype=np.float64)}
        block = _null_block({"auc": payload["auc"]}, null_arrays, n_assign, False, n_perm_required)
        null_payload = {"status": "ok", **block, "n_assignments": n_assign, "exhaustive": False, "n_distinct_drawn": len({tuple(x) for x in perms}), "n_perm": len(perms),
                        "null_unit": unit, "null_unit_reason": reason, "verdict": block.get("auc", {}).get("verdict")}
    elif split in ("order", "anchor_idle", "idle_early_late"):
        null_payload = {"status": V.not_run("null not requested"), "n_assignments": None, "null_unit": ""}
    payload["null"] = null_payload
    # G-CAL inside the split (3.3.8)
    gc = payload.get("gcal") or {}
    spread = null_payload.get("tpr05", {}).get("spread") if isinstance(null_payload.get("tpr05"), dict) else None
    pf, po = gc.get("per_fold_tpr05"), gc.get("pooled_tpr_at_fpr05")
    if isinstance(pf, float) and po is not None:
        diff = float(pf - po)
        if spread is None:
            gv = V.not_run("null spread unavailable")
        else:
            gv = V.GCAL_AGREE if abs(diff) <= spread else V.GCAL_PERFOLD
        payload["gcal"] = {"per_fold_tpr05": pf, "pooled_tpr_at_fpr05": po, "difference": diff, "null_spread": spread, "verdict": gv, "spread_rule": GCAL_SPREAD_RULE}
    elif class_label:
        payload["gcal"] = {"per_fold_tpr05": pf, "pooled_tpr_at_fpr05": po, "difference": None, "null_spread": spread,
                           "verdict": (pf if isinstance(pf, str) else V.not_applicable("no operating point")), "spread_rule": GCAL_SPREAD_RULE}
    # the re-run without the quarantined features (both readings kept; the headline stays the full model)
    payload["with_quarantine"] = None
    if quarantined and class_label:
        qset = {q["feature"] for q in quarantined}
        keep = [j for j, n in enumerate(names) if n not in qset]
        if keep:
            res_q = run_operating_point(X[:, keep], lab, folds, y_override=y_override, **op_kw)
            summ_q = summarize_operating_point(res_q["records"], res_q["folds"], fb, fpr_declared=fpr_declared, fpr_limit=fpr_limit, positive=positive)
            recs_q = summ_q.pop("records")
            wq = {k: summ_q[k] for k in ("tpr_05", "fpr_05_realized", "tpr_01", "fpr_01_realized", "auc", "tpr_at_fpr05_pooled", "per_member", "per_family_fpr", "per_workload_outcome")}
            wq.update({"quarantined_features": sorted(qset), "feature_count": len(keep), "status": "ok"})
            if null_arrays and split in ("lowo", "loco"):
                nq = null_distribution(X[:, keep], lab, folds, subsets, floor_by_cell=fb, null_jobs=n_jobs, l1=False, **op_kw)
                wq["null"] = _null_block({"tpr05": wq["tpr_05"] if isinstance(wq["tpr_05"], float) else None, "auc": wq["auc"]}, nq, n_assign, exhaustive, n_perm_required)
            payload["with_quarantine"] = wq
            S.write_csv(d / M.PRED_WITH_QUARANTINE, PRED_COLUMNS, _pred_rows(recs_q))
        else:
            payload["with_quarantine"] = {"quarantined_features": sorted(qset), "status": V.not_run("every feature quarantined")}
    S.write_csv(d / "predictions.csv", PRED_COLUMNS, _pred_rows(recs))
    if payload.get("roc"):
        S.write_csv(d / "roc.csv", ("fpr", "tpr", "threshold"), [{"fpr": a, "tpr": b, "threshold": c} for a, b, c in zip(payload["roc"]["fpr"], payload["roc"]["tpr"], payload["roc"]["threshold"])])
    else:
        S.write_csv(d / "roc.csv", ("fpr", "tpr", "threshold"), [])
    payload.pop("roc", None)
    S.write_json(d / "folds.json", "plan11.detection.folds.v1", params, CITATION_OP, {"folds": res["folds"]})
    S.write_json(d / "null.json", "plan11.detection.null.v1", params, CITATION_NULL,
                 {**{k: v for k, v in null_payload.items()}, "arrays": {k: v.tolist() for k, v in null_arrays.items()}})
    S.write_json(d / "scores.json", "plan11.detection.scores.v1", params, CITATION_OP, payload)
    return d


def read_scores(out: Path, rung: str, grid_id: str, name: str, normalized: bool = True) -> dict | None:
    p = split_dir(out, rung, grid_id, name, normalized) / "scores.json"
    return S.read_json(p) if p.is_file() else None


def strongest_single_rung(out: Path, grid_ids: dict, *, statistic: str = NULL_VERDICT_STATISTIC) -> tuple:
    """The strongest single rung under LOWO (norm) by the verdict statistic, AUC the tie-break
    (CR3 2.32's G-DIM; SPEC 3.7.7): returns (rung, d*) or (None, None)."""
    best = None
    for r in ("apf", "wapf", "persist", "content"):
        gid = grid_ids.get(r)
        if not gid:
            continue
        sc = read_scores(out, r, gid, "lowo", True)
        if not sc or sc.get("status") != "ok":
            continue
        t = sc.get("tpr_05") if statistic == "tpr05" else sc.get("auc")
        key = (t if isinstance(t, (int, float)) else -1.0, sc.get("auc") if isinstance(sc.get("auc"), (int, float)) else -1.0)
        if best is None or key > best[0]:
            best = (key, r, int(sc.get("feature_count_used") or sc.get("feature_count")))
    return (best[1], best[2]) if best else (None, None)


# --------------------------------------------------------------------------- the one-class run (3.3.6)

def make_one_class(model: str = ONE_CLASS_MODEL, seed: int = SEED_FOREST, n_estimators: int = N_ESTIMATORS):
    """'isolation_forest': Pipeline(SimpleImputer(median), StandardScaler(), IsolationForest(n_estimators=300,
    contamination='auto', random_state=seed)); score = -score_samples (higher = more anomalous).
    'gmm': GaussianMixture(n_components=4, covariance_type='diag', n_init=5, random_state=seed) on the
    standardized rows, score = -score_samples (the diagonal covariance is the ML review's condition
    for a full-rank density on about 100 rows, for-the-author item 6). 'ocsvm': OneClassSVM(kernel='rbf',
    gamma='scale', nu=0.05), score = -decision_function. One is primary (ONE_CLASS_MODEL); any other
    is run only as secondary and labelled G1C_SECONDARY. Returns (pipeline, score_fn). Citation: CITATION_1C."""
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    if model == "isolation_forest":
        from sklearn.ensemble import IsolationForest
        est = IsolationForest(n_estimators=int(n_estimators), contamination="auto", random_state=int(seed))
        fn = lambda p, X: -p.score_samples(X)
    elif model == "gmm":
        from sklearn.mixture import GaussianMixture
        est = GaussianMixture(n_components=4, covariance_type="diag", n_init=5, random_state=int(seed))
        fn = lambda p, X: -p.score_samples(X)
    elif model == "ocsvm":
        from sklearn.svm import OneClassSVM
        est = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
        fn = lambda p, X: -p.decision_function(X)
    else:
        raise ValueError(model)
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("m", est)]), fn


def _one_class_scores(X, lab, folds, *, model, seed, n_estimators, threshold_source, fpr, fpr_limit, method, y_rows) -> dict:
    """Per benign-workload fold the model is fitted on the other benign cells and scores the held-out
    workload; the final fold's model is fitted on every benign cell and scores every positive cell.
    Threshold 'inner_lowo': the (1 - fpr) quantile of the benign cells' out-of-fold scores from the
    per-workload folds; 'train_in_sample': the quantile of the final model's training scores."""
    ben_oof: dict = {}
    pos_scores: dict = {}
    fin_train: dict = {}
    for f in folds:
        tr, te = np.asarray(f["train"], dtype=np.int64), np.asarray(f["test"], dtype=np.int64)
        tr = tr[y_rows[tr] != POSITIVE]
        if len(tr) < 2 or len(te) == 0:
            continue
        pipe, fn = make_one_class(model, seed, n_estimators)
        n_comp = getattr(pipe.named_steps["m"], "n_components", None)
        if n_comp is not None and len(tr) < n_comp:
            pipe.named_steps["m"].set_params(n_components=max(1, len(tr)))
        pipe.fit(X[tr])
        sc = cell_scores(lab["cell_id"][te], fn(pipe, X[te]), "window_mean_proba")
        if f["name"].endswith("/final"):
            pos_scores.update(sc)
            fin_train = cell_scores(lab["cell_id"][tr], fn(pipe, X[tr]), "window_mean_proba")
        else:
            ben_oof.update(sc)
    src = ben_oof if threshold_source == "inner_lowo" else fin_train
    t05 = in_fold_threshold(src, fpr, method); t01 = in_fold_threshold(src, fpr_limit, method)
    return {"ben_oof": ben_oof, "pos": pos_scores, "t05": t05, "t01": t01}


def run_one_class(out: Path, rung: str, grid_id: str | None = None, *, model: str = ONE_CLASS_MODEL, primary: bool = True,
                  normalized: bool = True, threshold_source: str = ONE_CLASS_THRESHOLD_SOURCE, fpr_declared: float = FPR_DECLARED,
                  fpr_limit: float = FPR_RESOLUTION_LIMIT, seed: int = SEED_FOREST, seed_offset: int = 0, n_jobs: int = 1,
                  n_perm: int = N_PERM, n_estimators: int = N_ESTIMATORS, threshold_quantile_method: str = THRESHOLD_QUANTILE_METHOD,
                  n_perm_required: int = N_PERM, row_unit: str = ROW_UNIT) -> Path:
    """The one-class run (CR3 2.19; ML 1.3; K3 section 5 point 1): fold_one_class; per benign
    workload fold the model is fitted on the other benign cells and scores the held-out workload
    (its FPR); the final model is fitted on every benign cell and scores every sandbox and
    test_only cell, so every sandbox cell is scored by a model that never saw a sandbox cell.
    Threshold: 'inner_lowo' (the honest analogue of the forest's in-fold rule; the same threshold
    serves the final model) or 'train_in_sample'. Null (ML review 2.7; al-Kindi 2.2): the same
    workload-level subsets as LOWO (workload_label_permutations, the same seed); per subset the
    permuted-benign workloads are the training pool and the permuted positives are scored; tpr05
    and auc recorded, null_verdict as 3.3.4. Writes the split directory one_class (primary) or
    one_class__<model> (secondary) with predictions.csv, scores.json (model, primary, g1c_label,
    null), null.json, roc.csv, folds.json. Citation: CITATION_1C."""
    out = Path(out)
    seed = int(seed) + int(seed_offset); seed_null = SEED_LABEL_NULL + int(seed_offset)
    gid, gsrc = _grid(out, rung, grid_id)
    name = "one_class" if primary else f"one_class__{model}"
    params = {"rung": rung, "grid_id": gid, "grid_source": gsrc, "split": "one_class", "model": model, "primary": bool(primary),
              "one_class_model_declared": ONE_CLASS_MODEL, "threshold_source": threshold_source, "fpr_declared": fpr_declared, "fpr_limit": fpr_limit,
              "seed": seed, "seed_null": seed_null, "seed_offset": int(seed_offset), "n_perm": int(n_perm), "n_estimators": int(n_estimators),
              "threshold_quantile_method": threshold_quantile_method, "row_unit": row_unit, "normalized": bool(normalized),
              "null_denominator_rule": NULL_DENOMINATOR_RULE, "excluded_by_declaration": list(EXCLUDED_BY_DECLARATION), "n_perm_required": int(n_perm_required)}
    if gid is None:
        d = split_dir(out, rung, "no_selection", name, normalized); d.mkdir(parents=True, exist_ok=True)
        return _write_na(d, V.not_run(f"no selection for {rung} (run classes inherit-selection)"), params, "one_class", CITATION_1C)
    d = split_dir(out, rung, gid, name, normalized); d.mkdir(parents=True, exist_ok=True)
    if rung == "combined" and not normalized:
        return _write_na(d, V.not_applicable("raw variant of combined not defined (SPEC 3.1.1)"), params, "one_class", CITATION_1C)
    try:
        data = load_detection_data(out, rung, gid, normalized=normalized, row_unit=row_unit)
    except FileNotFoundError as e:
        return _write_na(d, V.not_run(f"input missing: {Path(str(e)).name}"), params, "one_class", CITATION_1C)
    if data["floor_by_cell"] is None:
        return _write_na(d, V.not_run("gates/detection/gk0_cells.csv missing (run gk0-cells)"), params, "one_class", CITATION_1C)
    lab, X, fb = data["lab"], data["X"], data["floor_by_cell"]
    params["inputs_sha256"] = S.inputs_sha256(data["inputs"], out); params["class_counts"] = _class_counts(lab)
    S_n, B_n, pool = _s_b(lab)
    params.update({"S": S_n, "B": B_n, "n_assignments": count_assignments(S_n, B_n)})
    if B_n == 0 or S_n == 0:
        return _write_na(d, V.not_applicable("one-class needs at least one benign and one positive workload"), params, "one_class", CITATION_1C)
    folds = DS.fold_one_class(lab)
    y = lab["y"].astype(str)
    r = _one_class_scores(X, lab, folds, model=model, seed=seed, n_estimators=n_estimators, threshold_source=threshold_source, fpr=fpr_declared,
                          fpr_limit=fpr_limit, method=threshold_quantile_method, y_rows=y)
    recs = []
    fold_of = {}
    for f in folds:
        for i in f["test"]:
            fold_of[str(lab["cell_id"][i])] = f["name"]
    for i in range(lab["n"]):
        c = str(lab["cell_id"][i])
        s = r["pos"].get(c, r["ben_oof"].get(c, float("nan")))
        recs.append(_record(lab, i, fold_of.get(c, ""), y[i], s, r["t05"], r["t01"], "ok" if np.isfinite(s) else NO_SCORE))
    summ = summarize_operating_point(recs, [], fb, fpr_declared=fpr_declared, fpr_limit=fpr_limit)
    recs2 = summ.pop("records")
    payload = {"status": "ok", "split": "one_class", "dir_name": name, "model": model, "primary": bool(primary),
               "g1c_label": V.G1C_PRIMARY if primary else V.G1C_SECONDARY, "threshold_source": threshold_source, "threshold_05": r["t05"],
               "threshold_01": r["t01"], "n_folds": len(folds), "grid_source": gsrc, "seed": seed, "n_perm": int(n_perm), "feature_count": int(X.shape[1]),
               "feature_count_used": int(X.shape[1]), "dim_status": V.GDIM_FULL, "score_source": SCORE_SOURCE_FULL, "row_unit": row_unit, "S": S_n, "B": B_n}
    payload.update(summ)
    payload["gop"] = {"note": V.not_applicable("one-class threshold set on the benign out-of-fold scores, not on training setters")}
    null_payload = {"status": V.not_run("null not requested"), "n_assignments": params["n_assignments"]}
    arrays = {}
    if int(n_perm) > 0:
        rng = np.random.default_rng(seed_null)
        subsets, exhaustive = workload_label_permutations(pool, S_n, int(n_perm), rng)
        n_assign = count_assignments(S_n, B_n)

        def one(sub):
            yp = _y_from_subset(lab, sub)
            lab_p = dict(lab); lab_p["y"] = yp
            fp = DS.fold_one_class(lab_p)
            rp = _one_class_scores(X, lab, fp, model=model, seed=seed, n_estimators=n_estimators, threshold_source=threshold_source,
                                   fpr=fpr_declared, fpr_limit=fpr_limit, method=threshold_quantile_method, y_rows=yp)
            rr = []
            for i in range(lab["n"]):
                c = str(lab["cell_id"][i]); s = rp["pos"].get(c, rp["ben_oof"].get(c, float("nan")))
                rr.append(_record(lab, i, "", yp[i], s, rp["t05"], rp["t01"], "ok" if np.isfinite(s) else NO_SCORE))
            sp = summarize_operating_point(rr, [], fb, fpr_declared=fpr_declared, fpr_limit=fpr_limit)
            return {"tpr05": sp["tpr_05"] if isinstance(sp["tpr_05"], float) else np.nan, "auc": sp["auc"] if sp["auc"] is not None else np.nan}
        if subsets:
            if int(n_jobs) > 1 and len(subsets) > 1:
                from joblib import Parallel, delayed
                vals = Parallel(n_jobs=int(n_jobs))(delayed(one)(s) for s in subsets)
            else:
                vals = [one(s) for s in subsets]
            arrays = {k: np.asarray([v[k] for v in vals], dtype=np.float64) for k in ("tpr05", "auc")}
            block = _null_block({"tpr05": payload["tpr_05"] if isinstance(payload["tpr_05"], float) else None, "auc": payload["auc"]}, arrays, n_assign, exhaustive, n_perm_required)
            null_payload = {"status": "ok", **block, "n_assignments": n_assign, "exhaustive": exhaustive, "n_distinct_drawn": len(set(subsets)),
                            "n_perm": len(subsets), "S": S_n, "B": B_n, "verdict": block.get(NULL_VERDICT_STATISTIC, {}).get("verdict")}
    payload["null"] = null_payload
    S.write_csv(d / "predictions.csv", PRED_COLUMNS, _pred_rows(recs2))
    roc = payload.pop("roc", None)
    S.write_csv(d / "roc.csv", ("fpr", "tpr", "threshold"), [{"fpr": a, "tpr": b, "threshold": c} for a, b, c in zip(roc["fpr"], roc["tpr"], roc["threshold"])] if roc else [])
    S.write_json(d / "folds.json", "plan11.detection.folds.v1", params, CITATION_1C, {"folds": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in f.items()} for f in folds]})
    S.write_json(d / "null.json", "plan11.detection.null.v1", params, CITATION_NULL, {**null_payload, "arrays": {k: v.tolist() for k, v in arrays.items()}})
    S.write_json(d / "scores.json", "plan11.detection.scores.v1", params, CITATION_1C, payload)
    return d


# --------------------------------------------------------------------------- the time-to-detect ladder (3.3.9)

def prefix_rows(n_pairs_cell: int, prefix_s: int, dt) -> int:
    """The prefix length in rows: round(prefix_s / dt) with dt a fixed spacing (LADDER_DT_S =
    0.644 s, the derived guest spacing; 0.500 s the configured interval), so that a 30 s prefix is
    47 rows, 60 s 93, 120 s 186, 300 s 466, 600 s 932 (K3 move 17 names about 46, 93, 186, 465 and
    930 at 645 ms), capped at the cell's extract length ``n_pairs_cell`` (al-Farabi M6: the capped
    prefix is the whole cell, so the 600 s reading equals the headline's features); with dt =
    "per_cell" the cell's own spacing 600 / n_pairs_cell is used (round(prefix_s * n_pairs_cell / 600))."""
    n = int(n_pairs_cell)
    if dt == "per_cell":
        k = int(round(float(prefix_s) * n / float(schema.DURATION_S)))
    else:
        k = int(round(float(prefix_s) / float(dt)))
    return max(0, min(k, n))


def boundary_start(cell_id: str, boundaries: dict, seq_first: int) -> int:
    """The row index of the first [SUSTAIN] boundary from inputs/iteration_boundaries.csv
    (cell_id, boundary_seqs as ';'-separated ascending seqs), or 0 when the cell has no boundary (a
    kernel honouring --duration has one iteration and nothing to drop; CR3 2.29 applies the drop
    rule to every cell uniformly)."""
    b = boundaries.get(cell_id)
    if not b:
        return 0
    return max(0, int(b[0]) - int(seq_first))


def load_boundaries(path: Path) -> dict:
    if not Path(path).is_file():
        return {}
    out = {}
    for r in S.read_csv(path):
        seqs = [int(float(x)) for x in str(r.get("boundary_seqs", "")).split(";") if x.strip()]
        if seqs:
            out[r["cell_id"]] = sorted(seqs)
    return out


def slice_extract(ex: dict, start: int, n: int) -> dict:
    """The extract dict sliced to rows [start, start + n) on every array; ``_n_rows`` = the rows kept."""
    out = {}
    lo, hi = int(start), int(start) + int(n)
    for k, v in ex.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == int(ex["_n_rows"]):
            out[k] = v[lo:hi]
        else:
            out[k] = v
    out["_n_rows"] = int(max(0, min(hi, int(ex["_n_rows"])) - lo)) if lo < int(ex["_n_rows"]) else 0
    return out


def prefix_features_path(out: Path, rung: str, grid_id: str, normalized: bool, prefix_s: int, reading: str) -> Path:
    return Path(out) / "detection" / "features" / rung / f"{grid_id}_{'norm' if normalized else 'raw'}_prefix{int(prefix_s)}s_{reading}.npz"


def build_prefix_features(out: Path, rung: str, grid_id: str, normalized: bool, n_rows_by_cell: dict, reading: str, *,
                          prefix_s: int, cells=None, norm: str = LADDER_NORM, head_drop=None, wapf_norm: str = S.WAPF_NORM_DEFAULT) -> Path:
    """The ladder's prefix features (SPEC_DETECTION 3.1.3): each cell's extract truncated to rows
    [start, start + n) (slice_extract), then series.rung_series and series.window_features exactly
    as series.build_features does, so the normalization of a prefix uses the prefix's own median K
    (``norm = "prefix"``; ``"whole_cell"`` divides by the whole cell's median instead). Under the
    ``from_boundary`` reading the boundary start replaces the head drop (LADDER_HEAD_DROP_RULE).
    The npz has the layout of SPEC 3.1.5 plus the scalars prefix_s, reading, n_rows_by_cell_json."""
    out = Path(out)
    if cells is None:
        cells = S.load_cells(out / "cells.csv")
    W, H = S.parse_grid_id(grid_id)
    names = S.feature_names(rung, normalized)
    Xs, meta = [], {k: [] for k in ("cell_id", "kernel", "archetype", "campaign", "role", "rep", "win_start", "n_series_cell")}
    n_dropped = 0
    for c in cells:
        if c["cell_id"] not in n_rows_by_cell:
            continue
        start, n = n_rows_by_cell[c["cell_id"]]
        ex = S.load_extract_cached(out, c["cell_id"])
        hd = S.head_drop_for(head_drop, c["kernel"]) if reading == "from_pair1" else 0
        exs = slice_extract(ex, start, n)
        if norm == "whole_cell" and normalized and rung in ("apf", "wapf", "combined"):
            # divide by the whole cell's median K: rescale the prefix series by kmed_prefix / kmed_whole
            Sx = S.rung_series(exs, rung, normalized=True, head_drop=hd, wapf_norm=wapf_norm)
            kp, kw = S.k_median_cell(exs, hd), S.k_median_cell(ex, hd)
            if rung == "apf":
                Sx = Sx * (kp / kw) if kw else Sx
            elif rung == "wapf":
                Sx = Sx * (kp / kw) if kw else Sx
            else:
                Sx[:, 0] = Sx[:, 0] * (kp / kw) if kw else Sx[:, 0]
                Sx[:, 1] = Sx[:, 1] * (kp / kw) if kw else Sx[:, 1]
        else:
            Sx = S.rung_series(exs, rung, normalized=normalized, head_drop=hd, wapf_norm=wapf_norm)
        ns = Sx.shape[0]
        if ns == 0:
            continue
        w, h = S.resolve_wh(W, H, ns)
        F, starts, nd = S.window_features(Sx, rung, w, h)
        n_dropped += nd
        if len(F) == 0:
            continue
        Xs.append(F); k = len(F)
        meta["cell_id"] += [c["cell_id"]] * k; meta["kernel"] += [c["kernel"]] * k; meta["archetype"] += [c["archetype_predicted"]] * k
        meta["campaign"] += [c["campaign"]] * k; meta["role"] += [c["role"]] * k; meta["rep"] += [int(c["rep"])] * k
        meta["win_start"] += starts.tolist(); meta["n_series_cell"] += [ns] * k
    X = np.concatenate(Xs, axis=0) if Xs else np.zeros((0, len(names)))
    p = prefix_features_path(out, rung, grid_id, normalized, prefix_s, reading)
    p.parent.mkdir(parents=True, exist_ok=True)
    hd_json = json.dumps(head_drop if isinstance(head_drop, dict) else {"all": head_drop or 0}, sort_keys=True)
    np.savez(p, X=X, feature_names=np.array(names), cell_id=np.array(meta["cell_id"]), kernel=np.array(meta["kernel"]),
             archetype=np.array(meta["archetype"]), campaign=np.array(meta["campaign"]), role=np.array(meta["role"]),
             rep=np.array(meta["rep"], dtype=np.int64), win_start=np.array(meta["win_start"], dtype=np.int64),
             n_series_cell=np.array(meta["n_series_cell"], dtype=np.int64), W=np.array(-1 if W is None else W), H=np.array(-1 if H is None else H),
             grid_id=np.array(grid_id), normalized=np.array(bool(normalized)), head_drop_json=np.array(hd_json), n_windows_dropped=np.array(n_dropped),
             wapf_norm=np.array(wapf_norm), prefix_s=np.array(int(prefix_s)), reading=np.array(reading),
             n_rows_by_cell_json=np.array(json.dumps({k: list(v) for k, v in n_rows_by_cell.items()}, sort_keys=True)))
    return p


LADDER_COLUMNS = ("rung", "grid_id", "reading", "prefix_s", "dt", "n_rows_median", "n_windows_median", "n_cells_with_window",
                  "tpr_05", "fpr_05_realized", "auc", "null_verdict", "note")


def run_ladder(out: Path, rung: str, grid_id: str | None = None, *, prefixes_s=LADDER_PREFIXES_S, dt=LADDER_DT_S, readings=LADDER_READINGS,
               norm: str = LADDER_NORM, n_perm: int = 0, normalized: bool = True, n_jobs: int = 1, seed_offset: int = 0, **split_kw) -> Path:
    """The time-to-detect ladder (K3 move 17; CR3 2.29; N3 Sec. 1 RQ1, Figure 6). For each reading
    and each prefix: build the prefix features (build_prefix_features), run run_detection_split
    with split lowo (and its null only when n_perm > 0) into ladder/<rung>/<grid_id>/<reading>/prefix<T>s/,
    and write ladder.csv (one row per rung, reading, prefix: rung, grid_id, reading, prefix_s, dt,
    n_rows_median, n_windows_median, n_cells_with_window, tpr_05, fpr_05_realized, auc,
    null_verdict, note) and ladder.json (the per-cell pair counts at both bracket spacings and
    per cell). The reading 'from_boundary' runs only when inputs/iteration_boundaries.csv exists and
    names at least one admissible cell; otherwise its rows carry note = LADDER_FROM_PAIR1_ONLY and
    empty numbers. A prefix shorter than one window at the rung's (W, H) for every cell reads
    not_applicable('prefix shorter than one window (n = <k> < W = <W>)'); cells without a window at
    a prefix are dropped for that prefix and counted in n_cells_with_window. Citation: CITATION_LADDER."""
    out = Path(out)
    gid, gsrc = _grid(out, rung, grid_id)
    lp = C.det_dir(out) / "ladder.csv"
    old = [r for r in S.read_csv(lp) if r.get("rung") != rung] if lp.is_file() else []
    params = {"rung": rung, "grid_id": gid, "grid_source": gsrc, "prefixes_s": list(prefixes_s), "dt": dt, "readings": list(readings),
              "ladder_norm": norm, "ladder_head_drop_rule": LADDER_HEAD_DROP_RULE, "n_perm": int(n_perm), "normalized": bool(normalized),
              "dt_bracket_s": list(schema.DT_BRACKET_S), "seed_offset": int(seed_offset)}
    rows = []
    if gid is None:
        rows.append({"rung": rung, "grid_id": "", "reading": "", "prefix_s": "", "dt": dt, "tpr_05": V.not_run(f"no selection for {rung} (run classes inherit-selection)"),
                     "null_verdict": "", "note": ""})
        S.write_csv(lp, LADDER_COLUMNS, old + rows); S.write_params(lp, "plan11.detection.ladder.v1", params, CITATION_LADDER)
        return lp
    cells = S.load_cells(out / "cells.csv")
    adm = C.admissible_ids(out) or set()
    cells = [c for c in cells if c["cell_id"] in adm]
    W, H = S.parse_grid_id(gid)
    bpath = out / "inputs" / "iteration_boundaries.csv"
    boundaries = load_boundaries(bpath)
    hd = S.load_head_drop(out / "inputs" / "head_drop.csv")
    per_cell = {}
    for c in cells:
        ex = S.load_extract_cached(out, c["cell_id"])
        n = int(ex["_n_rows"])
        sc = S.load_sidecar(out, c["cell_id"])
        per_cell[c["cell_id"]] = {"n_rows": n, "seq_first": int(sc.get("seq_first") or ex["seq"][0] if n else 0),
                                  "pairs_at_0500": {str(p): prefix_rows(n, p, schema.DT_BRACKET_S[0]) for p in prefixes_s},
                                  "pairs_at_0644": {str(p): prefix_rows(n, p, schema.DT_BRACKET_S[1]) for p in prefixes_s},
                                  "pairs_per_cell": {str(p): prefix_rows(n, p, "per_cell") for p in prefixes_s},
                                  "boundary_start": boundary_start(c["cell_id"], boundaries, int(sc.get("seq_first") or 0))}
    have_boundary = any(per_cell[c]["boundary_start"] > 0 for c in per_cell)
    for reading in readings:
        if reading == "from_boundary" and not (bpath.is_file() and have_boundary):
            for p in prefixes_s:
                rows.append({"rung": rung, "grid_id": gid, "reading": reading, "prefix_s": p, "dt": dt, "tpr_05": "", "fpr_05_realized": "", "auc": "",
                             "null_verdict": "", "note": V.LADDER_FROM_PAIR1_ONLY})
            continue
        for p in prefixes_s:
            nr = {}
            for c in cells:
                info = per_cell[c["cell_id"]]
                start = info["boundary_start"] if reading == "from_boundary" else 0
                n = prefix_rows(info["n_rows"] - start, p, dt)
                nr[c["cell_id"]] = (start, n)
            n_rows_list = [n for _, n in nr.values()]
            w_eff = W if W is not None else max(n_rows_list)
            if all((n - 1) < (W or 1) for n in n_rows_list):
                k = int(np.median(n_rows_list)) if n_rows_list else 0
                rows.append({"rung": rung, "grid_id": gid, "reading": reading, "prefix_s": p, "dt": dt, "n_rows_median": int(np.median(n_rows_list)) if n_rows_list else "",
                             "n_windows_median": 0, "n_cells_with_window": 0, "tpr_05": V.not_applicable(f"prefix shorter than one window (n = {max(k, 0)} < W = {w_eff})"),
                             "fpr_05_realized": "", "auc": "", "null_verdict": "", "note": ""})
                continue
            fp = build_prefix_features(out, rung, gid, normalized, nr, reading, prefix_s=p, cells=cells, norm=norm, head_drop=hd)
            feat = S.load_features(fp)
            n_cells_win = len(set(feat["cell_id"].tolist()))
            counts = [int(np.sum(feat["cell_id"] == c)) for c in set(feat["cell_id"].tolist())]
            d = run_detection_split(out, rung, gid, "lowo", normalized=normalized, n_perm=n_perm, run_null=n_perm > 0, n_jobs=n_jobs, seed_offset=seed_offset,
                                    dir_name=f"ladder_{reading}_prefix{int(p)}s", feat_path=fp,
                                    extra_params={"ladder_prefix_s": int(p), "ladder_reading": reading, "ladder_dt": dt, "ladder_norm": norm}, **split_kw)
            sc = S.read_json(d / "scores.json")
            nv = (sc.get("null") or {}).get("verdict") if sc.get("status") == "ok" else ""
            rows.append({"rung": rung, "grid_id": gid, "reading": reading, "prefix_s": p, "dt": dt, "n_rows_median": int(np.median(n_rows_list)),
                         "n_windows_median": int(np.median(counts)) if counts else 0, "n_cells_with_window": n_cells_win,
                         "tpr_05": sc.get("tpr_05") if sc.get("status") == "ok" else sc.get("status"), "fpr_05_realized": sc.get("fpr_05_realized") if sc.get("status") == "ok" else "",
                         "auc": sc.get("auc") if sc.get("status") == "ok" else "", "null_verdict": nv or V.not_run("ladder null not requested"), "note": ""})
            # the split directory lives under ladder/<rung>/<grid>/<reading>/prefix<T>s/ as SPEC 1.4 says: move it
            target = C.det_dir(out) / "ladder" / rung / gid / reading / f"prefix{int(p)}s"
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                import shutil
                shutil.rmtree(target)
            Path(d).rename(target)
    S.write_csv(lp, LADDER_COLUMNS, old + rows)
    S.write_params(lp, "plan11.detection.ladder.v1", {**params, "inputs_sha256": S.inputs_sha256([out / "cells.csv", bpath, out / "gates" / "selection.json"], out)}, CITATION_LADDER)
    jp = C.det_dir(out) / "ladder.json"
    jdoc = S.read_json(jp) if jp.is_file() else {"per_rung": {}}
    jdoc.setdefault("per_rung", {})[rung] = {"rows": rows, "per_cell": per_cell, "have_boundary": have_boundary}
    S.write_json(jp, "plan11.detection.ladder.v1", params, CITATION_LADDER, {"per_rung": jdoc["per_rung"]})
    return lp


# --------------------------------------------------------------------------- CLI

def _parse_list(s: str | None) -> tuple:
    if not s:
        return ()
    return tuple(x.strip() for x in s.split(",") if x.strip())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="detection_metrics.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("splits")
    a.add_argument("--out", required=True); a.add_argument("--rung", required=True, choices=S.RUNGS); a.add_argument("--grid-id", default=None)
    g = a.add_mutually_exclusive_group(required=True)
    g.add_argument("--split", choices=("lowo", "loco", "lofo")); g.add_argument("--all-splits", action="store_true")
    v = a.add_mutually_exclusive_group()
    v.add_argument("--raw", action="store_true"); v.add_argument("--norm", action="store_true"); v.add_argument("--raw-and-norm", action="store_true")
    a.add_argument("--null-perm", type=int, default=N_PERM); a.add_argument("--null-splits", default="lowo,loco")
    a.add_argument("--n-jobs", type=int, default=1); a.add_argument("--n-estimators", type=int, default=N_ESTIMATORS)
    a.add_argument("--min-samples-leaf", type=int, default=MIN_SAMPLES_LEAF); a.add_argument("--class-weight", default="none", choices=("none", "balanced"))
    a.add_argument("--threshold-source", default=THRESHOLD_SOURCE, choices=THRESHOLD_SOURCES)
    a.add_argument("--score-aggregation", default=SCORE_AGGREGATION, choices=("window_mean_proba", "window_median_proba", "vote_fraction"))
    a.add_argument("--row-unit", default=ROW_UNIT, choices=("cell", "window"))
    a.add_argument("--train-on-at-floor", default="false" if not TRAIN_ON_AT_FLOOR else "true", choices=("true", "false"))
    a.add_argument("--loco-mode", default=DS.LOCO_MODE, choices=("cell", "rep_index")); a.add_argument("--reduce-to-strongest", action="store_true")
    a.add_argument("--feature-drop", default=""); a.add_argument("--seed-offset", type=int, default=0); a.add_argument("--n-perm-required", type=int, default=N_PERM)
    b = sub.add_parser("one-class")
    b.add_argument("--out", required=True); b.add_argument("--rung", required=True, choices=S.RUNGS); b.add_argument("--grid-id", default=None)
    b.add_argument("--model", default=ONE_CLASS_MODEL, choices=ONE_CLASS_MODELS); b.add_argument("--secondary", action="store_true")
    b.add_argument("--threshold-source", default=ONE_CLASS_THRESHOLD_SOURCE, choices=("inner_lowo", "train_in_sample"))
    b.add_argument("--null-perm", type=int, default=N_PERM); b.add_argument("--n-jobs", type=int, default=1); b.add_argument("--n-estimators", type=int, default=N_ESTIMATORS)
    b.add_argument("--seed-offset", type=int, default=0); b.add_argument("--n-perm-required", type=int, default=N_PERM); b.add_argument("--raw", action="store_true")
    c = sub.add_parser("ladder")
    c.add_argument("--out", required=True); c.add_argument("--rung", required=True, choices=S.RUNGS); c.add_argument("--grid-id", default=None)
    c.add_argument("--prefixes", default=",".join(str(p) for p in LADDER_PREFIXES_S)); c.add_argument("--dt", default=str(LADDER_DT_S))
    c.add_argument("--norm", default=LADDER_NORM, choices=("prefix", "whole_cell")); c.add_argument("--null-perm", type=int, default=0)
    c.add_argument("--n-jobs", type=int, default=1); c.add_argument("--seed-offset", type=int, default=0); c.add_argument("--n-estimators", type=int, default=N_ESTIMATORS)
    c.add_argument("--threshold-source", default=THRESHOLD_SOURCE, choices=THRESHOLD_SOURCES)
    args = ap.parse_args(argv)
    out = Path(args.out)
    if not (out / "cells.csv").is_file():
        print(f"missing input: {out / 'cells.csv'}", file=sys.stderr); return 2
    try:
        if args.cmd == "splits":
            gid = args.grid_id or S.selected_grid_id(out, args.rung, None)[0]
            variants = ("raw", "norm") if args.raw_and_norm else (("raw",) if args.raw else ("norm",))
            splits_ = ("lowo", "loco", "lofo") if args.all_splits else (args.split,)
            null_splits = set(_parse_list(args.null_splits))
            cw = None if args.class_weight == "none" else "balanced"
            reduce_to = None; dir_name = None
            if args.reduce_to_strongest:
                if args.rung != "combined":
                    print("usage: --reduce-to-strongest applies to --rung combined", file=sys.stderr); return 3
                grid_ids = {r: S.selected_grid_id(out, r, None)[0] for r in ("apf", "wapf", "persist", "content")}
                strongest, dstar = strongest_single_rung(out, grid_ids)
                if strongest is None:
                    d = split_dir(out, "combined", gid or "no_selection", "lowo_matched", True); d.mkdir(parents=True, exist_ok=True)
                    _write_na(d, V.not_run("no single-rung LOWO scores to match (run the four single rungs first)"), {"rung": "combined"}, "lowo")
                    print(d); return 0
                reduce_to, dir_name = dstar, "lowo_matched"
                splits_ = ("lowo",)
            for var in variants:
                for sp in splits_:
                    d = run_detection_split(out, args.rung, gid, sp, normalized=(var == "norm"), n_perm=args.null_perm, run_null=(sp in null_splits),
                                            n_jobs=args.n_jobs, n_estimators=args.n_estimators, min_samples_leaf=args.min_samples_leaf, class_weight=cw,
                                            threshold_source=args.threshold_source, score_aggregation=args.score_aggregation, loco_mode=args.loco_mode,
                                            reduce_to=reduce_to, dir_name=dir_name, feature_drop=_parse_list(args.feature_drop), seed_offset=args.seed_offset,
                                            row_unit=args.row_unit, train_on_at_floor=(args.train_on_at_floor == "true"), n_perm_required=args.n_perm_required,
                                            extra_params={"matched_to_rung": strongest} if args.reduce_to_strongest else None)
                    print(d)
            return 0
        if args.cmd == "one-class":
            gid = args.grid_id or S.selected_grid_id(out, args.rung, None)[0]
            primary = not args.secondary
            if not primary and args.model == ONE_CLASS_MODEL:
                print(f"usage: --secondary cannot name the declared primary model {ONE_CLASS_MODEL}", file=sys.stderr); return 3
            if primary and args.model != ONE_CLASS_MODEL:
                prim = split_dir(out, args.rung, gid or "no_selection", "one_class", not args.raw)
                if prim.is_dir():
                    print("usage: a second one-class model needs --secondary", file=sys.stderr); return 3
                print(f"usage: {args.model} is not the declared primary ({ONE_CLASS_MODEL}); pass --secondary", file=sys.stderr); return 3
            d = run_one_class(out, args.rung, gid, model=args.model, primary=primary, normalized=not args.raw, threshold_source=args.threshold_source,
                              n_perm=args.null_perm, n_jobs=args.n_jobs, n_estimators=args.n_estimators, seed_offset=args.seed_offset, n_perm_required=args.n_perm_required)
            print(d); return 0
        if args.cmd == "ladder":
            gid = args.grid_id or S.selected_grid_id(out, args.rung, None)[0]
            dt = "per_cell" if args.dt == "per_cell" else float(args.dt)
            p = run_ladder(out, args.rung, gid, prefixes_s=tuple(int(x) for x in _parse_list(args.prefixes)), dt=dt, norm=args.norm, n_perm=args.null_perm,
                           n_jobs=args.n_jobs, seed_offset=args.seed_offset, n_estimators=args.n_estimators, threshold_source=args.threshold_source)
            print(p); return 0
    except FileNotFoundError as e:
        print(f"missing input: {e}", file=sys.stderr); return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
