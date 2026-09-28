#!/usr/bin/env python3
"""models.py -- the forest, the one-feature threshold model (B1's L1), unit aggregation, scores,
the split stage with B1-G1 / B1-G3 / B1-G6 at the unit, and clustering with ARI and NMI
(SPEC sections 4.2 to 4.5).

Citation: P2_STRUCTURE.md section V 'Models' and 'The splits'; SPEC 4.2 (the forest), 4.3 (L1),
4.4 (clustering), 4.5 (the split stage); CR 2.1 items 9, 10, 11 (B1-G1, B1-G3, B1-G6 restated at
the unit of the split); CR 2.2 item 32 (G-DIM's reduction).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plan11_encoding_ladder import series as S
from plan11_encoding_ladder import splits as SP
from plan11_encoding_ladder import nulls as NL
from plan11_encoding_ladder import verdicts as V
from plan11_encoding_ladder.nulls import SEED_FOREST, SEED_LABEL_NULL

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

N_ESTIMATORS = 300                       # section 8 item 23
UNIT_AGGREGATION = "cell_majority"       # section 8 item 22
B1G1_MIN_PERM = 500                      # CR 2.1 item 9
B1G3_MAX_DISAGREE = 1                    # section 8 item 24
DIM_MATCH_METHOD = "train_importance"    # section 8 item 28
CLUSTER_ALGOS = ("kmeans", "gmm", "agglomerative")
CLUSTER_PRIMARY = "kmeans"               # section 8 item 31
GN_HEADLINE_MIN_KERNELS = 3              # CR 2.2 item 25
CITATION_SPLIT = ("P2 Sec. V 'Models', 'The splits'; SPEC 4.2-4.5; CR 2.1 items 9 (B1-G1), 10 (B1-G3), "
                  "11 (B1-G6); CR 2.2 item 32 (G-DIM reduction)")
CITATION_CLUSTER = "P2 Sec. V 'Models' (clustering, k fixed, one primary algorithm, ARI and NMI against the unit-level null); SPEC 4.4"


# --------------------------------------------------------------------------- models (4.2, 4.3)

def make_forest(seed: int = SEED_FOREST, n_jobs: int = 1, n_estimators: int = N_ESTIMATORS, *,
                oob_score: bool = False, min_samples_leaf: int = 1, class_weight=None) -> Pipeline:
    """Pipeline(SimpleImputer(strategy='median'), StandardScaler(),
    RandomForestClassifier(n_estimators=300, max_depth=None, max_features='sqrt',
    min_samples_leaf=1, class_weight=None, bootstrap=True, random_state=seed,
    n_jobs=n_jobs)). n_estimators=300 and the imputer+scaler wrapping follow
    plan04_classify.py line 131 and plan05_campaign/peakvar_lift.py line 41; the scaler is
    fitted on the training fold only (b1_ae.py's rule). Citation: P2 Sec. V 'Models'.
    ``n_estimators`` is a parameter so a smoke run can be cheap; the default is the SPEC's 300
    and the value used is written into ``params``.

    The keyword-only ``oob_score``, ``min_samples_leaf`` and ``class_weight`` are the additive
    extension of SPEC_DETECTION.md 1.3 for the detection layer (out-of-bag scoring for the
    in-fold threshold, ML 1.6 item 2; the leaf size and class weight declared before the data,
    ML 3.7 question 12); their defaults reproduce the present forest exactly."""
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=n_estimators, max_depth=None, max_features="sqrt",
                                      min_samples_leaf=int(min_samples_leaf), class_weight=class_weight,
                                      bootstrap=True, oob_score=bool(oob_score),
                                      random_state=seed, n_jobs=n_jobs)),
    ])


def make_l1(n_classes: int, seed: int = SEED_FOREST) -> Pipeline:
    """B1's L1: ``DecisionTreeClassifier(max_depth=None, max_leaf_nodes=n_classes,
    random_state=seed)`` on one feature column (median-imputed), the feature chosen on the
    training fold by training accuracy (B1 doc, model L1). With more than two classes a single
    threshold cannot express the labels, so the tree carries at most n_classes - 1 thresholds
    (SPEC 4.3; section 8 item 24)."""
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("tree", DecisionTreeClassifier(max_depth=None, max_leaf_nodes=max(2, n_classes), random_state=seed)),
    ])


# --------------------------------------------------------------------------- reduction (G-DIM, 3.7.7)

def _reduce_fit(Xtr, ytr, d_target: int, method: str, seed: int, n_estimators: int):
    """Fit a per-fold feature reduction on the training fold only. Returns a transform closure."""
    d = Xtr.shape[1]
    if d_target >= d:
        return lambda X: X
    if method == "train_importance":
        f = make_forest(seed, 1, n_estimators).fit(Xtr, ytr)
        imp = f.named_steps["rf"].feature_importances_
        keep = np.sort(np.argsort(-imp, kind="stable")[:d_target])
        return lambda X: X[:, keep]
    if method == "pca":
        pre = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]).fit(Xtr)
        pca = PCA(n_components=d_target, random_state=seed).fit(pre.transform(Xtr))
        return lambda X: pca.transform(pre.transform(X))
    raise ValueError(method)


reduce_fit = _reduce_fit     # public alias for the detection layer (SPEC_DETECTION.md 1.3)


# --------------------------------------------------------------------------- unit aggregation (4.2)

def aggregate_units(cell_ids, y_pred_win, proba_win, classes, rule: str = UNIT_AGGREGATION) -> dict:
    """The unit prediction is the majority vote over the cell's windows (``cell_majority``; ties
    broken by the higher mean predicted probability, then by class name order); ``cell_mean_proba``
    takes the argmax of the mean probability. Returns {cell: (y_pred, vote_fraction)}.
    Citation: P2 Sec. V 'The splits' (unit = cell); SPEC 4.2 (unit aggregation); SPEC section 8 item 22."""
    cell_ids = np.asarray(cell_ids).astype(str)
    classes = list(classes)
    out = {}
    for c in dict.fromkeys(cell_ids.tolist()):
        m = cell_ids == c
        preds = np.asarray(y_pred_win)[m]
        mp = np.asarray(proba_win)[m].mean(axis=0)
        if rule == "cell_mean_proba":
            j = int(np.argmax(mp))
            out[c] = (classes[j], float(np.mean(preds == classes[j])))
            continue
        vals, counts = np.unique(preds, return_counts=True)
        top = counts.max()
        cands = [v for v, n in zip(vals, counts) if n == top]
        if len(cands) > 1:
            cands.sort(key=lambda v: (-mp[classes.index(v)], v))
        out[c] = (str(cands[0]), float(top / len(preds)))
    return out


def fit_predict_units(X, y_win, folds, cell_ids, *, seed=SEED_FOREST, n_jobs=1,
                      n_estimators=N_ESTIMATORS, reduce_to=None, reduce_method=DIM_MATCH_METHOD,
                      auto_reduce=True, rule=UNIT_AGGREGATION, model="forest") -> dict:
    """Fit on window rows of the training cells; predict every window of the test cells; aggregate
    per unit (SPEC 4.2). Returns {cell: {'y_pred', 'vote_fraction', 'fold', 'd_used'}}. When
    ``reduce_to`` is set, or when ``auto_reduce`` and d exceeds the fold's training cell count,
    the features are reduced per fold on the training fold only (SPEC 3.7.7)."""
    cell_ids = np.asarray(cell_ids).astype(str)
    y_win = np.asarray(y_win).astype(str)
    res = {}
    for f in folds:
        tr, te = f["train"], f["test"]
        if len(tr) == 0 or len(te) == 0:
            continue
        Xtr, Xte = X[tr], X[te]
        n_train_cells = len(set(cell_ids[tr].tolist()))
        d_target = Xtr.shape[1]
        if reduce_to is not None:
            d_target = min(d_target, int(reduce_to))
        if auto_reduce and d_target > n_train_cells:
            d_target = n_train_cells
        if d_target < Xtr.shape[1]:
            T = _reduce_fit(Xtr, y_win[tr], d_target, reduce_method, seed, n_estimators)
            Xtr, Xte = T(Xtr), T(Xte)
        if model == "forest":
            clf = make_forest(seed, n_jobs, n_estimators).fit(Xtr, y_win[tr])
        else:
            clf = model.fit(Xtr, y_win[tr])
        proba = clf.predict_proba(Xte)
        classes = list(clf.classes_)
        pred = np.array(classes)[np.argmax(proba, axis=1)]
        agg = aggregate_units(cell_ids[te], pred, proba, classes, rule)
        for c, (p, vf) in agg.items():
            res[c] = {"y_pred": p, "vote_fraction": vf, "fold": f["name"], "d_used": int(Xtr.shape[1])}
    return res


def score_units(y_true: dict, y_pred: dict, kernel_of: dict, headline_classes=None) -> dict:
    """Scores at the unit (SPEC 4.2): ``accuracy`` (fraction of test cells correctly labelled),
    ``recall_per_class``, ``macro_recall`` (over the headline classes when given, else all classes),
    ``recall_per_kernel`` (fraction of the kernel's cells correct)."""
    cells = [c for c in y_true if c in y_pred]
    if not cells:
        return {"accuracy": None, "recall_per_class": {}, "macro_recall": None, "recall_per_kernel": {}, "n_units": 0}
    correct = {c: (y_true[c] == y_pred[c]) for c in cells}
    acc = float(np.mean([correct[c] for c in cells]))
    rpc = {}
    for cls in sorted(set(y_true[c] for c in cells)):
        m = [correct[c] for c in cells if y_true[c] == cls]
        rpc[cls] = float(np.mean(m))
    if headline_classes is not None:
        hc = [c for c in headline_classes if c in rpc]
        macro = float(np.mean([rpc[c] for c in hc])) if hc else None
    else:
        macro = float(np.mean(list(rpc.values())))
    rpk = {}
    for k in sorted(set(kernel_of[c] for c in cells)):
        m = [correct[c] for c in cells if kernel_of[c] == k]
        rpk[k] = float(np.mean(m))
    return {"accuracy": acc, "recall_per_class": rpc, "macro_recall": macro, "recall_per_kernel": rpk,
            "n_units": len(cells)}


def majority_baseline(y_unit: dict, kernel_of: dict, arche_of: dict, split: str, labelspace: str, folds, lab) -> float:
    """B1-G6 at the unit (CR 2.1 item 11; SPEC 3.7.3). LOKO/archetype: the most populous archetype
    by kernel count among the training kernels of each fold, scored on the held-out kernel; LORO and
    within-trace, kernel space: the most populous kernel by cell count; archetype space under LORO
    or within-trace: the most populous archetype by cell count."""
    cells = list(y_unit)
    if split == "loko":
        correct = []
        for f in folds:
            tr_cells = set(lab["cell_id"][f["train"]].tolist())
            te_cells = [c for c in cells if c in set(lab["cell_id"][f["test"]].tolist())]
            tr_kernels = {kernel_of[c]: y_unit[c] for c in tr_cells}   # the label space in use (archetype; campaign under G-X, CHECK_2 M15)
            counts = {}
            for k, a in tr_kernels.items():
                counts[a] = counts.get(a, 0) + 1
            top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if counts else None
            correct += [y_unit[c] == top for c in te_cells]
        return float(np.mean(correct)) if correct else None
    counts = {}
    for c in cells:
        counts[y_unit[c]] = counts.get(y_unit[c], 0) + 1
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return float(np.mean([y_unit[c] == top for c in cells]))


def b1_g1_verdict(observed: float, null: np.ndarray, n_perm_required: int = B1G1_MIN_PERM) -> tuple:
    """B1-G1 at the unit (CR 2.1 item 9; SPEC 3.7.1): ``pass`` when the observed unit score strictly
    exceeds the null's 95th percentile (ties fail), else ``NEAR_UNFALSIFIABLE``; fewer than 500
    permutations -> ``not run: N permutations < 500`` (a smoke run). Returns (verdict, summary)."""
    summ = NL.null_summary(observed, null)
    n = summ["n"]
    if n < n_perm_required:
        return V.not_run(f"{n} permutations < {n_perm_required}"), summ
    return (V.PASS if summ["exceeds"] else V.NEAR_UNFALSIFIABLE), summ


def headline_classes_of(arche_of_kernel: dict) -> list[str]:
    """Archetypes with at least three kernels (G-N headline rows; CR 2.2 item 25)."""
    counts = {}
    for k, a in arche_of_kernel.items():
        counts[a] = counts.get(a, 0) + 1
    return sorted(a for a, n in counts.items() if n >= GN_HEADLINE_MIN_KERNELS)


# --------------------------------------------------------------------------- the split stage (4.5)

def prepare_split_data(out: Path, rung: str, grid_id: str, *, normalized: bool = True,
                       feature_drop=(), include_idle: bool = False, cells=None, feat=None) -> dict:
    """Load the feature file and apply, at read time, admissibility (preconditions.csv, the
    failed-verdict exclusion for pair rungs), the role filter, G-K0's relabelling and the feature
    drop. Nothing here is stored back into the feature file."""
    out = Path(out)
    if feat is None:
        p = S.features_path(out, rung, grid_id, normalized)
        if not p.is_file():
            raise FileNotFoundError(p)
        feat = S.load_features(p)
    if cells is None:
        cells = S.load_cells(out / "cells.csv")
    kept, ex_hard, ex_pair, pre_present = S.admissible_cells(out, cells, rung)
    kept_ids = {c["cell_id"] for c in kept}
    mask = np.array([cid in kept_ids for cid in feat["cell_id"]])
    if not include_idle:
        mask &= feat["role"] == "kernel"
    relabel = S.gk0_relabel(out)
    archetype = feat["archetype"].copy()
    for k, a in relabel.items():
        archetype[feat["kernel"] == k] = a
    names = list(feat["feature_names"])
    keep_cols = [j for j, n in enumerate(names) if n.split(".")[-1] not in set(feature_drop)]
    X = feat["X"][:, keep_cols]
    names = [names[j] for j in keep_cols]
    lab = SP.make_labels({**feat, "archetype": archetype}, mask)
    rows = lab["_rows"]
    cells_in_play = [c for c in kept if (include_idle or c["role"] == "kernel")]
    cells_no_windows = [c["cell_id"] for c in cells_in_play if c["cell_id"] not in set(lab["cell_id"].tolist())]
    return {
        "X": X[rows], "names": names, "lab": lab, "feat": feat, "cells": cells_in_play,
        "cells_no_windows": cells_no_windows, "excluded_hard": ex_hard, "excluded_pair_rungs": ex_pair,
        "preconditions_present": pre_present, "relabelled_kernels": sorted(relabel), "gk0_applied": bool(relabel),
        "feature_drop": list(feature_drop),
    }


def _unit_labels(lab: dict, labelspace: str) -> tuple:
    cells = list(dict.fromkeys(lab["cell_id"].tolist()))
    kernel_of = {c: str(lab["kernel"][lab["cell_id"] == c][0]) for c in cells}
    arche_of = {c: str(lab["archetype"][lab["cell_id"] == c][0]) for c in cells}
    camp_of = {c: str(lab["campaign"][lab["cell_id"] == c][0]) for c in cells}
    rep_of = {c: int(lab["rep"][lab["cell_id"] == c][0]) for c in cells}
    if labelspace == "kernel":
        y_unit = dict(kernel_of)
    elif labelspace == "archetype":
        y_unit = dict(arche_of)
    elif labelspace == "campaign":
        y_unit = dict(camp_of)
    else:
        raise ValueError(labelspace)
    return cells, y_unit, kernel_of, arche_of, camp_of, rep_of


def _rows_from_units(lab: dict, y_unit: dict) -> np.ndarray:
    return np.array([y_unit[c] for c in lab["cell_id"]])


def _point_score(X, lab, y_unit, folds, kernel_of, headline, **fit_kw) -> tuple:
    preds = fit_predict_units(X, _rows_from_units(lab, y_unit), folds, lab["cell_id"], **fit_kw)
    y_pred = {c: v["y_pred"] for c, v in preds.items()}
    return score_units(y_unit, y_pred, kernel_of, headline), preds


def _null_scores(X, lab, y_unit, folds, kernel_of, split, labelspace, n_perm, seed_null, n_jobs, fit_kw) -> np.ndarray:
    """B1-G1's unit-level shuffle null: n_perm permutations of the per-cell label vector (CR 2.1
    item 9), rows inherit, the same folds, the same model and seed; the score is unit accuracy."""
    cells = list(y_unit)
    kern = np.array([kernel_of[c] for c in cells])
    labs = np.array([y_unit[c] for c in cells])
    rng = np.random.default_rng(seed_null)
    perms = [NL.shuffle_labels_units(np.array(cells), kern, labs, split, labelspace, rng) for _ in range(n_perm)]

    def one(pl):
        yu = {c: str(pl[i]) for i, c in enumerate(cells)}
        sc, _ = _point_score(X, lab, yu, folds, kernel_of, None, **fit_kw)
        return sc["accuracy"] if sc["accuracy"] is not None else np.nan
    if n_jobs and n_jobs > 1 and n_perm > 1:
        from joblib import Parallel, delayed
        vals = Parallel(n_jobs=n_jobs)(delayed(one)(pl) for pl in perms)
    else:
        vals = [one(pl) for pl in perms]
    return np.asarray(vals, dtype=np.float64)


def quarantine_l1(X, names, lab, y_unit, folds, full_preds: dict, *, seed=SEED_FOREST,
                  max_disagree: int = B1G3_MAX_DISAGREE) -> list[dict]:
    """B1-G3 restated at the unit (CR 2.1 item 10; SPEC 3.7.2): for each feature, a one-feature
    threshold model (4.3) fitted on the training fold; the feature whose model reproduces the full
    model's per-unit predictions on all but at most ``max_disagree`` units (over all folds) is
    quarantined. Returns [{feature, n_disagree, n_units}] for the quarantined features."""
    y_win = _rows_from_units(lab, y_unit)
    n_classes = len(set(y_unit.values()))
    out = []
    for j, name in enumerate(names):
        preds = fit_predict_units(X[:, [j]], y_win, folds, lab["cell_id"], seed=seed,
                                  model=make_l1(n_classes, seed), auto_reduce=False)
        units = [c for c in full_preds if c in preds]
        n_dis = sum(1 for c in units if preds[c]["y_pred"] != full_preds[c]["y_pred"])
        if units and n_dis <= max_disagree:
            out.append({"feature": name, "n_disagree": int(n_dis), "n_units": len(units)})
    return out


def effective_scores(sc: dict | None) -> dict | None:
    """The rung's score once B1-G3 has run (SPEC 3.7.2, CR 2.1 item 10: "the split is re-run without
    the quarantined features and both scores are kept ... Table rows use the re-run"). When
    ``scores.json["with_quarantine"]`` carries a non-empty ``quarantined_features``, the re-run's
    keys (accuracy, macro_recall, recall_per_class, recall_per_kernel, majority, b1_g1, b1_g1_rank,
    null_p95, feature_count, status) are merged over the top-level (full-model) keys and
    ``quarantined_features`` is carried; otherwise the document is returned as written. Every
    consumer of a split's score reads through this function: the comparison gates G-L, G-DIM, G-M
    (gates_comparison._scores), the tables (_report_common.effective_scores delegates here) and
    the excluded-rows list (run_split_stage), so that one score is the rung's score (CHECK_2.md B1).
    The full model's keys stay in the document under their own names for the record. When every
    feature was quarantined there is no re-run (``with_quarantine.status`` = ``not run: every feature
    quarantined``): the score keys become None and ``b1_g1`` carries that string, so no consumer
    judges or prints the full model as the rung's score."""
    if sc is None:
        return None
    wq = sc.get("with_quarantine")
    if isinstance(wq, dict) and wq.get("quarantined_features"):
        merged = dict(sc)
        if "accuracy" not in wq:                      # every feature quarantined: no re-run exists
            merged.update({"accuracy": None, "macro_recall": None, "recall_per_class": {}, "recall_per_kernel": {},
                           "majority": None, "b1_g1_rank": None, "null_p95": None,
                           "b1_g1": wq.get("status") or V.not_run("every feature quarantined")})
        for k, v in wq.items():
            if k != "quarantined_features":
                merged[k] = v
        merged["quarantined_features"] = list(wq["quarantined_features"])
        merged["score_source"] = SCORE_SOURCE_QUARANTINE
        return merged
    return sc


SCORE_SOURCE_QUARANTINE = "with_quarantine (the re-run without the quarantined features; SPEC 3.7.2)"
SCORE_SOURCE_FULL = "full model (no feature quarantined by B1-G3)"
PRED_WITH_QUARANTINE = "predictions_with_quarantine.csv"


def split_dir(out: Path, rung: str, grid_id: str, split: str, labelspace: str, base: str = "splits",
              normalized: bool = True) -> Path:
    """The split stage directory (SPEC 4.5): ``gates/<base>/<rung>/<grid_id>/<split>__<labelspace>/``
    for the normalized run and ``<split>__<labelspace>__raw/`` for the raw (level-inclusive) run, so
    that APF's ``--raw-and-norm`` keeps both (Table 6's raw columns, SPEC 6.2; G-L (i)'s
    level-inclusive ceiling, CR 2.2 item 24; CHECK_1.md B1). The report layer reads ``__raw`` first."""
    return Path(out) / "gates" / base / rung / grid_id / (f"{split}__{labelspace}" + ("" if normalized else "__raw"))


def run_split_stage(out: Path, rung: str, grid_id: str, split: str, labelspace: str, *,
                    normalized: bool = True, n_perm: int = B1G1_MIN_PERM, seed: int = SEED_FOREST,
                    n_jobs: int = 1, feature_drop: tuple = (), include_idle: bool = False,
                    n_estimators: int = N_ESTIMATORS, run_null: bool = True, seed_offset: int = 0,
                    reduce_to: int | None = None, reduce_method: str = DIM_MATCH_METHOD,
                    test_frac: float = SP.TEST_FRAC, loro_mode: str = SP.LORO_MODE,
                    base_dir: str = "splits", label_override: dict | None = None,
                    quarantine: bool = True, unit_rule: str = UNIT_AGGREGATION,
                    grid_source: str = "argument") -> Path:
    """The split stage (SPEC 4.5): writes ``gates/<base_dir>/<rung>/<grid_id>/<split>__<labelspace>/``
    (``<split>__<labelspace>__raw/`` when ``normalized`` is false; CHECK_1.md B1)
    ``predictions.csv`` (cell_id, kernel, archetype, campaign, rep, fold, y_true, y_pred, n_windows,
    vote_fraction, held_out_campaign), ``scores.json`` (accuracy, macro_recall, recall_per_class,
    recall_per_kernel, majority, b1_g1, b1_g1_rank, null_p95, with_quarantine, feature_count,
    dim_status, seed, n_perm, n_cells_no_windows), ``null.json`` and ``l1_quarantine.json``.
    Unit = cell (P2 Sec. V 'The splits'). LOKO in kernel space is ``not applicable: held-out label
    unseen``; within-trace at the whole-cell point is ``not applicable: one window per cell``.
    ``label_override`` = {cell_id: label} replaces the label space (G-X's campaign labels).
    ``params`` records gk0_applied, relabelled_kernels, excluded_cells_pair_rungs, gc_verdict and
    inputs_sha256 (SPEC_review_al_farabi.md items 2.3, 2.4, 2.5, 2.8). When B1-G3 quarantines a
    feature (SPEC 3.7.2) the re-run's per-unit predictions are written beside the full model's as
    ``predictions_with_quarantine.csv`` (same columns; the tables read it first, CHECK_2.md B1) and
    the NEAR_UNFALSIFIABLE exclusion follows the re-run's verdict (effective_scores).
    Build epoch 2 (SPEC_epoch2 3.5.2; E1 sec. 4 M5; CHECK_2 M3): ``scores.json`` and
    ``with_quarantine`` also carry ``feature_count_used`` (the ``d_used`` the forest saw: an int when
    every fold used the same width, else ``"<min>-<max>"``) and ``feature_count_used_per_fold``
    (``{fold: d_used}``); ``effective_scores`` carries them like every other key and Table 7's
    ``feature count`` prints ``feature_count_used`` when present (the pre-reduction width stays in
    ``feature_count`` and in the G-DIM cell). Additive only; the signature is unchanged."""
    out = Path(out)
    d = split_dir(out, rung, grid_id, split, labelspace, base_dir, normalized=normalized)
    d.mkdir(parents=True, exist_ok=True)
    feat_path = S.features_path(out, rung, grid_id, normalized)
    data = prepare_split_data(out, rung, grid_id, normalized=normalized, feature_drop=feature_drop, include_idle=include_idle)
    X, names, lab = data["X"], data["names"], data["lab"]
    params = {
        "rung": rung, "grid_id": grid_id, "split": split, "labelspace": labelspace, "normalized": normalized,
        "n_perm": n_perm, "seed": seed + seed_offset, "seed_label_null": SEED_LABEL_NULL + seed_offset,
        "seed_offset": seed_offset, "n_jobs": n_jobs, "n_estimators": n_estimators,
        "feature_drop": list(feature_drop), "include_idle": include_idle, "unit_aggregation": unit_rule,
        "test_frac": test_frac, "loro_mode": loro_mode, "b1g3_max_disagree": B1G3_MAX_DISAGREE,
        "b1g1_min_perm": B1G1_MIN_PERM, "reduce_to": reduce_to, "reduce_method": reduce_method,
        "gk0_applied": data["gk0_applied"], "relabelled_kernels": data["relabelled_kernels"],
        "excluded_cells_hard": data["excluded_hard"], "excluded_cells_pair_rungs": data["excluded_pair_rungs"],
        "preconditions_present": data["preconditions_present"], "gc_verdict": S.gc_verdict(out, rung),
        "label_override": bool(label_override), "quarantine": quarantine,
        "grid_source": grid_source,      # SPEC_epoch2 B18 (CERT 6.13, 7.5): "selection.json" from the splits CLI, else "argument"
        "score_after_quarantine": "scores.json with_quarantine is the rung's score and " + PRED_WITH_QUARANTINE
                                  + " its predictions when a feature is quarantined (SPEC 3.7.2; models.effective_scores)",
        "inputs_sha256": S.inputs_sha256([out / "cells.csv", feat_path, out / "gates" / "preconditions.csv",
                                          out / "gates" / "gk0.csv", out / "gates" / "gc.csv"], out),
    }

    def write_na(reason: str) -> Path:
        S.write_json(d / "scores.json", "plan11.scores.v1", params, CITATION_SPLIT, {
            "status": V.not_applicable(reason), "accuracy": None, "macro_recall": None,
            "recall_per_class": {}, "recall_per_kernel": {}, "majority": None,
            "b1_g1": V.not_applicable(reason), "b1_g1_rank": None, "null_p95": None,
            "with_quarantine": None, "feature_count": len(names), "dim_status": None,
            "seed": seed + seed_offset, "n_perm": n_perm, "n_cells_no_windows": len(data["cells_no_windows"])})
        S.write_csv(d / "predictions.csv", PRED_COLUMNS, [])
        S.write_json(d / "null.json", "plan11.null.v1", params, CITATION_SPLIT, {"status": V.not_applicable(reason), "scores": []})
        S.write_json(d / "l1_quarantine.json", "plan11.l1_quarantine.v1", params, CITATION_SPLIT, {"status": V.not_applicable(reason), "quarantined": []})
        return d

    if split == "loko" and labelspace == "kernel":
        return write_na("held-out label unseen")
    if lab["n"] == 0:
        return write_na("no admissible cell with windows")
    one_per_cell = all(np.sum(lab["cell_id"] == c) == 1 for c in set(lab["cell_id"].tolist()))
    if split == "within_trace" and (grid_id == S.WHOLE_GRID_ID or one_per_cell):
        return write_na("one window per cell")

    cells, y_unit, kernel_of, arche_of, camp_of, rep_of = _unit_labels(lab, labelspace if labelspace != "campaign" else "campaign")
    if label_override:
        y_unit = {c: str(label_override[c]) for c in cells}
        lab = {**lab, "archetype": np.array([y_unit[c] for c in lab["cell_id"]])}
    folds = SP.folds_for(split, lab, test_frac=test_frac, loro_mode=loro_mode)
    headline = headline_classes_of({k: arche_of[c] for c, k in kernel_of.items() for k in [kernel_of[c]]}) if labelspace == "archetype" else None
    fit_kw = dict(seed=seed + seed_offset, n_jobs=1, n_estimators=n_estimators, reduce_to=reduce_to,
                  reduce_method=reduce_method, rule=unit_rule)
    d_full = X.shape[1]
    min_train_cells = min(len(set(lab["cell_id"][f["train"]].tolist())) for f in folds)
    dim_status = V.GDIM_REDUCED if (d_full > min_train_cells or (reduce_to is not None and reduce_to < d_full)) else V.GDIM_FULL

    def used_dims(preds: dict) -> tuple:
        """(feature_count_used, feature_count_used_per_fold): the width the forest saw per fold
        (``d_used`` of fit_predict_units); an int when every fold used the same width, else
        ``"<min>-<max>"`` (SPEC_epoch2 3.5.2; E1 sec. 4 M5; CHECK_2 M3)."""
        per_fold = {}
        for v in preds.values():
            per_fold.setdefault(str(v["fold"]), int(v["d_used"]))
        if not per_fold:
            return None, {}
        lo, hi = min(per_fold.values()), max(per_fold.values())
        return (lo if lo == hi else f"{lo}-{hi}"), per_fold

    def run_point(Xm, nm):
        sc, preds = _point_score(Xm, lab, y_unit, folds, kernel_of, headline, **fit_kw)
        maj = majority_baseline(y_unit, kernel_of, arche_of, split, labelspace, folds, lab)
        null = np.zeros(0)
        if run_null and n_perm > 0:
            null = _null_scores(Xm, lab, y_unit, folds, kernel_of, split, labelspace, n_perm,
                                SEED_LABEL_NULL + seed_offset, n_jobs, fit_kw)
            verdict, summ = b1_g1_verdict(sc["accuracy"], null)
        elif not run_null:
            verdict, summ = V.not_run("null not requested (--null-splits)"), NL.null_summary(sc["accuracy"], null)
        else:
            verdict, summ = b1_g1_verdict(sc["accuracy"], null)
        return sc, preds, maj, null, verdict, summ

    sc, preds, maj, null, verdict, summ = run_point(X, names)
    quar = quarantine_l1(X, names, lab, y_unit, folds, preds, seed=seed + seed_offset) if quarantine else []
    with_q, preds2 = None, None
    if quar:
        qnames = {q["feature"] for q in quar}
        keep = [j for j, n in enumerate(names) if n not in qnames]
        if keep:
            sc2, preds2, maj2, null2, verdict2, summ2 = run_point(X[:, keep], [names[j] for j in keep])
            fcu2, fcu2_per_fold = used_dims(preds2)
            with_q = {**sc2, "majority": maj2, "b1_g1": verdict2, "b1_g1_rank": summ2["rank"],
                      "null_p95": summ2["p95"], "quarantined_features": sorted(qnames),
                      "feature_count": len(keep), "feature_count_used": fcu2, "feature_count_used_per_fold": fcu2_per_fold}
        else:
            with_q = {"status": V.not_run("every feature quarantined"), "quarantined_features": sorted(qnames)}

    def pred_rows(pr: dict) -> list[dict]:
        rows = []
        for c in cells:
            p = pr.get(c)
            rows.append({"cell_id": c, "kernel": kernel_of[c], "archetype": arche_of[c], "campaign": camp_of[c],
                         "rep": rep_of[c], "fold": p["fold"] if p else "", "y_true": y_unit[c],
                         "y_pred": p["y_pred"] if p else V.not_run("no windows"),
                         "n_windows": int(np.sum(lab["cell_id"] == c)),
                         "vote_fraction": p["vote_fraction"] if p else None,
                         "held_out_campaign": camp_of[c] if split == "loro" else ""})
        for c in data["cells_no_windows"]:
            info = next(x for x in data["cells"] if x["cell_id"] == c)
            rows.append({"cell_id": c, "kernel": info["kernel"], "archetype": info["archetype_predicted"],
                         "campaign": info["campaign"], "rep": info["rep"], "fold": "", "y_true": "",
                         "y_pred": V.not_run("no windows"), "n_windows": 0, "vote_fraction": None, "held_out_campaign": ""})
        return rows
    S.write_csv(d / "predictions.csv", PRED_COLUMNS, pred_rows(preds))
    if preds2 is not None:
        S.write_csv(d / PRED_WITH_QUARANTINE, PRED_COLUMNS, pred_rows(preds2))
    elif (d / PRED_WITH_QUARANTINE).is_file():
        (d / PRED_WITH_QUARANTINE).unlink()          # a stale re-run file from an earlier run must not outlive its quarantine
    # the exclusion follows the rung's score (the re-run when a feature is quarantined; SPEC 3.7.1 with 3.7.2)
    eff = effective_scores({"accuracy": sc["accuracy"], "b1_g1": verdict, "with_quarantine": with_q})
    excluded_row = None
    if eff["b1_g1"] == V.NEAR_UNFALSIFIABLE:
        excluded_row = {"rung": rung, "grid_id": grid_id, "split": split, "labelspace": labelspace,
                        "normalized": normalized, "score": eff["accuracy"], "verdict": eff["b1_g1"]}
        _append_excluded(out, excluded_row)
    fcu, fcu_per_fold = used_dims(preds)
    S.write_json(d / "scores.json", "plan11.scores.v1", params, CITATION_SPLIT, {
        "status": "ok", **sc, "majority": maj, "b1_g1": verdict, "b1_g1_rank": summ["rank"],
        "b1_g1_rank_text": (f"rank {summ['rank']} of {summ['n']}" if summ["rank"] is not None else ""),
        "null_p95": summ["p95"], "null_summary": summ, "with_quarantine": with_q,
        "feature_count": d_full, "dim_status": dim_status, "min_train_cells": min_train_cells,
        # epoch 2 (SPEC_epoch2 3.5.2): the width the forest saw, per fold and summarised; Table 7 prints it
        "feature_count_used": fcu, "feature_count_used_per_fold": fcu_per_fold,
        "seed": seed + seed_offset, "n_perm": int(len(null)), "n_folds": len(folds),
        "n_cells_no_windows": len(data["cells_no_windows"]), "headline_classes": headline,
        "excluded_row": excluded_row,
        "score_source": SCORE_SOURCE_QUARANTINE if (with_q and with_q.get("quarantined_features")) else SCORE_SOURCE_FULL,
        "predictions_file": PRED_WITH_QUARANTINE if preds2 is not None else "predictions.csv"})
    S.write_json(d / "null.json", "plan11.null.v1", params, CITATION_SPLIT, {"summary": summ, "scores": null.tolist()})
    S.write_json(d / "l1_quarantine.json", "plan11.l1_quarantine.v1", params, CITATION_SPLIT,
                 {"quarantined": quar, "n_features": d_full})
    return d


PRED_COLUMNS = ("cell_id", "kernel", "archetype", "campaign", "rep", "fold", "y_true", "y_pred",
                "n_windows", "vote_fraction", "held_out_campaign")
EXCLUDED_COLUMNS = ("rung", "grid_id", "split", "labelspace", "normalized", "score", "verdict")


def _append_excluded(out: Path, row: dict):
    p = Path(out) / "gates" / "excluded_rows.csv"
    rows = S.read_csv(p) if p.is_file() else []
    rows = [r for r in rows if not (r["rung"] == row["rung"] and r["grid_id"] == row["grid_id"]
                                    and r["split"] == row["split"] and r["labelspace"] == row["labelspace"]
                                    and r["normalized"] == S.fmt_num(row["normalized"]))]
    rows.append(row)
    S.write_csv(p, EXCLUDED_COLUMNS, rows)


# --------------------------------------------------------------------------- clustering (4.4)

def cell_vectors(data: dict) -> tuple:
    """Per-cell vectors: the mean over the cell's windows of the (normalized) features; returns
    (cells, kernels, archetypes, Xcell)."""
    lab, X = data["lab"], data["X"]
    cells = list(dict.fromkeys(lab["cell_id"].tolist()))
    Xc = np.stack([np.nanmean(X[lab["cell_id"] == c], axis=0) for c in cells])
    kern = [str(lab["kernel"][lab["cell_id"] == c][0]) for c in cells]
    arch = [str(lab["archetype"][lab["cell_id"] == c][0]) for c in cells]
    return cells, kern, arch, Xc


def cluster_cells(Xcell: np.ndarray, k: int, algo: str = CLUSTER_PRIMARY, seed: int = SEED_FOREST) -> np.ndarray:
    """Per-cell vectors standardized over cells; ``k`` = the number of predicted archetypes present
    among the kernel cells after G-K0; primary ``KMeans(n_clusters=k, n_init=10, random_state=seed)``;
    alternatives ``gmm`` (GaussianMixture(n_components=k, n_init=5)) and ``agglomerative``
    (ward) (SPEC 4.4; section 8 item 31)."""
    Z = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]).fit_transform(Xcell)
    if algo == "kmeans":
        return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)
    if algo == "gmm":
        return GaussianMixture(n_components=k, n_init=5, random_state=seed).fit(Z).predict(Z)
    if algo == "agglomerative":
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)
    raise ValueError(algo)


def ari_nmi(labels_true, labels_pred) -> tuple:
    """(ARI, NMI) from sklearn.metrics (SPEC 4.4)."""
    return float(adjusted_rand_score(labels_true, labels_pred)), float(normalized_mutual_info_score(labels_true, labels_pred))


CLUSTER_PERM_FLOOR_CLI = B1G1_MIN_PERM     # SPEC_epoch2 B3: the `cluster` CLI default; the function default is 0


def run_clustering(out: Path, rung: str = "combined", algos=CLUSTER_ALGOS, n_perm: int = 500,
                   grid_id: str | None = None, seed_offset: int = 0, primary: str = CLUSTER_PRIMARY,
                   perm_floor: int = 0) -> Path:
    """gates/clustering.csv and clustering.json (SPEC 4.4): per algorithm ARI and NMI against the
    predicted archetype labels (after G-K0), the unit-level null = ``n_perm`` archetype-label
    permutations across kernels (cells inherit) with ARI and NMI recomputed against the fixed
    clustering, ``null_summary`` on each. ``perm_floor`` (SPEC_epoch2 B3; CHECK_3 M3; P2 Sec. V 5.1
    Plan 08 "at least 500 permutations", the null's own floor as ``b1_g1_verdict`` applies it): when
    ``0 < n_perm < perm_floor`` the ``exceeds_ari`` and ``exceeds_nmi`` cells read ``not run: <n>
    permutations < <perm_floor>`` and every number stays in its column. Two defaults on purpose:
    0 at the function (judge on any count, the epoch-1 contract of the direct calls) and
    ``CLUSTER_PERM_FLOOR_CLI = 500`` on the ``cluster`` CLI, which the driver runs."""
    out = Path(out)
    gid, src = (grid_id, "argument") if grid_id else S.selected_grid_id(out, rung, default=None)
    cols = ("rung", "algo", "k", "ari", "ari_null_p95", "ari_rank", "nmi", "nmi_null_p95", "nmi_rank",
            "exceeds_ari", "exceeds_nmi", "primary", "grid_id", "status")
    params = {"rung": rung, "grid_id": gid, "grid_source": src, "n_perm": n_perm, "algos": list(algos),
              "primary": primary, "seed": SEED_FOREST + seed_offset, "seed_label_null": SEED_LABEL_NULL + seed_offset,
              "perm_floor": perm_floor}
    if gid is None:
        rows = [{"rung": rung, "algo": a, "status": V.not_run(f"no selection for {rung}"), "primary": a == primary} for a in algos]
        S.write_csv(out / "gates" / "clustering.csv", cols, rows)
        S.write_json(out / "gates" / "clustering.json", "plan11.clustering.v1", params, CITATION_CLUSTER,
                     {"status": V.not_run(f"no selection for {rung}")})
        return out / "gates" / "clustering.csv"
    data = prepare_split_data(out, rung, gid, normalized=True)
    params.update({"gk0_applied": data["gk0_applied"], "relabelled_kernels": data["relabelled_kernels"],
                   "inputs_sha256": S.inputs_sha256([out / "cells.csv", S.features_path(out, rung, gid, True), out / "gates" / "gk0.csv"], out)})
    cells, kern, arch, Xc = cell_vectors(data)
    k = len(set(arch))
    uk = list(dict.fromkeys(kern))
    arch_of_k = {kk: arch[kern.index(kk)] for kk in uk}
    rng = np.random.default_rng(SEED_LABEL_NULL + seed_offset)
    rows, payload = [], {"k": k, "cells": cells, "kernels": kern, "archetype": arch, "per_algo": {}}
    for algo in algos:
        lab_pred = cluster_cells(Xc, k, algo, SEED_FOREST + seed_offset)
        ari, nmi = ari_nmi(arch, lab_pred)
        na, nn = [], []
        for _ in range(n_perm):
            perm = rng.permutation(len(uk))
            newmap = {uk[i]: arch_of_k[uk[perm[i]]] for i in range(len(uk))}
            a2 = [newmap[kk] for kk in kern]
            x, y = ari_nmi(a2, lab_pred)
            na.append(x); nn.append(y)
        sa, sn = NL.null_summary(ari, np.array(na)), NL.null_summary(nmi, np.array(nn))
        under = bool(perm_floor and 0 < sa["n"] < perm_floor)     # SPEC_epoch2 B3: the floor of B1-G1's null
        ex_a = V.not_run(f"{sa['n']} permutations < {perm_floor}") if under else sa["exceeds"]
        ex_n = V.not_run(f"{sn['n']} permutations < {perm_floor}") if under else sn["exceeds"]
        rows.append({"rung": rung, "algo": algo, "k": k, "ari": ari, "ari_null_p95": sa["p95"], "ari_rank": sa["rank"],
                     "nmi": nmi, "nmi_null_p95": sn["p95"], "nmi_rank": sn["rank"], "exceeds_ari": ex_a,
                     "exceeds_nmi": ex_n, "primary": algo == primary, "grid_id": gid, "status": "ok"})
        counts = {}
        for a, c in zip(arch, lab_pred):
            counts.setdefault(a, {})
            counts[a][f"c{int(c)}"] = counts[a].get(f"c{int(c)}", 0) + 1
        payload["per_algo"][algo] = {"labels": [int(v) for v in lab_pred], "ari": sa, "nmi": sn,
                                     "cluster_by_predicted_archetype": counts}
    S.write_csv(out / "gates" / "clustering.csv", cols, rows)
    S.write_json(out / "gates" / "clustering.json", "plan11.clustering.v1", params, CITATION_CLUSTER, payload)
    return out / "gates" / "clustering.csv"


# --------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="models.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("splits")
    a.add_argument("--out", required=True)
    a.add_argument("--rung", required=True, choices=S.RUNGS)
    a.add_argument("--grid-id", default=None)
    g = a.add_mutually_exclusive_group(required=True)
    g.add_argument("--split", choices=SP.SPLITS)
    g.add_argument("--all-splits", action="store_true")
    a.add_argument("--labelspace", default="all", choices=("kernel", "archetype", "all"))
    v = a.add_mutually_exclusive_group()
    v.add_argument("--raw", action="store_true")
    v.add_argument("--norm", action="store_true")
    v.add_argument("--raw-and-norm", action="store_true")
    a.add_argument("--null-perm", type=int, default=B1G1_MIN_PERM)
    a.add_argument("--null-splits", default="loko,loro,within_trace")
    a.add_argument("--n-jobs", type=int, default=1)
    a.add_argument("--n-estimators", type=int, default=N_ESTIMATORS)
    a.add_argument("--feature-drop", default="")
    a.add_argument("--include-idle", action="store_true")
    a.add_argument("--seed-offset", type=int, default=0)
    a.add_argument("--base-dir", default="splits",
                   help="the directory under gates/ the runs are written to (SPEC_epoch2 B10: the driver's G-L (ii) re-run uses splits_gl2drop)")
    c = sub.add_parser("cluster")
    c.add_argument("--out", required=True)
    c.add_argument("--rung", default="combined", choices=S.RUNGS)
    c.add_argument("--grid-id", default=None)
    c.add_argument("--algo", default="all", choices=("all",) + CLUSTER_ALGOS)
    c.add_argument("--null-perm", type=int, default=500)
    c.add_argument("--seed-offset", type=int, default=0)
    c.add_argument("--perm-floor", type=int, default=CLUSTER_PERM_FLOOR_CLI,
                   help="exceeds_ari / exceeds_nmi read `not run: N permutations < floor` when 0 < null-perm < floor (SPEC_epoch2 B3)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if not (out / "cells.csv").is_file():
        print(f"missing input: {out / 'cells.csv'}", file=sys.stderr)
        return 2
    if args.cmd == "cluster":
        algos = CLUSTER_ALGOS if args.algo == "all" else (args.algo,)
        try:
            print(run_clustering(out, args.rung, algos, args.null_perm, args.grid_id, args.seed_offset, perm_floor=args.perm_floor))
        except FileNotFoundError as e:
            print(f"missing input: {e}", file=sys.stderr)
            return 2
        return 0
    # SPEC_epoch2 B18 (CERT 3 third marker, 6.13, 7.5; E1 6.56): without --grid-id the split stage runs at the
    # rung's selected point and refuses when there is none; the silent W8_H4 fallback is gone
    if args.grid_id:
        gid, grid_source = args.grid_id, "argument"
    else:
        gid, _ = S.selected_grid_id(out, args.rung, None)
        grid_source = "selection.json"
        if gid is None:
            print(f"missing input: gates/selection.json has no entry for {args.rung} (run gates_temporal select first)", file=sys.stderr)
            return 2
    variants = [True]
    if args.raw:
        variants = [False]
    elif args.raw_and_norm:
        variants = [False, True]
    if args.rung == "combined":
        variants = [True]
    splits_ = SP.SPLITS if args.all_splits else (args.split,)
    null_splits = set(x for x in args.null_splits.split(",") if x)
    fdrop = tuple(x for x in args.feature_drop.split(",") if x)
    try:
        for norm in variants:
            for sp in splits_:
                spaces = ("archetype",) if sp == "loko" else (("kernel", "archetype") if args.labelspace == "all" else (args.labelspace,))
                for ls in spaces:
                    d = run_split_stage(out, args.rung, gid, sp, ls, normalized=norm, n_perm=args.null_perm,
                                        n_jobs=args.n_jobs, feature_drop=fdrop, include_idle=args.include_idle,
                                        n_estimators=args.n_estimators, run_null=sp in null_splits,
                                        seed_offset=args.seed_offset, base_dir=args.base_dir, grid_source=grid_source)
                    print(d)
    except FileNotFoundError as e:
        print(f"missing input: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
