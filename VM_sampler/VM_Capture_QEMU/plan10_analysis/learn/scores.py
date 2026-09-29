#!/usr/bin/env python3
"""scores.py -- every score a model's kind allows, the null, the bootstrap, and what explains
the result. Pure functions over arrays; the executor decides what to call.

  classification    accuracy, balanced accuracy, macro F1, kappa, MCC, per-class precision /
                    recall / F1 / support, the confusion matrix, top-2, log loss, Brier, ECE
  detection         AUROC, AUPRC, recall at a fixed FPR, the threshold curve
  clustering        ARI and NMI against family and workload, silhouette / Davies-Bouldin /
                    Calinski-Harabasz on the rows, purity, the cluster x label contingency
  embedding_quality silhouette by label and kNN purity on an embedding
  null_permutation  the score's distribution when the test labels are permuted against the
                    predictions (n times, seeded), and the observed score's percentile in it
  bootstrap         a CI from resampling recordings (groups) with replacement
  importance        permutation importance per column (rows) or per block (paths)

Numbers are floats or None (undefined: one class, no positives, too few rows), never NaN.
"""
from __future__ import annotations

import warnings

import numpy as np


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true).astype(str), np.asarray(y_pred).astype(str)
    return float((y_true == y_pred).mean()) if y_true.size else 0.0


def balanced_accuracy(y_true, y_pred):
    from sklearn.metrics import balanced_accuracy_score
    y_true, y_pred = np.asarray(y_true).astype(str), np.asarray(y_pred).astype(str)
    if y_true.size == 0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # a novelty fold predicts classes absent from y_true, by construction
        return _f(balanced_accuracy_score(y_true, y_pred))


def macro_f1(y_true, y_pred):
    from sklearn.metrics import f1_score
    y_true, y_pred = np.asarray(y_true).astype(str), np.asarray(y_pred).astype(str)
    if y_true.size == 0:
        return None
    return _f(f1_score(y_true, y_pred, average="macro", zero_division=0))


def classification(y_true, y_pred, proba=None, classes=None, top_k: int = 2, calibration_bins: int = 10) -> dict:
    with warnings.catch_warnings():
        return _classification(y_true, y_pred, proba, classes, top_k, calibration_bins)


def _classification(y_true, y_pred, proba, classes, top_k, calibration_bins) -> dict:
    from sklearn.metrics import cohen_kappa_score, confusion_matrix, log_loss, matthews_corrcoef, precision_recall_fscore_support
    y_true, y_pred = np.asarray(y_true).astype(str), np.asarray(y_pred).astype(str)
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()) | set(classes or []))
    out = {"n": int(y_true.size), "accuracy": _f(accuracy(y_true, y_pred)), "balanced_accuracy": balanced_accuracy(y_true, y_pred),
           "macro_f1": macro_f1(y_true, y_pred)}
    warnings.simplefilter("ignore")
    if y_true.size:
        out["kappa"] = _f(cohen_kappa_score(y_true, y_pred)) if len(set(y_true.tolist())) > 1 else None
        out["mcc"] = _f(matthews_corrcoef(y_true, y_pred)) if len(set(y_true.tolist())) > 1 else None
        p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
        out["per_class"] = {lab: {"precision": _f(p[i]), "recall": _f(r[i]), "f1": _f(f[i]), "support": int(s[i])} for i, lab in enumerate(labels)}
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        out["confusion"] = {"labels": labels, "matrix": cm.tolist(), "row_recall": [(_f(cm[i, i] / cm[i].sum()) if cm[i].sum() else None) for i in range(len(labels))]}
    else:
        out.update(kappa=None, mcc=None, per_class={}, confusion={"labels": labels, "matrix": [], "row_recall": []})
    if proba is not None and classes is not None and y_true.size and len(classes) == np.asarray(proba).shape[1]:
        P = np.asarray(proba, dtype=np.float64)
        P = np.clip(P, 1e-9, 1)
        P = P / P.sum(axis=1, keepdims=True)
        ci = {c: i for i, c in enumerate(classes)}
        known = np.array([t in ci for t in y_true])
        if known.any():
            ti = np.array([ci[t] for t in y_true[known]])
            Pk = P[known]
            k = min(top_k, len(classes))
            top = np.argsort(-Pk, axis=1)[:, :k]
            out[f"top{k}_accuracy"] = _f((top == ti[:, None]).any(axis=1).mean())
            onehot = np.zeros_like(Pk)
            onehot[np.arange(ti.size), ti] = 1
            out["brier"] = _f(((Pk - onehot) ** 2).sum(axis=1).mean())
            try:
                out["log_loss"] = _f(log_loss(ti, Pk, labels=list(range(len(classes)))))
            except ValueError:
                out["log_loss"] = None
            out["calibration"] = calibration(ti, Pk, calibration_bins)
            out["p_true_mean"] = _f(Pk[np.arange(ti.size), ti].mean())
        out["unknown_class_rows"] = int((~known).sum())
    return out


def calibration(ti: np.ndarray, P: np.ndarray, bins: int = 10) -> dict:
    """Reliability of the top prediction: confidence vs accuracy per bin, and ECE."""
    conf = P.max(axis=1)
    pred = P.argmax(axis=1)
    correct = (pred == ti).astype(float)
    edges = np.linspace(0, 1, int(bins) + 1)
    rows, ece = [], 0.0
    for i in range(int(bins)):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= edges[i]) & (conf <= edges[i + 1])
        if m.any():
            acc, c = float(correct[m].mean()), float(conf[m].mean())
            ece += m.mean() * abs(acc - c)
            rows.append({"lo": _f(edges[i]), "hi": _f(edges[i + 1]), "n": int(m.sum()), "confidence": _f(c), "accuracy": _f(acc)})
    return {"ece": _f(ece), "bins": rows}


def detection(is_positive, score, fpr: float = 0.05, n_curve: int = 50) -> dict:
    """Binary detection from a score where higher means more positive (more novel)."""
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
    y, s = np.asarray(is_positive).astype(bool), np.asarray(score, dtype=np.float64)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    out = {"n": int(y.size), "n_positive": int(y.sum()), "n_negative": int((~y).sum())}
    if y.size == 0 or y.all() or (~y).all():
        out.update(auroc=None, auprc=None, recall_at_fpr=None, threshold_at_fpr=None, curve=[], why="both classes are needed")
        return out
    out["auroc"] = _f(roc_auc_score(y, s))
    out["auprc"] = _f(average_precision_score(y, s))
    fp, tp, th = roc_curve(y, s)
    ok_i = np.flatnonzero(fp <= fpr)
    i = ok_i[-1] if ok_i.size else 0
    out["recall_at_fpr"] = _f(tp[i])
    out["threshold_at_fpr"] = _f(th[i]) if np.isfinite(th[i]) else None
    step = max(1, len(fp) // n_curve)
    out["curve"] = [{"fpr": _f(fp[j]), "tpr": _f(tp[j])} for j in list(range(0, len(fp), step)) + ([len(fp) - 1] if (len(fp) - 1) % step else [])]
    return out


def detection_vs_reference(novel_scores, reference_scores, fpr: float = 0.05) -> dict:
    """Novel test tiles against a reference of benign scores (the training tiles', in-sample and
    therefore optimistic): AUROC between the two sets, and the recall of the novel tiles above the
    reference's (1 - fpr) quantile. Says what the reference is."""
    a, b = np.asarray(novel_scores, dtype=np.float64), np.asarray(reference_scores, dtype=np.float64)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    out = {"n_novel": int(a.size), "n_reference": int(b.size), "reference": "training tiles, in-sample (optimistic)"}
    if a.size == 0 or b.size == 0:
        out.update(auroc=None, recall_at_fpr=None, threshold=None, why="needs novel test tiles and training tiles")
        return out
    d = detection(np.r_[np.ones(a.size, bool), np.zeros(b.size, bool)], np.r_[a, b], fpr)
    thr = float(np.quantile(b, 1 - fpr))
    out.update(auroc=d["auroc"], auprc=d["auprc"], threshold=_f(thr), recall_at_fpr=_f((a > thr).mean()), curve=d["curve"])
    return out


def clustering(clusters, y_family, y_workload=None, X=None) -> dict:
    from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, normalized_mutual_info_score, silhouette_score
    c = np.asarray(clusters).astype(int)
    out = {"n": int(c.size), "k_present": int(len(set(c.tolist())))}      # clusters the test tiles fell into; the model's k is in its own record
    for name, y in (("family", y_family), ("workload", y_workload)):
        if y is None:
            continue
        y = np.asarray(y).astype(str)
        out[f"ari_{name}"] = _f(adjusted_rand_score(y, c)) if c.size else None
        out[f"nmi_{name}"] = _f(normalized_mutual_info_score(y, c)) if c.size else None
        labs = sorted(set(y.tolist()))
        cl = sorted(set(c.tolist()))
        table = [[int(((y == lab) & (c == k)).sum()) for k in cl] for lab in labs]
        out[f"contingency_{name}"] = {"labels": labs, "clusters": cl, "matrix": table}
        out[f"purity_{name}"] = _f(sum(max(col) for col in zip(*table)) / c.size) if c.size and table else None
    if X is not None and c.size > out["k_present"] > 1:
        X2 = np.asarray(X).reshape(c.size, -1)
        try:
            out["silhouette"] = _f(silhouette_score(X2, c))
            out["davies_bouldin"] = _f(davies_bouldin_score(X2, c))
            out["calinski_harabasz"] = _f(calinski_harabasz_score(X2, c))
        except ValueError:
            out.update(silhouette=None, davies_bouldin=None, calinski_harabasz=None)
    else:
        out.update(silhouette=None, davies_bouldin=None, calinski_harabasz=None)
    return out


def embedding_quality(Z, y, k: int = 5) -> dict:
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
    Z, y = np.asarray(Z).reshape(len(y), -1), np.asarray(y).astype(str)
    out = {"n": int(y.size), "dim": int(Z.shape[1])}
    labs = set(y.tolist())
    if y.size <= 2 or len(labs) < 2 or len(labs) >= y.size:
        out.update(silhouette=None, knn_purity=None)
        return out
    try:
        out["silhouette"] = _f(silhouette_score(Z, y))
    except ValueError:
        out["silhouette"] = None
    kk = max(1, min(k, y.size - 1))
    nn = NearestNeighbors(n_neighbors=kk + 1).fit(Z)
    _, idx = nn.kneighbors(Z)
    out["knn_purity"] = _f((y[idx[:, 1:]] == y[:, None]).mean())
    out["knn_k"] = kk
    return out


def null_permutation(y_true, y_pred, n: int = 200, seed: int = 0, metric=None) -> dict:
    """The score when the test labels are permuted against the predictions: what chance gives
    with this class balance. Cheap, exact for a predictor that never saw test labels."""
    metric = metric or accuracy
    y_true, y_pred = np.asarray(y_true).astype(str), np.asarray(y_pred).astype(str)
    if y_true.size == 0 or n <= 0:
        return {"n": 0, "observed": _f(metric(y_true, y_pred)) if y_true.size else None}
    if len(set(y_true.tolist())) < 2:
        return {"n": 0, "observed": _f(metric(y_true, y_pred)),
                "why": "one class in the test fold: permuting its labels changes nothing; read the pooled null of the split"}
    rng = np.random.default_rng(seed)
    obs = float(metric(y_true, y_pred))
    dist = np.array([float(metric(rng.permutation(y_true), y_pred)) for _ in range(int(n))])
    return {"n": int(n), "observed": _f(obs), "mean": _f(dist.mean()), "std": _f(dist.std()),
            "p5": _f(np.percentile(dist, 5)), "p95": _f(np.percentile(dist, 95)), "max": _f(dist.max()),
            "percentile_of_observed": _f((dist < obs).mean() * 100), "p_value": _f(((dist >= obs).sum() + 1) / (n + 1)),
            "hist": np.histogram(dist, bins=20, range=(0, 1))[0].tolist() if 0 <= dist.min() and dist.max() <= 1 else None}


def null_permutation_grouped(y_true, y_pred, groups, n: int = 500, seed: int = 0, metric=None) -> dict:
    """The score when the labels are permuted ACROSS GROUPS, not across rows: every group (a
    kernel) carries one label, the groups' labels are shuffled among the groups, and every row
    inherits its group's permuted label. The predictions are the pooled out-of-fold ones as they
    stand: no refit. This is the null for a target defined per kernel (the archetype), where
    permuting rows would break the one-label-per-kernel structure the design has."""
    metric = metric or accuracy
    y_true, y_pred, groups = np.asarray(y_true).astype(str), np.asarray(y_pred).astype(str), np.asarray(groups).astype(str)
    what = "kernel-level permutation, no refit"
    if y_true.size == 0 or n <= 0:
        return {"n": 0, "observed": _f(metric(y_true, y_pred)) if y_true.size else None, "what": what}
    ug = np.unique(groups)
    label_of = {}
    for g in ug:
        labs = set(y_true[groups == g].tolist())
        if len(labs) != 1:
            return {"n": 0, "observed": _f(metric(y_true, y_pred)), "what": what,
                    "why": f"group {g} carries {len(labs)} labels; a kernel-level null needs one label per kernel"}
        label_of[g] = labs.pop()
    if len(set(label_of.values())) < 2:
        return {"n": 0, "observed": _f(metric(y_true, y_pred)), "what": what, "why": "one label across the kernels: permuting changes nothing"}
    rng = np.random.default_rng(seed)
    obs = float(metric(y_true, y_pred))
    glabels = np.array([label_of[g] for g in ug])
    gi = {g: i for i, g in enumerate(ug)}
    row_group = np.array([gi[g] for g in groups])
    dist = np.array([float(metric(rng.permutation(glabels)[row_group], y_pred)) for _ in range(int(n))])
    return {"n": int(n), "observed": _f(obs), "mean": _f(dist.mean()), "std": _f(dist.std()),
            "p5": _f(np.percentile(dist, 5)), "p95": _f(np.percentile(dist, 95)), "max": _f(dist.max()),
            "percentile_of_observed": _f((dist < obs).mean() * 100), "p_value": _f(((dist >= obs).sum() + 1) / (n + 1)),
            "hist": np.histogram(dist, bins=20, range=(0, 1))[0].tolist() if 0 <= dist.min() and dist.max() <= 1 else None,
            "n_groups": int(ug.size), "what": what}


def archetype_report(y_true, y_pred, groups, table: dict) -> dict:
    """Beside the accuracy, for a target defined per kernel: the majority-archetype baseline
    (always predict the most frequent archetype of the pooled rows), recall per archetype, and
    macro recall over the archetypes with at least three kernels IN THIS RUN, computed from the
    table restricted to the kernels present. An archetype with a single kernel cannot be
    predicted under leave-one-kernel-out (its only kernel is the held-out one, so no training row
    carries its label): its recall is not a number, and the row says so."""
    y_true, y_pred, groups = np.asarray(y_true).astype(str), np.asarray(y_pred).astype(str), np.asarray(groups).astype(str)
    out: dict = {"n": int(y_true.size)}
    if y_true.size == 0:
        return out
    kernels_present = sorted(set(groups.tolist()))
    kernels_of = {}
    for k in kernels_present:
        a = table.get(k)
        if a:
            kernels_of.setdefault(a, []).append(k)
    archetypes = sorted(set(y_true.tolist()) | set(kernels_of))
    counts = {a: int((y_true == a).sum()) for a in archetypes}
    maj = max(counts, key=counts.get)
    out["majority"] = maj
    out["majority_baseline"] = _f(counts[maj] / y_true.size)
    rows = {}
    for a in archetypes:
        nk = len(kernels_of.get(a, []))
        m = y_true == a
        if nk <= 1:
            rows[a] = {"n_kernels": nk, "n_rows": int(m.sum()), "recall": None,
                       "note": "one kernel: cannot be predicted under leave-one-kernel-out (no training row carries this archetype when it is held out)"}
        else:
            rows[a] = {"n_kernels": nk, "n_rows": int(m.sum()), "recall": _f((y_pred[m] == a).mean()) if m.any() else None, "note": ""}
    out["recall"] = rows
    eligible = [a for a in archetypes if len(kernels_of.get(a, [])) >= 3 and rows[a]["recall"] is not None]
    out["macro_recall_3plus"] = _f(float(np.mean([rows[a]["recall"] for a in eligible]))) if eligible else None
    out["macro_recall_3plus_over"] = eligible
    out["kernels_per_archetype"] = {a: len(v) for a, v in kernels_of.items()}
    out["what"] = ("majority-archetype baseline: always predict the most frequent archetype of the pooled rows; "
                   "macro recall over the archetypes with at least three kernels in this run")
    return out


def bootstrap(y_true, y_pred, groups, n: int = 200, seed: int = 0, metric=None) -> dict:
    """Resample the groups (recordings) with replacement; the score's spread over resamples."""
    metric = metric or accuracy
    y_true, y_pred, groups = np.asarray(y_true).astype(str), np.asarray(y_pred).astype(str), np.asarray(groups).astype(str)
    ug = np.unique(groups)
    if y_true.size == 0 or n <= 0 or ug.size < 2:
        return {"n": 0, "why": "fewer than two groups" if ug.size < 2 else "nothing to resample"}
    rng = np.random.default_rng(seed)
    idx_of = {g: np.flatnonzero(groups == g) for g in ug}
    vals = []
    for _ in range(int(n)):
        pick = rng.choice(ug, size=ug.size, replace=True)
        idx = np.concatenate([idx_of[g] for g in pick])
        vals.append(float(metric(y_true[idx], y_pred[idx])))
    vals = np.array(vals)
    return {"n": int(n), "groups": int(ug.size), "mean": _f(vals.mean()), "std": _f(vals.std()), "ci95": [_f(np.percentile(vals, 2.5)), _f(np.percentile(vals, 97.5))]}


def importance(predict, X, y, metric=None, seed: int = 0, n_repeats: int = 3, axis: str = "columns", names=None) -> dict:
    """Permutation importance: the drop in the metric when one column (rows) or one block (path,
    the last axis) is shuffled across tiles. `predict` maps X -> labels."""
    metric = metric or accuracy
    rng = np.random.default_rng(seed)
    X, y = np.asarray(X), np.asarray(y).astype(str)
    base = float(metric(y, predict(X)))
    n_cols = X.shape[-1]
    drops = []
    for j in range(n_cols):
        d = []
        for _ in range(int(n_repeats)):
            Xp = X.copy()
            perm = rng.permutation(X.shape[0])
            Xp[..., j] = X[perm][..., j]
            d.append(base - float(metric(y, predict(Xp))))
        drops.append(_f(np.mean(d)))
    order = np.argsort([-(v if v is not None else -np.inf) for v in drops])
    return {"baseline": _f(base), "axis": axis, "names": list(names) if names is not None else [str(j) for j in range(n_cols)],
            "drop": drops, "ranked": [int(i) for i in order]}


def per_class_score_stats(score, y, ) -> dict:
    y, s = np.asarray(y).astype(str), np.asarray(score, dtype=np.float64)
    out = {}
    for lab in sorted(set(y.tolist())):
        v = s[y == lab]
        v = v[np.isfinite(v)]
        out[lab] = {"n": int(v.size), "median": _f(np.median(v)) if v.size else None, "p25": _f(np.percentile(v, 25)) if v.size else None,
                    "p75": _f(np.percentile(v, 75)) if v.size else None, "mean": _f(v.mean()) if v.size else None}
    return out
