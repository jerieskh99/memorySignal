#!/usr/bin/env python3
"""results.py -- what a Learn run wrote, as frames and views the console draws.

Two frames feed the five views Explore already has (results_view.aggregate_frame):
  scores  one row per (configuration, split, fold, seed); columns are the scores
  tiles   one row per scored test tile; columns correct, p_true, p_max, novelty, cluster

And the views learning needs on top: the confusion matrix, ROC curves, the embedding in 2-D
(PCA on the bridge; UMAP when installed), saliency and importance maps, the null next to the
observed score, calibration, training curves, and a path or image tile as drawn.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from plan10_analysis import results_view as RV
from plan10_analysis.learn.executor import AGG_KEYS, _dig

SCORE_COLS = [k for k, _ in AGG_KEYS]


class ResultsError(ValueError):
    pass


def load_results(run_dir: Path) -> dict:
    f = Path(run_dir) / "learn_results.json"
    if not f.exists():
        raise ResultsError(f"{Path(run_dir).name}: no learn_results.json (the run has not finished)")
    return json.loads(f.read_text())


def load_sidecar(run_dir: Path) -> dict:
    f = Path(run_dir) / "sidecar.json"
    try:
        return json.loads(f.read_text()) if f.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def load_predictions(run_dir: Path):
    f = Path(run_dir) / "predictions.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=False)
    return z["keys"], z["pred"]


def _conf(res: dict, config: str) -> dict:
    c = next((c for c in res["configurations"] if c["id"] == config), None)
    if c is None:
        raise ResultsError(f"no configuration {config!r}; there are {', '.join(x['id'] for x in res['configurations'])}")
    return c


def _split(c: dict, split: str) -> dict:
    s = c["splits"].get(split)
    if s is None:
        raise ResultsError(f"configuration {c['id']} has no split {split!r}; it has {', '.join(c['splits'])}")
    return s


def summary(run_dir: Path) -> dict:
    res, side = load_results(run_dir), load_sidecar(run_dir)
    confs = []
    for c in res["configurations"]:
        per = {}
        for sp, s in c["splits"].items():
            m = s["aggregate"].get("metrics", {})
            per[sp] = {"n_folds": s["aggregate"].get("n_folds", 0), "skipped": len(s.get("skipped", [])), "label": s.get("label"),
                       "metrics": {k: {"mean": v["mean"], "std": v["std"], "n": v["n"]} for k, v in m.items()},
                       "pooled_accuracy": s["aggregate"].get("pooled_accuracy"), "pooled": s["aggregate"].get("pooled"),
                       "novelty_folds": s["aggregate"].get("novelty_folds", [])}
        confs.append({"id": c["id"], "name": c["name"], "model": c["model"], "kind": c["kind"], "input": c["input"],
                      "steps": [{"tier": st["tier"], "module": st["module"], "params": st["params"]} for st in c["steps"] if st["tier"] not in ("score", "output")],
                      "splits": per})
    return {"label": res.get("label"), "target": res.get("target"), "archetype": res.get("archetype"), "sweep": res.get("sweep"), "fits_done": res.get("fits_done"),
            "seconds": res.get("seconds"), "configurations": confs, "splits": list(side.get("splits") or (confs[0]["splits"] if confs else [])),
            "seeds": side.get("seeds"), "inputs": side.get("inputs"), "versions": {k: side.get(k) for k in ("python", "numpy", "sklearn", "torch", "torch_device")},
            "written_at": side.get("written_at"), "acknowledged": [a.get("id") for a in side.get("acknowledged") or []],
            "score_columns": SCORE_COLS}


def scores_frame(run_dir: Path) -> RV.Frame:
    res = load_results(run_dir)
    rows, keys = [], {k: [] for k in ("config", "name", "model", "kind", "split", "fold", "seed", "run", "held_out", "family", "novelty_fold")}
    for c in res["configurations"]:
        for sp, s in c["splits"].items():
            for f in s["folds"]:
                rows.append([_dig(f, path) for _, path in AGG_KEYS])
                keys["config"].append(c["id"]); keys["name"].append(c["name"]); keys["model"].append(c["model"] or ""); keys["kind"].append(c["kind"] or "")
                keys["split"].append(sp); keys["fold"].append(f["fold"]); keys["seed"].append(int(f["seed"])); keys["run"].append(c["input"]["run"])
                keys["held_out"].append(str(f.get("held_out") or "")); keys["family"].append(str(f.get("family") or _fold_family(f)))
                keys["novelty_fold"].append("novelty" if f.get("novelty_fold") else "known")
    if not rows:
        raise ResultsError("no scored fold in this run")
    X = np.array([[np.nan if v is None else float(v) for v in r] for r in rows], dtype=np.float64)
    keys = {k: np.asarray(v) for k, v in keys.items()}
    return RV.Frame(X, list(SCORE_COLS), keys, [res.get("label")], [])


def _fold_family(f: dict) -> str:
    tc = f.get("test_classes") or []
    return tc[0] if len(tc) == 1 else ""


def tiles_frame(run_dir: Path) -> RV.Frame:
    res = load_results(run_dir)
    pk = load_predictions(run_dir)
    if pk is None:
        raise ResultsError("no predictions.npz (keep_predictions was off)")
    K, Pr = pk
    name_of = {c["id"]: c["name"] for c in res["configurations"]}
    model_of = {c["id"]: c["model"] or "" for c in res["configurations"]}
    X = np.stack([Pr["correct"].astype(np.float64), Pr["p_true"].astype(np.float64), Pr["p_max"].astype(np.float64),
                  Pr["novelty"].astype(np.float64), Pr["cluster"].astype(np.float64)], axis=1)
    X[:, 0] = np.where(Pr["correct"] < 0, np.nan, X[:, 0])
    X[:, 4] = np.where(Pr["cluster"] < 0, np.nan, X[:, 4])
    keys = {k: np.asarray(K[k]) for k in ("config", "split", "fold", "seed", "recording", "workload", "family", "block", "t_index", "seq_start")}
    keys["name"] = np.asarray([name_of.get(c, c) for c in K["config"]])
    keys["model"] = np.asarray([model_of.get(c, "") for c in K["config"]])
    keys["y_true"] = np.asarray(Pr["y_true"]); keys["y_pred"] = np.asarray(Pr["y_pred"])
    keys["run"] = np.asarray([res.get("label")] * len(K))
    return RV.Frame(X, ["correct", "p_true", "p_max", "novelty", "cluster"], keys, [res.get("label")], [])


def frame(run_dir: Path, which: str = "scores") -> RV.Frame:
    if which == "scores":
        return scores_frame(run_dir)
    if which == "tiles":
        return tiles_frame(run_dir)
    raise ResultsError(f"frame is scores or tiles, not {which!r}")


def agg(run_dir: Path, which: str, view: str, **kw) -> dict:
    fr = frame(run_dir, which)
    out = RV.aggregate_frame(fr, view, **kw)
    out["frame"] = which
    return out


def confusion(run_dir: Path, config: str, split: str, fold: str | None = None) -> dict:
    c = _conf(load_results(run_dir), config)
    s = _split(c, split)
    if fold:
        f = next((x for x in s["folds"] if x["fold"] == fold), None)
        if f is None:
            raise ResultsError(f"no fold {fold!r} in {config}/{split}")
        cm = f.get("classification", {}).get("confusion")
        src = f"fold {fold}"
    else:
        cm = s["aggregate"].get("confusion")
        src = f"pooled over {s['aggregate'].get('n_folds', 0)} folds"
    if not cm:
        raise ResultsError(f"{config}/{split}: no confusion matrix (the model is a {c.get('kind')})")
    Mx = np.asarray(cm["matrix"], dtype=float)
    return RV._clean({"config": config, "name": c["name"], "split": split, "source": src, "labels": cm["labels"], "matrix": cm["matrix"],
                      "row_recall": cm.get("row_recall"), "n": int(Mx.sum()), "accuracy": float(np.trace(Mx) / Mx.sum()) if Mx.sum() else None,
                      "row_normalised": (Mx / np.maximum(Mx.sum(axis=1, keepdims=True), 1)).tolist(),
                      "novelty_folds": s["aggregate"].get("novelty_folds", [])})


def curves(run_dir: Path, config: str, split: str) -> dict:
    c = _conf(load_results(run_dir), config)
    s = _split(c, split)
    out = {"config": config, "name": c["name"], "split": split, "kind": c.get("kind"), "series": []}
    for f in s["folds"]:
        for key, label in (("novelty_detection", "novel vs benign in the test fold"), ("novelty_vs_training", "novel test tiles vs training tiles (in-sample reference)"),
                           ("confidence_detects_errors", "confidence as a detector of the model's own errors")):
            d = f.get(key) or {}
            if d.get("curve"):
                out["series"].append({"fold": f["fold"], "what": label, "auroc": d.get("auroc"), "auprc": d.get("auprc"), "recall_at_fpr": d.get("recall_at_fpr"),
                                      "x": [p["fpr"] for p in d["curve"]], "y": [p["tpr"] for p in d["curve"]]})
    if not out["series"]:
        out["why"] = "no detection curve: the folds hold no novel tiles, or the model gives no score"
    return RV._clean(out)


def embedding(run_dir: Path, config: str, split: str, method: str = "pca", max_points: int = 5000) -> dict:
    f = Path(run_dir) / "embeddings.npz"
    if not f.exists():
        raise ResultsError("no embeddings.npz (the models kept none, or keep_embeddings was off)")
    z = np.load(f, allow_pickle=False)
    kz, kk = f"{config}__{split}__Z", f"{config}__{split}__keys"
    if kz not in z:
        have = sorted({k.rsplit("__", 1)[0].replace("__", "/") for k in z.files})
        raise ResultsError(f"no embedding for {config}/{split}; there are: {', '.join(have) or 'none'}")
    Z, K = np.asarray(z[kz], dtype=np.float64), z[kk]
    n = Z.shape[0]
    stride = max(1, int(np.ceil(n / max_points)))
    Zs, Ks = Z[::stride], K[::stride]
    if Zs.shape[1] == 1:
        P = np.column_stack([Zs[:, 0], np.zeros(Zs.shape[0])])
        how = "the one embedding dimension on x"
        ev = None
    elif method == "umap":
        try:
            import umap
        except ImportError as e:
            raise ResultsError("umap-learn is not installed; use pca") from e
        P = umap.UMAP(n_components=2, random_state=0).fit_transform(Zs)
        how, ev = "UMAP (2-D, seed 0)", None
    else:
        from sklearn.decomposition import PCA
        p = PCA(n_components=2, random_state=0).fit(Zs)
        P = p.transform(Zs)
        how, ev = "PCA, first two components", [float(v) for v in p.explained_variance_ratio_]
    shorts = RV.short_names(np.unique(Ks["recording"]))
    return RV._clean({"config": config, "split": split, "method": how, "explained": ev, "dim": int(Z.shape[1]), "n": int(n), "n_shown": int(Zs.shape[0]),
                      "stride": stride, "points": P.tolist(),
                      "family": Ks["family"].tolist(), "workload": Ks["workload"].tolist(), "recording": [shorts.get(r, r) for r in Ks["recording"].tolist()],
                      "fold": Ks["fold"].tolist(), "t_index": Ks["t_index"].tolist()})


def _weighted_mean_maps(folds: list[dict], key: str, sub: str | None = None):
    maps, ws = [], []
    for f in folds:
        d = f.get(key)
        if not d:
            continue
        arr = d.get("mean") if sub is None else (d.get("by_true_class") or {}).get(sub)
        if arr is None:
            continue
        maps.append(np.asarray(arr, dtype=float))
        ws.append(float(f.get("n_test", 1)))
    if not maps or len({m.shape for m in maps}) != 1:
        return None
    W = np.asarray(ws) / sum(ws)
    return np.tensordot(W, np.stack(maps), axes=1)


def saliency(run_dir: Path, config: str, split: str) -> dict:
    c = _conf(load_results(run_dir), config)
    s = _split(c, split)
    mean = _weighted_mean_maps(s["folds"], "saliency")
    if mean is None:
        raise ResultsError(f"{config}/{split}: no saliency (only torch models give input gradients)")
    shape = next(f["saliency"]["shape"] for f in s["folds"] if f.get("saliency"))
    classes = sorted({k for f in s["folds"] for k in (f.get("saliency", {}).get("by_true_class") or {})})
    per = {k: _weighted_mean_maps(s["folds"], "saliency", k) for k in classes}
    return RV._clean({"config": config, "name": c["name"], "split": split, "shape": shape, "mean": mean.tolist(),
                      "by_true_class": {k: v.tolist() for k, v in per.items() if v is not None},
                      "what": "|d(max output) / d input|, averaged over test tiles, weighted by fold size"})


def importance(run_dir: Path, config: str, split: str) -> dict:
    c = _conf(load_results(run_dir), config)
    s = _split(c, split)
    folds = [f for f in s["folds"] if f.get("importance")]
    if not folds:
        raise ResultsError(f"{config}/{split}: no importance (classifiers and banks over at most 64 columns)")
    names = folds[0]["importance"]["names"]
    drops = np.array([[np.nan if v is None else v for v in f["importance"]["drop"]] for f in folds if f["importance"]["names"] == names], dtype=float)
    mean = np.nanmean(drops, axis=0)
    order = np.argsort(-np.nan_to_num(mean, nan=-np.inf))
    return RV._clean({"config": config, "name": c["name"], "split": split, "axis": folds[0]["importance"]["axis"], "names": names,
                      "mean_drop": mean.tolist(), "std_drop": np.nanstd(drops, axis=0).tolist(), "n_folds": int(drops.shape[0]),
                      "ranked": [int(i) for i in order], "baseline_mean": float(np.mean([f["importance"]["baseline"] for f in folds])),
                      "what": "drop in accuracy when the column is permuted across test tiles, mean over folds"})


def null(run_dir: Path, config: str, split: str) -> dict:
    c = _conf(load_results(run_dir), config)
    s = _split(c, split)
    per = []
    hist = None
    for f in s["folds"]:
        n = f.get("null") or {}
        if not n.get("n"):
            continue
        per.append({"fold": f["fold"], "observed": n.get("observed"), "null_mean": n.get("mean"), "null_p95": n.get("p95"), "null_max": n.get("max"),
                    "percentile": n.get("percentile_of_observed"), "p_value": n.get("p_value"), "metric": n.get("metric", "accuracy")})
        if n.get("hist"):
            hist = (np.asarray(n["hist"]) if hist is None else hist + np.asarray(n["hist"]))
    pooled = (s["aggregate"].get("pooled") or {})
    pn = pooled.get("null") or {}
    if not per and not pn.get("n"):
        raise ResultsError(f"{config}/{split}: no null (null_permutations was 0, or nothing was scored)")
    return RV._clean({"config": config, "name": c["name"], "split": split, "folds": per, "metric": (per[0]["metric"] if per else pn.get("metric", "accuracy")),
                      "hist": hist.tolist() if hist is not None else None, "edges": np.linspace(0, 1, 21).tolist(),
                      "observed_mean": float(np.mean([p["observed"] for p in per if p["observed"] is not None])) if any(p["observed"] is not None for p in per) else None,
                      "pooled": {"observed": pooled.get("accuracy", pooled.get("ari_family")), "n": pn.get("n"), "mean": pn.get("mean"), "p95": pn.get("p95"), "max": pn.get("max"),
                                 "p_value": pn.get("p_value"), "percentile": pn.get("percentile_of_observed"), "hist": pn.get("hist"), "why": pn.get("why"),
                                 "n_tiles": pooled.get("n"), "n_classes": pooled.get("n_classes")},
                      "what": (pn["what"] + ": the archetype labels permuted across the kernels, every row inheriting its kernel's, the pooled out-of-fold predictions scored as they stand"
                               if pn.get("what") else
                               "test labels permuted against the predictions: per fold where the fold holds several classes, and pooled over every fold's out-of-fold predictions")})


def calibration(run_dir: Path, config: str, split: str) -> dict:
    c = _conf(load_results(run_dir), config)
    s = _split(c, split)
    bins: dict[tuple, dict] = {}
    eces = []
    for f in s["folds"]:
        cal = (f.get("classification") or {}).get("calibration")
        if not cal:
            continue
        if cal.get("ece") is not None:
            eces.append(cal["ece"])
        for b in cal["bins"]:
            k = (b["lo"], b["hi"])
            d = bins.setdefault(k, {"lo": b["lo"], "hi": b["hi"], "n": 0, "conf_sum": 0.0, "acc_sum": 0.0})
            d["n"] += b["n"]; d["conf_sum"] += b["confidence"] * b["n"]; d["acc_sum"] += b["accuracy"] * b["n"]
    if not bins:
        raise ResultsError(f"{config}/{split}: no calibration (the model gives no probabilities)")
    rows = [{"lo": d["lo"], "hi": d["hi"], "n": d["n"], "confidence": d["conf_sum"] / d["n"], "accuracy": d["acc_sum"] / d["n"]} for d in sorted(bins.values(), key=lambda d: d["lo"])]
    return RV._clean({"config": config, "name": c["name"], "split": split, "bins": rows, "ece_mean": float(np.mean(eces)) if eces else None,
                      "proba_note": next((f.get("proba_note") for f in s["folds"] if f.get("proba_note")), None)})


def train_curves(run_dir: Path, config: str, split: str) -> dict:
    c = _conf(load_results(run_dir), config)
    s = _split(c, split)
    series = []
    for f in s["folds"]:
        if f.get("train_curve"):
            series.append({"fold": f["fold"], "x": [int(e) for e, _ in f["train_curve"]], "y": [float(l) for _, l in f["train_curve"]]})
        for st in f.get("fitted", []):
            if st.get("train_curve"):
                series.append({"fold": f["fold"] + " / " + st["module"], "x": [int(e) for e, _ in st["train_curve"]], "y": [float(l) for _, l in st["train_curve"]]})
    if not series:
        raise ResultsError(f"{config}/{split}: no training curve (only torch models keep one)")
    return RV._clean({"config": config, "name": c["name"], "split": split, "series": series, "what": "training loss per epoch"})


def tile(runs_root: Path, run: str, recording: str | None = None, t_index: int | None = None, index: int | None = None) -> dict:
    """One tile from a run's tiles.npz, as drawn: the (W, k) matrix and, for a path, the most
    active block per frame."""
    f = Path(runs_root) / run / "tiles.npz"
    if not f.exists():
        raise ResultsError(f"{run}: no tiles.npz")
    z = np.load(f, allow_pickle=False)
    K = z["tile_keys"]
    if index is None:
        m = np.ones(len(K), bool)
        if recording:
            m &= (K["recording"] == recording) | np.array([RV.short_recording(r) == recording for r in K["recording"]])
        if t_index is not None:
            m &= K["t_index"] == int(t_index)
        idx = np.flatnonzero(m)
        if idx.size == 0:
            raise ResultsError("no such tile")
        index = int(idx[0])
    X = z["X"][index]
    if np.iscomplexobj(X):
        X = np.abs(X)
    shape = str(z["shape"])
    out = {"run": run, "index": int(index), "shape": "path" if shape == "series" else shape, "w": int(z["w"]), "h": int(z["h"]),
           "recording": RV.short_recording(str(K["recording"][index])), "recording_id": str(K["recording"][index]), "t_index": int(K["t_index"][index]),
           "seq_start": int(K["seq_start"][index]), "family": str(K["family"][index]), "workload": str(K["workload"][index]),
           "matrix": (X if X.ndim == 2 else X[:, None]).astype(float).tolist(), "n_tiles": int(len(K)),
           "channels": [str(c) for c in z["channels"]], "n_blocks": int(z["n_blocks"])}
    if out["shape"] == "path":
        M = X if X.ndim == 2 else X[:, None]
        out["most_active_block"] = np.abs(M).argmax(axis=1).astype(int).tolist()
    if "pages" in z:
        out["pages"] = np.asarray(z["pages"]).astype(int).tolist()
    return RV._clean(out)


def tiles_index(runs_root: Path, run: str) -> dict:
    f = Path(runs_root) / run / "tiles.npz"
    if not f.exists():
        raise ResultsError(f"{run}: no tiles.npz")
    z = np.load(f, allow_pickle=False)
    K = z["tile_keys"]
    recs = {}
    for i, r in enumerate(K["recording"].tolist()):
        d = recs.setdefault(r, {"short": RV.short_recording(r), "family": str(K["family"][i]), "workload": str(K["workload"][i]), "t_index": []})
        d["t_index"].append(int(K["t_index"][i]))
    return {"run": run, "shape": "path" if str(z["shape"]) == "series" else str(z["shape"]), "n_tiles": int(len(K)), "recordings": recs}
