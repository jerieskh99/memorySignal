#!/usr/bin/env python3
"""executor.py -- run every configuration of a Learn pipeline; status, control, outputs, sidecar.

    python3 -m plan10_analysis.learn.executor run PIPELINE.json --out-dir OUT --runs-root RUNS
        [--acknowledge-all]

Refuses what pipeline.py refuses (hard: exit 1; unacknowledged soft: exit 2). Then, for every
configuration x split x seed x fold: the training rows fit every fitted step (preprocess,
represent, model) and the test rows are scored; nothing sees a test label before scoring.
status.json is rewritten after every fold and control.json is read between fits
({"command": "stop" | "pause" | "run"}), the same protocol as the extraction runner.

Output directory (OUT/<label>/):
  learn_results.json   every score per configuration, split, fold and seed, and per (configuration,
                       split) the mean and std over folds, the pooled confusion, the null, the bootstrap
  predictions.npz      one row per scored test tile: keys, y_true, y_pred, p_true, p_max, novelty
  embeddings.npz       test-tile embeddings per (configuration, split), when the model has them
  sidecar.json         the pipeline, the input runs and their sidecar hashes, folds per split
                       (recording ids), seeds, versions, device, wall time, the sweep size
  status.json, run.log, control.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
QEMU_DIR = HERE.parent.parent
if str(QEMU_DIR) not in sys.path:
    sys.path.insert(0, str(QEMU_DIR))

from plan10_analysis.learn import data as D, models as M, pipeline as PL, preprocess as P, scores as SC, splits as SP  # noqa: E402

KEY_DT = np.dtype([("config", "U16"), ("split", "U16"), ("fold", "U256"), ("seed", "i4"), ("recording", "U256"), ("workload", "U128"),
                   ("family", "U32"), ("block", "i4"), ("t_index", "i4"), ("seq_start", "i4")])
PRED_DT = np.dtype([("y_true", "U64"), ("y_pred", "U64"), ("p_true", "f4"), ("p_max", "f4"), ("novelty", "f4"), ("correct", "i1"), ("cluster", "i4")])


class Stop(Exception):
    pass


class RunRefused(RuntimeError):
    def __init__(self, code, verdict):
        super().__init__(f"pipeline refused (exit {code})")
        self.code, self.verdict = code, verdict


class Status:
    def __init__(self, out_dir: Path, label: str):
        self.path, self.control, self.log = out_dir / "status.json", out_dir / "control.json", out_dir / "run.log"
        self.d = {"schema": "plan10.learn_status.v1", "label": label, "state": "starting", "phase": None, "configuration": None,
                  "configuration_index": 0, "n_configurations": 0, "split": None, "fold": None, "fold_index": 0, "n_folds": 0,
                  "fits_done": 0, "fits_total": 0, "message": "", "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "updated_at": None, "errors": [], "per_configuration": {}}
        self.write()

    def write(self, **kw):
        self.d.update(kw)
        self.d["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.d, indent=1, default=str))
        os.replace(tmp, self.path)

    def logline(self, msg: str):
        line = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {msg}\n"
        with self.log.open("a") as f:
            f.write(line)
        print(msg, flush=True)

    def check_control(self):
        if not self.control.exists():
            return
        try:
            cmd = json.loads(self.control.read_text()).get("command")
        except (OSError, json.JSONDecodeError):
            return
        if cmd == "stop":
            raise Stop()
        while cmd == "pause":
            self.write(state="paused")
            time.sleep(1.0)
            try:
                cmd = json.loads(self.control.read_text()).get("command")
            except (OSError, json.JSONDecodeError):
                cmd = "run"
            if cmd == "stop":
                raise Stop()
        if self.d["state"] == "paused":
            self.write(state="running")


def context_for(runs_root: Path, registry=None) -> PL.Context:
    """What every run under the root offers, with the counts validation and the estimate need."""
    runs = {}
    for d in sorted(Path(runs_root).iterdir()) if Path(runs_root).is_dir() else []:
        if not d.is_dir():
            continue
        a = D.available(d)
        if not (a["features"] or a["tiles"]):
            continue
        try:
            k = D.keys_only(d, "features" if a["features"] else "tiles")
            a.update(n_families=len(set(k["family"].tolist())), n_workloads=len(set(k["workload"].tolist())),
                     n_recordings=len(set(k["recording"].tolist())), n_campaigns=len(set(k["campaign"].tolist())),
                     families=sorted(set(k["family"].tolist())))
            # what target archetype would keep: kernels of the twelve, their archetypes, and how many rows fall outside
            try:
                table = D.archetype_table()
                arch = np.asarray([D.archetype_of_workload(w, table) for w in k["workload"]])
                kernels = sorted({D.kernel_of_workload(w) for w, x in zip(k["workload"], arch) if x})
                per = {}
                for kr in kernels:
                    per.setdefault(table[kr], []).append(kr)
                a.update(n_kernels=len(kernels), archetypes=sorted(per), archetype_rows_dropped=int((arch == "").sum()),
                         single_kernel_archetypes=sum(1 for v in per.values() if len(v) == 1))
            except D.DataError as e:
                a["archetype_error"] = str(e)
        except D.DataError as e:
            a["error"] = str(e)
        runs[d.name] = a
    return PL.Context(runs, registry)


def run(pipeline_path: Path, out_dir: Path, runs_root: Path, acknowledge_all: bool = False) -> int:
    pipeline_path = Path(pipeline_path)
    pl = json.loads(pipeline_path.read_text())
    out_dir = Path(out_dir) / pl.get("label", "learn")
    out_dir.mkdir(parents=True, exist_ok=True)
    st = Status(out_dir, pl.get("label", "learn"))
    try:
        return _run(pl, pipeline_path, out_dir, Path(runs_root), st, acknowledge_all)
    except Stop:
        st.write(state="stopped", message="stopped by control.json")
        st.logline("stopped")
        return 130
    except RunRefused as e:
        st.write(state="refused", message=str(e), verdict=e.verdict)
        st.logline(f"refused: {json.dumps(e.verdict)[:600]}")
        return e.code
    except Exception as e:  # noqa: BLE001
        st.write(state="failed", message=f"{type(e).__name__}: {e}")
        st.logline("failed:\n" + traceback.format_exc())
        return 1


def _sha(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def _run(pl, pipeline_path, out_dir, runs_root, st, acknowledge_all) -> int:
    t0 = time.time()
    ctx = context_for(runs_root)
    issues = PL.validate(pl, ctx)
    if acknowledge_all:
        pl = dict(pl, acknowledged=list(pl.get("acknowledged", [])) + [
            {"id": i["id"], "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "note": "acknowledge-all at launch"}
            for i in issues if i["sev"] == "soft" and i.get("id") and i["id"] not in {a.get("id") for a in pl.get("acknowledged", [])}])
    code, verdict = PL.verdict(issues, pl)
    if code:
        raise RunRefused(code, {k: v for k, v in verdict.items() if k in ("hard", "soft_unacknowledged")})
    confs = PL.configurations(pl)
    runp = pl.get("run") or {}
    splits_wanted, seeds = list(runp.get("splits") or ["loro"]), [int(s) for s in (runp.get("seeds") or [0])]
    test_frac, target = float(runp.get("test_frac", 0.2)), runp.get("target", "family")
    est = PL.estimate(pl, ctx)
    st.write(state="running", phase="plan", n_configurations=len(confs), fits_total=est["fits"],
             verdict={"notes": [n["msg"] for n in verdict["notes"]], "soft_acknowledged": [s["msg"] for s in verdict["soft_acknowledged"]]})
    st.logline(f"{len(confs)} configuration(s) x {len(splits_wanted)} split(s) x {len(seeds)} seed(s); target {target}; runs {est['runs']}")

    datasets: dict[tuple, D.Dataset] = {}
    results = {"schema": "plan10.learn_results.v1", "label": pl.get("label"), "target": target, "configurations": [], "sweep": est}
    pred_keys, pred_rows = [], []
    emb_store: dict[str, dict] = {}
    folds_used: dict[str, dict] = {}
    fits_done = 0
    for ci, conf in enumerate(confs, 1):
        st.check_control()
        st.write(phase="fit", configuration=conf["id"], configuration_index=ci)
        steps = conf["steps"]
        inp = next(s for s in steps if s["module"] == "input")
        ip = PL.params_of(ctx.mod["input"], inp.get("params"))
        dkey = (ip["run"], ip.get("source", "features"))
        if dkey not in datasets:
            datasets[dkey] = D.load(runs_root / ip["run"], ip.get("source", "features"))
        ds = datasets[dkey]
        arch_info = None
        if target == "archetype":
            # rows without an archetype (the idle cells, anything not among the twelve kernels) leave
            # the run here, before any fold is cut; how many is recorded, never which
            keep = ds.archetype_rows()
            arch_info = {"rows_dropped": int((~keep).sum()), "rows_kept": int(keep.sum()), "rows_total": int(ds.n),
                         "table": "plan11_encoding_ladder/schema.py ARCHETYPE_OF"}
            if keep.sum() == 0:
                raise ValueError(f"{ip['run']}: no row belongs to one of the twelve kernels; target archetype has nothing to learn")
            ds = ds.subset(keep, "no archetype: not one of the twelve kernels")
            table = D.archetype_table()
            kernels = sorted({D.kernel_of_workload(w) for w in ds.keys["workload"]})
            per = {}
            for kr in kernels:
                per.setdefault(table[kr], []).append(kr)
            arch_info.update(kernels=kernels, archetypes=sorted(per), kernels_per_archetype={a: len(v) for a, v in per.items()})
            results.setdefault("archetype", arch_info)
            st.logline(f"target archetype: {arch_info['rows_dropped']} of {arch_info['rows_total']} rows have no archetype and are dropped; "
                       f"{len(kernels)} kernels over {len(per)} archetypes; leave-one-workload-out is leave one kernel out; "
                       f"null = kernel-level permutation, no refit")
        y_all = ds.labels(target)
        fam_all = ds.labels("family")
        fold_keys = dict(ds.keys)
        if target == "archetype":
            fold_keys["archetype"] = y_all
        crec = {"id": conf["id"], "name": conf["name"], "input": {"run": ip["run"], "source": ip.get("source", "features"), "shape": ds.shape,
                                                                     "n": ds.n, "tile_shape": list(ds.X.shape[1:])},
                "steps": [{"tier": ctx.mod[s["module"]]["tier"], "module": s["module"], "params": PL.params_of(ctx.mod[s["module"]], s.get("params"))} for s in steps],
                "model": next((s["module"] for s in steps if ctx.mod[s["module"]]["tier"] == "model"), None),
                "kind": next((ctx.mod[s["module"]]["kind"] for s in steps if ctx.mod[s["module"]]["tier"] == "model"), None),
                "splits": {}}
        if arch_info:
            crec["input"]["archetype"] = {k: arch_info[k] for k in ("rows_dropped", "rows_kept", "rows_total")}
        score_p = PL.params_of(ctx.mod["score"], next((s.get("params") for s in steps if s["module"] == "score"), {}))
        out_p = PL.params_of(ctx.mod["write"], next((s.get("params") for s in steps if s["module"] == "write"), {}))
        for split in splits_wanted:
            folds = SP.folds_for(fold_keys, split, test_frac, novelty_key="archetype" if target == "archetype" else "family")
            SP.assert_grouped(folds, ds.keys, SP.GROUP_OF[split])
            folds_used.setdefault(ip["run"], {})[split] = [{"name": f["name"], "held_out": f["held_out"], "novelty": bool(f.get("novelty")),
                                                            "n_train": int(f["train"].size), "n_test": int(f["test"].size)} for f in folds]
            srec = {"folds": [], "skipped": [], "aggregate": {}, "label": SP.split_label(split, target)}
            pooled = {"y_true": [], "y_pred": [], "recording": [], "workload": [], "family": [], "cluster": [], "kind": None, "seed": seeds[0]}
            for seed in seeds:
                for fi, fold in enumerate(folds, 1):
                    st.check_control()
                    st.write(split=split, fold=fold["name"], fold_index=fi, n_folds=len(folds), fits_done=fits_done)
                    tr, te = fold["train"], fold["test"]
                    if tr.size == 0 or te.size == 0:
                        srec["skipped"].append({"fold": fold["name"], "seed": seed, "why": "no training rows" if tr.size == 0 else "no test rows"})
                        continue
                    t1 = time.time()
                    try:
                        frec, tile_rows, tile_keys, Z = _fit_fold(ctx, ds, steps, y_all, fam_all, tr, te, seed, target, score_p, conf, split, fold)
                    except (M.ModelError, P.PreprocessError, ValueError) as e:
                        srec["skipped"].append({"fold": fold["name"], "seed": seed, "why": f"{type(e).__name__}: {e}"})
                        st.logline(f"[{conf['id']}] {split} {fold['name']} seed {seed}: skipped ({e})")
                        continue
                    frec["seconds"] = round(time.time() - t1, 3)
                    srec["folds"].append(frec)
                    pooled["kind"] = frec["kind"]
                    pooled["y_true"].extend(r[0] for r in tile_rows); pooled["y_pred"].extend(r[1] for r in tile_rows)
                    pooled["cluster"].extend(r[6] for r in tile_rows)
                    pooled["recording"].extend(k[4] for k in tile_keys); pooled["family"].extend(k[6] for k in tile_keys)
                    pooled["workload"].extend(k[5] for k in tile_keys)
                    if out_p.get("keep_predictions", True):
                        pred_rows.extend(tile_rows)
                        pred_keys.extend(tile_keys)
                    if Z is not None and out_p.get("keep_embeddings", True):
                        e = emb_store.setdefault(f"{conf['id']}|{split}", {"Z": [], "keys": []})
                        e["Z"].append(Z)
                        e["keys"].extend(tile_keys)
                    fits_done += 1
                    st.d["per_configuration"].setdefault(conf["id"], {})[split] = f"{len(srec['folds'])}/{len(folds) * len(seeds)} folds"
            srec["aggregate"] = _aggregate(srec["folds"])
            srec["aggregate"]["pooled"] = _pooled(pooled, int(score_p.get("null_permutations", 200)), int(score_p.get("bootstrap", 200)),
                                                  target, int(score_p.get("kernel_null_permutations", 500)))
            crec["splits"][split] = srec
            st.logline(f"[{conf['id']}] {conf['name']} | {split}: " + _summary_line(srec))
        results["configurations"].append(crec)
    st.write(phase="write", fits_done=fits_done)
    if pred_rows:
        np.savez_compressed(out_dir / "predictions.npz", keys=np.array(pred_keys, dtype=KEY_DT), pred=np.array(pred_rows, dtype=PRED_DT))
    if emb_store:
        packed = {}
        for k, e in emb_store.items():
            Z = np.concatenate(e["Z"]).astype(np.float32)
            packed[k.replace("|", "__") + "__Z"] = Z
            packed[k.replace("|", "__") + "__keys"] = np.array(e["keys"], dtype=KEY_DT)
        np.savez_compressed(out_dir / "embeddings.npz", **packed)
    results["fits_done"] = fits_done
    results["seconds"] = round(time.time() - t0, 2)
    (out_dir / "learn_results.json").write_text(json.dumps(results, indent=1, default=_json_default))
    try:
        import sklearn
        skv = sklearn.__version__
    except ImportError:
        skv = None
    try:
        import torch
        tv, dev = torch.__version__, os.environ.get("PLAN10_TORCH_DEVICE", "cpu")
    except ImportError:
        tv, dev = None, None
    sidecar = {"schema": "plan10.learn_sidecar.v1", "label": pl.get("label"), "pipeline": pl, "pipeline_file": str(pipeline_path),
               "acknowledged": pl.get("acknowledged", []), "runs_root": str(runs_root),
               "inputs": {lab: {"dir": str(runs_root / lab), "sidecar_sha256_16": _sha(runs_root / lab / "sidecar.json"),
                                "features_sha256_16": _sha(runs_root / lab / "features.npz") if (runs_root / lab / "features.npz").exists() else None,
                                "tiles_sha256_16": _sha(runs_root / lab / "tiles.npz") if (runs_root / lab / "tiles.npz").exists() else None,
                                "folds": folds_used.get(lab, {})} for lab in est["runs"]},
               "target": target, "splits": splits_wanted, "seeds": seeds, "test_frac": test_frac,
               "sweep": est, "fits_done": fits_done, "configurations": [{"id": c["id"], "name": c["name"]} for c in confs],
               "python": platform.python_version(), "numpy": np.__version__, "sklearn": skv, "torch": tv, "torch_device": dev,
               "determinism": "seeded (numpy, torch); torch on the CPU by default", "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "seconds": results["seconds"]}
    (out_dir / "sidecar.json").write_text(json.dumps(sidecar, indent=1, default=_json_default))
    st.write(state="done", phase="done", fits_done=fits_done, message=f"{len(confs)} configuration(s), {fits_done} fits in {results['seconds']}s")
    st.logline(st.d["message"])
    return 0


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _sub_keys(keys: dict, idx: np.ndarray) -> dict:
    return {k: np.asarray(v)[idx] for k, v in keys.items()}


def _fit_fold(ctx, ds, steps, y_all, fam_all, tr, te, seed, target, score_p, conf, split, fold):
    """One fold of one configuration: fit on train, score on test. Returns the fold record, the
    per-tile prediction rows and keys, and the test embedding (or None)."""
    Xtr, Xte = ds.X[tr], ds.X[te]
    ytr, yte = y_all[tr], y_all[te]
    ktr, kte = _sub_keys(ds.keys, tr), _sub_keys(ds.keys, te)
    shape = ds.shape
    fitted = []
    model = None
    model_id = None
    for s in steps:
        m = ctx.mod[s["module"]]
        p = PL.params_of(m, s.get("params"))
        if m["tier"] == "preprocess":
            t = P.make(s["module"], p, seed).fit(Xtr, ytr, ktr)
            Xtr, Xte = t.transform(Xtr, ktr), t.transform(Xte, kte)
            shape = P.out_shape(s["module"], shape)
            fitted.append({"module": s["module"], **t.describe()})
        elif m["tier"] == "represent":
            enc = M.make(s["module"], p, seed, {}, shape).fit(Xtr, ytr, ktr)
            Xtr, Xte = enc.embed(Xtr), enc.embed(Xte)
            shape = "embedding"
            fitted.append({"module": s["module"], "dim": int(Xtr.shape[1]), "train_curve": enc.train_curve})
        elif m["tier"] == "model":
            model_id = s["module"]
            model = M.make(s["module"], p, seed, {"n_families": len(set(fam_all[tr].tolist())), "n_workloads": len(set(ktr["workload"].tolist()))}, shape)
            model.fit(Xtr, ytr, ktr)
    if model is None:
        raise M.ModelError("no model in the configuration")
    kind = model.kind
    rec = {"fold": fold["name"], "split": split, "seed": seed, "held_out": fold.get("held_out"), "novelty_fold": bool(fold.get("novelty")),
           "n_train": int(tr.size), "n_test": int(te.size), "train_classes": sorted(set(ytr.tolist())), "test_classes": sorted(set(yte.tolist())),
           "fitted": fitted, "model": {"module": model_id, **model.describe()}, "kind": kind}
    sc = model.scores(Xte)
    n_te = int(te.size)
    y_pred = np.array([""] * n_te, dtype="U64")
    p_true = np.full(n_te, np.nan, dtype=np.float32)
    p_max = np.full(n_te, np.nan, dtype=np.float32)
    novelty = np.asarray(sc["novelty"], dtype=np.float32) if "novelty" in sc else np.full(n_te, np.nan, dtype=np.float32)
    cluster = np.full(n_te, -1, dtype=np.int32)
    Z = model.embed(Xte)
    if kind in ("classifier", "reconstructor"):
        y_pred = np.asarray(model.predict(Xte)).astype("U64")
        proba, classes = sc.get("proba"), model.classes_
        cls = SC.classification(yte, y_pred, proba, classes, calibration_bins=int(score_p.get("calibration_bins", 10)))
        rec["classification"] = cls
        rec["train_accuracy"] = SC._f(SC.accuracy(ytr, model.predict(Xtr)))
        rec["gap"] = SC._f(rec["train_accuracy"] - cls["accuracy"]) if rec["train_accuracy"] is not None and cls["accuracy"] is not None else None
        n_null, n_boot = int(score_p.get("null_permutations", 200)), int(score_p.get("bootstrap", 200))
        rec["null"] = SC.null_permutation(yte, y_pred, n_null, seed)
        rec["null_balanced"] = SC.null_permutation(yte, y_pred, n_null, seed, SC.balanced_accuracy) if n_null else {"n": 0}
        rec["bootstrap"] = SC.bootstrap(yte, y_pred, kte["recording"], n_boot, seed)
        if proba is not None and classes is not None and len(classes) == proba.shape[1]:
            ci = {c: i for i, c in enumerate(classes)}
            p_max = proba.max(axis=1).astype(np.float32)
            p_true = np.array([proba[i, ci[t]] if t in ci else np.nan for i, t in enumerate(yte)], dtype=np.float32)
            if sc.get("proba_note"):
                rec["proba_note"] = sc["proba_note"]
            # the confidence as a detector of its own mistakes, and of novelty where the fold has any
            rec["confidence_detects_errors"] = SC.detection(y_pred != yte, -p_max, float(score_p.get("fpr", 0.05)))
        if "novelty" in sc:
            train_fams = set(fam_all[tr].tolist())
            is_novel = np.array([f not in train_fams for f in fam_all[te]])
            rec["novelty_detection"] = SC.detection(is_novel, novelty, float(score_p.get("fpr", 0.05)))
            rec["novelty_by_class"] = SC.per_class_score_stats(novelty, yte)
            if is_novel.any():
                rec["novelty_vs_training"] = SC.detection_vs_reference(novelty[is_novel], np.asarray(model.scores(Xtr)["novelty"]), float(score_p.get("fpr", 0.05)))
        if "recon_error" in sc:
            E = np.asarray(sc["recon_error"])
            rec["recon_error"] = {"members": sc.get("members"), "by_true_class": {c: [SC._f(v) for v in E[yte == c].mean(axis=0)] for c in sorted(set(yte.tolist()))}}
        if score_p.get("importance", True) and Xte.shape[-1] <= 64 and n_te >= 4:
            names = ds.names if (shape == "rows" and Xte.shape[1] == len(ds.names)) else ([f"block{j}" for j in range(Xte.shape[-1])] if shape == "path" else None)
            rec["importance"] = SC.importance(model.predict, Xte, yte, seed=seed, n_repeats=2, axis="columns" if Xte.ndim == 2 else "last axis", names=names)
        if score_p.get("saliency", True):
            sal = model.saliency(Xte)
            if sal is not None:
                rec["saliency"] = {"mean": np.asarray(sal).mean(axis=0).astype(float).tolist(), "shape": list(np.asarray(sal).shape[1:]),
                                   "by_true_class": {c: np.asarray(sal)[yte == c].mean(axis=0).astype(float).tolist() for c in sorted(set(yte.tolist()))}}
    elif kind == "clusterer":
        cluster = np.asarray(model.predict(Xte)).astype(np.int32)
        rec["clustering"] = SC.clustering(cluster, fam_all[te], kte["workload"], Xte)
        n_null = int(score_p.get("null_permutations", 200))
        from sklearn.metrics import adjusted_rand_score
        rec["null"] = SC.null_permutation(fam_all[te], cluster.astype(str), n_null, seed, lambda yt, yp: adjusted_rand_score(yt, yp))
        rec["null"]["metric"] = "ari_family"
        y_pred = cluster.astype("U64")
        if "proba" in sc:
            p_max = np.asarray(sc["proba"]).max(axis=1).astype(np.float32)
    elif kind == "novelty":
        train_fams = set(fam_all[tr].tolist())
        is_novel = np.array([f not in train_fams for f in fam_all[te]])
        rec["novelty_detection"] = SC.detection(is_novel, novelty, float(score_p.get("fpr", 0.05)))
        rec["novelty_by_class"] = SC.per_class_score_stats(novelty, yte)
        train_nov = np.asarray(model.scores(Xtr)["novelty"])
        rec["novelty_train"] = SC.per_class_score_stats(train_nov, ytr)
        if is_novel.any():
            rec["novelty_vs_training"] = SC.detection_vs_reference(novelty[is_novel], train_nov, float(score_p.get("fpr", 0.05)))
        if score_p.get("saliency", True):
            sal = model.saliency(Xte)
            if sal is not None:
                rec["saliency"] = {"mean": np.asarray(sal).mean(axis=0).astype(float).tolist(), "shape": list(np.asarray(sal).shape[1:])}
    if Z is not None:
        rec["embedding"] = SC.embedding_quality(Z, yte)
    rec["train_curve"] = model.train_curve
    correct = (y_pred == yte).astype(np.int8) if kind in ("classifier", "reconstructor") else np.full(n_te, -1, dtype=np.int8)
    tile_keys = [(conf["id"], split, fold["name"], seed, kte["recording"][i], kte["workload"][i], kte["family"][i], int(kte.get("block", np.full(n_te, -1))[i]),
                  int(kte["t_index"][i]), int(kte["seq_start"][i])) for i in range(n_te)]
    tile_rows = [(yte[i], y_pred[i], float(p_true[i]), float(p_max[i]), float(novelty[i]), int(correct[i]), int(cluster[i])) for i in range(n_te)]
    return rec, tile_rows, tile_keys, (np.asarray(Z, dtype=np.float32) if Z is not None else None)


AGG_KEYS = [("accuracy", ("classification", "accuracy")), ("balanced_accuracy", ("classification", "balanced_accuracy")),
            ("macro_f1", ("classification", "macro_f1")), ("kappa", ("classification", "kappa")), ("mcc", ("classification", "mcc")),
            ("top2_accuracy", ("classification", "top2_accuracy")), ("brier", ("classification", "brier")), ("log_loss", ("classification", "log_loss")),
            ("ece", ("classification", "calibration", "ece")), ("train_accuracy", ("train_accuracy",)), ("gap", ("gap",)),
            ("null_mean", ("null", "mean")), ("null_p95", ("null", "p95")), ("null_p_value", ("null", "p_value")),
            ("bootstrap_lo", ("bootstrap", "ci95", 0)), ("bootstrap_hi", ("bootstrap", "ci95", 1)),
            ("auroc", ("novelty_detection", "auroc")), ("auprc", ("novelty_detection", "auprc")), ("recall_at_fpr", ("novelty_detection", "recall_at_fpr")),
            ("error_auroc", ("confidence_detects_errors", "auroc")),
            ("auroc_vs_training", ("novelty_vs_training", "auroc")), ("recall_at_fpr_vs_training", ("novelty_vs_training", "recall_at_fpr")),
            ("ari_family", ("clustering", "ari_family")), ("nmi_family", ("clustering", "nmi_family")), ("ari_workload", ("clustering", "ari_workload")),
            ("nmi_workload", ("clustering", "nmi_workload")), ("purity_family", ("clustering", "purity_family")), ("silhouette", ("clustering", "silhouette")),
            ("davies_bouldin", ("clustering", "davies_bouldin")), ("emb_silhouette", ("embedding", "silhouette")), ("emb_knn_purity", ("embedding", "knn_purity")),
            ("seconds", ("seconds",))]


def _dig(d, path):
    for k in path:
        if isinstance(d, dict):
            d = d.get(k)
        elif isinstance(d, list) and isinstance(k, int) and k < len(d):
            d = d[k]
        else:
            return None
    return d if isinstance(d, (int, float)) and not isinstance(d, bool) else None


def _aggregate(folds: list[dict]) -> dict:
    out = {"n_folds": len(folds), "metrics": {}}
    for name, path in AGG_KEYS:
        vals = [v for v in (_dig(f, path) for f in folds) if v is not None and np.isfinite(v)]
        if vals:
            out["metrics"][name] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "min": float(np.min(vals)), "max": float(np.max(vals)), "n": len(vals)}
    # pooled confusion over folds (labels unioned)
    labs = sorted({l for f in folds for l in (f.get("classification", {}).get("confusion", {}).get("labels") or [])})
    if labs:
        Mx = np.zeros((len(labs), len(labs)), dtype=int)
        for f in folds:
            c = f.get("classification", {}).get("confusion")
            if not c or not c.get("matrix"):
                continue
            for i, a in enumerate(c["labels"]):
                for j, b in enumerate(c["labels"]):
                    Mx[labs.index(a), labs.index(b)] += c["matrix"][i][j]
        out["confusion"] = {"labels": labs, "matrix": Mx.tolist(), "row_recall": [(float(Mx[i, i] / Mx[i].sum()) if Mx[i].sum() else None) for i in range(len(labs))]}
        pooled_n = int(Mx.sum())
        out["pooled_accuracy"] = float(np.trace(Mx) / pooled_n) if pooled_n else None
    nov = [f["novelty_detection"]["curve"] for f in folds if f.get("novelty_detection", {}).get("curve")]
    if nov:
        out["novelty_curves"] = nov
    out["novelty_folds"] = [f["fold"] for f in folds if f.get("novelty_fold")]
    return out


def _pooled(pooled: dict, n_null: int, n_boot: int, target: str = "family", n_kernel_null: int = 500) -> dict:
    """Out-of-fold predictions of every fold of a split, scored together: the accuracy, the
    permutation null and the bootstrap over recordings that a leave-one-out design needs, since
    a single fold often holds one class and has no null of its own.

    For target archetype the null is the kernel-level permutation (the archetype labels
    permuted across the kernels, every row inheriting its kernel's, no refit), and beside the
    accuracy: the majority-archetype baseline, recall per archetype, macro recall over the
    archetypes with at least three kernels."""
    yt, yp = np.asarray(pooled["y_true"]).astype(str), np.asarray(pooled["y_pred"]).astype(str)
    if yt.size == 0:
        return {"n": 0}
    out = {"n": int(yt.size), "n_recordings": int(len(set(pooled["recording"]))), "n_classes": int(len(set(yt.tolist())))}
    if pooled["kind"] in ("classifier", "reconstructor"):
        out["accuracy"] = SC._f(SC.accuracy(yt, yp))
        out["balanced_accuracy"] = SC.balanced_accuracy(yt, yp)
        out["macro_f1"] = SC.macro_f1(yt, yp)
        if target == "archetype":
            kernels = np.asarray([D.kernel_of_workload(w) for w in pooled["workload"]])
            out["null"] = SC.null_permutation_grouped(yt, yp, kernels, n_kernel_null, pooled["seed"])
            out["null_balanced"] = SC.null_permutation_grouped(yt, yp, kernels, n_kernel_null, pooled["seed"], SC.balanced_accuracy)
            out["archetype"] = SC.archetype_report(yt, yp, kernels, D.archetype_table())
            out["n_kernels"] = int(len(set(kernels.tolist())))
        else:
            out["null"] = SC.null_permutation(yt, yp, n_null, pooled["seed"])
            out["null_balanced"] = SC.null_permutation(yt, yp, n_null, pooled["seed"], SC.balanced_accuracy)
        out["bootstrap"] = SC.bootstrap(yt, yp, pooled["recording"], n_boot, pooled["seed"])
    elif pooled["kind"] == "clusterer":
        from sklearn.metrics import adjusted_rand_score
        fam, cl = np.asarray(pooled["family"]).astype(str), np.asarray(pooled["cluster"]).astype(str)
        out["ari_family"] = SC._f(adjusted_rand_score(fam, cl))
        out["null"] = SC.null_permutation(fam, cl, n_null, pooled["seed"], lambda a, b: adjusted_rand_score(a, b))
        out["null"]["metric"] = "ari_family"
        out["note"] = "cluster ids are per fold; pooled ARI compares within-fold structure across folds"
    return out


def _summary_line(srec: dict) -> str:
    m = srec["aggregate"].get("metrics", {})
    parts = []
    for k in ("accuracy", "balanced_accuracy", "macro_f1", "auroc", "auroc_vs_training", "recall_at_fpr_vs_training", "ari_family", "nmi_family"):
        if k in m:
            parts.append(f"{k} {m[k]['mean']:.3f}±{m[k]['std']:.3f}")
    pooled = srec["aggregate"].get("pooled") or {}
    if pooled.get("accuracy") is not None:
        nl = pooled.get("null") or {}
        parts.append(f"pooled {pooled['accuracy']:.3f}" + (f" vs null {nl['mean']:.3f} (p {nl['p_value']:.3f}{', ' + nl['what'] if nl.get('what') else ''})" if nl.get("n") else ""))
        ar = pooled.get("archetype") or {}
        if ar.get("majority_baseline") is not None:
            parts.append(f"majority-archetype baseline {ar['majority_baseline']:.3f}"
                         + (f", macro recall over {len(ar['macro_recall_3plus_over'])} archetype(s) with 3+ kernels {ar['macro_recall_3plus']:.3f}" if ar.get("macro_recall_3plus") is not None else ""))
    elif "null_mean" in m and "accuracy" in m:
        parts.append(f"null {m['null_mean']['mean']:.3f}")
    if srec["skipped"]:
        parts.append(f"{len(srec['skipped'])} skipped")
    return ", ".join(parts) if parts else "nothing scored"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("pipeline", type=Path)
    r.add_argument("--out-dir", type=Path, required=True)
    r.add_argument("--runs-root", type=Path, required=True)
    r.add_argument("--acknowledge-all", action="store_true")
    a = ap.parse_args()
    return run(a.pipeline, a.out_dir, a.runs_root, a.acknowledge_all)


if __name__ == "__main__":
    raise SystemExit(main())
