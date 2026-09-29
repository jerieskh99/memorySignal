#!/usr/bin/env python3
"""plan10_analysis.learn: the palette, the folds, the data, the validator, the executor end to
end over synthetic rows, paths and images, and the results views.

No differ, no corpus: synthetic run directories with known labels. The torch models run for two
epochs; what is checked is that every piece returns the shapes and refusals it promises, that
nothing leaks across a fold, and that the numbers are the ones numpy would give.

Run:  python3 tests/test_plan10_learn.py
      pytest tests/test_plan10_learn.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

QEMU_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QEMU_DIR))
from plan10_analysis.learn import data as D, executor as E, models as M, pipeline as PL, preprocess as P, registry as R, results as LR, scores as SC, splits as SP  # noqa: E402

KEY_DT = np.dtype([("recording", "U256"), ("workload", "U128"), ("family", "U32"), ("block", "i4"), ("t_index", "i4"), ("seq_start", "i4")])


def _rec(fam, wl, seed, camp):
    return f"{fam}/{wl}/args_--seed_{seed}_{seed:08x}/rep001__{camp}"


def make_runs(root: Path, n_t: int = 12, seed: int = 0):
    """Three families, two workloads each, two recordings per workload in two campaigns; rows,
    paths and images with a family-dependent mean so a fair model can tell them apart."""
    rng = np.random.default_rng(seed)
    fams = {"cpu": ["cpu_a", "cpu_b"], "io": ["io_a", "io_b"], "mem": ["mem_a", "mem_b"]}
    keys, rows, paths, imgs = [], [], [], []
    for fi, (fam, wls) in enumerate(fams.items()):
        for wl in wls:
            for ri, camp in enumerate(("exp-1", "exp-2")):
                for t in range(n_t):
                    keys.append((_rec(fam, wl, 100 * fi + ri, camp), wl, fam, -1, t, 1 + 4 * t))
                    rows.append(rng.normal(3 * fi, 1.0, size=6))
                    paths.append(rng.normal(3 * fi, 1.0, size=(8, 4)))
                    imgs.append(rng.normal(3 * fi, 1.0, size=(8, 20)))
    tk = np.array(keys, dtype=KEY_DT)
    out = {}
    for name, X, shape in (("rows_run", np.asarray(rows, np.float32), None), ("path_run", np.asarray(paths, np.float32), "path"), ("image_run", np.asarray(imgs, np.float32), "image")):
        d = root / name
        d.mkdir(parents=True)
        (d / "scheme.json").write_text("{}")
        (d / "status.json").write_text(json.dumps({"state": "done", "label": name}))
        side = {"label": name, "written_at": "2026-09-17T00:00:00+00:00", "speed": 2, "acknowledged": [], "n_rows": len(keys), "n_features": 6,
                "scheme": {"nodes": [{"module": "cells"}, {"module": "write"}]}, "recordings": []}
        if shape is None:
            np.savez_compressed(d / "features.npz", X=X, feature_names=np.array(["m0", "m1", "m2", "m3", "m4", "m5"], dtype="U64"), tile_keys=tk)
        else:
            extra = {"pages": np.arange(20, dtype=np.int32)} if shape == "image" else {}
            np.savez_compressed(d / "tiles.npz", X=X, tile_keys=tk, shape=np.array(shape), w=np.array(8), h=np.array(4),
                                channels=np.array(["h"], dtype="U64"), n_blocks=np.array(4 if shape == "path" else 0), complex=np.array(False), **extra)
            side["tiles"] = {"shape": shape, "n_tiles": len(keys), "w": 8, "h": 4, "n_blocks": 4 if shape == "path" else 0, "tile_shape": list(X.shape[1:])}
            side["n_rows"], side["n_features"] = 0, 0
        (d / "sidecar.json").write_text(json.dumps(side))
        out[name] = d
    return out, tk


def test_registry_names_what_it_lacks():
    reg = R.build_registry()
    ids = {m["id"]: m for m in reg["modules"]}
    assert ids["logreg"]["available"] and ids["minirocket"]["available"] and ids["ecod"]["available"]
    assert not ids["tabpfn"]["available"] and "tabpfn" in ids["tabpfn"]["why"]
    assert not ids["diffusion"]["available"] and "deferred" in ids["diffusion"]["why"]
    assert ids["lstm"]["accepts"] == ["path"] and ids["cnn2d"]["accepts"] == ["image", "path"]
    assert all(m["tier"] in {t["k"] for t in reg["tiers"]} for m in reg["modules"])
    assert all("torch" in m["needs"] for m in reg["modules"] if m["id"] in ("lstm", "vae", "deep_svdd", "cnn2d"))


def test_folds_never_straddle():
    with tempfile.TemporaryDirectory() as td:
        runs, tk = make_runs(Path(td))
        ds = D.load(runs["rows_run"])
        assert ds.shape == "rows" and ds.n == 3 * 2 * 2 * 12 and ds.names == ["m0", "m1", "m2", "m3", "m4", "m5"]
        assert sorted(set(ds.keys["campaign"].tolist())) == ["exp-1", "exp-2"]
        s = SP.summary(ds.keys, ["within_trace", "loro", "lowo", "loco"])
        assert s["loro"]["n_folds"] == 12 and s["lowo"]["n_folds"] == 6 and s["loco"]["n_folds"] == 2 and s["within_trace"]["n_folds"] == 1
        assert s["lowo"]["novelty"] == []                       # every family has two workloads
        wt = SP.fold_within_trace(ds.keys, 0.25)[0]
        t = ds.keys["t_index"]
        for r in np.unique(ds.keys["recording"]):
            m = ds.keys["recording"] == r
            assert t[np.intersect1d(np.flatnonzero(m), wt["test"])].min() > t[np.intersect1d(np.flatnonzero(m), wt["train"])].max()
        # a family with one workload is a novelty fold under lowo
        k2 = {k: v[np.isin(ds.keys["workload"], ["cpu_a", "io_a", "io_b"])] for k, v in ds.keys.items()}
        lw = SP.fold_lowo(k2)
        assert [f["held_out"] for f in lw if f["novelty"]] == ["cpu_a"]
        try:
            SP.folds_for(ds.keys, "nope")
            assert False
        except ValueError:
            pass


def test_data_reads_tiles_as_paths_and_images():
    with tempfile.TemporaryDirectory() as td:
        runs, tk = make_runs(Path(td))
        p = D.load(runs["path_run"], "tiles")
        assert p.shape == "path" and p.X.shape == (144, 8, 4) and p.meta["n_blocks"] == 4
        i = D.load(runs["image_run"], "tiles")
        assert i.shape == "image" and i.X.shape == (144, 8, 20) and i.meta["pages"] == list(range(20))
        av = D.available(runs["path_run"])
        assert av["tiles"] and not av["features"] and av["tiles_shape"] == "path" and av["n_tiles"] == 144
        try:
            D.load(runs["path_run"], "features")
            assert False
        except D.DataError as e:
            assert "Write" in str(e)
        # a series tile set reads as a one-block path
        d = Path(td) / "series_run"
        d.mkdir()
        np.savez_compressed(d / "tiles.npz", X=np.zeros((144, 8), np.float32), tile_keys=tk, shape=np.array("series"), w=np.array(8), h=np.array(4),
                            channels=np.array(["h"], dtype="U64"), n_blocks=np.array(0), complex=np.array(False))
        s = D.load(d, "tiles")
        assert s.shape == "path" and s.X.shape == (144, 8, 1)


def test_validator_refuses_by_name():
    with tempfile.TemporaryDirectory() as td:
        runs, _ = make_runs(Path(td))
        ctx = E.context_for(Path(td))
        assert ctx.runs["rows_run"]["n_workloads"] == 6 and ctx.runs["path_run"]["tiles_shape"] == "path"
        base = lambda slots, run=None: {"schema": PL.SCHEMA, "label": "t", "acknowledged": [], "slots": slots,
                                        "run": run or {"splits": ["loro"], "seeds": [0], "test_frac": 0.2, "target": "family"}}
        inp = lambda r, src: {"tier": "input", "alts": [{"module": "input", "params": {"run": r, "source": src}}]}
        tail = [{"tier": "score", "alts": [{"module": "score"}]}, {"tier": "output", "alts": [{"module": "write"}]}]
        ok = base([inp("rows_run", "features"), {"tier": "model", "alts": [{"module": "logreg"}]}] + tail)
        assert PL.verdict(PL.validate(ok, ctx), ok)[0] == 0
        # shape mismatch names both sides
        bad = base([inp("rows_run", "features"), {"tier": "model", "alts": [{"module": "lstm"}]}] + tail)
        iss = PL.validate(bad, ctx)
        assert any(i["sev"] == "hard" and "reads path" in i["msg"] and "rows reaches it" in i["msg"] for i in iss)
        # flatten fixes it
        good = base([inp("path_run", "tiles"), {"tier": "preprocess", "alts": [{"module": "flatten"}]}, {"tier": "model", "alts": [{"module": "logreg"}]}] + tail)
        assert PL.verdict(PL.validate(good, ctx), good)[0] == 0
        # unavailable and unbuilt modules, unknown module, missing tiers, bad run panel
        for slots, needle in (([inp("rows_run", "features"), {"tier": "model", "alts": [{"module": "tabpfn"}]}] + tail, "tabpfn"),
                              ([inp("rows_run", "features"), {"tier": "model", "alts": [{"module": "nope"}]}] + tail, "unknown module"),
                              ([inp("rows_run", "features")] + tail, "no model slot"),
                              ([inp("nope", "features"), {"tier": "model", "alts": [{"module": "logreg"}]}] + tail, "no run"),
                              ([inp("rows_run", "tiles"), {"tier": "model", "alts": [{"module": "logreg"}]}] + tail, "no tiles.npz")):
            iss = PL.validate(base(slots), ctx)
            assert any(i["sev"] == "hard" and needle in i["msg"] for i in iss), (needle, iss)
        iss = PL.validate(base(ok["slots"], {"splits": ["nope"], "seeds": [0]}), ctx)
        assert any("unknown split" in i["msg"] for i in iss)
        # soft: the ceiling alone, per-recording z, a sweep; every soft carries an id and acknowledgment clears it
        sw = base([inp("rows_run", "features"), {"tier": "preprocess", "alts": [{"module": "per_recording_z"}]},
                   {"tier": "model", "alts": [{"module": m} for m in ("logreg", "knn", "rf", "extratrees", "hgb", "svm_linear", "mlp")]}] + tail,
                  {"splits": ["within_trace"], "seeds": [0, 1], "test_frac": 0.2, "target": "family"})
        iss = PL.validate(sw, ctx)
        soft = {i["id"] for i in iss if i["sev"] == "soft"}
        assert "ceiling_only" in soft and "level_removed" in soft and any(s.startswith("sweep:14") for s in soft)
        code, v = PL.verdict(iss, sw)
        assert code == 2 and len(v["soft_unacknowledged"]) == 3
        sw["acknowledged"] = [{"id": s, "at": "t"} for s in soft]
        assert PL.verdict(PL.validate(sw, ctx), sw)[0] == 0
        assert len(PL.configurations(sw)) == 7 and PL.estimate(sw, ctx)["fits"] == 7 * 2 * 1
        # optional slots multiply the sweep by (alternatives + 1)
        opt = base([inp("rows_run", "features"), {"tier": "preprocess", "alts": [{"module": "scale"}, {"module": "log1p"}], "optional": True},
                    {"tier": "model", "alts": [{"module": "logreg"}, {"module": "knn"}]}] + tail)
        assert len(PL.configurations(opt)) == 6


def test_executor_end_to_end_three_shapes():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        runs, _ = make_runs(td)
        inp = lambda r, src: {"tier": "input", "alts": [{"module": "input", "params": {"run": r, "source": src}}]}
        tail = [{"tier": "score", "alts": [{"module": "score", "params": {"null_permutations": 30, "bootstrap": 30}}]}, {"tier": "output", "alts": [{"module": "write"}]}]
        pl = {"schema": PL.SCHEMA, "label": "e2e", "acknowledged": [],
              "slots": [inp("rows_run", "features"), {"tier": "preprocess", "alts": [{"module": "scale"}]},
                        {"tier": "model", "alts": [{"module": "logreg"}, {"module": "ae_bank", "params": {"min_train": 4, "max_iter": 50}}, {"module": "kmeans"}, {"module": "ecod"}]}] + tail,
              "run": {"splits": ["within_trace", "loro", "lowo", "loco"], "seeds": [0], "test_frac": 0.25, "target": "family"}}
        pp = td / "e2e.json"
        pp.write_text(json.dumps(pl))
        assert E.run(pp, td / "learn", td, acknowledge_all=True) == 0, (td / "learn" / "e2e" / "status.json").read_text()
        out = td / "learn" / "e2e"
        res = json.loads((out / "learn_results.json").read_text())
        assert [c["model"] for c in res["configurations"]] == ["logreg", "ae_bank", "kmeans", "ecod"]
        lr = res["configurations"][0]["splits"]
        # separable synthetic families: the linear floor is near perfect on every honest split, and the null is near chance
        for sp in ("loro", "lowo", "loco"):
            m = lr[sp]["aggregate"]["metrics"]
            assert m["accuracy"]["mean"] > 0.9, (sp, m["accuracy"])
            assert lr[sp]["aggregate"]["n_folds"] == {"loro": 12, "lowo": 6, "loco": 2}[sp]
            pooled = lr[sp]["aggregate"]["pooled"]
            assert pooled["n"] == 144 and pooled["accuracy"] > 0.9 and pooled["null"]["n"] == 30 and pooled["null"]["mean"] < 0.6, (sp, pooled)
            assert pooled["bootstrap"]["groups"] == 12
        # a one-class fold (loro, lowo) has no null of its own and says so; a two-campaign fold does
        assert "null_mean" not in lr["loro"]["aggregate"]["metrics"] and "one class" in lr["loro"]["folds"][0]["null"]["why"]
        assert lr["loco"]["aggregate"]["metrics"]["null_mean"]["mean"] < 0.6
        f = lr["loro"]["folds"][0]
        assert f["n_train"] + f["n_test"] == 144 and f["classification"]["confusion"]["labels"] == ["cpu", "io", "mem"]
        assert f["bootstrap"]["n"] == 0                               # one recording in a loro test fold: nothing to resample
        assert lr["loco"]["folds"][0]["bootstrap"]["groups"] == 6     # a campaign holds six recordings
        assert f["importance"]["names"] == ["m0", "m1", "m2", "m3", "m4", "m5"]
        assert f["train_accuracy"] is not None and f["gap"] is not None
        # the bank names its members and reconstructs; kmeans has ARI and a null on it; ecod has scores per class
        bank = res["configurations"][1]["splits"]["loro"]["folds"][0]
        assert bank["model"]["members"] == ["cpu", "io", "mem"] and "recon_error" in bank and bank["classification"]["accuracy"] > 0.8
        km = res["configurations"][2]["splits"]["lowo"]["folds"][0]
        assert km["model"]["k"] == 3 and km["clustering"]["k_present"] >= 1 and km["null"]["metric"] == "ari_family"
        assert res["configurations"][2]["splits"]["lowo"]["aggregate"]["pooled"]["ari_family"] > 0.3
        ec = res["configurations"][3]["splits"]["lowo"]["folds"][0]
        assert set(ec["novelty_by_class"]) <= {"cpu", "io", "mem"} and ec["novelty_detection"]["auroc"] is None    # nothing novel under lowo here
        # predictions and embeddings landed with keys; the sidecar names inputs, folds, versions, the sweep
        K, Pr = LR.load_predictions(out)
        assert len(K) == len(Pr) and set(K["config"].tolist()) == {"c001", "c002", "c003", "c004"}
        side = json.loads((out / "sidecar.json").read_text())
        assert side["inputs"]["rows_run"]["features_sha256_16"] and side["inputs"]["rows_run"]["folds"]["lowo"][0]["n_test"] == 24
        assert side["sweep"]["configurations"] == 4 and side["numpy"] and side["fits_done"] == 4 * (1 + 12 + 6 + 2)
        # results views over that run
        s = LR.summary(out)
        assert s["configurations"][0]["splits"]["loro"]["metrics"]["accuracy"]["n"] == 12
        fr = LR.scores_frame(out)
        assert fr.X.shape[0] == 4 * 21 and "accuracy" in fr.names
        d = LR.agg(out, "scores", "distribution", y="accuracy", group=["model", "split"])
        assert any(g["label"] == "logreg / lowo" and g["n"] == 6 for g in d["groups"])
        t = LR.agg(out, "tiles", "time", y="correct", group=["model"], x="t_index", stat="mean")
        assert any(s_["label"] == "logreg" and len(s_["x"]) == 12 for s_ in t["series"])
        cm = LR.confusion(out, "c001", "lowo")
        assert cm["n"] == 144 and cm["labels"] == ["cpu", "io", "mem"]
        emb = LR.embedding(out, "c002", "loro")
        assert emb["n"] == 144 and len(emb["points"]) == 144 and emb["dim"] == 3
        assert LR.importance(out, "c001", "loro")["names"] == ["m0", "m1", "m2", "m3", "m4", "m5"]
        nl = LR.null(out, "c001", "loro")
        assert nl["folds"] == [] and nl["pooled"]["n"] == 30 and nl["pooled"]["observed"] > 0.9
        assert LR.calibration(out, "c001", "loro")["ece_mean"] is not None
        for bad in (lambda: LR.confusion(out, "c003", "loro"), lambda: LR.saliency(out, "c001", "loro"), lambda: LR.train_curves(out, "c001", "loro")):
            try:
                bad()
                assert False
            except LR.ResultsError:
                pass
        # paths and images, with the torch models for two epochs, the numpy MiniRocket, and a stop through control.json
        pl2 = {"schema": PL.SCHEMA, "label": "shapes", "acknowledged": [],
               "slots": [{"tier": "input", "alts": [{"module": "input", "params": {"run": "path_run", "source": "tiles"}},
                                                    {"module": "input", "params": {"run": "image_run", "source": "tiles"}}]},
                         {"tier": "preprocess", "alts": [{"module": "scale"}]},
                         {"tier": "model", "alts": [{"module": "cnn2d", "params": {"epochs": 2}}, {"module": "conv_ae_bank", "params": {"epochs": 2, "min_train": 4}}]}] + tail,
               "run": {"splits": ["loco"], "seeds": [0], "test_frac": 0.2, "target": "family"}}
        pp2 = td / "shapes.json"
        pp2.write_text(json.dumps(pl2))
        assert E.run(pp2, td / "learn", td, acknowledge_all=True) == 0, (td / "learn" / "shapes" / "status.json").read_text()
        res2 = json.loads((td / "learn" / "shapes" / "learn_results.json").read_text())
        assert len(res2["configurations"]) == 4
        c = res2["configurations"][0]
        assert c["input"]["shape"] == "path" and c["splits"]["loco"]["folds"][0]["saliency"]["shape"] == [8, 4]
        assert c["splits"]["loco"]["folds"][0]["train_curve"] and len(c["splits"]["loco"]["folds"][0]["train_curve"]) == 2
        sal = LR.saliency(td / "learn" / "shapes", "c001", "loco")
        assert np.asarray(sal["mean"]).shape == (8, 4)
        assert LR.train_curves(td / "learn" / "shapes", "c001", "loco")["series"][0]["x"] == [1, 2]
        tile = LR.tile(td, "path_run", t_index=3)
        assert tile["shape"] == "path" and np.asarray(tile["matrix"]).shape == (8, 4) and len(tile["most_active_block"]) == 8
        assert LR.tiles_index(td, "image_run")["shape"] == "image"
        pl3 = {"schema": PL.SCHEMA, "label": "rocket", "acknowledged": [],
               "slots": [{"tier": "input", "alts": [{"module": "input", "params": {"run": "path_run", "source": "tiles"}}]},
                         {"tier": "model", "alts": [{"module": "minirocket", "params": {"n_kernels": 168}}, {"module": "lstm", "params": {"epochs": 2}}]}] + tail,
               "run": {"splits": ["lowo"], "seeds": [0], "test_frac": 0.2, "target": "family"}}
        pp3 = td / "rocket.json"
        pp3.write_text(json.dumps(pl3))
        assert E.run(pp3, td / "learn", td, acknowledge_all=True) == 0
        res3 = json.loads((td / "learn" / "rocket" / "learn_results.json").read_text())
        assert res3["configurations"][0]["splits"]["lowo"]["aggregate"]["metrics"]["accuracy"]["mean"] > 0.8
        assert res3["configurations"][1]["splits"]["lowo"]["folds"][0]["importance"]["names"] == ["block0", "block1", "block2", "block3"]
        # a stop before the first fit leaves a stopped status, not a failed one
        (td / "learn" / "stopme").mkdir(parents=True)
        (td / "learn" / "stopme" / "control.json").write_text(json.dumps({"command": "stop"}))
        pl4 = dict(pl3, label="stopme")
        pp4 = td / "stopme.json"
        pp4.write_text(json.dumps(pl4))
        assert E.run(pp4, td / "learn", td, acknowledge_all=True) == 130
        assert json.loads((td / "learn" / "stopme" / "status.json").read_text())["state"] == "stopped"
        # refusals leave a status that says why
        pl5 = dict(pl3, label="refused", slots=[pl3["slots"][0], {"tier": "model", "alts": [{"module": "logreg"}]}] + tail)
        pp5 = td / "refused.json"
        pp5.write_text(json.dumps(pl5))
        assert E.run(pp5, td / "learn", td, acknowledge_all=True) == 1
        st = json.loads((td / "learn" / "refused" / "status.json").read_text())
        assert st["state"] == "refused" and "reads rows" in st["verdict"]["hard"][0]["msg"]


def test_scores_and_models_agree_with_numpy():
    rng = np.random.default_rng(1)
    y = np.array(["a"] * 20 + ["b"] * 20)
    yp = y.copy()
    yp[:5] = "b"
    c = SC.classification(y, yp)
    assert c["accuracy"] == 35 / 40 and c["confusion"]["matrix"] == [[15, 5], [0, 20]] and c["per_class"]["a"]["recall"] == 0.75
    n = SC.null_permutation(y, yp, 100, 0)
    assert 0.4 < n["mean"] < 0.6 and n["percentile_of_observed"] > 90
    b = SC.bootstrap(y, yp, np.repeat(np.arange(8), 5), 100, 0)
    assert b["groups"] == 8 and b["ci95"][0] <= b["mean"] <= b["ci95"][1]
    d = SC.detection(y == "a", np.r_[rng.normal(2, 1, 20), rng.normal(0, 1, 20)])
    assert d["auroc"] > 0.85 and 0 <= d["recall_at_fpr"] <= 1 and d["curve"][0] == {"fpr": 0.0, "tpr": 0.0}
    assert SC.detection(np.ones(10, bool), rng.normal(size=10))["auroc"] is None
    e = M.make("ecod", {}, 0).fit(rng.normal(size=(100, 3)))
    far = e.scores(np.array([[0, 0, 0], [8, 8, 8]], dtype=np.float32))["novelty"]
    assert far[1] > far[0]
    # a preprocess never sees test rows: the scaler's mean is the training mean
    Xtr, Xte = rng.normal(5, 1, size=(50, 2)).astype(np.float32), rng.normal(-5, 1, size=(50, 2)).astype(np.float32)
    t = P.make("scale", {"method": "standard"}, 0).fit(Xtr)
    assert abs(t.transform(Xtr).mean()) < 0.05 and t.transform(Xte).mean() < -5


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
