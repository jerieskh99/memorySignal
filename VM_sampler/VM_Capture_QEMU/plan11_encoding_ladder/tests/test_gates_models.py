"""models.py: the forest, B1-G1 / B1-G3 / B1-G6 at the unit, G-DIM's reduction, clustering, the
failed-count exclusion and the no-windows row (SPEC 4.2-4.5; CR 2.1 items 9-11; al-Farabi 2.4, 2.9 (c))."""
import numpy as np
import pytest

from _b2_common import S, V, SY, corpus, tmp_out, rows, jload
from plan11_encoding_ladder import models as M
from plan11_encoding_ladder import gates_precondition as GP
from plan11_encoding_ladder import gates_temporal as GT

N_EST = 10        # smoke-size forest; the SPEC default 300 is the module constant (params records the value used)


def _feats(out, rung, W=16, H=16):
    S.build_features(out, None, rung, W, H, True)
    S.build_features(out, None, rung, W, H, False) if rung != "combined" else None
    return S.schema.grid_id(W, H)


def test_make_forest_is_the_declared_pipeline():
    f = M.make_forest()
    assert [n for n, _ in f.steps] == ["imp", "sc", "rf"]
    rf = f.named_steps["rf"]
    assert rf.n_estimators == 300 and rf.max_features == "sqrt" and rf.random_state == M.SEED_FOREST and rf.bootstrap
    assert f.named_steps["imp"].strategy == "median"
    assert M.make_l1(4).named_steps["tree"].max_leaf_nodes == 4


def test_b1_g6_majority_is_half_under_loko_on_twelve_kernels():
    out = corpus(reps=1, idle=0, n_pairs=60)
    gid = _feats(out, "apf")
    d = M.run_split_stage(out, "apf", gid, "loko", "archetype", n_perm=0, run_null=False, n_estimators=N_EST, quarantine=False)
    sc = jload(d / "scores.json")
    assert sc["majority"] == 0.5 and sc["n_folds"] == 12 and sc["headline_classes"] == ["SCATTER", "WORKING-SET"]
    assert sc["b1_g1"].startswith("not run") and sc["feature_count"] == 8 and sc["dim_status"] == V.GDIM_FULL
    pr = rows(d / "predictions.csv")
    assert len(pr) == 12 and all(r["y_true"] in S.schema.ARCHETYPES for r in pr)


def test_b1_g1_passes_on_an_archetype_consistent_corpus_and_refuses_on_one_preset():
    # must pass: every kernel of an archetype shares a preset (SPEC 5.3 'distinct presets'), LOKO/archetype, 500 unit-level shuffles
    out = corpus(reps=2, idle=0, n_pairs=60, archetype_consistent=True)
    gid = _feats(out, "apf")
    d = M.run_split_stage(out, "apf", gid, "loko", "archetype", n_perm=500, n_estimators=N_EST, n_jobs=4, quarantine=False)
    sc = jload(d / "scores.json")
    assert sc["n_perm"] == 500 and sc["b1_g1"] == V.PASS, (sc["accuracy"], sc["null_p95"])
    assert sc["b1_g1_rank_text"].startswith("rank ") and jload(d / "null.json")["summary"]["n"] == 500
    # must refuse: one preset for every kernel, only the seed differs
    out2 = corpus(reps=2, idle=0, n_pairs=60, one_preset=True)
    gid = _feats(out2, "apf")
    d2 = M.run_split_stage(out2, "apf", gid, "loko", "archetype", n_perm=500, n_estimators=N_EST, n_jobs=4, quarantine=False)
    sc2 = jload(d2 / "scores.json")
    assert sc2["b1_g1"] == V.NEAR_UNFALSIFIABLE, (sc2["accuracy"], sc2["null_p95"])
    ex = rows(out2 / "gates" / "excluded_rows.csv")
    assert ex and ex[0]["verdict"] == V.NEAR_UNFALSIFIABLE and ex[0]["rung"] == "apf"


def test_b1_g3_quarantines_the_level_feature_on_raw_apf_of_a_level_only_corpus():
    # must refuse (a feature is the label): level-only corpus, raw APF, kernel space within-trace
    out = corpus(reps=2, idle=0, n_pairs=60, level_only=True)
    gid = _feats(out, "apf")
    d = M.run_split_stage(out, "apf", gid, "loko", "archetype", normalized=False, n_perm=0, run_null=False, n_estimators=N_EST)
    q = jload(d / "l1_quarantine.json")["quarantined"]
    assert any(x["feature"] == "apf.k_over_n.mean" for x in q), q
    sc = jload(d / "scores.json")
    assert sc["with_quarantine"]["quarantined_features"]
    # CHECK_2.md B1: the re-run's predictions are written beside the full model's, and scores.json
    # names which score is the rung's score
    assert (d / "predictions_with_quarantine.csv").is_file() and sc["predictions_file"] == "predictions_with_quarantine.csv"
    assert sc["score_source"].startswith("with_quarantine") and len(rows(d / "predictions_with_quarantine.csv")) == len(rows(d / "predictions.csv"))
    assert M.effective_scores(sc)["accuracy"] == sc["with_quarantine"]["accuracy"]
    # must pass (empty quarantine): the normalized features of the standard corpus
    out2 = corpus(reps=2, idle=0, n_pairs=60)
    gid = _feats(out2, "apf")
    d2 = M.run_split_stage(out2, "apf", gid, "loko", "archetype", n_perm=0, run_null=False, n_estimators=N_EST)
    assert jload(d2 / "l1_quarantine.json")["quarantined"] == [] and jload(d2 / "scores.json")["with_quarantine"] is None
    assert not (d2 / "predictions_with_quarantine.csv").exists() and jload(d2 / "scores.json")["predictions_file"] == "predictions.csv"


def test_raw_and_norm_write_two_directories_and_keep_both(tmp_path):
    # CHECK_1.md B1 / al-Farabi certification 7.3: the raw (level-inclusive) APF split lands in
    # <split>__<labelspace>__raw/ beside the normalized one and neither overwrites the other
    out = corpus(reps=2, idle=0, n_pairs=60)
    gid = _feats(out, "apf")
    rc = M.main(["splits", "--out", str(out), "--rung", "apf", "--grid-id", gid, "--split", "loko",
                 "--raw-and-norm", "--null-perm", "0", "--null-splits", "none", "--n-estimators", str(N_EST)])
    assert rc == 0
    d_norm = M.split_dir(out, "apf", gid, "loko", "archetype")
    d_raw = M.split_dir(out, "apf", gid, "loko", "archetype", normalized=False)
    assert d_norm.name == "loko__archetype" and d_raw.name == "loko__archetype__raw"
    assert (d_norm / "scores.json").is_file() and (d_raw / "scores.json").is_file()
    assert jload(d_norm / "scores.json")["params"]["normalized"] is True
    assert jload(d_raw / "scores.json")["params"]["normalized"] is False
    for f in ("predictions.csv", "null.json", "l1_quarantine.json"):
        assert (d_norm / f).is_file() and (d_raw / f).is_file()
    # builder 3's reader finds the raw run at its first candidate path
    from plan11_encoding_ladder import _report_common as RC
    assert RC.split_dir(out, "apf", gid, "loko", "archetype", raw=True) == d_raw
    assert RC.split_dir(out, "apf", gid, "loko", "archetype", raw=False) == d_norm


def test_gdim_reduction_when_dimension_exceeds_training_cells():
    out = corpus(reps=1, idle=0, n_pairs=60)            # 12 cells: n_train = 11 < 60
    gid = _feats(out, "combined")
    d = M.run_split_stage(out, "combined", gid, "loko", "archetype", n_perm=0, run_null=False, n_estimators=N_EST, quarantine=False)
    sc = jload(d / "scores.json")
    assert sc["dim_status"] == V.GDIM_REDUCED and sc["feature_count"] == 60 and sc["min_train_cells"] == 11
    out2 = corpus(reps=8, idle=0, n_pairs=60)           # 96 cells: n_train = 88 > 60
    gid = _feats(out2, "combined")
    d2 = M.run_split_stage(out2, "combined", gid, "loko", "archetype", n_perm=0, run_null=False, n_estimators=N_EST, quarantine=False)
    assert jload(d2 / "scores.json")["dim_status"] == V.GDIM_FULL
    d3 = M.run_split_stage(out2, "combined", gid, "loko", "archetype", n_perm=0, run_null=False, n_estimators=N_EST, quarantine=False,
                           reduce_to=8, reduce_method="pca", base_dir="splits_matched")
    assert jload(d3 / "scores.json")["dim_status"] == V.GDIM_REDUCED


def test_not_applicable_forms():
    out = corpus(reps=1, idle=0, kernels=["gemm", "gibbs"], n_pairs=60)
    gid = _feats(out, "apf")
    d = M.run_split_stage(out, "apf", gid, "loko", "kernel", n_perm=0, run_null=False, n_estimators=N_EST)
    assert jload(d / "scores.json")["b1_g1"] == V.not_applicable("held-out label unseen")
    S.build_features(out, None, "apf", None, None, True)
    d = M.run_split_stage(out, "apf", "Wall_Hall", "within_trace", "kernel", n_perm=0, run_null=False, n_estimators=N_EST)
    assert jload(d / "scores.json")["b1_g1"] == V.not_applicable("one window per cell")
    d = M.run_split_stage(out, "apf", "Wall_Hall", "loro", "kernel", n_perm=0, run_null=False, n_estimators=N_EST, quarantine=False)
    assert jload(d / "scores.json")["accuracy"] is not None


def test_failed_count_excludes_the_cell_from_the_persist_split_only():
    out = corpus(reps=2, idle=0, kernels=["gemm", "gibbs", "fft"], n_pairs=60, overrides={"gibbs": dict(K0=6000)})
    SY.write_corpus(out, SY.corpus_specs(reps=2, idle=0, kernels=["gemm", "gibbs", "fft"], n_pairs=60,
                                         overrides={"gibbs": dict(K0=6000), "fft": dict(failed_count=2)}))
    GP.gate_preconditions(out, c1_activity_min=0.0)
    for rung in ("persist", "apf"):
        gid = _feats(out, rung)
        d = M.run_split_stage(out, rung, gid, "loro", "kernel", n_perm=0, run_null=False, n_estimators=N_EST, quarantine=False)
        ids = {r["cell_id"] for r in rows(d / "predictions.csv")}
        params = jload(d / "scores.json")["params"]
        if rung == "persist":
            assert not any(c.startswith("fft") for c in ids) and params["excluded_cells_pair_rungs"] == ["fft__rep00__01c", "fft__rep01__01c"]
        else:
            assert any(c.startswith("fft") for c in ids) and params["excluded_cells_pair_rungs"] == []


def test_cell_with_no_windows_is_written_not_dropped():
    out = corpus(reps=1, idle=0, kernels=["gemm", "gibbs", "fft"], n_pairs=60, overrides={"fft": dict(n_pairs=10)})
    gid = _feats(out, "apf")                             # W = 16 > 9 series rows for fft
    d = M.run_split_stage(out, "apf", gid, "loro", "kernel", n_perm=0, run_null=False, n_estimators=N_EST, quarantine=False)
    pr = {r["cell_id"]: r for r in rows(d / "predictions.csv")}
    assert pr["fft__rep00__01c"]["y_pred"] == V.not_run("no windows") and jload(d / "scores.json")["n_cells_no_windows"] == 1


def test_clustering_exceeds_null_on_archetype_consistent_and_not_on_one_preset():
    out = corpus(reps=3, idle=0, n_pairs=60, archetype_consistent=True)
    GT.gate_grid(out, "apf", n_surrogates=2); GT.select(out, "apf")
    M.run_clustering(out, "apf", n_perm=100)
    r = {x["algo"]: x for x in rows(out / "gates" / "clustering.csv")}
    assert r["kmeans"]["k"] == "4" and r["kmeans"]["primary"] == "true" and r["kmeans"]["exceeds_ari"] == "true", r["kmeans"]
    out2 = corpus(reps=3, idle=0, n_pairs=60, one_preset=True)
    GT.gate_grid(out2, "apf", n_surrogates=2); GT.select(out2, "apf")
    M.run_clustering(out2, "apf", n_perm=100)
    r2 = {x["algo"]: x for x in rows(out2 / "gates" / "clustering.csv")}
    assert r2["kmeans"]["exceeds_ari"] == "false"
    out3 = tmp_out(); SY.write_corpus(out3, SY.corpus_specs(reps=1, idle=0, kernels=["gemm"], n_pairs=60))
    M.run_clustering(out3, "apf", n_perm=5)
    assert rows(out3 / "gates" / "clustering.csv")[0]["status"] == V.not_run("no selection for apf")


def test_aggregate_units_tie_break():
    classes = ["A", "B"]
    cells = np.array(["c", "c"]); pred = np.array(["A", "B"]); proba = np.array([[0.6, 0.4], [0.3, 0.7]])
    assert M.aggregate_units(cells, pred, proba, classes)["c"] == ("B", 0.5)    # higher mean proba wins the tie
    proba = np.array([[0.5, 0.5], [0.5, 0.5]])
    assert M.aggregate_units(cells, pred, proba, classes)["c"][0] == "A"        # then class name order


def test_epoch2_b5_feature_count_used_after_a_declared_reduction():
    """SPEC_epoch2 B5 (CHECK_3 M5; E1 6.49): a run with reduce_to=3 on the 60-feature combined file
    writes feature_count == 60 (the vector) and feature_count_used == 3 (what every fold trained on);
    effective_scores carries the key through."""
    out = corpus(reps=1, idle=0, n_pairs=60)
    gid = _feats(out, "combined")
    d = M.run_split_stage(out, "combined", gid, "loko", "archetype", n_perm=0, run_null=False, n_estimators=N_EST,
                          quarantine=False, reduce_to=3, base_dir="splits_matched")
    sc = jload(d / "scores.json")
    assert sc["feature_count"] == 60 and sc["feature_count_used"] == 3
    assert M.effective_scores(sc)["feature_count_used"] == 3
    d2 = M.run_split_stage(out, "combined", gid, "loko", "archetype", n_perm=0, run_null=False, n_estimators=N_EST, quarantine=False)
    sc2 = jload(d2 / "scores.json")
    assert sc2["feature_count"] == 60 and sc2["feature_count_used"] == 11          # the training-cell cap of a 12-cell corpus
