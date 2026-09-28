"""gates_comparison.py: G-L, G-N, G-X, G-DIM, G-M and the B1 helpers (SPEC 3.7, 5.3)."""
import json

from _b2_common import S, V, SY, corpus, tmp_out, rows, row_where, jload
from plan11_encoding_ladder import gates_comparison as GX
from plan11_encoding_ladder import gates_temporal as GT
from plan11_encoding_ladder import models as M

N_EST = 10


def _prep(out, rung, **split_kw):
    GT.gate_grid(out, rung, n_surrogates=2); GT.select(out, rung)
    gid = S.selected_grid_id(out, rung)[0]
    M.run_split_stage(out, rung, gid, "loko", "archetype", n_estimators=N_EST, quarantine=False, **split_kw)
    return gid


def _fake_scores(out, rung, gid, acc, p95, n_perm=500, with_quarantine=None, split="loko", ls="archetype"):
    d = M.split_dir(out, rung, gid, split, ls); d.mkdir(parents=True, exist_ok=True)
    verdict = V.PASS if acc > p95 else V.NEAR_UNFALSIFIABLE
    (d / "scores.json").write_text(json.dumps({"schema": "plan11.scores.v1", "params": {}, "citation": "test", "accuracy": acc,
                                               "null_p95": p95, "n_perm": n_perm, "b1_g1": verdict, "feature_count": 8, "dim_status": V.GDIM_FULL,
                                               "with_quarantine": with_quarantine, "recall_per_kernel": {"gemm": 1.0, "fft": 0.0}}))
    return d


def test_gl_gdim_gm_judge_the_rerun_after_a_b1_g3_quarantine_and_the_tables_print_the_same_number():
    # CHECK_2.md B1 (SPEC 3.7.2 'Table rows use the re-run'; CR 2.1 item 10): once B1-G3 has quarantined a
    # feature the rung's score is the re-run's, for the comparison gates and for the tables alike. The
    # level-only reading: the full model (8 features, the level feature among them) exceeds the null,
    # the re-run without it does not.
    from plan11_encoding_ladder import _report_common as RC
    from plan11_encoding_ladder import tables as T
    out = corpus(reps=2, idle=0, n_pairs=60, level_only=True)
    gid = _prep(out, "apf", n_perm=0, run_null=False)
    wq = {"accuracy": 0.5, "null_p95": 0.6, "b1_g1": V.NEAR_UNFALSIFIABLE, "b1_g1_rank": 300, "macro_recall": 0.5,
          "recall_per_kernel": {"gemm": 0.5, "fft": 0.5}, "majority": 0.5, "feature_count": 6,
          "quarantined_features": ["apf.k_over_med.mean", "apf.k_over_med.peak2med"]}
    _fake_scores(out, "apf", gid, 0.8, 0.6, with_quarantine=wq)
    sc = M.effective_scores(jload(M.split_dir(out, "apf", gid, "loko", "archetype") / "scores.json"))
    assert sc["accuracy"] == 0.5 and sc["b1_g1"] == V.NEAR_UNFALSIFIABLE and sc["feature_count"] == 6 and sc["quarantined_features"] == wq["quarantined_features"]
    assert RC.effective_scores(jload(M.split_dir(out, "apf", gid, "loko", "archetype") / "scores.json")) == sc   # builder 3 reads through the same function
    # builder 3's fallback copy (a process without models.py) must give the same reading on every shape
    import sys
    from unittest import mock
    docs = [{"accuracy": 0.8, "b1_g1": V.PASS, "with_quarantine": None}, {"accuracy": 0.8, "b1_g1": V.PASS, "with_quarantine": wq},
            {"accuracy": 0.8, "b1_g1": V.PASS, "with_quarantine": {"status": V.not_run("every feature quarantined"), "quarantined_features": ["a"]}}, None]
    with mock.patch.dict(sys.modules, {"plan11_encoding_ladder.models": None}):
        assert [RC.effective_scores(d) for d in docs] == [M.effective_scores(d) for d in docs]
    GX.gate_gl(out)
    r = row_where(out / "gates" / "gl.csv", rung="apf", part="i")
    assert r["verdict"] == V.GL_LEVEL_ONLY and float(r["score_norm"]) == 0.5 and float(r["null_p95"]) == 0.6
    assert "with_quarantine" in jload(out / "gates" / "gl.params.json")["params"]["score_source"]
    # G-DIM's d* and G-M's per-kernel recalls come from the re-run too
    GX.gate_gdim(out, n_perm=0, n_estimators=N_EST)
    assert row_where(out / "gates" / "gdim.csv", rung="apf")["d"] == "6"
    assert "with_quarantine" in jload(out / "gates" / "gdim.params.json")["params"]["score_source"]
    # the table prints the number the gate judged: Table 7's APF LOKO row shows the re-run's accuracy
    # as a near_unfalsifiable block (its score is never printed) and the same feature count
    T.table7(out)
    t7 = row_where(out / "report" / "tables" / "table7.csv", rung="apf", split="LOKO")
    assert t7["accuracy"] == V.NEAR_UNFALSIFIABLE and t7["feature count"] == "6" and V.GL_LEVEL_ONLY in t7["G-L"]
    # the other reading, no quarantine: the full model's number, in both places
    _fake_scores(out, "apf", gid, 0.8, 0.6, with_quarantine=None)
    GX.gate_gl(out)
    assert float(row_where(out / "gates" / "gl.csv", rung="apf", part="i")["score_norm"]) == 0.8
    T.table7(out)
    assert row_where(out / "report" / "tables" / "table7.csv", rung="apf", split="LOKO")["accuracy"] == "0.800"
    # every feature quarantined: no re-run exists, so no consumer judges the full model
    _fake_scores(out, "apf", gid, 0.8, 0.6, with_quarantine={"status": V.not_run("every feature quarantined"), "quarantined_features": ["a"]})
    GX.gate_gl(out)
    assert row_where(out / "gates" / "gl.csv", rung="apf", part="i")["verdict"] == V.not_run("every feature quarantined")


def test_gl_part1_pass_and_level_only_and_part2_random_vs_shot():
    out = corpus(reps=2, idle=0, n_pairs=60, cv_case="random")
    gid = _prep(out, "apf", n_perm=0, run_null=False)
    _fake_scores(out, "apf", gid, 0.8, 0.6)
    GX.gate_gl(out)
    p = out / "gates" / "gl.csv"
    assert row_where(p, rung="apf", part="i")["verdict"] == V.PASS
    assert row_where(p, rung="apf", part="ii")["verdict"] == V.PASS and float(row_where(p, rung="apf", part="ii")["r2"]) <= 0.5
    assert row_where(p, rung="wapf", part="i")["verdict"] == V.not_run("no selection for wapf")
    _fake_scores(out, "apf", gid, 0.5, 0.6)
    GX.gate_gl(out)
    assert row_where(p, rung="apf", part="i")["verdict"] == V.GL_LEVEL_ONLY
    _fake_scores(out, "apf", gid, 0.8, 0.6, n_perm=20)
    GX.gate_gl(out)
    assert row_where(p, rung="apf", part="i")["verdict"] == V.PASS         # the rule reads the written b1_g1 outcome
    # part (ii) must refuse when CV follows 1 / sqrt(K): every kernel k_noise = 2 / sqrt(K0), no pulse or trend
    out2 = corpus(reps=2, idle=0, n_pairs=60, cv_case="shot",
                  overrides={k: dict(pulse_period=None, pulse_extra=0, trend=0.0, K0=max(SY.PRESETS[k].get("K0", 2048), 256)) for k in SY.PRESETS})
    _prep(out2, "apf", n_perm=0, run_null=False)
    GX.gate_gl(out2)
    r = row_where(out2 / "gates" / "gl.csv", rung="apf", part="ii")
    assert r["verdict"] == V.GL_SHOT_NOISE and float(r["r2"]) > 0.5
    assert GX.gl_part2_regression({"a": 1.0, "b": 2.0, "c": 3.0}, {"a": 1.0, "b": 2.0, "c": 3.0})["verdict"] == V.GL_SHOT_NOISE


def test_gn_statuses():
    assert GX.gn_status(6) == V.GN_HEADLINE and GX.gn_status(3) == V.GN_HEADLINE and GX.gn_status(2) == V.GN_ONE_TRAIN
    assert GX.gn_status(1) == V.GN_NOVELTY and GX.gn_status(0) == V.GN_NO_ROW
    out = corpus(reps=1, idle=0, n_pairs=60)
    GX.gate_gn(out)
    p = out / "gates" / "gn.csv"
    assert [row_where(p, archetype=a)["status"] for a in ("WORKING-SET", "SCATTER", "SEQUENTIAL-GROW", "FRONTIER-CHURN", "IDLE")] == \
        [V.GN_HEADLINE, V.GN_HEADLINE, V.GN_ONE_TRAIN, V.GN_NOVELTY, V.GN_NO_ROW]
    S.write_csv(out / "gates" / "gk0.csv", ("kernel", "verdict", "archetype_measured"), [{"kernel": "lexer", "verdict": V.GK0_IDLE_MEASURED, "archetype_measured": "IDLE"}])
    GX.gate_gn(out)
    assert row_where(p, archetype="IDLE")["status"] == V.GN_NOVELTY and row_where(p, archetype="SEQUENTIAL-GROW")["status"] == V.GN_NOVELTY


def test_gx_pooling_stands_vs_leak_with_total_confound():
    out = corpus(reps=3, idle=0, n_pairs=60, campaign_of=lambda k, r: ["01c", "01c1", "dwarfs1"][r % 3])
    _prep(out, "apf", n_perm=0, run_null=False)
    GX.gate_gx(out, "apf", n_perm=40, n_estimators=N_EST)
    r = row_where(out / "gates" / "gx.csv", rung="apf")
    assert r["leak_verdict"] == V.GX_POOLING_STANDS and r["confound_verdict"] == V.GX_CONFOUND_NONE and r["n_labels"] == "3"
    # WORKING-SET all in 01c, SCATTER all in 01c1, a per-campaign jitter offset visible after normalization
    camp = lambda k, r: "01c" if SY.KMAP[k] == "WORKING-SET" else ("01c1" if SY.KMAP[k] == "SCATTER" else "dwarfs1")
    out2 = corpus(reps=3, idle=0, n_pairs=60, campaign_of=camp, campaign_overrides={"01c": dict(k_noise=0.02), "01c1": dict(k_noise=0.30), "dwarfs1": dict(k_noise=0.15)})
    _prep(out2, "apf", n_perm=0, run_null=False)
    GX.gate_gx(out2, "apf", n_perm=40, n_estimators=N_EST)
    r2 = row_where(out2 / "gates" / "gx.csv", rung="apf")
    assert r2["leak_verdict"] == V.GX_LEAK and r2["confound_verdict"] == V.GX_CONFOUND_TOTAL and r2["headline_mark"] == V.refused("campaign leak with total confound")
    assert jload(out2 / "gates" / "gx.json")["archetype_campaign_sets"]["WORKING-SET"] == ["01c"]
    assert GX.gx_confound({"a": {"x"}, "b": {"x"}, "c": {"x", "y"}, "d": {"y"}}, {"a": "A", "b": "A", "c": "B", "d": "B"})[0] == V.GX_CONFOUND_PARTIAL
    assert GX.gx_confound({"a": {"x"}, "b": {"y"}}, {"a": "A", "b": "B"})[0] == V.GX_CONFOUND_NONE


def test_gdim_full_vs_declared_reduction_and_matched_row():
    out = corpus(reps=8, idle=0, n_pairs=60)                       # 96 cells
    for rung in ("apf", "combined"):
        _prep(out, rung, n_perm=0, run_null=False)
    GX.gate_gdim(out, n_perm=0, n_estimators=N_EST)
    p = out / "gates" / "gdim.csv"
    assert row_where(p, rung="combined")["status"] == V.GDIM_FULL and row_where(p, rung="combined")["d"] == "60"
    m = row_where(p, rung="combined (matched)")
    assert m["status"] == V.GDIM_REDUCED and m["d_matched"] == "8" and m["method"] == "train_importance" and m["matched_to"] == "apf"
    assert (M.split_dir(out, "combined", S.selected_grid_id(out, "combined")[0], "loko", "archetype", "splits_matched") / "scores.json").is_file()
    out2 = corpus(reps=1, idle=0, n_pairs=60)                      # 12 cells: d = 60 > 11 training cells
    _prep(out2, "combined", n_perm=0, run_null=False)
    GX.gate_gdim(out2, n_perm=0, n_estimators=N_EST)
    assert row_where(out2 / "gates" / "gdim.csv", rung="combined")["status"] == V.GDIM_REDUCED


def test_gm_beats_vs_difference_with_margin():
    ks = [f"k{i}" for i in range(12)]
    perfect = {"accuracy": 1.0, "recall_per_kernel": {k: 1.0 for k in ks}}
    chance = {"accuracy": 0.25, "recall_per_kernel": {k: (1.0 if i < 3 else 0.0) for i, k in enumerate(ks)}}
    r = GX.gm_compare(perfect, chance, spread=0.05)
    assert r["verdict"] == V.GM_BEATS and r["improving"] == 9 and r["worsening"] == 0 and r["ties"] == 3
    same = {"accuracy": 0.9, "recall_per_kernel": {k: (1.0 if i != 4 else 0.0) for i, k in enumerate(ks)}}
    other = {"accuracy": 0.85, "recall_per_kernel": {k: (1.0 if i != 5 else 0.0) for i, k in enumerate(ks)}}
    assert GX.gm_compare(same, other, spread=0.1)["verdict"] == V.GM_DIFFERENCE       # diff inside the spread
    assert GX.gm_compare(perfect, {"accuracy": 0.7, "recall_per_kernel": {k: (1.0 if i < 8 else 0.0) for i, k in enumerate(ks)}}, 0.05)["verdict"] == V.GM_DIFFERENCE  # 4 improving only
    out = corpus(reps=2, idle=0, n_pairs=60)
    for rung in ("apf", "content"):
        _prep(out, rung, n_perm=0, run_null=False)
    GX.gate_gm(out, n_seeds=2, n_estimators=N_EST)
    rs = rows(out / "gates" / "gm.csv")
    assert {(r["rung_a"], r["rung_b"]) for r in rs if r["split"] == "loko"} == {("apf", "content"), ("content", "apf")}
    assert all(r["verdict"] in (V.GM_BEATS, V.GM_DIFFERENCE) for r in rs) and (out / "gates" / "gm_runs" / "seed1").is_dir()


def test_b1_helpers_are_exported():
    import numpy as np
    assert GX.b1_g1_verdict(0.9, np.linspace(0, 0.5, 500))[0] == V.PASS
    assert GX.b1_g1_verdict(0.3, np.linspace(0, 0.5, 500))[0] == V.NEAR_UNFALSIFIABLE
    assert GX.b1_g1_verdict(0.9, np.linspace(0, 0.5, 20))[0] == V.not_run("20 permutations < 500")
    assert GX.headline_classes_of({"a": "X", "b": "X", "c": "X", "d": "Y"}) == ["X"]


# ---------------------------------------------------------------------------------------------
# build epoch 2, builder 3 (SPEC_epoch2 3.5.1, 3.5.2, 6.3 items 7 and 8): the matched comparison for
# every Table 7 (split, label space), and feature_count_used in scores.json
# ---------------------------------------------------------------------------------------------

def test_gdim_matched_all_splits_and_feature_count_used():
    out = corpus(reps=2, idle=0, n_pairs=60)                       # 24 cells: d = 60 > the LOKO training cell count
    for rung in ("apf", "combined"):
        _prep(out, rung, n_perm=0, run_null=False)
    cg = S.selected_grid_id(out, "combined")[0]
    GX.gate_gdim(out, n_perm=0, n_estimators=N_EST, null_splits=set())
    p = out / "gates" / "gdim.csv"
    rs = rows(p)
    assert list(rs[0].keys())[-2:] == ["split", "labelspace"]      # the two appended columns (SPEC_epoch2 3.5.1)
    matched = [r for r in rs if r["rung"] == "combined (matched)"]
    assert [(r["split"], r["labelspace"]) for r in matched] == [list(c) for c in GX.MATCHED_COMBOS] or \
           [(r["split"], r["labelspace"]) for r in matched] == [tuple(c) for c in GX.MATCHED_COMBOS]
    assert matched[0]["split"] == "loko" and matched[0]["labelspace"] == "archetype"      # the first match is the LOKO row
    for r in matched:
        assert r["status"] == V.GDIM_REDUCED and r["d"] == "60" and r["d_matched"] == "8" and r["matched_to"] == "apf"
        d = M.split_dir(out, "combined", cg, r["split"], r["labelspace"], "splits_matched")
        assert (d / "scores.json").is_file(), d
    assert matched[0]["loko_score"] not in ("", None) and all(r["loko_score"] == "" for r in matched[1:])
    for r in rs:
        if r["rung"] in S.RUNGS:
            assert r["split"] == "" and r["labelspace"] == ""
    # the first-match reading of tables._gdim_text is the LOKO row, as in epoch 1
    assert row_where(p, rung="combined (matched)")["split"] == "loko"
    prm = jload(out / "gates" / "gdim.params.json")["params"]
    assert prm["matched_splits"] == "all" and len(prm["matched_dirs"]) == 5 and prm["null_splits"] == []
    # within-trace at the whole-cell point keeps the split stage's own string; LORO/kernel is a number
    wt = jload(M.split_dir(out, "combined", cg, "within_trace", "kernel", "splits_matched") / "scores.json")
    lk = jload(M.split_dir(out, "combined", cg, "loro", "kernel", "splits_matched") / "scores.json")
    assert (wt["status"].startswith("not applicable: one window per cell")) if cg == S.WHOLE_GRID_ID else isinstance(wt["accuracy"], float)
    assert isinstance(lk["accuracy"], float) and lk["params"]["reduce_to"] == 8 and lk["params"]["n_perm"] == 0
    # feature_count_used (SPEC_epoch2 3.5.2): the width the forest saw, per fold and summarised
    lo = jload(M.split_dir(out, "combined", cg, "loko", "archetype", "splits_matched") / "scores.json")
    assert lo["feature_count"] == 60 and lo["feature_count_used"] == 8
    assert set(lo["feature_count_used_per_fold"].values()) == {8} and len(lo["feature_count_used_per_fold"]) == lo["n_folds"]
    assert M.effective_scores(lo)["feature_count_used"] == 8
    # the unreduced combined run on 24 cells: d = 60 exceeds the LOKO training cell count, so the auto-reduction
    # writes feature_count 60 and feature_count_used = the training cell count of the folds (a range when they differ)
    full = jload(M.split_dir(out, "combined", cg, "loko", "archetype") / "scores.json")
    assert full["feature_count"] == 60 and full["dim_status"] == V.GDIM_REDUCED
    fcu = full["feature_count_used"]
    assert (isinstance(fcu, int) and fcu < 60) or (isinstance(fcu, str) and "-" in fcu)
    # the epoch-1 single run stays reachable
    GX.gate_gdim(out, n_perm=0, n_estimators=N_EST, matched_splits="loko", null_splits="loko")
    m1 = [r for r in rows(p) if r["rung"] == "combined (matched)"]
    assert len(m1) == 1 and m1[0]["split"] == "loko"
    assert jload(out / "gates" / "gdim.params.json")["params"]["null_splits"] == ["loko"]
    # the CLI carries both flags
    assert GX.main(["gdim", "--out", str(out), "--null-perm", "0", "--n-estimators", str(N_EST), "--matched-splits", "all", "--null-splits", "loko"]) == 0
    assert len([r for r in rows(p) if r["rung"] == "combined (matched)"]) == 5
