"""series.py: the copied shape features against the original, rung series, windows, feature files,
admissibility (SPEC 3.1; SPEC_review_al_farabi.md items 2.3, 2.4)."""
import numpy as np
import pytest

from _b2_common import S, V, SY, corpus, tmp_out, rows, jload, PKG
from plan11_encoding_ladder import gates_precondition as GP


def test_shape_features_equal_original_b1_features():
    import importlib.util, sys
    orig = PKG.parent / "plan08_b1" / "b1_features.py"
    if not orig.is_file():
        pytest.skip("plan08_b1/b1_features.py not importable here")
    spec = importlib.util.spec_from_file_location("b1_features_orig", orig)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    rng = np.random.default_rng(0)
    for n in (1, 2, 7, 8, 16, 33):
        x = rng.random(n) * rng.choice([1e-3, 1.0, 1e3])
        want = np.array([mod.features(x.tolist())[k] for k in S.FEAT])
        got = S.shape_features(x)
        assert np.allclose(want, got, rtol=1e-12, atol=1e-15)
    x = rng.random(100)
    per = np.stack([S.shape_features(x[i * 4:i * 4 + 8]) for i in range(S.n_windows(100, 8, 4))])
    assert np.allclose(per, S.shape_features_windows(x, 8, 4), rtol=1e-12, atol=1e-15)


def test_n_windows_matches_plan02_rule():
    assert S.n_windows(7, 8, 4) == 0 and S.n_windows(8, 8, 4) == 1 and S.n_windows(931, 8, 4) == 231 and S.n_windows(931, 64, 32) == 28


def test_grid_points_are_the_declared_thirteen():
    pts = S.grid_points_ids()
    assert len(pts) == 13 and pts[0] == ("W8_H2", 8, 2) and pts[1] == ("W8_H4", 8, 4) and pts[-1] == ("Wall_Hall", None, None)
    assert S.parse_grid_id("W16_H8") == (16, 8) and S.parse_grid_id("Wall_Hall") == (None, None)


def test_rung_series_shapes_and_normalization():
    out = corpus(reps=1, idle=0, kernels=["gemm"])
    ex = S.load_extract(out, "gemm__rep00__01c")
    n = ex["_n_rows"]
    for rung, d in (("apf", 1), ("wapf", 1), ("persist", 1), ("content", 15), ("combined", 18)):
        X = S.rung_series(ex, rung, normalized=True)
        assert X.shape == (n - 1, d), rung
    raw = S.rung_series(ex, "apf", normalized=False)[:, 0]
    assert np.allclose(raw, ex["K"][:-1] / S.N)
    norm = S.rung_series(ex, "apf", normalized=True, head_drop=3)
    assert norm.shape[0] == n - 1 - 3 and abs(np.median(ex["K"][3:]) * norm[0, 0] - ex["K"][3]) < 1e-6
    assert np.allclose(S.rung_series(ex, "content", normalized=False), S.rung_series(ex, "content", normalized=True), equal_nan=True)
    assert len(S.feature_names("combined")) == 60 and len(S.feature_names("content")) == 36
    with pytest.raises(ValueError):
        S.rung_series(ex, "combined", normalized=False)


def test_build_features_stores_predicted_archetype_only():
    out = corpus(reps=1, idle=1, kernels=["gemm", "lexer"])
    p = S.build_features(out, None, "apf", 8, 4, True)
    f = S.load_features(p)
    assert f["X"].shape[1] == 8 and f["grid_id"] == "W8_H4" and set(f["archetype"]) == {"WORKING-SET", "SEQUENTIAL-GROW", "IDLE"}
    assert list(f["feature_names"])[:2] == ["apf.k_over_med.mean", "apf.k_over_med.std"]
    # G-K0's relabelling never enters the file: relabel lexer and rebuild, the bytes stay the same
    before = p.read_bytes()
    S.write_csv(out / "gates" / "gk0.csv", ("kernel", "verdict", "archetype_measured"),
                [{"kernel": "lexer", "verdict": V.GK0_IDLE_MEASURED, "archetype_measured": "IDLE"}])
    S.build_features(out, None, "apf", 8, 4, True)
    assert S.load_features(p)["archetype"].tolist() == f["archetype"].tolist()
    assert S.gk0_relabel(out) == {"lexer": "IDLE"}
    pw = S.build_features(out, None, "combined", None, None, True)
    fw = S.load_features(pw)
    assert fw["grid_id"] == "Wall_Hall" and all(np.sum(fw["cell_id"] == c) == 1 for c in set(fw["cell_id"]))


def test_admissible_cells_applies_failed_verdict_to_pair_rungs_only():
    out = corpus(reps=1, idle=0, kernels=["gemm", "gibbs"], overrides={"gibbs": dict(failed_count=2, K0=6000)})
    GP.gate_preconditions(out, c1_activity_min=0.0)
    cells = S.load_cells(out / "cells.csv")
    kept, ex_hard, ex_pair, present = S.admissible_cells(out, cells, "persist")
    assert present and ex_pair == ["gibbs__rep00__01c"] and [c["cell_id"] for c in kept] == ["gemm__rep00__01c"]
    kept2, _, ex_pair2, _ = S.admissible_cells(out, cells, "apf")
    assert ex_pair2 == [] and len(kept2) == 2
    assert jload(out / "gates" / "preconditions.json")["excluded_cells_pair_rungs"] == ["gibbs__rep00__01c"]


def test_head_drop_template_and_cell_headline():
    out = tmp_out()
    p = S.write_head_drop_template(out / "inputs" / "head_drop.csv")
    hd = S.load_head_drop(p)
    assert hd["lexer"] == 0 and hd["gemm"] == 0
    out = corpus(reps=1, idle=0, kernels=["gemm"])
    ex = S.load_extract(out, "gemm__rep00__01c")
    assert 0.01 < S.cell_headline(ex, "apf") < 0.03 and 0.5 < S.cell_headline(ex, "persist") <= 1.0 and S.cell_headline(ex, "content") > 0.05
