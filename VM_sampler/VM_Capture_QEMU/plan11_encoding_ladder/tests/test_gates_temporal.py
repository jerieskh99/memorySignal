"""gates_temporal.py: G1-G5 as amended, G3 flag, G-ORD, roll-up and selection (SPEC 3.5, 5.3;
SPEC_review_al_kindi.md items 3, 7; SPEC_review_al_farabi.md items 2.1, 2.2, 2.11 (b))."""
import importlib
import numpy as np
import pytest

from _b2_common import S, V, SY, corpus, tmp_out, rows, row_where, jload, pass_table_with, PKG
from plan11_encoding_ladder import gates_temporal as GT
from plan11_encoding_ladder import gates_calibration as GC
from plan11_encoding_ladder import nulls as NL


def test_stationarity_copy_equals_plan03_original_when_importable():
    import sys
    sys.path.insert(0, str(PKG.parent)); sys.path.insert(0, str(PKG.parent.parent.parent))
    try:
        orig = importlib.import_module("plan03_metric_kernel")
    except Exception:
        orig = None
    rng = np.random.default_rng(0)
    for n, W, H in ((50, 8, 4), (120, 16, 8), (120, 64, 64), (7, 8, 4)):
        x = rng.random(n)
        mine = GT.stationarity_per_window(x.tolist(), W, H)
        if orig is not None:
            assert mine == orig.stationarity_per_window(x.tolist(), W, H)
        batch = GT.stationarity_batch(x[None, :], W, H)[0]
        if mine is None:
            assert np.isnan(batch)
        else:
            assert abs(mine - batch) < 1e-12
    assert GT.stationarity_per_window([1.0] * 20, 8, 4) == 1.0 and GT.stationarity_batch(np.ones((1, 20)), 8, 4)[0] == 1.0


def _cell_x(spec):
    out = tmp_out(); SY.write_corpus(out, [spec])
    ex = S.load_extract(out, SY.cell_id_of(spec))
    return S.temporal_series(ex, "apf"), ex


def test_g1_pass_trend_present_and_fail():
    x, _ = _cell_x(SY.SynthSpec("nbody", 42, **SY.PRESETS["nbody"]))
    sur = NL.surrogates(x, 50)
    r = GT.g1_cell(x, sur, 8, 4)
    assert r["pf_obs"] >= 0.8 and not r["trend"]
    assert GT.g1_kernel([r, r])[0] == V.PASS
    xt, _ = _cell_x(SY.SynthSpec("rmat_gen", 42, K0=2048, content="double", trend=2.0))
    rt = GT.g1_cell(xt, NL.surrogates(xt, 50), 8, 4)
    assert rt["trend"] and abs(rt["drift_sd"]) > 1.0
    assert GT.g1_kernel([rt, rt, r])[0] == V.TREND_PRESENT               # more than half the cells trend
    # a step with equal halves is a least-squares drift of about 1.7 sd, so it reads 'trend present' under the
    # SPEC's drift rule; the non-stationary series that reads 'fail' is a plateau (no linear drift)
    xh, _ = _cell_x(SY.SynthSpec("spmm", 42, K0=2048, content="double", step=True))
    rh = GT.g1_cell(xh, NL.surrogates(xh, 50), 8, 4)
    assert rh["pf_obs"] < 0.8 and rh["trend"] and GT.g1_kernel([rh, rh])[0] == V.TREND_PRESENT
    xs, _ = _cell_x(SY.SynthSpec("spmm", 42, K0=2048, content="double", step=True, step_mode="middle"))
    rs = GT.g1_cell(xs, NL.surrogates(xs, 50), 8, 4)
    assert rs["pf_obs"] < 0.8 and not rs["trend"] and GT.g1_kernel([rs, rs])[0] == V.FAIL


def test_g2_cases_in_seconds_and_pairs():
    e = lambda p, k="declared": GC.PassEntry(p, k)
    r = GT.g2_kernel(e(100), [32] * 8, [931] * 8)
    assert r["G2_0500"] == V.PASS and r["G2_0644"] == V.PASS and r["G2"] == V.PASS and abs(r["coverage_0500"] - 2.667) < 1e-2
    assert abs(r["coverage_pairs"] - 32 * 100 / 931) < 1e-9 and r["G2_pairs"] == V.PASS
    assert GT.g2_kernel(e(6147), [8] * 8, [931] * 8)["G2"] == V.G2_ABOVE_NYQUIST
    r = GT.g2_kernel(e(300), [8] * 8, [931] * 8)
    assert r["G2_0500"] == V.PASS and r["G2_0644"] == V.PASS and abs(r["coverage_0500"] - 2.0) < 1e-9
    r = GT.g2_kernel(e(240), [8] * 8, [931] * 8)
    assert r["G2_0500"] == V.FAIL and r["G2_0644"] == V.PASS and r["G2"] == V.G2_UNDETERMINED
    assert GT.g2_kernel(e(None, "undeclared"), [8], [931])["G2"] == V.GP_UNDECLARED
    assert GT.g2_kernel(e(100, "inferred"), [32], [931])["G2"] == V.PASS + " (INFERRED)"


def test_g3_flag_present_absent_and_the_quefrency_floor():
    spec = SY.SynthSpec("gemm", 42, n_pairs=240, **SY.PRESETS["gemm"])          # al-Kindi item 3 (b): 240 pairs
    x, ex = _cell_x(spec)
    r = GT.g3_cell(x, S.k_median_cell(ex), n_surrogates=100)
    # at 240 pairs the floor n // 8 = 30 sits above the period 24, so the rahmonic 48 is the peak (al-Kindi section 3 item 5)
    assert r["flag_cell"] == V.G3_PRESENT and r["ceps_peak_idx"] % 24 == 0 and r["ceps_peak_idx"] <= len(x) // 2
    assert abs(r["ceps_peak_freq_cyc_per_pair"] - 1 / r["ceps_peak_idx"]) < 1e-9 and r["cv_ratio"] > 0
    x1, ex1 = _cell_x(SY.SynthSpec("gemm", 42, n_pairs=120, **SY.PRESETS["gemm"]))
    r1 = GT.g3_cell(x1, S.k_median_cell(ex1), n_surrogates=100)
    assert r1["ceps_peak_idx"] == 24 and r1["quefrency_floor"] == len(x1) // 8 == 14
    x2, ex2 = _cell_x(SY.SynthSpec("nbody", 42, n_pairs=240, **SY.PRESETS["nbody"]))
    assert GT.g3_cell(x2, S.k_median_cell(ex2), n_surrogates=100)["flag_cell"] == V.G3_ABSENT
    # the quefrency floor n // 8 = 29 at 239 series rows: a period-8 PULSE TRAIN is still flagged through its
    # rahmonics 32 and 40 (the cepstrum is a harmonic-comb detector), while a period-8 SINUSOID is not
    x3, ex3 = _cell_x(SY.SynthSpec("gemm", 42, n_pairs=240, K0=4096, pulse_period=8, pulse_extra=4096, churn=0.02))
    r3 = GT.g3_cell(x3, S.k_median_cell(ex3), n_surrogates=100)
    assert r3["quefrency_floor"] == 29 and r3["flag_cell"] == V.G3_PRESENT and r3["ceps_peak_idx"] % 8 == 0 and r3["ceps_peak_idx"] >= 29
    x4, ex4 = _cell_x(SY.SynthSpec("gemm", 42, n_pairs=240, K0=4096, churn=0.02, k_sine_period=8, k_sine_amp=0.5, k_noise=0.02))
    assert GT.g3_cell(x4, S.k_median_cell(ex4), n_surrogates=100)["flag_cell"] == V.G3_ABSENT
    # the phase-randomized null would tie the observed SNR to machine precision (al-Kindi item 3)
    rp = GT.g3_cell(x, S.k_median_cell(ex), n_surrogates=20, null="phase_randomize")
    assert abs(rp["ceps_snr_db"] - rp["snr_surrogate_p95"]) < 1e-6 and rp["flag_cell"] == V.G3_ABSENT


def test_g3_file_and_kernel_flag():
    out = corpus(reps=3, idle=0, kernels=["gemm", "nbody"], n_pairs=240)
    GT.gate_g3(out, "apf", n_surrogates=50, min_cells=3)
    p = out / "gates" / "g3_flags.csv"
    assert all(r["flag_kernel"] == V.G3_PRESENT for r in rows(p) if r["kernel"] == "gemm")
    assert all(r["flag_kernel"] == V.G3_ABSENT for r in rows(p) if r["kernel"] == "nbody")


def test_g4_and_g5():
    assert GT.g4_pass(8, 2) and GT.g4_pass(8, 4) and not GT.g4_pass(8, 8)
    out = corpus(reps=2, idle=0, kernels=["gemm", "gibbs"], n_pairs=60)
    GT.gate_grid(out, "apf", n_surrogates=5)
    g = lambda gid: row_where(out / "gates" / "grid" / "apf" / gid / "temporal_per_kernel.csv", kernel="gemm")
    assert g("W8_H2")["G4"] == V.PASS and g("W8_H8")["G4"] == V.FAIL and g("Wall_Hall")["G4"] == V.FAIL
    assert g("W8_H4")["G5"] == V.PASS and int(g("W8_H4")["n_windows_min"]) == 13 and float(g("W8_H4")["n_windows_nonoverlap_median"]) == 7
    assert g("W64_H64")["G5"] == V.FAIL          # 59 series rows: no window at W = 64
    assert g("Wall_Hall")["GORD"] == V.GORD_ORDER_BLIND_BY_CONSTRUCTION and g("W8_H4")["GORD"].startswith("pending")
    # every grid point is on disk with its feature files (al-Farabi 2.2)
    for gid, W, H in S.grid_points_ids():
        assert (out / "gates" / "grid" / "apf" / gid / "temporal_per_kernel.csv").is_file()
        assert (out / "gates" / "grid" / "apf" / gid / "g1_surrogates.npz").is_file()
        assert S.features_path(out, "apf", gid, True).is_file() and S.features_path(out, "apf", gid, False).is_file()


def test_gord_resolution_vs_order_blind():
    out = corpus(reps=3, idle=0, n_pairs=90, ord_case=True)
    GT.gate_grid(out, "apf", n_surrogates=2)
    GT.gate_gord(out, "apf", n_order_perm=3, null_perm=12, n_estimators=10)
    j = jload(out / "gates" / "grid" / "apf" / "W16_H8" / "gord.json")
    assert j["GORD"] == V.GORD_RESOLUTION, j
    assert row_where(out / "gates" / "grid" / "apf" / "W16_H4" / "temporal_per_kernel.csv", kernel="gemm")["GORD"] == V.GORD_RESOLUTION
    out2 = corpus(reps=3, idle=0, n_pairs=90, level_only=True)
    GT.gate_grid(out2, "apf", n_surrogates=2)
    GT.gate_gord(out2, "apf", n_order_perm=3, null_perm=12, n_estimators=10)
    assert jload(out2 / "gates" / "grid" / "apf" / "W16_H8" / "gord.json")["GORD"] == V.GORD_ORDER_BLIND


def test_select_rule_and_rollup_of_kernel_refusals():
    out = corpus(reps=2, idle=1, kernels=["gemm", "gibbs", "nbody"], n_pairs=120)
    pass_table_with(out, {"gemm": (100, "declared: synth"), "nbody": (6147, "declared: synth")})
    GT.gate_grid(out, "apf", n_surrogates=5)
    GT.select(out, "apf")
    sel = jload(out / "gates" / "selection.json")["apf"]
    grid = {r["grid_id"]: r for r in rows(out / "gates" / "table5_grid.csv")}
    # nbody above Nyquist is not applicable for G2 (al-Farabi 2.1); gemm at T = 6 s needs W * 0.5 / 6 >= 2 -> W >= 24:
    # the smallest passing W is 32 and the tie-break takes the hop ratio nearest 0.5 -> W32_H16
    assert sel["grid_id"] == "W32_H16" and sel["passes_acceptance"] is True and sel["W"] == 32
    assert grid["W32_H16"]["selected"] == "selected" and grid["W8_H4"]["G2"] == V.FAIL
    assert grid["W8_H4"]["n_kernels_na_G2"] == "1" and grid["W8_H4"]["n_kernels_undeclared_G2"] == "1"
    assert grid["Wall_Hall"]["G4"] == V.FAIL and grid["Wall_Hall"]["selected"] == ""
    assert grid["W32_H16"]["gates_passed"] == "3 of 3" and grid["W32_H8"]["gates_passed"] == "3 of 3" and grid["W32_H8"]["selected"] == ""
    assert len(rows(out / "gates" / "table5_long.csv")) == 13 * 4 and jload(out / "gates" / "grid_complete.json")["apf"]["complete"] is True
    # under 'blocks' the above-Nyquist kernel blocks G2 everywhere: best-feasible
    GT.select(out, "apf", kernel_refusals="blocks")
    sel2 = jload(out / "gates" / "selection.json")["apf"]
    assert sel2["passes_acceptance"] is False and sel2["selected_by"] == "best-feasible" and sel2["grid_id"] == "W8_H4"
    assert {r["grid_id"]: r for r in rows(out / "gates" / "table5_grid.csv")}["W8_H4"]["selected"] == "selected: best-feasible"
    # a trending kernel is not applicable for G1 under the default and blocks under 'blocks'
    out2 = corpus(reps=2, idle=0, kernels=["gemm", "gibbs", "rmat_gen"], n_pairs=120, overrides={"rmat_gen": dict(trend=2.0)})
    GT.gate_grid(out2, "apf", n_surrogates=5)
    assert row_where(out2 / "gates" / "grid" / "apf" / "W8_H4" / "temporal_per_kernel.csv", kernel="rmat_gen")["G1"] == V.TREND_PRESENT
    GT.select(out2, "apf")
    g = {r["grid_id"]: r for r in rows(out2 / "gates" / "table5_grid.csv")}
    assert g["W8_H4"]["G1"] == V.PASS and g["W8_H4"]["n_kernels_na_G1"] == "1"
    GT.select(out2, "apf", kernel_refusals="blocks")
    assert {r["grid_id"]: r for r in rows(out2 / "gates" / "table5_grid.csv")}["W8_H4"]["G1"] == V.FAIL


def test_gc_verdict_propagates_into_the_grid_refusal_column():
    out = corpus(reps=2, idle=0, kernels=["gemm", "gibbs", "histogram"], n_pairs=120, break_pulse=True, overrides={"gemm": dict(K0=1024)})
    GC.gate_gc(out, rung="apf")
    GT.gate_grid(out, "apf", n_surrogates=2); GT.select(out, "apf")
    assert all(r["refusal"] == V.GC_DISCONNECTED for r in rows(out / "gates" / "table5_grid.csv"))
    assert jload(out / "gates" / "grid" / "apf" / "W8_H4" / "temporal.params.json")["params"]["gc_verdict"] == V.GC_DISCONNECTED


# ---------------------------------------------------------------------------------------------
# build epoch 2, builder 3 (SPEC_epoch2 section 5.2 and 6.3 item 3): G-ORD honours --n-jobs and its
# numbers do not depend on the job count
# ---------------------------------------------------------------------------------------------

def test_gord_equal_for_one_and_four_jobs():
    import json as _json
    import shutil
    out = corpus(reps=2, idle=0, n_pairs=60, kernels=["gemm", "gibbs", "fft", "lexer"])
    GT.gate_grid(out, "apf", n_surrogates=2)
    GT.gate_gord(out, "apf", n_order_perm=2, null_perm=4, n_estimators=10, n_jobs=1)
    keep = tmp_out()
    ws = [w for w in GT.schema.GRID_WINDOWS if w != "whole"]
    one = {}
    for W in ws:
        gid = GT.schema.grid_id(W, W // 2)
        src = out / "gates" / "grid" / "apf" / gid / "gord.json"
        shutil.copy(src, keep / f"{gid}.json")
        one[gid] = _json.loads(src.read_text())
        assert one[gid]["params"]["n_jobs"] == 1 and one[gid]["params"]["parallel_backend"] == "joblib threads"
    per_kernel_one = {gid: rows(out / "gates" / "grid" / "apf" / gid / "temporal_per_kernel.csv") for gid, _, _ in S.grid_points_ids()}
    GT.gate_gord(out, "apf", n_order_perm=2, null_perm=4, n_estimators=10, n_jobs=4)
    for W in ws:
        gid = GT.schema.grid_id(W, W // 2)
        four = _json.loads((out / "gates" / "grid" / "apf" / gid / "gord.json").read_text())
        assert four["params"]["n_jobs"] == 4
        p1 = {k: v for k, v in one[gid]["params"].items() if k != "n_jobs"}
        p4 = {k: v for k, v in four["params"].items() if k != "n_jobs"}
        assert p1 == p4
        for key in ("score_ordered", "score_shuffled", "score_shuffled_mean", "null_spread", "null_summary", "GORD", "W", "H"):
            assert one[gid].get(key) == four.get(key), (gid, key, one[gid].get(key), four.get(key))
        if four.get("score_ordered") is not None:            # W above the series length reads `not run: fewer than two archetypes with windows`
            assert isinstance(four["score_shuffled"], list) and len(four["score_shuffled"]) == 2
            assert four["null_summary"]["n"] == 4
    assert any(one[GT.schema.grid_id(W, W // 2)].get("score_ordered") is not None for W in ws)
    for gid, _, _ in S.grid_points_ids():
        after = rows(out / "gates" / "grid" / "apf" / gid / "temporal_per_kernel.csv")
        for a, b in zip(per_kernel_one[gid], after):
            for col in ("gord_score_ordered", "gord_score_shuffled_mean", "gord_null_spread", "GORD"):
                assert a[col] == b[col], (gid, col)
    # the pre-drawn order permutations are epoch 1's stream: repetition outer, cell inner, one permutation of the
    # cell's row count each (what nulls.order_shuffle drew through featurize(rng_o))
    rng_a = NL.default_rng(NL.SEED_ORDER) if hasattr(NL, "default_rng") else np.random.default_rng(NL.SEED_ORDER)
    rng_b = np.random.default_rng(NL.SEED_ORDER)
    lens = [10, 7, 12]
    drawn = [[rng_a.permutation(n) for n in lens] for _ in range(2)]
    legacy = [[NL.order_shuffle(np.arange(n), rng_b) for n in lens] for _ in range(2)]
    for rep in range(2):
        for i in range(len(lens)):
            assert list(drawn[rep][i]) == list(legacy[rep][i])
    # `grid` accepts --n-jobs and records the job count it ran with (builder B's B6 parallelized it; this
    # builder's SPEC_epoch2 5.2 line "accepted and unused" is superseded by that record)
    j = jload(out / "gates" / "grid" / "apf" / "W8_H4" / "temporal.params.json")["params"]
    assert "n_jobs" in j or "n_jobs_used" in j


def test_epoch2_b6_grid_identical_at_any_n_jobs():
    """SPEC_epoch2 B6 (CHECK_3 M6; E1 6.28): gate_grid with n_jobs=2 writes every temporal_per_kernel.csv
    byte-identical to n_jobs=1 on the same corpus (the surrogates come from each cell's own seeded
    generator, the scoring runs through joblib)."""
    out = corpus(reps=2, idle=1, kernels=["gemm", "gibbs"], n_pairs=60)
    GT.gate_grid(out, "apf", n_surrogates=5, n_jobs=1)
    one = {gid: (out / "gates" / "grid" / "apf" / gid / "temporal_per_kernel.csv").read_bytes() for gid, _, _ in S.grid_points_ids()}
    GT.gate_grid(out, "apf", n_surrogates=5, n_jobs=2)
    for gid, _, _ in S.grid_points_ids():
        assert (out / "gates" / "grid" / "apf" / gid / "temporal_per_kernel.csv").read_bytes() == one[gid], gid
    prm = jload(out / "gates" / "grid" / "apf" / "W8_H4" / "temporal.params.json")["params"]
    assert prm["n_jobs"] == 2 and prm["duration_source"] == "sidecar duration_s_declared"


def test_epoch2_b12_g2_reads_the_declared_duration_of_a_synthetic_cell():
    """SPEC_epoch2 B12 (CHECK_3 M12; E1 6.62): a synth.py cell set at duration_s = 77.28, extracted at
    --duration-s 77.28, gemm declared at 5 passes: G2 passes at W64 (coverage 2.07 and 2.67) and fails at
    W32; the sidecars of tests/_synth_b2.py (600 s) leave the existing selection tests unchanged."""
    from plan11_encoding_ladder import synth, extract
    root = tmp_out(); out = root / "out"
    for rep_dir, seed in ((1, 42), (2, 1000)):
        synth.write_cell(synth.SynthSpec(name="gemm", seed=seed, n_pairs=120, rep_dir=rep_dir, duration_s=77.28, **synth.PRESETS["gemm"]),
                         root / "root", compress=False)
    extract.build_index(root / "root", out / "cells.csv")
    extract.extract_all(out / "cells.csv", out, duration_s=77.28)
    assert all(S.load_sidecar(out, c["cell_id"])["duration_s_declared"] == 77.28 for c in S.load_cells(out / "cells.csv"))
    pass_table_with(out, {"gemm": (5, "declared: synth")})
    GT.gate_grid(out, "apf", n_surrogates=2)
    r64 = row_where(out / "gates" / "grid" / "apf" / "W64_H32" / "temporal_per_kernel.csv", kernel="gemm")
    assert r64["G2_0500"] == V.PASS and r64["G2_0644"] == V.PASS and r64["G2"] == V.PASS
    assert abs(float(r64["coverage_0500"]) - 2.0704) < 1e-3 and abs(float(r64["coverage_0644"]) - 2.6667) < 1e-3
    r32 = row_where(out / "gates" / "grid" / "apf" / "W32_H16" / "temporal_per_kernel.csv", kernel="gemm")
    assert r32["G2_0500"] == V.FAIL and r32["G2_0644"] == V.FAIL
    prm = jload(out / "gates" / "grid" / "apf" / "W64_H32" / "temporal.params.json")["params"]
    assert prm["duration_s_per_kernel"]["gemm"]["median"] == 77.28 and prm["duration_s_fallback_cells"] == []
    # G-P reads the same sidecar: T_seconds = 77.28 / 5
    GC.gate_gp(out)
    assert abs(float(row_where(out / "gates" / "gp.csv", kernel="gemm", cell_id="all")["T_seconds"]) - 77.28 / 5) < 1e-9
    assert jload(out / "gates" / "gp.params.json")["params"]["duration_source"] == "sidecar duration_s_declared"
    # the same declared count against the constant 600 (the in-test generator's sidecars): W64 fails
    assert GT.g2_kernel(GC.PassEntry(5, "declared"), [64, 64], [120, 120])["G2_0500"] == V.FAIL
