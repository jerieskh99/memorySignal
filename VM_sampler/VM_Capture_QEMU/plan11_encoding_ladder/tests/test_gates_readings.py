"""gates_readings.py: G-J and G-DEC (SPEC 3.6, 5.3; SPEC_review_al_kindi.md items 4 and 9)."""
import numpy as np

from _b2_common import S, V, SY, corpus, tmp_out, rows, row_where, jload, pass_table_with
from plan11_encoding_ladder import gates_readings as GR
from plan11_encoding_ladder import gates_calibration as GC


def test_gj_interpretable_floor_overlap_and_floor_unmeasured():
    out = corpus(reps=2, idle=3, kernels=["gemm", "lexer"])
    GR.gate_gj(out)
    p = out / "gates" / "gj.csv"
    g = row_where(p, cell_id="gemm__rep00__01c"); l = row_where(p, cell_id="lexer__rep00__01c")
    assert g["mask_verdict"] == V.GJ_INTERPRETABLE and float(g["frac_interpretable"]) == 1.0 and float(g["J_q50"]) > 0.9
    assert l["mask_verdict"] == V.GJ_FLOOR_OVERLAP and float(l["frac_interpretable"]) == 0.0
    assert float(g["k_threshold"]) == 3.0 * float(g["floor_median_K"])
    m = np.load(out / "gates" / "gj_mask" / "gemm__rep00__01c.npy")
    assert set(m.dtype.names) == {"mask_K", "mask_persist"} and m["mask_K"].all() and m["mask_persist"].all()
    assert not np.load(out / "gates" / "gj_mask" / "lexer__rep00__01c.npy")["mask_K"].any()
    j = jload(out / "gates" / "gj.json")
    assert len(j["idle_J"]["J"]) == 5 and j["params"]["floor_subtraction"].startswith("none")
    out2 = corpus(reps=2, idle=0, kernels=["gemm", "lexer"])
    GR.gate_gj(out2)
    assert all(r["mask_verdict"] == V.GJ_FLOOR_UNMEASURED for r in rows(out2 / "gates" / "gj.csv"))


def _dec_corpus(**kw):
    out = corpus(reps=8, idle=3, kernels=["floyd", "gibbs"], **kw)
    pass_table_with(out, {"floyd": (12, "declared: synth")})
    GC.gate_gp(out)
    return out


def test_gdec_decay_and_no_decay_control():
    out = _dec_corpus(overrides={"floyd": dict(decay_factor=0.6)})
    GR.gate_gdec(out, n_surrogates=50)
    p = out / "gates" / "gdec.csv"
    assert row_where(p, kernel="floyd", cell_id="all")["verdict"] == V.GDEC_DECAY
    assert row_where(p, kernel="gibbs", cell_id="all")["verdict"].startswith("no slope")
    fl = row_where(p, kernel="floyd", cell_id="floyd__rep00__01c")
    assert int(fl["n_passes"]) >= 10 and float(fl["slope_l0"]) < 0 and float(fl["l0_rel_drop"]) > float(fl["k_rel_drop"])
    GR.gate_gdec(out, n_surrogates=50, boundary_source="period")
    assert row_where(p, kernel="floyd", cell_id="all")["verdict"] == V.GDEC_DECAY


def test_gdec_refusals():
    # undeclared pass period -> decay not resolved
    out = corpus(reps=8, idle=3, kernels=["floyd", "gibbs"])
    GC.gate_gp(out)
    GR.gate_gdec(out, n_surrogates=20)
    assert row_where(out / "gates" / "gdec.csv", kernel="floyd", cell_id="all")["verdict"] == V.GDEC_NOT_RESOLVED
    # K also decaying inside the pass -> no decay beyond breadth
    out = _dec_corpus(overrides={"floyd": dict(k_decay=True)})
    GR.gate_gdec(out, n_surrogates=20)
    assert row_where(out / "gates" / "gdec.csv", kernel="floyd", cell_id="all")["verdict"] == V.GDEC_NO_BEYOND_BREADTH
    # the idle cells with a negative l0 trend -> no decay beyond floor or host
    out = _dec_corpus(overrides={"floyd": dict(decay_factor=0.6)}, idle_overrides=dict(l0_trend=-0.5, idle_l0_max=16))
    GR.gate_gdec(out, n_surrogates=20)
    assert row_where(out / "gates" / "gdec.csv", kernel="floyd", cell_id="all")["verdict"] == V.GDEC_NO_BEYOND_FLOOR
    # no boundary at all (no pulse) -> decay not resolved: no pass boundary
    out = corpus(reps=8, idle=3, kernels=["floyd", "gibbs"], overrides={"floyd": dict(pulse_period=None, pulse_extra=0)})
    pass_table_with(out, {"floyd": (12, "declared: synth")}); GC.gate_gp(out)
    GR.gate_gdec(out, n_surrogates=20)
    assert row_where(out / "gates" / "gdec.csv", kernel="floyd", cell_id="all")["verdict"] == V.GDEC_NO_BOUNDARY
    # no idle cell -> the suffix
    out = corpus(reps=8, idle=0, kernels=["floyd", "gibbs"], overrides={"floyd": dict(decay_factor=0.6)})
    pass_table_with(out, {"floyd": (12, "declared: synth")}); GC.gate_gp(out)
    GR.gate_gdec(out, n_surrogates=20)
    assert row_where(out / "gates" / "gdec.csv", kernel="floyd", cell_id="all")["verdict"] == V.GDEC_DECAY + " (floor unmeasured)"
