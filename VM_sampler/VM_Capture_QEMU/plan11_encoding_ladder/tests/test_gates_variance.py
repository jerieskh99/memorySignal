"""variance.py: G-V estimable vs LOKO not estimable (SPEC 3.8; CR 2.3 item 35)."""
import numpy as np

from _b2_common import S, V, corpus, rows, row_where
from plan11_encoding_ladder import variance as GV
from plan11_encoding_ladder import gates_temporal as GT


def test_variance_levels_arithmetic():
    X = np.array([[0.0], [0.0], [10.0], [10.0], [20.0], [20.0], [30.0], [30.0]])
    lv = GV.variance_levels(X, ["a", "a", "b", "b", "c", "c", "d", "d"], ["A", "A", "A", "A", "B", "B", "B", "B"])
    assert lv["L0"][0] == 0.0 and lv["L2"][0] == 25.0 and lv["L3"][0] == 100.0 and lv["n_archetypes_multi"] == 2


def test_gv_estimable_vs_not_estimable():
    out = corpus(reps=3, idle=0, n_pairs=60, archetype_consistent=True)
    GT.gate_grid(out, "apf", n_surrogates=2); GT.select(out, "apf")
    GV.gate_gv(out)
    s = row_where(out / "gates" / "gv_summary.csv", rung="apf")
    assert s["verdict"] == V.GV_ESTIMABLE and int(s["n_features_L0_gt_L3"]) < 8
    assert len([r for r in rows(out / "gates" / "gv.csv") if r["rung"] == "apf"]) == 8
    assert row_where(out / "gates" / "gv_summary.csv", rung="wapf")["verdict"] == V.not_run("no selection for wapf")
    out2 = corpus(reps=3, idle=0, n_pairs=60, one_preset=True)
    GT.gate_grid(out2, "apf", n_surrogates=2); GT.select(out2, "apf")
    GV.gate_gv(out2)
    s2 = row_where(out2 / "gates" / "gv_summary.csv", rung="apf")
    assert s2["verdict"] == V.GV_NOT_ESTIMABLE and int(s2["n_features_L0_gt_L3"]) == 8 - int(s2["n_features_constant"])
