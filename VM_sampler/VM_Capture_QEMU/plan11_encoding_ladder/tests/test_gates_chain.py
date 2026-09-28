"""The gate chain on builder 1's own pipeline (synth.py corpus -> extract.py all) when both modules are
present; skipped otherwise. Checks that every gate runs on the real extract interface and that the
verdicts agree with the in-test generator's on the same presets."""
import importlib
import subprocess
import sys

import pytest

from _b2_common import S, V, PKG, tmp_out, rows, row_where, jload, admissibility, pass_table_with
from plan11_encoding_ladder import (gates_precondition as GP, gates_calibration as GC, gates_temporal as GT,
                                    gates_readings as GR, gates_comparison as GX, variance as GV, models as M)


def _have_builder1():
    return (PKG / "synth.py").is_file() and (PKG / "extract.py").is_file()


@pytest.mark.skipif(not _have_builder1(), reason="builder 1's synth.py / extract.py not present")
def test_chain_on_builder1_extracts():
    root = tmp_out(); out = root / "out"
    env = {"PYTHONPATH": str(PKG.parent)}
    subprocess.run([sys.executable, "-m", "plan11_encoding_ladder.synth", "corpus", "--root", str(root / "root"), "--n-pairs", "60",
                    "--reps", "2", "--idle", "2", "--jobs", "4"], check=True, cwd=str(PKG.parent), env={**env, "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"})
    subprocess.run([sys.executable, "-m", "plan11_encoding_ladder.extract", "index", "--root", str(root / "root"), "--out", str(out)], check=True, cwd=str(PKG.parent))
    subprocess.run([sys.executable, "-m", "plan11_encoding_ladder.extract", "all", "--cells-csv", str(out / "cells.csv"), "--out", str(out), "--jobs", "4"], check=True, cwd=str(PKG.parent))
    cells = S.load_cells(out / "cells.csv")
    assert len(cells) == 26 and {c["role"] for c in cells} == {"kernel", "idle"}
    GP.gate_preconditions(out, assume_failed_zero=True, assume_reason="AA A5", c1_activity_min=0.0005)
    pre = rows(out / "gates" / "preconditions.csv")
    assert all(r["all_hard_pass"] == "true" for r in pre) and all(r["failed_verdict"] == V.PASS for r in pre)
    pass_table_with(out, {"floyd": (6, "declared: synth"), "gemm": (2, "declared: synth")})
    admissibility(out)
    GC.gate_gp(out)
    for rung in ("apf", "persist", "wapf", "content", "combined"):
        GC.gate_gc(out, rung=rung)
    assert all(r["verdict"] == V.PASS for r in rows(out / "gates" / "gc.csv") if r["rep"] == "all")
    GP.gate_gk0(out)
    assert row_where(out / "gates" / "gk0.csv", kernel="lexer")["verdict"] == V.GK0_IDLE_MEASURED
    assert row_where(out / "gates" / "gk0.csv", kernel="gemm")["verdict"] == V.GK0_ABOVE_FLOOR
    GP.gate_gf(out, rung="apf", n_perm=10, n_estimators=10)
    assert row_where(out / "gates" / "gf.csv", part="i", kernel="idle")["verdict"] == V.GF_INSEPARABLE
    assert row_where(out / "gates" / "gf.csv", part="ii", kernel="lexer")["verdict"] == V.GF_AT_FLOOR
    GT.gate_grid(out, "apf", n_surrogates=10); GT.gate_g3(out, "apf", n_surrogates=10, min_cells=2)
    GT.select(out, "apf")
    sel = jload(out / "gates" / "selection.json")["apf"]
    assert sel["grid_id"] and row_where(out / "gates" / "preconditions.csv", cell_id=cells[0]["cell_id"])["C7"] in (V.PASS, V.FAIL)
    gid = sel["grid_id"]
    M.run_split_stage(out, "apf", gid, "loko", "archetype", n_perm=5, n_estimators=10, quarantine=False)
    sc = jload(M.split_dir(out, "apf", gid, "loko", "archetype") / "scores.json")
    assert sc["majority"] == 0.5 and sc["params"]["gk0_applied"] and sc["params"]["relabelled_kernels"] == ["lexer"]
    GR.gate_gj(out)
    assert row_where(out / "gates" / "gj.csv", cell_id="lexer__rep00__synth")["mask_verdict"] == V.GJ_FLOOR_OVERLAP
    GR.gate_gdec(out, n_surrogates=10)
    assert row_where(out / "gates" / "gdec.csv", kernel="floyd", cell_id="all")["verdict"] in (V.GDEC_DECAY, V.GDEC_NO_DECAY, V.GDEC_NO_BEYOND_BREADTH)
    GX.gate_gn(out)
    assert row_where(out / "gates" / "gn.csv", archetype="IDLE")["kernels"] == "lexer"
    GV.gate_gv(out)
    assert row_where(out / "gates" / "gv_summary.csv", rung="apf")["verdict"] in (V.GV_ESTIMABLE, V.GV_NOT_ESTIMABLE)
