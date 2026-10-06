"""Builder A's tests for comparators.py, the comparator rows of tables.py, the sweep figure and the
driver's move 14 (SPEC_epoch2.md Part 1.10). Every test uses synthetic data from synth.py (a
trajectory-level corpus written by function call, extracted by extract.py) or report_fixtures.py;
``n_estimators = 10`` and ``n_perm <= 10`` wherever a forest runs. Runs under ``python3 -m pytest -q``.

No sandbox workload is named here; the synthetic corpus has none."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from _b2_common import S, V, PKG, tmp_out, rows, row_where, jload
from plan11_encoding_ladder import comparators as C
from plan11_encoding_ladder import schema, synth, extract, models as M, tables, figures, run_moves
from plan11_encoding_ladder import gates_precondition as GP

sys.path.insert(0, str(PKG / "tests"))
from report_fixtures import make_out  # noqa: E402

SIX = ("gemm", "floyd", "gibbs", "histogram", "fft", "lexer")
N_PAIRS = 48
SPLIT_FLAGS = ["--null-perm", "5", "--n-estimators", "10", "--null-splits", "loko"]
PY = sys.executable


def _write_corpus(root: Path, out: Path, *, kernels=SIX, reps: int = 2, idle: int = 2, n_pairs: int = N_PAIRS, scale: int = 4) -> Path:
    """A synth.py corpus with trajectories (plain text, `compress=False`) and extract.py's extracts:
    `reps` cells of each kernel from `synth.PRESETS` with K0 and pulse_extra divided by `scale`
    (the comparators do not depend on the level; a smaller cell writes and streams faster) plus
    `idle` idle cells; `cells.csv` from `extract.build_index`, so `path` and `traj_file` are real."""
    specs = []
    for ki, k in enumerate(kernels):
        p = dict(synth.PRESETS[k])
        p["K0"] = int(p.get("K0", 2048)) // scale
        if p.get("pulse_extra"):
            p["pulse_extra"] = int(p["pulse_extra"]) // scale
        for r in range(reps):
            specs.append(synth.SynthSpec(name=k, seed=synth.rep_seed(r, ki), n_pairs=n_pairs, rep_dir=1, label="synth", **p))
    for r in range(idle):
        p = dict(synth.IDLE_PRESET)
        name = p.pop("name")
        specs.append(synth.SynthSpec(name=name, seed=synth.rep_seed(r, len(schema.KERNELS)), n_pairs=n_pairs, rep_dir=1, **p))
    for s_ in specs:
        synth.write_cell(s_, root, compress=False, truth_sides=("t",))
    extract.build_index(root, out / "cells.csv")
    extract.extract_all(out / "cells.csv", out, jobs=1)
    return out


@pytest.fixture(scope="module")
def corpus():
    """One corpus per module; each test copies `out` (the extracts are small; the trajectories stay
    in the shared read-only root that `cells.csv` points to)."""
    base = Path(tempfile.mkdtemp(prefix="plan11_cmp_"))
    _write_corpus(base / "root", base / "out")
    yield base / "out"
    shutil.rmtree(base, ignore_errors=True)


def _fresh(corpus: Path) -> Path:
    out = tmp_out() / "out"
    shutil.copytree(corpus, out)
    return out


def _one_cell_out(spec: synth.SynthSpec, **write_kw) -> tuple[Path, dict, Path]:
    """(out, truth, cell_dir) for one synth cell written, indexed and extracted."""
    root = tmp_out()
    cell = synth.write_cell(spec, root / "root", compress=False, **write_kw)
    truth = json.loads((cell / "truth.json").read_text())
    out = root / "out"
    extract.build_index(root / "root", out / "cells.csv")
    extract.extract_all(out / "cells.csv", out)
    return out, truth, cell


# --------------------------------------------------------------------------- 1. Savoldi
def test_savoldi_matches_truth():
    out, truth, _ = _one_cell_out(synth.SynthSpec(name="gemm", seed=42, n_pairs=40))
    cid = S.load_cells(out / "cells.csv")[0]["cell_id"]
    ex = S.load_extract(out, cid)
    r = C.savoldi_cell(ex, head_drop=3)
    K = np.asarray(truth["K"], dtype=float)[3:]
    assert abs(r["K_mean"] - float(np.mean(K))) < 1e-9
    assert abs(r["K_sd"] - float(np.std(K, ddof=1))) < 1e-9
    assert r["n_rows_used"] == len(K) and r["status"] == "ok"
    assert abs(r["K_median"] - float(np.median(K))) < 1e-9
    assert abs(r["norm"][0] - r["K_mean"] / r["K_median"]) < 1e-12
    r2 = C.savoldi_cell(ex, head_drop=3, rows="rung_series")
    assert r2["n_rows_used"] == len(K) - 1
    assert abs(r2["K_mean"] - float(np.mean(K[:-1]))) < 1e-9
    assert re.match(r"^\d+(\.\d+)?% \+/- \d+(\.\d+)?%$", r["U_text"]), r["U_text"]
    assert abs(C.savoldi_cell(ex, 3, ddof=0)["K_sd"] - float(np.std(K))) < 1e-9
    p = C.run_savoldi(out, no_splits=True)
    prm = jload(out / "gates" / "comparators" / "savoldi.params.json")
    assert prm["schema"] == "plan11.comparators.savoldi.v1" and prm["params"]["ddof"] == 1 and prm["params"]["rows"] == "all_after_head_drop"
    assert prm["citation"] == C.CIT_SAVOLDI and "norm_row_meaning" in prm["params"] and "interval_note" in prm["params"]
    row = rows(p)[0]
    assert row["status"] == "ok" and row["admissible"] == "preconditions not run"
    pk = rows(out / "gates" / "comparators" / "savoldi_per_kernel.csv")
    assert len(pk) == 1 and pk[0]["kernel"] == "gemm" and pk[0]["U_text_median"] == row["U_text"]
    # fewer than two rows: the SD is undefined and the row says so
    r3 = C.savoldi_cell(ex, head_drop=39)
    assert r3["n_rows_used"] == 1 and r3["status"] == "not run: fewer than two rows" and np.isnan(r3["K_sd"])


# --------------------------------------------------------------------------- 2. Dhodapkar-Smith
def test_dhodapkar_delta_is_one_minus_J_and_the_sweep_is_whole():
    spec = synth.SynthSpec(name="floyd", seed=7, n_pairs=60, pulse_period=10, pulse_extra=4096, K0=2048, churn=0.02)
    out, truth, _ = _one_cell_out(spec)
    cid = S.load_cells(out / "cells.csv")[0]["cell_id"]
    ex = S.load_extract(out, cid)
    d = C.dhodapkar_cell(ex, 0)
    J = np.array([np.nan if v is None else v for v in truth["J"]], dtype=float)[:-1]
    delta = 1.0 - J[~np.isnan(J)]
    n = len(delta)
    assert d["n_pairs_used"] == n and d["n_pairs_blank"] == 0
    # the extract writes J with ten significant digits (SPEC 2.2), so the generator's J is matched to
    # 1e-9; the module's delta is 1 - ex["J"] exactly (the identity of C14 cand. 3)
    assert np.max(np.abs((1.0 - ex["J"][:-1]) - delta)) < 1e-9
    assert abs(d["delta_q50"] - float(np.quantile(1.0 - ex["J"][:-1], 0.5))) < 1e-12
    assert abs(d["delta_mean"] - float(np.mean(1.0 - ex["J"][:-1]))) < 1e-12
    at = {s_["delta_th"]: s_ for s_ in d["sweep"]}
    B = int(np.sum(delta > 0.3))
    # every pulse boundary lights the pulse set for one snapshot (SPEC 5.1), so the J dip toward
    # K0 / (K0 + A) = 1/3 appears on the pair entering the boundary and on the pair leaving it:
    # two phase changes per boundary at delta_th = 0.3
    n_bound = sum(1 for b in truth["boundaries"] if truth["seq"][0] <= b < truth["seq"][-1])
    assert B == 2 * n_bound == at[0.3]["n_boundaries"], (B, n_bound, at[0.3])
    assert abs(at[0.3]["stability"] - (1 - B / n)) < 1e-12
    assert abs(at[0.3]["mean_phase_length_pairs"] - n / (B + 1)) < 1e-12
    assert at[0.04]["is_default"] and sum(1 for s_ in d["sweep"] if s_["is_default"]) == 1
    # the interior rule: a phase runs from one boundary pair to the pair before the next
    di = C.dhodapkar_cell(ex, 0, phase_length_rule="interior")
    b_idx = np.flatnonzero(delta > 0.3)
    exp = (b_idx[-1] - b_idx[0]) / (len(b_idx) - 1)
    assert abs({s_["delta_th"]: s_ for s_ in di["sweep"]}[0.3]["mean_phase_length_pairs"] - exp) < 1e-12
    assert np.isnan({s_["delta_th"]: s_ for s_ in di["sweep"]}[0.9]["mean_phase_length_pairs"])   # fewer than two boundaries
    # the "ge" rule counts ties
    assert C.dhodapkar_cell(ex, 0, boundary_rule="ge")["sweep"][0]["n_boundaries"] >= d["sweep"][0]["n_boundaries"]
    # the sweep file is whole: every grid point per cell, exactly one default, equal to params.default
    C.run_dhodapkar(out, no_splits=True)
    sw = rows(out / "gates" / "comparators" / "dhodapkar_sweep.csv")
    prm = jload(out / "gates" / "comparators" / "dhodapkar.params.json")["params"]
    assert len(sw) == len(C.DHODAPKAR_GRID) and sorted(float(r["delta_th"]) for r in sw) == sorted(C.DHODAPKAR_GRID)
    marked = [r for r in sw if r["is_default"] == "true"]
    assert len(marked) == 1 and float(marked[0]["delta_th"]) == prm["default"] == 0.04
    assert prm["default_source"] == C.DHODAPKAR_DEFAULT_SOURCE and prm["default_is_declared"] is True and prm["default_appended_to_grid"] is False
    assert prm["grid_source"] == C.DHODAPKAR_GRID_SOURCE
    main = rows(out / "gates" / "comparators" / "dhodapkar.csv")[0]
    assert int(main["n_boundaries"]) == at[0.04]["n_boundaries"] and main["delta_th_default"] == "0.04"
    # the record follows the value (al-Farabi review 2 (a)): a default off the grid is refused, or appended and recorded
    with pytest.raises(ValueError):
        C.run_dhodapkar(out, grid=(0.1, 0.2), default=0.04, no_splits=True)
    C.run_dhodapkar(out, grid=(0.1, 0.2), default=0.3, off_grid_rule="append", no_splits=True)
    prm = jload(out / "gates" / "comparators" / "dhodapkar.params.json")["params"]
    assert prm["default_appended_to_grid"] is True and prm["grid"] == [0.1, 0.2, 0.3]
    assert prm["default_source"].startswith("CLI --delta-th-default 0.3") and prm["default_is_declared"] is False
    assert C.main(["dhodapkar", "--out", str(out), "--grid", "0.1,0.2", "--delta-th-default", "0.04", "--no-splits"]) == 2


# --------------------------------------------------------------------------- 3. Law
def _brute_force(sets, N, L):
    dyn = [len(set.intersection(*sets[t - L + 1:t + 1])) for t in range(L - 1, len(sets))]
    sta = [N - len(set.union(*sets[t - L + 1:t + 1])) for t in range(L - 1, len(sets))]
    return dyn, sta


def _ever(sets, L):
    n = 0
    for pg in set().union(*sets):
        best = cur = 0
        for s_ in sets:
            cur = cur + 1 if pg in s_ else 0
            best = max(best, cur)
        n += best >= L
    return n


def test_law_streaming_pass_matches_brute_force():
    spec = synth.SynthSpec(name="gibbs", seed=5, n_pairs=30, K0=64, churn=0.3, floor_F=0, gap_seqs=(7,))
    out, truth, cell = _one_cell_out(spec, keep_sets=True)
    traj = cell / truth["traj_file"]
    sets = [set(s) for s in truth["sets"]]
    N = spec.N
    gap_t = 7 - truth["seq"][0]
    for unit in ("dumps", "pairs"):
        res = C.law_stream(traj, x_grid=(2, 4, 8), x_unit=unit, head_drop=0)
        assert res["status"] == "ok" and res["n_seq_gaps"] == 1 and res["gap_seqs"] == [7]
        for X in (2, 4, 8):
            L = X - 1 if unit == "dumps" else X
            assert res["L"][X] == L and res["t_first"][X] == L - 1
            dyn, sta = _brute_force(sets, N, L)
            assert res["dyn"][X].tolist() == dyn and res["sta"][X].tolist() == sta, (unit, X)
            assert res["dyn_ever"][X] == _ever(sets, L)
        if unit == "dumps":
            assert res["dyn"][2][gap_t] == 0 and res["sta"][2][gap_t] == N            # the gap is an empty snapshot
    res = C.law_stream(traj, x_grid=(2, 4, 8), head_drop=0)
    cid = S.load_cells(out / "cells.csv")[0]["cell_id"]
    ex = S.load_extract(out, cid)
    assert C.check_x2_equals_K(res, ex) == "true"
    assert C.check_x2_equals_K(C.law_stream(traj, x_grid=(2, 4), x_unit="pairs"), ex).startswith("not applicable")
    # the head-drop rules: the state restarts under reset_at_head_drop, the first window after it is recorded
    hd = 9
    res_r = C.law_stream(traj, x_grid=(2, 4, 8), head_drop=hd, head_drop_rule="reset_at_head_drop")
    assert res_r["t_first"][4] == hd + 2 and res_r["dyn"][4][0] == len(set.intersection(*sets[hd:hd + 3]))
    assert res_r["dyn"][4].tolist() == _brute_force(sets[hd:], N, 3)[0] and res_r["dyn_ever"][8] == _ever(sets[hd:], 7)
    res_d = C.law_stream(traj, x_grid=(2, 4, 8), head_drop=hd)
    assert res_d["t_first"][4] == hd and res_d["dyn"][4][0] == len(set.intersection(*sets[hd - 2:hd + 1]))
    # a corrupt trajectory is refused with the extractor's string, and no series file survives it
    spec_c = synth.SynthSpec(name="gibbs", seed=6, n_pairs=12, K0=64, floor_F=0, corrupt="seq_reverse")
    root_c = tmp_out()
    cell_c = synth.write_cell(spec_c, root_c, compress=False)
    tc = json.loads((cell_c / "truth.json").read_text())
    with pytest.raises(C.Refusal, match=r"^seq not monotone at row \d+$"):
        C.law_stream(cell_c / tc["traj_file"])
    sp = root_c / "law_series" / "x.npz"
    sp.parent.mkdir(parents=True)
    sp.write_bytes(b"stale")
    rec = C._law_worker({"cell_id": "x", "traj_path": str(cell_c / tc["traj_file"]), "out": str(out), "series_path": str(sp),
                         "pass_params": C._law_pass_params((2, 4), "dumps", 0, "runs_from_seq_first", N)})
    assert re.match(r"^refused: seq not monotone at row \d+$", rec["status"]) and not sp.exists()


def test_law_memory_model_and_resume():
    spec = synth.SynthSpec(name="gibbs", seed=11, n_pairs=30, K0=64, churn=0.2, floor_F=0)
    out, truth, cell = _one_cell_out(spec)
    traj = cell / truth["traj_file"]
    seen = []

    def hook(w):
        assert isinstance(w, C._PageList) and isinstance(w.pages, np.ndarray)
        assert len(C._LIVE_PAGE_ARRAYS) <= 1, len(C._LIVE_PAGE_ARRAYS)
        seen.append(int(w.pages.shape[0]))
    C.LAW_STEP_HOOK = hook
    try:
        res = C.law_stream(traj, x_grid=(2, 4))
    finally:
        C.LAW_STEP_HOOK = None
    assert len(seen) == 30 and seen == truth["K"]
    mm = res["memory_model"]
    assert mm["n_length_arrays"] == 4 and mm["page_lists_live_max"] == 1
    assert mm["nbytes_total_N_length_state"] == 4 * spec.N * 4
    # the per-cell resume: a second run skips the cell, --force re-runs it
    C.run_law(out, no_splits=True)
    j1 = jload(out / "gates" / "comparators" / "law_cells.json")
    cid = list(j1["cells"])[0]
    assert j1["cells"][cid]["status"] == "ok" and j1["cells"][cid]["check_x2_equals_K"] == "true"
    assert (out / "gates" / "comparators" / "law_series" / f"{cid}.npz").is_file()
    C.run_law(out, no_splits=True)
    j2 = jload(out / "gates" / "comparators" / "law_cells.json")
    assert j2["cells"][cid]["elapsed_s"] == j1["cells"][cid]["elapsed_s"] and j2["cells"][cid]["resumed"] is True
    assert j2["params"]["n_run"] == 0 and j2["params"]["n_resumed"] == 1
    C.run_law(out, no_splits=True, force=True)
    j3 = jload(out / "gates" / "comparators" / "law_cells.json")
    assert j3["params"]["n_run"] == 1 and "resumed" not in j3["cells"][cid]
    # a changed pass parameter is not a valid resume
    C.run_law(out, no_splits=True, x_unit="pairs")
    assert jload(out / "gates" / "comparators" / "law_cells.json")["params"]["n_run"] == 1
    prm = jload(out / "gates" / "comparators" / "law.params.json")["params"]
    assert prm["x_unit"] == "pairs" and prm["x_default_source"] == C.LAW_X_DEFAULT_SOURCE and prm["memory_model"]["n_length_arrays"] == 4
    sw = rows(out / "gates" / "comparators" / "law_sweep.csv")
    assert len(sw) == len(C.LAW_X_GRID) and [r for r in sw if r["is_default"] == "true"][0]["X"] == "4"
    assert rows(out / "gates" / "comparators" / "law.csv")[0]["check_x2_equals_K"].startswith("not applicable")


# --------------------------------------------------------------------------- 5. feature files and the split layout
def _loaded(out, name, normalized):
    return S.load_features(S.features_path(out, name, "Wall_Hall", normalized))


def test_feature_files_and_split_layout(corpus):
    out = _fresh(corpus)
    assert C.main(["savoldi", "--out", str(out), *SPLIT_FLAGS]) == 0
    f = _loaded(out, "cmp_savoldi", True)
    cells = S.load_cells(out / "cells.csv")
    for k in ("X", "feature_names", "cell_id", "kernel", "archetype", "campaign", "role", "rep", "win_start", "n_series_cell",
              "W", "H", "grid_id", "normalized", "head_drop_json", "n_windows_dropped", "wapf_norm"):
        assert k in f, k
    assert f["X"].shape == (len(cells), 2) and f["grid_id"] == "Wall_Hall" and f["normalized"] is True
    assert int(f["W"]) == -1 and int(f["H"]) == -1 and str(f["wapf_norm"]) == ""
    assert list(f["cell_id"]) == [c["cell_id"] for c in cells] and np.all(f["win_start"] == 0)
    idle = f["role"] == "idle"
    assert idle.sum() == 2 and set(f["archetype"][idle]) == {"IDLE"} and set(f["kernel"][idle]) == {"idle"}
    assert list(f["feature_names"]) == list(C.SAVOLDI_NAMES_NORM)
    assert list(_loaded(out, "cmp_savoldi", False)["feature_names"]) == list(C.SAVOLDI_NAMES_RAW)
    assert np.all(f["n_series_cell"] == N_PAIRS)
    sc = jload(M.split_dir(out, "cmp_savoldi", "Wall_Hall", "loko", "archetype") / "scores.json")
    assert sc["params"]["grid_id"] == "Wall_Hall" and sc["feature_count"] == 2 and sc["params"]["gc_verdict"] is None
    wt = jload(M.split_dir(out, "cmp_savoldi", "Wall_Hall", "within_trace", "kernel") / "scores.json")
    assert wt["status"].startswith("not applicable: one window per cell")
    assert (M.split_dir(out, "cmp_savoldi", "Wall_Hall", "loko", "archetype", normalized=False) / "scores.json").is_file()
    assert M.split_dir(out, "cmp_savoldi", "Wall_Hall", "loko", "archetype", normalized=False).name == "loko__archetype__raw"
    # the pair-rung exclusion (the PAIR_RUNGS line): a refused failed count keeps a cell out of the
    # two comparators that read pair adjacency and leaves it in Savoldi
    GP.gate_preconditions(out, assume_failed_zero=True, assume_reason="test", c1_activity_min=0.0)
    victim = next(c["cell_id"] for c in cells if c["kernel"] == "gemm")
    pre = rows(out / "gates" / "preconditions.csv")
    for r in pre:
        if r["cell_id"] == victim:
            r["failed_verdict"] = V.refused("failed count 3 > 0")
    S.write_csv(out / "gates" / "preconditions.csv", list(pre[0].keys()), pre)
    pj = jload(out / "gates" / "preconditions.json")
    pj["excluded_cells_pair_rungs"] = [victim]
    (out / "gates" / "preconditions.json").write_text(json.dumps(pj))
    assert C.main(["savoldi", "--out", str(out), *SPLIT_FLAGS]) == 0
    assert C.main(["dhodapkar", "--out", str(out), *SPLIT_FLAGS]) == 0
    assert C.main(["law", "--out", str(out), *SPLIT_FLAGS]) == 0
    for name, present in (("cmp_savoldi", True), ("cmp_dhodapkar", False), ("cmp_law", False)):
        ids = {r["cell_id"] for r in rows(M.split_dir(out, name, "Wall_Hall", "loko", "archetype") / "predictions.csv")}
        assert (victim in ids) is present, (name, present)
    dh = row_where(out / "gates" / "comparators" / "dhodapkar.csv", cell_id=victim)
    assert dh["excluded_pair_rung"] == "true" and dh["admissible"] == "true"
    assert row_where(out / "gates" / "comparators" / "law.csv", cell_id=victim)["excluded_pair_rung"] == "true"
    law_rows = rows(out / "gates" / "comparators" / "law.csv")
    assert all(r["status"] == "ok" and r["check_x2_equals_K"] == "true" for r in law_rows)


# --------------------------------------------------------------------------- 6. the gates on the comparator rows
def _apf_baseline(out: Path):
    hd = S.load_head_drop(out / "inputs" / "head_drop.csv")
    for norm in (False, True):
        S.build_features(out, None, "apf", 8, 4, norm, hd)
        for split, ls in (("loko", "archetype"), ("loro", "kernel")):
            M.run_split_stage(out, "apf", "W8_H4", split, ls, normalized=norm, n_perm=5, n_estimators=10, run_null=split == "loko")
    (out / "gates").mkdir(exist_ok=True)
    (out / "gates" / "selection.json").write_text(json.dumps({"schema": "plan11.selection.v1", "params": {}, "citation": "test",
                                                             "apf": {"grid_id": "W8_H4"}}))
    (out / "gates" / "gm.params.json").write_text(json.dumps({"schema": "plan11.gm.v1", "params": {"spread": 0.04}, "citation": "test"}))


def test_gates_for_comparators(corpus):
    out = _fresh(corpus)
    _apf_baseline(out)
    assert C.main(["all", "--out", str(out), *SPLIT_FLAGS]) == 0
    d = out / "gates" / "comparators"
    gl = rows(d / "gl.csv")
    assert [r["rung"] for r in gl] == list(schema.COMPARATOR_NAMES) and all(r["part"] == "i" for r in gl)
    # at five permutations B1-G1 reads `not run: 5 permutations < 500` (SPEC 3.7.1) and G-L (i) inherits
    # it exactly as gates_comparison.gate_gl does for a rung; the score and the null p95 stay in their columns
    assert all(r["verdict"] == "not run: 5 permutations < 500" for r in gl), gl
    assert all(re.match(r"^[01](\.\d+)?$", r["score_norm"]) and r["null_p95"] != "" for r in gl), gl
    # with the null's verdict written as `pass` the same reading gives pass / level only
    sc = jload(M.split_dir(out, "cmp_savoldi", "Wall_Hall", "loko", "archetype") / "scores.json")
    assert C.gl_part1_for({**sc, "b1_g1": V.PASS}, "")["verdict"] in (V.PASS, V.GL_LEVEL_ONLY)
    assert C.gl_part1_for({**sc, "b1_g1": V.PASS, "accuracy": 0.9, "null_p95": 0.5}, "")["verdict"] == V.PASS
    assert C.gl_part1_for({**sc, "b1_g1": V.PASS, "accuracy": 0.5, "null_p95": 0.5}, "")["verdict"] == V.GL_LEVEL_ONLY
    assert jload(d / "gl.params.json")["params"]["part_ii"] == C.GL_PART_II
    gd = {r["rung"]: r for r in rows(d / "gdim.csv")}
    assert gd["cmp_savoldi"]["status"] == V.GDIM_FULL and gd["cmp_savoldi"]["d"] == "2" and gd["cmp_savoldi__raw"]["d"] == "2"
    # d is the rung's feature count through effective_scores: the re-run's count when B1-G3 quarantined a
    # feature (SPEC 3.7.2), the full count otherwise; Law has four features and Dhodapkar-Smith three
    for name, full in (("cmp_law", 4), ("cmp_dhodapkar", 3)):
        for norm in (True, False):
            sc_ = M.effective_scores(jload(M.split_dir(out, name, "Wall_Hall", "loko", "archetype", normalized=norm) / "scores.json"))
            key = name if norm else f"{name}__raw"
            assert gd[key]["d"] == str(sc_["feature_count"]) and int(gd[key]["d"]) <= full and gd[key]["d_matched"] == gd[key]["d"], (key, gd[key])
            assert jload(M.split_dir(out, name, "Wall_Hall", "loko", "archetype", normalized=norm) / "scores.json")["feature_count"] == full
    assert gd["cmp_savoldi"]["d_matched"] == "2"
    gm = rows(d / "gm.csv")
    assert {r["rung_b"] for r in gm} == {"apf", "apf__raw"}
    loko = [r for r in gm if r["split"] == "loko"]
    assert len(loko) == 6 and all(r["verdict"] in (V.GM_BEATS, V.GM_DIFFERENCE) for r in loko), loko
    assert all(r["spread"] == "0.04" for r in loko)
    wt = [r for r in gm if r["split"] == "within_trace"]
    assert all(r["verdict"].startswith("not applicable: one window per cell") for r in wt)         # al-Farabi review 5.5
    gx = row_where(out / "gates" / "gx.csv", rung="cmp_savoldi")
    # the synthetic corpus has one campaign label, so G-X writes its not-applicable row (no forest runs)
    assert gx["grid_id"] == "Wall_Hall" and gx["leak_verdict"] == "not applicable: one campaign label"
    assert {r["rung"] for r in rows(out / "gates" / "gx.csv")} >= set(schema.COMPARATOR_NAMES)
    vd = rows(d / "verdicts.csv")
    assert len(vd) == 30 and {r["rung"] for r in vd} == set(schema.COMPARATOR_NAMES)
    lk = row_where(d / "verdicts.csv", rung="cmp_savoldi", variant="norm", split="loko", labelspace="archetype")
    assert lk["feature_count"] == "2" and lk["gl"] == "not run: 5 permutations < 500" and lk["gdim"] == V.GDIM_FULL
    assert row_where(d / "verdicts.csv", rung="cmp_savoldi", variant="raw", split="loko", labelspace="archetype")["gl"] == C.GL_RAW_TEXT
    # without the measured spread the G-M verdict names move 12's file
    (out / "gates" / "gm.params.json").unlink()
    C.run_gates(out, null_perm=5, n_estimators=10)
    assert all(r["verdict"] == "not run: gates/gm.params.json missing (move 12)" for r in rows(d / "gm.csv") if r["split"] == "loko")
    # a missing comparator gets not run: rows, and the gate files still carry their params block
    shutil.rmtree(out / "gates" / "splits" / "cmp_law")
    C.run_gates(out, null_perm=5, n_estimators=10)
    assert row_where(d / "gl.csv", rung="cmp_law")["verdict"].startswith("not run: gates/splits/cmp_law/Wall_Hall/loko__archetype/scores.json missing")
    assert jload(d / "gm.params.json")["schema"] == "plan11.comparators.gm.v1"


# --------------------------------------------------------------------------- 7. the tables
def test_table7_comparators_and_summary_table():
    out = make_out(tmp_out() / "out", n_pairs=40)
    tables.run(out, ["table7_comparators", "table_comparators"])
    t = rows(out / "report" / "tables" / "table7_comparators.csv")
    assert len(t) == 30 and list(t[0].keys()) == tables.TABLE7_COLUMNS
    for r in t:
        for c in ("accuracy", "macro recall (headline rows)", "null p95", "rank", "majority", "feature count"):
            assert re.match(r"^not run: gates/splits/cmp_(savoldi|dhodapkar|law)/Wall_Hall/(loko|loro|within_trace)__(kernel|archetype)(__raw)?/scores\.json missing \(move 14\)$", r[c]), r[c]
        assert r["G-C"] == tables.CMP_GC_TEXT and r["G-F (i)"] == tables.CMP_GF1_TEXT
        assert r["resolution (W x H)"] == tables.CMP_RESOLUTION_TEXT
    assert t[0]["rung"] == "Savoldi 2010 (as published; U = mean +/- SD of K)"
    assert t[5]["rung"] == "Savoldi 2010 (level-normalized; U = mean +/- SD of K)"
    assert t[10]["rung"] == "Dhodapkar-Smith 2003 (as published; delta_th = 0.04, declared default)"
    assert t[25]["rung"] == "Law 2010 (level-normalized; X = 4, declared default)"
    assert [r["split"] for r in t[:5]] == ["LOKO", "LORO", "within-trace", "LORO", "within-trace"]
    assert t[0]["G-L"] == tables.CMP_GL_RAW_TEXT and t[5]["G-L"] == "not run: gates/comparators/gl.csv missing"
    assert t[5]["G-M vs APF"] == "not run: gates/comparators/gm.csv missing"
    assert t[5]["G-DIM"] == "not run: gates/comparators/gdim.csv missing"
    assert t[5]["G-X"].startswith("not run: gates/gx.csv missing or has no row for cmp_savoldi")
    tables.table7(out)
    assert len(rows(out / "report" / "tables" / "table7.csv")) == 30
    # a hand-written LOKO norm run: the re-run's numbers through effective_scores, feature count = the count used
    d = M.split_dir(out, "cmp_savoldi", "Wall_Hall", "loko", "archetype")
    d.mkdir(parents=True)
    (d / "scores.json").write_text(json.dumps({
        "schema": "plan11.scores.v1", "params": {"normalized": True, "grid_id": "Wall_Hall"}, "citation": "test", "status": "ok",
        "accuracy": 0.75, "macro_recall": 0.7, "null_p95": 0.5, "b1_g1": "pass", "b1_g1_rank": 480, "n_perm": 500,
        "majority": 0.5, "feature_count": 2, "feature_count_used": 2, "dim_status": "full vector",
        "with_quarantine": {"accuracy": 0.6, "macro_recall": 0.55, "null_p95": 0.45, "b1_g1": "pass", "b1_g1_rank": 470,
                            "majority": 0.5, "feature_count": 1, "feature_count_used": 1, "quarantined_features": ["cmp_savoldi.k_over_med.mean"]}}))
    (out / "gates" / "comparators").mkdir(exist_ok=True)
    S.write_csv(out / "gates" / "comparators" / "gm.csv", C.GM_COLUMNS,
                [{"split": "loko", "rung_a": "cmp_savoldi", "rung_b": "apf", "score_a": 0.6, "score_b": 0.7, "diff": -0.1, "spread": 0.04,
                  "improving": 2, "worsening": 8, "ties": 2, "verdict": V.GM_DIFFERENCE},
                 {"split": "within_trace", "rung_a": "cmp_savoldi", "rung_b": "apf", "verdict": "not applicable: one window per cell"}])
    tables.run(out, ["table7_comparators"])
    t = rows(out / "report" / "tables" / "table7_comparators.csv")
    lk = t[5]
    assert lk["accuracy"] == "0.600" and lk["feature count"] == "1" and lk["null p95"] == "0.450" and lk["rank"] == "rank 470 of 500"
    assert lk["G-M vs APF"] == "difference with margin (diff -0.100 <= spread 0.040; 2 up, 8 down)"
    assert t[7]["G-M vs APF"] == "not applicable: one window per cell"
    assert t[0]["G-M vs APF"] == "not run: gates/comparators/gm.csv has no (loko, cmp_savoldi__raw, apf__raw) row"
    # the per-kernel table: not run without the files, the medians with hand-written ones
    tc = rows(out / "report" / "tables" / "table_comparators.csv")
    assert [r["kernel"] for r in tc] == list(schema.KERNEL_NAMES) + ["idle"]
    assert tc[0]["Savoldi U (median cell)"] == "not run: gates/comparators/savoldi_per_kernel.csv missing (move 14)"
    assert tc[0]["D-S boundaries (median, delta_th = 0.04)"] == "not run: gates/comparators/dhodapkar_per_kernel.csv missing (move 14)"
    assert tc[0]["Law dynamic-for-X (median, X = 4)"] == "not run: gates/comparators/law_per_kernel.csv missing (move 14)"
    S.write_csv(out / "gates" / "comparators" / "savoldi_per_kernel.csv", C.SAVOLDI_KERNEL_COLUMNS,
                [{"kernel": "gemm", "n_cells": 8, "K_mean_median": 4200.5, "K_mean_min": 4000, "K_mean_max": 4400, "K_sd_median": 30.25, "U_text_median": "1.602% +/- 0.0115%"}])
    S.write_csv(out / "gates" / "comparators" / "dhodapkar_per_kernel.csv", C.DHODAPKAR_KERNEL_COLUMNS,
                [{"kernel": "gemm", "n_cells": 8, "n_boundaries_median": 10, "stability_median": 0.83, "mean_phase_length_median": 5.36, "delta_q50_median": 0.04}])
    S.write_csv(out / "gates" / "comparators" / "law_per_kernel.csv", C.LAW_KERNEL_COLUMNS,
                [{"kernel": "gemm", "n_cells": 8, "dyn_mean_median": 4000.0, "dyn_frac_mean_median": 0.0153, "sta_frac_mean_median": 0.98, "dyn_ever_median": 4500}])
    tables.run(out, ["table_comparators"])
    g = row_where(out / "report" / "tables" / "table_comparators.csv", kernel="gemm")
    assert g["Savoldi U (median cell)"] == "1.602% +/- 0.0115%" and g["n cells"] == "8"
    assert g["D-S stability (median)"] == "0.83" and g["Law dynamic ever (median)"] == "4500"
    assert (out / "report" / "tables" / "table_comparators.tex").is_file() and (out / "report" / "tables" / "table7_comparators.json").is_file()


# --------------------------------------------------------------------------- 8. the figure
def test_figure_dhodapkar_sweep():
    out = make_out(tmp_out() / "out", n_pairs=40)
    w = figures.run(out, ["dhodapkar_sweep"])
    fd = out / "report" / "figures"
    assert (fd / "fig_dhodapkar_sweep.png").is_file() and (fd / "fig_dhodapkar_sweep.pdf").is_file()
    assert jload(fd / "figures.json")["status"] == "ok" and w["dhodapkar_sweep"]["n_panels"] == 0
    d = out / "gates" / "comparators"
    d.mkdir(parents=True)
    sw = []
    for k in ("gemm", "floyd"):
        for rep in (0, 1):
            for th in C.DHODAPKAR_GRID:
                sw.append({"cell_id": f"{k}__rep0{rep}__dwarfs1", "kernel": k, "role": "kernel", "rep": rep, "campaign": "dwarfs1",
                           "admissible": "true", "excluded_pair_rung": "false", "delta_th": th, "is_default": th == 0.04,
                           "n_pairs_used": 39, "n_pairs_blank": 0, "n_boundaries": int(round(30 * (1 - th))), "stability": th,
                           "mean_phase_length_pairs": 39 / (int(round(30 * (1 - th))) + 1)})
    S.write_csv(d / "dhodapkar_sweep.csv", C.DHODAPKAR_SWEEP_COLUMNS, sw)
    (d / "dhodapkar.params.json").write_text(json.dumps({"schema": "plan11.comparators.dhodapkar.v1", "citation": "test",
                                                        "params": {"default": 0.04, "default_is_declared": True, "grid": list(C.DHODAPKAR_GRID)}}))
    w = figures.run(out, ["dhodapkar_sweep"])["dhodapkar_sweep"]
    assert w["n_panels"] == 2 and w["n_cells_drawn"] == 4 and w["default"] == 0.04
    assert jload(fd / "figures.json")["status"] == "ok"


# --------------------------------------------------------------------------- 9. the driver's move 14
def _ns(**kw):
    import argparse
    base = dict(out="/x/out", root="/x/root", cells_csv=None, moves="0-13", assume_failed_zero=False, assume_reason=None,
                n_jobs=1, null_perm=500, null_splits="loko,loro,within_trace", seed_offset=0, force=False, dry_run=False,
                skip_missing_modules=False, only_modules=None, persist_side=None, failed_counts=None, table8_rung="combined",
                piano_cell=None, piano_stride=16, standalone_tex=None, cmd="plan")
    base.update(kw)
    return argparse.Namespace(**base)


def _driver_accepts(flag: str) -> bool:
    """Whether the driver's `run` parser knows `flag` (builder B's epoch-2 flags, `--n-estimators` among
    them, may or may not have landed; the test tolerates both states, SPEC_epoch2 Part 3.2)."""
    import argparse
    ap = argparse.ArgumentParser()
    run_moves._add_run_args(ap)
    return any(flag in a.option_strings for a in ap._actions)


def test_driver_move14_plan_and_run(corpus):
    assert run_moves.parse_moves("0-14") == list(range(15))
    with pytest.raises(ValueError):
        run_moves.parse_moves("18")
    P = [c for c in run_moves.build_plan(_ns()) if c["move"] == 14]
    assert [c["name"] for c in P] == ["comparators savoldi", "comparators dhodapkar", "comparators law", "comparators gates",
                                      "tables table7_comparators,table_comparators", "figures dhodapkar_sweep", "tables manifest (after comparators)"]
    args = {c["name"]: c["args"] for c in P}
    assert args["comparators dhodapkar"][-2:] == ["--delta-th-default", "0.04"] and "--x-default" in args["comparators law"]
    assert args["comparators law"][args["comparators law"].index("--x-default") + 1] == "4"
    assert args["comparators savoldi"][-2:] == ["--rows", "all_after_head_drop"]
    assert "--n-estimators" in args["comparators gates"] and "--seed-offset" in args["comparators gates"]
    assert "gates/preconditions.json" in P[0]["inputs"] and "json:gates/selection.json:apf" in P[0]["inputs"]
    P2 = {c["name"]: c for c in run_moves.build_plan(_ns(delta_th_default=0.3, law_x_default=8, comparator_jobs=3, n_jobs=2)) if c["move"] == 14}
    assert P2["comparators dhodapkar"]["args"][-1] == "0.3" and P2["comparators law"]["args"][-1] == "3"
    assert run_moves.main(["plan", "--out", "/x", "--moves", "14"]) == 0
    # the real run on the corpus
    out = _fresh(corpus)
    GP.gate_preconditions(out, assume_failed_zero=True, assume_reason="test", c1_activity_min=0.0)
    GP.gate_gk0(out)
    argv = ["run", "--out", str(out), "--moves", "14", "--null-perm", "5", "--null-splits", "loko"]
    if _driver_accepts("--n-estimators"):
        argv += ["--n-estimators", "10"]
    rc = run_moves.main(argv)
    L = jload(out / "driver_state.json")
    m14 = [c for c in L["commands"] if c["move"] == 14]
    assert rc == 0, [(c["name"], c["status"], c.get("stderr_tail")) for c in m14]
    assert len(m14) == 7 and all(c["status"] == "done" for c in m14), [(c["name"], c["status"]) for c in m14]
    t = rows(out / "report" / "tables" / "table7_comparators.csv")
    lk = [r for r in t if r["split"] == "LOKO"]
    assert len(lk) == 6 and all(r["G-M vs APF"] == "not run: gates/gm.params.json missing (move 12)" for r in lk), [r["G-M vs APF"] for r in lk]
    assert all(re.match(r"^[01](\.\d+)?$", r["accuracy"]) for r in lk), [r["accuracy"] for r in lk]
    assert "gates/comparators/verdicts.csv" in jload(out / "report" / "manifest.json")["files"]
    assert (out / "report" / "figures" / "fig_dhodapkar_sweep.pdf").is_file()
    # a second run skips every move-14 command (outputs exist, inputs unchanged)
    rc = run_moves.main(argv)
    L = jload(out / "driver_state.json")
    last = [c for c in L["commands"] if c["move"] == 14][-7:]
    assert rc == 0 and all(c["status"].startswith("skipped") for c in last), [(c["name"], c["status"]) for c in last]


# --------------------------------------------------------------------------- 10. CLI exit codes
def test_cli_exit_codes(corpus, capsys):
    empty = tmp_out()
    assert C.main(["savoldi", "--out", str(empty)]) == 2
    assert "cells.csv" in capsys.readouterr().err
    out = _fresh(corpus)
    assert C.main(["all", "--out", str(out), "--no-splits"]) == 0
    d = out / "gates" / "comparators"
    for f in ("savoldi.csv", "savoldi_per_kernel.csv", "dhodapkar.csv", "dhodapkar_sweep.csv", "dhodapkar_per_kernel.csv",
              "law.csv", "law_sweep.csv", "law_per_kernel.csv", "law_cells.json", "savoldi.params.json", "dhodapkar.params.json", "law.params.json"):
        assert (d / f).is_file(), f
    for name in schema.COMPARATOR_NAMES:
        for norm in (False, True):
            assert S.features_path(out, name, "Wall_Hall", norm).is_file()
    assert not (out / "gates" / "splits").exists()
    assert C.main(["law", "--out", str(out), "--x-grid", "2,8", "--x-default", "4", "--no-splits"]) == 2
    assert "is not on the grid" in capsys.readouterr().err
    assert C.main(["law", "--out", str(out), "--x-grid", "2,8", "--x-default", "4", "--off-grid-default", "append", "--no-splits"]) == 0
    prm = jload(d / "law.params.json")["params"]
    assert prm["x_grid"] == [2, 4, 8] and prm["default_appended_to_grid"] is True and prm["x_grid_source"] == "CLI --x-grid"
    # the per-kernel summaries are over admissible cells only
    GP.gate_preconditions(out, assume_failed_zero=True, assume_reason="test", c1_activity_min=0.0)
    pre = rows(out / "gates" / "preconditions.csv")
    for r in pre:
        if r["cell_id"].startswith("gibbs"):
            r["all_hard_pass"] = "false"
    S.write_csv(out / "gates" / "preconditions.csv", list(pre[0].keys()), pre)
    assert C.main(["savoldi", "--out", str(out), "--no-splits"]) == 0
    assert "gibbs" not in {r["kernel"] for r in rows(d / "savoldi_per_kernel.csv")}
    assert row_where(d / "savoldi.csv", kernel="gibbs", rep=0)["admissible"] == "false"


# --------------------------------------------------------------------------- 11. the chain on builder 1's pipeline
def _have_builder1():
    return (PKG / "synth.py").is_file() and (PKG / "extract.py").is_file()


@pytest.mark.skipif(not _have_builder1(), reason="builder 1's synth.py / extract.py not present")
def test_chain_with_law_on_builder1_pipeline():
    root = tmp_out()
    out = root / "out"
    env = {"PYTHONPATH": str(PKG.parent), "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"}
    cwd = str(PKG.parent)
    subprocess.run([PY, "-m", "plan11_encoding_ladder.synth", "corpus", "--root", str(root / "root"), "--n-pairs", "60",
                    "--reps", "2", "--idle", "2", "--jobs", "4"], check=True, cwd=cwd, env=env)
    subprocess.run([PY, "-m", "plan11_encoding_ladder.extract", "index", "--root", str(root / "root"), "--out", str(out)], check=True, cwd=cwd, env=env)
    subprocess.run([PY, "-m", "plan11_encoding_ladder.extract", "all", "--cells-csv", str(out / "cells.csv"), "--out", str(out), "--jobs", "4"],
                   check=True, cwd=cwd, env=env)
    GP.gate_preconditions(out, assume_failed_zero=True, assume_reason="AA A5", c1_activity_min=0.0)
    r = subprocess.run([PY, "-m", "plan11_encoding_ladder.comparators", "all", "--out", str(out), "--null-perm", "5", "--n-estimators", "10",
                        "--null-splits", "loko", "--jobs", "2"], cwd=cwd, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    d = out / "gates" / "comparators"
    for f in ("savoldi.csv", "savoldi.params.json", "savoldi_per_kernel.csv", "dhodapkar_sweep.csv", "dhodapkar.csv", "dhodapkar.params.json",
              "dhodapkar_per_kernel.csv", "law_sweep.csv", "law.csv", "law.params.json", "law_cells.json", "law_per_kernel.csv",
              "gl.csv", "gl.params.json", "gdim.csv", "gdim.params.json", "gm.csv", "gm.params.json", "verdicts.csv", "verdicts.params.json"):
        assert (d / f).is_file(), f
    cells = S.load_cells(out / "cells.csv")
    assert len(cells) == 26 and len(list((d / "law_series").glob("*.npz"))) == 26
    law = rows(d / "law.csv")
    assert len(law) == 26 and all(r["check_x2_equals_K"] == "true" and r["status"] == "ok" for r in law)
    assert len(rows(d / "law_sweep.csv")) == 26 * len(C.LAW_X_GRID) and len(rows(d / "dhodapkar_sweep.csv")) == 26 * len(C.DHODAPKAR_GRID)
    assert len(rows(d / "verdicts.csv")) == 30
    for name in schema.COMPARATOR_NAMES:
        for norm in (False, True):
            for split, ls in C.SPLIT_PLAN:
                assert (M.split_dir(out, name, "Wall_Hall", split, ls, normalized=norm) / "scores.json").is_file(), (name, norm, split, ls)
    assert row_where(out / "gates" / "gx.csv", rung="cmp_law")["leak_verdict"].startswith("not applicable: one campaign label")
    # no file under gates/comparators/ names a sandbox workload (the corpus has none); the kernels present are P2 Table 3's
    names = {r["kernel"] for r in rows(d / "savoldi.csv")}
    assert names <= set(schema.KERNEL_NAMES) | {"sleep"}
    idle = [r for r in rows(d / "savoldi.csv") if r["role"] == "idle"]
    assert len(idle) == 2 and row_where(d / "savoldi_per_kernel.csv", kernel="idle")["n_cells"] == "2"
