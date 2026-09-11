#!/usr/bin/env python3
"""results_view.py: every view returns the numbers numpy would, over one run and over several.

Builds two synthetic runs with known values (no differ, no corpus), then checks the summary,
each of the five views, the multi-run frame (feature alignment, the `run` key, exclusion of a
run lacking the metric), the natural ordering of groups, the 6-significant-digit rounding,
and the refusals.

Run:  python3 tests/test_plan10_results_view.py
      pytest tests/test_plan10_results_view.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

QEMU_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QEMU_DIR))
from plan10_analysis import results_view as RV   # noqa: E402

KEY_DT = np.dtype([("recording", "U256"), ("workload", "U128"), ("family", "U32"), ("block", "i4"), ("t_index", "i4"), ("seq_start", "i4")])


def _write_run(d: Path, label: str, names, rows, keys, sidecar_extra=None):
    d.mkdir(parents=True)
    X = np.asarray(rows, dtype=np.float32)
    tk = np.array(keys, dtype=KEY_DT)
    np.savez_compressed(d / "features.npz", X=X, feature_names=np.array(names, dtype="U64"), tile_keys=tk)
    side = {"schema": "plan10.sidecar.v1", "label": label, "written_at": "2026-09-11T00:00:00+00:00", "speed": 2,
            "max_pairs": None, "acknowledged": [{"id": "no_substrate", "at": "t"}], "source": {"kind": "local", "root": "/r"},
            "scheme": {"nodes": [{"module": "cells"}, {"module": "window", "params": {"w": 8, "h": 4, "edge": "drop"}}, {"module": "write"}]},
            "recordings": [{"id": k[0]} for k in {k[0]: None for k in keys}], "n_rows": len(rows), "n_features": len(names),
            "format": "npz+csv"}
    side.update(sidecar_extra or {})
    (d / "sidecar.json").write_text(json.dumps(side))
    (d / "scheme.json").write_text("{}")


def _rec(fam, wl, seed, camp="exp-1"):
    return f"{fam}/{wl}/args_--seed_{seed}_deadbeef/rep001__{camp}"


def make_runs(td: Path):
    # run A: two families, a recording each with 4 windows; a third recording with a NaN
    names = ["mean", "std", "duty"]
    rows, keys = [], []
    for t in range(4):
        rows.append([1.0 + t, 0.5, 0.25 * (t + 1)]); keys.append((_rec("cpu", "cpu_wl", 42), "cpu_wl", "cpu", -1, t, 1 + 4 * t))
    for t in range(4):
        rows.append([10.0 + t, 5.0, 1.0]); keys.append((_rec("mem", "mem_wl", 7), "mem_wl", "mem", -1, t, 1 + 4 * t))
    rows.append([float("nan"), 2.0, 0.5]); keys.append((_rec("mem", "mem_wl", 8), "mem_wl", "mem", -1, 0, 1))
    _write_run(td / "run_a", "run_a", names, rows, keys)
    # run B: shares mean and duty, lacks std, has an extra metric; page-resolution blocks
    names_b = ["mean", "duty", "extra"]
    rows, keys = [], []
    for blk in range(2):
        for t in range(3):
            rows.append([100.0 + 10 * blk + t, 0.5, 3.0]); keys.append((_rec("cpu", "cpu_wl", 42, "exp-2"), "cpu_wl", "cpu", blk, t, 1 + 4 * t))
    _write_run(td / "run_b", "run_b", names_b, rows, keys)
    return RV.get_run(td / "run_a"), RV.get_run(td / "run_b")


def test_summary_and_rounding():
    with tempfile.TemporaryDirectory() as td:
        a, _b = make_runs(Path(td))
        s = RV.summary(a)
        assert s["n_rows"] == 9 and s["n_features"] == 3
        mean = next(f for f in s["features"] if f["name"] == "mean")
        assert mean["n"] == 8 and mean["n_nan"] == 1
        vals = [1, 2, 3, 4, 10, 11, 12, 13]
        assert mean["median"] == float(np.median(vals)) and mean["min"] == 1 and mean["max"] == 13
        assert mean["std"] == float(f"{np.std(vals, ddof=1):.6g}")
        assert mean["p25"] == float(f"{np.percentile(vals, 25):.6g}")
        assert s["keys"]["family"]["n_unique"] == 2 and s["keys"]["recording"]["n_unique"] == 3
        assert s["keys"]["t_index"] == {"n_unique": 4, "min": 0, "max": 3}
        rec = s["keys"]["recording"]["values"]
        assert all(r["short"].startswith("seed_") for r in rec) and len({r["short"] for r in rec}) == 3
        assert s["facts"]["window"] == {"w": 8, "h": 4, "edge": "drop"} and s["facts"]["acknowledged"] == ["no_substrate"]
        assert s["facts"]["modules"] == ["cells", "window", "write"] and s["facts"]["extraction_ran"] == "local"
        # 6 significant digits, never more
        assert RV._r(0.123456789) == 0.123457 and RV._r(1234567.89) == 1234570.0 and RV._r(float("nan")) is None
        assert RV._r(np.int32(7)) == 7 and isinstance(RV._r(np.int32(7)), int)


def test_distribution_groups_and_histogram():
    with tempfile.TemporaryDirectory() as td:
        a, _b = make_runs(Path(td))
        d = RV.aggregate([a], "distribution", y="mean", group=["family"], bins=4)
        assert [g["label"] for g in d["groups"]] == ["cpu", "mem"]
        cpu, mem = d["groups"]
        assert cpu["n"] == 4 and cpu["median"] == 2.5 and cpu["mean"] == 2.5 and cpu["min"] == 1 and cpu["max"] == 4
        assert mem["n"] == 4 and mem["n_nan"] == 1 and mem["median"] == 11.5
        assert len(d["edges"]) == 5 and d["edges"][0] == 1 and d["edges"][-1] == 13
        assert sum(cpu["counts"]) == 4 and sum(mem["counts"]) == 4
        assert cpu["counts"] == [3, 1, 0, 0] and mem["counts"] == [0, 0, 0, 4]     # edges 1,4,7,10,13: 4 falls in [4,7), the last bin is closed
        # no group: one row named all; recording groups carry the short label and the full id in key
        assert RV.aggregate([a], "distribution", y="mean")["groups"][0]["label"] == "all"
        r = RV.aggregate([a], "distribution", y="mean", group=["recording"])["groups"]
        assert r[0]["label"].startswith("seed_42") and r[0]["key"]["recording"].startswith("cpu/cpu_wl/")
        # log scale: non-positive excluded from the histogram and counted
        rows = [[0.0, 1, 1], [1.0, 1, 1], [10.0, 1, 1]]
        keys = [(_rec("cpu", "w", 1), "w", "cpu", -1, t, t) for t in range(3)]
        _write_run(Path(td) / "run_c", "run_c", ["mean", "std", "duty"], rows, keys)
        c = RV.aggregate([RV.get_run(Path(td) / "run_c")], "distribution", y="mean", scale="log", bins=2)
        assert c["n_excluded_nonpositive"] == 1 and c["edges"][0] == 1 and c["edges"][-1] == 10 and c["groups"][0]["counts"] == [1, 1]
        assert c["groups"][0]["min"] == 0                       # the stats still see every finite value


def test_time_series_aggregates_blocks():
    with tempfile.TemporaryDirectory() as td:
        a, b = make_runs(Path(td))
        t = RV.aggregate([a], "time", y="mean", group=["recording"], stat="median")
        assert len(t["series"]) == 3 and t["x"] == "t_index"
        s0 = t["series"][0]
        assert s0["x"] == [0, 1, 2, 3] and s0["y"] == [1, 2, 3, 4] and s0["n"] == [1, 1, 1, 1] and s0["lo"] == [None] * 4
        # run B has two blocks per t_index: the stat pools them and the band is p25-p75
        tb = RV.aggregate([b], "time", y="mean", group=["recording"], stat="mean")
        s = tb["series"][0]
        assert s["x"] == [0, 1, 2] and s["y"] == [105, 106, 107] and s["n"] == [2, 2, 2] and s["lo"] == [102.5, 103.5, 104.5]
        # grouping by workload pools the NaN recording in: n counts finite values only
        tw = RV.aggregate([a], "time", y="mean", group=["workload"], stat="max")
        mem = next(x for x in tw["series"] if x["label"] == "mem_wl")
        assert mem["n"] == [1, 1, 1, 1] and mem["y"] == [10, 11, 12, 13]
        # seq_start is the other x; anything else is refused
        assert RV.aggregate([a], "time", y="mean", x="seq_start")["series"][0]["x"] == [1, 5, 9, 13]
        try:
            RV.aggregate([a], "time", y="mean", x="family"); assert False
        except RV.ViewError:
            pass


def test_matrix_scatter_table():
    with tempfile.TemporaryDirectory() as td:
        a, b = make_runs(Path(td))
        m = RV.aggregate([a], "matrix", y="mean", rows="family", cols="t_index", stat="median")
        assert m["row_labels"] == ["cpu", "mem"] and m["col_labels"] == ["0", "1", "2", "3"]
        assert m["values"] == [[1, 2, 3, 4], [10, 11, 12, 13]] and m["n"][1][0] == 1 and m["n_empty"] == 0
        sc = RV.aggregate([a], "scatter", x="mean", y="duty", group=["family"])
        cpu = sc["groups"][0]
        assert cpu["n"] == 4 and cpu["n_shown"] == 4 and cpu["r_pearson"] == 1 and cpu["rho_spearman"] == 1
        mem = sc["groups"][1]
        assert mem["n"] == 4 and mem["r_pearson"] is None                  # duty is constant there
        assert sc["n"] == 8 and sc["stride"] == 1 and cpu["points"][0] == [1, 0.25]
        # a cap subsamples by a fixed stride and says so
        big = RV.scatter(RV.make_frame([a]), "mean", "duty", [], max_points=3)
        assert big["stride"] == 3 and big["groups"][0]["n_shown"] == 3 and big["groups"][0]["n"] == 8
        tb = RV.aggregate([a], "table", group=["family"], stat="mean")
        assert tb["features"] == ["mean", "std", "duty"] and tb["rows"][0]["values"] == [2.5, 0.5, 0.625] and tb["rows"][1]["n"] == 5
        assert RV.aggregate([a], "table", stat="count")["rows"][0]["values"] == [8, 9, 9]


def test_several_runs_align_by_name():
    with tempfile.TemporaryDirectory() as td:
        a, b = make_runs(Path(td))
        d = RV.aggregate([a, b], "distribution", y="mean", group=["run"])
        assert d["features"] == ["mean", "duty"] and d["runs"] == ["run_a", "run_b"] and d["excluded"] == []
        assert [g["label"] for g in d["groups"]] == ["run_a", "run_b"]      # the order given, not lexical
        assert d["groups"][1]["n"] == 6 and d["groups"][1]["median"] == 106
        d2 = RV.aggregate([b, a], "distribution", y="mean", group=["run", "family"])
        assert [g["label"] for g in d2["groups"]] == ["run_b / cpu", "run_a / cpu", "run_a / mem"]
        assert d2["groups"][0]["key"] == {"run": "run_b", "family": "cpu"}
        # a run lacking the metric is excluded and named, not silently dropped
        e = RV.aggregate([a, b], "distribution", y="std", group=["run"])
        assert e["runs"] == ["run_a"] and e["excluded"] == [{"label": "run_b", "missing": ["std"]}]
        # a matrix of run x family, a table per run over the shared metrics
        m = RV.aggregate([a, b], "matrix", y="mean", cols="family", stat="count")
        assert m["rows"] == "run" and m["values"] == [[4, 4], [6, 0]] and m["n_empty"] == 1
        t = RV.aggregate([a, b], "table", group=["run"], stat="max")
        assert t["features"] == ["mean", "duty"] and t["rows"][1]["values"] == [112, 0.5]
        # defaults are echoed: no y means the first shared metric; scatter takes the first two
        assert RV.aggregate([a, b], "distribution")["y"] == "mean"
        s = RV.aggregate([a, b], "scatter")
        assert (s["x"], s["y"]) == ("mean", "duty")


def test_refusals_and_cache():
    with tempfile.TemporaryDirectory() as td:
        a, _b = make_runs(Path(td))
        for kw in (dict(view="nope"), dict(view="distribution", y="nope"), dict(view="distribution", group=["nope"]),
                   dict(view="table", stat="nope"), dict(view="distribution", scale="sqrt")):
            try:
                RV.aggregate([a], **kw); assert False, kw
            except RV.ViewError as e:
                assert str(e)
        try:
            RV.load_run(Path(td) / "missing"); assert False
        except RV.ViewError as e:
            assert "features.npz" in str(e)
        # the cache serves the same object until the file changes
        assert RV.get_run(Path(td) / "run_a") is a
        (Path(td) / "run_a" / "sidecar.json").write_text(json.dumps({"written_at": "later"}))
        import os
        os.utime(Path(td) / "run_a" / "sidecar.json", (1, 1))
        assert RV.get_run(Path(td) / "run_a") is not a
        # short names: a collision is broken by the variant hash
        ids = ["cpu/w/a_--seed_1_aaaaaaaa/rep001__x", "cpu/w/a_--seed_1_bbbbbbbb/rep001__x", "io/w/hash_cccccccc/rep001__y"]
        sn = RV.short_names(ids)
        assert sn[ids[0]] == "seed_1 x aaaaaaaa" and sn[ids[1]] == "seed_1 x bbbbbbbb" and sn[ids[2]] == "cccccccc y"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
