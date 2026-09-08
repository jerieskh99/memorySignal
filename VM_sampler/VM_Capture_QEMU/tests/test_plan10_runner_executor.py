#!/usr/bin/env python3
"""executor.py: the three examples run end to end on a synthetic corpus and write what they promise.

Also checks the stages on known inputs: Collapse in changed_fraction mode reproduces the
fixture's APF exactly, Window tiles have the arithmetic the console shows, and a stop
written to control.json is honoured.

Needs the differ binary and zstd; skips with a message otherwise.

Run:  python3 tests/test_plan10_runner_executor.py
      pytest tests/test_plan10_runner_executor.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

QEMU_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QEMU_DIR))
from plan10_analysis import corpus_manifest, scheme as S       # noqa: E402
from plan10_analysis.runner import chain, differ, executor, stages  # noqa: E402
from plan10_analysis.testing import synth                        # noqa: E402


def _have_tools() -> bool:
    try:
        differ.find_differ()
    except differ.DifferError as e:
        print(f"skip: {e}")
        return False
    if not chain.zstd_available():
        print("skip: zstd not on PATH")
        return False
    return True


def _corpus(td: Path, n_snapshots=10):
    root = td / "corpus"
    changes = synth.make_corpus(root, n_snapshots=n_snapshots)
    manifest = corpus_manifest.scan(root)
    mp = td / "manifest.json"
    mp.write_text(json.dumps(manifest))
    return root, manifest, mp, changes


def _example(manifest, k, **cells_over):
    s = S.make_examples(manifest)[k]
    cells = next(n for n in s["nodes"] if n["module"] == "cells")
    cells["params"].update(min_pairs=1, **cells_over)
    cells["params"]["sel"] = [r["id"] for r in manifest["recordings"]]
    s["acknowledged"] = [{"id": "no_substrate", "at": "test", "note": "synthetic chains"}]
    return s


def test_stages_on_known_inputs():
    seq = np.array([1, 1, 1, 2, 2, 3], dtype=np.int32)
    field = {"seq": seq, "page_index": np.array([3, 4, 5, 3, 9, 7], dtype=np.int32),
             "cols": {"hamming": np.array([8, 16, 4, 2, 2, 100], dtype=np.float32)}, "z": None,
             "channels": ["hamming"], "n_pairs": 3, "n_pages": 100, "block": None}
    s = stages.collapse(field, "zero", "changed_fraction")
    assert np.allclose(s["values"], [3 / 100, 2 / 100, 1 / 100]) and s["channels"] == ["changed_fraction"]
    m = stages.collapse(field, "zero", "mean")
    assert np.allclose(m["values"], [28 / 100, 4 / 100, 100 / 100])
    e = stages.collapse(field, "excluded", "mean")
    assert np.allclose(e["values"], [28 / 3, 4 / 2, 100 / 1])
    t = stages.window({"values": np.arange(10, dtype=np.float32), "channels": ["x"], "complex": False, "block": None, "n_pages": 1}, 4, 2)
    assert t["X"].shape == (4, 4) and t["keys"][0] == (None, 0, 1) and t["keys"][-1] == (None, 3, 7)
    tz = stages.window({"values": np.arange(10, dtype=np.float32), "channels": ["x"], "complex": False, "block": None, "n_pages": 1}, 4, 4, edge="zero")
    assert tz["X"].shape == (3, 4) and tz["X"][-1].tolist() == [8, 9, 0, 0]
    f = stages.stats(t, ["mean", "max", "duty"])
    assert f["names"] == ["mean", "max", "duty"] and f["rows"].shape == (4, 3) and f["rows"][0, 1] == 3
    # complex on matching rows; phase conventions
    dirf = dict(field, cols={"cosine": np.array([0, 1, 0.5, 0, 1, 0.5], dtype=np.float32)}, channels=["cosine"])
    z = stages.complex_field(field, dirf, "pi")
    assert z["z"] is not None and np.isclose(np.angle(z["z"][1]), np.pi)
    z2 = stages.complex_field(field, dirf, "2pi")
    assert np.isclose(np.angle(z2["z"][1]), 0.0, atol=1e-6)   # the collision: distance 1 lands on angle 0


def test_b1_example_end_to_end_and_apf_equals_fixture():
    if not _have_tools():
        return
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        root, manifest, mp, changes = _corpus(td, n_snapshots=10)
        s = _example(manifest, "b1")
        win = next(n for n in s["nodes"] if n["module"] == "window")
        win["params"].update(w=4, h=2)
        sp = td / "b1.json"
        sp.write_text(json.dumps(s))
        rc = executor.run(sp, td / "out", {"kind": "local", "root": str(root)}, td / "l1", speed=2, manifest_path=mp)
        assert rc == 0, (td / "out" / s["label"] / "run.log").read_text()
        out = td / "out" / s["label"]
        st = json.loads((out / "status.json").read_text())
        assert st["state"] == "done" and st["n_recordings"] == 3
        z = np.load(out / "features.npz")
        names = z["feature_names"].tolist()
        assert names == ["mean", "std", "cov", "median", "max", "p95", "peak2med", "duty"]
        n_windows_per_rec = (9 - 4) // 2 + 1
        assert z["X"].shape == (3 * n_windows_per_rec, 8)
        keys = z["tile_keys"]
        assert set(keys["workload"].tolist()) == {"mem_synth_a_v2", "cpu_synth_b_v2"}
        # APF from the fixture: 7 changed pages per pair -> 7/1024, constant, so mean == max == 7/1024
        assert np.allclose(z["X"][:, 0], 7 / 1024) and np.allclose(z["X"][:, 4], 7 / 1024)
        side = json.loads((out / "sidecar.json").read_text())
        assert side["acknowledged"][0]["id"] == "no_substrate" and side["speed"] == 2 and side["n_rows"] == z["X"].shape[0]
        assert side["source"]["kind"] == "local" and side["differ"]["path"].endswith("live_delta_calc_modular")
        assert (out / "features.csv").read_text().splitlines()[0].startswith("recording,workload,family,block,t_index,seq_start,mean")


def test_complex_and_plv_examples_end_to_end():
    if not _have_tools():
        return
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        root, manifest, mp, _ = _corpus(td, n_snapshots=12)
        # complex -> fft + cepstrum -> concat
        s = _example(manifest, "complex")
        next(n for n in s["nodes"] if n["module"] == "window")["params"].update(w=8, h=4)
        sp = td / "cx.json"
        sp.write_text(json.dumps(s))
        rc = executor.run(sp, td / "out", {"kind": "local", "root": str(root)}, td / "l1", speed=2, manifest_path=mp)
        assert rc == 0, (td / "out" / s["label"] / "run.log").read_text()
        z = np.load(td / "out" / s["label"] / "features.npz")
        names = z["feature_names"].tolist()
        assert names == ["fft_band0", "fft_band1", "fft_band2", "fft_band3", "cepstral_peak_idx", "ceps_peak_snr_db"], names
        assert z["X"].shape[0] == 3 * ((11 - 8) // 4 + 1)
        # plv with a baseline fitted on the first recording (L1 store reused from the run above)
        s = _example(manifest, "plv")
        next(n for n in s["nodes"] if n["module"] == "window")["params"].update(w=8, h=4)
        sp = td / "plv.json"
        sp.write_text(json.dumps(s))
        rc = executor.run(sp, td / "out", {"kind": "local", "root": str(root)}, td / "l1", speed=2, manifest_path=mp)
        assert rc == 0, (td / "out" / s["label"] / "run.log").read_text()
        z = np.load(td / "out" / s["label"] / "features.npz")
        assert all(n.startswith("plv_num_") for n in z["feature_names"].tolist()) and z["X"].shape[0] == 3
        log = (td / "out" / s["label"] / "run.log").read_text()
        assert "L1 store reused" in log and "baseline fitted on" in log


def test_refusal_and_stop():
    if not _have_tools():
        return
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        root, manifest, mp, _ = _corpus(td, n_snapshots=6)
        s = _example(manifest, "b1")
        next(n for n in s["nodes"] if n["module"] == "window")["params"].update(w=4, h=2)
        s["acknowledged"] = []                      # unacknowledged warning -> refused, exit 2
        sp = td / "b1.json"
        sp.write_text(json.dumps(s))
        rc = executor.run(sp, td / "out", {"kind": "local", "root": str(root)}, td / "l1", speed=2, manifest_path=mp)
        assert rc == 2
        st = json.loads((td / "out" / s["label"] / "status.json").read_text())
        assert st["state"] == "refused"
        # --acknowledge-all lets it through; a stop written while extracting is honoured
        out = td / "out2" / s["label"]
        out.mkdir(parents=True)
        (out / "control.json").write_text(json.dumps({"command": "stop"}))
        rc = executor.run(sp, td / "out2", {"kind": "local", "root": str(root)}, td / "l1b", speed=2, manifest_path=mp, acknowledge_all=True)
        assert rc == 130 and json.loads((out / "status.json").read_text())["state"] == "stopped"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
