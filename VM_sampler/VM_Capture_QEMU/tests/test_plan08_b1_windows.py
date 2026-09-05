#!/usr/bin/env python3
"""Smoke + correctness test for plan08_b1/b1_windows.py.

Hand-builds trajectories whose window values, labels, rep order, and counts are
all known, then checks the emitted npz against them: window count = (T-W)//H+1,
X_apf/X_wapf row-aligned, exact window values (incl. 50% overlap), APF=K/N,
wAPF=ham_sum/(N*BITS_P), workload-prefix stripped, family + rep grouping, and
the min-pairs filter.

Run:  python3 tests/test_plan08_b1_windows.py     (plain asserts, no pytest dep)
      pytest tests/test_plan08_b1_windows.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

QEMU_DIR = Path(__file__).resolve().parent.parent
WIN = QEMU_DIR / "plan08_b1" / "b1_windows.py"
N = 10
BITS_P = 4096 * 8
W, H = 8, 4


def _make_cell(root: Path, name: str, source: str, T: int):
    """Trajectory with n_changed[t]=t+1, ham_sum[t]=(t+1)*1000 -> every window
    value is identifiable. Returns the apf/wapf series for cross-checking."""
    d = root / name
    d.mkdir(parents=True)
    (d / "provenance.json").write_text(json.dumps(
        {"workload": "IGNORED", "source": source, "n_pages": N,
         "page_size": 4096, "bits_per_page": BITS_P, "n_pairs": T}))
    apf, wapf = [], []
    with (d / "b1_trajectory.jsonl").open("w") as f:
        for t in range(T):
            k = t + 1
            hs = (t + 1) * 1000
            f.write(json.dumps({"seq": t, "n_pages": N, "n_changed": k, "ham_sum": hs}) + "\n")
            apf.append(k / N)
            wapf.append(hs / (N * BITS_P))
        f.write(json.dumps({"final": True, "n_pairs": T}) + "\n")
    return apf, wapf


def _run(root: Path, out: Path, min_pairs: int):
    cmd = [sys.executable, str(WIN), str(root), "--window", str(W), "--hop", str(H),
           "--min-pairs", str(min_pairs), "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"b1_windows failed:\n{r.stdout}\n{r.stderr}"


def test_windows(tmp_path=None):
    tmp = Path(tmp_path or tempfile.mkdtemp())
    root = tmp / "b1_out"; root.mkdir()
    # two reps of one workload (prefix to strip; run_index 5 then 9), one of another
    aA, wA = _make_cell(root, "cA", "run_matrix_test5_mem_writemag_sweep_v2.npy.substrate_trajectory.csv", 12)
    aB, wB = _make_cell(root, "cB", "run_matrix_test9_mem_writemag_sweep_v2.npy.substrate_trajectory.csv.zst", 12)
    aC, wC = _make_cell(root, "cC", "run_matrix_test3_cpu_hash_loop_v2.npy.substrate_trajectory.csv", 10)

    out = tmp / "b1_windows.npz"
    _run(root, out, min_pairs=10)
    d = np.load(out, allow_pickle=False)

    Xa, Xw = d["X_apf"], d["X_wapf"]
    # counts: T=12 -> (12-8)//4+1 = 2 windows; T=10 -> 1. total 5.
    assert Xa.shape == (5, W), Xa.shape
    assert Xw.shape == (5, W), Xw.shape
    for key in ("cell_id", "workload", "family", "rep", "win_start"):
        assert len(d[key]) == 5, (key, len(d[key]))
    assert int(d["window"]) == W and int(d["hop"]) == H

    # locate cA's two windows by cell_id, in win_start order
    rows = {}
    for i in range(5):
        rows.setdefault(str(d["cell_id"][i]), []).append(i)
    ia = sorted(rows["cA"], key=lambda i: d["win_start"][i])
    assert [int(d["win_start"][i]) for i in ia] == [0, 4], "hop/win_start wrong"

    # exact window values: window0 = series[0:8], window1 = series[4:12]
    assert np.allclose(Xa[ia[0]], aA[0:8]), (Xa[ia[0]], aA[0:8])
    assert np.allclose(Xa[ia[1]], aA[4:12])
    assert np.allclose(Xw[ia[0]], wA[0:8])
    assert np.allclose(Xw[ia[1]], wA[4:12])
    # 50% overlap: last 4 of window0 == first 4 of window1
    assert np.allclose(Xa[ia[0]][4:8], Xa[ia[1]][0:4]), "windows not overlapping by W-H"

    # APF/wAPF definitions hold on a concrete value: seq0 -> K=1, ham_sum=1000
    assert abs(Xa[ia[0]][0] - 1 / N) < 1e-12
    assert abs(Xw[ia[0]][0] - 1000 / (N * BITS_P)) < 1e-12

    # labels: prefix stripped, family mapped, rep ranked by run_index
    assert str(d["workload"][ia[0]]) == "mem_writemag_sweep_v2", d["workload"][ia[0]]
    assert str(d["family"][ia[0]]) == "mem"
    reps = {str(d["cell_id"][i]): int(d["rep"][i]) for i in range(5)}
    assert reps["cA"] == 1 and reps["cB"] == 2, reps        # test5 -> rep1, test9 -> rep2
    assert reps["cC"] == 1                                   # different workload -> its own rep1
    assert str(d["family"][rows["cC"][0]]) == "cpu"

    # min-pairs filter: raise above T=10 -> cell cC drops, cpu family gone
    out2 = tmp / "b1_windows_strict.npz"
    _run(root, out2, min_pairs=11)
    d2 = np.load(out2, allow_pickle=False)
    assert d2["X_apf"].shape == (4, W), d2["X_apf"].shape   # only the two T=12 cells
    assert "cpu" not in set(d2["family"].tolist())

    print("PASS: window counts, exact values, overlap, APF/wAPF defs, labels/reps, min-pairs all OK")


def test_family_of_prefix():
    """Regression: family is the first token, not a substring. io_read_cache_hit
    must be io, never cache."""
    sys.path.insert(0, str(QEMU_DIR / "plan08_b1"))
    from b1_features import family_of
    assert family_of("io_read_cache_hit_v2") == "io", "io_read_cache_hit misfiled"
    assert family_of("io_direct_write_like_v2") == "io"
    assert family_of("cache_hot_loop_v2") == "cache"
    assert family_of("cache_cold_scan_v2") == "cache"
    assert family_of("mem_random_write_pages_v2") == "mem"
    assert family_of("sandbox_stealth_paced") == "sandbox"
    assert family_of("cpu_hash_loop_v2") == "cpu"
    print("PASS: family_of is prefix-based (io_read_cache_hit -> io, not cache)")


if __name__ == "__main__":
    test_family_of_prefix()
    test_windows()
