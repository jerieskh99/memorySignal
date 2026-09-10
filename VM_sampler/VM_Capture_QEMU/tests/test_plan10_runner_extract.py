#!/usr/bin/env python3
"""chain.py, differ.py, extract.py on a synthetic corpus with known answers.

The load-bearing checks: a chain reconstructs byte-exact through the rolling window; the
differ's hamming on a pair equals popcount(xor) page by page; the changed-page count it
reports equals the fixture's; and APF derived from the rows equals
plan02_apf_helper._compute_active_page_fraction on the same pair, exactly.

Needs the differ binary (PLAN10_DIFFER or the release build) and the zstd CLI; skips
with a message otherwise.

Run:  python3 tests/test_plan10_runner_extract.py
      pytest tests/test_plan10_runner_extract.py
"""
from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np

QEMU_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QEMU_DIR))
import plan02_apf_helper as helper                       # noqa: E402
from plan10_analysis.runner import chain, differ, extract  # noqa: E402
from plan10_analysis.testing import synth                 # noqa: E402

PAGE = 4096


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


def test_chain_reconstructs_byte_exact():
    if not _have_tools():
        return
    with tempfile.TemporaryDirectory() as td:
        rec = Path(td) / "rec"
        snaps = [synth.base_dump(7)]
        for k in range(1, 4):
            snaps.append(synth.next_snapshot(snaps[-1], k, 7)[0])
        synth.write_chain(rec, snaps)
        assert [p.name for p in chain.chain_files(rec)] == ["000000.zst", "000001.zst", "000002.zst", "000003.zst"]
        got = []
        for seq, prev, curr in chain.walk_chain(rec, Path(td) / "work"):
            got.append((seq, hashlib.sha256(prev.read_bytes()).hexdigest(), hashlib.sha256(curr.read_bytes()).hexdigest()))
        want = [hashlib.sha256(s.tobytes()).hexdigest() for s in snaps]
        assert [g[0] for g in got] == [1, 2, 3]
        assert [g[1] for g in got] == want[:3] and [g[2] for g in got] == want[1:]
        assert not list((Path(td) / "work").glob("*.raw"))
        # start_seq and max_pairs
        assert [s for s, _, _ in chain.walk_chain(rec, Path(td) / "w2", start_seq=2)] == [2, 3]
        assert [s for s, _, _ in chain.walk_chain(rec, Path(td) / "w3", max_pairs=1)] == [1]


def test_differ_hamming_equals_popcount_and_apf_equals_helper():
    if not _have_tools():
        return
    with tempfile.TemporaryDirectory() as td:
        a = synth.base_dump(3)
        b, changed = synth.next_snapshot(a, 1, 3)
        pa, pb = Path(td) / "a.raw", Path(td) / "b.raw"
        pa.write_bytes(a.tobytes())
        pb.write_bytes(b.tobytes())
        rows = differ.diff_pair(pa, pb, 2, ["hamming", "cosine"], Path(td) / "w")
        assert set(rows["page_index"].tolist()) == changed
        xor = np.bitwise_xor(a, b).reshape(-1, PAGE)
        pop = np.unpackbits(xor, axis=1).sum(axis=1)
        for p, h in zip(rows["page_index"], rows["hamming"]):
            assert int(h) == int(pop[p]), (p, h, pop[p])
        assert np.all(rows["cosine"] >= 0)
        apf_rows = len(rows["page_index"]) / synth.N_PAGES
        apf_helper = helper._compute_active_page_fraction(pa, pb, PAGE)
        assert abs(apf_rows - apf_helper) < 1e-12, (apf_rows, apf_helper)
        # the 4 MiB guard
        small = Path(td) / "small.raw"
        small.write_bytes(a.tobytes()[: 64 * PAGE])
        try:
            differ.check_dump_size(small)
            assert False, "must refuse a 256 KB dump"
        except differ.DifferError:
            pass


def test_extract_store_and_reuse():
    if not _have_tools():
        return
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "corpus"
        changes = synth.make_corpus(root, n_snapshots=4, workloads=(("mem", "mem_synth_a_v2", 1),))
        rec_id = next(iter(changes))
        store = Path(td) / "l1"
        seen = []
        p = extract.extract(rec_id, root / rec_id, 2, ["hamming"], store, progress=lambda d, n: seen.append((d, n)))
        assert seen == [(1, 3), (2, 3), (3, 3)]
        d = extract.load(p)
        assert d["n_pairs"] == 3 and d["n_pages"] == synth.N_PAGES
        for seq in (1, 2, 3):
            got = set(d["page_index"][d["seq"] == seq].tolist())
            assert got == changes[rec_id][seq], (seq, got, changes[rec_id][seq])
        # reuse: a superset store at the same speed is found; a different speed is not
        assert extract.existing(store, rec_id, 2, ["hamming"], None) == p
        assert extract.existing(store, rec_id, 0, ["hamming"], None) is None
        assert extract.existing(store, rec_id, 2, ["cosine"], None) is None
        p2 = extract.extract(rec_id, root / rec_id, 2, ["hamming"], store)
        assert p2 == p
        # max_pairs is part of the key
        p3 = extract.extract(rec_id, root / rec_id, 2, ["hamming", "cosine"], store, max_pairs=2)
        assert p3 != p and extract.load(p3)["n_pairs"] == 2


def test_extract_from_trajectory_matches_the_rows():
    """The capture's own rows become an L1 store shaped like extract()'s: seq counted from 1
    the way walk_chain numbers pairs, requested columns kept, max_pairs honoured, reused by
    existing(), and a missing column refused rather than filled."""
    import json
    import numpy as np
    from plan10_analysis.runner import trajectory
    with tempfile.TemporaryDirectory() as td:
        csv = Path(td) / "run_matrix_test1_wl.npy.substrate_trajectory.csv"
        csv.write_text("seq,page_index,hamming,cosine\n"
                       "0,5,255,0.5\n0,9,11,0.1\n"
                       "1,5,2,0.9\n"
                       "2,7,7,0.2\n2,9,3,0.3\n")
        assert trajectory.columns(csv) == ["hamming", "cosine"]
        store = Path(td) / "l1"
        rid = "mem/wl/var/rep001__x"
        seen = []
        p = extract.extract_from_trajectory(rid, csv, 2, ["hamming"], store, n_pages=100,
                                            progress=lambda d, n: seen.append(d))
        d = extract.load(p)
        assert seen == [1, 2, 3] and d["n_pairs"] == 3 and d["n_pages"] == 100
        assert d["seq"].tolist() == [1, 1, 2, 3, 3]                       # 0-based file, 1-based store
        assert d["page_index"].tolist() == [5, 9, 5, 7, 9]
        assert d["hamming"].dtype == np.float32 and d["hamming"].tolist() == [255, 11, 2, 7, 3]
        assert "cosine" not in d
        meta = json.loads(p.with_name(p.name.replace(".npz", ".meta.json")).read_text())
        assert meta["source"] == "substrate_csv" and meta["speed_assumed"] and meta["complete"]
        assert extract.existing(store, rid, 2, ["hamming"], None) == p
        p2 = extract.extract_from_trajectory(rid, csv, 2, ["hamming", "cosine"], store, max_pairs=2, n_pages=100)
        d2 = extract.load(p2)
        assert p2 != p and d2["n_pairs"] == 2 and d2["seq"].tolist() == [1, 1, 2] and d2["cosine"].tolist()[0] == np.float32(0.5)
        # n_pages never claims fewer pages than the data addresses
        p3 = extract.extract_from_trajectory(rid, csv, 3, ["hamming"], store, n_pages=4)
        assert extract.load(p3)["n_pages"] == 10
        try:
            trajectory.read(csv, ["nope"])
            assert False, "a column the file lacks must be refused"
        except trajectory.TrajectoryError:
            pass


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
