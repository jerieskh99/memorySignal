#!/usr/bin/env python3
"""corpus_manifest.py: the migrated trace tree, scanned; nothing invented.

Builds a synthetic tree in a temp dir shaped like the real one
(family/workload/variant/repNNN__label/NNNNNN.zst) and checks counts, statuses,
the unrecorded fields, the substrate-CSV join by workload name, and that a
missing or empty root is refused rather than filled.

Run:  python3 tests/test_plan10_corpus_manifest.py
      pytest tests/test_plan10_corpus_manifest.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

QEMU_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QEMU_DIR))
from plan10_analysis import corpus_manifest as cm  # noqa: E402


def _tree(root: Path) -> None:
    def rep(path: Path, n: int, partial: bool = False):
        path.mkdir(parents=True)
        for i in range(n):
            (path / f"{i:06d}.zst").write_bytes(b"x" * (1000 if i == 0 else 10))
        if partial:
            (path / ".000003.zst.abc123").write_bytes(b"p")
    rep(root / "mem" / "mem_alpha_v2" / "working-set-mb_256_--duration_300_--seed_db1ee788" / "rep001__runA", 11, partial=True)
    rep(root / "mem" / "mem_alpha_v2" / "working-set-mb_256_--duration_450_--seed_4b530e9e" / "rep002__runA", 6)
    rep(root / "cpu" / "cpu_beta_v2" / "duration_450_--seed_42_--phase-markers" / "rep001__runB", 1)     # base only
    rep(root / "cpu" / "cpu_beta_v2" / "duration_600_--seed_43_--phase-markers" / "rep001__runB", 0)     # empty
    (root / "io" / "io_gamma_v2" / "size_1_--duration_60" / "notarep").mkdir(parents=True)


def test_scan_counts_statuses_and_unrecorded_fields():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "zstd_local"
        _tree(root)
        m = cm.scan(root)
        assert m["schema"] == cm.SCHEMA
        assert m["n_recordings"] == 4
        assert m["n_with_chain"] == 2
        assert m["n_with_substrate_csv"] == 0
        assert m["n_workloads"] == 2
        assert m["families"] == ["cpu", "mem"]
        by = {r["id"]: r for r in m["recordings"]}
        a = by["mem/mem_alpha_v2/working-set-mb_256_--duration_300_--seed_db1ee788/rep001__runA"]
        assert a["n_snapshots"] == 11 and a["n_pairs"] == 10 and a["status"] == "ok" and a["has"]["chain"]
        assert a["variant"]["duration_s"] == 300 and a["variant"]["seed"] == "db1ee788" and a["rep"] == 1 and a["run_label"] == "runA"
        assert a["base_bytes"] == 1000 and a["bytes"] == 1100
        assert a["speed"] is None and "unrecorded" in a["speed_source"]
        assert a["iv_ms"] is None and "unrecorded" in a["iv_source"]
        assert a["n_pages"] is None and "unrecorded" in a["n_pages_source"]
        b = [r for r in m["recordings"] if r["n_snapshots"] == 1][0]
        assert b["status"] == "base_only" and not b["has"]["chain"] and b["has"]["base_only"]
        e = [r for r in m["recordings"] if r["n_snapshots"] == 0][0]
        assert e["status"] == "empty" and e["n_pairs"] == 0
        assert len(m["partial_files"]) == 1 and m["partial_files"][0].endswith(".000003.zst.abc123")
        assert any("not a rep dir" in w for w in m["warnings"])


def test_substrate_join_by_workload_name():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "zstd_local"
        _tree(root)
        mroot = Path(td) / "metrics"
        (mroot / "x").mkdir(parents=True)
        (mroot / "x" / "run_matrix_test7_mem_alpha_v2.substrate_trajectory.csv.zst").write_bytes(b"z")
        (mroot / "x" / "hc_field.csv.zst").write_bytes(b"z")
        m = cm.scan(root, mroot)
        assert m["n_with_substrate_csv"] == 2          # both mem_alpha_v2 recordings, joined by name
        r = [r for r in m["recordings"] if r["workload"] == "mem_alpha_v2"][0]
        assert r["has"]["substrate_csv"] and r["has"]["substrate_join"] == "workload-name"
        assert m["unjoined_metrics_artifacts"]["hc_field"] == 1
        assert cm._workload_from_metrics_name("run_matrix_test3_mem_x_v2.npy.substrate_trajectory.csv.gz") == "mem_x_v2"


def test_missing_and_empty_roots_are_refused():
    with tempfile.TemporaryDirectory() as td:
        try:
            cm.scan(Path(td) / "nope")
            assert False, "missing root must raise"
        except cm.CorpusMissing:
            pass
        empty = Path(td) / "empty"
        empty.mkdir()
        try:
            cm.scan(empty)
            assert False, "empty root must raise"
        except cm.CorpusMissing:
            pass
        r = subprocess.run([sys.executable, str(QEMU_DIR / "plan10_analysis" / "corpus_manifest.py"), "--root", str(empty)],
                           capture_output=True, text=True)
        assert r.returncode == 2 and "REFUSED" in r.stderr


def test_default_root_comes_from_console_sh():
    p = cm.default_root()
    assert str(p).endswith("thesis_traces/zstd_local"), p


def test_substrate_in_chain_is_seen_from_the_listing():
    """A trajectory the capture left beside its chain needs no metrics root: the listing
    carries it, for a remote source too. It joins as that recording's own, not by name."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "zstd_local"
        _tree(root)
        m0 = cm.scan(root)
        rec = next(r for r in m0["recordings"] if r["has"]["chain"])
        (root / rec["id"] / "run_matrix_test2_x.npy.substrate_trajectory.csv.zst").write_bytes(b"z")
        m = cm.scan(root)
        r = next(x for x in m["recordings"] if x["id"] == rec["id"])
        assert r["has"]["substrate_csv"] and r["has"]["substrate_join"] == "in-chain"
        assert r["has"]["substrate_csv_paths"] == [rec["id"] + "/run_matrix_test2_x.npy.substrate_trajectory.csv.zst"]
        assert r["has"]["substrate_csv_bytes"] == 1               # its size, so a fetch of it can be measured
        assert r["n_snapshots"] == rec["n_snapshots"] and r["bytes"] == rec["bytes"]   # not counted as a snapshot
        assert m["n_with_substrate_csv"] == m0["n_with_substrate_csv"] + 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
