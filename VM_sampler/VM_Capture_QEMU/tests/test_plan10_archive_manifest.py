#!/usr/bin/env python3
"""archive_manifest.py: the archive's own inventory, written by whoever adds to it.

Over the synthetic tree: register one recording and the entry equals what a full walk says
about it (one parser); a trajectory beside the chain gives has.substrate_columns from its
header; re-registering replaces rather than duplicates; unregister forgets; rebuild
reconciles a hand-copied recording the registrations missed; two writers serialise on the
lock and both updates survive; a stale lock from a dead writer is broken, a live one waited on.

Run:  python3 tests/test_plan10_archive_manifest.py
      pytest tests/test_plan10_archive_manifest.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from plan10_analysis import archive_manifest as am          # noqa: E402
from plan10_analysis import corpus_manifest as cm          # noqa: E402
from plan10_analysis.testing import synth                   # noqa: E402


def _corpus(td: Path) -> tuple[Path, list[str]]:
    root = td / "zstd_local"
    changes = synth.make_corpus(root, n_snapshots=4, workloads=(("mem", "mem_synth_a_v2", 2), ("cpu", "cpu_synth_b_v2", 1)))
    return root, sorted(changes)


def test_register_matches_the_walk_and_reads_the_trajectory_header():
    with tempfile.TemporaryDirectory() as td:
        root, ids = _corpus(Path(td))
        rid = ids[0]
        (root / rid / "run_matrix_test1_mem_synth_a_v2.npy.substrate_trajectory.csv").write_text(
            "seq,page_index,hamming,cosine\n0,1,3,0.5\n")
        e = am.register(root, rid)
        walk = next(r for r in cm.scan(root)["recordings"] if r["id"] == rid)
        for k in ("id", "family", "workload", "variant", "rep", "n_snapshots", "n_pairs", "bytes", "base_bytes", "contiguous"):
            assert e[k] == walk[k], (k, e[k], walk[k])
        assert e["has"]["substrate_csv"] and e["has"]["substrate_join"] == "in-chain"
        assert e["has"]["substrate_columns"] == ["hamming", "cosine"]
        m = am.load(root)
        assert m["schema"] == cm.SCHEMA and m["n_recordings"] == 1 and m["n_with_substrate_csv"] == 1
        assert m["archive_manifest"]["registered_since_rebuild"] == 1
        # a second registration of the same recording replaces, never duplicates
        am.register(root, rid)
        assert am.load(root)["n_recordings"] == 1
        # a recording without a trajectory says so, with None not []
        e2 = am.register(root, ids[1])
        assert e2["has"]["substrate_csv"] is False and e2["has"]["substrate_columns"] is None
        assert am.load(root)["n_recordings"] == 2


def test_unregister_and_rebuild_reconcile():
    with tempfile.TemporaryDirectory() as td:
        root, ids = _corpus(Path(td))
        am.register(root, ids[0])
        assert am.unregister(root, ids[0]) is True and am.load(root)["n_recordings"] == 0
        assert am.unregister(root, ids[0]) is False
        # the other two were never registered: a hand copy. rebuild finds all three.
        m = am.rebuild(root)
        assert m["n_recordings"] == 3 and m["archive_manifest"]["registered_since_rebuild"] == 0
        assert m["archive_manifest"]["rebuilt_at"] == m["scanned_at"]
        assert all(r["has"]["substrate_columns"] is None for r in m["recordings"])
        # and the manifest is a corpus manifest the console can consume as-is
        ctx_recs = {r["id"] for r in am.load(root)["recordings"]}
        assert ctx_recs == set(ids)


def test_describe_refuses_escapes_and_missing():
    with tempfile.TemporaryDirectory() as td:
        root, ids = _corpus(Path(td))
        assert am.describe(root, "mem/nope/x/rep001__y") is None
        try:
            am.describe(root, "../" + ids[0])
            assert False, "must refuse a path that escapes the root"
        except am.ArchiveError:
            pass


def test_two_writers_serialise_on_the_lock():
    with tempfile.TemporaryDirectory() as td:
        root, ids = _corpus(Path(td))
        errs: list[BaseException] = []

        def w(rid):
            try:
                am.register(root, rid)
            except BaseException as e:      # noqa: BLE001
                errs.append(e)

        ts = [threading.Thread(target=w, args=(r,)) for r in ids]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert not errs, errs
        assert am.load(root)["n_recordings"] == 3            # nobody's update was lost
        assert not (root / am.MANIFEST_DIR / am.LOCK_NAME).exists()


def test_stale_lock_is_broken_live_lock_is_waited_on():
    with tempfile.TemporaryDirectory() as td:
        root, ids = _corpus(Path(td))
        lock = root / am.MANIFEST_DIR / am.LOCK_NAME
        lock.mkdir(parents=True)
        old = time.time() - am.LOCK_STALE_S - 5
        os.utime(lock, (old, old))                            # a writer that died long ago
        am.register(root, ids[0])                             # broken, not waited on
        assert am.load(root)["n_recordings"] == 1
        # a live lock: released after 0.5 s by another thread; register must wait, not fail
        lock.mkdir()
        threading.Timer(0.5, lambda: lock.rmdir()).start()
        t0 = time.time()
        am.register(root, ids[1])
        assert time.time() - t0 >= 0.4
        assert am.load(root)["n_recordings"] == 2


def test_rebuild_keeps_a_registration_made_during_its_walk():
    """A writer that registers while rebuild is walking must not be erased by the rewrite."""
    from plan10_analysis import sources as S
    with tempfile.TemporaryDirectory() as td:
        root, ids = _corpus(Path(td))
        late = ids[2]
        real = S.LocalSource.listing

        def listing_then_register(self):
            out = real(self)                       # the walk has "seen" the tree...
            time.sleep(1.1)                        # registered_at must sort after t0 (second resolution)
            am.register(root, late)                # ...and a writer registers meanwhile
            return out

        S.LocalSource.listing = listing_then_register
        try:
            m = am.rebuild(root)
        finally:
            S.LocalSource.listing = real
        ids_after = {r["id"] for r in am.load(root)["recordings"]}
        assert late in ids_after and len(ids_after) == 3
        assert m["archive_manifest"]["kept_from_live_during_walk"] == [late]
        assert m["archive_manifest"]["registered_since_rebuild"] == 1


def test_release_survives_a_file_the_client_will_not_let_go_of():
    """NFS silly-renames an unlinked file that is still open (.nfsXXXX inside lock.d), so a
    plain rmdir fails 'Directory not empty'. The first real rebuild crashed there AFTER writing
    the manifest. Release renames the lock dir away first, so the lock is gone no matter what
    is inside; the litter is removed on a later release."""
    with tempfile.TemporaryDirectory() as td:
        root, ids = _corpus(Path(td))
        lock = root / am.MANIFEST_DIR / am.LOCK_NAME
        real_unlink = Path.unlink
        stray = ".nfs0000deadbeef"

        def stubborn_unlink(self, *a, **k):
            if self.name == stray:
                raise OSError(16, "Device or resource busy")
            return real_unlink(self, *a, **k)

        real_mkdir = Path.mkdir

        def mkdir_then_litter(self, *a, **k):
            real_mkdir(self, *a, **k)
            if self == lock:
                (self / stray).write_text("held open by the nfs client")

        Path.mkdir, Path.unlink = mkdir_then_litter, stubborn_unlink
        try:
            am.register(root, ids[0])                         # must not raise
        finally:
            Path.mkdir, Path.unlink = real_mkdir, real_unlink
        assert am.load(root)["n_recordings"] == 1
        assert not lock.exists(), "the lock must be released even though the dir could not be emptied"
        litter = list(lock.parent.glob(am.LOCK_NAME + ".released-*"))
        assert len(litter) == 1                               # parked, not blocking
        am.register(root, ids[1])                             # the next release sweeps it
        assert not lock.exists() and not list(lock.parent.glob(am.LOCK_NAME + ".released-*"))
        assert am.load(root)["n_recordings"] == 2


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
