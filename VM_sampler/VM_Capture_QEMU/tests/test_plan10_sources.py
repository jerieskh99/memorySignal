#!/usr/bin/env python3
"""sources.py: local and ssh sources present one interface; the manifest is the same over both.

No live ssh here: the ssh source is checked on the command lines it would run and on the
listing format it parses. A local tree is listed and fetched for real.

Run:  python3 tests/test_plan10_sources.py
      pytest tests/test_plan10_sources.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

QEMU_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QEMU_DIR))
from plan10_analysis import corpus_manifest as cm  # noqa: E402
from plan10_analysis.sources import Entry, LocalSource, SshSource, SourceError, make_source  # noqa: E402


def _tree(root: Path) -> None:
    d = root / "mem" / "mem_alpha_v2" / "ws_256_--duration_300_--seed_ab" / "rep001__run"
    d.mkdir(parents=True)
    for i in range(4):
        (d / f"{i:06d}.zst").write_bytes(b"x" * (100 if i == 0 else 5))
    (d / ".000002.zst.tmp1").write_bytes(b"p")
    (root / "cpu" / "cpu_beta_v2" / "d_450" / "rep001__run").mkdir(parents=True)


def test_local_listing_and_fetch():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _tree(root)
        src = LocalSource(root)
        ok, msg = src.test()
        assert ok, msg
        lst = src.listing()
        rels = {e.relpath for e in lst}
        assert "mem/mem_alpha_v2/ws_256_--duration_300_--seed_ab/rep001__run/000000.zst" in rels
        assert "mem/mem_alpha_v2/ws_256_--duration_300_--seed_ab/rep001__run" in {e.relpath for e in lst if e.is_dir}
        p = src.fetch("mem/mem_alpha_v2/ws_256_--duration_300_--seed_ab/rep001__run")
        assert (p / "000003.zst").exists()
        try:
            src.fetch("nope/nope/nope/rep001")
            assert False
        except SourceError:
            pass


def test_manifest_same_over_walk_and_listing():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _tree(root)
        a = cm.scan(root)
        b = cm.scan_listing(LocalSource(root).listing(), str(root))
        for k in ("n_recordings", "n_with_chain", "n_workloads", "families", "partial_files"):
            assert a[k] == b[k], k
        assert [r["id"] for r in a["recordings"]] == [r["id"] for r in b["recordings"]]
        assert a["recordings"][1]["n_pairs"] == 3 and a["recordings"][0]["status"] == "empty"


def test_ssh_command_lines_and_listing_parse():
    s = SshSource(host="srv.example", remote_root="/data/zstd_local/", user="jeries", key="~/.ssh/id_x", port=2222)
    argv = s.ssh_argv("echo hi")
    assert argv[0] == "ssh" and argv[-1] == "echo hi" and "jeries@srv.example" in argv
    assert "-p" in argv and argv[argv.index("-p") + 1] == "2222"
    assert "-i" in argv and argv[argv.index("-i") + 1].endswith("/.ssh/id_x")
    assert "BatchMode=yes" in " ".join(argv)
    r = s.rsync_argv("mem/wl/var/rep001__x", Path("/tmp/cache/mem/wl/var/rep001__x"))
    assert r[0] == "rsync" and "--include=*.zst" in r and r[-2].startswith("jeries@srv.example:")
    assert r[-2].endswith("/rep001__x/") and r[-1].endswith("/rep001__x/")
    assert "find . -mindepth 1" in s.listing_cmd() and "%P" in s.listing_cmd()
    lst = SshSource.parse_listing("d\t0\tmem\nd\t0\tmem/wl\nd\t0\tmem/wl/var\nd\t0\tmem/wl/var/rep001__r\n"
                                  "f\t1000\tmem/wl/var/rep001__r/000000.zst\nf\t10\tmem/wl/var/rep001__r/000001.zst\n"
                                  "f\t5\t.hidden\nbad line\n")
    assert len(lst) == 6 and lst[4] == Entry("mem/wl/var/rep001__r/000000.zst", 1000, False)
    m = cm.scan_listing(lst, "srv:/data/zstd_local", source=s.describe())
    assert m["n_recordings"] == 1 and m["recordings"][0]["n_pairs"] == 1 and m["source"]["kind"] == "ssh"
    assert m["source"]["key"].endswith("/.ssh/id_x")


def test_factory():
    assert make_source({"kind": "local", "root": "/tmp"}).kind == "local"
    assert make_source({"kind": "ssh", "host": "h", "remote_root": "/r"}).kind == "ssh"
    for bad in ({"kind": "ssh"}, {"kind": "local"}, {"kind": "ftp", "root": "/"}):
        try:
            make_source(bad)
            assert False, bad
        except SourceError:
            pass


def test_listing_survives_find_exit_1_on_a_live_corpus():
    """A corpus being written to makes find exit 1 without making the listing wrong.

    find returns 1 when an entry vanishes between readdir and stat, which is what an rsync temp
    file does while an upload is running. It still lists everything else. Treating that as fatal
    made the whole scan refuse against a corpus that was merely in use.
    """
    import subprocess as _sp
    from plan10_analysis import sources as _src

    rows = ("d\t0\tmem\nd\t0\tmem/wl\nd\t0\tmem/wl/var\nd\t0\tmem/wl/var/rep001__r\n"
            "f\t100\tmem/wl/var/rep001__r/000000.zst\nf\t5\tmem/wl/var/rep001__r/000001.zst\n")
    vanished = "find: './mem/wl/var/rep001__r/.000002.zst.pmsrkx': No such file or directory"
    s = SshSource("h", "/r", user="u")
    real = _sp.run
    try:
        # partial: non-zero exit, but rows came back -> keep them, and say so
        _src.subprocess.run = lambda *a, **k: _sp.CompletedProcess(a[0] if a else [], 1, rows, vanished)
        lst = s.listing()
        assert len(lst) == 6, lst
        assert s.listing_warning and "partial listing" in s.listing_warning
        assert "No such file" in s.listing_warning
        # the vanished temp file never enters the manifest either way
        m = cm.scan_listing(lst, "srv:/r")
        assert m["n_recordings"] == 1, m

        # clean: the warning clears rather than sticking from the previous call
        _src.subprocess.run = lambda *a, **k: _sp.CompletedProcess(a[0] if a else [], 0, rows, "")
        assert len(s.listing()) == 6 and s.listing_warning is None

        # real failure: non-zero exit and nothing usable -> still refuses
        _src.subprocess.run = lambda *a, **k: _sp.CompletedProcess(a[0] if a else [], 255, "", "Permission denied")
        try:
            s.listing()
            assert False, "a listing with no usable rows must raise"
        except SourceError as e:
            assert "255" in str(e)
    finally:
        _src.subprocess.run = real


def test_trajectory_fetch_is_its_own_narrow_rsync():
    """Pulling the trajectory alone must not touch the chain filter, and a local source
    finds the file in place or reports none."""
    import tempfile
    from plan10_analysis.sources import SshSource, make_source
    s = SshSource("srv.example", "/project/zstd", user="jeries")
    r = s.rsync_trajectory_argv("mem/wl/var/rep001__x", Path("/tmp/cache/mem/wl/var/rep001__x"))
    assert r[0] == "rsync" and "--include=*substrate_trajectory*" in r and "--exclude=*" in r
    assert "--include=*.zst" not in r and r[-2].startswith("jeries@srv.example:")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "mem" / "wl" / "var" / "rep001__x"
        d.mkdir(parents=True)
        (d / "000000.zst").write_bytes(b"z")
        loc = make_source({"kind": "local", "root": td})
        assert loc.fetch_trajectory("mem/wl/var/rep001__x") is None
        (d / "run_matrix_test1_wl.npy.substrate_trajectory.csv").write_text("seq,page_index,hamming\n")
        assert loc.fetch_trajectory("mem/wl/var/rep001__x").name.endswith("substrate_trajectory.csv")


def test_sources_read_and_rebuild_the_archive_manifest():
    """A local source hands over the archive's own manifest when one exists and walks when
    not; reconcile always walks and leaves a manifest behind. The ssh commands are the
    same two operations spelled for a remote shell."""
    import json, tempfile
    from plan10_analysis import corpus_manifest as cm
    from plan10_analysis.sources import SshSource, make_source
    from plan10_analysis.testing import synth
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "zstd_local"
        synth.make_corpus(root, n_snapshots=4, workloads=(("mem", "mem_synth_a_v2", 1),))
        loc = make_source({"kind": "local", "root": str(root)})
        assert loc.manifest() is None                                  # no manifest yet: caller walks
        walked = cm.scan_source(loc)
        assert "archive_manifest" not in walked and walked["n_recordings"] == 1
        rebuilt = cm.scan_source(loc, reconcile=True)                  # the walk that writes
        assert rebuilt["archive_manifest"]["rebuilt_at"] and (root / ".manifest" / "manifest.json").is_file()
        read = cm.scan_source(loc)                                     # now read, not walked
        assert read["archive_manifest"]["read_at"] and read["n_recordings"] == 1
        assert read["source"] == loc.describe() and read["root"] == str(root)
        # a registration by a writer shows up on the next read with no walk
        from plan10_analysis import archive_manifest as am
        rid = read["recordings"][0]["id"]
        (root / rid / "run_matrix_test1_mem_synth_a_v2.npy.substrate_trajectory.csv").write_text("seq,page_index,hamming\n0,1,2\n")
        am.register(root, rid)
        again = cm.scan_source(loc)
        assert again["n_with_substrate_csv"] == 1 and again["recordings"][0]["has"]["substrate_columns"] == ["hamming"]
        assert again["archive_manifest"]["registered_since_rebuild"] == 1
    s = SshSource("srv.example", "/project/zstd", user="jeries", remote_repo="$HOME/repo", remote_python="python3")
    assert s.manifest_cmd() == "cat /project/zstd/.manifest/manifest.json"
    assert s.reconcile_cmd() == "cd $HOME/repo && python3 plan10_analysis/archive_manifest.py rebuild /project/zstd"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
