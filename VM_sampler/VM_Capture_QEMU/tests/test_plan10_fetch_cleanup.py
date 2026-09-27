#!/usr/bin/env python3
"""Fetched files leave the cache only once the server is shown to hold identical copies.

In ssh fetch mode the executor deletes a recording's fetched trajectory or chain members after
its L1 store is complete, but first one read-only ssh round trip checks every file: present at
the path it was fetched from, same size, same sha256. A size mismatch, a checksum mismatch, a
missing remote file and an unreachable server each keep every local file. Only paths under the
cache are ever removed; a local source is never touched; a later scheme that needs a column the
store lacks fetches again. The setting and the per-recording outcome land in the sidecar.

Run:  python3 tests/test_plan10_fetch_cleanup.py
      pytest tests/test_plan10_fetch_cleanup.py
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from plan10_analysis import sources as S                                     # noqa: E402
from plan10_analysis.runner import executor                                  # noqa: E402
from test_plan10_runner_executor import _corpus, _example, _have_tools        # noqa: E402


class CannedTransport:
    """Stands in for ssh: records every command and answers as the server would, by reading the
    'remote' directory the command names. Nothing is ever pulled: cleanup must not need to."""

    def __init__(self, rc: int = 0, err: str = "", raise_exc: Exception | None = None):
        self.commands: list[str] = []
        self.rc, self.err, self.raise_exc = rc, err, raise_exc

    def run(self, remote_cmd: str, timeout: int = 7200):
        self.commands.append(remote_cmd)
        if self.raise_exc:
            raise self.raise_exc
        return self.rc, self.answer(remote_cmd), self.err

    @staticmethod
    def answer(cmd: str) -> str:
        d = Path(shlex.split(cmd.split(" 2>/dev/null", 1)[0])[1])
        if not d.is_dir():
            return "__NODIR__\n"
        names = shlex.split(cmd.split("for f in ", 1)[1].split("; do", 1)[0])
        lines = []
        for n in names:
            f = d / n
            if f.is_file():
                lines.append(f"{f.stat().st_size} {hashlib.sha256(f.read_bytes()).hexdigest()} {n}")
            else:
                lines.append(f"MISSING - {n}")
        return "\n".join(lines) + "\n"

    def pull(self, remote_path, local_path):
        raise AssertionError("cleanup must never pull anything")


REL = "kernel/kernel_x_v2/args_--seed_1/rep001__t"


def _pair(td: Path, transport=None):
    """A 'remote' recording and its fetched copy in the cache."""
    remote, cache = td / "remote", td / "cache"
    rd = remote / REL
    rd.mkdir(parents=True)
    for i in range(3):
        (rd / f"{i:06d}.zst").write_bytes(os.urandom(1000 + 700 * i))
    (rd / "run_matrix_test1_kernel_x_v2.npy.substrate_trajectory.csv.zst").write_bytes(os.urandom(2500))
    shutil.copytree(rd, cache / REL)
    src = S.SshSource(host="server", remote_root=str(remote), cache=cache, transport=transport or CannedTransport())
    return src, rd, cache / REL


def _bytes(d: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(d.iterdir()) if p.is_file()}


def test_verify_command_is_one_read_only_round_trip():
    with tempfile.TemporaryDirectory() as td:
        src, rd, cd = _pair(Path(td))
        cmd = src.verify_cmd(REL, [p.name for p in src.fetched_files(REL)])
        assert "sha256sum" in cmd and "wc -c" in cmd and str(rd) in cmd
        scrubbed = cmd.replace("2>/dev/null", "")
        for bad in ("rm ", "mv ", "touch ", "tee ", ">", "rsync"):
            assert bad not in scrubbed, bad
        before = _bytes(rd)
        rec = src.cleanup_fetched(REL)
        assert rec["deleted"] and len(src.transport.commands) == 1          # one round trip
        assert _bytes(rd) == before                                          # the server is untouched


def test_verified_copies_are_deleted_and_empty_dirs_pruned():
    with tempfile.TemporaryDirectory() as td:
        src, rd, cd = _pair(Path(td))
        total = sum(p.stat().st_size for p in cd.iterdir())
        rec = src.cleanup_fetched(REL)
        assert rec["verified"] and rec["deleted"] and rec["files"] == 4 and rec["failed_check"] is None
        assert abs(rec["freed_mb"] - total / 1e6) < 0.06 and set(rec["checks"].values()) == {"verified"}
        assert not cd.exists() and not (src.cache / "kernel").exists() and src.cache.is_dir()
        assert len(_bytes(rd)) == 4


def test_size_mismatch_keeps_everything():
    with tempfile.TemporaryDirectory() as td:
        src, rd, cd = _pair(Path(td))
        (rd / "000001.zst").write_bytes(b"short")
        local = _bytes(cd)
        rec = src.cleanup_fetched(REL)
        assert not rec["deleted"] and not rec["verified"] and rec["failed_check"].startswith("size: 000001.zst")
        assert rec["checks"]["000001.zst"] == "size" and rec["checks"]["000000.zst"] == "verified"
        assert _bytes(cd) == local and rec["freed_mb"] == 0.0 and rec["reason"].startswith("kept:")


def test_checksum_mismatch_keeps_everything():
    with tempfile.TemporaryDirectory() as td:
        src, rd, cd = _pair(Path(td))
        f = rd / "000002.zst"
        b = bytearray(f.read_bytes()); b[10] ^= 0xFF; f.write_bytes(bytes(b))            # same size, one byte off
        local = _bytes(cd)
        rec = src.cleanup_fetched(REL)
        assert not rec["deleted"] and rec["failed_check"].startswith("sha256: 000002.zst")
        assert rec["checks"]["000002.zst"] == "sha256" and _bytes(cd) == local


def test_missing_remote_file_keeps_everything():
    with tempfile.TemporaryDirectory() as td:
        src, rd, cd = _pair(Path(td))
        (rd / "run_matrix_test1_kernel_x_v2.npy.substrate_trajectory.csv.zst").unlink()
        local = _bytes(cd)
        rec = src.cleanup_fetched(REL)
        assert not rec["deleted"] and rec["failed_check"].startswith("missing: run_matrix_test1")
        assert _bytes(cd) == local
        shutil.rmtree(rd)                                                         # the whole recording gone
        rec = src.cleanup_fetched(REL)
        assert not rec["deleted"] and rec["failed_check"].startswith("missing:") and _bytes(cd) == local


def test_unreachable_server_keeps_everything():
    with tempfile.TemporaryDirectory() as td:
        src, rd, cd = _pair(Path(td), CannedTransport(raise_exc=OSError("No route to host")))
        local = _bytes(cd)
        rec = src.cleanup_fetched(REL)
        assert not rec["deleted"] and rec["failed_check"].startswith("unreachable:") and "No route" in rec["failed_check"]
        assert _bytes(cd) == local
        src2, rd2, cd2 = _pair(Path(td) / "b", CannedTransport(rc=255, err="ssh: connect to host server port 22: Connection refused"))
        rec = src2.cleanup_fetched(REL)
        assert not rec["deleted"] and rec["failed_check"].startswith("unreachable: ssh exit 255") and len(_bytes(cd2)) == 4


def test_nothing_outside_the_cache_is_ever_removed():
    with tempfile.TemporaryDirectory() as td:
        src, rd, cd = _pair(Path(td))
        outside = Path(td) / "elsewhere.zst"
        outside.write_bytes(b"not yours")
        (cd / "000009.zst").symlink_to(outside)                                   # a fetched-looking name pointing out
        (cd / "notes.txt").write_text("derived, not fetched")
        assert [p.name for p in src.fetched_files(REL)] == ["000000.zst", "000001.zst", "000002.zst", "000009.zst",
                                                            "run_matrix_test1_kernel_x_v2.npy.substrate_trajectory.csv.zst"]
        try:
            src.cleanup_fetched(REL)
            assert False, "must refuse"
        except AssertionError as e:
            assert "outside the cache" in str(e)
        assert outside.read_bytes() == b"not yours" and (cd / "000000.zst").exists() and (cd / "notes.txt").exists()
        assert src.transport.commands == []                                       # refused before any round trip


def test_executor_cleans_after_extraction_and_refetches_a_missing_column():
    """Through the executor, on the chain path: every fetched member goes once the store is
    complete; a scheme that needs a column the store lacks fetches again; keep_fetched keeps."""
    if not _have_tools():
        return
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        root, manifest, mp, changes = _corpus(td, n_snapshots=10)              # the 'server'
        cache = td / "cache"
        src = S.SshSource(host="server", remote_root=str(root), cache=cache, transport=CannedTransport())
        fetched: list[str] = []

        def fetch(rel):                                                        # rsync, without the network
            fetched.append(rel)
            dest = cache / rel
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(root / rel, dest)
            return dest

        src.fetch = fetch
        src.fetch_trajectory = lambda rel: None
        real_make = executor.make_source
        executor.make_source = lambda spec: src
        try:
            def scheme(label, chans):
                s = _example(manifest, "b1")
                next(n for n in s["nodes"] if n["module"] == "channels")["params"]["chans"] = chans
                next(n for n in s["nodes"] if n["module"] == "window")["params"].update(w=4, h=2)
                s["label"] = label
                sp = td / f"{label}.json"
                sp.write_text(json.dumps(s))
                return sp

            rc = executor.run(scheme("one", ["hamming"]), td / "out", src.describe(), td / "l1", speed=2, manifest_path=mp)
            assert rc == 0, (td / "out" / "one" / "run.log").read_text()
            side = json.loads((td / "out" / "one" / "sidecar.json").read_text())
            fc = side["fetch_cleanup"]
            assert fc["enabled"] and len(fc["per_recording"]) == 3 and len(fetched) == 3
            for rid, rec in fc["per_recording"].items():
                assert rec["verified"] and rec["deleted"] and rec["freed_mb"] > 0 and rec["failed_check"] is None, (rid, rec)
            assert not any(p.is_file() for p in cache.rglob("*"))                # the cache is empty again
            assert any("fetched files deleted" in l for l in (td / "out" / "one" / "run.log").read_text().splitlines())
            assert list((td / "l1").glob("*.npz"))                                # the stores stay

            # the store lacks l1: the executor must fetch again, and clean again
            rc = executor.run(scheme("two", ["hamming", "l1"]), td / "out", src.describe(), td / "l1", speed=2, manifest_path=mp)
            assert rc == 0 and len(fetched) == 6 and not any(p.is_file() for p in cache.rglob("*"))

            # keep_fetched: the files stay, and the sidecar says the setting was off
            rc = executor.run(scheme("three", ["hamming", "l2"]), td / "out", src.describe(), td / "l1", speed=2, manifest_path=mp, keep_fetched=True)
            assert rc == 0 and len(fetched) == 9
            side = json.loads((td / "out" / "three" / "sidecar.json").read_text())
            assert side["fetch_cleanup"]["enabled"] is False and side["fetch_cleanup"]["setting"] == "keep_fetched"
            assert all(v is None for v in side["fetch_cleanup"]["per_recording"].values())
            assert len([p for p in cache.rglob("*.zst")]) == 3 * 10
        finally:
            executor.make_source = real_make


def test_local_source_is_never_touched():
    if not _have_tools():
        return
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        root, manifest, mp, changes = _corpus(td, n_snapshots=10)
        before = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
        s = _example(manifest, "b1")
        next(n for n in s["nodes"] if n["module"] == "window")["params"].update(w=4, h=2)
        sp = td / "b1.json"
        sp.write_text(json.dumps(s))
        rc = executor.run(sp, td / "out", {"kind": "local", "root": str(root)}, td / "l1", speed=2, manifest_path=mp)
        assert rc == 0
        side = json.loads((td / "out" / s["label"] / "sidecar.json").read_text())
        assert side["fetch_cleanup"]["enabled"] is False and "local source" in side["fetch_cleanup"]["setting"]
        assert all(v is None for v in side["fetch_cleanup"]["per_recording"].values())
        assert sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()) == before


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
