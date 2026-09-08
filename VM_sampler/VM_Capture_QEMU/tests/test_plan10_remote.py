#!/usr/bin/env python3
"""Remote execution: the extraction runs where the recordings are, only the L1 file comes back.

There is no ssh daemon on this machine, so the transport is the one part that cannot be
exercised here: its argv is asserted instead. Everything above it -- the remote command, the
CLI that runs on the server, its JSON reply, the pull, the executor's use of all three, and
the failure paths -- is driven for real through a transport that runs the same commands in a
local shell and copies the same files.

Run:  python3 tests/test_plan10_remote.py
      pytest tests/test_plan10_remote.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

QEMU_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QEMU_DIR))
from plan10_analysis import corpus_manifest, scheme as S            # noqa: E402
from plan10_analysis.runner import chain, differ, executor, extract  # noqa: E402
from plan10_analysis.sources import SourceError, SshSource, make_source  # noqa: E402
from plan10_analysis.testing import synth                            # noqa: E402


def _have_tools() -> bool:
    try:
        differ.find_differ()
    except differ.DifferError as e:
        print(f"skip: {e}")
        return False
    return chain.zstd_available()


class LocalTransport:
    """Runs the remote command in a local shell and copies instead of rsyncing.

    Stands in for ssh so the remote path is executed for real. It records every command, so a
    test can assert what would have crossed the wire.
    """

    def __init__(self):
        self.commands: list[str] = []
        self.pulls: list[tuple[str, str]] = []

    def run(self, remote_cmd: str, timeout: int = 7200):
        self.commands.append(remote_cmd)
        r = subprocess.run(["bash", "-lc", remote_cmd], capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr

    def pull(self, remote_path: str, local_path: Path) -> None:
        self.pulls.append((remote_path, str(local_path)))
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(remote_path, local_path)


def _src(root: Path, repo: Path, store: Path, transport=None) -> SshSource:
    # remote_python is this interpreter: a login shell on the server may resolve a different
    # python than the one carrying numpy, which is exactly why the parameter exists
    return SshSource(host="server.example", remote_root=str(root), user="jeries",
                     mode="remote", remote_repo=str(repo), remote_store=str(store),
                     remote_python=sys.executable, transport=transport or LocalTransport())


def test_remote_command_and_argv():
    """What the runner would send: the command line, and the ssh/rsync invocations around it."""
    s = _src(Path("/data/zstd_local"), Path("/srv/repo"), Path("/srv/l1"))
    cmd = s.extract_cmd("mem/wl/var/rep001__r", 2, ["hamming", "cosine"], 40)
    assert cmd.startswith(f"cd /srv/repo && {sys.executable} plan10_analysis/runner/extract_cli.py")
    assert SshSource(host="h", remote_root="/r", mode="remote").extract_cmd("r", 2, ["hamming"], None).startswith(
        "cd $HOME/memorySignal/VM_sampler/VM_Capture_QEMU && python3 ")
    for frag in ("--root /data/zstd_local", "--rec-id mem/wl/var/rep001__r", "--speed 2",
                 "--columns hamming,cosine", "--store /srv/l1", "--max-pairs 40"):
        assert frag in cmd, frag
    assert "--max-pairs" not in s.extract_cmd("r", 2, ["hamming"], None)
    assert s.extract_cmd("r", 2, ["hamming"], None, probe=True).endswith("--probe")
    # a recording id with a space cannot break out of the command
    assert "'weird dir/rep001'" in s.extract_cmd("weird dir/rep001", 2, ["hamming"], None)
    # the transport that would carry it
    real = SshSource(host="h", remote_root="/r", user="u", key="/k", port=2222, mode="remote")
    argv = real.ssh_argv(real.extract_cmd("r", 2, ["hamming"], None))
    assert argv[0] == "ssh" and "u@h" in argv and argv[argv.index("-p") + 1] == "2222"
    assert argv[-1].startswith("cd ")
    assert real.describe()["mode"] == "remote"
    assert make_source({"kind": "ssh", "host": "h", "remote_root": "/r"}).mode == "fetch"
    assert make_source({"kind": "ssh", "host": "h", "remote_root": "/r", "mode": "remote"}).mode == "remote"
    try:
        make_source({"kind": "ssh", "host": "h", "remote_root": "/r", "mode": "sideways"})
        assert False
    except SourceError as e:
        assert "fetch" in str(e)


def test_probe_reports_what_the_server_has():
    if not _have_tools():
        return
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        root = td / "corpus"
        synth.make_corpus(root, n_snapshots=4, workloads=(("mem", "mem_synth_a_v2", 1),))
        s = _src(root, QEMU_DIR, td / "l1")
        p = s.probe_remote()
        assert p["zstd"] and p["numpy"] and p["root_exists"] and "error" not in p["differ"]
        assert p["executable"] == sys.executable
        # the probe REPORTS a python without numpy rather than dying of it, which is the
        # condition it exists to detect
        bare = SshSource(host="h", remote_root=str(root), mode="remote", remote_repo=str(QEMU_DIR),
                         remote_store=str(td / "l1"), remote_python="/usr/bin/python3",
                         transport=LocalTransport()).probe_remote()
        assert "numpy" in bare and "differ" in bare
        assert p["differ"]["path"].endswith("live_delta_calc_modular")
        # a root that is not there is reported, not guessed
        assert _src(td / "nope", QEMU_DIR, td / "l1").probe_remote()["root_exists"] is False


def test_remote_extract_moves_only_the_l1_file():
    if not _have_tools():
        return
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        root = td / "corpus"
        ch = synth.make_corpus(root, n_snapshots=8, workloads=(("mem", "mem_synth_a_v2", 1),))
        rid = next(iter(ch))
        remote_store, local_store = td / "remote_l1", td / "local_l1"
        tr = LocalTransport()
        s = _src(root, QEMU_DIR, remote_store, tr)
        npz = s.remote_extract(rid, 2, ["hamming"], local_store, max_pairs=None)

        assert npz.parent == local_store and npz.exists()
        d = extract.load(npz)
        assert d["n_pairs"] == 7 and d["n_pages"] == synth.N_PAGES
        # the extraction happened on the "server": its store holds the originals
        assert (remote_store / npz.name).exists()
        assert json.loads((local_store / npz.name.replace(".npz", ".meta.json")).read_text())["rec_id"] == rid
        # only the npz and its meta crossed; no .zst did
        assert len(tr.pulls) == 2 and all(p[0].endswith((".npz", ".meta.json")) for p in tr.pulls)
        assert not any(p[0].endswith(".zst") for p in tr.pulls)
        # and that is far smaller than the chain it came from
        chain_bytes = sum(f.stat().st_size for f in (root / rid).glob("*.zst"))
        assert npz.stat().st_size < chain_bytes

        # a second call reuses the server's store: still one command, and it says so
        tr.commands.clear()
        again = s.remote_extract(rid, 2, ["hamming"], local_store)
        assert again == npz and len(tr.commands) == 1

        # failures surface with the remote's own message
        try:
            s.remote_extract("no/such/recording/rep001", 2, ["hamming"], local_store)
            assert False
        except SourceError as e:
            assert "recording not found" in str(e)


def test_executor_runs_a_scheme_remotely():
    """The whole path: the executor takes the remote branch and never fetches a chain."""
    if not _have_tools():
        return
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        root = td / "corpus"
        synth.make_corpus(root, n_snapshots=10, workloads=(("mem", "mem_synth_a_v2", 1), ("cpu", "cpu_synth_b_v2", 1)))
        man = corpus_manifest.scan(root)
        mp = td / "m.json"
        mp.write_text(json.dumps(man))
        tr = LocalTransport()
        src = _src(root, QEMU_DIR, td / "remote_l1", tr)

        def node(i, m, **p):
            return {"id": i, "module": m, "params": p, "x": 0, "y": 0}

        sch = {"schema": "plan10.scheme.v1", "label": "remote_run",
               "acknowledged": [{"id": "no_substrate", "at": "t"}],
               "nodes": [node("c", "cells", sel=[r["id"] for r in man["recordings"]], min_pairs=1),
                         node("ch", "channels", chans=["hamming"]), node("co", "collapse", reduce="changed_fraction"),
                         node("wi", "window", w=4, h=2), node("st", "stats", feats=["mean", "max"]),
                         node("w", "write")],
               "pipes": [{"from": ["c", "cells"], "to": ["ch", "cells"]}, {"from": ["ch", "field"], "to": ["co", "in"]},
                         {"from": ["co", "out"], "to": ["wi", "in"]}, {"from": ["wi", "out"], "to": ["st", "in"]},
                         {"from": ["st", "out"], "to": ["w", "in"]}]}
        sp = td / "s.json"
        sp.write_text(json.dumps(sch))

        # the executor is given the source object's spec, but must use OUR transport, so patch
        # make_source for the call: this is the one seam a real run resolves over ssh
        import plan10_analysis.runner.executor as ex
        real_make = ex.make_source
        ex.make_source = lambda spec: src
        try:
            rc = ex.run(sp, td / "out", src.describe(), td / "local_l1", speed=2, manifest_path=mp)
        finally:
            ex.make_source = real_make
        out = td / "out" / "remote_run"
        assert rc == 0, (out / "run.log").read_text()
        z = np.load(out / "features.npz")
        assert z["X"].shape == (2 * ((9 - 4) // 2 + 1), 2)
        log = (out / "run.log").read_text()
        assert "remote host ready" in log and "extracting on server.example" in log
        assert "extract_cli.py" in log
        st = json.loads((out / "status.json").read_text())
        assert all(v.get("where") == "remote" for v in st["per_recording"].values())
        # the sidecar records that this ran remotely, and where
        side = json.loads((out / "sidecar.json").read_text())
        assert side["source"]["mode"] == "remote" and side["source"]["host"] == "server.example"
        assert side["extraction_ran"] == "remote"
        # and the differ it names is the one that actually ran, read from the store's own
        # meta rather than from whatever binary happens to sit on this machine. Against the
        # real server this first recorded the laptop's path.
        meta = json.loads((td / "local_l1" / next(iter(
            p for p in (td / "local_l1").iterdir() if p.name.endswith(".meta.json"))).name).read_text())
        assert side["differ"] == meta["differ"]
        assert side["differ_per_recording"] is None      # one differ for every recording here
        # one extraction command per recording, and nothing pulled but L1 files
        assert sum(1 for c in tr.commands if "--probe" not in c) == 2
        assert all(p[0].endswith((".npz", ".meta.json")) for p in tr.pulls)


def test_executor_refuses_when_the_server_is_not_ready():
    if not _have_tools():
        return
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        root = td / "corpus"
        synth.make_corpus(root, n_snapshots=6, workloads=(("mem", "mem_synth_a_v2", 1),))
        man = corpus_manifest.scan(root)
        mp = td / "m.json"
        mp.write_text(json.dumps(man))
        src = _src(td / "not_there", QEMU_DIR, td / "l1")      # trace root missing on the "server"
        sch = {"schema": "plan10.scheme.v1", "label": "bad_remote", "acknowledged": [{"id": "no_substrate", "at": "t"}],
               "nodes": [{"id": "c", "module": "cells", "params": {"sel": [r["id"] for r in man["recordings"]], "min_pairs": 1}, "x": 0, "y": 0},
                         {"id": "ch", "module": "channels", "params": {"chans": ["hamming"]}, "x": 0, "y": 0},
                         {"id": "co", "module": "collapse", "params": {"reduce": "changed_fraction"}, "x": 0, "y": 0},
                         {"id": "wi", "module": "window", "params": {"w": 4, "h": 2}, "x": 0, "y": 0},
                         {"id": "st", "module": "stats", "params": {"feats": ["mean"]}, "x": 0, "y": 0},
                         {"id": "w", "module": "write", "params": {}, "x": 0, "y": 0}],
               "pipes": [{"from": ["c", "cells"], "to": ["ch", "cells"]}, {"from": ["ch", "field"], "to": ["co", "in"]},
                         {"from": ["co", "out"], "to": ["wi", "in"]}, {"from": ["wi", "out"], "to": ["st", "in"]},
                         {"from": ["st", "out"], "to": ["w", "in"]}]}
        sp = td / "s.json"
        sp.write_text(json.dumps(sch))
        import plan10_analysis.runner.executor as ex
        real_make = ex.make_source
        ex.make_source = lambda spec: src
        try:
            rc = ex.run(sp, td / "out", src.describe(), td / "local_l1", speed=2, manifest_path=mp)
        finally:
            ex.make_source = real_make
        assert rc == 1
        st = json.loads((td / "out" / "bad_remote" / "status.json").read_text())
        assert st["state"] == "refused" and "the trace root" in json.dumps(st["verdict"])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
