#!/usr/bin/env python3
"""analysis_bridge.py: every endpoint answers, and a run launched through it reaches done.

Starts the bridge in a thread on a free port over a synthetic corpus, then drives it the
way the page does: health, scan, validate, run, status until done, results, and a stop on
a second run. Needs the differ and zstd for the run part; the endpoint part runs anyway.

Run:  python3 tests/test_plan10_bridge.py
      pytest tests/test_plan10_bridge.py
"""
from __future__ import annotations

import json
import socket
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

QEMU_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QEMU_DIR))
from plan10_analysis import scheme as S                  # noqa: E402
from plan10_analysis.runner import chain, differ          # noqa: E402
from plan10_analysis.testing import synth                # noqa: E402
from plan10_analysis.ui import analysis_bridge as AB     # noqa: E402


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _call(port, token, path, body=None):
    sep = "&" if "?" in path else "?"
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}{sep}token={token}", method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _start(td: Path, root: Path):
    port = _free_port()
    token = "t3st-token"
    out = td / "runs"
    th = threading.Thread(target=AB.serve, args=(port, {"kind": "local", "root": str(root)}, out, td / "l1", False, False, token), daemon=True)
    th.start()
    for _ in range(50):
        try:
            if _call(port, token, "/health")[0] == 200:
                break
        except (urllib.error.URLError, ConnectionError):
            time.sleep(0.1)
    return port, token, out


def test_endpoints_and_a_full_run():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        root = td / "corpus"
        synth.make_corpus(root, n_snapshots=8)
        port, token, out = _start(td, root)
        code, h = _call(port, token, "/health")
        assert code == 200 and h["ok"]
        assert _call(port, "wrong", "/health")[0] == 401
        code, m = _call(port, token, "/manifest")
        assert code == 200 and m["n_recordings"] == 3
        code, t = _call(port, token, "/source/test", {"source": {"kind": "local", "root": str(root)}})
        assert code == 200 and t["ok"]
        code, t = _call(port, token, "/source/test", {"source": {"kind": "local", "root": str(td / "nope")}})
        assert code == 200 and not t["ok"]
        code, m2 = _call(port, token, "/scan", {"source": {"kind": "local", "root": str(root)}})
        assert code == 200 and m2["n_recordings"] == 3
        assert _call(port, token, "/scan", {"source": {"kind": "local", "root": str(td / "nope")}})[0] == 400
        # validate: the b1 example over this corpus stops on the substrate warning
        s = S.make_examples(m2)["b1"]
        next(n for n in s["nodes"] if n["module"] == "cells")["params"].update(min_pairs=1, sel=[r["id"] for r in m2["recordings"]])
        next(n for n in s["nodes"] if n["module"] == "window")["params"].update(w=4, h=2)
        code, v = _call(port, token, "/validate", {"scheme": s})
        assert code == 200 and v["exit"] == 2 and v["estimate"]["effective_n_workloads"] == 2
        assert _call(port, token, "/run", {"scheme": s})[0] == 409          # refused: unacknowledged
        s["acknowledged"] = [{"id": "no_substrate", "at": "test"}]
        code, v = _call(port, token, "/validate", {"scheme": s})
        assert v["exit"] == 0
        try:
            differ.find_differ()
            have = chain.zstd_available()
        except differ.DifferError:
            have = False
        if not have:
            print("skip: run part needs the differ and zstd")
            return
        code, r = _call(port, token, "/run", {"scheme": s, "speed": 2})
        assert code == 200 and r["label"] == s["label"], r
        assert _call(port, token, "/run", {"scheme": s})[0] == 409          # exists, no force
        st = None
        for _ in range(600):
            code, st = _call(port, token, "/status?label=" + s["label"])
            if st.get("state") in ("done", "failed", "refused", "stopped"):
                break
            time.sleep(0.5)
        assert st and st["state"] == "done", st
        code, res = _call(port, token, "/results?label=" + s["label"] + "&rows=5")
        assert code == 200 and res["n_rows_total"] == 3 * ((7 - 4) // 2 + 1) and res["header"][6] == "mean"
        assert res["sidecar"]["acknowledged"][0]["id"] == "no_substrate"
        code, runs = _call(port, token, "/runs")
        assert any(x["label"] == s["label"] and x["state"] == "done" and x["has_features"] for x in runs["runs"])
        # a second run at another speed (so the L1 store is not reused and there is time to stop it),
        # stopped through the control endpoint right after launch
        s2 = dict(s, label="stop_me")
        code, r = _call(port, token, "/run", {"scheme": s2, "speed": 0})
        assert code == 200
        code, c = _call(port, token, "/control", {"label": "stop_me", "command": "stop"})
        assert code == 200 and c["ok"]
        for _ in range(200):
            code, st = _call(port, token, "/status?label=stop_me")
            if st.get("state") in ("stopped", "done", "failed"):
                break
            time.sleep(0.3)
        assert st["state"] == "stopped", st
        assert _call(port, token, "/control", {"label": "stop_me", "command": "nope"})[0] == 400
        assert _call(port, token, "/control", {"label": "no_such_run", "command": "stop"})[0] == 404


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
