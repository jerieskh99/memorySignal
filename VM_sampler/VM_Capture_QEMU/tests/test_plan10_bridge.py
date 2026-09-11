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


def test_results_summary_and_agg_over_http():
    """The Explore endpoints answer over HTTP for a run directory that already holds features,
    without a differ or a corpus run: summary, one view per kind, two runs as one frame, and
    the refusals (unknown run 404, unknown metric 400)."""
    import numpy as np
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        root = td / "corpus"
        synth.make_corpus(root, n_snapshots=8)
        port, token, out = _start(td, root)
        key_dt = np.dtype([("recording", "U256"), ("workload", "U128"), ("family", "U32"), ("block", "i4"), ("t_index", "i4"), ("seq_start", "i4")])
        for label, base in (("done_a", 1.0), ("done_b", 10.0)):
            d = out / label
            d.mkdir(parents=True)
            rows, keys = [], []
            for fam, wl, seed in (("cpu", "wl_one", 1), ("mem", "wl_two", 2)):
                for t in range(3):
                    rows.append([base + t, 0.5 * (t + 1)])
                    keys.append((f"{fam}/{wl}/args_--seed_{seed}_abcdef12/rep001__x", wl, fam, -1, t, 1 + 4 * t))
            np.savez_compressed(d / "features.npz", X=np.asarray(rows, dtype=np.float32), feature_names=np.array(["mean", "duty"], dtype="U64"), tile_keys=np.array(keys, dtype=key_dt))
            (d / "scheme.json").write_text("{}")
            (d / "status.json").write_text(json.dumps({"state": "done", "label": label}))
            (d / "sidecar.json").write_text(json.dumps({"label": label, "written_at": "2026-09-11T00:00:00+00:00", "speed": 2, "acknowledged": [],
                                                       "source": {"kind": "local", "root": str(root)}, "scheme": {"nodes": [{"module": "cells"}, {"module": "write"}]},
                                                       "recordings": [{"id": k[0]} for k in keys[::3]], "n_rows": len(rows), "n_features": 2}))
        code, runs = _call(port, token, "/runs")
        assert code == 200 and {r["label"]: (r["n_rows"], r["n_features"]) for r in runs["runs"]}["done_a"] == (6, 2)
        code, s = _call(port, token, "/results/summary?label=done_a")
        assert code == 200 and s["n_rows"] == 6 and [f["name"] for f in s["features"]] == ["mean", "duty"]
        assert s["features"][0]["median"] == 2.0 and s["keys"]["family"]["n_unique"] == 2 and s["facts"]["modules"] == ["cells", "write"]
        assert _call(port, token, "/results/summary?label=nope")[0] == 404
        assert _call(port, token, "/results/summary?label=../etc")[0] == 404
        code, d = _call(port, token, "/results/agg?labels=done_a&view=distribution&y=mean&group=family&bins=3")
        assert code == 200 and [g["label"] for g in d["groups"]] == ["cpu", "mem"] and d["groups"][0]["n"] == 3 and d["groups"][0]["median"] == 2.0
        code, t = _call(port, token, "/results/agg?labels=done_a&view=time&y=mean&x=t_index&group=workload&stat=max")
        assert code == 200 and t["series"][0]["x"] == [0, 1, 2] and t["series"][0]["y"] == [1, 2, 3]
        code, m = _call(port, token, "/results/agg?labels=done_a,done_b&view=matrix&y=mean&rows=run&cols=family&stat=mean")
        assert code == 200 and m["row_labels"] == ["done_a", "done_b"] and m["values"] == [[2.0, 2.0], [11.0, 11.0]]
        code, sc = _call(port, token, "/results/agg?labels=done_a&view=scatter&x=mean&y=duty&group=")
        assert code == 200 and sc["groups"][0]["n"] == 6 and sc["groups"][0]["r_pearson"] is not None
        code, tb = _call(port, token, "/results/agg?labels=done_a,done_b&view=table&group=run&stat=median")
        assert code == 200 and tb["rows"][1]["label"] == "done_b" and tb["rows"][1]["values"] == [11.0, 1.0]
        code, e = _call(port, token, "/results/agg?labels=done_a&view=distribution&y=nope")
        assert code == 400 and "nope" in e["error"]
        assert _call(port, token, "/results/agg?labels=done_a,missing&view=table")[0] == 404
        assert _call(port, token, "/results/agg?view=table")[0] == 400
        assert _call(port, token, "/results/agg?labels=done_a&view=table&bins=x")[0] == 400


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
