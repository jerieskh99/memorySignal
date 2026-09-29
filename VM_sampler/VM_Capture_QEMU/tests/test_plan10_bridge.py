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


def test_learn_endpoints_over_http():
    """The Learn routes answer over HTTP: the palette, what runs offer, validation with its
    refusals, a launched run reaching done, its status and results, the frames, the views, and
    a stop through control."""
    import numpy as np
    sys.path.insert(0, str(QEMU_DIR / "tests"))
    import test_plan10_learn as TL
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        root = td / "corpus"
        synth.make_corpus(root, n_snapshots=8)
        port, token, out = _start(td, root)
        TL.make_runs(out)                                   # rows_run, path_run, image_run under the runs dir
        code, reg = _call(port, token, "/learn/modules")
        assert code == 200 and any(m["id"] == "minirocket" and m["available"] for m in reg["modules"])
        code, inp = _call(port, token, "/learn/inputs")
        assert code == 200 and inp["runs"]["rows_run"]["features"] and inp["runs"]["path_run"]["tiles_shape"] == "path" and "floors" in inp["examples"]
        pl = inp["examples"]["floors"]
        pl["label"] = "floors_http"
        pl["slots"][2]["alts"] = [{"module": "logreg"}, {"module": "knn", "params": {"k": 3}}]
        pl["slots"][3]["alts"][0]["params"] = {"null_permutations": 20, "bootstrap": 20}
        pl["run"]["splits"] = ["loco"]
        code, v = _call(port, token, "/learn/validate", {"pipeline": pl})
        assert code == 200 and v["exit"] == 0 and v["estimate"]["configurations"] == 2 and [c["id"] for c in v["configurations"]] == ["c001", "c002"]
        bad = dict(pl, slots=[pl["slots"][0], {"tier": "model", "alts": [{"module": "lstm"}]}] + pl["slots"][3:])
        code, v = _call(port, token, "/learn/validate", {"pipeline": bad})
        assert code == 200 and v["exit"] == 1 and "reads path" in v["verdict"]["hard"][0]["msg"]
        assert _call(port, token, "/learn/run", {"pipeline": bad})[0] == 409
        code, r = _call(port, token, "/learn/run", {"pipeline": pl})
        assert code == 200 and r["label"] == "floors_http", r
        assert _call(port, token, "/learn/run", {"pipeline": pl})[0] == 409          # exists, no force
        st = None
        for _ in range(600):
            code, st = _call(port, token, "/learn/status?label=floors_http")
            if st.get("state") in ("done", "failed", "refused", "stopped"):
                break
            time.sleep(0.3)
        assert st and st["state"] == "done", st
        code, runs = _call(port, token, "/learn/runs")
        assert any(x["label"] == "floors_http" and x["has_results"] for x in runs["runs"])
        code, res = _call(port, token, "/learn/results?label=floors_http")
        assert code == 200 and [c["model"] for c in res["configurations"]] == ["logreg", "knn"]
        assert res["configurations"][0]["splits"]["loco"]["pooled"]["accuracy"] > 0.9
        code, a = _call(port, token, "/learn/agg?label=floors_http&frame=scores&view=table&group=model&stat=mean")
        assert code == 200 and [r["label"] for r in a["rows"]] == ["knn", "logreg"] and "accuracy" in a["features"]
        code, t = _call(port, token, "/learn/agg?label=floors_http&frame=tiles&view=time&y=correct&x=t_index&group=model&stat=mean")
        assert code == 200 and len(t["series"]) == 2
        code, cm = _call(port, token, "/learn/view?label=floors_http&kind=confusion&config=c001&split=loco")
        assert code == 200 and cm["labels"] == ["cpu", "io", "mem"] and cm["n"] == 144
        code, nl = _call(port, token, "/learn/view?label=floors_http&kind=null&config=c001&split=loco")
        assert code == 200 and nl["pooled"]["n"] == 20
        assert _call(port, token, "/learn/view?label=floors_http&kind=saliency&config=c001&split=loco")[0] == 400
        assert _call(port, token, "/learn/view?label=floors_http&kind=nope&config=c001&split=loco")[0] == 400
        code, tile = _call(port, token, "/learn/tile?run=path_run&t_index=2")
        assert code == 200 and tile["shape"] == "path" and len(tile["matrix"]) == 8
        assert _call(port, token, "/learn/tiles_index?run=image_run")[1]["shape"] == "image"
        assert _call(port, token, "/learn/status?label=nope")[0] == 404
        # a second run, stopped through control right after launch
        pl2 = dict(pl, label="stop_learn")
        pl2["slots"][2]["alts"] = [{"module": m} for m in ("logreg", "knn", "rf", "extratrees", "hgb")]
        pl2["run"]["splits"] = ["loro", "lowo", "loco"]
        code, r = _call(port, token, "/learn/run", {"pipeline": pl2})
        assert code == 200
        code, c = _call(port, token, "/learn/control", {"label": "stop_learn", "command": "stop"})
        assert code == 200 and c["ok"]
        for _ in range(200):
            code, st = _call(port, token, "/learn/status?label=stop_learn")
            if st.get("state") in ("stopped", "done", "failed"):
                break
            time.sleep(0.3)
        assert st["state"] in ("stopped", "done"), st
        assert _call(port, token, "/learn/control", {"label": "stop_learn", "command": "nope"})[0] == 400


def test_encoding_panel_endpoints_over_http():
    """The Encoding paper routes: configuration with the driver's own flags, the board, a launch of
    move 0 through the driver, the cells table, the toolkit's files served as they are (and nothing
    outside <out>), the params blocks, the runbook sections, the log and the launch record."""
    import urllib.request
    from plan10_analysis import encoding_panel as EP
    if not EP.toolkit_present():
        print("skip: the toolkit is not on disk")
        return
    sys.path.insert(0, str(QEMU_DIR / "tests"))
    import test_plan10_encoding_panel as TE
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        root = td / "corpus"
        synth.make_corpus(root, n_snapshots=8)
        port, token, out = _start(td, root)
        AB.ST.encoding = EP.Panel(td / "encoding_cfg.json")          # this test's own configuration file, not the user's
        corpus = TE.make_corpus(td / "enc")                          # its own directory: td/corpus holds this test's plan10 corpus
        eout = td / "eout"
        code, c = _call(port, token, "/encoding/config", {"out": str(eout), "root": str(corpus), "preset": "smoke", "flags": {"duration_s": 25.76, "n_jobs": 2}})
        assert code == 200 and c["preset"] == "custom" and c["flags"]["null_perm"] == 20 and any(f["dest"] == "c1_rule" for f in c["driver_flags"])
        assert _call(port, token, "/encoding/config", {"flags": {"nope": 1}})[0] == 400
        code, b = _call(port, token, "/encoding/board")
        assert code == 200 and len(b["moves"]) == 17 and b["moves"][0]["runnable"] and not b["moves"][1]["runnable"] and b["ledger"]["exists"] is False
        code, rb = _call(port, token, "/encoding/runbook")
        assert code == 200 and rb["sections"]["0"]["title"] == "the cell index" and rb["present"]
        code, pt = _call(port, token, "/encoding/plan_text?move=3")
        assert code == 200 and pt["text"].count("gates_calibration gc") == 5
        assert _call(port, token, "/encoding/run", {"move": 2})[0] == 400              # waits for moves 0 and 1
        code, r = _call(port, token, "/encoding/run", {"move": 0})
        assert code == 200 and r["move"] == 0 and r["shell"][0].startswith("python3 -m plan11_encoding_ladder.run_moves run")
        assert _call(port, token, "/encoding/run", {"move": 0})[0] == 400              # one process at a time
        for _ in range(400):
            code, b = _call(port, token, "/encoding/board")
            if not b["running"]:
                break
            time.sleep(0.3)
        assert b["moves"][0]["state"] == "done" and b["moves"][1]["runnable"], b["moves"][0]
        code, lr = _call(port, token, "/encoding/launch?id=" + r["id"])
        assert code == 200 and lr["exit_code"] == 0 and lr["driver_run"]["moves"] == [0] and lr["toolkit"]["content_fingerprint"]
        code, lg = _call(port, token, "/encoding/log?launch=" + r["id"] + "&tail=50")
        assert code == 200 and any("extract index" in ln for ln in lg["lines"]) and lg["running"] is False
        code, cells = _call(port, token, "/encoding/cells")
        assert code == 200 and cells["n_kernel"] == 12 and cells["n_idle"] == 3 and cells["n_other_dirs"] == 1 and cells["rows"][0]["n_pairs"] == ""
        code, t = _call(port, token, "/encoding/text?path=cells.csv")
        assert code == 200 and t["kind"] == "csv" and "cell_id" in t["header"] and t["n_rows"] == 16
        assert _call(port, token, "/encoding/text?path=../cells.csv")[0] == 400
        assert _call(port, token, "/encoding/text?path=/etc/hosts")[0] == 400
        code, ls = _call(port, token, "/encoding/list?path=")
        assert code == 200 and any(e["name"] == "cells.csv" for e in ls["entries"])
        code, v = _call(port, token, "/encoding/views")
        assert code == 200 and any(x["id"] == "calibration" and not x["any"] for x in v["views"])
        code, pb = _call(port, token, "/encoding/params")
        assert code == 200 and pb["driver_params"]["null_perm"] == 20 and any(x["path"] == "cells.index.json" for x in pb["blocks"])
        # the file route: bytes with a content type, nothing outside <out>
        req = urllib.request.Request(f"http://127.0.0.1:{port}/encoding/file?path=cells.csv&token={token}")
        with urllib.request.urlopen(req, timeout=30) as resp:
            assert resp.status == 200 and resp.headers.get("Content-Type", "").startswith("text/csv") and resp.read().startswith(b"cell_id,")
        assert _call(port, token, "/encoding/file?path=../x")[0] == 400
        assert _call(port, token, "/encoding/file?path=report/figures/fig_apf_per_kernel.png")[0] == 404
        assert _call(port, token, "/encoding/stop", {})[0] == 400                      # nothing running


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
