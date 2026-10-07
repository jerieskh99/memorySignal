#!/usr/bin/env python3
"""analysis_bridge.py -- the local backend the console talks to. Stdlib only.

ThreadingHTTPServer bound to 127.0.0.1 with a per-process token (printed once as
`token=...`, never on a command line), the same shape as plan07_campaign/ui/console_bridge.py.
It serves the built console and exposes:

  GET  /health                       liveness
  GET  /manifest                     the manifest of the current source
  POST /source/test    {source}      can the source be reached
  POST /scan           {source}      scan the source; becomes the current manifest
  POST /validate       {scheme}      scheme.py verdict + estimate over the current manifest
  POST /run            {scheme, speed?, max_pairs?, force?}
                                     write runs/<label>/scheme.json and manifest.json, spawn
                                     the executor, return {label, out_dir}
  GET  /status?label=L               status.json + the log tail
  POST /control        {label, command: stop|pause|run}
  GET  /runs                         every run under the output directory
  GET  /results?label=L&rows=N       sidecar + the first N feature rows
  GET  /learn/modules                the Learn palette (registry.py): what is built, unavailable, unbuilt
  GET  /learn/inputs                 what every run offers Learn (rows, tiles, counts) and the worked examples
  POST /learn/validate {pipeline}    pipeline.py verdict, estimate, the configurations of the sweep
  POST /learn/run      {pipeline, force?}   spawn the Learn executor; returns {label, out_dir}
  GET  /learn/runs, /learn/status?label=L, POST /learn/control {label, command}
  GET  /learn/results?label=L        summary of a Learn run (results.py)
  GET  /learn/agg?label=L&frame=scores|tiles&view=...  the five views over a Learn run's frames
  GET  /learn/view?label=L&kind=confusion|curves|embedding|saliency|importance|null|calibration|train&config=&split=
  GET  /learn/tile?run=R&recording=&t_index=  one path or image tile as drawn; /learn/tiles_index?run=R
  GET  /encoding/config, POST /encoding/config {out, root, preset, flags, standalone_tex}
                                     the Encoding paper panel: where the toolkit reads and writes, the driver's flags
  GET  /encoding/board               the moves in runbook order with their ledger states, the running process, the launches
  POST /encoding/run {move, force?}  launch the toolkit's driver for ONE move (`run_moves run --moves N`); POST /encoding/stop
  GET  /encoding/cells, /encoding/views, /encoding/params, /encoding/runbook, /encoding/plan_text?move=N
  GET  /encoding/text?path=P, /encoding/file?path=P, /encoding/list?path=P    the toolkit's files under <out>, as they are
  GET  /encoding/log?launch=ID&tail=N, /encoding/launch?id=ID
  GET  /grounding/config, POST /grounding/config {out, root, preset, flags}
                                     the Grounding paper panel (plan12_grounding): the same shape as /encoding/*
  GET  /grounding/board              the moves 0 to 10 with their record-book states, then 11 and 12 (the idle class check, the new-block test:
                                     their own records in the sibling folders <out>_idle13, <out>_newblocks), the running process, the launches
  POST /grounding/run {move, force?} launch the engine's driver for ONE move; POST /grounding/stop
  GET  /grounding/cells, /grounding/views, /grounding/params, /grounding/runbook, /grounding/plan_text?move=N
  GET  /grounding/text?path=P, /grounding/file?path=P, /grounding/list?path=P    the engine's files under <out>, as they are
  GET  /grounding/log?launch=ID&tail=N, /grounding/launch?id=ID
  GET  /grounding/encoding_table2    read only: the named encoding run's matching numbers next to move 6
  GET  /random/config, POST /random/config {paper}
                                     the Random-signal paper panel: the paper's server_runs/ folder read as it is (never the server)
  GET  /random/board                 the moves (server moves: done when their results are home; laptop moves: run from the tab), the launches
  POST /random/run {move, force?}    run ONE laptop move (analysis/analyze_lag.py, analysis/figures_lag.py); POST /random/stop
  GET  /random/views, /random/text?path=P, /random/file?path=P, /random/list?path=P, /random/log?launch=ID&tail=N, /random/launch?id=ID
  Environment: PLAN10_SERVED_HTML names another served page to read (a scratch build, for a check on a second port)
  GET  /results/summary?label=L      what a run holds: per-metric stats, keys, sidecar facts (Explore)
  GET  /results/agg?labels=A,B&view=V&y=F&x=F|key&group=k1,k2&stat=S&scale=linear|log&bins=N&rows=k&cols=k
                                     one view over one run or several, aggregated in numpy (results_view.py)

Sources are {"kind":"local","root":...} or {"kind":"ssh","host":...,"user":...,"key":...,
"remote_root":...,"port":22}. The executor runs as a subprocess so a run outlives a page
reload; the bridge only reads its status file. Nothing here runs analysis in-process.

Run:  python3 plan10_analysis/ui/analysis_bridge.py [--port 8766] [--source-json S] [--out-dir D] [--open]
"""
from __future__ import annotations

import argparse
import json
import shlex
import re
import os
import secrets
import shutil
import subprocess
import sys
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent            # .../plan10_analysis/ui
PKG = HERE.parent
QEMU_DIR = PKG.parent
sys.path.insert(0, str(QEMU_DIR))

from plan10_analysis import channel_roster, corpus_manifest, scheme as S   # noqa: E402
from plan10_analysis import results_view as RV                            # noqa: E402
from plan10_analysis.learn import executor as LE, pipeline as LP, registry as LREG, results as LR   # noqa: E402
from plan10_analysis import encoding_panel as EPN                         # noqa: E402
from plan10_analysis import grounding_panel as GPN                        # noqa: E402
from plan10_analysis import random_signal_panel as RSP                    # noqa: E402
from plan10_analysis.runner import trajectory, extract                     # noqa: E402
from plan10_analysis.modules import build_modules                         # noqa: E402
from plan10_analysis.sources import SourceError, make_source              # noqa: E402

SERVED_HTML = Path(os.environ["PLAN10_SERVED_HTML"]) if os.environ.get("PLAN10_SERVED_HTML") else HERE / "analysis_console.served.html"
BUILD = HERE / "build_analysis_console.py"
EXECUTOR = PKG / "runner" / "executor.py"
LEARN_EXECUTOR = PKG / "learn" / "executor.py"
DEFAULT_OUT = Path(os.path.expanduser("~/.cache/plan10/runs"))
LABEL_OK = __import__("re").compile(r"^[A-Za-z0-9_-]+$")


class State:
    def __init__(self, out_dir: Path, source: dict, store: Path | None, learn_dir: Path | None = None):
        self.out_dir = out_dir
        self.learn_dir = learn_dir or (out_dir.parent / "learn")
        self.encoding = EPN.Panel()
        self.grounding = GPN.Panel()
        self.random_signal = RSP.Panel()
        self.store = store
        self.source = source
        self.manifest: dict | None = None
        self.lock = threading.Lock()
        self.procs: dict[str, subprocess.Popen] = {}
        self.ctx_cache: tuple | None = None
        self.traj_cols: dict[str, list[str]] = {}   # recording id -> its trajectory's header, once read

    def context(self) -> S.Context:
        if self.manifest is None:
            raise SourceError("no manifest yet; scan a source first")
        key = (id(self.manifest),)
        if self.ctx_cache and self.ctx_cache[0] == key:
            return self.ctx_cache[1]
        ctx = S.Context(channel_roster.build_roster(), self.manifest, build_modules(), S.load_config())
        self.ctx_cache = (key, ctx)
        return ctx


ST: State


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------

def ep_health(_q, _b):
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "out_dir": str(ST.out_dir)}


def ep_manifest(_q, _b):
    if ST.manifest is None:
        return {"error": "no manifest yet; POST /scan"}, 404
    return ST.manifest


def ep_source_test(_q, body):
    try:
        src = make_source(body.get("source") or ST.source)
        ok, msg = src.test()
        out = {"ok": ok, "message": msg, "source": src.describe()}
        if ok and getattr(src, "mode", None) == "remote":
            # a remote run needs more than a reachable host: report what the server has
            try:
                probe = src.probe_remote()
                missing = [k for k, got in (("numpy", probe.get("numpy")), ("zstd", probe.get("zstd")),
                                            ("differ", "error" not in (probe.get("differ") or {})),
                                            ("trace root", probe.get("root_exists"))) if not got]
                out["probe"] = probe
                out["ok"] = not missing
                out["message"] = (msg + "; server ready" if not missing
                                  else msg + "; server is missing " + ", ".join(missing))
            except SourceError as e:
                out["ok"] = False
                out["message"] = f"{msg}; remote probe failed: {e}"
        return out
    except SourceError as e:
        return {"ok": False, "message": str(e)}


def ep_scan(_q, body):
    """Read the archive's own manifest (default: one file, instant), or with reconcile=true walk
    the archive and rewrite that manifest from what is actually there (the Scan button)."""
    try:
        src = make_source(body.get("source") or ST.source)
        m = corpus_manifest.scan_source(src, reconcile=bool(body.get("reconcile")))
    except (SourceError, corpus_manifest.CorpusMissing) as e:
        return {"error": str(e)}, 400
    with ST.lock:
        ST.source = src.describe()
        ST.manifest = m
        ST.ctx_cache = None
    return m


def ep_validate(_q, body):
    sch = body.get("scheme")
    if not isinstance(sch, dict):
        return {"error": "scheme missing"}, 400
    try:
        ctx = ST.context()
    except SourceError as e:
        return {"error": str(e)}, 400
    issues = S.validate(sch, ctx)
    code, v = S.verdict(issues, sch)
    return {"exit": code, "verdict": v, "estimate": S.estimate(sch, ctx), "issues": issues}


def ep_run(_q, body):
    sch = body.get("scheme")
    if not isinstance(sch, dict):
        return {"error": "scheme missing"}, 400
    label = str(sch.get("label") or "")
    if not LABEL_OK.match(label):
        return {"error": f"label must match {LABEL_OK.pattern}"}, 400
    try:
        ctx = ST.context()
    except SourceError as e:
        return {"error": str(e)}, 400
    code, v = S.verdict(S.validate(sch, ctx), sch)
    if code:
        return {"error": "scheme refused", "exit": code, "verdict": {k: v[k] for k in ("hard", "soft_unacknowledged")}}, 409
    run_dir = ST.out_dir / label
    if run_dir.exists() and not body.get("force"):
        return {"error": f"run {label!r} already exists; choose another label or pass force"}, 409
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    (run_dir / "scheme.json").write_text(json.dumps(sch, indent=1))
    (run_dir / "manifest.json").write_text(json.dumps(ST.manifest))
    (run_dir / "source.json").write_text(json.dumps(ST.source))
    argv = [sys.executable, str(EXECUTOR), "run", str(run_dir / "scheme.json"), "--out-dir", str(ST.out_dir),
            "--source-json", json.dumps(ST.source), "--manifest", str(run_dir / "manifest.json")]
    if ST.store:
        argv += ["--store", str(ST.store)]
    if body.get("speed") is not None:
        argv += ["--speed", str(int(body["speed"]))]
    if body.get("max_pairs"):
        argv += ["--max-pairs", str(int(body["max_pairs"]))]
    if body.get("keep_fetched"):
        argv += ["--keep-fetched"]
    log = (run_dir / "bridge.log").open("a")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    p = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, cwd=str(QEMU_DIR), env=env)
    with ST.lock:
        ST.procs[label] = p
    return {"label": label, "out_dir": str(run_dir), "pid": p.pid, "argv": argv}


def _status_of(run_dir: Path) -> dict:
    st = run_dir / "status.json"
    d = json.loads(st.read_text()) if st.exists() else {"state": "starting", "label": run_dir.name}
    p = ST.procs.get(run_dir.name)
    if p is not None:
        rc = p.poll()
        d["process"] = "running" if rc is None else f"exited {rc}"
        if rc is not None and d.get("state") in ("starting", "running"):
            d["state"] = "failed" if rc else "done"
    return d


def ep_status(q, _b):
    label = (q.get("label") or [""])[0]
    run_dir = ST.out_dir / label
    if not label or not run_dir.is_dir():
        return {"error": "unknown run"}, 404
    d = _status_of(run_dir)
    n = int((q.get("lines") or ["40"])[0])
    log = run_dir / "run.log"
    d["log_tail"] = log.read_text().splitlines()[-n:] if log.exists() else []
    blog = run_dir / "bridge.log"
    if blog.exists() and not log.exists():
        d["log_tail"] = blog.read_text().splitlines()[-n:]
    return d


def ep_trajectory_columns(q, _b):
    """Header of a recording's substrate trajectory: the channels it can serve without a re-diff.

    Read in place -- locally, or over ssh in one round trip -- never fetched. The header is the
    honest inventory: the manifest knows a trajectory exists, only the file knows what it holds.
    """
    rid = (q.get("rec") or [""])[0]
    if ST.manifest is None:
        return {"error": "no manifest yet; scan a source first"}, 400
    rec = next((r for r in ST.manifest.get("recordings", []) if r["id"] == rid), None)
    if rec is None:
        return {"error": "unknown recording"}, 404
    if rid in ST.traj_cols:
        return {"rec": rid, "columns": ST.traj_cols[rid], "cached": True}
    paths = rec["has"].get("substrate_csv_paths") or []
    if not paths:
        return {"rec": rid, "trajectory": None, "columns": []}
    rel = paths[0]
    try:
        src = make_source(ST.source)
    except SourceError as e:
        return {"error": str(e)}, 400
    if rec["has"].get("substrate_join") == "in-chain" and src.kind == "ssh":
        f = shlex.quote(src.remote_root + "/" + rel)
        cmd = (f"zstd -dc {f} 2>/dev/null | head -1" if rel.endswith(".zst")
               else f"gzip -dc {f} 2>/dev/null | head -1" if rel.endswith(".gz") else f"head -1 {f}")
        try:
            r = subprocess.run(src.ssh_argv(cmd), capture_output=True, text=True, timeout=90)
        except subprocess.TimeoutExpired:
            return {"error": "timed out reading the trajectory header on the server"}, 504
        head = (r.stdout.splitlines() or [""])[0].split(",")
        if len(head) < 3 or head[0] != "seq":
            return {"error": f"not a substrate trajectory header: {head[:3]}"}, 502
        cols = [c for c in head[2:] if c]
    else:
        fp = Path(rel) if Path(rel).is_absolute() else Path(getattr(src, "root", ".")) / rel
        try:
            cols = trajectory.columns(fp)
        except (trajectory.TrajectoryError, OSError) as e:
            return {"error": str(e)}, 502
    ST.traj_cols[rid] = cols
    return {"rec": rid, "trajectory": rel, "columns": cols}


def ep_rundetail(q, _b):
    """Everything about a run that is not its live status: what it was pointed at, what it
    selected, and what it has produced. status.json carries the moving parts; this carries
    the fixed ones, so the monitor can name traces instead of numbering them."""
    label = (q.get("label") or [""])[0]
    run_dir = ST.out_dir / label
    if not label or not run_dir.is_dir():
        return {"error": "unknown run"}, 404

    def _load(name):
        f = run_dir / name
        try:
            return json.loads(f.read_text()) if f.exists() else None
        except (json.JSONDecodeError, OSError):
            return None

    src = _load("source.json") or {}
    sch = _load("scheme.json") or {}
    man = _load("manifest.json") or {}

    # the recordings the scheme selected, resolved against the manifest it ran on
    sel, chans = [], []
    for n in (sch.get("nodes") or []):
        pr = n.get("params") or {}
        if n.get("module") == "cells":
            sel = list(pr.get("sel") or [])
        elif n.get("module") == "channels":
            chans += list(pr.get("chans") or [])
    by_id = {r["id"]: r for r in (man.get("recordings") or [])}
    # An ssh source rsyncs each chain into a local cache before extracting, and that fetch
    # reports no progress of its own -- the pair counter stays 0 throughout it. Size the cache
    # against the archive's own byte count so the monitor can show the half that is moving.
    # Both sides count only NNNNNN.zst, the same rule corpus_manifest uses for "bytes", so a
    # substrate CSV riding along in the chain directory cannot skew the ratio.
    cache = None
    if src.get("kind") == "ssh" and src.get("cache"):
        cache = Path(os.path.expanduser(str(src["cache"])))
    snap = re.compile(r"^\d{6}\.zst$")

    def _fetched(rid):
        """(snapshot files, snapshot bytes, trajectory bytes incl. an in-flight rsync temp)"""
        if cache is None:
            return None, None, None
        d = cache / rid
        if not d.is_dir():
            return 0, 0, 0
        n = b = t = 0
        try:
            for f in d.iterdir():
                if not f.is_file():
                    continue
                if snap.match(f.name):
                    n += 1
                    b += f.stat().st_size
                elif "substrate_trajectory" in f.name:      # done, or rsync's dotted temp file
                    t += f.stat().st_size
        except OSError:
            pass
        return n, b, t

    # size of the trajectory on the archive side, recorded by the scan: what a fast-path fetch pulls
    traj_bytes = {r["id"]: (r.get("has") or {}).get("substrate_csv_bytes")
                  for r in (man.get("recordings") or []) if (r.get("has") or {}).get("substrate_csv")}

    cells = []
    for rid in sel:
        r = by_id.get(rid) or {}
        fn, fb, ft = _fetched(rid)
        cells.append({"id": rid, "workload": r.get("workload"), "family": r.get("family"),
                      "rep": r.get("rep"), "run_label": r.get("run_label"),
                      "variant": (r.get("variant") or {}).get("raw"),
                      "n_pairs": r.get("n_pairs"), "bytes": r.get("bytes"),
                      "n_snapshots": r.get("n_snapshots"),
                      "fetched_files": fn, "fetched_bytes": fb,
                      "has_trajectory": rid in traj_bytes, "trajectory_bytes": traj_bytes.get(rid),
                      "fetched_trajectory_bytes": ft})

    # speed is an executor argv, not a status field; the run log states it on its first line
    speed = None
    log = run_dir / "run.log"
    if log.exists():
        try:
            head = log.open().readline()
            m = re.search(r"speed (\d+)", head)
            if m:
                speed = int(m.group(1))
        except OSError:
            pass

    files = []
    for f in sorted(run_dir.iterdir()):
        if f.is_file():
            files.append({"name": f.name, "bytes": f.stat().st_size})

    nodes = [{"id": n.get("id"), "module": n.get("module")} for n in (sch.get("nodes") or [])]
    return {"label": label, "source": src, "speed": speed, "channels": sorted(set(chans)),
            "cells": cells, "n_cells": len(cells), "nodes": nodes,
            "n_pipes": len(sch.get("pipes") or []), "files": files,
            "acknowledged": sch.get("acknowledged") or [],
            "run_dir": str(run_dir), "manifest_root": man.get("root")}


def ep_control(_q, body):
    label, cmd = str(body.get("label") or ""), body.get("command")
    run_dir = ST.out_dir / label
    if not run_dir.is_dir():
        return {"error": "unknown run"}, 404
    if cmd not in ("stop", "pause", "run"):
        return {"error": "command must be stop, pause or run"}, 400
    tmp = run_dir / "control.json.tmp"
    tmp.write_text(json.dumps({"command": cmd, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}))
    os.replace(tmp, run_dir / "control.json")
    return {"ok": True, "label": label, "command": cmd}


_SNAP_RE = re.compile(r"^\d{6}\.zst$")


def _cache_scan(label: str = ""):
    """Every recording in the fetch cache, with what would make it droppable.

    Droppable = snapshots present, a COMPLETE L1 store for it (any run: a relaunch with force
    forgets what its first attempt extracted, and the chain is no less finished for that), and
    no running run on it. Whether the ARCHIVE still holds it is a separate, ssh question that
    only the drop itself asks.
    """
    src_spec = ST.source
    if label and (ST.out_dir / label / "source.json").is_file():
        try:
            src_spec = json.loads((ST.out_dir / label / "source.json").read_text())
        except (OSError, json.JSONDecodeError):
            pass
    if (src_spec or {}).get("kind") != "ssh" or not src_spec.get("cache"):
        return src_spec, None, []
    cache = Path(os.path.expanduser(str(src_spec["cache"])))
    if not cache.is_dir():
        return src_spec, cache, []
    active: set[str] = set()
    if ST.out_dir.is_dir():
        for rd in ST.out_dir.iterdir():
            f = rd / "status.json"
            if f.is_file():
                try:
                    st = json.loads(f.read_text())
                    if st.get("state") in ("starting", "running") and st.get("recording"):
                        active.add(st["recording"])
                except (OSError, json.JSONDecodeError):
                    pass
    extracted: set[str] = set()
    for meta in extract.store_dir(ST.store).glob("*.meta.json"):
        try:
            m = json.loads(meta.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if m.get("complete") and m.get("rec_id"):
            extracted.add(m["rec_id"])
    out = []
    for d in sorted(p for p in cache.rglob("rep0*__*") if p.is_dir()):
        rid = str(d.relative_to(cache))
        n = b = t = 0
        for f in d.iterdir():
            if not f.is_file():
                continue
            if _SNAP_RE.match(f.name):
                n += 1
                b += f.stat().st_size
            elif "substrate_trajectory" in f.name:
                t += f.stat().st_size
        why = (None if (n and rid in extracted and rid not in active)
               else "trajectory only; a re-run re-fetches it" if not n
               else "a running run is on this recording" if rid in active
               else "no complete L1 store for it; dropping would force a re-fetch")
        out.append({"recording": rid, "snapshots": n, "bytes": b, "trajectory_bytes": t,
                    "extracted": rid in extracted, "active": rid in active,
                    "droppable": why is None, "why": why})
    return src_spec, cache, out


def ep_cache_status(q, _b):
    """What the local fetch cache holds and what could be reclaimed; no ssh, no deletion."""
    label = (q.get("label") or [""])[0]
    _, cache, items = _cache_scan(label)
    drop = [i for i in items if i["droppable"]]
    return {"cache": str(cache) if cache else None, "items": items,
            "n_droppable": len(drop), "droppable_bytes": sum(i["bytes"] for i in drop),
            "cached_bytes": sum(i["bytes"] + i["trajectory_bytes"] for i in items)}


def ep_cache_drop(_q, body):
    """Delete fetched chains from the local cache, but only where the archive still holds them.

    An ssh fetch is a read-only copy: the archive is authoritative and nothing here writes back
    to it, so "returning" a chain is meaningless -- the only safe question is whether the archive
    copy is still intact. Each droppable candidate (see _cache_scan) is re-stat'ed on the server,
    not trusted from the scan, and its snapshots are deleted only when the remote count and byte
    total match the local ones exactly. The trajectory CSV beside them stays.
    """
    label = str(body.get("label") or "")
    only = body.get("recording")
    src_spec, cache, items = _cache_scan(label)
    if cache is None:
        return {"error": "not an ssh fetch source; nothing is cached locally"}, 400
    try:
        src = make_source(src_spec)
    except SourceError as e:
        return {"error": str(e)}, 400

    def remote_stats(rid):
        cmd = (f"cd {shlex.quote(src.remote_root + '/' + rid)} 2>/dev/null && "
               "find . -maxdepth 1 -name '[0-9][0-9][0-9][0-9][0-9][0-9].zst' -printf '%s\\n' "
               "| awk '{n++; b+=$1} END {print (n+0), (b+0)}'")
        r = subprocess.run(src.ssh_argv(cmd), capture_output=True, text=True, timeout=120)
        parts = (r.stdout or "").split()
        if r.returncode != 0 or len(parts) != 2:
            return None
        return int(parts[0]), int(parts[1])

    results, freed = [], 0
    for it in items:
        rid = it["recording"]
        if only and rid != only:
            continue
        if not it["droppable"]:
            results.append({"recording": rid, "action": "kept", "why": it["why"]})
            continue
        rem = remote_stats(rid)
        if rem is None:
            results.append({"recording": rid, "action": "kept", "why": "could not read the archive copy"})
            continue
        if rem != (it["snapshots"], it["bytes"]):
            results.append({"recording": rid, "action": "kept",
                            "why": f"archive {rem[0]} files/{rem[1]} B vs local {it['snapshots']}/{it['bytes']}"})
            continue
        for f in (cache / rid).iterdir():
            if f.is_file() and _SNAP_RE.match(f.name):
                f.unlink()
        freed += it["bytes"]
        results.append({"recording": rid, "action": "dropped", "files": it["snapshots"], "bytes": it["bytes"],
                        "verified": f"archive {rem[0]} files/{rem[1]} B == local"})
    return {"label": label, "results": results, "freed_bytes": freed,
            "n_dropped": sum(1 for r in results if r["action"] == "dropped")}


def ep_runs(_q, _b):
    out = []
    if ST.out_dir.is_dir():
        for d in sorted(ST.out_dir.iterdir()):
            if d.is_dir() and (d / "scheme.json").exists():
                s = _status_of(d)
                item = {"label": d.name, "state": s.get("state"), "updated_at": s.get("updated_at"), "message": s.get("message"),
                        "has_features": (d / "features.npz").exists()}
                try:
                    side = json.loads((d / "sidecar.json").read_text()) if (d / "sidecar.json").exists() else {}
                    item.update(n_rows=side.get("n_rows"), n_features=side.get("n_features"), written_at=side.get("written_at"))
                except (json.JSONDecodeError, OSError):
                    pass
                out.append(item)
    return {"runs": out}


def ep_results(q, _b):
    label = (q.get("label") or [""])[0]
    run_dir = ST.out_dir / label
    side = run_dir / "sidecar.json"
    if not side.exists():
        return {"error": "no results for this run"}, 404
    n = int((q.get("rows") or ["50"])[0])
    csv_path = run_dir / "features.csv"
    lines = csv_path.read_text().splitlines() if csv_path.exists() else []
    return {"sidecar": json.loads(side.read_text()), "header": lines[0].split(",") if lines else [],
            "rows": [l.split(",") for l in lines[1:n + 1]], "n_rows_total": max(0, len(lines) - 1),
            "files": {k: str(run_dir / k) for k in ("features.npz", "features.csv", "sidecar.json") if (run_dir / k).exists()}}


def _run_dir_of(label: str):
    if not label or not LABEL_OK.match(label):
        return None
    d = ST.out_dir / label
    return d if d.is_dir() else None


def ep_results_summary(q, _b):
    """What one run holds, for the Explore view: every metric's statistics over all rows, the
    keys and their cardinalities, and the sidecar facts a figure must not be separated from."""
    label = (q.get("label") or [""])[0]
    run_dir = _run_dir_of(label)
    if run_dir is None:
        return {"error": "unknown run"}, 404
    try:
        return RV.summary(RV.get_run(run_dir))
    except RV.ViewError as e:
        return {"error": str(e)}, 400


def ep_results_agg(q, _b):
    """One view over one run or several, computed in numpy from features.npz. The page draws
    what comes back; it never aggregates rows itself. Unknown metric, key, view or stat is a 400
    that names what exists."""
    g = lambda k, d="": (q.get(k) or [d])[0]
    labels = [x for x in g("labels").split(",") if x]
    if not labels:
        return {"error": "labels required"}, 400
    runs = []
    for lab in labels:
        d = _run_dir_of(lab)
        if d is None:
            return {"error": f"unknown run {lab!r}"}, 404
        try:
            runs.append(RV.get_run(d))
        except RV.ViewError as e:
            return {"error": str(e)}, 400
    try:
        bins = int(g("bins", "30"))
    except ValueError:
        return {"error": "bins must be an integer"}, 400
    try:
        return RV.aggregate(runs, g("view", "distribution"), y=g("y") or None, x=g("x") or None,
                            group=[k for k in g("group").split(",") if k], stat=g("stat", "median"),
                            scale=g("scale", "linear"), bins=bins, rows=g("rows") or None, cols=g("cols") or None)
    except RV.ViewError as e:
        return {"error": str(e)}, 400


# ---------------------------------------------------------------------------
# Learn: pipelines over what runs wrote
# ---------------------------------------------------------------------------

def _learn_ctx():
    return LE.context_for(ST.out_dir)


def ep_learn_modules(_q, _b):
    return LREG.build_registry()


def ep_learn_inputs(_q, _b):
    ctx = _learn_ctx()
    return {"runs": ctx.runs, "examples": LP.examples(ctx), "runs_root": str(ST.out_dir), "learn_dir": str(ST.learn_dir)}


def ep_learn_validate(_q, body):
    pl = body.get("pipeline")
    if not isinstance(pl, dict):
        return {"error": "pipeline missing"}, 400
    ctx = _learn_ctx()
    issues = LP.validate(pl, ctx)
    code, v = LP.verdict(issues, pl)
    confs = LP.configurations(pl) if code != 1 else []
    return {"exit": code, "verdict": v, "issues": issues, "estimate": LP.estimate(pl, ctx),
            "configurations": [{"id": c["id"], "name": c["name"]} for c in confs]}


def ep_learn_run(_q, body):
    pl = body.get("pipeline")
    if not isinstance(pl, dict):
        return {"error": "pipeline missing"}, 400
    label = str(pl.get("label") or "")
    if not LABEL_OK.match(label):
        return {"error": f"label must match {LABEL_OK.pattern}"}, 400
    ctx = _learn_ctx()
    code, v = LP.verdict(LP.validate(pl, ctx), pl)
    if code:
        return {"error": "pipeline refused", "exit": code, "verdict": {k: v[k] for k in ("hard", "soft_unacknowledged")}}, 409
    run_dir = ST.learn_dir / label
    if run_dir.exists() and not body.get("force"):
        return {"error": f"Learn run {label!r} already exists; choose another label or pass force"}, 409
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    (run_dir / "pipeline.json").write_text(json.dumps(pl, indent=1))
    argv = [sys.executable, str(LEARN_EXECUTOR), "run", str(run_dir / "pipeline.json"), "--out-dir", str(ST.learn_dir), "--runs-root", str(ST.out_dir)]
    log = (run_dir / "bridge.log").open("a")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    p = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, cwd=str(QEMU_DIR), env=env)
    with ST.lock:
        ST.procs["learn:" + label] = p
    return {"label": label, "out_dir": str(run_dir), "pid": p.pid, "argv": argv}


def _learn_dir_of(label: str):
    if not label or not LABEL_OK.match(label):
        return None
    d = ST.learn_dir / label
    return d if d.is_dir() else None


def _learn_status_of(run_dir: Path) -> dict:
    st = run_dir / "status.json"
    d = json.loads(st.read_text()) if st.exists() else {"state": "starting", "label": run_dir.name}
    p = ST.procs.get("learn:" + run_dir.name)
    if p is not None:
        rc = p.poll()
        d["process"] = "running" if rc is None else f"exited {rc}"
        if rc is not None and d.get("state") in ("starting", "running"):
            d["state"] = "failed" if rc else "done"
    return d


def ep_learn_runs(_q, _b):
    out = []
    if ST.learn_dir.is_dir():
        for d in sorted(ST.learn_dir.iterdir()):
            if d.is_dir() and (d / "pipeline.json").exists():
                s = _learn_status_of(d)
                item = {"label": d.name, "state": s.get("state"), "updated_at": s.get("updated_at"), "message": s.get("message"),
                        "has_results": (d / "learn_results.json").exists(), "n_configurations": s.get("n_configurations"),
                        "fits_done": s.get("fits_done"), "fits_total": s.get("fits_total")}
                out.append(item)
    return {"runs": out}


def ep_learn_status(q, _b):
    label = (q.get("label") or [""])[0]
    run_dir = _learn_dir_of(label)
    if run_dir is None:
        return {"error": "unknown Learn run"}, 404
    d = _learn_status_of(run_dir)
    n = int((q.get("lines") or ["40"])[0])
    log = run_dir / "run.log"
    d["log_tail"] = log.read_text().splitlines()[-n:] if log.exists() else []
    blog = run_dir / "bridge.log"
    if blog.exists() and not log.exists():
        d["log_tail"] = blog.read_text().splitlines()[-n:]
    return d


def ep_learn_control(_q, body):
    label, cmd = str(body.get("label") or ""), body.get("command")
    run_dir = _learn_dir_of(label)
    if run_dir is None:
        return {"error": "unknown Learn run"}, 404
    if cmd not in ("stop", "pause", "run"):
        return {"error": "command must be stop, pause or run"}, 400
    tmp = run_dir / "control.json.tmp"
    tmp.write_text(json.dumps({"command": cmd, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}))
    os.replace(tmp, run_dir / "control.json")
    return {"ok": True, "label": label, "command": cmd}


def ep_learn_results(q, _b):
    run_dir = _learn_dir_of((q.get("label") or [""])[0])
    if run_dir is None:
        return {"error": "unknown Learn run"}, 404
    try:
        return LR.summary(run_dir)
    except LR.ResultsError as e:
        return {"error": str(e)}, 400


def ep_learn_agg(q, _b):
    g = lambda k, d="": (q.get(k) or [d])[0]
    run_dir = _learn_dir_of(g("label"))
    if run_dir is None:
        return {"error": "unknown Learn run"}, 404
    try:
        bins = int(g("bins", "30"))
    except ValueError:
        return {"error": "bins must be an integer"}, 400
    try:
        return LR.agg(run_dir, g("frame", "scores"), g("view", "distribution"), y=g("y") or None, x=g("x") or None,
                      group=[k for k in g("group").split(",") if k], stat=g("stat", "median"), scale=g("scale", "linear"), bins=bins,
                      rows=g("rows") or None, cols=g("cols") or None)
    except (LR.ResultsError, RV.ViewError) as e:
        return {"error": str(e)}, 400


def ep_learn_view(q, _b):
    g = lambda k, d="": (q.get(k) or [d])[0]
    run_dir = _learn_dir_of(g("label"))
    if run_dir is None:
        return {"error": "unknown Learn run"}, 404
    kind, config, split = g("kind"), g("config"), g("split")
    fns = {"confusion": lambda: LR.confusion(run_dir, config, split, g("fold") or None), "curves": lambda: LR.curves(run_dir, config, split),
           "embedding": lambda: LR.embedding(run_dir, config, split, g("method", "pca")), "saliency": lambda: LR.saliency(run_dir, config, split),
           "importance": lambda: LR.importance(run_dir, config, split), "null": lambda: LR.null(run_dir, config, split),
           "calibration": lambda: LR.calibration(run_dir, config, split), "train": lambda: LR.train_curves(run_dir, config, split)}
    if kind not in fns:
        return {"error": f"kind must be one of {', '.join(fns)}"}, 400
    try:
        return fns[kind]()
    except LR.ResultsError as e:
        return {"error": str(e)}, 400


def ep_learn_tile(q, _b):
    g = lambda k, d="": (q.get(k) or [d])[0]
    run = g("run")
    if not run or not LABEL_OK.match(run) or not (ST.out_dir / run).is_dir():
        return {"error": "unknown run"}, 404
    try:
        idx = g("index")
        return LR.tile(ST.out_dir, run, g("recording") or None, int(g("t_index")) if g("t_index") else None, int(idx) if idx else None)
    except (LR.ResultsError, ValueError) as e:
        return {"error": str(e)}, 400


def ep_learn_tiles_index(q, _b):
    run = (q.get("run") or [""])[0]
    if not run or not LABEL_OK.match(run) or not (ST.out_dir / run).is_dir():
        return {"error": "unknown run"}, 404
    try:
        return LR.tiles_index(ST.out_dir, run)
    except LR.ResultsError as e:
        return {"error": str(e)}, 400


# ---------------------------------------------------------------------------
# Encoding paper: the plan11_encoding_ladder toolkit, launched and read, never reimplemented
# ---------------------------------------------------------------------------

def _enc(fn):
    try:
        return fn()
    except EPN.PanelError as e:
        return {"error": str(e)}, 400
    except FileNotFoundError as e:
        return {"error": f"missing: {e}"}, 404


def ep_enc_config(_q, _b):
    return _enc(lambda: ST.encoding.config())


def ep_enc_set_config(_q, body):
    return _enc(lambda: ST.encoding.set_config(body or {}))


def ep_enc_board(_q, _b):
    return _enc(lambda: ST.encoding.board())


def ep_enc_run(_q, body):
    move = (body or {}).get("move")
    if move is None:
        return {"error": "move required"}, 400
    return _enc(lambda: ST.encoding.launch(move, bool((body or {}).get("force")), (body or {}).get("to_move")))


def ep_enc_stop(_q, _b):
    return _enc(lambda: ST.encoding.stop())


def ep_enc_cells(_q, _b):
    return _enc(lambda: EPN.cells(Path(ST.encoding.cfg.get("out") or "")) if ST.encoding.cfg.get("out") else {"exists": False, "why": "set <out> first"})


def ep_enc_views(_q, _b):
    def go():
        out = ST.encoding.cfg.get("out") or ""
        if not out:
            return {"views": [], "why": "set <out> first"}
        return {"views": EPN.views(Path(out), []), "out": out}
    return _enc(go)


def ep_enc_params(_q, _b):
    return _enc(lambda: EPN.params_blocks(Path(ST.encoding.cfg.get("out") or "")) if ST.encoding.cfg.get("out") else {"blocks": [], "why": "set <out> first"})


def ep_enc_runbook(_q, _b):
    return {"sections": {str(k): v for k, v in EPN.runbook_sections().items()}, "path": str(EPN.RUNBOOK), "present": EPN.RUNBOOK.exists()}


def ep_enc_plan_text(q, _b):
    mv = (q.get("move") or ["0"])[0]
    def go():
        c = ST.encoding.cfg
        if not c.get("out"):
            raise EPN.PanelError("set <out> first")
        return {"move": mv, "text": EPN.plan_text(c["out"], c.get("root") or "<root required>", c.get("flags", {}), mv)}
    return _enc(go)


def ep_enc_text(q, _b):
    rel = (q.get("path") or [""])[0]
    return _enc(lambda: EPN.text_file(Path(ST.encoding.cfg.get("out") or ""), rel))


def ep_enc_list(q, _b):
    rel = (q.get("path") or [""])[0]
    return _enc(lambda: EPN.listing(Path(ST.encoding.cfg.get("out") or ""), rel))


def ep_enc_file(q, _b):
    """The toolkit's file bytes (a figure, a PDF, a CSV) from under <out>; nothing outside it."""
    rel = (q.get("path") or [""])[0]
    try:
        p = EPN.safe_path(Path(ST.encoding.cfg.get("out") or ""), rel)
    except EPN.PanelError as e:
        return {"error": str(e)}, 400
    if not p.is_file():
        return {"error": f"no such file under the output root: {rel}"}, 404
    ctype = EPN.BINARY_TYPES.get(p.suffix.lower()) or {"csv": "text/csv", "json": "application/json", "md": "text/markdown", "tex": "text/plain",
                                                         "text": "text/plain"}.get(EPN.TEXT_KINDS.get(p.suffix.lower(), "text"), "application/octet-stream")
    return p.read_bytes(), 200, ctype


def ep_enc_log(q, _b):
    lid = (q.get("launch") or [""])[0] or None
    n = int((q.get("tail") or ["200"])[0])
    return _enc(lambda: ST.encoding.log_tail(lid, n))


def ep_enc_launch(q, _b):
    lid = (q.get("id") or [""])[0]
    return _enc(lambda: ST.encoding.launch_record(lid))


# ---------------------------------------------------------------------------
# Grounding paper: the plan12_grounding engine, launched and read, never reimplemented (the Encoding panel's pattern)
# ---------------------------------------------------------------------------

def _gp(fn):
    try:
        return fn()
    except GPN.PanelError as e:
        return {"error": str(e)}, 400
    except FileNotFoundError as e:
        return {"error": f"missing: {e}"}, 404


def _gp_out() -> Path:
    """The panel's <out>; a PanelError while it is unset, so no route ever serves the bridge's own
    working directory in its place."""
    out = ST.grounding.cfg.get("out") or ""
    if not out:
        raise GPN.PanelError("set <out> first")
    return Path(os.path.expanduser(out))


def ep_gp_config(_q, _b):
    return _gp(lambda: ST.grounding.config())


def ep_gp_set_config(_q, body):
    return _gp(lambda: ST.grounding.set_config(body or {}))


def ep_gp_board(_q, _b):
    return _gp(lambda: ST.grounding.board())


def ep_gp_run(_q, body):
    move = (body or {}).get("move")
    if move is None:
        return {"error": "move required"}, 400
    return _gp(lambda: ST.grounding.launch(move, bool((body or {}).get("force"))))


def ep_gp_stop(_q, _b):
    return _gp(lambda: ST.grounding.stop())


def ep_gp_cells(_q, _b):
    return _gp(lambda: GPN.cells(_gp_out()) if ST.grounding.cfg.get("out") else {"exists": False, "why": "set <out> first"})


def ep_gp_views(_q, _b):
    return _gp(lambda: {"views": GPN.views(_gp_out()), "out": str(_gp_out())} if ST.grounding.cfg.get("out") else {"views": [], "why": "set <out> first"})


def ep_gp_params(_q, _b):
    return _gp(lambda: GPN.params_blocks(_gp_out()) if ST.grounding.cfg.get("out") else {"blocks": [], "why": "set <out> first"})


def ep_gp_runbook(_q, _b):
    return {"sections": {str(k): v for k, v in GPN.runbook_sections().items()}, "path": str(GPN.RUNBOOK), "present": GPN.RUNBOOK.exists()}


def ep_gp_plan_text(q, _b):
    mv = (q.get("move") or ["0"])[0]
    def go():
        c = ST.grounding.cfg
        if not c.get("out"):
            raise GPN.PanelError("set <out> first")
        return {"move": mv, "text": GPN.plan_text(str(_gp_out()), c.get("root") or "", c.get("flags", {}), mv)}
    return _gp(go)


def ep_gp_text(q, _b):
    rel = (q.get("path") or [""])[0]
    return _gp(lambda: GPN.text_file(_gp_out(), rel))


def ep_gp_list(q, _b):
    rel = (q.get("path") or [""])[0]
    return _gp(lambda: GPN.listing(_gp_out(), rel))


def ep_gp_file(q, _b):
    """The engine's file bytes (a figure, a CSV, a gallery page) from under <out>; nothing outside it;
    nothing while <out> is unset; a CSV under inputs/ with its kernel and idle rows only."""
    rel = (q.get("path") or [""])[0]
    try:
        data, ctype = GPN.file_bytes(_gp_out(), rel)
    except GPN.PanelError as e:
        return {"error": str(e)}, 400
    except FileNotFoundError:
        return {"error": f"no such file under the output folder: {rel}"}, 404
    return data, 200, ctype


def ep_gp_log(q, _b):
    lid = (q.get("launch") or [""])[0] or None
    n = int((q.get("tail") or ["200"])[0])
    return _gp(lambda: ST.grounding.log_tail(lid, n))


def ep_gp_launch(q, _b):
    lid = (q.get("id") or [""])[0]
    return _gp(lambda: ST.grounding.launch_record(lid))


def ep_gp_encoding_table2(_q, _b):
    return _gp(lambda: GPN.encoding_table2(_gp_out()) if ST.grounding.cfg.get("out") else {"exists": False, "why": "set <out> first"})


# ---------------------------------------------------------------------------
# Random-signal paper: the paper's runs folder read as it is, the laptop moves run from the tab, the server never contacted
# ---------------------------------------------------------------------------

def _rs(fn):
    try:
        return fn()
    except RSP.PanelError as e:
        return {"error": str(e)}, 400
    except FileNotFoundError as e:
        return {"error": f"missing: {e}"}, 404


def ep_rs_config(_q, _b):
    return _rs(lambda: ST.random_signal.config())


def ep_rs_set_config(_q, body):
    return _rs(lambda: ST.random_signal.set_config(body or {}))


def ep_rs_board(_q, _b):
    return _rs(lambda: ST.random_signal.board())


def ep_rs_run(_q, body):
    move = (body or {}).get("move")
    if move is None:
        return {"error": "move required"}, 400
    return _rs(lambda: ST.random_signal.launch(move, bool((body or {}).get("force"))))


def ep_rs_stop(_q, _b):
    return _rs(lambda: ST.random_signal.stop())


def ep_rs_views(_q, _b):
    return _rs(lambda: {"views": RSP.views(ST.random_signal.paper), "paper": str(ST.random_signal.paper)})


def ep_rs_text(q, _b):
    rel = (q.get("path") or [""])[0]
    return _rs(lambda: RSP.text_file(ST.random_signal.paper, rel))


def ep_rs_list(q, _b):
    rel = (q.get("path") or [""])[0]
    return _rs(lambda: RSP.listing(ST.random_signal.paper, rel))


def ep_rs_file(q, _b):
    """A file of the paper's runs folder (a figure, a CSV, a text), nothing outside it."""
    rel = (q.get("path") or [""])[0]
    try:
        data, ctype = RSP.file_bytes(ST.random_signal.paper, rel)
    except RSP.PanelError as e:
        return {"error": str(e)}, 400
    except FileNotFoundError:
        return {"error": f"no such file under the paper's runs folder: {rel}"}, 404
    return data, 200, ctype


def ep_rs_log(q, _b):
    lid = (q.get("launch") or [""])[0] or None
    n = int((q.get("tail") or ["200"])[0])
    return _rs(lambda: ST.random_signal.log_tail(lid, n))


def ep_rs_launch(q, _b):
    lid = (q.get("id") or [""])[0]
    return _rs(lambda: ST.random_signal.launch_record(lid))


ROUTES_GET = {"/health": ep_health, "/manifest": ep_manifest, "/status": ep_status, "/runs": ep_runs, "/results": ep_results,
               "/results/summary": ep_results_summary, "/results/agg": ep_results_agg,
               "/learn/modules": ep_learn_modules, "/learn/inputs": ep_learn_inputs, "/learn/runs": ep_learn_runs, "/learn/status": ep_learn_status,
               "/learn/results": ep_learn_results, "/learn/agg": ep_learn_agg, "/learn/view": ep_learn_view, "/learn/tile": ep_learn_tile,
               "/learn/tiles_index": ep_learn_tiles_index,
               "/encoding/config": ep_enc_config, "/encoding/board": ep_enc_board, "/encoding/cells": ep_enc_cells, "/encoding/views": ep_enc_views,
               "/encoding/params": ep_enc_params, "/encoding/runbook": ep_enc_runbook, "/encoding/plan_text": ep_enc_plan_text,
               "/encoding/text": ep_enc_text, "/encoding/list": ep_enc_list, "/encoding/file": ep_enc_file, "/encoding/log": ep_enc_log,
               "/encoding/launch": ep_enc_launch,
               "/grounding/config": ep_gp_config, "/grounding/board": ep_gp_board, "/grounding/cells": ep_gp_cells, "/grounding/views": ep_gp_views,
               "/grounding/params": ep_gp_params, "/grounding/runbook": ep_gp_runbook, "/grounding/plan_text": ep_gp_plan_text,
               "/grounding/text": ep_gp_text, "/grounding/list": ep_gp_list, "/grounding/file": ep_gp_file, "/grounding/log": ep_gp_log,
               "/grounding/launch": ep_gp_launch, "/grounding/encoding_table2": ep_gp_encoding_table2,
               "/random/config": ep_rs_config, "/random/board": ep_rs_board, "/random/views": ep_rs_views, "/random/text": ep_rs_text,
               "/random/list": ep_rs_list, "/random/file": ep_rs_file, "/random/log": ep_rs_log, "/random/launch": ep_rs_launch,
               "/rundetail": ep_rundetail, "/trajectory_columns": ep_trajectory_columns,
               "/cache/status": ep_cache_status}
ROUTES_POST = {"/source/test": ep_source_test, "/scan": ep_scan, "/validate": ep_validate, "/run": ep_run, "/control": ep_control,
                "/cache/drop": ep_cache_drop, "/learn/validate": ep_learn_validate, "/learn/run": ep_learn_run, "/learn/control": ep_learn_control,
                "/encoding/config": ep_enc_set_config, "/encoding/run": ep_enc_run, "/encoding/stop": ep_enc_stop,
                "/grounding/config": ep_gp_set_config, "/grounding/run": ep_gp_run, "/grounding/stop": ep_gp_stop,
                "/random/config": ep_rs_set_config, "/random/run": ep_rs_run, "/random/stop": ep_rs_stop}

TOKEN = ""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet
        return

    def _send(self, obj, code=200, ctype="application/json"):
        data = obj if isinstance(obj, bytes) else json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith("text") else ""))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _auth(self, q) -> bool:
        tok = (q.get("token") or [""])[0] or self.headers.get("X-Token", "")
        return secrets.compare_digest(tok, TOKEN)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            if not self._auth(q):
                return self._send({"error": "token required: open the URL the bridge printed"}, 401)
            html = SERVED_HTML.read_text() if SERVED_HTML.exists() else "<p>console not built; run build_analysis_console.py --served</p>"
            inject = f"<script>window.__BRIDGE__={json.dumps({'token': TOKEN, 'source': ST.source})};</script>"
            return self._send(html.replace("</head>", inject + "</head>", 1).encode(), 200, "text/html")
        fn = ROUTES_GET.get(u.path)
        if not fn:
            return self._send({"error": "no such route"}, 404)
        if not self._auth(q):
            return self._send({"error": "bad token"}, 401)
        out = fn(q, None)
        if isinstance(out, tuple):
            return self._send(out[0], out[1], out[2]) if len(out) == 3 else self._send(out[0], out[1])
        return self._send(out)

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        fn = ROUTES_POST.get(u.path)
        if not fn:
            return self._send({"error": "no such route"}, 404)
        if not self._auth(q):
            return self._send({"error": "bad token"}, 401)
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._send({"error": "body is not JSON"}, 400)
        out = fn(q, body)
        if isinstance(out, tuple):
            return self._send(out[0], out[1])
        return self._send(out)


def serve(port: int, source: dict, out_dir: Path, store: Path | None, open_browser: bool, build: bool, token: str | None = None,
          learn_dir: Path | None = None):
    global ST, TOKEN
    TOKEN = token or secrets.token_urlsafe(24)
    out_dir.mkdir(parents=True, exist_ok=True)
    ST = State(out_dir, source, store, learn_dir)
    ST.learn_dir.mkdir(parents=True, exist_ok=True)
    try:
        src = make_source(source)
        ST.manifest = corpus_manifest.scan_source(src)
        ST.source = src.describe()
        am = ST.manifest.get("archive_manifest") or {}
        how = (f"archive manifest (rebuilt {am.get('rebuilt_at')}, updated {am.get('updated_at')}, "
               f"{am.get('registered_since_rebuild', 0)} registered since)" if am else "full walk")
        print(f"[bridge] {ST.manifest['n_recordings']} recordings from {ST.source.get('root') or ST.source.get('host')} via {how}")
    except (SourceError, corpus_manifest.CorpusMissing) as e:
        print(f"[bridge] no manifest at start: {e} (scan from the console)")
    if build:
        mpath = out_dir / "manifest.current.json"
        if ST.manifest:
            mpath.write_text(json.dumps(ST.manifest))
            r = subprocess.run([sys.executable, str(BUILD), "--served", "--manifest", str(mpath)], capture_output=True, text=True, cwd=str(QEMU_DIR))
            print(r.stdout.strip())
            if r.returncode:
                print(r.stderr.strip(), file=sys.stderr)
        else:
            print("[bridge] console not rebuilt (no manifest); serving the existing served build if any")
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/?token={TOKEN}"
    print(f"[bridge] listening on 127.0.0.1:{port}  token={TOKEN}", flush=True)
    print(f"[bridge] open: {url}", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--source-json", type=str, default=None, help='{"kind":"local","root":...} or an ssh spec')
    ap.add_argument("--root", type=Path, default=None, help="shorthand for a local source")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--store", type=Path, default=None, help="L1 store directory (default ~/.cache/plan10/l1)")
    ap.add_argument("--learn-dir", type=Path, default=None, help="where Learn runs land (default: beside --out-dir, ~/.cache/plan10/learn)")
    ap.add_argument("--no-build", action="store_true", help="do not rebuild the served console at start")
    ap.add_argument("--open", action="store_true", help="open the browser")
    ap.add_argument("--token", type=str, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.source_json:
        source = json.loads(a.source_json)
    else:
        root = a.root or corpus_manifest.default_root()
        source = {"kind": "local", "root": str(root)}
    serve(a.port, source, a.out_dir, a.store, a.open, not a.no_build, a.token, a.learn_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
