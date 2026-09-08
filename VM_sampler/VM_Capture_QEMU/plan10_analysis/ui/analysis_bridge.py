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

Sources are {"kind":"local","root":...} or {"kind":"ssh","host":...,"user":...,"key":...,
"remote_root":...,"port":22}. The executor runs as a subprocess so a run outlives a page
reload; the bridge only reads its status file. Nothing here runs analysis in-process.

Run:  python3 plan10_analysis/ui/analysis_bridge.py [--port 8766] [--source-json S] [--out-dir D] [--open]
"""
from __future__ import annotations

import argparse
import json
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
from plan10_analysis.modules import build_modules                         # noqa: E402
from plan10_analysis.sources import SourceError, make_source              # noqa: E402

SERVED_HTML = HERE / "analysis_console.served.html"
BUILD = HERE / "build_analysis_console.py"
EXECUTOR = PKG / "runner" / "executor.py"
DEFAULT_OUT = Path(os.path.expanduser("~/.cache/plan10/runs"))
LABEL_OK = __import__("re").compile(r"^[A-Za-z0-9_-]+$")


class State:
    def __init__(self, out_dir: Path, source: dict, store: Path | None):
        self.out_dir = out_dir
        self.store = store
        self.source = source
        self.manifest: dict | None = None
        self.lock = threading.Lock()
        self.procs: dict[str, subprocess.Popen] = {}
        self.ctx_cache: tuple | None = None

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
    try:
        src = make_source(body.get("source") or ST.source)
        m = corpus_manifest.scan_source(src)
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


def ep_runs(_q, _b):
    out = []
    if ST.out_dir.is_dir():
        for d in sorted(ST.out_dir.iterdir()):
            if d.is_dir() and (d / "scheme.json").exists():
                s = _status_of(d)
                out.append({"label": d.name, "state": s.get("state"), "updated_at": s.get("updated_at"), "message": s.get("message"),
                            "has_features": (d / "features.npz").exists()})
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


ROUTES_GET = {"/health": ep_health, "/manifest": ep_manifest, "/status": ep_status, "/runs": ep_runs, "/results": ep_results}
ROUTES_POST = {"/source/test": ep_source_test, "/scan": ep_scan, "/validate": ep_validate, "/run": ep_run, "/control": ep_control}

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
            return self._send(out[0], out[1])
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


def serve(port: int, source: dict, out_dir: Path, store: Path | None, open_browser: bool, build: bool, token: str | None = None):
    global ST, TOKEN
    TOKEN = token or secrets.token_urlsafe(24)
    out_dir.mkdir(parents=True, exist_ok=True)
    ST = State(out_dir, source, store)
    try:
        src = make_source(source)
        ST.manifest = corpus_manifest.scan_source(src)
        ST.source = src.describe()
        print(f"[bridge] scanned {ST.manifest['n_recordings']} recordings from {ST.source.get('root') or ST.source.get('host')}")
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
    ap.add_argument("--no-build", action="store_true", help="do not rebuild the served console at start")
    ap.add_argument("--open", action="store_true", help="open the browser")
    ap.add_argument("--token", type=str, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.source_json:
        source = json.loads(a.source_json)
    else:
        root = a.root or corpus_manifest.default_root()
        source = {"kind": "local", "root": str(root)}
    serve(a.port, source, a.out_dir, a.store, a.open, not a.no_build, a.token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
