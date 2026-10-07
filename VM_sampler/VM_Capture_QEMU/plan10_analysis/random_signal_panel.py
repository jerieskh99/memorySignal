#!/usr/bin/env python3
"""random_signal_panel.py -- the console's window onto the random-signal paper's runs (2026-10-07).

The paper (working name "the random-signal paper"; `memorySignal/random_signal_paper/`, not in git; it was
called "the grounding paper" until 2026-10-06, so older files use that name) keeps its runs under
`server_runs/`: scripts the author runs on the research server by hand, and two analysis scripts he runs
on the laptop. This module applies the Encoding paper and Grounding paper tabs' idea to that folder: a
board of moves, each with its state, its Run (laptop moves only), its log, and its results and figures
shown as the files write them. Nothing here computes a number, and nothing here contacts the server.

  - A server move shows the command the author runs there (the README's own words) and the rsync line
    that brings the results home; its state comes from the files under `server_runs/results/`: done
    when they are home, "not home yet" until then. The tab never runs anything on the server.
  - A laptop move (`analysis/analyze_lag.py`, `analysis/figures_lag.py`) runs from the tab, exactly as
    the author runs it, one process at a time; before a re-run of either, the tab keeps a copy of
    `analysis/out/` beside it (`analysis/out.pre_<time>/`), since both write there.
  - The paper's files are never edited. The console's own records (the launch records and logs) live
    under `~/.cache/plan10/random_signal/`, never under the paper's folder; the configuration (the paper's
    folder) in `~/.cache/plan10/random_signal_config.json`.
  - The declaration (`DECLARATION.md`, fixed 2026-10-06) is shown with its checksum checked against
    `DECLARATION.md.sha256`; the reading in words (`analysis/RESULTS.md`) is a view of its own.

New moves are added to MOVES below, the same way: a number, a title, where it runs, the command, the
files that mean "done", and the files to show.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
QEMU_DIR = HERE.parent
PAPER_DEFAULT = Path("/Users/jeries/Desktop/projects/thesis/memorySignal/random_signal_paper")
RUNS_DIR = "server_runs"                                    # under the paper's folder: the scripts, results/ and analysis/
DEFAULT_CONFIG = Path(os.path.expanduser("~/.cache/plan10/random_signal_config.json"))
CONSOLE_DIR = Path(os.path.expanduser("~/.cache/plan10/random_signal"))     # the console's own records, never under the paper's folder
SERVER = "jeries@cybersecurity.ac.upc.edu"
RSYNC_SCRIPTS = f"rsync -a ~/Desktop/projects/thesis/memorySignal/random_signal_paper/server_runs/ {SERVER}:random_signal_runs/"
RSYNC_HOME = f"rsync -a {SERVER}:random_signal_runs/results/ ~/Desktop/projects/thesis/memorySignal/random_signal_paper/server_runs/results/"
RSYNC_FP_CHECK = f"rsync -a ~/Desktop/projects/thesis/memorySignal/random_signal_paper/server_runs/fp_check.py {SERVER}:random_signal_runs/"
LEXER_42_CHAIN = ("/mnt/nfs/jeries/memory_traces/zstd_local/kernel/kernel_lexer_v2/input-mb_32_--duration_600_--seed_42_--phase-markers_--max-m_e2a0a0b0/"
                  "rep001__sandbox_deepdive_01c1")
N_LAG_RECORDINGS = 16

# The moves, in the README's order (server_runs/README.md). `where` is "server" (the author runs the command there and
# brings the results home with RSYNC_HOME) or "laptop" (the tab runs `argv` from `cwd`, the paper's server_runs/ folder).
# `done` lists file patterns under server_runs/; the move is done when every pattern matches at least `min` files (a third
# element excludes files ending with that suffix from the count: the lag scan's `_content.csv` beside each recording's csv).
# `files` are the files the Look tab shows for the move (patterns, under server_runs/). `requires` names the moves
# whose files the move reads; a laptop move's Run waits for them.
MOVES = [
    {"move": 0, "title": "Platform record", "where": "server", "script": "platform_check.sh",
     "what": "records the server's versions, the VM's settings and the free space before the tests (read only; the declaration asks for it)",
     "command": "bash ~/random_signal_runs/platform_check.sh", "done": [("results/platform_*.txt", 1)],
     "files": ["results/platform_*.txt"], "requires": [], "record": "Done 2026-10-06."},
    {"move": 1, "title": "Paused test and resume test, three runs", "where": "server", "script": "controls.sh via run_all.sh",
     "what": "the paused test (three copies in one pause: P1 against P2 must be identical; P0 against P1 and P2 is the epsilon) and the resume test "
             "(one resume, an immediate pause, two copies), three times, five minutes apart",
     "command": "bash ~/random_signal_runs/controls.sh", "done": [("results/controls_*/counts.txt", 3)],
     "files": ["results/controls_*/counts.txt", "results/controls_*/timing.txt"], "requires": [0], "record": "Done 2026-10-06 (three runs, via run_all.sh)."},
    {"move": 2, "title": "Lag scan, 16 recordings", "where": "server", "script": "run_lag_scan.sh with IDLE_ALL=1 CONTENT=1",
     "what": "for each recording, snapshot t against snapshot t+k for k = 1 to 128, at eight starting snapshots after the start-up burst, plus the dense page "
             "curve; all eight idle recordings (IDLE_ALL=1) and the optional content curve (CONTENT=1)",
     "command": "IDLE_ALL=1 CONTENT=1 bash ~/random_signal_runs/run_lag_scan.sh", "done": [("results/lag/*.csv", N_LAG_RECORDINGS, "_content.csv")],
     "log": "results/run_all_20261006T153404.log", "done_marker": "run_all end",
     "files": ["results/run_all_*.log", "results/lag/*.log"], "requires": [0], "record": "Done 2026-10-06; its log is results/run_all_20261006T153404.log."},
    {"move": 3, "title": "The declared reading (DECLARATION.md item 7.4)", "where": "laptop", "script": "analysis/analyze_lag.py",
     "what": "reads results/lag/ under the declaration's agreement rule (item 7.4, fixed 2026-10-06): the verdict per recording, the median bit curves "
             "and the page curves, into analysis/out/",
     "argv": [sys.executable, "analysis/analyze_lag.py"], "done": [("analysis/out/lag_rules_result.json", 1), ("analysis/out/*_page_curve.csv", 1)],
     "files": ["analysis/out/lag_rules_result.json", "analysis/out/*_page_curve.csv", "analysis/out/*_median_bits.csv"], "requires": [2],
     "keep_copy": "analysis/out", "record": "Done 2026-10-07."},
    {"move": 4, "title": "Figures", "where": "laptop", "script": "analysis/figures_lag.py",
     "what": "draws fig1 (the lag curves by recording) and fig2 (the kernels against idle) from analysis/out/",
     "argv": [sys.executable, "analysis/figures_lag.py"], "done": [("analysis/out/fig1_*.png", 1), ("analysis/out/fig2_*.png", 1)],
     "files": ["analysis/out/fig1_*.png", "analysis/out/fig2_*.png"], "requires": [3], "keep_copy": "analysis/out", "record": "Done 2026-10-07."},
    {"move": 5, "title": "The check of lexer_42's page-count fault", "where": "server", "script": "fp_check.py",
     "what": "finds the page behind lexer_42's four page-count mismatches (RESULTS.md, rule 2) and prints what changed in it, with lag_scan.py's own "
             "walker and fingerprint; about 8 minutes on the server",
     "command": f"python3 ~/random_signal_runs/fp_check.py {LEXER_42_CHAIN} ~/random_signal_runs/work/fp_check_lexer_42 | tee ~/random_signal_runs/results/fp_check_lexer_42.txt",
     "pre_command": RSYNC_FP_CHECK, "done": [("results/fp_check_lexer_42.txt", 1)],
     "files": ["results/fp_check_lexer_42.txt"], "requires": [2], "record": "Running on the server (2026-10-07); done when results/fp_check_lexer_42.txt is home."},
]
VIEWS = [
    {"id": "results_md", "title": "the reading in words (analysis/RESULTS.md)", "files": ["analysis/RESULTS.md"]},
    {"id": "declaration", "title": "the declaration (DECLARATION.md, fixed 2026-10-06) and its checksum", "files": ["../DECLARATION.md", "../DECLARATION.md.sha256"]},
    {"id": "figures", "title": "the figures (move 4)", "files": ["analysis/out/fig1_*.png", "analysis/out/fig2_*.png"]},
    {"id": "reading", "title": "the declared reading's verdicts (move 3)", "files": ["analysis/out/lag_rules_result.json"]},
    {"id": "controls", "title": "the paused and resume tests (move 1)", "files": ["results/controls_*/counts.txt", "results/controls_*/timing.txt"]},
    {"id": "platform", "title": "the platform record (move 0)", "files": ["results/platform_*.txt"]},
    {"id": "lag_logs", "title": "the lag scan's logs (move 2)", "files": ["results/run_all_*.log"]},
    {"id": "fp_check", "title": "the lexer_42 check (move 5)", "files": ["results/fp_check_lexer_42.txt"]},
    {"id": "readme", "title": "the runs' README (the commands, in the author's words)", "files": ["README.md"]},
]
TEXT_KINDS = {".csv": "csv", ".json": "json", ".md": "md", ".txt": "text", ".log": "text", ".py": "text", ".sh": "text", ".sha256": "text"}
BINARY_TYPES = {".png": "image/png", ".pdf": "application/pdf", ".npy": "application/octet-stream", ".npz": "application/octet-stream"}
MAX_TEXT_BYTES = 8 * 1024 * 1024
MAX_CSV_ROWS = 20000
STATES_NOT_RUNNABLE = ("running",)


class PanelError(ValueError):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# the paper's folder, read as it is
# ---------------------------------------------------------------------------

def runs_dir(paper: Path) -> Path:
    return Path(paper) / RUNS_DIR


def safe_path(paper: Path, rel: str) -> Path:
    """A path under the paper's `server_runs/` folder, or, with the one allowed step up, a file in the paper's
    folder itself (`../DECLARATION.md`, `../DECLARATION.md.sha256`, `../IDEA.md`); anything else is refused."""
    rel = str(rel or "")
    if not rel or rel.startswith(("/", "\\")):
        raise PanelError(f"not a path under the paper's runs folder: {rel!r}")
    parts = Path(rel).parts
    base = runs_dir(paper)
    if parts and parts[0] == "..":
        if len(parts) != 2 or parts[1].startswith(".") or not parts[1].endswith((".md", ".sha256")):
            raise PanelError(f"only the paper's own .md and .sha256 files may be read one level up: {rel!r}")
        p = (Path(paper) / parts[1]).resolve()
        if p.parent != Path(paper).resolve():
            raise PanelError(f"not a file of the paper's folder: {rel!r}")
        return p
    if ".." in parts or any(x.startswith(".") for x in parts) or "__pycache__" in parts:
        raise PanelError(f"not a path under the paper's runs folder: {rel!r}")
    p = (base / rel).resolve()
    b = base.resolve()
    if p != b and b not in p.parents:
        raise PanelError(f"not a path under the paper's runs folder: {rel!r}")
    return p


def expand(paper: Path, pattern: str) -> list[str]:
    """The files matching a pattern under server_runs/ (or `../<file>` in the paper's folder), as relative paths."""
    if pattern.startswith("../"):
        p = Path(paper) / pattern[3:]
        return [pattern] if p.is_file() else []
    base = runs_dir(paper)
    out = []
    for p in sorted(base.glob(pattern)):
        if p.is_file() and not any(x.startswith(".") for x in p.relative_to(base).parts):
            out.append(str(p.relative_to(base)))
    return out


def read_csv_rows(path: Path, limit: int = MAX_CSV_ROWS) -> tuple[list[str], list[list[str]], int]:
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        r = csv.reader(fh)
        header = next(r, [])
        rows, total = [], 0
        for row in r:
            total += 1
            if len(rows) < limit:
                rows.append(row)
    return header, rows, total


def text_file(paper: Path, rel: str) -> dict:
    """A CSV as header and rows, a JSON as its object, anything else as text; the path and size always."""
    p = safe_path(paper, rel)
    if not p.exists():
        return {"path": rel, "exists": False, "kind": None}
    if p.is_dir():
        return {"path": rel, "exists": True, "kind": "dir", "entries": listing(paper, rel)["entries"]}
    size = p.stat().st_size
    kind = TEXT_KINDS.get(p.suffix.lower())
    d = {"path": rel, "exists": True, "kind": kind, "bytes": size, "mtime": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")}
    if kind is None:
        d["kind"] = "binary"
        d["content_type"] = BINARY_TYPES.get(p.suffix.lower(), "application/octet-stream")
        return d
    if size > MAX_TEXT_BYTES:
        d["error"] = f"{size} bytes: over the {MAX_TEXT_BYTES} byte limit for inline display; open the file"
        return d
    if kind == "csv":
        header, rows, total = read_csv_rows(p)
        d.update(header=header, rows=rows, n_rows=total, truncated=total > len(rows))
    elif kind == "json":
        try:
            d["json"] = read_json(p)
        except json.JSONDecodeError as e:
            d["error"] = f"not valid JSON: {e}"
            d["text"] = p.read_text(encoding="utf-8", errors="replace")
    else:
        d["text"] = p.read_text(encoding="utf-8", errors="replace")
    return d


def file_bytes(paper: Path, rel: str) -> tuple[bytes, str]:
    p = safe_path(paper, rel)
    if not p.is_file():
        raise FileNotFoundError(rel)
    ctype = BINARY_TYPES.get(p.suffix.lower()) or {"csv": "text/csv", "json": "application/json", "md": "text/markdown", "text": "text/plain"}.get(
        TEXT_KINDS.get(p.suffix.lower(), "text"), "application/octet-stream")
    return p.read_bytes(), ctype


def listing(paper: Path, rel: str = "") -> dict:
    p = safe_path(paper, rel) if rel else runs_dir(paper).resolve()
    if not p.exists():
        return {"path": rel, "exists": False, "entries": []}
    ents = []
    for c in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name)):
        if c.name.startswith(".") or c.name == "__pycache__":
            continue
        st = c.stat()
        ents.append({"name": c.name, "dir": c.is_dir(), "bytes": None if c.is_dir() else st.st_size,
                     "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds"),
                     "path": f"{rel}/{c.name}" if rel else c.name})
    return {"path": rel, "exists": True, "entries": ents}


def declaration(paper: Path) -> dict:
    """DECLARATION.md and its checksum: the sha256 recorded in DECLARATION.md.sha256 against the file as it is now."""
    paper = Path(paper)
    p, s = paper / "DECLARATION.md", paper / "DECLARATION.md.sha256"
    res = {"path": "../DECLARATION.md", "exists": p.is_file(), "sha256_file": "../DECLARATION.md.sha256", "sha256_file_exists": s.is_file(),
           "expected": None, "computed": None, "matches": None, "status_line": None}
    if p.is_file():
        res["computed"] = sha256_file(p)
        res["bytes"] = p.stat().st_size
        res["mtime"] = datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
        with contextlib.suppress(OSError):
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("**Status:"):
                    res["status_line"] = line.strip("* ").strip()
                    break
    if s.is_file():
        tok = s.read_text(encoding="utf-8", errors="replace").split()
        res["expected"] = tok[0] if tok else None
    if res["expected"] and res["computed"]:
        res["matches"] = res["expected"] == res["computed"]
    return res


# ---------------------------------------------------------------------------
# the board: each move's state from the files it writes
# ---------------------------------------------------------------------------

def done_check(paper: Path, m: dict) -> dict:
    parts = []
    for rule in m["done"]:
        pattern, n_min, exclude = (rule + (None,))[:3]
        found = [f for f in expand(paper, pattern) if not (exclude and f.endswith(exclude))]
        parts.append({"pattern": pattern + (f" (not *{exclude})" if exclude else ""), "min": n_min, "found": len(found), "ok": len(found) >= n_min})
    return {"done": all(x["ok"] for x in parts) and bool(parts), "parts": parts}


def log_marker(paper: Path, m: dict) -> dict | None:
    """A server move with a log and a done marker: whether the log, once home, ends with the marker."""
    if not m.get("log"):
        return None
    p = runs_dir(paper) / m["log"]
    if not p.is_file():
        return {"log": m["log"], "home": False, "ended": None}
    tail = p.read_text(encoding="utf-8", errors="replace")[-4000:]
    return {"log": m["log"], "home": True, "ended": (m.get("done_marker") or "") in tail, "last_line": tail.strip().splitlines()[-1][:200] if tail.strip() else ""}


def move_states(paper: Path, running: dict | None, launches: list[dict]) -> list[dict]:
    """One row per move: its state from the files (server: done or not home yet; laptop: done, running, failed or
    not run), the command the author runs or the tab runs, the files present, and what it waits for."""
    paper = Path(paper)
    last_launch = {}
    for l in launches:                                   # newest first
        last_launch.setdefault(l.get("move"), l)
    rows = []
    for m in MOVES:
        dc = done_check(paper, m)
        files = []
        for pat in m["files"]:
            files += expand(paper, pat)
        ll = last_launch.get(m["move"])
        if running and running.get("move") == m["move"]:
            state = "running"
        elif dc["done"]:
            state = "done"
        elif m["where"] == "server":
            state = "not home yet"
        elif ll and ll.get("exit_code") not in (None, 0):
            state = "failed"
        else:
            state = "not run"
        row = {"move": m["move"], "title": m["title"], "where": m["where"], "script": m["script"], "what": m["what"], "state": state,
               "command": m.get("command") or " ".join(("python3" if i == 0 else t) for i, t in enumerate(m.get("argv", []))),
               "pre_command": m.get("pre_command"), "rsync_home": RSYNC_HOME if m["where"] == "server" else None,
               "done_check": dc, "log_marker": log_marker(paper, m), "files": files, "n_files": len(files), "requires": list(m["requires"]),
               "record": m.get("record"), "keep_copy": m.get("keep_copy"),
               "last_launch": ({k: ll.get(k) for k in ("id", "started_at", "finished_at", "exit_code")} if ll else None),
               "last_finished_at": (ll or {}).get("finished_at")}
        rows.append(row)
    for r in rows:
        if r["where"] != "laptop":
            r["runnable"] = False
            r["waits_for"] = "the author runs it on the server and brings the results home (the tab never contacts the server)"
            continue
        need = [q for q in r["requires"] if not any(x["move"] == q and x["state"] == "done" for x in rows)]
        r["runnable"] = (not running) and not need
        r["waits_for"] = (f"move(s) {', '.join(str(q) for q in need)} (their files are not home yet)" if need else None)
    return rows


def views(paper: Path) -> list[dict]:
    outs = []
    for v in VIEWS:
        files = []
        for pat in v["files"]:
            for f in expand(paper, pat):
                p = safe_path(paper, f)
                files.append({"path": f, "exists": p.exists(), "kind": ("image" if p.suffix == ".png" else TEXT_KINDS.get(p.suffix, "binary")),
                              "bytes": p.stat().st_size if p.exists() else None})
            if not expand(paper, pat):
                files.append({"path": pat, "exists": False, "kind": None, "bytes": None})
        outs.append({**{k: v[k] for k in ("id", "title")}, "files": files, "any": any(f["exists"] for f in files)})
    return outs


# ---------------------------------------------------------------------------
# launching a laptop move, one process at a time
# ---------------------------------------------------------------------------

class Panel:
    """The bridge's state for the panel: the paper's folder and the one running process."""

    def __init__(self, config_path: Path = DEFAULT_CONFIG, console_dir: Path = CONSOLE_DIR):
        self.config_path = Path(config_path)
        self.console_dir = Path(console_dir)
        self.lock = threading.RLock()
        self.proc: subprocess.Popen | None = None
        self.running: dict | None = None
        self.cfg = self._load()

    def _load(self) -> dict:
        d = {"paper": str(PAPER_DEFAULT)}
        if self.config_path.exists():
            with contextlib.suppress(OSError, json.JSONDecodeError):
                d.update(read_json(self.config_path))
        return d

    def save(self):
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.config_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.cfg, indent=1))
        os.replace(tmp, self.config_path)

    def set_config(self, body: dict) -> dict:
        if "paper" in body:
            self.cfg["paper"] = str(body["paper"] or "").strip() or str(PAPER_DEFAULT)
        self.save()
        return self.config()

    @property
    def paper(self) -> Path:
        return Path(os.path.expanduser(self.cfg.get("paper") or str(PAPER_DEFAULT)))

    def config(self) -> dict:
        p = self.paper
        return {"paper": str(p), "paper_exists": p.is_dir(), "runs_dir": str(runs_dir(p)), "runs_dir_exists": runs_dir(p).is_dir(),
                "console_dir": str(self.console_dir), "config_path": str(self.config_path), "server": SERVER,
                "rsync_scripts": RSYNC_SCRIPTS, "rsync_home": RSYNC_HOME, "readme": "README.md" if (runs_dir(p) / "README.md").is_file() else None,
                "declaration": declaration(p), "python": sys.executable,
                "note": ("server moves: the author runs the command on the server and brings the results home with the rsync line; "
                         "laptop moves run from this tab, one at a time, from the paper's server_runs/ folder; the paper's files are never edited; "
                         "before a re-run of move 3 or 4 the tab keeps a copy of analysis/out/ beside it")}

    def _need(self) -> Path:
        p = self.paper
        if not runs_dir(p).is_dir():
            raise PanelError(f"the paper's runs folder is not on disk: {runs_dir(p)}")
        return p

    def board(self) -> dict:
        p = self._need()
        self._reap()
        running = dict(self.running) if self.running else None
        launches = self.launches()
        return {"paper": str(p), "runs_dir": str(runs_dir(p)), "moves": move_states(p, running, launches), "running": running,
                "launches": launches[:20], "declaration": declaration(p), "rsync_home": RSYNC_HOME, "rsync_scripts": RSYNC_SCRIPTS}

    def _launch_dir(self) -> Path:
        d = self.console_dir / "launches"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def keep_copy(self, rel: str) -> str | None:
        """Before a re-run that rewrites `rel` (analysis/out), keep a copy beside it: analysis/out.pre_<time>/ (the house
        rule for a file, applied to the folder). Returns the copy's relative path, or None when there was nothing to keep."""
        src = runs_dir(self.paper) / rel
        if not src.is_dir() or not any(src.iterdir()):
            return None
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dst = src.parent / f"{src.name}.pre_{stamp}"
        n = 1
        while dst.exists():
            n += 1
            dst = src.parent / f"{src.name}.pre_{stamp}-{n}"
        shutil.copytree(src, dst)
        return str(dst.relative_to(runs_dir(self.paper)))

    def launch(self, move, force: bool = False) -> dict:
        p = self._need()
        with self.lock:
            self._reap()
            if self.running:
                raise PanelError(f"move {self.running['move']} is still running (launch {self.running['id']}); one process at a time")
            try:
                move = int(move)
            except (TypeError, ValueError):
                raise PanelError("move must be a number")
            m = next((x for x in MOVES if x["move"] == move), None)
            if m is None:
                raise PanelError(f"no move {move}; the moves are 0 to {MOVES[-1]['move']}")
            if m["where"] != "laptop":
                raise PanelError(f"move {move} runs on the server: the tab never contacts the server; run the command shown there and bring the results home with the rsync line")
            row = next(r for r in move_states(p, None, self.launches()) if r["move"] == move)
            if row["state"] == "done" and not force:
                raise PanelError(f"move {move} is done; Re-run re-runs it (a copy of {m.get('keep_copy') or 'its outputs'} is kept first)")
            if not row["runnable"]:
                raise PanelError(f"move {move} waits for {row['waits_for']}")
            kept = self.keep_copy(m["keep_copy"]) if m.get("keep_copy") and row["state"] in ("done", "failed") else None
            argv = list(m["argv"])
            ldir = self._launch_dir()
            base = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_move{move}"
            lid, n = base, 1
            while (ldir / f"{lid}.json").exists():
                n += 1
                lid = f"{base}-{n}"
            log = ldir / f"{lid}.log"
            rec = {"id": lid, "move": move, "title": m["title"], "force": bool(force), "argv": argv, "shell": " ".join(("python3" if i == 0 else t) for i, t in enumerate(argv)),
                   "cwd": str(runs_dir(p)), "started_at": now_iso(), "finished_at": None, "exit_code": None, "log": str(log), "kept_copy": kept,
                   "note": "started from the analysis console, from the paper's server_runs/ folder, as the README runs it"}
            (ldir / f"{lid}.json").write_text(json.dumps(rec, indent=1))
            fh = open(log, "ab")
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            self.proc = subprocess.Popen(argv, cwd=str(runs_dir(p)), stdout=fh, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            fh.close()
            self.running = {"id": lid, "move": move, "pid": self.proc.pid, "started_at": rec["started_at"], "force": bool(force)}
            return rec

    def _reap(self):
        with self.lock:
            if self.proc is None or self.running is None:
                return
            rc = self.proc.poll()
            if rc is None:
                return
            p = self._launch_dir() / f"{self.running['id']}.json"
            try:
                rec = read_json(p)
            except (OSError, json.JSONDecodeError):
                rec = {"id": self.running["id"], "move": self.running["move"]}
            rec["finished_at"] = now_iso()
            rec["exit_code"] = rc
            p.write_text(json.dumps(rec, indent=1))
            self.proc = None
            self.running = None

    def stop(self) -> dict:
        with self.lock:
            self._reap()
            if not self.running or self.proc is None:
                raise PanelError("nothing is running")
            proc, rid = self.proc, self.running["id"]
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                proc.terminate()
            t0 = time.time()
            while proc.poll() is None and time.time() - t0 < 10:
                time.sleep(0.2)
            if proc.poll() is None:
                with contextlib.suppress(Exception):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=5)
            self._reap()
            return {"stopped": rid, "note": "the process group was terminated; what the script had written stays; run it again when ready"}

    def launches(self) -> list[dict]:
        d = self.console_dir / "launches"
        if not d.is_dir():
            return []
        recs = []
        for p in sorted(d.glob("*.json"), reverse=True):
            try:
                r = read_json(p)
            except (OSError, json.JSONDecodeError):
                continue
            recs.append({k: r.get(k) for k in ("id", "move", "title", "force", "shell", "started_at", "finished_at", "exit_code", "log", "kept_copy")})
        return recs

    def launch_record(self, lid: str) -> dict:
        if not re.match(r"^[0-9TZ]+_move[0-9]+(-[0-9]+)?$", lid or ""):
            raise PanelError("not a launch id")
        p = self._launch_dir() / f"{lid}.json"
        if not p.exists():
            raise PanelError(f"no launch {lid}")
        return read_json(p)

    def log_tail(self, lid: str | None, n: int = 200) -> dict:
        self._reap()
        if not lid:
            lid = self.running["id"] if self.running else ((self.launches() or [{}])[0].get("id"))
        if not lid:
            return {"id": None, "lines": [], "running": False}
        if not re.match(r"^[0-9TZ]+_move[0-9]+(-[0-9]+)?$", lid):
            raise PanelError("not a launch id")
        p = self._launch_dir() / f"{lid}.log"
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()[-n:] if p.exists() else []
        return {"id": lid, "lines": lines, "running": bool(self.running and self.running["id"] == lid), "log": str(p)}
