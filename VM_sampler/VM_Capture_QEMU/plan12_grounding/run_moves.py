#!/usr/bin/env python3
"""run_moves.py -- the driver of plan12_grounding (SPEC section 4, 7): the moves in order, one record
book per output folder.

  python3 -m plan12_grounding.run_moves run    --out O --moves N [--root R | --ssh USER@HOST --remote-root RR]
                                               [--store S] [--cache C] [--force] [--dry-run] [--room-removal]
  python3 -m plan12_grounding.run_moves plan   --out O --moves N ...      (print the command list, run nothing)
  python3 -m plan12_grounding.run_moves status --out O                    (the record book, move by move)

Modelled on `plan11_encoding_ladder/run_moves.py`: every command is a subprocess of one module
(`python3 -m plan12_grounding.<module> <sub> ...`, cwd VM_Capture_QEMU/), recorded in
`<out>/driver_state.json` with its argv, the sha256 of every declared input, its outputs (and their
sha256 once written), status, exit code and times; the record book also carries the driver's params
and the toolkit fingerprint (sha256 of every plan12_grounding/*.py). A command whose outputs exist
is skipped unless an input's hash or its arguments changed since the `done` record (the encoding
toolkit's staleness rule); `--force` re-runs everything selected. A second driver on the same
`<out>` is refused (`<out>/.driver.lock`, the writer's pid). A move whose last attempt was stopped
mid-run reads "not run" in `status` until a later attempt records every one of its commands.

Slice 1 (2026-09-30) plans moves 0 to 2; later slices append their moves to `build_plan`.
The server is never contacted by this file: the modules do, one recording at a time, in fetch mode
only (SPEC 1.2), and `--dry-run` makes them print their commands instead.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

from plan12_grounding import PACKAGE_NAME, __version__, toolkit_fingerprint  # noqa: E402

LEDGER = "driver_state.json"
LOCK = ".driver.lock"
MAX_MOVE = 10                                # SPEC section 4: moves 0 to 10
CUT_DECLARED = 16                            # SPEC section 2: the declared start-up cut (declared/head_drop_values.csv)
CUT_MEASURED = 112                           # SPEC section 2: the council's measured end of the start-up burst
DEFAULT_STORE = "~/.cache/plan10/l1"         # plan10_analysis/runner/extract.py DEFAULT_STORE: the console's L1 store
DEFAULT_CACHE = "~/.cache/plan10/chains"     # plan10_analysis/sources.py DEFAULT_CACHE: where a fetched trajectory lands
DECLARED_DIR = _HERE.parent / "plan11_encoding_ladder" / "declared"     # read in place, never copied as logic (SPEC 2)
DECLARED_FILES = ("keep_first_pairs.csv", "seed_map.csv", "head_drop_values.csv")
CITATION = "plan12_grounding/SPEC.md sections 4 and 7; modelled on plan11_encoding_ladder/run_moves.py (its ledger and staleness rule)"

MOVE_NAMES = {0: "inputs and index", 1: "extract", 2: "sanity", 3: "every run", 4: "kernel portraits",
              5: "how similar", 6: "classification", 7: "noise floor", 8: "start-up", 9: "noise-floor removal",
              10: "summary"}


# ---------------------------------------------------------------------------------------------
# small helpers (kept here so the move modules import them from one place)
# ---------------------------------------------------------------------------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path, obj) -> Path:
    """Atomic: written beside the target, then renamed over it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=str)
    tmp.replace(path)
    return path


def parse_moves(spec: str, max_move: int = MAX_MOVE) -> list[int]:
    """`0-2`, `1,2`, `0-1,3` -> a sorted list of move numbers within 0..max_move."""
    moves = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            moves.update(range(int(a), int(b) + 1))
        else:
            moves.add(int(part))
    bad = [m for m in moves if m < 0 or m > max_move]
    if bad:
        raise ValueError(f"moves outside 0-{max_move}: {bad}")
    return sorted(moves)


# ---------------------------------------------------------------------------------------------
# the source (SPEC 2): a local root, or the server in fetch mode; the same flags on every module
# ---------------------------------------------------------------------------------------------
def add_source_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--root", default=None, help="a local retention root (the smoke corpus)")
    ap.add_argument("--ssh", default=None, help="USER@HOST of the research server (fetch mode only; SPEC 1.2)")
    ap.add_argument("--remote-root", default=None, help="the retention root on the server")
    ap.add_argument("--ssh-port", type=int, default=22)
    ap.add_argument("--ssh-key", default=None)
    ap.add_argument("--cache", default=DEFAULT_CACHE, help="where a fetched trajectory lands before extraction (removed after verification)")


def source_flags(o: argparse.Namespace) -> list[str]:
    """The source flags as given, for the module subprocesses."""
    toks: list[str] = []
    if getattr(o, "root", None):
        toks += ["--root", str(o.root)]
    if getattr(o, "ssh", None):
        toks += ["--ssh", str(o.ssh)]
    if getattr(o, "remote_root", None):
        toks += ["--remote-root", str(o.remote_root)]
    if getattr(o, "ssh_port", 22) != 22:
        toks += ["--ssh-port", str(o.ssh_port)]
    if getattr(o, "ssh_key", None):
        toks += ["--ssh-key", str(o.ssh_key)]
    if getattr(o, "cache", None) and str(o.cache) != DEFAULT_CACHE:
        toks += ["--cache", str(o.cache)]
    return toks


def source_from_args(o: argparse.Namespace):
    """`LocalSource(root)` or `SshSource(..., mode="fetch")` (plan10_analysis/sources.py). Building the
    ssh source contacts nothing; only listing / fetch_trajectory / verify do, one at a time."""
    from plan10_analysis import sources
    root, ssh = getattr(o, "root", None), getattr(o, "ssh", None)
    if root and ssh:
        raise SystemExit("give --root (local) or --ssh with --remote-root (fetch mode), not both")
    if root:
        return sources.LocalSource(root)
    if ssh:
        if not getattr(o, "remote_root", None):
            raise SystemExit("--ssh needs --remote-root")
        user, _, host = str(ssh).rpartition("@")
        return sources.SshSource(host=host, remote_root=str(o.remote_root), user=user or None,
                                 key=getattr(o, "ssh_key", None), port=int(getattr(o, "ssh_port", 22) or 22),
                                 cache=getattr(o, "cache", None) or DEFAULT_CACHE, mode="fetch")
    raise SystemExit("a source is required: --root DIR, or --ssh USER@HOST --remote-root DIR")


def describe_source(src) -> dict:
    """The source for the record, without the key path (the record book is shared)."""
    d = dict(src.describe())
    d.pop("key", None)
    return d


# ---------------------------------------------------------------------------------------------
# the command table (SPEC section 4); slice 1: moves 0 to 2
# ---------------------------------------------------------------------------------------------
def _cmd(move: int, name: str, module: str, sub: str | None, args: list, *, outputs=(), inputs=()) -> dict:
    return {"move": move, "name": name, "module": module, "sub": sub, "args": [str(a) for a in args],
            "outputs": list(outputs), "inputs": list(inputs)}


def build_plan(o: argparse.Namespace) -> list[dict]:
    out = str(Path(o.out))
    O = ["--out", out]
    S = source_flags(o)
    declared = [str(DECLARED_DIR / f) for f in DECLARED_FILES]
    P: list[dict] = []
    # ---- move 0: inputs and index (SPEC move 0)
    a0 = [*O, *S, "--cut-declared", o.cut_declared, "--cut-measured", o.cut_measured]
    if getattr(o, "encoding_out", None):
        a0 += ["--encoding-out", o.encoding_out]
    if getattr(o, "allow_unmatched_declared", False):
        a0.append("--allow-unmatched-declared")
    if getattr(o, "room_removal", False):
        a0.append("--room-removal")
    P.append(_cmd(0, "inputs and index", "inputs", "index", a0,
                  outputs=["cells.csv", "params.json", "inputs/sha256.json", "moves/00_index/index.json"], inputs=declared))
    # ---- move 1: extract (SPEC move 1), one recording at a time, resumable per recording
    a1 = [*O, *S, "--store", o.store]
    P.append(_cmd(1, "extract", "extract", "extract", a1,
                  outputs=["moves/01_extract/extract.json"], inputs=["cells.csv", "params.json"]))
    # ---- move 2: sanity (SPEC move 2); a violation stops the run
    P.append(_cmd(2, "sanity", "sanity", "check", [*O],
                  outputs=["moves/02_sanity/sanity.json"], inputs=["cells.csv", "moves/01_extract/extract.json"]))
    return P


# ---------------------------------------------------------------------------------------------
# the record book (SPEC 7): the encoding toolkit's ledger shape
# ---------------------------------------------------------------------------------------------
def load_ledger(out: Path) -> dict:
    p = Path(out) / LEDGER
    if p.exists():
        try:
            return read_json(p)
        except Exception:
            pass
    return {"schema": "plan12.driver_state.v1", "params": {}, "citation": CITATION, "package_version": __version__,
            "toolkit_fingerprint": None, "runs": [], "commands": []}


def save_ledger(out: Path, ledger: dict) -> None:
    write_json(Path(out) / LEDGER, ledger)


def _cmd_key(c: dict) -> str:
    return f"{c['move']}:{c['name']}"


def _last_done(ledger: dict, key: str) -> dict | None:
    for e in reversed(ledger.get("commands", [])):
        if e.get("key") == key and e.get("status") == "done":
            return e
    return None


def _output_exists(out: Path, spec: str) -> bool:
    """`path` under <out>, or `a|b` (either)."""
    if "|" in spec:
        return any(_output_exists(out, s) for s in spec.split("|"))
    return (Path(out) / spec).exists()


def _input_hash(out: Path, spec: str) -> str:
    """The sha256 of one declared input: a path under <out>, or an absolute path (the declared files);
    `absent` when it does not exist."""
    p = Path(spec) if Path(spec).is_absolute() else Path(out) / spec
    return sha256_file(p) if p.is_file() else "absent"


def _inputs_sha256(out: Path, specs: list[str]) -> dict:
    return {spec: _input_hash(out, spec) for spec in specs}


def _outputs_sha256(out: Path, specs: list[str]) -> dict:
    res = {}
    for spec in specs:
        for s in spec.split("|"):
            p = Path(out) / s
            if p.is_file():
                res[s] = sha256_file(p)
            elif p.is_dir():
                res[s] = "directory"
            else:
                res[s] = "absent"
    return res


def _argv_signature(argv) -> list:
    """argv without the volatile tokens (none yet in plan12; kept for the rule's shape)."""
    return list(argv or [])


def _stale_reason(out: Path, prev: dict, inputs: list[str], argv) -> str | None:
    cur = _inputs_sha256(out, inputs)
    old = prev.get("inputs_sha256", {})
    for k, v in cur.items():
        if old.get(k, "absent") != v:
            return f"stale: {Path(k).name} changed since move {prev.get('move')} ({prev.get('finished_at', '?')})"
    if argv is not None and prev.get("argv") is not None and _argv_signature(argv) != _argv_signature(prev.get("argv")):
        return f"stale: arguments changed since move {prev.get('move')} ({prev.get('finished_at', '?')})"
    return None


def _say(rec: dict) -> None:
    print(f"[move {rec['move']:>2}] {rec['name']:<28} {rec.get('status', '')}" + (f"  ({rec['stale']})" if rec.get("stale") else ""), flush=True)


# ---------------------------------------------------------------------------------------------
# one writer per record book (SPEC 1.4)
# ---------------------------------------------------------------------------------------------
def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (TypeError, ValueError):
        return False
    return True


def acquire_lock(out: Path) -> Path:
    """Refuse a second driver on the same <out>: the lock names the writer's pid; a lock whose pid is
    gone (a crash, a power cut) is taken over and said so."""
    out.mkdir(parents=True, exist_ok=True)
    lock = out / LOCK
    if lock.exists():
        try:
            old = read_json(lock)
        except Exception:
            old = {}
        pid = old.get("pid")
        if pid and _pid_alive(pid) and int(pid) != os.getpid():
            raise SystemExit(f"refused: a driver is already writing this record book ({lock}: pid {pid}, started {old.get('started_at')}); "
                             "one writer per output folder (SPEC 1.4)")
        print(f"[driver] stale lock from pid {pid} ({old.get('started_at')}) taken over", flush=True)
    write_json(lock, {"pid": os.getpid(), "started_at": now_iso(), "argv": sys.argv})
    return lock


def other_writers(out: Path) -> list[dict]:
    """Drivers of this package on this machine writing the same <out> (a `ps` check, as the console's
    Encoding paper tab does), whatever started them."""
    try:
        p = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    found = []
    needle = f"{PACKAGE_NAME}.run_moves"
    target = str(Path(out).resolve())
    for line in p.stdout.splitlines():
        line = line.strip()
        if needle not in line or " run " not in line:
            continue
        try:
            pid_s, cmd = line.split(None, 1)
            pid = int(pid_s)
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        toks = cmd.split()
        try:
            o = toks[toks.index("--out") + 1]
        except (ValueError, IndexError):
            continue
        if str(Path(os.path.expanduser(o)).resolve()) == target:
            found.append({"pid": pid, "command": cmd[:300]})
    return found


# ---------------------------------------------------------------------------------------------
# running the plan
# ---------------------------------------------------------------------------------------------
def _module_path(module: str) -> Path:
    return _HERE / f"{module}.py"


def run_plan(o: argparse.Namespace, plan: list[dict]) -> int:
    out = Path(os.path.expanduser(str(o.out)))
    out.mkdir(parents=True, exist_ok=True)
    others = other_writers(out)
    if others:
        print(f"refused: another driver is running on this output root (pid {others[0]['pid']}): {others[0]['command']}", file=sys.stderr)
        return 3
    lock = acquire_lock(out)
    try:
        return _run_plan_locked(o, plan, out)
    finally:
        try:
            if lock.exists() and read_json(lock).get("pid") == os.getpid():
                lock.unlink()
        except Exception:
            pass


def _run_plan_locked(o: argparse.Namespace, plan: list[dict], out: Path) -> int:
    ledger = load_ledger(out)
    ledger["params"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(o).items() if k not in ("func",)}
    ledger["params"].pop("ssh_key", None)                       # a path on the author's machine; not for the shared record
    ledger["toolkit_fingerprint"] = toolkit_fingerprint()
    ledger["package_version"] = __version__
    run_rec = {"started_at": now_iso(), "moves": parse_moves(o.moves), "argv": sys.argv, "dry_run": bool(o.dry_run),
               "pid": os.getpid(), "fingerprint": ledger["toolkit_fingerprint"]["sha256"]}
    ledger.setdefault("runs", []).append(run_rec)
    save_ledger(out, ledger)
    selected = set(parse_moves(o.moves))
    rc_final = 0
    for c in plan:
        if c["move"] not in selected:
            continue
        key = _cmd_key(c)
        rec = {"key": key, "move": c["move"], "name": c["name"], "module": c["module"], "sub": c["sub"],
               "started_at": now_iso(), "inputs_sha256": _inputs_sha256(out, c["inputs"]), "outputs": list(c["outputs"]),
               "fingerprint": ledger["toolkit_fingerprint"]["sha256"]}
        argv = [sys.executable, "-m", f"{PACKAGE_NAME}.{c['module']}"] + ([c["sub"]] if c["sub"] else []) + c["args"]
        rec["argv"] = argv
        rec["cwd"] = str(_HERE.parent)
        prev = _last_done(ledger, key)
        if not o.force and prev is not None and c["outputs"] and all(_output_exists(out, s) for s in c["outputs"]):
            stale = _stale_reason(out, prev, c["inputs"], argv)
            if stale is None:
                rec.update(status="skipped: outputs exist and inputs unchanged", exit_code=0, finished_at=now_iso())
                ledger["commands"].append(rec); save_ledger(out, ledger); _say(rec); continue
            rec["stale"] = stale
        if not _module_path(c["module"]).exists():
            rec.update(status=f"failed: module {c['module']}.py absent", exit_code=2, finished_at=now_iso())
            ledger["commands"].append(rec); save_ledger(out, ledger); _say(rec)
            print(f"missing module: {_module_path(c['module'])}", file=sys.stderr)
            return 2
        if o.dry_run:
            # the module prints what it would do (for an ssh source: the listing / fetch commands) and touches nothing
            proc = subprocess.run(argv + ["--dry-run"], cwd=str(_HERE.parent), capture_output=True, text=True)
            rec.update(status="dry-run", exit_code=proc.returncode, finished_at=now_iso(),
                       stdout_tail=proc.stdout[-6000:], stderr_tail=proc.stderr[-2000:])
            ledger["commands"].append(rec); save_ledger(out, ledger); _say(rec)
            if proc.stdout.strip():
                print(proc.stdout.rstrip(), flush=True)
            continue
        t0 = time.time()
        proc = subprocess.run(argv, cwd=str(_HERE.parent), capture_output=True, text=True)
        rec.update(exit_code=proc.returncode, finished_at=now_iso(), elapsed_s=round(time.time() - t0, 3),
                   stdout_tail=proc.stdout[-4000:], stderr_tail=proc.stderr[-4000:],
                   status="done" if proc.returncode == 0 else f"failed: exit {proc.returncode}",
                   outputs_sha256=_outputs_sha256(out, c["outputs"]))
        ledger["commands"].append(rec); save_ledger(out, ledger); _say(rec)
        if proc.returncode != 0:
            print(proc.stderr[-2000:], file=sys.stderr)
            print(f"stopped at move {c['move']} ({c['name']}), exit {proc.returncode}", file=sys.stderr)
            rc_final = proc.returncode if proc.returncode in (1, 2) else 1
            break
    run_rec["finished_at"] = now_iso()
    run_rec["exit_code"] = rc_final
    save_ledger(out, ledger)
    return rc_final


# ---------------------------------------------------------------------------------------------
# status: the record book, move by move, with the stopped-mid-run rule (SPEC 7)
# ---------------------------------------------------------------------------------------------
def interrupted_moves(ledger: dict, plan: list[dict]) -> dict:
    """{move: started_at} of the moves whose latest attempt was cut short: a `runs` entry with no
    `finished_at` whose pid is gone, and the first of its selected moves with a command that got no
    record during it; cleared once a later attempt records every command of the move."""
    runs = ledger.get("runs", []) or []
    cmds = ledger.get("commands", []) or []
    res: dict = {}
    for i, r in enumerate(runs):
        if r.get("finished_at"):
            continue
        if r.get("pid") and _pid_alive(r["pid"]):
            continue                                              # the live one
        t0 = r.get("started_at") or ""
        t1 = runs[i + 1].get("started_at") if i + 1 < len(runs) else None
        during = {rec.get("key") for rec in cmds if (rec.get("started_at") or "") >= t0 and (t1 is None or (rec.get("started_at") or "") < t1)}
        for m in r.get("moves") or []:
            keys = [c["key"] for c in [{"key": _cmd_key(c), "move": c["move"]} for c in plan] if c["move"] == m]
            if keys and any(k not in during for k in keys):
                later = {rec.get("key") for rec in cmds if (rec.get("started_at") or "") > t0 and (t1 is None or (rec.get("started_at") or "") >= t1)}
                if not all(k in later for k in keys):
                    res[m] = t0
                break
    return res


def move_states(out: Path, plan: list[dict]) -> list[dict]:
    ledger = load_ledger(out)
    last: dict = {}
    for rec in ledger.get("commands", []):
        if str(rec.get("status")) == "dry-run" and rec.get("key") in last:
            continue                                              # a dry run records the command, it does not undo a result
        last[rec.get("key")] = rec
    cut = interrupted_moves(ledger, plan)
    rows = []
    for m in sorted({c["move"] for c in plan}):
        cmds = [c for c in plan if c["move"] == m]
        sts = []
        for c in cmds:
            rec = last.get(_cmd_key(c))
            s = str((rec or {}).get("status") or "not run")
            sts.append("done" if s.startswith(("done", "skipped", "kept")) else "failed" if s.startswith("failed") else
                       "not run" if s.startswith(("not run", "dry-run")) else "other")
        if m in cut:
            state = "not run"
        elif all(s == "not run" for s in sts):
            state = "not run"
        elif any(s == "failed" for s in sts):
            state = "failed"
        elif all(s == "done" for s in sts):
            state = "done"
        else:
            state = "partial"
        rows.append({"move": m, "name": MOVE_NAMES.get(m, f"move {m}"), "state": state,
                     "done": sum(1 for s in sts if s == "done"), "n": len(cmds), "interrupted_at": cut.get(m)})
    return rows


def print_status(out: Path, plan: list[dict]) -> int:
    ledger = load_ledger(out)
    fp = ledger.get("toolkit_fingerprint") or {}
    print(f"record book: {Path(out) / LEDGER}")
    print(f"runs: {len(ledger.get('runs', []))}   command records: {len(ledger.get('commands', []))}   "
          f"fingerprint: {str(fp.get('sha256', '?'))[:16]} ({fp.get('n_py_files', '?')} files)   now: {toolkit_fingerprint()['sha256'][:16]}")
    for r in move_states(out, plan):
        extra = f"  (stopped mid-run at {r['interrupted_at']}: run it again; the driver resumes what finished)" if r["interrupted_at"] else ""
        print(f"  [move {r['move']:>2}] {r['name']:<28} {r['state']:<8} {r['done']}/{r['n']}{extra}")
    for rec in ledger.get("commands", [])[-8:]:
        print(f"    {rec.get('started_at', '')[11:19]} move {rec.get('move')} {rec.get('name'):<20} {str(rec.get('status'))[:60]}")
    return 0


def print_plan(plan: list[dict], moves: str) -> None:
    sel = set(parse_moves(moves))
    for c in plan:
        if c["move"] in sel:
            print(f"[move {c['move']:>2}] {c['name']:<28} python3 -m {PACKAGE_NAME}.{c['module']} " + (c["sub"] + " " if c["sub"] else "") + " ".join(c["args"]))


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------
def _add_run_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--out", required=True, help="the output folder (SPEC 7; decision D4)")
    ap.add_argument("--moves", default="0-2", help="moves to run, e.g. 0-2, 1, 0-1,2 (slice 1 plans 0 to 2)")
    add_source_args(ap)
    ap.add_argument("--store", default=DEFAULT_STORE, help="the L1 store the per-page extracts go to (default: the console's)")
    ap.add_argument("--encoding-out", default=None,
                    help="decision D2: the named encoding run's output folder, read only (its preconditions decide admissibility as there; its extract is move 6's baseline)")
    ap.add_argument("--cut-declared", type=int, default=CUT_DECLARED, help="the declared start-up cut in pairs (SPEC 2)")
    ap.add_argument("--cut-measured", type=int, default=CUT_MEASURED, help="the measured start-up cut in pairs (SPEC 2)")
    ap.add_argument("--allow-unmatched-declared", action="store_true",
                    help="do not stop when a declared keep-first row names no recording (the smoke corpus)")
    ap.add_argument("--room-removal", action="store_true", help="switch move 9 on (decision D1; off by default, used from slice 3)")
    ap.add_argument("--force", action="store_true", help="re-run every selected command")
    ap.add_argument("--dry-run", action="store_true", help="record every command; the modules print what they would do and touch nothing")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="plan12_grounding driver: the grounding paper's moves in order")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); _add_run_args(r)
    p = sub.add_parser("plan"); _add_run_args(p)
    s = sub.add_parser("status"); _add_run_args(s)
    o = ap.parse_args(argv)
    try:
        moves = parse_moves(o.moves)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    if o.cmd == "status":
        return print_status(Path(os.path.expanduser(o.out)), build_plan(o))
    if 0 in moves and not (o.root or o.ssh):
        print("missing input: move 0 needs a source: --root DIR, or --ssh USER@HOST --remote-root DIR", file=sys.stderr)
        return 2
    if o.ssh and not o.remote_root:
        print("--ssh needs --remote-root", file=sys.stderr)
        return 2
    plan = build_plan(o)
    if o.cmd == "plan":
        print_plan(plan, o.moves)
        return 0
    if 0 not in moves and not (Path(os.path.expanduser(o.out)) / "cells.csv").exists():
        print(f"missing input: {Path(o.out) / 'cells.csv'} (run move 0 first)", file=sys.stderr)
        return 2
    try:
        return run_plan(o, plan)
    except SystemExit as e:
        if e.code not in (None, 0):
            print(str(e), file=sys.stderr)
        return int(e.code or 0) if isinstance(e.code, int) else 3
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
