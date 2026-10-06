#!/usr/bin/env python3
"""encoding_panel.py -- the console's window onto plan11_encoding_ladder, the encoding paper's toolkit.

The toolkit is the source of truth for the paper's numbers and must stay runnable from the shell
exactly as its RUNBOOK.md says. So this module

  - launches the toolkit's own driver, one move at a time, in the runbook's order
    (`python3 -m plan11_encoding_ladder.run_moves run --out O --root R --moves N <flags>` from
    VM_Capture_QEMU/, the driver's own cwd), with the flags the author set and nothing else;
    the commands a move will run are the driver's own plan (`run_moves plan`), never a copy;
  - reads the toolkit's files as they are: the ledger (`driver_state.json`), `cells.csv`, the
    extract sidecars, the gate CSV/JSON records, the report tables and figures, the params blocks;
    nothing here computes a number, and a refusal string is passed through as written;
  - records every launch (the toolkit's commit state, the command line, the params block the
    driver wrote) under `<out>/.console/launches/`, a directory the toolkit does not read;
  - never writes under plan11_encoding_ladder/ and never touches a server.

Everything the page shows comes through these functions; the bridge only maps them to routes.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
QEMU_DIR = HERE.parent
TOOLKIT_NAME = "plan11_encoding_ladder"
TOOLKIT = QEMU_DIR / TOOLKIT_NAME
RUNBOOK = TOOLKIT / "RUNBOOK.md"
DEFAULT_CONFIG = Path(os.path.expanduser("~/.cache/plan10/encoding_config.json"))
CONSOLE_DIR = ".console"                 # under <out>; the toolkit neither reads nor hashes it
MAX_MOVE = 17                            # move 15: the optional LORO luck checks (SPEC_epoch2 Part 4 item 29); move 16: the
                                         # corrected instrument check, the idle common-ground test and the second Table 2
                                         # (added 2026-10-05, item 30); move 17: the new-block test on the five readings (added
                                         # 2026-10-06, item 31); the EUSIPCO row still follows move 14, all three being optional
OPTIONAL_MOVES = (15, 16, 17)            # the waiting rule (2026-10-06): an optional move waits for the moves whose files it reads
                                         # (the producers of its declared inputs), not for the move numbered before it; moves 0 to 14
                                         # keep the runbook's order
EUSIPCO_KEY = "eusipco"                  # the two runbook commands the driver does not schedule

# RUNBOOK.md section 1 (the paper run) and section 0b (the smoke run): the flag values those two
# command lines carry. Presets set exactly these; the author can change any of them.
PRESETS = {
    "paper": {"label": "paper run (RUNBOOK section 1)",
              "flags": {"assume_failed_zero": True, "assume_reason": "AA A5: any failed job re-runs the whole cell",
                        "keep_first_pairs": "plan11_encoding_ladder/declared/keep_first_pairs.csv",
                        "seed_map": "plan11_encoding_ladder/declared/seed_map.csv",
                        "n_jobs": 4, "null_perm": 500, "null_splits": "loko,within_trace"}},
    "smoke": {"label": "smoke run on the synthetic corpus (RUNBOOK section 0b)",
              "flags": {"null_perm": 20, "n_jobs": 2, "duration_s": 77.28, "assume_failed_zero": True, "assume_reason": "smoke run"}},
}

# The views of the brief, each a list of the toolkit's own files. `move` is the move that writes
# the first file; a missing file is reported by path, never drawn.
VIEWS = [
    {"id": "calibration", "title": "the calibration pulses, G-C per rung", "move": 3,
     "files": ["gates/gc.csv", "report/figures/fig_apf_per_kernel.png"],
     "note": "gc.csv is the toolkit's record of the pulse per rung and rep (gemm's jump and dip, stat_a to stat_c, the verdict); "
             "the toolkit draws no separate pulse figure, so the count per kernel over time stands beside it."},
    {"id": "floor", "title": "the floor from the idle cells, G-K0 and G-F", "move": 4,
     "files": ["gates/gk0.csv", "gates/gf.csv", "gates/gf_floors.json"]},
    {"id": "count", "title": "the count over time per kernel, reps overlaid", "move": 5,
     "files": ["report/figures/fig_apf_per_kernel.png"]},
    {"id": "level_matched", "title": "the level-matched sets side by side", "move": 5,
     "files": ["report/figures/fig_level_matched.png"]},
    {"id": "temporal_grid", "title": "the temporal-gate grid, the selected point marked", "move": 6,
     "files": ["gates/table5_grid.csv", "gates/selection.json", "gates/table5_long.csv", "report/figures/fig_table5_grid.png", "gates/g3_flags.csv", "gates/alias.csv"],
     "note": "every grid point is kept; the selected (W, H) per rung is read from selection.json and only highlighted."},
    {"id": "table6", "title": "Table 6: APF under the three splits", "move": 7,
     "files": ["report/tables/table6.csv", "report/tables/table6.md", "gates/gl.csv", "gates/gn.csv", "gates/gx.csv", "gates/gl2_rerun.json"]},
    {"id": "fused_plane", "title": "the fused plane, one panel per kernel, idle cells overlaid", "move": 8,
     "files": ["report/figures/fig_fused_plane.png", "gates/gj.csv"]},
    {"id": "j_hist", "title": "the overlap histograms with both nulls", "move": 9,
     "files": ["report/figures/fig_j_hist.png"]},
    {"id": "ratios", "title": "the amount ratios, G-DEC", "move": 10,
     "files": ["report/figures/fig_ratio_hist.png", "report/figures/fig_floyd_decay.png", "gates/gdec.csv"]},
    {"id": "wapf", "title": "wAPF over APF", "move": 11,
     "files": ["report/tables/table_wapf_over_apf.csv", "report/tables/table_wapf_over_apf.md"]},
    {"id": "table5", "title": "Table 5 and Table 5-G3", "move": 12,
     "files": ["report/tables/table5.csv", "report/tables/table5.md", "report/tables/table5_g3.csv"]},
    {"id": "table7", "title": "Table 7", "move": 12,
     "files": ["report/tables/table7.csv", "report/tables/table7.md", "gates/gf_check.json"]},
    {"id": "table8", "title": "Table 8, read as assignments with counts", "move": 12,
     "files": ["report/tables/table8.csv", "report/tables/table8.md", "gates/clustering.csv"]},
    {"id": "variance", "title": "the variance table, G-V", "move": 12,
     "files": ["report/tables/tablegv.csv", "report/tables/tablegv.md", "gates/gv_summary.csv", "gates/gv.csv"]},
    {"id": "comparisons", "title": "the comparison gates: G-L, G-DIM, G-M, G-X", "move": 12,
     "files": ["gates/gl.csv", "gates/gdim.csv", "gates/gm.csv", "gates/gx.csv"]},
    {"id": "status_tables", "title": "Table 4 status, the preconditions copy, the manifest", "move": 12,
     "files": ["report/tables/table4_status.csv", "report/tables/preconditions.csv", "report/manifest.json"]},
    {"id": "comparators", "title": "the comparator rows: Savoldi 2010, Dhodapkar-Smith 2003, Law 2010", "move": 14,
     "files": ["report/tables/table7_comparators.csv", "report/tables/table_comparators.csv", "gates/comparators/verdicts.csv",
               "report/figures/fig_dhodapkar_sweep.png", "gates/comparators/savoldi_per_kernel.csv", "gates/comparators/law.csv"]},
    {"id": "eusipco", "title": "the EUSIPCO tables (Table 2, Table 3)", "move": EUSIPCO_KEY,
     "files": ["report/tables/eusipco_table2.csv", "report/tables/eusipco_table2.md", "report/tables/eusipco_table3.csv", "report/tables/eusipco_table3.md"]},
    {"id": "figures_all", "title": "every figure the toolkit wrote", "move": 12,
     "files": ["report/figures/figures.json", "report/figures/fig_piano_roll.png", "report/figures/SKIPPED.txt"]},
    {"id": "new_blocks", "title": "the new-block test: accuracy by level and reading, by position, per kernel, the run level beside idle, the kernel confusions (move 17)", "move": 17,
     "note": "added 2026-10-06; no label-shuffle null (the scores' null column says why); chance and the majority class are the baselines; the comparators take no part",
     "files": ["report/figures/new_blocks_accuracy.svg", "report/figures/new_blocks_by_position.svg", "report/figures/new_blocks_per_kernel.svg",
               "report/figures/new_blocks_run_level.svg", "report/figures/new_blocks_confusion_apf.svg", "report/figures/new_blocks_confusion_content.svg",
               "report/tables/new_blocks_scores.md", "report/tables/new_blocks_margins.md", "report/tables/new_blocks_run_level.md", "report/tables/new_blocks_by_position.md",
               "gates/added/new_blocks.csv", "gates/added/new_blocks/W64_H32/summary.json", "gates/added/new_blocks/own/summary.json"]},
]

TEXT_KINDS = {".csv": "csv", ".json": "json", ".md": "md", ".tex": "tex", ".txt": "text", ".log": "text", ".py": "text"}
BINARY_TYPES = {".png": "image/png", ".pdf": "application/pdf", ".npz": "application/octet-stream", ".npy": "application/octet-stream",
                ".zst": "application/zstd", ".svg": "image/svg+xml"}
MAX_TEXT_BYTES = 8 * 1024 * 1024
MAX_CSV_ROWS = 20000


class PanelError(ValueError):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# the toolkit on disk: its driver's flags, its plan, its identity, its runbook
# ---------------------------------------------------------------------------

def toolkit_present() -> bool:
    return (TOOLKIT / "run_moves.py").exists() and RUNBOOK.exists()


def _import_driver():
    if str(QEMU_DIR) not in sys.path:
        sys.path.insert(0, str(QEMU_DIR))
    import importlib
    return importlib.import_module(f"{TOOLKIT_NAME}.run_moves")


def _parser():
    RM = _import_driver()
    ap = argparse.ArgumentParser(prog="run_moves.py run", add_help=False)
    RM._add_run_args(ap)
    return ap


def driver_flags() -> list[dict]:
    """Every flag of the driver's `run` subcommand, from the toolkit's own argparse definition:
    name, destination, default, type, choices, help. The page builds its flag form from this."""
    ap = _parser()
    out = []
    for a in ap._actions:
        if not a.option_strings or a.dest in ("out", "root", "moves", "force", "dry_run", "help"):
            continue
        kind = "bool" if a.nargs == 0 else ("int" if a.type is int else "float" if a.type is float else "str")
        out.append({"flag": a.option_strings[-1], "dest": a.dest, "kind": kind, "default": a.default,
                    "choices": [c for c in (a.choices or []) if c is not None] or None, "help": a.help or ""})
    return out


def flag_tokens(flags: dict) -> list[str]:
    """The command-line tokens for the flags the author set (every set flag is emitted, whether or
    not it equals the default, so the argv of a move is the same on every launch: the driver
    compares argv to decide staleness)."""
    ap = _parser()
    by_dest = {a.dest: a for a in ap._actions if a.option_strings}
    toks: list[str] = []
    for dest, val in (flags or {}).items():
        a = by_dest.get(dest)
        if a is None or val is None or val == "":
            continue
        opt = a.option_strings[-1]
        if a.nargs == 0:
            if val in (True, "true", "True", 1, "1"):
                toks.append(opt)
            continue
        if a.type is int:
            val = int(float(val))
        elif a.type is float:
            val = float(val)
        toks += [opt, str(val)]
    return toks


def driver_argv(sub: str, out: str, root: str, move: int | str, flags: dict, force: bool = False) -> list[str]:
    """`python3 -m plan11_encoding_ladder.run_moves <sub> --out O --root R --moves N <flags> [--force]`."""
    argv = [sys.executable, "-m", f"{TOOLKIT_NAME}.run_moves", sub, "--out", str(out), "--root", str(root), "--moves", str(move)]
    argv += flag_tokens(flags)
    if force:
        argv.append("--force")
    return argv


def eusipco_argv(out: str, standalone: str | None = None) -> list[list[str]]:
    """The two EUSIPCO commands of RUNBOOK.md ('The EUSIPCO outputs'), which the driver does not
    schedule; `--standalone` only when the author gives the path (it writes outside <out>)."""
    a = [sys.executable, "-m", f"{TOOLKIT_NAME}.tables_eusipco", "--out", str(out)]
    b = [sys.executable, "-m", f"{TOOLKIT_NAME}.latex_skeleton_eusipco", "--out", str(out)]
    if standalone:
        b += ["--standalone", str(standalone)]
    return [a, b]


def shell_line(argv: list[str]) -> str:
    """The argv as a shell line, `python3` for the interpreter the way the runbook writes it."""
    toks = ["python3" if i == 0 else t for i, t in enumerate(argv)]
    return " ".join(shlex.quote(t) for t in toks)


def build_plan(out: str, root: str, flags: dict) -> list[dict]:
    """The driver's own move table for these flags (`run_moves.build_plan`), each command with its
    module, subcommand, args, declared outputs and inputs; the same function the driver runs."""
    RM = _import_driver()
    ap = _parser()
    o = ap.parse_args(["--out", str(out), "--root", str(root)] + flag_tokens(flags))
    plan = RM.build_plan(o)
    for c in plan:
        c["key"] = f"{c['move']}:{c['name']}"
        c["argv"] = None if c["internal"] else [sys.executable, "-m", f"{TOOLKIT_NAME}.{c['module']}"] + ([c["sub"]] if c["sub"] else []) + list(c["args"])
        c["line"] = f"(internal) {c['sub']} {' '.join(c['args'])}" if c["internal"] else shell_line(c["argv"])
    return plan


def plan_text(out: str, root: str, flags: dict, move: int) -> str:
    """What `run_moves plan --moves N` prints: the exact argument lists, in the driver's words."""
    argv = driver_argv("plan", out, root, move, flags)
    p = subprocess.run(argv, cwd=str(QEMU_DIR), capture_output=True, text=True, timeout=120)
    return (p.stdout or "") + (("\n" + p.stderr) if p.returncode else "")


def runbook_sections() -> dict:
    """The runbook's own words per move: the `### Move N: <title>` heading and the section's text
    (its commands, what it writes, what to look at). Read, never paraphrased."""
    if not RUNBOOK.exists():
        return {}
    text = RUNBOOK.read_text(encoding="utf-8", errors="replace")
    heads = list(re.finditer(r"^### (?:Move (\d+): (.*)|(The EUSIPCO outputs.*))$", text, flags=re.M))
    out = {}
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else text.find("\n## ", m.end())
        body = text[m.end():end if end > 0 else None].strip()
        if m.group(1) is not None:
            key, title = int(m.group(1)), m.group(2).strip()
        else:
            key, title = EUSIPCO_KEY, m.group(3).strip()
        look = ""
        mm = re.search(r"^Look at:(.*?)(?=\n\n|\Z)", body, flags=re.S | re.M)
        if mm:
            look = "Look at:" + mm.group(1).strip()
        out[key] = {"title": title, "body": body, "look_at": look}
    return out


def _git(args: list[str]) -> str:
    try:
        p = subprocess.run(["git", *args], cwd=str(QEMU_DIR), capture_output=True, text=True, timeout=20)
        return p.stdout.strip() if p.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def toolkit_identity() -> dict:
    """What the launch record names: the repository's HEAD, whether the toolkit's files are tracked
    and clean at that commit, the toolkit's package version, and a content fingerprint (sha256
    over the sorted sha256 of every top-level .py file) that identifies the code that ran even
    while it is not committed."""
    head = _git(["rev-parse", "HEAD"])
    tracked = bool(_git(["ls-files", "--error-unmatch", f"{TOOLKIT_NAME}/run_moves.py"]))
    porcelain = _git(["status", "--porcelain", "--", TOOLKIT_NAME])
    dirty = [ln for ln in porcelain.splitlines() if ln.strip()]
    files = sorted(p for p in TOOLKIT.glob("*.py") if p.is_file()) if TOOLKIT.is_dir() else []
    h = hashlib.sha256()
    per = {}
    for p in files:
        d = hashlib.sha256(p.read_bytes()).hexdigest()
        per[p.name] = d
        h.update(f"{p.name}:{d}\n".encode())
    version = None
    rc = TOOLKIT / "_report_common.py"
    if rc.exists():
        m = re.search(r'^PACKAGE_VERSION\s*=\s*["\']([^"\']+)["\']', rc.read_text(errors="replace"), flags=re.M)
        if m:
            version = m.group(1)
    if version is None and (TOOLKIT / "__init__.py").exists():
        m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', (TOOLKIT / "__init__.py").read_text(errors="replace"))
        if m:
            version = m.group(1)
    return {"path": str(TOOLKIT), "present": toolkit_present(), "repo_head": head or None, "repo_head_short": head[:12] if head else None,
            "tracked": tracked, "dirty_entries": len(dirty), "dirty_sample": dirty[:5],
            "committed": bool(head) and tracked and not dirty,
            "content_fingerprint": h.hexdigest()[:16] if files else None, "n_py_files": len(files), "package_version": version,
            "python": sys.executable, "cwd": str(QEMU_DIR)}


# ---------------------------------------------------------------------------
# the toolkit's files, read as they are
# ---------------------------------------------------------------------------

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


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_path(out: Path, rel: str) -> Path:
    """A path under <out>, or a PanelError; `..` and absolute paths are refused."""
    rel = str(rel or "")
    if not rel or rel.startswith(("/", "\\")) or ".." in Path(rel).parts:
        raise PanelError(f"not a path under the output root: {rel!r}")
    p = (Path(out) / rel).resolve()
    o = Path(out).resolve()
    if p != o and o not in p.parents:
        raise PanelError(f"not a path under the output root: {rel!r}")
    return p


def text_file(out: Path, rel: str) -> dict:
    """A CSV as header and rows, a JSON as its object, anything else as text; the path and size
    always; cells untouched (a refusal string stays a string, an empty cell stays empty)."""
    p = safe_path(out, rel)
    if not p.exists():
        return {"path": rel, "exists": False, "kind": None}
    if p.is_dir():
        return {"path": rel, "exists": True, "kind": "dir", "entries": listing(out, rel)["entries"]}
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
        # the sidecar the toolkit writes beside a CSV-only result: `<name>.params.json`, or one per rung / grid (`gc.apf.params.json`)
        sib = sorted(p.parent.glob(p.stem + ".*params.json")) + sorted(p.parent.glob(p.stem + ".params.json"))
        seen = []
        for pj in sib:
            rel = str(pj.resolve().relative_to(Path(out).resolve()))
            if rel not in seen:
                seen.append(rel)
        if seen:
            d["params_file"] = seen[0] if len(seen) == 1 else None
            d["params_files"] = seen
    elif kind == "json":
        try:
            d["json"] = read_json(p)
        except json.JSONDecodeError as e:
            d["error"] = f"not valid JSON: {e}"
            d["text"] = p.read_text(encoding="utf-8", errors="replace")
    else:
        d["text"] = p.read_text(encoding="utf-8", errors="replace")
    return d


def listing(out: Path, rel: str = "") -> dict:
    p = safe_path(out, rel) if rel else Path(out).resolve()
    if not p.exists():
        return {"path": rel, "exists": False, "entries": []}
    ents = []
    for c in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name)):
        if c.name == CONSOLE_DIR and not rel:
            continue
        st = c.stat()
        ents.append({"name": c.name, "dir": c.is_dir(), "bytes": None if c.is_dir() else st.st_size,
                     "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds"),
                     "path": f"{rel}/{c.name}" if rel else c.name})
    return {"path": rel, "exists": True, "entries": ents}


def params_blocks(out: Path) -> dict:
    """Every params block the toolkit wrote under <out>: each JSON file with a top-level `params`
    (result records and the `*.params.json` sidecars), the driver's own params from the ledger,
    and the author's input files as they are. Read-only."""
    o = Path(out)
    blocks = []
    if o.is_dir():
        for p in sorted(o.rglob("*.json")):
            rel = p.relative_to(o)
            if rel.parts and rel.parts[0] == CONSOLE_DIR:
                continue
            if rel.parts and rel.parts[0] == "extract" and p.name == "sidecar.json":
                continue                                    # one per cell; the cells table shows them
            try:
                j = read_json(p)
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(j, dict) and isinstance(j.get("params"), dict):
                blocks.append({"path": str(rel), "schema": j.get("schema"), "citation": j.get("citation"), "params": j["params"],
                               "written_at": j.get("written_at") or j["params"].get("written_at")})
    ledger = load_ledger(o)
    inputs = {}
    for name in ("pass_table.csv", "gk0_source.csv", "head_drop.csv", "failed_counts.csv", "idle_admissibility.json", "cell_order.csv"):
        p = o / "inputs" / name
        inputs[name] = text_file(o, f"inputs/{name}") if p.exists() else {"path": f"inputs/{name}", "exists": False}
    sidecar_n = len(list((o / "extract").glob("*/sidecar.json"))) if (o / "extract").is_dir() else 0
    return {"driver_params": (ledger or {}).get("params"), "package_version": (ledger or {}).get("package_version"),
            "inputs": inputs, "blocks": blocks, "n_blocks": len(blocks), "n_sidecars_not_listed": sidecar_n,
            "manifest": "report/manifest.json" if (o / "report" / "manifest.json").exists() else None}


# ---------------------------------------------------------------------------
# the ledger, the moves, their states
# ---------------------------------------------------------------------------

def load_ledger(out: Path) -> dict | None:
    p = Path(out) / "driver_state.json"
    if not p.exists():
        return None
    try:
        return read_json(p)
    except (OSError, json.JSONDecodeError):
        return {"error": "driver_state.json is not readable JSON", "commands": [], "runs": []}


COMPLETE_PREFIXES = ("done", "skipped:", "kept:")


def classify(status: str | None) -> str:
    """The driver's status string -> one of the board's states; the string itself is kept beside it."""
    if status is None:
        return "not run"
    s = str(status)
    if s.startswith(COMPLETE_PREFIXES):
        return "done"
    if s.startswith("failed"):
        return "failed"
    if s.startswith("refused"):
        return "refused"
    if s.startswith("dry-run"):
        return "not run"
    if s.startswith("not run"):
        return "not run"
    return "other"


def interrupted_moves(ledger: dict, plan: list[dict], running: dict | None) -> dict:
    """Moves whose latest attempt was cut short: a ledger `runs` entry with no `finished_at` that
    belongs to no live process (the driver writes `finished_at` only when it ends on its own, so a
    stopped or crashed driver leaves the entry open). The interrupted move of such a run is the
    first of its selected moves with a command that got no record during it. Cleared once every
    command of the move has a record from a later attempt. {move: started_at of the cut attempt}."""
    runs = ledger.get("runs", []) or []
    cmds = ledger.get("commands", []) or []
    out: dict = {}
    for i, r in enumerate(runs):
        if r.get("finished_at"):
            continue
        if running and i == len(runs) - 1:
            continue                                   # the live one
        t0 = r.get("started_at") or ""
        t1 = runs[i + 1].get("started_at") if i + 1 < len(runs) else None
        during = {rec.get("key") for rec in cmds if (rec.get("started_at") or "") >= t0 and (t1 is None or (rec.get("started_at") or "") < t1)}
        for m in _moves_of(",".join(str(x) for x in (r.get("moves") or [])) or None):
            keys = [c["key"] for c in plan if c["move"] == m]
            if keys and any(k not in during for k in keys):
                later = {rec.get("key") for rec in cmds if (rec.get("started_at") or "") > t0 and (t1 is None or (rec.get("started_at") or "") >= t1)}
                if not all(k in later for k in keys):
                    out[m] = t0
                break
    return out


def move_states(out: Path, plan: list[dict], running: dict | None) -> list[dict]:
    """One record per move 0..14 plus the EUSIPCO row: the runbook title, the commands (from the
    driver's plan) with their last ledger record, the move's state, and which earlier moves it
    reads from (the producers of its declared inputs). A move whose last attempt was stopped
    mid-run reads `not run` with the time it was cut, whatever its earlier records say."""
    ledger = load_ledger(out) or {}
    last: dict[str, dict] = {}
    for rec in ledger.get("commands", []):
        last[rec.get("key")] = rec
    cut = interrupted_moves(ledger, plan, running)
    producers = _producers(plan)
    sections = runbook_sections()
    outp = Path(out)
    rows = []
    for m in range(MAX_MOVE + 1):
        cmds = [c for c in plan if c["move"] == m]
        crows, states = [], []
        reads = set()
        for c in cmds:
            rec = last.get(c["key"])
            st = classify(rec.get("status") if rec else None)
            verdict = (rec or {}).get("verdict")
            if verdict and str(verdict).startswith("refused"):
                st = "refused"
            states.append(st)
            for spec in c.get("inputs", []):
                pm = _producer_of(producers, _spec_file(spec))
                if pm is not None and pm < m:
                    reads.add(pm)
            crows.append({"key": c["key"], "name": c["name"], "module": c["module"], "sub": c["sub"], "internal": c["internal"], "template": c["template"],
                          "line": c["line"], "outputs": c.get("outputs", []), "inputs": c.get("inputs", []),
                          "status": (rec or {}).get("status"), "state": st, "verdict": verdict, "stale": (rec or {}).get("stale"),
                          "exit_code": (rec or {}).get("exit_code"), "started_at": (rec or {}).get("started_at"),
                          "finished_at": (rec or {}).get("finished_at"), "elapsed_s": (rec or {}).get("elapsed_s"),
                          "stderr_tail": ((rec or {}).get("stderr_tail") or "")[-600:] if st in ("failed", "refused") else None,
                          "outputs_present": [s for s in c.get("outputs", []) if _output_exists(outp, s)]})
        batch = (running or {}).get("batch") or {}
        if running and running.get("move") == m:
            state = "running"
        elif m in (batch.get("moves") or []) and isinstance(running.get("move"), int) and m > running["move"] \
                and not all(s == "done" and (c.get("started_at") or "") >= (batch.get("started_at") or "")
                            for s, c in zip(states, crows)):
            state = "queued"                           # selected in the running batch, after the move it is on; its old state is shown beside it
        elif m in cut:
            state = "not run"                          # stopped mid-run: whatever finished earlier does not make the move done
        elif not states or all(s == "not run" for s in states):
            state = "not run"
        elif any(s == "failed" for s in states):
            state = "failed"
        elif any(s == "refused" for s in states):
            state = "refused"
        elif all(s == "done" for s in states):
            state = "done"
        elif any(s == "done" for s in states):
            state = "partial"
        else:
            state = "other"
        missing = [s for c in cmds for s in c.get("outputs", []) if state == "done" and not _output_exists(outp, s)]
        sec = sections.get(m, {})
        rows.append({"move": m, "title": sec.get("title") or f"move {m}", "state": state, "commands": crows,
                     "state_before_batch": (None if state != "queued" else
                                            ("not run" if not states or all(s == "not run" for s in states) else
                                             "done" if all(s == "done" for s in states) else "partial")),
                     "reads_from": sorted(reads), "requires": (sorted(reads) if m in OPTIONAL_MOVES else ([m - 1] if m > 0 else [])),
                     "waits_rule": ("the moves whose files it reads (optional move)" if m in OPTIONAL_MOVES else "the move before it (the runbook's order)"),
                     "outputs_missing": missing, "last_finished_at": max([c["finished_at"] or "" for c in crows] or [""]) or None,
                     "interrupted_at": cut.get(m), "look_at": sec.get("look_at", ""), "output_dirs": _output_dirs(cmds)})
    # the two EUSIPCO commands: no ledger; state from the files they write
    e_files = ["report/tables/eusipco_table2.csv", "report/tables/eusipco_table3.csv", "report/p2e_skeleton.tex"]
    present = [f for f in e_files if (outp / f).exists()]
    sec = sections.get(EUSIPCO_KEY, {})
    rows.append({"move": EUSIPCO_KEY, "title": sec.get("title") or "The EUSIPCO outputs", "state": "running" if running and running.get("move") == EUSIPCO_KEY
                 else ("done" if len(present) == len(e_files) else "partial" if present else "not run"),
                 "commands": [], "reads_from": [12, 14], "requires": [14], "outputs_missing": [f for f in e_files if f not in present],
                 "last_finished_at": None, "look_at": sec.get("look_at", ""), "output_dirs": ["report/tables", "report"],
                 "note": "not scheduled by the driver; the runbook says to run these by hand once move 14 is done"})
    for r in rows:
        req = r["requires"]
        r["runnable"] = (not running) and all(any(x["move"] == q and x["state"] == "done" for x in rows) for q in req)
    return rows


def _spec_file(spec: str) -> str:
    if spec.startswith(("json:", "csv:")):
        return spec.split(":", 2)[1]
    return spec.split("|")[0]


def _producer_of(producers: dict, path: str) -> int | None:
    """The move that writes `path`: the exact file when declared, else the earliest move declaring a file in the
    same folder (the feature files are declared one grid point per rung folder; a grid point read later is the
    same move's output). None when nothing in the plan writes there (an author input, or a file the move itself writes)."""
    if path in producers:
        return producers[path]
    folder = str(Path(path).parent)
    cands = [mv for f, mv in producers.items() if str(Path(f).parent) == folder]
    return min(cands) if cands else None


def _producers(plan: list[dict]) -> dict:
    prod: dict[str, int] = {}
    for c in plan:
        for spec in c.get("outputs", []):
            for f in spec.split("|"):
                f = _spec_file(f)
                prod.setdefault(f, c["move"])
                if f == "extract":
                    prod.setdefault("extract/", c["move"])
    return prod


def _output_exists(out: Path, spec: str) -> bool:
    if "|" in spec and not spec.startswith(("json:", "csv:")):
        return any(_output_exists(out, s) for s in spec.split("|"))
    if spec.startswith("json:"):
        _, path, key = spec.split(":", 2)
        p = out / path
        if not p.exists():
            return False
        try:
            j = read_json(p)
        except (OSError, json.JSONDecodeError):
            return False
        return key in j or any(isinstance(j.get(k), dict) and key in j[k] for k in ("selection", "rungs", "grid_complete"))
    if spec.startswith("csv:"):
        _, path, cond = spec.split(":", 2)
        p = out / path
        if not p.exists():
            return False
        col, val = cond.split("=", 1)
        try:
            header, rows, _ = read_csv_rows(p)
            i = header.index(col)
            return any(len(r) > i and r[i] == val for r in rows)
        except (ValueError, OSError):
            return False
    return (out / spec).exists()


def _output_dirs(cmds: list[dict]) -> list[str]:
    dirs = []
    for c in cmds:
        for spec in c.get("outputs", []):
            f = _spec_file(spec.split("|")[0])
            d = f if f in ("extract", "cells.csv") else str(Path(f).parent)
            if d == ".":
                d = ""
            if d not in dirs:
                dirs.append(d)
    return dirs


# ---------------------------------------------------------------------------
# the cells table: cells.csv joined with the sidecars and the preconditions, for display
# ---------------------------------------------------------------------------

def cells(out: Path) -> dict:
    o = Path(out)
    p = o / "cells.csv"
    if not p.exists():
        return {"exists": False, "path": "cells.csv", "why": "not yet written by the toolkit: run move 0 (extract index)"}
    header, rows, _ = read_csv_rows(p)
    idx = {h: i for i, h in enumerate(header)}
    pre_p = o / "gates" / "preconditions.csv"
    pre: dict[str, dict] = {}
    pre_cols: list[str] = []
    if pre_p.exists():
        ph, prow, _ = read_csv_rows(pre_p)
        pre_cols = ph
        for r in prow:
            d = dict(zip(ph, r))
            pre[d.get("cell_id", "")] = d
    excluded = set()
    pj = o / "gates" / "preconditions.json"
    if pj.exists():
        try:
            excluded = set(read_json(pj).get("excluded_cells") or [])
        except (OSError, json.JSONDecodeError):
            pass
    out_rows, n_unknown, unknown_status = [], 0, {}
    for r in rows:
        d = {h: (r[i] if i < len(r) else "") for h, i in idx.items()}
        if d.get("role") not in ("kernel", "idle"):
            # the paper's corpus is the kernels plus the idle cells; other directories under the root are
            # counted here and left in cells.csv, never listed by name in this panel
            n_unknown += 1
            unknown_status[d.get("status", "")] = unknown_status.get(d.get("status", ""), 0) + 1
            continue
        cid = d.get("cell_id", "")
        sc = o / "extract" / cid / "sidecar.json"
        side = {}
        if sc.exists():
            try:
                side = read_json(sc)
            except (OSError, json.JSONDecodeError):
                side = {"status": "sidecar.json unreadable"}
        d["n_pairs"] = side.get("n_pairs", "")
        d["extract_status"] = side.get("status", "")
        d["n_seq_gaps"] = side.get("n_seq_gaps", "")
        d["dt_est_s"] = side.get("dt_est_s", "")
        d["K_max"] = side.get("K_max", "")
        pr = pre.get(cid, {})
        d["all_hard_pass"] = pr.get("all_hard_pass", "")
        d["C1"] = pr.get("C1", "")
        d["C1_rule"] = pr.get("C1_rule", "")
        d["failed_verdict"] = pr.get("failed_verdict", "")
        d["excluded"] = "excluded" if cid in excluded else ""
        d["precondition_row"] = pr
        out_rows.append(d)
    order = {"kernel": 0, "idle": 1}
    out_rows.sort(key=lambda d: (order.get(d.get("role"), 2), d.get("kernel", ""), int(d.get("rep") or 0)))
    n_k = sum(1 for d in out_rows if d["role"] == "kernel")
    n_i = sum(1 for d in out_rows if d["role"] == "idle")
    return {"exists": True, "path": "cells.csv", "columns": header, "rows": out_rows, "n_kernel": n_k, "n_idle": n_i,
            "n_ok": sum(1 for d in out_rows if d.get("status") == "ok"), "n_other_dirs": n_unknown, "other_dirs_status": unknown_status,
            "preconditions": {"exists": pre_p.exists(), "columns": pre_cols, "path": "gates/preconditions.csv", "excluded": sorted(excluded)},
            "sidecars": {"n": sum(1 for d in out_rows if d["extract_status"]), "path": "extract/<cell_id>/sidecar.json"},
            "index_params": (read_json(o / "cells.index.json").get("params") if (o / "cells.index.json").exists() else None)}


def views(out: Path, plan: list[dict]) -> list[dict]:
    o = Path(out)
    outs = []
    for v in VIEWS:
        files = []
        for f in v["files"]:
            p = o / f
            files.append({"path": f, "exists": p.exists(), "kind": ("image" if p.suffix == ".png" else TEXT_KINDS.get(p.suffix, "binary")),
                          "bytes": p.stat().st_size if p.exists() else None,
                          "pdf": (f[:-4] + ".pdf") if f.endswith(".png") and (o / (f[:-4] + ".pdf")).exists() else None})
        outs.append({**{k: v[k] for k in ("id", "title", "move")}, "note": v.get("note"), "files": files,
                     "any": any(f["exists"] for f in files)})
    return outs


def external_drivers(out: Path, exclude_pid: int | None = None) -> list[dict]:
    """Driver processes on this machine writing to the same <out> that this console did not start
    (a shell run): found by their command line, so a second writer to the ledger is refused."""
    try:
        p = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    found = []
    needle = f"{TOOLKIT_NAME}.run_moves"
    for line in p.stdout.splitlines():
        line = line.strip()
        if not line or needle not in line or " run " not in line:
            continue
        try:
            pid_s, cmd = line.split(None, 1)
            pid = int(pid_s)
        except ValueError:
            continue
        if exclude_pid is not None and pid == exclude_pid:
            continue
        toks = shlex.split(cmd) if "'" in cmd or '"' in cmd else cmd.split()
        try:
            o = toks[toks.index("--out") + 1]
        except (ValueError, IndexError):
            continue
        if Path(o).resolve() != Path(out).resolve():
            continue
        moves = None
        if "--moves" in toks:
            with contextlib.suppress(IndexError):
                moves = toks[toks.index("--moves") + 1]
        found.append({"pid": pid, "moves": moves, "command": cmd[:300]})
    return found


def _moves_of(spec: str | None) -> list[int]:
    """`6-14`, `2,4` -> the move numbers, in order (the driver's own syntax)."""
    if not spec:
        return [0]
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        try:
            if "-" in part:
                a, b = part.split("-", 1)
                out.extend(range(int(a), int(b) + 1))
            elif part:
                out.append(int(part))
        except ValueError:
            continue
    return out


def _first_move(spec: str | None) -> int | None:
    m = _moves_of(spec)
    return m[0] if m else None


def running_batch(out: Path, plan: list[dict]) -> dict | None:
    """The batch the live driver is working through, from the ledger's last `runs` entry while it is
    open (the driver writes `started_at` and `moves` when it starts and `finished_at` when it ends):
    its moves, its start, and the move it is on (the first selected move with a command that has no
    record started since the batch began). None when the ledger has no open run."""
    runs = (load_ledger(out) or {}).get("runs") or []
    if not runs or runs[-1].get("finished_at"):
        return None
    r = runs[-1]
    t0 = r.get("started_at") or ""
    moves = [int(m) for m in (r.get("moves") or []) if isinstance(m, int) or str(m).isdigit()]
    seen = {rec.get("key") for rec in (load_ledger(out) or {}).get("commands", []) if (rec.get("started_at") or "") >= t0}
    current = next((m for m in moves if any(c["key"] not in seen for c in plan if c["move"] == m)), None)
    return {"moves": moves, "started_at": t0, "current": current,
            "spec": (f"{moves[0]}-{moves[-1]}" if moves and moves == list(range(moves[0], moves[-1] + 1)) and len(moves) > 1 else ",".join(map(str, moves)))}


def _external_move(out: Path, plan: list[dict], spec: str | None) -> int | None:
    """Which move a shell-started driver is on: the first of its selected moves with a command
    that has no ledger record yet (the driver appends a record when a command ends)."""
    ledger = load_ledger(out) or {}
    keys = {rec.get("key") for rec in ledger.get("commands", [])}
    moves = _moves_of(spec)
    for m in moves:
        if any(c["key"] not in keys for c in plan if c["move"] == m):
            return m
    return moves[-1] if moves else None


# ---------------------------------------------------------------------------
# launching the driver, one move at a time
# ---------------------------------------------------------------------------

class Panel:
    """The bridge's state for the panel: the persisted configuration and the one running process."""

    def __init__(self, config_path: Path = DEFAULT_CONFIG):
        self.config_path = Path(config_path)
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.running: dict | None = None
        self.cfg = self._load()

    # ---- configuration
    def _load(self) -> dict:
        d = {"out": "", "root": "", "preset": "paper", "flags": dict(PRESETS["paper"]["flags"]), "standalone_tex": ""}
        if self.config_path.exists():
            try:
                d.update(read_json(self.config_path))
            except (OSError, json.JSONDecodeError):
                pass
        d["flags"] = dict(d.get("flags") or {})
        return d

    def save(self):
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.config_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.cfg, indent=1))
        os.replace(tmp, self.config_path)

    def set_config(self, body: dict) -> dict:
        if "preset" in body and body["preset"] in PRESETS:
            self.cfg["preset"] = body["preset"]
            self.cfg["flags"] = dict(PRESETS[body["preset"]]["flags"])
        for k in ("out", "root", "standalone_tex"):
            if k in body:
                self.cfg[k] = str(body[k] or "").strip()
        if isinstance(body.get("flags"), dict):
            valid = {f["dest"] for f in driver_flags()}
            for k, v in body["flags"].items():
                if k not in valid:
                    raise PanelError(f"not a driver flag: {k}")
                if v in (None, "", False):
                    self.cfg["flags"].pop(k, None)
                else:
                    self.cfg["flags"][k] = v
            self.cfg["preset"] = "custom" if self.cfg["flags"] != PRESETS.get(self.cfg.get("preset", ""), {}).get("flags") else self.cfg["preset"]
        self.save()
        return self.config()

    def config(self) -> dict:
        out, root = self.cfg.get("out") or "", self.cfg.get("root") or ""
        return {"out": out, "root": root, "preset": self.cfg.get("preset"), "flags": self.cfg.get("flags", {}), "standalone_tex": self.cfg.get("standalone_tex", ""),
                "out_exists": bool(out) and Path(out).is_dir(), "root_exists": bool(root) and Path(root).is_dir(),
                "presets": {k: {"label": v["label"], "flags": v["flags"]} for k, v in PRESETS.items()},
                "driver_flags": driver_flags() if toolkit_present() else [], "toolkit": toolkit_identity(),
                "run_line": shell_line(driver_argv("run", out or "<out>", root or "<root>", "N", self.cfg.get("flags", {}))) if toolkit_present() else None,
                "config_path": str(self.config_path)}

    def _need(self) -> tuple[Path, str]:
        if not toolkit_present():
            raise PanelError(f"the toolkit is not on disk at {TOOLKIT}")
        out, root = self.cfg.get("out") or "", self.cfg.get("root") or ""
        if not out:
            raise PanelError("set the output root <out> first")
        return Path(out), root

    # ---- the plan and the board
    def plan(self) -> list[dict]:
        out, root = self._need()
        return build_plan(str(out), root or "<root required>", self.cfg.get("flags", {}))

    def board(self) -> dict:
        out, root = self._need()
        self._reap()
        plan = self.plan()
        ext = external_drivers(out, self.proc.pid if self.proc else None)
        running = dict(self.running) if self.running else None
        if not running and ext:
            # a shell run of the driver on the same <out>: shown as running on its first selected move, and the console's launch is refused
            running = {"id": None, "move": _external_move(out, plan, ext[0]["moves"]), "pid": ext[0]["pid"], "started_at": None, "external": True,
                       "moves": ext[0]["moves"], "command": ext[0]["command"]}
        if running and running.get("move") != EUSIPCO_KEY:
            b = running_batch(out, plan)
            if b:
                b["source"] = "shell" if running.get("external") else "console"
                running["batch"] = b
                if b.get("current") is not None:
                    running["move"] = b["current"]
        return {"out": str(out), "root": root, "moves": move_states(out, plan, running), "running": running, "external": ext,
                "ledger": self._ledger_summary(out), "launches": self.launches()[:20]}

    def _ledger_summary(self, out: Path) -> dict:
        led = load_ledger(out)
        if not led:
            return {"exists": False, "path": "driver_state.json"}
        runs = led.get("runs", [])
        return {"exists": True, "path": "driver_state.json", "n_runs": len(runs), "n_commands": len(led.get("commands", [])),
                "package_version": led.get("package_version"), "last_run": runs[-1] if runs else None, "params": led.get("params")}

    # ---- launches
    def _launch_dir(self, out: Path) -> Path:
        d = out / CONSOLE_DIR / "launches"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def launch(self, move, force: bool = False, to_move=None) -> dict:
        """One driver run: move `move`, or with `to_move` the batch `--moves move-to_move` (the driver runs
        them in order and stops at the first failure). The runbook's order applies to the batch's first move."""
        out, root = self._need()
        with self.lock:
            self._reap()
            if self.running:
                raise PanelError(f"move {self.running['move']} is still running (launch {self.running['id']}); one toolkit process at a time")
            ext = external_drivers(out)
            if ext:
                raise PanelError(f"a driver started outside the console is running on this output root (pid {ext[0]['pid']}, moves {ext[0]['moves']}); "
                                 "two writers to driver_state.json are refused")
            plan = self.plan()
            board = move_states(out, plan, None)
            if move == EUSIPCO_KEY:
                row = next(r for r in board if r["move"] == EUSIPCO_KEY)
            else:
                move = int(move)
                if move < 0 or move > MAX_MOVE:
                    raise PanelError(f"moves are 0 to {MAX_MOVE}")
                if to_move is not None and to_move != "":
                    to_move = int(to_move)
                    if to_move < move or to_move > MAX_MOVE:
                        raise PanelError(f"a batch runs moves N to M with {move} <= M <= {MAX_MOVE}")
                    if to_move == move:
                        to_move = None
                else:
                    to_move = None
                row = next(r for r in board if r["move"] == move)
            if not row["runnable"]:
                need = [q for q in row["requires"] if not any(x["move"] == q and x["state"] == "done" for x in board)]
                rule = row.get("waits_rule") or "the runbook's order"
                raise PanelError(f"move {move} waits for move(s) {', '.join(str(q) for q in need)} to be done ({rule})")
            if move == 0 and not root:
                raise PanelError("move 0 needs the retention root <root>")
            if move == EUSIPCO_KEY:
                argvs = eusipco_argv(str(out), self.cfg.get("standalone_tex") or None)
            else:
                argvs = [driver_argv("run", str(out), root or "<root required>", f"{move}-{to_move}" if to_move is not None else move,
                                     self.cfg.get("flags", {}), force)]
            ldir = self._launch_dir(out)
            base = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + (f"_moves{move}-{to_move}" if to_move is not None else f"_move{move}")
            lid, n = base, 1
            while (ldir / f"{lid}.json").exists():          # two launches within one second keep two records
                n += 1
                lid = f"{base}-{n}"
            log = ldir / f"{lid}.log"
            rec = {"id": lid, "move": move, "moves": (f"{move}-{to_move}" if to_move is not None else None), "force": bool(force), "argv": argvs if len(argvs) > 1 else argvs[0], "shell": [shell_line(a) for a in argvs],
                   "cwd": str(QEMU_DIR), "started_at": now_iso(), "finished_at": None, "exit_code": None, "log": str(log),
                   "toolkit": toolkit_identity(), "config": {"preset": self.cfg.get("preset"), "flags": dict(self.cfg.get("flags", {})), "out": str(out), "root": root},
                   "driver_params": None, "driver_run": None, "note": "started from the analysis console; the same command line the runbook gives"}
            (ldir / f"{lid}.json").write_text(json.dumps(rec, indent=1))
            fh = open(log, "ab")
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            if len(argvs) == 1:
                self.proc = subprocess.Popen(argvs[0], cwd=str(QEMU_DIR), stdout=fh, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            else:
                # the EUSIPCO pair: one shell sequence so the second command runs after the first, as the runbook lists them
                script = " && ".join(" ".join(shlex.quote(t) for t in a) for a in argvs)
                self.proc = subprocess.Popen(["/bin/sh", "-c", script], cwd=str(QEMU_DIR), stdout=fh, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            fh.close()
            self.running = {"id": lid, "move": move, "moves": rec["moves"], "pid": self.proc.pid, "started_at": rec["started_at"], "out": str(out), "force": bool(force)}
            return rec

    def _reap(self):
        if self.proc is None or self.running is None:
            return
        rc = self.proc.poll()
        if rc is None:
            return
        out = Path(self.running["out"])
        ldir = self._launch_dir(out)
        p = ldir / f"{self.running['id']}.json"
        try:
            rec = read_json(p)
        except (OSError, json.JSONDecodeError):
            rec = {"id": self.running["id"], "move": self.running["move"]}
        rec["finished_at"] = now_iso()
        rec["exit_code"] = rc
        led = load_ledger(out) or {}
        rec["driver_params"] = led.get("params")
        rec["driver_run"] = (led.get("runs") or [None])[-1]
        rec["package_version"] = led.get("package_version")
        p.write_text(json.dumps(rec, indent=1))
        self.proc = None
        self.running = None

    def stop(self) -> dict:
        with self.lock:
            self._reap()
            if not self.running or self.proc is None:
                raise PanelError("nothing is running")
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                self.proc.terminate()
            t0 = time.time()
            while self.proc.poll() is None and time.time() - t0 < 10:
                time.sleep(0.2)
            if self.proc.poll() is None:
                with contextlib.suppress(Exception):
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            rid = self.running["id"]
            self._reap()
            return {"stopped": rid, "note": "the driver was terminated with its process group; its ledger keeps what finished, the running command left no record"}

    def launches(self) -> list[dict]:
        out = Path(self.cfg.get("out") or "")
        d = out / CONSOLE_DIR / "launches"
        if not d.is_dir():
            return []
        recs = []
        for p in sorted(d.glob("*.json"), reverse=True):
            try:
                r = read_json(p)
            except (OSError, json.JSONDecodeError):
                continue
            recs.append({k: r.get(k) for k in ("id", "move", "force", "shell", "started_at", "finished_at", "exit_code", "log")}
                        | {"toolkit": {k: (r.get("toolkit") or {}).get(k) for k in ("repo_head_short", "committed", "content_fingerprint", "package_version")},
                           "preset": (r.get("config") or {}).get("preset")})
        return recs

    def launch_record(self, lid: str) -> dict:
        out = Path(self.cfg.get("out") or "")
        if not re.match(r"^[0-9TZ]+_move[0-9a-z]+(-[0-9]+)?$", lid or ""):
            raise PanelError("not a launch id")
        p = out / CONSOLE_DIR / "launches" / f"{lid}.json"
        if not p.exists():
            raise PanelError(f"no launch {lid}")
        return read_json(p)

    def log_tail(self, lid: str | None, n: int = 200) -> dict:
        self._reap()
        out = Path(self.cfg.get("out") or "")
        if not lid:
            lid = self.running["id"] if self.running else ((self.launches() or [{}])[0].get("id"))
        if not lid:
            return {"id": None, "lines": [], "running": False}
        if not re.match(r"^[0-9TZ]+_move[0-9a-z]+(-[0-9]+)?$", lid):
            raise PanelError("not a launch id")
        p = out / CONSOLE_DIR / "launches" / f"{lid}.log"
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()[-n:] if p.exists() else []
        return {"id": lid, "lines": lines, "running": bool(self.running and self.running["id"] == lid), "log": str(p)}
