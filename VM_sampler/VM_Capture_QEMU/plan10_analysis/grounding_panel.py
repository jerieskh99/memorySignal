#!/usr/bin/env python3
"""grounding_panel.py -- the console's window onto plan12_grounding, the grounding paper's engine.

The Encoding paper panel's pattern (encoding_panel.py), applied to plan12: the engine is the source
of truth for the paper's numbers and must stay runnable from the shell exactly as its RUNBOOK.md
says. So this module

  - launches the engine's own driver, one move at a time, in the runbook's order
    (`python3 -m plan12_grounding.run_moves run --out O [--root R] --moves N <flags>` from
    VM_Capture_QEMU/, the driver's own cwd), with the flags the author set and nothing else;
    the commands a move will run are the driver's own plan (`run_moves plan`), never a copy;
  - reads the engine's files as they are: the record book (`driver_state.json`), `cells.csv`,
    `params.json`, the move records under `moves/`, the figures and tables, the report;
    nothing here computes a number, and a refusal string is passed through as written;
  - records every launch (the engine's commit state, the command line, the params block the
    driver wrote) under `<out>/.console/launches/`, a directory the engine does not read;
  - writes exactly one file outside `<out>/.console`: its own configuration,
    `~/.cache/plan10/grounding_config.json` (where `<out>` is, the root, the preset and the flags),
    which cannot live under `<out>` because it is what names `<out>`; this is the one exception to
    SPEC 8.2, named here;
  - never writes under plan12_grounding/ and never touches a server (a move the author starts
    may fetch from it, read only, as the engine's own runbook says).

Recordings outside the corpus are never named: `cells()` counts them, and every CSV under
`<out>/inputs/` is served with its kernel and idle rows only (the engine already copies only those;
this is defence in depth), the rows left out counted.

Move 9 (the noise-floor removal, decision D1) is a switch, off by default: the driver plans it only
with `--room-removal`; the board shows it as off until the author switches it on. Next to move 6
the panel offers, read only, the named encoding run's matching numbers (its Table 2 rows for the
combined rung), read from the encoding run's own files.

Two moves live beside the run, not in its record book (2026-10-06): move 11, the idle class check
(`plan12_grounding/idle_class.py`, steps 1 and 2) and move 12, the new-block test
(`plan12_grounding/new_blocks.py`). Each is its own command with its own record (`record.json`) in
its own sibling folder, `<out>_idle13` and `<out>_newblocks`, so that nothing in the run's record
book turns stale; the panel launches them with their own command lines, shows their state from their
records, and reads their files through the same path guard, extended to exactly those two siblings.
The cut folders of every view come from the run's params.json (`cut16`, `cut192`, ...), never from a
fixed list.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
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
TOOLKIT_NAME = "plan12_grounding"
TOOLKIT = QEMU_DIR / TOOLKIT_NAME
RUNBOOK = TOOLKIT / "RUNBOOK.md"
DEFAULT_CONFIG = Path(os.path.expanduser("~/.cache/plan10/grounding_config.json"))
CONSOLE_DIR = ".console"                 # under <out>; the engine neither reads nor hashes it
MAX_MOVE = 10                            # the driver's moves (run_moves.MAX_MOVE); 11 and 12 live beside the run (EXTRA_MOVES)
ROOM_MOVE = 9                            # decision D1: planned only with --room-removal
FALLBACK_CUTS = (16, 112)                # the cut folders when params.json is absent (the SPEC's two values)
# The moves beside the run (2026-10-06): their own module, command, record and sibling folder; never in run_moves.py.
EXTRA_MOVES = {
    11: {"title": "idle as a 13th class: the lexer check (steps 1 and 2)", "module": "idle_class", "suffix": "_idle13", "record_keys": ("step1_loro", "step2_within_trace"),
         "steps": ({"name": "idle class, step 1 (LORO, 13 classes)", "key": "step1_loro", "args": ["--step", "1"], "dir": "step1_loro"},
                   {"name": "idle class, step 2 (within-trace, 13 classes)", "key": "step2_within_trace", "args": ["--step", "2"], "dir": "step2_within_trace"}),
         "what": "idle_class.py: LORO (step 1) and within-trace (step 2) with the 12 kernels plus idle, the four feature sets as move 6 builds them, both cuts, no null; "
                 "writes <out>_idle13/step1_loro/cut<C>/ and step2_within_trace/cut<C>/ (scores.csv, recall_per_class.csv, confusion_<E>.csv and .svg, margins.csv, "
                 "predictions_<E>.csv, summary.json), <step>/step.json, e0_idle_check.json and record.json",
         "note": "beside the run, not in its record book: its own record.json in the sibling folder <out>_idle13; reads moves 0, 1 and 6; refuses to start (exit 3) while a move runs",
         "requires": [6], "reads_from": [0, 1, 6]},
    12: {"title": "the new-block test: seen 80%, a gap, then new blocks named at three levels", "module": "new_blocks", "suffix": "_newblocks", "record_keys": ("new_blocks",),
         "steps": ({"name": "new blocks (archetype, kernel, run; both cuts)", "key": "new_blocks", "args": [], "dir": ""},),
         "what": "new_blocks.py: every recording's first 80% of windows seen, a one-window gap, the rest new blocks; one forest per level (archetype, kernel, run), feature set and "
                 "cut on the seen windows; single blocks and pools of 2 and 3; chance and majority baselines, no null; writes <out>_newblocks/cut<C>/ (split.csv, scores.csv, "
                 "recall_per_kernel.csv, run_level.csv, by_position.csv, margins.csv, predictions_<level>_<E>.csv, pools_<level>_<E>.csv, confusion_kernel_<E>.csv/.svg, "
                 "accuracy_by_level.svg, accuracy_by_position.svg, recall_per_kernel.svg, summary.json), new_blocks.json, e0_idle_check.json and record.json",
         "note": "beside the run, not in its record book: its own record.json in the sibling folder <out>_newblocks; reads moves 0, 1 and 6; refuses to start (exit 3) while a move runs",
         "requires": [6], "reads_from": [0, 1, 6]},
    13: {"title": "the per-page Fourier test: every page's complex row, phi = 2 theta (step 1) and phi = theta (step 2)", "module": "page_fourier", "suffix": "_pagefourier",
         "record_keys": ("step1_doubled", "step2_single"), "lock": ".page_fourier.lock", "flags": (),
         "steps": ({"name": "page Fourier, step 1 (phi = 2 theta, the headline)", "key": "step1_doubled", "args": ["--step", "1"], "dir": "step1_doubled"},
                   {"name": "page Fourier, step 2 (phi = theta, the control)", "key": "step2_single", "args": ["--step", "2"], "dir": "step2_single"}),
         "what": "page_fourier.py: the letter's complex matrix read row by row: per page that changes after the cut, its complex row over the pairs (h e^{j phi}), Welch two-sided "
                 "(128-pair Hann segments), the spectrum normalised to sum 1; per recording the page-averaged spectrum (all pages, busy, rare, rest) and the asymmetry; between "
                 "recordings move 5's spectral test (within against between, the label null) with move 5's angle rows beside it; the identity check (the page arrows reproduce "
                 "N, H, C, S) before anything; writes <out>_pagefourier/<step>/cut<C>/ (recordings.csv, spectra/, pages/, spectra_by_group.csv, between.csv, per_group.csv, "
                 "asymmetry.csv, identity_check.csv, spectrum_by_group.svg, within_between.svg, asymmetry_by_kernel.svg, pages_image_<cell>.svg/.png, summary.json), "
                 "<step>/step.json and record.json",
         "note": "beside the run, not in its record book: its own record.json in the sibling folder <out>_pagefourier; reads moves 0, 1 and 5 and the L1 stores; refuses to "
                 "start (exit 3) while a move runs; the author's phase is the doubled angle (step 1), step 2 is the control",
         "requires": [1], "reads_from": [0, 1, 5]},
}
# the grid (2026-10-10): moves 14 to 16 are move 13 with its two switches turned (page_fourier.py --move N), in the same sibling folder and record
_GRID = {14: ("raw spectra, level kept", "the plain mean of the pages' power spectra (pages with more bits weigh more); no mean removed, the f = 0 bin kept"),
         15: ("each page counts once, level kept", "each page's spectrum normalised to sum 1 before the average; no mean removed, the f = 0 bin kept"),
         16: ("raw spectra, level removed", "the plain mean of the pages' power spectra; the mean removed per segment, the f = 0 bin dropped in the between-recordings test")}
for _mv, (_short, _long) in _GRID.items():
    EXTRA_MOVES[_mv] = {
        "title": f"the per-page Fourier test, {_short} (move 13's grid)", "module": "page_fourier", "suffix": "_pagefourier",
        "record_keys": (f"move{_mv}_step1_doubled", f"move{_mv}_step2_single"), "lock": ".page_fourier.lock", "flags": (),
        "steps": ({"name": f"page Fourier move {_mv}, step 1 (phi = 2 theta)", "key": f"move{_mv}_step1_doubled", "args": ["--move", str(_mv), "--step", "1"], "dir": f"move{_mv}/step1_doubled"},
                  {"name": f"page Fourier move {_mv}, step 2 (phi = theta)", "key": f"move{_mv}_step2_single", "args": ["--move", str(_mv), "--step", "2"], "dir": f"move{_mv}/step2_single"}),
        "what": f"page_fourier.py --move {_mv}: {_long}; everything else as move 13 (the data, the cuts, the identity check, the activity groups, the asymmetry, move 5's test "
                f"and its null, the figures); writes <out>_pagefourier/move{_mv}/<step>/cut<C>/ with move 13's files",
        "note": f"beside the run: the same sibling folder and record.json as move 13 (keys move{_mv}_<step>); reads moves 0, 1 and 5 and the L1 stores; refuses to start (exit 3) while a move runs",
        "requires": [1], "reads_from": [0, 1, 5]}
EXTRA_MOVES[16]["steps"] = EXTRA_MOVES[16]["steps"] + ({"name": "the grid's summary: moves 13 to 16, two steps, both cuts, side by side", "key": "grid_summary", "sub": "summary", "args": [],
                                                       "dir": "", "done_file": "grid_summary.json"},)
EXTRA_MOVES[16]["what"] += "; its third step, `page_fourier summary`, writes <out>_pagefourier/grid_summary.csv, .svg and .json (a variant not yet run reads missing)"
EXTRA_FLAGS = ("n_jobs", "n_estimators")  # the tab's flags passed to the moves beside the run unless the move names its own list (the paper preset's values are move 6's own)
CONFIG_FIELDS = ("out", "root")         # the two paths the config bar holds; every other driver flag is in the flag form

# RUNBOOK.md section 2 (the real run) and section 1 (the smoke run): the flag values those command
# lines carry. Presets set exactly these; the author can change any of them.
PRESETS = {
    "paper": {"label": "the real run (RUNBOOK section 2): the server in fetch mode, the encoding run named",
              "flags": {"ssh": "jeries@cybersecurity.ac.upc.edu", "remote_root": "/mnt/nfs/jeries/memory_traces/zstd_local",
                        "store": "~/.cache/plan10/l1", "encoding_out": "", "n_jobs": 4, "null_perm": 500, "null_splits": "loko,within_trace",
                        "n_estimators": 300, "n_shuffles": 1000, "room_removal": False}},
    "smoke": {"label": "the smoke run on the synthetic corpus (RUNBOOK section 1)",
              "flags": {"store": "~/.cache/plan12/smoke/l1", "encoding_out": "~/.cache/plan12/smoke/encoding_out", "allow_unmatched_declared": True,
                        "n_shuffles": 200, "null_perm": 5, "n_estimators": 30, "n_jobs": 4, "room_removal": False}},
}

# The views, each a list of the engine's own files under <out>. `move` is the move that writes the
# first file; a missing file is reported by path, never drawn. `{cut}` stands for each cut folder of the
# run (params.json: cut16, cut192, ...; `views()` expands it); `{idle13}` and `{newblocks}` for the two
# sibling folders of the moves beside the run.
VIEWS = [
    {"id": "index", "title": "the recordings indexed, the decisions", "move": 0,
     "files": ["cells.csv", "params.json", "inputs/sha256.json", "moves/00_index/index.json"]},
    {"id": "extract", "title": "the series written, one per recording", "move": 1,
     "files": ["moves/01_extract/recordings.csv", "moves/01_extract/extract.json"]},
    {"id": "sanity", "title": "the sanity counts and violations", "move": 2,
     "files": ["moves/02_sanity/counts.csv", "moves/02_sanity/violations.csv", "moves/02_sanity/sanity.json"]},
    {"id": "every_run", "title": "every run, per kernel and idle (the gallery)", "move": 3,
     "files": ["moves/03_every_run/index.html", "moves/03_every_run/figures.json"], "gallery": "moves/03_every_run"},
    {"id": "portraits", "title": "the kernel portraits", "move": 4,
     "files": ["moves/04_portraits/index.html", "moves/04_portraits/figures.json"], "gallery": "moves/04_portraits"},
    {"id": "similarity", "title": "how similar the runs are (ICC, the map, the spectra, leave-one-seed-out)", "move": 5,
     "files": ["moves/05_similarity/{cut}/icc_bars.svg", "moves/05_similarity/{cut}/pca_map.svg", "moves/05_similarity/{cut}/spectral_similarity.svg",
               "moves/05_similarity/{cut}/loso.svg", "moves/05_similarity/{cut}/icc.csv", "moves/05_similarity/{cut}/loso_summary.csv",
               "moves/05_similarity/hand_check.txt", "moves/05_similarity/similarity.json"]},
    {"id": "classify", "title": "classification: the four encodings under the three splits", "move": 6,
     "files": ["moves/06_classify/{cut}/bars.svg", "moves/06_classify/{cut}/scores.csv", "moves/06_classify/{cut}/margins.csv", "moves/06_classify/{cut}/gap.csv",
               "moves/06_classify/{cut}/confusion_loko.svg", "moves/06_classify/{cut}/confusion_loro.svg", "moves/06_classify/{cut}/excluded_cells.csv",
               "moves/06_classify/e0_identity.json", "moves/06_classify/e0_declared.json", "moves/06_classify/classify.json"]},
    {"id": "floor", "title": "the noise floor, each kernel against the idle runs", "move": 7,
     "files": ["moves/07_floor/{cut}/floor_N_despiked.svg", "moves/07_floor/{cut}/floor_H_despiked.svg", "moves/07_floor/{cut}/floor_A_despiked.svg",
               "moves/07_floor/{cut}/floor_bins.csv", "moves/07_floor/{cut}/floor_means.csv", "moves/07_floor/floor.json"]},
    {"id": "startup", "title": "the start-up: the spike rate against the pair index", "move": 8,
     "files": ["moves/08_startup/startup.svg", "moves/08_startup/bins.csv", "moves/08_startup/spikes_per_run.csv", "moves/08_startup/startup.json"]},
    {"id": "removed", "title": "the noise-floor removal: before and after (move 9, when switched on)", "move": 9,
     "files": ["moves/09_removed/before_after.svg", "moves/09_removed/before_after.csv", "moves/09_removed/removal_set.csv",
               "moves/09_removed/pages_per_recording.csv", "moves/09_removed/removal.json"]},
    {"id": "summary", "title": "the summary: the per-kernel table, the figure set, the manifest", "move": 10,
     "files": ["report/table_per_kernel.csv", "report/table_overall.csv", "report/figures/index.csv", "report/manifest.json", "moves/10_summary/summary.json"]},
    {"id": "idle13", "title": "idle as a 13th class: LORO and within-trace with the lexer and idle rows highlighted (move 11, beside the run)", "move": 11,
     "note": "written by idle_class.py into the sibling folder <out>_idle13; the rows and columns of each confusion follow cells.csv's class order, idle last",
     "files": ["{idle13}/step1_loro/{cut}/confusion_E2.svg", "{idle13}/step1_loro/{cut}/confusion_E_new.svg", "{idle13}/step1_loro/{cut}/scores.csv",
               "{idle13}/step1_loro/{cut}/recall_per_class.csv", "{idle13}/step1_loro/{cut}/margins.csv",
               "{idle13}/step2_within_trace/{cut}/confusion_E2.svg", "{idle13}/step2_within_trace/{cut}/scores.csv", "{idle13}/step2_within_trace/{cut}/recall_per_class.csv",
               "{idle13}/step1_loro/step.json", "{idle13}/step2_within_trace/step.json", "{idle13}/e0_idle_check.json", "{idle13}/record.json"]},
    {"id": "pagefourier", "title": "the per-page Fourier test: the spectrum per kernel and idle, within against between, the asymmetry, one recording's pages (move 13, beside the run)", "move": 13,
     "note": "written by page_fourier.py into the sibling folder <out>_pagefourier; step 1 is the author's phase (phi = 2 theta), step 2 the control (phi = theta)",
     "files": ["{pagefourier}/step1_doubled/{cut}/spectrum_by_group.svg", "{pagefourier}/step1_doubled/{cut}/within_between.svg", "{pagefourier}/step1_doubled/{cut}/asymmetry_by_kernel.svg",
               "{pagefourier}/step1_doubled/{cut}/between.csv", "{pagefourier}/step1_doubled/{cut}/recordings.csv", "{pagefourier}/step1_doubled/{cut}/identity_check.csv",
               "{pagefourier}/step2_single/{cut}/spectrum_by_group.svg", "{pagefourier}/step2_single/{cut}/within_between.svg", "{pagefourier}/step2_single/{cut}/asymmetry_by_kernel.svg",
               "{pagefourier}/step2_single/{cut}/between.csv", "{pagefourier}/step1_doubled/step.json", "{pagefourier}/step2_single/step.json", "{pagefourier}/record.json"]},
    {"id": "pagefourier_grid", "title": "the grid of the per-page Fourier test: moves 13 to 16 side by side, and each variant's within-against-between and asymmetry (moves 14 to 16, beside the run)", "move": 16,
     "note": "written by page_fourier.py into the sibling folder <out>_pagefourier: the summary after move 16's third step, then step 1 of each variant per cut",
     "files": ["{pagefourier}/grid_summary.svg", "{pagefourier}/grid_summary.csv",
               "{pagefourier}/move14/step1_doubled/{cut}/within_between.svg", "{pagefourier}/move14/step1_doubled/{cut}/asymmetry_by_kernel.svg", "{pagefourier}/move14/step1_doubled/{cut}/spectrum_by_group.svg",
               "{pagefourier}/move15/step1_doubled/{cut}/within_between.svg", "{pagefourier}/move15/step1_doubled/{cut}/asymmetry_by_kernel.svg", "{pagefourier}/move15/step1_doubled/{cut}/spectrum_by_group.svg",
               "{pagefourier}/move16/step1_doubled/{cut}/within_between.svg", "{pagefourier}/move16/step1_doubled/{cut}/asymmetry_by_kernel.svg", "{pagefourier}/move16/step1_doubled/{cut}/spectrum_by_group.svg",
               "{pagefourier}/grid_summary.json"]},
    {"id": "newblocks", "title": "the new-block test: accuracy by level, by position, per kernel; the kernel confusions (move 12, beside the run)", "move": 12,
     "note": "written by new_blocks.py into the sibling folder <out>_newblocks; no label-shuffle null (the scores.csv null column says why); chance and majority are the baselines",
     "files": ["{newblocks}/{cut}/accuracy_by_level.svg", "{newblocks}/{cut}/accuracy_by_position.svg", "{newblocks}/{cut}/recall_per_kernel.svg",
               "{newblocks}/{cut}/confusion_kernel_E2.svg", "{newblocks}/{cut}/confusion_kernel_E_new.svg", "{newblocks}/{cut}/scores.csv",
               "{newblocks}/{cut}/margins.csv", "{newblocks}/{cut}/run_level.csv", "{newblocks}/{cut}/by_position.csv", "{newblocks}/{cut}/split.csv",
               "{newblocks}/new_blocks.json", "{newblocks}/e0_idle_check.json", "{newblocks}/record.json"]},
]

TEXT_KINDS = {".csv": "csv", ".json": "json", ".md": "md", ".txt": "text", ".log": "text", ".py": "text", ".html": "html", ".svg": "svg"}
BINARY_TYPES = {".png": "image/png", ".pdf": "application/pdf", ".npz": "application/octet-stream", ".npy": "application/octet-stream",
                ".zst": "application/zstd", ".svg": "image/svg+xml", ".html": "text/html"}
MAX_TEXT_BYTES = 8 * 1024 * 1024
MAX_CSV_ROWS = 20000


class PanelError(ValueError):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# the engine on disk: its driver's flags, its plan, its identity, its runbook
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
    """Every flag of the driver's `run` subcommand, from the engine's own argparse definition:
    name, destination, default, type, choices, help. The page builds its flag form from this."""
    ap = _parser()
    out = []
    for a in ap._actions:
        if not a.option_strings or a.dest in (*CONFIG_FIELDS, "moves", "force", "dry_run", "help"):
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
        if a is None or _unset(val) or dest in CONFIG_FIELDS:
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
    """`python3 -m plan12_grounding.run_moves <sub> --out O [--root R] --moves N <flags> [--force]`; the
    root is passed only when set (the ssh source is a flag pair, --ssh and --remote-root)."""
    argv = [sys.executable, "-m", f"{TOOLKIT_NAME}.run_moves", sub, "--out", str(out)]
    if root:
        argv += ["--root", str(root)]
    argv += ["--moves", str(move)]
    argv += flag_tokens(flags)
    if force:
        argv.append("--force")
    return argv


def shell_line(argv: list[str]) -> str:
    toks = ["python3" if i == 0 else t for i, t in enumerate(argv)]
    return " ".join(shlex.quote(t) for t in toks)


def build_plan(out: str, root: str, flags: dict) -> list[dict]:
    """The driver's own move table for these flags (`run_moves.build_plan`), each command with its
    module, subcommand, args, declared outputs and inputs; the same function the driver runs."""
    RM = _import_driver()
    ap = _parser()
    args = ["--out", str(out)] + (["--root", str(root)] if root else []) + flag_tokens(flags)
    o = ap.parse_args(args)
    plan = RM.build_plan(o)
    for c in plan:
        c.setdefault("internal", False)
        c.setdefault("template", False)
        c["key"] = f"{c['move']}:{c['name']}"
        c["argv"] = [sys.executable, "-m", f"{TOOLKIT_NAME}.{c['module']}"] + ([c["sub"]] if c["sub"] else []) + list(c["args"])
        c["line"] = shell_line(c["argv"])
    return plan


def plan_text(out: str, root: str, flags: dict, move: int) -> str:
    """What `run_moves plan --moves N` prints: the exact argument lists, in the driver's words. For a
    move beside the run (11, 12), what its own `--dry-run` prints: the recordings, the cuts, the window,
    the E0 check and, for the new-block test, the windows, the seen windows and the new blocks per cut;
    a dry run writes nothing and skips the live-lock refusal."""
    mv = int(move) if str(move).lstrip("-").isdigit() else None
    if mv in EXTRA_MOVES:
        text = ""
        for st in EXTRA_MOVES[mv]["steps"]:
            argv = extra_argv(Path(os.path.expanduser(out)), mv, st, flags, dry_run=True)
            text += f"$ {shell_line(argv)}\n"
            try:
                p = subprocess.run(argv, cwd=str(QEMU_DIR), capture_output=True, text=True, timeout=300)
                text += (p.stdout or "") + (("\n" + p.stderr) if p.returncode else "") + "\n"
            except subprocess.TimeoutExpired:
                text += "(the dry run did not finish within 300 s)\n"
        return text
    argv = driver_argv("plan", out, root, move, flags)
    p = subprocess.run(argv, cwd=str(QEMU_DIR), capture_output=True, text=True, timeout=120)
    return (p.stdout or "") + (("\n" + p.stderr) if p.returncode else "")


def move_names() -> dict:
    RM = _import_driver()
    return {int(k): v for k, v in RM.MOVE_NAMES.items()}


def runbook_sections() -> dict:
    """The runbook's own words per move: its name from the driver's table and the row of the
    runbook's section 4 table (`| N | module | writes |`). Read, never paraphrased."""
    names = move_names()
    out = {m: {"title": t, "body": "", "look_at": ""} for m, t in names.items()}
    if not RUNBOOK.exists():
        return out
    text = RUNBOOK.read_text(encoding="utf-8", errors="replace")
    for m in re.finditer(r"^\|\s*(\d+)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*$", text, flags=re.M):
        mv = int(m.group(1))
        if mv in out:
            out[mv]["body"] = f"{m.group(2)}: {m.group(3)}"
    for mv, x in EXTRA_MOVES.items():                    # beside the run: not in RUNBOOK.md's table; their modules' own words
        out[mv] = {"title": x["title"], "body": x["what"], "look_at": f"the sibling folder <out>{x['suffix']} (the Look tab's view for move {mv})"}
    return out


def _git(args: list[str]) -> str:
    try:
        p = subprocess.run(["git", *args], cwd=str(QEMU_DIR), capture_output=True, text=True, timeout=20)
        return p.stdout.strip() if p.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def toolkit_identity() -> dict:
    """What the launch record names: the repository's HEAD, whether the engine's files are tracked
    and clean at that commit, the package version, and the engine's own fingerprint (sha256 over
    every plan12_grounding/*.py, the same value its record book carries)."""
    head = _git(["rev-parse", "HEAD"])
    tracked = bool(_git(["ls-files", "--error-unmatch", f"{TOOLKIT_NAME}/run_moves.py"]))
    porcelain = _git(["status", "--porcelain", "--", TOOLKIT_NAME])
    dirty = [ln for ln in porcelain.splitlines() if ln.strip()]
    fp, n_files, version = None, 0, None
    if toolkit_present():
        try:
            if str(QEMU_DIR) not in sys.path:
                sys.path.insert(0, str(QEMU_DIR))
            import importlib
            pkg = importlib.import_module(TOOLKIT_NAME)
            f = pkg.toolkit_fingerprint()
            fp, n_files, version = f["sha256"][:16], f.get("n_py_files"), getattr(pkg, "__version__", None)
        except Exception:                                   # noqa: BLE001  the identity is informative; a broken import is reported below
            pass
    return {"path": str(TOOLKIT), "present": toolkit_present(), "repo_head": head or None, "repo_head_short": head[:12] if head else None,
            "tracked": tracked, "dirty_entries": len(dirty), "dirty_sample": dirty[:5],
            "committed": bool(head) and tracked and not dirty,
            "content_fingerprint": fp, "n_py_files": n_files, "package_version": version,
            "python": sys.executable, "cwd": str(QEMU_DIR)}


# ---------------------------------------------------------------------------
# the engine's files, read as they are
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


def cut_dirs(out: Path) -> list[str]:
    """The run's cut folders, from its params.json (`cuts.declared_pairs`, `cuts.measured_pairs`:
    cut16 and cut192 in the real run), else the SPEC's two values; never a fixed list."""
    cuts = list(FALLBACK_CUTS)
    pj = Path(out) / "params.json"
    if pj.is_file():
        with contextlib.suppress(OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            c = read_json(pj).get("cuts") or {}
            cuts = [int(c["declared_pairs"]), int(c["measured_pairs"])]
    seen, res = set(), []
    for c in cuts:
        if c not in seen:
            seen.add(c); res.append(f"cut{c}")
    return res


def sibling_dirs(out: Path) -> dict:
    """{sibling folder name: its path} for the moves beside the run (EXTRA_MOVES): exactly `<out>_idle13`
    and `<out>_newblocks`, nothing else next to the run."""
    o = Path(out)
    return {o.name + m["suffix"]: o.parent / (o.name + m["suffix"]) for m in EXTRA_MOVES.values()}


def safe_path(out: Path, rel: str) -> Path:
    """A path under <out>, or under one of the two sibling folders of the moves beside the run
    (`<out>_idle13`, `<out>_newblocks`, named by their folder name as the first part of `rel`), or a
    PanelError; `..`, absolute paths, and any top-level dot entry other than `.console` are refused
    (a dot entry is scratch, never one of the engine's outputs; the siblings' lock files too)."""
    rel = str(rel or "")
    if not rel or rel.startswith(("/", "\\")) or ".." in Path(rel).parts:
        raise PanelError(f"not a path under the output root: {rel!r}")
    parts = Path(rel).parts
    first = parts[0] if parts else ""
    sibs = sibling_dirs(out)
    if first in sibs:
        base, sub = sibs[first], Path(*parts[1:]) if len(parts) > 1 else Path("")
        if sub.parts and sub.parts[0].startswith("."):
            raise PanelError(f"not one of the engine's outputs: {rel!r}")
        p = (base / sub).resolve()
        b = base.resolve()
        if p != b and b not in p.parents:
            raise PanelError(f"not a path under the sibling folder {first}: {rel!r}")
        return p
    if first.startswith(".") and first != CONSOLE_DIR:
        raise PanelError(f"not one of the engine's outputs: {rel!r}")
    p = (Path(out) / rel).resolve()
    o = Path(out).resolve()
    if p != o and o not in p.parents:
        raise PanelError(f"not a path under the output root: {rel!r}")
    return p


def _unset(v) -> bool:
    """A flag value that means "unset": None, the empty string, or false itself; 0 is a value."""
    return v is None or (isinstance(v, str) and v == "") or v is False


_IDLE_PATH_RE = re.compile(r"(^|/)sleep/sleep/sleep_600/rep00[0-9]__idle_01c/?$")
_IDLE_CELL_RE = re.compile(r"^idle__rep0[0-9]__idle_01c$")
_KERNELS = ("gemm", "floyd", "gibbs", "nbody", "spmm", "stencil_jacobi", "fft", "histogram", "fem_assembly", "lexer", "rmat_gen", "bnb_tsp")


def _corpus_row_local(row: dict, corpus_ids=None) -> bool:
    """The engine's exact corpus rule (plan12_grounding.inputs.corpus_row), the same patterns, used
    only when the engine cannot be imported: idle by the path's tail, a kernel by its name and path."""
    role = row.get("role")
    path = str(row.get("path") or row.get("rec_rel") or "")
    if path:
        if role == "idle":
            return bool(_IDLE_PATH_RE.search(path))
        if role == "kernel":
            k = str(row.get("kernel") or "")
            return k in _KERNELS and bool(re.search(r"(^|/)kernel/kernel_" + re.escape(k) + r"_v2/[^/]+/[^/]+/?$", path))
        return False
    cid = str(row.get("cell_id") or "")
    if corpus_ids is not None:
        return cid in corpus_ids
    if role == "idle":
        return bool(_IDLE_CELL_RE.match(cid))
    if role == "kernel":
        return "__rep" in cid and cid.split("__", 1)[0] in _KERNELS
    return False


def corpus_rule():
    """The engine's own rule when it is on disk (one source of truth), else the same patterns here."""
    try:
        if str(QEMU_DIR) not in sys.path:
            sys.path.insert(0, str(QEMU_DIR))
        import importlib
        return importlib.import_module(f"{TOOLKIT_NAME}.inputs").corpus_row
    except Exception:                                   # noqa: BLE001
        return _corpus_row_local


def corpus_csv(path: Path) -> tuple[list[str], list[list[str]], int, int]:
    """A CSV under inputs/ with only the rows that pass the engine's exact corpus rule (the layout of
    the path, never the role alone, since the encoding toolkit calls any "sleep" or "idle" label
    idle): (header, rows, n_total, n_left_out). A file without a path column (preconditions.csv) is
    judged by its cell_ids against the kept rows of the cells.csv beside it. A CSV without a `role`
    column is returned whole."""
    header, rows, total = read_csv_rows(path)
    if "role" not in header:
        return header, rows, total, 0
    rule = corpus_rule()
    ids = None
    sib = path.parent / "cells.csv"
    if "path" not in header and sib.is_file() and sib.resolve() != path.resolve():
        h2, r2, _ = read_csv_rows(sib)
        ids = {dict(zip(h2, r)).get("cell_id", "") for r in r2 if rule(dict(zip(h2, r)))}
    kept = [r for r in rows if rule(dict(zip(header, r)), ids)]
    return header, kept, total, total - len(kept)


def file_bytes(out: Path, rel: str) -> tuple[bytes, str]:
    """The engine's file bytes from under <out>, nothing outside it; a CSV under inputs/ is served with
    its kernel and idle rows only."""
    p = safe_path(out, rel)
    if not p.is_file():
        raise FileNotFoundError(rel)
    ctype = BINARY_TYPES.get(p.suffix.lower()) or {"csv": "text/csv", "json": "application/json", "md": "text/markdown", "html": "text/html",
                                                   "svg": "image/svg+xml", "text": "text/plain"}.get(TEXT_KINDS.get(p.suffix.lower(), "text"), "application/octet-stream")
    if p.suffix.lower() == ".csv" and Path(rel).parts and Path(rel).parts[0] == "inputs":
        header, rows, total, left = corpus_csv(p)
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
        return buf.getvalue().encode("utf-8"), "text/csv"
    return p.read_bytes(), ctype


def text_file(out: Path, rel: str) -> dict:
    """A CSV as header and rows, a JSON as its object, an HTML or SVG as its text, anything else as
    text; the path and size always; cells untouched (a refusal string stays a string)."""
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
        if Path(rel).parts and Path(rel).parts[0] == "inputs":
            header, rows, total, left = corpus_csv(p)      # never a recording outside the corpus, counted instead
            d.update(header=header, rows=rows, n_rows=len(rows), truncated=False, rows_left_out_counted_not_named=left)
        else:
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


def listing(out: Path, rel: str = "") -> dict:
    p = safe_path(out, rel) if rel else Path(out).resolve()
    if not p.exists():
        return {"path": rel, "exists": False, "entries": []}
    ents = []
    in_sibling = bool(rel) and Path(rel).parts[0] in sibling_dirs(out)
    for c in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name)):
        if (not rel or (in_sibling and len(Path(rel).parts) == 1)) and c.name.startswith(".") and c.name != CONSOLE_DIR:
            continue                                     # a top-level dot entry is scratch (a lock, a tmp), never one of the engine's outputs
        st = c.stat()
        ents.append({"name": c.name, "dir": c.is_dir(), "bytes": None if c.is_dir() else st.st_size,
                     "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds"),
                     "path": f"{rel}/{c.name}" if rel else c.name})
    if not rel:
        for name, sp in sibling_dirs(out).items():        # the moves beside the run: their folders, browsable by name
            if sp.is_dir():
                st = sp.stat()
                ents.append({"name": name, "dir": True, "bytes": None, "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds"),
                             "path": name, "beside_the_run": True})
    return {"path": rel, "exists": True, "entries": ents}


def params_blocks(out: Path) -> dict:
    """Every params block the engine wrote under <out>: `params.json` (the decisions D1 to D4 and
    the cuts), each move record with a `params` key, the driver's own params from the record
    book, and the author's declared inputs as copied. Read-only."""
    o = Path(out)
    blocks = []
    if o.is_dir():
        for p in sorted(o.rglob("*.json")):
            rel = p.relative_to(o)
            if rel.parts and rel.parts[0] in (CONSOLE_DIR, "series"):
                continue
            if rel.parts and rel.parts[0] == "report" and p.name == "manifest.json":
                continue
            try:
                p = safe_path(o, str(rel))                   # a link that leaves <out> is refused, a dot path too
            except PanelError:
                continue
            try:
                j = read_json(p)
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(j, dict) and isinstance(j.get("params"), dict):
                blocks.append({"path": str(rel), "schema": j.get("schema"), "citation": j.get("citation"), "params": j["params"],
                               "written_at": j.get("written_at")})
    ledger = load_ledger(o)
    decisions = None
    if (o / "params.json").exists():
        with contextlib.suppress(OSError, json.JSONDecodeError):
            decisions = read_json(o / "params.json")
    inputs = {}
    if (o / "inputs").is_dir():
        for p in sorted((o / "inputs").rglob("*")):
            if p.is_file():
                rel = str(p.relative_to(o))
                try:
                    inputs[rel] = text_file(o, rel)          # a CSV under inputs/ comes back with its kernel and idle rows only
                except PanelError:
                    continue
    return {"driver_params": (ledger or {}).get("params"), "package_version": (ledger or {}).get("package_version"),
            "toolkit_fingerprint": ((ledger or {}).get("toolkit_fingerprint") or {}).get("sha256"),
            "decisions": decisions, "inputs": inputs, "blocks": blocks, "n_blocks": len(blocks),
            "manifest": "report/manifest.json" if (o / "report" / "manifest.json").exists() else None}


# ---------------------------------------------------------------------------
# the record book, the moves, their states
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
    if s.startswith(("dry-run", "not run")):
        return "not run"
    return "other"


def _moves_of(spec: str | None) -> list[int]:
    """`6-10`, `2,4` -> the move numbers, in order (the driver's own syntax)."""
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


def interrupted_moves(ledger: dict, plan: list[dict], running: dict | None) -> dict:
    """Moves whose latest attempt was cut short: a record-book `runs` entry with no `finished_at`
    that belongs to no live process. The interrupted move of such a run is the first of its
    selected moves with a command that got no record during it. Cleared once every command of the
    move has a record from a later attempt. {move: started_at of the cut attempt}."""
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
        for m in [int(x) for x in (r.get("moves") or []) if isinstance(x, int) or str(x).isdigit()]:
            keys = [c["key"] for c in plan if c["move"] == m]
            if keys and any(k not in during for k in keys):
                later = {rec.get("key") for rec in cmds if (rec.get("started_at") or "") > t0 and (t1 is None or (rec.get("started_at") or "") >= t1)}
                if not all(k in later for k in keys):
                    out[m] = t0
                break
    return out


def _spec_file(spec: str) -> str:
    return spec.split("|")[0]


def _producers(plan: list[dict]) -> dict:
    prod: dict[str, int] = {}
    for c in plan:
        for spec in c.get("outputs", []):
            for f in spec.split("|"):
                prod.setdefault(f, c["move"])
    return prod


def _output_exists(out: Path, spec: str) -> bool:
    if "|" in spec:
        return any(_output_exists(out, s) for s in spec.split("|"))
    return (out / spec).exists()


def _output_dirs(cmds: list[dict]) -> list[str]:
    dirs = []
    for c in cmds:
        for spec in c.get("outputs", []):
            f = _spec_file(spec)
            d = f if f in ("cells.csv", "params.json") else str(Path(f).parent)
            if d == ".":
                d = ""
            if d not in dirs:
                dirs.append(d)
    return dirs


def progress(out: Path, move: int | None) -> dict | None:
    """A countable signal for the running move, from the files it writes: move 1 the series written
    of the admissible recordings; move 3 and 4 the cuts drawn; move 6 the prediction files of the
    24 a cut yields; move 9 the sub-moves recomputed. Nothing here is a result."""
    o = Path(out)
    if move is None:
        return None
    if move == 1:
        n_adm = 0
        if (o / "cells.csv").exists():
            h, rows, _ = read_csv_rows(o / "cells.csv")
            ia = h.index("admissible") if "admissible" in h else -1
            n_adm = sum(1 for r in rows if ia >= 0 and len(r) > ia and r[ia].lower() == "true")
        n = len(list((o / "series").glob("*.npz"))) if (o / "series").is_dir() else 0
        return {"move": 1, "done": n, "total": n_adm or None, "unit": "series written"}
    cuts = cut_dirs(o)
    if move in (3, 4):
        d = o / "moves" / ("03_every_run" if move == 3 else "04_portraits")
        n = sum(1 for c in cuts if (d / c / "index.html").exists()) if d.is_dir() else 0
        return {"move": move, "done": n, "total": len(cuts), "unit": "cuts drawn"}
    if move == 6:
        d = o / "moves" / "06_classify"
        n = sum(len(list((d / c).glob("predictions_*.csv"))) for c in cuts if (d / c).is_dir()) if d.is_dir() else 0
        return {"move": 6, "done": n, "total": 12 * len(cuts), "unit": "split and encoding predictions (12 per cut at most; a cut without windows writes none)"}
    if move == 11:
        sd = sibling_dirs(o)[o.name + EXTRA_MOVES[11]["suffix"]]
        n = sum(1 for st in EXTRA_MOVES[11]["steps"] for c in cuts if (sd / st["dir"] / c / "summary.json").exists()) if sd.is_dir() else 0
        return {"move": 11, "done": n, "total": len(EXTRA_MOVES[11]["steps"]) * len(cuts), "unit": "step and cut summaries"}
    if move == 12:
        sd = sibling_dirs(o)[o.name + EXTRA_MOVES[12]["suffix"]]
        n = sum(len(list((sd / c).glob("predictions_*.csv"))) for c in cuts if (sd / c).is_dir()) if sd.is_dir() else 0
        return {"move": 12, "done": n, "total": 12 * len(cuts), "unit": "level and feature-set predictions (12 per cut)"}
    if move == 13:
        sd = sibling_dirs(o)[o.name + EXTRA_MOVES[13]["suffix"]]
        n = sum(len(list((sd / st["dir"] / c / "spectra").glob("*.csv"))) for st in EXTRA_MOVES[13]["steps"] for c in cuts if (sd / st["dir"] / c / "spectra").is_dir()) if sd.is_dir() else 0
        n_rec = 0
        if (o / "cells.csv").exists():
            h, rows, _ = read_csv_rows(o / "cells.csv")
            ia = h.index("admissible") if "admissible" in h else -1
            n_rec = sum(1 for r in rows if ia >= 0 and len(r) > ia and r[ia].lower() == "true")
        return {"move": 13, "done": n, "total": (2 * len(cuts) * n_rec) or None, "unit": "recording spectra written (both steps, every cut)"}
    if move in (14, 15, 16):
        sd = sibling_dirs(o)[o.name + EXTRA_MOVES[move]["suffix"]]
        n = sum(len(list((sd / st["dir"] / c / "spectra").glob("*.csv"))) for st in EXTRA_MOVES[move]["steps"] if st["dir"] for c in cuts if (sd / st["dir"] / c / "spectra").is_dir()) if sd.is_dir() else 0
        n_rec = 0
        if (o / "cells.csv").exists():
            h, rows, _ = read_csv_rows(o / "cells.csv")
            ia = h.index("admissible") if "admissible" in h else -1
            n_rec = sum(1 for r in rows if ia >= 0 and len(r) > ia and r[ia].lower() == "true")
        return {"move": move, "done": n, "total": (2 * len(cuts) * n_rec) or None, "unit": "recording spectra written (both steps, every cut)"}
    if move == 9:
        d = o / "moves" / "09_removed"
        subs = ["series", "03_every_run", "04_portraits", "05_similarity", "06_classify", "07_floor"]
        n = sum(1 for s in subs if (d / s).is_dir()) if d.is_dir() else 0
        return {"move": 9, "done": n, "total": len(subs), "unit": "parts (the series, then moves 3 to 7)"}
    return None


def move_states(out: Path, plan: list[dict], running: dict | None, flags: dict | None = None) -> list[dict]:
    """One record per move 0..10: the driver's name for it, the commands (from the driver's plan)
    with their last record, the move's state, and which earlier moves it reads from. Move 9 not in
    the plan reads `off` (decision D1). A move whose last attempt was stopped mid-run reads `not run`
    with the time it was cut, whatever its earlier records say. Then one row per move beside the run
    (11, 12; `extra_move_states`), from the sibling folders' own records."""
    ledger = load_ledger(out) or {}
    last: dict[str, dict] = {}
    for rec in ledger.get("commands", []):
        if str(rec.get("status")) == "dry-run" and rec.get("key") in last:
            continue                                   # the driver's own rule: a dry run never demotes a result
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
            states.append(st)
            for spec in c.get("inputs", []):
                pm = producers.get(_spec_file(spec))
                if pm is not None and pm < m:
                    reads.add(pm)
            crows.append({"key": c["key"], "name": c["name"], "module": c["module"], "sub": c["sub"], "internal": False, "template": False,
                          "line": c["line"], "outputs": c.get("outputs", []), "inputs": c.get("inputs", []),
                          "status": (rec or {}).get("status"), "state": st, "stale": (rec or {}).get("stale"),
                          "exit_code": (rec or {}).get("exit_code"), "started_at": (rec or {}).get("started_at"),
                          "finished_at": (rec or {}).get("finished_at"), "elapsed_s": (rec or {}).get("elapsed_s"),
                          "stderr_tail": ((rec or {}).get("stderr_tail") or "")[-600:] if st in ("failed", "refused") else None,
                          "stdout_tail": ((rec or {}).get("stdout_tail") or "")[-1200:] if st == "done" else None,
                          "outputs_present": [s for s in c.get("outputs", []) if _output_exists(outp, s)]})
        batch = (running or {}).get("batch") or {}
        off = (m == ROOM_MOVE and not cmds)
        if off:
            state = "off"
        elif running and running.get("move") == m:
            state = "running"
        elif m in (batch.get("moves") or []) and isinstance(running.get("move"), int) and m > running["move"] \
                and not all(s == "done" and (c.get("started_at") or "") >= (batch.get("started_at") or "") for s, c in zip(states, crows)):
            state = "queued"
        elif m in cut:
            state = "not run"
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
        requires = [] if m == 0 else ([8] if m == 10 else [m - 1])       # the runbook's order; move 10 follows move 8 (move 9 is optional)
        rows.append({"move": m, "title": sec.get("title") or f"move {m}", "what": sec.get("body", ""), "state": state, "commands": crows,
                     "state_before_batch": (None if state != "queued" else
                                            ("not run" if not states or all(s == "not run" for s in states) else "done" if all(s == "done" for s in states) else "partial")),
                     "reads_from": sorted(reads), "requires": requires,
                     "outputs_missing": missing, "last_finished_at": max([c["finished_at"] or "" for c in crows] or [""]) or None,
                     "interrupted_at": cut.get(m), "look_at": sec.get("look_at", ""), "output_dirs": _output_dirs(cmds),
                     "note": ("decision D1: off by default; switch it on with the room removal flag (--room-removal) to plan it" if off else
                              "optional (decision D1): recomputes moves 3 to 7 without the idle core into moves/09_removed/" if m == ROOM_MOVE else None)})
    rows += extra_move_states(out, running, sections, flags)
    for r in rows:
        req = r["requires"]
        r["runnable"] = (not running) and r["state"] != "off" and all(any(x["move"] == q and x["state"] == "done" for x in rows) for q in req)
    return rows


def sibling_record(out: Path, move: int) -> dict | None:
    """The record.json of a move beside the run (its own record book, in its sibling folder), as written."""
    sd = sibling_dirs(out)[Path(out).name + EXTRA_MOVES[move]["suffix"]]
    p = sd / "record.json"
    if not p.is_file():
        return None
    try:
        return read_json(p)
    except (OSError, json.JSONDecodeError):
        return {"error": "record.json is not readable JSON", "entries": []}


def extra_argv(out: Path, move: int, step: dict, flags: dict, force: bool = False, dry_run: bool = False) -> list[str]:
    """The move's own command: `python3 -m plan12_grounding.<module> run --run <out> [--step N] [--n-jobs J]
    [--n-estimators T] [--force] [--dry-run]`; the tab's job and tree flags are passed when set (the paper
    preset's values are move 6's own); everything else is the module's default, read from the run's records."""
    x = EXTRA_MOVES[move]
    argv = [sys.executable, "-m", f"{TOOLKIT_NAME}.{x['module']}", step.get("sub", "run"), "--run", str(out)] + list(step["args"])
    for dest in x.get("flags", EXTRA_FLAGS):
        v = (flags or {}).get(dest)
        if not _unset(v):
            argv += [f"--{dest.replace('_', '-')}", str(int(float(v)))]
    if force:
        argv.append("--force")
    if dry_run:
        argv.append("--dry-run")
    return argv


def extra_move_states(out: Path, running: dict | None, sections: dict | None = None, flags: dict | None = None) -> list[dict]:
    """One board row per move beside the run (11, 12): its commands from EXTRA_MOVES with their last
    record in the sibling folder's record.json, the state from those records, the files present."""
    o = Path(out)
    sections = sections or {}
    rows = []
    for mv, x in sorted(EXTRA_MOVES.items()):
        sib = o.name + x["suffix"]
        sd = sibling_dirs(o)[sib]
        rec = sibling_record(o, mv) or {}
        last: dict = {}
        for e in rec.get("entries") or []:
            last[e.get("key")] = e
        crows, states = [], []
        for st in x["steps"]:
            e = last.get(st["key"]) or {}
            status = e.get("status")
            if status == "running":
                # an open entry: the move is running (this console's launch, a shell run holding the sibling's lock), or it was cut short
                lock = sd / x.get("lock", ".idle_class.lock" if mv == 11 else ".new_blocks.lock")
                held = False
                if lock.is_file():
                    with contextlib.suppress(OSError, json.JSONDecodeError, TypeError, ValueError):
                        held = _pid_alive(int(read_json(lock).get("pid") or 0))
                st_ = "running" if ((running and running.get("move") == mv) or held) else "failed"
                status = (status + (" (from a shell; the sibling's lock is held)" if held and not (running and running.get("move") == mv) else "")) if st_ == "running" \
                    else "failed: the record is open and no process holds it (stopped mid-run)"
            else:
                st_ = classify(status)
            states.append(st_)
            outs = [f"{sib}/{f}" for f in (e.get("outputs") or [])]
            key_files = [f"{sib}/{st['done_file']}"] if st.get("done_file") else ([f"{sib}/{st['dir']}/step.json"] if st["dir"] else [f"{sib}/new_blocks.json"])
            crows.append({"key": f"{mv}:{st['name']}", "name": st["name"], "module": x["module"], "sub": "run", "internal": False, "template": False,
                          "line": shell_line(extra_argv(o, mv, st, flags or {})), "outputs": key_files, "inputs": [],
                          "status": status, "state": st_, "stale": None, "exit_code": e.get("exit_code"), "started_at": e.get("started_at"),
                          "finished_at": e.get("finished_at"), "elapsed_s": e.get("elapsed_s"),
                          "stderr_tail": (e.get("error") or "")[-600:] if st_ in ("failed", "refused") else None, "stdout_tail": None,
                          "outputs_present": [f for f in key_files if (o.parent / f).exists()], "n_outputs_recorded": len(outs)})
        if running and running.get("move") == mv:
            state = "running"
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
        missing = [f for c in crows for f in c["outputs"] if state == "done" and not (o.parent / f).exists()]
        sec = sections.get(mv, {})
        rows.append({"move": mv, "title": sec.get("title") or x["title"], "what": sec.get("body") or x["what"], "state": state, "commands": crows,
                     "state_before_batch": None, "reads_from": list(x["reads_from"]), "requires": list(x["requires"]), "outputs_missing": missing,
                     "last_finished_at": max([c["finished_at"] or "" for c in crows] or [""]) or None, "interrupted_at": None,
                     "look_at": sec.get("look_at", ""), "output_dirs": [sib], "beside_the_run": True, "sibling": sib, "sibling_exists": sd.is_dir(),
                     "record": f"{sib}/record.json" if (sd / "record.json").is_file() else None, "note": x["note"]})
    return rows


# ---------------------------------------------------------------------------
# the cells table: cells.csv joined with the series records, for display
# ---------------------------------------------------------------------------

CELL_COLUMNS = ["cell_id", "role", "kernel", "archetype_predicted", "seed", "rep", "rep_dir", "campaign", "index_status", "seed_from_map",
                "keep_first_pairs", "head_drop_pairs", "admissible", "reason", "series_status", "n_pairs", "n_rows"]


def cells(out: Path) -> dict:
    o = Path(out)
    p = o / "cells.csv"
    if not p.exists():
        return {"exists": False, "path": "cells.csv", "why": "not yet written by the engine: run move 0 (inputs and index)"}
    header, rows, _ = read_csv_rows(p)
    idx = {h: i for i, h in enumerate(header)}
    ex = {}
    ep = o / "moves" / "01_extract" / "extract.json"
    if ep.exists():
        with contextlib.suppress(OSError, json.JSONDecodeError):
            ex = read_json(ep).get("recordings") or {}
    counts = {}
    if (o / "params.json").exists():
        with contextlib.suppress(OSError, json.JSONDecodeError):
            counts = read_json(o / "params.json").get("counts") or {}
    out_rows, n_unknown = [], 0
    for r in rows:
        d = {h: (r[i] if i < len(r) else "") for h, i in idx.items()}
        if d.get("role") not in ("kernel", "idle"):
            n_unknown += 1                              # never listed by name: the corpus is the kernels plus idle
            continue
        rec = ex.get(d.get("cell_id", ""), {})
        d["series_status"] = rec.get("status", "")
        d["n_pairs"] = rec.get("n_pairs", "")
        d["n_rows"] = rec.get("n_rows", "")
        out_rows.append(d)
    order = {"kernel": 0, "idle": 1}
    out_rows.sort(key=lambda d: (order.get(d.get("role"), 2), d.get("kernel", ""), int(float(d.get("rep") or 0))))
    return {"exists": True, "path": "cells.csv", "columns": CELL_COLUMNS, "rows": out_rows,
            "n_kernel": sum(1 for d in out_rows if d["role"] == "kernel"), "n_idle": sum(1 for d in out_rows if d["role"] == "idle"),
            "n_admissible": sum(1 for d in out_rows if str(d.get("admissible", "")).lower() == "true"),
            "n_other_in_csv": n_unknown,
            "n_other_counted_by_move_0": counts.get("other_recordings_counted_not_named"),
            "other_by_index_status": counts.get("other_recordings_by_index_status"),
            "series": {"n": sum(1 for d in out_rows if d["series_status"]), "path": "moves/01_extract/extract.json"}}


def expand_files(out: Path, specs: list[str]) -> list[str]:
    """`{cut}` -> every cut folder of the run (params.json), `{idle13}` and `{newblocks}` -> the sibling
    folders' names; the order of the specs kept, each cut in turn."""
    o = Path(out)
    subs = {"idle13": o.name + EXTRA_MOVES[11]["suffix"], "newblocks": o.name + EXTRA_MOVES[12]["suffix"], "pagefourier": o.name + EXTRA_MOVES[13]["suffix"]}
    files = []
    for f in specs:
        for cut in (cut_dirs(o) if "{cut}" in f else [None]):
            g = f.replace("{cut}", cut or "")
            for k, v in subs.items():
                g = g.replace("{" + k + "}", v)
            if g not in files:
                files.append(g)
    return files


def views(out: Path) -> list[dict]:
    o = Path(out)
    outs = []
    for v in VIEWS:
        files = []
        for f in expand_files(o, v["files"]):
            try:
                p = safe_path(o, f)
            except PanelError:
                continue
            files.append({"path": f, "exists": p.exists(), "kind": TEXT_KINDS.get(p.suffix, "binary"), "bytes": p.stat().st_size if p.exists() else None})
        outs.append({**{k: v[k] for k in ("id", "title", "move")}, "note": v.get("note"), "gallery": v.get("gallery"), "files": files,
                     "any": any(f["exists"] for f in files)})
    return outs


def _effective_scores(doc: dict) -> dict:
    """plan11's own reading of a split's score (models.effective_scores: the re-run without the
    quarantined features when B1-G3 quarantined any); the document as written when plan11 is absent."""
    try:
        if str(QEMU_DIR) not in sys.path:
            sys.path.insert(0, str(QEMU_DIR))
        from plan11_encoding_ladder import models as M11
        return M11.effective_scores(doc)
    except Exception:                                   # noqa: BLE001  plan11 not importable: the document as written, said so
        return {**doc, "score_source": "scores.json as written (plan11 models.effective_scores not importable)"}


def encoding_table2(out: Path) -> dict:
    """Next to move 6, read only: the named encoding run's matching numbers for the combined rung at
    the grid point move 6 used, the D2 folder and the D3 grid point both read from
    `moves/06_classify/classify.json` (what move 6 ran with, not today's files), each split's score
    read through plan11's `models.effective_scores`, the engine's own `table2_check.json`, and the
    encoding run's EUSIPCO Table 2 files when it wrote them. Nothing else outside <out> is read."""
    o = Path(out)
    cj = o / "moves" / "06_classify" / "classify.json"
    if not cj.exists():
        return {"exists": False, "why": "move 6 has not run yet: the D2 folder and the D3 grid point are read from moves/06_classify/classify.json"}
    try:
        prm = read_json(cj).get("params") or {}
    except (OSError, json.JSONDecodeError):
        return {"exists": False, "why": "moves/06_classify/classify.json unreadable"}
    enc = prm.get("D2_encoding_out")
    if not enc:
        return {"exists": False, "why": "move 6 ran without an encoding run (decision D2): E0 was not computed"}
    e = Path(os.path.expanduser(enc))
    if not e.is_dir():
        return {"exists": False, "why": f"the encoding run move 6 used is not on this machine: {enc}", "encoding_out": enc}
    grid = prm.get("D3_grid_id") or (prm.get("D3_window") or {}).get("grid_id")
    if not grid:
        return {"exists": False, "why": "classify.json names no D3 grid point", "encoding_out": str(e)}
    res = {"exists": True, "encoding_out": str(e), "rung": "combined", "rows": [], "table2": {}, "grid_id": grid,
           "grid_source": f"move 6's own record (classify.json: {(prm.get('D3_window') or {}).get('source', 'D3')})",
           "e0_status": prm.get("e0_status"), "e0_identity": prm.get("e0_identity"), "cut_declared": (prm.get("cuts") or {}).get("declared")}
    for split, lab in (("within_trace", "kernel"), ("loro", "kernel"), ("loko", "archetype")):
        d = e / "gates" / "splits" / "combined" / grid / f"{split}__{lab}"
        sc = d / "scores.json"
        row = {"split": split, "labelspace": lab, "path": str(sc.relative_to(e)), "exists": sc.exists()}
        if sc.exists():
            with contextlib.suppress(OSError, json.JSONDecodeError):
                j = _effective_scores(read_json(sc))
                row.update({k: j.get(k) for k in ("status", "accuracy", "macro_recall", "majority", "null_p95", "b1_g1", "n_perm", "feature_count", "feature_count_used", "n_units",
                                                   "score_source", "quarantined_features")})
        res["rows"].append(row)
    tc = o / "moves" / "06_classify" / "table2_check.json"
    if tc.exists():
        with contextlib.suppress(OSError, json.JSONDecodeError):
            res["table2_check"] = read_json(tc)
    for name in ("eusipco_table2.csv", "eusipco_table2.md"):
        p = e / "report" / "tables" / name
        if p.exists():
            if name.endswith(".csv"):
                h, rows, total = read_csv_rows(p)
                res["table2"][name] = {"header": h, "rows": rows, "n_rows": total}
            else:
                res["table2"][name] = {"text": p.read_text(encoding="utf-8", errors="replace")[:20000]}
    res["note"] = ("E0 of move 6 at the declared cut is the encoding run's own combined-rung features at this grid point, under the same settings, "
                   "order and quarantine; the engine's table2_check.json says whether each row is reproduced exactly")
    return res


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


def external_drivers(out: Path, exclude_pid: int | None = None) -> list[dict]:
    """Driver processes on this machine writing to the same <out> that this console did not start
    (a shell run): found by their command line, so a second writer to the record book is refused."""
    try:
        p = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    found = []
    needle = f"{TOOLKIT_NAME}.run_moves"
    beside = {f"{TOOLKIT_NAME}.{x['module']}": mv for mv, x in EXTRA_MOVES.items()}
    for line in p.stdout.splitlines():
        line = line.strip()
        if not line or " run " not in line or (needle not in line and not any(n in line for n in beside)):
            continue
        try:
            pid_s, cmd = line.split(None, 1)
            pid = int(pid_s)
        except ValueError:
            continue
        if exclude_pid is not None and pid == exclude_pid:
            continue
        toks = shlex.split(cmd) if "'" in cmd or '"' in cmd else cmd.split()
        mv_beside = next((mv for n, mv in beside.items() if n in toks), None)
        try:
            o = toks[toks.index("--run" if mv_beside is not None else "--out") + 1]
        except (ValueError, IndexError):
            continue
        if Path(os.path.expanduser(o)).resolve() != Path(out).resolve():
            continue
        if "--dry-run" in toks:
            continue                                       # a dry run writes nothing and holds no lock
        moves = None
        if mv_beside is not None:
            moves = str(mv_beside)
        elif "--moves" in toks:
            with contextlib.suppress(IndexError):
                moves = toks[toks.index("--moves") + 1]
        found.append({"pid": pid, "moves": moves, "command": cmd[:300], "beside_the_run": mv_beside is not None})
    return found


def running_batch(out: Path, plan: list[dict]) -> dict | None:
    """The batch the live driver is working through, from the record book's last `runs` entry while
    it is open: its moves, its start, and the move it is on (the first selected move with a command
    that has no record started since the batch began). None when the record book has no open run."""
    led = load_ledger(out) or {}
    runs = led.get("runs") or []
    if not runs or runs[-1].get("finished_at"):
        return None
    r = runs[-1]
    t0 = r.get("started_at") or ""
    moves = [int(m) for m in (r.get("moves") or []) if isinstance(m, int) or str(m).isdigit()]
    seen = {rec.get("key") for rec in led.get("commands", []) if (rec.get("started_at") or "") >= t0}
    current = next((m for m in moves if any(c["key"] not in seen for c in plan if c["move"] == m)), None)
    return {"moves": moves, "started_at": t0, "current": current,
            "spec": (f"{moves[0]}-{moves[-1]}" if moves and moves == list(range(moves[0], moves[-1] + 1)) and len(moves) > 1 else ",".join(map(str, moves)))}


def _external_move(out: Path, plan: list[dict], spec: str | None) -> int | None:
    if spec is not None and str(spec).isdigit() and int(spec) in EXTRA_MOVES:
        return int(spec)
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
        self.lock = threading.RLock()                 # re-entrant: _reap takes it too, from stop() and from the readers
        self.proc: subprocess.Popen | None = None
        self.running: dict | None = None
        self.cfg = self._load()

    # ---- configuration
    def _load(self) -> dict:
        d = {"out": "", "root": "", "preset": "paper", "flags": dict(PRESETS["paper"]["flags"])}
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
        for k in CONFIG_FIELDS:
            if k in body:
                self.cfg[k] = str(body[k] or "").strip()
        if isinstance(body.get("flags"), dict):
            valid = {f["dest"] for f in driver_flags()}
            for k, v in body["flags"].items():
                if k not in valid:
                    raise PanelError(f"not a driver flag: {k}")
                if _unset(v):
                    self.cfg["flags"].pop(k, None)
                else:
                    self.cfg["flags"][k] = v
            norm = lambda d: {k: v for k, v in (d or {}).items() if not _unset(v)}   # noqa: E731  an unset flag and a false switch are the same thing; 0 is a value
            self.cfg["preset"] = "custom" if norm(self.cfg["flags"]) != norm(PRESETS.get(self.cfg.get("preset", ""), {}).get("flags")) else self.cfg["preset"]
        self.save()
        return self.config()

    def config(self) -> dict:
        out, root = self.cfg.get("out") or "", self.cfg.get("root") or ""
        flags = self.cfg.get("flags", {})
        return {"out": out, "root": root, "preset": self.cfg.get("preset"), "flags": flags,
                "out_exists": bool(out) and Path(os.path.expanduser(out)).is_dir(), "root_exists": bool(root) and Path(os.path.expanduser(root)).is_dir(),
                "room_removal": bool(flags.get("room_removal")), "source": ("ssh" if flags.get("ssh") else "root" if root else "none"),
                "presets": {k: {"label": v["label"], "flags": v["flags"]} for k, v in PRESETS.items()},
                "driver_flags": driver_flags() if toolkit_present() else [], "toolkit": toolkit_identity(),
                "run_line": shell_line(driver_argv("run", out or "<out>", root, "N", flags)) if toolkit_present() else None,
                "config_path": str(self.config_path)}

    def _need(self) -> tuple[Path, str]:
        if not toolkit_present():
            raise PanelError(f"the engine is not on disk at {TOOLKIT}")
        out, root = self.cfg.get("out") or "", self.cfg.get("root") or ""
        if not out:
            raise PanelError("set the output folder <out> first")
        return Path(os.path.expanduser(out)), root

    # ---- the plan and the board
    def plan(self) -> list[dict]:
        out, root = self._need()
        return build_plan(str(out), root, self.cfg.get("flags", {}))

    def board(self) -> dict:
        out, root = self._need()
        self._reap()
        plan = self.plan()
        ext = external_drivers(out, self.proc.pid if self.proc else None)
        running = dict(self.running) if self.running else None
        if not running and ext:
            running = {"id": None, "move": _external_move(out, plan, ext[0]["moves"]), "pid": ext[0]["pid"], "started_at": None, "external": True,
                       "moves": ext[0]["moves"], "command": ext[0]["command"]}
        if running:
            b = running_batch(out, plan)
            if b:
                b["source"] = "shell" if running.get("external") else "console"
                running["batch"] = b
                if b.get("current") is not None:
                    running["move"] = b["current"]
            running["progress"] = progress(out, running.get("move") if isinstance(running.get("move"), int) else None)
        return {"out": str(out), "root": root, "room_removal": bool(self.cfg.get("flags", {}).get("room_removal")),
                "moves": move_states(out, plan, running, self.cfg.get("flags", {})), "running": running, "external": ext,
                "ledger": self._ledger_summary(out), "launches": self.launches()[:20],
                "beside_the_run": {str(mv): {"sibling": out.name + x["suffix"], "exists": sibling_dirs(out)[out.name + x["suffix"]].is_dir()} for mv, x in EXTRA_MOVES.items()}}

    def _ledger_summary(self, out: Path) -> dict:
        led = load_ledger(out)
        if not led:
            return {"exists": False, "path": "driver_state.json"}
        runs = led.get("runs", [])
        return {"exists": True, "path": "driver_state.json", "n_runs": len(runs), "n_commands": len(led.get("commands", [])),
                "package_version": led.get("package_version"), "toolkit_fingerprint": (led.get("toolkit_fingerprint") or {}).get("sha256"),
                "last_run": runs[-1] if runs else None, "params": led.get("params")}

    # ---- launches
    def _launch_dir(self, out: Path) -> Path:
        d = out / CONSOLE_DIR / "launches"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def launch(self, move, force: bool = False) -> dict:
        """One driver run of one move: `run_moves run --out O [--root R] --moves N <flags> [--force]`."""
        out, root = self._need()
        with self.lock:
            self._reap()
            if self.running:
                raise PanelError(f"move {self.running['move']} is still running (launch {self.running['id']}); one engine process at a time")
            ext = external_drivers(out)
            if ext:
                if ext[0].get("beside_the_run"):
                    raise PanelError(f"move {ext[0]['moves']} is running beside this run from a shell (pid {ext[0]['pid']}); one engine process at a time")
                raise PanelError(f"a driver started outside the console is running on this output folder (pid {ext[0]['pid']}, moves {ext[0]['moves']}); "
                                 "two writers to driver_state.json are refused")
            move = int(move)
            if move < 0 or (move > MAX_MOVE and move not in EXTRA_MOVES):
                raise PanelError(f"moves are 0 to {MAX_MOVE}, and {', '.join(str(m) for m in EXTRA_MOVES)} beside the run")
            plan = self.plan()
            flags = self.cfg.get("flags", {})
            board = move_states(out, plan, None, flags)
            row = next(r for r in board if r["move"] == move)
            if row["state"] == "off":
                raise PanelError("move 9 is off (decision D1): switch the room removal flag on first")
            if not row["runnable"]:
                need = [q for q in row["requires"] if not any(x["move"] == q and x["state"] == "done" for x in board)]
                raise PanelError(f"move {move} waits for move(s) {', '.join(str(q) for q in need)} to be done (the runbook's order)")
            if move == 0 and not root and not flags.get("ssh"):
                raise PanelError("move 0 needs a source: the local root <root>, or the ssh flags (--ssh and --remote-root)")
            if move == 0 and self.cfg.get("preset") == "paper" and _unset(flags.get("encoding_out")):
                raise PanelError("the paper preset needs the encoding run named (--encoding-out, decision D2): without it admissibility comes from the index alone "
                                 "and lexer seed 6898 is not refused; set the flag in the driver flags, or choose another preset")
            if row.get("interrupted_at") and not force:
                force = True                                # a stopped move runs again as if forced: nothing half-rewritten can be skipped over
            if move in EXTRA_MOVES:
                # beside the run: the module's own command, never the driver; move 11's two steps run one after the other in one shell,
                # so one launch, one log and one Stop (the process group) cover both; the module refuses (exit 3) while a move holds the run's lock
                lock = out / ".driver.lock"
                if lock.is_file():
                    with contextlib.suppress(OSError, json.JSONDecodeError):
                        pid = read_json(lock).get("pid")
                        if pid and _pid_alive(int(pid)):
                            raise PanelError(f"a move is running on this run ({lock.name}: pid {pid}); move {move} refuses to start beside a running move")
                steps = [extra_argv(out, move, st, flags, force) for st in EXTRA_MOVES[move]["steps"]]
                if len(steps) == 1:
                    argv = steps[0]
                else:
                    argv = ["/bin/sh", "-c", " && ".join(" ".join(shlex.quote(t) for t in a) for a in steps)]
                shell = [shell_line(a) for a in steps]
            else:
                argv = driver_argv("run", str(out), root, move, flags, force)
                shell = [shell_line(argv)]
            ldir = self._launch_dir(out)
            base = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_move{move}"
            lid, n = base, 1
            while (ldir / f"{lid}.json").exists():
                n += 1
                lid = f"{base}-{n}"
            log = ldir / f"{lid}.log"
            rec = {"id": lid, "move": move, "force": bool(force), "argv": argv, "shell": shell,
                   "cwd": str(QEMU_DIR), "started_at": now_iso(), "finished_at": None, "exit_code": None, "log": str(log),
                   "toolkit": toolkit_identity(), "config": {"preset": self.cfg.get("preset"), "flags": dict(flags), "out": str(out), "root": root},
                   "driver_params": None, "driver_run": None,
                   "beside_the_run": (f"{out.name}{EXTRA_MOVES[move]['suffix']}/record.json" if move in EXTRA_MOVES else None),
                   "note": ("started from the analysis console; the move's own command, its record in the sibling folder (not the run's record book)" if move in EXTRA_MOVES
                            else "started from the analysis console; the same command line the runbook gives")}
            (ldir / f"{lid}.json").write_text(json.dumps(rec, indent=1))
            fh = open(log, "ab")
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            self.proc = subprocess.Popen(argv, cwd=str(QEMU_DIR), stdout=fh, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            fh.close()
            self.running = {"id": lid, "move": move, "pid": self.proc.pid, "started_at": rec["started_at"], "out": str(out), "force": bool(force)}
            return rec

    def _reap(self):
        """Close the launch record once the process has ended. Under the lock, so a Stop and a
        reader never race on `self.proc`. The record book's run and params are attached only when
        that run is this launch's own (its pid is the launched process), never another run's."""
        with self.lock:
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
            own = [r for r in (led.get("runs") or []) if r.get("pid") == self.proc.pid]
            mv = self.running.get("move")
            if mv in EXTRA_MOVES:
                srec = sibling_record(out, mv) or {}
                t0 = rec.get("started_at") or ""
                mine = [e for e in (srec.get("entries") or []) if (e.get("started_at") or "") >= t0]
                rec["driver_run"] = None
                rec["driver_params"] = None
                rec["step_records"] = [{k: e.get(k) for k in ("key", "status", "exit_code", "started_at", "finished_at", "elapsed_s", "error")} for e in mine]
                rec["driver_run_note"] = (f"a move beside the run: its record is {out.name}{EXTRA_MOVES[mv]['suffix']}/record.json ({len(mine)} entr{'y' if len(mine) == 1 else 'ies'} from this launch)"
                                          if mine else "the module wrote no record entry for this launch (a refusal before its record: the run's lock held, or a bad argument)")
            elif own:
                rec["driver_run"] = own[-1]
                rec["driver_params"] = led.get("params")
                rec["package_version"] = led.get("package_version")
            else:
                rec["driver_run"] = None
                rec["driver_params"] = None
                rec["driver_run_note"] = "the driver wrote no run record for this launch (it ended before its record book entry: a refusal, a stop in its first moment, or a bad argument)"
            p.write_text(json.dumps(rec, indent=1))
            self.proc = None
            self.running = None

    def stop(self) -> dict:
        with self.lock:
            self._reap()
            if not self.running or self.proc is None:
                raise PanelError("nothing is running")
            proc, rid = self.proc, self.running["id"]              # local references: a reader's reap cannot pull them from under us
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
            return {"stopped": rid, "note": "the driver was terminated with its process group; its record book keeps what finished, the running command left no record; "
                                            "the move reads not run, and its next Run launches with --force"}

    def launches(self) -> list[dict]:
        out = Path(os.path.expanduser(self.cfg.get("out") or ""))
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
        out = Path(os.path.expanduser(self.cfg.get("out") or ""))
        if not re.match(r"^[0-9TZ]+_move[0-9]+(-[0-9]+)?$", lid or ""):
            raise PanelError("not a launch id")
        p = out / CONSOLE_DIR / "launches" / f"{lid}.json"
        if not p.exists():
            raise PanelError(f"no launch {lid}")
        return read_json(p)

    def log_tail(self, lid: str | None, n: int = 200) -> dict:
        self._reap()
        out = Path(os.path.expanduser(self.cfg.get("out") or ""))
        if not lid:
            lid = self.running["id"] if self.running else ((self.launches() or [{}])[0].get("id"))
        if not lid:
            return {"id": None, "lines": [], "running": False}
        if not re.match(r"^[0-9TZ]+_move[0-9]+(-[0-9]+)?$", lid):
            raise PanelError("not a launch id")
        p = out / CONSOLE_DIR / "launches" / f"{lid}.log"
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()[-n:] if p.exists() else []
        return {"id": lid, "lines": lines, "running": bool(self.running and self.running["id"] == lid), "log": str(p)}
