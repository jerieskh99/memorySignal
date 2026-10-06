#!/usr/bin/env python3
"""idle_class.py -- idle as a 13th class: steps 1 and 2 beside move 6 of plan12_grounding (2026-10-06).

  python3 -m plan12_grounding.idle_class run --run <main run out> --step 1|2 [--out <dir>]
        [--encoding-out <D2>] [--cuts declared,measured] [--n-jobs N] [--n-estimators N] [--force] [--dry-run]
  (run from VM_sampler/VM_Capture_QEMU/; the prompt is plan12_grounding/PROMPT_idle13_steps.md)

THE QUESTION THE STEPS ANSWER
In the real run, move 6's LORO (leave one recording out) recognizes all 12 kernels perfectly, lexer
included (7 of 7). But lexer barely writes memory: its run means sit at idle's level (pages 1.04 times
idle's, bits 0.99, angle 0.96), and idle is not a class in move 6 (95 kernel runs only, as in the encoding
paper's Table 2). The worry: the model may know lexer only as "the quiet one". These steps add idle as a
13th class and ask whether lexer is still recognized when "quiet" is no longer enough:
  step 1  LORO with 13 classes (the 12 kernels plus idle), at both cuts, on the four feature sets E0, E1,
          E2, E_new, built exactly as move 6 builds them. No label-shuffle null at all.
  step 2  within-trace with the same 13 classes, feature sets and cuts. No null.
Nothing else: no LOKO, no nulls.

THE DATA
The run's cells.csv rows with admissible true and role kernel or idle, each with the complete series move 1
wrote (`stats.load_runs`). Idle is exactly sleep/sleep/sleep_600/rep00N__idle_01c (8 cells). Its class label
is "idle" and its archetype "control" (the encoding run's cells.csv archetype_predicted). The kernel cells
keep move 6's order (the encoding run's cells.csv order); the 8 idle cells follow them in this run's
cells.csv order. The cuts (params.json cuts: declared_pairs, measured_pairs), the window (decision D3) and
the forest settings (trees, seed offset; n-jobs by default) are read from the run's params.json and
moves/06_classify/classify.json, so the 13-class numbers are comparable with move 6's.

WHAT IS REUSED (never copied) AND WHAT IS LOCAL
- classify.py's feature building: `classify.build_encodings` and `classify.cell_windows` (the E0 windows of
  series.window_features at the D3 window, the per-window statistics of stats.window_stats, the four column
  sets, move 6's exclusions) are called as they are. build_encodings takes its cell list from the module-level
  name `ordered_runs` (classify.py line 196), which admits kernel runs only (line 180); `ordered_runs_13` below
  is the local variant that differs ONLY in admitting idle, and `build_encodings_13` binds that one name to it
  for the duration of the call. Move 6's E0 gate (classify.run_classify lines 635-659: the declared files, the
  G-K0 relabel, the pair-rung exclusion, move 1's identity record) is repeated with the same calls.
- plan11's splits (`splits.folds_for` for loro and within_trace), models (the forest, unit scoring by cell
  majority, auto_reduce, the B1-G3 quarantine with the re-run and `models.effective_scores`) and the seeds
  (`nulls.SEED_FOREST` + move 6's seed offset) are reached through `classify.run_split`, called with no null.
- E0 for idle: the encoding run (D2) has extract folders for the 8 idle cells. Before anything runs, each is
  checked under move 6's own rules (classify.build_encodings lines 198-214): status ok in the encoding run's
  cells.csv, all_hard_pass and a failed_verdict that is not a refusal (series.admissible_cells, the pair-rung
  exclusion), an ok sidecar with an extract.csv, and cell_windows' row identity with this run's series after
  each cut (pair = seq + 1). All 8 pass: all four feature sets run. None passes, or move 6's own E0 gate
  fails: E_new only, the exact reason written into scores.csv, step.json and summary.json. Some pass and
  some do not: the step stops and prints the question (move 6's rule would drop those idle cells from every
  feature set, which is not a rule the prompt gives for idle), exit 2.

THE RULES THAT PROTECT THE LIVE RUN (the prompt, 2026-10-06)
- This is ONE new file; nothing existing in plan12_grounding/ is edited. Every move's code fingerprint
  (run_moves.module_closure) covers only the modules a move imports, so a module nothing imports changes no
  fingerprint and marks no finished move stale.
- The run folder is read only here: cells.csv, params.json, series/, moves/01_extract, moves/06_classify and
  the encoding run's files are read; nothing under <run> is ever written. `--out` defaults to the sibling
  "<run>_idle13" and a path inside the run folder is refused (exit 2).
- The step refuses to start (exit 3) while <run>/.driver.lock exists and its pid is alive, so it can never
  compete with a running move. `--dry-run` skips the refusal, writes nothing (no folder, no lock, no record)
  and prints what it would do: the cells found (the real run: 103 = 95 kernel + 8 idle), the cuts, D2, D3,
  and whether E0 can be built for idle.
- One writer per output folder: <out>/.idle_class.lock, a stale lock from a dead process is taken over.
  `run_moves.install_sigterm` is installed, so a Stop (SIGTERM) ends the step through its finally blocks.

OUTPUTS, under <out>/step1_loro/cut<C>/ and <out>/step2_within_trace/cut<C>/
  scores.csv            feature set, status, accuracy, macro recall, majority baseline, n units, n windows,
                        n classes, feature count, feature count used (after auto_reduce), score source,
                        quarantined features, n folds, and the null column saying no null ran
  recall_per_class.csv  13 classes x feature sets (rows in the run's cells.csv class order, idle last)
  confusion_<E>.csv     13 x 13 per feature set (rows true, columns predicted), the same class order
  margins.csv           E1-E0, E2-E0, E2-E1, E_new-E0: accuracy and macro-recall differences, cells gained and
                        lost; no null, and the status column says so
  confusion_<E>.svg     one per feature set run (E2 and E_new at least), lexer and idle rows highlighted, in
                        the engine's SVG style (figures.svg_open); confusion_<E>.csv is its data
  predictions_<E>.csv   per cell: class, fold, y_true, y_pred, windows, vote fraction (the re-run's when a
                        feature was quarantined, as move 6; predictions_full_model_<E>.csv keeps the full model's)
  features.npz, excluded_cells.csv   as move 6 writes them
  summary.json          lexer's and idle's recall per feature set, what each is mistaken for and what is taken
                        for it, the margins, and the comparison with move 6 (12 classes) read from the run's
                        moves/06_classify/cut<C>/ (LORO: lexer 7 of 7)
  <out>/<step>/step.json          the step's record: command, params, the E0 check, the cells, both cuts
  <out>/<step>/e0_idle_check.json the per-cell E0 check for idle at every cut
  <out>/record.json     one entry per step run: command, inputs with sha256 (the run's files and the
                        encoding run's), params, the sha256 of idle_class.py and of each plan12 module it
                        imports (run_moves.code_fingerprint), started_at, finished_at, exit status. A re-run
                        of a finished step is skipped unless --force, or its inputs or code changed.

THE REAL RUN (after move 9 ends; the console must not be restarted while a move runs)
  cd /Users/jeries/Desktop/projects/thesis/memorySignal/mem_sig/VM_sampler/VM_Capture_QEMU
  python3 -m plan12_grounding.idle_class run \\
      --run /Users/jeries/Desktop/projects/thesis/memorySignal/spl_paper/results/run_20261005 --step 1
  python3 -m plan12_grounding.idle_class run \\
      --run /Users/jeries/Desktop/projects/thesis/memorySignal/spl_paper/results/run_20261005 --step 2
  Defaults there: --out /Users/jeries/Desktop/projects/thesis/memorySignal/spl_paper/results/run_20261005_idle13,
  the encoding run and the window of move 6's classify.json (/Users/jeries/encoding_run_20260929, W64_H32),
  cuts 16 and 192, 300 trees and 4 processes (move 6's recorded values). Add --dry-run first to see the plan.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import argparse  # noqa: E402
import csv  # noqa: E402
import html  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

from plan12_grounding import __version__, toolkit_fingerprint  # noqa: E402
from plan12_grounding import classify as C  # noqa: E402  (move 6: the feature building, the split runner, the E0 gate; reused, never copied)
from plan12_grounding.run_moves import LOCK as DRIVER_LOCK, _pid_alive, code_fingerprint, install_sigterm, now_iso, read_json, sha256_file, write_json  # noqa: E402
from plan12_grounding.stats import cuts_of, load_runs  # noqa: E402
from plan12_grounding.figures import svg_open, write_csv  # noqa: E402
from plan11_encoding_ladder import schema  # noqa: E402
from plan11_encoding_ladder import series as S11  # noqa: E402
from plan11_encoding_ladder import verdicts as V  # noqa: E402
from plan11_encoding_ladder.nulls import SEED_FOREST  # noqa: E402

CITATION = ("plan12_grounding/PROMPT_idle13_steps.md (2026-10-06); plan12_grounding/classify.py move 6 (build_encodings, cell_windows, run_split, "
            "declare_e0, identity_record; its E0 gate, lines 635-659; ordered_runs, lines 177-184, varied here to admit idle); plan11_encoding_ladder "
            "splits.py folds_for, models.py fit_predict_units / score_units / majority_baseline / quarantine_l1 / effective_scores, nulls.py SEED_FOREST "
            "(the encoding paper's Table 2 settings as move 6 applies them; 13 classes = the 12 kernels plus idle; no label-shuffle null)")
STEPS = {1: ("step1_loro", "loro"), 2: ("step2_within_trace", "within_trace")}
IDLE_LABEL = "idle"
IDLE_ARCHETYPE = "control"                               # the encoding run's cells.csv archetype_predicted for the 8 idle cells
IDLE_REC_LAYOUT = "sleep/sleep/sleep_600/rep00N__idle_01c"
N_IDLE_EXPECTED = 8
ENCODINGS = C.ENCODINGS
MARGIN_PAIRS = C.MARGIN_PAIRS
CLASS_ORDER = list(schema.KERNEL_NAMES) + [IDLE_LABEL]   # the run's cells.csv order (move 0: schema.KERNEL_NAMES, then idle)
HIGHLIGHT = ("lexer", IDLE_LABEL)
NO_NULL = V.not_run("no label-shuffle null in these steps (none was asked for: the prompt of 2026-10-06)")
OUT_SUFFIX = "_idle13"
LOCK_NAME = ".idle_class.lock"
RECORD_NAME = "record.json"
EXIT_OK, EXIT_ERROR, EXIT_MISSING, EXIT_REFUSED = 0, 1, 2, 3
MODE_ALL = "all four feature sets (E0 can be built for every idle cell under move 6's rules)"
MODE_E_NEW_GATE = "E_new only (move 6's own E0 gate refuses E0)"
MODE_E_NEW_IDLE = "E_new only (E0 cannot be built for any idle cell under move 6's rules)"


class Stop(Exception):
    """A plain refusal with its exit code; main prints the message and returns the code."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def fmt(x) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


# ---------------------------------------------------------------------------------------------
# the run folder (read only), the output folder, the driver's lock
# ---------------------------------------------------------------------------------------------
def default_out(run: Path) -> Path:
    return run.parent / (run.name + OUT_SUFFIX)


def resolve_out(run: Path, out_arg: str | None) -> Path:
    out = Path(os.path.expanduser(out_arg)).resolve() if out_arg else default_out(run)
    if out == run or run in out.parents:
        raise Stop(EXIT_MISSING, f"refused: --out {out} lies inside the run folder {run}, which is read only here; the default is {default_out(run)}")
    return out


def driver_lock_state(run: Path) -> dict:
    """<run>/.driver.lock as run_moves.acquire_lock writes it: the writer's pid, and whether it is alive."""
    p = run / DRIVER_LOCK
    if not p.exists():
        return {"path": str(p), "exists": False, "pid": None, "alive": False, "started_at": None, "argv_tail": None}
    try:
        d = read_json(p)
    except Exception:                                      # noqa: BLE001  an unreadable lock is reported as held, pid unknown
        d = {}
    pid = d.get("pid")
    argv = d.get("argv") or []
    return {"path": str(p), "exists": True, "pid": pid, "alive": bool(pid and _pid_alive(pid)), "started_at": d.get("started_at"),
            "argv_tail": " ".join(str(a) for a in argv[-6:]) if argv else None}


def load_move6(run: Path, enc_override: str | None) -> dict:
    """What move 6 recorded and these steps repeat: the cuts (params.json), the encoding run (D2) and the window (D3) of
    moves/06_classify/classify.json, and its forest settings."""
    if not (run / "params.json").is_file():
        raise Stop(EXIT_MISSING, f"missing input: {run / 'params.json'} (run move 0 first)")
    params = read_json(run / "params.json")
    cj_path = run / "moves" / "06_classify" / "classify.json"
    if not cj_path.is_file():
        raise Stop(EXIT_MISSING, f"missing input: {cj_path} (move 6 has not run; its D3 window and forest settings are read from it)")
    cj = read_json(cj_path)
    p6 = cj.get("params") or {}
    enc_recorded = p6.get("D2_encoding_out") or (params.get("D2_baseline") or {}).get("encoding_out")
    enc_out = Path(os.path.expanduser(enc_override)).resolve() if enc_override else (Path(enc_recorded) if enc_recorded else None)
    win = p6.get("D3_window")
    if not isinstance(win, dict) or not win.get("grid_id"):
        raise Stop(EXIT_MISSING, f"{cj_path}: params.D3_window is missing; the window of move 6 is what these steps must repeat")
    return {"cuts": cuts_of(run), "cut_convention": (params.get("cuts") or {}).get("convention"),
            "encoding_out": enc_out, "encoding_out_recorded": enc_recorded, "encoding_out_overridden": bool(enc_override),
            "win": {"grid_id": win["grid_id"], "W": win.get("W"), "H": win.get("H"), "whole_cell": bool(win.get("whole_cell")), "source": win.get("source")},
            "n_jobs": int(p6.get("n_jobs") or 1), "n_estimators": int(p6.get("n_estimators") or 300), "seed_offset": int(p6.get("seed_offset") or 0),
            "gk0_relabelled_kernels": p6.get("gk0_relabelled_kernels"), "pair_rung_excluded_cells": p6.get("pair_rung_excluded_cells"),
            "e0_status_move6": p6.get("e0_status"), "e0_identity_move6": p6.get("e0_identity"), "classify_json": str(cj_path)}


def parse_cuts(spec: str | None, cuts: dict) -> dict:
    if not spec:
        return dict(cuts)
    names = [s.strip() for s in str(spec).split(",") if s.strip()]
    bad = [n for n in names if n not in cuts]
    if bad:
        raise Stop(EXIT_MISSING, f"--cuts names {bad}; the run's params.json knows {sorted(cuts)} (declared = {cuts['declared']} pairs, measured = {cuts['measured']} pairs)")
    return {n: int(cuts[n]) for n in names}


# ---------------------------------------------------------------------------------------------
# the cells: move 6's kernel runs, then idle
# ---------------------------------------------------------------------------------------------
def cells_positions(run: Path) -> dict:
    with open(run / "cells.csv", newline="") as fh:
        return {r["cell_id"]: i for i, r in enumerate(csv.DictReader(fh))}


def select_runs(run: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Every admissible recording with a complete series (stats.load_runs), split by role; each idle run carries its
    cells.csv row position for the order of ordered_runs_13."""
    runs = load_runs(run)
    pos = cells_positions(run)
    for r in runs:
        r["_cells_pos"] = pos.get(r["cell_id"], 10 ** 9)
    kernel = [r for r in runs if r["role"] == "kernel"]
    idle = sorted((r for r in runs if r["role"] == "idle"), key=lambda r: (r["_cells_pos"], r["cell_id"]))
    return runs, kernel, idle


ORDERED_RUNS_MOVE6 = C.ordered_runs        # move 6's own cell filter, kept by value here: build_encodings_13 rebinds the name in classify for a call


def ordered_runs_13(runs: list[dict], enc_rows: list[dict] | None) -> list[dict]:
    """classify.ordered_runs (classify.py lines 177-184) admitting idle. It differs ONLY in admitting idle: the kernel runs
    are exactly classify.ordered_runs's list (the encoding run's cells.csv order, else this run's cells.csv order); the
    idle runs follow them in this run's cells.csv order, each carrying the class label "idle" in place of the test label
    cells.csv gives it ("sleep"), as plan11's build_features labels idle too (series.py line 651). Nothing else changes."""
    kernel = ORDERED_RUNS_MOVE6(runs, enc_rows)
    idle = sorted((r for r in runs if r["role"] == "idle"), key=lambda r: (r.get("_cells_pos", 10 ** 9), r["cell_id"]))
    return kernel + [{**r, "kernel": IDLE_LABEL} for r in idle]


def build_encodings_13(runs: list[dict], enc_out: Path | None, enc_rows: list[dict] | None, cut: int, win: dict, relabel: dict,
                       pair_rung_excluded: set) -> dict:
    """classify.build_encodings reused as it is (the feature building that must stay identical to move 6's: cell_windows,
    the E0 windows, the per-window statistics, the four column sets, the exclusions), with the one name it reads for its
    cell list, `ordered_runs` (classify.py line 196), bound to ordered_runs_13 for the duration of the call."""
    if "ordered_runs" not in C.build_encodings.__code__.co_names:
        raise Stop(EXIT_ERROR, "classify.build_encodings no longer takes its cell list from the name ordered_runs; idle_class.build_encodings_13 must be revisited before it runs")
    orig = C.ordered_runs
    C.ordered_runs = ordered_runs_13
    try:
        return C.build_encodings(runs, enc_out, enc_rows, cut, win, relabel, pair_rung_excluded)
    finally:
        C.ordered_runs = orig


def move6_cells(run: Path, cut: int) -> list[str] | None:
    """Move 6's cells at this cut, in its order (the unique cell_ids of its features.npz), or None when absent."""
    p = run / "moves" / "06_classify" / f"cut{cut}" / "features.npz"
    if not p.is_file():
        return None
    with np.load(p, allow_pickle=False) as z:
        return list(dict.fromkeys(z["cell_id"].astype(str).tolist()))


# ---------------------------------------------------------------------------------------------
# E0: move 6's gate, then the check for idle
# ---------------------------------------------------------------------------------------------
def e0_gate(run: Path, enc_out: Path | None, cell_ids: list[str]) -> tuple:
    """Move 6's E0 gate, the same calls in the same order (classify.run_classify lines 635-659): the encoding run's rows,
    the declared files with sha256, G-K0's relabel, the pair-rung exclusion, and move 1's identity record (E0 only when it
    passed and the reference extract is unchanged). Returns (enc_rows, declared, relabel, pair_excluded, e0_status, identity)."""
    enc_rows, declared, relabel, pair_excluded, e0_status = None, None, {}, set(), None
    identity = C.identity_record(run)
    if enc_out is not None:
        if not (enc_out / "cells.csv").is_file():
            raise Stop(EXIT_MISSING, f"missing input: {enc_out / 'cells.csv'} (decision D2 names an encoding run without a cells.csv)")
        enc_rows = C.encoding_cells(enc_out)
        declared = C.declare_e0(enc_out, cell_ids)
        relabel = S11.gk0_relabel(enc_out)
        _, _, ex_pair, _ = S11.admissible_cells(enc_out, [r for r in enc_rows if r.get("status", "ok") == "ok"], C.E0_RUNG)
        pair_excluded = set(ex_pair)
        if not identity.get("passed"):
            e0_status = V.not_run(f"E0 identity check not passed in move 1 ({identity.get('status')}{': ' + str(identity.get('reason')) if identity.get('reason') else ''})")
        else:
            ref = (identity.get("reference") or {}).get("sha256")
            now = declared["files"].get(f"extract/{identity.get('cell_id')}/extract.csv")
            if ref and now and ref != now:
                e0_status = V.not_run(f"the encoding run's extract of {identity.get('cell_id')} changed since move 1's identity check (run move 1 again)")
    else:
        e0_status = V.not_run("no encoding run named (decision D2)")
    return enc_rows, declared, relabel, pair_excluded, e0_status, identity


E0_IDLE_RULE = ("move 6's rules for a cell's E0 (classify.build_encodings lines 198-214): status ok in the encoding run's cells.csv; not in the "
                "pair-rung exclusion (all_hard_pass in gates/preconditions.csv and a failed_verdict that is not a refusal, series.admissible_cells with the "
                "combined rung); an ok sidecar.json and an extract.csv under extract/<cell>/; and cell_windows' row identity with this run's series after "
                "the cut (the same row count, pair = seq + 1 on every row)")


def idle_e0_check(idle_runs: list[dict], enc_out: Path | None, enc_rows: list[dict] | None, pair_excluded: set, e0_status: str | None,
                  win: dict, cuts: dict) -> dict:
    """Whether E0 can be built for the idle cells under move 6's own rules, cell by cell and cut by cut."""
    if enc_out is None or e0_status is not None:
        return {"can_build": False, "n_idle": len(idle_runs), "n_idle_with_e0": 0, "reason": e0_status or V.not_run("no encoding run named (decision D2)"),
                "rule": E0_IDLE_RULE, "per_cell": []}
    enc_by_id = {r["cell_id"]: r for r in (enc_rows or [])}
    per_cell = []
    for r in idle_runs:
        cid, reasons, windows = r["cell_id"], [], {}
        ec = enc_by_id.get(cid)
        if ec is None:
            reasons.append("not a cell of the encoding run's cells.csv")
        elif ec.get("status") != "ok":
            reasons.append(f"encoding run status: {ec.get('status')}")
        if cid in pair_excluded:
            reasons.append("the encoding run's pair-rung exclusion (failed_verdict refused; series.admissible_cells)")
        sc = C.sidecar_of(enc_out, cid)
        if sc is None or sc.get("status") != "ok" or not S11.extract_path(enc_out, cid).is_file():
            reasons.append("no ok extract in the encoding run")
        if not reasons:
            ex = S11.load_extract(enc_out, cid)
            for name, cut in cuts.items():
                F, Xs, _, _, why = C.cell_windows(r, ex, cut, win)
                windows[name] = {"cut": int(cut), "ok": why is None, "reason": why,
                                 "n_windows": int(Xs.shape[0]) if Xs is not None else 0, "n_e0_windows": int(F.shape[0]) if F is not None else 0}
                if why is not None:
                    reasons.append(f"cut {cut}: {why}")
        per_cell.append({"cell_id": cid, "can_build": not reasons, "reasons": reasons, "windows": windows})
    n_ok = sum(1 for c in per_cell if c["can_build"])
    if per_cell and n_ok == len(per_cell):
        reason = None
    elif n_ok == 0:
        first = next((c for c in per_cell if c["reasons"]), None)
        reason = V.not_run("E0 cannot be built for any idle cell under move 6's rules" + (f" (first: {first['cell_id']}: {'; '.join(first['reasons'])})" if first else " (no idle cell)"))
    else:
        reason = f"E0 can be built for {n_ok} of {len(per_cell)} idle cells"
    return {"can_build": bool(per_cell) and n_ok == len(per_cell), "n_idle": len(per_cell), "n_idle_with_e0": n_ok, "reason": reason,
            "rule": E0_IDLE_RULE, "per_cell": per_cell}


def decide_mode(idle_runs: list[dict], e0_status: str | None, check: dict) -> tuple:
    """(mode, feature sets to run, the reason the others do not run). The prompt gives two rules: every feature set when E0
    can be built for idle, E_new only when it cannot. A mixed case is not covered: the step stops with the question."""
    if not idle_runs:
        raise Stop(EXIT_MISSING, f"stopped: no idle cell with an admissible series in this run (expected {N_IDLE_EXPECTED} at {IDLE_REC_LAYOUT}): "
                                 "the 13th class is absent, so these steps have nothing to answer")
    if e0_status is not None:
        return MODE_E_NEW_GATE, ("E_new",), e0_status
    if check["can_build"]:
        return MODE_ALL, tuple(ENCODINGS), None
    if check["n_idle_with_e0"] == 0:
        return MODE_E_NEW_IDLE, ("E_new",), check["reason"]
    bad = [c for c in check["per_cell"] if not c["can_build"]]
    raise Stop(EXIT_MISSING, f"stopped: {check['reason']} ({'; '.join(c['cell_id'] + ': ' + '; '.join(c['reasons']) for c in bad)}). Move 6's rule excludes a "
                             "cell without an ok E0 from every feature set (classify.build_encodings lines 198-214), which would leave the 13th class "
                             f"with {check['n_idle_with_e0']} cells; the prompt gives a rule only for all 8 (every feature set) or none (E_new only). "
                             "The question for the author: exclude those idle cells as move 6 would, or run E_new only on all 8? Nothing was written.")


# ---------------------------------------------------------------------------------------------
# the record book of the output folder, the lock, the code and input identities
# ---------------------------------------------------------------------------------------------
def acquire_out_lock(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    lock = out / LOCK_NAME
    if lock.exists():
        try:
            old = read_json(lock)
        except Exception:                                  # noqa: BLE001
            old = {}
        pid = old.get("pid")
        if pid and _pid_alive(pid) and int(pid) != os.getpid():
            raise Stop(EXIT_REFUSED, f"refused: another idle_class step is writing {out} ({lock}: pid {pid}, started {old.get('started_at')}); one writer per output folder")
        print(f"[idle13] stale lock from pid {pid} ({old.get('started_at')}) taken over", flush=True)
    write_json(lock, {"pid": os.getpid(), "started_at": now_iso(), "argv": sys.argv})
    return lock


def release_out_lock(lock: Path) -> None:
    try:
        if lock.exists() and read_json(lock).get("pid") == os.getpid():
            lock.unlink()
    except Exception:                                      # noqa: BLE001
        pass


def load_record(out: Path) -> dict:
    p = out / RECORD_NAME
    if p.is_file():
        try:
            return read_json(p)
        except Exception:                                  # noqa: BLE001  an unreadable record is started afresh, the old file renamed
            p.rename(p.with_suffix(".json.unreadable_" + now_iso().replace(":", "")))
    return {"schema": "plan12.idle_class.record.v1", "citation": CITATION, "package_version": __version__, "entries": []}


def save_record(out: Path, rec: dict) -> None:
    write_json(out / RECORD_NAME, rec)


def last_done(rec: dict, key: str) -> dict | None:
    for e in reversed(rec.get("entries") or []):
        if e.get("key") == key and e.get("status") == "done":
            return e
    return None


def code_identity() -> dict:
    """The sha256 of idle_class.py and of each plan12 module it imports (run_moves.module_closure), and one fingerprint over
    them (run_moves.code_fingerprint, the driver's own per-command rule)."""
    fp = code_fingerprint("idle_class")
    files = toolkit_fingerprint()["files"]
    return {"fingerprint": fp["sha256"], "modules": fp["modules"], "sha256": {f"{m}.py": files.get(f"{m}.py") for m in fp["modules"]}}


def inputs_identity(run: Path, enc_out: Path | None, declared: dict | None, runs: list[dict], cuts: dict, split: str) -> dict:
    """sha256 of every input a step reads: the run's files (cells.csv, params.json, move 1's records, move 6's records and the
    comparison files at each cut, the series of every selected cell) and the encoding run's files (classify.declare_e0)."""
    rels = ["cells.csv", "params.json", "moves/01_extract/extract.json", "moves/01_extract/e0_identity.json", "moves/06_classify/classify.json"]
    for cut in cuts.values():
        rels += [f"moves/06_classify/cut{cut}/scores.csv", f"moves/06_classify/cut{cut}/recall_per_kernel.csv", f"moves/06_classify/cut{cut}/features.npz"]
        rels += [f"moves/06_classify/cut{cut}/predictions_{split}_{e}.csv" for e in ENCODINGS]
    ex = read_json(run / "moves" / "01_extract" / "extract.json") if (run / "moves" / "01_extract" / "extract.json").is_file() else {}
    recs = ex.get("recordings") or {}
    series = {}
    for r in runs:
        p = Path((recs.get(r["cell_id"]) or {}).get("series") or (run / "series" / f"{r['cell_id']}.npz"))
        key = str(p.relative_to(run)) if str(p).startswith(str(run)) else str(p)
        series[key] = sha256_file(p) if p.is_file() else "absent"
    files = {rel: (sha256_file(run / rel) if (run / rel).is_file() else "absent") for rel in rels}
    files.update(series)
    return {"run": str(run), "run_files": files, "encoding_out": str(enc_out) if enc_out else None,
            "encoding_run_files": dict((declared or {}).get("files") or {})}


# ---------------------------------------------------------------------------------------------
# one cut of one step
# ---------------------------------------------------------------------------------------------
def class_order(labels_present) -> list[str]:
    present = set(labels_present)
    return [c for c in CLASS_ORDER if c in present] + sorted(present - set(CLASS_ORDER))


def confusion_of(pred: dict, labels: list[str]) -> np.ndarray:
    idx = {l: i for i, l in enumerate(labels)}
    Cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for v in pred.values():
        if v["y_pred"] and v["y_pred"] in idx and v["y_true"] in idx:
            Cm[idx[v["y_true"]], idx[v["y_pred"]]] += 1
    return Cm


def class_report(pred: dict, cls: str) -> dict:
    """One class of one feature set: its cells, its recall, what it is mistaken for (its wrong predictions) and what is
    taken for it (other classes' cells predicted as it)."""
    mine = {c: v for c, v in pred.items() if v["y_true"] == cls}
    n = len(mine)
    correct = sum(1 for v in mine.values() if v["y_pred"] == cls)
    mistaken: dict = {}
    for v in mine.values():
        if v["y_pred"] != cls:
            mistaken[v["y_pred"]] = mistaken.get(v["y_pred"], 0) + 1
    taken: dict = {}
    for v in pred.values():
        if v["y_true"] != cls and v["y_pred"] == cls:
            taken[v["y_true"]] = taken.get(v["y_true"], 0) + 1
    return {"n_cells": n, "n_correct": correct, "recall": (correct / n) if n else None, "text": (f"{correct} of {n}" if n else "absent (no cell of this class)"),
            "mistaken_for": dict(sorted(mistaken.items())), "taken_for_it": dict(sorted(taken.items())),
            "cells_missed": sorted(c for c, v in mine.items() if v["y_pred"] != cls)}


def confusion_svg_13(labels: list[str], Cm: np.ndarray, enc: str, split: str, cut: int, r: dict) -> str:
    """One feature set's confusion at the unit (cell), rows true, columns predicted, in the engine's style
    (figures.svg_open; the shading of classify.confusion_svg); the lexer and idle rows highlighted."""
    n = len(labels)
    cell, left, top = 22, 124, 100
    W_ = left + n * cell + 40
    H_ = top + n * cell + 70
    out = svg_open(W_, H_, f"Confusion at the unit (cell): {split}, {enc}, {n} classes, cut of {cut} pairs",
                   "rows: the true class; columns: the predicted class; the number of cells; shade: the share of the row; the lexer and idle rows are highlighted")
    px, py = left, top
    for i, l in enumerate(labels):
        if l in HIGHLIGHT:
            out.append(f'<rect x="{px - 116}" y="{py + i * cell}" width="{116 + n * cell}" height="{cell}" fill="#fff3c4" stroke="#d9822b" stroke-width="0.8"/>')
    for j, l in enumerate(labels):
        bold = ' font-weight="bold"' if l in HIGHLIGHT else ""
        out.append(f'<text x="{px + j * cell + cell / 2:.1f}" y="{py - 5}" text-anchor="end" font-size="9"{bold} '
                   f'transform="rotate(-60 {px + j * cell + cell / 2:.1f},{py - 5})">{html.escape(l)}</text>')
    for i, l in enumerate(labels):
        bold = ' font-weight="bold"' if l in HIGHLIGHT else ""
        out.append(f'<text x="{px - 6}" y="{py + i * cell + cell * 0.7:.1f}" text-anchor="end" font-size="9"{bold}>{html.escape(l)}</text>')
        row = int(Cm[i].sum())
        for j in range(n):
            v = int(Cm[i, j])
            share = v / row if row else 0.0
            fill = f"rgb({int(255 - 180 * share)},{int(255 - 120 * share)},{int(255 - 60 * share)})" if v else ("none" if l in HIGHLIGHT else "white")
            out.append(f'<rect x="{px + j * cell}" y="{py + i * cell}" width="{cell}" height="{cell}" fill="{fill}" stroke="#ddd"/>')
            if v:
                out.append(f'<text x="{px + j * cell + cell / 2:.1f}" y="{py + i * cell + cell * 0.7:.1f}" text-anchor="middle" font-size="9">{v}</text>')
    yb = py + n * cell + 16
    acc, mac = r.get("accuracy"), r.get("macro_recall")
    out.append(f'<text x="{px - 116}" y="{yb}" fill="#555" font-size="9">accuracy {fmt(acc)}, macro recall {fmt(mac)}, {r.get("n_units")} cells, '
               f'{n} classes; score: {html.escape(str(r.get("score_source") or ""))[:70]}</text>')
    out.append(f'<text x="{px - 116}" y="{yb + 14}" fill="#555" font-size="9">{html.escape(NO_NULL)}</text>')
    hl = ", ".join(HIGHLIGHT)
    out.append(f'<text x="{px - 116}" y="{yb + 28}" fill="#555" font-size="9">rows and columns in the class order of cells.csv (schema.KERNEL_NAMES), idle last; highlighted: {hl}</text>')
    out.append("</svg>")
    return "\n".join(out)


def move6_comparison(run: Path, cut: int, split: str) -> dict:
    """Move 6's 12-class numbers for this split and cut, read from the run's moves/06_classify/cut<C>/ (scores.csv, the
    predictions per feature set): accuracy, macro recall, n units, and lexer's recall as 'k of n'."""
    d = run / "moves" / "06_classify" / f"cut{cut}"
    res = {"source": str(d), "n_classes": 12, "split": split, "per_feature_set": {}}
    scores = {}
    if (d / "scores.csv").is_file():
        with open(d / "scores.csv", newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("split") == split:
                    scores[row.get("encoding")] = row
    for e in ENCODINGS:
        s = scores.get(e) or {}
        entry = {"status": s.get("status"), "accuracy": S11.to_float(s.get("accuracy")) if s else None,
                 "macro_recall": S11.to_float(s.get("macro_recall")) if s else None, "n_units": s.get("n_units"), "score_source": s.get("score_source"),
                 "predictions_file": None, "lexer": None}
        if entry["accuracy"] is not None and np.isnan(entry["accuracy"]):
            entry["accuracy"] = None
        if entry["macro_recall"] is not None and np.isnan(entry["macro_recall"]):
            entry["macro_recall"] = None
        p = d / f"predictions_{split}_{e}.csv"
        if p.is_file():
            with open(p, newline="") as fh:
                pred = {row["cell_id"]: {"y_true": row.get("y_true"), "y_pred": row.get("y_pred")} for row in csv.DictReader(fh)}
            entry["predictions_file"] = str(p.relative_to(run))
            entry["lexer"] = class_report(pred, "lexer")
        res["per_feature_set"][e] = entry
    return res


def one_cut(step_dir: Path, cut_name: str, cut: int, split: str, runs: list[dict], enc_out: Path | None, enc_rows, win: dict, relabel: dict,
            pair_excluded: set, mode: str, e_sets: tuple, reason: str | None, run: Path, *, n_jobs: int, n_est: int, seed: int) -> dict:
    d = step_dir / f"cut{cut}"
    d.mkdir(parents=True, exist_ok=True)
    m6_order = move6_cells(run, cut)
    if mode == MODE_E_NEW_IDLE:
        # E0 exists for no idle cell: the features come from the series alone (no encoding run named to build_encodings, so no
        # cell is excluded for its extract); the kernel cells are exactly move 6's cells at this cut, in its order
        keep = set(m6_order or [r["cell_id"] for r in runs if r["role"] == "kernel"])
        enc = build_encodings_13([r for r in runs if r["role"] == "idle" or r["cell_id"] in keep], None, enc_rows, cut, win, relabel, set())
    else:
        enc = build_encodings_13(runs, enc_out, enc_rows, cut, win, relabel, pair_excluded)
    X, names, lab = enc["X"], enc["names"], enc["lab"]
    cells = list(dict.fromkeys(lab["cell_id"].tolist()))
    label_of = {}
    for c in cells:
        label_of[c] = str(lab["kernel"][int(np.flatnonzero(lab["cell_id"] == c)[0])])
    kernel_cells = [c for c in cells if label_of[c] != IDLE_LABEL]
    idle_cells = [c for c in cells if label_of[c] == IDLE_LABEL]
    order_check = {"move6_features_npz": str(run / "moves" / "06_classify" / f"cut{cut}" / "features.npz"),
                   "kernel_order_matches_move6": (kernel_cells == m6_order) if m6_order is not None else None,
                   "n_kernel_cells_move6": len(m6_order) if m6_order is not None else None}
    np.savez_compressed(d / "features.npz", X=X, feature_names=np.array(names, dtype=str), cell_id=lab["cell_id"], kernel=lab["kernel"],
                        archetype=lab["archetype"], rep=lab["rep"], win_start=lab["win_start"], pair_start=np.array(enc["meta"]["pair_start"], dtype=np.int64),
                        campaign=lab["campaign"], cols_json=np.array(json.dumps(enc["cols"])), W=np.int64(win["W"] or 0), H=np.int64(win["H"] or 0),
                        grid_id=np.array(win["grid_id"]), cut=np.int64(cut))
    write_csv(d / "excluded_cells.csv", ["cell_id", "reason"], [[e["cell_id"], e["reason"]] for e in enc["excluded"]])
    labels = class_order(label_of.values())
    results: dict = {}
    for e in ENCODINGS:
        if e not in e_sets:
            results[e] = {"split": split, "labelspace": "kernel", "status": reason, "accuracy": None, "macro_recall": None, "n_units": 0, "n_windows": int(lab["n"])}
            print(f"[idle13] cut {cut} {e:<5} {split:<13} {str(reason)[:80]}", flush=True)
            continue
        cols = enc["cols"][e]
        t0 = time.time()
        r = C.run_split(X[:, cols], [names[j] for j in cols], lab, split, perms=[], run_null=False, seed=seed, n_jobs=n_jobs, n_est=n_est)
        r["elapsed_s"] = round(time.time() - t0, 1)
        if str(r.get("status", "")).startswith("ok"):
            r["status"] = "ok"                              # plan11's "null not run (--null-splits)" wording: the null column says it here
        results[e] = r
        print(f"[idle13] cut {cut} {e:<5} {split:<13} {str(r.get('status'))[:36]:<36} acc {fmt(r.get('accuracy'))} macro {fmt(r.get('macro_recall'))}"
              f" n {r.get('n_units')}{' quarantined ' + str(len(r['quarantined_features'])) if r.get('quarantined_features') else ''} ({r['elapsed_s']} s)", flush=True)
    # tables
    write_csv(d / "scores.csv", ["feature_set", "status", "accuracy", "macro_recall", "majority", "n_units", "n_windows", "n_classes", "feature_count",
                                 "feature_count_used", "score_source", "quarantined_features", "n_folds", "null"],
              [[e, r.get("status"), r.get("accuracy"), r.get("macro_recall"), r.get("majority"), r.get("n_units"), r.get("n_windows"), len(labels),
                r.get("feature_count"), r.get("feature_count_used"), r.get("score_source"), ";".join(r.get("quarantined_features") or []), r.get("n_folds"), NO_NULL]
               for e, r in results.items()])
    n_cells_of = {l: sum(1 for c in cells if label_of[c] == l) for l in labels}
    write_csv(d / "recall_per_class.csv", ["class", *ENCODINGS, "n_cells"],
              [[l] + [(results[e].get("recall_per_class") or {}).get(l) for e in ENCODINGS] + [n_cells_of[l]] for l in labels])
    confusions = {}
    for e, r in results.items():
        if not r.get("predictions"):
            continue
        Cm = confusion_of(r["predictions"], labels)
        confusions[e] = Cm
        write_csv(d / f"confusion_{e}.csv", ["true\\predicted"] + labels, [[l] + Cm[i].tolist() for i, l in enumerate(labels)])
        (d / f"confusion_{e}.svg").write_text(confusion_svg_13(labels, Cm, e, split, cut, r))
        write_csv(d / f"predictions_{e}.csv", ["cell_id", "class", "archetype", "rep", "fold", "y_true", "y_pred", "n_windows", "vote_fraction"],
                  [[c, v["kernel"], v["archetype"], v["rep"], v["fold"], v["y_true"], v["y_pred"], v["n_windows"], v["vote_fraction"]] for c, v in r["predictions"].items()])
        if r.get("predictions_full"):
            write_csv(d / f"predictions_full_model_{e}.csv", ["cell_id", "y_true", "y_pred"], [[c, v["y_true"], v["y_pred"]] for c, v in r["predictions_full"].items()])
    margin_rows = []
    for b, a in MARGIN_PAIRS:
        rb, ra = results[b], results[a]
        if ra.get("accuracy") is None or rb.get("accuracy") is None:
            margin_rows.append([f"{b}-{a}", None, None, None, None, "not run: " + str(ra.get("status") if ra.get("accuracy") is None else rb.get("status"))])
            continue
        pa, pb = ra["predictions"], rb["predictions"]
        gained = sum(1 for c in pb if c in pa and pb[c]["y_pred"] == pb[c]["y_true"] and pa[c]["y_pred"] != pa[c]["y_true"])
        lost = sum(1 for c in pb if c in pa and pb[c]["y_pred"] != pb[c]["y_true"] and pa[c]["y_pred"] == pa[c]["y_true"])
        dmac = (rb["macro_recall"] - ra["macro_recall"]) if (ra.get("macro_recall") is not None and rb.get("macro_recall") is not None) else None
        margin_rows.append([f"{b}-{a}", rb["accuracy"] - ra["accuracy"], dmac, gained, lost, "ok; " + NO_NULL])
    write_csv(d / "margins.csv", ["comparison", "delta_accuracy", "delta_macro_recall", "n_cells_gained", "n_cells_lost", "status"], margin_rows)
    # the summary: lexer and idle, and move 6's 12-class row beside each feature set
    m6 = move6_comparison(run, cut, split)
    focus = {cls: {e: (class_report(results[e]["predictions"], cls) if results[e].get("predictions") else {"status": results[e].get("status")}) for e in ENCODINGS}
             for cls in HIGHLIGHT}
    comparison = {}
    for e in ENCODINGS:
        r, t = results[e], (m6["per_feature_set"].get(e) or {})
        lx13 = (focus["lexer"].get(e) or {}).get("recall")
        lx12 = ((t.get("lexer") or {}).get("recall"))
        comparison[e] = {"accuracy_13": r.get("accuracy"), "accuracy_12_move6": t.get("accuracy"),
                         "accuracy_13_minus_12": (r["accuracy"] - t["accuracy"]) if (r.get("accuracy") is not None and t.get("accuracy") is not None) else None,
                         "macro_recall_13": r.get("macro_recall"), "macro_recall_12_move6": t.get("macro_recall"),
                         "lexer_recall_13": lx13, "lexer_13_text": (focus["lexer"].get(e) or {}).get("text"),
                         "lexer_recall_12_move6": lx12, "lexer_12_text": (t.get("lexer") or {}).get("text"),
                         "lexer_recall_13_minus_12": (lx13 - lx12) if (lx13 is not None and lx12 is not None) else None,
                         "idle_recall_13": (focus[IDLE_LABEL].get(e) or {}).get("recall"), "idle_13_text": (focus[IDLE_LABEL].get(e) or {}).get("text"),
                         "n_units_13": r.get("n_units"), "n_units_12_move6": t.get("n_units")}
    summary = {"schema": "plan12.idle_class_cut.v1", "citation": CITATION, "split": split, "cut": int(cut), "cut_name": cut_name,
               "mode": mode, "feature_sets_run": list(e_sets), "reason_others_not_run": reason, "null": NO_NULL,
               "n_cells": len(cells), "n_kernel_cells": len(kernel_cells), "n_idle_cells": len(idle_cells), "n_windows": int(lab["n"]),
               "n_classes": len(labels), "classes": labels, "n_cells_per_class": n_cells_of,
               "class_order": "the run's cells.csv order (schema.KERNEL_NAMES), idle last; the same order in every confusion table here",
               "cell_order": "move 6's kernel cells in its order (classify.ordered_runs), then the idle cells in this run's cells.csv order",
               "order_check": order_check, "idle_cells": idle_cells, "n_excluded": len(enc["excluded"]), "excluded": enc["excluded"],
               "e0_available": enc["e0_available"], "n_e0_windows": enc["n_e0_windows"], "feature_counts": {e: len(enc["cols"][e]) for e in ENCODINGS},
               "window": win, "n_estimators": n_est, "n_jobs": n_jobs, "seed_forest": seed,
               "scores": {e: {k: results[e].get(k) for k in ("status", "accuracy", "macro_recall", "majority", "n_units", "n_windows", "feature_count",
                                                            "feature_count_used", "score_source", "quarantined_features", "n_folds", "elapsed_s")} for e in ENCODINGS},
               "lexer": focus["lexer"], "idle": focus[IDLE_LABEL],
               "margins": [dict(zip(("comparison", "delta_accuracy", "delta_macro_recall", "n_cells_gained", "n_cells_lost", "status"), r)) for r in margin_rows],
               "move6_12_classes": m6, "comparison_with_move6": comparison,
               "files": sorted(p.name for p in d.iterdir())}
    write_json(d / "summary.json", summary)
    return summary


# ---------------------------------------------------------------------------------------------
# the step
# ---------------------------------------------------------------------------------------------
def run_step(o: argparse.Namespace) -> int:
    run = Path(os.path.expanduser(o.run)).resolve()
    if not (run / "cells.csv").is_file():
        raise Stop(EXIT_MISSING, f"missing input: {run / 'cells.csv'} (--run must name a plan12_grounding output folder with moves 0, 1 and 6 done)")
    step = int(o.step)
    step_name, split = STEPS[step]
    out = resolve_out(run, o.out)
    lock = driver_lock_state(run)
    if lock["alive"] and not o.dry_run:
        raise Stop(EXIT_REFUSED, f"refused: a move is running on this run ({lock['path']}: pid {lock['pid']}, started {lock['started_at']}, "
                                 f"{lock['argv_tail']}); these steps never compete with a running move. Run again after it ends; --dry-run shows the plan meanwhile.")
    m6 = load_move6(run, o.encoding_out)
    cuts = parse_cuts(o.cuts, m6["cuts"])
    n_jobs = int(o.n_jobs) if o.n_jobs is not None else m6["n_jobs"]
    n_est = int(o.n_estimators) if o.n_estimators is not None else m6["n_estimators"]
    seed = SEED_FOREST + m6["seed_offset"]
    enc_out = m6["encoding_out"]
    win = m6["win"]
    runs, kernel_runs, idle_runs = select_runs(run)
    cell_ids = [r["cell_id"] for r in kernel_runs] + [r["cell_id"] for r in idle_runs]
    enc_rows, declared, relabel, pair_excluded, e0_status, identity = e0_gate(run, enc_out, cell_ids)
    check = idle_e0_check(idle_runs, enc_out, enc_rows, pair_excluded, e0_status, win, cuts)
    mode, e_sets, reason = decide_mode(idle_runs, e0_status, check)
    n_classes = len({r["kernel"] for r in kernel_runs}) + 1
    # the plan, in plain words
    print(f"[idle13] step {step}: {split} with {n_classes} classes (the kernels present plus idle), feature sets {', '.join(e_sets)}; {NO_NULL}")
    print(f"[idle13] run {run} (read only); out {out}{' (exists)' if out.exists() else ' (to be created)'}")
    print(f"[idle13] driver lock: " + (f"held by pid {lock['pid']} ({'alive' if lock['alive'] else 'dead'}), started {lock['started_at']}: {lock['argv_tail']}" if lock["exists"] else "absent")
          + ("; the refusal is skipped by --dry-run" if (lock["alive"] and o.dry_run) else ""))
    print(f"[idle13] cells found: {len(kernel_runs) + len(idle_runs)} = {len(kernel_runs)} kernel ({len({r['kernel'] for r in kernel_runs})} kernels) + {len(idle_runs)} idle "
          f"(the real run expects 103 = 95 kernel + {N_IDLE_EXPECTED} idle); idle = {IDLE_REC_LAYOUT}, label {IDLE_LABEL!r}, archetype {IDLE_ARCHETYPE!r}")
    print(f"[idle13] cuts: {', '.join(f'{k} = {v} pairs' for k, v in cuts.items())} (params.json)")
    print(f"[idle13] D2 encoding run: {enc_out}{' (overrides ' + str(m6['encoding_out_recorded']) + ')' if m6['encoding_out_overridden'] else ' (move 6 classify.json)'}; "
          f"exists {bool(enc_out and enc_out.is_dir())}; G-K0 relabel {sorted(relabel)}; pair-rung excluded {len(pair_excluded)}")
    print(f"[idle13] D3 window: {win['grid_id']} (W {win['W']}, H {win['H']}; {win['source']}); forest: {n_est} trees, {n_jobs} processes, seed {seed} (move 6's values unless overridden)")
    print(f"[idle13] move 6's E0 gate: {'E0 usable (identity check passed on ' + str(identity.get('cell_id')) + ')' if e0_status is None else e0_status}")
    e0_idle_text = ("can be built for all " + str(check["n_idle"]) + " idle cells") if check["can_build"] else str(check["reason"])
    if check["can_build"] and check["per_cell"]:
        w0 = check["per_cell"][0]["windows"] or {}
        e0_idle_text += " (E0 windows of the first idle cell: " + ", ".join(f"cut {w['cut']}: {w['n_e0_windows']}" for w in w0.values()) + ")"
    print(f"[idle13] E0 for idle: {e0_idle_text}")
    for c in check["per_cell"]:
        if not c["can_build"]:
            print(f"[idle13]   {c['cell_id']}: {'; '.join(c['reasons'])}")
    print(f"[idle13] mode: {mode}")
    if o.dry_run:
        print(f"[idle13] dry run: would write {out / step_name}/cut<C>/ (scores.csv, recall_per_class.csv, confusion_<E>.csv and .svg, margins.csv, predictions_<E>.csv, "
              f"summary.json), {out / step_name / 'step.json'}, {out / step_name / 'e0_idle_check.json'} and {out / RECORD_NAME}; nothing written")
        return EXIT_OK
    # the output folder: one writer, the record book, the skip rule
    out_lock = acquire_out_lock(out)
    code = code_identity()
    entry = {"key": step_name, "step": step, "split": split, "command": sys.argv, "cwd": str(_HERE.parent), "started_at": now_iso(), "status": "running",
             "exit_code": None, "finished_at": None,
             "params": {"run": str(run), "out": str(out), "cuts": cuts, "encoding_out": str(enc_out) if enc_out else None, "window": win, "n_estimators": n_est,
                        "n_jobs": n_jobs, "seed_forest": seed, "seed_offset": m6["seed_offset"], "mode": mode, "feature_sets": list(e_sets), "reason_others_not_run": reason,
                        "n_kernel_cells": len(kernel_runs), "n_idle_cells": len(idle_runs), "force": bool(o.force)},
             "inputs_sha256": inputs_identity(run, enc_out, declared, kernel_runs + idle_runs, cuts, split),
             "code": {"idle_class.py": code["sha256"].get("idle_class.py"), "fingerprint": code["fingerprint"], "modules": code["modules"], "sha256": code["sha256"],
                      "toolkit_fingerprint": toolkit_fingerprint()["sha256"]},
             "outputs": []}
    sig = {k: v for k, v in entry["params"].items() if k not in ("n_jobs", "force")}
    rec = load_record(out)
    prev = last_done(rec, step_name)
    step_dir = out / step_name
    outputs_exist = (step_dir / "step.json").is_file() and all((step_dir / f"cut{c}" / "summary.json").is_file() for c in cuts.values())
    try:
        if prev is not None and not o.force and outputs_exist:
            why = None
            if prev.get("inputs_sha256") != entry["inputs_sha256"]:
                why = "an input changed"
            elif {k: v for k, v in (prev.get("params") or {}).items() if k not in ("n_jobs", "force")} != sig:
                why = "the parameters changed"
            elif (prev.get("code") or {}).get("fingerprint") != code["fingerprint"]:
                why = "the code changed"
            if why is None:
                entry.update(status="skipped: outputs exist and inputs, parameters and code unchanged", exit_code=0, finished_at=now_iso(),
                             made_by={"finished_at": prev.get("finished_at"), "code_fingerprint": (prev.get("code") or {}).get("fingerprint")}, outputs=prev.get("outputs") or [])
                rec["entries"].append(entry)
                save_record(out, rec)
                print(f"[idle13] skipped: {step_dir} stands as made on {prev.get('finished_at')} (inputs, parameters and code unchanged; --force re-runs)")
                return EXIT_OK
            print(f"[idle13] re-running: {why} since {prev.get('finished_at')}")
        rec["entries"].append(entry)
        save_record(out, rec)
        step_dir.mkdir(parents=True, exist_ok=True)
        write_json(step_dir / "e0_idle_check.json", {"schema": "plan12.idle_class_e0_check.v1", "citation": CITATION, "written_at": now_iso(), "encoding_out": str(enc_out) if enc_out else None,
                                                     "move6_e0_gate": e0_status or "E0 usable", "identity": {k: identity.get(k) for k in ("cell_id", "passed", "status", "reason", "checked_at")},
                                                     "e0_declared_files": (declared or {}).get("n_files"), **check})
        t0 = time.time()
        results = [one_cut(step_dir, name, cut, split, runs, enc_out, enc_rows, win, relabel, pair_excluded, mode, e_sets, reason, run,
                           n_jobs=n_jobs, n_est=n_est, seed=seed) for name, cut in cuts.items()]
        step_rec = {"schema": "plan12.idle_class_step.v1", "citation": CITATION, "package_version": __version__, "written_at": now_iso(), "command": sys.argv,
                    "step": step, "split": split, "question": __doc__.split("THE QUESTION THE STEPS ANSWER", 1)[1].split("THE DATA", 1)[0].strip(),
                    "params": entry["params"], "cut_convention": m6["cut_convention"], "classes": results[0]["classes"] if results else None,
                    "cells": {"n_kernel": len(kernel_runs), "n_idle": len(idle_runs), "kernels": sorted({r["kernel"] for r in kernel_runs}),
                              "idle_cells": [r["cell_id"] for r in idle_runs], "idle_label": IDLE_LABEL, "idle_archetype": IDLE_ARCHETYPE,
                              "order": "move 6's kernel cells in its order, then the idle cells in this run's cells.csv order"},
                    "e0": {"move6_gate": e0_status or "E0 usable", "idle": {k: check[k] for k in ("can_build", "n_idle", "n_idle_with_e0", "reason")}},
                    "mode": mode, "feature_sets_run": list(e_sets), "null": NO_NULL, "code": entry["code"], "elapsed_s": round(time.time() - t0, 1),
                    "results": results}
        write_json(step_dir / "step.json", step_rec)
        entry["outputs"] = sorted(str(p.relative_to(out)) for p in step_dir.rglob("*") if p.is_file())
        entry.update(status="done", exit_code=0, finished_at=now_iso(), elapsed_s=step_rec["elapsed_s"])
        for s in results:
            comp = s["comparison_with_move6"]
            line = "; ".join(f"{e}: acc {fmt(s['scores'][e]['accuracy'])} (move 6 {fmt(comp[e]['accuracy_12_move6'])}), lexer {s['lexer'][e].get('text', '-')} "
                             f"(move 6 {comp[e]['lexer_12_text'] or '-'}), idle {s['idle'][e].get('text', '-')}" for e in e_sets)
            print(f"[idle13] cut {s['cut']} ({s['cut_name']}): {s['n_cells']} cells ({s['n_kernel_cells']} kernel + {s['n_idle_cells']} idle), {s['n_windows']} windows, "
                  f"{s['n_classes']} classes; {line}")
            for cls in HIGHLIGHT:
                for e in e_sets:
                    f = s[cls][e]
                    if f.get("mistaken_for"):
                        print(f"[idle13]   cut {s['cut']} {e}: {cls} mistaken for {f['mistaken_for']}; taken for it: {f['taken_for_it'] or 'none'}")
        print(f"[idle13] step {step} done in {step_rec['elapsed_s']} s: {step_dir}")
        return EXIT_OK
    except BaseException as exc:                           # noqa: BLE001  the record says how the step ended; the exception continues to main
        code_ = getattr(exc, "code", None)
        entry.update(status=f"failed: {type(exc).__name__}" + (f" (exit {code_})" if isinstance(code_, int) else ""), exit_code=code_ if isinstance(code_, int) else EXIT_ERROR,
                     finished_at=now_iso(), error=str(exc)[:500])
        raise
    finally:
        if entry["status"] == "running":
            entry.update(status="failed: stopped before the end", finished_at=now_iso())
        rec = load_record(out)
        rec["entries"] = [e for e in rec["entries"] if not (e.get("key") == entry["key"] and e.get("started_at") == entry["started_at"])] + [entry]
        save_record(out, rec)
        release_out_lock(out_lock)


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------
def add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run", required=True, help="the main run's output folder (read only here)")
    p.add_argument("--step", required=True, type=int, choices=sorted(STEPS), help="1 = LORO with 13 classes; 2 = within-trace with 13 classes")
    p.add_argument("--out", default=None, help="the output folder (default: a sibling of the run, <run>_idle13; never inside the run)")
    p.add_argument("--encoding-out", default=None, help="decision D2 (default: the encoding run move 6 recorded in classify.json)")
    p.add_argument("--cuts", default=None, help="which cuts, by name: declared,measured (default both; the values come from params.json)")
    p.add_argument("--n-jobs", type=int, default=None, help="processes for the forest (default: move 6's recorded value)")
    p.add_argument("--n-estimators", type=int, default=None, help="trees (default: move 6's recorded value, Table 2's 300)")
    p.add_argument("--force", action="store_true", help="re-run a finished step")
    p.add_argument("--dry-run", action="store_true", help="print what would be done; write nothing; skip the live-lock refusal")


def main(argv: list[str] | None = None) -> int:
    install_sigterm()
    ap = argparse.ArgumentParser(prog="plan12_grounding.idle_class", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    add_run_args(sub.add_parser("run", help="step 1 (LORO) or step 2 (within-trace) with idle as the 13th class"))
    o = ap.parse_args(argv)
    try:
        return run_step(o)
    except Stop as exc:
        print(str(exc), file=sys.stderr)
        return exc.code
    except FileNotFoundError as exc:
        print(f"missing input: {exc}", file=sys.stderr)
        return EXIT_MISSING
    except SystemExit as exc:
        if exc.code not in (None, 0):
            print(f"stopped: exit {exc.code}", file=sys.stderr)
            return int(exc.code) if isinstance(exc.code, int) else EXIT_ERROR
        return EXIT_OK
    except Exception:
        traceback.print_exc()
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
