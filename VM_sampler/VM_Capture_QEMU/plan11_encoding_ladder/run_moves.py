#!/usr/bin/env python3
"""run_moves.py -- the driver: al-Kindi's blind moves in order over an extract directory,
resumable, every command recorded in a ledger (SPEC section 7; P2 Sec. 5; K2 Sec. 5).

Builder 3 (report), 2026-09-16. The build brief names this file `run_moves.py`; SPEC section 1
names it `driver.py`, so `driver.py` is a one-line alias of this module. Every move is a
subprocess invocation of the owning module's CLI (`python3 -m plan11_encoding_ladder.<module>
...` from the package's parent directory; never an import of another builder's functions
beyond `schema`/`verdicts`), with its command line, start, end and exit code recorded in
`<out>/driver_state.json`; the driver skips a move whose outputs exist unless `--force` and
stops at the first non-zero exit (SPEC section 7).

Corrections from the SPEC reviews implemented in the move table:
  - al-Kindi 6: `gates_calibration gp` runs at move 2 (it needs only the pass table and
    `n_pairs`); move 6 keeps `alias`.
  - al-Kindi 5: `gates_calibration alias` runs again at the end of move 7 (idempotent; the
    `table6_feature` rows are appended once Table 6's features exist).
  - al-Kindi 8: `gates_comparison gx --rung <rung>` runs for every rung (moves 7, 9 to 12).
  - al-Farabi 2.2: the first command of every temporal stage is
    `series features --all-grid --both` (norm only for `combined`), and the driver refuses the
    split stage of a rung unless all 13 `temporal_per_kernel.csv` and the 13 feature npz files
    exist (`gates/grid_complete.json` records the check).
  - al-Farabi 2.6: `gates_precondition gf --all-rungs` runs inside move 12 before `tables`;
    move 13 is the check that every Table 7 row carries its G-F (i) verdict
    (`gates/gf_check.json`).
  - al-Farabi 2.8: every command's `inputs_sha256` (the author's `inputs/*` and `cells.csv`)
    is recorded in the ledger; the skip test compares them with the current files and on any
    difference writes `stale: <file> changed since move <n>` and re-runs the move. `--force`
    remains the unconditional re-run. The author's input templates (`inputs/*`) are never
    overwritten once they exist (`kept: author input exists`).
  - SPEC 8 item 37: the driver never assumes a zero failed count without
    `--assume-failed-zero --assume-reason "<text>"`; the runbook gives the AA A5 text.

Build epoch 2 (SPEC_epoch2 sections 4 and 5; builder 3):
  - Keyed staleness (E1 sec. 4 M2; CERTIFY_al_farabi.md 7.3): an input spec may be a plain path,
    `json:<path>:<key>` or `csv:<path>:<col>=<value>`; `_input_hash` hashes exactly the part the
    command reads and the ledger keys `inputs_sha256` by the spec string. `splits <rung>` and
    `gx <rung>` declare `json:gates/selection.json:<rung>`, so a resume with nothing changed
    re-runs no split stage; every other command keeps its whole-file inputs. The invariant: a
    command is skipped only when a `done` record exists for its key, every declared output
    exists, and every declared input's keyed hash equals the recorded one. The first resume of a
    ledger written before epoch 2 re-runs the split stages once (the recorded key changed from
    the absolute path to the spec string). A command whose arguments differ from its `done`
    record's, `--n-jobs` / `--jobs` aside, is `stale: arguments changed` (al-Farabi's review of
    SPEC_epoch2, section 6 item 3), so a changed `--c1-rule` or `--null-perm` re-runs the step.
  - C1's rule (AD 2026-09-17; section 4): `--c1-rule`, `--c1-abs-fraction`, `--c1-idle-percentile`,
    `--c1-activity-min` are passed to `preconditions` only when the author sets them; the
    preconditions command's own default is `auto` and it records the rule in force regardless.
  - `gdim` carries `--null-splits` (the matched comparison runs for every Table 7 split; 3.5.1).
  - Builder B's items of SPEC_epoch2.md Part 2: the two `alias` steps read the APF rows of
    `g3_flags.csv` (`csv:gates/g3_flags.csv:rung=apf`, B2); the internal step `gl2-rerun` at move 7
    runs SPEC 3.7.4's feature-drop re-run into `gates/splits_gl2drop/` after a G-L (ii) refusal
    (`--gl2-rerun auto|manual`, B10); `--duration-s` reaches `extract all` (B12), `--wapf-norm`
    reaches `series features` and `gates_temporal grid` (B19), `--n-estimators` reaches the
    `NEST_COMMANDS` (B25), `--gord-n-order-perm` / `--gord-null-perm` reach `gord` (B26) and
    `--c1-activity-min-pages` reaches `preconditions` (B1); each only when it departs from the
    module's own default, so a default run's argv is stable across resumes.

CLI (SPEC 7.1, extended):
  run_moves.py run    --out O [--root R] [--cells-csv CSV] [--moves 0-13]
                      [--assume-failed-zero --assume-reason TEXT] [--n-jobs 1] [--null-perm 500]
                      [--null-splits loko,loro,within_trace] [--seed-offset 0] [--force]
                      [--dry-run] [--skip-missing-modules] [--only-modules M,...] [--persist-side t|t+1]
                      [--failed-counts CSV] [--table8-rung combined] [--piano-cell ID]
                      [--piano-stride 16] [--standalone-tex PATH]
                      [--c1-rule auto|idle_floor|absolute|legacy_apf_max] [--c1-abs-fraction 0.001]
                      [--c1-idle-percentile 95] [--c1-activity-min 0.02]
  run_moves.py status --out O
  run_moves.py plan   --out O [--moves 0-13] ...   (print the command list without running)
Exit 0 on success, 2 when a required input is missing (its path on stderr), 1 when a move
fails (the failing command and its exit code are in the ledger).
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import argparse
import json
import subprocess
import time
import traceback

from plan11_encoding_ladder._report_common import (  # noqa: E402
    GF_DEFAULT_GRID, GRID_IDS, PACKAGE_NAME, PACKAGE_VERSION, RUNGS, inputs_sha256, now_iso,
    read_csv, read_json, refused, result_json, write_json,
)

CITATION = "P2 Sec. 5 (the blind-move order); K2 Sec. 5; SPEC section 7; SPEC reviews al-Kindi 5, 6, 8 and al-Farabi 2.2, 2.6, 2.8"
LEDGER = "driver_state.json"
RANDOM_COMMANDS = {("gates_precondition", "gf"), ("gates_temporal", "grid"), ("gates_temporal", "g3"),
                   ("gates_temporal", "gord"), ("models", "splits"), ("models", "cluster"),
                   ("gates_comparison", "gx"), ("gates_comparison", "gm"), ("gates_comparison", "gdim"), ("gates_readings", "gdec"),
                   ("gates_idle_common_ground", "run"), ("new_blocks", "run")}
NJOBS_COMMANDS = {("gates_precondition", "gf"), ("gates_temporal", "grid"), ("gates_temporal", "gord"),
                  ("models", "splits"), ("gates_comparison", "gx"), ("gates_comparison", "gdim"), ("gates_comparison", "gm"),
                  ("gates_idle_common_ground", "run"), ("new_blocks", "run")}
# SPEC_epoch2 B25 (E1 6.61): the commands that take --n-estimators; the driver passes its value to every one of them
NEST_COMMANDS = {("gates_temporal", "gord"), ("models", "splits"), ("gates_precondition", "gf"),
                 ("gates_comparison", "gx"), ("gates_comparison", "gdim"), ("gates_comparison", "gm"),
                 ("gates_idle_common_ground", "run"), ("new_blocks", "run")}
# SPEC_epoch2 B10 (CHECK_3 M10; CERT 6.9; SPEC 3.7.4): the G-L (ii) consequence the driver runs after `gl` at move 7
GL2_FEATURE_DROP = "cov,std,peak2med"
GL2_BASE_DIR = "splits_gl2drop"
GL2_SHOT_NOISE = "refused: shot noise explains CV"      # verdicts.GL_SHOT_NOISE, spelled here so the driver imports only _report_common
# al-Farabi certification (cycle 2) 7.1: the admissibility record is a declared input of every step
# from move 3 on, so a re-run of the preconditions makes them stale. The record is
# gates/preconditions.json (excluded_cells, excluded_cells_pair_rungs, written by move 2 only), not
# preconditions.csv: the CSV's C7 column is refreshed in place after `select apf`
# (gates_temporal._refresh_c7), which would mark every step that ran before move 6 stale on the
# first resume although the admissible set did not change.
ADMISSIBILITY = "gates/preconditions.json"
INPUT_FILES = ("cells.csv", "inputs/pass_table.csv", "inputs/gk0_source.csv", "inputs/head_drop.csv",
               "inputs/failed_counts.csv", "inputs/idle_admissibility.json", "inputs/cell_order.csv")
MAX_MOVE = 17      # epoch 2: move 14 = the comparators (SPEC_epoch2.md Part 1.7); move 15 = the optional LORO
                   # luck checks (SPEC_epoch2 Part 4 item 29); move 16 = the corrected instrument check, the idle
                   # common-ground test and the second Table 2 (added 2026-10-05, after the run of 2026-09-29;
                   # SPEC_epoch2 Part 4 item 30; A20 to A22); move 17 = the new-block test on the five readings
                   # (added 2026-10-06; SPEC_epoch2 Part 4 item 31; A25). None of 15, 16, 17 is in the default --moves 0-14.
NEW_BLOCKS_PRIMARY_GRID = "W64_H32"      # move 17's primary window (A25); the own windows come from gates/selection.json
LORO_FULL_NULL_SPLITS = "loko,loro,within_trace"   # the paper value; move 15 restores LORO's null under it


def parse_moves(spec: str, max_move: int = MAX_MOVE) -> list[int]:
    """`0-13`, `6,7`, `2-4,12` -> a sorted list of move numbers. `max_move` (SPEC_DETECTION 1.3,
    an additive extension; default 13 keeps the epoch-1 behaviour) is the last valid move."""
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


def _cmd(move: int, name: str, module: str, sub: str | None, args: list, *, outputs=(), inputs=(),
         template: bool = False, internal: bool = False) -> dict:
    return {"move": move, "name": name, "module": module, "sub": sub, "args": [str(a) for a in args],
            "outputs": list(outputs), "inputs": list(inputs), "template": template, "internal": internal}


def build_plan(o: argparse.Namespace) -> list[dict]:
    """The move table (SPEC section 7 with the review corrections in the module docstring)."""
    out = str(Path(o.out))
    O = ["--out", out]
    cells = o.cells_csv or str(Path(o.out) / "cells.csv")
    P = []
    # ---- move 0: the cell index (pip install is a runbook instruction, not a driver command)
    idx_args = ["--root", o.root or "<root required>", *O]
    # AA 2026-09-28 (SPEC_epoch2 Part 4 item 28): the declared seed map, passed as given and declared
    # as an input by its resolved path, so its sha256 is in the ledger and a change re-runs move 0
    m0_inputs = []
    if getattr(o, "seed_map", None):
        idx_args += ["--seed-map", o.seed_map]
        smp = Path(o.seed_map)
        m0_inputs.append(str(smp if smp.is_absolute() else (_HERE.parent / smp).resolve()))
    P.append(_cmd(0, "extract index", "extract", "index", idx_args, outputs=["cells.csv"], inputs=m0_inputs))
    # ---- move 1: the per-cell extracts
    a = ["--cells-csv", cells, *O, "--jobs", o.n_jobs]
    if o.persist_side:
        a += ["--persist-side", o.persist_side]
    if o.failed_counts:
        a += ["--failed-counts", o.failed_counts]
    # AA A8 (2026-09-28; SPEC_epoch2 Part 4 item 26): the declared "keep only the first N pairs" file,
    # passed as given so a shell launch and a console launch build the same argv; declared as an input
    # by its resolved path so its sha256 is in the ledger and a change makes move 1 stale
    m1_inputs = ["cells.csv", "inputs/failed_counts.csv"]
    if getattr(o, "keep_first_pairs", None):
        a += ["--keep-first-pairs", o.keep_first_pairs]
        kfp = Path(o.keep_first_pairs)
        m1_inputs.append(str(kfp if kfp.is_absolute() else (_HERE.parent / kfp).resolve()))
    # SPEC_epoch2 B12: every cell's declared duration (600 on the real corpus; the smoke corpus passes n_pairs x 0.644);
    # passed only when it departs from the extractor's default, so a default run's argv is stable across resumes
    if float(getattr(o, "duration_s", 600) or 600) != 600:
        a += ["--duration-s", getattr(o, "duration_s")]
    P.append(_cmd(1, "extract all", "extract", "all", a, outputs=["extract"], inputs=m1_inputs))
    # ---- move 2: preconditions, the templates, G-P (al-Kindi review 6)
    a = [*O]
    if o.assume_failed_zero:
        a += ["--assume-failed-zero", "--assume-reason", o.assume_reason]
    if o.failed_counts:
        a += ["--failed-counts", o.failed_counts]
    # epoch 2 (SPEC_epoch2 section 4; AD 2026-09-17): the C1 flags travel only when the author sets them; the
    # preconditions command's own default is `--c1-rule auto` and it records the values used regardless
    for flag, attr in (("--c1-rule", "c1_rule"), ("--c1-abs-fraction", "c1_abs_fraction"),
                       ("--c1-idle-percentile", "c1_idle_percentile"), ("--c1-activity-min", "c1_activity_min"),
                       ("--c1-activity-min-pages", "c1_activity_min_pages")):      # SPEC_epoch2 B1: AA T1's 200 pages, when set
        if getattr(o, attr, None) is not None:
            a += [flag, getattr(o, attr)]
    P.append(_cmd(2, "preconditions", "gates_precondition", "preconditions", a,
                  outputs=["gates/preconditions.csv", "gates/preconditions.json"], inputs=["cells.csv", "inputs/failed_counts.csv"]))
    P.append(_cmd(2, "pass-table template", "gates_calibration", "pass-table", [*O], outputs=["inputs/pass_table.csv"], template=True))
    P.append(_cmd(2, "gk0 template", "gates_precondition", "gk0-template", [*O], outputs=["inputs/gk0_source.csv"], template=True))
    P.append(_cmd(2, "idle admissibility template", "gates_precondition", "idle-admissibility-template", [*O],
                  outputs=["inputs/idle_admissibility.json"], template=True))
    P.append(_cmd(2, "head-drop template", "series", "head-drop-template", [*O], outputs=["inputs/head_drop.csv"], template=True))
    P.append(_cmd(2, "gp", "gates_calibration", "gp", [*O], outputs=["gates/gp.csv"], inputs=["cells.csv", "inputs/pass_table.csv"]))
    # ---- move 3: G-C per rung; combined last, its verdict is read from the other four rows
    #      (SPEC 3.4.3; CR 2.2 item 22; CHECK_1.md B3)
    for rung in ("apf", "persist", "content", "wapf", "combined"):
        P.append(_cmd(3, f"gc {rung}", "gates_calibration", "gc", [*O, "--rung", rung], outputs=["gates/gc.csv"],
                      inputs=["cells.csv", "inputs/head_drop.csv", ADMISSIBILITY]))
    # ---- move 4: G-K0 and G-F at the default grid (SPEC 8 item 38)
    P.append(_cmd(4, "gk0", "gates_precondition", "gk0", [*O], outputs=["gates/gk0.csv"],
                  inputs=["cells.csv", "inputs/gk0_source.csv", "inputs/head_drop.csv", ADMISSIBILITY]))
    P.append(_cmd(4, "gf all rungs at " + GF_DEFAULT_GRID, "gates_precondition", "gf", [*O, "--all-rungs", "--grid-id", GF_DEFAULT_GRID, "--n-perm", o.null_perm],
                  outputs=["gates/gf.csv"], inputs=["cells.csv", "inputs/idle_admissibility.json", "inputs/head_drop.csv", ADMISSIBILITY]))
    # ---- move 5: the first two figures
    P.append(_cmd(5, "figures apf_per_kernel,level_matched", "figures", None, [*O, "--only", "apf_per_kernel,level_matched"],
                  outputs=["report/figures/fig_apf_per_kernel.pdf|report/figures/SKIPPED.txt"], inputs=["cells.csv", ADMISSIBILITY]))

    def temporal(move: int, rung: str):
        both = "--norm" if rung == "combined" else "--both"
        grid_inputs = ["cells.csv", "inputs/head_drop.csv", "inputs/pass_table.csv", ADMISSIBILITY]
        # SPEC_epoch2 B19: wAPF's normalization rule travels to the feature files and the grid when it departs from
        # the modules' own default (median_K), so a default run's argv is stable across resumes
        wn = ["--wapf-norm", getattr(o, "wapf_norm")] if getattr(o, "wapf_norm", "median_K") not in (None, "median_K") else []
        P.append(_cmd(move, f"features {rung} all grid", "series", "features", [*O, "--rung", rung, "--all-grid", both, *wn],
                      outputs=[f"features/{rung}/Wall_Hall_norm.npz"], inputs=["cells.csv", "inputs/head_drop.csv", ADMISSIBILITY]))
        P.append(_cmd(move, f"grid {rung}", "gates_temporal", "grid", [*O, "--rung", rung, *wn],
                      outputs=[f"gates/grid/{rung}/Wall_Hall/temporal_per_kernel.csv"], inputs=grid_inputs))
        P.append(_cmd(move, f"g3 {rung}", "gates_temporal", "g3", [*O, "--rung", rung], outputs=["gates/g3_flags.csv"],
                      inputs=["cells.csv", "inputs/head_drop.csv", ADMISSIBILITY]))
        # G-ORD writes its label into the 13 per-kernel grid CSVs, which `grid` rewrites with
        # `pending: gate_gord`; its inputs are therefore a superset of the grid's triggers so that any
        # rebuild of the grid re-runs G-ORD before `select` reads the label (al-Farabi certification
        # cycle 2, 7.2)
        ga = [*O, "--rung", rung]
        # SPEC_epoch2 B26: G-ORD's cost flags, passed when they depart from the gord CLI's defaults (20 and 100)
        if int(getattr(o, "gord_n_order_perm", 20) or 20) != 20:
            ga += ["--n-order-perm", getattr(o, "gord_n_order_perm")]
        if int(getattr(o, "gord_null_perm", 100) or 100) != 100:
            ga += ["--null-perm", getattr(o, "gord_null_perm")]
        P.append(_cmd(move, f"gord {rung}", "gates_temporal", "gord", ga,
                      outputs=[f"gates/grid/{rung}/W8_H4/gord.json"], inputs=grid_inputs + ["gates/gk0.csv"]))
        # the selection's inputs are the 13 grid CSVs plus gk0.csv and gc.csv (what `select` hashes
        # into selection.json params.inputs_sha256_<rung>), so a rebuilt grid makes it stale
        # (al-Farabi certification 7.1; SPEC 3.5.7)
        P.append(_cmd(move, f"select {rung}", "gates_temporal", "select", [*O, "--rung", rung],
                      outputs=[f"json:gates/selection.json:{rung}"],
                      inputs=["inputs/pass_table.csv", "gates/gk0.csv", "gates/gc.csv", ADMISSIBILITY]
                             + [f"gates/grid/{rung}/{g}/temporal_per_kernel.csv" for g in GRID_IDS]))

    def splits(move: int, rung: str):
        P.append(_cmd(move, f"grid complete {rung}", "driver", "grid-check", [rung], internal=True,
                      outputs=[f"json:gates/grid_complete.json:{rung}"]))
        a = [*O, "--rung", rung, "--all-splits", "--raw-and-norm" if rung == "apf" else "--norm",
             "--null-perm", o.null_perm, "--null-splits", o.null_splits]
        # epoch 2 (SPEC_epoch2 section 5.1; E1 sec. 4 M2): the split stage and G-X read the rung's own selection
        # entry (series.selected_grid_id reads doc[rung] only), so their trigger is the keyed
        # `json:gates/selection.json:<rung>`, not the whole file that grows through moves 6 to 12
        P.append(_cmd(move, f"splits {rung}", "models", "splits", a, outputs=[f"gates/splits/{rung}"],
                      inputs=["cells.csv", "inputs/head_drop.csv", f"json:gates/selection.json:{rung}", "gates/gk0.csv", ADMISSIBILITY]))
        P.append(_cmd(move, f"gx {rung}", "gates_comparison", "gx", [*O, "--rung", rung, "--null-perm", o.null_perm],
                      outputs=[f"csv:gates/gx.csv:rung={rung}"], inputs=["cells.csv", f"json:gates/selection.json:{rung}", "inputs/cell_order.csv", ADMISSIBILITY]))

    # ---- move 6: the temporal gate on APF
    temporal(6, "apf")
    # SPEC_epoch2 B2 (CHECK_3 M2): the two alias steps read the APF rows of g3_flags.csv, which grows by rung
    P.append(_cmd(6, "alias", "gates_calibration", "alias", [*O], outputs=["gates/alias.csv"], inputs=["inputs/pass_table.csv", "csv:gates/g3_flags.csv:rung=apf"]))
    # ---- move 7: Table 6
    splits(7, "apf")
    # `gl` keeps the whole selection.json as its input (SPEC_epoch2 B2: "keep the whole file for gl (both runs)"), so the
    # move-7 run is re-run once on a resume after the other rungs' selections arrive (BUILD_epoch2_fixes.md, B28)
    P.append(_cmd(7, "gl", "gates_comparison", "gl", [*O], outputs=["gates/gl.csv"], inputs=["cells.csv", "gates/selection.json", ADMISSIBILITY]))
    # SPEC_epoch2 B10 (SPEC 3.7.4's consequence of a G-L (ii) refusal): the driver's own step, right after `gl`
    P.append(_cmd(7, "gl2 rerun (feature drop after G-L (ii))", "driver", "gl2-rerun", [], internal=True,
                  outputs=["gates/gl2_rerun.json"], inputs=["csv:gates/gl.csv:part=ii", "json:gates/selection.json:apf", ADMISSIBILITY]))
    P.append(_cmd(7, "gn", "gates_comparison", "gn", [*O], outputs=["gates/gn.csv"], inputs=["cells.csv", "gates/gk0.csv", ADMISSIBILITY]))
    P.append(_cmd(7, "alias (again, table6 features)", "gates_calibration", "alias", [*O], outputs=["gates/alias.csv"],
                  inputs=["inputs/pass_table.csv", "csv:gates/g3_flags.csv:rung=apf", "json:gates/selection.json:apf"]))
    P.append(_cmd(7, "tables table6", "tables", None, [*O, "--only", "table6"], outputs=["report/tables/table6.csv"],
                  inputs=["cells.csv", "gates/selection.json", ADMISSIBILITY]))
    # ---- move 8: the fused plane
    P.append(_cmd(8, "gj", "gates_readings", "gj", [*O], outputs=["gates/gj.csv"], inputs=["cells.csv", "inputs/head_drop.csv", ADMISSIBILITY]))
    P.append(_cmd(8, "figures fused_plane", "figures", None, [*O, "--only", "fused_plane"],
                  outputs=["report/figures/fig_fused_plane.pdf|report/figures/SKIPPED.txt"], inputs=["cells.csv", ADMISSIBILITY]))
    # ---- move 9: persistence
    temporal(9, "persist")
    splits(9, "persist")
    P.append(_cmd(9, "figures j_hist", "figures", None, [*O, "--only", "j_hist"],
                  outputs=["report/figures/fig_j_hist.pdf|report/figures/SKIPPED.txt"], inputs=["cells.csv", ADMISSIBILITY]))
    # ---- move 10: content-change
    temporal(10, "content")
    splits(10, "content")
    P.append(_cmd(10, "gdec", "gates_readings", "gdec", [*O], outputs=["gates/gdec.csv"],
                  inputs=["cells.csv", "inputs/pass_table.csv", "gates/gp.csv", ADMISSIBILITY]))
    P.append(_cmd(10, "figures ratio_hist,floyd_decay", "figures", None, [*O, "--only", "ratio_hist,floyd_decay"],
                  outputs=["report/figures/fig_ratio_hist.pdf|report/figures/SKIPPED.txt"], inputs=["cells.csv", "gates/gdec.csv", ADMISSIBILITY]))
    # ---- move 11: wAPF
    temporal(11, "wapf")
    splits(11, "wapf")
    P.append(_cmd(11, "tables wapf_over_apf", "tables", None, [*O, "--only", "wapf_over_apf"],
                  outputs=["report/tables/table_wapf_over_apf.csv"], inputs=["cells.csv", ADMISSIBILITY]))
    # ---- move 12: combined, G-F at the selected points (al-Farabi 2.6), the comparisons, the report
    temporal(12, "combined")
    splits(12, "combined")
    # G-L (i) is per rung (P2 Sec. V 5.2 G-L; CR 2.2 item 24): the move-7 run fills APF's row
    # early; this run fills the other four once their selections exist (CHECK_1.md B2)
    P.append(_cmd(12, "gl (all rungs)", "gates_comparison", "gl", [*O], outputs=["gates/gl.csv"], inputs=["cells.csv", "gates/selection.json", ADMISSIBILITY]))
    P.append(_cmd(12, "gf all rungs at the selected points", "gates_precondition", "gf", [*O, "--all-rungs", "--n-perm", o.null_perm],
                  outputs=["gates/gf.csv"], inputs=["cells.csv", "inputs/idle_admissibility.json", "inputs/head_drop.csv", "gates/selection.json", ADMISSIBILITY]))
    # the matched comparison runs for every Table 7 (split, label space) at the driver's --null-splits
    # (SPEC_epoch2 3.5.1; E1 sec. 4 M4)
    P.append(_cmd(12, "gdim", "gates_comparison", "gdim", [*O, "--null-perm", o.null_perm, "--null-splits", o.null_splits],
                  outputs=["gates/gdim.csv"], inputs=["cells.csv", "gates/selection.json", ADMISSIBILITY]))
    P.append(_cmd(12, "gm", "gates_comparison", "gm", [*O], outputs=["gates/gm.csv"], inputs=["cells.csv", "gates/selection.json", ADMISSIBILITY]))
    P.append(_cmd(12, "variance", "variance", None, [*O], outputs=["gates/gv.csv", "gates/gv_summary.csv"], inputs=["cells.csv", "gates/selection.json", "gates/gk0.csv", ADMISSIBILITY]))
    P.append(_cmd(12, "cluster", "models", "cluster", [*O, "--rung", o.table8_rung, "--null-perm", o.null_perm],
                  outputs=["gates/clustering.csv", "gates/clustering.json"], inputs=["cells.csv", "gates/selection.json", "gates/gk0.csv", ADMISSIBILITY]))
    P.append(_cmd(12, "tables (all)", "tables", None, [*O, "--table8-rung", o.table8_rung],
                  outputs=["report/tables/table5.csv", "report/tables/table7.csv", "report/tables/table8.csv", "report/tables/tablegv.csv"],
                  inputs=["cells.csv", "gates/selection.json", ADMISSIBILITY]))
    sk = [*O]
    if o.standalone_tex:
        sk += ["--standalone", o.standalone_tex]
    P.append(_cmd(12, "latex skeleton", "latex_skeleton", None, sk, outputs=["report/paper2_skeleton.tex"], inputs=["inputs/pass_table.csv"]))
    fa = [*O]
    if o.piano_cell:
        fa += ["--piano-cell", o.piano_cell]
    fa += ["--piano-stride", o.piano_stride]
    P.append(_cmd(12, "figures (all)", "figures", None, fa,
                  outputs=["report/figures/fig_table5_grid.pdf|report/figures/SKIPPED.txt"], inputs=["cells.csv", ADMISSIBILITY]))
    P.append(_cmd(12, "tables manifest", "tables", None, [*O, "--only", "manifest"], outputs=["report/manifest.json"], inputs=["cells.csv", ADMISSIBILITY]))
    # ---- move 13: the tripwire check on every table row (al-Farabi 2.6)
    P.append(_cmd(13, "gf check on Table 7", "driver", "gf-check", [], internal=True, outputs=["gates/gf_check.json"]))
    # ---- move 14: the exact-input comparators (SPEC_epoch2.md Part 1; builder A owns this block)
    cmp_common = [*O, "--null-perm", o.null_perm, "--null-splits", o.null_splits, "--n-jobs", o.n_jobs,
                  "--n-estimators", getattr(o, "n_estimators", 300), "--seed-offset", getattr(o, "seed_offset", 0)]
    cmp_inputs = ["cells.csv", "inputs/head_drop.csv", ADMISSIBILITY, "gates/gk0.csv",
                  "json:gates/selection.json:apf", "gates/gm.params.json"]
    P.append(_cmd(14, "comparators savoldi", "comparators", "savoldi",
                  cmp_common + ["--rows", getattr(o, "savoldi_rows", "all_after_head_drop")],
                  outputs=["gates/comparators/savoldi.csv", "gates/splits/cmp_savoldi/Wall_Hall/loko__archetype/scores.json"],
                  inputs=cmp_inputs))
    P.append(_cmd(14, "comparators dhodapkar", "comparators", "dhodapkar",
                  cmp_common + ["--delta-th-default", getattr(o, "delta_th_default", 0.04)],
                  outputs=["gates/comparators/dhodapkar_sweep.csv", "gates/splits/cmp_dhodapkar/Wall_Hall/loko__archetype/scores.json"],
                  inputs=cmp_inputs))
    P.append(_cmd(14, "comparators law", "comparators", "law",
                  cmp_common + ["--x-default", getattr(o, "law_x_default", 4),
                                "--jobs", getattr(o, "comparator_jobs", None) or o.n_jobs],
                  outputs=["gates/comparators/law_sweep.csv", "gates/splits/cmp_law/Wall_Hall/loko__archetype/scores.json"],
                  inputs=cmp_inputs))
    P.append(_cmd(14, "comparators gates", "comparators", "gates", cmp_common,
                  outputs=["gates/comparators/gm.csv", "gates/comparators/verdicts.csv"],
                  inputs=cmp_inputs + ["gates/comparators/savoldi.csv", "gates/comparators/dhodapkar.csv", "gates/comparators/law.csv"]))
    P.append(_cmd(14, "tables table7_comparators,table_comparators", "tables", None,
                  [*O, "--only", "table7_comparators,table_comparators"],
                  outputs=["report/tables/table7_comparators.csv", "report/tables/table_comparators.csv"],
                  inputs=cmp_inputs + ["gates/comparators/verdicts.csv"]))
    # the admissibility record is a declared input of every step from move 3 on (CERT 7.1;
    # tests/test_driver.py asserts it), so the two report steps carry it beside their own input
    P.append(_cmd(14, "figures dhodapkar_sweep", "figures", None, [*O, "--only", "dhodapkar_sweep"],
                  outputs=["report/figures/fig_dhodapkar_sweep.pdf|report/figures/SKIPPED.txt"],
                  inputs=["gates/comparators/dhodapkar_sweep.csv", ADMISSIBILITY]))
    P.append(_cmd(14, "tables manifest (after comparators)", "tables", None, [*O, "--only", "manifest"],
                  outputs=["report/manifest.json"], inputs=["gates/comparators/verdicts.csv", ADMISSIBILITY]))
    # ---- move 15 (optional; SPEC_epoch2 Part 4 item 29, 2026-09-29): the LORO luck checks that a run under the
    # runbook's fallback `--null-splits loko,within_trace` skipped. LORO's split stage is re-run with its null for
    # every rung at the rung's selected point (the same command as moves 7 and 9 to 12, `--split loro` and
    # `--null-splits loro`; same seeds, so the scores are the same and only the null is added), then every step
    # whose own LORO null follows the flag (G-DIM, the comparators) under the full paper value, then the readers
    # and tables that print them. Only run when selected (`--moves 15`); the default `--moves 0-14` leaves it out.
    for rung in ("apf", "persist", "content", "wapf", "combined"):
        P.append(_cmd(15, f"loro luck check {rung}", "models", "splits",
                      [*O, "--rung", rung, "--split", "loro", "--labelspace", "all",
                       "--raw-and-norm" if rung == "apf" else "--norm", "--null-perm", o.null_perm, "--null-splits", "loro"],
                      outputs=[f"gates/splits/{rung}"],
                      inputs=["cells.csv", "inputs/head_drop.csv", f"json:gates/selection.json:{rung}", "gates/gk0.csv", ADMISSIBILITY]))
    P.append(_cmd(15, "gdim (with LORO luck check)", "gates_comparison", "gdim",
                  [*O, "--null-perm", o.null_perm, "--null-splits", LORO_FULL_NULL_SPLITS],
                  outputs=["gates/gdim.csv"], inputs=["cells.csv", "gates/selection.json", ADMISSIBILITY]))
    cmp_full = [*O, "--null-perm", o.null_perm, "--null-splits", LORO_FULL_NULL_SPLITS, "--n-jobs", o.n_jobs,
                "--n-estimators", getattr(o, "n_estimators", 300), "--seed-offset", getattr(o, "seed_offset", 0)]
    P.append(_cmd(15, "comparators savoldi (with LORO luck check)", "comparators", "savoldi",
                  cmp_full + ["--rows", getattr(o, "savoldi_rows", "all_after_head_drop")],
                  outputs=["gates/comparators/savoldi.csv"], inputs=cmp_inputs))
    P.append(_cmd(15, "comparators dhodapkar (with LORO luck check)", "comparators", "dhodapkar",
                  cmp_full + ["--delta-th-default", getattr(o, "delta_th_default", 0.04)],
                  outputs=["gates/comparators/dhodapkar_sweep.csv"], inputs=cmp_inputs))
    P.append(_cmd(15, "comparators law (with LORO luck check)", "comparators", "law",
                  cmp_full + ["--x-default", getattr(o, "law_x_default", 4),
                              "--jobs", getattr(o, "comparator_jobs", None) or o.n_jobs],
                  outputs=["gates/comparators/law_sweep.csv"], inputs=cmp_inputs))
    P.append(_cmd(15, "comparators gates (with LORO luck check)", "comparators", "gates", cmp_full,
                  outputs=["gates/comparators/gm.csv", "gates/comparators/verdicts.csv"],
                  inputs=cmp_inputs + ["gates/comparators/savoldi.csv", "gates/comparators/dhodapkar.csv", "gates/comparators/law.csv"]))
    P.append(_cmd(15, "gl (after LORO luck checks)", "gates_comparison", "gl", [*O], outputs=["gates/gl.csv"],
                  inputs=["cells.csv", "gates/selection.json", ADMISSIBILITY]))
    P.append(_cmd(15, "gm (after LORO luck checks)", "gates_comparison", "gm", [*O], outputs=["gates/gm.csv"],
                  inputs=["cells.csv", "gates/selection.json", ADMISSIBILITY]))
    P.append(_cmd(15, "tables (all, after LORO luck checks)", "tables", None, [*O, "--table8-rung", o.table8_rung],
                  outputs=["report/tables/table5.csv", "report/tables/table7.csv", "report/tables/table8.csv", "report/tables/tablegv.csv"],
                  inputs=["cells.csv", "gates/selection.json", ADMISSIBILITY]))
    P.append(_cmd(15, "tables table7_comparators,table_comparators (after LORO luck checks)", "tables", None,
                  [*O, "--only", "table7_comparators,table_comparators"],
                  outputs=["report/tables/table7_comparators.csv", "report/tables/table_comparators.csv"],
                  inputs=cmp_inputs + ["gates/comparators/verdicts.csv"]))
    P.append(_cmd(15, "tables manifest (after LORO luck checks)", "tables", None, [*O, "--only", "manifest"],
                  outputs=["report/manifest.json"], inputs=["gates/comparators/verdicts.csv", ADMISSIBILITY]))
    # ---- move 16 (optional; added 2026-10-05, after the run of 2026-09-29: SPEC_epoch2 Part 4 item 30; P2_AUTHOR_ANSWERS
    # A20 to A22). (1) The corrected instrument check for content, per changed byte, written beside gates/gc.csv and
    # never over it; (2) the idle common-ground test: leave-one-run-out with the 8 idle runs as a 13th class, per rung
    # at its selected point, the LORO null's label shuffles; (3) the second Table 2 under the corrected check and the
    # idle test's void rule. The declared tables and every existing gate file are left as they are. Only when selected
    # (`--moves 16`); the default `--moves 0-14` leaves it out, as it leaves move 15 out.
    P.append(_cmd(16, "gc corrected (content, per changed byte)", "gates_calibration", "gc-corrected", [*O],
                  outputs=["gates/added/gc_corrected.csv"], inputs=["cells.csv", "gates/gc.csv", ADMISSIBILITY]))
    P.append(_cmd(16, "idle common ground", "gates_idle_common_ground", "run", [*O, "--null-perm", o.null_perm],
                  outputs=["gates/added/idle_common_ground.csv"],
                  inputs=["cells.csv", "inputs/head_drop.csv", "gates/selection.json", "gates/gk0.csv", ADMISSIBILITY]))
    # 2026-10-06: the comparator rows it reads (report/tables/table7_comparators.csv, made by move 14) were not declared; they are now, so on a
    # run where move 16 ran this one command is reported stale, which is correct; no other existing command changes its inputs or arguments
    P.append(_cmd(16, "tables_eusipco table2_corrected", "tables_eusipco", None, [*O, "--only", "table2_corrected"],
                  outputs=["report/tables/eusipco_table2_corrected.csv"],
                  inputs=["report/tables/table7.csv", "report/tables/table7_comparators.csv", "gates/added/gc_corrected.csv", "gates/added/idle_common_ground.csv",
                          "gates/selection.json", ADMISSIBILITY]))
    # ---- move 17 (optional; added 2026-10-06: SPEC_epoch2 Part 4 item 31; P2_AUTHOR_ANSWERS A25): the new-block test on the five
    # readings, the encoding paper's version of the SPL paper's test (plan12_grounding/PROMPT_new_blocks_test.md): the first 80% of
    # every recording's windows seen, a one-window gap, the rest new blocks named at three levels (archetype, kernel, run), alone and in
    # pools of 2 and 3; (1) at the primary window W64_H32 for every reading; (2) at each reading's own selected window (gates/selection.json;
    # a reading whose own window is the primary reads "same as primary"); (3) the tables and figures. Every file each command reads is
    # declared: the own-window command's feature files are resolved from the selection when it exists (else the primary). Only when
    # selected (`--moves 17`); the default `--moves 0-14` leaves it out, as it leaves 15 and 16 out.
    nb_feats = [f"features/{rung}/{NEW_BLOCKS_PRIMARY_GRID}_norm.npz" for rung in RUNGS]
    P.append(_cmd(17, f"new blocks ({NEW_BLOCKS_PRIMARY_GRID})", "new_blocks", "run", [*O, "--window", NEW_BLOCKS_PRIMARY_GRID],
                  outputs=[f"gates/added/new_blocks/{NEW_BLOCKS_PRIMARY_GRID}/summary.json"],
                  inputs=["cells.csv", "inputs/head_drop.csv", ADMISSIBILITY] + nb_feats))
    from plan11_encoding_ladder import series as _series        # local: the selection reader, for the own-window feature files
    own_feats = []
    for rung in RUNGS:
        gid, _ = _series.selected_grid_id(Path(out), rung, default=None)
        own_feats.append(f"features/{rung}/{gid or NEW_BLOCKS_PRIMARY_GRID}_norm.npz")
    P.append(_cmd(17, "new blocks (own windows)", "new_blocks", "run", [*O, "--window", "own"],
                  outputs=["gates/added/new_blocks/own/summary.json"],
                  inputs=["cells.csv", "inputs/head_drop.csv", ADMISSIBILITY, "gates/selection.json", f"gates/added/new_blocks/{NEW_BLOCKS_PRIMARY_GRID}/scores.csv"]
                         + sorted(set(own_feats))))
    nb_read = [f"gates/added/new_blocks/{w}/{n}" for w in (NEW_BLOCKS_PRIMARY_GRID, "own")
               for n in ("scores.csv", "margins.csv", "per_kernel.csv", "by_position.csv", "run_level.csv", "confusion_kernel_apf.csv", "confusion_kernel_content.csv", "summary.json")]
    P.append(_cmd(17, "new blocks tables and figures", "new_blocks", "tables", [*O],
                  outputs=["gates/added/new_blocks.csv", "report/tables/new_blocks_scores.csv", "report/tables/new_blocks_margins.csv",
                           "report/tables/new_blocks_run_level.csv", "report/tables/new_blocks_by_position.csv", "report/figures/new_blocks_accuracy.svg",
                           "report/figures/new_blocks_by_position.svg", "report/figures/new_blocks_per_kernel.svg", "report/figures/new_blocks_run_level.svg",
                           "report/figures/new_blocks_confusion_apf.svg", "report/figures/new_blocks_confusion_content.svg"],
                  inputs=nb_read + [ADMISSIBILITY]))          # the admissibility record is an input of every step from move 3 on (al-Farabi certification 7.1)
    # seed offsets and n-jobs on the commands that take them (SPEC 7.1)
    for c in P:
        key = (c["module"], c["sub"])
        if key in RANDOM_COMMANDS and int(o.seed_offset or 0) != 0:
            c["args"] += ["--seed-offset", str(o.seed_offset)]
        if key in NJOBS_COMMANDS and int(o.n_jobs) != 1:
            c["args"] += ["--n-jobs", str(o.n_jobs)]
        if key in NEST_COMMANDS and int(getattr(o, "n_estimators", 300) or 300) != 300:     # SPEC_epoch2 B25: when it departs from the SPEC default
            c["args"] += ["--n-estimators", str(getattr(o, "n_estimators"))]
    return P


# ----------------------------------------------------------------------------------------------
# ledger
# ----------------------------------------------------------------------------------------------
def load_ledger(out: Path, name: str = LEDGER) -> dict:
    """The ledger `<out>/<name>`; `name` (SPEC_DETECTION 1.3, additive) lets the detection driver
    keep its own file (`driver_detection_state.json`)."""
    p = Path(out) / name
    if p.exists():
        try:
            return read_json(p)
        except Exception:
            pass
    return result_json("driver_state", {}, CITATION, {"package_version": PACKAGE_VERSION, "runs": [], "commands": []})


def save_ledger(out: Path, ledger: dict, name: str = LEDGER) -> None:
    write_json(Path(out) / name, ledger)


def _cmd_key(c: dict) -> str:
    return f"{c['move']}:{c['name']}"


def _last_done(ledger: dict, key: str) -> dict | None:
    for e in reversed(ledger.get("commands", [])):
        if e.get("key") == key and e.get("status") == "done":
            return e
    return None


def _output_exists(out: Path, spec: str) -> bool:
    """`path`, `a|b` (either), `json:path:key` (the key exists in the JSON, at the top level or
    under `selection`/`rungs`), `csv:path:col=value` (a row with that value exists)."""
    if "|" in spec and not spec.startswith(("json:", "csv:")):
        return any(_output_exists(out, s) for s in spec.split("|"))
    if spec.startswith("json:"):
        _, path, key = spec.split(":", 2)
        p = Path(out) / path
        if not p.exists():
            return False
        try:
            j = read_json(p)
        except Exception:
            return False
        if key in j:
            return True
        return any(isinstance(j.get(k), dict) and key in j[k] for k in ("selection", "rungs", "grid_complete"))
    if spec.startswith("csv:"):
        _, path, cond = spec.split(":", 2)
        p = Path(out) / path
        if not p.exists():
            return False
        col, val = cond.split("=", 1)
        try:
            return any(r.get(col) == val for r in read_csv(p))
        except Exception:
            return False
    return (Path(out) / spec).exists()


def _input_hash(out: Path, spec: str) -> str:
    """The keyed input hash of one input spec (SPEC_epoch2 section 5.1; E1 sec. 4 M2; CERTIFY_al_farabi.md
    section 7 item 3). Forms: a plain ``path`` hashes the file's bytes (``"absent"`` when it does not
    exist, as before); ``json:<path>:<key>`` hashes exactly what a per-rung command reads from a JSON
    document that grows by rung: ``{"key", "entry": doc[key], "params": doc["params"][key],
    "params_inputs": doc["params"]["inputs_sha256_" + key]}`` (the entry is looked up at the top level,
    then under ``selection`` / ``rungs``, as ``_output_exists`` does), serialised with sorted keys;
    ``csv:<path>:<col>=<value>`` hashes the header plus the rows whose ``col`` equals ``value``. A
    change to another rung's entry or row therefore does not change the hash, and a change to any
    part the command reads does."""
    import hashlib
    p_spec = spec
    if spec.startswith("json:"):
        _, path, key = spec.split(":", 2)
        p = Path(out) / path
        if not p.exists():
            return "absent"
        try:
            j = read_json(p)
        except Exception:
            return "unreadable"
        entry = j.get(key)
        if entry is None:
            for k in ("selection", "rungs", "grid_complete"):
                if isinstance(j.get(k), dict) and key in j[k]:
                    entry = j[k][key]
                    break
        params = j.get("params") if isinstance(j.get("params"), dict) else {}
        obj = {"key": key, "entry": entry, "params": params.get(key), "params_inputs": params.get("inputs_sha256_" + key)}
        blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()
    if spec.startswith("csv:"):
        _, path, cond = spec.split(":", 2)
        p = Path(out) / path
        if not p.exists():
            return "absent"
        col, val = cond.split("=", 1)
        try:
            rows = read_csv(p)
        except Exception:
            return "unreadable"
        header = list(rows[0].keys()) if rows else []
        obj = {"header": header, "rows": [r for r in rows if r.get(col) == val]}
        blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return inputs_sha256([Path(out) / p_spec])[str(Path(out) / p_spec)]


def _inputs_sha256(out: Path, specs: list[str]) -> dict:
    """{spec string: keyed hash} for a command's declared inputs (the ledger's ``inputs_sha256``)."""
    return {spec: _input_hash(out, spec) for spec in specs}


def _stale_part(spec: str) -> str:
    """The part of an input a stale message names: ``selection.json[apf]``, ``g3_flags.csv[rung=apf]``,
    or the file name."""
    if spec.startswith(("json:", "csv:")):
        _, path, key = spec.split(":", 2)
        return f"{Path(path).name}[{key}]"
    return Path(spec).name


ARGV_COST_FLAGS = ("--n-jobs", "--jobs")     # flags that change cost, never a number: ignored by the argument comparison


def _argv_signature(argv) -> list[str] | None:
    """A command's argv without the interpreter and without the cost-only flags and their values
    (SPEC_epoch2_review_al_farabi.md section 6 item 3, the second form)."""
    if not argv:
        return None
    toks = [str(t) for t in list(argv)[1:]]
    out, skip = [], False
    for t in toks:
        if skip:
            skip = False
            continue
        if t in ARGV_COST_FLAGS:
            skip = True
            continue
        out.append(t)
    return out


def _stale_reason(out: Path, prev: dict, inputs: list[str], argv=None) -> str | None:
    """al-Farabi 2.8 with the epoch-2 refinement (SPEC_epoch2 section 5.1): a command is skipped only when
    a ``done`` ledger record exists for its key, every declared output exists, and for every declared
    input spec the keyed hash now equals the hash recorded then. Compared spec by spec; the message
    names the part (``stale: selection.json[apf] changed since move 7 (...)``). Also (the hazard of
    SPEC_epoch2_review_al_farabi.md section 6 item 3, sharper now that the C1 flags exist): when
    ``argv`` is given and differs from the ``done`` record's argv after dropping ``--n-jobs`` /
    ``--jobs`` and their values, the command is ``stale: arguments changed since move <n>``, so a
    resume with another ``--c1-rule`` or ``--null-perm`` re-runs the step instead of keeping the old
    result under the new flag."""
    cur = _inputs_sha256(out, inputs)
    old = prev.get("inputs_sha256", {})
    for k, v in cur.items():
        if old.get(k, "absent") != v:
            return f"stale: {_stale_part(k)} changed since move {prev.get('move')} ({prev.get('finished_at', '?')})"
    if argv is not None and prev.get("argv") is not None:
        a, b = _argv_signature(argv), _argv_signature(prev.get("argv"))
        if a != b:
            return f"stale: arguments changed since move {prev.get('move')} ({prev.get('finished_at', '?')})"
    return None


# ----------------------------------------------------------------------------------------------
# internal steps
# ----------------------------------------------------------------------------------------------
def grid_check(out: Path, rung: str) -> tuple[str, dict]:
    """al-Farabi 2.2 / SPEC 3.5.7: refuse the split stage of a rung unless all 13
    `gates/grid/<rung>/<grid_id>/temporal_per_kernel.csv` and the 13 feature npz files
    (`features/<rung>/<grid_id>_{raw,norm}.npz`; norm only for `combined`) exist. Written to
    `gates/grid_complete.json` under the rung key."""
    out = Path(out)
    missing = []
    for gid in GRID_IDS:
        if not (out / "gates" / "grid" / rung / gid / "temporal_per_kernel.csv").exists():
            missing.append(f"gates/grid/{rung}/{gid}/temporal_per_kernel.csv")
        variants = ("norm",) if rung == "combined" else ("raw", "norm")
        for v in variants:
            if not (out / "features" / rung / f"{gid}_{v}.npz").exists():
                missing.append(f"features/{rung}/{gid}_{v}.npz")
    verdict = "complete" if not missing else refused(f"grid incomplete for {rung}: {len(missing)} files missing")
    p = out / "gates" / "grid_complete.json"
    j = read_json(p) if p.exists() else result_json("grid_complete", {"grid_ids": list(GRID_IDS)},
                                                    "SPEC 3.5.7; al-Farabi review 2.2", {"grid_complete": {}})
    j.setdefault("grid_complete", {})[rung] = {"verdict": verdict, "missing": missing, "checked_at": now_iso()}
    write_json(p, j)
    return verdict, {"missing": missing}


def gf_check(out: Path) -> tuple[str, dict]:
    """al-Farabi 2.6 (move 13): every row of `report/tables/table7.csv` carries its G-F (i)
    verdict; a row whose `G-F (i)` cell is empty is listed. Written to `gates/gf_check.json`."""
    out = Path(out)
    p = out / "report" / "tables" / "table7.csv"
    if not p.exists():
        verdict, detail = f"not run: {p.relative_to(out)} missing", {}
    else:
        rows = read_csv(p)
        empty = [f"{r.get('rung')}/{r.get('split')}/{r.get('label space')}" for r in rows if not str(r.get("G-F (i)", "")).strip()]
        per_rung = {}
        for r in rows:
            per_rung.setdefault(r.get("rung"), set()).add(r.get("G-F (i)", ""))
        detail = {"n_rows": len(rows), "rows_without_gf": empty, "gf_verdicts_per_rung": {k: sorted(v) for k, v in per_rung.items()}}
        verdict = "pass" if rows and not empty else refused(f"{len(empty)} Table 7 rows without a G-F (i) verdict" if rows else "Table 7 has no rows")
    write_json(out / "gates" / "gf_check.json", result_json("gf_check", {"table7": str(p)}, "SPEC 3.3.4; al-Farabi review 2.6 (move 13)",
                                                            {"verdict": verdict, **detail}))
    return verdict, detail


def gl2_rerun(out: Path, o: argparse.Namespace, rec: dict | None = None) -> tuple[str, dict]:
    """SPEC 3.7.4's consequence of a G-L (ii) refusal, as the driver's own step at move 7 right after
    `gl` (SPEC_epoch2 B10; CHECK_3 M10; CERT 6.9; E1 6.38). Reads `gates/gl.csv` part (ii) (APF's
    shot-noise route): when the verdict is `refused: shot noise explains CV` and `--gl2-rerun auto`
    (the default, SPEC 3.7.4's sentence as written) it runs, as a nested subprocess recorded in the
    ledger under `argv_nested`, `models splits --out O --rung apf --all-splits --raw-and-norm
    --null-perm <n> --null-splits <s> --feature-drop cov,std,peak2med --base-dir splits_gl2drop`
    and writes `gates/gl2_rerun.json` (`status = "done"`, the base dir, the drop set); when the
    verdict is `pass` it records `not run: G-L (ii) passed`; under `manual` it records `not run:
    manual (--gl2-rerun manual)` and the author runs the command by hand (RUNBOOK move 7). The
    tables are not changed: the refused run's numbers stay printed with the G-L refusal beside them
    and the re-run lives under `gates/splits_gl2drop/apf/` for the author (Part 4 item 18)."""
    out = Path(out)
    mode = getattr(o, "gl2_rerun", "auto") or "auto"
    p = out / "gates" / "gl.csv"
    part2 = None
    if p.exists():
        for r in read_csv(p):
            if r.get("rung") == "apf" and str(r.get("part", "")).strip().lower() in ("ii", "2", "(ii)"):
                part2 = r.get("verdict", "")
    detail = {"mode": mode, "gl_part_ii": part2, "feature_drop": GL2_FEATURE_DROP, "base_dir": GL2_BASE_DIR, "argv_nested": None,
              "citation": "SPEC 3.7.4; SPEC_epoch2 B10; P2 Sec. V 5.2 G-L"}
    if part2 is None:
        status = "not run: gates/gl.csv missing or has no part (ii) row (move 7)"
    elif part2 == "pass":
        status = "not run: G-L (ii) passed"
    elif part2 != GL2_SHOT_NOISE:
        status = f"not run: G-L (ii) reads {part2}"
    elif mode == "manual":
        status = "not run: manual (--gl2-rerun manual)"
    else:
        argv_nested = [sys.executable, "-m", f"{PACKAGE_NAME}.models", "splits", "--out", str(out), "--rung", "apf",
                       "--all-splits", "--raw-and-norm", "--null-perm", str(getattr(o, "null_perm", 500)),
                       "--null-splits", str(getattr(o, "null_splits", "loko,loro,within_trace")),
                       "--feature-drop", GL2_FEATURE_DROP, "--base-dir", GL2_BASE_DIR]
        if getattr(o, "n_estimators", None) is not None:
            argv_nested += ["--n-estimators", str(getattr(o, "n_estimators"))]
        if int(getattr(o, "n_jobs", 1) or 1) != 1:
            argv_nested += ["--n-jobs", str(getattr(o, "n_jobs"))]
        if int(getattr(o, "seed_offset", 0) or 0) != 0:
            argv_nested += ["--seed-offset", str(getattr(o, "seed_offset"))]
        detail["argv_nested"] = argv_nested
        if rec is not None:
            rec["argv_nested"] = argv_nested
        proc = subprocess.run(argv_nested, cwd=str(_HERE.parent), capture_output=True, text=True)
        detail["exit_code"] = proc.returncode
        detail["stderr_tail"] = (proc.stderr or "")[-2000:]
        status = "done" if proc.returncode == 0 else refused(f"gl2 re-run failed: exit {proc.returncode}")
    write_json(out / "gates" / "gl2_rerun.json", result_json("gl2_rerun", {"mode": mode, "feature_drop": GL2_FEATURE_DROP.split(","),
                                                                          "base_dir": GL2_BASE_DIR},
                                                              "SPEC 3.7.4; SPEC_epoch2 B10", {"status": status, **detail}))
    return status, detail


# ----------------------------------------------------------------------------------------------
# execution
# ----------------------------------------------------------------------------------------------
def _module_path(module: str) -> Path:
    return _HERE / f"{module}.py"


def run_plan(o: argparse.Namespace, plan: list[dict], *, max_move: int = MAX_MOVE, ledger_name: str = LEDGER,
             internal_steps: dict | None = None) -> int:
    """Run `plan` (a list of `_cmd` dicts). The three keyword arguments are SPEC_DETECTION 1.3's
    additive extensions, with defaults that keep the epoch-1 behaviour: `max_move` bounds
    `--moves`, `ledger_name` names the ledger file, and `internal_steps` maps an internal `sub`
    name to a callable `(out: Path, args: list[str]) -> (verdict: str, detail: dict)` consulted
    before the built-in `grid-check` / `gf-check` steps."""
    out = Path(o.out)
    out.mkdir(parents=True, exist_ok=True)
    internal_steps = dict(internal_steps or {})
    ledger = load_ledger(out, ledger_name)
    ledger["params"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(o).items() if k != "func"}
    run_rec = {"started_at": now_iso(), "moves": parse_moves(o.moves, max_move), "argv": sys.argv, "dry_run": bool(o.dry_run)}
    ledger.setdefault("runs", []).append(run_rec)
    save_ledger(out, ledger, ledger_name)
    selected = set(parse_moves(o.moves, max_move))
    only_modules = set(x.strip() for x in (o.only_modules or "").split(",") if x.strip())
    rc_final = 0
    for c in plan:
        if c["move"] not in selected:
            continue
        key = _cmd_key(c)
        rec = {"key": key, "move": c["move"], "name": c["name"], "module": c["module"], "sub": c["sub"],
               "started_at": now_iso(), "inputs_sha256": _inputs_sha256(out, c["inputs"])}
        argv = None if c["internal"] else [sys.executable, "-m", f"{PACKAGE_NAME}.{c['module']}"] + ([c["sub"]] if c["sub"] else []) + c["args"]
        rec["argv"] = argv
        rec["cwd"] = str(_HERE.parent)
        # the author's inputs are never overwritten
        if c["template"] and all(_output_exists(out, s) for s in c["outputs"]):
            rec.update(status="kept: author input exists", exit_code=0, finished_at=now_iso())
            ledger["commands"].append(rec); save_ledger(out, ledger, ledger_name); _say(rec); continue
        # skip rule with the staleness test (al-Farabi 2.8)
        prev = _last_done(ledger, key)
        if not o.force and prev is not None and c["outputs"] and all(_output_exists(out, s) for s in c["outputs"]):
            stale = _stale_reason(out, prev, c["inputs"], argv)
            if stale is None:
                rec.update(status="skipped: outputs exist and inputs unchanged", exit_code=0, finished_at=now_iso())
                ledger["commands"].append(rec); save_ledger(out, ledger, ledger_name); _say(rec); continue
            rec["stale"] = stale
        if o.dry_run:
            rec.update(status="dry-run", exit_code=0, finished_at=now_iso())
            ledger["commands"].append(rec); save_ledger(out, ledger, ledger_name); _say(rec); continue
        if only_modules and c["module"] not in only_modules:
            rec.update(status=f"not run: module {c['module']} filtered by --only-modules", exit_code=0, finished_at=now_iso())
            ledger["commands"].append(rec); save_ledger(out, ledger, ledger_name); _say(rec); continue
        if c["internal"]:
            if c["sub"] in internal_steps:
                verdict, detail = internal_steps[c["sub"]](out, list(c["args"]))
            elif c["sub"] == "grid-check":
                verdict, detail = grid_check(out, c["args"][0])
            elif c["sub"] == "gf-check":
                verdict, detail = gf_check(out)
            elif c["sub"] == "gl2-rerun":
                verdict, detail = gl2_rerun(out, o, rec)
            else:
                verdict, detail = refused(f"unknown internal step {c['sub']}"), {}
            rec.update(status="done" if not verdict.startswith("refused") else verdict, verdict=verdict, detail=detail,
                       exit_code=0, finished_at=now_iso())
            if c["sub"] == "gl2-rerun" and verdict.startswith("refused"):
                rec["exit_code"] = int((detail or {}).get("exit_code") or 1)
            ledger["commands"].append(rec); save_ledger(out, ledger, ledger_name); _say(rec)
            splits_will_run = (_module_path("models").exists() and (not only_modules or "models" in only_modules))
            if verdict.startswith("refused") and c["sub"] == "grid-check" and splits_will_run:
                print(f"stopped: {verdict}", file=sys.stderr)
                return 1
            continue
        if not _module_path(c["module"]).exists():
            if o.skip_missing_modules:
                rec.update(status=f"not run: module {c['module']}.py absent", exit_code=0, finished_at=now_iso())
                ledger["commands"].append(rec); save_ledger(out, ledger, ledger_name); _say(rec); continue
            rec.update(status=f"failed: module {c['module']}.py absent", exit_code=2, finished_at=now_iso())
            ledger["commands"].append(rec); save_ledger(out, ledger, ledger_name); _say(rec)
            print(f"missing module: {_module_path(c['module'])}", file=sys.stderr)
            return 2
        t0 = time.time()
        proc = subprocess.run(argv, cwd=str(_HERE.parent), capture_output=True, text=True)
        rec.update(exit_code=proc.returncode, finished_at=now_iso(), elapsed_s=round(time.time() - t0, 3),
                   stdout_tail=proc.stdout[-4000:], stderr_tail=proc.stderr[-4000:],
                   status="done" if proc.returncode == 0 else f"failed: exit {proc.returncode}")
        ledger["commands"].append(rec); save_ledger(out, ledger, ledger_name); _say(rec)
        if proc.returncode != 0:
            print(proc.stderr[-2000:], file=sys.stderr)
            print(f"stopped at move {c['move']} ({c['name']}), exit {proc.returncode}", file=sys.stderr)
            rc_final = proc.returncode if proc.returncode in (1, 2) else 1
            break
    run_rec["finished_at"] = now_iso()
    run_rec["exit_code"] = rc_final
    save_ledger(out, ledger, ledger_name)
    return rc_final


def _say(rec: dict) -> None:
    print(f"[move {rec['move']:>2}] {rec['name']:<44} {rec.get('status', '')}" + (f"  ({rec['stale']})" if rec.get("stale") else ""))


def print_status(out: Path, name: str = LEDGER) -> int:
    p = Path(out) / name
    if not p.exists():
        print(f"missing input: {p}", file=sys.stderr)
        return 2
    ledger = read_json(p)
    print(f"driver_state: {p}")
    print(f"runs: {len(ledger.get('runs', []))}")
    for rec in ledger.get("commands", []):
        print(f"  [move {rec.get('move'):>2}] {rec.get('name', ''):<44} {rec.get('status', '')}  "
              f"{rec.get('started_at', '')} .. {rec.get('finished_at', '')}")
    return 0


def print_plan(plan: list[dict], moves: str) -> None:
    sel = set(parse_moves(moves))
    for c in plan:
        if c["move"] in sel:
            argv = "(internal) " + c["sub"] + " " + " ".join(c["args"]) if c["internal"] else \
                f"python3 -m {PACKAGE_NAME}.{c['module']} " + (c["sub"] + " " if c["sub"] else "") + " ".join(c["args"])
            print(f"[move {c['move']:>2}] {c['name']:<44} {argv}")


def _add_run_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=None, help="the retention root (required when move 0 runs)")
    ap.add_argument("--cells-csv", default=None)
    ap.add_argument("--moves", default="0-14",
                    help="moves to run (default 0-14); move 15 (the optional LORO luck checks), move 16 (the corrected instrument check, "
                         "the idle common-ground test and the second Table 2, added 2026-10-05) and move 17 (the new-block test on the five "
                         "readings, added 2026-10-06) run only when named")
    # epoch 2, move 14 (SPEC_epoch2.md Part 1.7 (b); builder A): the comparators' declared defaults and the Law pass's job count
    ap.add_argument("--delta-th-default", type=float, default=0.04, help="Dhodapkar-Smith delta_th default (AA 2026-09-17: 0.04)")
    ap.add_argument("--law-x-default", type=int, default=4, help="Law 2010 X default (SPEC_epoch2 Part 4 item 6)")
    ap.add_argument("--savoldi-rows", default="all_after_head_drop", choices=("all_after_head_drop", "rung_series"))
    ap.add_argument("--comparator-jobs", type=int, default=None, help="processes for the Law trajectory pass (default: --n-jobs)")
    ap.add_argument("--assume-failed-zero", action="store_true")
    ap.add_argument("--assume-reason", default=None)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--null-perm", type=int, default=500)
    ap.add_argument("--null-splits", default="loko,loro,within_trace")
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="record every command in the ledger without running it")
    ap.add_argument("--skip-missing-modules", action="store_true",
                    help="record `not run: module absent` and continue when a module is not on disk (smoke tests only)")
    ap.add_argument("--only-modules", default=None,
                    help="comma-separated module names to run (tables,figures,latex_skeleton,driver,...); the others are recorded as not run")
    ap.add_argument("--persist-side", default=None, choices=[None, "t", "t+1"])
    ap.add_argument("--failed-counts", default=None)
    ap.add_argument("--keep-first-pairs", default=None,
                    help="CSV path, keep_first_pairs, reason (AA A8): move 1 reads only the first N pairs of each listed cell")
    ap.add_argument("--seed-map", default=None,
                    help="CSV path, seed[, source] (declared/seed_map.csv): move 0 takes each listed kernel cell's seed from it, not the cut folder name")
    ap.add_argument("--table8-rung", default="combined", choices=list(RUNGS))
    ap.add_argument("--piano-cell", default=None)
    ap.add_argument("--piano-stride", type=int, default=16)
    ap.add_argument("--standalone-tex", default=None, help="also write the LaTeX skeleton to this path")
    # epoch 2 (SPEC_epoch2 section 4): C1's rule, passed to `preconditions` only when set (its own default is auto)
    ap.add_argument("--c1-rule", default=None, choices=[None, "auto", "idle_floor", "absolute", "legacy_apf_max"],
                    help="C1 for kernel cells (AD 2026-09-17): auto (default of the preconditions command) | idle_floor | absolute | legacy_apf_max")
    ap.add_argument("--c1-abs-fraction", type=float, default=None, help="the absolute rule's fraction of memory (default 0.001 = 262 pages)")
    ap.add_argument("--c1-idle-percentile", type=float, default=None, help="the idle floor's percentile of K (default 95, G-K0's edge)")
    ap.add_argument("--c1-activity-min", type=float, default=None, help="the inherited apf_max threshold, used only under --c1-rule legacy_apf_max")
    ap.add_argument("--c1-activity-min-pages", type=int, default=None,
                    help="the absolute rule's page count (AA T1: 200; SPEC_epoch2 B1), passed to preconditions when set")
    # SPEC_epoch2 B10, B12, B19, B25, B26 (builder B's flags; read with getattr in build_plan)
    ap.add_argument("--gl2-rerun", default="auto", choices=("auto", "manual"),
                    help="run SPEC 3.7.4's feature-drop re-run after a G-L (ii) refusal (auto) or leave it to the author (manual)")
    ap.add_argument("--duration-s", type=float, default=600, help="every cell's declared duration for extract all (600 on the real corpus)")
    ap.add_argument("--wapf-norm", default="median_K", choices=("median_K", "median_self"), help="wAPF's normalization for features and grid")
    ap.add_argument("--n-estimators", type=int, default=300, help="forest size passed to gord, splits, gf, gx, gdim, gm")
    ap.add_argument("--gord-n-order-perm", type=int, default=20, help="G-ORD order shuffles per cell (gord --n-order-perm)")
    ap.add_argument("--gord-null-perm", type=int, default=100, help="G-ORD label-shuffle null size (gord --null-perm)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="plan11 driver: al-Kindi's moves in order (builder 3)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); _add_run_args(r)
    s = sub.add_parser("status"); s.add_argument("--out", required=True)
    pl = sub.add_parser("plan"); _add_run_args(pl)
    o = ap.parse_args(argv)
    if o.cmd == "status":
        return print_status(Path(o.out))
    try:
        moves = parse_moves(o.moves)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    if o.assume_failed_zero and not o.assume_reason:
        print("--assume-failed-zero requires --assume-reason TEXT (SPEC 3.3.2; section 8 item 37)", file=sys.stderr)
        return 2
    if 0 in moves and not o.root and o.cmd == "run":
        print("missing input: --root is required when move 0 (extract index) runs", file=sys.stderr)
        return 2
    plan = build_plan(o)
    if o.cmd == "plan":
        print_plan(plan, o.moves)
        return 0
    if 1 in moves and 0 not in moves and not (Path(o.cells_csv or Path(o.out) / "cells.csv")).exists():
        print(f"missing input: {Path(o.cells_csv or Path(o.out) / 'cells.csv')}", file=sys.stderr)
        return 2
    try:
        return run_plan(o, plan)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
