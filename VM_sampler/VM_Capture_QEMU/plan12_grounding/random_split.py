#!/usr/bin/env python3
"""random_split.py -- the random 80/20 split of the new-block test, beside the run of plan12_grounding (2026-10-10;
move 18 in the console's Grounding paper tab, after the time-shuffled control (17)).

  python3 -m plan12_grounding.random_split run --run <main run out> --step 1|2 [--out <dir>] [--seen 0.8]
        [--cuts declared,measured] [--encoding-out <D2>] [--n-jobs N] [--n-estimators N] [--bootstrap 1000]
        [--seed-split 20261010] [--newblocks-out <dir>] [--force] [--dry-run]
  (run from VM_sampler/VM_Capture_QEMU/)

THE QUESTION. The new-block test (new_blocks.py, move 12) trains on the first 80% of every recording's windows in
time order and names the blocks that come after a gap. How much of what it scores is the ORDER of the split, that
is, having to name the future of a recording from its past? This test keeps everything of move 12 and changes
one thing: per recording and cut, the windows go to training (80%) and testing (20%) AT RANDOM, seeded. A test
window then has training windows on both sides of it in time, and a neighbouring training window shares half its
pairs with it (W64_H32: windows start every 32 pairs and span 64). Two steps separate the two effects:
  step 1  the plain random split;
  step 2  the same split (the same test windows), with every training window that shares a pair with a test
          window REMOVED from training; the test says how many windows that removes, per recording and in all.
So step 1 against move 12 is "random against time order, pairs shared"; step 2 against move 12 is "random against
time order, no pair shared"; step 1 against step 2 is what the shared pairs alone are worth.

EXACTLY MOVE 12'S, imported from new_blocks.py and never copied where a function exists: the data (the run's
admissible recordings with a complete series, both cuts, decision D3's window, the feature sets E0, E1, E2 and E_new
built by idle_class.build_encodings_13 as move 6 builds them, move 6's E0 gate and its check for idle), the three
levels (archetype with idle as "idle" and no G-K0 relabel, kernel, run), the forest and its settings (move 6's trees,
processes and seed), the dimension rule and the B1-G3 quarantine with the test window as the unit (new_blocks.
fit_level), the vote and the scores on single test windows (pool_votes with the pool of 1, score_pools: accuracy,
macro recall over the classes present, chance 1/K, the majority class, recall per kernel), the run level's right
run / same kernel other run / other kernel (run_level_shares), the margins E1-E0, E2-E1, E_new-E0 with move 12's
bootstrap over recordings and its seed (bootstrap_margin, SEED_BOOTSTRAP), the confusion at the kernel level and
the figures (new_blocks' own figure functions, their titles reworded from "new blocks" to "test windows").
The data-building lines of new_blocks.one_cut that precede its split are repeated here line for line (they are not
a function there); the split itself is this module's.

NOT APPLICABLE HERE, said in the outputs and replaced by nothing: the pools of 2 and 3 consecutive new blocks
(the test windows are scattered at random through a recording, so no consecutive run of test windows exists by
design: single test windows only), the accuracy by position after the gap (there is no gap), and the label-shuffle
null, for move 12's reason (labels shuffled consistently within a recording are learned from that recording's own
training windows; chance and the majority class are the baselines).

THE SPLIT, per recording and cut (--seen 0.8, --seed-split 20261010)
The recording's windows in time order; n_train = round(seen * n) as move 12 counts its seen windows; the n - n_train
test windows are drawn without replacement by numpy's default_rng seeded with [--seed-split, cut, sha256(cell)[:4]]
(the same test windows in both steps and whatever --cuts or recordings are selected); the test windows are numbered
1, 2, 3 ... in time order (the block id "<cell>|b<k>", as move 12's). Step 2 then removes from training every
window whose pair range start .. start + W - 1 meets a test window's (|start difference| < W); split.csv says how
many per recording, summary.json in all.

THE OUTPUTS, under <out>/<step>/cut<C>/ (<step> = step1_random or step2_random_nooverlap; <out> defaults to <run>_randomsplit)
  split.csv             per recording: the windows, training, test, removed (step 2), the test windows' positions
  split_indices.npz     the row indices of the split (training, test, removed), the block ids, the window and pair starts
  scores.csv            per level and feature set: accuracy, macro recall, chance, majority, the test window count,
                        the class count, the feature counts, the quarantine, the training windows (and those removed),
                        the null column (move 12's reason), and MOVE 12's single-block scores beside (its scores.csv,
                        pool size 1, read from <run>_newblocks; "missing" when move 12 has not run)
  recall_per_kernel.csv at every level, per kernel and idle, with move 12's single-block recall beside
  run_level.csv         the run level, per kernel: right run, same kernel other run, other kernel; move 12's beside
  margins.csv           E1-E0, E2-E1, E_new-E0 with the 95% bootstrap interval over recordings; move 12's beside
  predictions_<level>_<E>.csv   every test window: cell, kernel, archetype, rep, position, window start, pair range,
                        truth, prediction, the predicted class's probability and the true class's
  confusion_kernel_<E>.csv (and .svg for E2 and E_new), accuracy_by_level.svg/.csv (this step against move 12),
  recall_per_kernel.svg/.csv, excluded_cells.csv, summary.json
  <out>/<step>/step.json, <out>/e0_idle_check.json, <out>/record.json (one entry per run: command, inputs with
  sha256, params, the code fingerprint of this module's closure, status; a finished step is skipped unless --force
  or an input, a parameter or the code changed). Move 12's results are read, not hashed.

THE RULES THAT PROTECT THE LIVE RUN (as new_blocks.py)
- ONE new file; nothing existing in plan12_grounding/ is edited; new_blocks.py is imported as it is (its recorded
  code fingerprint stays); nothing imports this module. The run folder and <run>_newblocks are read only here;
  --out inside the run is refused (exit 2). The test refuses to start (exit 3) while <run>/.driver.lock belongs to
  a live process; --dry-run skips that refusal, writes nothing and prints the plan (per cut: the windows, the
  training and test windows, and for step 2 how many training windows the removal takes, counted from the series
  with regular window starts; the run's split.csv is the record). One writer per output folder:
  <out>/.random_split.lock; run_moves.install_sigterm.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

from plan12_grounding import __version__, toolkit_fingerprint  # noqa: E402
from plan12_grounding import classify as C  # noqa: E402  (move 6's ENCODINGS, ENCODING_TEXT, ENC_COLOUR)
from plan12_grounding import idle_class as I  # noqa: E402  (build_encodings_13, the E0 gate and check, the lock state; imported, never edited)
from plan12_grounding import new_blocks as NB  # noqa: E402  (move 12: fit_level, pool_votes, score_pools, run_level_shares, bootstrap_margin, the figures, the record pattern)
from plan12_grounding.run_moves import _pid_alive, code_fingerprint, install_sigterm, now_iso, read_json, write_json  # noqa: E402
from plan12_grounding.stats import cut_series  # noqa: E402
from plan12_grounding.figures import svg_open, write_csv  # noqa: E402
from plan11_encoding_ladder import series as S11  # noqa: E402
from plan11_encoding_ladder import verdicts as V  # noqa: E402
from plan11_encoding_ladder.nulls import SEED_FOREST  # noqa: E402

CITATION = ("plan12_grounding/new_blocks.py (move 12: the data, the levels, fit_level with the dimension rule and B1-G3, pool_votes, score_pools, run_level_shares, "
            "bootstrap_margin, the figures; its one_cut's data-building lines repeated here); the author's design of the random split (2026-10-10): 80/20 at random "
            "per recording and cut, seeded; step 2 removes the training windows that share a pair with a test window")
STEPS = {1: ("step1_random", "the plain random split: round(seen * n) of each recording's windows train, the rest test, drawn at random"),
         2: ("step2_random_nooverlap", "the same random split, every training window that shares a pair with a test window removed from training")}
LEVELS = NB.LEVELS
LEVEL_TEXT = NB.LEVEL_TEXT
ENCODINGS = NB.ENCODINGS
MARGIN_PAIRS = NB.MARGIN_PAIRS
IDLE_LABEL = NB.IDLE_LABEL
HIGHLIGHT = NB.HIGHLIGHT
ENC_COLOUR = C.ENC_COLOUR
OUT_SUFFIX = "_randomsplit"
LOCK_NAME = ".random_split.lock"
RECORD_NAME = "record.json"
SEED_SPLIT = 20261010                                     # the split's own seed (the date of the design); the forest keeps move 6's, the bootstrap move 12's
DEFAULT_SEEN, DEFAULT_BOOTSTRAP = NB.DEFAULT_SEEN, NB.DEFAULT_BOOTSTRAP
EXIT_OK, EXIT_ERROR, EXIT_MISSING, EXIT_REFUSED = NB.EXIT_OK, NB.EXIT_ERROR, NB.EXIT_MISSING, NB.EXIT_REFUSED
NO_NULL = NB.NO_NULL
POOLS_NOTE = V.not_applicable("pools of 2 and 3 consecutive new blocks: the test windows are scattered at random through a recording, so no consecutive run of test "
                              "windows exists by design; single test windows only")
POSITION_NOTE = V.not_applicable("the accuracy by position after the gap: there is no gap in a random split; the test windows are numbered in time order for their ids only")
SPLIT_RULE = ("per recording and cut: the windows in time order; n_train = round(seen * n); n - n_train test windows drawn without replacement by numpy default_rng([seed_split, "
              "cut, sha256(cell_id)[:4]]).choice; the test windows numbered 1, 2, 3 ... in time order; step 2 removes every training window whose pair range "
              "start .. start + W - 1 meets a test window's (|start difference| < W)")
MOVE6_RULES = {**NB.MOVE6_RULES,
               "splits": "not applicable: move 6's LORO, LOKO and within-trace folds are replaced by the random split of this test (" + SPLIT_RULE + "), which no plan11 split expresses",
               "unit": "replaced: move 6's unit is the cell (majority vote over all its windows); here the unit is the single test window (move 12's single block); " + POOLS_NOTE}
Stop = NB.Stop


def fmt(x) -> str:
    return I.fmt(x)


def cell_seed(cell_id: str) -> int:
    return int.from_bytes(hashlib.sha256(cell_id.encode("utf-8")).digest()[:4], "big")


# ---------------------------------------------------------------------------------------------
# the split: training and test at random per recording; step 2's removal
# ---------------------------------------------------------------------------------------------
def random_split_rows(lab: dict, pair_start: np.ndarray, W: int, seen: float, seed: int, cut: int, step: int) -> dict:
    """Per cell (first-seen order): the row indices of its windows in time order; round(seen * n) train, the rest test
    at random (seeded by the cell and the cut); step 2 removes the training windows that share a pair with a test
    window. Returns new_blocks.split_rows' shape ({cells, per_cell, train_idx, test_idx, block_pos, block_id,
    n_new_total}) plus removed_idx and n_removed_total."""
    cell_id = lab["cell_id"]
    cells = list(dict.fromkeys(cell_id.tolist()))
    per_cell, train, test, removed, pos, bid = {}, [], [], [], [], []
    for c in cells:
        idx = np.flatnonzero(cell_id == c)
        order = np.argsort(lab["win_start"][idx], kind="stable")
        idx = idx[order]
        n = int(idx.size)
        n_train = int(np.floor(seen * n + 0.5))           # round(seen * n), half up, as move 12 counts its seen windows
        n_test = n - n_train
        rng = np.random.default_rng([int(seed), int(cut), cell_seed(c)])
        test_pos = np.sort(rng.choice(n, size=n_test, replace=False)) if n_test > 0 else np.zeros(0, dtype=np.int64)
        is_test = np.zeros(n, dtype=bool)
        is_test[test_pos] = True
        train_pos = np.flatnonzero(~is_test)
        removed_pos = np.zeros(0, dtype=np.int64)
        if step == 2 and n_test and train_pos.size:
            ps_tr, ps_te = pair_start[idx[train_pos]], pair_start[idx[test_pos]]
            shares = np.abs(ps_tr[:, None] - ps_te[None, :]).min(axis=1) < W      # the ranges start .. start + W - 1 meet
            removed_pos, train_pos = train_pos[shares], train_pos[~shares]
        per_cell[c] = {"cell_id": c, "n_windows": n, "n_train_drawn": n_train, "n_test": n_test, "n_removed": int(removed_pos.size), "n_train_used": int(train_pos.size),
                       "test_positions": " ".join(str(int(p)) for p in test_pos.tolist()), "removed_positions": " ".join(str(int(p)) for p in removed_pos.tolist()),
                       "first_pair": int(pair_start[idx[0]]), "last_pair": int(pair_start[idx[-1]] + W - 1),
                       "note": "" if n_test else "no test window: too few windows"}
        train += idx[train_pos].tolist()
        removed += idx[removed_pos].tolist()
        for k, j in enumerate(idx[test_pos].tolist(), start=1):
            test.append(j); pos.append(k); bid.append(f"{c}|b{k}")
    return {"cells": cells, "per_cell": per_cell, "train_idx": np.array(train, dtype=np.int64), "test_idx": np.array(test, dtype=np.int64),
            "removed_idx": np.array(removed, dtype=np.int64), "block_pos": np.array(pos, dtype=np.int64), "block_id": np.array(bid, dtype=str),
            "n_new_total": len(test), "n_removed_total": len(removed)}


def plan_counts(runs: list[dict], cut: int, win: dict, seen: float, seed: int, step: int) -> dict:
    """The dry run's counts from the series alone (no feature built): windows, training, test and, for step 2, the
    removal with regular window starts (window i starts i * H pairs after the first: |i - j| * H < W shares a pair)."""
    W, H = int(win["W"] or 0), int(win["H"] or 0)
    tot = {"recordings": 0, "windows": 0, "train": 0, "test": 0, "removed": 0, "recordings_without_test_window": 0}
    for r in runs:
        n_rows = int(cut_series(r, cut)["pair"].size)
        n = S11.n_windows(n_rows, W, H) if W else (1 if n_rows else 0)
        n_train = int(np.floor(seen * n + 0.5))
        n_test = n - n_train
        rng = np.random.default_rng([int(seed), int(cut), cell_seed(r["cell_id"])])
        test_pos = np.sort(rng.choice(n, size=n_test, replace=False)) if n_test > 0 else np.zeros(0, dtype=np.int64)
        is_test = np.zeros(n, dtype=bool); is_test[test_pos] = True
        train_pos = np.flatnonzero(~is_test)
        n_removed = int(np.sum(np.abs(train_pos[:, None] - test_pos[None, :]).min(axis=1) * H < W)) if (step == 2 and n_test and train_pos.size and H) else 0
        tot["recordings"] += 1; tot["windows"] += n; tot["train"] += n_train - n_removed; tot["test"] += n_test; tot["removed"] += n_removed
        if n_test == 0:
            tot["recordings_without_test_window"] += 1
    return tot


# ---------------------------------------------------------------------------------------------
# move 12's single-block results, read as they stand
# ---------------------------------------------------------------------------------------------
def _rows(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    with open(p, newline="") as fh:
        return [r for r in csv.DictReader(fh) if r.get("pool_size", "1") == "1"]


def move12_results(nb_out: Path, cut: int) -> dict:
    d = nb_out / f"cut{cut}"
    if not (d / "scores.csv").is_file():
        return {"status": "missing (move 12 has not run)", "folder": str(d), "scores": {}, "recall": {}, "run_level": {}, "margins": {}}
    return {"status": "ok", "folder": str(d),
            "scores": {(r["level"], r["feature_set"]): r for r in _rows(d / "scores.csv")},
            "recall": {(r["level"], r["feature_set"], r["kernel"]): r for r in _rows(d / "recall_per_kernel.csv")},
            "run_level": {(r["feature_set"], r["kernel"]): r for r in _rows(d / "run_level.csv")},
            "margins": {(r["level"], r["comparison"]): r for r in _rows(d / "margins.csv")}}


def _f(x):
    try:
        return float(x) if x not in (None, "") else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------
def relabel(svg: str) -> str:
    """new_blocks' figures speak of new blocks; here the unit is the random test window."""
    return svg.replace("New blocks named right", "Test windows named right").replace("single new blocks", "single test windows (random split)") \
              .replace("new blocks", "test windows").replace("new block", "test window")


def accuracy_by_level_svg(scores: dict, m12: dict, cut: int, e_sets: tuple, step_text: str) -> str:
    """Three panels (the levels); per feature set this step's accuracy on single test windows as a bar and move 12's
    single-block accuracy as an outlined bar beside it; chance 1/K grey, the majority class dashed red (this step's)."""
    W_, H_ = 940, 330
    out = svg_open(W_, H_, f"Test windows named right, by level and feature set, cut of {cut} pairs: {step_text[:60]}",
                   "filled bar: this step (random split, single test windows); outlined bar: move 12 (time split, single new blocks); grey line: chance 1/K; dashed red: the majority class")
    x0, y0, w, h = 60, 60, 270, 200
    for li, level in enumerate(LEVELS):
        px = x0 + li * (w + 25)
        out.append(f'<text x="{px}" y="{y0 - 6}" font-weight="bold">{level}</text>')
        out.append(f'<rect x="{px}" y="{y0}" width="{w}" height="{h}" fill="none" stroke="#ccc"/>')
        for t in (0.0, 0.5, 1.0):
            yy = y0 + h - h * t
            out.append(f'<line x1="{px}" y1="{yy:.1f}" x2="{px + w}" y2="{yy:.1f}" stroke="#eee"/><text x="{px - 4}" y="{yy + 3:.1f}" text-anchor="end">{t:.1f}</text>')
        bw = w / len(ENCODINGS)
        for ei, enc in enumerate(ENCODINGS):
            bx = px + ei * bw + 6
            half = (bw - 14) / 2
            s1 = ((scores.get(level) or {}).get(enc) or {}).get(1) or {}
            acc = s1.get("accuracy")
            if enc not in e_sets or acc is None:
                out.append(f'<text x="{bx + half / 2:.1f}" y="{y0 + h / 2:.1f}" text-anchor="middle" fill="#999" font-size="9">not run</text>')
            else:
                out.append(f'<rect x="{bx:.1f}" y="{y0 + h - h * acc:.1f}" width="{half - 2:.1f}" height="{h * acc:.1f}" fill="{ENC_COLOUR[enc]}" fill-opacity="0.85"/>')
                out.append(f'<text x="{bx + half / 2:.1f}" y="{y0 + h - h * acc - 3:.1f}" text-anchor="middle" font-size="9">{acc:.2f}</text>')
            m = (m12.get("scores") or {}).get((level, enc)) or {}
            a12 = _f(m.get("accuracy"))
            if a12 is None:
                out.append(f'<text x="{bx + half + half / 2:.1f}" y="{y0 + h / 2 + 12:.1f}" text-anchor="middle" fill="#999" font-size="8">move 12 {"missing" if m12.get("status") != "ok" else "not run"}</text>')
            else:
                out.append(f'<rect x="{bx + half:.1f}" y="{y0 + h - h * a12:.1f}" width="{half - 2:.1f}" height="{h * a12:.1f}" fill="none" stroke="{ENC_COLOUR[enc]}" stroke-width="1.5"/>')
                out.append(f'<text x="{bx + half + half / 2:.1f}" y="{y0 + h - h * a12 - 3:.1f}" text-anchor="middle" font-size="9" fill="#555">{a12:.2f}</text>')
            if s1.get("chance") is not None:
                cy = y0 + h - h * float(s1["chance"])
                out.append(f'<line x1="{bx - 3:.1f}" y1="{cy:.1f}" x2="{bx + 2 * half:.1f}" y2="{cy:.1f}" stroke="#888"/>')
            if s1.get("majority") is not None:
                my = y0 + h - h * float(s1["majority"])
                out.append(f'<line x1="{bx - 3:.1f}" y1="{my:.1f}" x2="{bx + 2 * half:.1f}" y2="{my:.1f}" stroke="#b03a2e" stroke-dasharray="3,2"/>')
            out.append(f'<text x="{bx + half:.1f}" y="{y0 + h + 12}" text-anchor="middle">{enc}</text>')
        any_s = next((v for e in ENCODINGS for v in [((scores.get(level) or {}).get(e) or {}).get(1)] if v), {})
        out.append(f'<text x="{px}" y="{y0 + h + 26}" fill="#555" font-size="9">{any_s.get("n_classes", "?")} classes, {any_s.get("n_pools", "?")} test windows</text>')
    out.append(f'<text x="{x0}" y="{H_ - 22}" fill="#555">{"; ".join(f"{e}: {C.ENCODING_TEXT[e]}" for e in ENCODINGS)}</text>')
    out.append(f'<text x="{x0}" y="{H_ - 10}" fill="#555">{NO_NULL[:150]}</text></svg>')
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------
# one cut of one step
# ---------------------------------------------------------------------------------------------
def one_cut(step_dir: Path, step: int, cut_name: str, cut: int, runs: list[dict], enc_out: Path | None, enc_rows, win: dict, pair_excluded: set, mode: str,
            e_sets: tuple, reason: str | None, run: Path, nb_out: Path, *, seen: float, seed_split: int, n_boot: int, n_jobs: int, n_est: int, seed: int) -> dict:
    step_name, step_text = STEPS[step]
    d = step_dir / f"cut{cut}"
    d.mkdir(parents=True, exist_ok=True)
    t_cut = time.time()
    # --- the data, as new_blocks.one_cut builds it before its split (repeated line for line; cited above) ---
    m6_order = I.move6_cells(run, cut)
    if mode == I.MODE_E_NEW_IDLE:
        keep = set(m6_order or [r["cell_id"] for r in runs if r["role"] == "kernel"])
        enc = I.build_encodings_13([r for r in runs if r["role"] == "idle" or r["cell_id"] in keep], None, enc_rows, cut, win, {}, set())
    else:
        enc = I.build_encodings_13(runs, enc_out, enc_rows, cut, win, {}, pair_excluded)      # relabel = {}: no G-K0 relabel (the author's design)
    X, names, lab = enc["X"], enc["names"], enc["lab"]
    if lab["n"] == 0:
        raise Stop(EXIT_ERROR, f"stopped: no window at cut {cut} (W {win['W']}, H {win['H']}); nothing to split")
    if win.get("whole_cell"):
        raise Stop(EXIT_ERROR, "stopped: the D3 window is the whole cell (one window per recording), so no recording can be split into training and test windows")
    W = int(win["W"])
    pair_start = np.array(enc["meta"]["pair_start"], dtype=np.int64)
    kernel_rows = lab["kernel"].astype(str)
    arche_rows = np.where(kernel_rows == IDLE_LABEL, IDLE_LABEL, lab["archetype"].astype(str)).astype(str)
    run_rows = lab["cell_id"].astype(str)
    labels_of = {"archetype": arche_rows, "kernel": kernel_rows, "run": run_rows}
    # --- the split: this module's ---
    sp = random_split_rows(lab, pair_start, W, seen, seed_split, cut, step)
    sp["cell_id_rows"] = run_rows
    cells = sp["cells"]
    kernel_of_cell = {c: str(kernel_rows[np.flatnonzero(run_rows == c)[0]]) for c in cells}
    arche_of_cell = {c: str(arche_rows[np.flatnonzero(run_rows == c)[0]]) for c in cells}
    rep_of_cell = {c: int(lab["rep"][np.flatnonzero(run_rows == c)[0]]) for c in cells}
    kernel_cells = [c for c in cells if kernel_of_cell[c] != IDLE_LABEL]
    idle_cells = [c for c in cells if kernel_of_cell[c] == IDLE_LABEL]
    labels_kernel = I.class_order(set(kernel_of_cell.values()))
    np.savez_compressed(d / "split_indices.npz", train_idx=sp["train_idx"], test_idx=sp["test_idx"], removed_idx=sp["removed_idx"], block_id=sp["block_id"], block_pos=sp["block_pos"],
                        cell_id=lab["cell_id"], win_start=lab["win_start"], pair_start=pair_start, W=np.int64(W), H=np.int64(win["H"] or 0), cut=np.int64(cut), step=np.int64(step),
                        seed_split=np.int64(seed_split))
    write_csv(d / "excluded_cells.csv", ["cell_id", "reason"], [[e["cell_id"], e["reason"]] for e in enc["excluded"]])
    write_csv(d / "split.csv", ["cell_id", "kernel", "n_windows", "n_train_drawn", "n_removed", "n_train_used", "n_test", "first_pair", "last_pair", "test_positions", "removed_positions", "note"],
              [[c, kernel_of_cell[c], *[sp["per_cell"][c][k] for k in ("n_windows", "n_train_drawn", "n_removed", "n_train_used", "n_test", "first_pair", "last_pair", "test_positions",
                                                                       "removed_positions", "note")]] for c in cells])
    print(f"[randomsplit] step {step} cut {cut}: {len(cells)} recordings ({len(kernel_cells)} kernel + {len(idle_cells)} idle), {int(lab['n'])} windows of W {W}; training "
          f"{int(sp['train_idx'].size)}" + (f" after {sp['n_removed_total']} removed for sharing a pair with a test window" if step == 2 else "") + f", test {sp['n_new_total']} "
          f"({min(v['n_test'] for v in sp['per_cell'].values())} to {max(v['n_test'] for v in sp['per_cell'].values())} per recording)", flush=True)
    majority_class = {}
    for level in LEVELS:
        counts: dict = {}
        for c in cells:
            lv = {"archetype": arche_of_cell, "kernel": kernel_of_cell, "run": {c: c}}[level][c]
            counts[lv] = counts.get(lv, 0) + 1
        majority_class[level] = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    te = sp["test_idx"]
    pools = [1]
    results: dict = {lv: {} for lv in LEVELS}
    pools_by: dict = {lv: {} for lv in LEVELS}
    for level in LEVELS:
        y = labels_of[level]
        for e in ENCODINGS:
            if e not in e_sets:
                results[level][e] = {"status": reason, "scores": {1: {"accuracy": None, "macro_recall": None, "n_pools": 0}}}
                print(f"[randomsplit] step {step} cut {cut} {level:<9} {e:<5} {str(reason)[:80]}", flush=True)
                continue
            cols = enc["cols"][e]
            t0 = time.time()
            fit = NB.fit_level(X[:, cols], [names[j] for j in cols], y, sp, seed=seed, n_jobs=n_jobs, n_est=n_est)
            use = fit["use"]
            if "pred" not in use:
                results[level][e] = {"status": fit["status"], "scores": {1: {"accuracy": None, "macro_recall": None, "n_pools": 0}},
                                     "quarantine": fit["quarantine"], "quarantined_features": fit["quarantined_features"], "score_source": fit["score_source"]}
                continue
            pv = NB.pool_votes(sp, use["pred"], use["proba"], use["classes"], y[te], pools)
            pools_by[level][e] = pv
            sc = {1: NB.score_pools(pv[1], fit["classes_train"], kernel_of_cell, majority_class[level])}
            res = {"status": "ok", "scores": sc, "feature_count": use["feature_count"], "feature_count_used": use["feature_count_used"], "reduced": use["reduced"],
                   "score_source": fit["score_source"], "quarantine": fit["quarantine"], "quarantined_features": fit["quarantined_features"],
                   "full_model": ({"accuracy": NB.score_pools(NB.pool_votes(sp, fit["full"]["pred"], fit["full"]["proba"], fit["full"]["classes"], y[te], [1])[1],
                                                              fit["classes_train"], kernel_of_cell, majority_class[level])["accuracy"]} if fit["with_quarantine"] else None),
                   "n_train_windows": fit["n_train_windows"], "n_train_cells": fit["n_train_cells"], "n_test_windows": fit["n_new_blocks"], "classes_train": fit["classes_train"],
                   "run_level": NB.run_level_shares(pv[1], kernel_of_cell) if level == "run" else None, "elapsed_s": round(time.time() - t0, 1)}
            ci = {c: i for i, c in enumerate(use["classes"])}
            res["blocks"] = [{"block_id": b, "cell_id": b.split("|b")[0], "position": int(sp["block_pos"][i]), "row": int(te[i]), "win_start": int(lab["win_start"][te[i]]),
                              "pair_start": int(pair_start[te[i]]), "pair_end": int(pair_start[te[i]] + W - 1), "y_true": str(y[te[i]]), "y_pred": str(use["pred"][i]),
                              "p_pred": float(use["proba"][i].max()), "p_true": float(use["proba"][i][ci[str(y[te[i]])]]) if str(y[te[i]]) in ci else None}
                             for i, b in enumerate(sp["block_id"].tolist())]
            results[level][e] = res
            s1 = sc[1]
            print(f"[randomsplit] step {step} cut {cut} {level:<9} {e:<5} acc {fmt(s1['accuracy'])} macro {fmt(s1['macro_recall'])} chance {fmt(s1['chance'])} majority {fmt(s1['majority'])}"
                  + f" | features {use['feature_count_used']}/{use['feature_count']}" + (f" quarantined {len(fit['quarantined_features'])}" if fit["quarantined_features"] else "")
                  + f" ({res['elapsed_s']} s)", flush=True)
    # tables, move 12's single-block results beside
    m12 = move12_results(nb_out, cut)
    score_rows = []
    for level in LEVELS:
        for e in ENCODINGS:
            r = results[level][e]
            s = (r.get("scores") or {}).get(1) or {}
            m = (m12["scores"].get((level, e)) or {})
            score_rows.append([level, e, r.get("status"), s.get("n_pools"), s.get("n_classes"), s.get("accuracy"), s.get("macro_recall"), s.get("chance"), s.get("majority"),
                               r.get("feature_count"), r.get("feature_count_used"), r.get("score_source"), ";".join(r.get("quarantined_features") or []),
                               r.get("n_train_windows"), sp["n_removed_total"] if step == 2 else 0, r.get("n_train_cells"), NO_NULL,
                               m.get("status") if m else m12["status"], _f(m.get("accuracy")), _f(m.get("macro_recall")), _f(m.get("chance")), _f(m.get("majority")),
                               m.get("n_pools"), m.get("n_train_windows")])
    write_csv(d / "scores.csv", ["level", "feature_set", "status", "n_test_windows", "n_classes", "accuracy", "macro_recall", "chance", "majority", "feature_count",
                                 "feature_count_used", "score_source", "quarantined_features", "n_train_windows", "n_train_windows_removed", "n_train_recordings", "null",
                                 "move12_status", "move12_accuracy", "move12_macro_recall", "move12_chance", "move12_majority", "move12_n_new_blocks", "move12_n_train_windows"], score_rows)
    rk_rows = []
    for level in LEVELS:
        for e in ENCODINGS:
            s = (results[level][e].get("scores") or {}).get(1) or {}
            for kern in labels_kernel:
                if kern in (s.get("recall_per_kernel") or {}):
                    m = m12["recall"].get((level, e, kern)) or {}
                    rk_rows.append([level, e, kern, s["recall_per_kernel"][kern], sum(1 for p in pools_by[level].get(e, {}).get(1, []) if kernel_of_cell[p["cell_id"]] == kern),
                                    _f(m.get("recall")), m.get("n_pools")])
    write_csv(d / "recall_per_kernel.csv", ["level", "feature_set", "kernel", "recall", "n_test_windows", "move12_recall", "move12_n_new_blocks"], rk_rows)
    rl_rows = []
    for e in ENCODINGS:
        r = results["run"][e]
        if not r.get("run_level"):
            continue
        for kern in labels_kernel:
            if kern in r["run_level"]:
                v = r["run_level"][kern]
                m = m12["run_level"].get((e, kern)) or {}
                rl_rows.append([e, kern, v["n_pools"], v["right_run"], v["same_kernel_other_run"], v["other_kernel"],
                                _f(m.get("right_run")), _f(m.get("same_kernel_other_run")), _f(m.get("other_kernel"))])
    write_csv(d / "run_level.csv", ["feature_set", "kernel", "n_test_windows", "right_run", "same_kernel_other_run", "other_kernel", "move12_right_run", "move12_same_kernel_other_run",
                                    "move12_other_kernel"], rl_rows)
    margin_rows, margins = [], {lv: {} for lv in LEVELS}
    for level in LEVELS:
        for b, a in MARGIN_PAIRS:
            m = m12["margins"].get((level, f"{b}-{a}")) or {}
            pb, pa = pools_by[level].get(b, {}).get(1), pools_by[level].get(a, {}).get(1)
            if pb is None or pa is None:
                margin_rows.append([level, f"{b}-{a}", None, None, None, None, None, None, None,
                                    "not run: " + str(results[level][a].get("status") if pa is None else results[level][b].get("status")),
                                    _f(m.get("delta_accuracy")), _f(m.get("ci95_lo")), _f(m.get("ci95_hi")), m.get("status") if m else m12["status"]])
                continue
            bm = NB.bootstrap_margin(pb, pa, cells, n_boot, NB.SEED_BOOTSTRAP)
            margins[level][f"{b}-{a}"] = bm
            margin_rows.append([level, f"{b}-{a}", bm["delta"], bm["ci_lo"], bm["ci_hi"], bm["n_pools"], bm["n_recordings"], bm["gained"], bm["lost"], bm["status"],
                                _f(m.get("delta_accuracy")), _f(m.get("ci95_lo")), _f(m.get("ci95_hi")), m.get("status") if m else m12["status"]])
    write_csv(d / "margins.csv", ["level", "comparison", "delta_accuracy", "ci95_lo", "ci95_hi", "n_test_windows", "n_recordings", "n_windows_gained", "n_windows_lost", "status",
                                  "move12_delta_accuracy", "move12_ci95_lo", "move12_ci95_hi", "move12_status"], margin_rows)
    for level in LEVELS:
        for e in ENCODINGS:
            r = results[level][e]
            if not r.get("blocks"):
                continue
            write_csv(d / f"predictions_{level}_{e}.csv", ["block_id", "cell_id", "kernel", "archetype", "rep", "position", "win_start", "pair_start", "pair_end", "y_true", "y_pred", "p_pred", "p_true"],
                      [[b["block_id"], b["cell_id"], kernel_of_cell[b["cell_id"]], arche_of_cell[b["cell_id"]], rep_of_cell[b["cell_id"]], b["position"], b["win_start"], b["pair_start"],
                        b["pair_end"], b["y_true"], b["y_pred"], b["p_pred"], b["p_true"]] for b in r["blocks"]])
    confusions = {}
    for e in ENCODINGS:
        r = results["kernel"][e]
        if not r.get("blocks"):
            continue
        pred = {b["block_id"]: {"y_true": b["y_true"], "y_pred": b["y_pred"]} for b in r["blocks"]}
        Cm = I.confusion_of(pred, labels_kernel)
        confusions[e] = Cm
        write_csv(d / f"confusion_kernel_{e}.csv", ["true\\predicted"] + labels_kernel, [[l] + Cm[i].tolist() for i, l in enumerate(labels_kernel)])
        if e in ("E2", "E_new"):
            (d / f"confusion_kernel_{e}.svg").write_text(relabel(NB.confusion_svg(labels_kernel, Cm, e, cut, r["scores"][1])))
    scores_for_fig = {lv: {e: (results[lv][e].get("scores") or {}) for e in ENCODINGS} for lv in LEVELS}
    (d / "accuracy_by_level.svg").write_text(accuracy_by_level_svg(scores_for_fig, m12, cut, e_sets, step_text))
    write_csv(d / "accuracy_by_level.csv", ["level", "feature_set", "accuracy", "macro_recall", "chance", "majority", "n_test_windows", "move12_accuracy"],
              [[lv, e, (scores_for_fig[lv][e].get(1) or {}).get("accuracy"), (scores_for_fig[lv][e].get(1) or {}).get("macro_recall"), (scores_for_fig[lv][e].get(1) or {}).get("chance"),
                (scores_for_fig[lv][e].get(1) or {}).get("majority"), (scores_for_fig[lv][e].get(1) or {}).get("n_pools"), _f((m12["scores"].get((lv, e)) or {}).get("accuracy"))]
               for lv in LEVELS for e in ENCODINGS])
    (d / "recall_per_kernel.svg").write_text(relabel(NB.recall_per_kernel_svg(scores_for_fig, labels_kernel, cut, e_sets)))
    write_csv(d / "recall_per_kernel_figure.csv", ["level", "feature_set", "kernel", "recall"],
              [[lv, e, kern, ((scores_for_fig[lv][e].get(1) or {}).get("recall_per_kernel") or {}).get(kern)] for lv in LEVELS for e in ENCODINGS for kern in labels_kernel])
    focus = {}
    for cls in HIGHLIGHT:
        focus[cls] = {}
        for e in ENCODINGS:
            r = results["kernel"][e]
            if r.get("blocks"):
                pred = {b["block_id"]: {"y_true": b["y_true"], "y_pred": b["y_pred"]} for b in r["blocks"]}
                focus[cls][e] = I.class_report(pred, cls)
            else:
                focus[cls][e] = {"status": r.get("status")}
    summary = {"schema": "plan12.random_split_cut.v1", "citation": CITATION, "cut": int(cut), "cut_name": cut_name, "step": step, "step_name": step_name, "step_text": step_text,
               "mode": mode, "feature_sets_run": list(e_sets), "reason_others_not_run": reason, "null": NO_NULL, "pools": POOLS_NOTE, "by_position": POSITION_NOTE,
               "move6_rules": MOVE6_RULES, "levels": LEVEL_TEXT, "window": win,
               "split": {"seen": seen, "seed_split": seed_split, "rule": SPLIT_RULE, "removal_applied": step == 2,
                         "n_recordings": len(cells), "n_kernel_recordings": len(kernel_cells), "n_idle_recordings": len(idle_cells), "n_windows": int(lab["n"]),
                         "n_train_windows_drawn": int(sum(v["n_train_drawn"] for v in sp["per_cell"].values())), "n_train_windows_removed": sp["n_removed_total"],
                         "n_train_windows_used": int(sp["train_idx"].size), "n_test_windows": sp["n_new_total"],
                         "removed_per_recording": {"min": min(v["n_removed"] for v in sp["per_cell"].values()), "max": max(v["n_removed"] for v in sp["per_cell"].values()),
                                                   "mean": float(np.mean([v["n_removed"] for v in sp["per_cell"].values()]))},
                         "test_per_recording": {str(k): sum(1 for v in sp["per_cell"].values() if v["n_test"] == k) for k in sorted({v["n_test"] for v in sp["per_cell"].values()})},
                         "recordings_without_test_window": [c for c in cells if sp["per_cell"][c]["n_test"] == 0]},
               "majority_class": majority_class, "classes": {"archetype": sorted(set(arche_of_cell.values())), "kernel": labels_kernel, "n_run": len(cells)},
               "n_excluded": len(enc["excluded"]), "excluded": enc["excluded"], "e0_available": enc["e0_available"], "n_e0_windows": enc["n_e0_windows"],
               "feature_counts": {e: len(enc["cols"][e]) for e in ENCODINGS}, "n_estimators": n_est, "n_jobs": n_jobs, "seed_forest": seed, "seed_bootstrap": NB.SEED_BOOTSTRAP, "n_bootstrap": n_boot,
               "move12": {"out": str(nb_out), "status": m12["status"], "folder": m12["folder"]},
               "scores": {lv: {e: {"status": results[lv][e].get("status"), "feature_count": results[lv][e].get("feature_count"), "feature_count_used": results[lv][e].get("feature_count_used"),
                                   "score_source": results[lv][e].get("score_source"), "quarantined_features": results[lv][e].get("quarantined_features"),
                                   "full_model": results[lv][e].get("full_model"), "elapsed_s": results[lv][e].get("elapsed_s"), "n_train_windows": results[lv][e].get("n_train_windows"),
                                   "single": {kk: vv for kk, vv in ((results[lv][e].get("scores") or {}).get(1) or {}).items() if kk not in ("recall_per_class",)},
                                   "move12_single": {k: _f((m12["scores"].get((lv, e)) or {}).get(k)) for k in ("accuracy", "macro_recall", "chance", "majority")}}
                               for e in ENCODINGS} for lv in LEVELS},
               "run_level": {e: results["run"][e].get("run_level") for e in ENCODINGS}, "margins": margins, "lexer": focus["lexer"], "idle": focus[IDLE_LABEL],
               "elapsed_s": round(time.time() - t_cut, 1), "files": sorted(p.name for p in d.iterdir())}
    write_json(d / "summary.json", summary)
    return summary


# ---------------------------------------------------------------------------------------------
# the step: the output folder, the locks, the record (new_blocks' pattern)
# ---------------------------------------------------------------------------------------------
def default_out(run: Path) -> Path:
    return run.parent / (run.name + OUT_SUFFIX)


def resolve_out(run: Path, out_arg: str | None) -> Path:
    out = Path(os.path.expanduser(out_arg)).resolve() if out_arg else default_out(run)
    if out == run or run in out.parents:
        raise Stop(EXIT_MISSING, f"refused: --out {out} lies inside the run folder {run}, which is read only here; the default is {default_out(run)}")
    return out


def resolve_nb_out(run: Path, arg: str | None) -> Path:
    return Path(os.path.expanduser(arg)).resolve() if arg else NB.default_out(run)


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
            raise Stop(EXIT_REFUSED, f"refused: another random_split run is writing {out} ({lock}: pid {pid}, started {old.get('started_at')}); one writer per output folder")
        print(f"[randomsplit] stale lock from pid {pid} ({old.get('started_at')}) taken over", flush=True)
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
        except Exception:                                  # noqa: BLE001
            p.rename(p.with_suffix(".json.unreadable_" + now_iso().replace(":", "")))
    return {"schema": "plan12.random_split.record.v1", "citation": CITATION, "package_version": __version__, "entries": []}


def save_record(out: Path, rec: dict) -> None:
    write_json(out / RECORD_NAME, rec)


def last_done(rec: dict, key: str) -> dict | None:
    for e in reversed(rec.get("entries") or []):
        if e.get("key") == key and e.get("status") == "done":
            return e
    return None


def code_identity() -> dict:
    fp = code_fingerprint("random_split")
    files = toolkit_fingerprint()["files"]
    return {"fingerprint": fp["sha256"], "modules": fp["modules"], "sha256": {f"{m}.py": files.get(f"{m}.py") for m in fp["modules"]}}


def run_move(o: argparse.Namespace) -> int:
    run = Path(os.path.expanduser(o.run)).resolve()
    if not (run / "cells.csv").is_file():
        raise Stop(EXIT_MISSING, f"missing input: {run / 'cells.csv'} (--run must name a plan12_grounding output folder with moves 0, 1 and 6 done)")
    step = int(o.step)
    step_name, step_text = STEPS[step]
    out = resolve_out(run, o.out)
    nb_out = resolve_nb_out(run, o.newblocks_out)
    seen, n_boot, seed_split = float(o.seen), int(o.bootstrap), int(o.seed_split)
    if not (0.0 < seen < 1.0) or n_boot < 1:
        raise Stop(EXIT_MISSING, f"--seen must lie in (0, 1) (given {seen}), --bootstrap 1 or more (given {n_boot})")
    lock = I.driver_lock_state(run)
    if lock["alive"] and not o.dry_run:
        raise Stop(EXIT_REFUSED, f"refused: a move is running on this run ({lock['path']}: pid {lock['pid']}, started {lock['started_at']}, {lock['argv_tail']}); "
                                 "this test never competes with a running move. Run again after it ends; --dry-run shows the plan meanwhile.")
    m6 = I.load_move6(run, o.encoding_out)
    cuts = I.parse_cuts(o.cuts, m6["cuts"])
    n_jobs = int(o.n_jobs) if o.n_jobs is not None else m6["n_jobs"]
    n_est = int(o.n_estimators) if o.n_estimators is not None else m6["n_estimators"]
    seed = SEED_FOREST + m6["seed_offset"]
    enc_out, win = m6["encoding_out"], m6["win"]
    runs, kernel_runs, idle_runs = I.select_runs(run)
    cell_ids = [r["cell_id"] for r in kernel_runs] + [r["cell_id"] for r in idle_runs]
    enc_rows, declared, relabel_, pair_excluded, e0_status, identity = I.e0_gate(run, enc_out, cell_ids)
    check = I.idle_e0_check(idle_runs, enc_out, enc_rows, pair_excluded, e0_status, win, cuts)
    mode, e_sets, reason = I.decide_mode(idle_runs, e0_status, check)
    n_kernels = len({r["kernel"] for r in kernel_runs})
    counts = {name: plan_counts(runs, cut, win, seen, seed_split, step) for name, cut in cuts.items()}
    m12_status = {name: move12_results(nb_out, cut)["status"] for name, cut in cuts.items()}
    print(f"[randomsplit] step {step}: {step_text}; seen {seen:g} of each recording's windows train, the rest test, at random (seed {seed_split}); single test windows only "
          f"({POOLS_NOTE[:60]}...); three levels (archetype, kernel, run); feature sets {', '.join(e_sets)}; {NO_NULL[:60]}...")
    print(f"[randomsplit] run {run} (read only); move 12's results {nb_out} (read only; " + "; ".join(f"{k}: {v}" for k, v in m12_status.items()) + f"); out {out}{' (exists)' if out.exists() else ' (to be created)'}")
    print(f"[randomsplit] driver lock: " + (f"held by pid {lock['pid']} ({'alive' if lock['alive'] else 'dead'}), started {lock['started_at']}: {lock['argv_tail']}" if lock["exists"] else "absent")
          + ("; the refusal is skipped by --dry-run" if (lock["alive"] and o.dry_run) else ""))
    print(f"[randomsplit] recordings found: {len(kernel_runs) + len(idle_runs)} = {len(kernel_runs)} kernel ({n_kernels} kernels) + {len(idle_runs)} idle; classes: archetype "
          f"{len({r['archetype'] for r in kernel_runs}) + (1 if idle_runs else 0)}, kernel {n_kernels + (1 if idle_runs else 0)}, run {len(kernel_runs) + len(idle_runs)}")
    print(f"[randomsplit] cuts: {', '.join(f'{k} = {v} pairs' for k, v in cuts.items())} (params.json); D2 encoding run: {enc_out}; G-K0 relabel NOT applied; pair-rung excluded {len(pair_excluded)}")
    print(f"[randomsplit] D3 window: {win['grid_id']} (W {win['W']}, H {win['H']}); forest: {n_est} trees, {n_jobs} processes, seed {seed} (move 6's values unless overridden); "
          f"bootstrap {n_boot} resamples, seed {NB.SEED_BOOTSTRAP}; E0 gate: {'E0 usable' if e0_status is None else e0_status}; mode: {mode}")
    for name, cut in cuts.items():
        c = counts[name]
        print(f"[randomsplit] cut {cut} ({name}): {c['recordings']} recordings, {c['windows']} windows, {c['train']} training" + (f" after {c['removed']} removed" if step == 2 else "")
              + f", {c['test']} test" + (f"; {c['recordings_without_test_window']} without a test window" if c["recordings_without_test_window"] else "") + " (counted from the series)")
    step_dir = out / step_name
    if o.dry_run:
        print(f"[randomsplit] dry run: would write {step_dir}/cut<C>/ (split.csv, split_indices.npz, scores.csv, recall_per_kernel.csv, run_level.csv, margins.csv, predictions_<level>_<E>.csv, "
              f"confusion_kernel_<E>.csv/.svg, accuracy_by_level.svg, recall_per_kernel.svg, summary.json), {step_dir / 'step.json'}, {out / 'e0_idle_check.json'} and {out / RECORD_NAME}; nothing written")
        return EXIT_OK
    out_lock = acquire_out_lock(out)
    code = code_identity()
    entry = {"key": step_name, "step": step, "command": sys.argv, "cwd": str(_HERE.parent), "started_at": now_iso(), "status": "running", "exit_code": None, "finished_at": None,
             "params": {"run": str(run), "out": str(out), "newblocks_out": str(nb_out), "cuts": cuts, "encoding_out": str(enc_out) if enc_out else None, "window": win, "seen": seen,
                        "seed_split": seed_split, "removal": step == 2, "bootstrap": n_boot, "seed_bootstrap": NB.SEED_BOOTSTRAP, "n_estimators": n_est, "n_jobs": n_jobs,
                        "seed_forest": seed, "seed_offset": m6["seed_offset"], "mode": mode, "feature_sets": list(e_sets), "reason_others_not_run": reason,
                        "n_kernel_cells": len(kernel_runs), "n_idle_cells": len(idle_runs), "force": bool(o.force)},
             "inputs_sha256": NB.inputs_identity(run, enc_out, declared, kernel_runs + idle_runs, cuts),
             "code": {"random_split.py": code["sha256"].get("random_split.py"), "fingerprint": code["fingerprint"], "modules": code["modules"], "sha256": code["sha256"],
                      "toolkit_fingerprint": toolkit_fingerprint()["sha256"]},
             "outputs": []}
    sig = {k: v for k, v in entry["params"].items() if k not in ("n_jobs", "force", "newblocks_out")}
    rec = load_record(out)
    prev = last_done(rec, step_name)
    outputs_exist = (step_dir / "step.json").is_file() and all((step_dir / f"cut{c}" / "summary.json").is_file() for c in cuts.values())
    try:
        if prev is not None and not o.force and outputs_exist:
            why = None
            if prev.get("inputs_sha256") != entry["inputs_sha256"]:
                why = "an input changed"
            elif {k: v for k, v in (prev.get("params") or {}).items() if k not in ("n_jobs", "force", "newblocks_out")} != sig:
                why = "the parameters changed"
            elif (prev.get("code") or {}).get("fingerprint") != code["fingerprint"]:
                why = "the code changed"
            if why is None:
                entry.update(status="skipped: outputs exist and inputs, parameters and code unchanged", exit_code=0, finished_at=now_iso(),
                             made_by={"finished_at": prev.get("finished_at"), "code_fingerprint": (prev.get("code") or {}).get("fingerprint")}, outputs=prev.get("outputs") or [])
                rec["entries"].append(entry)
                save_record(out, rec)
                print(f"[randomsplit] skipped: {step_dir} stands as made on {prev.get('finished_at')} (inputs, parameters and code unchanged; --force re-runs)")
                return EXIT_OK
            print(f"[randomsplit] re-running: {why} since {prev.get('finished_at')}")
        rec["entries"].append(entry)
        save_record(out, rec)
        step_dir.mkdir(parents=True, exist_ok=True)
        write_json(out / "e0_idle_check.json", {"schema": "plan12.random_split_e0_check.v1", "citation": CITATION, "written_at": now_iso(), "encoding_out": str(enc_out) if enc_out else None,
                                                "move6_e0_gate": e0_status or "E0 usable", "identity": {k: identity.get(k) for k in ("cell_id", "passed", "status", "reason", "checked_at")},
                                                "e0_declared_files": (declared or {}).get("n_files"), **check})
        t0 = time.time()
        results = [one_cut(step_dir, step, name, cut, runs, enc_out, enc_rows, win, pair_excluded, mode, e_sets, reason, run, nb_out,
                           seen=seen, seed_split=seed_split, n_boot=n_boot, n_jobs=n_jobs, n_est=n_est, seed=seed) for name, cut in cuts.items()]
        step_rec = {"schema": "plan12.random_split_step.v1", "citation": CITATION, "package_version": __version__, "written_at": now_iso(), "command": sys.argv, "step": step,
                    "step_name": step_name, "step_text": step_text, "question": __doc__.split("THE QUESTION.", 1)[1].split("EXACTLY MOVE 12'S", 1)[0].strip(),
                    "params": entry["params"], "cut_convention": m6["cut_convention"], "plan_counts": counts, "levels": LEVEL_TEXT, "null": NO_NULL, "pools": POOLS_NOTE,
                    "by_position": POSITION_NOTE, "move6_rules": MOVE6_RULES,
                    "cells": {"n_kernel": len(kernel_runs), "n_idle": len(idle_runs), "kernels": sorted({r["kernel"] for r in kernel_runs}), "idle_cells": [r["cell_id"] for r in idle_runs],
                              "idle_label": IDLE_LABEL, "idle_archetype_in_cells_csv": I.IDLE_ARCHETYPE},
                    "e0": {"move6_gate": e0_status or "E0 usable", "idle": {k: check[k] for k in ("can_build", "n_idle", "n_idle_with_e0", "reason")}},
                    "mode": mode, "feature_sets_run": list(e_sets), "move12": {"out": str(nb_out), "status": m12_status}, "code": entry["code"],
                    "elapsed_s": round(time.time() - t0, 1), "results": results}
        write_json(step_dir / "step.json", step_rec)
        entry["outputs"] = sorted(str(p.relative_to(out)) for p in step_dir.rglob("*") if p.is_file())
        entry.update(status="done", exit_code=0, finished_at=now_iso(), elapsed_s=step_rec["elapsed_s"])
        for s in results:
            line = "; ".join(f"{lv}: " + ", ".join(f"{e} {fmt(((s['scores'][lv][e].get('single') or {}).get('accuracy')))}" for e in e_sets) for lv in LEVELS)
            print(f"[randomsplit] step {step} cut {s['cut']} ({s['cut_name']}): {s['split']['n_recordings']} recordings, {s['split']['n_test_windows']} test windows, "
                  f"{s['split']['n_train_windows_used']} training" + (f" ({s['split']['n_train_windows_removed']} removed)" if step == 2 else "") + f"; accuracy: {line}")
        print(f"[randomsplit] step {step} done in {step_rec['elapsed_s']} s: {step_dir}")
        return EXIT_OK
    except BaseException as exc:                           # noqa: BLE001  the record says how the run ended; the exception continues to main
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
    p.add_argument("--step", required=True, type=int, choices=sorted(STEPS), help="1 = the plain random split; 2 = the same split, the training windows sharing a pair with a test window removed")
    p.add_argument("--out", default=None, help="the output folder (default: a sibling of the run, <run>_randomsplit; never inside the run)")
    p.add_argument("--newblocks-out", default=None, help="where move 12 wrote its results (default <run>_newblocks; read only)")
    p.add_argument("--seen", type=float, default=DEFAULT_SEEN, help="the share of each recording's windows that train (default 0.8, rounded to windows; move 12's)")
    p.add_argument("--seed-split", type=int, default=SEED_SPLIT, help="the seed of the random split (default 20261010)")
    p.add_argument("--cuts", default=None, help="which cuts, by name: declared,measured (default both; the values come from params.json)")
    p.add_argument("--encoding-out", default=None, help="decision D2 (default: the encoding run move 6 recorded in classify.json)")
    p.add_argument("--n-jobs", type=int, default=None, help="processes for the forest (default: move 6's recorded value)")
    p.add_argument("--n-estimators", type=int, default=None, help="trees (default: move 6's recorded value, Table 2's 300)")
    p.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP, help="resamples of the recordings for the margins' 95%% interval (default 1000, move 12's)")
    p.add_argument("--force", action="store_true", help="re-run a finished step")
    p.add_argument("--dry-run", action="store_true", help="print the plan (the recordings, the cuts, the windows, training and test, the removal); write nothing; skip the live-lock refusal")


def main(argv: list[str] | None = None) -> int:
    install_sigterm()
    ap = argparse.ArgumentParser(prog="plan12_grounding.random_split", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    add_run_args(sub.add_parser("run", help="step 1 (the plain random split) or step 2 (the shared-pair training windows removed): the three levels, both cuts"))
    o = ap.parse_args(argv)
    try:
        return run_move(o)
    except I.Stop as exc:
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
