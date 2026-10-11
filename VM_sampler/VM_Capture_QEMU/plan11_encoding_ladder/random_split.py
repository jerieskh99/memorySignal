#!/usr/bin/env python3
"""random_split.py -- move 18 of plan11_encoding_ladder (optional; added 2026-10-11): the random 80/20 split, the
companion of move 17 (new_blocks.py, the new-block test). The split and the removal of the shared-pair training
windows are copied from plan12_grounding/random_split.py, commit c91b091, 2026-10-10 (the SPL paper's move 18);
everything else is move 17's, imported from new_blocks.py (plan11 never imports plan12; new_blocks.py is not edited).

  python3 -m plan11_encoding_ladder.random_split run --out <run> --step 1|2 [--window W64_H32|own|<grid_id>] [--seen 0.8]
        [--seed-split 20261010] [--levels archetype,kernel,run] [--rungs apf,wapf,persist,content,combined] [--n-jobs N]
        [--n-estimators N] [--bootstrap 1000] [--seed-offset 0] [--dry-run]
  python3 -m plan11_encoding_ladder.random_split tables --out <run> [--windows W64_H32,own]

THE QUESTION. Move 17 trains on the first 80% of every recording's windows in time order and names the blocks that
come after a gap. How much of what it scores is the ORDER of the split, that is, having to name a recording's future
from its past? This move keeps everything of move 17 and changes one thing: per recording, the windows go to training
(80%) and testing (20%) AT RANDOM, seeded. A test window then has training windows on both sides of it in time, and a
neighbouring training window shares half its pairs with it (at 64 x 32: windows start every 32 pairs and span 64).
Two steps separate the two effects:
  step 1  the plain random split;
  step 2  the same split (the same test windows), with every training window that shares a pair with a test window
          REMOVED from training; the outputs say how many windows that removes, per recording and in all.
So step 1 against move 17 is "random against time order, pairs shared"; step 2 against move 17 is "random against
time order, no pair shared"; step 1 against step 2 is what the shared pairs alone are worth.

EXACTLY MOVE 17'S, imported from new_blocks.py: the data (every admissible cell, the 8 idle runs as one more answer,
the five readings from the toolkit's own feature files features/<rung>/<grid>_norm.npz, the run's own cut), the
windows (W64_H32 the primary, `own` each reading's selected window with "same as primary" where that is W64_H32), the
three levels (archetype as the feature files carry it, kernel, run), the forest and its settings, the dimension rule
and the L1 quarantine at the window (new_blocks.fit_level), the vote and the scores on single test windows
(pool_votes with the pool of 1, score_pools: accuracy with its 95% bootstrap interval over recordings, macro recall
over the classes present, chance 1/K, the majority class, recall per kernel), the run level read against idle
(run_level_shares), the margins against APF on the same test windows with the bootstrap and its seed
(bootstrap_margin, SEED_BOOTSTRAP), the kernel confusions and the figures (new_blocks' own figure functions, their
titles reworded from "new blocks" to "test windows"). The comparators take no part, as in move 17.

NOT APPLICABLE HERE, said in the outputs and replaced by nothing: the pools of 2 and 3 consecutive new blocks (the
test windows are scattered at random through a recording, so no consecutive run of test windows exists by design:
single test windows only), the accuracy by position after the gap (there is no gap), and the label-shuffle null, for
move 17's reason (labels shuffled consistently within a recording are learned from that recording's own training
windows; chance and the majority class are the baselines).

THE SPLIT, per recording, reading and window (--seen 0.8, --seed-split 20261010)
The recording's windows in time order; n_train = round(seen * n) as move 17 counts its seen windows; the n - n_train
test windows are drawn without replacement by numpy's default_rng seeded with [--seed-split, W, H, sha256(cell)[:4]]
(the same test windows for every reading at the same window, in both steps, whatever readings are selected); the
test windows are numbered 1, 2, 3 ... in time order (the block id "<cell>|b<k>", as move 17's). Step 2 then removes
from training every window whose pair range start .. start + W - 1 meets a test window's (|start difference| < W);
split_<rung>.csv says how many per recording, summary.json in all.

MOVE 17 BESIDE. Every scores, margins, per_kernel and run_level row carries move 17's single-block numbers for the
same level, reading and window (its pool_size 1 rows under gates/added/new_blocks/<window>/), marked "missing (move
17 has not run at this window)" when they are absent. Those files are read, not declared as inputs: move 18 does not
wait for move 17, and a later move 17 does not make move 18 stale (`tables` re-reads them).

THE OUTPUTS, under gates/added/random_split/<window>/<step>/ (<step> = step1_random or step2_random_nooverlap)
  scores.csv, margins.csv, per_kernel.csv, run_level.csv   as move 17's, single test windows (pool_size 1), with the
                        training windows removed (step 2) and move 17's columns beside
  split_<rung>.csv      per recording: the windows, training drawn, removed, training used, test, the test positions
  predictions_<level>_<rung>.csv, confusion_kernel_<rung>.csv, summary.json, params.json
`tables`: gates/added/random_split.csv (one row per window, step, level and reading), report/tables/random_split_scores,
random_split_margins, random_split_run_level, random_split_split (.csv, .md, .tex), report/figures/random_split_accuracy.svg
(the levels by reading: step 1, step 2 and move 17 side by side, with chance and the majority class),
random_split_per_kernel_step1/2.svg, random_split_run_level_step1/2.svg, random_split_confusion_apf.svg and
random_split_confusion_content.svg (step 2), each with its data as CSV.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import argparse  # noqa: E402
import hashlib  # noqa: E402
import html  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from plan11_encoding_ladder import models as M  # noqa: E402
from plan11_encoding_ladder import new_blocks as NB  # noqa: E402  (move 17: the data, the forest, the scores, the figures; imported, never edited)
from plan11_encoding_ladder import series as S  # noqa: E402
from plan11_encoding_ladder import verdicts as V  # noqa: E402
from plan11_encoding_ladder._report_common import write_table  # noqa: E402
from plan11_encoding_ladder.nulls import SEED_FOREST  # noqa: E402

ADDED = "added 2026-10-11"
COPIED = "copied from plan12_grounding/random_split.py, commit c91b091, 2026-10-10 (the random split, the removal of the shared-pair training windows, the notes)"
CITATION = ("plan11_encoding_ladder/PROMPT_move18_random_split.md (the author's design, 2026-10-11); " + COPIED + "; plan11_encoding_ladder/new_blocks.py (move 17: "
            "load_window, resolve_windows, fit_level, pool_votes, score_pools, run_level_shares, bootstrap_margin, the figures; SPEC_epoch2 Part 4 item 31); "
            "plan11 models.py make_forest / _reduce_fit / aggregate_units / quarantine_l1, nulls.py SEED_FOREST; no label-shuffle null (NO_NULL)")
STEPS = {1: ("step1_random", "the plain random split: round(seen * n) of each recording's windows train, the rest test, drawn at random"),
         2: ("step2_random_nooverlap", "the same random split, every training window that shares a pair with a test window removed from training")}
LEVELS, LEVEL_TEXT, RUNGS = NB.LEVELS, NB.LEVEL_TEXT, NB.RUNGS
MARGIN_BASE, PRIMARY_GRID, WINDOW_OWN, IDLE_LABEL, HIGHLIGHT = NB.MARGIN_BASE, NB.PRIMARY_GRID, NB.WINDOW_OWN, NB.IDLE_LABEL, NB.HIGHLIGHT
SEED_BOOTSTRAP, NO_NULL, COMPARATORS_NOTE, SAME_AS_PRIMARY, ENC_COLOUR = NB.SEED_BOOTSTRAP, NB.NO_NULL, NB.COMPARATORS_NOTE, NB.SAME_AS_PRIMARY, NB.ENC_COLOUR
DEFAULT_SEEN, DEFAULT_BOOTSTRAP = NB.DEFAULT_SEEN, NB.DEFAULT_BOOTSTRAP
SEED_SPLIT = 20261010                                      # the split's own seed (the date of the design); the forest keeps the toolkit's, the bootstrap move 17's
OUT_DIR = Path("gates") / "added" / "random_split"
SUMMARY_CSV = Path("gates") / "added" / "random_split.csv"
NB_DIR = NB.OUT_DIR                                        # move 17's folder, read for the columns beside
POOLS_NOTE = V.not_applicable("pools of 2 and 3 consecutive new blocks: the test windows are scattered at random through a recording, so no consecutive run of test "
                              "windows exists by design; single test windows only")
POSITION_NOTE = V.not_applicable("the accuracy by position after the gap: there is no gap in a random split; the test windows are numbered in time order for their ids only")
SPLIT_RULE = ("per recording, reading and window: the windows in time order; n_train = round(seen * n); n - n_train test windows drawn without replacement by numpy "
              "default_rng([seed_split, W, H, sha256(cell_id)[:4]]).choice; the test windows numbered 1, 2, 3 ... in time order; step 2 removes every training window "
              "whose pair range start .. start + W - 1 meets a test window's (|start difference| < W)")
MISSING_17 = "missing (move 17 has not run at this window)"
RULES = {**NB.RULES,
         "splits": "not applicable: the toolkit's splits (within-trace, LORO, LOKO) and move 17's time split are replaced by the random split of this test (" + SPLIT_RULE + ")",
         "unit": "replaced: the split stage's unit is the cell; here the unit is the single test window (move 17's single block); " + POOLS_NOTE,
         "by_position": POSITION_NOTE,
         "move17_beside": "move 17's single-block rows (gates/added/new_blocks/<window>/, pool_size 1) are read for the columns beside, not declared as inputs"}
EXIT_OK, EXIT_ERROR, EXIT_MISSING = NB.EXIT_OK, NB.EXIT_ERROR, NB.EXIT_MISSING
Stop = NB.Stop
fmt, svg_open, _f, _read_rows, class_order, confusion_of = NB.fmt, NB.svg_open, NB._f, NB._read_rows, NB.class_order, NB.confusion_of


def cell_seed(cell_id: str) -> int:
    return int.from_bytes(hashlib.sha256(cell_id.encode("utf-8")).digest()[:4], "big")


# ---------------------------------------------------------------------------------------------
# the split (copied from plan12_grounding/random_split.py random_split_rows, commit c91b091; the pair ranges from the feature file)
# ---------------------------------------------------------------------------------------------
def random_split_rows(d: dict, seen: float, seed_split: int, step: int) -> dict:
    """Per cell (first-seen order): the row indices of its windows in time order; round(seen * n) train, the rest test
    at random (seeded by the window geometry and the cell); step 2 removes the training windows that share a pair with
    a test window. Returns new_blocks.split_rows' shape ({cells, per_cell, train_idx, test_idx, block_pos, block_id,
    n_new_total}) plus removed_idx and n_removed_total."""
    cell_id, pair_start, W, H = d["cell_id"], d["pair_start"], int(d["W"]), int(d["H"])
    cells = list(dict.fromkeys(cell_id.tolist()))
    per_cell, train, test, removed, pos, bid = {}, [], [], [], [], []
    for c in cells:
        idx = np.flatnonzero(cell_id == c)
        idx = idx[np.argsort(d["win_start"][idx], kind="stable")]
        n = int(idx.size)
        n_train = int(np.floor(seen * n + 0.5))           # round(seen * n), half up, as move 17 counts its seen windows
        n_test = n - n_train
        rng = np.random.default_rng([int(seed_split), W, H, cell_seed(c)])
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
                       "first_pair": int(pair_start[idx[0]]), "last_pair": int(pair_start[idx[-1]] + W - 1), "note": "" if n_test else "no test window: too few windows"}
        train += idx[train_pos].tolist()
        removed += idx[removed_pos].tolist()
        for k, j in enumerate(idx[test_pos].tolist(), start=1):
            test.append(j); pos.append(k); bid.append(f"{c}|b{k}")
    return {"cells": cells, "per_cell": per_cell, "train_idx": np.array(train, dtype=np.int64), "test_idx": np.array(test, dtype=np.int64),
            "removed_idx": np.array(removed, dtype=np.int64), "block_pos": np.array(pos, dtype=np.int64), "block_id": np.array(bid, dtype=str),
            "n_new_total": len(test), "n_removed_total": len(removed)}


def split_counts(sp: dict) -> dict:
    pc = sp["per_cell"].values()
    rem = [v["n_removed"] for v in pc]
    return {"n_recordings": len(sp["cells"]), "n_windows": int(sum(v["n_windows"] for v in pc)), "n_train_drawn": int(sum(v["n_train_drawn"] for v in pc)),
            "n_train_removed": int(sp["n_removed_total"]), "n_train_used": int(sp["train_idx"].size), "n_test": int(sp["n_new_total"]),
            "removed_per_recording": {"min": min(rem, default=0), "max": max(rem, default=0), "mean": float(np.mean(rem)) if rem else 0.0},
            "recordings_without_test_window": [c for c in sp["cells"] if sp["per_cell"][c]["n_test"] == 0]}


# ---------------------------------------------------------------------------------------------
# move 17's single-block rows, read as they stand
# ---------------------------------------------------------------------------------------------
def move17_results(out: Path, window: str) -> dict:
    d = out / NB_DIR / window
    rows = lambda name: [r for r in _read_rows(d / name) if r.get("pool_size", "1") == "1"]        # noqa: E731
    if not (d / "scores.csv").is_file():
        return {"status": MISSING_17, "folder": str(d), "scores": {}, "margins": {}, "per_kernel": {}, "run_level": {}}
    return {"status": "ok", "folder": str(d),
            "scores": {(r["level"], r["rung"]): r for r in rows("scores.csv")},
            "margins": {(r["level"], r["rung"], r["comparison"]): r for r in rows("margins.csv")},
            "per_kernel": {(r["level"], r["rung"], r["kernel"]): r for r in rows("per_kernel.csv")},
            "run_level": {(r["rung"], r["kernel"]): r for r in rows("run_level.csv")}}


# ---------------------------------------------------------------------------------------------
# one reading at one window, one step (move 17's one_rung with the random split and the pool of 1)
# ---------------------------------------------------------------------------------------------
def one_rung(out: Path, d: dict, rung: str, grid_id: str, grid_source: str, levels, step: int, *, seen: float, seed_split: int, n_boot: int,
             n_jobs: int, n_est: int, seed: int) -> dict:
    sp = random_split_rows(d, seen, seed_split, step)
    cells = sp["cells"]
    cell_rows = d["cell_id"]
    kernel_of = {c: str(d["kernel"][np.flatnonzero(cell_rows == c)[0]]) for c in cells}
    arche_of = {c: str(d["archetype"][np.flatnonzero(cell_rows == c)[0]]) for c in cells}
    rep_of = {c: int(d["rep"][np.flatnonzero(cell_rows == c)[0]]) for c in cells}
    labels_of = {"archetype": d["archetype"], "kernel": d["kernel"], "run": cell_rows}
    majority_class = {}
    for level in LEVELS:
        counts: dict = {}
        for c in cells:
            lv = {"archetype": arche_of, "kernel": kernel_of, "run": {c: c}}[level][c]
            counts[lv] = counts.get(lv, 0) + 1
        majority_class[level] = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    res = {"rung": rung, "grid_id": grid_id, "grid_source": grid_source, "W": d["W"], "H": d["H"], "step": step, "split": sp, "counts": split_counts(sp), "cells": cells,
           "kernel_of": kernel_of, "arche_of": arche_of, "rep_of": rep_of, "levels": {}, "pools_by": {}, "majority_class": majority_class, "n_windows": int(cell_rows.size),
           "labels_kernel": class_order(kernel_of.values())}
    te = sp["test_idx"]
    if sp["n_new_total"] == 0:
        n_w = sorted({v["n_windows"] for v in sp["per_cell"].values()})
        res["status"] = V.not_run(f"no test window at {grid_id}: {n_w[0]}" + (f" to {n_w[-1]}" if len(n_w) > 1 else "") + " windows per recording, too few for a 20% share")
        print(f"[randomsplit] step {step} {rung:<9} {grid_id}: {res['status']}", flush=True)
        return res
    if sp["train_idx"].size == 0:
        res["status"] = V.not_run(f"no training window left at {grid_id} after the removal")
        print(f"[randomsplit] step {step} {rung:<9} {grid_id}: {res['status']}", flush=True)
        return res
    res["status"] = "ok"
    c = res["counts"]
    print(f"[randomsplit] step {step} {rung:<9} {grid_id}: {len(cells)} recordings, {int(cell_rows.size)} windows of W {d['W']}; training {c['n_train_used']}"
          + (f" after {c['n_train_removed']} removed for sharing a pair with a test window" if step == 2 else "") + f", test {c['n_test']} "
          f"({min(v['n_test'] for v in sp['per_cell'].values())} to {max(v['n_test'] for v in sp['per_cell'].values())} per recording)", flush=True)
    for level in levels:
        y = labels_of[level]
        t0 = time.time()
        fit = NB.fit_level(d["X"], d["names"], y, sp, cell_rows, seed=seed, n_jobs=n_jobs, n_est=n_est)
        use = fit["use"]
        if "pred" not in use:
            res["levels"][level] = {"status": fit["status"], "scores": {}, "quarantine": fit["quarantine"], "quarantined_features": fit["quarantined_features"],
                                    "score_source": fit["score_source"]}
            continue
        pv = NB.pool_votes(sp, use["pred"], use["proba"], use["classes"], y[te], [1])
        res["pools_by"][level] = pv
        sc = {1: NB.score_pools(pv[1], fit["classes_train"], kernel_of, majority_class[level], cells, n_boot)}
        ci = {cc: i for i, cc in enumerate(use["classes"])}
        blocks = [{"block_id": b, "cell_id": b.split("|b")[0], "position": int(sp["block_pos"][i]), "win_start": int(d["win_start"][te[i]]),
                   "pair_start": int(d["pair_start"][te[i]]), "pair_end": int(d["pair_start"][te[i]] + d["W"] - 1), "y_true": str(y[te[i]]), "y_pred": str(use["pred"][i]),
                   "p_pred": float(use["proba"][i].max()), "p_true": float(use["proba"][i][ci[str(y[te[i]])]]) if str(y[te[i]]) in ci else None}
                  for i, b in enumerate(sp["block_id"].tolist())]
        res["levels"][level] = {"status": "ok", "scores": sc, "feature_count": use["feature_count"], "feature_count_used": use["feature_count_used"],
                                "reduced": use["reduced"], "score_source": fit["score_source"], "quarantine": fit["quarantine"], "quarantined_features": fit["quarantined_features"],
                                "full_model_accuracy": (NB.score_pools(NB.pool_votes(sp, fit["full"]["pred"], fit["full"]["proba"], fit["full"]["classes"], y[te], [1])[1],
                                                                       fit["classes_train"], kernel_of, majority_class[level], cells, 1)["accuracy"] if fit["with_quarantine"] else None),
                                "n_train_windows": fit["n_train_windows"], "n_train_cells": fit["n_train_cells"], "n_test_windows": fit["n_new_blocks"],
                                "classes_train": fit["classes_train"], "run_level": NB.run_level_shares(pv[1], kernel_of) if level == "run" else None, "blocks": blocks,
                                "elapsed_s": round(time.time() - t0, 1)}
        s1 = sc[1]
        print(f"[randomsplit] step {step} {rung:<9} {grid_id} {level:<9} acc {fmt(s1['accuracy'])} [{fmt(s1['ci_lo'])}, {fmt(s1['ci_hi'])}] macro {fmt(s1['macro_recall'])} "
              f"chance {fmt(s1['chance'])} majority {fmt(s1['majority'])} | features {use['feature_count_used']}/{use['feature_count']}"
              + (f" quarantined {len(fit['quarantined_features'])}" if fit["quarantined_features"] else "") + f" ({res['levels'][level]['elapsed_s']} s)", flush=True)
    return res


# ---------------------------------------------------------------------------------------------
# the window's tables for one step (move 17's columns, the removed count, move 17's numbers beside)
# ---------------------------------------------------------------------------------------------
M17_SCORE = ["move17_status", "move17_accuracy", "move17_ci95_lo", "move17_ci95_hi", "move17_macro_recall", "move17_chance", "move17_majority", "move17_n_new_blocks", "move17_n_train_windows"]
SCORE_COLUMNS = ["level", "rung", "grid_id", "grid_source", "pool_size", "status", "n_pools", "n_classes", "accuracy", "ci95_lo", "ci95_hi", "macro_recall", "chance", "majority",
                 "feature_count", "feature_count_used", "score_source", "quarantined_features", "n_train_windows", "n_train_windows_removed", "n_train_recordings", "null"] + M17_SCORE
MARGIN_COLUMNS = NB.MARGIN_COLUMNS + ["move17_delta_accuracy", "move17_ci95_lo", "move17_ci95_hi", "move17_status"]
PER_KERNEL_COLUMNS = NB.PER_KERNEL_COLUMNS + ["move17_recall", "move17_n_pools"]
RUN_LEVEL_COLUMNS = NB.RUN_LEVEL_COLUMNS + ["move17_right_run", "move17_same_kernel_other_run", "move17_other_kernel"]
SPLIT_COLUMNS = ["cell_id", "kernel", "grid_id", "n_windows", "n_train_drawn", "n_removed", "n_train_used", "n_test", "first_pair", "last_pair", "test_positions", "removed_positions", "note"]
TABLE_FILES = {"scores.csv": SCORE_COLUMNS, "margins.csv": MARGIN_COLUMNS, "per_kernel.csv": PER_KERNEL_COLUMNS, "run_level.csv": RUN_LEVEL_COLUMNS}


def primary_rows(out: Path, step_name: str, name: str, rung: str) -> list[list]:
    """This move's primary window rows of the step for a reading whose own window is the primary: copied with status 'same as primary'."""
    cols = TABLE_FILES[name]
    res = []
    for r in _read_rows(out / OUT_DIR / PRIMARY_GRID / step_name / name):
        if r.get("rung") != rung:
            continue
        r = dict(r)
        if "status" in r:
            r["status"] = SAME_AS_PRIMARY + (f"; {r['status']}" if r.get("status") not in ("ok", "") else "")
        if "grid_source" in r:
            r["grid_source"] = "selection.json = " + PRIMARY_GRID
        res.append([r.get(c) for c in cols])
    return res


def _m17_score(m17: dict, level: str, rung: str) -> list:
    m = m17["scores"].get((level, rung))
    if not m:
        return [m17["status"] if m17["status"] != "ok" else MISSING_17, None, None, None, None, None, None, None, None]
    return [m.get("status"), _f(m.get("accuracy")), _f(m.get("ci95_lo")), _f(m.get("ci95_hi")), _f(m.get("macro_recall")), _f(m.get("chance")), _f(m.get("majority")),
            m.get("n_pools"), m.get("n_train_windows")]


def write_window(out: Path, window: str, step: int, results: dict, same_as_primary: set, levels, n_boot: int, params: dict) -> dict:
    step_name, step_text = STEPS[step]
    d = out / OUT_DIR / window / step_name
    d.mkdir(parents=True, exist_ok=True)
    m17 = move17_results(out, window)
    score_rows, margin_rows, pk_rows, rl_rows = [], [], [], []
    for rung in RUNGS:
        if rung in same_as_primary:
            if (out / OUT_DIR / PRIMARY_GRID / step_name / "scores.csv").is_file():
                score_rows += primary_rows(out, step_name, "scores.csv", rung); margin_rows += primary_rows(out, step_name, "margins.csv", rung)
                pk_rows += primary_rows(out, step_name, "per_kernel.csv", rung); rl_rows += primary_rows(out, step_name, "run_level.csv", rung)
            else:
                for level in levels:
                    score_rows.append([level, rung, PRIMARY_GRID, "selection.json = " + PRIMARY_GRID, 1, SAME_AS_PRIMARY + " (run the primary window first)", None, None, None, None,
                                       None, None, None, None, None, None, None, "", None, None, None, NO_NULL] + _m17_score(m17, level, rung))
            continue
        r = results[rung]
        n_removed = r["counts"]["n_train_removed"]
        for level in levels:
            lv = r["levels"].get(level) or {}
            s = (lv.get("scores") or {}).get(1) or {}
            score_rows.append([level, rung, r["grid_id"], r["grid_source"], 1, lv.get("status") or r["status"], s.get("n_pools"), s.get("n_classes"), s.get("accuracy"),
                               s.get("ci_lo"), s.get("ci_hi"), s.get("macro_recall"), s.get("chance"), s.get("majority"), lv.get("feature_count"), lv.get("feature_count_used"),
                               lv.get("score_source"), ";".join(lv.get("quarantined_features") or []), lv.get("n_train_windows"), n_removed, lv.get("n_train_cells"), NO_NULL]
                              + _m17_score(m17, level, rung))
            for kern in r["labels_kernel"]:
                if kern in (s.get("recall_per_kernel") or {}):
                    m = m17["per_kernel"].get((level, rung, kern)) or {}
                    pk_rows.append([level, rung, r["grid_id"], 1, kern, s["recall_per_kernel"][kern],
                                    sum(1 for p in r["pools_by"].get(level, {}).get(1, []) if r["kernel_of"][p["cell_id"]] == kern), _f(m.get("recall")), m.get("n_pools")])
        rl = (r["levels"].get("run") or {})
        if rl.get("run_level") is not None:
            for kern in r["labels_kernel"]:
                if kern in rl["run_level"]:
                    v = rl["run_level"][kern]
                    m = m17["run_level"].get((rung, kern)) or {}
                    rl_rows.append([rung, r["grid_id"], 1, kern, v["n_pools"], v["right_run"], v["same_kernel_other_run"], v["other_kernel"],
                                    "idle: the boot's share (idle runs have a boot and no seed)" if kern == IDLE_LABEL else "",
                                    _f(m.get("right_run")), _f(m.get("same_kernel_other_run")), _f(m.get("other_kernel"))])
        base = results.get(MARGIN_BASE)
        for level in levels:
            comp = f"{rung}-{MARGIN_BASE}"
            if rung == MARGIN_BASE:
                continue
            mm = m17["margins"].get((level, rung, comp)) or {}
            beside = [_f(mm.get("delta_accuracy")), _f(mm.get("ci95_lo")), _f(mm.get("ci95_hi")), mm.get("status") if mm else (m17["status"] if m17["status"] != "ok" else MISSING_17)]
            if base is None or (MARGIN_BASE in same_as_primary and window != PRIMARY_GRID):
                margin_rows.append([level, rung, r["grid_id"], 1, comp, None, None, None, None, None, None, None,
                                    V.not_run(f"{MARGIN_BASE} not computed at this window (its own window is {PRIMARY_GRID}: see the primary folder)")] + beside)
                continue
            if base["grid_id"] != r["grid_id"]:
                margin_rows.append([level, rung, r["grid_id"], 1, comp, None, None, None, None, None, None, None,
                                    f"not comparable: different windows ({rung} at {r['grid_id']}, {MARGIN_BASE} at {base['grid_id']})"] + beside)
                continue
            pb, pa = r["pools_by"].get(level, {}).get(1), base["pools_by"].get(level, {}).get(1)
            if pb is None or pa is None:
                margin_rows.append([level, rung, r["grid_id"], 1, comp, None, None, None, None, None, None, None,
                                    "not run: " + str((r["levels"].get(level) or {}).get("status") or r["status"] if pb is None else (base["levels"].get(level) or {}).get("status") or base["status"])] + beside)
                continue
            bm = NB.bootstrap_margin(pb, pa, r["cells"], n_boot, SEED_BOOTSTRAP)
            margin_rows.append([level, rung, r["grid_id"], 1, comp, bm["delta"], bm["ci_lo"], bm["ci_hi"], bm["n_pools"], bm["n_recordings"], bm["gained"], bm["lost"], bm["status"]] + beside)
        S.write_csv(d / f"split_{rung}.csv", SPLIT_COLUMNS,
                    [[c, r["kernel_of"][c], r["grid_id"], *[r["split"]["per_cell"][c][x] for x in ("n_windows", "n_train_drawn", "n_removed", "n_train_used", "n_test", "first_pair", "last_pair",
                                                                                                  "test_positions", "removed_positions", "note")]] for c in r["cells"]])
        for level in levels:
            lv = r["levels"].get(level) or {}
            if not lv.get("blocks"):
                continue
            S.write_csv(d / f"predictions_{level}_{rung}.csv", ["block_id", "cell_id", "kernel", "archetype", "rep", "position", "win_start", "pair_start", "pair_end", "y_true", "y_pred", "p_pred", "p_true"],
                        [[b["block_id"], b["cell_id"], r["kernel_of"][b["cell_id"]], r["arche_of"][b["cell_id"]], r["rep_of"][b["cell_id"]], b["position"], b["win_start"], b["pair_start"],
                          b["pair_end"], b["y_true"], b["y_pred"], b["p_pred"], b["p_true"]] for b in lv["blocks"]])
        kl = r["levels"].get("kernel") or {}
        if kl.get("blocks"):
            Cm = confusion_of(kl["blocks"], r["labels_kernel"])
            S.write_csv(d / f"confusion_kernel_{rung}.csv", ["true\\predicted"] + r["labels_kernel"], [[l] + Cm[i].tolist() for i, l in enumerate(r["labels_kernel"])])
    S.write_csv(d / "scores.csv", SCORE_COLUMNS, score_rows)
    S.write_csv(d / "margins.csv", MARGIN_COLUMNS, margin_rows)
    S.write_csv(d / "per_kernel.csv", PER_KERNEL_COLUMNS, pk_rows)
    S.write_csv(d / "run_level.csv", RUN_LEVEL_COLUMNS, rl_rows)
    summary = {"added": ADDED, "window": window, "step": step, "step_name": step_name, "step_text": step_text, "levels": list(levels), "pools": POOLS_NOTE, "by_position": POSITION_NOTE,
               "null": NO_NULL, "comparators": COMPARATORS_NOTE, "rules": RULES, "split_rule": SPLIT_RULE, "same_as_primary": sorted(same_as_primary),
               "move17": {"folder": m17["folder"], "status": m17["status"]}, "readings": {}}
    for rung, r in results.items():
        summary["readings"][rung] = {"status": r["status"], "grid_id": r["grid_id"], "grid_source": r["grid_source"], "W": r["W"], "H": r["H"], "n_recordings": len(r["cells"]),
                                     "n_windows": r["n_windows"], "split": r["counts"], "majority_class": r["majority_class"], "classes_kernel": r["labels_kernel"],
                                     "levels": {lv: {k: v for k, v in (r["levels"].get(lv) or {}).items() if k not in ("blocks", "quarantine")}
                                                | {"scores": {str(k): {kk: vv for kk, vv in s.items() if kk != "recall_per_class"} for k, s in ((r["levels"].get(lv) or {}).get("scores") or {}).items()}}
                                                for lv in levels}}
    S.write_json(d / "summary.json", "plan11.random_split_window.v1", params, CITATION, summary)
    S.write_json(d / "params.json", "plan11.random_split.v1", params, CITATION, {})
    return {"dir": d, "n_score_rows": len(score_rows)}


# ---------------------------------------------------------------------------------------------
# the move: `run`
# ---------------------------------------------------------------------------------------------
def run_random_split(out: Path, *, step: int, window: str = PRIMARY_GRID, seen: float = DEFAULT_SEEN, seed_split: int = SEED_SPLIT, levels=LEVELS, rungs=RUNGS,
                     n_jobs: int = 1, n_estimators: int = M.N_ESTIMATORS, bootstrap: int = DEFAULT_BOOTSTRAP, seed_offset: int = 0, dry_run: bool = False) -> int:
    out = Path(out)
    step_name, step_text = STEPS[step]
    if not (out / "cells.csv").is_file():
        raise Stop(EXIT_MISSING, f"missing input: {out / 'cells.csv'}")
    if not (0.0 < seen < 1.0) or bootstrap < 1:
        raise Stop(EXIT_MISSING, f"--seen must lie in (0, 1) (given {seen}), --bootstrap 1 or more (given {bootstrap})")
    cells = S.load_cells(out / "cells.csv")
    windows = NB.resolve_windows(out, window, rungs)
    same_as_primary = {r for r, (g, _) in windows.items() if window == WINDOW_OWN and g == PRIMARY_GRID}
    seed = SEED_FOREST + int(seed_offset)
    data = {r: NB.load_window(out, r, windows[r][0], cells) for r in rungs if r not in same_as_primary}
    first = next(iter(data.values()), None)
    m17 = move17_results(out, window)
    print(f"[randomsplit] step {step}: {step_text}; window {window}; seen {seen:g} of each recording's windows train, the rest test, at random (seed {seed_split}); single test windows "
          f"({POOLS_NOTE[:50]}...); levels {', '.join(levels)}; readings {', '.join(rungs)}; {NO_NULL[:60]}...")
    print(f"[randomsplit] run {out}; forest: {n_estimators} trees, {n_jobs} processes, seed {seed}; bootstrap {bootstrap} resamples, seed {SEED_BOOTSTRAP}; move 17 beside: {m17['status']} ({m17['folder']})")
    if first is not None:
        ids = set(first["cell_id"].tolist())
        n_k = sum(1 for c in cells if c["role"] == "kernel" and c["cell_id"] in ids)
        n_i = sum(1 for c in cells if c["role"] == "idle" and c["cell_id"] in ids)
        print(f"[randomsplit] recordings found: {first['n_cells']} = {n_k} kernel + {n_i} idle (the real run expects 103 = 95 kernel + 8 idle); "
              f"hard-excluded {len(first['excluded_hard'])}, pair-rung excluded {len(first['excluded_pair_rungs'])}; the cut: head drop {first['head_drop']}")
    plan = {}
    for r in rungs:
        gid, src = windows[r]
        if r in same_as_primary:
            print(f"[randomsplit] {r:<9} own window {gid} ({src}): {SAME_AS_PRIMARY}")
            continue
        d = data[r]
        sp = random_split_rows(d, seen, seed_split, step)       # the split itself (no forest): the dry run's numbers are the run's
        c = split_counts(sp)
        plan[r] = c
        ns = [v["n_windows"] for v in sp["per_cell"].values()]; nt = [v["n_test"] for v in sp["per_cell"].values()]
        print(f"[randomsplit] {r:<9} {gid} ({src}): {d['n_cells']} recordings, {c['n_windows']} windows of W {d['W']} H {d['H']}, {min(ns)} to {max(ns)} per recording; "
              f"training {c['n_train_used']}" + (f" after {c['n_train_removed']} removed ({c['removed_per_recording']['min']} to {c['removed_per_recording']['max']} per recording, "
                                                 f"mean {c['removed_per_recording']['mean']:.1f})" if step == 2 else "")
              + f", test {c['n_test']} ({min(nt)} to {max(nt)} per recording)" + (f"; {len(c['recordings_without_test_window'])} recording(s) without a test window" if c["recordings_without_test_window"] else ""))
    if dry_run:
        print(f"[randomsplit] dry run: would write {out / OUT_DIR / window / step_name}/ (scores.csv, margins.csv, per_kernel.csv, run_level.csv, split_<rung>.csv, predictions_<level>_<rung>.csv, "
              f"confusion_kernel_<rung>.csv, summary.json, params.json); nothing written")
        return EXIT_OK
    results = {}
    for r in rungs:
        if r in same_as_primary:
            continue
        results[r] = one_rung(out, data[r], r, windows[r][0], windows[r][1], levels, step, seen=seen, seed_split=seed_split, n_boot=bootstrap, n_jobs=n_jobs, n_est=n_estimators, seed=seed)
    inputs = [out / "cells.csv", out / "inputs" / "head_drop.csv", out / "gates" / "preconditions.json", out / "gates" / "preconditions.csv"] + [d["path"] for d in data.values()]
    if window == WINDOW_OWN:
        inputs += [out / "gates" / "selection.json", out / OUT_DIR / PRIMARY_GRID / step_name / "scores.csv"]
    params = {"added": ADDED, "window": window, "step": step, "step_name": step_name, "step_text": step_text, "removal": step == 2,
              "grids": {r: {"grid_id": g, "source": s, "same_as_primary": r in same_as_primary} for r, (g, s) in windows.items()},
              "seen": seen, "seed_split": int(seed_split), "split_rule": SPLIT_RULE, "levels": list(levels), "rungs": list(rungs), "n_estimators": int(n_estimators), "n_jobs": int(n_jobs),
              "seed_forest": seed, "seed_offset": int(seed_offset), "bootstrap": int(bootstrap), "seed_bootstrap": SEED_BOOTSTRAP,
              "head_drop": (first["head_drop"] if first else {}), "cut": RULES["cut"], "idle_label": IDLE_LABEL, "idle_archetype_in_feature_files": NB.IDLE_FILE_ARCHETYPE,
              "admissibility": {"hard_excluded": sorted({c for d in data.values() for c in d["excluded_hard"]}), "pair_rung_excluded": {r: d["excluded_pair_rungs"] for r, d in data.items()},
                                "preconditions_present": bool(first and first["preconditions_present"])},
              "split_counts": plan, "rules": RULES, "pools": POOLS_NOTE, "by_position": POSITION_NOTE, "comparators": COMPARATORS_NOTE, "null": NO_NULL, "copied": COPIED,
              "move17_read_not_declared": {"folder": m17["folder"], "status": m17["status"]}, "inputs_sha256": S.inputs_sha256([p for p in inputs if p.is_file()], out)}
    w = write_window(out, window, step, results, same_as_primary, levels, bootstrap, params)
    for r, res in results.items():
        if res["status"] != "ok":
            continue
        line = "; ".join(f"{lv}: {fmt(((res['levels'].get(lv) or {}).get('scores') or {}).get(1, {}).get('accuracy'))}" for lv in levels)
        print(f"[randomsplit] step {step} {r:<9} {res['grid_id']}: {res['counts']['n_test']} test windows, {res['counts']['n_train_used']} training"
              + (f" ({res['counts']['n_train_removed']} removed)" if step == 2 else "") + f"; accuracy {line}")
    print(f"[randomsplit] written: {w['dir']} ({w['n_score_rows']} score rows)")
    return EXIT_OK


# ---------------------------------------------------------------------------------------------
# the tables and the figures: `tables`
# ---------------------------------------------------------------------------------------------
def _rows_step(out: Path, window: str, step: int, name: str) -> list[dict]:
    return _read_rows(out / OUT_DIR / window / STEPS[step][0] / name)


def relabel(svg: str, step: int, window: str) -> str:
    """new_blocks' figures speak of new blocks after a gap; here the unit is the random test window of one step."""
    return (svg.replace("New blocks named right", "Test windows named right").replace("single new blocks", "single test windows (random split)")
               .replace("new blocks", "test windows").replace("new block", "test window").replace(f"window {window}", f"window {window}, random split step {step}", 1))


def fig_accuracy(rows_by_step: dict, window: str) -> tuple:
    """Three panels (the levels); per reading three bars: step 1 (light), step 2 (full) and move 17's single blocks (outlined), each with its
    bootstrap interval; chance 1/K as a grey line and the majority class as a dashed red line (step 2's)."""
    W_, H_ = 1000, 340
    out = svg_open(W_, H_, f"Test windows named right, by level and reading, window {window}: the random split against move 17",
                   "per reading: step 1 (light bar, random split, pairs shared), step 2 (full bar, the shared-pair training windows removed), move 17 (outlined, time order); "
                   "lines: the 95% bootstrap interval; grey: chance 1/K; dashed red: the majority class")
    x0, y0, w, h = 60, 60, 292, 200
    data = []
    for li, level in enumerate(LEVELS):
        px = x0 + li * (w + 22)
        out.append(f'<text x="{px}" y="{y0 - 6}" font-weight="bold">{level}</text>')
        out.append(f'<rect x="{px}" y="{y0}" width="{w}" height="{h}" fill="none" stroke="#ccc"/>')
        for t in (0.0, 0.5, 1.0):
            yy = y0 + h - h * t
            out.append(f'<line x1="{px}" y1="{yy:.1f}" x2="{px + w}" y2="{yy:.1f}" stroke="#eee"/><text x="{px - 4}" y="{yy + 3:.1f}" text-anchor="end">{t:.1f}</text>')
        bw = w / len(RUNGS)
        for ri, rung in enumerate(RUNGS):
            gx = px + ri * bw + 4
            tw = (bw - 10) / 3
            s = {st: (next((r for r in rows_by_step.get(st, []) if r["level"] == level and r["rung"] == rung and r.get("pool_size", "1") == "1"), None) or {}) for st in (1, 2)}
            m17 = s[2] or s[1]
            vals = {"step1": _f(s[1].get("accuracy")), "step2": _f(s[2].get("accuracy")), "move17": _f(m17.get("move17_accuracy"))}
            data.append({"level": level, "rung": rung, "grid_id": s[2].get("grid_id") or s[1].get("grid_id"), **vals,
                         "step1_ci95_lo": _f(s[1].get("ci95_lo")), "step1_ci95_hi": _f(s[1].get("ci95_hi")), "step2_ci95_lo": _f(s[2].get("ci95_lo")), "step2_ci95_hi": _f(s[2].get("ci95_hi")),
                         "move17_ci95_lo": _f(m17.get("move17_ci95_lo")), "move17_ci95_hi": _f(m17.get("move17_ci95_hi")), "chance": _f(s[2].get("chance") or s[1].get("chance")),
                         "majority": _f(s[2].get("majority") or s[1].get("majority")), "status_step1": s[1].get("status"), "status_step2": s[2].get("status"), "status_move17": m17.get("move17_status")})
            for k, (key, lo_k, hi_k) in enumerate((("step1", "ci95_lo", "ci95_hi"), ("step2", "ci95_lo", "ci95_hi"), ("move17", "move17_ci95_lo", "move17_ci95_hi"))):
                bx = gx + k * tw
                v = vals[key]
                src = s[1] if key == "step1" else (s[2] if key == "step2" else m17)
                if v is None:
                    txt = "missing" if key == "move17" else (str(src.get("status") or "not run")[:10])
                    out.append(f'<text x="{bx + tw / 2:.1f}" y="{y0 + h / 2 + (k - 1) * 11:.1f}" text-anchor="middle" fill="#999" font-size="7">{html.escape(txt)}</text>')
                    continue
                if key == "move17":
                    out.append(f'<rect x="{bx:.1f}" y="{y0 + h - h * v:.1f}" width="{tw - 2:.1f}" height="{h * v:.1f}" fill="none" stroke="{ENC_COLOUR[rung]}" stroke-width="1.5"/>')
                else:
                    out.append(f'<rect x="{bx:.1f}" y="{y0 + h - h * v:.1f}" width="{tw - 2:.1f}" height="{h * v:.1f}" fill="{ENC_COLOUR[rung]}" fill-opacity="{0.45 if key == "step1" else 0.9}"/>')
                lo, hi = _f(src.get(lo_k)), _f(src.get(hi_k))
                if lo is not None and hi is not None:
                    cx = bx + (tw - 2) / 2
                    out.append(f'<line x1="{cx:.1f}" y1="{y0 + h - h * hi:.1f}" x2="{cx:.1f}" y2="{y0 + h - h * lo:.1f}" stroke="#222" stroke-width="1"/>')
                out.append(f'<text x="{bx + (tw - 2) / 2:.1f}" y="{y0 + h - h * v - 3:.1f}" text-anchor="middle" font-size="7">{v:.2f}</text>')
            ch, mj = data[-1]["chance"], data[-1]["majority"]
            if ch is not None:
                out.append(f'<line x1="{gx - 2:.1f}" y1="{y0 + h - h * ch:.1f}" x2="{gx + 3 * tw:.1f}" y2="{y0 + h - h * ch:.1f}" stroke="#888"/>')
            if mj is not None:
                out.append(f'<line x1="{gx - 2:.1f}" y1="{y0 + h - h * mj:.1f}" x2="{gx + 3 * tw:.1f}" y2="{y0 + h - h * mj:.1f}" stroke="#b03a2e" stroke-dasharray="3,2"/>')
            out.append(f'<text x="{gx + 1.5 * tw:.1f}" y="{y0 + h + 12}" text-anchor="middle">{rung}</text>')
    out.append(f'<text x="{x0}" y="{H_ - 24}" fill="#555" font-size="9">{html.escape(POOLS_NOTE[:140])}</text>')
    out.append(f'<text x="{x0}" y="{H_ - 10}" fill="#555" font-size="9">{html.escape(NO_NULL[:140])}...</text></svg>')
    return "\n".join(out), data


def write_tables(out: Path, windows=(PRIMARY_GRID, WINDOW_OWN)) -> dict:
    """gates/added/random_split.csv, report/tables/random_split_*.{csv,md,tex} and report/figures/random_split_*.svg (each with its CSV)."""
    out = Path(out)
    tdir = out / "report" / "tables"
    fdir = out / "report" / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    present = [(w, st) for w in windows for st in (1, 2) if (out / OUT_DIR / w / STEPS[st][0] / "scores.csv").is_file()]
    if not present:
        raise Stop(EXIT_MISSING, f"missing input: {out / OUT_DIR}/<window>/<step>/scores.csv for {', '.join(windows)} (run `random_split run --step 1` and `--step 2` first)")
    all_scores = [{"window": w, "step": st, **r} for w, st in present for r in _rows_step(out, w, st, "scores.csv")]
    S.write_csv(out / SUMMARY_CSV, ["window", "step"] + SCORE_COLUMNS, [[r.get(c) for c in ["window", "step"] + SCORE_COLUMNS] for r in all_scores])
    fm = lambda x, d=3: ("" if _f(x) is None else f"{_f(x):.{d}f}")        # noqa: E731
    iv = lambda lo, hi: (f"[{fm(lo)}, {fm(hi)}]" if _f(lo) is not None else "")                                     # noqa: E731
    rows_t = [{"window": r["window"], "step": r["step"], "level": r["level"], "reading": r["rung"], "grid": r["grid_id"], "status": r.get("status"), "test windows": r.get("n_pools"),
               "classes": r.get("n_classes"), "accuracy": fm(r.get("accuracy")), "95% interval": iv(r.get("ci95_lo"), r.get("ci95_hi")), "macro recall": fm(r.get("macro_recall")),
               "chance": fm(r.get("chance")), "majority": fm(r.get("majority")), "training windows": r.get("n_train_windows"), "removed": r.get("n_train_windows_removed"),
               "move 17 accuracy": fm(r.get("move17_accuracy")), "move 17 interval": iv(r.get("move17_ci95_lo"), r.get("move17_ci95_hi")), "move 17 status": r.get("move17_status")}
              for r in all_scores]
    written = {"scores": write_table(tdir, "random_split_scores", ["window", "step", "level", "reading", "grid", "status", "test windows", "classes", "accuracy", "95% interval", "macro recall",
                                                                   "chance", "majority", "training windows", "removed", "move 17 accuracy", "move 17 interval", "move 17 status"], rows_t,
                                     label="tab:random_split_scores",
                                     note_comment=f"the random 80/20 split ({ADDED}): step 1 plain, step 2 with the shared-pair training windows removed; single test windows; move 17's single "
                                                  f"blocks beside; {NO_NULL}; {POOLS_NOTE}; {COMPARATORS_NOTE}")}
    rows_m = [{"window": w, "step": st, "level": r["level"], "comparison": r["comparison"], "grid": r["grid_id"], "delta accuracy": fm(r.get("delta_accuracy")),
               "95% interval": iv(r.get("ci95_lo"), r.get("ci95_hi")), "test windows": r.get("n_pools"), "gained": r.get("n_pools_gained"), "lost": r.get("n_pools_lost"), "status": r.get("status"),
               "move 17 delta": fm(r.get("move17_delta_accuracy")), "move 17 interval": iv(r.get("move17_ci95_lo"), r.get("move17_ci95_hi")), "move 17 status": r.get("move17_status")}
              for w, st in present for r in _rows_step(out, w, st, "margins.csv")]
    written["margins"] = write_table(tdir, "random_split_margins", ["window", "step", "level", "comparison", "grid", "delta accuracy", "95% interval", "test windows", "gained", "lost", "status",
                                                                    "move 17 delta", "move 17 interval", "move 17 status"], rows_m, label="tab:random_split_margins",
                                     note_comment="each reading minus APF on the same test windows, the 95% bootstrap interval over recordings; move 17's margins beside")
    rows_r = [{"window": w, "step": st, "reading": r["rung"], "grid": r["grid_id"], "kernel": r["kernel"], "test windows": r.get("n_pools"), "right run": fm(r.get("right_run")),
               "same kernel, other run": fm(r.get("same_kernel_other_run")), "other kernel": fm(r.get("other_kernel")), "move 17 right run": fm(r.get("move17_right_run")),
               "move 17 same kernel": fm(r.get("move17_same_kernel_other_run")), "move 17 other kernel": fm(r.get("move17_other_kernel")), "note": r.get("note")}
              for w, st in present for r in _rows_step(out, w, st, "run_level.csv")]
    written["run_level"] = write_table(tdir, "random_split_run_level", ["window", "step", "reading", "grid", "kernel", "test windows", "right run", "same kernel, other run", "other kernel",
                                                                        "move 17 right run", "move 17 same kernel", "move 17 other kernel", "note"], rows_r, label="tab:random_split_run_level",
                                       note_comment="the run level read against idle; the idle row is the boot's share; move 17's shares beside")
    rows_s = []
    for w, st in present:
        for rung in RUNGS:
            sp_rows = _read_rows(out / OUT_DIR / w / STEPS[st][0] / f"split_{rung}.csv")
            if not sp_rows:
                continue
            rem = [int(r["n_removed"] or 0) for r in sp_rows]
            rows_s.append({"window": w, "step": st, "reading": rung, "grid": sp_rows[0]["grid_id"], "recordings": len(sp_rows), "windows": sum(int(r["n_windows"]) for r in sp_rows),
                           "training drawn": sum(int(r["n_train_drawn"]) for r in sp_rows), "removed": sum(rem), "removed per recording": f"{min(rem)} to {max(rem)}, mean {np.mean(rem):.1f}",
                           "training used": sum(int(r["n_train_used"]) for r in sp_rows), "test": sum(int(r["n_test"]) for r in sp_rows)})
    written["split"] = write_table(tdir, "random_split_split", ["window", "step", "reading", "grid", "recordings", "windows", "training drawn", "removed", "removed per recording", "training used", "test"],
                                   rows_s, label="tab:random_split_split", note_comment="the split per window, step and reading: the training windows drawn, those step 2 removes for sharing a pair with a test window, the test windows")
    # the figures: the primary window when it has scores, else the window with the most scored rows (a short corpus has few at W64_H32)
    n_ok = {ww: sum(1 for r in all_scores if r["window"] == ww and r.get("status") == "ok") for ww in {w for w, _ in present}}
    w = PRIMARY_GRID if n_ok.get(PRIMARY_GRID) else max(n_ok, key=lambda ww: n_ok[ww])
    figs = {}
    rows_by_step = {st: _rows_step(out, w, st, "scores.csv") for st in (1, 2)}
    svg, data = fig_accuracy(rows_by_step, w)
    (fdir / "random_split_accuracy.svg").write_text(svg)
    S.write_csv(fdir / "random_split_accuracy.csv", list(data[0].keys()) if data else ["note"], [[r.get(c) for c in data[0].keys()] for r in data] if data else [["no data"]])
    figs["random_split_accuracy"] = str(fdir / "random_split_accuracy.svg")
    for st in (1, 2):
        for name, fn, src in ((f"random_split_per_kernel_step{st}", lambda rows: NB.fig_per_kernel(rows, w), "per_kernel.csv"),
                              (f"random_split_run_level_step{st}", lambda rows: NB.fig_run_level(rows, w), "run_level.csv")):
            svg, data = fn(_rows_step(out, w, st, src))
            (fdir / f"{name}.svg").write_text(relabel(svg, st, w))
            cols = list(data[0].keys()) if data else ["note"]
            S.write_csv(fdir / f"{name}.csv", cols, [[r.get(c) for c in cols] for r in data] if data else [["no data: " + src + " empty"]])
            figs[name] = str(fdir / f"{name}.svg")
    scores_2 = rows_by_step[2]
    for rung in ("apf", "content"):
        s1 = next((r for r in scores_2 if r["level"] == "kernel" and r["rung"] == rung), None)
        svg, data = NB.fig_confusion(out / OUT_DIR / w / STEPS[2][0], rung, w, s1)
        name = f"random_split_confusion_{rung}"
        if svg is None:
            (fdir / f"{name}.svg").write_text("\n".join(svg_open(520, 60, f"Confusion, {rung}, window {w}, random split step 2",
                                                                 f"not drawn: no confusion_kernel_{rung}.csv under {OUT_DIR / w / STEPS[2][0]} (the reading was not run at this window)") + ["</svg>"]))
            S.write_csv(fdir / f"{name}.csv", ["note"], [[f"no confusion for {rung} at {w}"]])
        else:
            (fdir / f"{name}.svg").write_text(relabel(svg, 2, w))
            S.write_csv(fdir / f"{name}.csv", ["true", "predicted", "n_blocks"], [[r["true"], r["predicted"], r["n_blocks"]] for r in data])
        figs[name] = str(fdir / f"{name}.svg")
    S.write_params(tdir / "random_split_tables.csv", "plan11.random_split_tables.v1",
                   {"added": ADDED, "windows_and_steps_present": [f"{w}/{STEPS[st][0]}" for w, st in present], "figure_window": w, "pools": POOLS_NOTE, "by_position": POSITION_NOTE,
                    "null": NO_NULL, "comparators": COMPARATORS_NOTE, "move17_read_not_declared": {ww: move17_results(out, ww)["status"] for ww in {x for x, _ in present}},
                    "inputs_sha256": S.inputs_sha256([out / OUT_DIR / ww / STEPS[st][0] / n for ww, st in present for n in ("scores.csv", "margins.csv", "per_kernel.csv", "run_level.csv")
                                                      if (out / OUT_DIR / ww / STEPS[st][0] / n).is_file()], out)}, CITATION)
    print(f"[randomsplit] tables: {', '.join(str(v['csv']) for v in written.values())}; summary {out / SUMMARY_CSV}; figures: {', '.join(figs)} under {fdir}")
    return {"tables": written, "figures": figs, "summary_csv": str(out / SUMMARY_CSV)}


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="plan11_encoding_ladder.random_split", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="move 18: the random 80/20 split on the five readings at one window, step 1 (plain) or step 2 (the shared-pair training windows removed)")
    r.add_argument("--out", required=True)
    r.add_argument("--step", required=True, type=int, choices=sorted(STEPS), help="1 = the plain random split; 2 = the same split, the training windows sharing a pair with a test window removed")
    r.add_argument("--window", default=PRIMARY_GRID, help=f"{PRIMARY_GRID} (the primary, default), own (each reading's selected window), or a grid id")
    r.add_argument("--seen", type=float, default=DEFAULT_SEEN, help="the share of each recording's windows that train (default 0.8, rounded to windows; move 17's)")
    r.add_argument("--seed-split", type=int, default=SEED_SPLIT, help="the seed of the random split (default 20261010)")
    r.add_argument("--levels", default=",".join(LEVELS), help="archetype,kernel,run (default all three)")
    r.add_argument("--rungs", default=",".join(RUNGS), help="the readings (default all five)")
    r.add_argument("--n-jobs", type=int, default=1)
    r.add_argument("--n-estimators", type=int, default=M.N_ESTIMATORS)
    r.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP, help="resamples of the recordings for the 95%% intervals (default 1000, move 17's)")
    r.add_argument("--seed-offset", type=int, default=0)
    r.add_argument("--dry-run", action="store_true", help="print the plan (the recordings, the windows, training and test per reading, the removal); write nothing")
    t = sub.add_parser("tables", help="move 18: gates/added/random_split.csv, report/tables/random_split_*, report/figures/random_split_*")
    t.add_argument("--out", required=True)
    t.add_argument("--windows", default=f"{PRIMARY_GRID},{WINDOW_OWN}", help="the window folders to read (default W64_H32,own; a missing one is skipped)")
    a = ap.parse_args(argv)
    out = Path(os.path.expanduser(a.out))
    try:
        if a.cmd == "run":
            return run_random_split(out, step=int(a.step), window=a.window, seen=float(a.seen), seed_split=int(a.seed_split), levels=NB.parse_list(a.levels, LEVELS, "levels"),
                                    rungs=NB.parse_list(a.rungs, RUNGS, "rungs"), n_jobs=int(a.n_jobs), n_estimators=int(a.n_estimators), bootstrap=int(a.bootstrap),
                                    seed_offset=int(a.seed_offset), dry_run=bool(a.dry_run))
        write_tables(out, [w.strip() for w in str(a.windows).split(",") if w.strip()])
        return EXIT_OK
    except Stop as exc:
        print(str(exc), file=sys.stderr)
        return exc.code
    except FileNotFoundError as exc:
        print(f"missing input: {exc}", file=sys.stderr)
        return EXIT_MISSING


if __name__ == "__main__":
    sys.exit(main())
