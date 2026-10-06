#!/usr/bin/env python3
"""new_blocks.py -- move 17 of plan11_encoding_ladder (optional; added 2026-10-06): the new-block test on the
encoding paper's five readings, the encoding paper's version of the SPL paper's test (P2_AUTHOR_ANSWERS.md A25;
plan12_grounding/PROMPT_new_blocks_test.md). The split, the gap check, the pooling, the bootstrap and the scoring
are copied from plan12_grounding/new_blocks.py, commit 6c13f9e, 2026-10-07 (plan11 never imports plan12).

  python3 -m plan11_encoding_ladder.new_blocks run --out <run> [--window W64_H32|own|<grid_id>] [--seen 0.8] [--gap 1]
        [--pool 1,2,3] [--levels archetype,kernel,run] [--rungs apf,wapf,persist,content,combined] [--n-jobs N]
        [--n-estimators N] [--bootstrap 1000] [--seed-offset 0] [--dry-run]
  python3 -m plan11_encoding_ladder.new_blocks tables --out <run> [--windows W64_H32,own]

THE QUESTION THE TEST ANSWERS (the author's words: "we have seen things, and now we got a new thing: will we know
to which it belongs? which archetype, which kernel, which workload, which seed")
A model sees the first 80% of every recording's windows, in time order. Then new blocks of the same recordings
arrive, one window at a time. For each new block, and for 2 or 3 consecutive new blocks together, it must name the
archetype, the kernel (workload) and the run (seed). Example: after seeing 80% of all recordings, the rest of gemm's
run rep03 arrives block by block; the answers should be WORKING-SET, gemm, rep03.

THE DATA
Every admissible cell of the run: status ok in cells.csv, all_hard_pass in gates/preconditions.csv, and for the
pair rungs no refused failed_verdict (series.admissible_cells, exactly as the split stage reads the feature files;
the hard-excluded cells stay out, as in move 16); the real run: 95 kernel runs and 8 idle runs, 103. Idle is one more
possible answer at every level. The features are the toolkit's own files features/<rung>/<grid>_norm.npz, the
level-normalized form the forest receives in every split (SPEC 3.1.5); the cut is the run's own (the files were
built with inputs/head_drop.csv; recorded per kernel and for idle in params.json). The comparators take no part:
each reads one window per run, so it has no seen part and no new block to name (said in every output).

THE WINDOWS
  W64_H32   the primary result: the same window for every reading (SPEC 3.1.3: window i covers rows [i*H, i*H+W)).
  own       the secondary rows: each reading's own selected window from gates/selection.json (the run of 2026-09-29:
            apf and persist at W8_H4). A reading whose own window is W64_H32 is not computed again: its rows read
            "same as primary" and carry the primary's numbers when the primary folder exists.
  <grid_id> any grid point, for a smoke run on a short corpus (W8_H4).

THE SPLIT, per recording and reading (flags --seen 0.8 --gap 1, in windows)
The recording's windows in time order (win_start); the first round(seen * n) are SEEN and train the model; then
`gap` windows are skipped, so that no new window shares a pair with any seen window (asserted from the windows'
pair ranges, pairs = head_drop + win_start .. + W - 1; a failure stops the run; the toolkit's within-trace split has
no gap: at 64 x 32 its last training and first test windows share 32 pairs); the windows after the gap are the NEW
BLOCKS, numbered 1, 2, 3 ... by position. Written per recording to split_<rung>.csv.

THE LABELS (three levels)
  archetype  the source family as the feature files carry it (archetype_predicted of cells.csv; lexer is
             SEQUENTIAL-GROW), no G-K0 relabel (the split stage relabels through series.gk0_relabel; this test does
             not, as the SPL test does not); "idle" for idle (the files write "IDLE");
  kernel     the 12 kernels plus idle (13 classes);
  run        each recording its own class (103 in the real run; for a kernel the run is its seed, for idle its boot).

THE MODEL, one random forest per level, reading and window, trained on all seen windows, with the toolkit's settings
  the forest        models.make_forest (300 trees unless --n-estimators), seed nulls.SEED_FOREST + --seed-offset, --n-jobs;
  dimension rule    applied as the split stage applies it (models.fit_predict_units auto_reduce): one training fold of N
                    recordings; a reading with more features than N would be reduced to N by training-fold importance
                    (models._reduce_fit); the largest reading has 60 features and the run 103 recordings, so nothing is
                    reduced; feature_count_used records it;
  L1 quarantine     B1-G3 applied as written with the block as the unit and the one training fold (models.quarantine_l1,
                    max_disagree = models.B1G3_MAX_DISAGREE): a one-feature threshold tree per feature on the seen windows;
                    a feature whose tree reproduces the forest's block predictions on all but at most one block is
                    quarantined, the forest re-run without it, the re-run is the score and the full model kept beside it.
Where a toolkit rule does not apply at window level the outputs say so and nothing replaces it (params.json "rules").

THE SCORES, per level, reading, window and pool size (1, 2, 3 consecutive new blocks of one recording, every
consecutive run of that length; majority vote, a tie to the larger summed class probability, models.aggregate_units)
  scores.csv            accuracy with its 95% bootstrap interval over recordings (--bootstrap 1000 resamples, seed
                        SEED_BOOTSTRAP), macro recall over the classes present among the pools, chance 1/K, the
                        majority class (the most populous training class by recording count, scored on the pools),
                        the counts, the feature counts, the quarantine, the null column
  per_kernel.csv        the share of each kernel's (and idle's) pools named right, at every level
  by_position.csv       the single blocks' accuracy by position after the gap
  run_level.csv         the run level read against idle: per kernel, the share of its pools given the right run, given
                        another run of the same kernel, given a run of another kernel; the idle row is the boot's share
                        (idle runs have a boot and no seed), printed beside every kernel's row
  margins.csv           each reading minus APF (wapf, persist, content, combined) on the same pools, with the bootstrap
                        interval; under `own`, only readings at APF's own window are comparable
  predictions_<level>_<rung>.csv, pools_<level>_<rung>.csv, confusion_kernel_<rung>.csv, split_<rung>.csv
  summary.json, params.json (the inputs' sha256, the label "added 2026-10-06", the citations, the rules)
NO LABEL-SHUFFLE NULL: labels shuffled consistently within a recording are learned from that recording's own seen
windows, so a shuffled model scores like the real one (the within-trace null of the run of 2026-09-29, 0.62, is this
effect). Chance and the majority class are the baselines.

THE TABLES AND FIGURES (`tables`): gates/added/new_blocks.csv (one row per level, reading, window and pool size),
report/tables/new_blocks_scores, new_blocks_margins, new_blocks_run_level, new_blocks_by_position (.csv, .md, .tex as
the toolkit's other tables), report/figures/new_blocks_accuracy.svg (accuracy by level and reading with the chance
lines), new_blocks_by_position.svg, new_blocks_per_kernel.svg (kernels by readings), new_blocks_run_level.svg
(kernels beside idle), new_blocks_confusion_apf.svg and new_blocks_confusion_content.svg, each with its data as CSV.
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

import numpy as np  # noqa: E402

from plan11_encoding_ladder import models as M  # noqa: E402
from plan11_encoding_ladder import schema  # noqa: E402
from plan11_encoding_ladder import series as S  # noqa: E402
from plan11_encoding_ladder import verdicts as V  # noqa: E402
from plan11_encoding_ladder._report_common import write_table  # noqa: E402
from plan11_encoding_ladder.nulls import SEED_FOREST  # noqa: E402

ADDED = "added 2026-10-06"
COPIED = "copied from plan12_grounding/new_blocks.py, commit 6c13f9e, 2026-10-07 (the split, the gap check, the pooling, the bootstrap, the scoring, the SVG frame)"
CITATION = ("P2_AUTHOR_ANSWERS.md A25 (the encoding paper's new-block test); plan12_grounding/PROMPT_new_blocks_test.md (the SPL paper's test, 2026-10-06); "
            + COPIED + "; plan11 models.py make_forest / _reduce_fit / aggregate_units / quarantine_l1 / B1G3_MAX_DISAGREE, nulls.py SEED_FOREST; "
            "series.py features_path / load_features / admissible_cells (SPEC 3.1.3, 3.1.5, 3.3.1), 4.2 (the forest); no label-shuffle null (NO_NULL)")
LEVELS = ("archetype", "kernel", "run")
LEVEL_TEXT = {"archetype": "the source family as the feature files carry it (archetype_predicted; no G-K0 relabel); 'idle' for idle",
              "kernel": "the kernel (the 12 kernels plus idle)", "run": "the run (each recording its own class; for a kernel its seed, for idle its boot)"}
RUNGS = tuple(S.RUNGS)
MARGIN_BASE = "apf"
PRIMARY_GRID = "W64_H32"
WINDOW_OWN = "own"
IDLE_LABEL = "idle"
IDLE_FILE_ARCHETYPE = "IDLE"
HIGHLIGHT = ("lexer", IDLE_LABEL)
CLASS_ORDER = list(schema.KERNEL_NAMES) + [IDLE_LABEL]
SEED_BOOTSTRAP = 20261006
DEFAULT_SEEN, DEFAULT_GAP, DEFAULT_POOLS, DEFAULT_BOOTSTRAP = 0.8, 1, "1,2,3", 1000
OUT_DIR = Path("gates") / "added" / "new_blocks"
SUMMARY_CSV = Path("gates") / "added" / "new_blocks.csv"
NO_NULL = V.not_run("no label-shuffle null: labels shuffled consistently within a recording are learned from that recording's own seen windows, "
                    "so a shuffled model scores like the real one (the within-trace null of the run of 2026-09-29, 0.62, is this effect); "
                    "the baselines are chance 1/K and the majority class")
COMPARATORS_NOTE = ("the comparators (Savoldi 2010, Dhodapkar-Smith 2003, Law 2010) take no part: each reads one window per run (SPEC_epoch2 Part 1.7), "
                    "so it has no seen part and no new block to name")
SAME_AS_PRIMARY = f"same as primary ({PRIMARY_GRID})"
RULES = {
    "splits": "not applicable: the toolkit's splits (splits.folds_for: within-trace, LORO, LOKO) are replaced by the time split of this test (the first "
              "round(seen * n) windows of every recording seen, a gap of `gap` windows, the rest new blocks); the within-trace split has no gap (at 64 x 32 "
              "its last training and first test windows share 32 pairs), so it is not this test",
    "null": NO_NULL,
    "unit": "replaced: the split stage's unit is the cell (majority vote over all its windows, models.aggregate_units); here the unit is the block (one window) "
            "and the pools of 2 and 3 consecutive new blocks, which vote by the same rule (majority, ties by the mean predicted probability, then name order)",
    "headline_macro_recall": "not applied: G-N's macro recall over the headline archetypes (SPEC 3.7.5, archetypes with at least three kernels) exists because a "
                             "held-out kernel's archetype must be represented by other kernels; here every class is seen in training, so macro recall is over every "
                             "class present among the pools, at every level",
    "majority_baseline": "not applicable as written: LOKO's majority baseline is per fold (models.majority_baseline); here the majority is the most populous "
                         "training class by recording count (ties by name), scored on the pools",
    "dimension_rule": "applied as written (models.fit_predict_units auto_reduce, models._reduce_fit train_importance): one training fold of N recordings; a reading "
                      "with more features than N is reduced to N by training-fold importance; the largest reading (combined) has 60 features, fewer than the "
                      "recordings, so nothing is reduced; feature_count_used records it",
    "quarantine": "applied as written (models.quarantine_l1, max_disagree = models.B1G3_MAX_DISAGREE) with the block as the unit and the one training fold: a "
                  "one-feature threshold tree per feature (models.make_l1, max_leaf_nodes = the level's class count) fitted on the seen windows; a feature whose "
                  "tree reproduces the forest's block predictions on all but at most max_disagree blocks is quarantined, the forest re-run without it, the "
                  "re-run is the score and the full model is kept beside it",
    "gk0_relabel": "not applied: the split stage relabels the G-K0 kernels' archetype at read time (models.prepare_split_data, series.gk0_relabel); here the "
                   "archetype level is the feature file's archetype as carried (the author's design, as the SPL test)",
    "admissibility": "applied as the split stage applies it (series.admissible_cells): status ok, all_hard_pass, and for the pair rungs no refused failed_verdict",
    "comparators": COMPARATORS_NOTE,
    "seeds": "the forest's seed is the toolkit's (nulls.SEED_FOREST + --seed-offset), one fit per level, reading and window; the bootstrap has its own seed (SEED_BOOTSTRAP)",
    "cut": "the run's own: the feature files were built with inputs/head_drop.csv (recorded per kernel and for idle); one cut per run folder",
}
EXIT_OK, EXIT_ERROR, EXIT_MISSING = 0, 1, 2
ENC_COLOUR = {"apf": "#8a8a8a", "wapf": "#2b5d8a", "persist": "#7b52ab", "content": "#d9822b", "combined": "#3a9d5d"}
FONT = 'font-family="Helvetica,Arial,sans-serif" font-size="10"'


class Stop(Exception):
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
# the data: one reading at one window, the admissible cells, the labels, the pair ranges
# ---------------------------------------------------------------------------------------------
def head_drop_of(feat: dict) -> dict:
    try:
        return {k: int(v) for k, v in json.loads(str(feat["head_drop_json"])).items()}
    except (KeyError, TypeError, ValueError):
        return {}


def load_window(out: Path, rung: str, grid_id: str, cells: list[dict]) -> dict:
    """The feature file of one reading at one window with the split stage's admissibility applied at read time
    (series.admissible_cells; idle kept), the three label vectors and each window's pair range."""
    p = S.features_path(out, rung, grid_id, True)
    if not p.is_file():
        raise Stop(EXIT_MISSING, f"missing input: {p} (the feature file of {rung} at {grid_id}; moves 6 and 9 to 12 write every grid point)")
    feat = S.load_features(p)
    kept, ex_hard, ex_pair, pre_present = S.admissible_cells(out, cells, rung)
    kept_ids = {c["cell_id"] for c in kept}
    mask = np.array([cid in kept_ids and role in ("kernel", "idle") for cid, role in zip(feat["cell_id"], feat["role"])], dtype=bool)
    rows = np.flatnonzero(mask)
    hd = head_drop_of(feat)
    W, H = int(feat["W"]), int(feat["H"])
    if W < 0:
        raise Stop(EXIT_ERROR, f"stopped: {grid_id} is the whole-cell point (one window per recording), so no recording can be split into seen and new blocks")
    cell_id = feat["cell_id"][rows]
    kernel = feat["kernel"][rows]
    role = feat["role"][rows]
    arche = feat["archetype"][rows].copy()
    arche[kernel == IDLE_LABEL] = IDLE_LABEL
    win_start = feat["win_start"][rows].astype(np.int64)
    hd_row = np.array([hd.get("idle" if ro == "idle" else k, 0) for k, ro in zip(kernel, role)], dtype=np.int64)
    return {"path": p, "X": feat["X"][rows], "names": list(feat["feature_names"]), "cell_id": cell_id.astype(str), "kernel": kernel.astype(str),
            "archetype": arche.astype(str), "rep": feat["rep"][rows].astype(int), "win_start": win_start, "pair_start": win_start + hd_row,
            "W": W, "H": H, "grid_id": str(feat["grid_id"]), "head_drop": hd, "n_windows_dropped": int(feat.get("n_windows_dropped", 0)),
            "excluded_hard": ex_hard, "excluded_pair_rungs": ex_pair, "preconditions_present": pre_present,
            "n_cells": len(set(cell_id.tolist()))}


def resolve_windows(out: Path, window: str, rungs) -> dict:
    """{rung: (grid_id, source)}: the primary grid for every reading, each reading's own selected point under `own`
    (gates/selection.json; a missing selection reads the primary), or the grid id given."""
    res = {}
    for r in rungs:
        if window == WINDOW_OWN:
            gid, src = S.selected_grid_id(out, r, default=None)
            res[r] = (gid, "selection.json") if gid else (PRIMARY_GRID, "no selection: the primary window")
        else:
            res[r] = (window, "the primary window" if window == PRIMARY_GRID else "the window given")
    return res


# ---------------------------------------------------------------------------------------------
# the split (copied from plan12_grounding/new_blocks.py split_rows, commit 6c13f9e, 2026-10-07; pair ranges from the feature file)
# ---------------------------------------------------------------------------------------------
def split_rows(d: dict, seen: float, gap: int) -> dict:
    cell_id, pair_start, W = d["cell_id"], d["pair_start"], d["W"]
    cells = list(dict.fromkeys(cell_id.tolist()))
    per_cell, train, test, pos, bid, failures = {}, [], [], [], [], []
    for c in cells:
        idx = np.flatnonzero(cell_id == c)
        idx = idx[np.argsort(d["win_start"][idx], kind="stable")]
        n = int(idx.size)
        n_seen = int(np.floor(seen * n + 0.5))             # round(seen * n), half up
        first_new = n_seen + int(gap)
        seen_idx, gap_idx, new_idx = idx[:n_seen], idx[n_seen:first_new], idx[first_new:]
        last_seen_end = int(pair_start[seen_idx[-1]] + W - 1) if seen_idx.size else None
        first_new_start = int(pair_start[new_idx[0]]) if new_idx.size else None
        ok = True
        if new_idx.size and seen_idx.size:
            seen_hi = int(pair_start[seen_idx].max() + W - 1)
            ok = int(pair_start[new_idx].min()) > seen_hi       # no new window's range meets any seen window's range
        rec = {"cell_id": c, "n_windows": n, "n_seen": int(seen_idx.size), "n_gap": int(gap_idx.size), "n_new": int(new_idx.size),
               "first_seen_pair": int(pair_start[seen_idx[0]]) if seen_idx.size else None, "last_seen_pair": last_seen_end,
               "first_new_pair": first_new_start, "last_new_pair": int(pair_start[new_idx[-1]] + W - 1) if new_idx.size else None,
               "gap_ok": bool(ok), "note": "" if new_idx.size else "no new block: too few windows for seen + gap + 1"}
        if not ok:
            failures.append(rec)
        per_cell[c] = rec
        train += seen_idx.tolist()
        for k, j in enumerate(new_idx.tolist(), start=1):
            test.append(j); pos.append(k); bid.append(f"{c}|b{k}")
    if failures:
        f = failures[0]
        raise Stop(EXIT_ERROR, f"stopped: the gap assertion failed for {f['cell_id']} (last seen pair {f['last_seen_pair']}, first new pair {f['first_new_pair']}; "
                               f"{len(failures)} recording(s)); a new window shares a pair with a seen one. Nothing scored.")
    return {"cells": cells, "per_cell": per_cell, "train_idx": np.array(train, dtype=np.int64), "test_idx": np.array(test, dtype=np.int64),
            "block_pos": np.array(pos, dtype=np.int64), "block_id": np.array(bid, dtype=str), "n_new_total": len(test)}


# ---------------------------------------------------------------------------------------------
# the forest on the seen windows (copied from plan12_grounding/new_blocks.py fit_level, commit 6c13f9e, 2026-10-07)
# ---------------------------------------------------------------------------------------------
def fit_level(X: np.ndarray, names: list[str], y: np.ndarray, sp: dict, cell_rows: np.ndarray, *, seed: int, n_jobs: int, n_est: int) -> dict:
    tr, te = sp["train_idx"], sp["test_idx"]
    n_train_cells = len(set(cell_rows[tr].tolist()))
    classes_train = sorted(set(y[tr].tolist()))

    def one(Xs):
        Xtr, Xte = Xs[tr], Xs[te]
        d_target = Xtr.shape[1]
        if d_target > n_train_cells:                       # the dimension rule, as fit_predict_units applies it (auto_reduce)
            d_target = n_train_cells
            T = M._reduce_fit(Xtr, y[tr], d_target, M.DIM_MATCH_METHOD, seed, n_est)
            Xtr, Xte = T(Xtr), T(Xte)
        clf = M.make_forest(seed, n_jobs, n_est).fit(Xtr, y[tr])
        proba = clf.predict_proba(Xte)
        classes = [str(c) for c in clf.classes_]
        pred = np.array(classes, dtype=str)[np.argmax(proba, axis=1)]
        return {"pred": pred, "proba": proba, "classes": classes, "feature_count": int(Xs.shape[1]), "feature_count_used": int(Xtr.shape[1]),
                "reduced": bool(Xtr.shape[1] < Xs.shape[1])}

    full = one(X)
    unit_rows = cell_rows.astype(object)                   # object dtype: a fixed-width string array would truncate the block ids
    unit_rows[te] = sp["block_id"].astype(object)
    unit_rows = unit_rows.astype(str)
    lab_u = {"cell_id": unit_rows, "n": int(unit_rows.size)}
    y_unit = {}
    for j in tr.tolist():
        y_unit.setdefault(unit_rows[j], str(y[j]))
    for j in range(len(y)):                                # the gap rows keep their cell's label (they are in no fold)
        y_unit.setdefault(unit_rows[j], str(y[j]))
    for j, b in zip(te.tolist(), sp["block_id"].tolist()):
        y_unit[b] = str(y[j])
    fold = [{"name": "seen_vs_new", "train": tr, "test": te}]
    full_preds = {b: {"y_pred": str(p)} for b, p in zip(sp["block_id"].tolist(), full["pred"].tolist())}
    quar = M.quarantine_l1(X, names, lab_u, y_unit, fold, full_preds, seed=seed)
    with_q = None
    if quar:
        qnames = {q["feature"] for q in quar}
        keep = [j for j, n in enumerate(names) if n not in qnames]
        if keep:
            with_q = {**one(X[:, keep]), "quarantined_features": sorted(qnames)}
        else:
            with_q = {"status": V.not_run("every feature quarantined"), "quarantined_features": sorted(qnames)}
    use = with_q if (with_q and "pred" in with_q) else full
    return {"full": full, "with_quarantine": with_q, "use": use, "quarantine": quar,
            "score_source": (M.SCORE_SOURCE_QUARANTINE if (with_q and "pred" in with_q) else M.SCORE_SOURCE_FULL),
            "quarantined_features": sorted({q["feature"] for q in quar}), "classes_train": classes_train, "n_train_cells": n_train_cells,
            "n_train_windows": int(tr.size), "n_new_blocks": int(te.size), "status": ("ok" if "pred" in use else str(with_q.get("status")))}


# ---------------------------------------------------------------------------------------------
# the pools and the scores (copied from plan12_grounding/new_blocks.py, commit 6c13f9e, 2026-10-07; the accuracy's bootstrap added)
# ---------------------------------------------------------------------------------------------
def pool_votes(sp: dict, pred: np.ndarray, proba: np.ndarray, classes: list[str], y_true_blocks: np.ndarray, pools: list[int]) -> dict:
    block_id, block_pos = sp["block_id"], sp["block_pos"]
    cells = list(dict.fromkeys(b.split("|b")[0] for b in block_id.tolist()))
    rows_of = {c: np.flatnonzero(np.char.startswith(block_id, c + "|b")) for c in cells}
    out = {k: [] for k in pools}
    for c in cells:
        rows = rows_of[c][np.argsort(block_pos[rows_of[c]], kind="stable")]
        m = int(rows.size)
        for k in pools:
            for s in range(0, m - k + 1):
                rr = rows[s:s + k]
                pid = f"{c}|p{k}|{int(block_pos[rr[0]])}"
                agg = M.aggregate_units(np.array([pid] * k), pred[rr], proba[rr], classes)
                lab, vf = agg[pid]
                out[k].append({"cell_id": c, "start_pos": int(block_pos[rr[0]]), "size": k, "y_true": str(y_true_blocks[rr[0]]), "y_pred": str(lab),
                               "vote_fraction": float(vf), "block_rows": rr.tolist()})
    return out


def bootstrap_accuracy(pl: list[dict], cells: list[str], n_boot: int, seed: int) -> tuple:
    """The accuracy's 95% bootstrap interval over recordings (the recordings resampled with replacement)."""
    if not pl:
        return None, None
    cid = np.array([p["cell_id"] for p in pl], dtype=str)
    correct = np.array([p["y_true"] == p["y_pred"] for p in pl], dtype=float)
    present = [c for c in cells if np.any(cid == c)]
    idx = {c: np.flatnonzero(cid == c) for c in present}
    nc = np.array([correct[idx[c]].sum() for c in present]); nn = np.array([idx[c].size for c in present], dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(present), size=(int(n_boot), len(present)))
    tot = nn[draws].sum(axis=1)
    acc = nc[draws].sum(axis=1) / np.where(tot > 0, tot, 1.0)
    return float(np.quantile(acc, 0.025)), float(np.quantile(acc, 0.975))


def score_pools(pl: list[dict], classes_train: list[str], kernel_of_cell: dict, majority_class: str, cells: list[str], n_boot: int) -> dict:
    n = len(pl)
    if n == 0:
        return {"n_pools": 0, "accuracy": None, "ci_lo": None, "ci_hi": None, "macro_recall": None, "chance": (1.0 / len(classes_train)) if classes_train else None,
                "majority": None, "n_classes": len(classes_train), "recall_per_class": {}, "recall_per_kernel": {}}
    correct = np.array([p["y_true"] == p["y_pred"] for p in pl], dtype=bool)
    rpc = {}
    for cls in sorted({p["y_true"] for p in pl}):
        m = np.array([p["y_true"] == cls for p in pl], dtype=bool)
        rpc[cls] = float(correct[m].mean())
    rpk = {}
    for k in sorted({kernel_of_cell[p["cell_id"]] for p in pl}):
        m = np.array([kernel_of_cell[p["cell_id"]] == k for p in pl], dtype=bool)
        rpk[k] = float(correct[m].mean())
    lo, hi = bootstrap_accuracy(pl, cells, n_boot, SEED_BOOTSTRAP)
    return {"n_pools": n, "accuracy": float(correct.mean()), "ci_lo": lo, "ci_hi": hi, "macro_recall": float(np.mean(list(rpc.values()))),
            "chance": 1.0 / len(classes_train), "majority": float(np.mean([p["y_true"] == majority_class for p in pl])), "n_classes": len(classes_train),
            "recall_per_class": rpc, "recall_per_kernel": rpk}


def run_level_shares(pl: list[dict], kernel_of_cell: dict) -> dict:
    out = {}
    for k in sorted({kernel_of_cell[p["cell_id"]] for p in pl}):
        mine = [p for p in pl if kernel_of_cell[p["cell_id"]] == k]
        right = sum(1 for p in mine if p["y_pred"] == p["y_true"])
        same = sum(1 for p in mine if p["y_pred"] != p["y_true"] and kernel_of_cell.get(p["y_pred"]) == k)
        other = len(mine) - right - same
        out[k] = {"n_pools": len(mine), "right_run": right / len(mine), "same_kernel_other_run": same / len(mine), "other_kernel": other / len(mine)}
    return out


def by_position(pl: list[dict]) -> dict:
    out = {}
    for pos in sorted({p["start_pos"] for p in pl}):
        mine = [p for p in pl if p["start_pos"] == pos]
        out[pos] = {"n_blocks": len(mine), "accuracy": float(np.mean([p["y_true"] == p["y_pred"] for p in mine]))}
    return out


def bootstrap_margin(pl_b: list[dict], pl_a: list[dict], cells: list[str], n_boot: int, seed: int) -> dict:
    key = lambda p: (p["cell_id"], p["start_pos"], p["size"])     # noqa: E731
    a = {key(p): p for p in pl_a}
    common = [p for p in pl_b if key(p) in a]
    if not common:
        return {"delta": None, "ci_lo": None, "ci_hi": None, "n_pools": 0, "n_recordings": 0, "gained": 0, "lost": 0, "status": V.not_run("no common pool")}
    cb = np.array([p["y_true"] == p["y_pred"] for p in common], dtype=float)
    ca = np.array([a[key(p)]["y_true"] == a[key(p)]["y_pred"] for p in common], dtype=float)
    cid = np.array([p["cell_id"] for p in common], dtype=str)
    present = [c for c in cells if np.any(cid == c)]
    idx = {c: np.flatnonzero(cid == c) for c in present}
    nb = np.array([cb[idx[c]].sum() for c in present]); na = np.array([ca[idx[c]].sum() for c in present]); nn = np.array([idx[c].size for c in present], dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(present), size=(int(n_boot), len(present)))
    tot = nn[draws].sum(axis=1)
    dd = (nb[draws].sum(axis=1) - na[draws].sum(axis=1)) / np.where(tot > 0, tot, 1.0)
    return {"delta": float(cb.mean() - ca.mean()), "ci_lo": float(np.quantile(dd, 0.025)), "ci_hi": float(np.quantile(dd, 0.975)), "n_pools": int(cb.size),
            "n_recordings": len(present), "gained": int(np.sum((cb == 1) & (ca == 0))), "lost": int(np.sum((cb == 0) & (ca == 1))), "n_resamples": int(n_boot),
            "seed": int(seed), "status": "ok"}


def class_order(labels_present) -> list[str]:
    present = set(labels_present)
    return [c for c in CLASS_ORDER if c in present] + sorted(present - set(CLASS_ORDER))


def confusion_of(blocks: list[dict], labels: list[str]) -> np.ndarray:
    idx = {l: i for i, l in enumerate(labels)}
    Cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for b in blocks:
        if b["y_pred"] in idx and b["y_true"] in idx:
            Cm[idx[b["y_true"]], idx[b["y_pred"]]] += 1
    return Cm


# ---------------------------------------------------------------------------------------------
# one reading at one window
# ---------------------------------------------------------------------------------------------
def one_rung(out: Path, d: dict, rung: str, grid_id: str, grid_source: str, levels, pools: list[int], *, seen: float, gap: int, n_boot: int,
             n_jobs: int, n_est: int, seed: int) -> dict:
    sp = split_rows(d, seen, gap)
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
    res = {"rung": rung, "grid_id": grid_id, "grid_source": grid_source, "W": d["W"], "H": d["H"], "split": sp, "cells": cells, "kernel_of": kernel_of,
           "arche_of": arche_of, "rep_of": rep_of, "levels": {}, "pools_by": {}, "majority_class": majority_class, "n_windows": int(cell_rows.size),
           "labels_kernel": class_order(kernel_of.values())}
    te = sp["test_idx"]
    if sp["n_new_total"] == 0:
        n_w = sorted({v["n_windows"] for v in sp["per_cell"].values()})
        res["status"] = V.not_run(f"no new block at {grid_id}: {n_w[0]}" + (f" to {n_w[-1]}" if len(n_w) > 1 else "") + " windows per recording, too few for seen + gap + 1")
        print(f"[newblocks] {rung:<9} {grid_id}: {res['status']}", flush=True)
        return res
    res["status"] = "ok"
    print(f"[newblocks] {rung:<9} {grid_id}: {len(cells)} recordings, {int(cell_rows.size)} windows of W {d['W']}; seen {int(sp['train_idx'].size)}, new blocks "
          f"{sp['n_new_total']} ({min(v['n_new'] for v in sp['per_cell'].values())} to {max(v['n_new'] for v in sp['per_cell'].values())} per recording); the gap assertion held", flush=True)
    for level in levels:
        y = labels_of[level]
        t0 = time.time()
        fit = fit_level(d["X"], d["names"], y, sp, cell_rows, seed=seed, n_jobs=n_jobs, n_est=n_est)
        use = fit["use"]
        if "pred" not in use:
            res["levels"][level] = {"status": fit["status"], "scores": {}, "quarantine": fit["quarantine"], "quarantined_features": fit["quarantined_features"],
                                    "score_source": fit["score_source"]}
            continue
        pv = pool_votes(sp, use["pred"], use["proba"], use["classes"], y[te], pools)
        res["pools_by"][level] = pv
        sc = {k: score_pools(pv[k], fit["classes_train"], kernel_of, majority_class[level], cells, n_boot) for k in pools}
        ci = {c: i for i, c in enumerate(use["classes"])}
        blocks = [{"block_id": b, "cell_id": b.split("|b")[0], "position": int(sp["block_pos"][i]), "win_start": int(d["win_start"][te[i]]),
                   "pair_start": int(d["pair_start"][te[i]]), "pair_end": int(d["pair_start"][te[i]] + d["W"] - 1), "y_true": str(y[te[i]]), "y_pred": str(use["pred"][i]),
                   "p_pred": float(use["proba"][i].max()), "p_true": float(use["proba"][i][ci[str(y[te[i]])]]) if str(y[te[i]]) in ci else None}
                  for i, b in enumerate(sp["block_id"].tolist())]
        res["levels"][level] = {"status": "ok", "scores": sc, "feature_count": use["feature_count"], "feature_count_used": use["feature_count_used"],
                                "reduced": use["reduced"], "score_source": fit["score_source"], "quarantine": fit["quarantine"], "quarantined_features": fit["quarantined_features"],
                                "full_model_accuracy": (score_pools(pool_votes(sp, fit["full"]["pred"], fit["full"]["proba"], fit["full"]["classes"], y[te], [1])[1],
                                                                    fit["classes_train"], kernel_of, majority_class[level], cells, 1)["accuracy"] if fit["with_quarantine"] else None),
                                "n_train_windows": fit["n_train_windows"], "n_train_cells": fit["n_train_cells"], "n_new_blocks": fit["n_new_blocks"],
                                "classes_train": fit["classes_train"], "by_position": by_position(pv[1]) if 1 in pv else {},
                                "run_level": run_level_shares(pv[1], kernel_of) if (level == "run" and 1 in pv) else None, "blocks": blocks,
                                "elapsed_s": round(time.time() - t0, 1)}
        s1 = sc[1] if 1 in sc else next(iter(sc.values()))
        print(f"[newblocks] {rung:<9} {grid_id} {level:<9} acc {fmt(s1['accuracy'])} [{fmt(s1['ci_lo'])}, {fmt(s1['ci_hi'])}] macro {fmt(s1['macro_recall'])} "
              f"chance {fmt(s1['chance'])} majority {fmt(s1['majority'])}" + "".join(f" | pool {k}: {fmt(sc[k]['accuracy'])}" for k in pools if k != 1)
              + f" | features {use['feature_count_used']}/{use['feature_count']}" + (f" quarantined {len(fit['quarantined_features'])}" if fit["quarantined_features"] else "")
              + f" ({res['levels'][level]['elapsed_s']} s)", flush=True)
    return res


# ---------------------------------------------------------------------------------------------
# the window's tables (the rung results, the margins against APF, the "same as primary" rows)
# ---------------------------------------------------------------------------------------------
SCORE_COLUMNS = ["level", "rung", "grid_id", "grid_source", "pool_size", "status", "n_pools", "n_classes", "accuracy", "ci95_lo", "ci95_hi", "macro_recall",
                 "chance", "majority", "feature_count", "feature_count_used", "score_source", "quarantined_features", "n_train_windows", "n_train_recordings", "null"]
MARGIN_COLUMNS = ["level", "rung", "grid_id", "pool_size", "comparison", "delta_accuracy", "ci95_lo", "ci95_hi", "n_pools", "n_recordings", "n_pools_gained", "n_pools_lost", "status"]
PER_KERNEL_COLUMNS = ["level", "rung", "grid_id", "pool_size", "kernel", "recall", "n_pools"]
BY_POSITION_COLUMNS = ["level", "rung", "grid_id", "position", "n_blocks", "accuracy", "chance"]
RUN_LEVEL_COLUMNS = ["rung", "grid_id", "pool_size", "kernel", "n_pools", "right_run", "same_kernel_other_run", "other_kernel", "note"]


def _read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def primary_rows(out: Path, name: str, rung: str) -> list[list]:
    """The primary window's rows for a reading (its own window is the primary): copied with status 'same as primary'."""
    rows = _read_rows(out / OUT_DIR / PRIMARY_GRID / name)
    cols = {"scores.csv": SCORE_COLUMNS, "margins.csv": MARGIN_COLUMNS, "per_kernel.csv": PER_KERNEL_COLUMNS, "by_position.csv": BY_POSITION_COLUMNS,
            "run_level.csv": RUN_LEVEL_COLUMNS}[name]
    res = []
    for r in rows:
        if r.get("rung") != rung:
            continue
        r = dict(r)
        if "status" in r:
            r["status"] = SAME_AS_PRIMARY + (f"; {r['status']}" if r.get("status") not in ("ok", "") else "")
        if "grid_source" in r:
            r["grid_source"] = "selection.json = " + PRIMARY_GRID
        res.append([r.get(c) for c in cols])
    return res


def write_window(out: Path, window: str, results: dict, same_as_primary: dict, levels, pools: list[int], n_boot: int, params: dict) -> dict:
    d = out / OUT_DIR / window
    d.mkdir(parents=True, exist_ok=True)
    score_rows, margin_rows, pk_rows, bp_rows, rl_rows = [], [], [], [], []
    for rung in RUNGS:
        if rung in same_as_primary:
            has_primary = (out / OUT_DIR / PRIMARY_GRID / "scores.csv").is_file()
            if has_primary:
                score_rows += primary_rows(out, "scores.csv", rung); margin_rows += primary_rows(out, "margins.csv", rung)
                pk_rows += primary_rows(out, "per_kernel.csv", rung); bp_rows += primary_rows(out, "by_position.csv", rung); rl_rows += primary_rows(out, "run_level.csv", rung)
            else:
                for level in levels:
                    for k in pools:
                        score_rows.append([level, rung, PRIMARY_GRID, "selection.json = " + PRIMARY_GRID, k, SAME_AS_PRIMARY + " (run the primary window first)", None, None,
                                           None, None, None, None, None, None, None, None, None, "", None, None, NO_NULL])
            continue
        r = results[rung]
        for level in levels:
            lv = r["levels"].get(level) or {}
            for k in pools:
                s = (lv.get("scores") or {}).get(k) or {}
                score_rows.append([level, rung, r["grid_id"], r["grid_source"], k, lv.get("status") or r["status"], s.get("n_pools"), s.get("n_classes"), s.get("accuracy"),
                                   s.get("ci_lo"), s.get("ci_hi"), s.get("macro_recall"), s.get("chance"), s.get("majority"), lv.get("feature_count"), lv.get("feature_count_used"),
                                   lv.get("score_source"), ";".join(lv.get("quarantined_features") or []), lv.get("n_train_windows"), lv.get("n_train_cells"), NO_NULL])
                for kern in r["labels_kernel"]:
                    if kern in (s.get("recall_per_kernel") or {}):
                        pk_rows.append([level, rung, r["grid_id"], k, kern, s["recall_per_kernel"][kern],
                                        sum(1 for p in r["pools_by"].get(level, {}).get(k, []) if r["kernel_of"][p["cell_id"]] == kern)])
            if lv.get("by_position"):
                chance = lv["scores"][1]["chance"] if 1 in lv["scores"] else None
                for p, v in sorted(lv["by_position"].items()):
                    bp_rows.append([level, rung, r["grid_id"], p, v["n_blocks"], v["accuracy"], chance])
        rl = (r["levels"].get("run") or {})
        if rl.get("run_level") is not None:
            for k in pools:
                shares = run_level_shares(r["pools_by"]["run"][k], r["kernel_of"])
                for kern in r["labels_kernel"]:
                    if kern in shares:
                        v = shares[kern]
                        rl_rows.append([rung, r["grid_id"], k, kern, v["n_pools"], v["right_run"], v["same_kernel_other_run"], v["other_kernel"],
                                        "idle: the boot's share (idle runs have a boot and no seed)" if kern == IDLE_LABEL else ""])
        # the margins against APF on the same pools (the same window only)
        base = results.get(MARGIN_BASE)
        for level in levels:
            for k in pools:
                comp = f"{rung}-{MARGIN_BASE}"
                if rung == MARGIN_BASE:
                    continue
                if base is None or MARGIN_BASE in same_as_primary and window != PRIMARY_GRID:
                    margin_rows.append([level, rung, r["grid_id"], k, comp, None, None, None, None, None, None, None,
                                        V.not_run(f"{MARGIN_BASE} not computed at this window (its own window is {PRIMARY_GRID}: see the primary folder)")])
                    continue
                if base["grid_id"] != r["grid_id"]:
                    margin_rows.append([level, rung, r["grid_id"], k, comp, None, None, None, None, None, None, None,
                                        f"not comparable: different windows ({rung} at {r['grid_id']}, {MARGIN_BASE} at {base['grid_id']})"])
                    continue
                pb, pa = r["pools_by"].get(level, {}).get(k), base["pools_by"].get(level, {}).get(k)
                if pb is None or pa is None:
                    margin_rows.append([level, rung, r["grid_id"], k, comp, None, None, None, None, None, None, None,
                                        "not run: " + str((r["levels"].get(level) or {}).get("status") or r["status"] if pb is None else (base["levels"].get(level) or {}).get("status") or base["status"])])
                    continue
                bm = bootstrap_margin(pb, pa, r["cells"], n_boot, SEED_BOOTSTRAP)
                margin_rows.append([level, rung, r["grid_id"], k, comp, bm["delta"], bm["ci_lo"], bm["ci_hi"], bm["n_pools"], bm["n_recordings"], bm["gained"], bm["lost"], bm["status"]])
        # per rung files
        S.write_csv(d / f"split_{rung}.csv", ["cell_id", "kernel", "grid_id", "n_windows", "n_seen", "n_gap", "n_new", "first_seen_pair", "last_seen_pair", "first_new_pair", "last_new_pair", "gap_ok", "note"],
                    [[c, r["kernel_of"][c], r["grid_id"], *[r["split"]["per_cell"][c][x] for x in ("n_windows", "n_seen", "n_gap", "n_new", "first_seen_pair", "last_seen_pair", "first_new_pair",
                                                                                                  "last_new_pair", "gap_ok", "note")]] for c in r["cells"]])
        for level in levels:
            lv = r["levels"].get(level) or {}
            if not lv.get("blocks"):
                continue
            S.write_csv(d / f"predictions_{level}_{rung}.csv", ["block_id", "cell_id", "kernel", "archetype", "rep", "position", "win_start", "pair_start", "pair_end", "y_true", "y_pred", "p_pred", "p_true"],
                        [[b["block_id"], b["cell_id"], r["kernel_of"][b["cell_id"]], r["arche_of"][b["cell_id"]], r["rep_of"][b["cell_id"]], b["position"], b["win_start"], b["pair_start"],
                          b["pair_end"], b["y_true"], b["y_pred"], b["p_pred"], b["p_true"]] for b in lv["blocks"]])
            S.write_csv(d / f"pools_{level}_{rung}.csv", ["cell_id", "kernel", "pool_size", "start_position", "y_true", "y_pred", "vote_fraction"],
                        [[p["cell_id"], r["kernel_of"][p["cell_id"]], k, p["start_pos"], p["y_true"], p["y_pred"], p["vote_fraction"]] for k in pools for p in r["pools_by"][level][k]])
        kl = r["levels"].get("kernel") or {}
        if kl.get("blocks"):
            Cm = confusion_of(kl["blocks"], r["labels_kernel"])
            S.write_csv(d / f"confusion_kernel_{rung}.csv", ["true\\predicted"] + r["labels_kernel"], [[l] + Cm[i].tolist() for i, l in enumerate(r["labels_kernel"])])
    S.write_csv(d / "scores.csv", SCORE_COLUMNS, score_rows)
    S.write_csv(d / "margins.csv", MARGIN_COLUMNS, margin_rows)
    S.write_csv(d / "per_kernel.csv", PER_KERNEL_COLUMNS, pk_rows)
    S.write_csv(d / "by_position.csv", BY_POSITION_COLUMNS, bp_rows)
    S.write_csv(d / "run_level.csv", RUN_LEVEL_COLUMNS, rl_rows)
    summary = {"added": ADDED, "window": window, "levels": list(levels), "pools": pools, "null": NO_NULL, "comparators": COMPARATORS_NOTE, "rules": RULES,
               "same_as_primary": sorted(same_as_primary), "readings": {}}
    for rung, r in results.items():
        summary["readings"][rung] = {"status": r["status"], "grid_id": r["grid_id"], "grid_source": r["grid_source"], "W": r["W"], "H": r["H"], "n_recordings": len(r["cells"]),
                                     "n_windows": r["n_windows"], "n_seen_windows": int(r["split"]["train_idx"].size), "n_new_blocks": r["split"]["n_new_total"],
                                     "new_blocks_per_recording": {str(k): sum(1 for v in r["split"]["per_cell"].values() if v["n_new"] == k)
                                                                  for k in sorted({v["n_new"] for v in r["split"]["per_cell"].values()})},
                                     "majority_class": r["majority_class"], "classes_kernel": r["labels_kernel"],
                                     "levels": {lv: {k: v for k, v in (r["levels"].get(lv) or {}).items() if k not in ("blocks", "quarantine")}
                                                | {"scores": {str(k): {kk: vv for kk, vv in s.items() if kk != "recall_per_class"} for k, s in ((r["levels"].get(lv) or {}).get("scores") or {}).items()}}
                                                for lv in levels}}
    S.write_json(d / "summary.json", "plan11.new_blocks_window.v1", params, CITATION, summary)
    S.write_json(d / "params.json", "plan11.new_blocks.v1", params, CITATION, {})
    return {"dir": d, "n_score_rows": len(score_rows)}


# ---------------------------------------------------------------------------------------------
# the move: `run`
# ---------------------------------------------------------------------------------------------
def parse_list(spec: str, allowed, what: str) -> list[str]:
    items = [s.strip() for s in str(spec).split(",") if s.strip()]
    bad = [i for i in items if i not in allowed]
    if bad or not items:
        raise Stop(EXIT_MISSING, f"--{what} {spec!r}: choose from {','.join(allowed)}")
    return items


def parse_pools(spec: str) -> list[int]:
    try:
        pools = sorted({int(s) for s in str(spec).split(",") if s.strip()})
    except ValueError:
        raise Stop(EXIT_MISSING, f"--pool {spec!r}: a comma-separated list of positive integers (default {DEFAULT_POOLS})")
    if not pools or min(pools) < 1:
        raise Stop(EXIT_MISSING, f"--pool {spec!r}: a comma-separated list of positive integers (default {DEFAULT_POOLS})")
    return pools


def run_new_blocks(out: Path, *, window: str = PRIMARY_GRID, seen: float = DEFAULT_SEEN, gap: int = DEFAULT_GAP, pools=None, levels=LEVELS, rungs=RUNGS,
                   n_jobs: int = 1, n_estimators: int = M.N_ESTIMATORS, bootstrap: int = DEFAULT_BOOTSTRAP, seed_offset: int = 0, dry_run: bool = False) -> int:
    out = Path(out)
    pools = pools or [1, 2, 3]
    if not (out / "cells.csv").is_file():
        raise Stop(EXIT_MISSING, f"missing input: {out / 'cells.csv'}")
    if not (0.0 < seen < 1.0) or gap < 0 or bootstrap < 1:
        raise Stop(EXIT_MISSING, f"--seen must lie in (0, 1) (given {seen}), --gap be 0 or more (given {gap}), --bootstrap 1 or more (given {bootstrap})")
    cells = S.load_cells(out / "cells.csv")
    windows = resolve_windows(out, window, rungs)
    same_as_primary = {r for r, (g, _) in windows.items() if window == WINDOW_OWN and g == PRIMARY_GRID}
    seed = SEED_FOREST + int(seed_offset)
    data = {}
    for r in rungs:
        if r in same_as_primary:
            continue
        data[r] = load_window(out, r, windows[r][0], cells)
    # the plan, in plain words
    n_adm = {r: d["n_cells"] for r, d in data.items()}
    first = next(iter(data.values()), None)
    print(f"[newblocks] the new-block test, window {window}: seen {seen:g} of each recording's windows, a gap of {gap} window(s), then the new blocks; pools {pools}; "
          f"levels {', '.join(levels)}; readings {', '.join(rungs)}; {NO_NULL[:60]}...")
    print(f"[newblocks] run {out}; forest: {n_estimators} trees, {n_jobs} processes, seed {seed}; bootstrap {bootstrap} resamples, seed {SEED_BOOTSTRAP}")
    if first is not None:
        n_k = sum(1 for c in cells if c["role"] == "kernel" and c["cell_id"] in set(first["cell_id"].tolist()))
        n_i = sum(1 for c in cells if c["role"] == "idle" and c["cell_id"] in set(first["cell_id"].tolist()))
        print(f"[newblocks] recordings found: {first['n_cells']} = {n_k} kernel + {n_i} idle (the real run expects 103 = 95 kernel + 8 idle); "
              f"hard-excluded {len(first['excluded_hard'])}, pair-rung excluded {len(first['excluded_pair_rungs'])}; the cut: head drop {first['head_drop']}")
    for r in rungs:
        gid, src = windows[r]
        if r in same_as_primary:
            print(f"[newblocks] {r:<9} own window {gid} ({src}): {SAME_AS_PRIMARY}")
            continue
        d = data[r]
        sp_counts = {}
        for c in list(dict.fromkeys(d["cell_id"].tolist())):
            n = int(np.sum(d["cell_id"] == c)); n_seen = int(np.floor(seen * n + 0.5)); n_new = max(0, n - n_seen - gap)
            sp_counts[c] = (n, n_seen, n_new)
        ns = [v[0] for v in sp_counts.values()]; nn = [v[2] for v in sp_counts.values()]
        print(f"[newblocks] {r:<9} {gid} ({src}): {d['n_cells']} recordings (admissible {n_adm[r]}), {int(d['cell_id'].size)} windows of W {d['W']} H {d['H']}, "
              f"{min(ns)} to {max(ns)} per recording; seen {sum(v[1] for v in sp_counts.values())}, gap {gap} per recording, new blocks {sum(nn)} ({min(nn)} to {max(nn)} per recording)"
              + (f"; {sum(1 for x in nn if x == 0)} recording(s) without a new block" if any(x == 0 for x in nn) else ""))
    if dry_run:
        print(f"[newblocks] dry run: would write {out / OUT_DIR / window}/ (scores.csv, per_kernel.csv, by_position.csv, run_level.csv, margins.csv, predictions_<level>_<rung>.csv, "
              f"pools_<level>_<rung>.csv, confusion_kernel_<rung>.csv, split_<rung>.csv, summary.json, params.json); nothing written")
        return EXIT_OK
    results = {}
    for r in rungs:
        if r in same_as_primary:
            continue
        results[r] = one_rung(out, data[r], r, windows[r][0], windows[r][1], levels, pools, seen=seen, gap=gap, n_boot=bootstrap, n_jobs=n_jobs, n_est=n_estimators, seed=seed)
    inputs = [out / "cells.csv", out / "inputs" / "head_drop.csv", out / "gates" / "preconditions.json", out / "gates" / "preconditions.csv"] + [d["path"] for d in data.values()]
    if window == WINDOW_OWN:
        inputs += [out / "gates" / "selection.json", out / OUT_DIR / PRIMARY_GRID / "scores.csv"]
    params = {"added": ADDED, "window": window, "grids": {r: {"grid_id": g, "source": s, "same_as_primary": r in same_as_primary} for r, (g, s) in windows.items()},
              "seen": seen, "gap": gap, "pools": pools, "levels": list(levels), "rungs": list(rungs), "n_estimators": int(n_estimators), "n_jobs": int(n_jobs),
              "seed_forest": seed, "seed_offset": int(seed_offset), "bootstrap": int(bootstrap), "seed_bootstrap": SEED_BOOTSTRAP,
              "head_drop": (first["head_drop"] if first else {}), "cut": RULES["cut"], "idle_label": IDLE_LABEL, "idle_archetype_in_feature_files": IDLE_FILE_ARCHETYPE,
              "admissibility": {"hard_excluded": sorted({c for d in data.values() for c in d["excluded_hard"]}), "pair_rung_excluded": {r: d["excluded_pair_rungs"] for r, d in data.items()},
                                "preconditions_present": bool(first and first["preconditions_present"])},
              "rules": RULES, "comparators": COMPARATORS_NOTE, "null": NO_NULL, "copied": COPIED, "inputs_sha256": S.inputs_sha256([p for p in inputs if p.is_file()], out)}
    w = write_window(out, window, results, same_as_primary, levels, pools, bootstrap, params)
    for r, res in results.items():
        if res["status"] != "ok":
            continue
        line = "; ".join(f"{lv}: {fmt(((res['levels'].get(lv) or {}).get('scores') or {}).get(1, {}).get('accuracy'))}" for lv in levels)
        print(f"[newblocks] {r:<9} {res['grid_id']}: {res['split']['n_new_total']} new blocks; single-block accuracy {line}")
    print(f"[newblocks] written: {w['dir']} ({w['n_score_rows']} score rows)")
    return EXIT_OK


# ---------------------------------------------------------------------------------------------
# the tables and the figures: `tables`
# ---------------------------------------------------------------------------------------------
def svg_open(width: int, height: int, title: str, sub: str = "") -> list[str]:
    # copied from plan12_grounding/figures.py svg_open, commit 6c13f9e, 2026-10-07
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" {FONT}>',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="10" y="16" font-size="12" font-weight="bold">{html.escape(title)}</text>'
            + (f'<text x="10" y="30" fill="#555">{html.escape(sub)}</text>' if sub else "")]


def _f(x):
    try:
        return None if x in (None, "") else float(x)
    except (TypeError, ValueError):
        return None


def _rows_window(out: Path, window: str, name: str) -> list[dict]:
    return _read_rows(out / OUT_DIR / window / name)


def fig_accuracy(rows: list[dict], window: str, pools: list[int]) -> tuple:
    """Three panels (the levels); per reading the single-block accuracy as a bar with its bootstrap interval, the pools of 2 and 3 as
    ticks, chance 1/K as a grey line and the majority class as a dashed red line."""
    W_, H_ = 980, 330
    out = svg_open(W_, H_, f"New blocks named right, by level and reading, window {window}",
                   "bars: accuracy on single new blocks with the 95% bootstrap interval over recordings; ticks: pools of 2 and 3 consecutive blocks; grey: chance 1/K; dashed red: the majority class")
    x0, y0, w, h = 60, 60, 285, 200
    data = []
    for li, level in enumerate(LEVELS):
        px = x0 + li * (w + 25)
        out.append(f'<text x="{px}" y="{y0 - 6}" font-weight="bold">{level}</text>')
        out.append(f'<rect x="{px}" y="{y0}" width="{w}" height="{h}" fill="none" stroke="#ccc"/>')
        for t in (0.0, 0.5, 1.0):
            yy = y0 + h - h * t
            out.append(f'<line x1="{px}" y1="{yy:.1f}" x2="{px + w}" y2="{yy:.1f}" stroke="#eee"/><text x="{px - 4}" y="{yy + 3:.1f}" text-anchor="end">{t:.1f}</text>')
        bw = w / len(RUNGS)
        for ri, rung in enumerate(RUNGS):
            bx = px + ri * bw + 6
            s1 = next((r for r in rows if r["level"] == level and r["rung"] == rung and r["pool_size"] == "1"), None) or {}
            acc = _f(s1.get("accuracy"))
            data.append({"level": level, "rung": rung, "grid_id": s1.get("grid_id"), "status": s1.get("status"), "accuracy": acc, "ci95_lo": _f(s1.get("ci95_lo")),
                         "ci95_hi": _f(s1.get("ci95_hi")), "chance": _f(s1.get("chance")), "majority": _f(s1.get("majority")),
                         **{f"pool{k}": _f((next((r for r in rows if r["level"] == level and r["rung"] == rung and r["pool_size"] == str(k)), None) or {}).get("accuracy")) for k in pools}})
            if acc is None:
                out.append(f'<text x="{bx + (bw - 12) / 2:.1f}" y="{y0 + h / 2:.1f}" text-anchor="middle" fill="#999" font-size="8">{html.escape(str(s1.get("status") or "not run")[:18])}</text>')
            else:
                out.append(f'<rect x="{bx:.1f}" y="{y0 + h - h * acc:.1f}" width="{bw - 12:.1f}" height="{h * acc:.1f}" fill="{ENC_COLOUR[rung]}" fill-opacity="0.85"/>')
                lo, hi = _f(s1.get("ci95_lo")), _f(s1.get("ci95_hi"))
                if lo is not None and hi is not None:
                    cx = bx + (bw - 12) / 2
                    out.append(f'<line x1="{cx:.1f}" y1="{y0 + h - h * hi:.1f}" x2="{cx:.1f}" y2="{y0 + h - h * lo:.1f}" stroke="#222" stroke-width="1.2"/>')
                out.append(f'<text x="{bx + (bw - 12) / 2:.1f}" y="{y0 + h - h * acc - 4:.1f}" text-anchor="middle">{acc:.2f}</text>')
                for k in pools:
                    if k == 1:
                        continue
                    pk = _f((next((r for r in rows if r["level"] == level and r["rung"] == rung and r["pool_size"] == str(k)), None) or {}).get("accuracy"))
                    if pk is not None:
                        yy = y0 + h - h * pk
                        out.append(f'<line x1="{bx - 2:.1f}" y1="{yy:.1f}" x2="{bx + bw - 10:.1f}" y2="{yy:.1f}" stroke="#222" stroke-width="1"/><text x="{bx + bw - 9:.1f}" y="{yy + 3:.1f}" font-size="7">{k}</text>')
            ch, mj = _f(s1.get("chance")), _f(s1.get("majority"))
            if ch is not None:
                out.append(f'<line x1="{bx - 3:.1f}" y1="{y0 + h - h * ch:.1f}" x2="{bx + bw - 9:.1f}" y2="{y0 + h - h * ch:.1f}" stroke="#888"/>')
            if mj is not None:
                out.append(f'<line x1="{bx - 3:.1f}" y1="{y0 + h - h * mj:.1f}" x2="{bx + bw - 9:.1f}" y2="{y0 + h - h * mj:.1f}" stroke="#b03a2e" stroke-dasharray="3,2"/>')
            out.append(f'<text x="{bx + (bw - 12) / 2:.1f}" y="{y0 + h + 12}" text-anchor="middle">{rung}</text>')
    out.append(f'<text x="{x0}" y="{H_ - 10}" fill="#555">{html.escape(NO_NULL[:120])}...</text></svg>')
    return "\n".join(out), data


def fig_by_position(rows: list[dict], window: str) -> tuple:
    positions = sorted({int(r["position"]) for r in rows if r.get("position")})
    W_, H_ = 980, 300
    out = svg_open(W_, H_, f"New blocks named right, by their position after the gap, window {window}", "lines: accuracy of the single new blocks at position 1, 2, 3 ... after the gap, one line per reading; grey: chance 1/K")
    x0, y0, w, h = 60, 60, 285, 180
    P = max(positions) if positions else 1
    data = []
    for li, level in enumerate(LEVELS):
        px = x0 + li * (w + 25)
        out.append(f'<text x="{px}" y="{y0 - 6}" font-weight="bold">{level}</text>')
        out.append(f'<rect x="{px}" y="{y0}" width="{w}" height="{h}" fill="none" stroke="#ccc"/>')
        for t in (0.0, 0.5, 1.0):
            yy = y0 + h - h * t
            out.append(f'<line x1="{px}" y1="{yy:.1f}" x2="{px + w}" y2="{yy:.1f}" stroke="#eee"/><text x="{px - 4}" y="{yy + 3:.1f}" text-anchor="end">{t:.1f}</text>')
        X = lambda p: px + 12 + (w - 24) * ((p - 1) / max(1, P - 1))      # noqa: E731
        for p in positions:
            out.append(f'<text x="{X(p):.1f}" y="{y0 + h + 12}" text-anchor="middle">{p}</text>')
        chance = None
        for rung in RUNGS:
            pts = sorted([(int(r["position"]), _f(r["accuracy"])) for r in rows if r["level"] == level and r["rung"] == rung and r.get("position")], key=lambda t: t[0])
            pts = [(p, a) for p, a in pts if a is not None]
            for p, a in pts:
                data.append({"level": level, "rung": rung, "position": p, "accuracy": a})
            if not pts:
                continue
            out.append(f'<polyline points="{" ".join(f"{X(p):.1f},{y0 + h - h * a:.1f}" for p, a in pts)}" fill="none" stroke="{ENC_COLOUR[rung]}" stroke-width="2"/>')
            for p, a in pts:
                out.append(f'<circle cx="{X(p):.1f}" cy="{y0 + h - h * a:.1f}" r="2.5" fill="{ENC_COLOUR[rung]}"/>')
            chance = _f(next((r for r in rows if r["level"] == level and r["rung"] == rung and r.get("chance")), {}).get("chance")) or chance
        if chance is not None:
            out.append(f'<line x1="{px}" y1="{y0 + h - h * chance:.1f}" x2="{px + w}" y2="{y0 + h - h * chance:.1f}" stroke="#888"/>')
        out.append(f'<text x="{px + w / 2:.1f}" y="{y0 + h + 24}" text-anchor="middle" fill="#555" font-size="9">position after the gap</text>')
    legend = "; ".join('<tspan fill="%s">%s</tspan>' % (ENC_COLOUR[r], r) for r in RUNGS)
    out.append(f'<text x="{x0}" y="{H_ - 10}" fill="#555">{legend}</text></svg>')
    return "\n".join(out), data


def fig_per_kernel(rows: list[dict], window: str) -> tuple:
    """The kernel level: per kernel (and idle), the share of its single new blocks named right, grouped bars per reading."""
    labels = class_order({r["kernel"] for r in rows if r["level"] == "kernel" and r["pool_size"] == "1"})
    n = len(labels)
    gw = 5 * 7 + 6
    W_ = 70 + n * gw + 30
    ph = 140
    H_ = 50 + ph + 60
    out = svg_open(W_, H_, f"Each kernel's new blocks named right at the kernel level, by reading, window {window}",
                   "bars per kernel: apf, wapf, persist, content, combined (left to right), single new blocks; the lexer and idle groups are highlighted")
    px, py = 60, 50
    out.append(f'<rect x="{px}" y="{py}" width="{n * gw}" height="{ph}" fill="none" stroke="#ccc"/>')
    for t in (0.0, 0.5, 1.0):
        yy = py + ph - ph * t
        out.append(f'<line x1="{px}" y1="{yy:.1f}" x2="{px + n * gw}" y2="{yy:.1f}" stroke="#eee"/><text x="{px - 4}" y="{yy + 3:.1f}" text-anchor="end">{t:.1f}</text>')
    data = []
    for ki, k in enumerate(labels):
        gx = px + ki * gw
        if k in HIGHLIGHT:
            out.append(f'<rect x="{gx}" y="{py}" width="{gw}" height="{ph}" fill="#fff3c4" stroke="none"/>')
        for ri, rung in enumerate(RUNGS):
            v = _f((next((r for r in rows if r["level"] == "kernel" and r["pool_size"] == "1" and r["rung"] == rung and r["kernel"] == k), None) or {}).get("recall"))
            data.append({"kernel": k, "rung": rung, "recall": v})
            if v is not None:
                out.append(f'<rect x="{gx + 3 + ri * 7}" y="{py + ph - ph * v:.1f}" width="6" height="{ph * v:.1f}" fill="{ENC_COLOUR[rung]}" fill-opacity="0.9"/>')
        bold = ' font-weight="bold"' if k in HIGHLIGHT else ""
        out.append(f'<text x="{gx + gw / 2:.1f}" y="{py + ph + 10}" text-anchor="end" font-size="8"{bold} transform="rotate(-45 {gx + gw / 2:.1f},{py + ph + 10})">{html.escape(k)}</text>')
    legend = "; ".join('<tspan fill="%s">%s</tspan>' % (ENC_COLOUR[r], r) for r in RUNGS)
    out.append(f'<text x="{px}" y="{H_ - 8}" fill="#555">{legend}</text></svg>')
    return "\n".join(out), data


def fig_run_level(rows: list[dict], window: str) -> tuple:
    """The run level read against idle: per reading a panel; per kernel and idle a stacked bar of the shares of its single new blocks given the
    right run, another run of the same kernel (for idle, another boot), a run of another kernel."""
    labels = class_order({r["kernel"] for r in rows if r["pool_size"] == "1"})
    n = len(labels)
    pw, ph, left = 60 + n * 14, 110, 50
    W_ = 20 + pw + 40
    H_ = 50 + len(RUNGS) * (ph + 45)
    out = svg_open(W_, H_, f"The run level read against idle, window {window}",
                   "per reading: each kernel's single new blocks given the right run (dark), another run of the same kernel (mid; for idle, another boot), a run of another kernel (light)")
    data = []
    for pi, rung in enumerate(RUNGS):
        px, py = left, 50 + pi * (ph + 45)
        out.append(f'<text x="{px}" y="{py - 6}" font-weight="bold">{rung}</text>')
        out.append(f'<rect x="{px}" y="{py}" width="{n * 14}" height="{ph}" fill="none" stroke="#ccc"/>')
        for t in (0.0, 0.5, 1.0):
            yy = py + ph - ph * t
            out.append(f'<line x1="{px}" y1="{yy:.1f}" x2="{px + n * 14}" y2="{yy:.1f}" stroke="#eee"/><text x="{px - 4}" y="{yy + 3:.1f}" text-anchor="end">{t:.1f}</text>')
        for ki, k in enumerate(labels):
            r = next((x for x in rows if x["rung"] == rung and x["pool_size"] == "1" and x["kernel"] == k), None)
            gx = px + ki * 14
            if k in HIGHLIGHT:
                out.append(f'<rect x="{gx}" y="{py}" width="14" height="{ph}" fill="#fff3c4" stroke="none"/>')
            if r is None:
                continue
            right, same, other = _f(r.get("right_run")) or 0.0, _f(r.get("same_kernel_other_run")) or 0.0, _f(r.get("other_kernel")) or 0.0
            data.append({"rung": rung, "kernel": k, "n_pools": r.get("n_pools"), "right_run": right, "same_kernel_other_run": same, "other_kernel": other})
            y = py + ph
            for share, colour in ((right, ENC_COLOUR[rung]), (same, "#bbbbbb"), (other, "#e8e8e8")):
                hh = ph * share
                out.append(f'<rect x="{gx + 2}" y="{y - hh:.1f}" width="10" height="{hh:.1f}" fill="{colour}" stroke="#999" stroke-width="0.3"/>')
                y -= hh
            bold = ' font-weight="bold"' if k in HIGHLIGHT else ""
            out.append(f'<text x="{gx + 7:.1f}" y="{py + ph + 9}" text-anchor="end" font-size="7"{bold} transform="rotate(-45 {gx + 7:.1f},{py + ph + 9})">{html.escape(k)}</text>')
    out.append("</svg>")
    return "\n".join(out), data


def fig_confusion(out_dir: Path, rung: str, window: str, score_row: dict | None) -> tuple:
    rows = _read_rows(out_dir / f"confusion_kernel_{rung}.csv")
    if not rows:
        return None, []
    labels = [k for k in rows[0].keys() if k != "true\\predicted"]
    Cm = np.array([[int(float(r[l] or 0)) for l in labels] for r in rows], dtype=np.int64)
    n = len(labels)
    cell, left, top = 22, 124, 100
    W_ = left + n * cell + 40
    H_ = top + n * cell + 70
    out = svg_open(W_, H_, f"Confusion on single new blocks: the kernel level, {rung}, window {window}",
                   "rows: the true kernel; columns: the predicted kernel; the number of new blocks; shade: the share of the row; the lexer and idle rows are highlighted")
    px, py = left, top
    for i, l in enumerate(labels):
        if l in HIGHLIGHT:
            out.append(f'<rect x="{px - 116}" y="{py + i * cell}" width="{116 + n * cell}" height="{cell}" fill="#fff3c4" stroke="#d9822b" stroke-width="0.8"/>')
    for j, l in enumerate(labels):
        bold = ' font-weight="bold"' if l in HIGHLIGHT else ""
        out.append(f'<text x="{px + j * cell + cell / 2:.1f}" y="{py - 5}" text-anchor="end" font-size="9"{bold} transform="rotate(-60 {px + j * cell + cell / 2:.1f},{py - 5})">{html.escape(l)}</text>')
    data = []
    for i, l in enumerate(labels):
        bold = ' font-weight="bold"' if l in HIGHLIGHT else ""
        out.append(f'<text x="{px - 6}" y="{py + i * cell + cell * 0.7:.1f}" text-anchor="end" font-size="9"{bold}>{html.escape(l)}</text>')
        row = int(Cm[i].sum())
        for j in range(n):
            v = int(Cm[i, j])
            data.append({"true": l, "predicted": labels[j], "n_blocks": v})
            share = v / row if row else 0.0
            fill = f"rgb({int(255 - 180 * share)},{int(255 - 120 * share)},{int(255 - 60 * share)})" if v else ("none" if l in HIGHLIGHT else "white")
            out.append(f'<rect x="{px + j * cell}" y="{py + i * cell}" width="{cell}" height="{cell}" fill="{fill}" stroke="#ddd"/>')
            if v:
                out.append(f'<text x="{px + j * cell + cell / 2:.1f}" y="{py + i * cell + cell * 0.7:.1f}" text-anchor="middle" font-size="9">{v}</text>')
    yb = py + n * cell + 16
    s = score_row or {}
    out.append(f'<text x="{px - 116}" y="{yb}" fill="#555" font-size="9">accuracy {fmt(_f(s.get("accuracy")))} [{fmt(_f(s.get("ci95_lo")))}, {fmt(_f(s.get("ci95_hi")))}], '
               f'macro recall {fmt(_f(s.get("macro_recall")))}, {s.get("n_pools")} new blocks, {n} classes; chance {fmt(_f(s.get("chance")))}, majority {fmt(_f(s.get("majority")))}</text>')
    out.append(f'<text x="{px - 116}" y="{yb + 14}" fill="#555" font-size="9">{html.escape(NO_NULL[:150])}</text>')
    out.append(f'<text x="{px - 116}" y="{yb + 28}" fill="#555" font-size="9">rows and columns in the class order of cells.csv (schema.KERNEL_NAMES), idle last; highlighted: {", ".join(HIGHLIGHT)}</text>')
    out.append("</svg>")
    return "\n".join(out), data


def write_tables(out: Path, windows=(PRIMARY_GRID, WINDOW_OWN)) -> dict:
    """gates/added/new_blocks.csv, report/tables/new_blocks_*.{csv,md,tex} and report/figures/new_blocks_*.svg (each with its CSV)."""
    out = Path(out)
    tdir = out / "report" / "tables"
    fdir = out / "report" / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    present = [w for w in windows if (out / OUT_DIR / w / "scores.csv").is_file()]
    if not present:
        raise Stop(EXIT_MISSING, f"missing input: {out / OUT_DIR}/<window>/scores.csv for {', '.join(windows)} (run `new_blocks run` first)")
    all_scores = []
    for w in present:
        for r in _rows_window(out, w, "scores.csv"):
            all_scores.append({"window": w, **r})
    S.write_csv(SUMMARY_CSV if SUMMARY_CSV.is_absolute() else out / SUMMARY_CSV, ["window"] + SCORE_COLUMNS, [[r.get(c) for c in ["window"] + SCORE_COLUMNS] for r in all_scores])
    pools = sorted({int(r["pool_size"]) for r in all_scores if r.get("pool_size")})
    fm = lambda x, d=3: ("" if _f(x) is None else f"{_f(x):.{d}f}")        # noqa: E731
    rows_t = [{"window": r["window"], "level": r["level"], "reading": r["rung"], "grid": r["grid_id"], "pool": r["pool_size"], "status": r.get("status"),
               "new blocks": r.get("n_pools"), "classes": r.get("n_classes"), "accuracy": fm(r.get("accuracy")),
               "95% interval": (f"[{fm(r.get('ci95_lo'))}, {fm(r.get('ci95_hi'))}]" if _f(r.get("ci95_lo")) is not None else ""),
               "macro recall": fm(r.get("macro_recall")), "chance": fm(r.get("chance")), "majority": fm(r.get("majority")), "features used": r.get("feature_count_used")}
              for r in all_scores]
    written = {"scores": write_table(tdir, "new_blocks_scores", ["window", "level", "reading", "grid", "pool", "status", "new blocks", "classes", "accuracy", "95% interval",
                                                                 "macro recall", "chance", "majority", "features used"], rows_t, label="tab:new_blocks_scores",
                                     note_comment=f"the new-block test ({ADDED}): single blocks and pools of consecutive blocks; {NO_NULL}; {COMPARATORS_NOTE}")}
    rows_m = []
    for w in present:
        for r in _rows_window(out, w, "margins.csv"):
            rows_m.append({"window": w, "level": r["level"], "comparison": r["comparison"], "grid": r["grid_id"], "pool": r["pool_size"], "delta accuracy": fm(r.get("delta_accuracy")),
                           "95% interval": (f"[{fm(r.get('ci95_lo'))}, {fm(r.get('ci95_hi'))}]" if _f(r.get("ci95_lo")) is not None else ""), "pools": r.get("n_pools"),
                           "gained": r.get("n_pools_gained"), "lost": r.get("n_pools_lost"), "status": r.get("status")})
    written["margins"] = write_table(tdir, "new_blocks_margins", ["window", "level", "comparison", "grid", "pool", "delta accuracy", "95% interval", "pools", "gained", "lost", "status"],
                                     rows_m, label="tab:new_blocks_margins", note_comment="each reading minus APF on the same new blocks, the 95% bootstrap interval over recordings")
    rows_r = []
    for w in present:
        for r in _rows_window(out, w, "run_level.csv"):
            rows_r.append({"window": w, "reading": r["rung"], "grid": r["grid_id"], "pool": r["pool_size"], "kernel": r["kernel"], "new blocks": r.get("n_pools"),
                           "right run": fm(r.get("right_run")), "same kernel, other run": fm(r.get("same_kernel_other_run")), "other kernel": fm(r.get("other_kernel")), "note": r.get("note")})
    written["run_level"] = write_table(tdir, "new_blocks_run_level", ["window", "reading", "grid", "pool", "kernel", "new blocks", "right run", "same kernel, other run", "other kernel", "note"],
                                       rows_r, label="tab:new_blocks_run_level", note_comment="the run level read against idle; the idle row is the boot's share")
    rows_p = []
    for w in present:
        for r in _rows_window(out, w, "by_position.csv"):
            rows_p.append({"window": w, "level": r["level"], "reading": r["rung"], "grid": r["grid_id"], "position": r["position"], "blocks": r.get("n_blocks"),
                           "accuracy": fm(r.get("accuracy")), "chance": fm(r.get("chance"))})
    written["by_position"] = write_table(tdir, "new_blocks_by_position", ["window", "level", "reading", "grid", "position", "blocks", "accuracy", "chance"], rows_p,
                                         label="tab:new_blocks_by_position", note_comment="the single new blocks by their position after the gap")
    # the figures: the primary window when it has scores, else the window with the most scored rows (a short corpus has none at W64_H32)
    n_ok = {ww: sum(1 for r in all_scores if r["window"] == ww and r.get("status") == "ok") for ww in present}
    w = PRIMARY_GRID if n_ok.get(PRIMARY_GRID) else max(present, key=lambda ww: n_ok[ww])
    figs = {}
    for name, fn, src in (("new_blocks_accuracy", lambda rows: fig_accuracy(rows, w, pools), "scores.csv"), ("new_blocks_by_position", lambda rows: fig_by_position(rows, w), "by_position.csv"),
                          ("new_blocks_per_kernel", lambda rows: fig_per_kernel(rows, w), "per_kernel.csv"), ("new_blocks_run_level", lambda rows: fig_run_level(rows, w), "run_level.csv")):
        svg, data = fn(_rows_window(out, w, src))
        (fdir / f"{name}.svg").write_text(svg)
        cols = list(data[0].keys()) if data else ["note"]
        S.write_csv(fdir / f"{name}.csv", cols, [[r.get(c) for c in cols] for r in data] if data else [["no data: " + src + " empty"]])
        figs[name] = str(fdir / f"{name}.svg")
    scores_w = _rows_window(out, w, "scores.csv")
    for rung in ("apf", "content"):
        s1 = next((r for r in scores_w if r["level"] == "kernel" and r["rung"] == rung and r["pool_size"] == "1"), None)
        svg, data = fig_confusion(out / OUT_DIR / w, rung, w, s1)
        name = f"new_blocks_confusion_{rung}"
        if svg is None:
            (fdir / f"{name}.svg").write_text("\n".join(svg_open(500, 60, f"Confusion, {rung}, window {w}", f"not drawn: no confusion_kernel_{rung}.csv under {OUT_DIR / w} (the reading was not run at this window)") + ["</svg>"]))
            S.write_csv(fdir / f"{name}.csv", ["note"], [[f"no confusion for {rung} at {w}"]])
        else:
            (fdir / f"{name}.svg").write_text(svg)
            S.write_csv(fdir / f"{name}.csv", ["true", "predicted", "n_blocks"], [[r["true"], r["predicted"], r["n_blocks"]] for r in data])
        figs[name] = str(fdir / f"{name}.svg")
    S.write_params(tdir / "new_blocks_tables.csv", "plan11.new_blocks_tables.v1",
                   {"added": ADDED, "windows_present": present, "figure_window": w, "pools": pools, "null": NO_NULL, "comparators": COMPARATORS_NOTE,
                    "inputs_sha256": S.inputs_sha256([out / OUT_DIR / ww / n for ww in present for n in ("scores.csv", "margins.csv", "per_kernel.csv", "by_position.csv", "run_level.csv")
                                                      if (out / OUT_DIR / ww / n).is_file()], out)}, CITATION)
    print(f"[newblocks] tables: {', '.join(str(v['csv']) for v in written.values())}; summary {out / SUMMARY_CSV}; figures: {', '.join(figs)} under {fdir}")
    return {"tables": written, "figures": figs, "summary_csv": str(out / SUMMARY_CSV)}


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="plan11_encoding_ladder.new_blocks", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="move 17: the new-block test on the five readings at one window")
    r.add_argument("--out", required=True)
    r.add_argument("--window", default=PRIMARY_GRID, help=f"{PRIMARY_GRID} (the primary, default), own (each reading's selected window), or a grid id")
    r.add_argument("--seen", type=float, default=DEFAULT_SEEN, help="the share of each recording's windows the model sees, in time order (default 0.8, rounded to windows)")
    r.add_argument("--gap", type=int, default=DEFAULT_GAP, help="windows skipped between the seen windows and the first new block (default 1)")
    r.add_argument("--pool", default=DEFAULT_POOLS, help="pool sizes, consecutive new blocks voted together (default 1,2,3)")
    r.add_argument("--levels", default=",".join(LEVELS), help="archetype,kernel,run (default all three)")
    r.add_argument("--rungs", default=",".join(RUNGS), help="the readings (default all five)")
    r.add_argument("--n-jobs", type=int, default=1)
    r.add_argument("--n-estimators", type=int, default=M.N_ESTIMATORS)
    r.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP, help="resamples of the recordings for the 95%% intervals (default 1000)")
    r.add_argument("--seed-offset", type=int, default=0)
    r.add_argument("--dry-run", action="store_true", help="print the plan (the recordings, the windows, the seen windows, the new blocks); write nothing")
    t = sub.add_parser("tables", help="move 17: gates/added/new_blocks.csv, report/tables/new_blocks_*, report/figures/new_blocks_*")
    t.add_argument("--out", required=True)
    t.add_argument("--windows", default=f"{PRIMARY_GRID},{WINDOW_OWN}", help="the window folders to read (default W64_H32,own; a missing one is skipped)")
    a = ap.parse_args(argv)
    out = Path(os.path.expanduser(a.out))
    try:
        if a.cmd == "run":
            return run_new_blocks(out, window=a.window, seen=float(a.seen), gap=int(a.gap), pools=parse_pools(a.pool), levels=parse_list(a.levels, LEVELS, "levels"),
                                  rungs=parse_list(a.rungs, RUNGS, "rungs"), n_jobs=int(a.n_jobs), n_estimators=int(a.n_estimators), bootstrap=int(a.bootstrap),
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
