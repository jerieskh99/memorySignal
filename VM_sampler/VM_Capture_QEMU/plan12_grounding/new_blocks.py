#!/usr/bin/env python3
"""new_blocks.py -- the new-block test beside move 6 of plan12_grounding (2026-10-06; move 12 in the
console's Grounding paper tab, where the idle class check of idle_class.py is move 11).

  python3 -m plan12_grounding.new_blocks run --run <main run out> [--out <dir>] [--seen 0.8] [--gap 1]
        [--pools 1,2,3] [--cuts declared,measured] [--encoding-out <D2>] [--n-jobs N] [--n-estimators N]
        [--bootstrap 1000] [--force] [--dry-run]
  (run from VM_sampler/VM_Capture_QEMU/; the prompt is plan12_grounding/PROMPT_new_blocks_test.md)

THE QUESTION THE TEST ANSWERS (the author's words: "we have seen things, and now we got a new thing: will we
know to which it belongs? which archetype, which kernel, which workload, which seed")
A model sees the first 80% of every recording, in time order. Then new blocks of the same recordings arrive,
one window at a time. For each new block, and for 2 or 3 consecutive new blocks together, it must name the
archetype, the kernel and the run (the seed). Example: after seeing 80% of all 103 recordings, the rest of
gemm's run rep03 arrives block by block; the answers should be working-set, gemm, rep03.

THE DATA
The run's admissible recordings with a complete series (stats.load_runs; the real run: 103 = 95 kernel + 8 idle),
both cuts (params.json), the window of decision D3 (moves/06_classify/classify.json; the real run: W64_H32), the
feature sets E0, E1, E2 and E_new built exactly as move 6 builds them, idle admitted through
idle_class.build_encodings_13 (imported, never edited). Move 6's E0 gate and its check for idle are
idle_class's (e0_gate, idle_e0_check, decide_mode): every feature set when E0 can be built for every idle cell,
E_new only when it can be built for none, a stop with the question otherwise.

THE SPLIT, per recording and cut (flags --seen 0.8 --gap 1, in windows)
The recording's windows in time order (window start ascending); the first round(seen * n) of them are SEEN and
train the model; then `gap` windows are skipped, so that no new window shares a pair with any seen window
(asserted from the pair ranges: the first new window's first pair is after the last seen window's last pair,
pairs = start .. start + W - 1; a failure stops the run); the windows after the gap are the NEW BLOCKS,
numbered 1, 2, 3 ... by their position after the gap. Written per recording to cut<C>/split.csv.

THE LABELS (three levels)
  archetype  the encoding run's archetype_predicted (this run's cells.csv carries the same column, checked
             identical on the real run), no G-K0 relabel (move 6 relabels lexer; this test does not); "idle"
             for the idle cells (cells.csv calls their archetype "control", said in the record);
  kernel     the 12 kernels plus idle (13 classes);
  run        each recording its own class (the real run: 103; for a kernel the run is its seed).

THE MODEL: one forest per level, feature set and cut on all seen windows, with move 6's settings
  the forest       models.make_forest (300 trees in the real run, the record's value), seed nulls.SEED_FOREST +
                   move 6's seed offset, move 6's n_jobs unless overridden;
  dimension rule   applied as move 6 applies it (models.fit_predict_units auto_reduce): one training fold of
                   N recordings; a feature set with more features than N is reduced to N by training-fold
                   importance (models._reduce_fit, the same function); feature_count_used records it;
  B1-G3 quarantine applied as written with the block as the unit and the one training fold
                   (models.quarantine_l1: a one-feature threshold tree, max_leaf_nodes = the level's class
                   count, fitted on the seen windows; a feature whose tree reproduces the forest's block
                   predictions on all but at most models.B1G3_MAX_DISAGREE blocks is quarantined; the forest
                   is re-run without it and the re-run is the score, the full model kept beside it).
Where a move 6 rule does not apply at window level the outputs say so and nothing replaces it (summary.json
"move6_rules"): the splits (LORO, LOKO, within-trace) are replaced by the time split above; the label-shuffle
null and its B1-G1 verdict do not run (see NO_NULL); the unit "cell majority over all its windows" becomes
the block, and the pools of 2 and 3 consecutive blocks vote by the same rule (models.aggregate_units:
majority, ties by the mean predicted probability, then name order); LOKO's headline-class macro recall is
not applied (every class is seen in training); LOKO's per-fold majority baseline is not applicable, the
majority here is the most populous training class by recording count scored on the pools.

NO LABEL-SHUFFLE NULL. Labels shuffled consistently within a recording are learned from that recording's
own seen windows, so a shuffled model scores like the real one: move 6's within-trace null of 0.62 (cut 16,
null p95 0.621 for E0, E1, E2 in the real run) is this effect. Chance (1/K) and the majority class are the
baselines.

THE SCORES, per level, feature set, cut and pool size (1, 2, 3 consecutive new blocks of one recording,
every consecutive run of that length; majority vote, ties by the summed class probability)
  scores.csv            accuracy, macro recall (over the classes present among the pools), chance 1/K,
                        majority, the pool count, the class count, feature counts, the quarantine, the null column
  recall_per_kernel.csv at every level: the share of each kernel's pools (and idle's) labelled right at that level
  run_level.csv         at the run level, per kernel: the share of its pools given the right run, given another
                        run of the same kernel, given a run of another kernel
  by_position.csv       accuracy of the single blocks by their position after the gap (1st, 2nd, 3rd ...)
  margins.csv           E1-E0, E2-E1, E_new-E0: the accuracy difference on the same pools, with a 95% bootstrap
                        interval over recordings (--bootstrap 1000 resamples of the recordings, seed
                        SEED_BOOTSTRAP), pools gained and lost
  predictions_<level>_<E>.csv   every new block: cell, kernel, archetype, rep, position, window start, pair
                        range, truth, prediction, the predicted class's probability and the true class's
  pools_<level>_<E>.csv every pool: cell, size, first position, truth, prediction, vote fraction
  confusion_kernel_<E>.csv (and .svg for E2 and E_new, lexer and idle highlighted): the kernel level, single blocks
  accuracy_by_level.svg/.csv, accuracy_by_position.svg/.csv, recall_per_kernel.svg/.csv   the figures
  features.npz, excluded_cells.csv   as move 6 writes them; split.csv the split per recording
  summary.json          the cut's record: the counts, the scores, the margins, the rules applied and not
  <out>/new_blocks.json the move's record (command, params, the E0 check, both cuts); <out>/e0_idle_check.json
  <out>/record.json     one entry per run: command, inputs with sha256, params, the code fingerprint
                        (run_moves.code_fingerprint of this module's closure), status; a finished run is skipped
                        unless --force, or its inputs, parameters or code changed (idle_class's rule)

THE RULES THAT PROTECT THE LIVE RUN (as idle_class.py)
- ONE new file; nothing existing in plan12_grounding/ is edited. A move's code fingerprint covers only the
  modules it imports (run_moves.module_closure); nothing imports this module, so no finished move turns stale.
- The run folder is read only here. `--out` defaults to the sibling "<run>_newblocks"; a path inside the run is
  refused (exit 2). The test refuses to start (exit 3) while <run>/.driver.lock belongs to a live process;
  `--dry-run` skips that refusal, writes nothing and prints the plan (the recordings, the cuts, D2, D3, the E0
  check, and per cut the windows, the seen windows and the new blocks: the real run, about 309 at cut 192).
- One writer per output folder: <out>/.new_blocks.lock; run_moves.install_sigterm, so a Stop ends the run
  through its finally blocks.

THE REAL RUN (after move 9 ends; from the Grounding paper tab, move 12, or by hand)
  cd /Users/jeries/Desktop/projects/thesis/memorySignal/mem_sig/VM_sampler/VM_Capture_QEMU
  python3 -m plan12_grounding.new_blocks run \\
      --run /Users/jeries/Desktop/projects/thesis/memorySignal/spl_paper/results/run_20261005
  Defaults there: --out .../run_20261005_newblocks, the encoding run and the window of move 6's classify.json,
  cuts 16 and 192, 300 trees and 4 processes (move 6's recorded values), seen 0.8, gap 1, pools 1,2,3, 1000
  bootstrap resamples. Add --dry-run first to see the plan.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import argparse  # noqa: E402
import html  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

from plan12_grounding import __version__, toolkit_fingerprint  # noqa: E402
from plan12_grounding import classify as C  # noqa: E402  (move 6: the feature sets' column sets and names; reused, never copied)
from plan12_grounding import idle_class as I  # noqa: E402  (idle as a class: build_encodings_13, the E0 gate and check, the lock state; imported, never edited)
from plan12_grounding.run_moves import LOCK as DRIVER_LOCK, _pid_alive, code_fingerprint, install_sigterm, now_iso, read_json, sha256_file, write_json  # noqa: E402
from plan12_grounding.stats import cut_series, load_runs  # noqa: E402
from plan12_grounding.figures import PALETTE, svg_open, write_csv  # noqa: E402
from plan11_encoding_ladder import models as M  # noqa: E402
from plan11_encoding_ladder import series as S11  # noqa: E402
from plan11_encoding_ladder import verdicts as V  # noqa: E402
from plan11_encoding_ladder.nulls import SEED_FOREST  # noqa: E402

CITATION = ("plan12_grounding/PROMPT_new_blocks_test.md (2026-10-06, the author's design); plan12_grounding/classify.py move 6 (the feature sets, "
            "ENCODINGS and their column sets); plan12_grounding/idle_class.py (build_encodings_13, e0_gate, idle_e0_check, decide_mode, load_move6, "
            "select_runs); plan11_encoding_ladder models.py make_forest / _reduce_fit (the dimension rule) / aggregate_units (the vote) / quarantine_l1 "
            "(B1-G3) / B1G3_MAX_DISAGREE, nulls.py SEED_FOREST; no label-shuffle null (see NO_NULL)")
LEVELS = ("archetype", "kernel", "run")
LEVEL_TEXT = {"archetype": "the archetype (the encoding run's archetype_predicted, no G-K0 relabel; idle = 'idle')",
              "kernel": "the kernel (the 12 kernels plus idle)", "run": "the run (each recording its own class; for a kernel, its seed)"}
ENCODINGS = C.ENCODINGS
MARGIN_PAIRS = (("E1", "E0"), ("E2", "E1"), ("E_new", "E0"))
IDLE_LABEL = I.IDLE_LABEL
HIGHLIGHT = I.HIGHLIGHT
CLASS_ORDER = I.CLASS_ORDER
OUT_SUFFIX = "_newblocks"
LOCK_NAME = ".new_blocks.lock"
RECORD_NAME = "record.json"
RECORD_KEY = "new_blocks"
SEED_BOOTSTRAP = 20261006                                 # the bootstrap's own seed (the date of the design); the forest keeps move 6's
DEFAULT_SEEN, DEFAULT_GAP, DEFAULT_POOLS, DEFAULT_BOOTSTRAP = 0.8, 1, "1,2,3", 1000
EXIT_OK, EXIT_ERROR, EXIT_MISSING, EXIT_REFUSED = I.EXIT_OK, I.EXIT_ERROR, I.EXIT_MISSING, I.EXIT_REFUSED
NO_NULL = V.not_run("no label-shuffle null: labels shuffled consistently within a recording are learned from that recording's own seen windows, "
                    "so a shuffled model scores like the real one (move 6's within-trace null of 0.62 is this effect); the baselines are chance 1/K and the majority class")
MOVE6_RULES = {
    "splits": "not applicable: move 6's LORO, LOKO and within-trace folds (splits.folds_for) are replaced by the time split of this test (the first round(seen * n) "
              "windows of every recording seen, a gap of `gap` windows, the rest new blocks), which no plan11 split expresses",
    "null": NO_NULL,
    "unit": "replaced: move 6's unit is the cell (majority vote over all its windows, models.aggregate_units); here the unit is the block (one window) and the "
            "pools of 2 and 3 consecutive new blocks, which vote by the same rule (majority, ties by the mean predicted probability, then name order)",
    "headline_macro_recall": "not applied: LOKO's macro recall over the headline archetypes (models.headline_classes_of, archetypes with at least three kernels) exists "
                             "because a held-out kernel's archetype must be represented by other kernels; here every class is seen in training, so macro recall is over every "
                             "class present among the pools, at every level",
    "majority_baseline": "not applicable as written: LOKO's majority baseline is per fold (models.majority_baseline); here the majority is the most populous training class "
                         "by recording count (ties by name), scored on the pools",
    "dimension_rule": "applied as written (models.fit_predict_units auto_reduce, models._reduce_fit train_importance): one training fold of N recordings; a feature set with "
                      "more features than N is reduced to N features by training-fold importance; feature_count_used records it",
    "quarantine": "applied as written (models.quarantine_l1, max_disagree = models.B1G3_MAX_DISAGREE) with the block as the unit and the one training fold: a one-feature "
                  "threshold tree per feature (models.make_l1, max_leaf_nodes = the level's class count) fitted on the seen windows; a feature whose tree reproduces the "
                  "forest's block predictions on all but at most max_disagree blocks is quarantined, the forest re-run without it, the re-run is the score and the full "
                  "model is kept beside it",
    "gk0_relabel": "not applied: move 6 relabels the G-K0 kernels' archetype (series.gk0_relabel; the real run: lexer); the archetype level here is the encoding run's "
                   "archetype_predicted as written (the author's design)",
    "e0_gate": "applied as move 6 and idle_class apply it: the declared encoding-run files, the pair-rung exclusion, move 1's identity record; E0 for idle under "
               "idle_class.idle_e0_check (every feature set when E0 can be built for every idle cell, E_new only when for none)",
    "seeds": "the forest's seed is move 6's (nulls.SEED_FOREST + its seed offset), one fit per level, feature set and cut; the bootstrap has its own seed (SEED_BOOTSTRAP)",
}


class Stop(I.Stop):
    pass


def fmt(x) -> str:
    return I.fmt(x)


# ---------------------------------------------------------------------------------------------
# the output folder, the locks, the record
# ---------------------------------------------------------------------------------------------
def default_out(run: Path) -> Path:
    return run.parent / (run.name + OUT_SUFFIX)


def resolve_out(run: Path, out_arg: str | None) -> Path:
    out = Path(os.path.expanduser(out_arg)).resolve() if out_arg else default_out(run)
    if out == run or run in out.parents:
        raise Stop(EXIT_MISSING, f"refused: --out {out} lies inside the run folder {run}, which is read only here; the default is {default_out(run)}")
    return out


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
            raise Stop(EXIT_REFUSED, f"refused: another new_blocks run is writing {out} ({lock}: pid {pid}, started {old.get('started_at')}); one writer per output folder")
        print(f"[newblocks] stale lock from pid {pid} ({old.get('started_at')}) taken over", flush=True)
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
    return {"schema": "plan12.new_blocks.record.v1", "citation": CITATION, "package_version": __version__, "entries": []}


def save_record(out: Path, rec: dict) -> None:
    write_json(out / RECORD_NAME, rec)


def last_done(rec: dict, key: str) -> dict | None:
    for e in reversed(rec.get("entries") or []):
        if e.get("key") == key and e.get("status") == "done":
            return e
    return None


def code_identity() -> dict:
    fp = code_fingerprint("new_blocks")
    files = toolkit_fingerprint()["files"]
    return {"fingerprint": fp["sha256"], "modules": fp["modules"], "sha256": {f"{m}.py": files.get(f"{m}.py") for m in fp["modules"]}}


def inputs_identity(run: Path, enc_out: Path | None, declared: dict | None, runs: list[dict], cuts: dict) -> dict:
    """sha256 of every input the test reads: the run's files (cells.csv, params.json, move 1's records, move 6's record and
    its features at each cut, the series of every selected cell) and the encoding run's files (classify.declare_e0)."""
    rels = ["cells.csv", "params.json", "moves/01_extract/extract.json", "moves/01_extract/e0_identity.json", "moves/06_classify/classify.json"]
    rels += [f"moves/06_classify/cut{cut}/features.npz" for cut in cuts.values()]
    ex = read_json(run / "moves" / "01_extract" / "extract.json") if (run / "moves" / "01_extract" / "extract.json").is_file() else {}
    recs = ex.get("recordings") or {}
    files = {rel: (sha256_file(run / rel) if (run / rel).is_file() else "absent") for rel in rels}
    for r in runs:
        p = Path((recs.get(r["cell_id"]) or {}).get("series") or (run / "series" / f"{r['cell_id']}.npz"))
        key = str(p.relative_to(run)) if str(p).startswith(str(run)) else str(p)
        files[key] = sha256_file(p) if p.is_file() else "absent"
    return {"run": str(run), "run_files": files, "encoding_out": str(enc_out) if enc_out else None,
            "encoding_run_files": dict((declared or {}).get("files") or {})}


def parse_pools(spec: str) -> list[int]:
    try:
        pools = sorted({int(s) for s in str(spec).split(",") if s.strip()})
    except ValueError:
        raise Stop(EXIT_MISSING, f"--pools {spec!r}: a comma-separated list of positive integers (default {DEFAULT_POOLS})")
    if not pools or min(pools) < 1:
        raise Stop(EXIT_MISSING, f"--pools {spec!r}: a comma-separated list of positive integers (default {DEFAULT_POOLS})")
    return pools


# ---------------------------------------------------------------------------------------------
# the split: seen, the gap, the new blocks, per recording
# ---------------------------------------------------------------------------------------------
def split_rows(lab: dict, pair_start: np.ndarray, W: int, seen: float, gap: int) -> dict:
    """Per cell (first-seen order): the row indices of its windows in time order, the seen ones, the gap, the new blocks
    with their positions, and the pair-range check. Returns {cells, per_cell: {cell: {...}}, train_idx, test_idx, block_pos,
    block_id, n_new_total}; raises Stop when a new window shares a pair with a seen one."""
    cell_id = lab["cell_id"]
    cells = list(dict.fromkeys(cell_id.tolist()))
    per_cell, train, test, pos, bid, failures = {}, [], [], [], [], []
    for c in cells:
        idx = np.flatnonzero(cell_id == c)
        order = np.argsort(lab["win_start"][idx], kind="stable")
        idx = idx[order]
        n = int(idx.size)
        n_seen = int(np.floor(seen * n + 0.5))             # round(seen * n), half up
        first_new = n_seen + int(gap)
        seen_idx, gap_idx, new_idx = idx[:n_seen], idx[n_seen:first_new], idx[first_new:]
        last_seen_end = int(pair_start[seen_idx[-1]] + W - 1) if seen_idx.size else None
        first_new_start = int(pair_start[new_idx[0]]) if new_idx.size else None
        ok = True if (new_idx.size == 0 or seen_idx.size == 0) else (first_new_start > last_seen_end)
        if new_idx.size and seen_idx.size:
            # the assertion over every pair: no new window's range meets any seen window's range
            seen_lo, seen_hi = int(pair_start[seen_idx].min()), int(pair_start[seen_idx].max() + W - 1)
            new_lo = int(pair_start[new_idx].min())
            ok = ok and (new_lo > seen_hi) and (seen_lo <= seen_hi)
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


def plan_counts(runs: list[dict], cut: int, win: dict, seen: float, gap: int) -> dict:
    """The dry run's counts from the series alone (no feature built): windows, seen windows and new blocks per cut."""
    W, H = int(win["W"] or 0), int(win["H"] or 0)
    tot = {"recordings": 0, "windows": 0, "seen": 0, "gap": 0, "new_blocks": 0, "recordings_without_new_block": 0, "per_recording_new": {}}
    for r in runs:
        n_rows = int(cut_series(r, cut)["pair"].size)
        n = S11.n_windows(n_rows, W, H) if W else (1 if n_rows else 0)
        n_seen = int(np.floor(seen * n + 0.5))
        n_new = max(0, n - n_seen - int(gap))
        tot["recordings"] += 1; tot["windows"] += n; tot["seen"] += n_seen; tot["gap"] += min(int(gap), max(0, n - n_seen)); tot["new_blocks"] += n_new
        if n_new == 0:
            tot["recordings_without_new_block"] += 1
        tot["per_recording_new"][n_new] = tot["per_recording_new"].get(n_new, 0) + 1
    return tot


# ---------------------------------------------------------------------------------------------
# one level of one feature set: the forest on the seen windows, the blocks predicted, the quarantine
# ---------------------------------------------------------------------------------------------
def fit_level(X: np.ndarray, names: list[str], y: np.ndarray, sp: dict, *, seed: int, n_jobs: int, n_est: int) -> dict:
    """The forest of one level and feature set on the seen windows (the dimension rule as move 6's, the same seed), the new
    blocks predicted with their class probabilities, then B1-G3's quarantine at the block and the re-run without the
    quarantined features (the re-run is the score; the full model is kept)."""
    tr, te = sp["train_idx"], sp["test_idx"]
    cell_id = np.asarray(sp["cell_id_rows"]).astype(str)
    n_train_cells = len(set(cell_id[tr].tolist()))
    classes_train = sorted(set(y[tr].tolist()))

    def one(Xs, nm):
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

    full = one(X, names)
    # B1-G3 at the block: units = the seen cells (train rows) and the blocks (test rows); the one fold; the full model's block predictions
    unit_rows = cell_id.astype(object)                   # object dtype: a fixed-width string array would truncate the block ids
    unit_rows[te] = sp["block_id"].astype(object)
    unit_rows = unit_rows.astype(str)
    lab_u = {"cell_id": unit_rows, "n": int(unit_rows.size)}
    y_unit = {}
    for j in tr.tolist():
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
            with_q = {**one(X[:, keep], [names[j] for j in keep]), "quarantined_features": sorted(qnames)}
        else:
            with_q = {"status": V.not_run("every feature quarantined"), "quarantined_features": sorted(qnames)}
    use = with_q if (with_q and "pred" in with_q) else full
    return {"full": full, "with_quarantine": with_q, "use": use, "quarantine": quar,
            "score_source": (M.SCORE_SOURCE_QUARANTINE if (with_q and "pred" in with_q) else M.SCORE_SOURCE_FULL),
            "quarantined_features": sorted({q["feature"] for q in quar}), "classes_train": classes_train, "n_train_cells": n_train_cells,
            "n_train_windows": int(tr.size), "n_new_blocks": int(te.size),
            "status": ("ok" if "pred" in use else str(with_q.get("status")))}


# ---------------------------------------------------------------------------------------------
# the pools and the scores
# ---------------------------------------------------------------------------------------------
def pool_votes(sp: dict, pred: np.ndarray, proba: np.ndarray, classes: list[str], y_true_blocks: np.ndarray, pools: list[int]) -> dict:
    """{pool size: [ {cell_id, start_pos, size, y_true, y_pred, vote_fraction, block_rows} ]}: every consecutive run of `k` new blocks
    of one recording, voted by models.aggregate_units (majority; ties by the mean predicted probability, then name order)."""
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


def score_pools(pl: list[dict], classes_train: list[str], kernel_of_cell: dict, majority_class: str) -> dict:
    n = len(pl)
    if n == 0:
        return {"n_pools": 0, "accuracy": None, "macro_recall": None, "chance": (1.0 / len(classes_train)) if classes_train else None, "majority": None,
                "n_classes": len(classes_train), "recall_per_class": {}, "recall_per_kernel": {}}
    correct = np.array([p["y_true"] == p["y_pred"] for p in pl], dtype=bool)
    rpc = {}
    for cls in sorted({p["y_true"] for p in pl}):
        m = np.array([p["y_true"] == cls for p in pl], dtype=bool)
        rpc[cls] = float(correct[m].mean())
    rpk = {}
    for k in sorted({kernel_of_cell[p["cell_id"]] for p in pl}):
        m = np.array([kernel_of_cell[p["cell_id"]] == k for p in pl], dtype=bool)
        rpk[k] = float(correct[m].mean())
    return {"n_pools": n, "accuracy": float(correct.mean()), "macro_recall": float(np.mean(list(rpc.values()))), "chance": 1.0 / len(classes_train),
            "majority": float(np.mean([p["y_true"] == majority_class for p in pl])), "n_classes": len(classes_train),
            "recall_per_class": rpc, "recall_per_kernel": rpk}


def run_level_shares(pl: list[dict], kernel_of_cell: dict) -> dict:
    """At the run level, per kernel: the share of its pools given the right run, another run of the same kernel, a run of another kernel."""
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
    """The accuracy difference of two feature sets on the same pools, and its 95% bootstrap interval over recordings
    (the recordings resampled with replacement; each resample's difference is over its pools, paired)."""
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
    delta = float(cb.mean() - ca.mean())
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(present), size=(int(n_boot), len(present)))
    tot = nn[draws].sum(axis=1)
    d = (nb[draws].sum(axis=1) - na[draws].sum(axis=1)) / np.where(tot > 0, tot, 1.0)
    return {"delta": delta, "ci_lo": float(np.quantile(d, 0.025)), "ci_hi": float(np.quantile(d, 0.975)), "n_pools": int(cb.size), "n_recordings": len(present),
            "gained": int(np.sum((cb == 1) & (ca == 0))), "lost": int(np.sum((cb == 0) & (ca == 1))), "n_resamples": int(n_boot), "seed": int(seed), "status": "ok"}


# ---------------------------------------------------------------------------------------------
# figures (the engine's SVG style, figures.svg_open; every figure with its CSV)
# ---------------------------------------------------------------------------------------------
ENC_COLOUR = C.ENC_COLOUR


def accuracy_by_level_svg(scores: dict, pools: list[int], cut: int, e_sets: tuple) -> str:
    """Three panels (the levels); per feature set the single-block accuracy as a bar, the pools of 2 and 3 as ticks on the bar,
    chance 1/K as a grey line and the majority class as a dashed red line."""
    W_, H_ = 940, 320
    out = svg_open(W_, H_, f"New blocks named right, by level and feature set, cut of {cut} pairs",
                   "bars: accuracy on single new blocks; ticks on a bar: pools of 2 and 3 consecutive blocks; grey line: chance 1/K; dashed red: the majority class; " + NO_NULL[:60] + "...")
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
            bx = px + ei * bw + 8
            s1 = ((scores.get(level) or {}).get(enc) or {}).get(1) or {}
            acc = s1.get("accuracy")
            if enc not in e_sets or acc is None:
                out.append(f'<text x="{bx + (bw - 16) / 2:.1f}" y="{y0 + h / 2:.1f}" text-anchor="middle" fill="#999" font-size="9">not run</text>')
            else:
                out.append(f'<rect x="{bx:.1f}" y="{y0 + h - h * acc:.1f}" width="{bw - 16:.1f}" height="{h * acc:.1f}" fill="{ENC_COLOUR[enc]}" fill-opacity="0.85"/>')
                out.append(f'<text x="{bx + (bw - 16) / 2:.1f}" y="{y0 + h - h * acc - 3:.1f}" text-anchor="middle">{acc:.2f}</text>')
                for k in pools:
                    if k == 1:
                        continue
                    sk = ((scores.get(level) or {}).get(enc) or {}).get(k) or {}
                    if sk.get("accuracy") is not None:
                        yy = y0 + h - h * sk["accuracy"]
                        out.append(f'<line x1="{bx - 2:.1f}" y1="{yy:.1f}" x2="{bx + bw - 14:.1f}" y2="{yy:.1f}" stroke="#222" stroke-width="1.2"/>'
                                   f'<text x="{bx + bw - 12:.1f}" y="{yy + 3:.1f}" font-size="8">{k}</text>')
            if s1.get("chance") is not None:
                cy = y0 + h - h * float(s1["chance"])
                out.append(f'<line x1="{bx - 3:.1f}" y1="{cy:.1f}" x2="{bx + bw - 13:.1f}" y2="{cy:.1f}" stroke="#888"/>')
            if s1.get("majority") is not None:
                my = y0 + h - h * float(s1["majority"])
                out.append(f'<line x1="{bx - 3:.1f}" y1="{my:.1f}" x2="{bx + bw - 13:.1f}" y2="{my:.1f}" stroke="#b03a2e" stroke-dasharray="3,2"/>')
            out.append(f'<text x="{bx + (bw - 16) / 2:.1f}" y="{y0 + h + 12}" text-anchor="middle">{enc}</text>')
        any_s = next((v for e in ENCODINGS for v in [((scores.get(level) or {}).get(e) or {}).get(1)] if v), {})
        out.append(f'<text x="{px}" y="{y0 + h + 26}" fill="#555" font-size="9">{any_s.get("n_classes", "?")} classes, {any_s.get("n_pools", "?")} new blocks</text>')
    out.append(f'<text x="{x0}" y="{H_ - 10}" fill="#555">{"; ".join(f"{e}: {C.ENCODING_TEXT[e]}" for e in ENCODINGS)}</text></svg>')
    return "\n".join(out)


def accuracy_by_position_svg(bypos: dict, cut: int, e_sets: tuple) -> str:
    """Three panels (the levels): the single blocks' accuracy against their position after the gap, one line per feature set."""
    positions = sorted({p for lv in bypos.values() for e in lv.values() for p in e})
    W_, H_ = 940, 300
    out = svg_open(W_, H_, f"New blocks named right, by their position after the gap, cut of {cut} pairs",
                   "lines: accuracy of the single new blocks at position 1, 2, 3 ... after the gap, one line per feature set; grey: chance 1/K")
    x0, y0, w, h = 60, 60, 270, 180
    P = max(positions) if positions else 1
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
        for enc in ENCODINGS:
            d = (bypos.get(level) or {}).get(enc) or {}
            if enc not in e_sets or not d:
                continue
            pts = " ".join(f"{X(p):.1f},{y0 + h - h * d[p]['accuracy']:.1f}" for p in sorted(d))
            out.append(f'<polyline points="{pts}" fill="none" stroke="{ENC_COLOUR[enc]}" stroke-width="2"/>')
            for p in sorted(d):
                out.append(f'<circle cx="{X(p):.1f}" cy="{y0 + h - h * d[p]["accuracy"]:.1f}" r="2.5" fill="{ENC_COLOUR[enc]}"/>')
            chance = d[sorted(d)[0]].get("chance", chance)
        if chance is not None:
            cy = y0 + h - h * float(chance)
            out.append(f'<line x1="{px}" y1="{cy:.1f}" x2="{px + w}" y2="{cy:.1f}" stroke="#888"/>')
        out.append(f'<text x="{px + w / 2:.1f}" y="{y0 + h + 24}" text-anchor="middle" fill="#555" font-size="9">position after the gap</text>')
    legend = "; ".join('<tspan fill="%s">%s</tspan>' % (ENC_COLOUR[e], e) for e in ENCODINGS)
    out.append(f'<text x="{x0}" y="{H_ - 10}" fill="#555">{legend}</text></svg>')
    return "\n".join(out)


def recall_per_kernel_svg(scores: dict, labels: list[str], cut: int, e_sets: tuple) -> str:
    """Three rows (the levels): per kernel (and idle), the share of its single new blocks named right, grouped bars per feature set."""
    n = len(labels)
    gw = 4 * 7 + 6
    W_ = 70 + n * gw + 30
    ph, gap = 120, 50
    H_ = 50 + len(LEVELS) * (ph + gap)
    out = svg_open(W_, H_, f"Each kernel's new blocks named right, per level and feature set, cut of {cut} pairs",
                   "bars per kernel: E0, E1, E2, E_new (left to right), single new blocks; the lexer and idle groups are highlighted")
    for li, level in enumerate(LEVELS):
        px, py = 60, 50 + li * (ph + gap)
        out.append(f'<text x="{px}" y="{py - 6}" font-weight="bold">{level}</text>')
        out.append(f'<rect x="{px}" y="{py}" width="{n * gw}" height="{ph}" fill="none" stroke="#ccc"/>')
        for t in (0.0, 0.5, 1.0):
            yy = py + ph - ph * t
            out.append(f'<line x1="{px}" y1="{yy:.1f}" x2="{px + n * gw}" y2="{yy:.1f}" stroke="#eee"/><text x="{px - 4}" y="{yy + 3:.1f}" text-anchor="end">{t:.1f}</text>')
        for ki, k in enumerate(labels):
            gx = px + ki * gw
            if k in HIGHLIGHT:
                out.append(f'<rect x="{gx}" y="{py}" width="{gw}" height="{ph}" fill="#fff3c4" stroke="none"/>')
            for ei, enc in enumerate(ENCODINGS):
                s1 = ((scores.get(level) or {}).get(enc) or {}).get(1) or {}
                r = (s1.get("recall_per_kernel") or {}).get(k)
                if enc in e_sets and r is not None:
                    out.append(f'<rect x="{gx + 3 + ei * 7}" y="{py + ph - ph * r:.1f}" width="6" height="{ph * r:.1f}" fill="{ENC_COLOUR[enc]}" fill-opacity="0.9"/>')
            bold = ' font-weight="bold"' if k in HIGHLIGHT else ""
            out.append(f'<text x="{gx + gw / 2:.1f}" y="{py + ph + 10}" text-anchor="end" font-size="8"{bold} transform="rotate(-45 {gx + gw / 2:.1f},{py + ph + 10})">{html.escape(k)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def confusion_svg(labels: list[str], Cm: np.ndarray, enc: str, cut: int, sc: dict) -> str:
    """The kernel level's confusion on single new blocks (rows true, columns predicted), the lexer and idle rows highlighted."""
    n = len(labels)
    cell, left, top = 22, 124, 100
    W_ = left + n * cell + 40
    H_ = top + n * cell + 70
    out = svg_open(W_, H_, f"Confusion on single new blocks: the kernel level, {enc}, {n} classes, cut of {cut} pairs",
                   "rows: the true kernel; columns: the predicted kernel; the number of new blocks; shade: the share of the row; the lexer and idle rows are highlighted")
    px, py = left, top
    for i, l in enumerate(labels):
        if l in HIGHLIGHT:
            out.append(f'<rect x="{px - 116}" y="{py + i * cell}" width="{116 + n * cell}" height="{cell}" fill="#fff3c4" stroke="#d9822b" stroke-width="0.8"/>')
    for j, l in enumerate(labels):
        bold = ' font-weight="bold"' if l in HIGHLIGHT else ""
        out.append(f'<text x="{px + j * cell + cell / 2:.1f}" y="{py - 5}" text-anchor="end" font-size="9"{bold} transform="rotate(-60 {px + j * cell + cell / 2:.1f},{py - 5})">{html.escape(l)}</text>')
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
    out.append(f'<text x="{px - 116}" y="{yb}" fill="#555" font-size="9">accuracy {fmt(sc.get("accuracy"))}, macro recall {fmt(sc.get("macro_recall"))}, '
               f'{sc.get("n_pools")} new blocks, {n} classes; chance {fmt(sc.get("chance"))}, majority {fmt(sc.get("majority"))}</text>')
    out.append(f'<text x="{px - 116}" y="{yb + 14}" fill="#555" font-size="9">{html.escape(NO_NULL[:150])}</text>')
    out.append(f'<text x="{px - 116}" y="{yb + 28}" fill="#555" font-size="9">rows and columns in the class order of cells.csv (schema.KERNEL_NAMES), idle last; highlighted: {", ".join(HIGHLIGHT)}</text>')
    out.append("</svg>")
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------
# one cut
# ---------------------------------------------------------------------------------------------
def one_cut(out: Path, cut_name: str, cut: int, runs: list[dict], enc_out: Path | None, enc_rows, win: dict, pair_excluded: set, mode: str,
            e_sets: tuple, reason: str | None, run: Path, *, seen: float, gap: int, pools: list[int], n_boot: int, n_jobs: int, n_est: int, seed: int) -> dict:
    d = out / f"cut{cut}"
    d.mkdir(parents=True, exist_ok=True)
    t_cut = time.time()
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
        raise Stop(EXIT_ERROR, "stopped: the D3 window is the whole cell (one window per recording), so no recording can be split into seen and new blocks")
    W = int(win["W"])
    pair_start = np.array(enc["meta"]["pair_start"], dtype=np.int64)
    kernel_rows = lab["kernel"].astype(str)
    arche_rows = np.where(kernel_rows == IDLE_LABEL, IDLE_LABEL, lab["archetype"].astype(str)).astype(str)
    run_rows = lab["cell_id"].astype(str)
    labels_of = {"archetype": arche_rows, "kernel": kernel_rows, "run": run_rows}
    sp = split_rows(lab, pair_start, W, seen, gap)
    sp["cell_id_rows"] = run_rows
    cells = sp["cells"]
    kernel_of_cell = {c: str(kernel_rows[np.flatnonzero(run_rows == c)[0]]) for c in cells}
    arche_of_cell = {c: str(arche_rows[np.flatnonzero(run_rows == c)[0]]) for c in cells}
    rep_of_cell = {c: int(lab["rep"][np.flatnonzero(run_rows == c)[0]]) for c in cells}
    kernel_cells = [c for c in cells if kernel_of_cell[c] != IDLE_LABEL]
    idle_cells = [c for c in cells if kernel_of_cell[c] == IDLE_LABEL]
    labels_kernel = I.class_order(set(kernel_of_cell.values()))
    np.savez_compressed(d / "features.npz", X=X, feature_names=np.array(names, dtype=str), cell_id=lab["cell_id"], kernel=lab["kernel"],
                        archetype=lab["archetype"], rep=lab["rep"], win_start=lab["win_start"], pair_start=pair_start, campaign=lab["campaign"],
                        cols_json=np.array(json.dumps(enc["cols"])), W=np.int64(win["W"] or 0), H=np.int64(win["H"] or 0), grid_id=np.array(win["grid_id"]), cut=np.int64(cut),
                        train_idx=sp["train_idx"], test_idx=sp["test_idx"], block_id=sp["block_id"], block_pos=sp["block_pos"])
    write_csv(d / "excluded_cells.csv", ["cell_id", "reason"], [[e["cell_id"], e["reason"]] for e in enc["excluded"]])
    write_csv(d / "split.csv", ["cell_id", "kernel", "n_windows", "n_seen", "n_gap", "n_new", "first_seen_pair", "last_seen_pair", "first_new_pair", "last_new_pair", "gap_ok", "note"],
              [[c, kernel_of_cell[c], *[sp["per_cell"][c][k] for k in ("n_windows", "n_seen", "n_gap", "n_new", "first_seen_pair", "last_seen_pair", "first_new_pair", "last_new_pair", "gap_ok", "note")]] for c in cells])
    print(f"[newblocks] cut {cut}: {len(cells)} recordings ({len(kernel_cells)} kernel + {len(idle_cells)} idle), {int(lab['n'])} windows of W {W}; seen {int(sp['train_idx'].size)}, "
          f"new blocks {sp['n_new_total']} ({min(v['n_new'] for v in sp['per_cell'].values())} to {max(v['n_new'] for v in sp['per_cell'].values())} per recording); the gap assertion held", flush=True)
    # the training majority class per level (the most populous class by recording count, ties by name)
    majority_class = {}
    for level in LEVELS:
        counts: dict = {}
        for c in cells:
            lv = {"archetype": arche_of_cell, "kernel": kernel_of_cell, "run": {c: c}}[level][c]
            counts[lv] = counts.get(lv, 0) + 1
        majority_class[level] = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    te = sp["test_idx"]
    results: dict = {lv: {} for lv in LEVELS}
    pools_by: dict = {lv: {} for lv in LEVELS}
    for level in LEVELS:
        y = labels_of[level]
        for e in ENCODINGS:
            if e not in e_sets:
                results[level][e] = {"status": reason, "scores": {k: {"accuracy": None, "macro_recall": None, "n_pools": 0} for k in pools}}
                print(f"[newblocks] cut {cut} {level:<9} {e:<5} {str(reason)[:80]}", flush=True)
                continue
            cols = enc["cols"][e]
            t0 = time.time()
            fit = fit_level(X[:, cols], [names[j] for j in cols], y, sp, seed=seed, n_jobs=n_jobs, n_est=n_est)
            use = fit["use"]
            if "pred" not in use:
                results[level][e] = {"status": fit["status"], "scores": {k: {"accuracy": None, "macro_recall": None, "n_pools": 0} for k in pools},
                                     "quarantine": fit["quarantine"], "quarantined_features": fit["quarantined_features"], "score_source": fit["score_source"]}
                continue
            pv = pool_votes(sp, use["pred"], use["proba"], use["classes"], y[te], pools)
            pools_by[level][e] = pv
            sc = {k: score_pools(pv[k], fit["classes_train"], kernel_of_cell, majority_class[level]) for k in pools}
            res = {"status": "ok", "scores": sc, "feature_count": use["feature_count"], "feature_count_used": use["feature_count_used"], "reduced": use["reduced"],
                   "score_source": fit["score_source"], "quarantine": fit["quarantine"], "quarantined_features": fit["quarantined_features"],
                   "full_model": ({"accuracy": score_pools(pool_votes(sp, fit["full"]["pred"], fit["full"]["proba"], fit["full"]["classes"], y[te], [1])[1],
                                                           fit["classes_train"], kernel_of_cell, majority_class[level])["accuracy"]} if fit["with_quarantine"] else None),
                   "n_train_windows": fit["n_train_windows"], "n_train_cells": fit["n_train_cells"], "n_new_blocks": fit["n_new_blocks"], "classes_train": fit["classes_train"],
                   "by_position": by_position(pv[1]) if 1 in pv else {}, "run_level": run_level_shares(pv[1], kernel_of_cell) if (level == "run" and 1 in pv) else None,
                   "elapsed_s": round(time.time() - t0, 1)}
            # the blocks' predictions with the class probabilities
            ci = {c: i for i, c in enumerate(use["classes"])}
            res["blocks"] = [{"block_id": b, "cell_id": b.split("|b")[0], "position": int(sp["block_pos"][i]), "row": int(te[i]), "win_start": int(lab["win_start"][te[i]]),
                              "pair_start": int(pair_start[te[i]]), "pair_end": int(pair_start[te[i]] + W - 1), "y_true": str(y[te[i]]), "y_pred": str(use["pred"][i]),
                              "p_pred": float(use["proba"][i].max()), "p_true": float(use["proba"][i][ci[str(y[te[i]])]]) if str(y[te[i]]) in ci else None}
                             for i, b in enumerate(sp["block_id"].tolist())]
            results[level][e] = res
            s1 = sc[1] if 1 in sc else next(iter(sc.values()))
            print(f"[newblocks] cut {cut} {level:<9} {e:<5} acc {fmt(s1['accuracy'])} macro {fmt(s1['macro_recall'])} chance {fmt(s1['chance'])} majority {fmt(s1['majority'])}"
                  + "".join(f" | pool {k}: {fmt(sc[k]['accuracy'])}" for k in pools if k != 1)
                  + f" | features {use['feature_count_used']}/{use['feature_count']}" + (f" quarantined {len(fit['quarantined_features'])}" if fit["quarantined_features"] else "")
                  + f" ({res['elapsed_s']} s)", flush=True)
    # tables
    score_rows = []
    for level in LEVELS:
        for e in ENCODINGS:
            r = results[level][e]
            for k in pools:
                s = (r.get("scores") or {}).get(k) or {}
                score_rows.append([level, e, k, r.get("status"), s.get("n_pools"), s.get("n_classes"), s.get("accuracy"), s.get("macro_recall"), s.get("chance"), s.get("majority"),
                                   r.get("feature_count"), r.get("feature_count_used"), r.get("score_source"), ";".join(r.get("quarantined_features") or []),
                                   r.get("n_train_windows"), r.get("n_train_cells"), NO_NULL])
    write_csv(d / "scores.csv", ["level", "feature_set", "pool_size", "status", "n_pools", "n_classes", "accuracy", "macro_recall", "chance", "majority",
                                 "feature_count", "feature_count_used", "score_source", "quarantined_features", "n_train_windows", "n_train_recordings", "null"], score_rows)
    rk_rows = []
    for level in LEVELS:
        for e in ENCODINGS:
            for k in pools:
                s = (results[level][e].get("scores") or {}).get(k) or {}
                for kern in labels_kernel:
                    if kern in (s.get("recall_per_kernel") or {}):
                        rk_rows.append([level, e, k, kern, s["recall_per_kernel"][kern], sum(1 for p in pools_by[level].get(e, {}).get(k, []) if kernel_of_cell[p["cell_id"]] == kern)])
    write_csv(d / "recall_per_kernel.csv", ["level", "feature_set", "pool_size", "kernel", "recall", "n_pools"], rk_rows)
    rl_rows = []
    for e in ENCODINGS:
        r = results["run"][e]
        if not r.get("run_level"):
            continue
        for k in pools:
            shares = run_level_shares(pools_by["run"][e][k], kernel_of_cell)
            for kern in labels_kernel:
                if kern in shares:
                    v = shares[kern]
                    rl_rows.append([e, k, kern, v["n_pools"], v["right_run"], v["same_kernel_other_run"], v["other_kernel"]])
    write_csv(d / "run_level.csv", ["feature_set", "pool_size", "kernel", "n_pools", "right_run", "same_kernel_other_run", "other_kernel"], rl_rows)
    bp_rows, bypos = [], {lv: {} for lv in LEVELS}
    for level in LEVELS:
        for e in ENCODINGS:
            r = results[level][e]
            if not r.get("by_position"):
                continue
            chance = r["scores"][1]["chance"] if 1 in r["scores"] else None
            bypos[level][e] = {p: {**v, "chance": chance} for p, v in r["by_position"].items()}
            for p, v in sorted(r["by_position"].items()):
                bp_rows.append([level, e, p, v["n_blocks"], v["accuracy"], chance])
    write_csv(d / "by_position.csv", ["level", "feature_set", "position", "n_blocks", "accuracy", "chance"], bp_rows)
    margin_rows, margins = [], {lv: {} for lv in LEVELS}
    for level in LEVELS:
        for k in pools:
            for b, a in MARGIN_PAIRS:
                pb, pa = pools_by[level].get(b, {}).get(k), pools_by[level].get(a, {}).get(k)
                if pb is None or pa is None:
                    margin_rows.append([level, k, f"{b}-{a}", None, None, None, None, None, None, None,
                                        "not run: " + str(results[level][a].get("status") if pa is None else results[level][b].get("status"))])
                    continue
                bm = bootstrap_margin(pb, pa, cells, n_boot, SEED_BOOTSTRAP)
                margins[level][f"{k}:{b}-{a}"] = bm
                margin_rows.append([level, k, f"{b}-{a}", bm["delta"], bm["ci_lo"], bm["ci_hi"], bm["n_pools"], bm["n_recordings"], bm["gained"], bm["lost"], bm["status"]])
    write_csv(d / "margins.csv", ["level", "pool_size", "comparison", "delta_accuracy", "ci95_lo", "ci95_hi", "n_pools", "n_recordings", "n_pools_gained", "n_pools_lost", "status"], margin_rows)
    for level in LEVELS:
        for e in ENCODINGS:
            r = results[level][e]
            if not r.get("blocks"):
                continue
            write_csv(d / f"predictions_{level}_{e}.csv", ["block_id", "cell_id", "kernel", "archetype", "rep", "position", "win_start", "pair_start", "pair_end", "y_true", "y_pred", "p_pred", "p_true"],
                      [[b["block_id"], b["cell_id"], kernel_of_cell[b["cell_id"]], arche_of_cell[b["cell_id"]], rep_of_cell[b["cell_id"]], b["position"], b["win_start"], b["pair_start"],
                        b["pair_end"], b["y_true"], b["y_pred"], b["p_pred"], b["p_true"]] for b in r["blocks"]])
            write_csv(d / f"pools_{level}_{e}.csv", ["cell_id", "kernel", "pool_size", "start_position", "y_true", "y_pred", "vote_fraction"],
                      [[p["cell_id"], kernel_of_cell[p["cell_id"]], k, p["start_pos"], p["y_true"], p["y_pred"], p["vote_fraction"]] for k in pools for p in pools_by[level][e][k]])
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
            (d / f"confusion_kernel_{e}.svg").write_text(confusion_svg(labels_kernel, Cm, e, cut, r["scores"][1]))
    scores_for_fig = {lv: {e: (results[lv][e].get("scores") or {}) for e in ENCODINGS} for lv in LEVELS}
    (d / "accuracy_by_level.svg").write_text(accuracy_by_level_svg(scores_for_fig, pools, cut, e_sets))
    write_csv(d / "accuracy_by_level.csv", ["level", "feature_set", "pool_size", "accuracy", "macro_recall", "chance", "majority", "n_pools"],
              [[lv, e, k, (scores_for_fig[lv][e].get(k) or {}).get("accuracy"), (scores_for_fig[lv][e].get(k) or {}).get("macro_recall"), (scores_for_fig[lv][e].get(k) or {}).get("chance"),
                (scores_for_fig[lv][e].get(k) or {}).get("majority"), (scores_for_fig[lv][e].get(k) or {}).get("n_pools")] for lv in LEVELS for e in ENCODINGS for k in pools])
    (d / "accuracy_by_position.svg").write_text(accuracy_by_position_svg(bypos, cut, e_sets))
    write_csv(d / "accuracy_by_position.csv", ["level", "feature_set", "position", "n_blocks", "accuracy", "chance"], bp_rows)
    (d / "recall_per_kernel.svg").write_text(recall_per_kernel_svg(scores_for_fig, labels_kernel, cut, e_sets))
    write_csv(d / "recall_per_kernel_figure.csv", ["level", "feature_set", "kernel", "recall"],
              [[lv, e, kern, ((scores_for_fig[lv][e].get(1) or {}).get("recall_per_kernel") or {}).get(kern)] for lv in LEVELS for e in ENCODINGS for kern in labels_kernel])
    # the focus: lexer and idle at the kernel level, what each is mistaken for
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
    summary = {"schema": "plan12.new_blocks_cut.v1", "citation": CITATION, "cut": int(cut), "cut_name": cut_name, "mode": mode, "feature_sets_run": list(e_sets),
               "reason_others_not_run": reason, "null": NO_NULL, "move6_rules": MOVE6_RULES, "levels": LEVEL_TEXT, "window": win,
               "split": {"seen": seen, "gap": gap, "rule": "windows in time order; the first round(seen * n) seen; `gap` windows skipped; the rest new blocks numbered by position",
                         "gap_assertion": "held for every recording: the first new window's first pair is after the last seen window's last pair (pairs = start .. start + W - 1)",
                         "n_recordings": len(cells), "n_kernel_recordings": len(kernel_cells), "n_idle_recordings": len(idle_cells), "n_windows": int(lab["n"]),
                         "n_seen_windows": int(sp["train_idx"].size), "n_new_blocks": sp["n_new_total"],
                         "new_blocks_per_recording": {str(k): sum(1 for v in sp["per_cell"].values() if v["n_new"] == k) for k in sorted({v["n_new"] for v in sp["per_cell"].values()})},
                         "recordings_without_new_block": [c for c in cells if sp["per_cell"][c]["n_new"] == 0]},
               "pools": pools, "pool_rule": "every consecutive run of k new blocks of one recording; majority vote, ties by the mean (= summed) predicted probability, then name order (models.aggregate_units)",
               "majority_class": majority_class, "classes": {"archetype": sorted(set(arche_of_cell.values())), "kernel": labels_kernel, "n_run": len(cells)},
               "idle_archetype_in_cells_csv": sorted({str(lab["archetype"][i]) for i in np.flatnonzero(kernel_rows == IDLE_LABEL)}),
               "n_excluded": len(enc["excluded"]), "excluded": enc["excluded"], "e0_available": enc["e0_available"], "n_e0_windows": enc["n_e0_windows"],
               "feature_counts": {e: len(enc["cols"][e]) for e in ENCODINGS}, "n_estimators": n_est, "n_jobs": n_jobs, "seed_forest": seed, "seed_bootstrap": SEED_BOOTSTRAP, "n_bootstrap": n_boot,
               "scores": {lv: {e: {"status": results[lv][e].get("status"), "feature_count": results[lv][e].get("feature_count"), "feature_count_used": results[lv][e].get("feature_count_used"),
                                   "score_source": results[lv][e].get("score_source"), "quarantined_features": results[lv][e].get("quarantined_features"),
                                   "full_model": results[lv][e].get("full_model"), "elapsed_s": results[lv][e].get("elapsed_s"),
                                   "pools": {str(k): {kk: vv for kk, vv in ((results[lv][e].get("scores") or {}).get(k) or {}).items() if kk not in ("recall_per_class",)} for k in pools}}
                               for e in ENCODINGS} for lv in LEVELS},
               "run_level": {e: results["run"][e].get("run_level") for e in ENCODINGS}, "by_position": {lv: {e: results[lv][e].get("by_position") for e in ENCODINGS} for lv in LEVELS},
               "margins": margins, "lexer": focus["lexer"], "idle": focus[IDLE_LABEL], "elapsed_s": round(time.time() - t_cut, 1),
               "files": sorted(p.name for p in d.iterdir())}
    write_json(d / "summary.json", summary)
    return summary


# ---------------------------------------------------------------------------------------------
# the move
# ---------------------------------------------------------------------------------------------
def run_move(o: argparse.Namespace) -> int:
    run = Path(os.path.expanduser(o.run)).resolve()
    if not (run / "cells.csv").is_file():
        raise Stop(EXIT_MISSING, f"missing input: {run / 'cells.csv'} (--run must name a plan12_grounding output folder with moves 0, 1 and 6 done)")
    out = resolve_out(run, o.out)
    seen, gap, pools, n_boot = float(o.seen), int(o.gap), parse_pools(o.pools), int(o.bootstrap)
    if not (0.0 < seen < 1.0) or gap < 0 or n_boot < 1:
        raise Stop(EXIT_MISSING, f"--seen must lie in (0, 1) (given {seen}), --gap be 0 or more (given {gap}), --bootstrap 1 or more (given {n_boot})")
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
    enc_rows, declared, relabel, pair_excluded, e0_status, identity = I.e0_gate(run, enc_out, cell_ids)
    check = I.idle_e0_check(idle_runs, enc_out, enc_rows, pair_excluded, e0_status, win, cuts)
    mode, e_sets, reason = I.decide_mode(idle_runs, e0_status, check)
    n_kernels = len({r["kernel"] for r in kernel_runs})
    counts = {name: plan_counts(runs, cut, win, seen, gap) for name, cut in cuts.items()}
    # the plan, in plain words
    print(f"[newblocks] the new-block test: seen {seen:g} of each recording's windows, a gap of {gap} window(s), then the new blocks; pools {pools}; "
          f"three levels (archetype, kernel, run); feature sets {', '.join(e_sets)}; {NO_NULL[:60]}...")
    print(f"[newblocks] run {run} (read only); out {out}{' (exists)' if out.exists() else ' (to be created)'}")
    print(f"[newblocks] driver lock: " + (f"held by pid {lock['pid']} ({'alive' if lock['alive'] else 'dead'}), started {lock['started_at']}: {lock['argv_tail']}" if lock["exists"] else "absent")
          + ("; the refusal is skipped by --dry-run" if (lock["alive"] and o.dry_run) else ""))
    print(f"[newblocks] recordings found: {len(kernel_runs) + len(idle_runs)} = {len(kernel_runs)} kernel ({n_kernels} kernels) + {len(idle_runs)} idle "
          f"(the real run expects 103 = 95 kernel + {I.N_IDLE_EXPECTED} idle); classes: archetype {len({r['archetype'] for r in kernel_runs}) + (1 if idle_runs else 0)}, "
          f"kernel {n_kernels + (1 if idle_runs else 0)}, run {len(kernel_runs) + len(idle_runs)}")
    print(f"[newblocks] cuts: {', '.join(f'{k} = {v} pairs' for k, v in cuts.items())} (params.json)")
    print(f"[newblocks] D2 encoding run: {enc_out}{' (overrides ' + str(m6['encoding_out_recorded']) + ')' if m6['encoding_out_overridden'] else ' (move 6 classify.json)'}; "
          f"exists {bool(enc_out and enc_out.is_dir())}; G-K0 relabel NOT applied (move 6 relabels {sorted(relabel) or 'none'}); pair-rung excluded {len(pair_excluded)}")
    print(f"[newblocks] D3 window: {win['grid_id']} (W {win['W']}, H {win['H']}; {win['source']}); forest: {n_est} trees, {n_jobs} processes, seed {seed} (move 6's values unless overridden); "
          f"bootstrap {n_boot} resamples, seed {SEED_BOOTSTRAP}")
    print(f"[newblocks] move 6's E0 gate: {'E0 usable (identity check passed on ' + str(identity.get('cell_id')) + ')' if e0_status is None else e0_status}")
    print(f"[newblocks] E0 for idle: {('can be built for all ' + str(check['n_idle']) + ' idle cells') if check['can_build'] else str(check['reason'])}; mode: {mode}")
    for name, cut in cuts.items():
        c = counts[name]
        print(f"[newblocks] cut {cut} ({name}): {c['recordings']} recordings, {c['windows']} windows, {c['seen']} seen, {c['new_blocks']} new blocks "
              f"({', '.join(f'{n} recordings with {k}' for k, n in sorted(c['per_recording_new'].items()))})"
              + (f"; {c['recordings_without_new_block']} without a new block" if c["recordings_without_new_block"] else ""))
    if o.dry_run:
        print(f"[newblocks] dry run: would write {out}/cut<C>/ (split.csv, scores.csv, recall_per_kernel.csv, run_level.csv, by_position.csv, margins.csv, predictions_<level>_<E>.csv, "
              f"pools_<level>_<E>.csv, confusion_kernel_<E>.csv/.svg, accuracy_by_level.svg, accuracy_by_position.svg, recall_per_kernel.svg, summary.json), "
              f"{out / 'new_blocks.json'}, {out / 'e0_idle_check.json'} and {out / RECORD_NAME}; nothing written")
        return EXIT_OK
    out_lock = acquire_out_lock(out)
    code = code_identity()
    entry = {"key": RECORD_KEY, "command": sys.argv, "cwd": str(_HERE.parent), "started_at": now_iso(), "status": "running", "exit_code": None, "finished_at": None,
             "params": {"run": str(run), "out": str(out), "cuts": cuts, "encoding_out": str(enc_out) if enc_out else None, "window": win, "seen": seen, "gap": gap, "pools": pools,
                        "bootstrap": n_boot, "seed_bootstrap": SEED_BOOTSTRAP, "n_estimators": n_est, "n_jobs": n_jobs, "seed_forest": seed, "seed_offset": m6["seed_offset"],
                        "mode": mode, "feature_sets": list(e_sets), "reason_others_not_run": reason, "n_kernel_cells": len(kernel_runs), "n_idle_cells": len(idle_runs),
                        "force": bool(o.force)},
             "inputs_sha256": inputs_identity(run, enc_out, declared, kernel_runs + idle_runs, cuts),
             "code": {"new_blocks.py": code["sha256"].get("new_blocks.py"), "fingerprint": code["fingerprint"], "modules": code["modules"], "sha256": code["sha256"],
                      "toolkit_fingerprint": toolkit_fingerprint()["sha256"]},
             "outputs": []}
    sig = {k: v for k, v in entry["params"].items() if k not in ("n_jobs", "force")}
    rec = load_record(out)
    prev = last_done(rec, RECORD_KEY)
    outputs_exist = (out / "new_blocks.json").is_file() and all((out / f"cut{c}" / "summary.json").is_file() for c in cuts.values())
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
                print(f"[newblocks] skipped: {out} stands as made on {prev.get('finished_at')} (inputs, parameters and code unchanged; --force re-runs)")
                return EXIT_OK
            print(f"[newblocks] re-running: {why} since {prev.get('finished_at')}")
        rec["entries"].append(entry)
        save_record(out, rec)
        write_json(out / "e0_idle_check.json", {"schema": "plan12.new_blocks_e0_check.v1", "citation": CITATION, "written_at": now_iso(), "encoding_out": str(enc_out) if enc_out else None,
                                                "move6_e0_gate": e0_status or "E0 usable", "identity": {k: identity.get(k) for k in ("cell_id", "passed", "status", "reason", "checked_at")},
                                                "e0_declared_files": (declared or {}).get("n_files"), **check})
        t0 = time.time()
        results = [one_cut(out, name, cut, runs, enc_out, enc_rows, win, pair_excluded, mode, e_sets, reason, run,
                           seen=seen, gap=gap, pools=pools, n_boot=n_boot, n_jobs=n_jobs, n_est=n_est, seed=seed) for name, cut in cuts.items()]
        move_rec = {"schema": "plan12.new_blocks.v1", "citation": CITATION, "package_version": __version__, "written_at": now_iso(), "command": sys.argv,
                    "question": __doc__.split("THE QUESTION THE TEST ANSWERS", 1)[1].split("THE DATA", 1)[0].strip(),
                    "params": entry["params"], "cut_convention": m6["cut_convention"], "plan_counts": counts, "levels": LEVEL_TEXT, "null": NO_NULL, "move6_rules": MOVE6_RULES,
                    "cells": {"n_kernel": len(kernel_runs), "n_idle": len(idle_runs), "kernels": sorted({r["kernel"] for r in kernel_runs}), "idle_cells": [r["cell_id"] for r in idle_runs],
                              "idle_label": IDLE_LABEL, "idle_archetype_label": IDLE_LABEL, "idle_archetype_in_cells_csv": I.IDLE_ARCHETYPE,
                              "order": "move 6's kernel cells in its order, then the idle cells in this run's cells.csv order (idle_class.ordered_runs_13)"},
                    "e0": {"move6_gate": e0_status or "E0 usable", "idle": {k: check[k] for k in ("can_build", "n_idle", "n_idle_with_e0", "reason")}},
                    "mode": mode, "feature_sets_run": list(e_sets), "code": entry["code"], "elapsed_s": round(time.time() - t0, 1), "results": results}
        write_json(out / "new_blocks.json", move_rec)
        entry["outputs"] = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and p.name not in (RECORD_NAME, LOCK_NAME))
        entry.update(status="done", exit_code=0, finished_at=now_iso(), elapsed_s=move_rec["elapsed_s"])
        for s in results:
            line = "; ".join(f"{lv}: " + ", ".join(f"{e} {fmt(((s['scores'][lv][e].get('pools') or {}).get('1') or {}).get('accuracy'))}" for e in e_sets) for lv in LEVELS)
            print(f"[newblocks] cut {s['cut']} ({s['cut_name']}): {s['split']['n_recordings']} recordings, {s['split']['n_new_blocks']} new blocks; single-block accuracy: {line}")
        print(f"[newblocks] done in {move_rec['elapsed_s']} s: {out}")
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
    p.add_argument("--out", default=None, help="the output folder (default: a sibling of the run, <run>_newblocks; never inside the run)")
    p.add_argument("--seen", type=float, default=DEFAULT_SEEN, help="the share of each recording's windows the model sees, in time order (default 0.8, rounded to windows)")
    p.add_argument("--gap", type=int, default=DEFAULT_GAP, help="windows skipped between the seen windows and the first new block (default 1: no shared pair at W64_H32)")
    p.add_argument("--pools", default=DEFAULT_POOLS, help="pool sizes, consecutive new blocks voted together (default 1,2,3)")
    p.add_argument("--cuts", default=None, help="which cuts, by name: declared,measured (default both; the values come from params.json)")
    p.add_argument("--encoding-out", default=None, help="decision D2 (default: the encoding run move 6 recorded in classify.json)")
    p.add_argument("--n-jobs", type=int, default=None, help="processes for the forest (default: move 6's recorded value)")
    p.add_argument("--n-estimators", type=int, default=None, help="trees (default: move 6's recorded value, Table 2's 300)")
    p.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP, help="resamples of the recordings for the margins' 95%% interval (default 1000)")
    p.add_argument("--force", action="store_true", help="re-run a finished test")
    p.add_argument("--dry-run", action="store_true", help="print the plan (the recordings, the cuts, the windows, the seen windows, the new blocks); write nothing; skip the live-lock refusal")


def main(argv: list[str] | None = None) -> int:
    install_sigterm()
    ap = argparse.ArgumentParser(prog="plan12_grounding.new_blocks", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    add_run_args(sub.add_parser("run", help="the new-block test at the three levels, both cuts"))
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
