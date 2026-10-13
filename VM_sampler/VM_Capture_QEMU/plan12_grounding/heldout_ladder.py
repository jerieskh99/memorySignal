#!/usr/bin/env python3
"""heldout_ladder.py -- the held-out ladder beside the run of plan12_grounding (2026-10-13; move 21 in the console's
Grounding paper tab, after the arrow ladder (20)): move 20's ladder tested on whole recordings never seen in training,
with shorter and shorter pieces.

  python3 -m plan12_grounding.heldout_ladder run --run <main run out> --length 64|32|16|8|4|2 [--out <dir>] [--cuts declared,measured]
        [--rungs rung1,...] [--bootstrap 1000] [--encoding-out <D2>] [--n-jobs N] [--n-estimators N] [--arrowladder-out <dir>] [--force] [--dry-run]
  python3 -m plan12_grounding.heldout_ladder summary --run <main run out> [--out <dir>] [--dry-run]
  (run from VM_sampler/VM_Capture_QEMU/)

THE QUESTION. Moves 12 and 20 name new blocks of recordings the model has already seen 80% of. What does each rung
of the arrow ladder buy when the recording itself is new, and how short a piece of it suffices?

THE SPLIT, by whole recordings: 4 folds; in fold k every group (the 12 kernels and idle) holds out its runs with rep 2k
and 2k + 1 (lexer has no rep 6, so its fold 3 holds out one run) and trains on all its other runs. Every run is tested
exactly once, by a model that never saw it; the code asserts that no held-out recording has a window in training and
that every recording is held out exactly once.

THE PIECES: lengths 64, 32, 16, 8, 4 and 2 pairs, at both cuts. For each length the training windows are that length,
taken from the training runs every half length (every pair at length 2), and the test pieces are the held-out runs cut
into consecutive non-overlapping pieces of that length (the windows of the same grid whose start is a multiple of the
length). Each test piece is named on its own, with no vote over pieces.

THE INPUT SETS: move 20's rungs 0 to 5b, each on its own (rung 0 N, H, A; 1 H; 2 H and A; 3 C and S; 4 the doubled
angle; 5a the l0 arrow; 5b the l1 arrow), the series read from move 1's files and from move 20's series
(<run>_arrowladder/series/<cell>.npz; move 20's step `series` must be done); each piece's features from the statistic
sets move 20 uses (stats.series_stats, the phase set for A). A statistic that cannot be computed at a length (non-finite
for some window or piece of that length, in any fold) is left out at that length for every fold and piece, and
left_out.csv lists what was left out per length and rung.

THE LEVELS: kernel (13 classes with idle) and archetype (5 with idle); the run level does not apply, since the run is
never seen. The forest and its settings as move 12 (new_blocks.fit_level: move 6's trees, processes and seed, the
dimension rule, the B1-G3 quarantine with the piece as the unit); the scores as move 12 scores single blocks
(new_blocks.score_pools: accuracy, macro recall over the classes present, chance 1/K, the majority class by recording
count, recall per kernel); the margins between rungs (2 - 1, 3 - 2, 4 - 3, 5a - 3, 5b - 3, 2 - 0, as move 20) with move
12's bootstrap over recordings on the same pieces (new_blocks.bootstrap_margin, SEED_BOOTSTRAP).

CHECK, before anything else (the length-64 step): with the training grid of length 64 and hop 32, every rung's feature
rows must equal move 20's rows for the same windows (<run>_arrowladder/ladder/cut<C>/<rung>/features.npz, move 12's
window frame), bit for bit; a failure stops the step.

THE STEPS: one per length (64 first, 2 last; the short lengths are the long steps), both cuts and every rung inside,
resumable per cut and rung (a rung whose result.json stands with the same parameters is kept; the features of a length
are cached per rung); `summary` collects every length present into the tables and figures per cut (one figure per cut
and level: accuracy against piece length, one line per rung); every length step refreshes them too.

THE OUTPUTS, under <out> (default <run>_heldout)
  heldout/cut<C>/len<L>/     scores.csv, recall_per_kernel.csv, margins.csv, left_out.csv, folds.csv, summary.json;
                             <rung>/predictions_<level>.csv (every test piece: cell, fold, position, window start, pair
                             range, truth, prediction, probabilities), <rung>/result.json; features/<rung>.npz (the
                             windows of every recording at this length, cached)
  heldout/cut<C>/            scores_all_lengths.csv, margins_all_lengths.csv, left_out_all_lengths.csv,
                             accuracy_vs_length_kernel.svg/.csv, accuracy_vs_length_archetype.svg/.csv
  heldout/len<L>.json        the step's record; heldout/summary.json the summary's; record.json the record book (one
                             entry per step run: command, inputs with sha256, params, the code fingerprint of this
                             module's closure, status; a finished step is skipped unless --force or an input, a
                             parameter or the code changed)

THE RULES THAT PROTECT THE LIVE RUN (as arrow_ladder.py)
- ONE new file; nothing existing in plan12_grounding/ is edited; arrow_ladder.py, new_blocks.py, idle_class.py and
  stats.py are imported as they are (their recorded code fingerprints stay); nothing imports this module. The run
  folder, every other sibling folder and the L1 stores are read only here; --out inside the run is refused (exit 2).
  A step refuses to start (exit 3) while <run>/.driver.lock belongs to a live process, or while move 20's lock
  (<run>_arrowladder/.arrow_ladder.lock) belongs to a live process; --dry-run skips those refusals and writes nothing.
  One writer per output folder: <out>/.heldout_ladder.lock; run_moves.install_sigterm.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

from plan12_grounding import __version__, toolkit_fingerprint  # noqa: E402
from plan12_grounding import arrow_ladder as AL  # noqa: E402  (move 20: the rungs, the feature names, the series files; imported, never edited)
from plan12_grounding import idle_class as I  # noqa: E402  (load_move6, the lock state, class_order, confusion_of)
from plan12_grounding import new_blocks as NB  # noqa: E402  (move 12: fit_level, score_pools, bootstrap_margin)
from plan12_grounding import stats as ST  # noqa: E402  (series_stats, cut_series, load_runs, cuts_of)
from plan12_grounding.run_moves import _pid_alive, code_fingerprint, install_sigterm, now_iso, read_json, sha256_file, write_json  # noqa: E402
from plan12_grounding.figures import svg_open, write_csv  # noqa: E402
from plan11_encoding_ladder.nulls import SEED_FOREST  # noqa: E402

CITATION = ("plan12_grounding/arrow_ladder.py (move 20: the rungs and their series, E_new's statistic sets on each rung; its feature rows are the check); "
            "plan12_grounding/new_blocks.py (move 12: the forest with the dimension rule and the quarantine, the scores, the bootstrap); the author's design of the "
            "held-out ladder (2026-10-13): 4 folds by rep over whole recordings, pieces of 64 to 2 pairs, every piece named alone")
OUT_SUFFIX = "_heldout"
LOCK_NAME = ".heldout_ladder.lock"
RECORD_NAME = "record.json"
KEY_SUMMARY = "summary"
LENGTHS = (64, 32, 16, 8, 4, 2)
N_FOLDS = 4
LEVELS = ("kernel", "archetype")
LEVEL_TEXT = {"kernel": "the kernel (the 12 kernels plus idle; 13 classes)", "archetype": "the archetype (the encoding run's archetype_predicted, no G-K0 relabel; idle = 'idle'; 5 classes)",
              "run": "not applicable: the run is never seen (every recording is held out whole)"}
RUNG_ORDER, RUNG_KEYS, RUNG_TEXT, LADDER_PAIRS, STORED = AL.RUNG_ORDER, AL.RUNG_KEYS, AL.RUNG_TEXT, AL.LADDER_PAIRS, AL.STORED
RUNG_COLOUR = AL.RUNG_COLOUR
IDLE_LABEL = NB.IDLE_LABEL
SEED_BOOTSTRAP = NB.SEED_BOOTSTRAP
NO_NULL = NB.NO_NULL
DEFAULT_BOOTSTRAP = NB.DEFAULT_BOOTSTRAP
FEATURES_VERSION = "plan12.heldout_features.v1"
EXIT_OK, EXIT_ERROR, EXIT_MISSING, EXIT_REFUSED = I.EXIT_OK, I.EXIT_ERROR, I.EXIT_MISSING, I.EXIT_REFUSED
FOLD_RULE = "fold k holds out, in every group (the 12 kernels and idle), the runs with rep 2k and 2k + 1 (k = 0 .. 3) and trains on all the other runs of all groups"
PIECE_RULE = ("training windows of the length every half length (every pair at length 2) over the cut series of the training runs; test pieces = the windows of the same grid "
              "whose start is a multiple of the length (consecutive, non-overlapping) over the held-out runs; every piece named alone")
LEFT_OUT_RULE = "a statistic non-finite for any window or piece of the length (training or test, any fold) is left out at that length for every fold and piece"
Stop = I.Stop


def fmt(x) -> str:
    return I.fmt(x)


def hop_of(L: int) -> int:
    return max(1, L // 2)


# ---------------------------------------------------------------------------------------------
# the output folder, the locks, the record (arrow_ladder's pattern)
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
            raise Stop(EXIT_REFUSED, f"refused: another heldout_ladder step is writing {out} ({lock}: pid {pid}, started {old.get('started_at')}); one writer per output folder")
        print(f"[heldout] stale lock from pid {pid} ({old.get('started_at')}) taken over", flush=True)
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
    return {"schema": "plan12.heldout_ladder.record.v1", "citation": CITATION, "package_version": __version__, "entries": []}


def save_record(out: Path, rec: dict) -> None:
    write_json(out / RECORD_NAME, rec)


def last_done(rec: dict, key: str) -> dict | None:
    for e in reversed(rec.get("entries") or []):
        if e.get("key") == key and e.get("status") == "done":
            return e
    return None


def code_identity() -> dict:
    fp = code_fingerprint("heldout_ladder")
    files = toolkit_fingerprint()["files"]
    return {"fingerprint": fp["sha256"], "modules": fp["modules"], "sha256": {f"{m}.py": files.get(f"{m}.py") for m in fp["modules"]}}


def refuse_if_busy(run: Path, al_out: Path, dry_run: bool) -> dict:
    lock = I.driver_lock_state(run)
    if lock["alive"] and not dry_run:
        raise Stop(EXIT_REFUSED, f"refused: a move is running on this run ({lock['path']}: pid {lock['pid']}, started {lock['started_at']}, {lock['argv_tail']}); "
                                 "this move never competes with a running move. Run again after it ends; --dry-run shows the plan meanwhile.")
    al_lock = al_out / AL.LOCK_NAME
    if al_lock.is_file() and not dry_run:
        try:
            old = read_json(al_lock)
        except Exception:                                  # noqa: BLE001
            old = {}
        if old.get("pid") and _pid_alive(int(old["pid"])):
            raise Stop(EXIT_REFUSED, f"refused: move 20 is still writing {al_out} ({al_lock}: pid {old['pid']}, started {old.get('started_at')}); run again after it ends")
    return lock


def _finish(out: Path, entry: dict, out_lock: Path) -> None:
    if entry["status"] == "running":
        entry.update(status="failed: stopped before the end", finished_at=now_iso())
    rec = load_record(out)
    rec["entries"] = [e for e in rec["entries"] if not (e.get("key") == entry["key"] and e.get("started_at") == entry["started_at"])] + [entry]
    save_record(out, rec)
    release_out_lock(out_lock)


def _skip_if_unchanged(out: Path, key: str, entry: dict, sig_skip: tuple, outputs_exist: bool, force: bool, tag: str) -> bool:
    rec = load_record(out)
    prev = last_done(rec, key)
    sig = {k: v for k, v in entry["params"].items() if k not in sig_skip}
    if prev is not None and not force and outputs_exist:
        why = None
        if prev.get("inputs_sha256") != entry["inputs_sha256"]:
            why = "an input changed"
        elif {k: v for k, v in (prev.get("params") or {}).items() if k not in sig_skip} != sig:
            why = "the parameters changed"
        elif (prev.get("code") or {}).get("fingerprint") != entry["code"]["fingerprint"]:
            why = "the code changed"
        if why is None:
            entry.update(status="skipped: outputs exist and inputs, parameters and code unchanged", exit_code=0, finished_at=now_iso(),
                         made_by={"finished_at": prev.get("finished_at"), "code_fingerprint": (prev.get("code") or {}).get("fingerprint")}, outputs=prev.get("outputs") or [])
            rec["entries"].append(entry)
            save_record(out, rec)
            print(f"[heldout] skipped: {tag} stands as made on {prev.get('finished_at')} (inputs, parameters and code unchanged; --force re-runs)")
            return True
        print(f"[heldout] re-running: {why} since {prev.get('finished_at')}")
    rec["entries"].append(entry)
    save_record(out, rec)
    return False


# ---------------------------------------------------------------------------------------------
# the recordings, their labels, the folds, the series
# ---------------------------------------------------------------------------------------------
def labelled_runs(run: Path) -> list[dict]:
    """The run's admissible recordings with a complete series (stats.load_runs), each with its group (the kernel, 'idle'
    for the idle runs), its archetype label ('idle' for idle, else the encoding run's archetype_predicted) and its rep."""
    runs = ST.load_runs(run)
    out = []
    for r in runs:
        idle = r["role"] == "idle"
        out.append({**r, "group": "idle" if idle else r["kernel"], "label_kernel": IDLE_LABEL if idle else r["kernel"],
                    "label_archetype": IDLE_LABEL if idle else str(r.get("archetype") or ""), "rep": int(r["rep"]) if r.get("rep") is not None else 0})
    return out


def make_folds(runs: list[dict]) -> list[dict]:
    folds = []
    held_count = {r["cell_id"]: 0 for r in runs}
    for k in range(N_FOLDS):
        held = [r["cell_id"] for r in runs if r["rep"] in (2 * k, 2 * k + 1)]
        train = [r["cell_id"] for r in runs if r["cell_id"] not in set(held)]
        if set(held) & set(train):
            raise Stop(EXIT_ERROR, f"stopped: fold {k} holds out a recording it also trains on")
        for c in held:
            held_count[c] += 1
        groups_held = sorted({next(r["group"] for r in runs if r["cell_id"] == c) for c in held})
        folds.append({"fold": k, "reps": [2 * k, 2 * k + 1], "heldout": held, "train": train, "n_heldout": len(held), "n_train": len(train), "groups_held_out": groups_held})
    never = [c for c, n in held_count.items() if n == 0]
    twice = [c for c, n in held_count.items() if n > 1]
    if never or twice:
        raise Stop(EXIT_ERROR, f"stopped: the folds do not test every recording exactly once (never: {never[:3]}, more than once: {twice[:3]}); reps outside 0 .. {2 * N_FOLDS - 1}?")
    return folds


def load_series(run: Path, al_out: Path, runs: list[dict]) -> dict:
    """{cell: the series dict of move 20 (move 1's N, H, A, C, S and move 20's C2, S2, L0C, L0S, L1C, L1S, L0, L1 on move 1's pair axis)}."""
    out = {}
    for r in runs:
        stored = r["arrays"]
        mine = AL.load_rung_series(al_out, r["cell_id"])
        if not np.array_equal(np.asarray(stored["pair"]), np.asarray(mine["pair"])):
            raise Stop(EXIT_ERROR, f"stopped: the pair axis of move 20's series for {r['cell_id']} differs from move 1's")
        out[r["cell_id"]] = {"pair": np.asarray(stored["pair"]), **{k: np.asarray(stored[k]) for k in STORED}, **{k: mine[k] for k in ("C2", "S2", "L0C", "L0S", "L1C", "L1S", "L0", "L1")}}
    return out


# ---------------------------------------------------------------------------------------------
# the windows of one length: every recording's windows on the half-length grid, per rung (cached)
# ---------------------------------------------------------------------------------------------
def features_path(out: Path, cut: int, L: int, rung: str) -> Path:
    return out / "heldout" / f"cut{cut}" / f"len{L}" / "features" / f"{rung}.npz"


def safe_series_stats(name: str, x: np.ndarray) -> tuple[dict, dict]:
    """stats.series_stats as it is; when one of its statistic sets raises on a short piece (numpy's histogram refuses 16 bins on
    a near-constant pair of large values, for one), the sets are computed one by one and the set that raised reads NaN for that
    window, so the column rule leaves it out at that length. Returns (stats, {set: error} for the sets that raised)."""
    try:
        return ST.series_stats(name, x), {}
    except Exception as first:                             # noqa: BLE001
        xf = np.asarray(x, dtype=np.float64)
        xf = xf[np.isfinite(xf)]
        out, failed = {}, {}
        nov_keys = ST.NOV_PHASE_STATS if name == "A" else ST.NOV_MAG_STATS
        for label, keys, fn in (("b1", ST.B1_FEAT, lambda: ST.b1_stats(xf)), ("deep", ST.DEEP_STATS, lambda: ST.deep_stats(xf)),
                                ("nov", nov_keys, lambda: (ST.nov_phase(xf) if name == "A" else ST.nov_magnitude(xf)))):
            for k in keys:
                out[f"{name}.{label}.{k}"] = float("nan")
            try:
                for k, v in fn().items():
                    out[f"{name}.{label}.{k}"] = v
            except Exception as e:                         # noqa: BLE001
                failed[f"{name}.{label}"] = f"{type(e).__name__}: {e}"[:120]
        if not failed:
            failed[f"{name}.?"] = f"{type(first).__name__}: {first}"[:120]
        return out, failed


def windows_of_length(series_by_cell: dict, cells: list[str], cut: int, L: int, keys: tuple) -> dict:
    hop = hop_of(L)
    names = AL.rung_feature_names(keys)
    X_rows, cell_ids, starts = [], [], []
    failures: dict = {}
    for c in cells:
        a = ST.cut_series({"arrays": series_by_cell[c]}, cut)
        n = int(a["pair"].size)
        t = 0
        while t + L <= n:
            st = {}
            for s in keys:
                vals, failed = safe_series_stats(s, a[s][t:t + L].astype(np.float64))
                st.update(vals)
                for k, msg in failed.items():
                    f = failures.setdefault(k, {"n_windows": 0, "first_error": msg})
                    f["n_windows"] += 1
            X_rows.append([st[k] for k in names])
            cell_ids.append(c); starts.append(t)
            t += hop
    X = np.asarray(X_rows, dtype=np.float64).reshape(len(X_rows), len(names))
    return {"X": X, "names": names, "cell_id": np.array(cell_ids, dtype=str), "win_start": np.array(starts, dtype=np.int64), "hop": hop, "L": L, "failures": failures}


def load_or_build_features(out: Path, series_by_cell: dict, cells: list[str], cut: int, L: int, rung: str, force: bool) -> tuple[dict, float, bool]:
    p = features_path(out, cut, L, rung)
    if p.is_file() and not force:
        try:
            with np.load(p, allow_pickle=False) as z:
                if str(z["version"]) == FEATURES_VERSION and int(z["L"]) == L and int(z["cut"]) == cut and set(z["cell_id"].astype(str).tolist()) == set(cells):
                    failures = json.loads(str(z["failures"])) if "failures" in z.files else {}
                    return {"X": z["X"], "names": z["names"].astype(str).tolist(), "cell_id": z["cell_id"].astype(str), "win_start": z["win_start"].astype(np.int64), "hop": int(z["hop"]),
                            "L": L, "failures": failures}, 0.0, True
        except Exception:                                  # noqa: BLE001
            pass
    t0 = time.time()
    f = windows_of_length(series_by_cell, cells, cut, L, RUNG_KEYS[rung])
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, X=f["X"], names=np.array(f["names"], dtype=str), cell_id=f["cell_id"], win_start=f["win_start"], hop=np.int64(f["hop"]), L=np.int64(L), cut=np.int64(cut),
                        version=np.array(FEATURES_VERSION), rung=np.array(rung), failures=np.array(json.dumps(f["failures"])))
    return f, round(time.time() - t0, 1), False


def check_against_move20(al_out: Path, cut: int, rung: str, f: dict) -> dict:
    """The check: at length 64 (hop 32) every rung's rows must equal move 20's rows for the same windows (move 12's frame)."""
    p = al_out / "ladder" / f"cut{cut}" / rung / "features.npz"
    if not p.is_file():
        return {"ok": None, "status": f"missing: {p} (move 20's ladder has not run at cut {cut})"}
    with np.load(p, allow_pickle=False) as z:
        X20, names20 = z["X"], z["feature_names"].astype(str).tolist()
        keys20 = list(zip(z["cell_id"].astype(str).tolist(), z["win_start"].astype(np.int64).tolist()))
    mine = {(c, int(t)): i for i, (c, t) in enumerate(zip(f["cell_id"].tolist(), f["win_start"].tolist()))}
    missing = [k for k in keys20 if k not in mine]
    if names20 != f["names"]:
        return {"ok": False, "status": "the feature names differ", "n_move20_rows": len(keys20)}
    if missing:
        return {"ok": False, "status": f"{len(missing)} of move 20's windows have no row here (first: {missing[0]})", "n_move20_rows": len(keys20)}
    idx = np.array([mine[k] for k in keys20], dtype=np.int64)
    equal = np.array_equal(f["X"][idx], X20, equal_nan=True)
    dev = float(np.nanmax(np.abs(f["X"][idx] - X20))) if not equal else 0.0
    return {"ok": bool(equal), "status": "ok" if equal else f"rows differ (largest absolute deviation {dev:.3g})", "n_move20_rows": len(keys20), "n_rows_here": int(f["X"].shape[0]),
            "max_abs_dev": dev, "file": str(p)}


# ---------------------------------------------------------------------------------------------
# one rung at one length and cut: the four folds, both levels
# ---------------------------------------------------------------------------------------------
def run_rung(d: Path, rung: str, f: dict, folds: list[dict], runs_by_id: dict, classes: dict, majority: dict, kernel_of: dict, *, L: int, cut: int, n_jobs: int, n_est: int,
             seed: int) -> dict:
    d.mkdir(parents=True, exist_ok=True)
    X_all, names_all, cell_rows, starts = f["X"], f["names"], f["cell_id"], f["win_start"]
    finite = np.isfinite(X_all).all(axis=0)
    left_out = [n for n, ok in zip(names_all, finite) if not ok]
    keep = np.flatnonzero(finite)
    X, names = X_all[:, keep], [names_all[j] for j in keep]
    if X.shape[1] == 0:
        raise Stop(EXIT_ERROR, f"stopped: every statistic of {rung} is non-finite at length {L} (cut {cut}); nothing to train on")
    res = {"rung": rung, "keys": list(RUNG_KEYS[rung]), "text": RUNG_TEXT[rung], "length": L, "hop": int(f["hop"]), "n_windows": int(X.shape[0]), "n_features": len(names),
           "left_out": left_out, "statistic_sets_that_raised": f.get("failures") or {}, "levels": {}}
    is_piece = (starts % L == 0)
    for level in LEVELS:
        y = np.array([runs_by_id[c][f"label_{level}"] for c in cell_rows.tolist()], dtype=str)
        pieces, fold_recs = [], []
        t0 = time.time()
        for fo in folds:
            held = set(fo["heldout"])
            tr = np.flatnonzero(~np.isin(cell_rows, list(held)))
            te = np.flatnonzero(np.isin(cell_rows, list(held)) & is_piece)
            if set(cell_rows[tr].tolist()) & held:
                raise Stop(EXIT_ERROR, f"stopped: fold {fo['fold']}: a held-out recording has a window in training")
            if te.size == 0:
                fold_recs.append({"fold": fo["fold"], "status": "no test piece", "n_train_windows": int(tr.size)})
                continue
            pos = np.zeros(te.size, dtype=np.int64)
            for c in held:
                m = cell_rows[te] == c
                pos[m] = np.arange(1, int(m.sum()) + 1)
            # the fold's rows: the training windows then the test pieces (the held-out runs' other windows take no part, so every row the forest sees is train or test)
            sel = np.concatenate([tr, te])
            sp = {"train_idx": np.arange(tr.size), "test_idx": np.arange(tr.size, sel.size), "cell_id_rows": cell_rows[sel],
                  "block_id": np.array([f"{c}|b{k}" for c, k in zip(cell_rows[te].tolist(), pos.tolist())], dtype=str), "block_pos": pos, "cells": sorted(set(cell_rows[sel].tolist()))}
            fit = NB.fit_level(X[sel], names, y[sel], sp, seed=seed, n_jobs=n_jobs, n_est=n_est)
            use = fit["use"]
            if "pred" not in use:
                fold_recs.append({"fold": fo["fold"], "status": fit["status"], "n_train_windows": int(tr.size), "quarantined_features": fit["quarantined_features"]})
                continue
            ci = {c: i for i, c in enumerate(use["classes"])}
            for i, j in enumerate(te.tolist()):
                pieces.append({"cell_id": str(cell_rows[j]), "fold": fo["fold"], "start_pos": int(pos[i]), "size": 1, "win_start": int(starts[j]), "y_true": str(y[j]),
                               "y_pred": str(use["pred"][i]), "vote_fraction": float(use["proba"][i].max()), "p_true": float(use["proba"][i][ci[str(y[j])]]) if str(y[j]) in ci else None,
                               "block_rows": [int(j)]})
            fold_recs.append({"fold": fo["fold"], "status": "ok", "n_train_windows": int(tr.size), "n_train_cells": fit["n_train_cells"], "n_test_pieces": int(te.size),
                              "feature_count": use["feature_count"], "feature_count_used": use["feature_count_used"], "reduced": use["reduced"], "score_source": fit["score_source"],
                              "quarantined_features": fit["quarantined_features"], "classes_train": fit["classes_train"]})
        sc = NB.score_pools(pieces, classes[level], kernel_of, majority[level]) if pieces else {"n_pools": 0, "accuracy": None, "macro_recall": None, "chance": None, "majority": None,
                                                                                                "n_classes": len(classes[level]), "recall_per_class": {}, "recall_per_kernel": {}}
        res["levels"][level] = {"status": "ok" if pieces else "not run: no test piece", "scores": {k: v for k, v in sc.items() if k != "recall_per_class"}, "folds": fold_recs,
                                "n_pieces": len(pieces), "quarantined_union": sorted({q for fr in fold_recs for q in (fr.get("quarantined_features") or [])}),
                                "elapsed_s": round(time.time() - t0, 1)}
        res["levels"][level]["_pieces"] = pieces
        write_csv(d / f"predictions_{level}.csv", ["cell_id", "kernel", "fold", "position", "win_start", "pair_start", "pair_end", "y_true", "y_pred", "p_pred", "p_true"],
                  [[p["cell_id"], kernel_of[p["cell_id"]], p["fold"], p["start_pos"], p["win_start"], int(runs_by_id[p["cell_id"]]["_pair0"][cut] + p["win_start"]),
                    int(runs_by_id[p["cell_id"]]["_pair0"][cut] + p["win_start"] + L - 1), p["y_true"], p["y_pred"], p["vote_fraction"], p["p_true"]] for p in pieces])
        print(f"[heldout] cut {cut} len {L:<2} {rung:<7} {level:<9} acc {fmt(sc.get('accuracy'))} macro {fmt(sc.get('macro_recall'))} chance {fmt(sc.get('chance'))} "
              f"majority {fmt(sc.get('majority'))} | {len(pieces)} pieces, {len(names)} features ({len(left_out)} left out) ({res['levels'][level]['elapsed_s']} s)", flush=True)
    write_json(d / "result.json", {"schema": "plan12.heldout_rung.v1", "citation": CITATION, "written_at": now_iso(),
                                   **{k: v for k, v in res.items() if k != "levels"}, "levels": {lv: {k: v for k, v in r.items() if k != "_pieces"} for lv, r in res["levels"].items()},
                                   "pieces": {lv: r["_pieces"] for lv, r in res["levels"].items()}})
    return res


def rung_stands(d: Path, params_sig: dict) -> dict | None:
    p = d / "result.json"
    if not p.is_file():
        return None
    try:
        r = read_json(p)
    except Exception:                                      # noqa: BLE001
        return None
    if r.get("params_sig") != params_sig:
        return None
    for lv, pieces in (r.get("pieces") or {}).items():
        r["levels"][lv]["_pieces"] = pieces
    return r


# ---------------------------------------------------------------------------------------------
# one length at one cut: the rungs, the tables
# ---------------------------------------------------------------------------------------------
def length_cut(out: Path, al_out: Path, cut_name: str, cut: int, L: int, runs: list[dict], series_by_cell: dict, folds: list[dict], rungs: list[str], *, n_boot: int, n_jobs: int,
               n_est: int, seed: int, force: bool, params_sig: dict, check_len: int = 64) -> dict:
    d = out / "heldout" / f"cut{cut}" / f"len{L}"
    d.mkdir(parents=True, exist_ok=True)
    t_cut = time.time()
    cells = [r["cell_id"] for r in runs]
    runs_by_id = {r["cell_id"]: r for r in runs}
    for r in runs:
        r.setdefault("_pair0", {})[cut] = int(ST.cut_series(r, cut)["pair"][0])      # the first pair after the cut, for the pieces' pair ranges
    kernel_of = {r["cell_id"]: r["label_kernel"] for r in runs}
    classes = {lv: sorted({r[f"label_{lv}"] for r in runs}) for lv in LEVELS}
    majority = {}
    for lv in LEVELS:
        counts: dict = {}
        for r in runs:
            counts[r[f"label_{lv}"]] = counts.get(r[f"label_{lv}"], 0) + 1
        majority[lv] = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    write_csv(d / "folds.csv", ["fold", "reps_held_out", "n_heldout", "n_train", "groups_held_out", "heldout_cells"],
              [[fo["fold"], " ".join(map(str, fo["reps"])), fo["n_heldout"], fo["n_train"], " ".join(fo["groups_held_out"]), " ".join(fo["heldout"])] for fo in folds])
    results, timing, checks = {}, {}, {}
    for rung in rungs:
        rd = d / rung
        prev = None if force else rung_stands(rd, params_sig)
        if prev is not None:
            results[rung] = prev
            timing[rung] = {"features_s": 0.0, "folds_s": 0.0, "kept": True}
            checks[rung] = prev.get("check_move20")
            print(f"[heldout] cut {cut} len {L} {rung}: stands from an earlier run (result.json matches the parameters); kept", flush=True)
            continue
        f, t_feat, cached = load_or_build_features(out, series_by_cell, cells, cut, L, rung, force)
        print(f"[heldout] cut {cut} len {L:<2} {rung:<7} {RUNG_TEXT[rung][:60]}: {f['X'].shape[0]} windows (hop {f['hop']}), {len(f['names'])} statistics "
              f"({'cached' if cached else f'{t_feat} s'})", flush=True)
        chk = None
        if L == check_len:                                 # the frame's window (W64_H32 on the real run): every rung's rows must equal move 20's for the same windows
            chk = check_against_move20(al_out, cut, rung, f)
            if chk["ok"] is False:
                raise Stop(EXIT_ERROR, f"stopped: the check failed for {rung} at cut {cut}: {chk['status']}; nothing else written")
            print(f"[heldout] cut {cut} len {L} {rung}: check against move 20's rows: {chk['status']} ({chk.get('n_move20_rows')} windows)", flush=True)
        t1 = time.time()
        res = run_rung(rd, rung, f, folds, runs_by_id, classes, majority, kernel_of, L=L, cut=cut, n_jobs=n_jobs, n_est=n_est, seed=seed)
        res["params_sig"] = params_sig
        res["check_move20"] = chk
        j = read_json(rd / "result.json"); j["params_sig"] = params_sig; j["check_move20"] = chk; write_json(rd / "result.json", j)
        results[rung] = res
        timing[rung] = {"features_s": t_feat, "folds_s": round(time.time() - t1, 1), "kept": False}
        checks[rung] = chk
    # the tables
    score_rows, rk_rows, lo_rows = [], [], []
    for lv in LEVELS:
        for rung in rungs:
            r = results[rung]
            s = (r["levels"].get(lv) or {}).get("scores") or {}
            fr = (r["levels"].get(lv) or {}).get("folds") or []
            ok_f = [x for x in fr if x.get("status") == "ok"]
            score_rows.append([cut, L, rung, RUNG_TEXT[rung], lv, (r["levels"].get(lv) or {}).get("status"), s.get("n_pools"), s.get("n_classes"), s.get("accuracy"), s.get("macro_recall"),
                               s.get("chance"), s.get("majority"), r["n_features"], len(r["left_out"]),
                               (min(x["feature_count_used"] for x in ok_f) if ok_f else None), (max(x["feature_count_used"] for x in ok_f) if ok_f else None),
                               ";".join((r["levels"].get(lv) or {}).get("quarantined_union") or []), (float(np.mean([x["n_train_windows"] for x in ok_f])) if ok_f else None), NO_NULL])
            for kern in I.class_order(set(kernel_of.values())):
                if kern in (s.get("recall_per_kernel") or {}):
                    rk_rows.append([cut, L, rung, lv, kern, s["recall_per_kernel"][kern], sum(1 for p in (r["levels"][lv].get("_pieces") or []) if kernel_of[p["cell_id"]] == kern)])
    for rung in rungs:
        raised = results[rung].get("statistic_sets_that_raised") or {}
        lo_rows.append([cut, L, rung, len(results[rung]["left_out"]), results[rung]["n_features"], ";".join(results[rung]["left_out"]),
                        "; ".join(f"{k}: {v['n_windows']} windows ({v['first_error']})" for k, v in raised.items())])
    write_csv(d / "scores.csv", ["cut", "length", "rung", "input_set", "level", "status", "n_pieces", "n_classes", "accuracy", "macro_recall", "chance", "majority", "n_features",
                                 "n_left_out", "feature_count_used_min", "feature_count_used_max", "quarantined_features_union", "n_train_windows_mean", "null"], score_rows)
    write_csv(d / "recall_per_kernel.csv", ["cut", "length", "rung", "level", "kernel", "recall", "n_pieces"], rk_rows)
    write_csv(d / "left_out.csv", ["cut", "length", "rung", "n_left_out", "n_kept", "statistics_left_out", "statistic_sets_that_raised"], lo_rows)
    margin_rows, margins = [], {lv: {} for lv in LEVELS}
    for lv in LEVELS:
        for b, a in LADDER_PAIRS:
            if b not in results or a not in results:
                continue
            pb, pa = results[b]["levels"][lv].get("_pieces"), results[a]["levels"][lv].get("_pieces")
            comp = f"{b}-{a}"
            if not pb or not pa:
                margin_rows.append([cut, L, lv, comp, None, None, None, None, None, None, None, "not run: no piece"])
                continue
            bm = NB.bootstrap_margin(pb, pa, cells, n_boot, SEED_BOOTSTRAP)
            margins[lv][comp] = bm
            margin_rows.append([cut, L, lv, comp, bm["delta"], bm["ci_lo"], bm["ci_hi"], bm["n_pools"], bm["n_recordings"], bm["gained"], bm["lost"], bm["status"]])
    write_csv(d / "margins.csv", ["cut", "length", "level", "comparison", "delta_accuracy", "ci95_lo", "ci95_hi", "n_pieces", "n_recordings", "n_pieces_gained", "n_pieces_lost", "status"], margin_rows)
    summary = {"schema": "plan12.heldout_length.v1", "citation": CITATION, "cut": int(cut), "cut_name": cut_name, "length": L, "hop": hop_of(L), "fold_rule": FOLD_RULE, "piece_rule": PIECE_RULE,
               "left_out_rule": LEFT_OUT_RULE, "levels": LEVEL_TEXT, "classes": classes, "majority_class": majority, "n_recordings": len(cells), "folds": folds,
               "n_estimators": n_est, "n_jobs": n_jobs, "seed_forest": seed, "seed_bootstrap": SEED_BOOTSTRAP, "n_bootstrap": n_boot, "null": NO_NULL,
               "check_move20": checks, "left_out": {rung: results[rung]["left_out"] for rung in rungs},
               "scores": {lv: {rung: {k: v for k, v in results[rung]["levels"][lv].items() if k not in ("_pieces",)} for rung in rungs} for lv in LEVELS},
               "margins": margins, "time_per_rung_s": timing, "elapsed_s": round(time.time() - t_cut, 1)}
    write_json(d / "summary.json", summary)
    line = "; ".join(f"{lv}: " + ", ".join(f"{r.replace('rung', 'r')} {fmt(((results[r]['levels'].get(lv) or {}).get('scores') or {}).get('accuracy'))}" for r in rungs) for lv in LEVELS)
    print(f"[heldout] cut {cut} len {L} done in {summary['elapsed_s']} s; accuracy on held-out pieces: {line}", flush=True)
    return summary


# ---------------------------------------------------------------------------------------------
# the summary across lengths: tables and figures per cut
# ---------------------------------------------------------------------------------------------
def _rows(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def _f(x):
    try:
        return float(x) if x not in (None, "") else None
    except (TypeError, ValueError):
        return None


def accuracy_vs_length_svg(rows: list[dict], cut: int, level: str, rungs: list[str]) -> str:
    """Accuracy on held-out pieces against the piece length (log axis, 2 to 64), one line per rung, chance as a grey line,
    the majority class dotted red; a length not yet run leaves a gap."""
    W_, H_ = 780, 360
    out = svg_open(W_, H_, f"The held-out ladder: accuracy against piece length, the {level} level, cut of {cut} pairs",
                   "every recording held out whole (4 folds by rep); each piece named alone; one line per rung; grey: chance 1/K; dotted red: the majority class")
    x0, y0, w, h = 60, 60, 480, 230
    xs = {L: x0 + 20 + (w - 40) * i / (len(LENGTHS) - 1) for i, L in enumerate(sorted(LENGTHS))}
    out.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="none" stroke="#ccc"/>')
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = y0 + h - h * t
        out.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0 + w}" y2="{yy:.1f}" stroke="#eee"/><text x="{x0 - 4}" y="{yy + 3:.1f}" text-anchor="end">{t:.2f}</text>')
    for L in LENGTHS:
        out.append(f'<text x="{xs[L]:.1f}" y="{y0 + h + 14}" text-anchor="middle">{L}</text>')
    out.append(f'<text x="{x0 + w / 2:.1f}" y="{y0 + h + 30}" text-anchor="middle" fill="#555">piece length (pairs)</text>')
    chance = next((_f(r["chance"]) for r in rows if r["level"] == level and _f(r["chance"]) is not None), None)
    majority = next((_f(r["majority"]) for r in rows if r["level"] == level and _f(r["majority"]) is not None), None)
    if chance is not None:
        out.append(f'<line x1="{x0}" y1="{y0 + h - h * chance:.1f}" x2="{x0 + w}" y2="{y0 + h - h * chance:.1f}" stroke="#888"/>')
    if majority is not None:
        out.append(f'<line x1="{x0}" y1="{y0 + h - h * majority:.1f}" x2="{x0 + w}" y2="{y0 + h - h * majority:.1f}" stroke="#b03a2e" stroke-dasharray="1,2"/>')
    for ri, rung in enumerate(rungs):
        pts = sorted([(int(r["length"]), _f(r["accuracy"])) for r in rows if r["level"] == level and r["rung"] == rung and _f(r["accuracy"]) is not None])
        if not pts:
            continue
        colour = RUNG_COLOUR.get(rung, "#333")
        points = " ".join(f"{xs[L]:.1f},{y0 + h - h * a:.1f}" for L, a in pts)
        dash = ' stroke-dasharray="4,3"' if rung == "rung0" else ""
        out.append(f'<polyline points="{points}" fill="none" stroke="{colour}" stroke-width="{2.2 if rung == "rung0" else 1.6}"{dash}/>')
        for L, a in pts:
            out.append(f'<circle cx="{xs[L]:.1f}" cy="{y0 + h - h * a:.1f}" r="2.5" fill="{colour}"/>')
        out.append(f'<text x="{x0 + w + 6}" y="{y0 + 12 + ri * 13}" font-size="9" fill="{colour}">{rung.replace("rung", "r")} {RUNG_TEXT[rung].split(":")[0][:22]}</text>')
    out.append(f'<text x="{x0}" y="{H_ - 10}" fill="#555" font-size="9">r0 N,H,A; r1 H; r2 H,A; r3 C,S; r4 doubled angle; r5a l0 arrow; r5b l1 arrow; {NO_NULL[:70]}...</text></svg>')
    return "\n".join(out)


def write_summary(out: Path, cuts: dict, rungs: list[str], quiet: bool = False) -> dict:
    present = {}
    for cut_name, cut in cuts.items():
        d = out / "heldout" / f"cut{cut}"
        sc, mg, lo = [], [], []
        for L in LENGTHS:
            ld = d / f"len{L}"
            if (ld / "summary.json").is_file():
                sc += _rows(ld / "scores.csv"); mg += _rows(ld / "margins.csv"); lo += _rows(ld / "left_out.csv")
                present.setdefault(cut, []).append(L)
        if not sc:
            continue
        d.mkdir(parents=True, exist_ok=True)
        write_csv(d / "scores_all_lengths.csv", list(sc[0].keys()), [[r.get(c) for c in sc[0].keys()] for r in sc])
        if mg:
            write_csv(d / "margins_all_lengths.csv", list(mg[0].keys()), [[r.get(c) for c in mg[0].keys()] for r in mg])
        if lo:
            write_csv(d / "left_out_all_lengths.csv", list(lo[0].keys()), [[r.get(c) for c in lo[0].keys()] for r in lo])
        for lv in LEVELS:
            (d / f"accuracy_vs_length_{lv}.svg").write_text(accuracy_vs_length_svg(sc, cut, lv, rungs))
            write_csv(d / f"accuracy_vs_length_{lv}.csv", ["cut", "length", "rung", "accuracy", "macro_recall", "chance", "majority", "n_pieces"],
                      [[r["cut"], r["length"], r["rung"], r["accuracy"], r["macro_recall"], r["chance"], r["majority"], r["n_pieces"]] for r in sc if r["level"] == lv])
    write_json(out / "heldout" / "summary.json", {"schema": "plan12.heldout_summary.v1", "citation": CITATION, "written_at": now_iso(), "cuts": cuts, "lengths": list(LENGTHS),
                                                  "lengths_present": {str(c): v for c, v in present.items()}, "rungs": rungs})
    if not quiet:
        print(f"[heldout] summary: " + "; ".join(f"cut {c}: lengths {v}" for c, v in present.items()) + (" (nothing yet)" if not present else ""))
    return present


# ---------------------------------------------------------------------------------------------
# the steps
# ---------------------------------------------------------------------------------------------
def prepare(o: argparse.Namespace, need_series: bool = True) -> dict:
    run = Path(os.path.expanduser(o.run)).resolve()
    if not (run / "cells.csv").is_file():
        raise Stop(EXIT_MISSING, f"missing input: {run / 'cells.csv'} (--run must name a plan12_grounding output folder with moves 0, 1 and 6 done)")
    out = resolve_out(run, o.out)
    al_out = Path(os.path.expanduser(o.arrowladder_out)).resolve() if getattr(o, "arrowladder_out", None) else AL.default_out(run)
    lock = refuse_if_busy(run, al_out, o.dry_run)
    m6 = I.load_move6(run, getattr(o, "encoding_out", None))
    cuts = I.parse_cuts(getattr(o, "cuts", None), m6["cuts"])
    runs = labelled_runs(run)
    folds = make_folds(runs)
    missing = [r["cell_id"] for r in runs if not AL.series_stands(al_out, r["cell_id"], None)] if need_series else []
    return {"run": run, "out": out, "al_out": al_out, "lock": lock, "m6": m6, "cuts": cuts, "runs": runs, "folds": folds, "missing_series": missing}


def run_length(o: argparse.Namespace) -> int:
    L = int(o.length)
    if L not in LENGTHS:
        raise Stop(EXIT_MISSING, f"--length {L}: the lengths are {', '.join(map(str, LENGTHS))}")
    P = prepare(o)
    run, out, al_out, m6, cuts, runs, folds = P["run"], P["out"], P["al_out"], P["m6"], P["cuts"], P["runs"], P["folds"]
    n_boot = int(o.bootstrap)
    rungs = [s.strip() for s in str(o.rungs).split(",") if s.strip()] if o.rungs else list(RUNG_ORDER)
    bad = [r for r in rungs if r not in RUNG_KEYS]
    if bad:
        raise Stop(EXIT_MISSING, f"--rungs {bad}: the rungs are {', '.join(RUNG_ORDER)}")
    n_jobs = int(o.n_jobs) if o.n_jobs is not None else m6["n_jobs"]
    n_est = int(o.n_estimators) if o.n_estimators is not None else m6["n_estimators"]
    seed = SEED_FOREST + m6["seed_offset"]
    n_idle = sum(1 for r in runs if r["group"] == "idle")
    check_len = int(m6["win"]["W"] or 64)                  # the check runs at the frame's window length (64 on the real run; the smoke's frame is shorter)
    if hop_of(check_len) != int(m6["win"]["H"] or hop_of(check_len)):
        raise Stop(EXIT_ERROR, f"stopped: move 6's window {m6['win']['grid_id']} has a hop that is not half its length, so the check against move 20's rows cannot be made on this grid")
    print(f"[heldout] length {L} (hop {hop_of(L)}): the held-out ladder on {len(runs)} recordings ({n_idle} idle) in {N_FOLDS} folds by rep, cuts {', '.join(f'{k} = {v}' for k, v in cuts.items())}; "
          f"rungs {', '.join(r.replace('rung', 'r') for r in rungs)}; levels {', '.join(LEVELS)}; forest {n_est} trees, {n_jobs} processes, seed {seed} (move 6's values unless overridden); "
          f"bootstrap {n_boot}; {NO_NULL[:50]}...")
    print(f"[heldout] run {run} (read only); move 20's series and ladder {al_out} (read only; series present for {len(runs) - len(P['missing_series'])} of {len(runs)}); out {out}"
          f"{' (exists)' if out.exists() else ' (to be created)'}; folds: " + "; ".join(f"{fo['fold']}: reps {fo['reps']} hold out {fo['n_heldout']} runs" for fo in folds))
    print(f"[heldout] driver lock: " + (f"held by pid {P['lock']['pid']} ({'alive' if P['lock']['alive'] else 'dead'})" if P["lock"]["exists"] else "absent")
          + ("; the refusal is skipped by --dry-run" if (P["lock"]["alive"] and o.dry_run) else "") + (f"; the check against move 20's rows runs at this length (the frame's window)" if L == check_len else ""))
    if o.dry_run:
        print(f"[heldout] dry run: would write {out}/heldout/cut<C>/len{L}/ (scores.csv, recall_per_kernel.csv, margins.csv, left_out.csv, folds.csv, <rung>/predictions_<level>.csv, "
              f"<rung>/result.json, features/<rung>.npz, summary.json), {out}/heldout/len{L}.json, the summary tables and figures and {out / RECORD_NAME}; nothing written")
        return EXIT_OK
    if P["missing_series"]:
        raise Stop(EXIT_MISSING, f"missing input: move 20's series for {len(P['missing_series'])} recording(s) (first: {P['missing_series'][0]}); run move 20's `series` step first")
    if L == check_len:
        absent = [f"cut{c}/{r}" for c in cuts.values() for r in rungs if not (al_out / "ladder" / f"cut{c}" / r / "features.npz").is_file()]
        if absent:
            raise Stop(EXIT_MISSING, f"missing input: move 20's feature rows for the check ({len(absent)}: {absent[:3]}); run move 20's `ladder` step first")
    out_lock = acquire_out_lock(out)
    code = code_identity()
    params = {"run": str(run), "out": str(out), "arrowladder_out": str(al_out), "cuts": cuts, "length": L, "hop": hop_of(L), "rungs": rungs, "levels": list(LEVELS), "n_folds": N_FOLDS,
              "fold_rule": FOLD_RULE, "bootstrap": n_boot, "seed_bootstrap": SEED_BOOTSTRAP, "n_estimators": n_est, "n_jobs": n_jobs, "seed_forest": seed, "seed_offset": m6["seed_offset"],
              "n_recordings": len(runs), "features_version": FEATURES_VERSION, "force": bool(o.force)}
    params_sig = {k: v for k, v in params.items() if k not in ("n_jobs", "force", "arrowladder_out", "rungs", "out")}
    entry = {"key": f"len{L}", "length": L, "command": sys.argv, "cwd": str(_HERE.parent), "started_at": now_iso(), "status": "running", "exit_code": None, "finished_at": None, "params": params,
             "inputs_sha256": {"cells.csv": sha256_file(run / "cells.csv"), "series": {r["cell_id"]: sha256_file(run / "series" / f"{r['cell_id']}.npz") for r in runs if (run / "series" / f"{r['cell_id']}.npz").is_file()},
                               "arrow_series": {r["cell_id"]: sha256_file(AL.series_path(al_out, r["cell_id"])) for r in runs}},
             "code": {"heldout_ladder.py": code["sha256"].get("heldout_ladder.py"), "fingerprint": code["fingerprint"], "modules": code["modules"], "sha256": code["sha256"],
                      "toolkit_fingerprint": toolkit_fingerprint()["sha256"]},
             "outputs": []}
    outputs_exist = (out / "heldout" / f"len{L}.json").is_file() and all((out / "heldout" / f"cut{c}" / f"len{L}" / "summary.json").is_file() for c in cuts.values())
    try:
        if _skip_if_unchanged(out, f"len{L}", entry, ("n_jobs", "force", "arrowladder_out"), outputs_exist, bool(o.force), f"{out}/heldout/len{L}"):
            return EXIT_OK
        t0 = time.time()
        series_by_cell = load_series(run, al_out, runs)
        results = [length_cut(out, al_out, name, cut, L, runs, series_by_cell, folds, rungs, n_boot=n_boot, n_jobs=n_jobs, n_est=n_est, seed=seed, force=bool(o.force), params_sig=params_sig,
                              check_len=check_len) for name, cut in cuts.items()]
        write_summary(out, cuts, rungs, quiet=True)
        step_rec = {"schema": "plan12.heldout_step.v1", "citation": CITATION, "package_version": __version__, "written_at": now_iso(), "command": sys.argv, "length": L, "params": params,
                    "question": __doc__.split("THE QUESTION.", 1)[1].split("THE SPLIT", 1)[0].strip(), "code": entry["code"], "elapsed_s": round(time.time() - t0, 1),
                    "check_length": check_len,
                    "time_per_cut_s": {s["cut_name"]: s["elapsed_s"] for s in results}, "check_move20": {s["cut_name"]: s["check_move20"] for s in results} if L == check_len else None,
                    "left_out": {s["cut_name"]: s["left_out"] for s in results}, "results": results}
        write_json(out / "heldout" / f"len{L}.json", step_rec)
        entry["outputs"] = sorted(str(p.relative_to(out)) for c in cuts.values() for p in (out / "heldout" / f"cut{c}" / f"len{L}").rglob("*") if p.is_file()) + [f"heldout/len{L}.json"]
        entry.update(status="done", exit_code=0, finished_at=now_iso(), elapsed_s=step_rec["elapsed_s"])
        per_cut = "; ".join(f"cut {s['cut']} {s['elapsed_s']} s" for s in results)
        print(f"[heldout] length {L} done in {step_rec['elapsed_s']} s ({per_cut}): {out / 'heldout'}")
        return EXIT_OK
    except BaseException as exc:                           # noqa: BLE001
        code_ = getattr(exc, "code", None)
        entry.update(status=f"failed: {type(exc).__name__}" + (f" (exit {code_})" if isinstance(code_, int) else ""), exit_code=code_ if isinstance(code_, int) else EXIT_ERROR,
                     finished_at=now_iso(), error=str(exc)[:500])
        raise
    finally:
        _finish(out, entry, out_lock)


def run_summary(o: argparse.Namespace) -> int:
    P = prepare(o, need_series=False)
    out, cuts = P["out"], P["cuts"]
    if o.dry_run:
        present = {c: [L for L in LENGTHS if (out / "heldout" / f"cut{c}" / f"len{L}" / "summary.json").is_file()] for c in cuts.values()}
        print(f"[heldout] summary dry run: lengths present {present}; would write {out}/heldout/cut<C>/scores_all_lengths.csv, margins_all_lengths.csv, left_out_all_lengths.csv, "
              f"accuracy_vs_length_<level>.svg/.csv and {out}/heldout/summary.json; nothing written")
        return EXIT_OK
    out_lock = acquire_out_lock(out)
    entry = {"key": KEY_SUMMARY, "command": sys.argv, "cwd": str(_HERE.parent), "started_at": now_iso(), "status": "running", "exit_code": None, "finished_at": None,
             "params": {"run": str(P["run"]), "out": str(out)}, "code": code_identity(), "outputs": ["heldout/summary.json"]}
    rec = load_record(out)
    rec["entries"].append(entry); save_record(out, rec)
    try:
        present = write_summary(out, cuts, list(RUNG_ORDER))
        if not present:
            raise Stop(EXIT_MISSING, f"nothing to summarise: no length has run in {out}")
        entry.update(status="done", exit_code=0, finished_at=now_iso(), lengths_present={str(c): v for c, v in present.items()})
        return EXIT_OK
    except BaseException as exc:                           # noqa: BLE001
        code_ = getattr(exc, "code", None)
        entry.update(status=f"failed: {type(exc).__name__}" + (f" (exit {code_})" if isinstance(code_, int) else ""), exit_code=code_ if isinstance(code_, int) else EXIT_ERROR,
                     finished_at=now_iso(), error=str(exc)[:500])
        raise
    finally:
        _finish(out, entry, out_lock)


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------
def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run", required=True, help="the main run's output folder (read only here)")
    p.add_argument("--out", default=None, help="the output folder (default: a sibling of the run, <run>_heldout; never inside the run)")
    p.add_argument("--arrowladder-out", default=None, help="where move 20 wrote its series and ladder (default <run>_arrowladder; read only)")
    p.add_argument("--n-jobs", type=int, default=None, help="processes for the forest (default: move 6's recorded value)")
    p.add_argument("--n-estimators", type=int, default=None, help="trees (default: move 6's recorded value)")
    p.add_argument("--force", action="store_true", help="re-run a finished step")
    p.add_argument("--dry-run", action="store_true", help="print the plan; write nothing; skip the live-lock refusals")


def main(argv: list[str] | None = None) -> int:
    install_sigterm()
    ap = argparse.ArgumentParser(prog="plan12_grounding.heldout_ladder", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="one piece length: the four folds, every rung, both levels, both cuts (resumable per cut and rung)")
    _common(r)
    r.add_argument("--length", required=True, type=int, choices=list(LENGTHS), help="the piece length in pairs: 64, 32, 16, 8, 4 or 2")
    r.add_argument("--cuts", default=None, help="which cuts, by name: declared,measured (default both; the values come from params.json)")
    r.add_argument("--rungs", default=None, help="the rungs to run (default all seven)")
    r.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP, help="move 12's bootstrap resamples for the margins (default 1000)")
    r.add_argument("--encoding-out", default=None, help="decision D2 (default: the encoding run move 6 recorded in classify.json)")
    s = sub.add_parser("summary", help="the tables and figures across the lengths present, per cut")
    _common(s)
    s.add_argument("--cuts", default=None)
    o = ap.parse_args(argv)
    try:
        return run_summary(o) if o.cmd == "summary" else run_length(o)
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
