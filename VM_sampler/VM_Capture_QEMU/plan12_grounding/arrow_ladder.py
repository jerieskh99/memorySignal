#!/usr/bin/env python3
"""arrow_ladder.py -- the arrow ladder beside the run of plan12_grounding (2026-10-13; move 20 in the console's
Grounding paper tab, after the window-size sweep (19)): move 12's new-block test repeated on a ladder of input sets
built from the arrow's size and angle, each rung an input set on its own, without the encoding readings.

  python3 -m plan12_grounding.arrow_ladder series --run <main run out> [--out <dir>] [--only cellA,cellB] [--force] [--dry-run]
  python3 -m plan12_grounding.arrow_ladder ladder --run <main run out> [--out <dir>] [--cuts declared,measured] [--rungs rung1,...]
        [--seen 0.8] [--gap 1] [--pools 1,2,3] [--bootstrap 1000] [--encoding-out <D2>] [--n-jobs N] [--n-estimators N]
        [--newblocks-out <dir>] [--force] [--dry-run]
  (run from VM_sampler/VM_Capture_QEMU/)

THE QUESTION. Move 12 names new blocks from E_new, the statistics of three per-pair series: N (changed pages), H (bits
flipped) and A (the bit-weighted mean angle). What does each part of the arrow buy? The ladder keeps everything of
move 12 and changes only the input set, one rung at a time:
  rung 0   N, H and A: E_new's own series, the check (it must reproduce move 12's E_new scores exactly)
  rung 1   bits alone: H
  rung 2   bits and angle, as two separate series: H and A (E_new without N)
  rung 3   the complex form, bits and the plain angle: C = sum(h cos theta) and S = sum(h sin theta), move 1's stored series
  rung 4   the complex form, bits and the doubled angle: C2 = sum(h cos 2 theta) and S2 = sum(h sin 2 theta)
  rung 5a  the complex form, bytes changed and the plain angle: L0C = sum(l0 cos theta) and L0S = sum(l0 sin theta)
  rung 5b  the complex form, the L1 distance and the plain angle: L1C = sum(l1 cos theta) and L1S = sum(l1 sin theta)
theta = arccos(clip(1 - d, 0, 1)) as in extract.py; every series is a per-pair sum over the changed pages (the store's
rows with h > 0), on move 1's pair axis (the keep-first rule, the pair axis of extract.build_series).

THE TWO STEPS
  series   one pass over the per-page stores (<run>/moves/01_extract/extract.json names them; read only): per recording
           the per-pair series of every rung, written to <out>/series/<cell>.npz (resumable: a recording whose file
           stands with the same store sha256 and series version is kept). CHECK 1, every recording: the pass must
           reproduce move 1's stored N, H, A, C and S (series/<cell>.npz) to float precision; a failure stops the step.
  ladder   move 12's test on every rung, both cuts: the window frame of move 12 (idle_class.build_encodings_13 with
           move 6's window, the same cells, windows and labels), each rung's window features = the statistic sets
           classify.py applies to N, H and A for E_new (stats.series_stats: B1's set, the deep set, the November
           magnitude set; the phase set for an angle series, which only A is), applied to the rung's series over the
           same windows; then exactly new_blocks.py: the 80/20 split in time order with its gap, idle as a class, the
           three levels, single blocks and pools, move 6's forest settings, the dimension rule, the B1-G3 quarantine.
           CHECK 2: rung 0's feature rows must equal move 12's E_new rows bit for bit and its scores must equal move
           12's E_new scores (read from <run>_newblocks, not recomputed) at both cuts; a failure stops the step.
           Resumable per cut and rung (a rung whose result.json stands with the same parameters is kept).

THE REPORT, per cut (<out>/ladder/cut<C>/)
  scores.csv             per level, rung and pool size, as move 12 reports them, with move 12's E0 and E_new rows beside
                         (read from <run>_newblocks/cut<C>/scores.csv; "missing" when move 12 has not run)
  margins.csv            each rung minus the rung before it (rung 2 - rung 1, 3 - 2, 4 - 3, 5a - 3, 5b - 3; and 2 - 0, the
                         cost of N) on the same pools, with move 12's bootstrap over recordings (new_blocks.bootstrap_margin,
                         SEED_BOOTSTRAP)
  recall_per_kernel.csv, run_level.csv, by_position.csv   as move 12's, per rung
  <rung>/predictions_<level>.csv, <rung>/pools_<level>.csv, <rung>/confusion_kernel.csv (.svg: single blocks, the kernel
                         level), <rung>/features.npz (the rung's rows and names), <rung>/result.json
  ladder_accuracy.svg    the summary figure: the rungs side by side per level (single blocks as bars, pools of 2 and 3
                         as ticks), E0 and E_new as reference lines, chance and the majority class
  summary.json           the cut's record, the two checks included
  <out>/ladder/arrow_ladder.json, <out>/series/series.json, <out>/record.json (one entry per step run: command, inputs
  with sha256, params, the code fingerprint of this module's closure, status; a finished step is skipped unless --force or
  an input, a parameter or the code changed)

THE RULES THAT PROTECT THE LIVE RUN (as new_blocks.py and page_fourier.py)
- ONE new file; nothing existing in plan12_grounding/ is edited; new_blocks.py, classify.py, idle_class.py, stats.py and
  extract.py are imported as they are (their recorded code fingerprints stay); nothing imports this module. The run
  folder, every other sibling folder and the L1 stores are read only here; --out inside the run is refused (exit 2). A
  step refuses to start (exit 3) while <run>/.driver.lock belongs to a live process; --dry-run skips that refusal and
  writes nothing. One writer per output folder: <out>/.arrow_ladder.lock; run_moves.install_sigterm.
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
from plan12_grounding import idle_class as I  # noqa: E402  (build_encodings_13, the E0 gate and check, the lock state; imported, never edited)
from plan12_grounding import new_blocks as NB  # noqa: E402  (move 12: split_rows, fit_level, pool_votes, score_pools, run_level_shares, by_position, bootstrap_margin, confusion_svg)
from plan12_grounding import stats as ST  # noqa: E402  (series_stats, cut_series, load_runs, cuts_of: the statistic sets and the cut, as move 6 applies them)
from plan12_grounding.run_moves import _pid_alive, code_fingerprint, install_sigterm, now_iso, read_json, sha256_file, write_json  # noqa: E402
from plan12_grounding.figures import svg_open, write_csv  # noqa: E402
from plan10_analysis.runner import extract as L1  # noqa: E402  (the console's L1 store loader; read only)
from plan11_encoding_ladder.nulls import SEED_FOREST  # noqa: E402

CITATION = ("plan12_grounding/new_blocks.py (move 12: the split, the forest, the scores, the bootstrap; its functions are called here on each rung's rows); "
            "plan12_grounding/classify.py build_encodings and stats.py series_stats / window_stats (E_new's statistic sets, applied here to each rung's series); "
            "plan12_grounding/extract.py build_series (theta = arccos(clip(1 - d, 0, 1)), the keep-first rule, the pair axis, the rows with h > 0; the store pass "
            "repeats it and adds the doubled angle and the l0 and l1 arrows); the author's design of the ladder (2026-10-13)")
SERIES_VERSION = "plan12.arrow_series.v1"
OUT_SUFFIX = "_arrowladder"
LOCK_NAME = ".arrow_ladder.lock"
RECORD_NAME = "record.json"
KEY_SERIES, KEY_LADDER = "series", "ladder"
RUNGS = (("rung0", ("N", "H", "A"), "N, H, A: E_new's own series (the check against move 12)"),
         ("rung1", ("H",), "bits alone: H"),
         ("rung2", ("H", "A"), "bits and angle, as two separate series: H and A (E_new without N)"),
         ("rung3", ("C", "S"), "the complex form, bits and the plain angle: C = sum(h cos theta), S = sum(h sin theta) (move 1's stored series)"),
         ("rung4", ("C2", "S2"), "the complex form, bits and the doubled angle: C2 = sum(h cos 2 theta), S2 = sum(h sin 2 theta)"),
         ("rung5a", ("L0C", "L0S"), "the complex form, bytes changed and the plain angle: L0C = sum(l0 cos theta), L0S = sum(l0 sin theta)"),
         ("rung5b", ("L1C", "L1S"), "the complex form, the L1 distance and the plain angle: L1C = sum(l1 cos theta), L1S = sum(l1 sin theta)"))
RUNG_TEXT = {r: t for r, _, t in RUNGS}
RUNG_KEYS = {r: k for r, k, _ in RUNGS}
RUNG_ORDER = [r for r, _, _ in RUNGS]
LADDER_PAIRS = (("rung2", "rung1"), ("rung3", "rung2"), ("rung4", "rung3"), ("rung5a", "rung3"), ("rung5b", "rung3"), ("rung2", "rung0"))
STORED = ("N", "H", "A", "C", "S")                         # read from move 1's series/<cell>.npz (rungs 0 to 3); the rest from this move's store pass
LEVELS = NB.LEVELS
IDLE_LABEL = NB.IDLE_LABEL
HIGHLIGHT = NB.HIGHLIGHT
SEED_BOOTSTRAP = NB.SEED_BOOTSTRAP
NO_NULL = NB.NO_NULL
DEFAULT_SEEN, DEFAULT_GAP, DEFAULT_POOLS, DEFAULT_BOOTSTRAP = NB.DEFAULT_SEEN, NB.DEFAULT_GAP, NB.DEFAULT_POOLS, NB.DEFAULT_BOOTSTRAP
RTOL_SERIES = 1e-9                                         # check 1: the pass against move 1's stored series (the same float64 arithmetic on the same rows)
EXIT_OK, EXIT_ERROR, EXIT_MISSING, EXIT_REFUSED = I.EXIT_OK, I.EXIT_ERROR, I.EXIT_MISSING, I.EXIT_REFUSED
RUNG_COLOUR = {"rung0": "#8a8a8a", "rung1": "#b8b8b8", "rung2": "#3a9d5d", "rung3": "#2b5d8a", "rung4": "#7b52ab", "rung5a": "#d9822b", "rung5b": "#b03a2e"}
Stop = I.Stop


def fmt(x) -> str:
    return I.fmt(x)


# ---------------------------------------------------------------------------------------------
# the store pass: every rung's per-pair series from one per-page store (extract.build_series repeated, the arrows added)
# ---------------------------------------------------------------------------------------------
def store_series(rec: dict) -> dict:
    """The per-pair series on move 1's pair axis: the keep-first rule (rec['keep_first_pairs']), the pair axis = every pair
    of the kept range, the rows with h > 0, theta = arccos(clip(1 - d, 0, 1)); N, H, A, C, S as build_series makes them,
    then C2, S2 (the doubled angle), L0C, L0S, L1C, L1S (the l0 and l1 arrows) and the plain sums L0, L1."""
    z = L1.load(Path(rec["store"]))
    seq = np.asarray(z["seq"]).astype(np.int64)
    h = np.asarray(z["hamming"]).astype(np.float64)
    d = np.asarray(z["cosine"]).astype(np.float64)
    l0 = np.asarray(z["l0"]).astype(np.float64)
    l1 = np.asarray(z["l1"]).astype(np.float64)
    keep = np.ones(seq.size, dtype=bool)
    kf = rec.get("keep_first_pairs")
    kf = int(kf) if kf not in (None, "", "None") else None
    if kf and seq.size:
        keep &= seq <= int(seq.min()) + kf - 1
    pairs = np.unique(seq[keep])
    n = int(pairs.size)
    m = keep & (h > 0)
    seq_k, h_k, d_k, l0_k, l1_k = seq[m], h[m], d[m], l0[m], l1[m]
    theta = np.arccos(np.clip(1.0 - d_k, 0.0, 1.0))
    inv = np.searchsorted(pairs, seq_k)
    N = np.bincount(inv, minlength=n).astype(np.int64)
    H = np.bincount(inv, weights=h_k, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        A = np.where(H > 0, np.bincount(inv, weights=h_k * theta, minlength=n) / np.where(H > 0, H, 1.0), np.nan)
    arrays = {"pair": pairs.astype(np.int64), "N": N, "H": H, "A": A,
              "C": np.bincount(inv, weights=h_k * np.cos(theta), minlength=n), "S": np.bincount(inv, weights=h_k * np.sin(theta), minlength=n),
              "C2": np.bincount(inv, weights=h_k * np.cos(2.0 * theta), minlength=n), "S2": np.bincount(inv, weights=h_k * np.sin(2.0 * theta), minlength=n),
              "L0C": np.bincount(inv, weights=l0_k * np.cos(theta), minlength=n), "L0S": np.bincount(inv, weights=l0_k * np.sin(theta), minlength=n),
              "L1C": np.bincount(inv, weights=l1_k * np.cos(theta), minlength=n), "L1S": np.bincount(inv, weights=l1_k * np.sin(theta), minlength=n),
              "L0": np.bincount(inv, weights=l0_k, minlength=n), "L1": np.bincount(inv, weights=l1_k, minlength=n)}
    return {"arrays": arrays, "n_rows_file": int(seq.size), "n_rows_kept": int(m.sum()), "keep_first_pairs": kf, "n_pairs": n}


def check_stored(run_rec: dict, mine: dict) -> dict:
    """Check 1: the pass against move 1's stored series (the run's series/<cell>.npz): the pair axis equal, N equal, and H, A,
    C, S within RTOL_SERIES (relative to max(1, |stored|)); A compared where both are finite, and undefined in the same pairs."""
    a = run_rec["arrays"]
    res = {"cell_id": run_rec["cell_id"], "ok": True, "n_pairs_stored": int(a["pair"].size), "n_pairs_pass": int(mine["pair"].size), "max_rel_dev": 0.0, "first_difference": None}
    if int(a["pair"].size) != int(mine["pair"].size) or not np.array_equal(a["pair"], mine["pair"]):
        res.update(ok=False, first_difference=f"the pair axis differs: stored {int(a['pair'].size)} pairs, the pass {int(mine['pair'].size)}")
        return res
    if not np.array_equal(a["N"].astype(np.int64), mine["N"].astype(np.int64)):
        j = int(np.flatnonzero(a["N"].astype(np.int64) != mine["N"].astype(np.int64))[0])
        res.update(ok=False, first_difference=f"N at pair {int(a['pair'][j])}: stored {int(a['N'][j])}, the pass {int(mine['N'][j])}")
        return res
    for key in ("H", "C", "S", "A"):
        x, y = np.asarray(a[key], dtype=np.float64), np.asarray(mine[key], dtype=np.float64)
        if key == "A":
            fx, fy = np.isfinite(x), np.isfinite(y)
            if not np.array_equal(fx, fy):
                j = int(np.flatnonzero(fx != fy)[0])
                res.update(ok=False, first_difference=f"A defined in different pairs (first at pair {int(a['pair'][j])})")
                return res
            x, y = x[fx], y[fy]
        dev = np.abs(x - y) / np.maximum(1.0, np.abs(x))
        res["max_rel_dev"] = max(res["max_rel_dev"], float(dev.max()) if dev.size else 0.0)
        if dev.size and dev.max() > RTOL_SERIES:
            j = int(np.argmax(dev))
            res.update(ok=False, first_difference=f"{key} at index {j}: stored {x[j]!r}, the pass {y[j]!r}")
            return res
    return res


# ---------------------------------------------------------------------------------------------
# the output folder, the locks, the record (page_fourier's pattern)
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
            raise Stop(EXIT_REFUSED, f"refused: another arrow_ladder step is writing {out} ({lock}: pid {pid}, started {old.get('started_at')}); one writer per output folder")
        print(f"[ladder] stale lock from pid {pid} ({old.get('started_at')}) taken over", flush=True)
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
    return {"schema": "plan12.arrow_ladder.record.v1", "citation": CITATION, "package_version": __version__, "entries": []}


def save_record(out: Path, rec: dict) -> None:
    write_json(out / RECORD_NAME, rec)


def last_done(rec: dict, key: str) -> dict | None:
    for e in reversed(rec.get("entries") or []):
        if e.get("key") == key and e.get("status") == "done":
            return e
    return None


def code_identity() -> dict:
    fp = code_fingerprint("arrow_ladder")
    files = toolkit_fingerprint()["files"]
    return {"fingerprint": fp["sha256"], "modules": fp["modules"], "sha256": {f"{m}.py": files.get(f"{m}.py") for m in fp["modules"]}}


def refuse_if_driver_live(run: Path, dry_run: bool) -> dict:
    lock = I.driver_lock_state(run)
    if lock["alive"] and not dry_run:
        raise Stop(EXIT_REFUSED, f"refused: a move is running on this run ({lock['path']}: pid {lock['pid']}, started {lock['started_at']}, {lock['argv_tail']}); "
                                 "this move never competes with a running move. Run again after it ends; --dry-run shows the plan meanwhile.")
    return lock


def _finish(out: Path, entry: dict, out_lock: Path) -> None:
    if entry["status"] == "running":
        entry.update(status="failed: stopped before the end", finished_at=now_iso())
    rec = load_record(out)
    rec["entries"] = [e for e in rec["entries"] if not (e.get("key") == entry["key"] and e.get("started_at") == entry["started_at"])] + [entry]
    save_record(out, rec)
    release_out_lock(out_lock)


def _skip_if_unchanged(out: Path, key: str, entry: dict, sig_skip: tuple, outputs_exist: bool, force: bool, tag: str) -> bool:
    """The skip rule of the moves beside the run: a finished step with unchanged inputs, parameters and code stands."""
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
            print(f"[ladder] skipped: {tag} stands as made on {prev.get('finished_at')} (inputs, parameters and code unchanged; --force re-runs)")
            return True
        print(f"[ladder] re-running: {why} since {prev.get('finished_at')}")
    rec["entries"].append(entry)
    save_record(out, rec)
    return False


# ---------------------------------------------------------------------------------------------
# step 1: `series`
# ---------------------------------------------------------------------------------------------
def series_path(out: Path, cell_id: str) -> Path:
    return out / "series" / f"{cell_id}.npz"


def series_stands(out: Path, cell_id: str, store_sha: str | None) -> bool:
    p = series_path(out, cell_id)
    if not p.is_file():
        return False
    try:
        with np.load(p, allow_pickle=False) as z:
            return str(z["series_version"]) == SERIES_VERSION and (store_sha is None or str(z["store_sha256"]) == str(store_sha)) and bool(z["check_ok"])
    except Exception:                                      # noqa: BLE001
        return False


def load_rung_series(out: Path, cell_id: str) -> dict:
    with np.load(series_path(out, cell_id), allow_pickle=False) as z:
        return {k: z[k] for k in ("pair", "N", "H", "A", "C", "S", "C2", "S2", "L0C", "L0S", "L1C", "L1S", "L0", "L1")}


def run_series(o: argparse.Namespace) -> int:
    run = Path(os.path.expanduser(o.run)).resolve()
    if not (run / "cells.csv").is_file():
        raise Stop(EXIT_MISSING, f"missing input: {run / 'cells.csv'} (--run must name a plan12_grounding output folder with moves 0 and 1 done)")
    out = resolve_out(run, o.out)
    lock = refuse_if_driver_live(run, o.dry_run)
    exj = read_json(run / "moves" / "01_extract" / "extract.json")
    ex_recs = exj.get("recordings") or {}
    runs = ST.load_runs(run)
    only = [s.strip() for s in str(o.only).split(",") if s.strip()] if o.only else None
    if only:
        missing = set(only) - {r["cell_id"] for r in runs}
        if missing:
            raise Stop(EXIT_MISSING, f"--only names cells without a complete series in this run: {sorted(missing)}")
        runs = [r for r in runs if r["cell_id"] in set(only)]
    missing_store = [r["cell_id"] for r in runs if not Path((ex_recs.get(r["cell_id"]) or {}).get("store") or "").is_file()]
    if missing_store:
        raise Stop(EXIT_MISSING, f"missing input: the L1 store of {len(missing_store)} recording(s) is not on this machine (first: {missing_store[0]}; extract.json names it)")
    stands = {r["cell_id"]: series_stands(out, r["cell_id"], (ex_recs[r["cell_id"]] or {}).get("store_sha256")) for r in runs}
    print(f"[ladder] series: one pass over the per-page stores of {len(runs)} recordings ({sum(1 for r in runs if r['role'] == 'idle')} idle); per recording N, H, A, C, S "
          f"(checked against move 1's series to {RTOL_SERIES:g}), C2, S2 (the doubled angle), L0C, L0S, L1C, L1S (the l0 and l1 arrows); {sum(stands.values())} stand from an earlier run"
          + (" (--force recomputes)" if o.force else ""))
    print(f"[ladder] run {run} (read only); stores {Path(ex_recs[runs[0]['cell_id']]['store']).parent} (read only); out {out}{' (exists)' if out.exists() else ' (to be created)'}")
    print(f"[ladder] driver lock: " + (f"held by pid {lock['pid']} ({'alive' if lock['alive'] else 'dead'}), started {lock['started_at']}: {lock['argv_tail']}" if lock["exists"] else "absent")
          + ("; the refusal is skipped by --dry-run" if (lock["alive"] and o.dry_run) else ""))
    if o.dry_run:
        print(f"[ladder] dry run: would write {out}/series/<cell>.npz, series/identity_check.csv, series/series.json and {out / RECORD_NAME}; nothing written")
        return EXIT_OK
    out_lock = acquire_out_lock(out)
    code = code_identity()
    entry = {"key": KEY_SERIES, "command": sys.argv, "cwd": str(_HERE.parent), "started_at": now_iso(), "status": "running", "exit_code": None, "finished_at": None,
             "params": {"run": str(run), "out": str(out), "only": only, "series_version": SERIES_VERSION, "rtol": RTOL_SERIES, "n_recordings": len(runs), "force": bool(o.force)},
             "inputs_sha256": {"extract.json": sha256_file(run / "moves" / "01_extract" / "extract.json"),
                               "stores": {r["cell_id"]: (ex_recs[r["cell_id"]].get("store_sha256") or sha256_file(ex_recs[r["cell_id"]]["store"])) for r in runs},
                               "series": {r["cell_id"]: sha256_file(ex_recs[r["cell_id"]].get("series") or (run / "series" / f"{r['cell_id']}.npz")) for r in runs}},
             "code": {"arrow_ladder.py": code["sha256"].get("arrow_ladder.py"), "fingerprint": code["fingerprint"], "modules": code["modules"], "sha256": code["sha256"],
                      "toolkit_fingerprint": toolkit_fingerprint()["sha256"]},
             "outputs": []}
    outputs_exist = (out / "series" / "series.json").is_file() and all(stands.values())
    try:
        if _skip_if_unchanged(out, KEY_SERIES, entry, ("force",), outputs_exist, bool(o.force), f"{out}/series"):
            return EXIT_OK
        (out / "series").mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        rows, n_done, n_kept = [], 0, 0
        for r in runs:
            cid = r["cell_id"]
            rec = ex_recs[cid]
            t1 = time.time()
            if stands[cid] and not o.force:
                n_kept += 1
                rows.append([cid, r["group"], None, None, True, None, 0.0, "kept from an earlier run"])
                continue
            s = store_series({**rec, "cell_id": cid})
            chk = check_stored(r, s["arrays"])
            rows.append([cid, r["group"], s["n_pairs"], s["n_rows_kept"], chk["ok"], chk["max_rel_dev"], round(time.time() - t1, 1), chk["first_difference"] or "ok"])
            if not chk["ok"]:
                write_csv(out / "series" / "identity_check.csv", ["cell_id", "group", "n_pairs", "n_rows_changed", "check_ok", "max_rel_dev", "time_s", "note"], rows)
                raise Stop(EXIT_ERROR, f"stopped: the store pass does not reproduce move 1's series for {cid}: {chk['first_difference']} (largest relative deviation "
                                       f"{chk['max_rel_dev']:.3g}); nothing else written")
            np.savez_compressed(series_path(out, cid), **{k: v for k, v in s["arrays"].items()}, cell_id=np.array(cid), series_version=np.array(SERIES_VERSION),
                                store_sha256=np.array(str(rec.get("store_sha256") or "")), keep_first_pairs=np.int64(s["keep_first_pairs"] or 0),
                                n_rows_file=np.int64(s["n_rows_file"]), n_rows_changed=np.int64(s["n_rows_kept"]), check_ok=np.bool_(chk["ok"]), check_max_rel_dev=np.float64(chk["max_rel_dev"]))
            n_done += 1
            print(f"[ladder] series {cid:<32} {s['n_pairs']} pairs, {s['n_rows_kept']} changed-page rows; check against move 1 ok (max rel dev {chk['max_rel_dev']:.2e}); {time.time() - t1:.1f} s", flush=True)
        write_csv(out / "series" / "identity_check.csv", ["cell_id", "group", "n_pairs", "n_rows_changed", "check_ok", "max_rel_dev", "time_s", "note"], rows)
        times = [x[6] for x in rows if x[7] == "ok"]
        sj = {"schema": "plan12.arrow_series.v1", "citation": CITATION, "package_version": __version__, "written_at": now_iso(), "command": sys.argv, "series_version": SERIES_VERSION,
              "series": {"N": "changed pages (rows with h > 0)", "H": "sum h", "A": "sum(h theta) / sum h", "C": "sum(h cos theta)", "S": "sum(h sin theta)",
                         "C2": "sum(h cos 2 theta)", "S2": "sum(h sin 2 theta)", "L0C": "sum(l0 cos theta)", "L0S": "sum(l0 sin theta)", "L1C": "sum(l1 cos theta)",
                         "L1S": "sum(l1 sin theta)", "L0": "sum l0", "L1": "sum l1", "theta": "arccos(clip(1 - d, 0, 1))", "pair_axis": "move 1's (extract.build_series)"},
              "check": {"rule": f"N equal; H, A, C, S within {RTOL_SERIES:g} of move 1's stored series relative to max(1, |stored|); A undefined in the same pairs",
                        "all_ok": all(x[4] for x in rows), "max_rel_dev": max((x[5] for x in rows if x[5] is not None), default=0.0)},
              "n_recordings": len(runs), "n_computed": n_done, "n_kept": n_kept, "time_per_recording_s": {"mean": float(np.mean(times)) if times else None, "max": max(times, default=None)},
              "elapsed_s": round(time.time() - t0, 1), "code": entry["code"]}
        write_json(out / "series" / "series.json", sj)
        entry["outputs"] = sorted(str(p.relative_to(out)) for p in (out / "series").rglob("*") if p.is_file())
        entry.update(status="done", exit_code=0, finished_at=now_iso(), elapsed_s=sj["elapsed_s"])
        print(f"[ladder] series done in {sj['elapsed_s']} s: {n_done} computed, {n_kept} kept; the check against move 1 passed on every computed recording (max rel dev "
              f"{sj['check']['max_rel_dev']:.2e}); {out / 'series'}")
        return EXIT_OK
    except BaseException as exc:                           # noqa: BLE001
        code_ = getattr(exc, "code", None)
        entry.update(status=f"failed: {type(exc).__name__}" + (f" (exit {code_})" if isinstance(code_, int) else ""), exit_code=code_ if isinstance(code_, int) else EXIT_ERROR,
                     finished_at=now_iso(), error=str(exc)[:500])
        raise
    finally:
        _finish(out, entry, out_lock)


# ---------------------------------------------------------------------------------------------
# step 2: `ladder`: the rung's window features on move 12's window frame, then move 12's test
# ---------------------------------------------------------------------------------------------
def rung_feature_names(keys: tuple) -> list[str]:
    names = []
    for s in keys:
        names += [f"{s}.b1.{k}" for k in ST.B1_FEAT] + [f"{s}.deep.{k}" for k in ST.DEEP_STATS]
        names += [f"{s}.nov.{k}" for k in (ST.NOV_PHASE_STATS if s == "A" else ST.NOV_MAG_STATS)]
    return names


def rung_rows(lab: dict, series_by_cell: dict, cut: int, W: int, keys: tuple) -> tuple[np.ndarray, list[str]]:
    """The rung's feature rows on move 12's window frame: for every row of `lab` (a cell and a window start, a row offset
    into the cut series), stats.series_stats on the rung's series over that window (the statistic sets of E_new, the phase
    set for A), in the order of rung_feature_names."""
    names = rung_feature_names(keys)
    X = np.full((int(lab["n"]), len(names)), np.nan)
    cells = list(dict.fromkeys(lab["cell_id"].tolist()))
    for c in cells:
        a = ST.cut_series({"arrays": series_by_cell[c]}, cut)
        rows = np.flatnonzero(lab["cell_id"] == c)
        n = int(a["pair"].size)
        for i in rows.tolist():
            t = int(lab["win_start"][i])
            if t + W > n:
                raise Stop(EXIT_ERROR, f"stopped: window at row offset {t} of {c} runs past the cut series ({n} rows, W {W}); the frame and the series disagree")
            st = {}
            for s in keys:
                st.update(ST.series_stats(s, a[s][t:t + W].astype(np.float64)))
            X[i] = [st[k] for k in names]
    return X, names


def move12_scores(nb_out: Path, cut: int) -> dict:
    p = nb_out / f"cut{cut}" / "scores.csv"
    if not p.is_file():
        return {"status": f"missing (move 12 has not run: {p})", "rows": {}}
    with open(p, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {"status": "ok", "rows": {(r["level"], r["feature_set"], int(r["pool_size"])): r for r in rows}}


def _f(x):
    try:
        return float(x) if x not in (None, "") else None
    except (TypeError, ValueError):
        return None


def ladder_svg(scores: dict, refs: dict, pools: list[int], cut: int, rungs: list[str]) -> str:
    """Three panels (the levels): the rungs side by side, the single-block accuracy as a bar, the pools of 2 and 3 as ticks,
    E0 and E_new (move 12, single blocks) as dashed reference lines, chance grey and the majority class dotted red."""
    W_, H_ = 1000, 340
    out = svg_open(W_, H_, f"The arrow ladder: new blocks named right, by level and rung, cut of {cut} pairs",
                   "bars: single new blocks; ticks: pools of 2 and 3; dashed lines: move 12's E0 (grey) and E_new (green) on single blocks; grey line: chance 1/K; dotted red: the majority class")
    x0, y0, w, h = 60, 60, 292, 200
    for li, level in enumerate(LEVELS):
        px = x0 + li * (w + 22)
        out.append(f'<text x="{px}" y="{y0 - 6}" font-weight="bold">{level}</text>')
        out.append(f'<rect x="{px}" y="{y0}" width="{w}" height="{h}" fill="none" stroke="#ccc"/>')
        for t in (0.0, 0.5, 1.0):
            yy = y0 + h - h * t
            out.append(f'<line x1="{px}" y1="{yy:.1f}" x2="{px + w}" y2="{yy:.1f}" stroke="#eee"/><text x="{px - 4}" y="{yy + 3:.1f}" text-anchor="end">{t:.1f}</text>')
        bw = w / len(rungs)
        for ri, rung in enumerate(rungs):
            bx = px + ri * bw + 6
            s1 = ((scores.get(level) or {}).get(rung) or {}).get(1) or {}
            acc = s1.get("accuracy")
            if acc is None:
                out.append(f'<text x="{bx + (bw - 12) / 2:.1f}" y="{y0 + h / 2:.1f}" text-anchor="middle" fill="#999" font-size="8">not run</text>')
            else:
                out.append(f'<rect x="{bx:.1f}" y="{y0 + h - h * acc:.1f}" width="{bw - 12:.1f}" height="{h * acc:.1f}" fill="{RUNG_COLOUR.get(rung, "#333")}" fill-opacity="0.85"/>')
                out.append(f'<text x="{bx + (bw - 12) / 2:.1f}" y="{y0 + h - h * acc - 3:.1f}" text-anchor="middle" font-size="8">{acc:.2f}</text>')
                for k in pools:
                    if k == 1:
                        continue
                    sk = ((scores.get(level) or {}).get(rung) or {}).get(k) or {}
                    if sk.get("accuracy") is not None:
                        yy = y0 + h - h * sk["accuracy"]
                        out.append(f'<line x1="{bx - 2:.1f}" y1="{yy:.1f}" x2="{bx + bw - 10:.1f}" y2="{yy:.1f}" stroke="#222" stroke-width="1"/><text x="{bx + bw - 9:.1f}" y="{yy + 3:.1f}" font-size="7">{k}</text>')
            if s1.get("chance") is not None:
                cy = y0 + h - h * float(s1["chance"])
                out.append(f'<line x1="{bx - 3:.1f}" y1="{cy:.1f}" x2="{bx + bw - 9:.1f}" y2="{cy:.1f}" stroke="#888"/>')
            if s1.get("majority") is not None:
                my = y0 + h - h * float(s1["majority"])
                out.append(f'<line x1="{bx - 3:.1f}" y1="{my:.1f}" x2="{bx + bw - 9:.1f}" y2="{my:.1f}" stroke="#b03a2e" stroke-dasharray="1,2"/>')
            out.append(f'<text x="{bx + (bw - 12) / 2:.1f}" y="{y0 + h + 12}" text-anchor="middle" font-size="8">{rung.replace("rung", "r")}</text>')
        kx = px + w                                        # the key of the reference lines, under the panel on the right, where no bar label sits
        for enc, colour in (("E_new", "#3a9d5d"), ("E0", "#8a8a8a")):
            v = _f(((refs.get(level) or {}).get(enc) or {}).get("accuracy"))
            if v is not None:
                yy = y0 + h - h * v
                out.append(f'<line x1="{px}" y1="{yy:.1f}" x2="{px + w}" y2="{yy:.1f}" stroke="{colour}" stroke-width="1.2" stroke-dasharray="5,3"/>')
                label = f"move 12 {enc} {v:.2f}"
            else:
                label = f"move 12 {enc}: missing"
            tw = 5 * len(label) + 4
            out.append(f'<text x="{kx}" y="{y0 + h + 39}" text-anchor="end" font-size="8" fill="{colour}">{label}</text>'
                       f'<line x1="{kx - tw - 14}" y1="{y0 + h + 36}" x2="{kx - tw - 2}" y2="{y0 + h + 36}" stroke="{colour}" stroke-width="1.2" stroke-dasharray="5,3"/>')
            kx -= tw + 22
        any_s = next((v for r in rungs for v in [((scores.get(level) or {}).get(r) or {}).get(1)] if v), {})
        out.append(f'<text x="{px}" y="{y0 + h + 26}" fill="#555" font-size="9">{any_s.get("n_classes", "?")} classes, {any_s.get("n_pools", "?")} new blocks</text>')
    out.append(f'<text x="{x0}" y="{H_ - 24}" fill="#555" font-size="9">r0 N,H,A (E_new, the check); r1 H; r2 H,A; r3 C,S; r4 C2,S2 (doubled angle); r5a L0C,L0S (bytes); r5b L1C,L1S (L1)</text>')
    out.append(f'<text x="{x0}" y="{H_ - 10}" fill="#555" font-size="9">{NO_NULL[:150]}</text></svg>')
    return "\n".join(out)


def one_rung(d: Path, rung: str, X: np.ndarray, names: list[str], frame: dict, pools: list[int], n_boot: int, n_jobs: int, n_est: int, seed: int, cut: int) -> dict:
    """Move 12's test (new_blocks.one_cut after its data) on one rung's rows; the per-rung files; the result dict with the
    pools per level and pool size (for the margins) and the scores."""
    d.mkdir(parents=True, exist_ok=True)
    sp, labels_of, kernel_of_cell, arche_of_cell, rep_of_cell, majority_class, labels_kernel, lab, pair_start, W = \
        frame["sp"], frame["labels_of"], frame["kernel_of_cell"], frame["arche_of_cell"], frame["rep_of_cell"], frame["majority_class"], frame["labels_kernel"], frame["lab"], frame["pair_start"], frame["W"]
    te = sp["test_idx"]
    res = {"rung": rung, "keys": list(RUNG_KEYS[rung]), "text": RUNG_TEXT[rung], "n_features": len(names), "levels": {}, "pools_by": {}}
    np.savez_compressed(d / "features.npz", X=X, feature_names=np.array(names, dtype=str), cell_id=lab["cell_id"], win_start=lab["win_start"], pair_start=pair_start,
                        train_idx=sp["train_idx"], test_idx=sp["test_idx"], block_id=sp["block_id"], block_pos=sp["block_pos"], cut=np.int64(cut), rung=np.array(rung))
    for level in LEVELS:
        y = labels_of[level]
        t0 = time.time()
        fit = NB.fit_level(X, names, y, sp, seed=seed, n_jobs=n_jobs, n_est=n_est)
        use = fit["use"]
        if "pred" not in use:
            res["levels"][level] = {"status": fit["status"], "scores": {}, "quarantined_features": fit["quarantined_features"], "score_source": fit["score_source"]}
            continue
        pv = NB.pool_votes(sp, use["pred"], use["proba"], use["classes"], y[te], pools)
        res["pools_by"][level] = {str(k): pv[k] for k in pools}
        sc = {k: NB.score_pools(pv[k], fit["classes_train"], kernel_of_cell, majority_class[level]) for k in pools}
        ci = {c: i for i, c in enumerate(use["classes"])}
        blocks = [{"block_id": b, "cell_id": b.split("|b")[0], "position": int(sp["block_pos"][i]), "win_start": int(lab["win_start"][te[i]]), "pair_start": int(pair_start[te[i]]),
                   "pair_end": int(pair_start[te[i]] + W - 1), "y_true": str(y[te[i]]), "y_pred": str(use["pred"][i]), "p_pred": float(use["proba"][i].max()),
                   "p_true": float(use["proba"][i][ci[str(y[te[i]])]]) if str(y[te[i]]) in ci else None} for i, b in enumerate(sp["block_id"].tolist())]
        res["levels"][level] = {"status": "ok", "scores": {str(k): {kk: vv for kk, vv in sc[k].items() if kk != "recall_per_class"} for k in pools},
                                "feature_count": use["feature_count"], "feature_count_used": use["feature_count_used"], "reduced": use["reduced"], "score_source": fit["score_source"],
                                "quarantined_features": fit["quarantined_features"], "n_train_windows": fit["n_train_windows"], "n_train_cells": fit["n_train_cells"],
                                "n_new_blocks": fit["n_new_blocks"], "classes_train": fit["classes_train"], "by_position": {str(p): v for p, v in NB.by_position(pv[1]).items()} if 1 in pv else {},
                                "run_level": NB.run_level_shares(pv[1], kernel_of_cell) if (level == "run" and 1 in pv) else None, "elapsed_s": round(time.time() - t0, 1)}
        write_csv(d / f"predictions_{level}.csv", ["block_id", "cell_id", "kernel", "archetype", "rep", "position", "win_start", "pair_start", "pair_end", "y_true", "y_pred", "p_pred", "p_true"],
                  [[b["block_id"], b["cell_id"], kernel_of_cell[b["cell_id"]], arche_of_cell[b["cell_id"]], rep_of_cell[b["cell_id"]], b["position"], b["win_start"], b["pair_start"],
                    b["pair_end"], b["y_true"], b["y_pred"], b["p_pred"], b["p_true"]] for b in blocks])
        write_csv(d / f"pools_{level}.csv", ["cell_id", "kernel", "pool_size", "start_position", "y_true", "y_pred", "vote_fraction"],
                  [[p["cell_id"], kernel_of_cell[p["cell_id"]], k, p["start_pos"], p["y_true"], p["y_pred"], p["vote_fraction"]] for k in pools for p in pv[k]])
        if level == "kernel":
            pred = {b["block_id"]: {"y_true": b["y_true"], "y_pred": b["y_pred"]} for b in blocks}
            Cm = I.confusion_of(pred, labels_kernel)
            write_csv(d / "confusion_kernel.csv", ["true\\predicted"] + labels_kernel, [[l] + Cm[i].tolist() for i, l in enumerate(labels_kernel)])
            (d / "confusion_kernel.svg").write_text(NB.confusion_svg(labels_kernel, Cm, rung, cut, sc[1]))
        s1 = sc[1] if 1 in sc else next(iter(sc.values()))
        print(f"[ladder] cut {cut} {rung:<7} {level:<9} acc {fmt(s1['accuracy'])} macro {fmt(s1['macro_recall'])} chance {fmt(s1['chance'])} majority {fmt(s1['majority'])}"
              + "".join(f" | pool {k}: {fmt(sc[k]['accuracy'])}" for k in pools if k != 1) + f" | features {use['feature_count_used']}/{use['feature_count']}"
              + (f" quarantined {len(fit['quarantined_features'])}" if fit["quarantined_features"] else "") + f" ({res['levels'][level]['elapsed_s']} s)", flush=True)
    write_json(d / "result.json", {"schema": "plan12.arrow_ladder_rung.v1", "citation": CITATION, "written_at": now_iso(), **res})
    return res


def rung_stands(d: Path, params_sig: dict) -> dict | None:
    p = d / "result.json"
    if not p.is_file():
        return None
    try:
        r = read_json(p)
    except Exception:                                      # noqa: BLE001
        return None
    return r if r.get("params_sig") == params_sig else None


def check_e_new(X0: np.ndarray, names0: list[str], enc: dict, res0: dict, m12: dict, pools: list[int]) -> dict:
    """Check 2: rung 0's rows against move 12's E_new rows (the same statistics on the same windows) and its scores against
    move 12's E_new scores read from <run>_newblocks."""
    cols = enc["cols"]["E_new"]
    Xe = enc["X"][:, cols]
    names_e = [enc["names"][j] for j in cols]
    rows_ok = (names0 == names_e) and X0.shape == Xe.shape and np.array_equal(X0, Xe, equal_nan=True)
    out = {"rows_identical": bool(rows_ok), "names_identical": names0 == names_e, "shape": [int(X0.shape[0]), int(X0.shape[1])], "move12": m12["status"], "scores": {}, "scores_identical": None}
    if not rows_ok and names0 == names_e and X0.shape == Xe.shape:
        diff = ~(np.isclose(X0, Xe, equal_nan=True))
        out["max_abs_dev"] = float(np.nanmax(np.abs(X0 - Xe))) if np.isfinite(X0).any() else None
        out["n_cells_differing"] = int(diff.sum())
    if m12["status"] != "ok":
        return out
    all_ok = True
    for level in LEVELS:
        for k in pools:
            mine = ((res0["levels"].get(level) or {}).get("scores") or {}).get(str(k)) or {}
            ref = m12["rows"].get((level, "E_new", k)) or {}
            cmp = {}
            for key in ("accuracy", "macro_recall", "chance", "majority"):
                a, b = mine.get(key), _f(ref.get(key))
                cmp[key] = {"mine": a, "move12": b, "equal": (a is None and b is None) or (a is not None and b is not None and abs(a - b) <= 1e-9)}
            cmp["n_pools"] = {"mine": mine.get("n_pools"), "move12": _f(ref.get("n_pools")), "equal": _f(mine.get("n_pools")) == _f(ref.get("n_pools"))}
            ok = all(v["equal"] for v in cmp.values())
            all_ok &= ok
            out["scores"][f"{level}/pool{k}"] = {"ok": ok, **cmp}
    out["scores_identical"] = bool(all_ok)
    return out


def ladder_cut(out: Path, cut_name: str, cut: int, runs: list[dict], enc_out, enc_rows, win: dict, pair_excluded: set, mode: str, run: Path, nb_out: Path, rungs: list[str], *,
               seen: float, gap: int, pools: list[int], n_boot: int, n_jobs: int, n_est: int, seed: int, force: bool, params_sig: dict) -> dict:
    d = out / "ladder" / f"cut{cut}"
    d.mkdir(parents=True, exist_ok=True)
    t_cut = time.time()
    # --- the window frame of move 12 (new_blocks.one_cut's data, through the same calls) ---
    m6_order = I.move6_cells(run, cut)
    if mode == I.MODE_E_NEW_IDLE:
        keep = set(m6_order or [r["cell_id"] for r in runs if r["role"] == "kernel"])
        enc = I.build_encodings_13([r for r in runs if r["role"] == "idle" or r["cell_id"] in keep], None, enc_rows, cut, win, {}, set())
    else:
        enc = I.build_encodings_13(runs, enc_out, enc_rows, cut, win, {}, pair_excluded)
    lab = enc["lab"]
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
    sp = NB.split_rows(lab, pair_start, W, seen, gap)
    sp["cell_id_rows"] = run_rows
    cells = sp["cells"]
    kernel_of_cell = {c: str(kernel_rows[np.flatnonzero(run_rows == c)[0]]) for c in cells}
    arche_of_cell = {c: str(arche_rows[np.flatnonzero(run_rows == c)[0]]) for c in cells}
    rep_of_cell = {c: int(lab["rep"][np.flatnonzero(run_rows == c)[0]]) for c in cells}
    labels_kernel = I.class_order(set(kernel_of_cell.values()))
    majority_class = {}
    for level in LEVELS:
        counts: dict = {}
        for c in cells:
            lv = {"archetype": arche_of_cell, "kernel": kernel_of_cell, "run": {c: c}}[level][c]
            counts[lv] = counts.get(lv, 0) + 1
        majority_class[level] = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    frame = {"sp": sp, "labels_of": labels_of, "kernel_of_cell": kernel_of_cell, "arche_of_cell": arche_of_cell, "rep_of_cell": rep_of_cell, "majority_class": majority_class,
             "labels_kernel": labels_kernel, "lab": lab, "pair_start": pair_start, "W": W}
    print(f"[ladder] cut {cut}: {len(cells)} recordings, {int(lab['n'])} windows of W {W}; seen {int(sp['train_idx'].size)}, new blocks {sp['n_new_total']}; the gap assertion held", flush=True)
    write_csv(d / "split.csv", ["cell_id", "kernel", "n_windows", "n_seen", "n_gap", "n_new", "first_seen_pair", "last_seen_pair", "first_new_pair", "last_new_pair", "gap_ok", "note"],
              [[c, kernel_of_cell[c], *[sp["per_cell"][c][k] for k in ("n_windows", "n_seen", "n_gap", "n_new", "first_seen_pair", "last_seen_pair", "first_new_pair", "last_new_pair", "gap_ok", "note")]] for c in cells])
    write_csv(d / "excluded_cells.csv", ["cell_id", "reason"], [[e["cell_id"], e["reason"]] for e in enc["excluded"]])
    # --- the rungs' series: move 1's stored N, H, A, C, S and this move's store pass ---
    by_run = {r["cell_id"]: r for r in runs}
    series_by_cell = {}
    for c in cells:
        stored = by_run[c]["arrays"]
        mine = load_rung_series(out, c)
        if not np.array_equal(np.asarray(stored["pair"]), np.asarray(mine["pair"])):
            raise Stop(EXIT_ERROR, f"stopped: the pair axis of this move's series for {c} differs from move 1's; re-run `series`")
        series_by_cell[c] = {"pair": np.asarray(stored["pair"]), **{k: np.asarray(stored[k]) for k in STORED}, **{k: mine[k] for k in ("C2", "S2", "L0C", "L0S", "L1C", "L1S", "L0", "L1")}}
    # --- the rungs ---
    results, timing, check2 = {}, {}, None
    m12 = move12_scores(nb_out, cut)
    for rung in rungs:
        rd = d / rung
        t1 = time.time()
        prev = None if force else rung_stands(rd, params_sig)
        if prev is not None and all((rd / f"pools_{lv}.csv").is_file() for lv in LEVELS if prev["levels"].get(lv, {}).get("status") == "ok"):
            results[rung] = prev
            print(f"[ladder] cut {cut} {rung}: stands from an earlier run (result.json matches the parameters); kept", flush=True)
            if rung == "rung0":
                check2 = prev.get("check_e_new")
            timing[rung] = 0.0
            continue
        X, names = rung_rows(lab, series_by_cell, cut, W, RUNG_KEYS[rung])
        print(f"[ladder] cut {cut} {rung}: {RUNG_TEXT[rung]}; {len(names)} features on {X.shape[0]} windows ({time.time() - t1:.1f} s to compute)", flush=True)
        res = one_rung(rd, rung, X, names, frame, pools, n_boot, n_jobs, n_est, seed, cut)
        res["params_sig"] = params_sig
        if rung == "rung0":
            check2 = check_e_new(X, names, enc, res, m12, pools)
            res["check_e_new"] = check2
            if not check2["rows_identical"] or check2["scores_identical"] is False:
                write_json(rd / "result.json", {"schema": "plan12.arrow_ladder_rung.v1", "citation": CITATION, "written_at": now_iso(), **{k: v for k, v in res.items() if k != "pools_by"}})
                raise Stop(EXIT_ERROR, f"stopped: rung 0 (N, H, A) does not reproduce move 12's E_new at cut {cut}: rows identical {check2['rows_identical']}, scores identical "
                                       f"{check2['scores_identical']} ({[k for k, v in check2['scores'].items() if not v['ok']][:4]}); nothing else written")
            print(f"[ladder] cut {cut} check 2: rung 0's rows equal move 12's E_new rows ({check2['shape'][0]} x {check2['shape'][1]}); scores "
                  f"{'identical to' if check2['scores_identical'] else 'NOT compared with (' + check2['move12'] + ')'} move 12's E_new", flush=True)
        write_json(rd / "result.json", {"schema": "plan12.arrow_ladder_rung.v1", "citation": CITATION, "written_at": now_iso(), **{k: v for k, v in res.items() if k != "pools_by"}, "params_sig": params_sig,
                                        "pools_by": res["pools_by"]})
        results[rung] = res
        timing[rung] = round(time.time() - t1, 1)
    # --- the tables: scores with move 12's E0 and E_new beside, margins between rungs, per kernel, run level, by position ---
    def pools_of(res, level, k):
        pb = (res.get("pools_by") or {}).get(level)
        return (pb or {}).get(str(k))
    score_rows = []
    for level in LEVELS:
        for rung in rungs:
            r = results[rung]
            for k in pools:
                lv = r["levels"].get(level) or {}
                s = (lv.get("scores") or {}).get(str(k)) or {}
                score_rows.append([level, rung, RUNG_TEXT[rung], k, lv.get("status"), s.get("n_pools"), s.get("n_classes"), s.get("accuracy"), s.get("macro_recall"), s.get("chance"),
                                   s.get("majority"), lv.get("feature_count"), lv.get("feature_count_used"), lv.get("score_source"), ";".join(lv.get("quarantined_features") or []),
                                   lv.get("n_train_windows"), lv.get("n_train_cells"), NO_NULL])
        for enc_name in ("E0", "E_new"):
            for k in pools:
                ref = m12["rows"].get((level, enc_name, k))
                if ref is None:
                    score_rows.append([level, f"move12_{enc_name}", f"move 12's {enc_name} ({m12['status']})", k, m12["status"] if m12["status"] != "ok" else "missing"] + [None] * 12 + [NO_NULL])
                else:
                    score_rows.append([level, f"move12_{enc_name}", f"move 12's {enc_name} (read from {nb_out.name})", k, ref.get("status"), _f(ref.get("n_pools")), _f(ref.get("n_classes")),
                                       _f(ref.get("accuracy")), _f(ref.get("macro_recall")), _f(ref.get("chance")), _f(ref.get("majority")), _f(ref.get("feature_count")),
                                       _f(ref.get("feature_count_used")), ref.get("score_source"), ref.get("quarantined_features"), _f(ref.get("n_train_windows")), _f(ref.get("n_train_recordings")), NO_NULL])
    write_csv(d / "scores.csv", ["level", "rung", "input_set", "pool_size", "status", "n_pools", "n_classes", "accuracy", "macro_recall", "chance", "majority", "feature_count",
                                 "feature_count_used", "score_source", "quarantined_features", "n_train_windows", "n_train_recordings", "null"], score_rows)
    margin_rows, margins = [], {lv: {} for lv in LEVELS}
    for level in LEVELS:
        for k in pools:
            for b, a in LADDER_PAIRS:
                if b not in results or a not in results:
                    continue
                pb, pa = pools_of(results[b], level, k), pools_of(results[a], level, k)
                comp = f"{b}-{a}"
                if pb is None or pa is None:
                    margin_rows.append([level, k, comp, None, None, None, None, None, None, None, "not run: " + str((results[a if pa is None else b]["levels"].get(level) or {}).get("status"))])
                    continue
                bm = NB.bootstrap_margin(pb, pa, cells, n_boot, SEED_BOOTSTRAP)
                margins[level][f"{k}:{comp}"] = bm
                margin_rows.append([level, k, comp, bm["delta"], bm["ci_lo"], bm["ci_hi"], bm["n_pools"], bm["n_recordings"], bm["gained"], bm["lost"], bm["status"]])
    write_csv(d / "margins.csv", ["level", "pool_size", "comparison", "delta_accuracy", "ci95_lo", "ci95_hi", "n_pools", "n_recordings", "n_pools_gained", "n_pools_lost", "status"], margin_rows)
    rk_rows, rl_rows, bp_rows = [], [], []
    for level in LEVELS:
        for rung in rungs:
            for k in pools:
                s = ((results[rung]["levels"].get(level) or {}).get("scores") or {}).get(str(k)) or {}
                for kern in labels_kernel:
                    if kern in (s.get("recall_per_kernel") or {}):
                        rk_rows.append([level, rung, k, kern, s["recall_per_kernel"][kern], sum(1 for p in (pools_of(results[rung], level, k) or []) if kernel_of_cell[p["cell_id"]] == kern)])
            bp = (results[rung]["levels"].get(level) or {}).get("by_position") or {}
            chance = (((results[rung]["levels"].get(level) or {}).get("scores") or {}).get("1") or {}).get("chance")
            for p, v in sorted(bp.items(), key=lambda kv: int(kv[0])):
                bp_rows.append([level, rung, int(p), v["n_blocks"], v["accuracy"], chance])
    for rung in rungs:
        rl = (results[rung]["levels"].get("run") or {}).get("run_level")
        if not rl:
            continue
        for k in pools:
            shares = NB.run_level_shares(pools_of(results[rung], "run", k), kernel_of_cell)
            for kern in labels_kernel:
                if kern in shares:
                    v = shares[kern]
                    rl_rows.append([rung, k, kern, v["n_pools"], v["right_run"], v["same_kernel_other_run"], v["other_kernel"]])
    write_csv(d / "recall_per_kernel.csv", ["level", "rung", "pool_size", "kernel", "recall", "n_pools"], rk_rows)
    write_csv(d / "run_level.csv", ["rung", "pool_size", "kernel", "n_pools", "right_run", "same_kernel_other_run", "other_kernel"], rl_rows)
    write_csv(d / "by_position.csv", ["level", "rung", "position", "n_blocks", "accuracy", "chance"], bp_rows)
    scores_fig = {lv: {rung: {k: ((results[rung]["levels"].get(lv) or {}).get("scores") or {}).get(str(k)) for k in pools} for rung in rungs} for lv in LEVELS}
    refs = {lv: {e: (m12["rows"].get((lv, e, 1)) or {}) for e in ("E0", "E_new")} for lv in LEVELS}
    (d / "ladder_accuracy.svg").write_text(ladder_svg(scores_fig, refs, pools, cut, rungs))
    write_csv(d / "ladder_accuracy.csv", ["level", "rung", "pool_size", "accuracy", "macro_recall", "chance", "majority", "n_pools", "move12_E0_accuracy", "move12_E_new_accuracy"],
              [[lv, rung, k, (scores_fig[lv][rung].get(k) or {}).get("accuracy"), (scores_fig[lv][rung].get(k) or {}).get("macro_recall"), (scores_fig[lv][rung].get(k) or {}).get("chance"),
                (scores_fig[lv][rung].get(k) or {}).get("majority"), (scores_fig[lv][rung].get(k) or {}).get("n_pools"), _f(refs[lv]["E0"].get("accuracy")), _f(refs[lv]["E_new"].get("accuracy"))]
               for lv in LEVELS for rung in rungs for k in pools])
    summary = {"schema": "plan12.arrow_ladder_cut.v1", "citation": CITATION, "cut": int(cut), "cut_name": cut_name, "mode": mode, "rungs": {r: {"keys": list(RUNG_KEYS[r]), "text": RUNG_TEXT[r]} for r in rungs},
               "ladder_pairs": [f"{b}-{a}" for b, a in LADDER_PAIRS], "window": win, "split": {"seen": seen, "gap": gap, "n_recordings": len(cells), "n_windows": int(lab["n"]),
               "n_seen_windows": int(sp["train_idx"].size), "n_new_blocks": sp["n_new_total"]}, "pools": pools, "majority_class": majority_class,
               "classes": {"archetype": sorted(set(arche_of_cell.values())), "kernel": labels_kernel, "n_run": len(cells)}, "n_excluded": len(enc["excluded"]),
               "n_estimators": n_est, "n_jobs": n_jobs, "seed_forest": seed, "seed_bootstrap": SEED_BOOTSTRAP, "n_bootstrap": n_boot, "null": NO_NULL,
               "move12": {"out": str(nb_out), "status": m12["status"]}, "check_e_new": check2,
               "scores": {lv: {rung: {k: v for k, v in (results[rung]["levels"].get(lv) or {}).items() if k not in ("by_position",)} for rung in rungs} for lv in LEVELS},
               "margins": margins, "time_per_rung_s": timing, "elapsed_s": round(time.time() - t_cut, 1), "files": sorted(p.name for p in d.iterdir())}
    write_json(d / "summary.json", summary)
    line = "; ".join(f"{lv}: " + ", ".join(f"{r.replace('rung', 'r')} {fmt((((results[r]['levels'].get(lv) or {}).get('scores') or {}).get('1') or {}).get('accuracy'))}" for r in rungs) for lv in LEVELS)
    print(f"[ladder] cut {cut} ({cut_name}) done in {summary['elapsed_s']} s; single-block accuracy: {line}", flush=True)
    return summary


def run_ladder(o: argparse.Namespace) -> int:
    run = Path(os.path.expanduser(o.run)).resolve()
    if not (run / "cells.csv").is_file():
        raise Stop(EXIT_MISSING, f"missing input: {run / 'cells.csv'} (--run must name a plan12_grounding output folder with moves 0, 1 and 6 done)")
    out = resolve_out(run, o.out)
    nb_out = Path(os.path.expanduser(o.newblocks_out)).resolve() if o.newblocks_out else NB.default_out(run)
    seen, gap, pools, n_boot = float(o.seen), int(o.gap), NB.parse_pools(o.pools), int(o.bootstrap)
    if not (0.0 < seen < 1.0) or gap < 0 or n_boot < 1:
        raise Stop(EXIT_MISSING, f"--seen must lie in (0, 1) (given {seen}), --gap be 0 or more (given {gap}), --bootstrap 1 or more (given {n_boot})")
    rungs = [s.strip() for s in str(o.rungs).split(",") if s.strip()] if o.rungs else list(RUNG_ORDER)
    bad = [r for r in rungs if r not in RUNG_KEYS]
    if bad:
        raise Stop(EXIT_MISSING, f"--rungs {bad}: the rungs are {', '.join(RUNG_ORDER)}")
    if "rung0" not in rungs:
        rungs = ["rung0"] + rungs                          # the check always runs
    lock = refuse_if_driver_live(run, o.dry_run)
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
    missing_series = [r["cell_id"] for r in runs if not series_stands(out, r["cell_id"], None)]
    print(f"[ladder] ladder: move 12's test on {len(rungs)} rungs ({', '.join(r.replace('rung', 'r') for r in rungs)}), both cuts {', '.join(f'{k} = {v}' for k, v in cuts.items())}; "
          f"seen {seen:g}, gap {gap}, pools {pools}; three levels; forest {n_est} trees, {n_jobs} processes, seed {seed} (move 6's values unless overridden); bootstrap {n_boot}; {NO_NULL[:50]}...")
    print(f"[ladder] run {run} (read only); move 12's results {nb_out} (read only; {', '.join(f'cut {c}: ' + move12_scores(nb_out, c)['status'] for c in cuts.values())}); "
          f"out {out}{' (exists)' if out.exists() else ' (to be created)'}; D3 window {win['grid_id']} (W {win['W']}, H {win['H']}); mode: {mode}")
    print(f"[ladder] recordings found: {len(runs)} = {len(kernel_runs)} kernel + {len(idle_runs)} idle; this move's series present for {len(runs) - len(missing_series)} of {len(runs)}"
          + (f" (missing: {missing_series[:3]}{'...' if len(missing_series) > 3 else ''}; run `series` first)" if missing_series else ""))
    print(f"[ladder] driver lock: " + (f"held by pid {lock['pid']} ({'alive' if lock['alive'] else 'dead'}), started {lock['started_at']}: {lock['argv_tail']}" if lock["exists"] else "absent")
          + ("; the refusal is skipped by --dry-run" if (lock["alive"] and o.dry_run) else ""))
    if o.dry_run:
        print(f"[ladder] dry run: would write {out}/ladder/cut<C>/ (scores.csv, margins.csv, recall_per_kernel.csv, run_level.csv, by_position.csv, split.csv, ladder_accuracy.svg, "
              f"<rung>/ (features.npz, predictions_<level>.csv, pools_<level>.csv, confusion_kernel.csv/.svg, result.json), summary.json), {out}/ladder/arrow_ladder.json and {out / RECORD_NAME}; nothing written")
        return EXIT_OK
    if missing_series:
        raise Stop(EXIT_MISSING, f"missing input: this move's series for {len(missing_series)} recording(s) (first: {missing_series[0]}); run `arrow_ladder series` first")
    out_lock = acquire_out_lock(out)
    code = code_identity()
    params = {"run": str(run), "out": str(out), "newblocks_out": str(nb_out), "cuts": cuts, "rungs": rungs, "encoding_out": str(enc_out) if enc_out else None, "window": win,
              "seen": seen, "gap": gap, "pools": pools, "bootstrap": n_boot, "seed_bootstrap": SEED_BOOTSTRAP, "n_estimators": n_est, "n_jobs": n_jobs, "seed_forest": seed,
              "seed_offset": m6["seed_offset"], "mode": mode, "n_kernel_cells": len(kernel_runs), "n_idle_cells": len(idle_runs), "series_version": SERIES_VERSION, "force": bool(o.force)}
    params_sig = {k: v for k, v in params.items() if k not in ("n_jobs", "force", "newblocks_out", "rungs", "out")}
    entry = {"key": KEY_LADDER, "command": sys.argv, "cwd": str(_HERE.parent), "started_at": now_iso(), "status": "running", "exit_code": None, "finished_at": None, "params": params,
             "inputs_sha256": {**NB.inputs_identity(run, enc_out, declared, kernel_runs + idle_runs, cuts), "arrow_series": {r["cell_id"]: sha256_file(series_path(out, r["cell_id"])) for r in runs}},
             "code": {"arrow_ladder.py": code["sha256"].get("arrow_ladder.py"), "fingerprint": code["fingerprint"], "modules": code["modules"], "sha256": code["sha256"],
                      "toolkit_fingerprint": toolkit_fingerprint()["sha256"]},
             "outputs": []}
    outputs_exist = (out / "ladder" / "arrow_ladder.json").is_file() and all((out / "ladder" / f"cut{c}" / "summary.json").is_file() for c in cuts.values())
    try:
        if _skip_if_unchanged(out, KEY_LADDER, entry, ("n_jobs", "force", "newblocks_out"), outputs_exist, bool(o.force), f"{out}/ladder"):
            return EXIT_OK
        t0 = time.time()
        results = [ladder_cut(out, name, cut, runs, enc_out, enc_rows, win, pair_excluded, mode, run, nb_out, rungs, seen=seen, gap=gap, pools=pools, n_boot=n_boot, n_jobs=n_jobs,
                              n_est=n_est, seed=seed, force=bool(o.force), params_sig=params_sig) for name, cut in cuts.items()]
        move_rec = {"schema": "plan12.arrow_ladder.v1", "citation": CITATION, "package_version": __version__, "written_at": now_iso(), "command": sys.argv,
                    "question": __doc__.split("THE QUESTION.", 1)[1].split("THE TWO STEPS", 1)[0].strip(), "params": params, "rungs": {r: {"keys": list(RUNG_KEYS[r]), "text": RUNG_TEXT[r]} for r in rungs},
                    "e0": {"move6_gate": e0_status or "E0 usable", "idle": {k: check[k] for k in ("can_build", "n_idle", "n_idle_with_e0", "reason")}}, "mode": mode,
                    "check_e_new": {s["cut_name"]: s["check_e_new"] for s in results}, "code": entry["code"], "elapsed_s": round(time.time() - t0, 1), "results": results}
        write_json(out / "ladder" / "arrow_ladder.json", move_rec)
        entry["outputs"] = sorted(str(p.relative_to(out)) for p in (out / "ladder").rglob("*") if p.is_file())
        entry.update(status="done", exit_code=0, finished_at=now_iso(), elapsed_s=move_rec["elapsed_s"])
        print(f"[ladder] ladder done in {move_rec['elapsed_s']} s: {out / 'ladder'}")
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
    p.add_argument("--out", default=None, help="the output folder (default: a sibling of the run, <run>_arrowladder; never inside the run)")
    p.add_argument("--n-jobs", type=int, default=None, help="processes for the forest (default: move 6's recorded value; ignored by `series`)")
    p.add_argument("--n-estimators", type=int, default=None, help="trees (default: move 6's recorded value; ignored by `series`)")
    p.add_argument("--force", action="store_true", help="re-run a finished step")
    p.add_argument("--dry-run", action="store_true", help="print the plan; write nothing; skip the live-lock refusal")


def main(argv: list[str] | None = None) -> int:
    install_sigterm()
    ap = argparse.ArgumentParser(prog="plan12_grounding.arrow_ladder", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("series", help="step 1: one pass over the per-page stores, every rung's per-pair series per recording, checked against move 1")
    _common(s)
    s.add_argument("--only", default=None, help="a comma-separated list of cell ids (a test on a few recordings)")
    la = sub.add_parser("ladder", help="step 2: move 12's test on every rung, both cuts, with the two checks and the summary figure")
    _common(la)
    la.add_argument("--cuts", default=None, help="which cuts, by name: declared,measured (default both; the values come from params.json)")
    la.add_argument("--rungs", default=None, help="the rungs to run (default all; rung0, the check, always runs)")
    la.add_argument("--seen", type=float, default=DEFAULT_SEEN, help="move 12's share of seen windows (default 0.8)")
    la.add_argument("--gap", type=int, default=DEFAULT_GAP, help="move 12's gap in windows (default 1)")
    la.add_argument("--pools", default=DEFAULT_POOLS, help="move 12's pool sizes (default 1,2,3)")
    la.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP, help="move 12's bootstrap resamples (default 1000)")
    la.add_argument("--encoding-out", default=None, help="decision D2 (default: the encoding run move 6 recorded in classify.json)")
    la.add_argument("--newblocks-out", default=None, help="where move 12 wrote its results (default <run>_newblocks; read only)")
    o = ap.parse_args(argv)
    try:
        return run_series(o) if o.cmd == "series" else run_ladder(o)
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
