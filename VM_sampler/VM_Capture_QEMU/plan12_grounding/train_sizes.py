#!/usr/bin/env python3
"""train_sizes.py -- the training-size loops beside the run of plan12_grounding (2026-10-13; move 22 in the console's
Grounding paper tab, after the held-out ladder (21)): move 21's test wrapped in loops over the number of training runs
and the overlap of the training pieces, with random rounds repeated.

  python3 -m plan12_grounding.train_sizes check    --run <main run out> [--check-length 64] [--heldout-out <dir>] [--out <dir>] ...
  python3 -m plan12_grounding.train_sizes plan     --run <main run out> [--sizes 1,2,3] [--repeats 5] [--seed S] [--lengths 64,32,...] [--overlaps 50] ...
  python3 -m plan12_grounding.train_sizes unit     --run <main run out> (--index I | --cut C --length L --overlap P --rung R [--size M --repeat K]) ...
  python3 -m plan12_grounding.train_sizes run      --run <main run out> --size M [the same choice flags] ...
  python3 -m plan12_grounding.train_sizes collect  --run <main run out> ...
  python3 -m plan12_grounding.train_sizes estimate --run <main run out> [the same choice flags] [--json]
  (run from VM_sampler/VM_Capture_QEMU/; common flags: --out, --arrowladder-out, --heldout-out, --encoding-out, --cuts, --rungs,
   --bootstrap, --n-jobs, --n-estimators, --force, --dry-run)

THE QUESTION. Move 21 names pieces of recordings never seen, with six of the eight runs of every kernel in training and a
new training window every half length. How many runs of each kernel does the naming need, and does the overlap of the
training pieces matter? Everything inside the loop is move 21's: the piece lengths, the ladder's rungs, the statistic
sets, the levels kernel and archetype, the forest, the scores, both cuts.

THE OUTER LOOP, the training size m (1 to 7): every group (the 12 kernels and idle) trains on m of its runs and holds
out the rest. A group with fewer runs trains on at most its runs minus one (lexer has 7, so at m = 7 it trains on 6);
the rounds file records it.

THE ROUNDS, per repeat and per group: the group's runs are shuffled with a seed made from (the base seed, the repeat
number, the group); consecutive blocks of (runs - m) of the shuffled list are held out, one block per round; when
that does not divide the runs, the last block is filled with runs drawn at random from the earlier blocks. Groups are
shuffled independently; round k holds out block k of every group (a group with fewer blocks than the round count cycles
through its blocks again, so its runs in the first blocks are tested twice). So in every repeat every run is tested at
least once, always by a model that never saw it; a run tested twice in one repeat counts once (its pieces weigh 1/2
each: the mean of its two tests). When the rounds cannot differ between repeats (every group leaves one run out), one
repeat runs and the record says so. Before any training the code asserts, for every repeat and group, that every run is
held out in at least one round and that no held-out run is in its round's training set; a failure stops the step.

THE OVERLAP of the training pieces, the third loop: 0%, 25%, 50% or 75% (move 21 uses 50%: a new piece every half
length). A piece starts on a whole pair, so an overlap that gives no whole-pair start at a length (25% and 75% at 2 pairs)
is skipped there and listed. The test pieces stay consecutive and non-overlapping.

THE REPEATS are numbered 1, 2, 3, ...; their seeds depend only on (the base seed, the repeat number): the same base seed
always gives the same repeats, and asking later for more repeats runs only the new ones. The base seed defaults to the
run's forest seed (move 6's, as move 21 uses it); repeat r's forest is seeded with base + r; the shuffles with sha256 of
(base, r, group).

THE UNITS. The nested loops are flattened into a list of parameter vectors, one per unit of work, (cut, size, overlap,
repeat, length, rung), each unit holding all the rounds of its repeat. The features of a (cut, length, overlap, rung) are
shared by every size and repeat: they are computed once, as their own units listed first, and the model units read them.
`plan` writes the list for the choices made (one line per unit, with its index); `unit` runs one unit by its index or
its parameters and writes only its own result file (no record entry, no folder lock: units can run alone, in any order,
on any machine, once the feature units are done); `run --size M` runs the feature units its model units need, then the
model units of that size, skipping what stands; `collect` merges whatever units have finished into the tables and the
figures. Nothing in a unit depends on another unit having run in the same process.

THE REPORT, per cut, size, overlap, piece length, rung and level: the mean over repeats and its spread (min, max,
standard deviation), with each repeat's value kept; the differences between rungs as move 21 (move 12's bootstrap over
recordings, per repeat, then the mean over repeats). One summary figure per cut and level: accuracy against the number
of training runs, one line per rung, one panel per piece length (and per overlap when more than one was run), the spread
over repeats as a band.

CHECK, before anything else: with m = 6, one repeat, 50% overlap and move 21's fixed pairs instead of the shuffle, the
scores must equal move 21's (read from <run>_heldout, or from --heldout-out, where move 21 ran on the same recordings),
at the check length (64 by default). The feature rows are compared too when move 21's features stand there. A size step
refuses to start until the check has passed in <out>.

THE OUTPUTS, under <out> (default <run>_trainsizes)
  trainsizes/plan.json, plan.csv            the units of the choices made, with their indices and what stands
  trainsizes/rounds.csv, rounds/size<M>_rep<K>.json   which runs were held out in each round of each repeat
  trainsizes/skipped.csv                    the (length, overlap) pairs with no whole-pair start
  trainsizes/cut<C>/len<L>/ov<P>/features/<rung>.npz           a feature unit (every recording's windows on the union of
                                                               the training grid and the test grid)
  trainsizes/cut<C>/size<M>/ov<P>/rep<K>/len<L>/<rung>/        a model unit: result.json, predictions_<level>.csv
  trainsizes/cut<C>/scores.csv, scores_per_repeat.csv, margins.csv, margins_per_repeat.csv, recall_per_kernel.csv,
            left_out.csv, accuracy_vs_trainsize_<level>.svg/.csv                      (collect)
  trainsizes/check_move21/check.json, check.csv and the check's own units   (check)
  trainsizes/size<M>.json, trainsizes/summary.json, record.json (one entry per step: check, size<M>, collect)

THE RULES THAT PROTECT THE LIVE RUN (as heldout_ladder.py)
- ONE new file; nothing existing in plan12_grounding/ is edited; heldout_ladder.py, arrow_ladder.py, new_blocks.py and
  stats.py are imported as they are (their recorded code fingerprints stay); nothing imports this module. The run folder,
  every other sibling folder and the L1 stores are read only here; --out inside the run or at another sibling is refused
  (exit 2). A step refuses to start (exit 3) while <run>/.driver.lock, move 20's lock or move 21's lock belongs to a live
  process; --dry-run skips those refusals and writes nothing. One writer per output folder for the steps:
  <out>/.train_sizes.lock; run_moves.install_sigterm.
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
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

from plan12_grounding import __version__, toolkit_fingerprint  # noqa: E402
from plan12_grounding import arrow_ladder as AL  # noqa: E402  (move 20: the rungs, the feature names, the series files)
from plan12_grounding import heldout_ladder as HL  # noqa: E402  (move 21: the recordings and labels, the folds, the series, the statistic sets, the figure style)
from plan12_grounding import idle_class as I  # noqa: E402
from plan12_grounding import new_blocks as NB  # noqa: E402  (move 12: fit_level)
from plan12_grounding import stats as ST  # noqa: E402
from plan12_grounding.run_moves import _pid_alive, code_fingerprint, install_sigterm, now_iso, read_json, sha256_file, write_json  # noqa: E402
from plan12_grounding.figures import svg_open, write_csv  # noqa: E402
from plan11_encoding_ladder.nulls import SEED_FOREST  # noqa: E402

CITATION = ("plan12_grounding/heldout_ladder.py (move 21: whole recordings held out, pieces of 64 to 2 pairs, each named alone; its scores are the check); "
            "plan12_grounding/arrow_ladder.py (move 20: the rungs); plan12_grounding/new_blocks.py (move 12: the forest, the dimension rule, the quarantine, the bootstrap); "
            "the author's design of the training-size loops (2026-10-13): m runs of every group in training, random rounds with a seed per (base, repeat, group), "
            "the overlap of the training pieces as a third loop, every unit of work its own parameter vector")
OUT_SUFFIX = "_trainsizes"
LOCK_NAME = ".train_sizes.lock"
RECORD_NAME = "record.json"
SIZES = (1, 2, 3, 4, 5, 6, 7)
OVERLAPS = (0, 25, 50, 75)
DEFAULT_OVERLAPS = (50,)
DEFAULT_REPEATS = 5
LENGTHS = HL.LENGTHS
LEVELS = HL.LEVELS
RUNG_ORDER, RUNG_KEYS, RUNG_TEXT, LADDER_PAIRS, RUNG_COLOUR = AL.RUNG_ORDER, AL.RUNG_KEYS, AL.RUNG_TEXT, AL.LADDER_PAIRS, AL.RUNG_COLOUR
IDLE_LABEL = NB.IDLE_LABEL
SEED_BOOTSTRAP = NB.SEED_BOOTSTRAP
NO_NULL = NB.NO_NULL
DEFAULT_BOOTSTRAP = NB.DEFAULT_BOOTSTRAP
FEATURES_VERSION = "plan12.trainsizes_features.v1"
EXIT_OK, EXIT_ERROR, EXIT_MISSING, EXIT_REFUSED = I.EXIT_OK, I.EXIT_ERROR, I.EXIT_MISSING, I.EXIT_REFUSED
Stop = I.Stop
KEY_CHECK, KEY_COLLECT = "check", "collect"
ROUND_RULE = ("per repeat and group: the group's runs shuffled with sha256(base seed, repeat, group); consecutive blocks of (runs - m) held out, one per round; "
              "a remainder block filled with runs drawn at random from the earlier blocks; round k holds out block k of every group, a group with fewer blocks cycling; "
              "a run tested twice in a repeat weighs 1/2 per test; a group with fewer runs trains on at most its runs minus one")
PIECE_RULE = ("training windows of the length every (1 - overlap) x length pairs over the cut series of the training runs (the training grid); test pieces = consecutive, "
              "non-overlapping windows of the length over the held-out runs (the test grid; start a multiple of the length); every piece named alone; a (length, overlap) "
              "with no whole-pair training start is skipped")
LEFT_OUT_RULE = HL.LEFT_OUT_RULE.replace("any fold", "any round")
# move 21's measured step times on the real run (2026-10-13, 300 trees, 4 processes; four folds, seven rungs, both levels): the basis of the estimate shown in the tab
# (features: the seven feature files of the length; folds: the forests of the seven rungs). Length 2 was measured at the declared cut only (a scratch run stopped before
# the measured cut); its measured-cut values are extrapolated with the ratio the other lengths show.
MEASURED_21 = {64: {"features_s": {"declared": 21.2, "measured": 21.0}, "folds_s": {"declared": 72.6, "measured": 59.1}, "measured": True},
               32: {"features_s": {"declared": 36.7, "measured": 36.2}, "folds_s": {"declared": 141.1, "measured": 109.9}, "measured": True},
               16: {"features_s": {"declared": 67.6, "measured": 69.3}, "folds_s": {"declared": 313.0, "measured": 255.3}, "measured": True},
               8: {"features_s": {"declared": 128.5, "measured": 128.0}, "folds_s": {"declared": 772.6, "measured": 544.2}, "measured": True},
               4: {"features_s": {"declared": 244.9, "measured": 243.0}, "folds_s": {"declared": 2040.5, "measured": 1459.6}, "measured": True},
               2: {"features_s": {"declared": 602.5, "measured": 480.0}, "folds_s": {"declared": 6085.5, "measured": 4450.0}, "measured": False}}
N_RUNGS_21, N_FOLDS_21 = 7, HL.N_FOLDS
# the share of a round's cost that follows the training rows (the forest's fit); the rest follows the held-out pieces (the prediction and the quarantine's
# work per piece). Calibrated on move 21 (six runs per group) against one measurement with two runs per group at lengths 64 and 32 (2026-10-13), where the
# cost per round rose threefold while the training rows fell to a third: at the long lengths the pieces dominate; at the short lengths the forest's fit grows.
W_FIT = {64: 0.1, 32: 0.1, 16: 0.3, 8: 0.3, 4: 0.5, 2: 0.5}


def fmt(x) -> str:
    return I.fmt(x)


def hop_of(L: int, overlap: int) -> int | None:
    """The training grid's step: (100 - overlap)% of the length, in whole pairs; None when that is not a whole pair."""
    h = L * (100 - int(overlap))
    if h % 100 != 0:
        return None
    return int(h // 100)


def grid_ratio(overlap: int) -> float:
    """Windows per recording relative to move 21's half-length grid: the union of the training grid and the test grid."""
    return {0: 0.5, 25: 1.0, 50: 1.0, 75: 2.0}[int(overlap)]


def stable_seed(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


# ---------------------------------------------------------------------------------------------
# the output folder, the locks, the record (heldout_ladder's pattern)
# ---------------------------------------------------------------------------------------------
def default_out(run: Path) -> Path:
    return run.parent / (run.name + OUT_SUFFIX)


def other_siblings(run: Path) -> set[Path]:
    """The sibling folders of the other moves beside the run (never written here)."""
    suffixes = ("_idle13", "_newblocks", "_pagefourier", "_timeshuffle", "_randomsplit", "_timeshuffle_sizes", AL.OUT_SUFFIX if hasattr(AL, "OUT_SUFFIX") else "_arrowladder", HL.OUT_SUFFIX)
    return {run.parent / (run.name + s) for s in suffixes}


def resolve_out(run: Path, out_arg: str | None) -> Path:
    out = Path(os.path.expanduser(out_arg)).resolve() if out_arg else default_out(run)
    if out == run or run in out.parents:
        raise Stop(EXIT_MISSING, f"refused: --out {out} lies inside the run folder {run}, which is read only here; the default is {default_out(run)}")
    for s in other_siblings(run):
        if out == s or s in out.parents:
            raise Stop(EXIT_MISSING, f"refused: --out {out} lies in another move's sibling folder {s}, which is read only here; the default is {default_out(run)}")
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
            raise Stop(EXIT_REFUSED, f"refused: another train_sizes step is writing {out} ({lock}: pid {pid}, started {old.get('started_at')}); one writer per output folder")
        print(f"[trainsizes] stale lock from pid {pid} ({old.get('started_at')}) taken over", flush=True)
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
    return {"schema": "plan12.train_sizes.record.v1", "citation": CITATION, "package_version": __version__, "entries": []}


def save_record(out: Path, rec: dict) -> None:
    write_json(out / RECORD_NAME, rec)


def last_done(rec: dict, key: str) -> dict | None:
    for e in reversed(rec.get("entries") or []):
        if e.get("key") == key and e.get("status") == "done":
            return e
    return None


def code_identity() -> dict:
    fp = code_fingerprint("train_sizes")
    files = toolkit_fingerprint()["files"]
    return {"fingerprint": fp["sha256"], "modules": fp["modules"], "sha256": {f"{m}.py": files.get(f"{m}.py") for m in fp["modules"]}}


def refuse_if_busy(run: Path, al_out: Path, ho_out: Path, dry_run: bool) -> dict:
    lock = HL.refuse_if_busy(run, al_out, dry_run)        # the run's driver lock and move 20's lock
    ho_lock = ho_out / HL.LOCK_NAME
    if ho_lock.is_file() and not dry_run:
        try:
            old = read_json(ho_lock)
        except Exception:                                  # noqa: BLE001
            old = {}
        if old.get("pid") and _pid_alive(int(old["pid"])):
            raise Stop(EXIT_REFUSED, f"refused: move 21 is still writing {ho_out} ({ho_lock}: pid {old['pid']}, started {old.get('started_at')}); run again after it ends")
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
            print(f"[trainsizes] skipped: {tag} stands as made on {prev.get('finished_at')} (inputs, parameters and code unchanged; --force re-runs)")
            return True
        print(f"[trainsizes] re-running: {why} since {prev.get('finished_at')}")
    rec["entries"].append(entry)
    save_record(out, rec)
    return False


# ---------------------------------------------------------------------------------------------
# the choices: sizes, repeats, seed, lengths, overlaps
# ---------------------------------------------------------------------------------------------
def _ints(spec, allowed: tuple, what: str, default: tuple) -> list[int]:
    if spec is None or str(spec).strip() == "":
        return list(default)
    try:
        vals = [int(s) for s in str(spec).replace(";", ",").split(",") if s.strip()]
    except ValueError as exc:
        raise Stop(EXIT_MISSING, f"--{what} {spec!r}: a comma-separated list of integers is expected") from exc
    bad = [v for v in vals if v not in allowed]
    if bad:
        raise Stop(EXIT_MISSING, f"--{what} {bad}: the {what} are {', '.join(map(str, allowed))}")
    seen: list[int] = []
    for v in vals:
        if v not in seen:
            seen.append(v)
    return seen


def choices_of(o: argparse.Namespace, base_default: int) -> dict:
    sizes = _ints(getattr(o, "sizes", None), SIZES, "sizes", SIZES)
    lengths = _ints(getattr(o, "lengths", None), LENGTHS, "lengths", LENGTHS)
    overlaps = _ints(getattr(o, "overlaps", None), OVERLAPS, "overlaps", DEFAULT_OVERLAPS)
    repeats = int(getattr(o, "repeats", None) or DEFAULT_REPEATS)
    if repeats < 1:
        raise Stop(EXIT_MISSING, f"--repeats {repeats}: at least one repeat")
    seed = int(o.seed) if getattr(o, "seed", None) not in (None, "") else int(base_default)
    return {"sizes": sorted(sizes), "lengths": sorted(lengths, reverse=True), "overlaps": sorted(overlaps), "repeats": repeats, "base_seed": seed}


def choice_tokens(ch: dict) -> list[str]:
    return ["--sizes", ",".join(map(str, ch["sizes"])), "--repeats", str(ch["repeats"]), "--seed", str(ch["base_seed"]),
            "--lengths", ",".join(map(str, ch["lengths"])), "--overlaps", ",".join(map(str, ch["overlaps"]))]


# ---------------------------------------------------------------------------------------------
# the rounds: per repeat and group, the held-out blocks
# ---------------------------------------------------------------------------------------------
def group_runs(runs: list[dict]) -> dict:
    """{group: [cell_id ...] in rep order} over the admissible recordings (stats.load_runs's order)."""
    g: dict = {}
    for r in runs:
        g.setdefault(r["group"], []).append(r["cell_id"])
    return {k: sorted(v, key=lambda c: (next(r["rep"] for r in runs if r["cell_id"] == c), c)) for k, v in g.items()}


def make_rounds(groups: dict, size: int, base_seed: int, repeat: int) -> dict:
    """The rounds of one repeat at training size m: per group the shuffled runs cut into held-out blocks of (runs - m),
    round k holding out block k of every group (cycling for a group with fewer blocks). Asserts, before any training,
    that every run is held out at least once and that no held-out run is in its round's training set."""
    per_group, n_blocks = {}, {}
    for g in sorted(groups):
        cells = list(groups[g])
        n = len(cells)
        m_used = min(int(size), n - 1)
        if m_used < 1:
            raise Stop(EXIT_ERROR, f"stopped: group {g} has {n} run(s); a training size of at least 1 leaves nothing to hold out")
        held = n - m_used
        rng = np.random.default_rng(stable_seed(base_seed, repeat, g))
        order = [cells[i] for i in rng.permutation(n).tolist()]
        full = n // held
        blocks = [order[j * held:(j + 1) * held] for j in range(full)]
        rem = order[full * held:]
        filled: list[str] = []
        if rem:
            earlier = order[:full * held]
            take = held - len(rem)
            pick = [earlier[i] for i in rng.choice(len(earlier), size=take, replace=False).tolist()] if take > 0 else []
            filled = pick
            blocks.append(rem + pick)
        per_group[g] = {"n_runs": n, "m_used": m_used, "n_heldout_per_round": held, "capped": m_used < int(size), "shuffled": order, "blocks": blocks,
                        "remainder": rem, "filled_from_earlier": filled, "seed": stable_seed(base_seed, repeat, g)}
        n_blocks[g] = len(blocks)
    n_rounds = max(n_blocks.values())
    rounds = []
    for k in range(n_rounds):
        held_all, by_group = [], {}
        for g in sorted(groups):
            b = per_group[g]["blocks"][k % n_blocks[g]]
            by_group[g] = {"heldout": list(b), "block": k % n_blocks[g], "cycled": k >= n_blocks[g]}
            held_all += list(b)
        train = [c for g in sorted(groups) for c in groups[g] if c not in set(held_all)]
        if set(held_all) & set(train):
            raise Stop(EXIT_ERROR, f"stopped: repeat {repeat}, size {size}, round {k}: a held-out run is in its round's training set")
        rounds.append({"round": k, "heldout": held_all, "train": train, "n_heldout": len(held_all), "n_train": len(train), "per_group": by_group})
    times_held = {c: 0 for g in groups for c in groups[g]}
    for ro in rounds:
        for c in ro["heldout"]:
            times_held[c] += 1
    never = [c for c, n in times_held.items() if n == 0]
    if never:
        raise Stop(EXIT_ERROR, f"stopped: repeat {repeat}, size {size}: {len(never)} run(s) never held out (first: {never[0]})")
    for g in sorted(groups):
        for ro in rounds:
            if len(ro["per_group"][g]["heldout"]) != per_group[g]["n_heldout_per_round"]:
                raise Stop(EXIT_ERROR, f"stopped: repeat {repeat}, size {size}, round {ro['round']}: group {g} holds out {len(ro['per_group'][g]['heldout'])} runs, not {per_group[g]['n_heldout_per_round']}")
    leave_one_out = all(v["n_heldout_per_round"] == 1 for v in per_group.values())
    return {"size": int(size), "repeat": int(repeat), "base_seed": int(base_seed), "n_rounds": n_rounds, "rounds": rounds, "groups": per_group, "times_held": times_held,
            "n_train_total": int(sum(v["m_used"] for v in per_group.values())), "leave_one_out": leave_one_out, "fixed": False, "rule": ROUND_RULE}


def fixed_rounds_move21(runs: list[dict], groups: dict) -> dict:
    """Move 21's folds as rounds (the check): m = 6 with lexer at 6, every run held out exactly once."""
    folds = HL.make_folds(runs)
    rounds = []
    for fo in folds:
        held = set(fo["heldout"])
        rounds.append({"round": fo["fold"], "heldout": list(fo["heldout"]), "train": list(fo["train"]), "n_heldout": fo["n_heldout"], "n_train": fo["n_train"],
                       "per_group": {g: {"heldout": [c for c in groups[g] if c in held], "block": fo["fold"], "cycled": False} for g in sorted(groups)}})
    return {"size": 6, "repeat": 0, "base_seed": None, "n_rounds": len(rounds), "rounds": rounds,
            "groups": {g: {"n_runs": len(groups[g]), "m_used": len(groups[g]) - 2 if len(groups[g]) >= 8 else len(groups[g]) - 1, "n_heldout_per_round": None, "capped": False} for g in sorted(groups)},
            "times_held": {c: 1 for g in groups for c in groups[g]}, "n_train_total": None, "leave_one_out": False, "fixed": True, "rule": HL.FOLD_RULE}


def repeats_of(size: int, groups: dict, ch: dict) -> list[int]:
    """The repeats to run at a size: 1 .. repeats, or only 1 when every group leaves one run out (the rounds cannot differ)."""
    probe = make_rounds(groups, size, ch["base_seed"], 1)
    return [1] if probe["leave_one_out"] else list(range(1, ch["repeats"] + 1))


def rounds_hash(rd: dict) -> str:
    return hashlib.sha256(json.dumps([ro["heldout"] for ro in rd["rounds"]], sort_keys=True).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------
# the feature units: every recording's windows on the union of the training grid and the test grid
# ---------------------------------------------------------------------------------------------
def features_path(out: Path, cut: int, L: int, overlap: int, rung: str) -> Path:
    return out / "trainsizes" / f"cut{cut}" / f"len{L}" / f"ov{overlap}" / "features" / f"{rung}.npz"


def windows_on_grids(series_by_cell: dict, cells: list[str], cut: int, L: int, hop: int, keys: tuple) -> dict:
    names = AL.rung_feature_names(keys)
    X_rows, cell_ids, starts, on_train, is_piece = [], [], [], [], []
    failures: dict = {}
    for c in cells:
        a = ST.cut_series({"arrays": series_by_cell[c]}, cut)
        n = int(a["pair"].size)
        if n < L:
            continue
        grid = sorted(set(range(0, n - L + 1, hop)) | set(range(0, n - L + 1, L)))
        for t in grid:
            st = {}
            for s in keys:
                vals, failed = HL.safe_series_stats(s, a[s][t:t + L].astype(np.float64))
                st.update(vals)
                for k, msg in failed.items():
                    f = failures.setdefault(k, {"n_windows": 0, "first_error": msg})
                    f["n_windows"] += 1
            X_rows.append([st[k] for k in names])
            cell_ids.append(c); starts.append(t); on_train.append(t % hop == 0); is_piece.append(t % L == 0)
    X = np.asarray(X_rows, dtype=np.float64).reshape(len(X_rows), len(names))
    return {"X": X, "names": names, "cell_id": np.array(cell_ids, dtype=str), "win_start": np.array(starts, dtype=np.int64), "on_train_grid": np.array(on_train, dtype=bool),
            "is_piece": np.array(is_piece, dtype=bool), "hop": int(hop), "L": int(L), "failures": failures}


def features_stand(p: Path, cells: list[str], cut: int, L: int, overlap: int) -> dict | None:
    if not p.is_file():
        return None
    try:
        with np.load(p, allow_pickle=False) as z:
            if str(z["version"]) == FEATURES_VERSION and int(z["L"]) == L and int(z["cut"]) == cut and int(z["overlap"]) == overlap and set(z["cell_id"].astype(str).tolist()) == set(cells):
                return {"X": z["X"], "names": z["names"].astype(str).tolist(), "cell_id": z["cell_id"].astype(str), "win_start": z["win_start"].astype(np.int64),
                        "on_train_grid": z["on_train_grid"].astype(bool), "is_piece": z["is_piece"].astype(bool), "hop": int(z["hop"]), "L": L,
                        "failures": json.loads(str(z["failures"])) if "failures" in z.files else {}}
    except Exception:                                      # noqa: BLE001
        return None
    return None


def run_feature_unit(out: Path, series_by_cell: dict, cells: list[str], cut: int, L: int, overlap: int, rung: str, force: bool) -> tuple[dict, float, bool]:
    p = features_path(out, cut, L, overlap, rung)
    hop = hop_of(L, overlap)
    if hop is None:
        raise Stop(EXIT_MISSING, f"length {L} with overlap {overlap}% has no whole-pair training start; this (length, overlap) is skipped")
    if not force:
        f = features_stand(p, cells, cut, L, overlap)
        if f is not None:
            return f, 0.0, True
    t0 = time.time()
    f = windows_on_grids(series_by_cell, cells, cut, L, hop, RUNG_KEYS[rung])
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, X=f["X"], names=np.array(f["names"], dtype=str), cell_id=f["cell_id"], win_start=f["win_start"], on_train_grid=f["on_train_grid"], is_piece=f["is_piece"],
                        hop=np.int64(hop), L=np.int64(L), overlap=np.int64(overlap), cut=np.int64(cut), version=np.array(FEATURES_VERSION), rung=np.array(rung),
                        failures=np.array(json.dumps(f["failures"])), written_at=np.array(now_iso()))
    return f, round(time.time() - t0, 1), False


# ---------------------------------------------------------------------------------------------
# the scores with weights (a run tested twice in a repeat counts once) and the bootstrap over recordings
# ---------------------------------------------------------------------------------------------
def score_weighted(pl: list[dict], classes_train: list[str], kernel_of_cell: dict, majority_class: str) -> dict:
    """new_blocks.score_pools with a weight per piece (1 / the times its run was tested in the repeat); unit weights give
    score_pools's numbers exactly."""
    n = len(pl)
    if n == 0:
        return {"n_pools": 0, "accuracy": None, "macro_recall": None, "chance": (1.0 / len(classes_train)) if classes_train else None, "majority": None,
                "n_classes": len(classes_train), "recall_per_class": {}, "recall_per_kernel": {}, "weight_sum": 0.0}
    correct = np.array([p["y_true"] == p["y_pred"] for p in pl], dtype=float)
    w = np.array([float(p.get("weight", 1.0)) for p in pl], dtype=float)
    rpc = {}
    for cls in sorted({p["y_true"] for p in pl}):
        m = np.array([p["y_true"] == cls for p in pl], dtype=bool)
        rpc[cls] = float((w[m] * correct[m]).sum() / w[m].sum())
    rpk = {}
    for k in sorted({kernel_of_cell[p["cell_id"]] for p in pl}):
        m = np.array([kernel_of_cell[p["cell_id"]] == k for p in pl], dtype=bool)
        rpk[k] = float((w[m] * correct[m]).sum() / w[m].sum())
    maj = np.array([p["y_true"] == majority_class for p in pl], dtype=float)
    return {"n_pools": n, "accuracy": float((w * correct).sum() / w.sum()), "macro_recall": float(np.mean(list(rpc.values()))), "chance": 1.0 / len(classes_train),
            "majority": float((w * maj).sum() / w.sum()), "n_classes": len(classes_train), "recall_per_class": rpc, "recall_per_kernel": rpk, "weight_sum": float(w.sum())}


def bootstrap_margin_weighted(pl_b: list[dict], pl_a: list[dict], cells: list[str], n_boot: int, seed: int) -> dict:
    """new_blocks.bootstrap_margin with the pieces paired by (recording, round, position) and weighted; unit weights and
    one test per recording give bootstrap_margin's numbers exactly (the same resamples)."""
    key = lambda p: (p["cell_id"], p.get("round", 0), p["start_pos"])     # noqa: E731
    a = {key(p): p for p in pl_a}
    common = [p for p in pl_b if key(p) in a]
    if not common:
        return {"delta": None, "ci_lo": None, "ci_hi": None, "n_pools": 0, "n_recordings": 0, "gained": 0, "lost": 0, "status": "not run: no common piece"}
    cb = np.array([p["y_true"] == p["y_pred"] for p in common], dtype=float)
    ca = np.array([a[key(p)]["y_true"] == a[key(p)]["y_pred"] for p in common], dtype=float)
    w = np.array([float(p.get("weight", 1.0)) for p in common], dtype=float)
    cid = np.array([p["cell_id"] for p in common], dtype=str)
    present = [c for c in cells if np.any(cid == c)]
    idx = {c: np.flatnonzero(cid == c) for c in present}
    nb = np.array([(w[idx[c]] * cb[idx[c]]).sum() for c in present]); na = np.array([(w[idx[c]] * ca[idx[c]]).sum() for c in present]); nn = np.array([w[idx[c]].sum() for c in present], dtype=float)
    delta = float((w * cb).sum() / w.sum() - (w * ca).sum() / w.sum())
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(present), size=(int(n_boot), len(present)))
    tot = nn[draws].sum(axis=1)
    d = (nb[draws].sum(axis=1) - na[draws].sum(axis=1)) / np.where(tot > 0, tot, 1.0)
    return {"delta": delta, "ci_lo": float(np.quantile(d, 0.025)), "ci_hi": float(np.quantile(d, 0.975)), "n_pools": int(cb.size), "n_recordings": len(present),
            "gained": int(np.sum((cb == 1) & (ca == 0))), "lost": int(np.sum((cb == 0) & (ca == 1))), "n_resamples": int(n_boot), "seed": int(seed), "status": "ok"}


# ---------------------------------------------------------------------------------------------
# the model units: one (cut, size, overlap, repeat, length, rung), all the rounds of the repeat, both levels
# ---------------------------------------------------------------------------------------------
def unit_dir(out: Path, cut: int, size: int, overlap: int, repeat: int, L: int, rung: str, check: bool = False) -> Path:
    base = out / "trainsizes" / "check_move21" if check else out / "trainsizes"
    return base / f"cut{cut}" / f"size{size}" / f"ov{overlap}" / f"rep{repeat}" / f"len{L}" / rung


def unit_signature(*, cut: int, size: int, overlap: int, repeat: int, L: int, rung: str, rd: dict, seed_forest: int, n_est: int, features_sha: str, n_recordings: int, check: bool) -> dict:
    return {"cut": int(cut), "size": int(size), "overlap": int(overlap), "repeat": int(repeat), "length": int(L), "rung": rung, "rounds_hash": rounds_hash(rd), "n_rounds": rd["n_rounds"],
            "seed_forest": int(seed_forest), "n_estimators": int(n_est), "features_sha256": features_sha, "features_version": FEATURES_VERSION, "levels": list(LEVELS),
            "n_recordings": int(n_recordings), "check": bool(check), "code_fingerprint": code_identity()["fingerprint"]}


def unit_stands(d: Path, sig: dict) -> dict | None:
    p = d / "result.json"
    if not p.is_file():
        return None
    try:
        r = read_json(p)
    except Exception:                                      # noqa: BLE001
        return None
    return r if r.get("signature") == sig and all((d / f"predictions_{lv}.csv").is_file() for lv in LEVELS) else None


def read_pieces(d: Path, level: str) -> list[dict]:
    p = d / f"predictions_{level}.csv"
    if not p.is_file():
        return []
    with open(p, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return [{"cell_id": r["cell_id"], "round": int(r["round"]), "start_pos": int(r["position"]), "size": 1, "win_start": int(r["win_start"]), "y_true": r["y_true"], "y_pred": r["y_pred"],
             "weight": float(r["weight"])} for r in rows]


def run_model_unit(d: Path, f: dict, rd: dict, runs_by_id: dict, classes: dict, majority: dict, kernel_of: dict, *, cut: int, L: int, rung: str, n_jobs: int, n_est: int,
                   seed_forest: int, sig: dict, tag: str) -> dict:
    d.mkdir(parents=True, exist_ok=True)
    t_unit = time.time()
    X_all, names_all, cell_rows, starts = f["X"], f["names"], f["cell_id"], f["win_start"]
    on_train, is_piece = f["on_train_grid"], f["is_piece"]
    finite = np.isfinite(X_all).all(axis=0)
    left_out = [n for n, ok in zip(names_all, finite) if not ok]
    keep = np.flatnonzero(finite)
    X, names = X_all[:, keep], [names_all[j] for j in keep]
    if X.shape[1] == 0:
        raise Stop(EXIT_ERROR, f"stopped: every statistic of {rung} is non-finite at length {L} (cut {cut}); nothing to train on")
    weight_of = {c: 1.0 / n for c, n in rd["times_held"].items() if n}
    res = {"schema": "plan12.trainsizes_unit.v1", "citation": CITATION, "signature": sig, "rung": rung, "keys": list(RUNG_KEYS[rung]), "text": RUNG_TEXT[rung], "cut": int(cut), "length": L,
           "hop": int(f["hop"]), "n_windows": int(X.shape[0]), "n_train_grid_windows": int(on_train.sum()), "n_features": len(names), "left_out": left_out,
           "statistic_sets_that_raised": f.get("failures") or {}, "rounds": [{k: v for k, v in ro.items() if k != "train"} for ro in rd["rounds"]], "n_rounds": rd["n_rounds"],
           "leave_one_out": rd["leave_one_out"], "fixed_rounds": rd["fixed"], "levels": {}}
    for level in LEVELS:
        y = np.array([runs_by_id[c][f"label_{level}"] for c in cell_rows.tolist()], dtype=str)
        pieces, round_recs = [], []
        t0 = time.time()
        for ro in rd["rounds"]:
            held = set(ro["heldout"])
            in_held = np.isin(cell_rows, list(held))
            tr = np.flatnonzero(~in_held & on_train)
            te = np.flatnonzero(in_held & is_piece)
            if set(cell_rows[tr].tolist()) & held:
                raise Stop(EXIT_ERROR, f"stopped: round {ro['round']}: a held-out recording has a window in training")
            if te.size == 0 or tr.size == 0:
                round_recs.append({"round": ro["round"], "status": "no test piece" if te.size == 0 else "no training window", "n_train_windows": int(tr.size)})
                continue
            pos = np.zeros(te.size, dtype=np.int64)
            for c in held:
                m = cell_rows[te] == c
                pos[m] = np.arange(1, int(m.sum()) + 1)
            sel = np.concatenate([tr, te])
            sp = {"train_idx": np.arange(tr.size), "test_idx": np.arange(tr.size, sel.size), "cell_id_rows": cell_rows[sel],
                  "block_id": np.array([f"{c}|b{k}" for c, k in zip(cell_rows[te].tolist(), pos.tolist())], dtype=str), "block_pos": pos, "cells": sorted(set(cell_rows[sel].tolist()))}
            fit = NB.fit_level(X[sel], names, y[sel], sp, seed=seed_forest, n_jobs=n_jobs, n_est=n_est)
            use = fit["use"]
            if "pred" not in use:
                round_recs.append({"round": ro["round"], "status": fit["status"], "n_train_windows": int(tr.size), "quarantined_features": fit["quarantined_features"]})
                continue
            ci = {c: i for i, c in enumerate(use["classes"])}
            for i, j in enumerate(te.tolist()):
                c = str(cell_rows[j])
                pieces.append({"cell_id": c, "round": ro["round"], "start_pos": int(pos[i]), "size": 1, "win_start": int(starts[j]), "y_true": str(y[j]), "y_pred": str(use["pred"][i]),
                               "vote_fraction": float(use["proba"][i].max()), "p_true": float(use["proba"][i][ci[str(y[j])]]) if str(y[j]) in ci else None, "weight": weight_of.get(c, 1.0)})
            round_recs.append({"round": ro["round"], "status": "ok", "n_train_windows": int(tr.size), "n_train_cells": fit["n_train_cells"], "n_test_pieces": int(te.size),
                               "feature_count": use["feature_count"], "feature_count_used": use["feature_count_used"], "reduced": use["reduced"], "score_source": fit["score_source"],
                               "quarantined_features": fit["quarantined_features"], "classes_train": fit["classes_train"],
                               "accuracy_this_round": float(np.mean([p["y_true"] == p["y_pred"] for p in pieces if p["round"] == ro["round"]]))})
        sc = score_weighted(pieces, classes[level], kernel_of, majority[level])
        res["levels"][level] = {"status": "ok" if pieces else "not run: no test piece", "scores": {k: v for k, v in sc.items() if k != "recall_per_class"}, "rounds": round_recs,
                                "n_pieces": len(pieces), "n_pieces_weighted": sc.get("weight_sum"), "quarantined_union": sorted({q for rr in round_recs for q in (rr.get("quarantined_features") or [])}),
                                "elapsed_s": round(time.time() - t0, 1)}
        write_csv(d / f"predictions_{level}.csv", ["cell_id", "kernel", "round", "position", "win_start", "pair_start", "pair_end", "y_true", "y_pred", "p_pred", "p_true", "weight"],
                  [[p["cell_id"], kernel_of[p["cell_id"]], p["round"], p["start_pos"], p["win_start"], int(runs_by_id[p["cell_id"]]["_pair0"][cut] + p["win_start"]),
                    int(runs_by_id[p["cell_id"]]["_pair0"][cut] + p["win_start"] + L - 1), p["y_true"], p["y_pred"], p["vote_fraction"], p["p_true"], p["weight"]] for p in pieces])
        print(f"[trainsizes] {tag} {level:<9} acc {fmt(sc.get('accuracy'))} macro {fmt(sc.get('macro_recall'))} chance {fmt(sc.get('chance'))} majority {fmt(sc.get('majority'))} | "
              f"{len(pieces)} pieces, {rd['n_rounds']} rounds, {len(names)} features ({len(left_out)} left out) ({res['levels'][level]['elapsed_s']} s)", flush=True)
    res["elapsed_s"] = round(time.time() - t_unit, 1)
    res["written_at"] = now_iso()
    write_json(d / "result.json", res)
    return res


# ---------------------------------------------------------------------------------------------
# the plan: the units of the choices made
# ---------------------------------------------------------------------------------------------
def make_plan(cuts: dict, ch: dict, rungs: list[str], groups: dict) -> dict:
    """The feature units first (cut, length, overlap, rung), then the model units (cut, size, overlap, repeat, length, rung),
    each with the index of the feature unit it reads; the (length, overlap) pairs skipped; the rounds per size."""
    units, skipped, fidx = [], [], {}
    for L in ch["lengths"]:
        for ov in ch["overlaps"]:
            if hop_of(L, ov) is None:
                skipped.append({"length": L, "overlap": ov, "why": f"no whole-pair training start: {100 - ov}% of {L} pairs is not a whole number"})
                continue
            for cn, cut in cuts.items():
                for rung in rungs:
                    fidx[(cut, L, ov, rung)] = len(units)
                    units.append({"index": len(units), "kind": "features", "cut": cut, "cut_name": cn, "size": None, "overlap": ov, "repeat": None, "length": L, "rung": rung, "needs": None,
                                  "file": f"trainsizes/cut{cut}/len{L}/ov{ov}/features/{rung}.npz"})
    per_size = {}
    for size in ch["sizes"]:
        reps = repeats_of(size, groups, ch)
        probe = make_rounds(groups, size, ch["base_seed"], 1)
        per_size[size] = {"repeats": reps, "n_rounds": probe["n_rounds"], "leave_one_out": probe["leave_one_out"], "n_train_total": probe["n_train_total"],
                          "capped_groups": sorted(g for g, v in probe["groups"].items() if v["capped"])}
        for ov in ch["overlaps"]:
            for rep in reps:
                for L in ch["lengths"]:
                    if hop_of(L, ov) is None:
                        continue
                    for rung in rungs:
                        for cn, cut in cuts.items():
                            units.append({"index": len(units), "kind": "model", "cut": cut, "cut_name": cn, "size": size, "overlap": ov, "repeat": rep, "length": L, "rung": rung,
                                          "needs": fidx[(cut, L, ov, rung)], "file": f"trainsizes/cut{cut}/size{size}/ov{ov}/rep{rep}/len{L}/{rung}/result.json"})
    return {"schema": "plan12.trainsizes_plan.v1", "citation": CITATION, "choices": ch, "cuts": cuts, "rungs": rungs, "units": units, "skipped": skipped, "per_size": per_size,
            "n_units": {"features": sum(1 for u in units if u["kind"] == "features"), "model": sum(1 for u in units if u["kind"] == "model")}}


def unit_done(out: Path, u: dict) -> bool:
    p = out / u["file"]
    return p.is_file()


def write_plan(out: Path, plan: dict) -> None:
    (out / "trainsizes").mkdir(parents=True, exist_ok=True)
    write_json(out / "trainsizes" / "plan.json", {**plan, "written_at": now_iso()})
    write_csv(out / "trainsizes" / "plan.csv", ["index", "kind", "cut", "size", "overlap", "repeat", "length", "rung", "needs", "file", "done"],
              [[u["index"], u["kind"], u["cut"], u["size"], u["overlap"], u["repeat"], u["length"], u["rung"], u["needs"], u["file"], unit_done(out, u)] for u in plan["units"]])
    write_csv(out / "trainsizes" / "skipped.csv", ["length", "overlap", "why"], [[s["length"], s["overlap"], s["why"]] for s in plan["skipped"]])


def write_rounds(out: Path, groups: dict, ch: dict, sizes: list[int]) -> None:
    rows = []
    for size in sizes:
        for rep in repeats_of(size, groups, ch):
            rd = make_rounds(groups, size, ch["base_seed"], rep)
            write_json(out / "trainsizes" / "rounds" / f"size{size}_rep{rep}.json", {"schema": "plan12.trainsizes_rounds.v1", "citation": CITATION, **rd})
            for ro in rd["rounds"]:
                for g in sorted(groups):
                    pg = rd["groups"][g]
                    rows.append([size, rep, ro["round"], g, pg["n_runs"], pg["m_used"], pg["capped"], " ".join(ro["per_group"][g]["heldout"]), ro["per_group"][g]["block"],
                                 ro["per_group"][g]["cycled"], " ".join(pg["filled_from_earlier"]) if ro["per_group"][g]["block"] == len(pg["blocks"]) - 1 else "", rd["n_rounds"],
                                 rd["leave_one_out"]])
    write_csv(out / "trainsizes" / "rounds.csv", ["size", "repeat", "round", "group", "n_runs", "m_used", "capped", "heldout_runs", "block", "cycled", "filled_from_earlier",
                                                  "n_rounds", "leave_one_out"], rows)


# ---------------------------------------------------------------------------------------------
# the estimate: from move 21's measured step times
# ---------------------------------------------------------------------------------------------
def estimate(cuts: dict, ch: dict, rungs: list[str], groups: dict, ref_train: float) -> dict:
    """Seconds per (size, overlap, repeat, length) and in total, from move 21's measured step times: a model unit costs
    move 21's forest time per rung and fold, times the rounds, times a mix of the training rows relative to move 21's (the
    runs in training over move 21's, the windows per run over move 21's grid; the share W_FIT of the cost) and the held-out
    runs per round relative to move 21's (the rest); a feature unit costs move 21's feature time per rung times the windows
    ratio. Length 2's measured cut is extrapolated."""
    plan = make_plan(cuts, ch, rungs, groups)
    per_cut_name = {}
    for cn in cuts:
        per_cut_name[cn] = "measured" if cn == "measured" else "declared"
    n_runs = sum(len(v) for v in groups.values())
    ref_held = max(n_runs - ref_train, 1.0)
    rows, total_feat, total_model = [], 0.0, 0.0
    for u in plan["units"]:
        cn = per_cut_name.get(u["cut_name"], "declared")
        m21 = MEASURED_21[u["length"]]
        if u["kind"] == "features":
            s = m21["features_s"][cn] / N_RUNGS_21 * grid_ratio(u["overlap"])
            total_feat += s
        else:
            ps = plan["per_size"][u["size"]]
            w = W_FIT[u["length"]]
            rows_ratio = (ps["n_train_total"] / max(ref_train, 1.0)) * grid_ratio(u["overlap"])
            pieces_ratio = (n_runs - ps["n_train_total"]) / ref_held
            s = m21["folds_s"][cn] / (N_RUNGS_21 * N_FOLDS_21) * ps["n_rounds"] * (w * rows_ratio + (1.0 - w) * pieces_ratio)
            total_model += s
        u["seconds_estimated"] = round(s, 1)
    by = {}
    for u in plan["units"]:
        if u["kind"] != "model":
            continue
        k = (u["size"], u["overlap"], u["repeat"], u["length"])
        by[k] = by.get(k, 0.0) + u["seconds_estimated"]
    for k, s in sorted(by.items()):
        rows.append({"size": k[0], "overlap": k[1], "repeat": k[2], "length": k[3], "seconds": round(s, 1)})
    by_size = {}
    for u in plan["units"]:
        if u["kind"] == "model":
            by_size[u["size"]] = by_size.get(u["size"], 0.0) + u["seconds_estimated"]
    return {"schema": "plan12.trainsizes_estimate.v1", "basis": "move 21's measured step times on the real run (2026-10-13; length 2 measured at the declared cut only), scaled by the "
                                                                "rounds, the training rows (the runs in training, the windows per run) and the held-out runs per round, in the shares one "
                                                                "measurement with two runs per group showed; one process at a time, move 6's trees and processes; a rough guide, not a promise",
            "measured_21": MEASURED_21, "w_fit": W_FIT,
            "ref_train_runs": ref_train, "choices": ch, "n_units": plan["n_units"], "skipped": plan["skipped"], "per_size": {str(k): v for k, v in plan["per_size"].items()},
            "seconds": {"features": round(total_feat, 1), "model": round(total_model, 1), "total": round(total_feat + total_model, 1)},
            "seconds_by_size": {str(k): round(v, 1) for k, v in by_size.items()}, "by_size_overlap_repeat_length": rows,
            "per_unit_seconds": [u["seconds_estimated"] for u in plan["units"]],
            "lengths_not_measured": [L for L in ch["lengths"] if not MEASURED_21[L]["measured"]]}


# ---------------------------------------------------------------------------------------------
# the collect: tables and figures across the units present
# ---------------------------------------------------------------------------------------------
def _f(x):
    try:
        return float(x) if x not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _stats(vals: list[float]) -> dict:
    v = [x for x in vals if x is not None]
    if not v:
        return {"mean": None, "min": None, "max": None, "sd": None, "n": 0}
    return {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)), "sd": (float(np.std(v, ddof=1)) if len(v) > 1 else None), "n": len(v)}


def find_units(out: Path, cut: int) -> list[dict]:
    """Every model unit result present under trainsizes/cut<C>/size*/ov*/rep*/len*/<rung>/result.json, with its parameters."""
    base = out / "trainsizes" / f"cut{cut}"
    found = []
    if not base.is_dir():
        return found
    for p in sorted(base.glob("size*/ov*/rep*/len*/*/result.json")):
        try:
            r = read_json(p)
        except Exception:                                  # noqa: BLE001
            continue
        s = r.get("signature") or {}
        if s.get("check"):
            continue
        found.append({"path": p, "dir": p.parent, "size": int(s["size"]), "overlap": int(s["overlap"]), "repeat": int(s["repeat"]), "length": int(s["length"]), "rung": s["rung"], "result": r})
    return found


def accuracy_vs_trainsize_svg(agg: list[dict], cut: int, level: str, lengths: list[int], overlaps: list[int], rungs: list[str]) -> str:
    """Accuracy on held-out pieces against the number of training runs: one panel per piece length (columns) and overlap
    (rows), one line per rung (the mean over repeats), the spread over repeats (min to max) as a band."""
    pw, ph, gx, gy, x0, y0 = 170, 120, 22, 44, 60, 60
    ncol, nrow = max(1, len(lengths)), max(1, len(overlaps))
    W_, H_ = x0 + ncol * (pw + gx) + 150, y0 + nrow * (ph + gy) + 30
    out = svg_open(W_, H_, f"The training-size loops: accuracy against the number of training runs, the {level} level, cut of {cut} pairs",
                   "every recording held out whole in random rounds; the line is the mean over repeats, the band the min to max; one panel per piece length (columns) and overlap (rows)")
    chance = next((r["chance"] for r in agg if r["level"] == level and r.get("chance") is not None), None)
    majority = next((r["majority"] for r in agg if r["level"] == level and r.get("majority") is not None), None)
    for oi, ov in enumerate(overlaps):
        for li, L in enumerate(lengths):
            px, py = x0 + li * (pw + gx), y0 + oi * (ph + gy)
            out.append(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" fill="none" stroke="#ccc"/>')
            out.append(f'<text x="{px}" y="{py - 6}" font-size="10" font-weight="bold">pieces of {L} pairs, overlap {ov}%</text>')
            for t in (0.0, 0.5, 1.0):
                yy = py + ph - ph * t
                out.append(f'<line x1="{px}" y1="{yy:.1f}" x2="{px + pw}" y2="{yy:.1f}" stroke="#eee"/>')
                if li == 0:
                    out.append(f'<text x="{px - 4}" y="{yy + 3:.1f}" text-anchor="end" font-size="9">{t:.1f}</text>')
            xs = {m: px + 12 + (pw - 24) * (m - 1) / 6 for m in SIZES}
            for m in SIZES:
                out.append(f'<text x="{xs[m]:.1f}" y="{py + ph + 12}" text-anchor="middle" font-size="9">{m}</text>')
            if chance is not None:
                out.append(f'<line x1="{px}" y1="{py + ph - ph * chance:.1f}" x2="{px + pw}" y2="{py + ph - ph * chance:.1f}" stroke="#888"/>')
            if majority is not None:
                out.append(f'<line x1="{px}" y1="{py + ph - ph * majority:.1f}" x2="{px + pw}" y2="{py + ph - ph * majority:.1f}" stroke="#b03a2e" stroke-dasharray="1,2"/>')
            for rung in rungs:
                pts = sorted([(r["size"], r["accuracy_mean"], r["accuracy_min"], r["accuracy_max"]) for r in agg
                              if r["level"] == level and r["rung"] == rung and r["length"] == L and r["overlap"] == ov and r.get("accuracy_mean") is not None])
                if not pts:
                    continue
                colour = RUNG_COLOUR.get(rung, "#333")
                if len(pts) > 1:
                    band = " ".join(f"{xs[m]:.1f},{py + ph - ph * hi:.1f}" for m, _, lo, hi in pts) + " " + " ".join(f"{xs[m]:.1f},{py + ph - ph * lo:.1f}" for m, _, lo, hi in reversed(pts))
                    out.append(f'<polygon points="{band}" fill="{colour}" fill-opacity="0.12" stroke="none"/>')
                    line = " ".join(f"{xs[m]:.1f},{py + ph - ph * a:.1f}" for m, a, _, _ in pts)
                    dash = ' stroke-dasharray="4,3"' if rung == "rung0" else ""
                    out.append(f'<polyline points="{line}" fill="none" stroke="{colour}" stroke-width="{2.0 if rung == "rung0" else 1.4}"{dash}/>')
                for m, a, lo, hi in pts:
                    out.append(f'<circle cx="{xs[m]:.1f}" cy="{py + ph - ph * a:.1f}" r="2.2" fill="{colour}"/>')
                    if lo != hi:
                        out.append(f'<line x1="{xs[m]:.1f}" y1="{py + ph - ph * lo:.1f}" x2="{xs[m]:.1f}" y2="{py + ph - ph * hi:.1f}" stroke="{colour}" stroke-width="0.8"/>')
            if oi == nrow - 1:
                out.append(f'<text x="{px + pw / 2:.1f}" y="{py + ph + 26}" text-anchor="middle" fill="#555" font-size="9">training runs per group</text>')
    lx = x0 + ncol * (pw + gx) + 6
    for ri, rung in enumerate(rungs):
        out.append(f'<text x="{lx}" y="{y0 + 12 + ri * 13}" font-size="9" fill="{RUNG_COLOUR.get(rung, "#333")}">{rung.replace("rung", "r")} {RUNG_TEXT[rung].split(":")[0][:22]}</text>')
    out.append(f'<text x="{lx}" y="{y0 + 12 + len(rungs) * 13 + 4}" font-size="9" fill="#888">grey: chance 1/K</text>')
    out.append(f'<text x="{lx}" y="{y0 + 12 + len(rungs) * 13 + 17}" font-size="9" fill="#b03a2e">dotted red: the majority class</text>')
    out.append(f'<text x="{x0}" y="{H_ - 10}" fill="#555" font-size="9">r0 N,H,A; r1 H; r2 H,A; r3 C,S; r4 doubled angle; r5a l0 arrow; r5b l1 arrow; the band is the spread over repeats; {NO_NULL[:60]}...</text></svg>')
    return "\n".join(out)


def collect_cut(out: Path, cut_name: str, cut: int, runs: list[dict], rungs: list[str], n_boot: int, quiet: bool) -> dict:
    d = out / "trainsizes" / f"cut{cut}"
    units = find_units(out, cut)
    kernel_of = {r["cell_id"]: r["label_kernel"] for r in runs}
    cells = [r["cell_id"] for r in runs]
    if not units:
        return {"cut": cut, "n_units": 0}
    d.mkdir(parents=True, exist_ok=True)
    # per repeat
    rep_rows, rk_acc, lo_rows, lo_seen = [], {}, [], set()
    for u in units:
        r = u["result"]
        for lv in LEVELS:
            s = (r["levels"].get(lv) or {}).get("scores") or {}
            rr = (r["levels"].get(lv) or {}).get("rounds") or []
            ok_r = [x for x in rr if x.get("status") == "ok"]
            rep_rows.append([cut, u["size"], u["overlap"], u["repeat"], u["length"], u["rung"], RUNG_TEXT[u["rung"]], lv, (r["levels"].get(lv) or {}).get("status"), s.get("n_pools"),
                             s.get("weight_sum"), s.get("n_classes"), s.get("accuracy"), s.get("macro_recall"), s.get("chance"), s.get("majority"), r["n_features"], len(r["left_out"]),
                             r["n_rounds"], (float(np.mean([x["n_train_windows"] for x in ok_r])) if ok_r else None),
                             ";".join(f"{x['round']}:{x['accuracy_this_round']:.4f}" for x in ok_r if x.get("accuracy_this_round") is not None),
                             ";".join((r["levels"].get(lv) or {}).get("quarantined_union") or []), r.get("elapsed_s"), NO_NULL])
            for kern, rec in (s.get("recall_per_kernel") or {}).items():
                rk_acc.setdefault((u["size"], u["overlap"], u["length"], u["rung"], lv, kern), []).append(rec)
        key = (u["length"], u["overlap"], u["rung"])
        if key not in lo_seen:
            lo_seen.add(key)
            raised = r.get("statistic_sets_that_raised") or {}
            lo_rows.append([cut, u["length"], u["overlap"], u["rung"], len(r["left_out"]), r["n_features"], ";".join(r["left_out"]),
                            "; ".join(f"{k}: {v['n_windows']} windows ({v['first_error']})" for k, v in raised.items())])
    write_csv(d / "scores_per_repeat.csv", ["cut", "size", "overlap", "repeat", "length", "rung", "input_set", "level", "status", "n_pieces", "n_pieces_weighted", "n_classes", "accuracy",
                                            "macro_recall", "chance", "majority", "n_features", "n_left_out", "n_rounds", "n_train_windows_mean", "accuracy_per_round",
                                            "quarantined_features_union", "unit_elapsed_s", "null"], rep_rows)
    write_csv(d / "left_out.csv", ["cut", "length", "overlap", "rung", "n_left_out", "n_kept", "statistics_left_out", "statistic_sets_that_raised"], sorted(lo_rows, key=lambda r: (-r[1], r[2], r[3])))
    # the aggregate over repeats
    groups_: dict = {}
    for row in rep_rows:
        k = (row[1], row[2], row[4], row[5], row[7])    # size, overlap, length, rung, level
        groups_.setdefault(k, []).append(row)
    agg_rows, agg = [], []
    for (size, ov, L, rung, lv), rows in sorted(groups_.items(), key=lambda kv: (kv[0][4], -kv[0][2], kv[0][1], kv[0][0], RUNG_ORDER.index(kv[0][3]) if kv[0][3] in RUNG_ORDER else 99)):
        acc, mac = _stats([_f(r[12]) for r in rows]), _stats([_f(r[13]) for r in rows])
        reps = sorted(int(r[3]) for r in rows)
        rec = {"cut": cut, "size": size, "overlap": ov, "length": L, "rung": rung, "level": lv, "n_repeats": len(rows), "repeats": reps, "accuracy_mean": acc["mean"], "accuracy_min": acc["min"],
               "accuracy_max": acc["max"], "accuracy_sd": acc["sd"], "macro_mean": mac["mean"], "macro_min": mac["min"], "macro_max": mac["max"], "macro_sd": mac["sd"],
               "chance": _f(rows[0][14]), "majority": _f(rows[0][15]), "n_rounds": rows[0][18], "n_features": rows[0][16], "n_left_out": rows[0][17],
               "accuracy_per_repeat": ";".join(f"{int(r[3])}:{_f(r[12]):.4f}" for r in rows if _f(r[12]) is not None)}
        agg.append(rec)
        agg_rows.append([cut, size, ov, L, rung, RUNG_TEXT[rung], lv, len(rows), " ".join(map(str, reps)), acc["mean"], acc["min"], acc["max"], acc["sd"], mac["mean"], mac["min"], mac["max"],
                         mac["sd"], rec["chance"], rec["majority"], rec["n_rounds"], rec["n_features"], rec["n_left_out"], rec["accuracy_per_repeat"], NO_NULL])
    write_csv(d / "scores.csv", ["cut", "size", "overlap", "length", "rung", "input_set", "level", "n_repeats", "repeats", "accuracy_mean", "accuracy_min", "accuracy_max", "accuracy_sd",
                                 "macro_recall_mean", "macro_recall_min", "macro_recall_max", "macro_recall_sd", "chance", "majority", "n_rounds", "n_features", "n_left_out",
                                 "accuracy_per_repeat", "null"], agg_rows)
    rk_rows = []
    for (size, ov, L, rung, lv, kern), vals in sorted(rk_acc.items(), key=lambda kv: (kv[0][4], -kv[0][2], kv[0][1], kv[0][0], kv[0][3], kv[0][5])):
        st = _stats(vals)
        rk_rows.append([cut, size, ov, L, rung, lv, kern, st["n"], st["mean"], st["min"], st["max"], st["sd"]])
    write_csv(d / "recall_per_kernel.csv", ["cut", "size", "overlap", "length", "rung", "level", "kernel", "n_repeats", "recall_mean", "recall_min", "recall_max", "recall_sd"], rk_rows)
    # the margins between rungs: per repeat (move 12's bootstrap over recordings, weighted), then the mean over repeats
    by_unit = {(u["size"], u["overlap"], u["repeat"], u["length"], u["rung"]): u for u in units}
    mrep_rows: list = []
    magg: dict = {}
    keys_sorl = sorted({(u["size"], u["overlap"], u["repeat"], u["length"]) for u in units})
    for size, ov, rep, L in keys_sorl:
        for lv in LEVELS:
            for b, a in LADDER_PAIRS:
                ub, ua = by_unit.get((size, ov, rep, L, b)), by_unit.get((size, ov, rep, L, a))
                if ub is None or ua is None:
                    continue
                pb, pa = read_pieces(ub["dir"], lv), read_pieces(ua["dir"], lv)
                comp = f"{b}-{a}"
                if not pb or not pa:
                    mrep_rows.append([cut, size, ov, rep, L, lv, comp, None, None, None, None, None, None, None, "not run: no piece"])
                    continue
                bm = bootstrap_margin_weighted(pb, pa, cells, n_boot, SEED_BOOTSTRAP)
                mrep_rows.append([cut, size, ov, rep, L, lv, comp, bm["delta"], bm["ci_lo"], bm["ci_hi"], bm["n_pools"], bm["n_recordings"], bm["gained"], bm["lost"], bm["status"]])
                magg.setdefault((size, ov, L, lv, comp), []).append((rep, bm["delta"], bm["ci_lo"], bm["ci_hi"]))
    write_csv(d / "margins_per_repeat.csv", ["cut", "size", "overlap", "repeat", "length", "level", "comparison", "delta_accuracy", "ci95_lo", "ci95_hi", "n_pieces", "n_recordings",
                                             "n_pieces_gained", "n_pieces_lost", "status"], mrep_rows)
    magg_rows = []
    for (size, ov, L, lv, comp), vals in sorted(magg.items(), key=lambda kv: (kv[0][3], -kv[0][2], kv[0][1], kv[0][0], kv[0][4])):
        st = _stats([v[1] for v in vals])
        magg_rows.append([cut, size, ov, L, lv, comp, st["n"], st["mean"], st["min"], st["max"], st["sd"], ";".join(f"{v[0]}:{v[1]:+.4f} [{v[2]:+.4f}, {v[3]:+.4f}]" for v in sorted(vals))])
    write_csv(d / "margins.csv", ["cut", "size", "overlap", "length", "level", "comparison", "n_repeats", "delta_mean", "delta_min", "delta_max", "delta_sd", "per_repeat_delta_ci95"], magg_rows)
    # the figures
    lengths = sorted({u["length"] for u in units}, reverse=True)
    overlaps = sorted({u["overlap"] for u in units})
    for lv in LEVELS:
        (d / f"accuracy_vs_trainsize_{lv}.svg").write_text(accuracy_vs_trainsize_svg(agg, cut, lv, lengths, overlaps, rungs))
        write_csv(d / f"accuracy_vs_trainsize_{lv}.csv", ["cut", "size", "overlap", "length", "rung", "n_repeats", "accuracy_mean", "accuracy_min", "accuracy_max", "accuracy_sd", "chance", "majority"],
                  [[cut, r["size"], r["overlap"], r["length"], r["rung"], r["n_repeats"], r["accuracy_mean"], r["accuracy_min"], r["accuracy_max"], r["accuracy_sd"], r["chance"], r["majority"]]
                   for r in agg if r["level"] == lv])
    summ = {"cut": cut, "cut_name": cut_name, "n_units": len(units), "sizes": sorted({u["size"] for u in units}), "overlaps": overlaps, "lengths": lengths,
            "repeats": sorted({u["repeat"] for u in units}), "rungs": sorted({u["rung"] for u in units}, key=lambda r: RUNG_ORDER.index(r) if r in RUNG_ORDER else 99)}
    if not quiet:
        print(f"[trainsizes] collect cut {cut}: {len(units)} model units; sizes {summ['sizes']}; overlaps {overlaps}; lengths {lengths}; repeats {summ['repeats']}", flush=True)
    return summ


def collect_all(out: Path, cuts: dict, runs: list[dict], rungs: list[str], n_boot: int, quiet: bool = False) -> dict:
    per_cut = {cn: collect_cut(out, cn, cut, runs, rungs, n_boot, quiet) for cn, cut in cuts.items()}
    summ = {"schema": "plan12.trainsizes_summary.v1", "citation": CITATION, "written_at": now_iso(), "cuts": cuts, "round_rule": ROUND_RULE, "piece_rule": PIECE_RULE, "left_out_rule": LEFT_OUT_RULE,
            "levels": {lv: HL.LEVEL_TEXT[lv] for lv in LEVELS}, "null": NO_NULL, "per_cut": per_cut, "n_units_total": sum(v.get("n_units", 0) for v in per_cut.values())}
    write_json(out / "trainsizes" / "summary.json", summ)
    return summ


# ---------------------------------------------------------------------------------------------
# the steps
# ---------------------------------------------------------------------------------------------
def prepare(o: argparse.Namespace, need_series: bool = True) -> dict:
    run = Path(os.path.expanduser(o.run)).resolve()
    if not (run / "cells.csv").is_file():
        raise Stop(EXIT_MISSING, f"missing input: {run / 'cells.csv'} (--run must name a plan12_grounding output folder with moves 0, 1 and 6 done)")
    out = resolve_out(run, o.out)
    al_out = Path(os.path.expanduser(o.arrowladder_out)).resolve() if getattr(o, "arrowladder_out", None) else AL.default_out(run)
    ho_out = Path(os.path.expanduser(o.heldout_out)).resolve() if getattr(o, "heldout_out", None) else HL.default_out(run)
    lock = refuse_if_busy(run, al_out, ho_out, o.dry_run)
    m6 = I.load_move6(run, getattr(o, "encoding_out", None))
    cuts = I.parse_cuts(getattr(o, "cuts", None), m6["cuts"])
    runs = HL.labelled_runs(run)
    groups = group_runs(runs)
    rungs = [s.strip() for s in str(o.rungs).split(",") if s.strip()] if getattr(o, "rungs", None) else list(RUNG_ORDER)
    bad = [r for r in rungs if r not in RUNG_KEYS]
    if bad:
        raise Stop(EXIT_MISSING, f"--rungs {bad}: the rungs are {', '.join(RUNG_ORDER)}")
    n_jobs = int(o.n_jobs) if getattr(o, "n_jobs", None) is not None else m6["n_jobs"]
    n_est = int(o.n_estimators) if getattr(o, "n_estimators", None) is not None else m6["n_estimators"]
    seed21 = SEED_FOREST + m6["seed_offset"]
    ch = choices_of(o, seed21)
    missing = [r["cell_id"] for r in runs if not AL.series_stands(al_out, r["cell_id"], None)] if need_series else []
    for r in runs:
        for cut in cuts.values():
            r.setdefault("_pair0", {})[cut] = int(ST.cut_series(r, cut)["pair"][0])
    return {"run": run, "out": out, "al_out": al_out, "ho_out": ho_out, "lock": lock, "m6": m6, "cuts": cuts, "runs": runs, "groups": groups, "rungs": rungs, "n_jobs": n_jobs,
            "n_est": n_est, "seed21": seed21, "choices": ch, "missing_series": missing, "n_boot": int(getattr(o, "bootstrap", None) or DEFAULT_BOOTSTRAP)}


def labels_of(runs: list[dict]) -> tuple[dict, dict, dict]:
    kernel_of = {r["cell_id"]: r["label_kernel"] for r in runs}
    classes = {lv: sorted({r[f"label_{lv}"] for r in runs}) for lv in LEVELS}
    majority = {}
    for lv in LEVELS:
        counts: dict = {}
        for r in runs:
            counts[r[f"label_{lv}"]] = counts.get(r[f"label_{lv}"], 0) + 1
        majority[lv] = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return kernel_of, classes, majority


def _inputs_sha(P: dict) -> dict:
    run, al_out, runs = P["run"], P["al_out"], P["runs"]
    return {"cells.csv": sha256_file(run / "cells.csv"), "series": {r["cell_id"]: sha256_file(run / "series" / f"{r['cell_id']}.npz") for r in runs if (run / "series" / f"{r['cell_id']}.npz").is_file()},
            "arrow_series": {r["cell_id"]: sha256_file(AL.series_path(al_out, r["cell_id"])) for r in runs if AL.series_path(al_out, r["cell_id"]).is_file()}}


def _entry(key: str, P: dict, params: dict, outputs: list[str]) -> dict:
    code = code_identity()
    return {"key": key, "command": sys.argv, "cwd": str(_HERE.parent), "started_at": now_iso(), "status": "running", "exit_code": None, "finished_at": None, "params": params,
            "inputs_sha256": _inputs_sha(P), "code": {"train_sizes.py": code["sha256"].get("train_sizes.py"), "fingerprint": code["fingerprint"], "modules": code["modules"], "sha256": code["sha256"],
                                                      "toolkit_fingerprint": toolkit_fingerprint()["sha256"]}, "outputs": outputs}


def _print_header(P: dict, what: str) -> None:
    ch = P["choices"]
    n_idle = sum(1 for r in P["runs"] if r["group"] == "idle")
    print(f"[trainsizes] {what}: {len(P['runs'])} recordings ({n_idle} idle) in {len(P['groups'])} groups; cuts {', '.join(f'{k} = {v}' for k, v in P['cuts'].items())}; "
          f"rungs {', '.join(r.replace('rung', 'r') for r in P['rungs'])}; levels {', '.join(LEVELS)}; forest {P['n_est']} trees, {P['n_jobs']} processes; "
          f"sizes {ch['sizes']}, repeats {ch['repeats']}, base seed {ch['base_seed']} (move 21's forest seed {P['seed21']}), lengths {ch['lengths']}, overlaps {ch['overlaps']}%; {NO_NULL[:50]}...")
    print(f"[trainsizes] run {P['run']} (read only); move 20's series {P['al_out']} (read only); move 21's results {P['ho_out']} (read only); out {P['out']}"
          f"{' (exists)' if P['out'].exists() else ' (to be created)'}; driver lock: " + (f"held by pid {P['lock']['pid']} ({'alive' if P['lock']['alive'] else 'dead'})" if P["lock"]["exists"] else "absent"))


def _fail(entry: dict, exc: BaseException) -> None:
    code_ = getattr(exc, "code", None)
    entry.update(status=f"failed: {type(exc).__name__}" + (f" (exit {code_})" if isinstance(code_, int) else ""), exit_code=code_ if isinstance(code_, int) else EXIT_ERROR,
                 finished_at=now_iso(), error=str(exc)[:500])


def check_passed(out: Path) -> dict | None:
    p = out / "trainsizes" / "check_move21" / "check.json"
    if not p.is_file():
        return None
    try:
        c = read_json(p)
    except Exception:                                      # noqa: BLE001
        return None
    return c if c.get("ok") is True else None


def run_check(o: argparse.Namespace) -> int:
    P = prepare(o)
    run, out, ho_out, cuts, runs, groups, rungs = P["run"], P["out"], P["ho_out"], P["cuts"], P["runs"], P["groups"], P["rungs"]
    L = int(o.check_length)
    if L not in LENGTHS:
        raise Stop(EXIT_MISSING, f"--check-length {L}: the lengths are {', '.join(map(str, LENGTHS))}")
    _print_header(P, f"the check against move 21 at length {L} (m = 6 with move 21's fixed pairs, one repeat, 50% overlap, move 21's forest seed)")
    absent = [f"cut{c}/len{L}/scores.csv" for c in cuts.values() if not (ho_out / "heldout" / f"cut{c}" / f"len{L}" / "scores.csv").is_file()]
    if absent and not o.dry_run:
        raise Stop(EXIT_MISSING, f"missing input: move 21's scores at length {L} in {ho_out} ({absent[:2]}); run move 21's length-{L} step there first (or name its folder with --heldout-out)")
    if o.dry_run:
        print(f"[trainsizes] dry run: would write {out}/trainsizes/check_move21/ (the check's units, check.json, check.csv) and {out / RECORD_NAME}; nothing written"
              + (f"; move 21's scores missing: {absent}" if absent else ""))
        return EXIT_OK
    if P["missing_series"]:
        raise Stop(EXIT_MISSING, f"missing input: move 20's series for {len(P['missing_series'])} recording(s) (first: {P['missing_series'][0]}); run move 20's `series` step first")
    out_lock = acquire_out_lock(out)
    params = {"run": str(run), "out": str(out), "heldout_out": str(ho_out), "cuts": cuts, "check_length": L, "rungs": rungs, "size": 6, "overlap": 50, "repeat": 0, "rounds": "move 21's folds",
              "n_estimators": P["n_est"], "n_jobs": P["n_jobs"], "seed_forest": P["seed21"], "n_recordings": len(runs), "features_version": FEATURES_VERSION, "force": bool(o.force)}
    entry = _entry(KEY_CHECK, P, params, ["trainsizes/check_move21/check.json"])
    outputs_exist = (out / "trainsizes" / "check_move21" / "check.json").is_file() and check_passed(out) is not None
    try:
        if _skip_if_unchanged(out, KEY_CHECK, entry, ("n_jobs", "force"), outputs_exist, bool(o.force), f"{out}/trainsizes/check_move21"):
            return EXIT_OK
        t0 = time.time()
        write_plan(out, make_plan(cuts, P["choices"], rungs, groups))   # the plan of the launch's choices, written first so the tab's progress has its total from the start
        series_by_cell = HL.load_series(run, P["al_out"], runs)
        cells = [r["cell_id"] for r in runs]
        runs_by_id = {r["cell_id"]: r for r in runs}
        kernel_of, classes, majority = labels_of(runs)
        rd = fixed_rounds_move21(runs, groups)
        rows, ok_all, per_cut = [], True, {}
        for cn, cut in cuts.items():
            s21 = {(r["rung"], r["level"]): r for r in HL._rows(ho_out / "heldout" / f"cut{cut}" / f"len{L}" / "scores.csv")}
            rk21: dict = {}
            for r in HL._rows(ho_out / "heldout" / f"cut{cut}" / f"len{L}" / "recall_per_kernel.csv"):
                rk21.setdefault((r["rung"], r["level"]), {})[r["kernel"]] = _f(r["recall"])
            per_rung = {}
            for rung in rungs:
                f, t_feat, cached = run_feature_unit(out, series_by_cell, cells, cut, L, 50, rung, bool(o.force))
                feat_chk = {"status": "move 21's features not present; rows not compared"}
                p21 = ho_out / "heldout" / f"cut{cut}" / f"len{L}" / "features" / f"{rung}.npz"
                if p21.is_file():
                    with np.load(p21, allow_pickle=False) as z:
                        X21, keys21 = z["X"], list(zip(z["cell_id"].astype(str).tolist(), z["win_start"].astype(np.int64).tolist()))
                        names21 = z["names"].astype(str).tolist()
                    mine = {(c, int(t)): i for i, (c, t) in enumerate(zip(f["cell_id"].tolist(), f["win_start"].tolist()))}
                    missing = [k for k in keys21 if k not in mine]
                    if names21 != f["names"] or missing or len(mine) != len(keys21):
                        feat_chk = {"ok": False, "status": f"the rows differ in names or windows ({len(missing)} of move 21's windows missing here; {len(mine)} rows here, {len(keys21)} there)"}
                    else:
                        idx = np.array([mine[k] for k in keys21], dtype=np.int64)
                        eq = np.array_equal(f["X"][idx], X21, equal_nan=True)
                        feat_chk = {"ok": bool(eq), "status": "rows equal move 21's bit for bit" if eq else f"rows differ (largest deviation {float(np.nanmax(np.abs(f['X'][idx] - X21))):.3g})", "n_rows": len(keys21)}
                    if feat_chk.get("ok") is False:
                        ok_all = False
                sha = sha256_file(features_path(out, cut, L, 50, rung))
                sig = unit_signature(cut=cut, size=6, overlap=50, repeat=0, L=L, rung=rung, rd=rd, seed_forest=P["seed21"], n_est=P["n_est"], features_sha=sha, n_recordings=len(runs), check=True)
                ud = unit_dir(out, cut, 6, 50, 0, L, rung, check=True)
                res = None if o.force else unit_stands(ud, sig)
                if res is None:
                    res = run_model_unit(ud, f, rd, runs_by_id, classes, majority, kernel_of, cut=cut, L=L, rung=rung, n_jobs=P["n_jobs"], n_est=P["n_est"], seed_forest=P["seed21"], sig=sig,
                                         tag=f"check cut {cut} len {L} {rung:<7}")
                else:
                    print(f"[trainsizes] check cut {cut} len {L} {rung}: stands from an earlier run; kept", flush=True)
                for lv in LEVELS:
                    mine_s = res["levels"][lv]["scores"]
                    ref = s21.get((rung, lv))
                    if ref is None:
                        rows.append([cut, L, rung, lv, "missing", None, None, None, None, None, None, "move 21 has no row for this rung and level"])
                        ok_all = False
                        continue
                    diffs = {k: (abs((mine_s.get(k) or 0.0) - (_f(ref.get(k2)) or 0.0)) if mine_s.get(k) is not None and _f(ref.get(k2)) is not None else None)
                             for k, k2 in (("accuracy", "accuracy"), ("macro_recall", "macro_recall"), ("chance", "chance"), ("majority", "majority"))}
                    n_same = int(mine_s.get("n_pools") or 0) == int(_f(ref.get("n_pieces")) or -1)
                    rk_mine = mine_s.get("recall_per_kernel") or {}
                    rk_ref = rk21.get((rung, lv)) or {}
                    rk_dev = max([abs(rk_mine[k] - rk_ref[k]) for k in rk_mine if k in rk_ref] or [0.0]) if rk_mine and rk_ref else None
                    rk_same_keys = set(rk_mine) == set(rk_ref)
                    worst = max([v for v in diffs.values() if v is not None] + [rk_dev or 0.0])
                    equal = n_same and rk_same_keys and all(v is not None and v <= 1e-9 for v in diffs.values()) and (rk_dev is not None and rk_dev <= 1e-9)
                    ok_all = ok_all and equal
                    rows.append([cut, L, rung, lv, "equal" if equal else "DIFFERENT", mine_s.get("accuracy"), _f(ref.get("accuracy")), mine_s.get("macro_recall"), _f(ref.get("macro_recall")),
                                 mine_s.get("n_pools"), _f(ref.get("n_pieces")), f"largest deviation {worst:.3g}; recall per kernel {'compared' if rk_ref else 'absent in move 21'}"
                                 + ("" if rk_same_keys else "; the kernels differ")])
                per_rung[rung] = {"features": feat_chk, "features_time_s": t_feat, "features_cached": cached, "unit_elapsed_s": res.get("elapsed_s")}
            per_cut[cn] = per_rung
        write_csv(out / "trainsizes" / "check_move21" / "check.csv", ["cut", "length", "rung", "level", "verdict", "accuracy_here", "accuracy_move21", "macro_here", "macro_move21", "n_pieces_here",
                                                                      "n_pieces_move21", "note"], rows)
        chk = {"schema": "plan12.trainsizes_check.v1", "citation": CITATION, "written_at": now_iso(), "ok": bool(ok_all), "check_length": L, "heldout_out": str(ho_out), "cuts": cuts,
               "rule": "m = 6 (lexer 6), one repeat, 50% overlap, move 21's fixed pairs, move 21's forest seed: every rung's and level's accuracy, macro recall, chance, majority, piece count "
                       "and recall per kernel must equal move 21's; the feature rows are compared bit for bit when move 21's stand",
               "per_cut": per_cut, "n_compared": len(rows), "n_equal": sum(1 for r in rows if r[4] == "equal"), "elapsed_s": round(time.time() - t0, 1), "code": entry["code"]}
        write_json(out / "trainsizes" / "check_move21" / "check.json", chk)
        entry["outputs"] = ["trainsizes/check_move21/check.json", "trainsizes/check_move21/check.csv"]
        if not ok_all:
            entry.update(status="failed: the check against move 21 did not pass", exit_code=EXIT_ERROR, finished_at=now_iso(), elapsed_s=chk["elapsed_s"])
            print(f"[trainsizes] the check FAILED: {chk['n_equal']} of {len(rows)} (cut, rung, level) rows equal move 21's; see {out}/trainsizes/check_move21/check.csv", file=sys.stderr)
            return EXIT_ERROR
        entry.update(status="done", exit_code=0, finished_at=now_iso(), elapsed_s=chk["elapsed_s"])
        print(f"[trainsizes] the check passed: {chk['n_equal']} of {len(rows)} (cut, rung, level) rows equal move 21's at length {L} ({chk['elapsed_s']} s): {out}/trainsizes/check_move21")
        return EXIT_OK
    except BaseException as exc:                           # noqa: BLE001
        _fail(entry, exc)
        raise
    finally:
        _finish(out, entry, out_lock)


def run_plan(o: argparse.Namespace) -> int:
    P = prepare(o, need_series=False)
    out, cuts, groups, rungs, ch = P["out"], P["cuts"], P["groups"], P["rungs"], P["choices"]
    plan = make_plan(cuts, ch, rungs, groups)
    _print_header(P, "the plan")
    est = estimate(cuts, ch, rungs, groups, ref_train=float(np.mean([fo["n_train"] for fo in HL.make_folds(P["runs"])])))
    print(f"[trainsizes] plan: {plan['n_units']['features']} feature units, {plan['n_units']['model']} model units; skipped {plan['skipped'] or 'none'}; rounds per size "
          + ", ".join(f"{k}: {v['n_rounds']}" + (" (leave one out: one repeat)" if v["leave_one_out"] else "") for k, v in plan["per_size"].items())
          + f"; estimated {est['seconds']['total'] / 3600:.1f} h in total")
    if o.dry_run:
        print(f"[trainsizes] dry run: would write {out}/trainsizes/plan.json, plan.csv, skipped.csv, rounds.csv and rounds/size<M>_rep<K>.json; nothing written")
        return EXIT_OK
    write_plan(out, plan)
    write_rounds(out, groups, ch, ch["sizes"])
    write_json(out / "trainsizes" / "estimate.json", est)
    print(f"[trainsizes] plan written: {out}/trainsizes/plan.json (.csv), rounds.csv, rounds/, skipped.csv, estimate.json")
    return EXIT_OK


def _run_one_unit(P: dict, u: dict, series_by_cell: dict, cells: list[str], runs_by_id: dict, kernel_of: dict, classes: dict, majority: dict, force: bool) -> dict:
    out, ch = P["out"], P["choices"]
    if u["kind"] == "features":
        f, t_feat, cached = run_feature_unit(out, series_by_cell, cells, u["cut"], u["length"], u["overlap"], u["rung"], force)
        print(f"[trainsizes] features cut {u['cut']} len {u['length']:<2} ov {u['overlap']:<2} {u['rung']:<7}: {f['X'].shape[0]} windows (hop {f['hop']}; {int(f['on_train_grid'].sum())} on the training grid), "
              f"{len(f['names'])} statistics ({'stand' if cached else f'{t_feat} s'})", flush=True)
        return {"kind": "features", "kept": cached, "elapsed_s": t_feat}
    fp = features_path(out, u["cut"], u["length"], u["overlap"], u["rung"])
    f = features_stand(fp, cells, u["cut"], u["length"], u["overlap"])
    if f is None:
        raise Stop(EXIT_MISSING, f"missing input: the feature unit {fp} (cut {u['cut']}, length {u['length']}, overlap {u['overlap']}%, {u['rung']}) has not run; run it first")
    rd = make_rounds(P["groups"], u["size"], ch["base_seed"], u["repeat"])
    seed_forest = ch["base_seed"] + int(u["repeat"])
    sig = unit_signature(cut=u["cut"], size=u["size"], overlap=u["overlap"], repeat=u["repeat"], L=u["length"], rung=u["rung"], rd=rd, seed_forest=seed_forest, n_est=P["n_est"],
                         features_sha=sha256_file(fp), n_recordings=len(cells), check=False)
    ud = unit_dir(out, u["cut"], u["size"], u["overlap"], u["repeat"], u["length"], u["rung"])
    tag = f"cut {u['cut']} size {u['size']} ov {u['overlap']:<2} rep {u['repeat']} len {u['length']:<2} {u['rung']:<7}"
    if not force:
        prev = unit_stands(ud, sig)
        if prev is not None:
            print(f"[trainsizes] {tag}: stands from an earlier run (result.json matches the parameters); kept", flush=True)
            return {"kind": "model", "kept": True, "elapsed_s": 0.0}
    res = run_model_unit(ud, f, rd, runs_by_id, classes, majority, kernel_of, cut=u["cut"], L=u["length"], rung=u["rung"], n_jobs=P["n_jobs"], n_est=P["n_est"], seed_forest=seed_forest, sig=sig, tag=tag)
    return {"kind": "model", "kept": False, "elapsed_s": res["elapsed_s"]}


def run_unit(o: argparse.Namespace) -> int:
    P = prepare(o)
    out, cuts, groups, rungs, ch = P["out"], P["cuts"], P["groups"], P["rungs"], P["choices"]
    plan = make_plan(cuts, ch, rungs, groups)
    if o.index is not None:
        if not (0 <= int(o.index) < len(plan["units"])):
            raise Stop(EXIT_MISSING, f"--index {o.index}: the plan of these choices has {len(plan['units'])} units (0 .. {len(plan['units']) - 1})")
        u = plan["units"][int(o.index)]
    else:
        need = [k for k in ("cut", "length", "overlap", "rung") if getattr(o, k, None) in (None, "")]
        if need:
            raise Stop(EXIT_MISSING, f"unit: give --index, or --cut --length --overlap --rung (and --size --repeat for a model unit); missing {need}")
        cut = int(o.cut)
        if cut not in cuts.values():
            raise Stop(EXIT_MISSING, f"--cut {cut}: the run's cuts are {cuts}")
        if hop_of(int(o.length), int(o.overlap)) is None:
            raise Stop(EXIT_MISSING, f"length {o.length} with overlap {o.overlap}% has no whole-pair training start; this (length, overlap) is skipped")
        cn = next(k for k, v in cuts.items() if v == cut)
        if getattr(o, "size", None) in (None, ""):
            u = {"index": None, "kind": "features", "cut": cut, "cut_name": cn, "size": None, "overlap": int(o.overlap), "repeat": None, "length": int(o.length), "rung": o.rung, "needs": None,
                 "file": f"trainsizes/cut{cut}/len{int(o.length)}/ov{int(o.overlap)}/features/{o.rung}.npz"}
        else:
            rep = int(o.repeat or 1)
            if rep < 1:
                raise Stop(EXIT_MISSING, f"--repeat {rep}: repeats are numbered from 1")
            u = {"index": None, "kind": "model", "cut": cut, "cut_name": cn, "size": int(o.size), "overlap": int(o.overlap), "repeat": rep, "length": int(o.length), "rung": o.rung, "needs": None,
                 "file": f"trainsizes/cut{cut}/size{int(o.size)}/ov{int(o.overlap)}/rep{rep}/len{int(o.length)}/{o.rung}/result.json"}
        if u["rung"] not in RUNG_KEYS:
            raise Stop(EXIT_MISSING, f"--rung {u['rung']}: the rungs are {', '.join(RUNG_ORDER)}")
        if u["length"] not in LENGTHS or u["overlap"] not in OVERLAPS or (u["size"] is not None and u["size"] not in SIZES):
            raise Stop(EXIT_MISSING, f"unit: the lengths are {LENGTHS}, the overlaps {OVERLAPS}, the sizes {SIZES}")
    _print_header(P, f"one unit: {u['kind']} cut {u['cut']} size {u['size']} overlap {u['overlap']}% repeat {u['repeat']} length {u['length']} {u['rung']}")
    if o.dry_run:
        print(f"[trainsizes] dry run: would write {out / u['file']}" + (" and its predictions" if u["kind"] == "model" else "") + "; nothing written (no record entry, no folder lock: a unit writes only its own file)")
        return EXIT_OK
    if P["missing_series"] and u["kind"] == "features":
        raise Stop(EXIT_MISSING, f"missing input: move 20's series for {len(P['missing_series'])} recording(s) (first: {P['missing_series'][0]}); run move 20's `series` step first")
    cells = [r["cell_id"] for r in P["runs"]]
    runs_by_id = {r["cell_id"]: r for r in P["runs"]}
    kernel_of, classes, majority = labels_of(P["runs"])
    series_by_cell = HL.load_series(P["run"], P["al_out"], P["runs"]) if u["kind"] == "features" else {}
    r = _run_one_unit(P, u, series_by_cell, cells, runs_by_id, kernel_of, classes, majority, bool(o.force))
    took = "kept" if r["kept"] else f"{r['elapsed_s']} s"
    print(f"[trainsizes] unit done ({took}): {out / u['file']}")
    return EXIT_OK


def run_size(o: argparse.Namespace) -> int:
    size = int(o.size)
    if size not in SIZES:
        raise Stop(EXIT_MISSING, f"--size {size}: the sizes are {', '.join(map(str, SIZES))}")
    P = prepare(o)
    run, out, cuts, groups, rungs, ch = P["run"], P["out"], P["cuts"], P["groups"], P["rungs"], P["choices"]
    ch_size = {**ch, "sizes": [size]}
    plan = make_plan(cuts, ch_size, rungs, groups)
    ps = plan["per_size"][size]
    _print_header(P, f"training size {size}: {len(ps['repeats'])} repeat(s) ({'leave one out: the rounds cannot differ, one repeat' if ps['leave_one_out'] else 'random rounds'}), "
                     f"{ps['n_rounds']} rounds per repeat, {ps['n_train_total']} runs in training" + (f"; capped groups {ps['capped_groups']}" if ps["capped_groups"] else ""))
    est = estimate(cuts, ch_size, rungs, groups, ref_train=float(np.mean([fo["n_train"] for fo in HL.make_folds(P["runs"])])))
    needed = {u["needs"] for u in plan["units"] if u["kind"] == "model"}
    f_units = [u for u in plan["units"] if u["kind"] == "features" and u["index"] in needed]
    m_units = [u for u in plan["units"] if u["kind"] == "model"]
    print(f"[trainsizes] size {size}: {len(f_units)} feature units ({sum(1 for u in f_units if unit_done(out, u))} stand), {len(m_units)} model units ({sum(1 for u in m_units if unit_done(out, u))} stand); "
          f"skipped {plan['skipped'] or 'none'}; estimated {est['seconds']['total'] / 60:.0f} min for the whole size (from move 21's measured times)")
    chk = check_passed(out)
    if chk is None and not o.dry_run:
        raise Stop(EXIT_MISSING, f"missing input: the check against move 21 has not passed in {out} (trainsizes/check_move21/check.json); run the `check` step first")
    if o.dry_run:
        print(f"[trainsizes] dry run: would write the units under {out}/trainsizes/cut<C>/ (features and size{size}/), {out}/trainsizes/size{size}.json, rounds/, the collect's tables and figures "
              f"and {out / RECORD_NAME}; nothing written")
        return EXIT_OK
    if P["missing_series"]:
        raise Stop(EXIT_MISSING, f"missing input: move 20's series for {len(P['missing_series'])} recording(s) (first: {P['missing_series'][0]}); run move 20's `series` step first")
    out_lock = acquire_out_lock(out)
    params = {"run": str(run), "out": str(out), "cuts": cuts, "size": size, "lengths": ch["lengths"], "overlaps": ch["overlaps"], "repeats": ps["repeats"], "repeats_asked": ch["repeats"],
              "base_seed": ch["base_seed"], "n_rounds": ps["n_rounds"], "leave_one_out": ps["leave_one_out"], "rungs": rungs, "levels": list(LEVELS), "round_rule": ROUND_RULE,
              "piece_rule": PIECE_RULE, "bootstrap": P["n_boot"], "seed_bootstrap": SEED_BOOTSTRAP, "n_estimators": P["n_est"], "n_jobs": P["n_jobs"], "n_recordings": len(P["runs"]),
              "features_version": FEATURES_VERSION, "check_passed_at": chk.get("written_at"), "force": bool(o.force)}
    entry = _entry(f"size{size}", P, params, [f"trainsizes/size{size}.json"])
    outputs_exist = (out / "trainsizes" / f"size{size}.json").is_file() and all(unit_done(out, u) for u in plan["units"])
    try:
        if _skip_if_unchanged(out, f"size{size}", entry, ("n_jobs", "force", "check_passed_at"), outputs_exist, bool(o.force), f"{out}/trainsizes/size{size}"):
            return EXIT_OK
        t0 = time.time()
        write_plan(out, make_plan(cuts, ch, rungs, groups))          # the plan of the whole launch's choices (every chosen size), so the tab's progress counts all its units
        write_rounds(out, groups, ch, [size])
        cells = [r["cell_id"] for r in P["runs"]]
        runs_by_id = {r["cell_id"]: r for r in P["runs"]}
        kernel_of, classes, majority = labels_of(P["runs"])
        series_by_cell = HL.load_series(run, P["al_out"], P["runs"]) if any(not unit_done(out, u) for u in f_units) or o.force else {}
        timing = []
        est_of = est.get("per_unit_seconds") or []
        for u in f_units + m_units:
            r = _run_one_unit(P, u, series_by_cell, cells, runs_by_id, kernel_of, classes, majority, bool(o.force))
            timing.append({**{k: u[k] for k in ("index", "kind", "cut", "size", "overlap", "repeat", "length", "rung")}, **r,
                           "estimated_s": est_of[u["index"]] if u["index"] is not None and u["index"] < len(est_of) else None})
        by = {}
        for t in timing:
            if t["kind"] != "model":
                continue
            k = (t["overlap"], t["repeat"], t["length"])
            by.setdefault(k, {"overlap": k[0], "repeat": k[1], "length": k[2], "seconds": 0.0, "n_units": 0, "n_kept": 0})
            by[k]["seconds"] += t["elapsed_s"]; by[k]["n_units"] += 1; by[k]["n_kept"] += int(t["kept"])
        collect_all(out, cuts, P["runs"], rungs, P["n_boot"], quiet=True)
        step = {"schema": "plan12.trainsizes_step.v1", "citation": CITATION, "package_version": __version__, "written_at": now_iso(), "command": sys.argv, "size": size, "params": params,
                "question": __doc__.split("THE QUESTION.", 1)[1].split("THE OUTER LOOP", 1)[0].strip(), "code": entry["code"], "elapsed_s": round(time.time() - t0, 1),
                "per_size": ps, "skipped": plan["skipped"], "n_units": {"features": len(f_units), "model": len(m_units)}, "time_per_unit": timing,
                "time_per_overlap_repeat_length_s": [{**v, "seconds": round(v["seconds"], 1)} for v in by.values()], "estimate": est}
        write_json(out / "trainsizes" / f"size{size}.json", step)
        entry["outputs"] = [u["file"] for u in plan["units"]] + [f"trainsizes/size{size}.json", "trainsizes/rounds.csv", "trainsizes/summary.json"]
        entry.update(status="done", exit_code=0, finished_at=now_iso(), elapsed_s=step["elapsed_s"])
        line = "; ".join(f"ov {v['overlap']} rep {v['repeat']} len {v['length']}: {v['seconds']:.0f} s" for v in by.values())
        print(f"[trainsizes] size {size} done in {step['elapsed_s']} s ({line}): {out}/trainsizes")
        return EXIT_OK
    except BaseException as exc:                           # noqa: BLE001
        _fail(entry, exc)
        raise
    finally:
        _finish(out, entry, out_lock)


def run_collect(o: argparse.Namespace) -> int:
    P = prepare(o, need_series=False)
    out, cuts, rungs = P["out"], P["cuts"], P["rungs"]
    if o.dry_run:
        n = {c: len(find_units(out, c)) for c in cuts.values()}
        print(f"[trainsizes] collect dry run: model units present {n}; would write {out}/trainsizes/cut<C>/scores.csv, scores_per_repeat.csv, margins.csv, margins_per_repeat.csv, "
              f"recall_per_kernel.csv, left_out.csv, accuracy_vs_trainsize_<level>.svg/.csv and {out}/trainsizes/summary.json; nothing written")
        return EXIT_OK
    out_lock = acquire_out_lock(out)
    entry = {"key": KEY_COLLECT, "command": sys.argv, "cwd": str(_HERE.parent), "started_at": now_iso(), "status": "running", "exit_code": None, "finished_at": None,
             "params": {"run": str(P["run"]), "out": str(out), "bootstrap": P["n_boot"]}, "code": code_identity(), "outputs": ["trainsizes/summary.json"]}
    rec = load_record(out)
    rec["entries"].append(entry); save_record(out, rec)
    try:
        t0 = time.time()
        summ = collect_all(out, cuts, P["runs"], rungs, P["n_boot"])
        if summ["n_units_total"] == 0:
            raise Stop(EXIT_MISSING, f"nothing to collect: no model unit has run in {out}")
        entry.update(status="done", exit_code=0, finished_at=now_iso(), elapsed_s=round(time.time() - t0, 1), n_units=summ["n_units_total"])
        print(f"[trainsizes] collect done in {entry['elapsed_s']} s: {summ['n_units_total']} model units into {out}/trainsizes/cut<C>/ and summary.json")
        return EXIT_OK
    except BaseException as exc:                           # noqa: BLE001
        _fail(entry, exc)
        raise
    finally:
        _finish(out, entry, out_lock)


def run_estimate(o: argparse.Namespace) -> int:
    P = prepare(o, need_series=False)
    est = estimate(P["cuts"], P["choices"], P["rungs"], P["groups"], ref_train=float(np.mean([fo["n_train"] for fo in HL.make_folds(P["runs"])])))
    if getattr(o, "json", False):
        print(json.dumps(est))
        return EXIT_OK
    _print_header(P, "the estimate")
    print(f"[trainsizes] {est['n_units']['features']} feature units, {est['n_units']['model']} model units; rounds per size "
          + ", ".join(f"{k}: {v['n_rounds']}" + (" (leave one out: one repeat)" if v["leave_one_out"] else "") for k, v in est["per_size"].items()))
    for r in est["by_size_overlap_repeat_length"]:
        print(f"[trainsizes]   size {r['size']} overlap {r['overlap']:>2}% repeat {r['repeat']} length {r['length']:>2}: about {r['seconds'] / 60:.1f} min")
    print(f"[trainsizes] total about {est['seconds']['total'] / 3600:.2f} h (features {est['seconds']['features'] / 60:.0f} min, models {est['seconds']['model'] / 3600:.2f} h)"
          + (f"; lengths not yet measured by move 21 (extrapolated): {est['lengths_not_measured']}" if est["lengths_not_measured"] else ""))
    return EXIT_OK


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------
def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run", required=True, help="the main run's output folder (read only here)")
    p.add_argument("--out", default=None, help="the output folder (default: a sibling of the run, <run>_trainsizes; never inside the run or another sibling)")
    p.add_argument("--arrowladder-out", default=None, help="where move 20 wrote its series (default <run>_arrowladder; read only)")
    p.add_argument("--heldout-out", default=None, help="where move 21 wrote its results (default <run>_heldout; read only; the check reads its scores)")
    p.add_argument("--encoding-out", default=None, help="decision D2 (default: the encoding run move 6 recorded in classify.json)")
    p.add_argument("--cuts", default=None, help="which cuts, by name: declared,measured (default both)")
    p.add_argument("--rungs", default=None, help="the rungs to run (default all seven)")
    p.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP, help="move 12's bootstrap resamples for the margins (default 1000)")
    p.add_argument("--n-jobs", type=int, default=None, help="processes for the forest (default: move 6's recorded value)")
    p.add_argument("--n-estimators", type=int, default=None, help="trees (default: move 6's recorded value)")
    p.add_argument("--sizes", default=None, help="the training sizes, comma-separated (1 to 7; default all)")
    p.add_argument("--repeats", type=int, default=None, help=f"the random repeats per size (default {DEFAULT_REPEATS}; one when every group leaves one run out)")
    p.add_argument("--seed", default=None, help="the base seed (default: the run's forest seed, move 6's)")
    p.add_argument("--lengths", default=None, help="the piece lengths, comma-separated (default all of move 21's: 64,32,16,8,4,2)")
    p.add_argument("--overlaps", default=None, help="the overlaps of the training pieces in percent, comma-separated (0, 25, 50, 75; default 50)")
    p.add_argument("--force", action="store_true", help="re-run what stands")
    p.add_argument("--dry-run", action="store_true", help="print the plan; write nothing; skip the live-lock refusals")


def main(argv: list[str] | None = None) -> int:
    install_sigterm()
    ap = argparse.ArgumentParser(prog="plan12_grounding.train_sizes", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="the check against move 21: m = 6 with its fixed pairs, one repeat, 50% overlap, at one length")
    _common(c)
    c.add_argument("--check-length", type=int, default=64, help="the length of the check (default 64; move 21 must have run it)")
    p = sub.add_parser("plan", help="write the units of the choices made, with their indices, the rounds and the estimate")
    _common(p)
    u = sub.add_parser("unit", help="run one unit by its index in the plan of these choices, or by its parameters; writes only its own file")
    _common(u)
    u.add_argument("--index", type=int, default=None)
    u.add_argument("--cut", type=int, default=None)
    u.add_argument("--size", type=int, default=None)
    u.add_argument("--overlap", type=int, default=None)
    u.add_argument("--repeat", type=int, default=None)
    u.add_argument("--length", type=int, default=None)
    u.add_argument("--rung", default=None)
    r = sub.add_parser("run", help="one training size: the feature units it needs, then its model units (every overlap, repeat, length, rung, cut), then the collect")
    _common(r)
    r.add_argument("--size", type=int, required=True, help="the training size (1 to 7)")
    k = sub.add_parser("collect", help="the tables and figures across the units present, per cut")
    _common(k)
    e = sub.add_parser("estimate", help="the time estimate for the choices made, from move 21's measured step times")
    _common(e)
    e.add_argument("--json", action="store_true", help="print the estimate as one JSON line (the tab's form reads it)")
    o = ap.parse_args(argv)
    try:
        return {"check": run_check, "plan": run_plan, "unit": run_unit, "run": run_size, "collect": run_collect, "estimate": run_estimate}[o.cmd](o)
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
