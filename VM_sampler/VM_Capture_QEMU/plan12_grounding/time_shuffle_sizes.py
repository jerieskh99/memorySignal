#!/usr/bin/env python3
"""time_shuffle_sizes.py -- the window-size sweep of the time-shuffled control, beside the run of plan12_grounding
(2026-10-10; move 19 in the console's Grounding paper tab, after the control itself (17) and the random split (18)).

  python3 -m plan12_grounding.time_shuffle_sizes run --run <main run out> --size 1|2|4|8|16|32|128 [--out <dir>]
        [--timeshuffle-out <dir>] [--pagefourier-out <dir>] [--cuts declared,measured] [--only cellA,cellB]
        [--chunk-pages 2000] [--n-time-shuffles 10] [--n-shuffles 1000] [--seed 20260930] [--check-unshuffled]
        [--force] [--dry-run]
  python3 -m plan12_grounding.time_shuffle_sizes summary --run <main run out> [--out <dir>] [--timeshuffle-out <dir>]
        [--pagefourier-out <dir>] [--dry-run]
  (run from VM_sampler/VM_Capture_QEMU/)

THE QUESTION. Move 17 shuffled the order of 64-pair windows (the same new order for every page, ten shuffles, the
eight variants of moves 13 to 16) and every within-minus-between gap held. A 64-pair window is half a Welch segment,
so the shuffle left the inside of every window intact. How small must the window be before the gap goes? This move
repeats move 17 exactly for windows of 1, 2, 4, 8, 16, 32 and 128 pairs. A window is always a stretch of consecutive
pairs across ALL pages; pages are never separated. Size 1 is a full shuffle of the pairs. Size 64 is move 17 and is
read from <run>_timeshuffle, never recomputed.

EXACTLY MOVE 17'S COMPUTATION. Every size runs time_shuffle.one_cut as it is (imported, never edited), with the block
size set: the same identity check, the same block orders drawn from default_rng([seed, cut, shuffle, sha256(cell)[:4]])
(so a size-64 run here draws move 17's orders), the same Welch passes (page_fourier's), the same between-recordings
test and label null (a fresh default_rng(seed) before every call), the same files per phase and cut. time_shuffle's
command line refuses sizes under 2; its block_order function handles size 1 (T blocks of one pair, no tail), and this
module checks after every step that each recording's orders are permutations of its T // size full blocks with the
right tail (step.json "order_check"). The accumulator check against recording_spectra runs on the first recording of
every cut and phase as in move 17; --check-unshuffled recomputes the unshuffled order too (the reproduction check:
on the same recording set the recomputed gap equals page_fourier's).

THE STEPS. One step per window size, both phases inside (step 1 phi = 2 theta, step 2 phi = theta, the four variants
each), both cuts; a step is resumable: a (phase, cut) whose summary.json stands with the same size, shuffles, seed and
recording count is kept, the rest recomputed. The last step, `summary`, collects every size present with move 17's
size 64 into the sweep's figure and table; every size step refreshes them too.

THE OUTPUTS, under <out> (default <run>_timeshuffle_sizes)
  size<B>/<phase>/cut<C>/   move 17's files for that size (recordings.csv, block_orders.csv, spectra/<cell>.npz, gaps.csv,
                            summary.csv, identity_check.csv, gaps_unshuffled_vs_shuffled.svg/.csv, summary.json)
  size<B>/<phase>/gaps_both_cuts.svg/.csv, size<B>/gaps_both_cuts.svg/.csv (the eight variants), size<B>/step.json
  sweep_summary.csv/.json   per variant (move, phase), cut and size (1 to 128, 64 from move 17): the shuffled gap as
                            mean, spread (std), min and max over the shuffles, the shuffled nulls' p95 mean, the
                            unshuffled gap and null p95 from <run>_pagefourier, the source folder; a size not yet run
                            reads "missing"
  sweep_summary.svg         per cut and phase a row of four panels (moves 13 to 16): the gap against the window size,
                            the unshuffled gap as a reference line, the label null's 95th percentile dashed red, the
                            shuffles' spread as whiskers; size 64 drawn hollow (move 17); missing sizes marked
  record.json               one entry per run (key size<B> or sweep_summary): command, inputs with sha256, params, the
                            code fingerprint of this module's closure, status; a finished size is skipped unless --force
                            or an input, a parameter or the code changed. Move 17's and page_fourier's results are read,
                            not hashed.

THE RULES THAT PROTECT THE LIVE RUN (as time_shuffle.py)
- ONE new file; nothing existing in plan12_grounding/ is edited; time_shuffle.py and page_fourier.py are imported as
  they are (their recorded code fingerprints stay); nothing imports this module. The run folder, every other sibling
  folder (<run>_idle13, _newblocks, _pagefourier, _timeshuffle, _randomsplit) and the L1 stores are read only here;
  --out inside the run is refused (exit 2). The step refuses to start (exit 3) while <run>/.driver.lock belongs to a
  live process; --dry-run skips that refusal and writes nothing. One writer per output folder:
  <out>/.time_shuffle_sizes.lock; run_moves.install_sigterm.
- --size 64 is accepted only for the reproduction check (it recomputes move 17 into this folder); the sweep always
  takes size 64 from <run>_timeshuffle when it is there.
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
from plan12_grounding import idle_class as I  # noqa: E402  (the lock state, the record pattern; imported, never edited)
from plan12_grounding import page_fourier as PF  # noqa: E402  (select_runs, inputs_identity, default_out, the variants and steps)
from plan12_grounding import time_shuffle as TS  # noqa: E402  (move 17: one_cut, plan_blocks, read_gaps, variant_rows, gaps_svg, SUMMARY_COLS; imported, never edited)
from plan12_grounding.run_moves import _pid_alive, code_fingerprint, install_sigterm, now_iso, read_json, write_json  # noqa: E402
from plan12_grounding.stats import cuts_of  # noqa: E402
from plan12_grounding.figures import svg_open, write_csv  # noqa: E402

CITATION = ("plan12_grounding/time_shuffle.py (move 17: the time-shuffled control, its one_cut run here with the block size set; its block orders, "
            "seeds, label null and files); plan12_grounding/page_fourier.py (moves 13 to 16); the author's design of the sweep (2026-10-10): windows of "
            "1, 2, 4, 8, 16, 32 and 128 pairs, 64 read from move 17, the gap against the window size")
STEPS = TS.STEPS                                           # 1: ("step1_doubled", 2.0, phi = 2 theta); 2: ("step2_single", 1.0, phi = theta)
PHASES = (1, 2)
VARIANT_LIST = TS.VARIANT_LIST
SIZES = (1, 2, 4, 8, 16, 32, 128)                          # the sweep's own sizes
MOVE17_SIZE = 64
ALL_SIZES = (1, 2, 4, 8, 16, 32, 64, 128)
OUT_SUFFIX = "_timeshuffle_sizes"
LOCK_NAME = ".time_shuffle_sizes.lock"
RECORD_NAME = "record.json"
SUMMARY_KEY = "sweep_summary"
DEFAULT_TIME_SHUFFLES = TS.DEFAULT_TIME_SHUFFLES
DEFAULT_CHUNK, DEFAULT_SHUFFLES, DEFAULT_SEED = TS.DEFAULT_CHUNK, TS.DEFAULT_SHUFFLES, TS.DEFAULT_SEED
EXIT_OK, EXIT_ERROR, EXIT_MISSING, EXIT_REFUSED = I.EXIT_OK, I.EXIT_ERROR, I.EXIT_MISSING, I.EXIT_REFUSED
SWEEP_COLS = ["size", "source", "status"] + TS.SUMMARY_COLS
Stop = I.Stop


def fmt(x) -> str:
    return I.fmt(x)


# ---------------------------------------------------------------------------------------------
# the output folder, the locks, the record (time_shuffle's pattern, this module's names)
# ---------------------------------------------------------------------------------------------
def default_out(run: Path) -> Path:
    return run.parent / (run.name + OUT_SUFFIX)


def resolve_out(run: Path, out_arg: str | None) -> Path:
    out = Path(os.path.expanduser(out_arg)).resolve() if out_arg else default_out(run)
    if out == run or run in out.parents:
        raise Stop(EXIT_MISSING, f"refused: --out {out} lies inside the run folder {run}, which is read only here; the default is {default_out(run)}")
    return out


def resolve_dir(run: Path, arg: str | None, default: Path) -> Path:
    return Path(os.path.expanduser(arg)).resolve() if arg else default


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
            raise Stop(EXIT_REFUSED, f"refused: another time_shuffle_sizes run is writing {out} ({lock}: pid {pid}, started {old.get('started_at')}); one writer per output folder")
        print(f"[sizes] stale lock from pid {pid} ({old.get('started_at')}) taken over", flush=True)
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
    return {"schema": "plan12.time_shuffle_sizes.record.v1", "citation": CITATION, "package_version": __version__, "entries": []}


def save_record(out: Path, rec: dict) -> None:
    write_json(out / RECORD_NAME, rec)


def last_done(rec: dict, key: str) -> dict | None:
    for e in reversed(rec.get("entries") or []):
        if e.get("key") == key and e.get("status") == "done":
            return e
    return None


def code_identity() -> dict:
    fp = code_fingerprint("time_shuffle_sizes")
    files = toolkit_fingerprint()["files"]
    return {"fingerprint": fp["sha256"], "modules": fp["modules"], "sha256": {f"{m}.py": files.get(f"{m}.py") for m in fp["modules"]}}


def _finish(out: Path, entry: dict, out_lock: Path) -> None:
    if entry["status"] == "running":
        entry.update(status="failed: stopped before the end", finished_at=now_iso())
    rec = load_record(out)
    rec["entries"] = [e for e in rec["entries"] if not (e.get("key") == entry["key"] and e.get("started_at") == entry["started_at"])] + [entry]
    save_record(out, rec)
    release_out_lock(out_lock)


# ---------------------------------------------------------------------------------------------
# the sweep's table and figure
# ---------------------------------------------------------------------------------------------
def size_source(out: Path, ts_out: Path, size: int, phase: int, cut: int) -> tuple[Path | None, str]:
    """Where a size's gaps come from: move 17's folder for 64 (this module's own size64 only when move 17's is absent)."""
    step_name = STEPS[phase][0]
    if size == MOVE17_SIZE:
        if (ts_out / step_name / f"cut{cut}" / "gaps.csv").is_file():
            return ts_out / step_name, f"move 17 ({ts_out.name})"
        if (out / f"size{size}" / step_name / f"cut{cut}" / "gaps.csv").is_file():
            return out / f"size{size}" / step_name, f"size{size} recomputed here (move 17's folder has no result)"
        return None, f"move 17 ({ts_out.name})"
    return out / f"size{size}" / step_name, f"size{size}"


def sweep_rows(out: Path, ts_out: Path, pf_out: Path, cuts: dict) -> list[dict]:
    rows = []
    for size in ALL_SIZES:
        for phase in PHASES:
            for cut_name, cut in cuts.items():
                src, source = size_source(out, ts_out, size, phase, cut)
                gaps = TS.read_gaps(src, cut) if src is not None else []
                if gaps:
                    for r in TS.variant_rows(phase, gaps, pf_out, cut):
                        rows.append({"size": size, "source": source, "status": "ok", **{k: v for k, v in r.items() if k != "shuffled_gaps"}})
                else:
                    for m, w, lv in VARIANT_LIST:
                        un = TS.pagefourier_result(pf_out, m, STEPS[phase][0], cut)
                        rows.append({"size": size, "source": source, "status": "missing (not run)", "move": m, "weighting": w, "level": lv, "step": phase, "phase": STEPS[phase][2],
                                     "cut": cut, "label": f"m{m} s{phase}", "n_recordings": None, "n_time_shuffles": 0, "unshuffled_status": un["status"], "unshuffled_gap": un["gap"],
                                     "unshuffled_null_p95": un["null_p95"], "unshuffled_p": un["p_value"], "shuffled_gap_mean": None, "shuffled_gap_std": None, "shuffled_gap_min": None,
                                     "shuffled_gap_max": None, "shuffled_null_p95_mean": None, "shuffled_p_min": None, "shuffled_p_max": None, "n_shuffles_at_or_above_unshuffled": None,
                                     "recomputed_gap": None, "recomputed_matches_pagefourier": None, "unshuffled_source": un["source"]})
    return rows


def sweep_svg(rows: list[dict], cuts: dict) -> str:
    """Per cut and phase a row of four panels (moves 13 to 16): the shuffled gap against the window size (1 to 128 on a
    log axis), the unshuffled gap as a horizontal line, the unshuffled label null's 95th percentile dashed red, the
    shuffles' spread as whiskers (min to max) and a thick bar (mean +- std); size 64 (move 17) drawn hollow."""
    pw, ph, gx, gy = 200, 120, 28, 62
    left, top = 50, 56
    W_ = left + 4 * (pw + gx) + 10
    H_ = top + len(cuts) * (2 * (ph + gy) + 26) + 10
    out = svg_open(W_, H_, "The window-size sweep of the time-shuffled control: the gap against the window size, per variant and cut",
                   "dots: the shuffled gap's mean over the shuffles (whiskers min to max, thick bar mean +- std); line: the unshuffled gap (page_fourier); dashed red: its label null's "
                   "95th percentile; hollow dot: size 64 = move 17; a size not yet run reads missing")
    xs = {s: i for i, s in enumerate(ALL_SIZES)}
    X = lambda px, s: px + 14 + xs[s] * (pw - 28) / (len(ALL_SIZES) - 1)       # noqa: E731
    for ci, (cut_name, cut) in enumerate(cuts.items()):
        by = top + ci * (2 * (ph + gy) + 26)
        out.append(f'<text x="{left}" y="{by - 8}" font-weight="bold">cut of {cut} pairs ({cut_name})</text>')
        for pi, phase in enumerate(PHASES):
            py = by + pi * (ph + gy) + 14
            colour = "#2b5d8a" if phase == 1 else "#d9822b"
            for mi, (m, w, lv) in enumerate(VARIANT_LIST):
                px = left + mi * (pw + gx)
                rs = {r["size"]: r for r in rows if r["move"] == m and r["step"] == phase and r["cut"] == cut}
                vals = []
                for r in rs.values():
                    vals += [v for v in (r.get("unshuffled_gap"), r.get("unshuffled_null_p95"), r.get("shuffled_gap_min"), r.get("shuffled_gap_max")) if v is not None]
                lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
                pad = max(0.02, 0.08 * (hi - lo))
                lo, hi = min(lo - pad, 0.0 - pad), hi + pad
                Y = lambda v: py + ph - ph * (v - lo) / (hi - lo)            # noqa: E731
                out.append(f'<text x="{px}" y="{py - 4}" font-size="9" font-weight="bold">m{m} s{phase}: {w}, {lv}</text>')
                out.append(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" fill="none" stroke="#ccc"/>')
                out.append(f'<line x1="{px}" y1="{Y(0):.1f}" x2="{px + pw}" y2="{Y(0):.1f}" stroke="#ddd"/>')
                for t in (lo + pad, hi - pad):
                    out.append(f'<text x="{px - 3}" y="{Y(t) + 3:.1f}" text-anchor="end" font-size="7">{t:.2f}</text>')
                any_row = next((r for r in rs.values() if r.get("unshuffled_gap") is not None), None)
                if any_row is not None:
                    ug = float(any_row["unshuffled_gap"])
                    out.append(f'<line x1="{px}" y1="{Y(ug):.1f}" x2="{px + pw}" y2="{Y(ug):.1f}" stroke="{colour}" stroke-width="1.5"/>')
                    out.append(f'<text x="{px + pw - 2}" y="{Y(ug) - 3:.1f}" text-anchor="end" font-size="7" fill="{colour}">unshuffled {ug:.3f}</text>')
                    if any_row.get("unshuffled_null_p95") is not None:
                        out.append(f'<line x1="{px}" y1="{Y(float(any_row["unshuffled_null_p95"])):.1f}" x2="{px + pw}" y2="{Y(float(any_row["unshuffled_null_p95"])):.1f}" '
                                   f'stroke="#b03a2e" stroke-width="1" stroke-dasharray="3,2"/>')
                pts = []
                for s in ALL_SIZES:
                    cx = X(px, s)
                    r = rs.get(s)
                    out.append(f'<text x="{cx:.1f}" y="{py + ph + 10}" text-anchor="middle" font-size="7">{s}</text>')
                    if r is None or r.get("shuffled_gap_mean") is None:
                        out.append(f'<text x="{cx:.1f}" y="{py + ph - 4}" text-anchor="middle" font-size="6" fill="#999">missing</text>')
                        continue
                    mu, sd = float(r["shuffled_gap_mean"]), float(r.get("shuffled_gap_std") or 0.0)
                    mn, mx = float(r["shuffled_gap_min"]), float(r["shuffled_gap_max"])
                    out.append(f'<line x1="{cx:.1f}" y1="{Y(mn):.1f}" x2="{cx:.1f}" y2="{Y(mx):.1f}" stroke="#333" stroke-width="0.8"/>')
                    out.append(f'<line x1="{cx:.1f}" y1="{Y(mu - sd):.1f}" x2="{cx:.1f}" y2="{Y(mu + sd):.1f}" stroke="#999" stroke-width="3"/>')
                    pts.append((cx, Y(mu)))
                    if s == MOVE17_SIZE:
                        out.append(f'<circle cx="{cx:.1f}" cy="{Y(mu):.1f}" r="2.8" fill="white" stroke="#111" stroke-width="1.2"/>')
                    else:
                        out.append(f'<circle cx="{cx:.1f}" cy="{Y(mu):.1f}" r="2.5" fill="#111"/>')
                if len(pts) > 1:
                    out.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + '" fill="none" stroke="#111" stroke-width="0.6" stroke-dasharray="2,2"/>')
                out.append(f'<text x="{px + pw / 2:.1f}" y="{py + ph + 20}" text-anchor="middle" font-size="7" fill="#555">window size (pairs)</text>')
    out.append(f'<text x="{left}" y="{H_ - 6}" fill="#555" font-size="8">s1 (blue line): phi = 2 theta; s2 (orange line): phi = theta; m13 once/removed, m14 raw/kept, m15 once/kept, m16 raw/removed; '
               f'{TS.SHUFFLE_RULE[:100]}...</text></svg>')
    return "\n".join(out)


def write_sweep(out: Path, ts_out: Path, pf_out: Path, cuts: dict, quiet: bool = False) -> dict:
    rows = sweep_rows(out, ts_out, pf_out, cuts)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "sweep_summary.csv", SWEEP_COLS, [[r.get(c) for c in SWEEP_COLS] for r in rows])
    (out / "sweep_summary.svg").write_text(sweep_svg(rows, cuts))
    missing = sorted({f"size {r['size']} {STEPS[r['step']][0]} cut {r['cut']}" for r in rows if r["status"] != "ok"}, key=lambda s: (int(s.split()[1]), s))
    write_json(out / "sweep_summary.json", {"schema": "plan12.time_shuffle_sizes_sweep.v1", "citation": CITATION, "written_at": now_iso(), "cuts": cuts, "sizes": list(ALL_SIZES),
                                            "move17_size": MOVE17_SIZE, "timeshuffle_out": str(ts_out), "pagefourier_out": str(pf_out), "rows": rows, "missing": missing})
    if not quiet:
        print(f"[sizes] sweep summary: {sum(1 for r in rows if r['status'] == 'ok')} of {len(rows)} variant-size rows present ({len(missing)} size/phase/cut results missing"
              f"{': ' + ', '.join(missing[:5]) + ('...' if len(missing) > 5 else '') if missing else ''}); {out / 'sweep_summary.csv'}, .svg, .json")
    return {"rows": rows, "missing": missing}


def size_figure(size_dir: Path, pf_out: Path, cuts: dict, size: int) -> None:
    """The eight variants of one size, both cuts, in move 17's figure."""
    panels = []
    for cut_name, cut in cuts.items():
        rows = []
        for phase in PHASES:
            rows += TS.variant_rows(phase, TS.read_gaps(size_dir / STEPS[phase][0], cut), pf_out, cut)
        rows.sort(key=lambda r: (r["move"], r["step"]))
        panels.append((cut_name, cut, rows))
    (size_dir / "gaps_both_cuts.svg").write_text(TS.gaps_svg(panels, f"The time-shuffled control with windows of {size} pair{'s' if size > 1 else ''}: the gap unshuffled against shuffled, both cuts"))
    write_csv(size_dir / "gaps_both_cuts.csv", TS.SUMMARY_COLS, [[r[c] for c in TS.SUMMARY_COLS] for _, _, rows in panels for r in rows])


# ---------------------------------------------------------------------------------------------
# the order check after a step: every order a permutation of the right number of full blocks
# ---------------------------------------------------------------------------------------------
def check_orders(size_dir: Path, size: int, cuts: dict) -> dict:
    n_rows, bad = 0, []
    for phase in PHASES:
        for cut in cuts.values():
            d = size_dir / STEPS[phase][0] / f"cut{cut}"
            if not (d / "recordings.csv").is_file() or not (d / "block_orders.csv").is_file():
                continue
            with open(d / "recordings.csv", newline="") as fh:
                T_of = {r["cell_id"]: int(r["n_pairs_cut"]) for r in csv.DictReader(fh) if r["n_pairs_cut"]}
            with open(d / "block_orders.csv", newline="") as fh:
                for r in csv.DictReader(fh):
                    T = T_of.get(r["cell_id"])
                    n_full, tail = int(r["n_full_blocks"]), int(r["tail_pairs"])
                    order = [int(x) for x in r["order_old_blocks_in_new_order"].split()] if r["order_old_blocks_in_new_order"].strip() else []
                    ok = (T is not None and n_full == T // size and tail == T - n_full * size and sorted(order) == list(range(n_full))
                          and (r["shuffled"] == "True") == (n_full >= 2))
                    n_rows += 1
                    if not ok:
                        bad.append(f"{STEPS[phase][0]} cut{cut} {r['cell_id']} shuffle {r['shuffle']}")
    return {"ok": not bad, "n_orders_checked": n_rows, "rule": f"n_full = T // {size}, tail = T - n_full * {size}, the order a permutation of 0 .. n_full - 1, shuffled iff n_full >= 2",
            "failures": bad[:20]}


# ---------------------------------------------------------------------------------------------
# one size: both phases, both cuts, resumable per (phase, cut)
# ---------------------------------------------------------------------------------------------
def cut_stands(d: Path, size: int, n_time: int, n_shuffles: int, seed: int, n_rec: int, phase_factor: float, check_unshuffled: bool) -> bool:
    sj = d / "summary.json"
    if not sj.is_file() or not (d / "gaps.csv").is_file():
        return False
    try:
        j = read_json(sj)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    ln = j.get("label_null") or {}
    return (j.get("status") == "ok" and j.get("block_pairs") == size and j.get("n_time_shuffles") == n_time and ln.get("n_shuffles") == n_shuffles and ln.get("seed") == seed
            and j.get("n_recordings") == n_rec and j.get("phase_factor") == phase_factor and (bool(j.get("check_unshuffled")) or not check_unshuffled))


def run_step(o: argparse.Namespace) -> int:
    run = Path(os.path.expanduser(o.run)).resolve()
    if not (run / "cells.csv").is_file():
        raise Stop(EXIT_MISSING, f"missing input: {run / 'cells.csv'} (--run must name a plan12_grounding output folder with moves 0 and 1 done)")
    size = int(o.size)
    if size not in ALL_SIZES:
        raise Stop(EXIT_MISSING, f"--size {size}: the sweep's sizes are {', '.join(str(s) for s in SIZES)} (64 is move 17, accepted here only for the reproduction check)")
    n_time = int(o.n_time_shuffles)
    if n_time < 1:
        raise Stop(EXIT_MISSING, f"--n-time-shuffles must be 1 or more (given {n_time})")
    out = resolve_out(run, o.out)
    ts_out = resolve_dir(run, o.timeshuffle_out, TS.default_out(run))
    pf_out = resolve_dir(run, o.pagefourier_out, PF.default_out(run))
    lock = I.driver_lock_state(run)
    if lock["alive"] and not o.dry_run:
        raise Stop(EXIT_REFUSED, f"refused: a move is running on this run ({lock['path']}: pid {lock['pid']}, started {lock['started_at']}, {lock['argv_tail']}); "
                                 "this sweep never competes with a running move. Run again after it ends; --dry-run shows the plan meanwhile.")
    cuts = I.parse_cuts(o.cuts, cuts_of(run))
    exj = read_json(run / "moves" / "01_extract" / "extract.json")
    ex_recs = exj.get("recordings") or {}
    only = [s.strip() for s in str(o.only).split(",") if s.strip()] if o.only else None
    runs = PF.select_runs(run, only)
    missing_store = [r["cell_id"] for r in runs if not Path((ex_recs.get(r["cell_id"]) or {}).get("store") or "").is_file()]
    if missing_store:
        raise Stop(EXIT_MISSING, f"missing input: the L1 store of {len(missing_store)} recording(s) is not on this machine (first: {missing_store[0]}; extract.json names it)")
    n_idle = sum(1 for r in runs if r["group"] == "idle")
    size_dir = out / f"size{size}"
    stands = {(ph, c): cut_stands(size_dir / STEPS[ph][0] / f"cut{c}", size, n_time, int(o.n_shuffles), int(o.seed), len(runs), STEPS[ph][1], bool(o.check_unshuffled))
              for ph in PHASES for c in cuts.values()}
    sizes_done = [s for s in ALL_SIZES if (out / f"size{s}" / "step.json").is_file()]
    m17 = all((ts_out / STEPS[ph][0] / f"cut{c}" / "gaps.csv").is_file() for ph in PHASES for c in cuts.values())
    m17_time = None
    if (ts_out / RECORD_NAME).is_file():
        try:
            m17_time = [e.get("elapsed_s") for e in read_json(ts_out / RECORD_NAME).get("entries", []) if e.get("status") == "done" and e.get("elapsed_s")]
        except (OSError, ValueError, json.JSONDecodeError):
            m17_time = None
    print(f"[sizes] window size {size}: move 17's computation with blocks of {size} pair{'s' if size > 1 else ''}, both phases (step 1 phi = 2 theta, step 2 phi = theta; the four "
          f"variants each), on {len(runs)} recordings ({len(runs) - n_idle} kernel + {n_idle} idle), cuts {', '.join(f'{k} = {v}' for k, v in cuts.items())}; {n_time} shuffles per "
          f"recording (seed {o.seed}); {o.n_shuffles} label shuffles per between test; chunks of {o.chunk_pages} pages"
          + ("; --check-unshuffled: the unshuffled order recomputed too" if o.check_unshuffled else "") + ("; size 64 is move 17's: this run is the reproduction check only" if size == MOVE17_SIZE else ""))
    print(f"[sizes] run {run} (read only); stores {Path(ex_recs[runs[0]['cell_id']]['store']).parent} (read only); move 17 {ts_out} ({'both phases present' if m17 else 'NOT complete'}; "
          f"read only); page_fourier {pf_out} (read only); out {out}{' (exists)' if out.exists() else ' (to be created)'}; sizes done here: {sizes_done or 'none'}")
    print(f"[sizes] driver lock: " + (f"held by pid {lock['pid']} ({'alive' if lock['alive'] else 'dead'}), started {lock['started_at']}: {lock['argv_tail']}" if lock["exists"] else "absent")
          + ("; the refusal is skipped by --dry-run" if (lock["alive"] and o.dry_run) else ""))
    for name, cut in cuts.items():
        pb = TS.plan_blocks(runs, cut, size)
        print(f"[sizes] cut {cut} ({name}): {pb['recordings']} recordings, {pb['full_blocks_min']} to {pb['full_blocks_max']} full blocks of {size}"
              + (f"; {pb['recordings_with_fewer_than_two_blocks']} with fewer than two blocks (order kept)" if pb["recordings_with_fewer_than_two_blocks"] else "")
              + "".join(f"; phase {ph} stands (resumed)" for ph in PHASES if stands[(ph, cut)]))
    if m17_time:
        print(f"[sizes] move 17 took {', '.join(f'{t:.0f}' for t in m17_time)} s per phase on this run; a size step is both phases, about {2 * max(m17_time) / 60:.0f} minutes")
    if o.dry_run:
        print(f"[sizes] dry run: would write {size_dir}/<phase>/cut<C>/ (move 17's files), {size_dir}/gaps_both_cuts.svg, {size_dir / 'step.json'}, {out}/sweep_summary.csv/.svg/.json "
              f"and {out / RECORD_NAME}; nothing written")
        return EXIT_OK
    out_lock = acquire_out_lock(out)
    code = code_identity()
    key = f"size{size}"
    entry = {"key": key, "size": size, "command": sys.argv, "cwd": str(_HERE.parent), "started_at": now_iso(), "status": "running", "exit_code": None, "finished_at": None,
             "params": {"run": str(run), "out": str(out), "timeshuffle_out": str(ts_out), "pagefourier_out": str(pf_out), "cuts": cuts, "size": size, "phases": list(PHASES),
                        "nperseg": PF.NPERSEG, "n_time_shuffles": n_time, "n_shuffles": int(o.n_shuffles), "seed": int(o.seed), "chunk_pages": int(o.chunk_pages), "only": only,
                        "check_unshuffled": bool(o.check_unshuffled), "rtol": PF.RTOL, "n_recordings": len(runs), "force": bool(o.force)},
             "inputs_sha256": PF.inputs_identity(run, runs, ex_recs, cuts),
             "code": {"time_shuffle_sizes.py": code["sha256"].get("time_shuffle_sizes.py"), "fingerprint": code["fingerprint"], "modules": code["modules"], "sha256": code["sha256"],
                      "toolkit_fingerprint": toolkit_fingerprint()["sha256"]},
             "outputs": []}
    skip_keys = ("force", "chunk_pages", "timeshuffle_out", "pagefourier_out")
    sig = {k: v for k, v in entry["params"].items() if k not in skip_keys}
    rec = load_record(out)
    prev = last_done(rec, key)
    outputs_exist = (size_dir / "step.json").is_file() and all((size_dir / STEPS[ph][0] / f"cut{c}" / "summary.json").is_file() for ph in PHASES for c in cuts.values())
    try:
        if prev is not None and not o.force and outputs_exist:
            why = None
            if prev.get("inputs_sha256") != entry["inputs_sha256"]:
                why = "an input changed"
            elif {k: v for k, v in (prev.get("params") or {}).items() if k not in skip_keys} != sig:
                why = "the parameters changed"
            elif (prev.get("code") or {}).get("fingerprint") != code["fingerprint"]:
                why = "the code changed"
            if why is None:
                entry.update(status="skipped: outputs exist and inputs, parameters and code unchanged", exit_code=0, finished_at=now_iso(),
                             made_by={"finished_at": prev.get("finished_at"), "code_fingerprint": (prev.get("code") or {}).get("fingerprint")}, outputs=prev.get("outputs") or [])
                rec["entries"].append(entry)
                save_record(out, rec)
                write_sweep(out, ts_out, pf_out, cuts)
                print(f"[sizes] skipped: {size_dir} stands as made on {prev.get('finished_at')} (inputs, parameters and code unchanged; --force re-runs)")
                return EXIT_OK
            print(f"[sizes] re-running: {why} since {prev.get('finished_at')}")
        rec["entries"].append(entry)
        save_record(out, rec)
        size_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        results, timing = {}, {}
        for ph in PHASES:
            step_name, phase_factor, phase_text = STEPS[ph]
            step_dir = size_dir / step_name
            step_dir.mkdir(parents=True, exist_ok=True)
            for name, cut in cuts.items():
                t1 = time.time()
                d = step_dir / f"cut{cut}"
                if stands[(ph, cut)] and not o.force:
                    results[f"{step_name}/cut{cut}"] = {**read_json(d / "summary.json"), "resumed": True}
                    print(f"[sizes] size {size} phase {ph} cut {cut}: stands from an earlier run (summary.json matches: {size}-pair blocks, {n_time} shuffles, seed {o.seed}, {len(runs)} recordings); kept", flush=True)
                else:
                    print(f"[sizes] size {size} phase {ph} ({phase_text}) cut {cut}: time_shuffle.one_cut with B = {size}", flush=True)
                    s = TS.one_cut(step_dir, name, cut, ph, runs, ex_recs, run, pf_out, chunk=int(o.chunk_pages), n_time=n_time, B=size, n_shuffles=int(o.n_shuffles),
                                   seed=int(o.seed), check_unshuffled=bool(o.check_unshuffled))
                    results[f"{step_name}/cut{cut}"] = {**s, "resumed": False}
                timing[f"{step_name}/cut{cut}"] = round(time.time() - t1, 1)
            TS.both_cuts_figure(size_dir, ph, cuts, pf_out)
        size_figure(size_dir, pf_out, cuts, size)
        order_check = check_orders(size_dir, size, cuts)
        if not order_check["ok"]:
            raise Stop(EXIT_ERROR, f"stopped: the block orders of size {size} are not permutations of the full blocks ({len(order_check['failures'])} failure(s), first: {order_check['failures'][0]})")
        step_rec = {"schema": "plan12.time_shuffle_sizes_step.v1", "citation": CITATION, "package_version": __version__, "written_at": now_iso(), "command": sys.argv, "size": size,
                    "phases": {ph: STEPS[ph][2] for ph in PHASES}, "question": __doc__.split("THE QUESTION.", 1)[1].split("EXACTLY MOVE 17'S", 1)[0].strip(),
                    "shuffle_rule": TS.SHUFFLE_RULE, "params": entry["params"], "recordings": [r["cell_id"] for r in runs], "code": entry["code"], "order_check": order_check,
                    "time_per_phase_and_cut_s": timing, "elapsed_s": round(time.time() - t0, 1), "results": results}
        write_json(size_dir / "step.json", step_rec)
        write_sweep(out, ts_out, pf_out, cuts)
        entry["outputs"] = sorted(str(p.relative_to(out)) for p in size_dir.rglob("*") if p.is_file()) + ["sweep_summary.csv", "sweep_summary.svg", "sweep_summary.json"]
        entry.update(status="done", exit_code=0, finished_at=now_iso(), elapsed_s=step_rec["elapsed_s"])
        print(f"[sizes] size {size} done in {step_rec['elapsed_s']} s ({'; '.join(f'{k} {v} s' for k, v in timing.items())}); orders checked: {order_check['n_orders_checked']} ok; {size_dir}")
        return EXIT_OK
    except BaseException as exc:                           # noqa: BLE001
        code_ = getattr(exc, "code", None)
        entry.update(status=f"failed: {type(exc).__name__}" + (f" (exit {code_})" if isinstance(code_, int) else ""), exit_code=code_ if isinstance(code_, int) else EXIT_ERROR,
                     finished_at=now_iso(), error=str(exc)[:500])
        raise
    finally:
        _finish(out, entry, out_lock)


def run_summary(o: argparse.Namespace) -> int:
    run = Path(os.path.expanduser(o.run)).resolve()
    if not (run / "params.json").is_file():
        raise Stop(EXIT_MISSING, f"missing input: {run / 'params.json'}")
    out = resolve_out(run, o.out)
    ts_out = resolve_dir(run, o.timeshuffle_out, TS.default_out(run))
    pf_out = resolve_dir(run, o.pagefourier_out, PF.default_out(run))
    lock = I.driver_lock_state(run)
    if lock["alive"] and not o.dry_run:
        raise Stop(EXIT_REFUSED, f"refused: a move is running on this run ({lock['path']}: pid {lock['pid']}); run again after it ends")
    cuts = cuts_of(run)
    if o.dry_run:
        rows = sweep_rows(out, ts_out, pf_out, cuts)
        present = sorted({(r["size"], r["step"], r["cut"]) for r in rows if r["status"] == "ok"})
        print(f"[sizes] summary dry run: {len(present)} of {len(ALL_SIZES) * len(PHASES) * len(cuts)} size/phase/cut results present (sizes {sorted({s for s, _, _ in present}) or 'none'}); "
              f"would write {out}/sweep_summary.csv, .svg, .json; nothing written")
        return EXIT_OK
    out_lock = acquire_out_lock(out)
    entry = {"key": SUMMARY_KEY, "command": sys.argv, "cwd": str(_HERE.parent), "started_at": now_iso(), "status": "running", "exit_code": None, "finished_at": None,
             "params": {"run": str(run), "out": str(out), "timeshuffle_out": str(ts_out), "pagefourier_out": str(pf_out)}, "code": code_identity(),
             "outputs": ["sweep_summary.csv", "sweep_summary.svg", "sweep_summary.json"]}
    rec = load_record(out)
    rec["entries"].append(entry); save_record(out, rec)
    try:
        res = write_sweep(out, ts_out, pf_out, cuts)
        entry.update(status="done", exit_code=0, finished_at=now_iso(), n_missing=len(res["missing"]))
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
def add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run", required=True, help="the main run's output folder (read only here)")
    p.add_argument("--size", required=True, type=int, choices=list(ALL_SIZES), help="the window size in pairs: 1, 2, 4, 8, 16, 32 or 128 (64 = move 17: accepted only for the reproduction check)")
    p.add_argument("--out", default=None, help="the output folder (default: a sibling of the run, <run>_timeshuffle_sizes; never inside the run)")
    p.add_argument("--timeshuffle-out", default=None, help="where move 17 wrote its results (default <run>_timeshuffle; read only)")
    p.add_argument("--pagefourier-out", default=None, help="where page_fourier wrote its results (default <run>_pagefourier; read only)")
    p.add_argument("--cuts", default=None, help="which cuts, by name: declared,measured (default both; the values come from params.json)")
    p.add_argument("--only", default=None, help="a comma-separated list of cell ids (a test on a few recordings)")
    p.add_argument("--chunk-pages", type=int, default=DEFAULT_CHUNK, help="pages per chunk when the rows are built (memory)")
    p.add_argument("--n-time-shuffles", type=int, default=DEFAULT_TIME_SHUFFLES, help="block orders drawn per recording (default 10, move 17's)")
    p.add_argument("--n-shuffles", type=int, default=DEFAULT_SHUFFLES, help="label shuffles of the between-recordings null (page_fourier: 1000)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="the seed of the block orders and of the label null (the run's: 20260930, move 17's)")
    p.add_argument("--check-unshuffled", action="store_true", help="also recompute the unshuffled order through time_shuffle's pass and its between test, beside page_fourier's")
    p.add_argument("--force", action="store_true", help="re-run a finished size (every phase and cut)")
    p.add_argument("--dry-run", action="store_true", help="print the plan; write nothing; skip the live-lock refusal")


def main(argv: list[str] | None = None) -> int:
    install_sigterm()
    ap = argparse.ArgumentParser(prog="plan12_grounding.time_shuffle_sizes", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    add_run_args(sub.add_parser("run", help="one window size: move 17's computation with that block size, both phases, both cuts (resumable)"))
    sm = sub.add_parser("summary", help="the sweep's table and figure from every size present, with move 17's size 64")
    sm.add_argument("--run", required=True)
    sm.add_argument("--out", default=None, help="the sibling folder (default <run>_timeshuffle_sizes)")
    sm.add_argument("--timeshuffle-out", default=None, help="where move 17 wrote its results (default <run>_timeshuffle)")
    sm.add_argument("--pagefourier-out", default=None, help="where page_fourier wrote its results (default <run>_pagefourier)")
    sm.add_argument("--dry-run", action="store_true")
    o = ap.parse_args(argv)
    try:
        return run_summary(o) if o.cmd == "summary" else run_step(o)
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
