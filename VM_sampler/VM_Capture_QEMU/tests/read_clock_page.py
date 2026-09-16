#!/usr/bin/env python3
"""Host-side clock page reader.

For each raw memory dump of one timing run, find the guest clock page written by
tests/guest_clock_page.py, read the guest's clock at the instant of that dump, and print
the guest's advance between consecutive dumps. If the run's snapshot_timings.jsonl is
given, join by image path and set the guest's advance beside the host's own record of the
running interval for the same pair, t0[n+1] - t5[n]; the difference is the per-frame sliver
the host cannot see (al-Kindi 01, section 4: outcomes A, B, C).

Hits. The marker may occur more than once in a dump (the record page itself; possibly a
stale copy on the Python heap). Every occurrence is parsed; a hit is valid when the head and
tail counters agree and the clock values are plausible; the valid hit with the highest
counter is the freshest and is the one used. Torn hits (head != tail) are counted.

Outputs: one line per dump on stdout, a table of pairs, and with --out a JSONL of every dump
record plus a summary JSON. Nothing is modified; dumps are opened read-only.

Usage:
    ./read_clock_page.py --dump-dir /var/lib/libvirt/qemu/dump \
        --timings timing_runs/<run>/snapshot_timings.jsonl --out timing_runs/<run>/clock_page
    ./read_clock_page.py dump1.raw dump2.raw ...   # explicit files, in order
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import mmap
import os
import statistics
import struct
import sys

MARKER = b"CLKPAGE-" + b"MEMSIG02"
FMT = "<16sQQQQQQ"
RECORD_SIZE = struct.calcsize(FMT)

# plausibility: monotonic/boot below ~30 years of ns, realtime between 2001 and 2096
MONO_MAX_NS = 30 * 365 * 86400 * 10**9
REAL_MIN_NS = 10**18
REAL_MAX_NS = 4 * 10**18


def parse_hits(mm: mmap.mmap, max_hits: int) -> tuple[list[dict], int, int]:
    """Return (valid_hits, n_hits, n_torn)."""
    hits, n_hits, n_torn = [], 0, 0
    pos = mm.find(MARKER)
    while pos >= 0 and n_hits < max_hits:
        n_hits += 1
        if pos + RECORD_SIZE <= len(mm):
            _, c1, mono, mono_raw, real, boot, c2 = struct.unpack_from(FMT, mm, pos)
            if c1 != c2:
                n_torn += 1
            elif 0 < mono < MONO_MAX_NS and REAL_MIN_NS < real < REAL_MAX_NS and 0 < boot < MONO_MAX_NS:
                hits.append({"offset": pos, "counter": c1, "mono_ns": mono, "mono_raw_ns": mono_raw,
                             "real_ns": real, "boot_ns": boot})
        pos = mm.find(MARKER, pos + 1)
    return hits, n_hits, n_torn


def read_dump(path: str, max_hits: int) -> dict:
    rec = {"path": path, "base": os.path.basename(path), "size": os.path.getsize(path)}
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            hits, n_hits, n_torn = parse_hits(mm, max_hits)
        finally:
            mm.close()
    rec["n_hits"], rec["n_torn"], rec["n_valid"] = n_hits, n_torn, len(hits)
    if hits:
        best = max(hits, key=lambda h: h["counter"])
        rec.update(best)
        rec["status"] = "OK"
    else:
        rec["status"] = "NO_CLOCK_PAGE" if n_hits == 0 else "ONLY_TORN"
    return rec


def load_timings(path: str) -> dict[str, dict]:
    """image basename -> snapshot record (t0..t5 host epoch seconds, seq). Backpressure
    lines (seq -1) are skipped."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("seq", -1) < 0 or "image_path" not in d:
                continue
            out[os.path.basename(d["image_path"])] = d
    return out


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)


def f(x, w=9, p=4):
    return f"{'n/a':>{w}}" if x is None else f"{x:{w}.{p}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dumps", nargs="*", help="dump files, in capture order")
    ap.add_argument("--dump-dir", default="", help="directory of memory_dump-*.raw, sorted by name")
    ap.add_argument("--glob", default="memory_dump-*.raw")
    ap.add_argument("--timings", default="", help="the run's snapshot_timings.jsonl, to join by image path")
    ap.add_argument("--out", default="", help="output prefix; writes <out>.jsonl and <out>.summary.json")
    ap.add_argument("--max-hits", type=int, default=64)
    a = ap.parse_args()

    paths = list(a.dumps)
    if a.dump_dir:
        paths += sorted(glob.glob(os.path.join(a.dump_dir, a.glob)))
    if not paths:
        print("no dumps given", file=sys.stderr)
        return 2

    timings = load_timings(a.timings) if a.timings else {}
    recs = []
    for p in paths:
        r = read_dump(p, a.max_hits)
        if timings:
            t = timings.get(r["base"])
            if t:
                r["seq"] = t["seq"]
                for k in ("t0_before_suspend", "t1_after_suspend", "t2_pmemsave_start",
                          "t3_pmemsave_end", "t4_before_resume", "t5_after_resume"):
                    r[k] = float(t[k])
        recs.append(r)
        if r["status"] == "OK":
            print(f"{r['base']} off={r['offset']} counter={r['counter']} hits={r['n_hits']} torn={r['n_torn']} "
                  f"mono_ns={r['mono_ns']} real_ns={r['real_ns']}"
                  + (f" seq={r['seq']}" if "seq" in r else ""))
        else:
            print(f"{r['base']} {r['status']} hits={r['n_hits']} torn={r['n_torn']}")

    # pairs of consecutive dumps that both have a reading
    pairs = []
    prev = None
    for r in recs:
        if r["status"] != "OK":
            prev = None
            continue
        if prev is not None:
            pr = {
                "from": prev["base"], "to": r["base"],
                "d_mono_s": (r["mono_ns"] - prev["mono_ns"]) / 1e9,
                "d_mono_raw_s": (r["mono_raw_ns"] - prev["mono_raw_ns"]) / 1e9,
                "d_real_s": (r["real_ns"] - prev["real_ns"]) / 1e9,
                "d_boot_s": (r["boot_ns"] - prev["boot_ns"]) / 1e9,
                "d_offset_s": ((r["real_ns"] - r["mono_ns"]) - (prev["real_ns"] - prev["mono_ns"])) / 1e9,
                "d_counter": r["counter"] - prev["counter"],
            }
            if "t5_after_resume" in prev and "t0_before_suspend" in r:
                host_running = r["t0_before_suspend"] - prev["t5_after_resume"]
                pr["host_running_s"] = host_running
                pr["epsilon_s"] = pr["d_mono_s"] - host_running
                pr["resume_bracket_prev_s"] = prev["t5_after_resume"] - prev["t4_before_resume"]
                pr["suspend_bracket_s"] = r["t1_after_suspend"] - r["t0_before_suspend"]
                pr["pause_s"] = r["t5_after_resume"] - r["t0_before_suspend"]
                pr["host_dt_s"] = r["t0_before_suspend"] - prev["t0_before_suspend"]
            pairs.append(pr)
        prev = r

    print()
    print(f"{'pair':>6} {'d_mono':>9} {'d_raw':>9} {'d_real':>9} {'d_off':>9} {'host_run':>9} {'eps':>9} {'sus_br':>9} {'res_br':>9} {'pause':>9}")
    for i, pr in enumerate(pairs):
        print(f"{i:>6} {f(pr['d_mono_s'])} {f(pr['d_mono_raw_s'])} {f(pr['d_real_s'])} {f(pr['d_offset_s'])} "
              f"{f(pr.get('host_running_s'))} {f(pr.get('epsilon_s'))} {f(pr.get('suspend_bracket_s'))} "
              f"{f(pr.get('resume_bracket_prev_s'))} {f(pr.get('pause_s'))}")

    summary = {
        "n_dumps": len(recs),
        "n_with_page": sum(1 for r in recs if r["status"] == "OK"),
        "n_no_page": sum(1 for r in recs if r["status"] == "NO_CLOCK_PAGE"),
        "n_only_torn": sum(1 for r in recs if r["status"] == "ONLY_TORN"),
        "n_pairs": len(pairs),
    }
    if pairs:
        dm = [p["d_mono_s"] for p in pairs]
        summary["d_mono_s"] = {"mean": statistics.fmean(dm), "median": statistics.median(dm),
                               "min": min(dm), "max": max(dm),
                               "stdev": statistics.pstdev(dm) if len(dm) > 1 else 0.0}
        summary["d_offset_s_max_abs"] = max(abs(p["d_offset_s"]) for p in pairs)
        summary["mono_minus_raw_drift_s"] = sum(p["d_mono_s"] - p["d_mono_raw_s"] for p in pairs)
        eps = [p["epsilon_s"] for p in pairs if "epsilon_s" in p]
        if eps:
            sb = [p["suspend_bracket_s"] for p in pairs if "epsilon_s" in p]
            pz = [p["pause_s"] for p in pairs if "epsilon_s" in p]
            summary["epsilon_s"] = {"mean": statistics.fmean(eps), "median": statistics.median(eps),
                                    "min": min(eps), "max": max(eps), "n": len(eps)}
            summary["corr_epsilon_vs_suspend_bracket"] = pearson(sb, eps)
            summary["corr_epsilon_vs_pause"] = pearson(pz, eps)
            summary["how_to_read"] = {
                "A": "epsilon within ~0.04 s at every pair: guest advance equals the host-observed running interval; the 931 residue is not a clock story.",
                "B": "epsilon positive and correlated with the suspend bracket (t1-t0): guest running time hidden inside the suspend request; host_running is a lower bound.",
                "C": "epsilon positive and correlated with the whole pause (t5-t0): the guest clock partially catches up on resume; the timing model needs the guest clock plumbed through.",
                "realtime_step": "any |d_offset_s| well above 0 between two dumps is a step of CLOCK_REALTIME.",
                "slew": "mono_minus_raw_drift_s well above 0 over the run is NTP frequency slew.",
            }
    print()
    print(json.dumps(summary, indent=2))

    if a.out:
        with open(a.out + ".jsonl", "w") as fo:
            for r in recs:
                fo.write(json.dumps(r) + "\n")
            for pr in pairs:
                fo.write(json.dumps({"kind": "pair", **pr}) + "\n")
        with open(a.out + ".summary.json", "w") as fo:
            json.dump(summary, fo, indent=2)
        print(f"wrote {a.out}.jsonl and {a.out}.summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
