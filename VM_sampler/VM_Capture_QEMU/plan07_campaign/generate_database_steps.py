#!/usr/bin/env python3
"""Generate a large capture database: many reps per workload, varied parameters.

Design goals (2026-08-11):

  * LONG traces. Default 300 s per cell, so a windowed analysis (the project's
    offline metrics use window=128/hop=64) gets many full windows per trace even
    for the heavy workloads whose backpressure throttles cadence to ~0.35 snap/s.

  * BALANCED PER FAMILY. The retention tree is family/workload/params/repNNN, so
    one dataset serves both lenses: per-workload folders, and per-family by
    aggregation. Families differ hugely in size (kernel 64 workloads, io 2), so
    reps-per-workload is derived from a per-family target rather than fixed --
    otherwise 'kernel' would swamp every other family.

  * PARAMETER VARIATION. Each rep of a workload perturbs its numeric size knobs
    (and its --seed) instead of repeating one configuration. That is what makes
    the dataset answer "does one behaviour still look like itself under different
    arguments?" rather than just measuring run-to-run noise. Scaling is generic
    (multiply the value found in the base command) so it needs no per-workload
    semantics, and every result is clamped to what the guest can actually run.

  * CONTEXT. Steps are emitted in a shuffled order (fixed seed => reproducible),
    so a workload is preceded by a different neighbour in each rep. Combined with
    the orchestrator's per-cell VM reboot this mostly isolates cells; run with
    chaining (see --chain) when you WANT the preceding workload's state to carry.

Usage:
    python3 plan07_campaign/generate_database_steps.py --per-family 100
        -> plan07_campaign/database_steps.txt
        -> plan07_campaign/database_manifest.csv

Then capture with CAPTURE_METRIC=apf_queue and NO ZSTD for the bulk (the metric
is ~10 MB per 101-cell pass; full raw retention is ~674 GiB per pass and caps you
at 3 passes in 2 TB). Use --raw-subset to emit a separate small steps file to run
WITH ZSTD=1, preserving the recompute-any-future-metric guarantee where it counts.
"""
from __future__ import annotations

import argparse
import csv
import shlex
import random
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_STEPS = HERE / "full_campaign_steps.txt"

# Numeric flags safe to scale for parameter variation. Everything else (paths,
# --duration, --seed, --max-mb, --phase-markers, mode/enum flags) is left alone.
SCALABLE = {
    "--working-set-mb", "--file-size-mb", "--input-size-mb", "--output-size-mb",
    "--block-kb", "--buffer-kb", "--table-kb", "--cpu-block-kb",
    "--dim", "--grid-n", "--width", "--height", "--n", "--nodes", "--particles",
    "--rows", "--cols", "--inner", "--samples", "--bins", "--paths", "--tokens",
    "--elements", "--cells", "--vertices", "--edges", "--seq-len", "--files",
    "--capacity", "--len-a", "--len-b", "--chain", "--steps", "--text-mb",
    "--input-mb", "--ensemble", "--bits", "--patterns", "--states",
}

# Hard ceilings the guest can honour (964 MB RAM, ~664 MB usable; scratch on
# /var/tmp has 45 GB but a single cell writing >1 GB is pointlessly slow).
CLAMP_MB = {
    "--working-set-mb": 384, "--file-size-mb": 512, "--input-size-mb": 256,
    "--output-size-mb": 256, "--mem-cap-mb": 256,
}
# Scale factors cycled across reps. 1.0 first so rep001 reproduces the campaign's
# own configuration, making it directly comparable to the runs already captured.
SCALES = [1.0, 0.5, 2.0, 0.25, 1.5, 0.75, 3.0, 0.33]


def family_of(binary: str) -> str:
    return binary.split("_", 1)[0] if "_" in binary else binary


def binary_of(cmd: str) -> str:
    m = re.search(r"/(?:bin|app_realistic|methodology)/([A-Za-z0-9_.-]+)", cmd)
    return re.sub(r"\.py$", "", m.group(1)) if m else "step"


def _scale_tokens(toks: list[str], scale: float, seed: int, duration: int) -> list[str]:
    """Rewrite duration/seed/size flags in an already-tokenised command."""
    out: list[str] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        nxt = toks[i + 1] if i + 1 < len(toks) else None
        if t == "--duration" and nxt is not None:
            out += [t, str(duration)]; i += 2; continue
        if t == "--seed" and nxt is not None:
            out += [t, str(seed)]; i += 2; continue
        # A nested command string (mp_phase_boundary_inference passes its child's
        # whole argv as one quoted value). shlex hands it over as a single token,
        # so recurse into it rather than letting the outer pass mangle it.
        if t == "--child-args" and nxt is not None:
            inner = _scale_tokens(shlex.split(nxt), scale, seed, duration)
            out += [t, " ".join(inner)]; i += 2; continue
        if t in SCALABLE and nxt is not None and nxt.isdigit():
            v = max(1, int(int(nxt) * scale))
            cap = CLAMP_MB.get(t)
            if cap:
                v = min(v, cap)
            out += [t, str(v)]; i += 2; continue
        out.append(t); i += 1
    return out


def scale_command(cmd: str, scale: float, seed: int, duration: int) -> str:
    """Apply a size scale + fresh seed + duration to one command.

    Tokenise with shlex, not str.split(): a naive split leaves the closing quote
    stuck to the last value inside a quoted group (`--seed 42"`), so rewriting
    that value silently deletes the quote and produces an unparseable command.
    Reassemble with shlex.quote so any token containing spaces is re-quoted.
    Shell operators (&&) are single tokens and pass through untouched.
    """
    toks = _scale_tokens(shlex.split(cmd), scale, seed, duration)
    return " ".join(t if t in ("&&", "||", ";") else shlex.quote(t) for t in toks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-family", type=int, default=100,
                    help="target traces per family (default 100)")
    ap.add_argument("--duration", type=int, default=300,
                    help="seconds per cell (default 300)")
    ap.add_argument("--raw-subset", type=int, default=0,
                    help="also emit N cells as a separate steps file to run WITH ZSTD=1")
    ap.add_argument("--seed", type=int, default=42, help="shuffle/seed base")
    ap.add_argument("--out", default="database_steps.txt")
    a = ap.parse_args()

    base = [l.strip() for l in BASE_STEPS.read_text().splitlines()
            if l.strip() and not l.strip().startswith("#")]

    by_family: dict[str, list[str]] = defaultdict(list)
    for cmd in base:
        by_family[family_of(binary_of(cmd))].append(cmd)

    rng = random.Random(a.seed)
    cells: list[tuple[str, str, str, int, float]] = []   # cmd, family, binary, rep, scale
    for fam, cmds in sorted(by_family.items()):
        reps = max(1, -(-a.per_family // len(cmds)))     # ceil division
        for cmd in cmds:
            b = binary_of(cmd)
            for rep in range(1, reps + 1):
                scale = SCALES[(rep - 1) % len(SCALES)]
                seed = a.seed + rep * 1000 + (hash(b) % 997)
                cells.append((scale_command(cmd, scale, seed, a.duration),
                              fam, b, rep, scale))

    # Shuffle so a workload's neighbour differs per rep: family is then not
    # confounded with position in a multi-day run (host drift would otherwise be
    # indistinguishable from a family signature).
    rng.shuffle(cells)

    out_p = HERE / a.out
    out_p.write_text("\n".join(c[0] for c in cells) + "\n")

    man_p = HERE / (Path(a.out).stem + "_manifest.csv")
    with man_p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["step_index", "family", "workload",
                                           "rep", "scale", "command"])
        w.writeheader()
        for i, (cmd, fam, b, rep, sc) in enumerate(cells, 1):
            w.writerow({"step_index": i, "family": fam, "workload": b,
                        "rep": rep, "scale": sc, "command": cmd})

    per_fam: dict[str, int] = defaultdict(int)
    for _, fam, _, _, _ in cells:
        per_fam[fam] += 1
    mins = len(cells) * (a.duration / 60 + 4.7)
    print(f"wrote {out_p.name}: {len(cells)} cells @ {a.duration}s")
    print(f"      {man_p.name}")
    print("per family:", dict(sorted(per_fam.items(), key=lambda x: -x[1])))
    print(f"estimated wall clock: {mins/60:.0f} h ({mins/60/24:.1f} days) "
          f"at ~{a.duration/60+4.7:.1f} min/cell")
    print(f"estimated APF-only size: ~{len(cells)*0.1:.0f} MB "
          f"(raw zstd would be ~{len(cells)*6.7:.0f} GiB)")

    if a.raw_subset:
        sub = cells[:a.raw_subset]
        sp = HERE / (Path(a.out).stem + "_rawsubset.txt")
        sp.write_text("\n".join(c[0] for c in sub) + "\n")
        print(f"wrote {sp.name}: {len(sub)} cells to run WITH ZSTD=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
