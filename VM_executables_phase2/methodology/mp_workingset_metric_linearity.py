#!/usr/bin/env python3
"""mp_workingset_metric_linearity

Methodology analysis: takes a list of metadata JSON files produced by
mem_workingset_sweep_v2 and mem_writemag_sweep_v2 runs and fits a
piecewise-linear / power-law model to each metric as a function of the
working set or write magnitude.

This script does NOT run the workload itself; it only does the analysis. The
sweep itself is performed by running the corresponding MEM executable multiple
times with different --working-set-mb (or --bytes-per-page) values.

Inputs: a directory of metadata JSON files. The script picks up files named
*workingset_sweep_v2_metadata.json or *writemag_sweep_v2_metadata.json by
default, but can be overridden via --metric-key and --x-key.
"""
from __future__ import annotations

import os
import sys
import json
import math
import glob
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.phase2_common import (
    base_argparser, phase, Metadata, log_info, log_warn,
)

TEST = "mp_workingset_metric_linearity"


def build_argparser() -> argparse.ArgumentParser:
    p = base_argparser(TEST, "Working-set / write-magnitude metric linearity analysis")
    p.add_argument("--inputs-dir", type=str, default=None,
                   help="Directory of metadata JSON files from sweep runs (required unless --dry-run)")
    p.add_argument("--family", choices=["workingset", "writemag"], default="workingset")
    p.add_argument("--metric-keys", type=str,
                   default="writes,passes,bytes_written,ops",
                   help="Comma-separated metric keys to fit")
    return p


def linear_fit(x: list[float], y: list[float]) -> dict:
    n = len(x)
    if n < 2:
        return {"slope": None, "intercept": None, "r2": None}
    xb = sum(x) / n
    yb = sum(y) / n
    sxx = sum((xi - xb) ** 2 for xi in x)
    sxy = sum((xi - xb) * (yi - yb) for xi, yi in zip(x, y))
    syy = sum((yi - yb) ** 2 for yi in y)
    if sxx == 0:
        return {"slope": 0.0, "intercept": yb, "r2": 0.0}
    slope = sxy / sxx
    intercept = yb - slope * xb
    r2 = (sxy ** 2) / (sxx * syy) if syy else 0.0
    return {"slope": slope, "intercept": intercept, "r2": r2}


def power_law_fit(x: list[float], y: list[float]) -> dict:
    pairs = [(xi, yi) for xi, yi in zip(x, y) if xi > 0 and yi > 0]
    if len(pairs) < 2:
        return {"alpha": None, "beta": None, "r2": None}
    lx = [math.log(p[0]) for p in pairs]
    ly = [math.log(p[1]) for p in pairs]
    fit = linear_fit(lx, ly)
    return {"alpha": fit["slope"], "log_beta": fit["intercept"], "r2": fit["r2"]}


def main() -> int:
    args = build_argparser().parse_args()
    meta = Metadata(TEST, "Python", args.output_dir)
    meta.set_param(inputs_dir=args.inputs_dir, family=args.family,
                   metric_keys=args.metric_keys, seed=args.seed)
    if args.dry_run:
        meta.set("status", "dry_run"); meta.write(); return 0

    if not args.inputs_dir:
        log_warn("--inputs-dir is required (or use --dry-run)")
        meta.set("status", "missing_inputs_dir"); meta.write(); return 2

    pattern = "mem_workingset_sweep_v2_*.json" if args.family == "workingset" \
              else "mem_writemag_sweep_v2_*.json"
    paths = sorted(glob.glob(os.path.join(args.inputs_dir, pattern)))
    if not paths:
        # also accept any *.json with the expected structure
        paths = sorted(glob.glob(os.path.join(args.inputs_dir, "*_metadata.json")))
    if not paths:
        log_warn(f"no metadata files found in {args.inputs_dir}")
        meta.set("status", "no_inputs"); meta.write(); return 1

    phase(TEST, "load")
    runs = []
    for p in paths:
        try:
            with open(p, "r") as f:
                obj = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            log_warn(f"skip {p}: {e}")
            continue
        runs.append(obj)
    if not runs:
        meta.set("status", "no_valid_inputs"); meta.write(); return 1

    x_key = "working_set_mb" if args.family == "workingset" else "bytes_per_page"

    metric_keys = [k.strip() for k in args.metric_keys.split(",") if k.strip()]
    phase(TEST, "fit")
    fits = {}
    for mk in metric_keys:
        x_y = []
        for r in runs:
            xv = r.get(x_key) if x_key in r else r.get("parameters", {}).get(x_key)
            yv = r.get(mk)
            if isinstance(xv, (int, float)) and isinstance(yv, (int, float)):
                x_y.append((float(xv), float(yv)))
        if not x_y:
            fits[mk] = {"note": f"no values for x={x_key} y={mk}"}
            continue
        x_y.sort()
        x = [a for a, _ in x_y]
        y = [b for _, b in x_y]
        fits[mk] = {
            "n_points": len(x),
            "x": x, "y": y,
            "linear": linear_fit(x, y),
            "power_law": power_law_fit(x, y),
        }

    meta.set("status", "ok")
    meta.set("inputs", paths)
    meta.set("x_key", x_key)
    meta.set("fits", fits)
    meta.add_limitation("Linear/power-law fits are first-pass; replace with piecewise or Bayesian fits as needed")
    meta.write()
    return 0


if __name__ == "__main__":
    sys.exit(main())
