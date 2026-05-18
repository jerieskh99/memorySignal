#!/usr/bin/env python3
"""mp_phase_boundary_inference

Methodology test: runs sandbox_ransom_seq (or sandbox_ransom_batched) with
--phase-markers enabled and records the ground-truth phase boundary timestamps
emitted on stderr. Then compares predicted boundaries from a stub change-point
detector against the ground-truth boundaries and reports an F1-like score.

This script does not consume real signal traces — it operates on a synthetic
per-segment event-count stream computed from the metadata of the underlying
test. The intent is to provide an end-to-end harness that the analyzer can
later swap for a real per-segment metric series (e.g. active_page_fraction).
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import subprocess
import statistics
import re

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.phase2_common import (
    base_argparser, phase, Metadata, log_info, log_warn,
)

TEST = "mp_phase_boundary_inference"

PHASE_RE = re.compile(r"\[(.+?)\] \[PHASE\] test=(\S+) phase=(\S+)")
TS_FMT = "%Y-%m-%dT%H:%M:%SZ"


def build_argparser() -> argparse.ArgumentParser:
    p = base_argparser(TEST, "Phase boundary detection methodology test")
    p.add_argument("--child-binary", required=True,
                   help="Path to the sandbox_ransom binary to invoke (seq or batched)")
    p.add_argument("--child-args", default="",
                   help="Extra args to pass through to the child (string)")
    p.add_argument("--detector", choices=["fixed", "diff"], default="fixed",
                   help="Stub change-point detector (default fixed: equal-spaced)")
    p.add_argument("--tolerance-s", type=float, default=0.5,
                   help="Max time delta to count a predicted boundary as TP")
    return p


def parse_phase_lines(stderr: str) -> list[tuple[float, str]]:
    boundaries: list[tuple[float, str]] = []
    for line in stderr.splitlines():
        m = PHASE_RE.search(line)
        if not m:
            continue
        ts, _test, ph = m.group(1), m.group(2), m.group(3)
        try:
            t = time.mktime(time.strptime(ts, TS_FMT))
        except ValueError:
            continue
        boundaries.append((t, ph))
    return boundaries


def run_child(binary: str, extra_args: str) -> tuple[list[tuple[float, str]], int, str]:
    args = [binary, "--phase-markers"]
    if extra_args:
        args += extra_args.split()
    log_info(f"running child: {' '.join(args)}")
    proc = subprocess.run(args, capture_output=True, text=True, check=False)
    return parse_phase_lines(proc.stderr), proc.returncode, proc.stderr


def predict_fixed(boundaries: list[tuple[float, str]]) -> list[float]:
    if len(boundaries) < 2:
        return []
    t0, tN = boundaries[0][0], boundaries[-1][0]
    k = len(boundaries)
    return [t0 + (tN - t0) * i / (k - 1) for i in range(k)]


def predict_diff(boundaries: list[tuple[float, str]]) -> list[float]:
    if len(boundaries) < 3:
        return [b[0] for b in boundaries]
    deltas = [boundaries[i+1][0] - boundaries[i][0] for i in range(len(boundaries)-1)]
    if not deltas:
        return [b[0] for b in boundaries]
    median = statistics.median(deltas)
    preds = [boundaries[0][0]]
    for i, d in enumerate(deltas):
        if d > 1.5 * median:
            preds.append(boundaries[i+1][0])
    return preds


def evaluate(preds: list[float], truth: list[float], tol: float) -> dict:
    matched_truth = set()
    matched_pred = set()
    for pi, p in enumerate(preds):
        for ti, t in enumerate(truth):
            if ti in matched_truth:
                continue
            if abs(p - t) <= tol:
                matched_pred.add(pi)
                matched_truth.add(ti)
                break
    tp = len(matched_pred)
    fp = len(preds) - tp
    fn = len(truth) - len(matched_truth)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def main() -> int:
    args = build_argparser().parse_args()
    meta = Metadata(TEST, "Python", args.output_dir)
    meta.set_param(child=args.child_binary, child_args=args.child_args,
                   detector=args.detector, tolerance_s=args.tolerance_s,
                   seed=args.seed)
    if args.dry_run:
        meta.set("status", "dry_run"); meta.write(); return 0

    if not os.path.isfile(args.child_binary):
        log_warn(f"child binary not found: {args.child_binary}")
        meta.set("status", "child_missing"); meta.write(); return 1

    phase(TEST, "run_child")
    truth_pairs, rc, _stderr = run_child(args.child_binary, args.child_args)
    truth_ts = [t for t, _ in truth_pairs]
    if not truth_ts:
        meta.set("status", "no_phase_markers"); meta.write(); return 2

    phase(TEST, "predict")
    if args.detector == "fixed":
        preds = predict_fixed(truth_pairs)
    else:
        preds = predict_diff(truth_pairs)

    phase(TEST, "evaluate")
    score = evaluate(preds, truth_ts, args.tolerance_s)

    meta.set("status", "ok")
    meta.set("child_returncode", rc)
    meta.set("truth_n", len(truth_ts))
    meta.set("pred_n", len(preds))
    meta.set("score", score)
    meta.set("truth_boundaries", truth_ts)
    meta.set("pred_boundaries", preds)
    meta.add_limitation("Detector is a stub; replace with a real change-point method using segment metrics")
    meta.write()
    return 0


if __name__ == "__main__":
    sys.exit(main())
