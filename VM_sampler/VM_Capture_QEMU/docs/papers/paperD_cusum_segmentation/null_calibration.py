#!/usr/bin/env python3
"""R-8 CUSUM null-calibration for Paper D.

Closes the last open offline reviewer item flagged in reviewer_memo_round2.

R-8  The marker-less phasic G3 gate validates a phasic cell by boundary-count
     PLAUSIBILITY: the cell passes iff the windowed CUSUM emits
     1 <= n_boundaries <= 20 (plan04_run.PLAUSIBILITY_MIN / PLAUSIBILITY_MAX;
     scored in plan04_run._score_cell, marker-less branch). At the achievable
     cadence the trajectories are short (median ~30 snap-pairs -> ~6 rolling-mean
     points at (W,H)=(8,4)), so the upper bound 20 never bites: the gate reduces
     to "did CUSUM fire at least once" (n_boundaries >= 1). The ONLY way to FAIL
     is n_boundaries == 0.

     Two facts make this band suspect, and R-8 asks us to quantify them:

       1. The PHASIC detector is family-tuned to be highly sensitive. The
          canonical Plan 04 run (plan04_overview.html: "--k 2.0 --h 4.0
          --k-phasic 0.1 --h-phasic 0.5") drives phasic cells at k=0.1, h=0.5 --
          a 20x-lower reference and 8x-lower threshold than the steady k=2.0,
          h=4.0 -- explicitly "to surface sustained activity into the
          plausibility band" (plan04_run.py D-87 comment). A detector tuned to
          fire, scored by a band that only excludes "never fired", is at risk of
          being unfalsifiable.

       2. The result JSON records only the steady defaults ("k": 2.0, "h": 4.0);
          the phasic k=0.1/h=0.5 override is applied per-cell in the runner and
          is invisible in the detector block. This script makes the real phasic
          detector explicit and reproducible.

     We run the EXACT phasic detector (plan04_cusum.detect_boundaries_full at
     k=0.1, h=0.5, min_separation=2, (W,H)=(8,4)) on three nulls that contain NO
     sustained phase structure and measure the [1,20] pass-rate:

       NULL-A  real STEADY cells pushed through the phasic (k=0.1, h=0.5, 8/4)
               detector. Empirical null: real non-phasic trajectories
               (reversible-XOR steady workloads), real cadence + measurement
               noise, no phase rhythm.
       NULL-B  temporal SHUFFLE of each phasic cell's APF values. Preserves the
               exact marginal (mean, std, full value multiset) and destroys
               temporal contiguity -> destroys any sustained regime. If phasic
               still passes at the shuffle rate, the gate keys on dispersion,
               not on phase rhythm.
       NULL-C  IID Gaussian surrogate matched to each phasic cell's (length,
               mean, std), clipped to [0,1]. Destroys distribution shape too.

     The band+detector is informative only if the real-phasic pass-rate sits
     materially above the null pass-rate. If the gap is ~0 the G3 plausibility
     verdict is near-unfalsifiable and phasic acceptance is not evidence of
     detected phase structure -- it is evidence the trajectory is not flat.

Reproduces the production per-workload plausibility_pass_fraction from
plan04_segmenter_results.json (batched 0.8333, selective 0.75, seq 0.75,
slowburn 0.9167, scanner 1.0) as a sanity check before reporting nulls.

Usage:
  python3 null_calibration.py [path/to/_v3_combined] > null_calibration.json
"""
import csv, json, sys
from pathlib import Path
import numpy as np

THIS = Path(__file__).resolve()
VM_CAPTURE_QEMU = THIS.parents[3]          # .../VM_Capture_QEMU
WORKTREE_ROOT = THIS.parents[5]            # .../clever-pasteur-a388b5
sys.path.insert(0, str(VM_CAPTURE_QEMU))
from plan04_cusum import detect_boundaries_full  # noqa: E402

DEFAULT_DATA = WORKTREE_ROOT / "_v3_combined"
PHASIC = {"sandbox_ransom_batched", "sandbox_ransom_selective",
          "sandbox_ransom_seq", "sandbox_ransom_slowburn",
          "sandbox_scanner_metadata"}
# Family-aware production detector (plan04_overview.html canonical invocation:
# --k 2.0 --h 4.0 --k-phasic 0.1 --h-phasic 0.5). plan04_run.py applies the
# phasic override per-cell (D-87); the result JSON records only the steady pair.
K_STEADY, H_STEADY = 2.0, 4.0
K_PHASIC, H_PHASIC = 0.1, 0.5
MIN_SEP = 2
# Marker-less plausibility band (plan04_run.PLAUSIBILITY_MIN / _MAX).
BAND_MIN, BAND_MAX = 1, 20
# Phasic regime: all 5 phasic workloads recommend (8,4). Nulls asking "would a
# structureless trajectory pass the PHASIC gate" use the phasic detector at (8,4).
PHASIC_WH = (8, 4)
N_SURROGATE = 1000
SEED = 0


def load_trajectory(data_dir, cell_id):
    """APF values sorted by seq, final sentinel dropped (mirrors
    plan02_metrics_per_cell.load_streaming_trajectory)."""
    p = data_dir / "cells" / "work" / cell_id / "apf_trajectory.jsonl"
    if not p.is_file():
        return []
    pairs = []
    for raw in p.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("final") is True:
            continue
        seq, apf = obj.get("seq"), obj.get("apf")
        if isinstance(seq, int) and isinstance(apf, (int, float)):
            pairs.append((seq, float(apf)))
    pairs.sort(key=lambda t: t[0])
    return [v for _, v in pairs]


def load_recommendation(data_dir):
    rec = json.loads((data_dir / "plan03" / "recommendation.json").read_text())
    out = {}
    for e in rec.get("recommendations", []):
        wl, w, h = e.get("workload"), e.get("recommended_window"), e.get("recommended_hop")
        if wl and w is not None and h is not None:
            out[wl] = (int(w), int(h))
    return out


def load_cells(data_dir):
    """Non-warmup, ok cells with a trajectory file. Returns list of dicts."""
    cells = []
    with open(data_dir / "manifest.csv") as fh:
        for r in csv.DictReader(fh):
            if str(r.get("is_warmup", "")).strip() == "1":
                continue
            if not str(r.get("status", "")).startswith("ok"):
                continue
            traj = load_trajectory(data_dir, r["cell_id"])
            if not traj:
                continue
            cells.append({"cell_id": r["cell_id"], "workload": r["workload"],
                          "family": "phasic" if r["workload"] in PHASIC else "steady",
                          "duration_s": r.get("duration_s"),
                          "replicate": r.get("replicate"), "traj": traj})
    return cells


def n_boundaries(traj, window, hop, k, h):
    return len(detect_boundaries_full(list(traj), window, hop,
                                      k=k, h=h, min_separation=MIN_SEP))


def in_band(n):
    return BAND_MIN <= n <= BAND_MAX


def summarize_counts(ns):
    """Distribution + band diagnostics for a list of integer boundary counts."""
    a = np.asarray(ns, dtype=float)
    n = a.size
    if n == 0:
        return {"n": 0}
    return {"n": int(n),
            "band_pass_fraction": round(float(np.mean([in_band(int(x)) for x in a])), 4),
            "p_n_eq_zero": round(float(np.mean(a == 0)), 4),
            "p_n_ge_1": round(float(np.mean(a >= 1)), 4),
            "p_n_gt_20": round(float(np.mean(a > BAND_MAX)), 4),
            "n_pred_min": int(a.min()), "n_pred_median": float(np.median(a)),
            "n_pred_mean": round(float(a.mean()), 4), "n_pred_max": int(a.max())}


def real_group(cells, params_of):
    """Per-cell real boundary counts. params_of(cell) -> (window,hop,k,h)."""
    ns, per_wl = [], {}
    for c in cells:
        w, h, k, hh = params_of(c)
        n = n_boundaries(c["traj"], w, h, k, hh)
        ns.append(n)
        per_wl.setdefault(c["workload"], []).append(n)
    out = summarize_counts(ns)
    out["per_workload"] = {w: {"n_cells": len(v),
                               "plausibility_pass_fraction": round(float(
                                   np.mean([in_band(x) for x in v])), 4),
                               "n_pred_median": float(np.median(v))}
                           for w, v in sorted(per_wl.items())}
    return out


def surrogate_group(cells, make_surrogate, rng, window, hop, k, h,
                    n_sur=N_SURROGATE):
    """Per-cell mean band-pass over n_sur surrogates run through the (window,
    hop, k, h) detector, then pooled.

    make_surrogate(traj, rng) -> surrogate trajectory. Reports the pooled
    pass-rate plus the per-cell distribution and the matched real pass for
    paired reading.
    """
    per_cell = []
    pooled_pass, pooled_zero, pooled_n = [], [], []
    for c in cells:
        real_n = n_boundaries(c["traj"], window, hop, k, h)
        passes, zeros, ncounts = [], [], []
        for _ in range(n_sur):
            sn = n_boundaries(make_surrogate(c["traj"], rng), window, hop, k, h)
            passes.append(in_band(sn))
            zeros.append(sn == 0)
            ncounts.append(sn)
        mp = float(np.mean(passes))
        per_cell.append({"cell_id": c["cell_id"], "workload": c["workload"],
                         "real_in_band": bool(in_band(real_n)),
                         "surrogate_band_pass": round(mp, 4),
                         "surrogate_p_n0": round(float(np.mean(zeros)), 4)})
        pooled_pass.extend(passes)
        pooled_zero.extend(zeros)
        pooled_n.extend(ncounts)
    pa = np.asarray(pooled_pass, dtype=float)
    na = np.asarray(pooled_n, dtype=float)
    real_passing = [d["surrogate_band_pass"] for d in per_cell if d["real_in_band"]]
    return {"n_cells": len(cells), "n_surrogates_per_cell": n_sur,
            "detector": {"window": window, "hop": hop, "k": k, "h": h},
            "pooled_band_pass_fraction": round(float(pa.mean()), 4),
            "pooled_p_n_eq_zero": round(float(np.mean(np.asarray(pooled_zero))), 4),
            "n_pred_median": float(np.median(na)),
            "n_pred_mean": round(float(na.mean()), 4),
            "mean_surrogate_pass_among_real_passing_cells": round(
                float(np.mean(real_passing)), 4) if real_passing else None,
            "per_cell": per_cell}


def shuffle_surrogate(traj, rng):
    return rng.permutation(np.asarray(traj, dtype=float))


def iid_surrogate(traj, rng):
    a = np.asarray(traj, dtype=float)
    return np.clip(rng.normal(a.mean(), a.std(), size=a.size), 0.0, 1.0)


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA
    rng = np.random.default_rng(SEED)
    cells = load_cells(data_dir)
    rec = load_recommendation(data_dir)
    phasic = [c for c in cells if c["family"] == "phasic"]
    steady = [c for c in cells if c["family"] == "steady"]

    def phasic_params(c):       # production phasic detector at own (8,4)
        w, h = rec.get(c["workload"], (8, 4))
        return (w, h, K_PHASIC, H_PHASIC)

    def steady_params(c):       # production steady detector at own (W,H)
        w, h = rec.get(c["workload"], (8, 4))
        return (w, h, K_STEADY, H_STEADY)

    def steady_through_phasic(c):   # NULL-A: steady traj, phasic detector, (8,4)
        return (PHASIC_WH[0], PHASIC_WH[1], K_PHASIC, H_PHASIC)

    # Real, production detectors: reproduces segmenter numbers.
    real_phasic = real_group(phasic, phasic_params)
    real_steady_own = real_group(steady, steady_params)
    # NULL-A: real steady through the phasic detector + band.
    real_steady_as_null = real_group(steady, steady_through_phasic)

    # NULL-B / NULL-C: phasic surrogates through the phasic detector.
    null_b = surrogate_group(phasic, shuffle_surrogate, rng,
                             PHASIC_WH[0], PHASIC_WH[1], K_PHASIC, H_PHASIC)
    null_c = surrogate_group(phasic, iid_surrogate, rng,
                             PHASIC_WH[0], PHASIC_WH[1], K_PHASIC, H_PHASIC)

    rp = real_phasic["band_pass_fraction"]
    a = real_steady_as_null["band_pass_fraction"]
    b = null_b["pooled_band_pass_fraction"]
    c = null_c["pooled_band_pass_fraction"]
    gaps = {"vs_steady_as_null": round(rp - a, 4),
            "vs_temporal_shuffle": round(rp - b, 4),
            "vs_iid_gaussian": round(rp - c, 4)}
    max_null = max(a, b, c)
    near_unfalsifiable = bool((rp - max_null) < 0.15 and max_null >= 0.5)

    out = {
        "schema": "paperD.cusum_null_calibration.v1",
        "numpy_version": np.__version__,
        "data_dir": str(data_dir),
        "detector": {
            "name": "cusum", "min_separation": MIN_SEP,
            "steady": {"k": K_STEADY, "h": H_STEADY},
            "phasic": {"k": K_PHASIC, "h": H_PHASIC},
            "phasic_regime_window_hop": list(PHASIC_WH),
            "note": ("family-aware: plan04_run.py applies k_phasic=0.1, h_phasic="
                     "0.5 to phasic cells (D-87, 'to surface sustained activity "
                     "into the plausibility band'); steady cells use k=2.0, "
                     "h=4.0. The result JSON records only the steady pair."),
        },
        "plausibility_band": {"min": BAND_MIN, "max": BAND_MAX,
                              "note": ("phasic G3 passes iff n_boundaries in "
                                       "[1,20]; short trajectories never reach "
                                       "20, so the gate is effectively "
                                       "n_boundaries >= 1.")},
        "n_phasic_cells": len(phasic), "n_steady_cells": len(steady),
        "reproduction_check": {
            "note": ("real_phasic.per_workload.plausibility_pass_fraction "
                     "matches plan04_segmenter_results.json per_workload "
                     "(batched 0.8333, selective 0.75, seq 0.75, slowburn "
                     "0.9167, scanner 1.0)."),
            "phasic_per_workload": real_phasic["per_workload"],
            "steady_per_workload_own_WH": real_steady_own["per_workload"],
        },
        "real": {
            "phasic": real_phasic,
            "steady_own_WH": real_steady_own,
            "steady_through_phasic_gate": real_steady_as_null,
        },
        "nulls": {
            "A_steady_as_null": {
                "description": ("real steady-workload trajectories scored by the "
                                "phasic detector (k=0.1, h=0.5, (8,4)); pass iff "
                                "n_boundaries in [1,20]"),
                "band_pass_fraction": real_steady_as_null["band_pass_fraction"],
                "p_n_eq_zero": real_steady_as_null["p_n_eq_zero"],
                "n_pred_median": real_steady_as_null["n_pred_median"]},
            "B_temporal_shuffle": null_b,
            "C_iid_gaussian": null_c,
        },
        "discrimination": {
            "real_phasic_band_pass": rp,
            "null_steady_as_null_band_pass": a,
            "null_temporal_shuffle_band_pass": b,
            "null_iid_gaussian_band_pass": c,
            "gaps_real_minus_null": gaps,
            "band_near_unfalsifiable": near_unfalsifiable,
            "interpretation": (
                "The phasic detector is tuned to k=0.1 specifically to enter the "
                "[1,20] band, which passes a cell whenever CUSUM fires at least "
                "once. If the null pass-rates (steady-as-null, temporal-shuffle, "
                "IID) are close to the real phasic pass-rate, the gate does not "
                "certify detected PHASE STRUCTURE -- it only certifies the "
                "trajectory is not flat. A small real-minus-null gap means the "
                "marker-less G3 acceptance is near-unfalsifiable and should be "
                "reported as a plausibility screen, not a phase-detection "
                "result."),
        },
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
