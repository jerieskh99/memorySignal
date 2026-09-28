#!/usr/bin/env python3
"""variance.py -- G-V, the variance decomposition report (SPEC section 3.8).

Citation: P2_STRUCTURE.md section V 5.2 'Reports and disclosures' G-V; CR 2.3 item 35;
SPEC_review_al_farabi.md items 2.3 (G-K0's relabelling applied at read time) and 2.9 (b).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plan11_encoding_ladder import series as S
from plan11_encoding_ladder import models as M
from plan11_encoding_ladder import verdicts as V

CIT_GV = "P2 Sec. V 5.2 G-V; CR 2.3 item 35"


def variance_levels(Xcell: np.ndarray, kernels, archetypes) -> dict:
    """The G-V levels (P2 Sec. V 5.2 G-V; CR 2.3 item 35; CIT_GV). Per feature: L0 = mean over
    kernels of the population variance across the kernel's cells; L2 = mean over archetypes with
    at least two kernels of the variance across the archetype's kernel means; L3 = the variance across archetype means (archetypes with at least one kernel).
    Population variances (section 8 item 30)."""
    kernels = np.asarray(kernels).astype(str); archetypes = np.asarray(archetypes).astype(str)
    uk = list(dict.fromkeys(kernels.tolist()))
    kmean = {k: np.nanmean(Xcell[kernels == k], axis=0) for k in uk}
    L0 = np.nanmean(np.stack([np.nanvar(Xcell[kernels == k], axis=0) for k in uk]), axis=0)
    ka = {k: archetypes[kernels == k][0] for k in uk}
    ua = list(dict.fromkeys(ka.values()))
    amean = {a: np.nanmean(np.stack([kmean[k] for k in uk if ka[k] == a]), axis=0) for a in ua}
    multi = [a for a in ua if sum(1 for k in uk if ka[k] == a) >= 2]
    L2 = np.nanmean(np.stack([np.nanvar(np.stack([kmean[k] for k in uk if ka[k] == a]), axis=0) for a in multi]), axis=0) if multi else np.full(Xcell.shape[1], np.nan)
    L3 = np.nanvar(np.stack([amean[a] for a in ua]), axis=0) if len(ua) >= 1 else np.full(Xcell.shape[1], np.nan)
    return {"L0": L0, "L2": L2, "L3": L3, "n_kernels": len(uk), "n_archetypes": len(ua), "n_archetypes_multi": len(multi)}


def gate_gv(out: Path) -> Path:
    """G-V (P2 Sec. V 5.2 G-V; CR 2.3 item 35). On the normalized features at the selected point of
    each rung, the per-cell vector is the mean over the cell's windows; per feature L0, L2, L3 as
    ``variance_levels`` (after G-K0's relabelling, applied at read time). A rung with L0 > L3 for
    every feature is GV_NOT_ESTIMABLE, else GV_ESTIMABLE; a constant feature (L0 = L3 = 0, e.g. a duty
    of 1.0 everywhere) carries no information and is left out of 'every feature' (``constant`` column,
    ``n_features_constant``). Output gates/gv.csv (rung, feature, L0, L2,
    L3, L0_over_L3) and gates/gv_summary.csv (rung, n_features, n_features_L0_gt_L3, verdict). A rung
    without a selection writes ``not run: no selection for <rung>``."""
    out = Path(out)
    rows, summ = [], []
    for rung in S.RUNGS:
        gid, _ = S.selected_grid_id(out, rung, None)
        if gid is None:
            summ.append({"rung": rung, "verdict": V.not_run(f"no selection for {rung}")}); continue
        try:
            data = M.prepare_split_data(out, rung, gid, normalized=True)
        except FileNotFoundError:
            summ.append({"rung": rung, "grid_id": gid, "verdict": V.not_run("feature file missing")}); continue
        cells, kern, arch, Xc = M.cell_vectors(data)
        if len(cells) < 2:
            summ.append({"rung": rung, "grid_id": gid, "verdict": V.not_run("fewer than two cells")}); continue
        lv = variance_levels(Xc, kern, arch)
        names = data["names"]
        n_gt, n_const = 0, 0
        for j, f in enumerate(names):
            l0, l2, l3 = float(lv["L0"][j]), float(lv["L2"][j]), float(lv["L3"][j])
            const = bool((np.isnan(l0) or l0 == 0.0) and (np.isnan(l3) or l3 == 0.0))
            n_const += const
            gt = bool(not const and not np.isnan(l0) and not np.isnan(l3) and l0 > l3)
            n_gt += gt
            rows.append({"rung": rung, "feature": f, "L0": l0, "L2": l2, "L3": l3, "L0_over_L3": (l0 / l3) if (l3 and not np.isnan(l3)) else None,
                         "constant": const})
        n_inf = len(names) - n_const
        summ.append({"rung": rung, "grid_id": gid, "n_features": len(names), "n_features_constant": n_const, "n_features_L0_gt_L3": n_gt,
                     "verdict": V.GV_NOT_ESTIMABLE if (n_inf > 0 and n_gt == n_inf) else V.GV_ESTIMABLE})
    p = S.write_csv(out / "gates" / "gv.csv", ("rung", "feature", "L0", "L2", "L3", "L0_over_L3", "constant"), rows)
    S.write_csv(out / "gates" / "gv_summary.csv", ("rung", "grid_id", "n_features", "n_features_constant", "n_features_L0_gt_L3", "verdict"), summ)
    S.write_params(p, "plan11.gv.v1", {"variance": "population", "L2_over": "archetypes with at least two kernels",
                                       "per_cell_vector": "mean over windows at the selected point", "gk0_applied_at_read_time": True,
                                       "inputs_sha256": S.inputs_sha256([out / "cells.csv", out / "gates" / "selection.json", out / "gates" / "gk0.csv"], out)}, CIT_GV)
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="variance.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed-offset", type=int, default=0)
    args = ap.parse_args(argv)
    out = Path(args.out)
    if not (out / "cells.csv").is_file():
        print(f"missing input: {out / 'cells.csv'}", file=sys.stderr); return 2
    print(gate_gv(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
