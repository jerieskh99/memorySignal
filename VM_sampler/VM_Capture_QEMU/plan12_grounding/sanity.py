#!/usr/bin/env python3
"""sanity.py -- move 2 of plan12_grounding (SPEC move 2): the checks on every row and every recording.

  python3 -m plan12_grounding.sanity check --out O [--dry-run]

On every kept row of every recording's L1 store (the rows the series was built from: the keep-first
cut applied, as move 1 recorded it): `l0 <= h <= 8 * l0`, `l0 <= l1 <= 255 * l0`, `0 <= d <= 1`,
finite values. Per recording: the series' pair indices consecutive with no gap or repeat, no pair
with zero changed pages (a pair of the kept range with no row of h > 0 is a gap), and the series'
page counts equal to the store's recount. Angle conventions counted, not judged: rows at d = 1,
rows at d = 0 with h > 0, the smallest positive d. Writes `<out>/moves/02_sanity/counts.csv` (one row
per recording), `violations.csv` (rule, count, up to five example rows) and `sanity.json`. Any
violation exits 1, which stops the driver (SPEC: a violation stops the run).
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
import traceback  # noqa: E402

import numpy as np  # noqa: E402

from plan12_grounding import __version__, toolkit_fingerprint  # noqa: E402
from plan12_grounding.run_moves import now_iso, read_json, write_json  # noqa: E402
from plan10_analysis.runner import extract as L1  # noqa: E402

CITATION = "plan12_grounding/SPEC.md move 2 (the magnitude rules, the pair rules, the angle conventions counted)"
RULES = ("finite", "l0 <= h <= 8*l0", "l0 <= l1 <= 255*l0", "0 <= d <= 1", "pairs consecutive, no repeat",
         "no pair with zero changed pages", "series page counts equal the store's recount")
COUNT_COLUMNS = ("cell_id", "n_rows_store", "n_rows_kept", "n_rows_dropped_keep_first", "n_pairs", "n_at_d1",
                 "n_at_d0_hpos", "min_positive_d", "n_violations", "violated_rules")
N_EXAMPLES = 5


def _examples(seq, page, h, d, l0, l1, mask, k: int = N_EXAMPLES) -> list[dict]:
    idx = np.flatnonzero(mask)[:k]
    return [{"seq": int(seq[i]), "page_index": int(page[i]), "h": float(h[i]), "d": float(d[i]), "l0": float(l0[i]), "l1": float(l1[i])} for i in idx]


def check_recording(cell_id: str, store_npz: Path, series_npz: Path) -> tuple[dict, list[dict]]:
    z = L1.load(store_npz)
    seq = np.asarray(z["seq"]).astype(np.int64)
    page = np.asarray(z["page_index"]).astype(np.int64)
    h, d, l0, l1 = (np.asarray(z[c]).astype(np.float64) for c in ("hamming", "cosine", "l0", "l1"))
    with np.load(series_npz, allow_pickle=False) as s:
        meta = json.loads(str(s["meta"]))
        pairs = np.asarray(s["pair"]).astype(np.int64)
        N = np.asarray(s["N"]).astype(np.int64)
    keep = np.ones(seq.size, dtype=bool)
    if meta.get("keep_first_bound_seq") is not None:
        keep &= seq <= int(meta["keep_first_bound_seq"])
    seq_k, page_k, h_k, d_k, l0_k, l1_k = seq[keep], page[keep], h[keep], d[keep], l0[keep], l1[keep]
    viol: list[dict] = []
    fin = np.isfinite(h_k) & np.isfinite(d_k) & np.isfinite(l0_k) & np.isfinite(l1_k)
    if not fin.all():
        viol.append({"cell_id": cell_id, "rule": RULES[0], "count": int((~fin).sum()), "examples": _examples(seq_k, page_k, h_k, d_k, l0_k, l1_k, ~fin)})
    bad = ~((l0_k <= h_k) & (h_k <= 8.0 * l0_k))
    if bad.any():
        viol.append({"cell_id": cell_id, "rule": RULES[1], "count": int(bad.sum()), "examples": _examples(seq_k, page_k, h_k, d_k, l0_k, l1_k, bad)})
    bad = ~((l0_k <= l1_k) & (l1_k <= 255.0 * l0_k))
    if bad.any():
        viol.append({"cell_id": cell_id, "rule": RULES[2], "count": int(bad.sum()), "examples": _examples(seq_k, page_k, h_k, d_k, l0_k, l1_k, bad)})
    bad = ~((d_k >= 0.0) & (d_k <= 1.0))
    if bad.any():
        viol.append({"cell_id": cell_id, "rule": RULES[3], "count": int(bad.sum()), "examples": _examples(seq_k, page_k, h_k, d_k, l0_k, l1_k, bad)})
    # per recording: the pair index
    if pairs.size:
        dif = np.diff(pairs)
        if not np.all(dif == 1):
            viol.append({"cell_id": cell_id, "rule": RULES[4], "count": int(np.sum(dif != 1)),
                         "examples": [{"after_pair": int(pairs[i]), "next_pair": int(pairs[i + 1])} for i in np.flatnonzero(dif != 1)[:N_EXAMPLES]]})
        expected = np.arange(int(pairs.min()), int(pairs.max()) + 1)
        missing = np.setdiff1d(expected, pairs)
        if missing.size or (N <= 0).any():
            viol.append({"cell_id": cell_id, "rule": RULES[5], "count": int(missing.size + int((N <= 0).sum())),
                         "examples": [{"pair": int(p)} for p in missing[:N_EXAMPLES]] + [{"pair": int(pairs[i]), "N": int(N[i])} for i in np.flatnonzero(N <= 0)[:N_EXAMPLES]]})
    hp = h_k > 0
    u, cnt = np.unique(seq_k[hp], return_counts=True)
    if u.size != pairs.size or not np.array_equal(u, pairs) or not np.array_equal(cnt, N):
        viol.append({"cell_id": cell_id, "rule": RULES[6], "count": int(max(u.size, pairs.size)),
                     "examples": [{"pairs_store": int(u.size), "pairs_series": int(pairs.size)}]})
    pos = d_k[d_k > 0]
    counts = {"cell_id": cell_id, "n_rows_store": int(seq.size), "n_rows_kept": int(seq_k.size),
              "n_rows_dropped_keep_first": int((~keep).sum()), "n_pairs": int(pairs.size),
              "n_at_d1": int((d_k == 1.0).sum()), "n_at_d0_hpos": int(((d_k == 0.0) & hp).sum()),
              "min_positive_d": (float(pos.min()) if pos.size else ""),
              "n_violations": int(sum(v["count"] for v in viol)), "violated_rules": "; ".join(v["rule"] for v in viol)}
    return counts, viol


def check(o: argparse.Namespace) -> int:
    out = Path(os.path.expanduser(o.out))
    ex = read_json(out / "moves" / "01_extract" / "extract.json")
    recs = {k: v for k, v in (ex.get("recordings") or {}).items() if str(v.get("status", "")).startswith(("done", "reused"))}
    if o.dry_run:
        print(f"[sanity] dry run: would check {len(recs)} recording(s)' stores and series under {out}; rules: {'; '.join(RULES)}")
        return 0
    mdir = out / "moves" / "02_sanity"
    mdir.mkdir(parents=True, exist_ok=True)
    all_counts, all_viol = [], []
    for i, (cid, r) in enumerate(sorted(recs.items()), 1):
        counts, viol = check_recording(cid, Path(r["store"]), Path(r["series"]))
        all_counts.append(counts)
        all_viol.extend(viol)
        flag = f"  VIOLATIONS: {counts['violated_rules']}" if viol else ""
        print(f"[sanity] [{i}/{len(recs)}] {cid}: {counts['n_rows_kept']} rows, {counts['n_pairs']} pairs, "
              f"d=1: {counts['n_at_d1']}, d=0 with h>0: {counts['n_at_d0_hpos']}, min d>0: {counts['min_positive_d']}{flag}", flush=True)
    with open(mdir / "counts.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(COUNT_COLUMNS))
        w.writeheader()
        for c in all_counts:
            w.writerow(c)
    with open(mdir / "violations.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell_id", "rule", "count", "examples"])
        for v in all_viol:
            w.writerow([v["cell_id"], v["rule"], v["count"], json.dumps(v["examples"])])
    total = int(sum(v["count"] for v in all_viol))
    mins = [c["min_positive_d"] for c in all_counts if c["min_positive_d"] != ""]
    summary = {
        "schema": "plan12.sanity.v1", "citation": CITATION, "package_version": __version__, "written_at": now_iso(),
        "toolkit_fingerprint": toolkit_fingerprint()["sha256"],
        "rules": list(RULES), "n_recordings": len(all_counts),
        "n_rows_checked": int(sum(c["n_rows_kept"] for c in all_counts)),
        "n_rows_dropped_keep_first": int(sum(c["n_rows_dropped_keep_first"] for c in all_counts)),
        "violations_total": total, "zero_violations": total == 0,
        "violations_by_rule": {r: int(sum(v["count"] for v in all_viol if v["rule"] == r)) for r in RULES},
        "recordings_with_violations": sorted({v["cell_id"] for v in all_viol}),
        "angle_conventions": {"rows_at_d1": int(sum(c["n_at_d1"] for c in all_counts)),
                              "rows_at_d0_with_h_positive": int(sum(c["n_at_d0_hpos"] for c in all_counts)),
                              "smallest_positive_d": (float(min(mins)) if mins else None)},
    }
    write_json(mdir / "sanity.json", summary)
    print(f"[sanity] {summary['n_recordings']} recordings, {summary['n_rows_checked']} rows checked: {total} violation(s); "
          f"rows at d=1: {summary['angle_conventions']['rows_at_d1']}, at d=0 with h>0: {summary['angle_conventions']['rows_at_d0_with_h_positive']}, "
          f"smallest positive d: {summary['angle_conventions']['smallest_positive_d']}", flush=True)
    if total:
        print(f"[sanity] stopped: {total} violation(s) in {len(summary['recordings_with_violations'])} recording(s); see {mdir / 'violations.csv'}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="plan12_grounding.sanity", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check", help="move 2: the sanity checks")
    p.add_argument("--out", required=True)
    p.add_argument("--dry-run", action="store_true")
    o = ap.parse_args(argv)
    try:
        return check(o)
    except FileNotFoundError as exc:
        print(f"missing input: {exc}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
