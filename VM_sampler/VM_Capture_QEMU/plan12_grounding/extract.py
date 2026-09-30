#!/usr/bin/env python3
"""extract.py -- move 1 of plan12_grounding (SPEC move 1 and section 3): the per-page store and the
per-pair series, one recording at a time, resumable per recording.

  python3 -m plan12_grounding.extract extract --out O (--root R | --ssh USER@HOST --remote-root RR)
      [--store S] [--only REGEX] [--limit N] [--dry-run]

For each admissible recording of `<out>/cells.csv`:
  1. `source.fetch_trajectory(rec_rel)`: the capture's substrate trajectory beside the chain (in place
     for a local root; one rsync into the cache for the server, read only);
  2. `plan10_analysis.runner.extract.extract_from_trajectory(rec_rel, csv, speed=2,
     columns=["hamming", "cosine", "l0", "l1"], store=--store, n_pages=262144)`: the console's own
     L1 store, keyed as the console keys it (the recording's four-part relative path), so the
     console's SPL work and this toolkit share stores;
  3. `source.cleanup_fetched(rec_rel)` (ssh only): the fetched copy leaves the cache once verified
     identical on the server, size and sha256 (the approved deletion, SPEC 1.3);
  4. the per-pair series of SPEC section 3 into `<out>/series/<cell_id>.npz`: `N_t`, `H_t`, `A_t`
     and the stored extras, over the rows with h > 0, every pair kept with its index. The keep-first
     cut of the three double recordings (declared/keep_first_pairs.csv, AA A8) is applied here with
     the encoding toolkit's rule (rows with seq <= seq_first + N - 1); the start-up cuts are NOT
     applied (analysis time; the convention is in the file's metadata).
Every recording's store path, its sha256 and its series file go to `<out>/moves/01_extract/extract.json`
(rewritten after each recording) and `recordings.csv`. A recording whose series exists and whose
store is complete is reused. A recording that cannot be fetched or read is recorded as refused and
the move goes on; the move exits 1 at the end when any recording was refused, so the driver stops
and the record book names it.
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
import re  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

from plan12_grounding import __version__, toolkit_fingerprint  # noqa: E402
from plan12_grounding.run_moves import (  # noqa: E402
    DEFAULT_STORE, add_source_args, describe_source, now_iso, read_json, sha256_file, source_from_args, write_json,
)
from plan10_analysis import sources  # noqa: E402
from plan10_analysis.runner import extract as L1  # noqa: E402  (the console's L1 store; imported, never copied)
from plan10_analysis.runner import trajectory  # noqa: E402
from plan11_encoding_ladder import schema  # noqa: E402

CITATION = ("plan12_grounding/SPEC.md move 1 and section 3 (the angle theta = arccos(clip(1 - d, 0, 1)); N_t, H_t, A_t); "
            "plan10_analysis/runner/extract.py extract_from_trajectory; plan11 extract._stream (the keep-first rule)")
COLUMNS = ["hamming", "cosine", "l0", "l1"]
SPEED = 2                    # the corpus's differ speed, assumed (config substrateSpeed; DATA_OPS_BRIEF section 1)
N_BINS = 18                  # SPEC section 3: an 18-bin histogram of theta over [0, pi/2]
SERIES_SCHEMA = "plan12.series.v1"
PAIR_CONVENTION = ("pair index = the L1 store's seq, 1-based: plan10_analysis/runner/trajectory.py adds one to the "
                   "trajectory's 0-based seq, so a pair here means the same pair it means everywhere in the console")
CUT_CONVENTION = ("start-up cuts are applied at analysis time, never here: a cut of H pairs drops the first H pairs of this "
                  "series in pair-index order (plan11_encoding_ladder/series.py rung_series: n_series = n_pairs - 1 - head_drop, "
                  "the first head_drop rows dropped), H = series.head_drop_for(load_head_drop(declared/head_drop_values.csv), "
                  "kernel, role): 16 declared, and 112 measured (SPEC section 2); both are reported side by side")
KEEP_FIRST_RULE = ("keep-first (AA A8, plan11 extract._stream): rows with seq <= seq_first + N - 1 are kept, a seq gap counts "
                   "as a pair; later rows are dropped here")


def load_cells(out: Path) -> list[dict]:
    with open(out / "cells.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def series_path(out: Path, cell_id: str) -> Path:
    return out / "series" / f"{cell_id}.npz"


def read_series_meta(path: Path) -> dict | None:
    try:
        with np.load(path, allow_pickle=False) as z:
            return json.loads(str(z["meta"]))
    except Exception:
        return None


def build_series(npz_path: Path, rec: dict, keep_first_pairs: int | None) -> tuple[dict, dict]:
    """The per-pair series of SPEC section 3 from one L1 store. Returns (arrays, meta)."""
    z = L1.load(npz_path)
    seq = np.asarray(z["seq"]).astype(np.int64)
    h = np.asarray(z["hamming"]).astype(np.float64)
    d = np.asarray(z["cosine"]).astype(np.float64)
    n_rows_file = int(seq.size)
    seq_first_file = int(seq.min()) if seq.size else None
    seq_last_file = int(seq.max()) if seq.size else None
    keep = np.ones(seq.size, dtype=bool)
    bound = None
    if keep_first_pairs and seq.size:
        bound = seq_first_file + int(keep_first_pairs) - 1
        keep &= seq <= bound
    hpos = h > 0
    m = keep & hpos
    seq_k, h_k, d_k = seq[m], h[m], d[m]
    theta = np.arccos(np.clip(1.0 - d_k, 0.0, 1.0))
    pairs, inv = np.unique(seq_k, return_inverse=True)
    n = int(pairs.size)
    N = np.bincount(inv, minlength=n).astype(np.int64)
    H = np.bincount(inv, weights=h_k, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        A = np.bincount(inv, weights=h_k * theta, minlength=n) / H
        A_unw = np.bincount(inv, weights=theta, minlength=n) / N
    C = np.bincount(inv, weights=h_k * np.cos(theta), minlength=n)
    S = np.bincount(inv, weights=h_k * np.sin(theta), minlength=n)
    n90 = np.bincount(inv, weights=(d_k >= 1.0).astype(np.float64), minlength=n).astype(np.int64)
    n0 = np.bincount(inv, weights=(d_k <= 0.0).astype(np.float64), minlength=n).astype(np.int64)
    edges = np.linspace(0.0, np.pi / 2, N_BINS + 1)
    b = np.clip(np.searchsorted(edges, theta, side="right") - 1, 0, N_BINS - 1)
    hist = np.zeros((n, N_BINS), dtype=np.int64)
    np.add.at(hist, (inv, b), 1)
    seq_first_kept = int(pairs.min()) if n else None
    seq_last_kept = int(pairs.max()) if n else None
    n_gaps = int((seq_last_kept - seq_first_kept + 1) - n) if n else 0
    arrays = {"pair": pairs.astype(np.int64), "N": N, "H": H, "A": A, "C": C, "S": S, "A_unweighted": A_unw,
              "n_at_90": n90, "n_at_0": n0, "hist18": hist, "hist_edges": edges}
    meta = {
        "schema": SERIES_SCHEMA, "citation": CITATION, "package_version": __version__, "complete": True,
        "cell_id": rec["cell_id"], "rec_rel": rec["rec_rel"], "kernel": rec["kernel"], "role": rec["role"],
        "seed": rec.get("seed"), "rep": rec.get("rep"), "campaign": rec.get("campaign"),
        "columns": COLUMNS, "speed": SPEED, "speed_assumed": True,
        "series": {"N": "page count: number of changed pages (rows with h > 0) in the pair",
                   "H": "bits flipped: sum of h", "A": "bit-weighted mean angle: sum(h * theta) / sum(h), theta = arccos(clip(1 - d, 0, 1)) in [0, pi/2]",
                   "C": "sum(h * cos theta)", "S": "sum(h * sin theta)", "A_unweighted": "mean theta over the pair's rows",
                   "n_at_90": "rows with d = 1 (theta = pi/2; the zero-page convention)", "n_at_0": "rows with d = 0 and h > 0 (angles under about 0.028 degrees are stored as 0)",
                   "hist18": "counts of theta in 18 equal bins over [0, pi/2] (hist_edges)"},
        "pair_convention": PAIR_CONVENTION, "cut_convention": CUT_CONVENTION,
        "cuts_pairs": {"declared": rec.get("cut_declared"), "measured": rec.get("cut_measured")},
        "head_drop_pairs_declared": rec.get("head_drop_pairs"),
        "keep_first_pairs": (int(keep_first_pairs) if keep_first_pairs else None), "keep_first_rule": KEEP_FIRST_RULE,
        "keep_first_bound_seq": bound,
        "n_rows_file": n_rows_file, "n_pairs_file": int(z["n_pairs"]), "seq_first_file": seq_first_file, "seq_last_file": seq_last_file,
        "n_rows_dropped_after_keep_first": int((~keep).sum()), "n_rows_h0_dropped": int((keep & ~hpos).sum()),
        "n_rows_kept": int(m.sum()), "n_pairs_series": n, "seq_first_kept": seq_first_kept, "seq_last_kept": seq_last_kept,
        "n_pairs_missing_in_kept_range": n_gaps,
        "store": {"npz": str(npz_path), "meta": str(npz_path.with_name(npz_path.name.replace(".npz", ".meta.json"))),
                  "rec_id": rec["rec_rel"], "n_pages": int(z["n_pages"])},
        "toolkit_fingerprint": toolkit_fingerprint()["sha256"], "written_at": now_iso(),
    }
    return arrays, meta


def save_series(path: Path, arrays: dict, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **arrays)
    os.replace(tmp, path)


def process(rec: dict, src, store: Path, out: Path, dry_run: bool) -> dict:
    cell_id, rec_rel = rec["cell_id"], rec["rec_rel"]
    kf = int(rec["keep_first_pairs"]) if str(rec.get("keep_first_pairs") or "").strip() else None
    sp = series_path(out, cell_id)
    r: dict = {"cell_id": cell_id, "rec_rel": rec_rel, "status": None, "series": str(sp), "keep_first_pairs": kf,
               "started_at": now_iso()}
    meta = read_series_meta(sp) if sp.exists() else None
    reusable = bool(meta and meta.get("complete") and Path(meta.get("store", {}).get("npz", "")).exists())
    if reusable and not dry_run:
        r.update(status="reused", store=meta["store"]["npz"], store_sha256=sha256_file(meta["store"]["npz"]),
                 n_pairs=meta.get("n_pairs_series"), n_rows=meta.get("n_rows_kept"), finished_at=now_iso())
        return r
    if dry_run:
        lines = [f"[extract] dry run {cell_id} ({rec_rel}):" + ("  (a complete series exists: a real run would reuse it and do none of this)" if reusable else "")]
        if src.kind == "ssh":
            dest = src.cache / rec_rel
            lines.append("  fetch (read only): " + " ".join(src.rsync_trajectory_argv(rec_rel, dest)))
            lines.append(f"  extract locally: extract_from_trajectory({rec_rel!r}, <fetched csv>, speed={SPEED}, columns={COLUMNS}, store={store})")
            lines.append("  verify on the server before removing the fetched copy (read only): " + src.verify_cmd(rec_rel, ["<the fetched trajectory file>"]))
        else:
            lines.append(f"  trajectory: {trajectory.find(src.root / rec_rel)}")
            lines.append(f"  extract locally: extract_from_trajectory({rec_rel!r}, ..., speed={SPEED}, columns={COLUMNS}, store={store})")
        lines.append(f"  series: {sp}" + (f" (keep-first {kf} pairs)" if kf else ""))
        print("\n".join(lines))
        r.update(status="dry-run", finished_at=now_iso())
        return r
    t0 = time.time()
    try:
        traj = src.fetch_trajectory(rec_rel)
    except sources.SourceError as e:
        r.update(status=f"refused: fetch failed: {str(e)[:300]}", finished_at=now_iso())
        return r
    if traj is None:
        r.update(status="refused: no trajectory beside the chain", finished_at=now_iso())
        return r
    r["trajectory"] = str(traj)
    r["fetched"] = src.kind == "ssh"
    try:
        npz = L1.extract_from_trajectory(rec_rel, traj, SPEED, COLUMNS, store, n_pages=schema.N_PAGES)
    except (trajectory.TrajectoryError, RuntimeError, OSError, ValueError) as e:
        r.update(status=f"refused: trajectory unreadable: {str(e)[:300]}")
        npz = None
    if src.kind == "ssh":
        try:
            r["cleanup"] = src.cleanup_fetched(rec_rel)
        except Exception as e:                       # noqa: BLE001  the fetched copy stays; nothing else is touched
            r["cleanup"] = {"verified": False, "deleted": False, "reason": f"cleanup error: {str(e)[:200]}"}
    if npz is None:
        r["finished_at"] = now_iso()
        return r
    r["store"] = str(npz)
    r["store_meta"] = str(npz.with_name(npz.name.replace(".npz", ".meta.json")))
    r["store_sha256"] = sha256_file(npz)
    rec2 = dict(rec)
    arrays, meta = build_series(npz, rec2, kf)
    save_series(sp, arrays, meta)
    r.update(status="done", n_pairs=meta["n_pairs_series"], n_rows=meta["n_rows_kept"], n_pairs_file=meta["n_pairs_file"],
             n_rows_h0_dropped=meta["n_rows_h0_dropped"], n_pairs_missing=meta["n_pairs_missing_in_kept_range"],
             series_sha256=sha256_file(sp), elapsed_s=round(time.time() - t0, 3), finished_at=now_iso())
    return r


def extract(o: argparse.Namespace) -> int:
    out = Path(os.path.expanduser(o.out))
    src = source_from_args(o)
    store = L1.store_dir(o.store)
    cells = load_cells(out)
    params = read_json(out / "params.json") if (out / "params.json").is_file() else {}
    cuts = params.get("cuts", {})
    rx = re.compile(o.only) if o.only else None
    todo = [c for c in cells if c.get("admissible", "").lower() == "true" and (rx is None or rx.search(c["cell_id"]))]
    if o.limit:
        todo = todo[: int(o.limit)]
    mdir = out / "moves" / "01_extract"
    mdir.mkdir(parents=True, exist_ok=True)
    jpath = mdir / "extract.json"
    doc = read_json(jpath) if jpath.is_file() and not o.dry_run else {}
    recs: dict = dict(doc.get("recordings", {})) if isinstance(doc.get("recordings"), dict) else {}
    print(f"[extract] {len(todo)} admissible recording(s); store {store}; source {describe_source(src)}", flush=True)
    for i, c in enumerate(todo, 1):
        c = dict(c)
        c["cut_declared"], c["cut_measured"] = cuts.get("declared_pairs"), cuts.get("measured_pairs")
        r = process(c, src, store, out, o.dry_run)
        if not o.dry_run:
            recs[c["cell_id"]] = r
            write_json(jpath, {"schema": "plan12.extract.v1", "citation": CITATION, "package_version": __version__,
                               "params": {"store": str(store), "columns": COLUMNS, "speed": SPEED, "speed_assumed": True,
                                          "n_pages": schema.N_PAGES, "source": describe_source(src),
                                          "pair_convention": PAIR_CONVENTION, "cut_convention": CUT_CONVENTION,
                                          "keep_first_rule": KEEP_FIRST_RULE, "toolkit_fingerprint": toolkit_fingerprint()["sha256"]},
                               "recordings": recs, "updated_at": now_iso()})
        extra = f" {r.get('n_pairs')} pairs, {r.get('n_rows')} rows" if r.get("status") in ("done", "reused") else ""
        cl = r.get("cleanup")
        print(f"[extract] [{i}/{len(todo)}] {c['cell_id']}: {r['status']}{extra}" + (f" · cache: {cl.get('reason')}" if cl else ""), flush=True)
    if o.dry_run:
        return 0
    rows = list(recs.values())
    cols = ["cell_id", "rec_rel", "status", "n_pairs", "n_rows", "n_pairs_file", "keep_first_pairs", "store", "store_sha256",
            "series", "series_sha256", "elapsed_s", "finished_at"]
    with open(mdir / "recordings.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
    bad = [r for r in rows if not str(r.get("status", "")).startswith(("done", "reused"))]
    n_done = sum(1 for r in rows if str(r.get("status", "")).startswith(("done", "reused")))
    print(f"[extract] {n_done} recording(s) with a complete store and series, {len(bad)} refused", flush=True)
    if bad:
        for r in bad:
            print(f"[extract]   {r['cell_id']}: {r['status']}", file=sys.stderr)
        return 1
    missing = [c["cell_id"] for c in todo if c["cell_id"] not in recs]
    if missing:
        print(f"[extract] not processed: {missing}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="plan12_grounding.extract", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("extract", help="move 1: the store and the per-pair series, per admissible recording")
    p.add_argument("--out", required=True)
    add_source_args(p)
    p.add_argument("--store", default=DEFAULT_STORE)
    p.add_argument("--only", default=None, help="regex on cell_id")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="print the fetch and extract commands; touch nothing")
    o = ap.parse_args(argv)
    try:
        return extract(o)
    except FileNotFoundError as exc:
        print(f"missing input: {exc}", file=sys.stderr)
        return 2
    except SystemExit as exc:
        if exc.code not in (None, 0):
            print(str(exc), file=sys.stderr)
        return 2 if exc.code not in (None, 0) else 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
