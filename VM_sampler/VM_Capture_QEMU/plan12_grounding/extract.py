#!/usr/bin/env python3
"""extract.py -- move 1 of plan12_grounding (SPEC move 1 and section 3): the per-page store and the
per-pair series, one recording at a time, resumable per recording.

  python3 -m plan12_grounding.extract extract --out O (--root R | --ssh USER@HOST --remote-root RR)
      [--store S] [--encoding-out E] [--identity-cell CELL] [--only REGEX] [--limit N] [--force] [--dry-run]

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
(rewritten after each recording) and `recordings.csv`. A recording's series is reused only when its
metadata matches the current values (the keep-first cut, the cut convention, the series format, the
sha256 of this module) and its store exists; `--force` rebuilds every series from its store. A
recording that cannot be fetched or read is recorded as refused and the move goes on; the move exits
1 at the end when any recording was refused, so the driver stops and the record book names it.

The E0 identity check (SPEC 2, decision D2) runs here, while the chosen recording's trajectory is in
the cache: the first admissible kernel recording with an ok sidecar in the named encoding run (or
`--identity-cell`) is recomputed with `plan11_encoding_ladder.extract.extract_cell` and the sidecar's
own parameters, and the result must be byte-identical to the encoding run's `extract/<cell>/extract.csv`.
The record is `<out>/moves/01_extract/e0_identity.json`; move 6 reads it and never fetches.
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
    DEFAULT_STORE, add_source_args, describe_source, install_sigterm, now_iso, read_json, sha256_file, source_from_args, write_json,
)
from plan10_analysis import sources  # noqa: E402
from plan10_analysis.runner import extract as L1  # noqa: E402  (the console's L1 store; imported, never copied)
from plan10_analysis.runner import trajectory  # noqa: E402
from plan11_encoding_ladder import extract as E11  # noqa: E402  (extract_cell, for the E0 identity check; read only)
from plan11_encoding_ladder import schema  # noqa: E402


def series_code_sha256() -> str:
    """The sha256 of this module: a series is reused only when it was built by this code."""
    return sha256_file(Path(__file__))

CITATION = ("plan12_grounding/SPEC.md move 1 and section 3 (the angle theta = arccos(clip(1 - d, 0, 1)); N_t, H_t, A_t); "
            "plan10_analysis/runner/extract.py extract_from_trajectory; plan11 extract._stream (the keep-first rule)")
COLUMNS = ["hamming", "cosine", "l0", "l1"]
SPEED = 2                    # the corpus's differ speed, assumed (config substrateSpeed; DATA_OPS_BRIEF section 1)
N_BINS = 18                  # SPEC section 3: an 18-bin histogram of theta over [0, pi/2]
SERIES_SCHEMA = "plan12.series.v1"
SERIES_FORMAT = 2            # 2 (2026-10-01): every pair of the kept range is on the axis; a pair without a changed page reads N = 0, A = NaN
IDENTITY_RECORD = "e0_identity.json"
PAIR_CONVENTION = ("pair index = the L1 store's seq, 1-based: plan10_analysis/runner/trajectory.py adds one to the "
                   "trajectory's 0-based seq, so a pair here means the same pair it means everywhere in the console")
CUT_CONVENTION = ("start-up cuts are applied at analysis time, never here: a cut of H pairs drops the first H pairs AND the last "
                  "pair of this series, in pair-index order, exactly the rows plan11_encoding_ladder/series.py rung_series drops "
                  "(lo = head_drop, hi = n_rows - 1: n_series = n_pairs - 1 - head_drop); H = 16 declared and 112 measured "
                  "(params.json cuts; declared/head_drop_values.csv gives 16 for every kernel and for idle); both are reported side by side")
PAIR_AXIS = ("every pair of the kept range is on the axis (every seq present in the store within the keep-first bound); a pair "
             "without a row of h > 0, or whose rows were all excluded (move 9), reads N = 0, H = 0, C = S = 0 and A undefined (NaN)")
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


def build_series(npz_path: Path, rec: dict, keep_first_pairs: int | None, exclude_pages=None) -> tuple[dict, dict]:
    """The per-pair series of SPEC section 3 from one L1 store. Returns (arrays, meta). `exclude_pages`
    (move 9, rule A) is a set of page indices whose rows are left out before anything is counted;
    the metadata records how many rows that removed."""
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
    pairs = np.unique(seq[keep])                    # the pair axis: every pair of the kept range, whatever its rows hold
    n = int(pairs.size)
    m = keep.copy()
    n_rows_excluded_pages = 0
    if exclude_pages:
        page = np.asarray(z["page_index"]).astype(np.int64)
        drop = np.isin(page, np.fromiter(exclude_pages, dtype=np.int64))
        n_rows_excluded_pages = int((m & drop).sum())
        m &= ~drop
    hpos = h > 0
    m &= hpos
    seq_k, h_k, d_k = seq[m], h[m], d[m]
    theta = np.arccos(np.clip(1.0 - d_k, 0.0, 1.0))
    inv = np.searchsorted(pairs, seq_k)
    N = np.bincount(inv, minlength=n).astype(np.int64)
    H = np.bincount(inv, weights=h_k, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        A = np.where(H > 0, np.bincount(inv, weights=h_k * theta, minlength=n) / np.where(H > 0, H, 1.0), np.nan)
        A_unw = np.where(N > 0, np.bincount(inv, weights=theta, minlength=n) / np.where(N > 0, N, 1), np.nan)
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
    n_pairs_empty = int((N == 0).sum())
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
        "series_format": SERIES_FORMAT, "series_code_sha256": series_code_sha256(),
        "pair_convention": PAIR_CONVENTION, "pair_axis": PAIR_AXIS, "cut_convention": CUT_CONVENTION,
        "cuts_pairs": {"declared": rec.get("cut_declared"), "measured": rec.get("cut_measured")},
        "head_drop_pairs_declared": rec.get("head_drop_pairs"),
        "keep_first_pairs": (int(keep_first_pairs) if keep_first_pairs else None), "keep_first_rule": KEEP_FIRST_RULE,
        "keep_first_bound_seq": bound,
        "n_rows_file": n_rows_file, "n_pairs_file": int(z["n_pairs"]), "seq_first_file": seq_first_file, "seq_last_file": seq_last_file,
        "n_rows_dropped_after_keep_first": int((~keep).sum()), "n_rows_h0_dropped": int((keep & ~hpos).sum()),
        "excluded_pages": ({"n_pages": len(exclude_pages), "n_rows_removed": n_rows_excluded_pages, "rule": "move 9 rule A"} if exclude_pages else None),
        "n_rows_kept": int(m.sum()), "n_pairs_series": n, "seq_first_kept": seq_first_kept, "seq_last_kept": seq_last_kept,
        "n_pairs_missing_in_kept_range": n_gaps, "n_pairs_empty": n_pairs_empty,
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


def reusable_series(meta: dict | None, kf: int | None) -> tuple[bool, str]:
    """A series is reused only when its metadata matches the current values; the reason otherwise."""
    if not meta or not meta.get("complete"):
        return False, "no complete series"
    if not Path((meta.get("store") or {}).get("npz", "")).exists():
        return False, "its store is missing"
    if (meta.get("keep_first_pairs") or None) != (kf or None):
        return False, f"keep-first changed ({meta.get('keep_first_pairs')} -> {kf})"
    if meta.get("cut_convention") != CUT_CONVENTION:
        return False, "the cut convention changed"
    if meta.get("series_format") != SERIES_FORMAT:
        return False, f"the series format changed ({meta.get('series_format')} -> {SERIES_FORMAT})"
    if meta.get("series_code_sha256") != series_code_sha256():
        return False, "extract.py changed since the series was built"
    return True, "metadata matches"


def identity_check(enc_out: Path, cell_id: str, rec_rel: str, traj: Path, work: Path) -> dict:
    """Recompute one recording with plan11's extract_cell from its trajectory, with the sidecar's own
    parameters, and compare byte for byte with the encoding run's extract.csv (SPEC 2, decision D2)."""
    res = {"cell_id": cell_id, "rec_rel": rec_rel, "checked_at": now_iso(), "passed": False, "status": "failed"}
    sc_path = enc_out / "extract" / cell_id / "sidecar.json"
    sc = read_json(sc_path) if sc_path.is_file() else None
    if sc is None or sc.get("status") != "ok":
        res["reason"] = f"the encoding run has no ok sidecar for {cell_id}"
        return res
    prm = sc.get("params", {})
    kw = dict(n_pages=int(prm.get("n_pages", sc.get("N"))), page_size=int(prm.get("page_size", sc.get("page_size"))),
              quantiles=tuple(float(q) for q in prm.get("quantiles", sc.get("quantiles"))),
              persist_side=prm.get("persist_side", sc.get("persist_side", "t")),
              duration_s=float(prm.get("duration_s", sc.get("duration_s_declared", 600))),
              failed_count=prm.get("failed_count"), failed_count_source=prm.get("failed_count_source"),
              role=prm.get("role_override") or sc.get("role"), cell_id=cell_id, rep=sc.get("rep"),
              archetype_predicted=prm.get("archetype_override") or None,
              idle_markers=tuple(prm.get("idle_markers") or schema.IDLE_MARKERS_DEFAULT),
              keep_first_pairs=sc.get("keep_first_pairs"), keep_first_reason=sc.get("keep_first_reason"),
              keep_first_source=sc.get("keep_first_source"))
    res["params"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in kw.items()}
    ref = enc_out / "extract" / cell_id / "extract.csv"
    res["reference"] = {"path": str(ref), "sha256": sha256_file(ref) if ref.is_file() else "absent", "sidecar_sha256": sha256_file(sc_path)}
    res["trajectory"] = {"path": str(traj), "sha256": sha256_file(traj), "bytes": int(Path(traj).stat().st_size)}
    t0 = time.time()
    work.mkdir(parents=True, exist_ok=True)
    side = E11.extract_cell(traj, work, **kw)
    rp = work / "extract" / cell_id / "extract.csv"
    res["recomputed"] = {"status": side.get("status"), "path": str(rp), "n_pairs": side.get("n_pairs"), "seq_first": side.get("seq_first"),
                         "seq_last": side.get("seq_last"), "K_median": side.get("K_median"), "elapsed_s": round(time.time() - t0, 1)}
    if side.get("status") != "ok" or not rp.is_file():
        res["reason"] = f"recompute refused: {side.get('status')}"
        return res
    res["recomputed"]["sha256"] = sha256_file(rp)
    res["passed"] = res["recomputed"]["sha256"] == res["reference"]["sha256"]
    res["status"] = "passed" if res["passed"] else "failed"
    res["reference"].update({"n_pairs": sc.get("n_pairs"), "seq_first": sc.get("seq_first"), "seq_last": sc.get("seq_last"), "K_median": sc.get("K_median")})
    if not res["passed"]:
        res["first_difference"] = _first_difference(ref, rp)
    return res


def _first_difference(a: Path, b: Path) -> dict:
    with open(a, newline="") as fa, open(b, newline="") as fb:
        ra, rb = csv.reader(fa), csv.reader(fb)
        for i, (la, lb) in enumerate(zip(ra, rb)):
            if la != lb:
                cols = [j for j, (x, y) in enumerate(zip(la, lb)) if x != y]
                return {"row": i, "columns": cols[:5], "a": [la[j] for j in cols[:5]], "b": [lb[j] for j in cols[:5]]}
        return {"row": None, "note": "one file is a prefix of the other"}


def process(rec: dict, src, store: Path, out: Path, dry_run: bool, force: bool = False, ident: dict | None = None) -> dict:
    """One recording. `ident` names the identity check's recording ({cell_id, enc_out, work, needed}): when
    this is it, the check runs here, with the trajectory this move fetched, before the fetched copy leaves."""
    cell_id, rec_rel = rec["cell_id"], rec["rec_rel"]
    kf = int(rec["keep_first_pairs"]) if str(rec.get("keep_first_pairs") or "").strip() else None
    sp = series_path(out, cell_id)
    r: dict = {"cell_id": cell_id, "rec_rel": rec_rel, "status": None, "series": str(sp), "keep_first_pairs": kf,
               "started_at": now_iso()}
    meta = read_series_meta(sp) if sp.exists() else None
    reusable, why = reusable_series(meta, kf)
    if reusable and force:
        reusable, why = False, "--force"
    r["reuse"] = why
    check_here = bool(ident and ident.get("needed") and ident.get("cell_id") == cell_id)
    store_hit = L1.existing(store, rec_rel, SPEED, COLUMNS, None)
    if reusable and not dry_run and not check_here:
        r.update(status="reused", store=meta["store"]["npz"], store_sha256=sha256_file(meta["store"]["npz"]),
                 n_pairs=meta.get("n_pairs_series"), n_rows=meta.get("n_rows_kept"), finished_at=now_iso())
        return r
    if dry_run:
        lines = [f"[extract] dry run {cell_id} ({rec_rel}):" + ("  (a series exists and its metadata matches: a real run would reuse it)" if reusable else f"  (series to build: {why})")
                 + ("  (the E0 identity check would run on this recording)" if check_here else "")]
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
    traj = None
    need_traj = store_hit is None or check_here           # a store already in place is read, not fetched again, unless the check needs the file
    if need_traj:
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
    npz = store_hit
    if npz is None:
        try:
            npz = L1.extract_from_trajectory(rec_rel, traj, SPEED, COLUMNS, store, n_pages=schema.N_PAGES)
        except (trajectory.TrajectoryError, RuntimeError, OSError, ValueError) as e:
            r.update(status=f"refused: trajectory unreadable: {str(e)[:300]}")
            npz = None
    if check_here and traj is not None:
        try:
            ident["result"] = identity_check(Path(ident["enc_out"]), cell_id, rec_rel, Path(traj), Path(ident["work"]))
        except Exception as e:                       # noqa: BLE001  recorded as a failed check, never silent
            ident["result"] = {"cell_id": cell_id, "rec_rel": rec_rel, "checked_at": now_iso(), "passed": False, "status": "failed",
                               "reason": f"recompute raised: {str(e)[:300]}"}
        r["e0_identity"] = ident["result"].get("status")
    if src.kind == "ssh" and traj is not None:
        try:
            r["cleanup"] = src.cleanup_fetched(rec_rel)
        except Exception as e:                       # noqa: BLE001  the fetched copy stays; nothing else is touched
            r["cleanup"] = {"verified": False, "deleted": False, "reason": f"cleanup error: {str(e)[:200]}"}
    if npz is None:
        r["finished_at"] = now_iso()
        return r
    if reusable:                                      # the check needed the file; the series itself stands
        r.update(status="reused", store=str(npz), store_sha256=sha256_file(npz), n_pairs=meta.get("n_pairs_series"),
                 n_rows=meta.get("n_rows_kept"), finished_at=now_iso())
        return r
    r["store"] = str(npz)
    r["store_meta"] = str(npz.with_name(npz.name.replace(".npz", ".meta.json")))
    r["store_sha256"] = sha256_file(npz)
    rec2 = dict(rec)
    arrays, meta = build_series(npz, rec2, kf)
    save_series(sp, arrays, meta)
    r.update(status="done", n_pairs=meta["n_pairs_series"], n_rows=meta["n_rows_kept"], n_pairs_file=meta["n_pairs_file"],
             n_rows_h0_dropped=meta["n_rows_h0_dropped"], n_pairs_missing=meta["n_pairs_missing_in_kept_range"], n_pairs_empty=meta["n_pairs_empty"],
             series_sha256=sha256_file(sp), elapsed_s=round(time.time() - t0, 3), finished_at=now_iso())
    return r


def identity_plan(out: Path, todo: list[dict], enc_out: Path | None, wanted: str | None, force: bool) -> dict | None:
    """Which recording the E0 identity check recomputes, and whether it must run: the first admissible
    kernel recording (cells.csv order) with an ok sidecar in the encoding run, or `--identity-cell`;
    not needed when the record exists, passed, and still names the encoding run's current extract."""
    if enc_out is None:
        return None
    work = out / "moves" / "01_extract" / "e0_identity"
    rec_path = out / "moves" / "01_extract" / IDENTITY_RECORD
    cands = [c for c in todo if c.get("role") == "kernel"]
    if wanted:
        cands = [c for c in cands if c["cell_id"] == wanted]
        if not cands:
            raise SystemExit(f"--identity-cell {wanted}: not an admissible kernel recording of cells.csv")
    chosen = None
    for c in cands:
        sc = enc_out / "extract" / c["cell_id"] / "sidecar.json"
        if sc.is_file() and read_json(sc).get("status") == "ok":
            chosen = c
            break
    if chosen is None:
        return {"cell_id": None, "enc_out": str(enc_out), "work": str(work), "needed": False, "record": str(rec_path),
                "result": {"passed": False, "status": "failed", "reason": "no admissible kernel recording with an ok sidecar in the encoding run"}}
    ref = enc_out / "extract" / chosen["cell_id"] / "extract.csv"
    ref_sha = sha256_file(ref) if ref.is_file() else "absent"
    needed = True
    old = read_json(rec_path) if rec_path.is_file() else None
    if old and old.get("passed") and old.get("cell_id") == chosen["cell_id"] and (old.get("reference") or {}).get("sha256") == ref_sha and not force:
        needed = False
    return {"cell_id": chosen["cell_id"], "enc_out": str(enc_out), "work": str(work), "needed": needed, "record": str(rec_path),
            "result": (old if not needed else None), "reference_sha256": ref_sha}


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
    enc_out = Path(os.path.expanduser(o.encoding_out)) if o.encoding_out else (
        Path(params["D2_baseline"]["encoding_out"]) if (params.get("D2_baseline") or {}).get("encoding_out") else None)
    ident = identity_plan(out, todo, enc_out, o.identity_cell, bool(o.force))
    mdir = out / "moves" / "01_extract"
    jpath = mdir / "extract.json"
    print(f"[extract] {len(todo)} admissible recording(s); store {store}; source {describe_source(src)}" + ("; --force: every series rebuilt" if o.force else ""), flush=True)
    if ident:
        print(f"[extract] E0 identity check: recording {ident['cell_id']}, " + ("to run with this move's fetch" if ident["needed"] else "record passed and current, not repeated"), flush=True)
    elif enc_out is None:
        print("[extract] E0 identity check: not run (no encoding run named, decision D2)", flush=True)
    if o.dry_run:
        for i, c in enumerate(todo, 1):
            c = dict(c); c["cut_declared"], c["cut_measured"] = cuts.get("declared_pairs"), cuts.get("measured_pairs")
            process(c, src, store, out, True, bool(o.force), ident)
        return 0
    mdir.mkdir(parents=True, exist_ok=True)
    doc = read_json(jpath) if jpath.is_file() else {}
    recs: dict = dict(doc.get("recordings", {})) if isinstance(doc.get("recordings"), dict) else {}

    def write_identity():
        if enc_out is None:
            write_json(mdir / IDENTITY_RECORD, {"schema": "plan12.e0_identity.v1", "citation": CITATION, "cell_id": None, "passed": False,
                                                "status": "not run: no encoding run named (decision D2)", "checked_at": now_iso()})
            return
        res = ident.get("result") or {"cell_id": ident["cell_id"], "passed": False, "status": "failed", "reason": "the recording was not reached in this run"}
        write_json(mdir / IDENTITY_RECORD, {"schema": "plan12.e0_identity.v1", "citation": CITATION, "encoding_out": str(enc_out), **res})

    for i, c in enumerate(todo, 1):
        c = dict(c)
        c["cut_declared"], c["cut_measured"] = cuts.get("declared_pairs"), cuts.get("measured_pairs")
        r = process(c, src, store, out, False, bool(o.force), ident)
        recs[c["cell_id"]] = r
        write_json(jpath, {"schema": "plan12.extract.v1", "citation": CITATION, "package_version": __version__,
                           "params": {"store": str(store), "columns": COLUMNS, "speed": SPEED, "speed_assumed": True,
                                      "n_pages": schema.N_PAGES, "source": describe_source(src), "force": bool(o.force),
                                      "series_format": SERIES_FORMAT, "series_code_sha256": series_code_sha256(),
                                      "pair_convention": PAIR_CONVENTION, "pair_axis": PAIR_AXIS, "cut_convention": CUT_CONVENTION,
                                      "keep_first_rule": KEEP_FIRST_RULE, "toolkit_fingerprint": toolkit_fingerprint()["sha256"],
                                      "e0_identity": IDENTITY_RECORD},
                           "recordings": recs, "updated_at": now_iso()})
        if ident and ident.get("cell_id") == c["cell_id"] and ident.get("result"):
            write_identity()
        extra = f" {r.get('n_pairs')} pairs, {r.get('n_rows')} rows" if r.get("status") in ("done", "reused") else ""
        cl = r.get("cleanup")
        print(f"[extract] [{i}/{len(todo)}] {c['cell_id']}: {r['status']}{extra}" + (f" · cache: {cl.get('reason')}" if cl else "")
              + (f" · E0 identity {r['e0_identity']}" if r.get("e0_identity") else ""), flush=True)
    write_identity()
    if ident and ident.get("result") is not None:
        res = ident["result"]
        print(f"[extract] E0 identity check on {res.get('cell_id')}: {res.get('status')}" + (f" ({res.get('reason')})" if res.get("reason") else ""), flush=True)
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
    if ident and not (ident.get("result") or {}).get("passed"):
        print(f"[extract] stopped: the E0 identity check did not pass ({(ident.get('result') or {}).get('reason') or (ident.get('result') or {}).get('first_difference')})", file=sys.stderr)
        return 1
    missing = [c["cell_id"] for c in todo if c["cell_id"] not in recs]
    if missing:
        print(f"[extract] not processed: {missing}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    install_sigterm()
    ap = argparse.ArgumentParser(prog="plan12_grounding.extract", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("extract", help="move 1: the store and the per-pair series, per admissible recording")
    p.add_argument("--out", required=True)
    add_source_args(p)
    p.add_argument("--store", default=DEFAULT_STORE)
    p.add_argument("--encoding-out", default=None, help="decision D2: the encoding run whose extract the identity check recomputes one recording of (default: params.json)")
    p.add_argument("--identity-cell", default=None, help="the recording of the E0 identity check (default: the first admissible kernel recording with an ok sidecar)")
    p.add_argument("--only", default=None, help="regex on cell_id")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true", help="rebuild every series from its store (the identity check runs again too)")
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
