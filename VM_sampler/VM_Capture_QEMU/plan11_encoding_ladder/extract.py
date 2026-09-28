#!/usr/bin/env python3
"""extract.py -- the streaming per-cell extractor of paper 2's analysis toolkit (builder 1).

Implements K2 move 1 (`council/10_al_kindi_revised.md` section 5 move 1; `P2_STRUCTURE.md`
section 5 move 1) as fixed in `plan11_encoding_ladder/SPEC.md` sections 2.1 to 2.7: one pass over
a cell's substrate trajectory (`*substrate_trajectory.csv[.zst|.gz]`, header
`seq,page_index,<64 metrics>`), one output row per `seq` from `seq_first` to `seq_last`
inclusive, two snapshots of page indices in memory at any moment, and a sidecar with the cell's
identity and the pass's counters. The sorted `page_index` set of K2 move 1 is not kept (SPEC 2.2;
al-Kindi's review section 3 item 3: nothing in this paper reads it).

Subcommands (SPEC 2.5, 7.1):
  extract.py index --root R --out O [--idle-marker M ...] [--role-overrides CSV]
  extract.py cell  --cell-dir D --out O [--persist-side t|t+1] [--failed-count N | --failed-dir D]
                   [--role kernel|idle] [--cell-id ID] [--force]
  extract.py all   --cells-csv CSV --out O [--jobs 1] [--only REGEX] [--failed-counts CSV]
                   [--persist-side t|t+1] [--force]

Exit codes (SPEC 7.1): 0 on success (a written refusal is a success), 2 when an input file is
missing (its path on stderr), 1 on an internal error.

No server path appears in this file. No sandbox workload is named.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse  # noqa: E402
import contextlib  # noqa: E402
import csv  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import weakref  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import numpy as np  # noqa: E402

from plan11_encoding_ladder import __version__, schema  # noqa: E402

CITATION_EXTRACT = (
    "P2 Sec. 5 move 1 (one streaming pass per cell, one row per seq); K2 Sec. 5 move 1; "
    "P2 Sec. IV rungs 0, 0', 1, 2 (K/N, sum hamming, J with the independence null K_t K_{t+1}/N, "
    "the three ratios l0/4096, l1/l0, hamming/l0 on the persistent pages); SPEC 2.1 to 2.4"
)
CITATION_INDEX = "SPEC 2.6 and 2.7 (cell identity from the retention path; rep index from the seed, AA A4)"

# Instrumentation hook for the memory test (SPEC section 2.4 "two snapshots"): if not None it is
# called as EMIT_HOOK(prev, cur) at every emitted row. `_LIVE_SNAPSHOTS` holds a weak reference
# to every Snapshot object that is alive.
EMIT_HOOK = None
_LIVE_SNAPSHOTS: "weakref.WeakSet[Snapshot]" = weakref.WeakSet()

_REQUIRED_COLUMNS = ("seq", "page_index", "hamming", "l0", "l1")


class Refusal(Exception):
    """A written refusal (SPEC 2.1): the reason becomes `status = "refused: <reason>"`."""


# ---------------------------------------------------------------------------
# Reading (copied from plan08_b1/b1_extract_hamming.py:open_text, 2026-09-16; extended per SPEC
# binding rules: zstd binary -> zstandard module -> gzip -> plain)
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def open_text(path: str):
    """Yield a text line stream for a plain / .gz / .zst file.

    copied from plan08_b1/b1_extract_hamming.py:open_text, 2026-09-16, extended: a `.zst` path
    is read through the `zstd` binary (`zstd -dc -q`) first and through
    `zstandard.ZstdDecompressor().stream_reader` when the binary is absent; `.gz` through
    `gzip`; anything else as plain text (SPEC 2.5 and the binding rules). If the caller leaves
    the block by an exception (a refusal mid-file), the decompressor is stopped and the
    caller's exception propagates unchanged; only a normal exit checks the binary's return code.
    """
    path = str(path)
    if path.endswith(".zst"):
        try:
            proc = subprocess.Popen(["zstd", "-dc", "-q", path], stdout=subprocess.PIPE)
        except FileNotFoundError:
            proc = None
        if proc is not None:
            body_ok = False
            try:
                yield io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace")
                body_ok = True
            finally:
                if proc.stdout:
                    proc.stdout.close()
                if not body_ok:
                    proc.kill()
                    proc.wait()
                elif proc.wait() != 0:
                    raise RuntimeError(f"zstd -dc failed on {path} (rc={proc.returncode})")
            return
        try:
            import zstandard  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "cannot read a .zst trajectory: neither the `zstd` binary nor the `zstandard` "
                "module is available (runbook move 0: install zstd or `pip install --user "
                "zstandard`)"
            ) from exc
        with open(path, "rb") as raw:
            dctx = zstandard.ZstdDecompressor()
            with dctx.stream_reader(raw) as reader:
                yield io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    elif path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            yield fh
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            yield fh


def sha256_file(path, chunk: int = 1 << 20) -> str:
    """sha256 of a file's bytes, read in chunks (never whole)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------
class Snapshot:
    """One finished snapshot (SPEC 2.4): `pages` sorted ascending as int64 with the three
    channel arrays permuted into the same order. Every live instance is tracked in
    `_LIVE_SNAPSHOTS` so a test can assert that at most two exist at any moment."""

    __slots__ = ("seq", "pages", "ham", "l0", "l1", "__weakref__")

    def __init__(self, seq: int, pages: np.ndarray, ham: np.ndarray, l0: np.ndarray, l1: np.ndarray):
        self.seq = int(seq)
        self.pages = pages
        self.ham = ham
        self.l0 = l0
        self.l1 = l1
        _LIVE_SNAPSHOTS.add(self)

    @property
    def K(self) -> int:
        return int(self.pages.shape[0])


def empty_snapshot(seq: int) -> Snapshot:
    """A K = 0 snapshot for a missing `seq` (SPEC 2.1: a gap is a K = 0 snapshot, not a failed job)."""
    e = np.empty(0, dtype=np.int64)
    return Snapshot(seq, e, e, e, e)


def finalize_buffer(seq: int, pages: list, ham: list, l0: list, l1: list) -> tuple[Snapshot, int]:
    """Turn the row buffer of one `seq` into a Snapshot: sort by `page_index`, drop duplicate
    pages keeping the first row, and return the count of dropped rows (SPEC 2.1, 2.4)."""
    if not pages:
        return empty_snapshot(seq), 0
    p = np.asarray(pages, dtype=np.int64)
    uniq, first_idx = np.unique(p, return_index=True)
    n_dup = int(p.shape[0] - uniq.shape[0])
    h = np.asarray(ham, dtype=np.int64)[first_idx]
    a = np.asarray(l0, dtype=np.int64)[first_idx]
    b = np.asarray(l1, dtype=np.int64)[first_idx]
    return Snapshot(seq, uniq, h, a, b), n_dup


def _quantiles(arr: np.ndarray, qs) -> list[float]:
    """numpy.quantile with its default (linear) method (SPEC 2.2)."""
    return [float(v) for v in np.quantile(arr, qs)]


def row_values(prev: Snapshot, cur, *, n_pages: int, quantiles, persist_side: str, page_size: int) -> list:
    """The 58 extract values for `prev.seq` in `schema.EXTRACT_COLUMNS` order (SPEC 2.2, 2.4).

    Definitions (SPEC 2.2): S_t the page set at seq t, K_t = |S_t|, P_t = S_t & S_{t+1}.
    n_persist = |P_t|; n_union = K_t + K_{t+1} - n_persist; J = n_persist / n_union, 0 when
    exactly one set is empty, blank when both are; J_null_inter = K_t K_{t+1} / N (P2 Sec. IV
    rung 1; K2 Sec. 2 rung 1); J_null = J_null_inter / (K_t + K_{t+1} - J_null_inter), 0 when
    the denominator is 0 (SPEC section 8 item 2). `_all` sums and quantiles over every row of
    seq t (quantiles blank at K = 0); `_per` sums and quantiles over the persistent pages with
    the channel values taken at t (persist_side = "t", SPEC section 8 item 1) or at t+1
    ("t+1"); the three ratio quantiles l0/4096, l1/l0, hamming/l0 on the persistent pages
    (P2 Sec. IV rung 2; K2 Sec. 2 rung 2 (b), (c)), the two divisions restricted to rows with
    l0 >= 1. When `cur` is None (the last seq) every pair column is blank."""
    K = prev.K
    out: list = [prev.seq, K]
    n_persist = 0
    ia = ib = None
    if cur is None:
        out += [None, None, None, None, None]
    else:
        Kc = cur.K
        if K and Kc:
            _, ia, ib = np.intersect1d(prev.pages, cur.pages, assume_unique=True, return_indices=True)
            n_persist = int(ia.shape[0])
        n_union = K + Kc - n_persist
        if K == 0 and Kc == 0:
            J = None
        elif K == 0 or Kc == 0:
            J = 0.0
        else:
            J = n_persist / n_union
        j_null_inter = (K * Kc) / float(n_pages)
        denom = K + Kc - j_null_inter
        j_null = (j_null_inter / denom) if denom != 0 else 0.0
        out += [n_persist, n_union, J, j_null_inter, j_null]
    for arr in (prev.ham, prev.l0, prev.l1):
        out.append(int(arr.sum()))
        out += _quantiles(arr, quantiles) if K > 0 else [None] * len(quantiles)
    if cur is None or n_persist == 0:
        out += [None] * (3 * (1 + len(quantiles)) + 3 * len(quantiles))
        return out
    if persist_side == "t":
        src, idx = prev, ia
    elif persist_side == "t+1":
        src, idx = cur, ib
    else:
        raise ValueError(f"persist_side must be 't' or 't+1', got {persist_side!r}")
    ham = src.ham[idx]
    l0 = src.l0[idx]
    l1 = src.l1[idx]
    for arr in (ham, l0, l1):
        out.append(int(arr.sum()))
        out += _quantiles(arr, quantiles)
    out += _quantiles(l0 / float(page_size), quantiles)
    mask = l0 >= 1
    if mask.any():
        l0m = l0[mask].astype(np.float64)
        out += _quantiles(l1[mask] / l0m, quantiles)
        out += _quantiles(ham[mask] / l0m, quantiles)
    else:
        out += [None] * (2 * len(quantiles))
    return out


# ---------------------------------------------------------------------------
# The streaming pass (SPEC 2.4)
# ---------------------------------------------------------------------------
def _stream(traj_path: Path, writer, *, n_pages: int, quantiles, persist_side: str, page_size: int) -> dict:
    """One pass over the trajectory. Writes one row per seq through `writer`; returns the
    counters of the sidecar (SPEC 2.3). Raises `Refusal` on a non-monotone seq, a missing
    column, or a file without data rows. Memory: two Snapshot objects plus the row buffer of
    the snapshot being read, never more (SPEC 2.4)."""
    stats = {
        "n_rows_in": 0, "n_rows_skipped": 0, "n_rows_dup_page": 0,
        "n_rows_zero_hamming": 0, "n_rows_zero_l0": 0,
        "seq_first": None, "seq_last": None, "n_seq_present": 0, "gap_seqs": [], "n_seq_gaps": 0,
        "K_list": [],
    }
    with open_text(str(traj_path)) as fin:
        reader = csv.reader(fin)
        try:
            header = next(reader)
        except StopIteration:
            raise Refusal("empty trajectory file")
        header = [h.strip() for h in header]
        stats["header_line"] = ",".join(header)
        stats["header_sha256"] = hashlib.sha256(",".join(header).encode("utf-8")).hexdigest()
        stats["header_ncols"] = len(header)
        col = {name: i for i, name in enumerate(header)}
        for req in _REQUIRED_COLUMNS:
            if req not in col:
                raise Refusal(f"header lacks column {req}")
        i_seq, i_pg, i_ham, i_l0, i_l1 = (col[c] for c in _REQUIRED_COLUMNS)
        stats["columns_used"] = {c: col[c] for c in _REQUIRED_COLUMNS}

        buf_pages: list[int] = []
        buf_ham: list[int] = []
        buf_l0: list[int] = []
        buf_l1: list[int] = []
        cur_seq = None
        prev = None
        n_present = 0
        gaps: list[int] = stats["gap_seqs"]
        K_list: list[int] = stats["K_list"]

        def emit(p: Snapshot, c) -> None:
            vals = row_values(p, c, n_pages=n_pages, quantiles=quantiles,
                              persist_side=persist_side, page_size=page_size)
            writer.writerow([schema.format_value(n, v) for n, v in zip(schema.EXTRACT_COLUMNS, vals)])
            K_list.append(p.K)
            if EMIT_HOOK is not None:
                EMIT_HOOK(p, c)

        def advance(p, snap: Snapshot) -> Snapshot:
            # Emits the row of `p` against `snap`, then hands `snap` back as the new `prev`;
            # the caller's reassignment releases the old `prev`, so two snapshots are live
            # during the emit and one afterwards.
            if p is not None:
                emit(p, snap)
            return snap

        for row in reader:
            stats["n_rows_in"] += 1
            try:
                s = int(row[i_seq]); pg = int(row[i_pg]); h = int(row[i_ham])
                a = int(row[i_l0]); b = int(row[i_l1])
            except (ValueError, IndexError):
                stats["n_rows_skipped"] += 1
                continue
            if h == 0:
                stats["n_rows_zero_hamming"] += 1
            if a == 0:
                stats["n_rows_zero_l0"] += 1
            if cur_seq is None:
                cur_seq = s
                stats["seq_first"] = s
                n_present = 1
            elif s != cur_seq:
                if s < cur_seq:
                    raise Refusal(f"seq not monotone at row {stats['n_rows_in']}")
                snap, n_dup = finalize_buffer(cur_seq, buf_pages, buf_ham, buf_l0, buf_l1)
                stats["n_rows_dup_page"] += n_dup
                buf_pages.clear(); buf_ham.clear(); buf_l0.clear(); buf_l1.clear()
                prev = advance(prev, snap)
                del snap
                for g in range(cur_seq + 1, s):
                    gaps.append(g)
                    prev = advance(prev, empty_snapshot(g))
                cur_seq = s
                n_present += 1
            buf_pages.append(pg); buf_ham.append(h); buf_l0.append(a); buf_l1.append(b)

        if cur_seq is None:
            raise Refusal("no data rows")
        snap, n_dup = finalize_buffer(cur_seq, buf_pages, buf_ham, buf_l0, buf_l1)
        stats["n_rows_dup_page"] += n_dup
        buf_pages.clear(); buf_ham.clear(); buf_l0.clear(); buf_l1.clear()
        prev = advance(prev, snap)
        del snap
        emit(prev, None)
        stats["seq_last"] = cur_seq
        stats["n_seq_present"] = n_present
    stats["n_seq_gaps"] = len(gaps)
    return stats


# ---------------------------------------------------------------------------
# Per-cell extraction (SPEC 2.5)
# ---------------------------------------------------------------------------
def find_trajectory(cell_dir: Path) -> tuple[list[Path], Path]:
    """The trajectory files of a cell directory by the glob of SPEC 2.1; a file path is
    accepted as the trajectory itself (its parent is the cell directory)."""
    cell_dir = Path(cell_dir)
    if cell_dir.is_file():
        return [cell_dir], cell_dir.parent
    if not cell_dir.is_dir():
        raise FileNotFoundError(str(cell_dir))
    files = sorted(p for p in cell_dir.glob(schema.TRAJ_GLOB) if p.is_file())
    return files, cell_dir


def derive_cell_id(cell_dir, *, role: str | None = None, rep: int | None = None,
                   idle_markers: tuple[str, ...] = schema.IDLE_MARKERS_DEFAULT) -> str:
    """The cell_id `extract_cell` would assign to `cell_dir` (SPEC 2.6) without reading the
    trajectory: from the path, with `role` and `rep` overriding the parsed role and `rep_dir - 1`."""
    _, cdir = find_trajectory(Path(cell_dir))
    ident = schema.parse_cell_path(cdir, idle_markers=idle_markers)
    r = ident["role"] if role is None else role
    rep_val = int(rep) if rep is not None else ((ident["rep_dir"] - 1) if ident["rep_dir"] is not None else 0)
    return schema.cell_id_of(ident["kernel"], r, rep_val, ident["campaign"])


def _count_files(d: Path) -> int:
    d = Path(d)
    if not d.is_dir():
        raise FileNotFoundError(str(d))
    return sum(1 for p in d.iterdir() if p.is_file())


def extract_cell(cell_dir, out_dir, *,
                 n_pages: int = schema.N_PAGES, page_size: int = schema.PAGE_SIZE,
                 quantiles: tuple[float, ...] = schema.QUANTILES,
                 persist_side: str = "t",
                 duration_s: float = schema.DURATION_S,
                 failed_count: int | None = None, failed_dir=None,
                 failed_count_source: str | None = None,
                 role: str | None = None,
                 cell_id: str | None = None,
                 rep: int | None = None,
                 archetype_predicted: str | None = None,
                 idle_markers: tuple[str, ...] = schema.IDLE_MARKERS_DEFAULT,
                 extra_inputs_sha256: dict | None = None) -> dict:
    """Extract one cell (SPEC 2.2 to 2.5; K2 move 1). `cell_dir` is the cell directory (or the
    trajectory file itself); `out_dir` is the `--out` root: the outputs are
    `<out_dir>/extract/<cell_id>/extract.csv` and `sidecar.json`. Returns the sidecar dict.

    Identity comes from `schema.parse_cell_path` unless `role` / `cell_id` / `rep` are given;
    `rep` is the paper's rep index assigned by `build_index` (SPEC 2.6); without it the sidecar
    uses `rep_dir - 1` and says so in `rep_source`. `failed_count` is a recorded input, never
    computed (SPEC 2.3; CR 2.1 item 2; AA A5): from `failed_count`, else the count of files in
    `failed_dir`, else null with `failed_count_source = "not recorded"`. `persist_side` is
    SPEC section 8 item 1 (default "t"). A refusal is written into the sidecar's `status` and
    no extract.csv is left behind (al-Farabi's condition 3: nothing silently absorbed)."""
    t0 = time.monotonic()
    started = _utc_now()
    cell_id_given = cell_id is not None
    if persist_side not in ("t", "t+1"):
        raise ValueError("persist_side must be 't' or 't+1'")
    files, cdir = find_trajectory(Path(cell_dir))
    ident = schema.parse_cell_path(cdir, idle_markers=idle_markers)
    if role is not None:
        ident["role"] = role
        if role == "idle":
            ident["archetype_predicted"] = "control"
        elif role == "kernel":
            ident["archetype_predicted"] = schema.ARCHETYPE_OF.get(ident["kernel"], "unknown")
    if archetype_predicted:
        ident["archetype_predicted"] = str(archetype_predicted)
    if rep is None:
        rep_val = (ident["rep_dir"] - 1) if ident["rep_dir"] is not None else 0
        rep_source = "rep_dir - 1 (no cells.csv rep given)"
    else:
        rep_val = int(rep)
        rep_source = "cells.csv"
    if cell_id is None:
        cell_id = schema.cell_id_of(ident["kernel"], ident["role"], rep_val, ident["campaign"])

    if failed_count is not None:
        fc = int(failed_count)
        fc_src = failed_count_source or "--failed-count"
    elif failed_dir is not None:
        fc = _count_files(Path(failed_dir))
        fc_src = f"--failed-dir {failed_dir}"
    else:
        fc = None
        fc_src = "not recorded"

    out_cell = Path(out_dir) / "extract" / cell_id
    out_cell.mkdir(parents=True, exist_ok=True)
    extract_path = out_cell / "extract.csv"
    tmp_path = out_cell / "extract.csv.tmp"
    sidecar_path = out_cell / "sidecar.json"

    params = {
        "n_pages": int(n_pages), "page_size": int(page_size), "quantiles": list(quantiles),
        "persist_side": persist_side, "duration_s": float(duration_s),
        "failed_count": fc, "failed_count_source": fc_src,
        "idle_markers": list(idle_markers), "role_override": role, "archetype_override": archetype_predicted,
        "cell_id_override": cell_id_given,
        "inputs_sha256": dict(extra_inputs_sha256 or {}),
    }
    sidecar: dict = {
        "schema": schema.SIDECAR_SCHEMA,
        "extractor_version": __version__,
        "cell_id": cell_id, "kernel": ident["kernel"], "role": ident["role"],
        "archetype_predicted": ident["archetype_predicted"], "seed": ident["seed"],
        "rep": rep_val, "rep_dir": ident["rep_dir"], "label": ident["label"],
        "campaign": ident["campaign"], "rep_source": rep_source,
        "path": str(cell_dir), "traj_file": None, "source_bytes": None, "source_sha256": None,
        "N": int(n_pages), "page_size": int(page_size), "bits_per_page": int(page_size) * 8,
        "duration_s_declared": float(duration_s),   # SPEC_epoch2 B12 (al-Farabi review 5.3): a float, so 77.28 is not recorded as 77
        "quantiles": list(quantiles),
        "persist_side": persist_side,
        "header_sha256": None, "header_ncols": None, "columns_used": None,
        "n_rows_in": 0, "n_rows_skipped": 0, "n_rows_dup_page": 0,
        "n_rows_zero_hamming": 0, "n_rows_zero_l0": 0,
        "seq_first": None, "seq_last": None, "n_seq_present": 0,
        "n_pairs": None, "n_seq_gaps": 0, "gap_seqs": [],
        "dt_est_s": None, "dt_bracket_s": list(schema.DT_BRACKET_S),
        "K_median": None, "K_max": None, "apf_max": None,
        "failed_count": fc, "failed_count_source": fc_src,
        "status": "ok",
        "started_at": started, "finished_at": None, "elapsed_s": None,
        "params": params,
        "citation": CITATION_EXTRACT,
    }

    def _finish(status: str) -> dict:
        sidecar["status"] = status
        sidecar["finished_at"] = _utc_now()
        sidecar["elapsed_s"] = round(time.monotonic() - t0, 3)
        with open(sidecar_path, "w") as fh:
            json.dump(sidecar, fh, indent=1)
        return sidecar

    if len(files) != 1:
        if tmp_path.exists():
            tmp_path.unlink()
        return _finish("refused: trajectory file count != 1")
    traj = files[0]
    sidecar["traj_file"] = traj.name
    sidecar["source_bytes"] = int(traj.stat().st_size)
    sidecar["source_sha256"] = sha256_file(traj)
    params["inputs_sha256"][traj.name] = sidecar["source_sha256"]

    try:
        with open(tmp_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(schema.EXTRACT_COLUMNS)
            stats = _stream(traj, writer, n_pages=n_pages, quantiles=quantiles,
                            persist_side=persist_side, page_size=page_size)
    except Refusal as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        if extract_path.exists():
            extract_path.unlink()
        return _finish(f"refused: {exc}")
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    tmp_path.replace(extract_path)

    K_arr = np.asarray(stats.pop("K_list"), dtype=np.int64)
    n_pairs = int(stats["seq_last"] - stats["seq_first"] + 1)
    sidecar.update({
        "header_sha256": stats["header_sha256"], "header_ncols": stats["header_ncols"],
        "columns_used": stats["columns_used"],
        "n_rows_in": stats["n_rows_in"], "n_rows_skipped": stats["n_rows_skipped"],
        "n_rows_dup_page": stats["n_rows_dup_page"],
        "n_rows_zero_hamming": stats["n_rows_zero_hamming"], "n_rows_zero_l0": stats["n_rows_zero_l0"],
        "seq_first": int(stats["seq_first"]), "seq_last": int(stats["seq_last"]),
        "n_seq_present": int(stats["n_seq_present"]),
        "n_pairs": n_pairs, "n_seq_gaps": int(stats["n_seq_gaps"]),
        "gap_seqs": [int(g) for g in stats["gap_seqs"][:100]],
        "dt_est_s": float(duration_s) / n_pairs,
        "K_median": float(np.median(K_arr)), "K_max": int(K_arr.max()),
        "apf_max": float(K_arr.max()) / float(n_pages),
    })
    return _finish("ok")


# ---------------------------------------------------------------------------
# The cell index (SPEC 2.6, 2.7)
# ---------------------------------------------------------------------------
def _load_role_overrides(path) -> dict[str, dict]:
    """`--role-overrides CSV` with columns `path, role[, archetype_predicted]`; `path` is the
    cell directory as listed by the index or its last four components."""
    out: dict[str, dict] = {}
    if path is None:
        return out
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(str(p))
    with open(p, newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row.get("path") or "").strip()
            if not key:
                continue
            out[key] = {k: (v or "").strip() for k, v in row.items() if k != "path"}
    return out


def _override_for(cell_dir: Path, overrides: dict[str, dict]) -> dict | None:
    if not overrides:
        return None
    parts = cell_dir.parts
    tail = "/".join(parts[-4:]) if len(parts) >= 4 else str(cell_dir)
    for key in (str(cell_dir), str(cell_dir.resolve()), tail):
        if key in overrides:
            return overrides[key]
    return None


def build_index(root, out_csv, *, idle_markers: tuple[str, ...] = schema.IDLE_MARKERS_DEFAULT,
                role_overrides=None) -> list[dict]:
    """The cell index `cells.csv` (SPEC 2.7): one row per directory `rep*__*` under `root`
    (found by `root.rglob("rep*__*")`, directories kept), with the identity of SPEC 2.6 and a
    status of `ok`, `refused: trajectory file count != 1`, `refused: duplicate seed`, or
    `refused: unknown kernel`. A directory with no trajectory file (or more than one) is listed
    with its refusal rather than dropped, so the author sees it.

    The paper's rep index (SPEC 2.6; P2 Sec. VI; AA A4): within one (role, kernel) group the
    cell with seed 42 is rep 0 and the remaining cells are numbered 1 upward by ascending seed
    (ties by path, both kept and flagged `refused: duplicate seed`); a cell without a parsed
    seed takes `rep = rep_dir - 1`. Roles come from the idle markers, then `role_overrides`
    (a CSV `path, role[, archetype_predicted]`); the author may also edit `cells.csv` by hand.
    Writes `cells.csv` and `cells.index.json` (the params block) next to it. Returns the rows."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(str(root))
    overrides = _load_role_overrides(role_overrides)
    rows: list[dict] = []
    for d in sorted(p for p in root.rglob("rep*__*") if p.is_dir()):
        try:
            ident = schema.parse_cell_path(d, idle_markers=idle_markers)
        except ValueError:
            continue
        ov = _override_for(d, overrides)
        if ov:
            if ov.get("role"):
                ident["role"] = ov["role"]
                if ov["role"] == "idle":
                    ident["archetype_predicted"] = "control"
                elif ov["role"] == "kernel":
                    ident["archetype_predicted"] = schema.ARCHETYPE_OF.get(ident["kernel"], "unknown")
            if ov.get("archetype_predicted"):
                ident["archetype_predicted"] = ov["archetype_predicted"]
        files = sorted(p for p in d.glob(schema.TRAJ_GLOB) if p.is_file())
        status = schema.STATUS_OK
        if len(files) != 1:
            status = schema.STATUS_TRAJ_COUNT
        elif ident["role"] == "unknown":
            status = schema.STATUS_UNKNOWN_KERNEL
        rows.append({
            **ident,
            "path": str(d),
            "traj_file": files[0].name if len(files) == 1 else "",
            "status": status,
        })

    # rep index per (role, kernel) group
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["role"], r["kernel"]), []).append(r)
    for _, members in groups.items():
        seeded = [m for m in members if m["seed"] is not None]
        unseeded = [m for m in members if m["seed"] is None]
        for m in unseeded:
            m["rep"] = (m["rep_dir"] - 1) if m["rep_dir"] is not None else 0
        rep0 = sorted((m for m in seeded if m["seed"] == schema.REP0_SEED), key=lambda m: m["path"])
        rest = sorted((m for m in seeded if m["seed"] != schema.REP0_SEED), key=lambda m: (m["seed"], m["path"]))
        ordered = rep0 + rest
        for i, m in enumerate(ordered):
            m["rep"] = i
        seen: dict[int, int] = {}
        for m in seeded:
            seen[m["seed"]] = seen.get(m["seed"], 0) + 1
        for m in seeded:
            if seen[m["seed"]] > 1 and m["status"] == schema.STATUS_OK:
                m["status"] = schema.STATUS_DUP_SEED
    for r in rows:
        r["cell_id"] = schema.cell_id_of(r["kernel"], r["role"], r["rep"], r["campaign"])

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_csv.with_suffix(out_csv.suffix + ".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(schema.CELLS_COLUMNS), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in schema.CELLS_COLUMNS})
    tmp.replace(out_csv)
    meta = {
        "schema": schema.CELLS_SCHEMA,
        "params": {"root": str(root), "idle_markers": list(idle_markers),
                   "role_overrides": str(role_overrides) if role_overrides else None,
                   "rep0_seed": schema.REP0_SEED, "traj_glob": schema.TRAJ_GLOB,
                   "inputs_sha256": ({str(role_overrides): sha256_file(role_overrides)} if role_overrides else {})},
        "citation": CITATION_INDEX,
        "n_rows": len(rows),
        "n_ok": sum(1 for r in rows if r["status"] == schema.STATUS_OK),
        "status_counts": _count_by(rows, "status"),
        "written_at": _utc_now(),
        "package_version": __version__,
    }
    with open(out_csv.parent / "cells.index.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    return rows


def _count_by(rows: list[dict], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[str(r.get(key))] = out.get(str(r.get(key)), 0) + 1
    return out


def read_cells_csv(path) -> list[dict]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(str(p))
    with open(p, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["rep"] = int(r["rep"]) if r.get("rep") not in (None, "") else None
        r["seed"] = int(r["seed"]) if r.get("seed") not in (None, "") else None
        r["rep_dir"] = int(r["rep_dir"]) if r.get("rep_dir") not in (None, "") else None
    return rows


# ---------------------------------------------------------------------------
# Batch mode (SPEC 2.5 `all`)
# ---------------------------------------------------------------------------
def _sidecar_ok(out_root: Path, cell_id: str) -> bool:
    p = out_root / "extract" / cell_id / "sidecar.json"
    if not p.is_file():
        return False
    try:
        with open(p) as fh:
            return json.load(fh).get("status") == "ok"
    except (OSError, ValueError):
        return False


def _worker(job: dict) -> dict:
    """One cell for the batch mode (a module-level function so multiprocessing can pickle it)."""
    try:
        sc = extract_cell(**job)
        return {"cell_id": sc["cell_id"], "status": sc["status"], "n_pairs": sc.get("n_pairs"),
                "n_seq_gaps": sc.get("n_seq_gaps"), "elapsed_s": sc.get("elapsed_s")}
    except Exception as exc:  # an internal error is reported per cell, not swallowed
        return {"cell_id": job.get("cell_id"), "status": f"error: {type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()}


def _load_failed_counts(path) -> dict[str, tuple[int, str]]:
    """`--failed-counts CSV` with columns `cell_id, failed_count, source` (SPEC 3.3.2 format)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(str(p))
    out: dict[str, tuple[int, str]] = {}
    with open(p, newline="") as fh:
        for row in csv.DictReader(fh):
            cid = (row.get("cell_id") or "").strip()
            if not cid:
                continue
            fc = row.get("failed_count", "")
            if fc in (None, ""):
                continue
            out[cid] = (int(fc), (row.get("source") or f"--failed-counts {p.name}").strip())
    return out


def extract_all(cells_csv, out_dir, *, jobs: int = 1, only: str | None = None,
                failed_counts=None, persist_side: str = "t", force: bool = False,
                duration_s: float = schema.DURATION_S) -> dict:
    """Run `extract_cell` for every `status == ok` row of `cells.csv` (SPEC 2.5), skipping a
    cell whose sidecar exists with `status == "ok"` unless `force`. `jobs > 1` uses
    multiprocessing (one process per cell at a time). Writes `<out>/extract/extract_all.json`
    (params, per-cell statuses) and returns it. `duration_s` (SPEC_epoch2 B12; CHECK_3 M12) is
    every cell's declared guest-running seconds, recorded in the sidecar as `duration_s_declared`
    (600, the real corpus; a synthetic corpus passes `n_pairs x 0.644`) and read by G2 and G-P."""
    cells_csv = Path(cells_csv)
    out_root = Path(out_dir)
    rows = read_cells_csv(cells_csv)
    fcs = _load_failed_counts(failed_counts) if failed_counts else {}
    rx = re.compile(only) if only else None
    inputs_sha = {cells_csv.name: sha256_file(cells_csv)}
    if failed_counts:
        inputs_sha[Path(failed_counts).name] = sha256_file(failed_counts)
    jobs_list: list[dict] = []
    skipped: list[dict] = []
    for r in rows:
        cid = r["cell_id"]
        if rx is not None and not rx.search(cid):
            continue
        if r["status"] != schema.STATUS_OK:
            skipped.append({"cell_id": cid, "reason": f"cells.csv status: {r['status']}"})
            continue
        if not force and _sidecar_ok(out_root, cid):
            skipped.append({"cell_id": cid, "reason": "sidecar ok (use --force to redo)"})
            continue
        fc, fsrc = fcs.get(cid, (None, None))
        jobs_list.append({
            "cell_dir": r["path"], "out_dir": str(out_root), "persist_side": persist_side,
            "failed_count": fc, "failed_count_source": fsrc,
            "role": r["role"] or None, "cell_id": cid, "rep": r["rep"],
            "archetype_predicted": r.get("archetype_predicted") or None,
            "extra_inputs_sha256": inputs_sha, "duration_s": float(duration_s),
        })
    results: list[dict] = []
    if jobs_list:
        if int(jobs) > 1:
            import multiprocessing as mp
            with mp.Pool(processes=int(jobs)) as pool:
                for res in pool.imap_unordered(_worker, jobs_list):
                    results.append(res)
                    print(f"[extract] {res['cell_id']}: {res['status']}", file=sys.stderr)
        else:
            for job in jobs_list:
                res = _worker(job)
                results.append(res)
                print(f"[extract] {res['cell_id']}: {res['status']}", file=sys.stderr)
    results.sort(key=lambda r: str(r.get("cell_id")))
    summary = {
        "schema": "plan11.extract_all.v1",
        "params": {"cells_csv": str(cells_csv), "out": str(out_root), "jobs": int(jobs), "only": only,
                   "failed_counts": str(failed_counts) if failed_counts else None,
                   "persist_side": persist_side, "force": bool(force), "duration_s": float(duration_s), "inputs_sha256": inputs_sha},
        "citation": CITATION_EXTRACT,
        "n_cells_in_csv": len(rows), "n_run": len(results), "n_skipped": len(skipped),
        "status_counts": _count_by(results, "status"),
        "results": results, "skipped": skipped,
        "written_at": _utc_now(), "package_version": __version__,
    }
    (out_root / "extract").mkdir(parents=True, exist_ok=True)
    with open(out_root / "extract" / "extract_all.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    return summary


# ---------------------------------------------------------------------------
# CLI (SPEC 7.1)
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="extract.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_idx = sub.add_parser("index", help="write <out>/cells.csv from a retention root (SPEC 2.7)")
    p_idx.add_argument("--root", required=True, help="retention root or any directory holding kernel/...")
    p_idx.add_argument("--out", required=True)
    p_idx.add_argument("--idle-marker", action="append", default=None,
                       help="substring of the test label that marks an idle cell (default: sleep, idle)")
    p_idx.add_argument("--role-overrides", default=None, help="CSV path, role[, archetype_predicted]")

    p_cell = sub.add_parser("cell", help="extract one cell directory or trajectory file (SPEC 2.5)")
    p_cell.add_argument("--cell-dir", required=True, help="cell directory (or the trajectory file)")
    p_cell.add_argument("--out", required=True)
    p_cell.add_argument("--persist-side", choices=("t", "t+1"), default="t")
    g = p_cell.add_mutually_exclusive_group()
    g.add_argument("--failed-count", type=int, default=None)
    g.add_argument("--failed-dir", default=None)
    p_cell.add_argument("--role", choices=("kernel", "idle"), default=None)
    p_cell.add_argument("--cell-id", default=None)
    p_cell.add_argument("--rep", type=int, default=None, help="the paper's rep index (from cells.csv)")
    p_cell.add_argument("--force", action="store_true")
    p_cell.add_argument("--duration-s", type=float, default=schema.DURATION_S,
                        help="the cell's declared guest-running seconds (sidecar duration_s_declared; SPEC_epoch2 B12)")

    p_all = sub.add_parser("all", help="extract every ok cell of cells.csv (SPEC 2.5)")
    p_all.add_argument("--cells-csv", required=True)
    p_all.add_argument("--out", required=True)
    p_all.add_argument("--jobs", type=int, default=1)
    p_all.add_argument("--only", default=None, help="regex on cell_id")
    p_all.add_argument("--failed-counts", default=None, help="CSV cell_id, failed_count, source")
    p_all.add_argument("--persist-side", choices=("t", "t+1"), default="t")
    p_all.add_argument("--force", action="store_true")
    p_all.add_argument("--duration-s", type=float, default=schema.DURATION_S,
                       help="every cell's declared guest-running seconds (600 on the real corpus; a synthetic corpus passes n_pairs x 0.644; SPEC_epoch2 B12)")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = _build_parser()
    args = ap.parse_args(argv)
    try:
        if args.cmd == "index":
            markers = tuple(args.idle_marker) if args.idle_marker else schema.IDLE_MARKERS_DEFAULT
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            rows = build_index(args.root, out / "cells.csv", idle_markers=markers,
                               role_overrides=args.role_overrides)
            n_ok = sum(1 for r in rows if r["status"] == schema.STATUS_OK)
            print(f"[index] {len(rows)} cells listed, {n_ok} ok -> {out / 'cells.csv'}")
            return 0
        if args.cmd == "cell":
            out = Path(args.out)
            if not args.force:
                cid = args.cell_id or derive_cell_id(args.cell_dir, role=args.role, rep=args.rep)
                if _sidecar_ok(out, cid):
                    print(f"[cell] {cid}: sidecar ok, skipped (use --force)")
                    return 0
            sc = extract_cell(args.cell_dir, out, persist_side=args.persist_side,
                              failed_count=args.failed_count, failed_dir=args.failed_dir,
                              role=args.role, cell_id=args.cell_id, rep=args.rep, duration_s=args.duration_s)
            print(f"[cell] {sc['cell_id']}: {sc['status']} (n_pairs={sc.get('n_pairs')}, "
                  f"gaps={sc.get('n_seq_gaps')}, {sc.get('elapsed_s')} s)")
            return 0
        if args.cmd == "all":
            summary = extract_all(args.cells_csv, args.out, jobs=args.jobs, only=args.only,
                                  failed_counts=args.failed_counts, persist_side=args.persist_side,
                                  force=args.force, duration_s=args.duration_s)
            print(f"[all] ran {summary['n_run']} cells, skipped {summary['n_skipped']}: "
                  f"{summary['status_counts']}")
            return 0
        ap.error(f"unknown command {args.cmd}")
        return 1
    except FileNotFoundError as exc:
        print(f"missing input: {exc}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
