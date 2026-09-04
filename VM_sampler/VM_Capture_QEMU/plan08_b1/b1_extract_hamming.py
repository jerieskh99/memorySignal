#!/usr/bin/env python3
"""b1_extract_hamming.py -- lean the fat substrate CSV to the hamming+cosine field.

Experiment B1 (the encoding floor). See docs/EXPERIMENT_B1_ENCODING_FLOOR.md.

The Aug campaign captured CAPTURE_METRIC=substrate, so each cell has a
`*.substrate_trajectory.csv`: one row per CHANGED page per snapshot, columns
`seq,page_index,hamming,cosine,+49`. These are tens of GB
(mem_random_write_pages_v2 = 28.29 GB), so we stream, never load, and keep only
the two channels the whole encoding family is built on -- magnitude (hamming)
and orientation (cosine) -- in ONE pass, so the 28 GB is never re-read.

Two outputs from one pass:
  hc_field.csv[.zst]    seq,page_index,hamming,cosine  -- the sparse page x time
                        field. hamming feeds B1 (APF/wAPF); cosine is stored,
                        UNREAD by B1, for the later complex-encoding paper.
                        zstd-compressed by default (--no-field to skip entirely).
  b1_trajectory.jsonl   per seq: sufficient stats (n_changed, ham_sum). B1 only.

cosine is stored RAW -- it is a DISTANCE (0 = identical). The 2*pi / phase
mapping is the complex-encoding paper's decision, made downstream, not here.

APF and wAPF are NOT stored; they are views on the sufficient stats:
  APF(seq)  = n_changed / n_pages
  wAPF(seq) = ham_sum / (n_pages * BITS_PER_PAGE)
Their per-cell means are printed as a sanity readout only. This derived APF is
identical by construction to plan02_apf_helper._compute_active_page_fraction
(both count pages with any byte changed); the equality is asserted in the test.

Source may be plain, .gz, or .zst -- decompressed transparently on read.

Usage:
  b1_extract_hamming.py <substrate_csv[.zst|.gz]> <out_dir>
      [--n-pages 262144] [--page-size 4096] [--no-field]
      [--cell-id ID] [--workload NAME]
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gzip
import io
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


@contextlib.contextmanager
def open_text(path: str):
    """Yield a text line stream for a plain / .gz / .zst file."""
    if path.endswith(".zst"):
        proc = subprocess.Popen(["zstd", "-dc", "-q", path], stdout=subprocess.PIPE)
        try:
            yield io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace")
        finally:
            if proc.stdout:
                proc.stdout.close()
            if proc.wait() != 0:
                raise RuntimeError(f"zstd -dc failed on {path} (rc={proc.returncode})")
    elif path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            yield fh
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            yield fh


@contextlib.contextmanager
def field_writer(path: Path, compress: bool):
    """Yield a text write handle for the field file (zstd-piped when compress)."""
    if not compress:
        with path.open("w", newline="") as fh:
            yield fh
        return
    proc = subprocess.Popen(["zstd", "-3", "-q", "-f", "-o", str(path)],
                            stdin=subprocess.PIPE)
    w = io.TextIOWrapper(proc.stdin, encoding="utf-8")
    try:
        yield w
    finally:
        w.flush()
        w.close()                       # closes proc.stdin -> zstd finalizes
        if proc.wait() != 0:
            raise RuntimeError(f"zstd compress failed for {path} (rc={proc.returncode})")


@contextlib.contextmanager
def _null_writer():
    yield None


def extract(src: str, out_dir: Path, n_pages: int, page_size: int,
            cell_id: str, workload: str, write_field: bool) -> int:
    bits_per_page = page_size * 8
    out_dir.mkdir(parents=True, exist_ok=True)
    field_path = out_dir / ("hc_field.csv.zst" if write_field else "hc_field.csv")
    traj_path = out_dir / "b1_trajectory.jsonl"
    prov_path = out_dir / "provenance.json"

    stats: dict[int, list[int]] = {}   # seq -> [n_changed, ham_sum]; ~hundreds of seqs
    n_rows = n_skipped = 0
    has_cosine = False

    fw_cm = field_writer(field_path, compress=True) if write_field else _null_writer()
    with open_text(src) as fin, fw_cm as ffield:
        reader = csv.reader(fin)
        try:
            header = next(reader)
        except StopIteration:
            sys.exit(f"empty source: {src}")
        col = {name.strip(): i for i, name in enumerate(header)}
        for req in ("seq", "page_index", "hamming"):
            if req not in col:
                sys.exit(f"source missing required column {req!r}; header={header}")
        i_seq, i_pg, i_ham = col["seq"], col["page_index"], col["hamming"]
        i_cos = col.get("cosine")               # optional; None on a pure-APF source
        has_cosine = i_cos is not None

        if ffield is not None:
            ffield.write("seq,page_index,hamming,cosine\n" if has_cosine
                         else "seq,page_index,hamming\n")
        for row in reader:
            try:
                seq = int(row[i_seq]); pg = int(row[i_pg]); ham = int(row[i_ham])
            except (IndexError, ValueError):
                n_skipped += 1
                continue
            if ham <= 0:                        # differ emits only changed pages; guard
                n_skipped += 1
                continue
            if ffield is not None:
                if has_cosine:
                    cos = row[i_cos].strip() if i_cos < len(row) else ""   # raw token
                    ffield.write(f"{seq},{pg},{ham},{cos}\n")
                else:
                    ffield.write(f"{seq},{pg},{ham}\n")
            acc = stats.get(seq)
            if acc is None:
                stats[seq] = [1, ham]
            else:
                acc[0] += 1; acc[1] += ham
            n_rows += 1

    apf_sum = wapf_sum = 0.0
    with traj_path.open("w") as ftraj:
        for seq in sorted(stats):
            k, ham_sum = stats[seq]
            ftraj.write(json.dumps({
                "seq": seq, "n_pages": n_pages,
                "n_changed": k, "ham_sum": ham_sum,
            }, separators=(",", ":")) + "\n")
            apf_sum += k / n_pages
            wapf_sum += ham_sum / (n_pages * bits_per_page)
        ftraj.write(json.dumps({"final": True, "n_pairs": len(stats)}) + "\n")

    n_seq = len(stats)
    prov = {
        "cell_id": cell_id, "workload": workload, "source": str(src),
        "source_bytes": Path(src).stat().st_size if Path(src).is_file() else None,
        "has_cosine": has_cosine, "field_written": write_field,
        "field_path": str(field_path) if write_field else None,
        "n_pages": n_pages, "page_size": page_size, "bits_per_page": bits_per_page,
        "n_pairs": n_seq, "n_rows": n_rows, "n_skipped": n_skipped,
        "tool": "b1_extract_hamming.py",
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }
    prov_path.write_text(json.dumps(prov, indent=2) + "\n")

    tag = cell_id or src
    if n_seq:
        print(f"[b1_extract] {tag}: {n_seq} snapshots, {n_rows} changed-page rows"
              f"{f', {n_skipped} skipped' if n_skipped else ''}"
              f"{'' if has_cosine else '  (no cosine column in source)'}")
        print(f"[b1_extract]   mean APF  = {apf_sum / n_seq:.6f}   (n_changed / {n_pages})")
        print(f"[b1_extract]   mean wAPF = {wapf_sum / n_seq:.9f}   (ham_sum / (N*{bits_per_page}))")
    else:
        print(f"[b1_extract] {tag}: no changed-page rows ({n_skipped} skipped) -- check the source")
    if write_field:
        print(f"[b1_extract]   -> {field_path}")
    print(f"[b1_extract]   -> {traj_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="substrate_trajectory.csv (plain, .gz, or .zst)")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--n-pages", type=int, default=262144,
                    help="total pages = guest RAM / page size (default 1 GiB / 4 KiB)")
    ap.add_argument("--page-size", type=int, default=4096)
    ap.add_argument("--no-field", action="store_true",
                    help="skip the hc field; write only the sufficient-stats trajectory")
    ap.add_argument("--cell-id", default="")
    ap.add_argument("--workload", default="")
    a = ap.parse_args()
    return extract(a.src, a.out_dir, a.n_pages, a.page_size,
                   a.cell_id, a.workload, write_field=not a.no_field)


if __name__ == "__main__":
    raise SystemExit(main())
