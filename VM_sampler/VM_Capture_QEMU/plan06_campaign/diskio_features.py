#!/usr/bin/env python3
"""Per-cell disk-I/O features from the Plan-06 diskio_trajectory.jsonl.

Each diskio trajectory line is {seq, t_emit_epoch, rd_bytes, wr_bytes} with the
byte counters CUMULATIVE since domain boot (from domblkstat). Per interval:
rate = (counter[i+1]-counter[i]) / (t[i+1]-t[i]); aggregated per cell. Counters
are monotonic; a negative delta (domain restart) clamps to 0.

  wr_rate_mean_mbs / wr_rate_max_mbs / wr_rate_p95_mbs  -- write rate (MB/s)
  rd_rate_mean_mbs                                       -- read rate (MB/s)
  wr_total_mb / rd_total_mb                              -- total written / read (MB)
  rd_frac          -- read share of total I/O = rd_total/(rd_total+wr_total).
                      ~0 for a pure writer (benign mmap, writes only), ~0.25-0.4
                      for an encrypt-in-place threat that reads plaintext + writes
                      ciphertext (ransom_seq/selective). Separates "pure writer"
                      from "encryptor" -- the angle the write-only view misses.

Note on reads: host domblkstat rd_bytes counts only reads that MISS the guest
page cache and hit the virtual disk. Memory-stress workloads keep their working
set in guest RAM, so most cells read ~0; reads carry signal mainly for the
encrypt-in-place threats (seq/selective), which is exactly what rd_frac captures.

Keyed by cell_id, mirroring extra_features.load_extra. Run directly for a
synthetic self-test.
"""
import json
import statistics as st
from pathlib import Path

DISKIO_FEATS = ["wr_rate_mean_mbs", "wr_rate_max_mbs", "wr_rate_p95_mbs",
                "rd_rate_mean_mbs", "wr_total_mb", "rd_total_mb", "rd_frac"]
_MB = 1_000_000.0
_EPS = 1e-9


def _pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return 0.0
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def cell_diskio(traj_path: Path) -> dict:
    rows = []
    with open(traj_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if all(k in o for k in ("seq", "t_emit_epoch", "rd_bytes", "wr_bytes")):
                rows.append(o)
    rows.sort(key=lambda o: o["seq"])
    wr_rates, rd_rates = [], []
    for a, b in zip(rows, rows[1:]):
        dt = float(b["t_emit_epoch"]) - float(a["t_emit_epoch"])
        if dt <= 0:
            continue
        dwr = float(b["wr_bytes"]) - float(a["wr_bytes"])
        drd = float(b["rd_bytes"]) - float(a["rd_bytes"])
        wr_rates.append(max(0.0, dwr) / dt / _MB)
        rd_rates.append(max(0.0, drd) / dt / _MB)
    wr_total = rd_total = 0.0
    if rows:
        wr_total = max(0.0, float(rows[-1]["wr_bytes"]) - float(rows[0]["wr_bytes"])) / _MB
        rd_total = max(0.0, float(rows[-1]["rd_bytes"]) - float(rows[0]["rd_bytes"])) / _MB
    rd_frac = rd_total / (rd_total + wr_total + _EPS)
    return {
        "wr_rate_mean_mbs": round(st.mean(wr_rates), 4) if wr_rates else 0.0,
        "wr_rate_max_mbs": round(max(wr_rates), 4) if wr_rates else 0.0,
        "wr_rate_p95_mbs": round(_pct(wr_rates, 0.95), 4) if wr_rates else 0.0,
        "rd_rate_mean_mbs": round(st.mean(rd_rates), 4) if rd_rates else 0.0,
        "wr_total_mb": round(wr_total, 2),
        "rd_total_mb": round(rd_total, 2),
        "rd_frac": round(rd_frac, 4),
    }


def load_diskio(cells_dir) -> dict:
    """cell_id -> {DISKIO_FEATS} for every work/<cell_id>/diskio_trajectory.jsonl."""
    out = {}
    work = Path(cells_dir) / "work"
    if not work.exists():
        return out
    for d in sorted(work.iterdir()):
        p = d / "diskio_trajectory.jsonl"
        if p.exists():
            out[d.name] = cell_diskio(p)
    return out


def _selftest():
    import tempfile
    import os
    # synthetic: wr climbs 5 MB every 0.6 s -> ~8.33 MB/s; rd flat.
    lines = []
    wr = 0
    for seq in range(10):
        lines.append(json.dumps({"seq": seq, "t_emit_epoch": 1000.0 + 0.6 * seq,
                                 "rd_bytes": 4096, "wr_bytes": wr}))
        wr += 5_000_000
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.write(fd, ("\n".join(lines) + "\n").encode()); os.close(fd)
    f = cell_diskio(Path(path))
    os.unlink(path)
    assert abs(f["wr_rate_mean_mbs"] - 8.3333) < 0.01, f
    assert f["rd_rate_mean_mbs"] == 0.0, f
    assert abs(f["wr_total_mb"] - 45.0) < 0.01, f   # 9 intervals * 5 MB
    assert f["rd_total_mb"] == 0.0, f               # reads flat -> 0
    assert f["rd_frac"] == 0.0, f                   # pure writer
    # second synthetic: encrypt-in-place, rd climbs 2 MB + wr climbs 4 MB per step.
    lines, rd, wr = [], 0, 0
    for seq in range(10):
        lines.append(json.dumps({"seq": seq, "t_emit_epoch": 1000.0 + 0.6 * seq,
                                 "rd_bytes": rd, "wr_bytes": wr}))
        rd += 2_000_000; wr += 4_000_000
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.write(fd, ("\n".join(lines) + "\n").encode()); os.close(fd)
    g = cell_diskio(Path(path))
    os.unlink(path)
    assert abs(g["rd_total_mb"] - 18.0) < 0.01, g   # 9 * 2 MB
    assert abs(g["wr_total_mb"] - 36.0) < 0.01, g   # 9 * 4 MB
    assert abs(g["rd_frac"] - (18.0 / 54.0)) < 0.001, g   # 0.3333 read share
    print("diskio_features self-test OK:", f, g)


if __name__ == "__main__":
    _selftest()
