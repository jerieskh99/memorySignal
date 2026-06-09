#!/usr/bin/env python3
"""Per-cell disk-I/O features from the Plan-06 diskio_trajectory.jsonl.

Each diskio trajectory line is {seq, t_emit_epoch, rd_bytes, wr_bytes} with the
byte counters CUMULATIVE since domain boot (from domblkstat). Per interval:
rate = (counter[i+1]-counter[i]) / (t[i+1]-t[i]); aggregated per cell. Counters
are monotonic; a negative delta (domain restart) clamps to 0.

  wr_rate_mean_mbs / wr_rate_max_mbs / wr_rate_p95_mbs  -- write rate (MB/s)
  rd_rate_mean_mbs                                       -- read rate (MB/s)
  wr_total_mb                                            -- total written (MB)

Keyed by cell_id, mirroring extra_features.load_extra. Run directly for a
synthetic self-test.
"""
import json
import statistics as st
from pathlib import Path

DISKIO_FEATS = ["wr_rate_mean_mbs", "wr_rate_max_mbs", "wr_rate_p95_mbs",
                "rd_rate_mean_mbs", "wr_total_mb"]
_MB = 1_000_000.0


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
    wr_total = 0.0
    if rows:
        wr_total = max(0.0, float(rows[-1]["wr_bytes"]) - float(rows[0]["wr_bytes"])) / _MB
    return {
        "wr_rate_mean_mbs": round(st.mean(wr_rates), 4) if wr_rates else 0.0,
        "wr_rate_max_mbs": round(max(wr_rates), 4) if wr_rates else 0.0,
        "wr_rate_p95_mbs": round(_pct(wr_rates, 0.95), 4) if wr_rates else 0.0,
        "rd_rate_mean_mbs": round(st.mean(rd_rates), 4) if rd_rates else 0.0,
        "wr_total_mb": round(wr_total, 2),
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
    print("diskio_features self-test OK:", f)


if __name__ == "__main__":
    _selftest()
