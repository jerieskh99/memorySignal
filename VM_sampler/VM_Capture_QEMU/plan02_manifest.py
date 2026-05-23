#!/usr/bin/env python3
"""plan02_manifest.py -- manifest CSV format + atomic state transitions.

Owner: SA (Senior Architect) + DE (Senior Data Engineer).
Reference: docs/tuning_plans/experiment_audit.md Decisions D-1, D-3, D-6, D-9.

The manifest is the SOURCE OF TRUTH for which Plan-02 cells need to run,
which have completed, which have failed, and how they should be ordered.

Columns (CSV):
  cell_id            sha1 prefix of (workload, iv, duration, replicate)
  manifest_id        uuid for this whole batch
  block_id           int. cells with the same block_id run in random order
                     within one ~4-hour wall-clock window. blocks run
                     in ascending order.
  workload           workload name (e.g. sandbox_ransom_batched)
  interval_ms        int. iv config
  duration_s         int. how long the cell runs
  replicate          int. 0..N-1 per (workload, iv, duration) tuple
  is_warmup          bool. discarded cell at session start
  status             pending | running | ok | failed | skipped
  expected_path      where the per-cell JSON should land
  retry_count        int. EN retry policy bumps this on per-cell crash
  notes              free-text. why a cell failed, etc

Operations:
  build_manifest(...)    plan the matrix from CLI args
  load(path)             read the CSV into memory
  save(path, rows)       write the CSV via temp+rename
  next_pending(rows)     pick the next cell to execute
  set_status(rows, id, status, **kwargs)   atomically mutate one row
  set_status_on_disk(path, id, status, **kw)  load + mutate + save

Crash-recovery semantics:
  On orchestrator restart, manifest is read fresh. Any row with
  status='running' is treated as crashed mid-flight; EN policy is to
  mark it 'failed' with a note, then continue from the next pending row.
  The operator can flip a 'failed' row back to 'pending' to retry.

Idempotency:
  Re-running an 'ok' row is a no-op. The orchestrator skips it.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from plan02_schema import cell_id


# ---------------------------------------------------------------------------
# Row dataclass
# ---------------------------------------------------------------------------

VALID_STATUSES = {"pending", "running", "ok", "failed", "skipped"}


@dataclass
class ManifestRow:
    cell_id: str
    manifest_id: str
    block_id: int
    workload: str
    interval_ms: int
    duration_s: int
    replicate: int
    is_warmup: bool = False
    status: str = "pending"
    expected_path: str = ""
    retry_count: int = 0
    notes: str = ""
    # Day-7 additions (D-19): workload-launching variables. Empty means
    # "do not launch a workload" (back-compat with the Step 1 pilot).
    workload_command: str = ""
    ssh_target: str = ""
    keep_dumps: bool = False

    def to_csv(self) -> dict[str, str]:
        d = asdict(self)
        d["is_warmup"] = "1" if self.is_warmup else "0"
        d["keep_dumps"] = "1" if self.keep_dumps else "0"
        for k, v in list(d.items()):
            d[k] = "" if v is None else str(v)
        return d

    @classmethod
    def from_csv(cls, row: dict[str, str]) -> "ManifestRow":
        return cls(
            cell_id=row["cell_id"],
            manifest_id=row["manifest_id"],
            block_id=int(row["block_id"]),
            workload=row["workload"],
            interval_ms=int(row["interval_ms"]),
            duration_s=int(row["duration_s"]),
            replicate=int(row["replicate"]),
            is_warmup=(row.get("is_warmup", "0") == "1"),
            status=row.get("status", "pending"),
            expected_path=row.get("expected_path", ""),
            retry_count=int(row.get("retry_count", "0") or 0),
            notes=row.get("notes", ""),
            workload_command=row.get("workload_command", ""),
            ssh_target=row.get("ssh_target", ""),
            keep_dumps=(row.get("keep_dumps", "0") == "1"),
        )


CSV_HEADER = [
    "cell_id", "manifest_id", "block_id", "workload", "interval_ms",
    "duration_s", "replicate", "is_warmup", "status",
    "expected_path", "retry_count", "notes",
    "workload_command", "ssh_target", "keep_dumps",
]


# ---------------------------------------------------------------------------
# Build / generate
# ---------------------------------------------------------------------------

def build_manifest(
    workloads: list[str],
    intervals_ms: list[int],
    durations_s: list[int],
    replicates: int,
    output_dir: Path,
    seed: int = 0,
    block_size: int = 24,
    add_warmup_per_block: bool = True,
    workload_commands: dict[str, str] | None = None,
    ssh_target: str = "",
    keep_dumps: bool = False,
) -> list[ManifestRow]:
    """Generate the cell list, assign block_ids, randomize within blocks.

    block_size = number of cells per ~4 h block. With ~5 min/cell average,
    24 cells per block stays close to 2 h (allows some retry budget per
    block to fit in 4 h wall-clock).

    add_warmup_per_block prepends a single warmup cell at the head of each
    block, using the first workload at the median interval/duration.
    Warmup output is discarded by the orchestrator.
    """
    if not workloads:
        raise ValueError("workloads must not be empty")
    if not intervals_ms:
        raise ValueError("intervals_ms must not be empty")
    if not durations_s:
        raise ValueError("durations_s must not be empty")
    if replicates < 1:
        raise ValueError("replicates must be >= 1")

    manifest_id = uuid.uuid4().hex[:12]
    rng = random.Random(seed)
    wcmds = workload_commands or {}

    cells: list[ManifestRow] = []
    for workload in workloads:
        for iv in intervals_ms:
            for dur in durations_s:
                for r in range(replicates):
                    cid = cell_id(workload, iv, dur, r)
                    expected = str(output_dir / f"cell_{cid}.json")
                    cells.append(ManifestRow(
                        cell_id=cid,
                        manifest_id=manifest_id,
                        block_id=-1,  # filled in below
                        workload=workload,
                        interval_ms=iv,
                        duration_s=dur,
                        replicate=r,
                        is_warmup=False,
                        status="pending",
                        expected_path=expected,
                        workload_command=wcmds.get(workload, ""),
                        ssh_target=ssh_target,
                        keep_dumps=keep_dumps,
                    ))

    # Randomize the cell list, then chunk into blocks.
    rng.shuffle(cells)
    blocked: list[ManifestRow] = []
    block_id = 0
    median_iv = sorted(intervals_ms)[len(intervals_ms) // 2]
    median_dur = sorted(durations_s)[len(durations_s) // 2]
    for i in range(0, len(cells), block_size):
        chunk = cells[i:i + block_size]
        for cell in chunk:
            cell.block_id = block_id
        if add_warmup_per_block:
            wcid = cell_id(f"warmup_block{block_id}", median_iv, median_dur, 0)
            blocked.append(ManifestRow(
                cell_id=wcid,
                manifest_id=manifest_id,
                block_id=block_id,
                workload=workloads[0],
                interval_ms=median_iv,
                duration_s=min(median_dur, 300),  # cap warmup at 5 min
                replicate=0,
                is_warmup=True,
                status="pending",
                expected_path=str(output_dir / f"warmup_block{block_id}.json"),
                notes="warmup; output discarded",
            ))
        blocked.extend(chunk)
        block_id += 1

    return blocked


# ---------------------------------------------------------------------------
# Atomic load / save
# ---------------------------------------------------------------------------

def save(path: Path, rows: Iterable[ManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writeheader()
        for row in rows:
            w.writerow(row.to_csv())
        f.flush()
        try:
            import os
            os.fsync(f.fileno())
        except OSError:
            pass
    tmp.replace(path)


def load(path: Path) -> list[ManifestRow]:
    if not path.is_file():
        return []
    with path.open() as f:
        reader = csv.DictReader(f)
        return [ManifestRow.from_csv(r) for r in reader]


# ---------------------------------------------------------------------------
# Status transitions
# ---------------------------------------------------------------------------

def next_pending(rows: list[ManifestRow]) -> ManifestRow | None:
    """Return the first pending row in CSV order (already randomized in
    build_manifest). Returns None if all rows are non-pending.
    """
    for r in rows:
        if r.status == "pending":
            return r
    return None


def crashed_running_to_failed(rows: list[ManifestRow]) -> int:
    """On orchestrator restart, any 'running' row is treated as crashed."""
    fixed = 0
    for r in rows:
        if r.status == "running":
            r.status = "failed"
            r.notes = (r.notes + " | crashed; auto-marked failed on restart").strip(" |")
            fixed += 1
    return fixed


def set_status(rows: list[ManifestRow], cid: str, status: str,
               notes_append: str | None = None,
               bump_retry: bool = False) -> None:
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid status: {status}")
    for r in rows:
        if r.cell_id == cid:
            r.status = status
            if notes_append:
                r.notes = (r.notes + " | " + notes_append).strip(" |")
            if bump_retry:
                r.retry_count += 1
            return
    raise KeyError(f"cell_id not found: {cid}")


def set_status_on_disk(path: Path, cid: str, status: str,
                       notes_append: str | None = None,
                       bump_retry: bool = False) -> None:
    rows = load(path)
    set_status(rows, cid, status, notes_append=notes_append,
               bump_retry=bump_retry)
    save(path, rows)


# ---------------------------------------------------------------------------
# Summary stats (for status reporting + audit log)
# ---------------------------------------------------------------------------

def summarize(rows: list[ManifestRow]) -> dict[str, int]:
    out: dict[str, int] = {s: 0 for s in VALID_STATUSES}
    out["warmup"] = 0
    out["total"] = len(rows)
    for r in rows:
        out[r.status] += 1
        if r.is_warmup:
            out["warmup"] += 1
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="plan02_manifest.py",
        description="Build, inspect, or mutate a Plan-02 manifest CSV.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build", help="generate a new manifest")
    pb.add_argument("--workloads", nargs="+", required=True)
    pb.add_argument("--intervals-ms", nargs="+", type=int, required=True)
    pb.add_argument("--durations-s", nargs="+", type=int, required=True)
    pb.add_argument("--replicates", type=int, default=3)
    pb.add_argument("--block-size", type=int, default=24)
    pb.add_argument("--no-warmup", action="store_true")
    pb.add_argument("--seed", type=int, default=0)
    pb.add_argument("--output", required=True, help="manifest CSV path")
    pb.add_argument("--cell-output-dir", required=True,
                    help="directory where per-cell JSONs will land")
    pb.add_argument("--workload-command", action="append", default=[],
                    help="workload=command pairs, repeated per workload. "
                         "Example: --workload-command "
                         "'sandbox_ransom_batched=/path/to/binary --rounds 5' "
                         "When non-empty, plan02_run.py SSHes to ssh_target "
                         "and launches the command with --phase-markers.")
    pb.add_argument("--ssh-target", default="",
                    help="ssh target for workload launch (e.g. kali@192.168.122.10)")
    pb.add_argument("--keep-dumps", action="store_true",
                    help="Mark every cell with keep_dumps=1 so the producer's "
                         "tail-cleanup does not delete dumps. Required for "
                         "post-cell analyzer integration. Disk impact: "
                         "ram_mb * snaps_per_cell per cell.")

    ps = sub.add_parser("summary", help="print manifest status summary")
    ps.add_argument("manifest")

    pl = sub.add_parser("list", help="list rows (optionally filter)")
    pl.add_argument("manifest")
    pl.add_argument("--status", default=None)

    pf = sub.add_parser("force-status", help="overwrite a cell's status")
    pf.add_argument("manifest")
    pf.add_argument("--cell-id", required=True)
    pf.add_argument("--status", required=True, choices=sorted(VALID_STATUSES))
    pf.add_argument("--note", default=None)

    args = p.parse_args(argv)

    if args.cmd == "build":
        # Parse --workload-command pairs into a dict
        wcmds: dict[str, str] = {}
        for spec in args.workload_command:
            if "=" not in spec:
                print(f"ERROR: --workload-command requires workload=command "
                      f"(got: {spec!r})", file=sys.stderr)
                return 2
            key, _, val = spec.partition("=")
            wcmds[key.strip()] = val.strip()

        rows = build_manifest(
            workloads=args.workloads,
            intervals_ms=args.intervals_ms,
            durations_s=args.durations_s,
            replicates=args.replicates,
            output_dir=Path(args.cell_output_dir),
            seed=args.seed,
            block_size=args.block_size,
            add_warmup_per_block=not args.no_warmup,
            workload_commands=wcmds,
            ssh_target=args.ssh_target,
            keep_dumps=args.keep_dumps,
        )
        save(Path(args.output), rows)
        print(f"wrote {len(rows)} rows to {args.output}", file=sys.stderr)
        for k, v in summarize(rows).items():
            print(f"  {k:>8}: {v}", file=sys.stderr)
        return 0

    if args.cmd == "summary":
        rows = load(Path(args.manifest))
        for k, v in summarize(rows).items():
            print(f"{k:>8}: {v}")
        return 0

    if args.cmd == "list":
        rows = load(Path(args.manifest))
        for r in rows:
            if args.status and r.status != args.status:
                continue
            print(f"{r.cell_id}  blk={r.block_id:>2}  "
                  f"{r.workload:<32}  iv={r.interval_ms:>4}  "
                  f"d={r.duration_s:>4}s  rep={r.replicate}  "
                  f"warm={r.is_warmup}  st={r.status}")
        return 0

    if args.cmd == "force-status":
        set_status_on_disk(Path(args.manifest), args.cell_id, args.status,
                           notes_append=args.note)
        print(f"set {args.cell_id} -> {args.status}", file=sys.stderr)
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(_main())
