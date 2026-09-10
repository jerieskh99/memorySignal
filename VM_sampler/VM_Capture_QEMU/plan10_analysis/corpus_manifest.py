#!/usr/bin/env python3
"""corpus_manifest.py -- what recordings exist, and what each one has.

Works over a LISTING, not a filesystem: `scan_listing(entries, root_label)` builds the
manifest from (relpath, size, is_dir) entries, so a local walk (sources.LocalSource) and a
remote GNU `find` (sources.SshSource) produce the same manifest through the same code.
`scan(root)` is the local convenience wrapper the console build uses.

The tree the migration agent writes (its default root is read from
plan07_campaign/ui/console.sh, TRACES_LOCAL_DIR) is laid out

    <root>/<family>/<workload>/<variant-args>/rep<NNN>__<run-label>/<NNNNNN>.zst

with 000000.zst the full base dump and every later file a `zstd --patch-from` delta
(reconstruct_zstd_chain.sh). Optionally a local metrics root is scanned for the substrate
CSVs the B1 extractor reads, joined to recordings BY WORKLOAD NAME ONLY.

Everything a recording does not carry is reported as unrecorded, never filled with a
default: capture speed and sampling interval are in neither the chain tree nor
runs/<label>.json (which records capture_metric and retention only); their capture-time
values live in config_qemu_upc.json. A missing or empty root is a loud failure (exit 2).

Run:  python3 plan10_analysis/corpus_manifest.py [--root R] [--metrics-root M] [--out m.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
QEMU_DIR = HERE.parent
if str(QEMU_DIR) not in sys.path:
    sys.path.insert(0, str(QEMU_DIR))
from plan10_analysis.sources import Entry, LocalSource  # noqa: E402

CONSOLE_SH = QEMU_DIR / "plan07_campaign" / "ui" / "console.sh"

SCHEMA = "plan10.corpus_manifest.v1"
_RE_REP = re.compile(r"^rep(\d+)(?:__(.+))?$")
_RE_SNAP = re.compile(r"^(\d{6})\.zst$")
_RE_PARTIAL = re.compile(r"^\.\d{6}\.zst\.")
_RE_DURATION = re.compile(r"--duration_(\d+)")
_RE_SEED = re.compile(r"--seed_([0-9a-zA-Z]+)")
_RE_TRAIL_HASH = re.compile(r"_([0-9a-f]{8})$")
_RE_SUBSTRATE = re.compile(r"substrate_trajectory\.csv(\.zst|\.gz)?$")


class CorpusMissing(RuntimeError):
    pass


def default_root() -> Path:
    """The migration agent's local destination, read from console.sh rather than typed."""
    text = CONSOLE_SH.read_text() if CONSOLE_SH.exists() else ""
    m = re.search(r'TRACES_LOCAL_DIR="\$\{TRACES_LOCAL_DIR:-([^}]+)\}"', text)
    if not m:
        raise CorpusMissing(f"TRACES_LOCAL_DIR default not found in {CONSOLE_SH}; pass --root")
    return Path(os.path.expandvars(m.group(1).replace("$HOME", os.environ.get("HOME", "~"))))


def parse_variant(dirname: str) -> dict:
    """Best-effort parse of the argument-encoded variant dir. The raw name is always kept."""
    out: dict = {"raw": dirname, "duration_s": None, "seed": None, "hash": None, "truncated": False}
    m = _RE_DURATION.search(dirname)
    if m:
        out["duration_s"] = int(m.group(1))
    m = _RE_SEED.search(dirname)
    if m:
        out["seed"] = m.group(1)
    m = _RE_TRAIL_HASH.search(dirname)
    if m:
        out["hash"] = m.group(1)
        out["truncated"] = ("--see_" in dirname) or ("--max-m_" in dirname) or ("--phase-marke" in dirname and "--phase-markers" not in dirname)
    return out


def _workload_from_metrics_name(basename: str) -> str:
    """Mirror of b1_extract_all.sh: strip run_matrix_testN_ and the trajectory suffix."""
    b = re.sub(r"\.(zst|gz)$", "", basename)
    b = re.sub(r"\.csv$", "", b)
    b = re.sub(r"^run_matrix_test\d+_", "", b)
    b = re.sub(r"(\.npy)?\.substrate_trajectory$", "", b)
    return b


def scan_metrics_root(metrics_root: Path | None) -> dict:
    """Index substrate CSVs by workload name (local only). Other B1 artifacts are counted, not joined."""
    idx: dict[str, list[str]] = {}
    others = {"hc_field": 0, "b1_trajectory": 0}
    if not metrics_root:
        return {"by_workload": idx, "others": others}
    if not metrics_root.exists():
        raise CorpusMissing(f"metrics root does not exist: {metrics_root}")
    for p in metrics_root.rglob("*"):
        if not p.is_file():
            continue
        if _RE_SUBSTRATE.search(p.name):
            idx.setdefault(_workload_from_metrics_name(p.name), []).append(str(p))
        elif p.name.startswith("hc_field.csv"):
            others["hc_field"] += 1
        elif p.name == "b1_trajectory.jsonl":
            others["b1_trajectory"] += 1
    return {"by_workload": idx, "others": others}


def scan_listing(entries: list[Entry], root_label: str, metrics: dict | None = None,
                 source: dict | None = None) -> dict:
    """Build the manifest from a listing. Depth-4 dirs named rep<NNN>[__label] are recordings."""
    metrics = metrics or {"by_workload": {}, "others": {"hc_field": 0, "b1_trajectory": 0}}
    files: dict[str, int] = {e.relpath: e.size for e in entries if not e.is_dir}
    dirs = {e.relpath for e in entries if e.is_dir}
    # files under each dir (direct children only)
    children: dict[str, list[tuple[str, int]]] = {}
    for rel, size in files.items():
        parent, _, name = rel.rpartition("/")
        children.setdefault(parent, []).append((name, size))

    recs: list[dict] = []
    partials: list[str] = []
    warnings: list[str] = []
    for d in sorted(dirs):
        parts = d.split("/")
        if len(parts) != 4:
            continue
        fam, wl, var, rep = parts
        if fam.startswith("."):
            continue
        m = _RE_REP.match(rep)
        if not m:
            warnings.append(f"not a rep dir, skipped: {d}")
            continue
        snaps: list[int] = []
        nbytes, base_bytes = 0, None
        inchain: list[str] = []      # a substrate trajectory the capture left beside its chain
        for name, size in children.get(d, []):
            if _RE_PARTIAL.match(name):
                partials.append(f"{d}/{name}")
                continue
            s = _RE_SNAP.match(name)
            if s:
                snaps.append(int(s.group(1)))
                nbytes += size
                if s.group(1) == "000000":
                    base_bytes = size
            elif _RE_SUBSTRATE.search(name):
                inchain.append(f"{d}/{name}")
        snaps.sort()
        n = len(snaps)
        contiguous = snaps == list(range(n))
        if not contiguous and n:
            warnings.append(f"snapshot numbering has gaps: {d}")
        # In-chain first: it is this recording's own output, not a workload-name guess. The
        # listing carries it for a remote source too, which a metrics root (local only) cannot.
        substrate = metrics["by_workload"].get(wl, [])
        join = "workload-name" if substrate else None
        if inchain:
            substrate = sorted(inchain) + substrate
            join = "in-chain"
        recs.append({
            "id": d,
            "family": fam,
            "workload": wl,
            "family_prefix_matches_dir": wl.split("_", 1)[0] == fam,
            "variant": parse_variant(var),
            "rep": int(m.group(1)),
            "run_label": m.group(2),
            "n_snapshots": n,
            "n_pairs": max(0, n - 1),
            "contiguous": contiguous,
            "bytes": nbytes,
            "base_bytes": base_bytes,
            "has": {
                "chain": n >= 2,
                "base_only": n == 1,
                "substrate_csv": bool(substrate),
                "substrate_csv_paths": substrate,
                "substrate_join": join,
            },
            "speed": None,
            "speed_source": "unrecorded: not in the chain tree nor in runs/<label>.json; "
                            "capture-time value is config_qemu_upc.json substrateSpeed",
            "iv_ms": None,
            "iv_source": "unrecorded: capture-time value is config_qemu_upc.json intervalMsec",
            "n_pages": None,
            "n_pages_source": "unrecorded: derivable as dump_size // page_size "
                              "(plan02_apf_helper) once the base dump is reconstructed",
            "status": "ok" if n >= 2 else ("base_only" if n == 1 else "empty"),
        })
    if not recs:
        raise CorpusMissing(f"no recordings under family/workload/variant/rep in: {root_label}")
    return {
        "schema": SCHEMA,
        "scanned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": root_label,
        "source": source or {"kind": "local", "root": root_label},
        "metrics_root": None,
        "n_recordings": len(recs),
        "n_with_chain": sum(1 for r in recs if r["has"]["chain"]),
        "n_with_substrate_csv": sum(1 for r in recs if r["has"]["substrate_csv"]),
        "n_workloads": len({r["workload"] for r in recs}),
        "families": sorted({r["family"] for r in recs}),
        "recordings": recs,
        "partial_files": partials,
        "unjoined_metrics_artifacts": metrics["others"],
        "warnings": warnings,
    }


def scan(root: Path, metrics_root: Path | None = None) -> dict:
    """Local convenience: walk `root` and build the manifest."""
    root = Path(root)
    if not root.is_dir():
        raise CorpusMissing(f"trace root does not exist: {root}")
    src = LocalSource(root)
    m = scan_listing(src.listing(), str(root), scan_metrics_root(metrics_root), src.describe())
    m["metrics_root"] = str(metrics_root) if metrics_root else None
    return m


def scan_source(src) -> dict:
    """Manifest over any Source (local or ssh)."""
    d = src.describe()
    label = d.get("root") or f"{d.get('host')}:{d.get('remote_root')}"
    return scan_listing(src.listing(), label, None, d)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=None, help="trace root (default: console.sh TRACES_LOCAL_DIR)")
    ap.add_argument("--metrics-root", type=Path, default=None, help="optional root holding substrate CSVs")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    try:
        root = a.root or default_root()
        m = scan(root, a.metrics_root)
    except CorpusMissing as e:
        print(f"[corpus_manifest] REFUSED: {e}", file=sys.stderr)
        return 2
    if a.out:
        a.out.write_text(json.dumps(m, indent=1) + "\n")
        print(f"[corpus_manifest] wrote {a.out}")
    print(f"[corpus_manifest] {m['n_recordings']} recordings, {m['n_with_chain']} with a chain, "
          f"{m['n_with_substrate_csv']} with a substrate CSV, {m['n_workloads']} workloads, "
          f"families {m['families']}")
    if m["partial_files"]:
        print(f"[corpus_manifest] {len(m['partial_files'])} partial transfer files ignored")
    for w in m["warnings"][:10]:
        print(f"[corpus_manifest] warning: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
