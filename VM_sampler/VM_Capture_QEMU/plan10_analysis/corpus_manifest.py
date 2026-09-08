#!/usr/bin/env python3
"""corpus_manifest.py -- what recordings exist on this machine, and what each one has.

Scans the migrated trace tree the migration agent writes to (its default root is
read from plan07_campaign/ui/console.sh, TRACES_LOCAL_DIR), which is laid out

    <root>/<family>/<workload>/<variant-args>/rep<NNN>__<run-label>/<NNNNNN>.zst

with 000000.zst the full base dump and every later file a `zstd --patch-from`
delta (reconstruct_zstd_chain.sh). Optionally also scans a metrics root for the
substrate CSVs the B1 extractor reads (`*substrate_trajectory.csv[.zst|.gz]`),
joined to recordings BY WORKLOAD NAME ONLY -- the join is recorded as such.

Everything a recording does not carry is reported as unrecorded, never filled
with a default. In particular the capture speed level and the sampling
interval are NOT in the chain tree and NOT in the run metadata
(plan07_campaign/runs/<label>.json records capture_metric and retention but
not substrateSpeed or intervalMsec); their capture-time values live in
config_qemu_upc.json. The manifest carries null for both and says why.

A missing or empty root is a loud failure (exit 2), not a sample corpus.

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
        # a trailing hash marks a name the orchestrator shortened; flags may be cut mid-word
        out["truncated"] = "--see_" in dirname or "--max-m_" in dirname or "--phase-marke" in dirname and "--phase-markers" not in dirname
    return out


def _workload_from_metrics_name(basename: str) -> str:
    """Mirror of b1_extract_all.sh: strip run_matrix_testN_ and the trajectory suffix."""
    b = re.sub(r"\.(zst|gz)$", "", basename)
    b = re.sub(r"\.csv$", "", b)
    b = re.sub(r"^run_matrix_test\d+_", "", b)
    b = re.sub(r"(\.npy)?\.substrate_trajectory$", "", b)
    return b


def scan_metrics_root(metrics_root: Path | None) -> dict:
    """Index substrate CSVs by workload name. Other B1 artifacts are counted, not joined."""
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


def scan(root: Path, metrics_root: Path | None = None) -> dict:
    if not root.exists() or not root.is_dir():
        raise CorpusMissing(f"trace root does not exist: {root}")
    metrics = scan_metrics_root(metrics_root)
    recs: list[dict] = []
    partials: list[str] = []
    warnings: list[str] = []
    for fam_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        for wl_dir in sorted(p for p in fam_dir.iterdir() if p.is_dir()):
            for var_dir in sorted(p for p in wl_dir.iterdir() if p.is_dir()):
                for rep_dir in sorted(p for p in var_dir.iterdir() if p.is_dir()):
                    m = _RE_REP.match(rep_dir.name)
                    if not m:
                        warnings.append(f"not a rep dir, skipped: {rep_dir.relative_to(root)}")
                        continue
                    snaps, nbytes, base_bytes = [], 0, None
                    for f in rep_dir.iterdir():
                        if _RE_PARTIAL.match(f.name):
                            partials.append(str(f.relative_to(root)))
                            continue
                        s = _RE_SNAP.match(f.name)
                        if s and f.is_file():
                            st = f.stat()
                            snaps.append(int(s.group(1)))
                            nbytes += st.st_size
                            if s.group(1) == "000000":
                                base_bytes = st.st_size
                    snaps.sort()
                    n = len(snaps)
                    contiguous = snaps == list(range(n))
                    if not contiguous and n:
                        warnings.append(f"snapshot numbering has gaps: {rep_dir.relative_to(root)}")
                    wl = wl_dir.name
                    variant = parse_variant(var_dir.name)
                    substrate = metrics["by_workload"].get(wl, [])
                    recs.append({
                        "id": str(rep_dir.relative_to(root)),
                        "family": fam_dir.name,
                        "workload": wl,
                        "family_prefix_matches_dir": wl.split("_", 1)[0] == fam_dir.name,
                        "variant": variant,
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
                            "substrate_join": "workload-name" if substrate else None,
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
        raise CorpusMissing(f"trace root has no recordings under family/workload/variant/rep: {root}")
    return {
        "schema": SCHEMA,
        "scanned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(root),
        "metrics_root": str(metrics_root) if metrics_root else None,
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
