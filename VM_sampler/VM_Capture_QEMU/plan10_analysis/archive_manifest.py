"""archive_manifest.py -- the archive keeps its own inventory; readers stop walking it.

The archive (family/workload/variant/repNNN__label/NNNNNN.zst, with the capture's substrate
trajectory beside each chain) is append-only at the file level: a snapshot is written once and
never rewritten. So the only party that knows when a recording is COMPLETE is the tool that put
it there and verified it. That tool registers the recording here, in
`<root>/.manifest/manifest.json`, and a reader -- the analysis console, first of all -- reads
that one file instead of stat-ing a hundred thousand snapshots over NFS. A recording that is
still arriving is not in the manifest and so cannot be selected; there is no stability guess.

The file IS a corpus manifest (`plan10.corpus_manifest.v1`, the same shape corpus_manifest.py
produces from a full walk), so the console consumes it unchanged. Each recording additionally
carries `has.substrate_columns`, the header of its trajectory, which the walk cannot give (a
listing has no file contents) and which is what the Channels module needs to know what a
recording can serve without a re-diff.

Two writers finishing in the same second must not clobber each other: updates take a lock (an
atomically created directory, which works on NFS where flock may not), rewrite to a temp file,
and rename over the old one, so a reader never sees a partial manifest.

`rebuild` is reconciliation: a full walk that rewrites the manifest from what is actually there.
It is the safety net for files copied in by hand without registering, and the only path that
ever walks the archive.

Run:  python3 plan10_analysis/archive_manifest.py register <root> <rel>
      python3 plan10_analysis/archive_manifest.py unregister <root> <rel>
      python3 plan10_analysis/archive_manifest.py rebuild <root>
      python3 plan10_analysis/archive_manifest.py show <root>
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
QEMU_DIR = HERE.parent
if str(QEMU_DIR) not in sys.path:
    sys.path.insert(0, str(QEMU_DIR))
from plan10_analysis import corpus_manifest as cm          # noqa: E402
from plan10_analysis.sources import Entry, LocalSource      # noqa: E402

MANIFEST_DIR = ".manifest"
MANIFEST_NAME = "manifest.json"
LOCK_NAME = "lock.d"
LOCK_STALE_S = 120          # a writer that died mid-update; break its lock after this
LOCK_WAIT_S = 60            # how long a second writer waits for the first


class ArchiveError(RuntimeError):
    pass


def manifest_path(root: Path) -> Path:
    return Path(root) / MANIFEST_DIR / MANIFEST_NAME


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- lock

class _Lock:
    """mkdir is atomic on every filesystem that matters, NFS included; flock is not."""

    def __init__(self, root: Path):
        self.d = Path(root) / MANIFEST_DIR / LOCK_NAME

    def __enter__(self):
        self.d.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        while True:
            try:
                self.d.mkdir()
                (self.d / "owner").write_text(f"{os.getpid()}@{os.uname().nodename} {_now()}\n")
                return self
            except FileExistsError:
                try:
                    age = time.time() - self.d.stat().st_mtime
                except FileNotFoundError:
                    continue
                if age > LOCK_STALE_S:
                    _rmdir(self.d)              # a dead writer; take it over
                    continue
                if time.time() - t0 > LOCK_WAIT_S:
                    raise ArchiveError(f"manifest lock held for {age:.0f}s by {self._owner()}: {self.d}")
                time.sleep(0.2)

    def __exit__(self, *_):
        _rmdir(self.d)

    def _owner(self) -> str:
        try:
            return (self.d / "owner").read_text().strip()
        except OSError:
            return "?"


def _rmdir(d: Path) -> None:
    """Release the lock. Rename the directory away FIRST: a rename is atomic and succeeds even
    when the NFS client has left a silly-renamed .nfs* file inside (that made a plain rmdir fail
    with 'Directory not empty' and crash the first rebuild AFTER it had written the manifest).
    Then delete the renamed directory with a few retries; a leftover lock.d.released-* is litter
    a later release sweeps up, never a lock."""
    gone = d.with_name(f"{d.name}.released-{os.getpid()}-{int(time.time() * 1000)}")
    try:
        os.rename(d, gone)
    except FileNotFoundError:
        return
    for stale in [gone] + [q for q in d.parent.glob(f"{d.name}.released-*") if q != gone]:
        for _ in range(10):
            try:
                for f in stale.iterdir():
                    f.unlink()
                stale.rmdir()
                break
            except FileNotFoundError:
                break
            except OSError:
                time.sleep(0.2)


# --------------------------------------------------------------------------- read / write

def load(root: Path) -> dict | None:
    p = manifest_path(root)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ArchiveError(f"unreadable manifest {p}: {e}")


def _write(root: Path, m: dict) -> None:
    p = manifest_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, indent=1))
    os.replace(tmp, p)


def _recount(m: dict) -> dict:
    recs = sorted(m["recordings"], key=lambda r: r["id"])
    m["recordings"] = recs
    m["n_recordings"] = len(recs)
    m["n_with_chain"] = sum(1 for r in recs if r["has"]["chain"])
    m["n_with_substrate_csv"] = sum(1 for r in recs if r["has"]["substrate_csv"])
    m["n_workloads"] = len({r["workload"] for r in recs})
    m["families"] = sorted({r["family"] for r in recs})
    return m


def _empty(root: Path) -> dict:
    return {"schema": cm.SCHEMA, "scanned_at": _now(), "root": str(root),
            "source": {"kind": "local", "root": str(root)}, "metrics_root": None,
            "n_recordings": 0, "n_with_chain": 0, "n_with_substrate_csv": 0, "n_workloads": 0,
            "families": [], "recordings": [], "partial_files": [], "unjoined_metrics_artifacts": {},
            "warnings": [], "archive_manifest": {"rebuilt_at": None, "updated_at": _now(),
                                                 "registered_since_rebuild": 0}}


# --------------------------------------------------------------------------- one recording

def _trajectory_columns(rec_dir: Path) -> list[str] | None:
    """The trajectory's header, read in place. None when there is no trajectory."""
    from plan10_analysis.runner import trajectory
    p = trajectory.find(rec_dir)
    if p is None:
        return None
    try:
        return trajectory.columns(p)
    except trajectory.TrajectoryError:
        return []


def describe(root: Path, rel: str) -> dict | None:
    """One recording's manifest entry, from its own directory. None if it is not there.

    Reuses corpus_manifest.scan_listing over a listing of just this directory, so the entry
    is exactly what a full walk would say about it -- one parser, not two.
    """
    root = Path(root)
    rel = rel.strip("/")
    if ".." in rel.split("/"):
        raise ArchiveError(f"refusing path that escapes the root: {rel}")
    d = root / rel
    if not d.is_dir():
        return None
    entries = [Entry(rel, 0, True)]
    for f in sorted(d.iterdir()):
        if f.is_file():
            entries.append(Entry(f"{rel}/{f.name}", f.stat().st_size, False))
    try:
        m = cm.scan_listing(entries, str(root), None, {"kind": "local", "root": str(root)})
    except cm.CorpusMissing:
        return None
    if not m["recordings"]:
        return None
    r = m["recordings"][0]
    r["has"]["substrate_columns"] = _trajectory_columns(d)
    r["registered_at"] = _now()
    return r


def register(root: Path, rel: str) -> dict:
    """Add or replace one recording's entry. Called by whatever wrote it, after verifying it."""
    root = Path(root)
    entry = describe(root, rel)
    if entry is None:
        raise ArchiveError(f"not a recording directory: {root / rel}")
    with _Lock(root):
        m = load(root) or _empty(root)
        m["recordings"] = [r for r in m["recordings"] if r["id"] != entry["id"]] + [entry]
        m = _recount(m)
        am = m.setdefault("archive_manifest", {"rebuilt_at": None, "registered_since_rebuild": 0})
        am["updated_at"] = _now()
        am["registered_since_rebuild"] = int(am.get("registered_since_rebuild") or 0) + 1
        am["last_registered"] = entry["id"]
        _write(root, m)
    return entry


def unregister(root: Path, rel: str) -> bool:
    """Forget a recording (after it was removed from the archive). True if it was listed."""
    root = Path(root)
    rel = rel.strip("/")
    with _Lock(root):
        m = load(root)
        if m is None:
            return False
        before = len(m["recordings"])
        m["recordings"] = [r for r in m["recordings"] if r["id"] != rel]
        if len(m["recordings"]) == before:
            return False
        m = _recount(m)
        m.setdefault("archive_manifest", {})["updated_at"] = _now()
        _write(root, m)
    return True


# --------------------------------------------------------------------------- reconciliation

def rebuild(root: Path, with_columns: bool = True) -> dict:
    """Walk the whole archive and rewrite the manifest from what is actually there.

    The walk takes minutes; a writer may register a recording while it runs. The walk saw
    that recording's directory either before it was complete or not at all, so at write time
    anything registered since the walk began is taken from the live manifest, not from the
    walk -- the writer's verified entry outranks a stale glimpse. Reconciliation removes only
    what is genuinely gone.
    """
    root = Path(root)
    t0 = _now()
    src = LocalSource(root)
    m = cm.scan_listing(src.listing(), str(root), None, src.describe())
    if with_columns:
        for r in m["recordings"]:
            r["has"]["substrate_columns"] = _trajectory_columns(root / r["id"]) if r["has"]["substrate_csv"] else None
    with _Lock(root):
        live = load(root)
        fresh = {r["id"]: r for r in (live or {"recordings": []})["recordings"]
                 if (r.get("registered_at") or "") >= t0}
        if fresh:
            m["recordings"] = [r for r in m["recordings"] if r["id"] not in fresh] + list(fresh.values())
            m = _recount(m)
        m["archive_manifest"] = {"rebuilt_at": m["scanned_at"], "updated_at": _now(),
                                 "registered_since_rebuild": len(fresh),
                                 "kept_from_live_during_walk": sorted(fresh)}
        _write(root, m)
    return m


# --------------------------------------------------------------------------- cli

def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] not in ("register", "unregister", "rebuild", "show"):
        print(__doc__.split("Run:")[1].strip(), file=sys.stderr)
        return 2
    cmd, root = argv[0], Path(argv[1])
    try:
        if cmd == "register":
            if len(argv) != 3:
                raise ArchiveError("register needs <root> <rel>")
            e = register(root, argv[2])
            print(f"registered {e['id']}: {e['n_snapshots']} snapshots, {e['bytes']} B, "
                  f"trajectory {'yes' if e['has']['substrate_csv'] else 'no'}"
                  + (f" ({len(e['has']['substrate_columns'])} columns)" if e['has'].get('substrate_columns') else ""))
        elif cmd == "unregister":
            if len(argv) != 3:
                raise ArchiveError("unregister needs <root> <rel>")
            print("removed" if unregister(root, argv[2]) else "was not listed")
        elif cmd == "rebuild":
            m = rebuild(root)
            print(f"rebuilt: {m['n_recordings']} recordings, {m['n_with_substrate_csv']} with a trajectory, "
                  f"{m['n_workloads']} workloads -> {manifest_path(root)}")
        else:
            m = load(root)
            if m is None:
                print("no manifest")
                return 1
            am = m.get("archive_manifest") or {}
            print(f"{m['n_recordings']} recordings, {m['n_with_substrate_csv']} with a trajectory; "
                  f"rebuilt {am.get('rebuilt_at')}, updated {am.get('updated_at')}, "
                  f"{am.get('registered_since_rebuild', 0)} registered since")
    except ArchiveError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
