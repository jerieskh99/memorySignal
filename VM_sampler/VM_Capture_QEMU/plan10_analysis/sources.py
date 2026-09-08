#!/usr/bin/env python3
"""sources.py -- where the recordings live: a local tree, or a tree behind SSH.

One interface, two implementations, so nothing downstream knows which it has:

    src = make_source({"kind": "local", "root": "/path/to/zstd_local"})
    src = make_source({"kind": "ssh", "host": "server", "user": "jeries",
                       "key": "~/.ssh/id_ed25519", "remote_root": "/project/.../zstd_local",
                       "port": 22, "cache": "~/.cache/plan10/chains"})

    src.listing()        -> [Entry(relpath, size, is_dir)]   the whole tree, one call
    src.fetch(rec_rel)   -> local Path holding that recording's .zst files
    src.describe()       -> what goes into a sidecar (never the key contents)
    src.test()           -> (ok, message)

The SSH source lists with GNU `find -printf` in one round trip and fetches one recording
at a time with rsync into a local cache; the executor then runs locally on the cache. It
never executes analysis remotely. Pure stdlib.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CACHE = "~/.cache/plan10/chains"
_RE_SNAP = re.compile(r"^\d{6}\.zst$")


@dataclass(frozen=True)
class Entry:
    relpath: str
    size: int
    is_dir: bool


class SourceError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# local
# ---------------------------------------------------------------------------

class LocalSource:
    kind = "local"

    def __init__(self, root: str | Path):
        self.root = Path(os.path.expanduser(str(root)))

    def describe(self) -> dict:
        return {"kind": "local", "root": str(self.root)}

    def test(self) -> tuple[bool, str]:
        if not self.root.is_dir():
            return False, f"not a directory: {self.root}"
        return True, f"ok: {self.root}"

    def listing(self) -> list[Entry]:
        if not self.root.is_dir():
            raise SourceError(f"trace root does not exist: {self.root}")
        out: list[Entry] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            base = Path(dirpath)
            for d in dirnames:
                out.append(Entry(str((base / d).relative_to(self.root)), 0, True))
            for f in sorted(filenames):
                p = base / f
                try:
                    out.append(Entry(str(p.relative_to(self.root)), p.stat().st_size, False))
                except OSError:
                    continue
        return out

    def fetch(self, rec_rel: str) -> Path:
        p = self.root / rec_rel
        if not p.is_dir():
            raise SourceError(f"recording not found: {p}")
        return p


# ---------------------------------------------------------------------------
# ssh
# ---------------------------------------------------------------------------

class SshSource:
    kind = "ssh"

    def __init__(self, host: str, remote_root: str, user: str | None = None, key: str | None = None,
                 port: int = 22, cache: str | Path = DEFAULT_CACHE):
        if not host or not remote_root:
            raise SourceError("ssh source needs host and remote_root")
        self.host = host
        self.user = user or None
        self.key = os.path.expanduser(key) if key else None
        self.port = int(port or 22)
        self.remote_root = remote_root.rstrip("/")
        self.cache = Path(os.path.expanduser(str(cache)))

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host

    def _ssh_opts(self) -> list[str]:
        opts = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15", "-p", str(self.port)]
        if self.key:
            opts += ["-i", self.key]
        return opts

    def ssh_argv(self, remote_cmd: str) -> list[str]:
        return ["ssh", *self._ssh_opts(), self.target, remote_cmd]

    def rsync_argv(self, rec_rel: str, dest: Path) -> list[str]:
        ssh_cmd = "ssh " + " ".join(shlex.quote(o) for o in self._ssh_opts())
        return ["rsync", "-a", "--partial", "--include=*.zst", "--exclude=*", "-e", ssh_cmd,
                f"{self.target}:{shlex.quote(self.remote_root + '/' + rec_rel)}/", str(dest) + "/"]

    def describe(self) -> dict:
        return {"kind": "ssh", "host": self.host, "user": self.user, "port": self.port,
                "remote_root": self.remote_root, "key": self.key, "cache": str(self.cache)}

    def test(self) -> tuple[bool, str]:
        try:
            r = subprocess.run(self.ssh_argv(f"test -d {shlex.quote(self.remote_root)} && echo ok || echo missing"),
                               capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as e:
            return False, f"ssh failed: {e}"
        if r.returncode != 0:
            return False, f"ssh exit {r.returncode}: {r.stderr.strip()[:300]}"
        if r.stdout.strip() != "ok":
            return False, f"remote root missing: {self.remote_root}"
        return True, f"ok: {self.target}:{self.remote_root}"

    # GNU find: %y type, %s size, %P path relative to the starting point
    def listing_cmd(self) -> str:
        return f"cd {shlex.quote(self.remote_root)} && find . -mindepth 1 \\( -type d -o -type f \\) -printf '%y\\t%s\\t%P\\n'"

    @staticmethod
    def parse_listing(text: str) -> list[Entry]:
        out: list[Entry] = []
        for line in text.splitlines():
            parts = line.split("\t", 2)
            if len(parts) != 3:
                continue
            ty, size, rel = parts
            if not rel or rel.startswith("."):
                continue
            out.append(Entry(rel, int(size or 0), ty == "d"))
        return out

    def listing(self) -> list[Entry]:
        r = subprocess.run(self.ssh_argv(self.listing_cmd()), capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise SourceError(f"remote listing failed (exit {r.returncode}): {r.stderr.strip()[:300]}")
        return self.parse_listing(r.stdout)

    def fetch(self, rec_rel: str) -> Path:
        dest = self.cache / rec_rel
        dest.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(self.rsync_argv(rec_rel, dest), capture_output=True, text=True)
        if r.returncode != 0:
            raise SourceError(f"rsync failed for {rec_rel} (exit {r.returncode}): {r.stderr.strip()[:300]}")
        if not any(_RE_SNAP.match(p.name) for p in dest.iterdir()):
            raise SourceError(f"fetched nothing for {rec_rel}")
        return dest


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------

def make_source(spec: dict):
    kind = (spec or {}).get("kind", "local")
    if kind == "local":
        if not spec.get("root"):
            raise SourceError("local source needs root")
        return LocalSource(spec["root"])
    if kind == "ssh":
        return SshSource(host=spec.get("host", ""), remote_root=spec.get("remote_root", ""), user=spec.get("user"),
                         key=spec.get("key"), port=spec.get("port", 22), cache=spec.get("cache") or DEFAULT_CACHE)
    raise SourceError(f"unknown source kind {kind!r}")
