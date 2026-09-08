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

import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CACHE = "~/.cache/plan10/chains"
# where the capture console already puts the repo on the server (plan07_campaign/ui/console.sh)
DEFAULT_REMOTE_REPO = "$HOME/memorySignal/VM_sampler/VM_Capture_QEMU"
DEFAULT_REMOTE_STORE = "~/.cache/plan10/l1"
_RE_SNAP = re.compile(r"^\d{6}\.zst$")


class SshTransport:
    """Runs a command on the server and pulls a file back. The only part that needs a server.

    Separated so everything above it -- the remote command, its JSON reply, where the pulled
    file lands, the failure paths -- is exercised in the tests through a transport that runs
    the same commands locally.
    """

    def __init__(self, src: "SshSource"):
        self.src = src

    def run(self, remote_cmd: str, timeout: int = 7200) -> tuple[int, str, str]:
        r = subprocess.run(self.src.ssh_argv(remote_cmd), capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr

    def pull(self, remote_path: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        ssh_cmd = "ssh " + " ".join(shlex.quote(o) for o in self.src._ssh_opts())
        r = subprocess.run(["rsync", "-a", "-e", ssh_cmd,
                            f"{self.src.target}:{remote_path}", str(local_path)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SourceError(f"rsync of {remote_path} failed (exit {r.returncode}): {r.stderr.strip()[:300]}")


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
                 port: int = 22, cache: str | Path = DEFAULT_CACHE, mode: str = "fetch",
                 remote_repo: str = DEFAULT_REMOTE_REPO, remote_store: str = DEFAULT_REMOTE_STORE,
                 remote_python: str = "python3", transport=None):
        if not host or not remote_root:
            raise SourceError("ssh source needs host and remote_root")
        if mode not in ("fetch", "remote"):
            raise SourceError(f"ssh mode must be 'fetch' or 'remote'; got {mode!r}")
        self.host = host
        self.user = user or None
        self.key = os.path.expanduser(key) if key else None
        self.port = int(port or 22)
        self.remote_root = remote_root.rstrip("/")
        self.cache = Path(os.path.expanduser(str(cache)))
        # fetch: pull each recording's chain here and extract locally.
        # remote: run the extraction on the server and pull back only the L1 npz.
        self.mode = mode
        self.remote_repo = remote_repo
        self.remote_store = remote_store
        # a login shell on the server may resolve a different python than the one carrying
        # numpy (the analysis env is often a venv, as plan08_b1/requirements.txt sets up).
        # --probe reports which interpreter answered, so a mismatch is visible before a run.
        self.remote_python = remote_python or "python3"
        # the transport is injectable so the remote path can be exercised without a server
        self.transport = transport or SshTransport(self)

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
                "remote_root": self.remote_root, "key": self.key, "cache": str(self.cache),
                "mode": self.mode, "remote_repo": self.remote_repo, "remote_store": self.remote_store,
                "remote_python": self.remote_python}

    # ---------------- remote execution ----------------

    def extract_cmd(self, rec_id: str, speed: int, columns: list[str], max_pairs: int | None,
                    probe: bool = False) -> str:
        """The command a remote run executes on the server."""
        q = shlex.quote
        argv = [self.remote_python, "plan10_analysis/runner/extract_cli.py",
                "--root", q(self.remote_root), "--rec-id", q(rec_id),
                "--speed", str(int(speed)), "--columns", q(",".join(columns)),
                "--store", q(self.remote_store)]
        if max_pairs:
            argv += ["--max-pairs", str(int(max_pairs))]
        if probe:
            argv.append("--probe")
        return f"cd {self.remote_repo} && " + " ".join(argv)

    def probe_remote(self) -> dict:
        """What the server has: python, numpy, zstd, the differ, the trace root."""
        code, out, err = self.transport.run(self.extract_cmd("_", 2, ["hamming"], None, probe=True), timeout=120)
        if code != 0:
            raise SourceError(f"remote probe failed (exit {code}): {(err or out).strip()[:300]}")
        try:
            return json.loads(out.strip().splitlines()[-1])
        except (ValueError, IndexError) as e:
            raise SourceError(f"remote probe did not return JSON: {out.strip()[:200]}") from e

    def remote_extract(self, rec_id: str, speed: int, columns: list[str], store: Path,
                       max_pairs: int | None = None, log=None) -> Path:
        """Extract on the server, pull back only the npz and its meta. Returns the local npz."""
        cmd = self.extract_cmd(rec_id, speed, sorted(set(columns) | {"hamming"}), max_pairs)
        if log:
            log(f"remote: {cmd}")
        code, out, err = self.transport.run(cmd)
        if code != 0:
            raise SourceError(f"remote extraction of {rec_id} failed (exit {code}): {(err or out).strip()[-400:]}")
        try:
            res = json.loads(out.strip().splitlines()[-1])
        except (ValueError, IndexError) as e:
            raise SourceError(f"remote extraction did not return JSON: {out.strip()[-300:]}") from e
        if "npz" not in res:
            raise SourceError(f"remote extraction reported: {res}")
        store = Path(store)
        store.mkdir(parents=True, exist_ok=True)
        local_npz = store / Path(res["npz"]).name
        self.transport.pull(res["npz"], local_npz)
        self.transport.pull(res["meta"], store / Path(res["meta"]).name)
        if log:
            log(f"remote extraction of {rec_id}: {res['n_pairs']} pairs"
                f"{' (reused on the server)' if res.get('reused') else ''}, pulled {local_npz.name}")
        return local_npz

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
                         key=spec.get("key"), port=spec.get("port", 22), cache=spec.get("cache") or DEFAULT_CACHE,
                         mode=spec.get("mode") or "fetch",
                         remote_repo=spec.get("remote_repo") or DEFAULT_REMOTE_REPO,
                         remote_store=spec.get("remote_store") or DEFAULT_REMOTE_STORE,
                         remote_python=spec.get("remote_python") or "python3")
    raise SourceError(f"unknown source kind {kind!r}")
