#!/usr/bin/env python3
"""One host-side command for the guest clock page experiment.

Steps, all from the host:
  1. copy tests/guest_clock_page.py into the guest over scp
  2. run run_timing_instrumentation_experiment.py (producer only, dumps kept) with the
     writer as the guest-side --test-command
  3. copy the writer's heartbeat file back from the guest
  4. run tests/read_clock_page.py on exactly the dumps this run produced (mtime >= run
     start), joined to this run's snapshot_timings.jsonl
  5. unless --keep-writer-in-guest, remove the writer and its heartbeat from the guest
  6. write <workdir>/RUN_RECORD.json: every command, path, and script checksum used

The writer and reader stay in the repository; the guest only ever holds a copy.

Where the dumps go. With useDomainDir true in the config (the AppArmor workaround: the
domain's libvirt-generated profile grants only /var/lib/libvirt/qemu/domain-<id>-<name>/),
the producer resolves that directory itself at startup via `virsh domid`. So this driver
does not trust the config's imageDir: it takes the dump paths from the image_path field
of this run's snapshot_timings.jsonl, which is the producer's own record. libvirt deletes
the per-domain directory when the domain shuts down, so the reader runs immediately and
the domain is resumed, never stopped, by this script. --delete-dumps-after removes this
run's dumps once the reader has finished.

If the timing driver's producer stop lands during a suspend the domain is left paused;
this script checks `virsh domstate` after the run and resumes it before touching the guest.

Environment, same conventions as run_files_controlled.py: SSH_KEY, SSH_PASS, SSH_OPTS.

Example:
    python3 tests/run_clock_page_experiment.py --ssh-target kali@192.168.222.63 \
        --duration 60 --interval-ms 500 --ram-mb 1024 \
        --workdir ~/memorySignal/timing_runs/clockpage_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
QEMU_DIR = HERE.parent
DRIVER = QEMU_DIR / "run_timing_instrumentation_experiment.py"
WRITER = HERE / "guest_clock_page.py"
READER = HERE / "read_clock_page.py"
DEFAULT_CONFIG = QEMU_DIR / "config_qemu_upc.json"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ssh_parts() -> tuple[list[str], list[str]]:
    """Return (ssh prefix without target, scp prefix) honouring SSH_KEY / SSH_PASS / SSH_OPTS."""
    key, pw, opts = os.environ.get("SSH_KEY", ""), os.environ.get("SSH_PASS", ""), os.environ.get("SSH_OPTS", "")
    pre = ["sshpass", "-p", pw] if pw else []
    common = ["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5"]
    if key:
        common += ["-i", key]
    if opts:
        common += opts.split()
    return pre + ["ssh"] + common, pre + ["scp", "-q"] + common


def run(cmd: list[str], record: list[dict], check: bool = True, **kw) -> subprocess.CompletedProcess:
    shown = " ".join(shlex.quote(c) for c in cmd)
    shown_safe = shown.replace(os.environ.get("SSH_PASS", "\0"), "***") if os.environ.get("SSH_PASS") else shown
    print(f"[clock-page-exp] $ {shown_safe}", flush=True)
    t = time.time()
    p = subprocess.run(cmd, text=True, **kw)
    record.append({"cmd": shown_safe, "rc": p.returncode, "host_time": t, "elapsed_s": time.time() - t})
    if check and p.returncode != 0:
        print(f"[clock-page-exp] command failed with rc={p.returncode}", file=sys.stderr)
        sys.exit(p.returncode)
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ssh-target", required=True, help="user@guest-ip")
    ap.add_argument("--duration", type=int, default=60, help="host wall-clock capture window, seconds")
    ap.add_argument("--interval-ms", type=int, default=500)
    ap.add_argument("--ram-mb", type=int, default=1024)
    ap.add_argument("--hz", type=int, default=1000, help="writer rate in the guest")
    ap.add_argument("--writer-seconds", type=int, default=None, help="default: duration + 30")
    ap.add_argument("--workdir", required=True, help="this run's directory (timings JSONL, driver JSON, reader output, record)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--dump-dir", default=None, help="where the producer writes dumps; default: imageDir from --config")
    ap.add_argument("--guest-path", default="~/guest_clock_page.py")
    ap.add_argument("--guest-heartbeat", default="~/clock_page_heartbeat.jsonl")
    ap.add_argument("--keep-writer-in-guest", action="store_true")
    ap.add_argument("--delete-dumps-after", action="store_true", help="remove this run's dumps once the reader has finished")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    a = ap.parse_args()

    workdir = Path(a.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(a.config).expanduser().read_text())
    dump_dir = Path(a.dump_dir).expanduser() if a.dump_dir else Path(cfg.get("imageDir", "/var/lib/libvirt/qemu/dump"))
    writer_seconds = a.writer_seconds if a.writer_seconds is not None else a.duration + 30
    for p in (DRIVER, WRITER, READER):
        if not p.exists():
            print(f"missing: {p}", file=sys.stderr)
            return 2

    record: dict = {
        "started_host_epoch": time.time(),
        "ssh_target": a.ssh_target, "domain": cfg.get("domain"), "config": str(a.config),
        "dump_dir": str(dump_dir), "workdir": str(workdir),
        "duration_s": a.duration, "interval_ms": a.interval_ms, "ram_mb": a.ram_mb,
        "writer_hz": a.hz, "writer_seconds": writer_seconds,
        "scripts": {"driver": {"path": str(DRIVER), "sha256": sha256(DRIVER)},
                    "writer": {"path": str(WRITER), "sha256": sha256(WRITER)},
                    "reader": {"path": str(READER), "sha256": sha256(READER)}},
        "commands": [],
    }
    cmds = record["commands"]

    print(f"[clock-page-exp] domain={cfg.get('domain')!r} guest={a.ssh_target} dumps->{dump_dir} workdir={workdir}")
    print(f"[clock-page-exp] {a.duration}s window at {a.interval_ms} ms, {a.ram_mb} MiB: roughly "
          f"{a.duration * 1000 // (a.interval_ms + 1500)}-{a.duration * 1000 // a.interval_ms} dumps of {a.ram_mb} MiB kept in {dump_dir}")
    if not a.yes:
        if input("[clock-page-exp] continue? [y/N] ").strip().lower() != "y":
            print("[clock-page-exp] aborted; nothing was done.")
            return 1

    ssh, scp = ssh_parts()

    # 1. writer into the guest
    run(scp + [str(WRITER), f"{a.ssh_target}:{a.guest_path}"], cmds)
    run(ssh + [a.ssh_target, f"python3 -m py_compile {a.guest_path} && echo writer-ok"], cmds)

    # 2. the timing experiment, writer as the guest-side workload
    run_start = time.time()
    record["run_start_host_epoch"] = run_start
    test_cmd = f"python3 {a.guest_path} --seconds {writer_seconds} --hz {a.hz} --heartbeat {a.guest_heartbeat} --quiet"
    driver_cmd = [sys.executable, str(DRIVER),
                  "--duration", str(a.duration), "--interval-ms", str(a.interval_ms), "--ram-mb", str(a.ram_mb),
                  "--keep-dumps", "--no-self-clean",
                  "--config", str(a.config),
                  "--ssh-target", a.ssh_target, "--test-command", test_cmd,
                  "--workdir", str(workdir), "--output-json", str(workdir / "timing_experiment.json")]
    if a.dump_dir:
        driver_cmd += ["--image-dir", str(dump_dir)]
    run(driver_cmd, cmds, cwd=str(QEMU_DIR))

    # 3. the producer's stop can land mid-suspend and leave the domain paused: resume it
    domain = cfg.get("domain", "")
    uri = cfg.get("virshUri", "qemu:///system")
    st = run(["virsh", "-c", uri, "domstate", domain], cmds, check=False, capture_output=True)
    state = (st.stdout or "").strip().lower()
    record["domstate_after_run"] = state
    if "paused" in state:
        run(["virsh", "-c", uri, "resume", domain], cmds, check=False)
        time.sleep(2)
        st = run(["virsh", "-c", uri, "domstate", domain], cmds, check=False, capture_output=True)
        record["domstate_after_resume"] = (st.stdout or "").strip().lower()
        print(f"[clock-page-exp] domain was left paused by the producer stop; resumed -> {record['domstate_after_resume']}")

    # 4. heartbeat back from the guest (best effort; the guest needs a moment after a resume)
    for attempt in range(6):
        p = run(scp + [f"{a.ssh_target}:{a.guest_heartbeat}", str(workdir / "guest_heartbeat.jsonl")], cmds, check=False)
        if p.returncode == 0:
            break
        time.sleep(5)

    # 5. read exactly this run's dumps: the paths the producer itself recorded
    dumps = []
    jsonl = workdir / "snapshot_timings.jsonl"
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("seq", -1) >= 0 and d.get("image_path"):
                dumps.append(Path(d["image_path"]))
    dumps = [p for p in dumps if p.exists()]
    if not dumps:  # fallback: anything new in the configured dir
        dumps = sorted(p for p in dump_dir.glob("memory_dump-*.raw") if p.stat().st_mtime >= run_start - 1)
    record["n_dumps_this_run"] = len(dumps)
    record["dumps"] = [str(p) for p in dumps]
    record["dump_dir_actual"] = str(dumps[0].parent) if dumps else None
    if not dumps:
        print(f"[clock-page-exp] no dumps found via snapshot_timings.jsonl image_path nor in {dump_dir}; check producer.log", file=sys.stderr)
    else:
        print(f"[clock-page-exp] {len(dumps)} dumps for this run under {dumps[0].parent}")
        run([sys.executable, str(READER), *map(str, dumps),
             "--timings", str(workdir / "snapshot_timings.jsonl"),
             "--out", str(workdir / "clock_page")], cmds, check=False)

    # 6. tidy the guest
    if not a.keep_writer_in_guest:
        run(ssh + [a.ssh_target, f"rm -f {a.guest_path} {a.guest_heartbeat}"], cmds, check=False)
        record["writer_removed_from_guest"] = True
    else:
        record["writer_removed_from_guest"] = False

    # optional: reclaim disk
    if a.delete_dumps_after and dumps:
        for p in dumps:
            try:
                p.unlink()
            except OSError as e:
                print(f"[clock-page-exp] could not delete {p}: {e}", file=sys.stderr)
        record["dumps_deleted_after"] = True

    record["finished_host_epoch"] = time.time()
    (workdir / "RUN_RECORD.json").write_text(json.dumps(record, indent=2))
    print(f"[clock-page-exp] done. record: {workdir / 'RUN_RECORD.json'}; reader output: {workdir / 'clock_page.summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
