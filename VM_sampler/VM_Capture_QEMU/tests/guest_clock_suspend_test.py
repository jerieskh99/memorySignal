#!/usr/bin/env python3
"""The 30-second suspend/resume clock test, run from the host.

Protocol (al-Farabi, time_axes_paper/council/02_al_farabi_three_timelines_form.md, section 7):
read the guest's clocks over SSH with a host timestamp on each side of the read, suspend the
domain for N seconds, resume, read again at once and then at a tail of later offsets, and
compare each guest delta with the host delta.

The guest read returns CLOCK_REALTIME, CLOCK_MONOTONIC, CLOCK_MONOTONIC_RAW and CLOCK_BOOTTIME.
MONOTONIC_RAW is not subject to NTP frequency slew, so a slewing sync client shows up as
MONOTONIC drifting away from MONOTONIC_RAW after the resume.

Every guest reading is bracketed by host time before the SSH command is sent and after its
output is received; the host time attributed to the reading is the midpoint and the bracket
width is recorded as the reading's uncertainty.

Nothing is written anywhere except the output files named on the command line. The only
change made to the VM is the one suspend/resume the operator is running this script to do.

Environment, same conventions as run_files_controlled.py:
  SSH_KEY   optional private key for the guest
  SSH_PASS  optional password (uses sshpass)
  SSH_OPTS  optional extra ssh options

Example:
  ./guest_clock_suspend_test.py --ssh-target jeries@192.168.122.50 \
      --domain "Kali Jeries" --suspend-seconds 30 --tail 0,60,180,600 \
      --out /tmp/clock_test_$(date +%Y%m%dT%H%M%S)
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time

GUEST_READ = (
    "python3 -c "
    + shlex.quote(
        "import time,json,os\n"
        "d={'real':time.time(),"
        "'mono':time.monotonic(),"
        "'mono_raw':time.clock_gettime(time.CLOCK_MONOTONIC_RAW),"
        "'boot':time.clock_gettime(time.CLOCK_BOOTTIME)}\n"
        "try:\n"
        "  d['clocksource']=open('/sys/devices/system/clocksource/clocksource0/current_clocksource').read().strip()\n"
        "except Exception as e:\n"
        "  d['clocksource']='ERR '+str(e)\n"
        "print(json.dumps(d))"
    )
)

GUEST_ENV = (
    "echo '--- clocksource'; cat /sys/devices/system/clocksource/clocksource0/current_clocksource; "
    "echo '--- available'; cat /sys/devices/system/clocksource/clocksource0/available_clocksource; "
    "echo '--- sync services (is-active)'; "
    "for s in chrony chronyd systemd-timesyncd ntp ntpd ntpsec qemu-guest-agent; do "
    "printf '%s: ' \"$s\"; systemctl is-active \"$s\" 2>/dev/null || echo unknown; done; "
    "echo '--- timedatectl'; timedatectl 2>/dev/null | head -20; "
    "echo '--- adjtimex (frequency/status)'; "
    "python3 -c \"import ctypes,ctypes.util\n"
    "class T(ctypes.Structure):\n"
    "  _fields_=[('modes',ctypes.c_int),('offset',ctypes.c_long),('freq',ctypes.c_long),('maxerror',ctypes.c_long),('esterror',ctypes.c_long),('status',ctypes.c_int),('constant',ctypes.c_long),('precision',ctypes.c_long),('tolerance',ctypes.c_long),('time_sec',ctypes.c_long),('time_usec',ctypes.c_long),('tick',ctypes.c_long),('pad',ctypes.c_long*20)]\n"
    "t=T(); libc=ctypes.CDLL(ctypes.util.find_library('c')); r=libc.adjtimex(ctypes.byref(t))\n"
    "print('ret',r,'freq_ppm',t.freq/65536.0,'status',t.status,'tick',t.tick)\" 2>/dev/null || echo 'adjtimex read failed'"
)


def host_now() -> float:
    return time.time()


def run(cmd: str, timeout: float = 30.0) -> tuple[int, str, str]:
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def ssh_base(target: str) -> str:
    parts: list[str] = []
    ssh_pass = os.environ.get("SSH_PASS", "")
    ssh_key = os.environ.get("SSH_KEY", "")
    ssh_opts = os.environ.get("SSH_OPTS", "")
    if ssh_pass:
        parts += ["sshpass", "-p", ssh_pass]
    parts += ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5", "-o", "BatchMode=" + ("no" if ssh_pass else "yes")]
    if ssh_key:
        parts += ["-i", ssh_key]
    if ssh_opts:
        parts += ssh_opts.split()
    parts.append(target)
    return " ".join(shlex.quote(p) for p in parts)


def guest_read(target: str, label: str, retries: int = 30, retry_sleep: float = 1.0) -> dict:
    """One bracketed guest clock reading. Retries while the guest's sshd is not yet answering
    (right after a resume the first attempt can fail); every attempt is logged."""
    base = ssh_base(target)
    attempts = []
    for i in range(1, retries + 1):
        t_send = host_now()
        rc, out, err = run(f"{base} {shlex.quote(GUEST_READ)}", timeout=20)
        t_recv = host_now()
        attempts.append({"attempt": i, "rc": rc, "host_send": t_send, "host_recv": t_recv, "stderr": err.strip()[-300:]})
        if rc == 0 and out.strip():
            try:
                g = json.loads(out.strip().splitlines()[-1])
            except json.JSONDecodeError:
                g = {"parse_error": out.strip()[-300:]}
            return {
                "label": label,
                "host_send": t_send,
                "host_recv": t_recv,
                "host_mid": (t_send + t_recv) / 2.0,
                "bracket_s": t_recv - t_send,
                "guest": g,
                "attempts": attempts,
            }
        time.sleep(retry_sleep)
    return {"label": label, "host_send": None, "host_recv": None, "host_mid": None, "bracket_s": None, "guest": None, "attempts": attempts}


def virsh(uri: str, sub: str, domain: str) -> tuple[int, str, str]:
    return run(f"virsh -c {shlex.quote(uri)} {sub} {shlex.quote(domain)}")


def wait_state(uri: str, domain: str, want: str, timeout: float = 60.0, poll: float = 0.05) -> tuple[bool, float, str]:
    deadline = host_now() + timeout
    last = ""
    while host_now() < deadline:
        rc, out, _ = virsh(uri, "domstate", domain)
        last = out.strip().lower()
        if rc == 0 and want in last:
            return True, host_now(), last
        time.sleep(poll)
    return False, host_now(), last


def fmt(x):
    return "n/a" if x is None else f"{x:.6f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ssh-target", required=True, help="user@guest-ip (the same target the capture console uses)")
    ap.add_argument("--domain", default=os.environ.get("VM_DOMAIN", "Kali Jeries"))
    ap.add_argument("--virsh-uri", default=os.environ.get("VIRSH_URI", "qemu:///system"))
    ap.add_argument("--suspend-seconds", type=float, default=30.0)
    ap.add_argument("--tail", default="0,60,180,600", help="seconds after resume at which to read the guest again")
    ap.add_argument("--pre-reads", type=int, default=3, help="baseline readings before the suspend, 1 s apart")
    ap.add_argument("--out", required=True, help="output prefix; writes <out>.jsonl, <out>.env.txt, <out>.summary.json")
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation before suspending")
    a = ap.parse_args()

    tail = [float(x) for x in a.tail.split(",") if x.strip()]
    jsonl = open(a.out + ".jsonl", "w")

    def log(rec: dict) -> None:
        rec.setdefault("host_logged", host_now())
        jsonl.write(json.dumps(rec) + "\n")
        jsonl.flush()

    # 0. Preconditions: the domain must be running, and we say so before touching it.
    rc, out, err = virsh(a.virsh_uri, "domstate", a.domain)
    state = out.strip().lower()
    print(f"[test] domain {a.domain!r} state: {state or err.strip()}")
    if rc != 0 or "running" not in state:
        print("[test] the domain is not running; start it first (virsh start) and re-run. Nothing was done.")
        return 2
    if not a.yes:
        ans = input(f"[test] This will suspend {a.domain!r} for {a.suspend_seconds:.0f} s and resume it. Continue? [y/N] ")
        if ans.strip().lower() != "y":
            print("[test] aborted; nothing was done.")
            return 1

    # 1. Environment on both sides, once.
    base = ssh_base(a.ssh_target)
    rc, genv, gerr = run(f"{base} {shlex.quote(GUEST_ENV)}", timeout=40)
    rc2, xml, xerr = run(f"virsh -c {shlex.quote(a.virsh_uri)} dumpxml {shlex.quote(a.domain)} | grep -A8 '<clock'")
    rc3, hostcs, _ = run("cat /sys/devices/system/clocksource/clocksource0/current_clocksource 2>/dev/null")
    with open(a.out + ".env.txt", "w") as f:
        f.write(f"host_time_utc_epoch {host_now()}\n")
        f.write(f"host clocksource: {hostcs.strip()}\n")
        f.write("=== guest environment ===\n" + genv + ("\n[stderr] " + gerr if gerr.strip() else "") + "\n")
        f.write("=== domain <clock> from dumpxml ===\n" + xml + ("\n[stderr] " + xerr if xerr.strip() else "") + "\n")
    print("[test] environment recorded to", a.out + ".env.txt")
    log({"kind": "env", "guest_env": genv, "domain_clock_xml": xml, "host_clocksource": hostcs.strip()})

    # 2. Baseline readings.
    pre = []
    for i in range(a.pre_reads):
        r = guest_read(a.ssh_target, f"pre{i}")
        log({"kind": "read", **r})
        pre.append(r)
        print(f"[test] {r['label']}: host_mid={fmt(r['host_mid'])} bracket={fmt(r['bracket_s'])}s guest={r['guest']}")
        if i < a.pre_reads - 1:
            time.sleep(1.0)
    if pre[-1]["guest"] is None:
        print("[test] could not read the guest before suspending; aborting, nothing was suspended.")
        return 3

    # 3. Suspend, wait, resume; every transition stamped and confirmed by domstate polling.
    t_suspend_cmd = host_now()
    rc, out, err = virsh(a.virsh_uri, "suspend", a.domain)
    ok_p, t_paused, st = wait_state(a.virsh_uri, a.domain, "paused")
    log({"kind": "suspend", "host_cmd": t_suspend_cmd, "rc": rc, "stdout": out.strip(), "stderr": err.strip(), "confirmed": ok_p, "host_confirmed": t_paused, "state": st})
    print(f"[test] suspend: cmd at {t_suspend_cmd:.6f}, paused confirmed at {t_paused:.6f} ({t_paused - t_suspend_cmd:.3f}s), ok={ok_p}")
    if not ok_p:
        print("[test] suspend not confirmed; resuming defensively.")
        virsh(a.virsh_uri, "resume", a.domain)
        return 4

    time.sleep(a.suspend_seconds)

    t_resume_cmd = host_now()
    rc, out, err = virsh(a.virsh_uri, "resume", a.domain)
    ok_r, t_running, st = wait_state(a.virsh_uri, a.domain, "running")
    log({"kind": "resume", "host_cmd": t_resume_cmd, "rc": rc, "stdout": out.strip(), "stderr": err.strip(), "confirmed": ok_r, "host_confirmed": t_running, "state": st})
    print(f"[test] resume: cmd at {t_resume_cmd:.6f}, running confirmed at {t_running:.6f} ({t_running - t_resume_cmd:.3f}s), ok={ok_r}")
    pause_host_s = t_running - t_paused

    # 4. Post readings at the tail offsets, measured from the confirmed resume.
    post = []
    for off in tail:
        target_t = t_running + off
        while host_now() < target_t:
            time.sleep(min(0.2, target_t - host_now()))
        r = guest_read(a.ssh_target, f"post+{off:g}s")
        log({"kind": "read", **r})
        post.append(r)
        print(f"[test] {r['label']}: host_mid={fmt(r['host_mid'])} bracket={fmt(r['bracket_s'])}s guest={r['guest']}")

    # 5. Arithmetic. Reference = last pre reading. For each post reading, guest deltas vs host delta.
    ref = pre[-1]
    rows = []
    for r in post:
        if r["guest"] is None or ref["guest"] is None:
            rows.append({"label": r["label"], "error": "no reading"})
            continue
        dh = r["host_mid"] - ref["host_mid"]
        g, g0 = r["guest"], ref["guest"]
        row = {
            "label": r["label"],
            "host_delta_s": dh,
            "expected_if_frozen_s": dh - pause_host_s,
            "guest_real_delta_s": g["real"] - g0["real"],
            "guest_mono_delta_s": g["mono"] - g0["mono"],
            "guest_mono_raw_delta_s": g["mono_raw"] - g0["mono_raw"],
            "guest_boot_delta_s": g["boot"] - g0["boot"],
            "uncertainty_s": (r["bracket_s"] or 0) / 2 + (ref["bracket_s"] or 0) / 2,
        }
        row["mono_minus_frozen_expectation_s"] = row["guest_mono_delta_s"] - row["expected_if_frozen_s"]
        row["real_minus_host_s"] = row["guest_real_delta_s"] - dh
        row["mono_minus_mono_raw_s"] = row["guest_mono_delta_s"] - row["guest_mono_raw_delta_s"]
        rows.append(row)

    summary = {
        "domain": a.domain,
        "ssh_target": a.ssh_target,
        "suspend_seconds_requested": a.suspend_seconds,
        "pause_host_s_confirmed": pause_host_s,
        "suspend_confirm_latency_s": t_paused - t_suspend_cmd,
        "resume_confirm_latency_s": t_running - t_resume_cmd,
        "reference_reading": ref["label"],
        "rows": rows,
        "how_to_read": {
            "frozen": "guest_mono_delta ~= expected_if_frozen (mono_minus_frozen_expectation ~ 0 within uncertainty) at every tail point; real_minus_host ~= -pause at every point: both clocks froze and nothing re-synced.",
            "realtime_resynced_at_resume": "mono frozen, but real_minus_host ~ 0 already at post+0s: something steps CLOCK_REALTIME on resume.",
            "realtime_stepped_later": "mono frozen; real_minus_host ~ -pause at post+0s, then ~0 at a later tail point: a stepping sync client, poll interval = when it jumped.",
            "slewing": "mono frozen at post+0s, but mono_minus_mono_raw grows over the tail (and mono_minus_frozen_expectation grows): a slewing sync client; guest runs fast against the host until the debt is paid.",
            "tracking": "guest_mono_delta ~= host_delta (not host minus pause): the clocksource tracks host time; the declaration fails. Not expected on kvm-clock.",
        },
    }
    with open(a.out + ".summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log({"kind": "summary", **summary})

    print("\n[test] pause on host (paused-confirmed to running-confirmed): %.3f s" % pause_host_s)
    print("[test] %-12s %12s %12s %12s %12s %12s %10s %10s %10s" % ("reading", "host_d", "frozen_exp", "mono_d", "mono_raw_d", "real_d", "mono-exp", "real-host", "mono-raw"))
    for row in rows:
        if "error" in row:
            print("[test] %-12s %s" % (row["label"], row["error"]))
            continue
        print("[test] %-12s %12.3f %12.3f %12.3f %12.3f %12.3f %10.3f %10.3f %10.4f" % (
            row["label"], row["host_delta_s"], row["expected_if_frozen_s"], row["guest_mono_delta_s"],
            row["guest_mono_raw_delta_s"], row["guest_real_delta_s"], row["mono_minus_frozen_expectation_s"],
            row["real_minus_host_s"], row["mono_minus_mono_raw_s"]))
    print("[test] wrote", a.out + ".jsonl", a.out + ".env.txt", a.out + ".summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
