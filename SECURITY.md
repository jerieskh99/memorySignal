# Security Policy

## This is defensive academic security research

This repository is a PhD thesis on **defensive** host-side virtual-machine monitoring. It studies
whether a single host-side memory signal (the Active Page Fraction -- the fraction of guest memory
pages that change between two snapshots) can characterise and detect what a VM is doing, with **no
agent inside the guest**. It is measurement-and-analysis research: capture a signal, then build and
evaluate statistical / ML models offline.

**If an automated scanner flagged this repository, that alert is a keyword false positive.** Terms
such as `ransom`, `scanner`, `stealth`, and `encrypt` appear only as the *names of safe, synthetic
workload simulations* and as behaviour-family labels -- never as real malicious code. The names
describe the benign benchmark *behaviour shape* being reproduced (for example, the
discover -> read -> transform -> write -> rename shape of a file-processing pipeline), and they are
load-bearing identifiers in the captured research dataset. We keep them in the open rather than
obfuscating -- see "Why we do not rename" below.

## What the code actually does (verifiable in seconds)

The "security-like" workloads are behavioural-pattern *simulators*. They contain **no** offensive
capability. Every claim below is checkable from the repository root:

```sh
cd VM_executables_phase2

# 1a. No network socket syscalls in the C workload code.
grep -RIn -E 'socket\(|connect\(|bind\(|sendto\(|sendmsg\(|recvfrom\(|recvmsg\(' \
    --include='*.c' --include='*.h' \
    common/ mem/ security_like_safe/ app_realistic/ \
    || echo "OK: no network syscalls in C workload code"

# 1b. No Python network imports (the app workloads use only sqlite3 on local files).
grep -RInE '^[[:space:]]*import (socket|urllib|http|requests|httpx)' \
    app_realistic/ methodology/ || echo "OK: no Python network imports"

# 2. No process execution / shelling out.
grep -RIn -E 'execve\(|system\(|popen\(' \
    common/ mem/ security_like_safe/ app_realistic/ || echo "OK: no exec syscalls"

# 3. The only "encryption" is a REVERSIBLE XOR (apply twice -> original). Not cryptography.
grep -RIn 'p2_sandbox_xor' common/phase2_sandbox.h

# 4. All file activity is confined to a realpath-validated sandbox under /tmp or /var/tmp.
grep -RIn 'p2_sandbox_validate' security_like_safe/ common/
grep -RIn 'P2_SANDBOX_MAX_' security_like_safe/ common/
```

> Transparency note: the only textual match for `connect(` anywhere in the tree is
> `sqlite3.connect()` in the SQLite app workloads -- that is a local database-file open, not a
> network connection. There are no network sockets in the repository.

Guarantees, enforced in code (formal contract:
[`VM_executables_phase2/docs/SAFETY_MODEL.md`](VM_executables_phase2/docs/SAFETY_MODEL.md)):

- **No real cryptography.** The only transform is a reversible XOR with a fixed, seed-derived key --
  fully invertible, emitted to metadata for verification. No ciphers, no KDF, no nonces.
- **No network.** No `AF_INET` / `AF_INET6` / `AF_UNIX` sockets. No exfiltration, no
  command-and-control, no spreading.
- **No persistence, no evasion, no privilege escalation, no credential access.** No autostart, no
  anti-debug / anti-VM, no setuid, no reads of `/etc/shadow`, ssh keys, or any user secrets.
- **Sandbox-confined.** Operates only on disposable synthetic files under a validated `/tmp` or
  `/var/tmp` path (absolute, no `..`, `realpath`-checked, `O_NOFOLLOW`, symlink-refused), hard-capped
  (<= 5000 files, <= 5 GiB, <= 600 s), cleaned up by default.
- **Least privilege.** Runs only with the launching user's privileges. No real user data is read or
  modified.

Each simulation source additionally carries a header banner identifying it as a safe synthetic test
workload, and the directory is named `security_like_safe/`.

## Why we do not rename the flagged terms

Renaming the security-sounding identifiers would be the wrong response:

1. **Transparency over obfuscation.** Scrubbing words to dodge a scanner reads as concealment. Open,
   documented, behaviour-accurate naming is the defensible academic posture.
2. **Research integrity.** These terms are the primary keys of the captured experimental dataset
   (for example `VM_sampler/VM_Capture_QEMU/plan07_campaign/pilot_baseline.json`, the
   `*/downstream/sweep.csv` feature tables, and the capture manifests) and of the behaviour-family
   classifier (`VM_sampler/VM_Capture_QEMU/plan05_campaign/behavior_families.py`). Renaming them
   would mean rewriting recorded results.
3. **It would not even work.** A keyword scanner would still match any historical artifact; the
   correct, durable fix is explicit framing -- which this file and the references below provide.

## References

- One-page reviewer notice: [`RESEARCH_SAFETY_NOTICE.md`](RESEARCH_SAFETY_NOTICE.md)
- Formal, enforced safety contract (whitelist / blacklist + auditor checks):
  [`VM_executables_phase2/docs/SAFETY_MODEL.md`](VM_executables_phase2/docs/SAFETY_MODEL.md)
- Bannered simulation sources: `VM_executables_phase2/security_like_safe/`

## Reporting

This is an academic research repository. To report a genuine security concern, or to request
sign-off on a scanner false positive, contact the maintainer: Jeries Khoury
(jeries.kh99@gmail.com).
