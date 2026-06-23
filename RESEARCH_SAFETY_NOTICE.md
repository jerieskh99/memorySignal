# Research and Safety Notice

**For reviewers and automated-scanning teams.** This repository belongs to an academic PhD
thesis on **defensive** security research. If an automated scanner flagged it, that alert is a
**keyword false positive**: terms such as "ransom" and "scanner" appear only as the *names of
safe, synthetic workload simulations*, never as real malicious code. This one-page notice
explains what the project is, why those terms appear, and the safety guarantees. The formal,
enforced safety contract is in
[`VM_executables_phase2/docs/SAFETY_MODEL.md`](VM_executables_phase2/docs/SAFETY_MODEL.md).

---

## What this project is

The thesis studies whether a single **host-side memory signal** can identify and detect what a
virtual machine is doing, with no software installed inside the guest. The signal is the
**Active Page Fraction (APF)** — the fraction of the guest's memory pages that change between
two consecutive memory snapshots. The research goal is **detection and characterisation of
workload behaviour** (telling apart memory-sweeps, file-processing pipelines, etc.) for
defensive monitoring. It is measurement-and-analysis research: capture a signal, then build and
evaluate statistical/ML models offline.

## Why security-sounding terms appear

To study whether the memory signal can recognise behaviour, the project needs workloads that
produce *recognisable, reproducible behavioural patterns* — including the **shape** of a
file-processing pipeline (discover → read → transform → write → rename). Those workloads are
named after the pattern they imitate (for example `sandbox_ransom_seq`, `sandbox_scanner_metadata`).
The names describe the **benchmark behaviour being simulated**, not any real capability.

## Safety guarantees (plain language)

The "security-like" workloads under `VM_executables_phase2/security_like_safe/` are
**behavioural-pattern simulators only**. They:

- perform **no real encryption** — the only transform is a **reversible XOR** with a fixed key,
  fully invertible, which is not cryptography;
- operate **only on disposable, synthetic files** generated inside a validated **sandbox**
  directory — no real user data, no documents, no system files are read or modified;
- use **no network** of any kind — no sockets, no exfiltration, no command-and-control, no
  spreading;
- have **no persistence** — no autostart, services, scheduled tasks, or startup entries;
- have **no evasion or stealth** — no anti-debug, anti-VM, anti-analysis, hiding, or rootkit
  behaviour;
- perform **no privilege escalation** and **no credential access**;
- are **resource-capped** and **time-bounded**, and run only with the launching user's
  privileges.

Each simulation's source file additionally carries a header banner stating "This is NOT
ransomware …", and the directory itself is named `security_like_safe/`.

## What we are doing about the alert (transparency, not concealment)

We are **not** renaming or obfuscating anything to avoid scanners — that would be the wrong
response. Instead we make the benign, defensive nature of the work **explicit and visible**
(this notice, the root `README.md`, and the formal `SAFETY_MODEL.md`), and we are pursuing
clearance through the proper institutional channel. Reviewers are welcome to read the
whitelist/blacklist in `SAFETY_MODEL.md` and the bannered source under
`VM_executables_phase2/security_like_safe/`.

## Pointers

- Formal safety contract (whitelist / blacklist, enforcement): `VM_executables_phase2/docs/SAFETY_MODEL.md`
- The simulation sources (each bannered): `VM_executables_phase2/security_like_safe/`
- Project overview and components: `README.md`
