# Memory-Signal Workload-Behaviour Research

> **Safety and scope (please read first).** This is an academic PhD research repository for
> **defensive** security work: detecting and characterising workload behaviour from a virtual
> machine's **memory signal** (measured host-side, black-box, with no guest agent). The
> workloads under `VM_executables_phase2/security_like_safe/` are **safe behavioural
> simulations** — they imitate the *shape* of ordinary file-processing workloads using a
> reversible XOR on disposable, synthetic files inside a sandbox. **They are not malware:** no
> real encryption, no network, no persistence, no evasion, no access to any real user data.
> See [`RESEARCH_SAFETY_NOTICE.md`](RESEARCH_SAFETY_NOTICE.md) and the formal
> [`SAFETY_MODEL.md`](VM_executables_phase2/docs/SAFETY_MODEL.md). Security-sounding terms such
> as "ransom" appear only as **names of these benign simulations**; an automated keyword alert
> on them is a false positive.

## About this repository

The thesis studies whether a single host-side **memory signal** — the Active Page Fraction
(APF), the fraction of guest memory pages that change between consecutive snapshots — can
**characterise** (identify the kind of) and **detect** workload behaviour running inside a VM,
without any guest agent. Controlled workload generators drive the guest with reproducible
memory- and I/O-access patterns; the host sampler captures the signal; offline analysis builds
and evaluates the models. All "security-like" workloads are safe simulations as described above
and in the safety documents linked at the top.

## Correctness-validation utility

This component contains small, focused utilities used to verify the correctness of the page-wise differential encoding used in the volatile memory signal model.

The primary goal of these checks is to ensure that the **magnitude component** (Hamming-based bit flip count) is **consistent, symmetric, and complete** across incremental memory snapshots.

## Files

- **flip_accounting_check.py** — Verifies that bit-flip accounting is correct:
  - Checks that forward and backward comparisons are symmetric.
  - Confirms that the sum of changes matches expected flip counts.

## Requirements

Create and activate the virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate 
```

Then install dependencies (after generating requirements.txt):

```bash
pip install -r requirements.txt
```

## Usage

```bash
python flip_accounting_check.py --prev <path_to_previous_snapshot> --curr <path_to_current_snapshot>
```

Snapshots must be page-aligned and of the same size.

## Purpose

This repository is used **only for correctness validation** — not full analysis, feature extraction, or modeling. It allows verifying that the differential signal representation behaves exactly as intended before using it in larger pipelines.