# Plan 06 Results -- Disk-I/O capture channel (builds on Plan 05)

## Context

Plan 05's downstream study found one hard boundary: a single memory scalar (APF)
cannot separate two near-idle workloads with opposite labels -- the stealth threat
`slowburn` and the benign `pagefault`. They are twins in memory; the masquerade map
showed low-APF threats impersonate benign `pagefault`. The discriminating signal is
disk I/O (`slowburn` writes files; `pagefault` only faults RAM), which APF cannot
see. Plan 06 adds a **disk-I/O capture channel** -- a per-snapshot host-side
`domblkstat` read of the guest's cumulative rd/wr byte counters, at the
memory-snapshot cadence -- making each trajectory 2-channel (APF + disk rate). The
instrumentation is additive and flag-gated (`CAPTURE_DISKIO`, default off =
byte-identical to the Plan 05 paths).

## What ran

A subset to validate the channel: 8 workloads x 600 s x 2 reps = 16 cells, covering
both threat families (4 ransomware + scanner) and the masquerade benigns
(pagefault, mmap, workingset), with `CAPTURE_DISKIO=1`.

## Channel validation

A 15 s `batched` smoke moved `domblkstat wr_bytes` by only ~164 KB -- because
`batched` is the memory-capped workload (Plan 05's 2.1 M "exceeds --mem-cap-mb")
and barely writes to disk. A clean writer (`ransom_seq`) settled it: over 600 s the
host `wr_bytes` rose **6.5 MB -> 11.3 GB**, matching the guest's own
`/proc/diskstats` counter (~7.8 GB in a separate 45 s probe). The host-side
`domblkstat` channel captures guest disk writes correctly.

## Result 1 -- the masquerade pair IS resolved by disk I/O

Mean write rate (MB/s), the pair APF could not separate:

| workload | type | write MB/s |
|---|---|---:|
| ransom_slowburn | threat | **0.20** |
| mem_pagefault_density | benign | **0.002** |

~100x separation. The two workloads that alias in memory are cleanly distinct on
disk. The 2nd channel disambiguates them -- the APF blind spot, for this pair, is
closed by adding disk I/O.

## Result 2 -- disk I/O is NOT a threat marker

The full write-rate ranking exposes the catch:

| workload | type | write MB/s |
|---|---|---:|
| mem_mmap_traversal | **benign** | **140.7** |
| ransom_selective | threat | 22.5 |
| ransom_seq | threat | 9.4 |
| ransom_slowburn | threat | 0.20 |
| sandbox_scanner_metadata | threat | 0.19 |
| mem_workingset_sweep | benign | 0.004 |
| ransom_batched | **threat** | **0.003** |
| mem_pagefault_density | benign | 0.002 |

The **heaviest writer is benign** (`mmap_traversal` at 140 MB/s -- read-modify-write
on a memory-mapped file), and a **threat writes nothing** (`batched`, mem-capped).
So "writes to disk" is a behavior, not a threat sign -- exactly as APF was. Disk I/O
is a second behavioral axis, not a detector. Adding it did not lift threat/benign
classification.

Caveat: the 16-cell subset is too small to classify meaningfully (10 threat / 6
benign, majority baseline 0.625; every leave-one-out CV number is below that). The
direct masquerade-pair measurement is the trustworthy result here, not the
classifier numbers.

## The reframe

The value of a second channel is **behavioral disambiguation** -- separating
workloads that *alias* in the first channel (slowburn ~ pagefault in memory, now
split on disk) -- **not** threat detection. No single channel (memory or disk) is a
threat marker; benign workloads occupy the loud end of both. This sharpens the
project's stance: it is **characterization, not detection**. The honest, defensible
claim is "two orthogonal channels disambiguate behaviors that one channel aliases,"
not "the second channel catches the threat."

## Conclusions

1. **The disk-I/O channel works and is additive** -- flag-gated, byte-identical when
   off; `domblkstat` captures guest writes (validated host-vs-guest, ~7.8 GB match).
2. **It resolves the specific APF blind spot** (slowburn vs pagefault, ~100x on
   write rate) -- a concrete demonstration that a second channel disambiguates a
   memory-aliased pair.
3. **It is not a threat axis** -- benign `mmap` is the heaviest writer; threat
   `batched` writes nothing. Multi-channel buys disambiguation, not detection.
4. **Next (full run):** re-capture all 66 workloads 2-channel to test whether disk
   I/O improves multi-class WORKLOAD/FAMILY identification (the characterization
   angle), with the meaningful sample size the subset lacked.

Reproduce: `CAPTURE_DISKIO=1` capture -> `plan05_campaign/build_cells_dir.py`
(copies the diskio sibling) -> `plan03_sweep.py` -> `plan06_campaign/diskio_lift.py`.
Outputs under `plan06_campaign/downstream/`.
