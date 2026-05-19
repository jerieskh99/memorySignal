# Snapshot Interval & Test-Duration Tuning Study

Audit + planning document for the QEMU/pmemsave capture pipeline. Goal: decide
whether the current 100 ms target interval is appropriate for Phase 2
workloads (especially MEM working-set sweeps over 1 GiB+ and SECURITY-LIKE
multi-phase tests), and produce concrete, testable next steps. No code,
config, or data files were modified by this study.

> **Status of cited files.** This study references
> `VM_executables_phase2/docs/RUNNING_PHASE2_TESTS.md` per the task brief.
> That file does not yet exist in the tree (only `IMPLEMENTATION_NOTES.md`,
> `SAFETY_MODEL.md`, `SMOKE_TEST_RESULTS.md`, `TEST_CATALOG.md` are present).
> The pieces of run-time guidance normally collected in such a "running"
> document are covered here by inference from the test catalog, the README,
> and the Phase 2 plan; once that document is written, this study should be
> revisited to keep them aligned.

---

## 0. Correction notice and the three time axes (added in revision)

> **Third-pass correction (after Experiment 1 Run 2 on `pcrserral`, 2026-05-20).**
> `bc` has been installed on the capture host. Re-running Experiment 1
> measured `guest_dt = 125 ms ± 5 ms` at `intervalMsec = 100` — Axis A is
> now stationary and matches configuration (target + 25 ms unavoidable
> bash bookkeeping). The per-family `intervalMsec` recommendations in
> this study are now **valid for Δt_frame interpretation**. However Run 2
> also revealed (a) consumer-driven backpressure saturates the queue
> within ~20 snapshots and (b) suspend latency is wildly non-stationary
> (0.075–80 s range, dominated by libvirtd/disk contention with the
> concurrent consumer). These affect host-side throughput, not Δt_frame.
> The wall-clock cost table in Section 3a′ should be re-derived from
> measured run-2 numbers (mean host_dt ≈ 5–9 s, not the 2.5 s planning
> estimate) before adopting any per-family wall-clock budget. See
> [`TIMING_EXPERIMENT_1_CONCLUSIONS.md`](./TIMING_EXPERIMENT_1_CONCLUSIONS.md)
> for full data.

> **What changed.** An earlier revision of this study conflated *host
> wall-clock cost per snapshot* with *analysis frame spacing in guest time*
> and concluded that the effective Δt for spectral analysis was 2.5–5.5 s.
> That conclusion was incorrect. Re-reading the producer
> (`capture_producer_qemu_pmemsave.sh`) line by line shows that the
> configured `intervalMsec` is the **guest-running gap between snapshots**,
> not the host wall-clock period. The VM is paused only across the
> `pmemsave` window itself, so guest workload time only advances during
> the `sleep intervalMsec/1000` step (line 161) — which is between
> `virsh resume` (line 154) and the next iteration's `virsh suspend`
> (line 92).
>
> The sections below have been rewritten to reflect the correct timing
> model. Anything stated in this study about "effective Δt" should be
> read as **host wall-clock cost per snapshot** (a throughput metric);
> the **frame spacing seen by the analyzer** is approximately
> `intervalMsec` plus small per-loop bookkeeping while the VM is running.
> See [SNAPSHOT_INTERVAL_QA.md](./SNAPSHOT_INTERVAL_QA.md) for an
> explanatory Q&A.

The capture pipeline has **three** distinct time axes; the original draft
collapsed them into one and produced misleading conclusions.

| Axis                              | Definition                                                                                                   | Set by                                                                                          |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------- |
| **A. Guest-running interval**     | Wall-clock gap between consecutive snapshots **measured by the guest's own monotonic clock** (i.e., excluding time when the VM is paused). | The `sleep intervalMsec/1000` step in the producer loop, post-resume. With `intervalMsec=100`, this is ≈ 100 ms plus per-loop bookkeeping. |
| **B. Host wall-clock per snapshot** | Wall-clock time on the host required to produce one frame: backpressure check + suspend + state-poll + pmemsave + 0.5 s flush + enqueue + resume + state-poll + `sleep intervalMsec`. | Dominated by the pmemsave I/O cost — for 1 GiB RAM on SSD, on the order of 2.5–5.5 s. This is the throughput of the experiment, not its sampling resolution. |
| **C. Analysis frame spacing (Δt_frame)** | The Δt the analyzer should use to interpret cepstral peak indices, MSC frequencies, and PLV time axes. | **Axis A**, not Axis B. The deltas computed by the consumer measure guest memory *change-per-guest-running-interval*; the pmemsave-induced pause does not advance guest time, so it does not introduce extra "silence" in the analysis frame. |

A fourth derived axis is worth naming for the duration discussion:

| Axis                              | Definition                                                                                                   | Set by                                                                                          |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------- |
| **D. Experiment throughput**      | Wall-clock budget for collecting `N` frames. `N × axis-B`. This is what makes the pipeline "feel slow." | Axis B (pmemsave cost), scaled by snapshot count.                                               |

**Why the distinction matters.** A guest workload that sleeps 100 ms inside
the VM (`nanosleep(100ms)`) reads its `CLOCK_MONOTONIC` and sees ~100 ms
elapsed. When the VM is paused during pmemsave, guest `CLOCK_MONOTONIC`
freezes; on resume it continues from where it stopped. So the workload's
own perception of inter-snapshot time matches Axis A, not Axis B. The
delta the analyzer computes between two consecutive memory snapshots
reflects what changed during that ~100 ms of *guest execution*, not the
~2.5–5.5 s of host wall-clock that elapsed while collecting the pair.

Spectral interpretation: a cepstral peak at frame index `k` corresponds to
a rhythm of period `k × Δt_frame = k × axis-A`, *not* `k × axis-B`. Phase 1
spectral results are calibrated to ~100 ms guest time, not ~3 s host
wall-clock, and remain valid under that interpretation.

---

## 1. Current capture flow and where the interval is configured

**Configured value.** `VM_sampler/VM_Capture_QEMU/config_qemu_upc.json`:

```json
"ramSizeMb":    1024,
"intervalMsec": 100,
"streaming": {
    "enabled": true,
    "minFramesForStreaming": 128,
    "deltaMetric": "cosine"
},
"rawRetention": {
    "rawMetrics": {
        "windowSize": 128,
        "stepSize":   64
    }
}
```

The `intervalMsec` value flows into the producer; `windowSize` / `stepSize`
flow into the offline metrics. Both default to the Phase 1 canonical pair
(`128, 64`).

**Producer loop** (`capture_producer_qemu_pmemsave.sh`):

```text
loop forever:
    check backpressure (queue size >= maxPendingJobs → sleep)
    virsh suspend <domain>                    # ~0.5 s settle
    wait until domstate == paused
    virsh qemu-monitor-command pmemsave 0..ramSizeBytes → newImage
    virsh resume <domain>
    enqueue (prev, new) job for consumer
    sleep intervalMsec / 1000
```

**Consumer loop** (`capture_consumer_qemu.sh`): per (prev, new) pair, runs the
Rust `live_delta_calc` binary, appends a 1-D delta-feature column to a
long-lived `run_matrix.npy`, optionally invokes the streaming metrics
pipeline (`coherence_temp_spec_stability.streaming_metrics`) and writes
results once at least `minFramesForStreaming = 128` columns have been
accumulated.

**Offline metrics** (`offline_step_metrics.py`):

- `window_size` / `step_size` come from CLI flags (`--window-size 128
  --step-size 64`) or from `config.rawRetention.rawMetrics`.
- Per-segment MSC/Cepstrum/PLV are computed on each sliding window of size
  `window_size`, hop `step_size`.
- The constant `_MIN_WINDOWS_PER_SEGMENT_DEFAULT = 50` (offline_step_metrics.py:246)
  sets the *minimum* number of windows the offline analyzer expects to see in
  any segment it scores. With `window_size=128`, `step_size=64`, that demands
  `min_frames = (50 − 1) * 64 + 128 = 3264` snapshots per scored segment.

**Where 100 ms enters the math (corrected).** The producer aims for one
snapshot per `intervalMsec` of *guest-running time*. Tracing the loop:

```
SUSPEND (line 92)  →  pmemsave (line 106, VM frozen)  →  sleep 0.5 (flush)
  → enqueue  → RESUME (line 154)  → sleep intervalMsec/1000 (line 161)
  → SUSPEND (next iteration) → ...
```

Steps between `SUSPEND` and `RESUME` happen with the VM paused; the guest's
monotonic clock does not advance. The `sleep intervalMsec/1000` happens
*after* `RESUME` and is the **only** part of the loop where guest time
moves. Therefore:

1. **Axis-A (guest-running interval) ≈ `intervalMsec`** (≈ 100 ms by
   default), plus a small bookkeeping overhead for backpressure check,
   `find` queue listing, and the bash interpreter itself between resume
   and the next suspend. That overhead is empirically a few ms — small
   compared to 100 ms.
2. **Axis-B (host wall-clock per snapshot) ≈ 2.5–5.5 s** for 1 GiB on
   commodity SSD. This is the production rate, *not* the analyzer's
   sampling rate. It is what limits experiment throughput.

100 ms is therefore *not* an upper-bound rate target on the wrong axis — it
is the configured **guest-time** target, which the producer honors.
The Axis-B cost is independent of `intervalMsec`: changing `intervalMsec`
shrinks or stretches the *guest-time* gap between snapshots, but does not
make any individual pmemsave dump cheaper.

---

## 2. Why a 100 ms interval may be slow with a 1 GiB VM image

A pmemsave snapshot is a synchronous *dump of the entire VM RAM* to a file on
the host, with the VM paused for the dump's duration. With `ramSizeMb=1024`:

| Sub-step                              | Typical cost (1 GiB image)              |
| ------------------------------------- | --------------------------------------- |
| `virsh suspend` + domstate poll       | ~0.5 s (`sleep 0.5` then poll @ 200 ms) |
| `pmemsave` write of 1 GiB             | 1–4 s, depending on disk (SSD vs HDD)   |
| Producer's `sleep 0.5` after suspend  | 0.5 s                                   |
| `virsh resume` + state poll           | ~0.5 s                                  |
| `sleep intervalMsec/1000`             | 0.1 s (the configured target)           |
| **Net wall-clock per snapshot**       | **≈ 2.5 – 5.5 s** (effective Δt)        |

Consequences (corrected):

1. **Experiment throughput is slow.** Each snapshot costs ~2.5–5.5 s of
   host wall-clock; collecting a 600-frame guest run (60 s at 100 ms
   guest-time interval) takes ~25–55 minutes of host time. This is a
   throughput/cost problem, not a sampling-rate problem.
2. **Guest CLOCK_MONOTONIC freezes during pause, but the analysis is
   unaffected.** QEMU pauses the VCPUs and pauses guest TSC during
   `virsh suspend`; guest `clock_gettime(CLOCK_MONOTONIC)` does not
   advance. From inside the VM, the pmemsave gap is invisible — the
   workload thread sleeps for 100 ms of guest time, runs whatever it
   does, the VM gets paused, then resumes, and the workload continues
   as if nothing happened. The pair `(snapshot_n, snapshot_n+1)`
   therefore brackets ≈ `intervalMsec` of *guest activity*, which is
   exactly what the delta should measure.
3. **Real-world wall-clock rhythms partially distort.** Anything driven
   by host wall-clock (e.g. a packet arriving on a host network, an NTP
   sync from the host) is delayed by the pause. Anything driven by
   guest-internal timers (sqlite checkpoint cadence in guest seconds,
   fsync writeback in guest seconds, in-workload `nanosleep`) is
   unaffected because guest time freezes too. Most Phase 2 workloads
   are guest-internal-clock-driven, so the spectral signatures they
   produce are still calibrated to `intervalMsec` of guest time.
4. **Raw I/O pressure on the host.** Each 1 GiB pmemsave is a 1 GiB
   sequential write. At ~0.3 Hz wall-clock (the actual production
   rate) that is still ~300 MiB/s sustained, plus the consumer
   re-reading prev + new for the delta. The disk I/O budget is the
   single biggest tuning lever for throughput.
5. **Backpressure introduces gaps in *both* time axes.**
   `backpressure.maxPendingJobs = 20`. When the queue saturates, the
   producer sleeps for `sleepOnBackpressureSeconds = 1` of **host
   wall-clock**, but during that sleep the VM is *not* suspended —
   the workload continues running. So backpressure inflates the
   guest-time gap from `intervalMsec` to whatever the wait was. This
   is the one mechanism that *does* corrupt the guest-time Δt, and
   it is the most important thing to instrument.
6. **Cepstrum / FFT semantics still depend on a stable Δt_frame.** If
   backpressure (or any other transient) inflates Axis-A on individual
   frames, the analyzer treats them as evenly spaced and aliases the
   true rhythm. Measuring Axis-A per frame is the *first* required
   instrumentation step.

A 100 ms target on a 1 GiB image is therefore not slow in the
**sampling-rate** sense — the analyzer gets a clean 100 ms guest-time
frame spacing as long as backpressure does not fire. It is slow in the
**throughput** sense: each guest second of data costs ~25–55 host
seconds to produce, and a 5-minute run takes 2+ hours of wall clock.
Tuning `intervalMsec` upward (e.g. 1 s) does not make any individual
snapshot cheaper, but does reduce the snapshot count needed for the
same *guest-time* run (because there are fewer snapshots per guest
second), and therefore cuts the wall-clock cost in proportion to the
interval increase.

---

## 3. Interval × duration sizing tables

> **Reading the tables (post-correction).** Throughout this section,
> "duration" means **guest-time duration** of the captured run — the
> amount of guest execution the analyzer ends up looking at. Snapshot
> count and window count are determined entirely by `intervalMsec`
> (Axis A) and guest-time duration. Host wall-clock cost (Axis B/D) is
> reported separately so the experiment-throughput vs analysis-fidelity
> trade-off is visible.

The following tables assume:

- `Δt_frame` = analysis frame spacing ≈ `intervalMsec` (Axis A). This is
  what the analyzer's spectral / window machinery sees.
- `T_guest` = guest-time duration of the run. Snapshot count
  `N_snap = T_guest / Δt_frame`.
- `wallclock_per_snapshot` (Axis B) ≈ 2.5 s for 1 GiB on commodity SSD.
  This is a planning estimate; the pilot in Section 6 measures it.
- Window count uses Phase 1 canonical `window=128, hop=64`:
  `N_windows = floor((N_snap − 128) / 64) + 1`.

### 3a. Snapshot count (per guest-time duration `T_guest`)

Snapshot count = `T_guest / intervalMsec`. This is also frame count `T`
into the analyzer.

|             | 1 min | 2 min | 5 min   | 10 min  |
| ----------- | ----: | ----: | ------: | ------: |
| 100 ms      |   600 | 1200  |   3000  |   6000  |
| 200 ms      |   300 |  600  |   1500  |   3000  |
| 250 ms      |   240 |  480  |   1200  |   2400  |
| 500 ms      |   120 |  240  |    600  |   1200  |
| 1 s         |    60 |  120  |    300  |    600  |

### 3a′. Wall-clock cost (host minutes) to collect the table above

This is **Axis D**: `snapshot_count × wallclock_per_snapshot`. With
`wallclock_per_snapshot ≈ 2.5 s` (1 GiB, SSD):

|             | 1 min guest | 2 min guest | 5 min guest | 10 min guest |
| ----------- | ----------: | ----------: | ----------: | -----------: |
| 100 ms      |    25 min   |    50 min   |   2 h 5 m   |   4 h 10 m   |
| 200 ms      |  12.5 min   |    25 min   |   62.5 min |   2 h 5 m    |
| 250 ms      |    10 min   |    20 min   |    50 min  |   100 min    |
| 500 ms      |     5 min   |    10 min   |    25 min  |    50 min    |
| 1 s         |   2.5 min   |     5 min   |  12.5 min  |    25 min    |

So `intervalMsec` is a knob for **trading temporal resolution against
throughput**: at the same guest-time duration, a 10× larger interval
needs 10× fewer snapshots and finishes in 10× less wall-clock — but
gives the analyzer 10× fewer frames per guest second.

### 3b. Raw memory exposure (sum of pmemsave bytes written to disk)

|             | 1 min  | 2 min  | 5 min   | 10 min  |
| ----------- | -----: | -----: | ------: | ------: |
| 100 ms      | 600 GiB | 1.2 TiB | 3.0 TiB | 6.0 TiB |
| 200 ms      | 300 GiB | 600 GiB | 1.5 TiB | 3.0 TiB |
| 250 ms      | 240 GiB | 480 GiB | 1.2 TiB | 2.4 TiB |
| 500 ms      | 120 GiB | 240 GiB | 600 GiB | 1.2 TiB |
| 1 s         |  60 GiB | 120 GiB | 300 GiB | 600 GiB |

(Each row = snapshot count × 1 GiB. This is producer-side writes only; the
consumer reads both prev + new, doubling the I/O traffic on the same path.
These figures are real — the snapshot count is set by Axis A, and each
snapshot really does write 1 GiB regardless of how slow the host is.)

### 3c. Number of (128, 64) windows (per guest-time duration)

Using `N_windows = floor((N_snap − 128) / 64) + 1`; values clipped to 0
when the trace is too short. `N_snap` comes from Table 3a.

|             | 1 min | 2 min | 5 min | 10 min |
| ----------- | ----: | ----: | ----: | -----: |
| 100 ms      |     8 |    17 |    45 |     92 |
| 200 ms      |     3 |     8 |    22 |     45 |
| 250 ms      |     2 |     6 |    17 |     35 |
| 500 ms      |     0 |     2 |     8 |     17 |
| 1 s         |     0 |     0 |     3 |      8 |

These are the **actual** window counts available to the analyzer given a
specified guest-time duration. The 100 ms × 5 min cell really does yield
45 windows; the wall-clock cost of producing those 45 windows is 12.5
minutes of host time (Table 3a′), not 5 minutes — but the analyzer still
sees 45 windows of 12.8 s guest activity each.

> **Correction note.** A prior revision of this table inserted a second
> "Realistic effective Δt" sub-table claiming that almost no 10-minute
> run yields a single window. That sub-table assumed `Δt_frame ≈ 3 s`,
> i.e. it confused Axis B with Axis A. It is removed. The correct
> reading is the table above: window count is set by guest-time
> duration and `intervalMsec`, not by wall-clock cost.

### 3d. Feasibility for k = 2 / 4 / 8 segmentation

A k-segment segmentation needs ≥ k × `min_windows_per_segment` windows
across the run. With the offline default of 50 windows per segment, the
minimum frame count `T_min = (50 − 1) × 64 + 128 = 3264` per segment, so
`T_total = k × 3264 − (k−1) × 0 ≈ k × 3264` frames (lower bound, assuming
no overlap between segments).

| k | min total frames | needed guest duration @ Δt = 100 ms | needed guest duration @ Δt = 500 ms | needed guest duration @ Δt = 1 s | wall-clock cost @ Δt = 100 ms | wall-clock cost @ Δt = 1 s |
|--:|-----------------:|------------------------------------:|------------------------------------:|---------------------------------:|------------------------------:|---------------------------:|
| 2 | 6 528            | 10.9 min                             | 54.4 min                            | 108.8 min                        | ≈ 4.5 hr                       | ≈ 45 min                   |
| 4 | 13 056           | 21.8 min                             | 108.8 min                           | 3.6 hr                           | ≈ 9 hr                         | ≈ 90 min                   |
| 8 | 26 112           | 43.5 min                             | 3.6 hr                              | 7.3 hr                           | ≈ 18 hr                        | ≈ 3 hr                     |

The trade-off (corrected): a fast 100 ms interval *is* feasible for
k = 2 segmentation in 10.9 minutes of **guest time**, but the host pays
~4.5 hours to produce it. A 1 s interval needs 108.8 min of guest time
but only 45 min of wall-clock. Either is technically possible; the
question is throughput, not whether the analyzer can window the data.

|       Family     | k=2 feasible? | k=4? | k=8? |
| ---------------- | :-----------: | :--: | :--: |
| MEM single-phase | N/A           | N/A  | N/A  |
| APP-REALISTIC    | yes (default 300 s guest is short of k=2 at 100 ms; lengthen or use larger interval) | borderline | impractical |
| SECURITY-LIKE batched (default ~600 s) | yes at any interval ≤ 500 ms | borderline | requires very long runs |
| SECURITY-LIKE seq (per-file rhythm) | yes; per-file phases need Δt_frame ≤ per-phase guest time / 2 | inherits | inherits |
| METHODOLOGY (paired) | inherits child | inherits child | inherits child |

So multi-segment analysis is **not** blocked by the pmemsave cost — it
is blocked by **experiment wall-clock budget**. The earlier conclusion
that the pipeline "cannot support multi-segment analysis" was wrong; the
correct statement is that *running enough hours of pipeline* is the
binding constraint, plus the per-test guest duration in the Phase 2 plan
may need to be lengthened (or `_MIN_WINDOWS_PER_SEGMENT_DEFAULT` lowered
with great care) to satisfy k ≥ 2.

---

## 4. Wall-clock meaning of window=128 / hop=64 at each interval

The window length determines the lowest-frequency rhythm the cepstral and
MSC features can resolve in a single segment; the hop sets the temporal
resolution of the sliding statistics. Both must be interpreted in
**guest time** (Axis A / Δt_frame), because the deltas the analyzer
consumes were computed across guest-time intervals.

| `intervalMsec` | Δt_frame | Window in guest time (128 × Δt) | Hop in guest time (64 × Δt) | Min phase resolvable as cepstral peak | Wall-clock to collect 1 window |
| -------------: | :------: | :-----------------------------: | :-------------------------: | :-----------------------------------: | :----------------------------: |
| 100 ms         | 0.1 s    |          12.8 s                 |          6.4 s              | ≈ 0.6 Hz (1.6 s period)               | ≈ 5.3 min (128 × 2.5 s)        |
| 200 ms         | 0.2 s    |          25.6 s                 |         12.8 s              | ≈ 0.3 Hz (3.1 s period)               | ≈ 5.3 min                      |
| 250 ms         | 0.25 s   |          32.0 s                 |         16.0 s              | ≈ 0.25 Hz (4.0 s period)              | ≈ 5.3 min                      |
| 500 ms         | 0.5 s    |          64.0 s                 |         32.0 s              | ≈ 0.125 Hz (8.0 s period)             | ≈ 5.3 min                      |
| 1 s            | 1.0 s    |         128.0 s                 |         64.0 s              | ≈ 0.0625 Hz (16 s period)             | ≈ 5.3 min                      |

Two takeaways:

1. **The window covers different *guest-time* durations depending on
   `intervalMsec`.** At 100 ms the window is 12.8 s of guest activity; at
   1 s it is 128 s. Whether 12.8 s or 128 s is the right "look" depends
   on the rhythm of interest (Section 5).
2. **Wall-clock cost to collect one window is roughly constant** (just
   `128 × wallclock_per_snapshot`) and dominated by pmemsave, not by
   `intervalMsec`. Cutting `intervalMsec` from 100 ms → 1 s does **not**
   reduce the wall-clock cost per window — it only reduces wall-clock
   cost per **guest-second** (because each window now covers 10× more
   guest activity, so the run needs fewer windows to cover the same
   guest duration).

> **Correction note.** A prior revision claimed "a 128-frame window
> covers ~6.4 minutes of *wall clock*" and that "sandbox_ransom_seq
> and batched finish before a single window closes". Both statements
> conflated wall-clock with guest time. The corrected reading is:
> sandbox_ransom_seq's per-file phases are short in *guest time*, and
> may sit below Nyquist for a chosen `intervalMsec`; this is a
> resolution problem, not a throughput problem.

---

## 5. Family-specific recommendations

The right interval depends on the dominant rhythm the test is trying to
isolate. The table below lists, per family, the rhythm of interest, the
fastest rhythm that needs to be captured (Nyquist-like 2× requirement), and
the resulting minimum sampling rate.

In this section, "Recommended `intervalMsec`" is Axis A — the guest-time
spacing between snapshots — which directly sets the analyzer's frame
spacing. Wall-clock cost (Axis D) is reported separately because it is
*independent* of `intervalMsec` per snapshot and depends only on guest
duration × wallclock_per_snapshot.

| Family / test                          | Dominant rhythm        | Fastest event of interest (in guest time) | Recommended `intervalMsec` | Notes |
| -------------------------------------- | ---------------------- | ----------------------------------------- | -------------------------- | ----- |
| MEM (single-phase steady-state) `mem_workingset_sweep_v2`, `mem_writemag_sweep_v2`, `mem_rmw_intensity_v2` | none (steady) | irrelevant | **1 s or larger** | Steady-state tests do not benefit from a fast Δt_frame. Larger interval → fewer snapshots for the same guest duration → less wall-clock cost. |
| MEM (transient) `mem_pagefault_density_v2`, `mem_mmap_traversal_v2` (msync rhythm) | first-touch decay, msync cadence | depends on `--msync-interval-ms` | ≤ msync-interval / 4 | Setting `--msync-interval-ms 250` requires `intervalMsec` ≤ 60 ms for Nyquist. The pipeline can do this in guest time (just slow in wall-clock). |
| APP-REALISTIC `app_sqlite_oltp_v2` | WAL append + checkpoint | checkpoint cadence (~10 s default) | ≤ 2 s | Checkpoint rhythm is the contribution. `intervalMsec=1000` resolves it; at `intervalMsec=2000` it sits near Nyquist. |
| APP-REALISTIC `app_compress_gzip_v2`, `app_decompress_gzip_v2`, `app_json_parse_v2`, `app_hashtable_intensive_v2` | continuous CPU+IO; two-phase trajectory | not rhythmic | 1 s adequate | Hashtable build→probe transition is one-shot; needs only enough resolution to locate it. |
| APP-REALISTIC `app_sqlite_analytical_v2` | read-heavy + temp-table writes | temp-table cadence (per-aggregate) | ≤ 1 s | Same comment as OLTP. |
| SECURITY-LIKE `sandbox_ransom_seq` | per-file 5-phase | per-file duration (varies with `--files`, `--file-size-bytes`) | ≤ per-file / 8 | At default `--files 1000 --file-size-bytes 1 MiB`, each per-file pass is short in guest time (tens of ms). For the *per-file* rhythm to be resolvable, `intervalMsec` would need to be ≤ ~10 ms — possible (huge wall-clock cost) or, more sensibly, reduce file count / increase file size so each per-file phase is several seconds of guest time. |
| SECURITY-LIKE `sandbox_ransom_batched` | five mechanism-distinct intervals | each interval ~tens of seconds in guest time at N=1000 | ≤ 1 s | Batched variant produces five coarse blocks. 1 s `intervalMsec` resolves the boundaries; sub-block detail is lost. |
| SECURITY-LIKE `sandbox_ransom_slowburn` | 1 file every 5 s | 0.2 Hz | ≤ 1 s | Easiest case: rhythm intentionally slow. `intervalMsec=1000` gives 5 frames per file. |
| SECURITY-LIKE `sandbox_ransom_selective` | discovery + 5-phase loop | inherits seq's per-file rhythm | inherits | Same per-file constraints as seq. |
| SECURITY-LIKE `sandbox_scanner_metadata` | metadata-only | per-stat is sub-ms | ≤ 1 s | Per-stat rhythm sits below any practical `intervalMsec`; aggregate metadata throughput is what we measure. |
| METHODOLOGY `mp_phase_boundary_inference` | inherited from child | inherited | match child | Detector needs ≥ k+1 windows to predict k boundaries. |
| METHODOLOGY `mp_workingset_metric_linearity` | analysis only | n/a | n/a | Analyzes prior metadata; no live capture. |
| MIXED / phasic (Phase 1 transition-stress runs, `steps_transition_stress.txt`) | abrupt step transitions | step duration | ≤ step / 8 | At 800 ms steps, `intervalMsec ≤ 100 ms` resolves transitions; with default 100 ms target each step yields 8 frames. |

The pattern (corrected): **`intervalMsec` is set by the guest-time rhythm
of interest** (Axis A). **Wall-clock cost is set independently by Axis B
(per-snapshot pmemsave cost) and Axis D (snapshot count).** The 1 GiB
pmemsave floor does **not** cap the achievable `intervalMsec` — it caps
how cheaply we can run experiments with many snapshots. The earlier
section claimed the pmemsave floor "caps the achievable rate"; this was
true only in the throughput sense, not the resolution sense.

> **Correction note.** The earlier `sandbox_ransom_seq` row claimed the
> rhythm was "impossible with full pmemsave for 1 GiB". That was based
> on confusing wall-clock per snapshot with Δt_frame. The correct
> statement is: at default `--files 1000`, per-file phases are tens of
> ms of *guest time*; at `intervalMsec=100ms` the analyzer sees a few
> frames per per-file phase, which is borderline. Lengthen the per-file
> phase (fewer files, larger files) or accept that the *per-file*
> rhythm is sub-Nyquist while the *aggregate* file-rate rhythm is
> resolvable.

---

## 6. Pilot experiment

Run a focused pilot **before** committing to a Phase 2 interval. Use the
existing `sandbox_ransom_batched` (well-defined five-phase structure) and
`mem_workingset_sweep_v2` (well-defined steady state) as the two probe
workloads.

### 6.1 Design

A 5 × 4 factorial: interval ∈ {100 ms, 250 ms, 500 ms, 1 s, 2 s} ×
**guest-time duration** ∈ {1 min, 2 min, 5 min, 10 min}, two replicates
per cell, two workloads (the two probes). Total runs: 5 × 4 × 2 × 2 =
80. Estimated wall-clock cost: each cell costs
`(guest_duration / intervalMsec) × wallclock_per_snapshot`; the heaviest
cell (100 ms × 10 min) is ~4 h 10 m, the lightest (2 s × 1 min) is
≈ 1.25 min. Total over all 80 cells ≈ 50 host-hours of workload run plus
≈ 2.7 h of inter-test idle (`120 s × 80`). Capture via `CAPTURE_MODE=1`
on `run_files_controlled.py`.

Hold constant: VM RAM (1 GiB), guest OS, host hardware, page size, all
per-test parameters (use defaults from `TEST_CATALOG.md`), random seed.

### 6.2 Measurements per run

Record per recording (CSV, one row per run). The instrumentation
distinguishes the three time axes from Section 0:

**Axis A (guest-running interval, Δt_frame):**

- `target_interval_ms` — value of `intervalMsec` in the config snapshot.
- `guest_dt_mean_s`, `guest_dt_std_s`, `guest_dt_p99_s` — the gap, in
  *guest* time, between consecutive frames. Computed from the
  `resume_t_host` and `next_suspend_t_host` timestamps below by
  subtracting the contiguous host-running window and asserting the
  guest TSC delta matches. With no backpressure events, this should
  equal `target_interval_ms` plus a small per-loop overhead.

**Axis B (host wall-clock per snapshot):**

- `host_dt_mean_s`, `host_dt_std_s`, `host_dt_p99_s` — wall-clock time
  per snapshot, suspend timestamp to next suspend timestamp.

**Per-snapshot instrumentation (new — must be added to the producer):**

- `t0_before_suspend_host_ns` — `clock_gettime(CLOCK_REALTIME)` *just
  before* `virsh suspend`.
- `t1_after_suspend_host_ns` — *just after* `virsh suspend` returns
  paused.
- `t2_pmemsave_start_host_ns` — *just before* invoking the pmemsave
  monitor command.
- `t3_pmemsave_end_host_ns` — *just after* the monitor command returns.
- `t4_before_resume_host_ns` — *just before* `virsh resume`.
- `t5_after_resume_host_ns` — *just after* domstate == running.
- (Optional) `guest_monotonic_ns_at_t5` — if the consumer can read the
  guest's `CLOCK_MONOTONIC` via an SSH probe; useful to confirm the
  guest sees no time elapse between `t0` and `t5`.

From these we derive:

- `host_dt = next_t0 − t0` (Axis B).
- `pmemsave_dt = t3 − t2`.
- `vm_pause_dt = t5 − t0` (wall-clock the VM was paused).
- `guest_run_dt = next_t0 − t5` (wall-clock the VM ran ≈ Axis A on the
  host side; equal to `sleep intervalMsec/1000` plus a few ms of bash
  bookkeeping when backpressure does not fire).

**Other measurements:**

- `actual_snapshots`: read from `run_matrix.npy` shape.
- `vm_pause_fraction = sum(vm_pause_dt) / sum(host_dt)` — fraction of
  wall-clock time the VM was paused.
- `backpressure_events`: count of "queue size ≥ maxPending" log lines.
  Important: each backpressure event inflates Axis A
  (`guest_run_dt`) by `sleepOnBackpressureSeconds`; a high count
  silently corrupts Δt_frame stationarity.
- `n_windows`: `floor((actual_snapshots − 128) / 64) + 1`.
- `n_segments_k2`, `n_segments_k4`, `n_segments_k8`: how many segments
  the segmenter actually produces.
- Per workload, the *defining metric*:
  - `sandbox_ransom_batched`: agreement F1 between ground-truth phase
    timestamps (emitted by the workload's `--phase-markers`) and
    detected boundaries via `mp_phase_boundary_inference`.
  - `mem_workingset_sweep_v2`: coefficient-of-variation (`CV`) of
    `active_page_fraction` across windows. Low CV is the goal for
    steady state.

### 6.3 Statistical analysis

For each defining metric, fit:

```
metric ~ log(target_interval_ms) + log(target_duration_s)
       + log(target_interval_ms) * log(target_duration_s)
       + workload + (1 | replicate)
```

A simple linear model is enough; the question is monotonicity, not exact
shape. Bonferroni-corrected pairwise comparisons across the five intervals
to find the *slowest* interval that still meets the acceptance criteria in
Section 7.

Also examine: residual correlation of `guest_dt_std_s` with
`backpressure_events` and with `n_segments`. The hypothesis is that
backpressure is the *only* mechanism that inflates Δt_frame (Axis A);
absent backpressure, `guest_dt_std_s` should be tiny (a few ms). If
this is confirmed, the analyzer can trust `intervalMsec` as a fixed Δt;
if not, the per-frame measured Δt must be plumbed through to the
spectral pipeline.

---

## 7. Acceptance criteria

Adopt a slower interval **only if** all of the following hold:

1. **Axis-A (Δt_frame) stationarity.**
   `guest_dt_std_s / guest_dt_mean_s < 0.10` *and*
   `guest_dt_p99_s < 1.25 × guest_dt_mean_s`. Cepstral features assume
   constant Δt_frame; large dispersion (typically driven by backpressure)
   breaks the spectral interpretation.
2. **Window count.** `n_windows ≥ 50` for the chosen guest-time duration.
   Anything less is below the existing
   `_MIN_WINDOWS_PER_SEGMENT_DEFAULT = 50` and either forces a constant
   default change *or* drops segment-level analysis for that workload.
3. **k = 2 segmentation succeeds.** For `sandbox_ransom_batched` the
   detector + ground-truth F1 ≥ 0.8 at k = 5. For steady-state tests, the
   single-segment summary has CV ≤ 0.15.
4. **No silent data loss from backpressure.**
   `backpressure_events / total_snapshots ≤ 0.01`. Above 1 %, the
   producer is gating between snapshots; the gating sleep is not a
   pmemsave-paused window — the VM is running — so it inflates Axis A,
   which directly hurts criterion (1).
5. **Throughput is acceptable for the planned dataset.** `host_dt_mean_s ×
   total_snapshots ≤ budget`. This is a separate (Axis D) criterion and
   does not affect data validity — only how long collection takes. Use
   it to rank-order interval choices among those that pass (1)–(4).
6. **Cross-family equivalence (optional).** If a single global interval
   is adopted, its acceptance must hold for *every* family in Section 5.
   If it only passes for, say, MEM steady-state but fails for batched,
   that is evidence for a per-family interval profile (Section 8).

A test that fails (1) or (4) is rejected outright; a test that fails (2)
or (3) is rejected for that workload only.

> **Correction note.** The earlier acceptance criterion 4
> ("`vm_pause_fraction ≤ 0.30` — above that, guest's wall-clock and
> host's wall-clock diverge enough that inter-snapshot guest timestamps
> are biased") was wrongly framed. Pause fraction directly indexes
> Axis D (throughput) and indexes Axis B (per-snapshot cost), but it
> **does not bias** the inter-snapshot *guest* timestamps — those
> exclude pause time by construction (QEMU freezes guest TSC).
> Replaced with the throughput criterion above.

---

## 8. Global interval vs per-family interval profile

Three options, with trade-offs:

**Option A. Keep a global interval.** Single number in
`config_qemu_upc.json`. Simplest pipeline; easiest cross-test
comparability; matches Phase 1's design choice. Cost (corrected): a
global interval that satisfies rhythmic tests will over-sample
steady-state MEM (wasted *wall-clock*, not wasted resolution), and one
that satisfies steady-state MEM will under-resolve any rhythmic
workload's guest-time rhythm.

**Option B. Per-family interval profile.** Three or four profiles
(e.g. `steady`, `phasic_fast`, `phasic_slow`, `methodology`). The
orchestrator (`run_files_controlled.py`) writes the right `intervalMsec`
into a per-step config snapshot before launching capture. Cost:
cross-test comparability becomes conditional ("with the steady profile");
analyzer must read Δt_frame from metadata, not from a global;
segment-counts diverge across families. This is what the
*resolution-vs-throughput* trade-off demands when families' rhythms
differ widely.

**Option C. Per-test interval profile.** Most flexible; most book-keeping.
Each Phase 2 binary's metadata JSON already carries the workload-side
parameters; the capture side could mirror them. Probably overkill for an
18-test catalog where most tests cluster into 3 rhythmic regimes.

**Recommendation.** Plan around **Option B**. Three profiles:

- `steady` — 1 s target interval, used for `mem_workingset_sweep_v2`,
  `mem_writemag_sweep_v2`, `mem_rmw_intensity_v2`,
  `app_sqlite_analytical_v2`, `app_compress_gzip_v2`,
  `app_decompress_gzip_v2`, `app_json_parse_v2`,
  `sandbox_scanner_metadata`, `sandbox_ransom_slowburn`.
- `phasic` — 500 ms `intervalMsec` (i.e. 500 ms of guest time per frame;
  ~5.3 min wall-clock per 128-frame window) for `sandbox_ransom_batched`,
  `sandbox_ransom_seq`, `sandbox_ransom_selective`, `app_sqlite_oltp_v2`,
  `app_hashtable_intensive_v2`, `mem_pagefault_density_v2`,
  `mem_mmap_traversal_v2`.
- `inherit` — methodology tests inherit from the child they call.

The fast 100 ms profile becomes a *Phase 1 legacy* setting, retained for
backward-compatibility runs but not used by Phase 2 unless an experiment
explicitly requires it (and accepts the trade-offs in Section 9).

The profile itself is just a number; the engineering cost is in
(a) plumbing the per-step config snapshot and (b) recording Δt in the
analyzer's metadata so cross-profile runs can be compared.

---

## 9. Risks of changing the interval (revised)

1. **Spectral feature meaning changes with `intervalMsec`.** Cepstral
   peak indices are reported in *frame* units. A change from
   `intervalMsec` = 100 ms to 1 s shifts the meaning of peak index `k`
   from `k × 100 ms` to `k × 1 s` of *guest time*. Any downstream
   consumer that does not also read Δt_frame will misinterpret peak
   positions by a factor of 10. (This risk is genuine and unchanged
   by the correction.)
2. **Phase 1 baseline incomparability.** Phase 1 results were collected
   at `intervalMsec` = 100 ms. If Phase 2 moves to a slower
   `intervalMsec`, comparing Phase 2 against Phase 1 reference rows
   requires either re-collecting Phase 1 at the new interval or
   documenting the interval-induced shift via the pilot.
3. **Short phases (in guest time) vanish under Nyquist.** Any rhythm
   shorter than `2 × intervalMsec` of *guest time* is invisible.
   `sandbox_ransom_seq` per-file phases are tens of ms of guest time
   at default parameters; at `intervalMsec` = 100 ms the per-file
   rhythm is borderline; at 1 s it is gone. (Earlier draft tied this
   to host wall-clock; the correct constraint is guest time.)
4. **Backpressure inflates Axis A and aliases the spectrum.** A
   `intervalMsec` close to the queue-drain rate makes the producer hit
   `backpressure_events` often; each hit adds
   `sleepOnBackpressureSeconds = 1` s of *guest* time to the next
   frame's spacing (because the VM keeps running during that sleep).
   This is the **only** mechanism that corrupts Δt_frame stationarity
   and is the most important thing to instrument.
5. **Inter-test idle coverage scales with `intervalMsec`.** A 120 s
   guest-time idle at `intervalMsec` = 100 ms gives 1200 frames; at
   `intervalMsec` = 1 s it gives 120 frames; at 3 s it gives 40
   (sub-window). For the "host quiet between tests" diagnostic to
   keep using the same window/hop, idle duration must scale with
   `intervalMsec`.
6. **Streaming pipeline behavior changes.** `minFramesForStreaming =
   128` in `config_qemu_upc.json` controls when streaming metrics
   begin to emit. The first emission happens after `128 × intervalMsec`
   of *guest* time — at 100 ms that is 12.8 s of guest activity
   (~5–6 min wall-clock), at 1 s that is 128 s of guest activity
   (~5–6 min wall-clock too, because wall-clock is dominated by
   pmemsave count, not `intervalMsec`). Tests with short guest
   duration emit no streaming metrics.
7. **`_MIN_WINDOWS_PER_SEGMENT_DEFAULT = 50`** is a hard floor that
   silently rejects segments. Slowing `intervalMsec` without
   lengthening *guest* duration pushes more runs below this floor.
8. **Cleanup overhead scales with snapshot count, not with
   `intervalMsec` directly.** A 6000-snapshot run at `intervalMsec` =
   100 ms (10 min guest) leaves a peak queue of ~20 × 1 GiB files
   (40 GiB live); the same 10 min guest duration at `intervalMsec` =
   1 s gives 600 snapshots and ~10 live files (10 GiB), which is more
   host-friendly. Slower `intervalMsec` always helps host I/O budget.

---

## 10. Final recommended next actions

In order of priority:

1. **Instrument the producer to measure all three time axes (Axis A,
   Axis B, derived Axis D).** Add the six per-snapshot timestamps
   listed in Section 6.2 (`t0`…`t5`) to a per-run log file. This is
   the prerequisite for every later step. Until this is done, the
   pilot cannot tell whether `intervalMsec` is being honored or
   inflated by backpressure. The plumbing already exists in the
   image-filename timestamps for `t2` (snapshot start); the other
   timestamps need to be added.
2. **Run the Section 6 pilot.** 80 runs, two workloads, five intervals,
   four durations. Outputs feed Sections 7 and 8.
3. **Adopt Option B profiles** if the pilot confirms cross-family
   divergence (highly likely). Document the per-step interval in the
   per-test metadata JSON so the analyzer can keep cross-test results
   commensurable.
4. **Reconsider the 1 GiB RAM size for throughput, not for resolution.**
   A 512 MiB or 256 MiB VM cuts the pmemsave dump time roughly in
   half / quarter, slashing **Axis B** and therefore Axis D. It does
   **not** affect `intervalMsec` (Axis A) — guest-time resolution is
   already a free parameter. Smaller VM RAM is the structural lever
   for "experiment feels slow"; `intervalMsec` is the structural
   lever for "rhythm of interest is unresolved". (Note that
   `mem_workingset_sweep_v2` defaults to `--working-set-mb 1024`,
   which would not fit in a 512 MiB VM; the sweep range is the
   binding constraint on RAM size, not the spectral analysis.)
5. **Tighten the analyzer's Δt_frame awareness.** Add `guest_dt_mean_s`,
   `guest_dt_std_s`, and `host_dt_mean_s` to every saved metric JSON,
   for both raw and streaming pipelines. Reject any segment whose
   internal `guest_dt` dispersion exceeds the Section 7 stationarity
   threshold rather than silently averaging across irregular gaps. The
   analyzer should *never* convert frames to seconds using
   `host_dt_mean_s`; that is Axis B and is the wrong axis for spectral
   interpretation.
6. **Write `VM_executables_phase2/docs/RUNNING_PHASE2_TESTS.md`** so the
   per-test runtime guidance (interval profile, duration target,
   expected windows, expected segments) lives next to the test catalog.
   This is the missing companion file referenced by the task brief.
7. **Re-baseline Phase 1 tests at the new profile.** A small re-run
   subset of Phase 1 workloads at the chosen Phase 2 interval makes
   cross-phase comparisons honest. The Phase 1 plan's confusion-matrix
   diagnostics can be re-computed cheaply.
8. **Update `_MIN_WINDOWS_PER_SEGMENT_DEFAULT`** only after the pilot:
   if a slower Δt reduces window count below 50, decide whether to (a)
   lower the threshold (and re-validate Phase 1 results survive the
   change), (b) lengthen Phase 2 test durations, or (c) reduce window
   size from 128 to 64 (and re-validate the Phase 1 cepstral / MSC
   results survive).
9. **Confirm with a reproducibility check.** Run the same workload three
   times back-to-back at the proposed interval; verify metrics align
   within Phase 1 CV thresholds. If they don't, the interval choice is
   too aggressive (or too lax) and needs another pilot pass.

No script, config, or data file has been modified by this study.
