# Experiment 1 — Timing Instrumentation: Conclusions

**Run 1 ID:** `20260519T000307Z_7428bb09` (pre-bc-fix, 3 snapshots, 15 s)
**Run 2 ID:** `20260519T222407Z_c169d947` (post-bc-fix, 19 snapshots + 85 backpressure events, 298 s)
**Host:** `pcrserral` (Linux capture host)
**Date analysed:** 2026-05-20

## Headline after Run 2

The `bc` fix worked. The configured `intervalMsec = 100` is now honoured: measured
`guest_dt = 125 ms ± 5 ms` (CV = 4.2 %). Analysis frame spacing is correct and
stationary. The original 15× mis-calibration is gone.

But Run 2 surfaced two new problems:

1. **Backpressure became binding.** The producer hit the queue cap (20) within 19
   snapshots, then spent 85 seconds in `sleep 1` waiting. The consumer cannot
   keep up with one 1 GiB pmemsave per ~2 s.
2. **Suspend latency is wildly non-stationary** once the consumer competes for
   disk and libvirtd. Range 0.075 s – **80.3 s** (single spike at seq=4).
   Most cycles 0.2–2 s. Pmemsave throughput also halved under contention
   (0.77 s → 1.5–3.9 s).

These do not corrupt **Δt_frame** (still stationary). They do corrupt
**experiment throughput predictability** and may cause irregular gaps in
**real** captures where consumer + producer run together. Plan 02 should
address consumer throughput before any per-family `intervalMsec` recommendation
is finalized.

## Executive summary (Run 1, kept for historical record)

The instrumentation worked. The data revealed two surprises and one confirmed
behaviour. The headline finding is that the configured `intervalMsec=100`
**did not** produce 100 ms of guest-running time between snapshots. The
measured guest-running gap is **~6.6 ms**, not 100 ms. The most plausible
explanation is that `bc` is not installed on the capture host, so the
producer's `sleep "$(echo "scale=3; intervalMsec/1000" | bc)"` falls back to
`sleep "$(( intervalMsec/1000 ))"` = `sleep 0`.

The second surprise is that `t1 − t0` (the "suspend" interval as instrumented)
grew from 0.97 s on the first snapshot to 5.80 s on the third. The producer's
`wait_state "paused"` polling loop is almost certainly contributing most of
this latency; the actual QEMU VCPU pause is much faster. Our `t1` therefore
measures "host wall-clock until libvirt reports paused", not "host wall-clock
until VCPU is actually paused".

The confirmed behaviour: `pmemsave` itself takes ~0.75 s for 1 GiB on this
host — faster than the 2.5 s planning estimate used in the original tuning
study.

## Measured configuration

| Field | Value | Source |
| --- | --- | --- |
| `intervalMsec` (config) | 100 | `config_timing_experiment.json` line 8 |
| `ramSizeMb` | 1024 | `config_timing_experiment.json` line 4 |
| `vm_domain` | `Kali Jeries` | `config_timing_experiment.json` line 2 |
| `imageDir` | `/var/lib/libvirt/qemu/dump` | line 5 |
| Backpressure max | 20 | line 41 |
| Poll interval (paused/running) | 200 ms | `vmStatePolling.pollIntervalMs` line 45 |
| Snapshots captured | 3 (seq 0, 1, 2) | `snapshot_timings.jsonl` |
| Backpressure events | 0 | grep `seq:-1` in JSONL |
| Run duration on host wall-clock | ~15 s (t0 of seq 0 → t5 of seq 2) | derived |

The orchestrator's `--duration` was 60 s, but the JSONL captured only 3
complete records. The producer was either killed mid-snapshot-4 by the
orchestrator's graceful shutdown, or the run ended before completing more
snapshots. `producer.log` shows a fourth `pmemsave` starting at
`memory_dump-20260519020529165.raw` (line 41), which never produced a JSONL
record — consistent with shutdown mid-snapshot.

## Measured timing table

All values in seconds. `host_dt_n` is `t0[n+1] − t0[n]`. `guest_run_n` is
`t0[n+1] − t5[n]` (gap during which the VM was running between snapshots).

| seq | suspend = t1−t0 | pmemsave = t3−t2 | flush+chown = t4−t3 | resume = t5−t4 | vm_pause = t5−t0 | host_dt = next_t0−t0 | guest_run = next_t0−t5 |
| --: | ---------------: | ----------------: | -------------------: | --------------: | ----------------: | ---------------------: | ----------------------: |
|   0 |           0.974 |             0.749 |                0.518 |           0.035 |             2.278 |                  2.285 |                  **0.0066** |
|   1 |           4.203 |             0.770 |                0.522 |           0.036 |             5.533 |                  5.540 |                  **0.0067** |
|   2 |           5.795 |             0.747 |                0.522 |           0.039 |             7.105 |                       — |                       — |

Aggregates (n=3 for component, n=2 for inter-snapshot deltas):

| Quantity | Mean | Std |
| --- | --: | --: |
| suspend (t1−t0) | 3.657 s | 2.005 s |
| pmemsave (t3−t2) | 0.755 s | 0.010 s |
| flush+chown (t4−t3) | 0.521 s | 0.002 s |
| resume (t5−t4) | 0.037 s | 0.002 s |
| vm_pause (t5−t0) | 4.972 s | 2.011 s |
| host_dt (next_t0−t0) | 3.912 s | 1.628 s |
| guest_run (next_t0−t5) | **0.0066 s** | 0.0001 s |
| vm_pause_fraction (Σ vm_pause / Σ host_dt) | **~0.998** | — |

The VM is paused **~99.8 % of host wall-clock time** during this run. The
guest spends ~6.6 ms running between every snapshot.

## Guest-running interval conclusion

**`intervalMsec` is not producing the expected guest-running interval.**

The measured gap between `t5_after_resume` of snapshot N and
`t0_before_suspend` of snapshot N+1 is **6.6 ms**, not 100 ms.

That gap should equal:

```
emit_timing JSONL append  (~ms)
sleep intervalMsec/1000   (intended 100 ms)
backpressure check        (2× find forks, ~ms)
timestamp + path setup    (~ms)
echo statements + echo to file  (~ms)
__t0=$(ts_ns)             (1× date fork, ~ms)
```

Total expected ≈ 100 ms + ~10–20 ms of bash bookkeeping. Measured: 6.6 ms.
The 100 ms term is missing.

**Most plausible cause:** `bc` is not installed on the capture host
(`pcrserral`). The producer line at `capture_producer_qemu_pmemsave.sh:190`
reads:

```bash
if command -v bc &>/dev/null; then
  sleep "$(echo "scale=3; $intervalMsec/1000" | bc)"
else
  sleep "$(( intervalMsec / 1000 ))"
fi
```

When `bc` is absent, the fallback runs `sleep "$(( 100/1000 ))"`. Bash
integer arithmetic: `100/1000 = 0`. So the fallback executes `sleep 0` —
which is exactly what the 6.6 ms gap (script overhead only) looks like.

**Action to confirm:** on the capture host, run `command -v bc`. If empty,
the hypothesis is verified.

A secondary candidate is that `sleep` does not accept the `.100` format
that `bc` emits (no leading zero). GNU coreutils `sleep` does accept it,
but BusyBox `sleep` does not. Kali is GNU-based so this is unlikely to be
the cause, but the bc-fallback hypothesis above covers both.

## Host capture-cost conclusion

`pmemsave` itself is faster than the planning estimate:

- Measured: **0.755 s ± 0.010** for a 1 GiB dump.
- Planning estimate in the original study: 2.5 s.

So pmemsave alone delivers ~1.36 GiB/s, consistent with a modest NVMe or
fast SATA SSD. The original study's "2.5 s per snapshot" was a conservative
estimate; the real number is roughly one-third of that.

But the full per-snapshot host cost is not equal to `pmemsave` alone. It is
dominated by the **suspend-confirmation polling**:

- `t1 − t0` (configured as "suspend done") averages 3.66 s but with large
  variance (0.97 s → 4.20 s → 5.80 s).
- This interval includes:
  1. `virsh suspend` fork+exec (~10–50 ms).
  2. libvirtd RPC to QEMU.
  3. QEMU pauses VCPUs (sub-ms).
  4. `wait_state "paused"` polling loop: repeated
     `virsh -c qemu:///system domstate "$domain"` calls separated by
     `sleep 0.2`.
  5. `virsh domstate` itself may take 100s of ms on a busy libvirtd.

We cannot tell from the present log whether QEMU is slow to pause or
whether the `virsh domstate` query is the dominant cost. Both are
possible. The instrumentation logs only `t0` and `t1` — it does not log
the time inside the polling loop.

`t4 − t3` (the gap between `pmemsave` returning and `virsh resume` being
called) averages 0.522 s. This is the `sleep 0.5` "flush" plus the
`sudo chown` plus the dump-existence check. The 0.5 s flush is almost
all of it.

`t5 − t4` (resume) averages 0.037 s. Resume + first successful poll for
"running" is fast.

**Total host wall-clock per snapshot (measured):** ~4 s mean, with a
strong upward drift across the three snapshots (2.3 s → 5.5 s → 7.1 s).
We do not yet know whether the drift continues indefinitely or stabilises;
we need a longer run.

## Corrected interpretation of `intervalMsec`

The original docs claimed: *"intervalMsec is the guest-running interval
between snapshots."* That claim was correct in theory but the present
build of the producer does not deliver it on this host.

| Layer | Claim | Status after Experiment 1 |
| --- | --- | --- |
| Source code intent | `intervalMsec` sets the post-resume sleep duration. | Confirmed (script reads the config field correctly). |
| Effective behavior with `bc` available | Guest-running gap ≈ `intervalMsec` ms. | **Not tested here** — `bc` likely absent. |
| Effective behavior on the capture host as configured | Guest-running gap ≈ 6.6 ms regardless of `intervalMsec`. | **Confirmed** by this experiment. |

Until `bc` is installed (or the sleep is rewritten without `bc`),
`intervalMsec` does not control the guest-running interval on this host.
Every spectral / cepstral / segmenter conclusion that assumed
Δt_frame ≈ 0.1 s is therefore mis-calibrated.

## Implications for analysis, windowing, segmentation

Under the measured Δt_frame ≈ 6.6 ms:

| Quantity | At 100 ms (assumed) | At 6.6 ms (measured) |
| --- | --: | --: |
| Window = 128 frames in guest time | 12.8 s | **0.845 s** |
| Hop = 64 frames in guest time | 6.4 s | **0.422 s** |
| Nyquist resolvable rhythm | 0.2 s | **0.013 s** (13 ms) |
| 1 s of guest activity | 10 frames | **152 frames** |
| 1 min of guest activity | 600 frames | **~9 100 frames** |

Consequences if the current host build remains unchanged:

- **Windows cover much less guest activity than designed.** A 128-frame
  window now spans 0.85 s of guest execution, not 12.8 s. Workload rhythms
  slower than ~1 s sit outside a single window.
- **Cepstral peak indices map differently to Hz.** A peak at index 10
  means a rhythm of `10 × 6.6 ms ≈ 66 ms`, not `10 × 100 ms = 1 s`. Any
  downstream consumer that assumed 100 ms is wrong by ~15×.
- **Segment count requirements blow up.** The 50-windows-per-segment
  floor still requires 3264 frames; at 6.6 ms each that is 21.5 s of
  guest time per segment, which is fine, but the **wall-clock cost**
  also grows because more snapshots are needed per unit of guest time.
  Actually: at 6.6 ms guest-gap and ~4 s host-cost per snapshot, the
  host pays ~600 s per second of guest time — 10× slower than the
  100 ms case would be if it worked.
- **Steady-state metrics see far more deltas per second of guest
  activity.** This may either help (denser sampling) or hurt (deltas
  capture finer structure that the cepstral models were not tuned for).

The single safest mitigation: fix the sleep so `intervalMsec` actually
takes effect, then re-collect. Anything done on data captured with the
broken sleep is calibrated to a 6.6 ms frame spacing, not to whatever
`intervalMsec` the operator set.

## What we know now

1. The instrumentation works: the six host timestamps are recorded
   accurately. The producer-side patch is sound.
2. The producer's post-resume sleep is not delivering the configured
   `intervalMsec` of guest-running time on this host. Measured guest
   gap: 6.6 ms.
3. `pmemsave` itself is fast (~0.75 s for 1 GiB), faster than the
   tuning-study's planning estimate (2.5 s).
4. Suspend latency as instrumented (`t1 − t0`) is dominated by
   confirmation polling, not by the actual QEMU pause. We measured
   0.97 s → 4.20 s → 5.80 s — increasing across the three snapshots.
5. The 0.5 s "flush" sleep + `sudo chown` adds ~0.52 s per snapshot
   between `t3` (pmemsave end) and `t4` (before resume).
6. VM pause fraction is ~99.8 % in this run.

## What remains uncertain

1. **Whether `bc` is actually missing on the capture host.** Need to
   verify with `command -v bc`. If `bc` is present, a different cause
   for the missing sleep must be found.
2. **Whether the suspend-latency growth (0.97 → 4.20 → 5.80 s) continues
   or saturates.** Three snapshots is too few. A longer run is needed
   to characterise the long-tail behaviour.
3. **What fraction of `t1 − t0` is the actual QEMU pause vs `virsh
   domstate` polling latency vs libvirtd RPC.** The current
   instrumentation cannot separate them. To split it, we would need
   to log each poll iteration's start/end inside `wait_state`.
4. **Whether `pmemsave_sec ≈ 0.75 s` is stable across longer runs and
   different RAM sizes.** Three samples is a tiny basis.
5. **Whether backpressure ever fires.** Zero events in this 15-s run,
   but the run was too short for the consumer queue to backlog.
6. **What the guest workload was doing during the 6.6 ms windows.**
   The orchestrator was invoked with no `--test-command`. The producer
   ran against an idle VM. Conclusions about backpressure and pmemsave
   under load remain unmeasured.

## Recommended next actions

1. **Verify `bc` presence on the capture host.** Run `command -v bc`.
   If empty, either `apt install bc` or replace the producer's sleep
   with a bc-free form. Both are out of scope for this analysis but
   are the obvious code-change next step.
2. **Re-run Experiment 1 with a longer duration** (e.g. `--duration 300`
   ≈ 5 min host wall-clock) and a workload (`--test-command "bash
   ~/VM_executables_phase2/scripts/run_phase2_min.sh /tmp/p2"
   --ssh-target ...`). Goals:
   - confirm whether the suspend-latency growth saturates;
   - characterise pmemsave under guest load;
   - observe backpressure events.
3. **Add inner-poll instrumentation to `wait_state`** so we can split
   `t1 − t0` into "virsh suspend RPC" vs "polling overhead". The plan
   text already allows this; the patch is small (log per-iteration
   `virsh domstate` start/end).
4. **Do not adopt any tuning recommendation from the original study
   until the sleep is fixed.** All "interval × duration" tables in
   `SNAPSHOT_INTERVAL_TUNING_STUDY.md` assume the sleep works. With
   the sleep broken, the per-family `intervalMsec` recommendations
   are meaningless on this host.
5. **Do not analyse any captured run-matrix yet.** Frame spacing is
   6.6 ms, not 100 ms; cepstral indices and segmenter results would
   be off by ~15×.
6. **Treat this Experiment 1 result as evidence for the value of the
   instrumentation itself.** Without the six timestamps, the broken
   sleep would have remained invisible and the entire downstream
   pipeline would have been mis-calibrated silently. Plan 01 from
   `tuning_plans/01_instrumentation_logging_plan.md` is therefore
   load-bearing and should remain in the pipeline.

## Appendix: raw data summary

`snapshot_timings.jsonl` contents (3 lines):

```text
seq=0  t0=...4114.231266234  t1=...4115.204750272  t2=...4115.206269402
       t3=...4115.955635278  t4=...4116.473852228  t5=...4116.509101253
seq=1  t0=...4116.515681855  t1=...4120.718916393  t2=...4120.721083463
       t3=...4121.491057158  t4=...4122.012835225  t5=...4122.048552066
seq=2  t0=...4122.055203100  t1=...4127.850189505  t2=...4127.851648669
       t3=...4128.599010674  t4=...4129.121040813  t5=...4129.160118056
```

`producer.log` ends mid-snapshot-4 (the line for
`memory_dump-20260519020529165.raw` shows a pmemsave start with no matching
"RAW memory dump OK" — consistent with orchestrator shutdown during the
dump).

---

# Run 2 — post bc-fix (2026-05-20)

**Run ID:** `20260519T222407Z_c169d947`
**Configuration:** `intervalMsec=100`, `ramSizeMb=1024`, idle VM (no `--test-command`).
**Duration:** ~298 s host wall-clock.
**Snapshots:** 19 complete (seq 0–18) + 85 backpressure-event records.
**Method:** Same instrumented producer; `bc` installed on capture host this time.

## Measured timing table (Run 2)

| seq | suspend | pmemsave | flush | resume | host_dt | guest_run | queue |
| --: | ------: | -------: | ----: | -----: | ------: | --------: | ----: |
|  0  |  0.185  |  0.789   | 0.518 | 0.038  |  1.659  | **0.126** |   2 |
|  1  |  0.292  |  0.770   | 0.517 | 0.035  |  1.740  | **0.126** |   2 |
|  2  |  0.201  |  0.770   | 0.522 | 0.035  |  1.656  | **0.126** |   3 |
|  3  |  1.832  |  0.752   | 0.522 | 0.036  |  3.270  | **0.126** |   4 |
|  4  | **80.318** | 0.753 | 0.521 | 0.037  | 81.744  | 0.113     |   5 |
|  5  |  1.843  |  0.769   | 0.521 | 0.036  |  3.297  | **0.126** |   6 |
|  6  |  8.241  |  0.771   | 0.522 | 0.546  | 10.209  | 0.128     |   7 |
|  7  |  1.497  |  0.765   | 0.522 | 0.035  |  2.946  | **0.126** |   8 |
|  8  |  1.902  |  3.020   | 0.522 | 0.034  |  8.303  | **0.126** |   9 |
|  9  |  3.427  |  3.920   | 0.524 | 0.037  |  8.036  | **0.126** |  10 |
| 10  |  2.836  |  3.076   | 0.522 | 0.036  |  6.597  | **0.126** |  11 |
| 11  |  0.873  |  2.024   | 0.522 | 0.039  |  3.570  | 0.110     |  12 |
| 12  | 10.453  |  1.865   | 0.523 | 0.038  | 13.622  | **0.126** |  13 |
| 13  |  3.949  |  3.236   | 0.520 | 0.037  |  7.870  | **0.126** |  14 |
| 14  |  0.480  |  2.350   | 0.522 | 0.036  |  3.516  | **0.126** |  15 |
| 15  |  1.662  |  1.474   | 0.521 | 0.035  |  3.808  | 0.114     |  16 |
| 16  |  0.075  |  1.526   | 0.522 | 0.036  |  2.287  | **0.126** |  17 |
| 17  |  0.253  |  1.438   | 0.559 | 0.040  |  2.418  | **0.126** |  18 |
| 18  |  0.908  |  2.609   | 0.538 | 0.041  |   —     | —         |  19 |

All times in seconds.

### Aggregates (n = 19 components, n = 18 inter-snapshot deltas)

| Quantity         |     mean |   median |      min |      max |      std |       CV |
| ---------------- | -------: | -------: | -------: | -------: | -------: | -------: |
| suspend          |  6.380 s |  1.662 s |  0.075 s | 80.318 s | 17.633 s |   276 %  |
| pmemsave         |  1.720 s |  1.474 s |  0.752 s |  3.920 s |  1.011 s |    59 %  |
| flush+chown      |  0.524 s |  0.522 s |  0.517 s |  0.559 s |  0.009 s |   1.8 %  |
| resume           |  0.064 s |  0.036 s |  0.034 s |  0.547 s |  0.114 s |   179 %  |
| host_dt          |  9.253 s |  3.543 s |  1.656 s | 81.744 s | 17.887 s |   193 %  |
| **guest_run** (Axis A) | **0.124 s** | **0.126 s** | **0.110 s** | **0.128 s** | **0.005 s** | **4.2 %** |

## What this confirms

**1. The `bc` hypothesis was correct.** Run 1 measured `guest_dt = 6.6 ms`
when configured `intervalMsec = 100`. Run 2 measures `guest_dt = 125 ms`. The
producer's post-resume sleep is now firing. The +25 ms over the 100 ms target
is bash-bookkeeping overhead between `t5_after_resume` and the next
`t0_before_suspend` (two `find` forks for the backpressure check, one `date`
fork for the timestamp, file writes, echo statements). Unavoidable without a
producer rewrite and small enough to ignore for spectral analysis.

**2. Axis A is stationary.** `guest_run_interval` CV = 4.2 %. Range
0.110–0.128 s. The 0.110 and 0.114 outliers correspond to seq=4 and seq=11
which had slightly compressed pre-suspend bookkeeping; not large enough to
break cepstral assumptions. Phase 1's `StabilityValidator` would accept this.

**3. The frame-spacing claim of the corrected timing model is now empirically
validated.** With `intervalMsec = 100` the analyzer truly sees ~125 ms per
frame. `window = 128` covers `128 × 0.125 = 16 s` of guest time, not the
0.85 s from Run 1. Slow rhythms (sqlite checkpoint, sandbox phases) can now
fit in a single window.

## What Run 2 also revealed (new problems)

**A. Backpressure is the binding constraint.** Queue depth grew monotonically
from 2 → 19 over the 19 snapshots, then hit cap=20 and the producer spent
85 s in `sleepOnBackpressureSeconds=1` waits while the consumer drained.
With `pmemsave ≈ 1–2 s`, the producer can issue snapshots at ~0.4 Hz; the
consumer (Rust delta + queue file management) clearly can't sustain that
when the prev dump is still being read for delta computation. **For any
real Phase 2 capture, this needs addressing before plan 02 is run** — either
raise `maxPendingJobs`, throttle the producer, or speed the consumer.

**B. Suspend latency is wildly non-stationary.** Range 0.075 s to 80.3 s.
The 80 s spike at seq=4 is one event but it utterly dominates the
host-side statistics. Median 1.66 s, mean 6.38 s, std 17.6 s. Most likely
cause: the consumer's concurrent `live_delta_calc` process reading the
previous dump from `/var/lib/libvirt/qemu/dump` competes with the
producer's `virsh suspend` / `domstate` calls (the libvirtd socket is
single-threaded for monitor commands on many builds). The 80 s spike may
be a libvirtd RPC backlog clearing.

**C. Pmemsave throughput halved under contention.** Seq 0–7 measured
~0.77 s (1.36 GiB/s, matching Run 1). Seq 8–18 measured 1.5–3.9 s
(0.27–0.71 GiB/s). The break is gradual but real. Almost certainly disk
bandwidth contention with the consumer reading prior 1 GiB dumps.

**D. VM pause fraction effectively unchanged.** With `guest_dt ≈ 0.125 s`
and `host_dt ≈ 5–10 s` (excluding the outlier), pause fraction is still
≈ 98.6 %. The bc fix did not change the host throughput because pmemsave +
suspend confirmation dominate, not the post-resume sleep.

## Implications for downstream analysis

**Δt_frame is now correct.** This unblocks the cepstral / MSC / PLV
interpretation. Window=128 hop=64 now means 16 s × 8 s of guest time at
`intervalMsec=100`. Slow rhythms (sqlite checkpoint ~10 s, sandbox phases
~10 s, slowburn 5 s/file) fit inside a window.

**Old datasets remain mis-calibrated.** Any data collected before the bc
fix has `Δt_frame ≈ 6.6 ms` and is unusable for spectral analysis at the
canonical (128, 64). Either re-collect or reanalyze with corrected Δt.

**Throughput is still bad.** With 9 s mean host_dt per snapshot, a 300 s
guest-time run needs ~2400 snapshots and ~6 host-hours. Backpressure
saturation makes the real number even worse. A 5 min Phase 2 workload at
`intervalMsec = 100` is impractical until the consumer keeps up.

**The host-side variability does not affect Δt_frame.** A snapshot whose
host cycle takes 80 s still represents 125 ms of guest activity between
prev and curr. The analyzer sees a clean frame; the operator just waits
longer. So *for analysis purposes* the bc fix is sufficient. For
*experiment throughput* the next bottleneck is consumer + libvirtd
contention.

## Updated recommended next actions

1. **Re-run Experiment 1 producer-only (consumer disabled)** to isolate
   pure pmemsave + suspend latency from the contention pattern. The current
   orchestrator already disables `streaming` and `rawRetention.enabled`,
   but the consumer process itself still runs because of the default
   capture launch path. Stop the consumer, or run the producer
   stand-alone, and re-measure suspend / pmemsave latency without the
   delta-calc disk contention.

2. **Profile the consumer** (`capture_consumer_qemu.sh` + `live_delta_calc`)
   to see why one dump takes longer than `intervalMsec + pmemsave + flush`
   to process. If the Rust delta calc reads both prev and curr at full
   1 GiB each, it does 2 GiB of sequential read per job. At a typical SSD
   read rate ~1 GiB/s that is 2 s — already at the producer's rate, no
   headroom. Either speed the calc or skip it in the timing experiment.

3. **Raise `backpressure.maxPendingJobs`** for diagnostic runs only. The
   default 20 is enough to mask a slow consumer for ~20 snapshots before
   the producer stalls. Raising it does not fix the consumer but lets
   longer runs collect more pre-stall data.

4. **Document the +25 ms script overhead** in the producer's comments
   (and `RUN_TIMING_INSTRUMENTATION.md`) so the operator knows the
   effective `Δt_frame ≈ intervalMsec + 25 ms` regardless of the
   configured value.

5. **Investigate the 80 s suspend spike.** Single spike out of 19 is
   not enough to characterise the tail. The next Run 2-equivalent
   experiment should run longer (≥ 5 min producer-only, large
   `maxPendingJobs`) so the tail of the suspend distribution is
   sampled with enough events to estimate.

6. **Plan 02 (per-family `intervalMsec` tuning) is now unblocked for
   analysis-side claims** (Δt_frame works). But it is still blocked for
   *throughput* claims until the consumer / contention story is resolved.
   The pilot's wall-clock cost table needs to be re-derived from
   measured Run 2 numbers (mean host_dt ≈ 5 s when stable, not 2.5 s as
   originally estimated).

## What is now known with confidence

- Configured `intervalMsec = 100` produces `guest_dt ≈ 125 ms` (target + 25 ms).
- `guest_dt` is stationary across the 19 snapshots (CV 4.2 %).
- pmemsave on quiet host: ~0.77 s for 1 GiB (1.36 GiB/s).
- pmemsave under consumer contention: 1.5–3.9 s.
- suspend confirmation: 0.2–2 s when uncontended, can spike to >80 s.
- backpressure starts firing at queue cap within ~20 snapshots if consumer is active.

## What remains uncertain

- Long-run distribution of suspend latency (single 80 s spike is not enough
  to estimate the tail).
- Whether removing the consumer brings suspend latency to a stable
  sub-second value or whether libvirtd alone introduces variance.
- Whether the +25 ms script overhead is truly a constant or scales with
  queue depth / disk load.
- Whether the 0.5 s flush sleep is still necessary now that we can measure
  pmemsave completion directly via the monitor return.

## Appendix: backpressure timing

Backpressure events fired starting at host time `1779229744.453` (after
seq=18 enqueued) and continued for 85 events over ~127 s of host
wall-clock. The orchestrator killed the producer at the duration cap with
the queue still saturated.
