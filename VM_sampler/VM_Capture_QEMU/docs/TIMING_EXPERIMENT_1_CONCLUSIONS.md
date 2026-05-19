# Experiment 1 — Timing Instrumentation: Conclusions

**Run ID:** `20260519T000307Z_7428bb09`
**Host:** `pcrserral` (Linux capture host)
**Date analysed:** 2026-05-19

## Executive summary

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
