# 01 — Producer Instrumentation & Timing Logging Plan

**Status:** plan, not yet implemented.
**Prerequisite for:** every subsequent plan in this folder.
**Estimated implementation effort:** ~1 day of producer + consumer + analyzer changes; 1 day of validation; 0 hours of new experiment runtime (the instrumentation runs alongside normal captures).

## Why this comes first

The corrected timing model in
[`../SNAPSHOT_INTERVAL_QA.md`](../SNAPSHOT_INTERVAL_QA.md) defines three
time axes. None of the downstream tuning experiments can answer their
own questions until the pipeline actually records all three axes per
snapshot. Right now:

- The producer logs the snapshot filename with a millisecond
  timestamp (which approximates `t2_pmemsave_start`), but not the
  suspend/resume boundary timestamps.
- The consumer logs delta-calculation completion time, which is
  Axis B-adjacent but not Axis B-precise.
- Nothing currently records how often backpressure fires per run —
  the only source is grepping `producer.log`.

The plan is to add six per-snapshot timestamps and a small per-run
summary, plumbed through the existing producer/consumer pipeline
without changing the captured-frame format.

## Goals

1. **Compute Axis A (guest-time frame spacing) per snapshot pair** so
   the analyzer can verify whether `intervalMsec` is being honored.
2. **Compute Axis B (host wall-clock per snapshot) per snapshot** so
   throughput tuning has measured data instead of estimates.
3. **Count and tag backpressure events** so the analyzer can quarantine
   frames produced during backpressure gating.
4. **Surface per-run aggregates** in a stable JSON schema that future
   plans depend on.

## Non-goals

- Changing `intervalMsec` or any capture parameter (that is plan 02).
- Changing `window_size` / `step_size` (plan 03).
- Modifying the captured RAW image format. Timestamps live in a
  *sidecar* log file, not in the dump.

## Six per-snapshot timestamps to log

All on the **host** via `clock_gettime(CLOCK_REALTIME)`. Nanosecond
precision. Wall-clock anchor (not monotonic) so timestamps remain
comparable across long runs.

| Name                          | Where in the producer loop                                              |
| ----------------------------- | ----------------------------------------------------------------------- |
| `t0_before_suspend_host_ns`   | Immediately before `virsh suspend` (`capture_producer_qemu_pmemsave.sh:92`).            |
| `t1_after_suspend_host_ns`    | Immediately after `wait_state "paused"` returns OK (line 98).            |
| `t2_pmemsave_start_host_ns`   | Immediately before the `virsh qemu-monitor-command pmemsave` call (line 106). |
| `t3_pmemsave_end_host_ns`     | Immediately after the same call returns.                                |
| `t4_before_resume_host_ns`    | Immediately before `virsh resume` (line 154).                            |
| `t5_after_resume_host_ns`     | Immediately after `wait_state "running"` returns OK (line 156).          |

A seventh derived field is logged too:

- `guest_run_dt_predicted_ns` = `intervalMsec × 10^6` (the sleep target
  that *will* run between `t5` and the next iteration's `t0`).

## Per-snapshot log file format

JSON Lines, one object per snapshot, written to
`<queueDir>/snapshot_timings.jsonl` (sidecar to the existing
`pending/processing/done` queue). Schema:

```json
{
  "seq": 1234,
  "dump_path": "/var/lib/libvirt/qemu/dump/memory_dump-20260518123456789.raw",
  "t0_before_suspend_host_ns":  170...,
  "t1_after_suspend_host_ns":   170...,
  "t2_pmemsave_start_host_ns":  170...,
  "t3_pmemsave_end_host_ns":    170...,
  "t4_before_resume_host_ns":   170...,
  "t5_after_resume_host_ns":    170...,
  "guest_run_dt_predicted_ns":  100000000,
  "backpressure_event":         false,
  "backpressure_wait_ms":       0,
  "dump_bytes":                 1073741824,
  "interval_msec_active":       100,
  "config_path":                "/.../config_qemu_upc.json",
  "vm_ram_mb_active":           1024
}
```

Per-snapshot derived (not stored, computed by the analyzer):

| Derived                       | Formula                                                            | Meaning                          |
| ----------------------------- | ------------------------------------------------------------------ | -------------------------------- |
| `suspend_latency_ns`          | `t1 − t0`                                                          | Time to enter paused state.      |
| `pmemsave_ns`                 | `t3 − t2`                                                          | Dump duration.                   |
| `resume_latency_ns`           | `t5 − t4`                                                          | Time to leave paused state.      |
| `vm_pause_ns`                 | `t5 − t0`                                                          | Host wall-clock the VM was paused. |
| `guest_run_dt_actual_ns`      | `next.t0 − t5`                                                     | Axis A on host side.             |
| `host_dt_ns` (Axis B)         | `next.t0 − t0`                                                     | Per-snapshot host wall-clock.    |

## Per-run aggregate written by the consumer

Appended to `outputDir/<run_id>/timing_summary.json` once the
consumer drains the queue at end-of-run:

```json
{
  "run_id":              "20260518_phase2_batched_seed42",
  "interval_msec":       100,
  "vm_ram_mb":           1024,
  "snapshot_count":      600,
  "guest_dt_mean_s":     0.103,
  "guest_dt_std_s":      0.004,
  "guest_dt_p99_s":      0.118,
  "host_dt_mean_s":      2.74,
  "host_dt_std_s":       0.31,
  "host_dt_p99_s":       3.55,
  "pmemsave_ns_mean":    2_120_000_000,
  "vm_pause_fraction":   0.96,
  "backpressure_events": 0,
  "first_t0_host_ns":    170...,
  "last_next_t0_host_ns": 170...
}
```

Acceptance for the instrumentation itself (not the data):

1. The producer writes exactly one JSON Line per snapshot, in
   order. The line is appended after `virsh resume` returns, never
   before — so a crash mid-dump never leaves a half-line that
   confuses downstream readers.
2. The consumer's aggregate is generated only on clean end-of-run
   (queue drained). Partial runs leave the JSONL but no aggregate.
3. The schema versions itself with a `schema_version` field; any
   future column addition is backward-compatible.

## Implementation surface

- **`capture_producer_qemu_pmemsave.sh`** — add a small bash helper
  `ts_ns() { date +%s%N; }` (Linux GNU date) and capture the six
  timestamps as bash variables before/after the relevant commands.
  Emit the JSON Line at the end of each iteration. ~30 lines.
- **`capture_consumer_qemu.sh`** — at queue-drain time, run a small
  Python aggregator that reads `snapshot_timings.jsonl` and emits
  `timing_summary.json`. ~50 lines.
- **`offline_step_metrics.py`** — read `guest_dt_mean_s` from the
  per-run summary, use it as Δt_frame for cepstral/MSC reporting.
  Reject any run whose `guest_dt_std_s / guest_dt_mean_s ≥ 0.10`
  (configurable). ~20 lines.
- **`config_qemu_upc.json`** — no schema change in this plan; the
  per-run summary subsumes it.

## Validation

A two-row pilot, both on `sandbox_ransom_batched` because its phase
structure is well-defined:

| Run | `intervalMsec` | Guest duration | Expected `guest_dt_mean_s`   | Expected `backpressure_events` |
| --: | -------------: | -------------: | ---------------------------- | ------------------------------ |
|   1 | 100 ms         | 1 min          | 0.100 ± 0.005 s              | 0                              |
|   2 | 1 s            | 1 min          | 1.000 ± 0.005 s              | 0                              |

If row 1's `guest_dt_mean_s` is not close to 0.100 s, either the
instrumentation is wrong or backpressure fired silently — both need
to be ruled out before plan 02 runs.

## Risks and pitfalls

- **`date +%s%N` is GNU-specific.** On BSD/macOS hosts it returns
  the literal string `%N`. The producer is running on the Linux
  capture host so this is fine; if the pipeline ever moves, switch
  to a small C helper.
- **Bash subshell overhead.** Each `$(date +%s%N)` forks once. At
  100 ms intervals that is six extra forks per iteration; bench it
  if perf becomes a concern, and consider an inline read of
  `/proc/uptime` or `EPOCHREALTIME` (bash ≥ 5.0) to avoid the fork.
- **Clock skew during a run.** `CLOCK_REALTIME` can step (NTP); use
  `CLOCK_MONOTONIC_RAW` if we ever need to compare across long runs.
  For this plan, `CLOCK_REALTIME` is fine because the run is finite
  and we only need *relative* timings within it.
- **Backpressure gating wakes the VM during the sleep.** The
  producer's `sleep "$sleepOnBackpressure"` (line 83) is taken with
  the VM running, so it inflates Axis A. The new
  `backpressure_event: true` flag lets the analyzer drop that frame
  if needed.
