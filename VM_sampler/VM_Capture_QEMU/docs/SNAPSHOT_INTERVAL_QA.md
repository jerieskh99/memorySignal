# Snapshot Interval Q&A

Companion to [SNAPSHOT_INTERVAL_TUNING_STUDY.md](./SNAPSHOT_INTERVAL_TUNING_STUDY.md).
Explains the corrected timing model in plain language. Read this if you
ever need to ask: *"what does `intervalMsec = 100` actually mean for my
analysis?"*

## Main takeaway

There are **three** different time axes in the pipeline. The earlier study
collapsed them and reached wrong conclusions about what `intervalMsec`
buys you.

| Axis                         | What it is                                                           | What it sets                                                       |
| ---------------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------ |
| **A. Guest-running interval** | Wall-clock gap *as seen by the guest's own monotonic clock*. The VM does **not** run during `pmemsave`. | The **frame spacing** the analyzer should use for cepstra/MSC/PLV. |
| **B. Host wall-clock per snapshot** | Real time on the host to produce one frame: suspend + pmemsave + resume. | The **production rate**; how fast the experiment can collect data. |
| **C. Experiment throughput**  | `total_snapshots × Axis-B`. How long the entire run takes.            | The **cost** of an experiment, in wall-clock hours.                |

`intervalMsec` sets **Axis A**, not Axis B. With `intervalMsec = 100`,
the analyzer sees ~100 ms of *guest activity* between consecutive frames,
even if each frame takes ~2.5 s of host wall-clock to materialize.

Choosing `intervalMsec` answers: *"what guest-time resolution do I
need?"* — not *"how long will this experiment take?"*

---

## Core Validation Questions

### 1. Does the current capture loop pause the VM during pmemsave?

Yes. The producer (`capture_producer_qemu_pmemsave.sh`) calls
`virsh suspend` before every snapshot and `virsh resume` after it. The
exact order (lines 91–164):

```
virsh suspend       (line 92)   ← VM goes paused
wait domstate=paused (line 98)
pmemsave 1 GiB      (line 106)  ← VM frozen, dump to disk
sleep 0.5           (line 114)  ← flush
enqueue prev/curr   (line 138)
virsh resume        (line 154)  ← VM goes running
wait domstate=running (line 156)
sleep intervalMsec  (line 161)  ← VM running
[next iteration]
```

The VM is paused from line 92 to line 154 — across the pmemsave window.

### 2. Does the configured `intervalMsec` represent guest-running time between snapshots, host wall-clock delay between captures, or something else?

It represents **guest-running time between snapshots** (Axis A). The
`sleep intervalMsec/1000` (line 161) happens *after* `virsh resume`,
while the VM is running. The next iteration's `virsh suspend` follows
that sleep. So between two consecutive pmemsave moments, the guest
executes for approximately `intervalMsec` of its own time, plus a few
milliseconds of bash bookkeeping.

### 3. Where exactly does sleep happen relative to resume/suspend?

After `resume`, before the next iteration's `suspend`. The sleep is
the **only** part of the loop where guest time advances. Everything
else either (a) happens before resume (so the VM is still paused) or
(b) happens after `resume` but is so brief that it adds only a few ms.

There is one additional `sleep 0.5` (line 114) *inside* the paused
window — a flush wait for pmemsave. That sleep happens while the VM is
paused, so it does not consume guest time; it adds 0.5 s to Axis B.

> **Clarification on "inflates Axis B".** Axis B is the additive sum
> of all host-side wall-clock costs per snapshot. The 0.5 s flush is
> one term in that sum; it is not double-counted anywhere. The choice
> of the word "inflates" was informal — what is meant is "this term
> adds to the per-snapshot wall-clock cost". The flush is also
> arguably *redundant*: `qemu-monitor-command pmemsave` returns
> synchronously after writing all bytes, so the 0.5 s belt-and-braces
> sleep is a safety margin from when this path was first prototyped,
> not a correctness requirement. Whether to keep it is a separate
> decision (see `docs/tuning_plans/02_interval_tuning_experiment.md`);
> mathematically the accounting is correct as stated.

### 4. During pmemsave, does the guest workload progress or is it frozen?

It is frozen. QEMU stops the VCPUs and pauses the guest TSC during
`virsh suspend`. Inside the VM, `clock_gettime(CLOCK_MONOTONIC)` does
not advance. A workload thread blocked on `nanosleep(100ms)` does not
wake during pause — it wakes at guest-time `T + 100ms`, which on the
host wall-clock may be `T + 100ms + (2 s of pmemsave + bookkeeping)`.

So from the *workload's* perspective, the snapshot is instantaneous.
The delta between consecutive snapshots reflects what the workload
changed during ~100 ms of its own execution — exactly the thing the
analyzer should measure.

### 5. Should analysis frame spacing be interpreted as guest-running interval or host wall-clock time?

**Guest-running interval.** The deltas computed by the consumer
measure guest memory change per `intervalMsec` of guest execution.
Cepstral peak indices, MSC frequencies, PLV time axes — all of these
should be reported using `Δt_frame = intervalMsec`, not the host
wall-clock per snapshot.

If a downstream consumer converts frame index `k` to seconds, it
should multiply by `intervalMsec / 1000`, not by `host_dt_mean_s`.

### 6. What timing values must be logged to prove this?

Six per-snapshot timestamps from the producer (`clock_gettime(CLOCK_REALTIME)`
on the host):

| Timestamp                    | Where in the loop                                  |
| ---------------------------- | -------------------------------------------------- |
| `t0_before_suspend_host_ns`  | Before `virsh suspend` (line 92).                  |
| `t1_after_suspend_host_ns`   | After `domstate == paused` (line 98).              |
| `t2_pmemsave_start_host_ns`  | Before invoking the pmemsave monitor cmd (line 106). |
| `t3_pmemsave_end_host_ns`    | After the monitor cmd returns.                      |
| `t4_before_resume_host_ns`   | Before `virsh resume` (line 154).                   |
| `t5_after_resume_host_ns`    | After `domstate == running` (line 156).             |

Derived per-snapshot:

- `host_dt = next_t0 − t0` — Axis B.
- `pmemsave_dt = t3 − t2`.
- `vm_pause_dt = t5 − t0` — host wall-clock the VM was paused.
- `guest_run_dt = next_t0 − t5` — Axis A on the host side (equals
  `sleep intervalMsec/1000` plus bash overhead when no backpressure).

Optional but useful: probe the guest's `CLOCK_MONOTONIC` over SSH at
`t5` and again at `next_t0`; the delta between them should match
`guest_run_dt` to within a few ms.

### 7. Which parts of the previous snapshot interval study remain correct?

- The capture loop description and the location of `intervalMsec` in
  `config_qemu_upc.json`.
- The fact that pmemsave is expensive for 1 GiB and dominates host
  wall-clock per snapshot.
- The raw I/O exposure numbers (snapshot_count × 1 GiB).
- Backpressure can corrupt sampling regularity (and the recommendation
  to instrument it).
- The pilot design's factorial structure, the per-workload defining
  metrics, and the use of monotonic seed/parameter holds.
- The Option B (per-family interval profile) recommendation, modulo
  the corrected justification.

### 8. Which parts are incorrect or overstated?

- "Effective sampling frequency is closer to 0.2–0.4 Hz, not 10 Hz" —
  incorrect when read as frame spacing. That's the **production
  rate** (Axis B), not the frame spacing (Axis A).
- "Realistic effective Δt ≈ 3 s" applied to spectral analysis — wrong
  axis.
- The "realistic" Section 3 sub-tables showing 0 windows for any 10-min
  run — wrong; window count is set by guest duration / `intervalMsec`,
  not by wall-clock cost.
- "k=2 segmentation requires 5.4 hr at realistic Δt" — wrong; 10.9 min
  of *guest time*, ~4.5 h of *wall clock* — different axes.
- "`window=128` covers ~6.4 minutes of wall clock at realistic Δt" —
  wrong; window is `128 × intervalMsec` of *guest* time.
- "`sandbox_ransom_seq` is infeasible at 1 GiB pmemsave" — wrong as a
  resolution claim. The per-file rhythm may be sub-Nyquist at chosen
  `intervalMsec`, which is a separate, fixable issue (lengthen
  per-file phase, or accept aggregate file-rate rhythm).
- Acceptance criterion 4 in the old Section 7 (`vm_pause_fraction ≤
  0.30` because of "guest/host wall-clock divergence") — wrongly
  framed; pause fraction does not bias guest-time frame spacing.

### 9. How should the interval/duration tuning plan be revised?

Three changes:

1. **Replace "Δt" with "Δt_frame" everywhere** and define it as
   `intervalMsec`.
2. **Add a parallel host-wall-clock-cost view** for every guest-time
   plan, so the throughput-vs-resolution trade-off is visible.
3. **Reframe the family recommendations** as "what guest-time
   resolution does each family need?" — the wall-clock cost is then
   set by the *guest duration* the test needs, not by the interval.

The Option B per-family profile still stands but is justified
differently: profiles select resolution (`intervalMsec`) and guest
duration jointly; the wall-clock cost falls out of those choices.

### 10. What does this imply for `window=128`, `hop=64`, and segment-level analysis?

- One window covers `128 × intervalMsec` of **guest** time. At 100 ms
  that is 12.8 s of guest activity; at 1 s it is 128 s of guest
  activity.
- One hop covers `64 × intervalMsec` of guest time.
- Spectral peaks are at frequencies `k / (window × intervalMsec)` Hz
  in **guest** time.
- A k-segment segmentation needs `≥ k × 50 ≈ 50k` windows; that
  translates to `(50k − 1) × 64 + 128 ≈ 3264k` frames; that translates
  to `3264k × intervalMsec` of guest time.
- The wall-clock cost to collect those frames is `3264k ×
  wallclock_per_snapshot ≈ 3264k × 2.5 s ≈ 8160k seconds ≈ 2.3 k
  hours` (independent of `intervalMsec`).
- So k=2 segmentation at any `intervalMsec` costs ≈ 4.6 host hours
  regardless of whether the chosen interval is 100 ms or 1 s. The
  interval controls what *resolution* you get in those hours, not
  how long they take.

---

## Practical Understanding Questions

### Is my 100 ms interval actually wrong?

No. It is a legitimate choice for guest-time resolution — you get
100 ms frame spacing in the analyzer. What is "wrong" is the
*expectation* that 100 ms means anything about how long the experiment
takes. At 1 GiB RAM you pay ~25 s of host wall-clock per second of
*guest* data.

### Does pmemsave time count as workload execution time?

No. While pmemsave runs, the VM is paused; the guest's monotonic
clock does not advance. Workload threads do not run; their internal
timers do not fire.

### What is the difference between guest-running time and host wall-clock time?

Guest-running time advances only while the VM is *running*. Host
wall-clock advances always. During pmemsave the VM is paused for
~2 s of host wall-clock, but the guest sees 0 ms.

A 60 s guest workload at `intervalMsec=100ms` makes 600 snapshots; at
2.5 s host cost per snapshot, the host pays ≈ 25 minutes to collect
60 s of guest activity.

### Why does it feel like the experiment is much longer than the guest run?

Because each snapshot is 25× more expensive than the gap it covers.
With `intervalMsec=100ms` and a ~2.5 s pmemsave, every 100 ms of
guest execution costs ≈ 2500 ms of host wall-clock. Over a 60 s
guest run, that ratio compounds into a 25-minute host run. The guest
log still shows 60 s of activity (correct, in guest time); the
operator's wall clock shows 25 minutes (also correct, in host time).
The two are not in conflict — they are the two axes from the main
takeaway.

### Why does it take hours and hours for a few minutes on the guest axis?

Same arithmetic, larger numbers:

| Guest duration | `intervalMsec` | Snapshots | Host wall-clock (≈2.5 s/snap) |
| -------------: | -------------: | --------: | ----------------------------: |
| 1 min          | 100 ms         |       600 | 25 min                        |
| 5 min          | 100 ms         |     3 000 | 2 h 5 min                     |
| 10 min         | 100 ms         |     6 000 | 4 h 10 min                    |
| 5 min          | 1 s            |       300 | 12 m 30 s                     |
| 10 min         | 1 s            |       600 | 25 min                        |

The two levers that move host wall-clock are:

1. **Snapshot count** — set by `guest_duration / intervalMsec`. Bigger
   `intervalMsec` → fewer snapshots → less wall-clock.
2. **Per-snapshot cost** — set by Axis B, dominated by 1 GiB pmemsave
   and the disk write bandwidth. Smaller VM RAM → cheaper per snapshot.

If the experiment "feels too long" the question is which lever to
pull. The tuning plans under
[`docs/tuning_plans/`](./tuning_plans/) plan that decision.

### If I increase `intervalMsec`, will the same guest-time run finish faster because the VM is paused fewer times?

Yes. The VM is paused once per snapshot, so fewer snapshots means
fewer pauses, fewer pmemsave dumps, fewer disk writes, and a
proportionally shorter host wall-clock. For a 5-minute guest run,
going from `intervalMsec=100ms` to `intervalMsec=1s` cuts snapshot
count from 3000 to 300 and host wall-clock from ~2 h to ~12 min.

The cost is in **guest-time resolution**: 1 s frame spacing cannot
see rhythms faster than ~2 s (Nyquist), and 1 s frame spacing makes
windows much coarser. The right joint choice of
`(intervalMsec, window, hop)` depends on the rhythm of interest and
is the subject of
[`02_interval_tuning_experiment.md`](./tuning_plans/02_interval_tuning_experiment.md)
and
[`03_window_hop_tuning_experiment.md`](./tuning_plans/03_window_hop_tuning_experiment.md).

### If the VM is paused during pmemsave, what does one frame mean?

It is a snapshot of guest RAM taken at a particular instant of guest
execution. The *delta* between consecutive frames is "what changed in
guest memory during the `intervalMsec` of guest time between the two
pmemsave moments". From the guest's perspective, that's about 100 ms
of execution.

### Why does the experiment feel slow if the guest only runs 100 ms between samples?

Because **producing** the snapshot is expensive. The host has to
suspend the VM, write 1 GiB to disk, resume the VM, run the delta,
append to the run matrix, optionally compute streaming metrics — all
of that takes ~2.5 s. The "slowness" is Axis B (production rate), not
Axis A (sampling resolution).

### Does increasing `intervalMsec` make each snapshot cheaper?

**No to each snapshot, but yes to the whole run.** Each pmemsave
still writes 1 GiB; each suspend/resume cycle still has the same
latency. The per-snapshot cost (Axis B) is set by RAM size and disk
speed — not by `intervalMsec`.

But a bigger `intervalMsec` means *fewer snapshots per guest second*,
so fewer pauses, fewer pmemsave dumps, and a proportionally shorter
**total** host wall-clock for the same guest duration. The cost is
in guest-time resolution.

### What does increasing `intervalMsec` actually change?

Two things:

1. **The guest-time spacing between samples** grows. The analyzer sees
   coarser temporal resolution — rhythms faster than `2 × intervalMsec`
   become invisible (Nyquist).
2. **The snapshot count for the same guest duration** drops
   proportionally, so the experiment finishes in proportionally less
   wall-clock time.

So `intervalMsec` is a **throughput-vs-resolution** knob, not a
"how-slow-is-each-snapshot" knob.

### How many frames do I need for `window=128` and `hop=64`?

At minimum 128 (one window). For useful sliding-window statistics,
the offline analyzer's default expects at least 50 windows per
segment, which is `(50 − 1) × 64 + 128 = 3264` frames per segment.
For k segments, ≈ `3264 k` frames.

> **Deferred.** The right values of `window` and `hop` (and the
> 50-windows-per-segment threshold) for Phase 2 are themselves
> open questions — see
> [`03_window_hop_tuning_experiment.md`](./tuning_plans/03_window_hop_tuning_experiment.md).
> This answer will be revisited once that tuning lands.

### What does a 128-frame window mean in guest time?

`128 × intervalMsec`.

| `intervalMsec` | Window in guest time |
| -------------: | -------------------: |
| 100 ms         |              12.8 s  |
| 250 ms         |              32.0 s  |
| 500 ms         |              64.0 s  |
| 1 s            |             128.0 s  |

### What does it mean in host collection time?

≈ `128 × wallclock_per_snapshot` ≈ `128 × 2.5 s` ≈ 5.3 min, almost
**independent** of `intervalMsec`. Each window costs roughly the same
wall-clock to produce regardless of which guest-time interval you
chose.

### How does this affect segment-level analysis?

Segment-level analysis is bounded by *frame count*, not by guest time
or wall-clock time directly. To get k segments × 50 windows × ~64
hop ≈ `3264 k` frames, choose:

- a small `intervalMsec` if you need fine guest-time resolution within
  each segment, and accept the wall-clock cost; or
- a larger `intervalMsec` if your phenomenon is slow in guest time,
  and pay less wall-clock.

In both cases the host pays roughly the same wall-clock to produce
the segmenter input (Axis B × frame count), so the choice is really
about *guest-time resolution per spent host-hour*.

### How should I choose 100 ms vs 250 ms vs 500 ms?

Three questions, in order:

1. **What is the fastest guest-time rhythm I care about?** Need
   `intervalMsec ≤ rhythm / 2` for Nyquist; `≤ rhythm / 4` to actually
   see the peak cleanly in cepstra.
2. **What is the slowest guest-time rhythm I care about?** Need
   `window × intervalMsec ≥ rhythm` (otherwise the window cannot
   contain a full period).
3. **What is my wall-clock budget?** Compute
   `snapshots × 2.5 s = guest_duration / intervalMsec × 2.5 s`. If
   this exceeds the budget, increase `intervalMsec` until it fits —
   accepting the resolution loss from (1).

### What timing should I log to prove the real behavior?

The six per-snapshot timestamps in question 6 above. Plus per-run
aggregates: `guest_dt_mean_s`, `guest_dt_std_s`, `host_dt_mean_s`,
`vm_pause_fraction`, `backpressure_events`.

### What claims from the old MD were wrong or overstated? Where have they been corrected?

See question 8 above. Short list:

- "Effective Δt = 2.5–5.5 s" applied to spectral analysis.
- "Realistic effective Δt ≈ 3 s" used to argue most runs have 0
  windows.
- "Window covers 6.4 min of wall clock at realistic Δt" — wrong axis.
- "k=2 segmentation impossible at realistic Δt" — confused guest time
  with wall-clock.
- "`sandbox_ransom_seq` is infeasible because of pmemsave cost" —
  wrong axis.
- "`vm_pause_fraction ≤ 0.30` because it biases inter-snapshot
  timestamps" — wrong mechanism (it indexes throughput, not bias).

**Corrected in:**

- `SNAPSHOT_INTERVAL_TUNING_STUDY.md` — full revision across all
  sections.
- `SNAPSHOT_INTERVAL_QA.md` (this file).
- `snapshot_interval_qa.html` (the standalone Q&A page).

A scan of the rest of the repo's documentation (`README.md` for the
capture pipeline, `RUN_CONTROLLED_CAPTURE.md`,
`RAW_CAPTURE_ALTERNATIVE.md`, `docs/QEMU_CAPTURE_PIPELINE.md`,
`VM_executables_phase2/docs/*`, and the workload test site HTML)
turned up no other documents propagating the wrong claims. The
website `VM_executables/workload_test_site/index.html` describes
test cards, not capture timing, and is unaffected.

### What claims remain correct?

- 1 GiB pmemsave is expensive and dominates wall-clock per snapshot.
- Raw I/O totals (1 GiB written per snapshot).
- Backpressure can corrupt sampling regularity.
- The per-family profile recommendation (the *justification* is
  revised; the recommendation stands).
- The pilot's factorial design and the choice of defining metrics.

### What should I do next experimentally?

1. Instrument the producer to log all six per-snapshot timestamps.
   Without that, you cannot tell whether Axis A is honored or
   corrupted.
2. Run a tiny 4-cell pilot: `(intervalMsec, guest_duration) ∈ {100 ms,
   1 s} × {1 min, 5 min}` on `sandbox_ransom_batched`. Verify
   `guest_dt_mean_s ≈ intervalMsec` and `guest_dt_std_s` is tiny
   (a few ms) when no backpressure fires.
3. If that holds, scale to the Section 6 pilot in the main study.
4. Decide per-family `intervalMsec` from rhythm requirements
   (Section 5 of the main study), not from wall-clock cost.
5. Set guest duration per family from the segmentation k requirement
   (`3264 k` frames per segment minimum).
6. Compute wall-clock cost as a *consequence* of (4) + (5), and
   decide whether to reduce RAM size to bring it down.
