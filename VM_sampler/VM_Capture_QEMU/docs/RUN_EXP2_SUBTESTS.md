# Experiment 2 — sub-tests 2a / 2b / 2c

Three orchestrators that target the two open mechanisms from
[`TIMING_EXPERIMENT_1_CONCLUSIONS.md`](./TIMING_EXPERIMENT_1_CONCLUSIONS.md):
non-stationary host Δt (iii) and the pause-fraction noise floor (iv).
Each script writes one JSON file with the data needed for analysis.

| Sub-test | Targets | Script | Default cost |
| -------- | ------- | ------ | ------------ |
| 2a · consumer isolation | mechanism iii | `run_exp2a_consumer_isolation.py` | 2 × 60 s ≈ 2 min |
| 2b · interval sweep     | mechanism iv  | `run_exp2b_interval_sweep.py`     | 4 × 60 s + cool-down ≈ 5 min |
| 2c · flush sensitivity  | bonus (host_dt −0.5 s) | `run_exp2c_flush_sensitivity.py` | 2 × 60 s ≈ 2 min |

All three use the **instrumented producer** patched in Experiment 1.
None of them start the consumer.

## Prerequisites

- `bc` installed (verified in Experiment 1 Run 2).
- Linux capture host with `virsh`, `pkill`, `pgrep` available.
- Free disk space under `imageDir` (each pass writes ~1 GiB / snapshot;
  scripts clean up afterwards unless `--keep-dumps`).
- `config_qemu_upc.json` (or another file via `--config`) with correct
  `domain`, `ramSizeMb`, `imageDir`, `queueDir`.

The producer was patched in this commit to add the `TIMING_NO_FLUSH`
environment-variable guard on the `sleep 0.5` flush; 2c needs that
patch to be live. 2c verifies it via `producer_has_flush_toggle` in
the output JSON; if `false`, patch the producer first.

## 2a — consumer isolation

Runs two passes back-to-back:

1. **consumer_on** — whatever the host state is when invoked.
2. **consumer_off** — `pkill -f capture_consumer_qemu.sh`, drain the
   queue, then run the producer alone.

```sh
python3 run_exp2a_consumer_isolation.py \
    --duration 60 \
    --output-json ./exp2a.json
```

Useful flags:

| Flag | Meaning |
| ---- | ------- |
| `--duration SEC`     | Per-pass duration (default 60). |
| `--interval-ms INT`  | Override `intervalMsec` in config snapshot. |
| `--ram-mb INT`       | Override `ramSizeMb` (must match live VM RAM for full dump). |
| `--skip-pass1`       | Skip the consumer-on pass (e.g. if consumer not running). |
| `--keep-dumps`       | Do not remove dumps written by this run. |
| `--dry-run`          | Validate paths, emit plan JSON, no VM action. |

Output JSON schema (top-level keys):

```text
experiment:  "2a_consumer_isolation"
config:      { interval_ms, duration_sec, ram_size_mb, ... }
passes.consumer_on:  { mean_host_snapshot_cycle_sec, mean_suspend_sec, ... , records: [...] }
passes.consumer_off: { ... }
comparison:  { mean_host_snapshot_cycle_sec: { on, off, delta, ratio_off_over_on }, ... }
notes:       [ ... ]
```

The headline number is `comparison.mean_host_snapshot_cycle_sec.delta`. If
negative (off < on), the consumer was indeed the contention source.
Expected from Run 2 data: `suspend` median drops from 1.66 s to ~0.2 s,
pmemsave from 1.47 s to 0.77 s.

## 2b — interval sweep

Producer-only (consumer killed up front). For each combination of
`intervalMsec` × pmemsave-size, runs the producer for `--duration`
seconds and records per-cell timing aggregates.

```sh
# Default: 4 intervals × 1 RAM = 4 cells
python3 run_exp2b_interval_sweep.py \
    --duration 60 \
    --intervals 100,250,500,1000 \
    --output-json ./exp2b.json

# With pmemsave-size sweep (caveat: VM RAM not actually changed)
python3 run_exp2b_interval_sweep.py \
    --duration 60 \
    --intervals 100,500,1000 \
    --ram-sweep 256,512,1024 \
    --output-json ./exp2b_full.json
```

Useful flags:

| Flag | Meaning |
| ---- | ------- |
| `--intervals "a,b,c"`         | Comma-separated intervalMsec values. |
| `--ram-sweep "a,b,c"`         | Comma-separated pmemsave sizes (MiB). Optional. |
| `--duration SEC`              | Per-cell duration (default 60). |
| `--inter-cell-cooldown SEC`   | Idle between cells (default 5). |
| `--dry-run`                   | Plan JSON, no work. |

Output JSON schema:

```text
experiment:  "2b_interval_sweep"
config:      { intervals_ms, rams_mb, per_cell_duration_sec, n_cells, ... }
cells:       [ { cell_id, interval_ms, ram_mb, mean_*, estimated_vm_pause_fraction, ... } ]
comparison:
  rows:                       [ { cell_id, interval_ms, ram_mb, pause_fraction, ... } ]
  pause_fraction_ranking:     [ top-5 best (lowest pause %) ]
  best_pause / worst_pause
notes:       [ ... ]
```

The headline result is `comparison.best_pause` (the cell with the lowest
VM pause fraction). Expected: 1 s × 256 MiB cell drops pause from ~98 %
to ~55 %.

> **Caveat repeated.** `--ram-sweep` shrinks the pmemsave dump size
> only; the VM still has its configured RAM. This is a *planning*
> estimate. For a real smaller-VM measurement, reboot the guest with
> the target RAM and re-run with `--ram-sweep` set to that value.

## 2c — flush sensitivity

Two passes (flush_on / flush_off), producer-only. `flush_off` sets
`TIMING_NO_FLUSH=1` so the producer skips the post-pmemsave 0.5 s
sleep. Both passes probe a sample of dumps for integrity (file size +
3 non-zero content samples at offsets 0 / RAM/2 / RAM − 4 KiB).

```sh
python3 run_exp2c_flush_sensitivity.py \
    --duration 60 \
    --probes-per-pass 5 \
    --output-json ./exp2c.json
```

Useful flags:

| Flag | Meaning |
| ---- | ------- |
| `--probes-per-pass N`  | Dumps to integrity-probe per pass (default 5). |
| `--cooldown-between-passes SEC` | Idle between passes (default 5). |
| `--duration SEC`       | Per-pass duration (default 60). |

Output JSON schema:

```text
experiment:  "2c_flush_sensitivity"
config:      { interval_ms, ram_size_mb, producer_has_flush_toggle, ... }
passes.flush_on:  { mean_host_snapshot_cycle_sec, integrity: {all_ok, probes: [...]}, ... }
passes.flush_off: { ... }
comparison:  { mean_host_snapshot_cycle_sec: { on, off, delta, ratio }, integrity_on, integrity_off }
recommendation: "REMOVE the flush — ..." | "KEEP the flush — ..." | "NEUTRAL — ..."
notes:       [ ... ]
```

The headline is the `recommendation` field:

- **REMOVE** if `flush_off` dumps stay intact and host_dt drops by ≥ 0.3 s.
- **KEEP** if any `flush_off` dump fails integrity.
- **NEUTRAL** if host_dt difference is within noise.

If REMOVE, follow up with a one-line patch removing the
`sleep 0.5` from the producer (now guarded by `TIMING_NO_FLUSH`).

## Order of execution

Recommended sequence in a single Linux session:

```sh
# 1. Confirm bc fix still healthy with a quick sanity (30 s)
python3 run_timing_instrumentation_experiment.py \
    --duration 30 --output-json ./sanity_post_fix.json

# 2. 2a — consumer isolation (~2 min)
python3 run_exp2a_consumer_isolation.py --duration 60 \
    --output-json ./exp2a.json

# 3. 2c — cheap (~2 min); needs consumer already killed by 2a, fine to chain
python3 run_exp2c_flush_sensitivity.py --duration 60 \
    --output-json ./exp2c.json

# 4. 2b — interval sweep (~5 min default; longer if --ram-sweep)
python3 run_exp2b_interval_sweep.py --duration 60 \
    --intervals 100,250,500,1000 \
    --output-json ./exp2b.json
```

Then ship the four JSON files back for analysis.

## What to send for analysis

Per sub-test, **one JSON file** (the `--output-json`):

- `exp2a.json` — consumer-on vs consumer-off timing comparison.
- `exp2b.json` — interval × ram cells + best/worst pause-fraction.
- `exp2c.json` — flush comparison + recommendation.

The scripts also write per-pass `snapshot_timings.jsonl` and
`producer.log` under each `--workdir` if deeper inspection is needed.

## Safety

- All three scripts only kill processes matching
  `capture_consumer_qemu.sh` (`pgrep -f`); they never touch other
  PIDs.
- Queue cleanup is restricted to `<queueDir>/pending` and
  `<queueDir>/processing`; nothing else.
- Dump cleanup removes only files matching `memory_dump-*.raw` under
  `imageDir` with mtime ≥ the run start (or per-cell start for 2b);
  older dumps are never touched.
- `--keep-dumps` preserves everything for forensic inspection.
- The original `config_qemu_upc.json` is never modified. Each script
  writes its own snapshot copy under `--workdir`.
- The producer's `TIMING_NO_FLUSH` guard is opt-in via env-var — does
  not affect any other capture path when unset.
