# Capture preflight QA

Catches, in minutes, the failures that otherwise surface hours into a campaign.

Every check here exists because the corresponding failure actually happened
during the 2026-08-09 campaign. Calibration test: run against that campaign's
original steps file and guest facts, the analyzer flags **9 of the 9** steps
that really failed, plus four more that "succeeded" while producing invalid
data.

## Run it

```bash
export SSH_TARGET=kali@192.168.222.63
export SSH_KEY=/project/homes/jeries/.ssh/id_ed25519
export ZSTD_DIR=/project/homes/jeries/memory_traces/zstd_local

./qa/preflight.sh                 # static + environment checks, ~1 min
./qa/preflight.sh --smoke         # also runs every workload briefly, ~15 min
./qa/preflight.sh --json qa.json  # machine-readable report
```

Exit `0` = ready (warnings allowed), `1` = do not start a campaign.

## What each check covers

| check | catches | seen as |
|---|---|---|
| `10_host` | missing zstd/virsh, bad config, host disk full, leftover 1 GiB dumps | disk fills mid-run |
| `20_guest` | guest unreachable, **scratch on tmpfs**, unbuilt binaries | exit 127; contaminated IO signal |
| `30_steps` | working set > guest RAM, payload > scratch free space, chain-identity collisions | `mmap` failure exit 1; `ENOSPC` |
| `40_smoke` | anything only observable by running it | segfaults, missing runtimes, bad flags |
| `50_cleanliness` | orphaned producers, paused domain, non-empty queue, prior-run residue | SSH flapping between steps; mixed data |

## The three contamination sources

The request that prompted this system named three, and they map to checks:

1. **From the test itself** — `30_steps` (declared sizes vs capacity) and
   `40_smoke` (empirical).
2. **From previous runs** — `50_cleanliness` (queue, guest scratch, stray
   processes, leftover dumps, mixed `run_id`s in the retention tree).
3. **From the system** — `10_host` and `20_guest`. The subtlest instance: the
   guest mounts `/tmp` as tmpfs sized at 50% of RAM. Workloads pointed there
   have their file writes land in RAM, so the host-side page-change signal
   records file I/O as memory activity — an `IO`-family workload silently
   measures memory instead. Nothing in the workload or the orchestrator is
   wrong; only a mount-type probe finds it.

## Adding a check

Drop `qa/checks/NN_name.sh` into place. It is sourced with `lib/common.sh`
already loaded, so it can call:

- `pass "..."`, `warn "..."`, `fail "..."`, `info "..."`, `section "..."`
- `gssh CMD` — run on the guest (bounded timeout)
- `cfg KEY` — read the capture config JSON
- `probe_guest_dir DIR` — `type=… avail_mb=… source=… writable=…`
- `load_facts` — `GUEST_RAM_MB`, `GUEST_AVAIL_MB`, `GUEST_SWAP_MB`, `GUEST_KERNEL`

No registration; the runner globs `checks/[0-9][0-9]_*.sh` in order. A check
must not `exit` — call `fail` and return, so later checks still report.

## Calibration

`analyze_steps.py` thresholds are derived from observed behaviour on the
964 MB guest, not from theory:

- `GUEST_OS_RESERVE_MB = 300` — budget is *total* RAM minus this, because the
  orchestrator reboots the guest before every step, so a workload meets a
  freshly booted system rather than whatever `MemAvailable` says at probe time.
  Observed: 384 MB ran fine (94 snapshots), 512 MB fine, 1024 MB swapped
  heavily, 2048 MB failed outright.
- `SCRATCH_HEADROOM_FRACTION = 0.80` — leave room for the filesystem and any
  concurrent writer.

If the guest's RAM or disk layout changes, re-check these against a real run
before trusting the verdict.

## Known gap

`40_smoke` shortens `--duration` and redirects scratch into a throwaway
sandbox, so it proves a workload *starts and exits cleanly* — not that it
behaves identically at full duration and full size. A workload that only fails
after 60s of allocation growth will pass smoke. Static analysis in `30_steps`
covers the size dimension; the two are complementary, neither is sufficient
alone.
