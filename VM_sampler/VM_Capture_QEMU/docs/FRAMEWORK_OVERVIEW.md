# Framework overview · tests, plans, rounds, bugs

**Purpose of this document:** orient the reader on the four kinds of moving parts in this investigation and on what "blocked / unblocked" means. Read this before any of the round-by-round results documents.

---

## Why this needs explaining

This investigation has four distinct kinds of moving parts. They look similar from outside but do completely different things. Conflating them is the most common way to get lost in the timeline.

Concretely: a **test** is not a **plan**, a **plan** is not a **round**, and a **bug** is not any of those. Each has its own role in the loop.

---

## The four actors

### 1. Tests — the measurements

Tests are scripts that **observe** the pipeline. They run the producer, collect timestamps, and write a JSON. **They do not change the pipeline.** They make defects visible; they do not fix them.

There are four tests in this investigation:

- **`sanity`** — `run_timing_instrumentation_experiment.py`. One instrumented producer run, 30 s, idle Kali VM. Answers: *"does the producer emit snapshots at the rate `intervalMsec` claims?"*
- **`2a`** — `run_exp2a_consumer_isolation.py`. Two producer runs (consumer-on, consumer-off). Answers: *"how much of the host-side variance is the consumer's fault?"*
- **`2b`** — `run_exp2b_interval_sweep.py`. Four producer runs across `intervalMsec ∈ {100, 250, 500, 1000}` ms. Answers: *"what is the pause-fraction cost at each interval?"*
- **`2c`** — `run_exp2c_flush_sensitivity.py`. Two producer runs (flush-on, flush-off). Answers: *"is the 0.5 s flush sleep in the producer necessary?"*

A test produces a JSON. The JSON is the artifact. No JSON, no claim.

### 2. Bugs and mechanisms — what tests find

- **Mechanisms i–vii** — theoretical hypotheses for why the classifier confused workload pairs it should have separated cleanly. Each one is something the analysis chain could be doing wrong. Numbered with lowercase Roman to keep them visually distinct from plans (Arabic) and bugs (uppercase Latin).
- **Engineering bugs D, E, F, G, H** — concrete defects discovered along the way. Lettered uppercase Latin. Not "mechanisms" because they are not hypotheses about analysis; they are observed failures in the orchestration code.

Tests do not fix bugs. They make bugs measurable. Once measurable, a bug gets a plan.

### 3. Plans — the code patches

A plan is a code change targeting one or more bugs. Each plan is one commit; each plan is small (typically 1–34 lines); each plan is verifiable by re-running the relevant test(s).

| Plan | Targets | What it does |
| ---- | ------- | ------------ |
| 1 | vi | Producer gets `TIMING_SELF_CLEAN` env-var. When set, the producer deletes the previous dump in-process. |
| 1b | vi (propagation) | Orchestrators 2a / 2b / 2c export `TIMING_SELF_CLEAN=1` by default. |
| 1c | E | Sanity script gets the same self-clean default plus pre-run queue drain. |
| 2 | A (cross-pass contamination) | *Superseded* by Plan 1 — would have manually purged between passes; Plan 1 makes that unnecessary. |
| 3 | vii | 2c bases the verdict on a stable-subset (first-N) mean instead of all-N mean. |
| 4 | D | `disk_free_check` takes `peak_concurrent_dumps` so the free-space requirement reflects the actual cleanup policy. |
| 5 | F | 2c overrides Plan 1b default — turns self-clean OFF so the integrity probe finds dumps to inspect. |
| 6 | G | 2c `--stable-n` default flipped 10 → 5 so the comparison window predates the dirty-page drift Plan 5 re-introduced. |
| Bug-H fix | H | `mkdir -p` parent dir of `--output-json` in 2b + 2c. |

`Plan 02` (with a leading zero, two digits) is a **different thing**: a higher-level experiment, not a fix. It is the per-family `intervalMsec` profile — the actual thesis-contribution experiment. The single-digit plans (1, 1b, 1c, 3, 4, 5, 6) fix the capture pipeline so Plan 02 can run reliably.

### 4. Rounds — the verification cycles

A round is a full re-run of relevant tests after one or more plans land. Round N tests the pipeline as it stands after plans #1 through #N have committed.

| Round | Tests run | What landed first | What it found |
| ----- | --------- | ----------------- | ------------- |
| Run 1 | sanity | — | mechanisms i, ii (bc missing) |
| Run 2 | sanity | bc installed | i, ii FIXED empirically |
| R1 | 2a + 2b + 2c | — | v (perm), vii (probe logic), B (queue) |
| R2 | 2a + 2b + 2c | v fix, probe rewrite, purge | iii FIXED, surfaced vi + vii nuance |
| R3 | sanity + 2a + 2b + 2c | Plans 1·1b·3·4 | vi, vii, D FIXED · surfaced E, F |
| R4 | sanity + 2c | Plans 1c·5 | E, F FIXED · surfaced G |
| R5 | 2c | Plan 6 + Bug-H fix | G FIXED · H FIXED · **REMOVE verdict locked** |

A round is **not** a new investigation. It is a verification step. The pipeline is treated as a black box; we check whether the JSONs come back the way the plans predicted.

---

## The cycle, in one sentence

> **Test exposes bug → bug gets a plan → plan is coded → round re-runs the test → either the bug is closed or a new one surfaces.**

When all rounds come back green, the higher-level experiment that was waiting on the pipeline is *unblocked*.

---

## What "blocked" and "unblocked" actually mean

Higher-level experiments depend on lower-level reliability.

**Plan 02** — the per-family `intervalMsec` profile — is the thesis contribution. It will pick the smallest `intervalMsec` per workload family such that each family's defining rhythm fits at least 4 periods inside the analyser's 128-sample window.

Plan 02 was **blocked** because four lower-level things had to be true for its conclusions to be trustworthy:

1. **`intervalMsec` must actually do something.** Pre-bc-fix the producer was sleeping 0 seconds regardless of the configured interval. (Closed by the bc install — mechanisms i, ii.)
2. **Timing must be stationary across a pass.** If `pmemsave` drifts 0.77 s → 4.7 s over a 21-snap pass, then "host Δt" has no well-defined value. (Closed by Plan 1 — mechanism vi.)
3. **The pause-fraction sweep must produce trustworthy numbers.** Without 2b reporting clean per-interval pause fractions, the per-family choice is unfounded. (Closed by R2 — mechanism iv reframed as a design knob.)
4. **The sanity script must run cleanly.** Without it we cannot re-verify the pipeline quickly when something changes. (Closed by Plan 1c — Bug E.)

Without those four, Plan 02 would pick interval values based on noise. Its conclusions would be artifacts of the capture pipeline, not properties of the workloads.

All four prerequisites are now satisfied. Plan 02 is **unblocked** — meaning the next experiment can run, and its conclusions will measure what the experiment claims to measure, not the pipeline's defects.

**"Blocked"** means: prerequisite chain unsound. **"Unblocked"** means: prerequisite chain structurally verified.

---

## The illustration

```
                                                              ┌─────────────────────┐
   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │  HIGHER-LEVEL       │
   │   TESTS      │    │    BUGS &    │    │    PLANS     │   │  EXPERIMENT         │
   │  (measure)   │──► │  MECHANISMS  │──► │   (fix)      │──►│  (unblocked when    │
   │              │    │  (revealed)  │    │              │   │   plans land)       │
   ├──────────────┤    ├──────────────┤    ├──────────────┤   ├─────────────────────┤
   │  sanity      │    │  i  bc         │  │  1   TIMING_   │ │                     │
   │  2a          │    │  ii bc         │  │      SELF_CLEAN│ │     Plan 02         │
   │  2b          │    │  iii consumer  │  │  1b  orch env  │ │     · per-family    │
   │  2c          │    │  iv  pause     │  │  1c  sanity    │ │       intervalMsec  │
   │              │    │  v   cleanup   │  │  3   stable-n  │ │     · the thesis    │
   │              │    │  vi  drift     │  │  4   disk-free │ │       result        │
   │              │    │  vii verdict   │  │  5   2c inverts│ │                     │
   │              │    │  D   disk-pess │  │  6   n=10→5    │ │                     │
   │              │    │  E   sanity    │  │  Bug-H mkdir   │ │                     │
   │              │    │  F   probe     │  │                │ │                     │
   │              │    │  G   coupling  │  │                │ │                     │
   │              │    │  H   parent    │  │                │ │                     │
   └──────┬───────┘    └──────────────┘    └──────────────┘   └─────────────────────┘
          │                                                              ▲
          │                                                              │
          └───────  ROUNDS = re-run all relevant tests · verify  ────────┘
                    Run 1 → Run 2 → R1 → R2 → R3 → R4 → R5
```

Read top-to-bottom: tests find bugs; bugs get plans; plans land; rounds verify; when all bugs in the dependency cone are closed, the higher-level experiment unblocks.

---

## Vocabulary cheat-sheet

| Word | What it is | Concrete example |
| ---- | ---------- | ---------------- |
| **test** | measurement script that observes the pipeline | `run_exp2c_flush_sensitivity.py` |
| **pass** | one run of a test (or one sub-run of an A/B test) | "2c pass A = flush_on" |
| **bug** or **mechanism** | a defect the tests reveal | mechanism vi = in-pass drift |
| **plan** | a code patch closing one or more bugs | Plan 1 = `TIMING_SELF_CLEAN` env-var |
| **round** | a re-run of relevant tests after plans land | Round 3 = first run on the Plan-1·1b·3·4 pipeline |
| **blocked** | a higher-level experiment cannot trust its inputs yet | "Plan 02 was blocked through Round 2" |
| **unblocked** | all prerequisites are met; higher-level experiment can run | "Plan 02 unblocked after Round 5" |
| **Plan 02** (note leading zero) | a future higher-level experiment, not a fix | per-family `intervalMsec` profile |

---

## How to use this when reading the rest of the docs

When you encounter:

- **A test name** (`sanity` / `2a` / `2b` / `2c`) → think *"what observation does this make?"*
- **A bug ID** (mechanism iii, Bug E, …) → think *"what defect did some test reveal?"*
- **A plan number** (Plan 1, Plan 6, …) → think *"what code change closes one or more bugs?"*
- **A round** (Round 3, Round 5, …) → think *"all relevant tests re-run after plans up to here landed."*
- **"Plan 02 unblocked"** → think *"the per-family `intervalMsec` thesis-contribution experiment can run because its capture-pipeline prerequisites are all met."*

If you keep these four lanes separate in your head, the rest of the documentation reads as a clean story rather than an alphabet soup.
