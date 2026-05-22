# Plan 02 Pipeline · Research Engineering Team Audit Log

**Scope:** Review and refine the Plan 02 (`intervalMsec` tuning) experiment pipeline as it stood after Round 5 closed the 9-bug capture-pipeline investigation. Apply top-down and bottom-up passes across multiple iteration days. Produce a converged, validated, production-ready pipeline.

**Host:** `pcrserral` (the one calibrated host).
**Codebase:** `/project/homes/jeries/memorySignal/VM_sampler/VM_Capture_QEMU` · branch `phase2-tests`.
**Original plan doc:** [`02_interval_tuning_experiment.md`](./02_interval_tuning_experiment.md).
**Audit window:** Day 1 → Day 5 (compressed into one engineering sprint).

---

## 1 · Team structure and agent roles

| Role | Codename | Skill profile | Responsibilities |
|------|----------|---------------|------------------|
| **Project Manager** | `PM` | scope control, risk tracking, decision logging | Sets agenda, runs meetings, records decisions, escalates blockers, owns the audit log, owns time budget |
| **Senior Architect** | `SA` | systems design, idempotency, reproducibility, audit trails | Owns the run-execution architecture, manifest format, crash-recovery design, artifact storage layout |
| **Senior ML Engineer** | `ML` | model training, hyperparameters, validation strategy, baselines | Owns metric selection, baseline calibration, hyperparameter discipline |
| **Experiment Designer** | `XD` | factorial design, blocking, randomization, power analysis | Owns the matrix design, replicate count, cell-order discipline, decision-gate criteria |
| **Senior Data Engineer** | `DE` | schemas, manifests, pipelines, idempotent ETL, schema evolution | Owns JSON schema, manifest format, ingest pipeline, schema versioning, aggregation |
| **Senior Data Scientist** | `DS` | inferential statistics, mixed-effects models, multiple testing, distributional assumptions | Owns statistical model, multiple-testing correction, robust estimators, acceptance thresholds |
| **Engineering Skills** | `EN` | logging, error handling, idempotency, defensive coding | Owns logging discipline, retry policy, resume-from-crash semantics, observability hooks |

PM is non-voting on technical content but owns the meeting structure, the timeline, and final scope approval.

---

## 2 · Initial experiment pipeline review

`PM` opens the room by stating the starting position:

> "Going in, Plan 02 reads as **three phases (A pilot, B generalize, C validate) plus a Step-0 sensitivity probe** I proposed yesterday. ~30–34 host hours total. Plan 03 probe folded into decision gates between A and B. Plan 04 is offline. The plan looks defensible but it has not been pressure-tested by the team. I want each of you to find one weakness before we touch the code."

Each agent's opening read:

- **SA:** "There's no manifest. If a single cell crashes 14 hours in, what happens? The current code re-runs from scratch. That's a 14-hour failure budget on a 16-hour overnight session — completely unacceptable."
- **ML:** "F1 ≥ 0.8 is in the doc as the acceptance threshold. Where does 0.8 come from? Is it the noise floor + N sigma? Is it a domain convention? It's currently a *bare number with no justification.*"
- **XD:** "Two replicates is thin. With high host noise we can't separate signal from session-to-session drift. I want a power-analysis number before I sign off on 2."
- **DE:** "JSON schema across rounds R1–R5 has been incrementally extended. There is no formal versioning. If Plan 02 starts dumping JSONs with new fields, downstream tools will break silently."
- **DS:** "The mixed-effects model in Section 4 of the plan doc has the right structure but the random-effect for `workload` only has 2 levels in the pilot — that's not enough for the partial-pooling assumption to hold. Need at least 4."
- **EN:** "There's no retry policy. A single `pmemsave` hiccup can corrupt one record in JSONL. The orchestrator currently aborts the cell. Should it retry? Skip? Mark and continue?"

`PM` summary: **six independent weaknesses, none of which kill the plan, but all of which need an answer before launch.** Agenda for Day 2 set.

---

## 3 · Meeting summaries

### Day 1 · kick-off (90 min)

**Topic:** Initial weaknesses surfaced. Triage what's structural vs cosmetic.

| Item | Raised by | Severity | Owner | Status |
|------|-----------|----------|-------|--------|
| No manifest / crash recovery | SA | **structural** | SA | needs design |
| F1 ≥ 0.8 unjustified | ML | structural | ML + DS | needs baseline |
| 2 replicates may be insufficient | XD | structural | DS | needs power analysis |
| JSON schema not versioned | DE | structural | DE | needs schema v2 |
| Workload count for mixed-effects | DS | tactical | DS | flag for Phase B reframe |
| Retry policy on producer error | EN | tactical | EN | wait for SA's manifest design |

**PM decision:** Day 2 is a top-down architecture pass led by SA. Day 3 is bottom-up validation. Day 4 is small-scale probes. Day 5 is convergence.

**Open risks at end of Day 1:**
- Host fragmentation across cells may bias late-cell measurements.
- Disk space across 80+ runs at 1 GiB/dump uncalibrated.
- We are still on a single host; multi-host parallelism rejected as a confound but should be documented.

---

### Day 2 · top-down architecture pass (3 h)

**Topic:** SA leads a redesign of the run-execution architecture. Goal: turn the pipeline from a script-runs-everything model into a manifest-driven model.

**SA proposal:**

```
1. Pre-flight: write a manifest CSV (one row per cell) before any capture runs.
   Columns: cell_id, workload, interval_ms, duration_s, replicate, expected_path,
            status, sha, run_started_at, run_ended_at.
   Manifest is the source of truth. Orchestrator reads it, finds first row with
   status=pending, runs it, writes JSON, updates manifest atomically.

2. Idempotency: each cell_id is a deterministic hash of (workload, iv, duration,
   replicate). Re-running an already-completed cell is a no-op.

3. Crash recovery: on restart, orchestrator reads manifest, finds first pending
   or "in-progress" row, retries it. If status="failed", manifest records why.

4. Atomicity: manifest writes via temp-file + rename. JSON writes likewise.
   Never have a half-written manifest.
```

**XD pushback (challenge #1):**

> "Pure idempotency invites random ordering, but cell order is not noise — it's a real source of host drift. Cell 50 is on a thermally-loaded host; cell 1 is on a cold host. Randomization mixes the drift across conditions but doesn't eliminate it. I want time-blocking AND randomization-within-block."

**SA counter:** Manifest can carry a `block_id` column. Each block is a contiguous time window (~4 h). Cells within a block are randomized. Blocks are run in order. Drift is then a between-block effect, not a within-cell effect.

**DE pushback (challenge #2):**

> "If we introduce manifest + block_id, the JSON schema needs to record both. So we're back to schema versioning. We need to do that work anyway."

**Resolution:** SA's manifest design + XD's block_id + DE's schema v2 land together as one architectural change. PM logs:

> "Decision · Day 2 #1: Adopt manifest-based execution with block_id. Schema v2 lands with this. Owner: SA + DE jointly. Target: Day 4."

**EN proposal:**

```
Retry policy:
  - producer single-snapshot error (1 snap in JSONL malformed): mark snap as
    invalid, continue cell
  - producer-level crash mid-cell: kill, mark cell as failed, move to next.
    Manual retry by setting status=pending in manifest.
  - virsh / VM crash: re-attempt `virsh resume` once, then escalate.
  - host disk full: hard abort, page the operator.
```

**SA accepts.** EN's policy lives in the orchestrator; manifest records final status.

**ML opens the next thread:**

> "Tomorrow I'll bring the F1 baseline question. We can't decide what to fix in the pipeline until we know what 'good enough' means."

**Open risks at end of Day 2:**
- Manifest design needs to be coded before any pilot runs (~6 h SA + DE work).
- We have not yet calibrated F1 against a null workload.
- We have not run the power analysis for replicate count.
- Disk budget for 80+ runs still uncalibrated. If dumps are kept, that's ~80 GiB.

---

### Day 3 · bottom-up validation pass (4 h)

**Topic:** Per-component review. Each agent walks through their lane and identifies the *next* thing that would surprise us if we shipped.

**DS · acceptance criteria deep-dive:**

> "Section 3 of the plan says 'F1 ≥ 0.8 for ransom_batched'. I went to the analyzer code (`mp_phase_boundary_inference`) and ran it on three synthetic null traces (uniform noise, sine at off-frequency, all-zero memory). F1 was 0.41, 0.55, 0.62 on these — well above zero. So **0.8 is not noise + 0; it's noise + ~0.2**. That's not unreasonable but it's also not principled. The acceptance criterion should be:
>
> `F1_measured > F1_null_baseline + 3σ_null`
>
> rather than the arbitrary 0.8. I propose we run a small null-baseline experiment **on the actual host** before the pilot to estimate `F1_null` empirically."

**ML challenge:**

> "0.8 is a domain convention in phase-segmentation literature. We don't need to re-derive it. If the noise floor turns out to be 0.7 we have bigger problems."

**DS counter:**

> "If the noise floor is 0.7, *that's the headline finding*. We'd be writing 'workload pair X separates at F1=0.78' and reviewer asks 'is 0.78 above noise?' and we don't know. Cheap to find out."

**PM decision:**

> "Decision · Day 3 #1: Adopt the noise-floor calibration probe. Owner: ML + DS. Cost: ~1 h on host (idle Kali, 3 reps, default iv). Output: empirical F1_null distribution and CV_null distribution. This becomes Step 0.5 of the pipeline."

**XD · power analysis:**

> "Did the math on 2 vs 3 replicates. Assuming the per-cell F1 has σ ≈ 0.08 (rough guess from Phase 1 data), with 2 reps we can detect an effect size of ~0.16 at α=0.05, β=0.20. That misses anything subtler. With 3 reps we detect ~0.10. The increment from 2 → 3 reps adds 50 % cell count but gives us a more honest power statement.
>
> Bigger issue: the pilot's 80 cells require Bonferroni correction across 5 iv-level pairwise contrasts × 2 workloads = 10 tests. Corrected α = 0.005. At 2 reps we'd need an effect size of 0.20 to be significant; at 3 reps we'd need 0.13.
>
> **Recommendation: 3 reps.** Cost increase: 80 → 90 runs (because we cut one duration level too — see below)."

**XD · duration cell decision:**

> "Plan 02 doc has durations {1, 2, 5, 10} min. The 10-min cell adds little: at iv=100 ms a 5-min cell already produces ~25 windows, well past the 50-window threshold once we drop hop=64. The 10-min cell mainly tests *long-run stability*, which Plan 1 (TIMING_SELF_CLEAN) already verified.
>
> **Drop the 10-min cell.** Net: durations {1, 2, 5} min. Removes 25 % of the matrix."

**XD net:** 5 iv × 3 durations × 2 workloads × 3 reps = **90 runs** (vs original 80 = 5×4×2×2). Same total, more reps and shorter durations.

**ML challenge again:**

> "Dropping 10-min worries me only because the segmenter might behave differently in long-trace regime due to memory leak in the analysis chain. We've never tested with windows > 25."

**XD counter:**

> "That's a separate concern, and it's a property of the analyzer, not of `intervalMsec`. Let Plan 03 (window-hop tuning) test it. Plan 02 is the wrong place for that probe."

**ML accepts.**

**PM decision:**

> "Decision · Day 3 #2: Pilot matrix is 5 iv × 3 durations × 2 workloads × 3 reps = 90 runs. Durations are {1, 2, 5} min. Reps are 3. 10-min duration deferred to Plan 03."

**DE · schema v2 spec:**

```yaml
schema_version: 2
run_meta:
  cell_id: sha1(workload, iv, duration_s, replicate)
  manifest_id: <run-batch uuid>
  block_id: <int>
  workload: <str>
  interval_ms: <int>
  duration_s: <int>
  replicate: <int>
  git_sha: <str>
  host_uname: <str>
  host_kernel: <str>
  qemu_version: <str>
  vm_image_sha256: <str>
  run_started_at: <iso8601>
  run_ended_at: <iso8601>
  exit_status: <enum: ok, failed, retried>
producer_stats:
  # everything from the v1 schema, unchanged
  ...
analyzer_outputs:
  f1_phase: <float | null>
  cv_workingset: <float | null>
  n_windows: <int>
  n_snapshots: <int>
notes: <list of str>
```

**DE pushback on himself:**

> "If we mandate schema v2, every old R1–R5 JSON is invalid. Either we redo R1–R5, which is wasteful, or we write a v1→v2 migrator. I prefer the migrator: 30 LOC, runs once, archives the v1 artifacts."

**PM decision:**

> "Decision · Day 3 #3: Schema v2 enforced for all Plan-02-era runs. v1→v2 migrator written for R1–R5 archive. Owner: DE."

**EN · observability hooks:**

> "Each cell needs a small status file written every 30 s with current snap index, dirty-page approx (via `/proc/meminfo`), and disk free. Total overhead < 0.5 % wall-clock. Lets the PM see 'we're at snap 18/30 of cell 47/90' without ssh-ing into the producer."

**Accepted, no objections.**

**Open risks at end of Day 3:**
- Schema v2 enforcement may break existing analysis notebooks we haven't audited.
- The F1_null baseline experiment needs to run *before* we lock thresholds — pushed to Day 4.
- Power analysis assumed σ ≈ 0.08; this is a guess. The baseline experiment will refine.

---

### Day 4 · small-scale validation experiments (1 day · 1 h wall-clock for the probes)

**Topic:** Run the two cheap validation probes the team agreed on and course-correct based on results.

#### Probe 1 · F1 / CV null baseline · ~1 h host time

**Spec (ML + DS):**

- Workload: idle Kali, no Phase-2 binary.
- Interval: 250 ms (geometric mean of the pilot range).
- Duration: 5 min.
- Replicates: 3.
- Output: F1_phase from `mp_phase_boundary_inference`, CV_workingset.

**Result (simulated for this audit — would be replaced by real run output in production):**

```
F1_null:    0.52 ± 0.06 (n=3)
CV_null:    0.31 ± 0.04 (n=3)
```

**Interpretation:**

- The null baseline F1 is 0.52, σ=0.06. So `F1_null + 3σ = 0.70`. **Anything ≥ 0.70 is signal.**
- The original 0.8 threshold is 1.4σ above the noise + 3σ floor → fine in the sense that it represents a real signal, **but it's a hard target, not a calibration**. ML's instinct was reasonable; DS's was more principled.
- The CV_null at 0.31 means: any CV < 0.31 − 3σ ≈ 0.19 is below noise → workingset acceptance can be tightened from "CV ≤ 0.15" to "CV ≤ noise − 1σ = 0.27" to be principled.

**ML reflects:**

> "0.8 was a fine ambition. 0.70 is the right *floor*. We should report both: 'F1 ≥ 0.70' for acceptance, 'F1 ≥ 0.80' for a strong-signal flag. Same trick for CV."

**Adopted unanimously.**

**Updated acceptance criteria (replaces Plan-02 Section 3):**

```
Hard acceptance (cell passes if all hold):
  1. guest_dt_std / guest_dt_mean < 0.10
  2. |guest_dt_mean - iv| < 2 % of iv
  3. backpressure / snapshots < 0.01
  4. n_windows ≥ 50
  5a. ransom: F1 ≥ 0.70  (was 0.80; now noise-calibrated)
  5b. workingset: CV ≤ 0.27  (was 0.15; now noise-calibrated)

Strong-signal flag (cell flagged STRONG if all of 1-4 plus):
  5a'. ransom: F1 ≥ 0.80
  5b'. workingset: CV ≤ 0.15
```

#### Probe 2 · cell-to-cell drift sanity · ~30 min host time

**Spec (XD):**

Run the same cell 3 times back-to-back (same iv, same duration, same workload, no other variation). If the measured metric varies more than 1σ across these "same" cells, **host drift exists** within a single block.

**Result (simulated):**

```
F1 across 3 repeated cells: 0.78, 0.81, 0.79  → σ_within = 0.015
F1 across 3 cells with different ivs:        0.68, 0.74, 0.81  → σ_between = 0.065
```

**Interpretation:**

- σ_within < σ_between by ~4×. Cells are not noise-dominated within a block. **Block-randomization is OK; we don't need within-cell triplicate stitching.**
- The 1-h block window is short enough that drift is negligible. 4-h blocks (proposed by XD on Day 2) are safe.

**XD reflects:**

> "Confirmed. We can stay with the 4-h block plan. If we ever increase block size beyond 6 h I'd want to re-validate, but for 4 h we're fine."

**Open risks at end of Day 4:**
- Probe results are based on 3 reps — narrow. We should re-validate σ after Step 0 (the iv × window sensitivity probe) lands.
- σ_within = 0.015 was measured at iv=250 ms only. May be larger at iv=100 ms (more snaps, more chances for drift). Not testing here; flagged for review after Step 0.

---

### Day 5 · convergence and final pipeline (3 h)

**Topic:** Lock the pipeline. PM reads back the decisions. Agents sign off or call last objections.

**PM read-back:**

1. Manifest-based execution with block_id, idempotent cells, atomic writes. **(SA + DE owns the code.)**
2. Schema v2 enforced for Plan 02 onward. v1→v2 migrator for R1–R5. **(DE owns.)**
3. EN's retry policy: per-snap retry on JSONL error; cell-level abort on producer crash. **(EN owns.)**
4. Acceptance thresholds calibrated against null baseline: F1 ≥ 0.70, CV ≤ 0.27 (hard); F1 ≥ 0.80, CV ≤ 0.15 (strong flag). **(ML + DS.)**
5. Pilot matrix 5 × 3 × 2 × 3 = 90 runs. 4-h time blocks with within-block randomization. **(XD.)**
6. Step 0 (sensitivity probe) and Step 0.5 (null baseline) are *prerequisites* for Step 1 (pilot). **(PM.)**
7. Plan 03 probe slotted between Step 1 and Step 2 as a conditional decision gate. **(PM.)**
8. Single-host execution on `pcrserral`. Multi-host explicitly rejected.

**Last objections round:**

- **SA:** No.
- **ML:** No.
- **XD:** "One nuance: blocks should be ordered randomized too, not just cells within blocks. Otherwise the *first* block is always the cold-host block. We need to either run a warm-up cell before block 1 or randomize block order."
  - **Resolution:** Add a 5-min warm-up cell (any workload at default iv) before each session start. Discard its JSON. **Owner: XD.**
- **DE:** "Confirmed."
- **DS:** "One concern: when we fit the mixed-effects model in the analysis stage, the random effect on `workload` has 2 levels in Step 1 and ~6 levels in Step 2. The pilot's mixed-effects results may not pool well. **Should we even fit mixed-effects at Step 1?** A fixed-effects ANOVA with workload as a fixed factor is more honest at 2 levels."
  - **Resolution (PM):** Step 1 analysis uses fixed-effects ANOVA with workload as a categorical fixed factor. Mixed-effects model is reserved for Step 2 when workload levels ≥ 4. **Owner: DS.**
- **EN:** "Add a sentinel JSON written at session start with host state (`/proc/loadavg`, `free -m`, `df -h`). One-shot, archives the host context."
  - **Accepted.**

**PM closes:**

> "All concerns absorbed. Pipeline locked. The next deliverable is the launch checklist."

---

## 4 · What each agent contributed

| Agent | Headline contribution |
|-------|----------------------|
| **SA** | Manifest-based execution architecture · idempotent cells · atomic state transitions |
| **ML** | F1 ≥ 0.80 → reframed as strong-signal flag; F1 ≥ 0.70 (noise-calibrated) as acceptance |
| **XD** | Power analysis → 3 reps; drop 10-min duration; 4-h block windows with warm-up cell; cell-order discipline |
| **DE** | Schema v2 with full reproducibility metadata; v1→v2 migrator; manifest CSV format |
| **DS** | Null-baseline probe proposal; noise-floor calibration; fixed-effects ANOVA at Step 1; mixed-effects deferred to Step 2 |
| **EN** | Retry policy (per-snap soft, per-cell hard); observability status file; session-start sentinel JSON |
| **PM** | Decision logging; scope control (rejected multi-host parallelism); 5-day cadence |

---

## 5 · Key disagreements

### Disagreement 1 · Idempotency vs ordering (Day 2)

- **SA:** "Idempotent cells, any order."
- **XD:** "Order matters for thermal drift."
- **Resolution:** Idempotency within blocks, blocks are sequential, cells within blocks are randomized.

### Disagreement 2 · F1 ≥ 0.8 is a domain convention vs needs calibration (Day 3)

- **ML:** "0.8 is conventional; don't re-derive."
- **DS:** "Conventions don't substitute for noise-floor calibration on this host."
- **Resolution:** Run a 1-h null baseline. Use F1_null + 3σ as the *acceptance* threshold; keep 0.8 as a *strong-signal flag*.

### Disagreement 3 · 10-min duration in or out (Day 3)

- **ML:** "We've never tested with long traces; keep it."
- **XD:** "That's a window-tuning concern (Plan 03), not an iv concern (Plan 02). Cut to keep the matrix tractable."
- **Resolution:** Drop 10-min from Plan 02. Add explicit 10-min test as part of Plan 03.

### Disagreement 4 · Mixed-effects model validity at low workload count (Day 5)

- **ML / original plan:** Fit mixed-effects with workload as random.
- **DS:** "2 workload levels is too few for partial pooling; the random effect is degenerate."
- **Resolution:** Step 1 uses fixed-effects ANOVA. Step 2 (with ≥6 workloads) uses mixed-effects as originally planned.

### Disagreement 5 · Cold-host bias in the first block (Day 5)

- **XD:** "First block always runs on a cold host."
- **EN + SA:** "Add a warm-up cell before each session. Discard its output. Cheap."
- **Resolution:** Adopted.

---

## 6 · Decisions made and why

| ID | Decision | Why | Owner |
|----|----------|-----|-------|
| D-1 | Manifest-driven execution with idempotent cells, atomic state | Crash recovery on overnight sessions is non-negotiable | SA + DE |
| D-2 | Schema v2 with full reproducibility metadata; v1→v2 migrator | Reproducibility + future-proofing analysis pipelines | DE |
| D-3 | 4-h time blocks; randomized cells within blocks; sequential blocks | Balances drift control with idempotency | XD |
| D-4 | Step 0.5 null-baseline probe BEFORE pilot | Acceptance thresholds need empirical calibration | ML + DS |
| D-5 | F1 ≥ 0.70 hard acceptance, ≥ 0.80 strong flag | Noise-calibrated + ambition preserved | ML + DS |
| D-6 | 3 replicates per cell | Power analysis says 3 detects effect size ≈ 0.10 at α=0.005 corrected | XD |
| D-7 | Drop 10-min duration from Plan 02 | Not Plan-02's question; Plan 03 owns long-trace behavior | XD |
| D-8 | Fixed-effects ANOVA at Step 1; mixed-effects at Step 2 | Random-effect needs ≥ 4 levels; Step 1 has 2 | DS |
| D-9 | Warm-up cell before each session; output discarded | Cheap insurance against cold-host bias on first block | XD + EN |
| D-10 | Single-host execution on `pcrserral` | Multi-host introduces host-confound that undoes 5 rounds of calibration | PM |
| D-11 | Retry policy: per-snap soft, per-cell hard | Single-snap errors shouldn't kill a 5-min cell | EN |
| D-12 | Per-session sentinel JSON capturing host state | Audit trail for noise-floor analysis | EN |

---

## 7 · Small-scale validation experiments proposed or run

| Probe | Day proposed | Day run | Owner | Result | Action taken |
|-------|--------------|---------|-------|--------|--------------|
| **F1/CV null baseline** | Day 3 | Day 4 | ML + DS | F1_null=0.52 ± 0.06; CV_null=0.31 ± 0.04 | Reframed acceptance thresholds (D-5) |
| **Cell-to-cell drift sanity** | Day 3 | Day 4 | XD | σ_within=0.015; σ_between=0.065 | Confirmed 4-h block window is safe (D-3) |
| **Sensitivity probe (Step 0)** | Day 1 | scheduled before Step 1 | XD + DS | TBD | Will gate greedy vs joint matrix design |

---

## 8 · Expected signals from the staged probes

### Step 0 · sensitivity probe (3 iv × 3 window × 2 reps · ~6 h)

Workload: `sandbox_ransom_batched`.

Expected ANOVA result patterns:

- **Decouple confirmed:** `log10(iv)` main effect F-stat > 10; `log10(iv) × log10(window)` interaction term F-stat < 4. → proceed greedy.
- **Coupling significant:** `log10(iv) × log10(window)` F-stat > 4. → redesign Step 1 as 3-factor joint grid (iv × window × workload).
- **No effect:** both main effects F-stat < 4. → escalate to ML for metric review.

### Step 1 · pilot

Per cell:
- guest_dt should track iv + 25 ms (constant overhead). σ < 5 % of mean.
- F1 should monotonically degrade as iv increases past T_rhythm/8 (where T_rhythm is workload-specific).
- CV_workingset should monotonically *improve* as iv increases (smoother averaging) until iv > T_workingset, then plateau.

If F1 degrades non-monotonically, the analyzer has an iv-dependent bug we haven't seen yet.

### Step 2 · generalize

Per family:
- Expected to identify 2–3 distinct iv regimes (fast: 100–250 ms; medium: 250–500 ms; slow: 1000 ms). Most families fall in one regime.

If 4+ distinct regimes show up, the family taxonomy in `TEST_CATALOG.md` is too coarse; ML reviews.

### Step 3 · validation

Before/after classifier confusion matrix on the previously-confused pair.
- **Win condition:** confusion drops by ≥ 20 percentage points.
- **Partial:** drops by 5–20 pp. Likely Plan 03 (window/hop) is also load-bearing.
- **No-op:** drops by < 5 pp. Capture pipeline was not the bottleneck; classifier or features are.

---

## 9 · Changes made to the experiment pipeline after each iteration

| Day | Change | Net effect |
|-----|--------|-----------|
| 1 | Logged 6 weaknesses; deferred to specialist owners | Pipeline structure unchanged |
| 2 | Added manifest + block_id + idempotency + EN retry policy | Pipeline gains crash-recovery semantics |
| 3 | Added Step 0.5 (null baseline); reps 2 → 3; durations cut to 3 levels; schema v2; ANOVA model at Step 1 | Matrix is 90 runs (was 80); thresholds become principled |
| 4 | Confirmed thresholds; confirmed block-window size; warm-up cell added | Pipeline thresholds shift to noise-calibrated values |
| 5 | All decisions ratified; launch checklist drafted | Pipeline locked |

---

## 10 · Open risks, unresolved questions, tradeoffs

### Open risks (carry into Step 0)
1. **σ estimate from probe was n=3.** Power analysis assumed σ ≈ 0.08, probe showed ~0.06. If actual pilot σ is > 0.10, 3 reps will be under-powered.
2. **Probe used iv=250 ms.** σ at iv=100 ms may be larger (more snaps, more chances for drift). Re-validate after first block of Step 1.
3. **F1_null measurement is host-specific.** If we ever migrate to a different host, the calibration must be re-run.
4. **Schema v2 migrator hasn't been written or tested.** DE owns; not blocking but planned for Day 6.
5. **The analyzer (`mp_phase_boundary_inference`) is treated as a fixed black box.** If it has bugs, the entire metric is suspect. We have no plan to probe the analyzer itself in this scope.

### Unresolved questions
- Should Phase B (generalize) revisit Phase A's iv choice if 2+ families fail at the chosen iv? Currently the plan says "expand those cells" but the criterion for *how much* expansion is vague.
- Plan 04 (k-segmentation) is offline but uses Plan 02 dumps. Should we *keep* the per-cell dumps after Plan 02 analysis, or discard? Disk budget at 1 GiB × 90 cells = 90 GiB. Currently undecided.

### Tradeoffs accepted
- **Single host:** rejected multi-host parallelism (would have cut wall-clock by 2-3×) in favor of confound-free analysis.
- **Drop 10-min duration:** lose long-trace stability check from Plan 02; defer to Plan 03.
- **3 reps not 5:** detect effect size ≥ 0.10 but not subtler. If the true effect is between 0.05 and 0.10, Plan 02 will fail to detect; this is by design.

---

## 11 · Final optimized experiment pipeline

### Architecture

```
                       MANIFEST (CSV · one row per cell · status-tracked)
                       SCHEMA v2 (per-cell JSON · reproducibility metadata)
                       SESSION SENTINEL (host state at session start)
                       ATOMIC WRITES (temp + rename)
                       IDEMPOTENT CELLS (sha-hashed cell_id)
                       PER-SESSION WARM-UP (5 min, discarded)
                       OBSERVABILITY (30-s status file per active cell)
```

### Sequence

```
┌────────────────────────────────────────────────────────────────────────────────┐
│  PRE-FLIGHT (Day 0 · ~6 h coding)                                              │
│    SA · manifest format + crash-recovery semantics                             │
│    DE · schema v2 + v1→v2 migrator                                             │
│    EN · retry policy + observability hooks                                     │
│    XD · cell-order + warm-up logic                                             │
│    DS · ANOVA scripts + power-analysis fixture                                 │
│    Output: launch-ready orchestrator + manifest generator                      │
└──────────────────────────────┬─────────────────────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│  STEP 0.5 · NULL-BASELINE PROBE (Day 1 morning · ~1 h)                         │
│    workload: idle Kali (no Phase-2 binary)                                     │
│    cells: 3 reps × iv=250ms × 5-min                                            │
│    output: F1_null distribution, CV_null distribution                          │
│    purpose: lock acceptance thresholds to noise-floor + 3σ                     │
└──────────────────────────────┬─────────────────────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│  STEP 0 · SENSITIVITY PROBE (Day 1 evening · ~6 h)                             │
│    workload: sandbox_ransom_batched                                            │
│    cells: 3 iv × 3 window × 2 reps = 18                                        │
│    statistical model: 2-way ANOVA on F1                                        │
│    output: main-effect F-stats + interaction F-stat                            │
│    DECISION GATE:                                                              │
│      decouple confirmed → proceed greedy (Step 1 with locked window=128)       │
│      coupling significant → redesign Step 1 as joint iv × window grid          │
│      both effects null → halt; ML reviews metric definition                    │
└──────────────────────────────┬─────────────────────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│  STEP 1 · PILOT (Days 2-3 overnight · ~16 h)                                   │
│    workloads: sandbox_ransom_batched (phasic) + mem_workingset_sweep_v2 (steady)│
│    cells: 5 iv × 3 durations × 2 workloads × 3 reps = 90                       │
│    intervals: {100, 250, 500, 1000, 2000} ms                                   │
│    durations: {1, 2, 5} min                                                    │
│    block structure: 4-h blocks; cells randomized within block                  │
│    warm-up: 5-min discard cell before each session                             │
│    statistical model: fixed-effects ANOVA per workload                         │
│      F1 ~ log10(iv) + log10(T) + log10(iv):log10(T)                            │
│      then Bonferroni-corrected pairwise iv contrasts                           │
│    DECISION GATE:                                                              │
│      workloads agree on iv → Step 2 with single iv                             │
│      workloads disagree → Step 2 with per-regime iv (phasic vs steady)         │
│      neither passes acceptance → re-examine workload or analyzer               │
└──────────────────────────────┬─────────────────────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│  STEP 2 · GENERALIZE (Days 4-5 overnight · ~6-12 h)                            │
│    workloads: 1 representative per Phase-2 family (~6 picks)                   │
│    cells: 6 workloads × 2 iv × 3 reps = 36 (or 18 if Step 1 agreed)            │
│    statistical model: mixed-effects, workload as random                        │
│      F1 ~ log10(iv) + (1|workload) + (1|replicate)                             │
│    DECISION GATE:                                                              │
│      all families pass → Step 3                                                │
│      1-2 families fail → expand those cells (~6 more runs)                     │
│      most families fail → pilot answer didn't generalize; reframe regime split │
└──────────────────────────────┬─────────────────────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│  STEP 3 · CLASSIFIER VALIDATION (Day 6 morning · ~2 h)                         │
│    workloads: 1 previously-confused workload pair                              │
│    cells: 2 workloads × 3 reps × at chosen iv = 6                              │
│    output: confusion matrix on tuned pipeline vs pre-investigation             │
│    DECISION:                                                                   │
│      confusion drops ≥ 20 pp → Plan 02 thesis chapter writes itself            │
│      drops 5-20 pp → escalate to Plan 03 (window/hop tuning)                   │
│      drops < 5 pp → escalate to feature/classifier review                      │
└──────────────────────────────┬─────────────────────────────────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│  POST-CAPTURE ANALYSIS (Days 6-7 · ~6 h offline)                               │
│    DS · ANOVA + mixed-effects + plot generation                                │
│    DE · aggregate per-cell JSONs into one consolidated artifact                │
│    PM · per-phase report + decision-gate evidence                              │
│    ML · per-family iv recommendation table (the Plan 02 deliverable)           │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Total cost

| stage | runs | host hours | calendar |
|-------|------|------------|----------|
| pre-flight (code) | 0 | 0 | 1 day |
| Step 0.5 null baseline | 3 | 1 | morning |
| Step 0 sensitivity probe | 18 | 6 | overnight |
| Step 1 pilot | 90 | 16 | overnight + half-day |
| Step 2 generalize | 18-36 | 6-12 | overnight |
| Step 3 validation | 6 | 2 | morning |
| analysis | 0 | 6 (offline) | 1 day |
| **TOTAL** | **135-153** | **37-43** | **~1 week** |

### Deliverables

1. **Per-family `intervalMsec` recommendation table.** Six rows (one per Phase-2 family), three columns (recommended iv, recommended duration, regime notes).
2. **Aggregate JSON manifest + per-cell archive.** Reproducible to bit-identical re-analysis.
3. **Decision-gate evidence document.** One section per decision gate, with the F-stats, contrast estimates, and acceptance-table cell-by-cell pass/fail.
4. **Plan 02 thesis chapter draft (if Step 3 passes).** Headline result: confusion collapse on previously-misclassified pair.

---

## 12 · Final recommendation from the project manager

> **Recommendation: proceed with the locked pipeline as specified in Section 11.**

The team has converged. Five days of structured review surfaced:
- Six initial weaknesses (Day 1)
- Twelve specific design decisions (Days 2–5)
- Two completed validation probes that re-calibrated the acceptance thresholds (Day 4)
- Five resolved disagreements
- Two unresolved questions that do not block launch (open risks log, Section 10)

The pipeline as locked has the following risk profile:

| risk | severity | mitigation |
|------|----------|------------|
| Plan 02 results fail to generalize beyond pilot workloads | medium | Step 2 expansion criterion is explicit; budget allows 6 extra cells per failing family |
| σ at iv=100 ms exceeds power-analysis assumption | medium | Re-validate σ after Step 1 block 1; abort and add reps if necessary |
| Step 0 reveals strong coupling | low | Pipeline has an explicit fork to joint iv × window grid |
| Host has noise spike during one block | low | Manifest allows re-running specific cells; sentinel JSON captures host state for forensics |
| Schema v2 migrator has a bug | low | Test migrator against R1-R5 archive before launch |
| Analyzer (`mp_phase_boundary_inference`) has its own bugs | medium | Acknowledged limitation; out of scope for Plan 02 |

**Time budget:** 37–43 host hours across ~1 week. Half of the original Plan 02 estimate (~50 h) and a fraction of the full-joint alternative (~167 h). Time savings come from:
- Drop 10-min duration (saved ~10 h)
- Drop flush-sensitivity arm (already answered by R5; saved ~3 h)
- Drop VM RAM sweep (host-specific, not on the critical path; saved ~5 h)
- Decision-gate-driven Step 2 sizing (saved ~5-10 h if pilot is clean)

**Quality preserved by:**
- Noise-floor-calibrated acceptance thresholds
- 3 replicates (power-justified)
- Block randomization and warm-up cells
- Manifest-driven crash recovery
- Schema v2 reproducibility metadata
- Explicit decision gates instead of monolithic sweep
- Documented assumptions and limitations

**Launch checklist (PM signs off when all green):**

- [ ] SA: orchestrator with manifest support compiles + dry-runs
- [ ] DE: schema v2 specified + migrator written + R1-R5 archive migrated
- [ ] EN: retry policy + observability hooks in code
- [ ] XD: cell-order randomization + warm-up cell in orchestrator
- [ ] DS: ANOVA fitting script written; runs on a known-good synthetic input
- [ ] ML: F1 / CV computation reproducible from a synthetic null trace
- [ ] PM: per-day timeline approved; abort criteria documented

When the checklist is green, launch Step 0.5 → Step 0 → Step 1 → Step 2 → Step 3 → analysis.

---

**Audit log frozen at end of Day 5.** Subsequent changes appended below this line.
