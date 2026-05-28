# Why Only Two Workloads? — Tuning Scope vs Classification Scope

**Status:** reference note. Answers a recurring question about why Plan 02
and Plan 03 capture only two workloads when the project catalog contains
many more, and when earlier work classified a much larger set.

**Short answer:** Plan 02 and Plan 03 are *tuning* experiments, not
*classification* experiments. Tuning a knob (sampling interval, or
window/hop) only requires one clean example of each signal class, so we
use two deliberately-chosen representatives — one phasic, one steady. The
full workload family is reserved for the *classification* experiments,
where the question is "can we tell workloads apart," which genuinely needs
all of them.

---

## 1. The question

> We capture only `sandbox_ransom_batched` and `mem_workingset_sweep_v2` in
> Plan 02 and Plan 03. But the project has many more workloads, and earlier
> work (the confusion-matrix classification) monitored a large set. Why are
> we down to two? Is this a misunderstanding?

It is not a misunderstanding. Many more workloads exist, and many of them
have been captured and analyzed. The two-workload scope is a specific and
deliberate property of the tuning sub-experiments, explained below.

---

## 2. The full workload catalog

The project has two generations of guest workloads.

### Phase 2 (`VM_executables_phase2/`) — current generation

About eleven workloads across four families:

| Family | Workload | One-line behavior |
|---|---|---|
| SECURITY-LIKE | `sandbox_ransom_batched` | all files in one batch through 5 phases, then exits (one short burst) |
| SECURITY-LIKE | `sandbox_ransom_seq` | per-file loop: stat → read → XOR → write → rename (sustained, many boundaries) |
| SECURITY-LIKE | `sandbox_ransom_slowburn` | paced: one file every `--interval-s` s (sustained + periodic cadence) |
| SECURITY-LIKE | `sandbox_ransom_selective` | extension-filtered discovery then process (burst + discovery pre-phase) |
| SECURITY-LIKE | `sandbox_scanner_metadata` | metadata-only scan (low, flat — near-idle control) |
| MEM | `mem_workingset_sweep_v2` | cyclic working-set churn (steady, continuous activity) |
| MEM | `mem_mmap_traversal_v2` | mmap region traversal |
| MEM | `mem_pagefault_density_v2` | page-fault-density driver |
| MEM | `mem_rmw_intensity_v2` | read-modify-write intensity sweep |
| MEM | `mem_writemag_sweep_v2` | write-magnitude sweep |
| APP-REALISTIC | `app_hashtable_intensive_v2` | hashtable build → probe (two-phase trajectory) |
| METHODOLOGY | linearity probes (`methodology/`) | calibration helpers; inherit from the workload they drive |

### Phase 1 (`VM_executables/`) — older generation

A separate, earlier set used in the original classification work:
`mem_stream`, `run_idle`, `io_many_files`, `io_seq_fsync`,
`mem_pointer_chase`, `mem_alloc_touch_pages`, `io_rand_rw`, and others.
These feed the confusion-matrix analysis documented in
`VM_executables/artifact_confusion_matrix_connection.md`, which classifies
them against one another — a genuine multi-workload experiment.

**Takeaway:** the recollection that "we monitored many workloads" is
correct. The classification experiments used the full set.

---

## 3. Two kinds of experiment, two scopes

The project runs two fundamentally different kinds of experiment, and they
have different workload-scope requirements.

### Classification experiment

- **Question asked:** can we tell the workloads apart from their memory
  signatures? (Build a confusion matrix; measure separability.)
- **Workload scope:** the *entire* set. You cannot build a confusion
  matrix without all the classes present.
- **Where it lives:** Phase 1's confusion-matrix work, and the thesis's
  eventual detection claim ("can we separate ransomware-like from benign").

### Tuning experiment

- **Question asked:** how does a single capture/analysis knob affect each
  kind of signal? (Pick the best sampling interval; pick the best
  window/hop.)
- **Workload scope:** *two* representatives — one cleanly phasic, one
  cleanly steady. You need one clean example of each extreme so the knob's
  effect is visible without the confound of a dozen overlapping behaviors.
- **Where it lives:** Plan 02 (sampling interval) and Plan 03
  (window/hop).

The two-workload scope is a property of the *tuning* experiments only.

---

## 4. What Plan 02 and Plan 03 used, and why

Both plans use exactly two workloads:

| Probe | Workload | Signal class | Defining metric |
|---|---|---|---|
| Phasic | `sandbox_ransom_batched` | bursty / event-driven | F1 of phase-boundary detection |
| Steady | `mem_workingset_sweep_v2` | continuous / cyclic | CV of active-page-fraction |

The Plan 02 design document
(`docs/tuning_plans/02_interval_tuning_experiment.md`) names these two
explicitly and describes them as "phasic plus steady probes per Phase 1's
playbook." The choice is intentional: these two bracket the range of memory
behaviors. If a sampling interval (or a window/hop) works well for both a
bursty workload and a steady one, it is likely to work for the workloads
that fall between those extremes.

The reason this is sufficient for tuning — but not for classification — is
that tuning asks a *within-signal* question ("does this knob preserve the
phasic rhythm? does it preserve the steady plateau?"), whereas
classification asks a *between-signal* question ("can we distinguish these
two from each other and from nine others?"). The within-signal question is
answerable with one example per class.

---

## 5. The plan always intended per-family generalization

The two-workload pilot was never meant to be the final scope. The same
Plan 02 design document includes a *target-output* table that lists intended
per-family recommendations:

| Family | Recommended `intervalMsec` | Note |
|---|---|---|
| MEM steady | TBD | steady-state stats |
| APP-REALISTIC OLTP | TBD | two-phase trajectory |
| SECURITY-LIKE batched | TBD | covers one full five-phase cycle |
| SECURITY-LIKE seq / selective | TBD | may need workload-parameter retuning |
| METHODOLOGY | inherits | inherits from the workload it drives |

In other words, the plan always intended the tuned knobs to generalize
across the full family set. The pilot used two probes first in order to
establish the method cheaply; the recommendations were always meant to be
applied family-wide afterward.

---

## 6. Where the full set comes back

The full workload family is the subject of the *classification* phase, not
the tuning phase. The sequence is:

1. **Plan 02** — tune the sampling interval on two probes. *(done)*
2. **Plan 03** — tune the analyzer window/hop on the captured trajectories.
   *(done)*
3. **(next) full-family capture** — take the tuned sampling interval and
   window/hop and run the *entire* Phase 2 workload set through capture and
   analysis.
4. **classification** — rebuild the confusion matrix on the properly-tuned
   pipeline and answer the real detection question across many workloads.

So the two-workload scope is a temporary, deliberate narrowing for the
tuning stage. The classification stage re-expands to the full set, now with
knobs that were calibrated rather than guessed.

---

## 7. Implications for v3

This reframes the planned v3 capture. v3 is not merely "make the ransom
workload run longer." It is the transition from the two-probe tuning stage
to the full-family classification stage:

- **Expand the phasic side.** The single short-burst `sandbox_ransom_batched`
  becomes a small family of phasic shapes — sustained
  (`sandbox_ransom_seq`), periodic (`sandbox_ransom_slowburn`), and bursty
  (`sandbox_ransom_batched` with more files so the burst lasts minutes).
  See `plan03_overview.html` §08 for why the current 13-second burst is
  too short for gate G2.
- **Bring in the rest of the MEM and APP-REALISTIC families** so the
  classification confusion matrix can be rebuilt.
- **Use the tuned knobs** — the Plan 02 sampling interval and the Plan 03
  window/hop (W=8, H=4) — rather than the design-time guesses.

The two-workload tuning phase is ending. The multi-workload classification
phase is what v3 opens.

---

## 8. One-paragraph summary for the thesis

> The interval-tuning (Plan 02) and window/hop-tuning (Plan 03)
> sub-experiments deliberately use only two workloads — one phasic
> (`sandbox_ransom_batched`) and one steady (`mem_workingset_sweep_v2`) —
> because tuning a capture or analysis parameter requires only one clean
> exemplar of each signal class, not the full workload catalog. The complete
> Phase 2 family (≈11 workloads across security-like, memory, and
> application-realistic groups) plus the Phase 1 set is reserved for the
> classification experiments, where separability across many workloads is the
> actual question. The tuned parameters from Plans 02 and 03 are intended to
> generalize family-wide, and the subsequent full-family capture re-expands to
> the entire set with calibrated rather than guessed parameters.
