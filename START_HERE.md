# Start here

You are picking up a doctoral project mid-flight. The capture half is built, validated, and driven
from a UI. The analysis half is not. Your job is to build and automate it, and to do so in a
specific way that this document exists to transmit.

Read this file first, then follow the ramp-up in order. Do not start proposing designs until you
have finished it. This project's standard is that context precedes opinion.

Branch: **`fullv5`**. Everything referenced here lives on it.

---

## 1. What the project is

A method for identifying what a virtual machine is doing by watching only its memory: no system
calls, no agent inside the guest, no semantic knowledge of the guest OS.

The instrument suspends a QEMU VM, dumps guest RAM, and diffs consecutive dumps. Each diff is
reduced, per page, to a vector of numeric channels (how much changed, in what direction, what the
new content looks like, where in the page). Stack those vectors over time and you have a signal.
The bet of the thesis is that the signal carries enough structure to say what the machine was
running.

The founding intuition, which matters more than any implementation detail:

- Memory is **listened to**, not read. The framing is acoustic (an ECG, a symphony). That is what
  licenses treating this as signal processing rather than forensics.
- A difference between two snapshots **is a derivative** of a state signal. That is the move that
  makes the signal-processing toolkit apply by right rather than by analogy.
- Therefore `F{f'(t)} = jw * F{f(t)}`, and spectral methods are legitimate on deltas.

If you propose something that quietly abandons one of those three, say so out loud rather than
sliding past it.

---

## 2. The mindset you are inheriting

This is the part that matters most. The engineering here is downstream of a way of not being
fooled. Entry 11 of the diary states it as seven moves. Internalise them; you are expected to apply
them without being asked.

1. **A perfect score is a symptom.** A number that looks too good is a report of a measurement bug
   until proven otherwise. This project has twice hit 100% and twice correctly torn it down.
2. **When everything fails the test, suspect the test.** A check that passes or fails uniformly is
   more likely broken than right. Turn the instrument on itself first.
3. **Budget the design by what the data can support.** Count degrees of freedom before choosing a
   method. More knobs than the data can constrain is a liability, not sophistication.
4. **Set the bar from measured noise, not convention.** Run the metric on deliberate nonsense and
   see what it scores. A threshold not anchored to a measured noise floor is decoration.
5. **Write the objection where the temptation will occur.** Put the prohibition in the file where
   the shortcut would be taken.
6. **Analogy, then object, then formula.** Physical picture first, convert to a mathematical
   object, only then write the equation. Never the reverse.
7. **The form is the contribution, not the realisation.** Separate what any valid version must
   preserve (roles, invariants, conditions of valid observation) from the one version that happens
   to be built. Test: could a different implementation still satisfy this sentence?

Two operational consequences for you:

- **Verify before asserting.** Do not claim a file, function, flag, or threshold exists without
  opening it. This project's record contains several confident claims that failed checking, and
  each is documented rather than quietly deleted. Do the same: if you could not verify something,
  label it unverified rather than smoothing it.
- **Tag evidence.** The project distinguishes `[traced]` (opened or ran it) from `[inferred]`
  (assembled, and the source does not say so). Carry that into anything you write.

---

## 3. Ramp-up: build your context in this order

### Phase A: the reasoning and the record

Open `docs/research-diary/index.html` and read the entries in order. Thirteen entries,
chronological, written to be read by a person. Fastest path to understanding both the project and
how its author thinks. It is deliberately harder on the work than any paper would be.

| # | Entry | Why it matters to you |
|---|-------|----------------------|
| 1 | The reduction | The founding chain: why listen, why a delta is a derivative, why two numbers per page become one. |
| 2 | The score he didn't believe | The original thesis and detector. A 100% result its own results chapter refused to accept. |
| 3 | Six weeks later, he threw the detector away | The March 2025 hinge: keeps the apparatus, deletes the detector, starts asking whether the sampling is even real. |
| 4 | The confusion that started it | Why the instrument, not the model, became the object of study. |
| 5 | The instrument was lying by 15x | A missing utility turned a sleep into `sleep 0`. Every frequency ever computed sat on the wrong axis. |
| 6 | The workload was a label, not a variable | Ninety cells passed every gate having never run a workload. |
| 7 | Two gates that measured the wrong thing | Where `W=8, H=4` comes from, and why that provenance is weaker than it looks. |
| 8 | The night the headline collapsed | A schema mismatch faked every gate pass; a perfect classification score was one leaked feature. |
| 9 | The failure that wasn't | Eighteen failed comparisons that, investigated, produced the largest single win in the project. |
| 10 | Building an instrument worth trusting | The 101-workload campaign, the dwarfs, predicted confusion. |
| 11 | **What carries forward** | The seven moves above. If you read only one entry, read this one. |
| 12 | The angle that hid an allocation | Stub, in progress. The phase collision (section 7). |
| 13 | **The whole experiment as a graph** | The mental model you will work inside. Defines the shape of your task. |

### Phase B: the doctoral plan (external to this repo)

`~/Desktop/plan/` holds the doctoral plan proper: `Document-1-CLEAN-v5/v6/v7.tex`,
`OBJECTIVES_PLAYBOOK.md`, `Doctoral-Plan-Respined-Outline-v7.md`, `CONTEXT_SYNTHESIS.md`,
`SOURCE_TO_DOC_CONNECTION_MAP.md`, and the deepdives. Objectives are numbered O1 through O8 and
stages M1 through M5.

**Read it.** The analysis stage you are about to build is objectives O3, O4, and O8. You cannot
scope the work correctly without knowing what those objectives actually commit to.

Note while reading: this material has **not** been verified against the code by anyone. At least
six claims in the in-repo methodology documents rest on it, including a non-invertibility proof
that a data-protection argument depends on, and the source decks are reported to be internally
inconsistent about whether that proof is established or merely anticipated. Treat the summaries in
this repo as unverified until you have checked them against those files yourself.

### Phase C: how this project plans and validates work

- Proposals: `VM_sampler/VM_Capture_QEMU/docs/plan01_overview.html` through `plan07_*`. The richer
  ones (`plan03_window_hop_proposal.html`, `plan04_segmenter_proposal.html`) show the full
  template.
- Implementations: `VM_sampler/VM_Capture_QEMU/plan0N_*.py`.
- Campaigns: `plan05_campaign/`, `plan06_campaign/`, `plan07_campaign/`.
- Tests: `VM_sampler/VM_Capture_QEMU/tests/test_plan0N_*.py`.
- Pre-registration and audit track: `VM_sampler/VM_Capture_QEMU/docs/tuning_plans/`. Caution: these
  still read "Status: plan, not yet implemented" and carry superseded thresholds. Do not cite them
  as the live pre-registration without checking the code.

The proposal template, from Plan 03 on: executive summary, simulated multi-role team review,
baseline, top-down analysis, bottom-up file-by-file with output schemas, an audit pass, **key
disputes and their resolutions**, improved spec with acceptance gates and a decision rule,
validation strategy, statistical design, risks, final recommendation with per-role votes.

The dispute mechanism is the important part. A real design fight is written as "the fight" and "the
resolution", and the resolution binds. Example from `plan04_segmenter_proposal.html`, choosing a
change-point detector: five candidates, and the deciding argument was not accuracy but "with
approximately 20 marker-rich cells per duration stratum, any detector with more than one knob is
over-parameterised". The winner had exactly one knob.

**The gate system.** This project does not trust a run because it finished. It trusts a run because
named, machine-checked claims about it passed.

- **C1 through C8**: session-level claims, evaluated in
  `VM_sampler/VM_Capture_QEMU/plan02_validate_session.py`. The verdict line is
  `ok = c1 and c2 and c4 and c5 and c6 and c7 and c8` (C3 informational, excluded).
- **G1 through G5**: analysis-quality gates, numbered per plan. C7 wraps Plan 03's, C8 wraps
  Plan 04's.
- Gates name a **claim**, not a code path. That is why C1 catches "ninety cells that ran nothing"
  where a return-code check would not.
- The principle to carry forward: a gate that reports is read past; a gate that refuses to write
  the file is not.

Read `plan02_validate_session.py` directly. It is the clearest single artifact of the standard.

### Phase D: the capture console, and what "finished" looks like here

`VM_sampler/VM_Capture_QEMU/plan07_campaign/ui/`

The most recent and most polished piece of the project. Study it, because it is the model for what
you are being asked to produce on the analysis side.

- `capture_console.template.html`, hand-edited source. Single-file vanilla HTML/CSS/JS, no
  framework, no build dependencies.
- `build_console.py` generates the shipped builds and injects live pipeline data at a marker,
  reading `full_campaign_steps.txt`, `generate_database_steps.py`, and `subset_run.py` so the UI
  cannot drift when the Python changes. That choice is characteristic. Note it.
- `console_bridge.py`, the backend. Python standard library only, `ThreadingHTTPServer` bound to
  127.0.0.1 with a token. No framework, no database, polling rather than websockets.
- `console.sh`, laptop-side launcher: ssh, build, bridge, port-forward, browser, migration agent.

What it does: pick a subset of workloads per family (modes `all`, `asc`, `desc`, `rand`, `named`),
set cadence (durations, scales, repetitions), set retention and capture metric, choose feature
groups, then **generate a plan** by crossing workload x duration x scale x rep into real guest
commands, with per-cell checkboxes and reordering. That curated cell list is what runs. Plus
preflight, launch, stop, cleanup, health check, live log tail, and pause/continue/skip of the
in-flight trace.

Live stats survive a UI or bridge restart because the producer writes `capture_status.json`
atomically each snapshot and the bridge just reads that file. State lives in `capture_status.json`,
`capture_control.json`, an append-only `.migration/ledger.jsonl`, and saved configs under
`plan07_campaign/configs/`.

It invokes `subset_run.py`, which composes a launch line for `run_files_controlled.py`, the
orchestrator, which drives `capture_producer_qemu_pmemsave.sh` and `capture_consumer_qemu.sh`.

**The console stops at capture and migration. There is no analysis view. That boundary is exactly
where your work begins.**

For a mechanical file-by-file map of the capture pipeline see `docs/QUICKSTART_FOR_AI_CONTEXT.md`
and `docs/ACTIVE_PIPELINE_FILE_MAP.md`. Accurate and time-saving, but inventory, not reasoning.

---

## 4. The mental model: the pipeline is a graph

Read Entry 13 (`docs/research-diary/part_graph.html`) for the figure. In summary:

The old picture was one pipe: Hamming and Cosine, combined into a complex number, into FFT or
Cepstrum or Wavelet, into a model. The current picture is a directed graph in tiers:

```
METRICS  ->  COMPOSE  ->  ANALYSE  ->  ORGANISE  ->  MODEL
```

- **METRICS**: the per-page channels the differ emits. Four families implemented (A amount,
  B direction, C content, D texture). A metric is not a primitive: it is a point in a designed
  space of choices (representation x reading x reference x unit). Swap one coordinate, get a
  different metric.
- **COMPOSE**: combining channels. The complex representation (magnitude from a Hamming-like
  channel, angle from a Cosine-like channel) is **one node here, not the core of the system**. That
  demotion is deliberate and is the main conceptual shift of Entry 13.
- **ANALYSE**: cepstrum, FFT, wavelet, scattering, PLV, MSC, CUSUM.
- **ORGANISE**: scale and reduce (flatten, standardise, PCA).
- **MODEL**: autoencoder, one-class SVM, novelty, random forest, LSTM.

The 2025 thesis pipeline is exactly **one traversal** of this graph. The open research question,
and the reason the graph is drawn at all, is **which path through it actually separates workloads**.
That is unanswered, and answering it is the point of the analysis stage.

---

## 5. What you are building: the analysis stage

The capture side produces the METRICS tier. Everything after is unbuilt. The decomposition, in
order:

**Step 0. Optional combining.** Compose channels into derived representations. The complex number
built from a magnitude-family channel and a direction-family channel is the known example, but it
is optional and it is one option among several. Keeping magnitude and direction as two separate
real channels is a legitimate alternative that has never actually been ablated.

**Step 1. Organising the data.** Windowing and blocking along **two axes**:
- the **temporal** axis (windows over snapshots, with a window size and hop), and
- the **spatial** axis (the memory address axis, cut into blocks or tiles).

A genuine subtlety lives here, so do not flatten it: **there are two distinct organising
operations**, and the Entry 13 figure only shows the second.
- Organising **before** analysis: windowing and blocking the raw metric time-series. This is Step 1.
- Organising **after** analysis: scaling, flattening, reducing the resulting features (the ORGANISE
  tier in the figure).

The figure needs a "pre-something" layer to represent the first. Treat that as an open modelling
question you are expected to resolve, not a drafting error to paper over.

**Step 2. The analysis itself.** Cepstrum, FFT, wavelet and the rest, applied to the organised
data. The choice is not independent of Step 1: a window that does not span the rhythm cannot show
the rhythm.

**Step 3. Retention.** What is kept from the analysis output, in what form, at what cost. Not to be
confused with capture-side zstd retention of raw dumps; this is retention of derived features.

---

## 6. Required reading in this repo, with a health warning

Two documents already propose this stage:

- `docs/ANALYSIS_PIPELINE_METHODOLOGY.md` (771 lines), the forward-looking proposal. Proposes
  analysis-side gates A1 through A7.
- `docs/METHODOLOGY_AS_EXECUTED.md` (564 lines), a reconciliation of that proposal against what
  Plans 01 through 07 actually built.

**Read them. Do not accept them uncritically.** Read them together with the companion critique:

- **`docs/RAJA_REVIEW_ANALYSIS_METHODOLOGY.md`**

That review checked both documents against the code and found confirmed defects, including places
where the reconciliation document is less accurate than the document it was reconciling. It also
identifies what those documents get right and should keep. Section 7 below is the short version;
the companion has the detail, the evidence, and the priority split.

---

## 7. Landmines: verified facts that contradict the current draft

Each confirmed by opening the code. Stated here so you neither rediscover them the hard way nor
inherit the errors.

1. **The captured corpus has 51 metric channels, not 64.** `config_qemu_upc.json` sets
   `substrateSpeed: 2`; the consumer passes it through; `metrics/mod.rs` documents that speed 2
   drops thirteen channels (`lz_change` plus the heavy twelve: FFT spatial, autocorrelation,
   high-frequency, four GLCM, Kendall, bigram, `ncd`). Those thirteen still emit zeros into an
   unchanged 64-column schema, so the schema lies about the data. Any design that reasons about
   `ncd`, `cross_corr_lag`, `phase_corr`, or the GLCM channels is reasoning about channels that are
   identically zero in the corpus. **Re-derive the channel roster before designing anything
   per-channel.**

2. **Reconstruct-plus-differ cost is already measured**, in `main.rs`, the same file the draft
   cites for a different figure: about 39 s/pair at speed 0, 5.5 at speed 1, 2.1 at speed 2, 1.2 at
   speed 4, on a 1 GB dump pair at roughly 5% activity. At a few hundred pairs per trace that is
   hours per trace at speed 0. A hard architectural constraint on any design that re-diffs the
   corpus, not a soft preference.

3. **The phase fix is about thirty sites, not one.** The collision is real: the differ's `cosine`
   field is a **distance** in [0,1] (`family_b/structure.rs`: "cosine DISTANCE (0 = identical)",
   all-zero side maps to 1), and `make_complex` multiplies it by `2*pi`, so distance 0 and distance
   1 land on the same angle. A page rewritten in place becomes indistinguishable from a page
   freshly allocated from zeros. But the same `mag * exp(1j * 2 * pi * phase)` expression appears
   inline roughly thirty more times in `VMsig_featureExctraction/block_feature_extractor.py`.
   Patching only the function leaves a feature vector mixing two conventions.

4. **Gate C7 passes trivially today, and is one filename away from failing everything.** C7
   requires every entry in `plan03_recommendation.json` to have `passes_acceptance == true`. No
   such file exists, so C7 evaluates not-applicable and passes. The artifact that would plausibly
   fill that name, `plan05_campaign/downstream/recommendation.json`, has 7 of 11 entries at
   `passes_acceptance: false`. Creating or renaming it flips every cell to `ok = false`.

5. **C3 is informational for scope reasons, not because it failed.** The code comment attributes it
   to D-25 (low window count is a duration-matrix issue owned by Plan 03). The draft's claim that
   it "was demoted when it failed everything" is unsupported.

Before trusting any count: the real executed corpus in the record is **11 workloads and 66 cells**,
while the forward-looking draft computes sample counts, payload sizes, and cache budgets from an
assumed **100 workloads x 3 repetitions x about 700 snapshots**. Those have never been reconciled,
and every derived number scales linearly off the assumption.

On the window: `W=8, H=4` is the live setting everywhere, but it came from a sweep that saw only
one averaged channel, on traces too short to test the largest windows, and it won partly on a
"smallest W" tiebreak. `W=128` was never in the sweep grid. Treat it as a default with weak
provenance, not a validated choice.

---

## 8. What this stage actually is

This is **the next stage of the pipeline, to be built and automated**. It is not a document
exercise. The goal is that an analysis run can be composed, launched, monitored, and trusted the
same way a capture run can be today.

The capture side is the worked example of what that means. It went: plans, implementation, named
gates, an orchestrator that enforces the run lifecycle, then a console that lets a human compose a
capture scheme (which workloads, which cadence, which retention), launch it, and watch it. The
analysis side needs the same arc, and the graph in Entry 13 tells you what its equivalent of
"compose a scheme" is: **choosing a path through the graph**.

Concretely, the stage should end up able to:

- compose an analysis scheme (which channels, which optional combination, which windowing and
  blocking in time and space, which analysis, what is retained),
- execute it automatically over the captured corpus, without hand-run notebooks,
- gate it, with analysis-side claims in the C/G tradition (the A1 through A7 proposal is a starting
  point, not a settled design),
- and make the comparison between paths a first-class operation, since which path separates
  workloads best is the actual research question.

Sequence to get there:

1. Read the diary in order. Entries 1, 11, 13 are non-negotiable.
2. Read `~/Desktop/plan/` for objectives O3, O4, O8, which are what this stage owes the thesis.
3. Read `plan02_validate_session.py` and one full proposal to internalise the gate system and the
   plan template.
4. Follow one path through the capture console, from a UI click to a capture script, so you know
   concretely what the substrate is and where it stops.
5. Read both methodology documents beside `docs/RAJA_REVIEW_ANALYSIS_METHODOLOGY.md`.
6. Independently verify the channel roster and the real corpus manifest. Do not take either from a
   document, including this one.
7. Then design and build, in this project's idiom: disputes recorded and resolved, gates named
   before results are seen, thresholds anchored to measured noise, evidence tagged, decision rules
   stated in advance.

One last thing, and the reason this document exists in this form. The goal is not only to build the
pipeline. It is to make the way of thinking explicit enough that it can be carried by something
other than its author. Work in a way that would survive being read back later as evidence of how
the decision was made.
