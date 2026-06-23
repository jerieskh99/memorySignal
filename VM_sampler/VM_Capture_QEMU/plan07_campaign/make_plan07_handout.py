#!/usr/bin/env python3
"""Plan 07 comprehensive handout -> docs/plan07_handout.html + docs/plan07_handout.pdf.

One content model, two renderers, shared figures (plan07_figs) -> the HTML and the
PDF are the same document. All data numbers are verified project results; design
diagrams are labelled schematic. Run: python3 plan07_campaign/make_plan07_handout.py
"""
import html as _html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table,
                                TableStyle, PageBreak, KeepTogether)
from reportlab.lib.utils import ImageReader

import plan07_figs as F

DOCS = Path(__file__).resolve().parent.parent / "docs"
FIGS = F.build_all()
P = F.PAL
DATE = "June 2026"

# ---------------------------------------------------------------- content model
def p(t): return ("p", t)
def h3(t): return ("h3", t)
def fig(name, cap): return ("fig", name, cap)
def ul(items): return ("ul", items)
def callout(kind, title, text): return ("callout", kind, title, text)
def table(header, rows, cap=None): return ("table", header, rows, cap)

FAM_TABLE = table(
    ["Family", "Kind", "Distinct workloads", "Cells", "Per-family block validatable now?"],
    [["ransomware", "threat", "4", "24", "Yes -- 4 within-family folds"],
     ["mem_sweep", "benign", "3", "18", "Marginal -- 3 folds (pilot-grade)"],
     ["mem_fault", "benign", "2", "12", "No -- 1 degenerate fold"],
     ["app", "benign", "1", "6", "No -- family-of-one (memorises one workload)"],
     ["scanner", "threat", "1", "6", "No -- family-of-one; and a threat family"]],
    "Family composition of the CURRENT 66-cell dataset. These figures are provisional -- they "
    "grow as we capture more workloads (see the catalogue below). Size, in distinct workloads, "
    "decides whether a per-family model can be honestly validated.")

METRIC_TABLE = table(
    ["Metric", "Group", "Status", "What it captures"],
    [["apf_mean, apf_std, cepstral, stat_pass_frac, n_pairs, n_windows", "AGNOSTIC (7)", "existing", "leakage-free level + periodicity + quality"],
     ["apf_max, apf_p95", "peakvar -- magnitude", "existing", "level / loudness of churn"],
     ["apf_cov, peak2med, duty_gt05", "peakvar -- scale-invariant", "existing", "shape / burstiness (unitless)"],
     ["wr_rate, rd_rate, rd_frac", "disk (Plan 06)", "ablation only", "a second-channel comparison -- NOT part of the memory-only spine"],
     ["APF (changed-page fraction)", "Hamming -- baseline", "kept", "the existing signal, retained for comparison"],
     ["Hamming sum", "Hamming -- NEW", "additive", "total bits flipped per snapshot"],
     ["mean Hamming / changed page", "Hamming -- NEW", "additive", "intensity per touched page -- the encryption tell"],
     ["p95 / max Hamming per page", "Hamming -- NEW", "additive", "is any page heavily rewritten"]],
    "Feature inventory. The new per-page Hamming scalars are computed additively at capture "
    "(XOR + popcount in the page comparison APF already performs) -- a few scalars, no per-page storage.")

CATALOG_TABLE = table(
    ["Family", "Captured (66-cell)", "Available now (built)", "Also specced (design docs)"],
    [["app", "1 (hashtable)", "6: + sqlite_oltp / sqlite_analytical / gzip_compress / gzip_decompress / json_parse", "--"],
     ["mem", "5 (sweep + fault)", "8: + phase-1 mem_stream / mem_pointer_chase / mem_alloc_touch_pages", "mem_random_write_pages, mem_stride_sweep_large"],
     ["IO", "0", "3: phase-1 io_seq_fsync / io_rand_rw / io_many_files", "next-gen IO family"],
     ["IDLE", "0", "0", "idle baselines (negative / reference family)"],
     ["ransomware", "4", "4 (seq / batched / slowburn / selective)", "--"],
     ["scanner", "1", "1 (scanner_metadata)", "additional scanner variants"]],
    "The available catalogue is much larger than the 66-cell subset: roughly 22 built workloads "
    "(16 in VM_executables_phase2 + 6 phase-1) plus whole specced families (IO, IDLE) we never "
    "captured. The capture plan should draw on all of it, not just fill the thin families.")

CAPTURE_TABLE = table(
    ["Family", "Captured", "Capture plan (priority order)"],
    [["app", "1", "capture all 6 built programs -- no new code; takes app 1 -> 6"],
     ["IO", "0", "capture the 3 built IO programs -- an entire family currently absent"],
     ["mem", "5", "add the 3 built phase-1 mem programs (+ specced variants) for within-family diversity"],
     ["IDLE", "0", "capture idle baselines as a reference / negative family"],
     ["scanner", "1", "author ~2 more scanner programs, then capture"],
     ["ransomware", "4", "sufficient; optionally add variants for tighter intervals"]],
    "Capture broadly for workload and family diversity (the real binding constraint), "
    "prioritising families with too few distinct workloads and the entirely-absent IO and IDLE "
    "families. Most of this is capture-only (the executables exist); only scanner needs new code. "
    "All counts are current and grow with capture.")

WINDOWS_TABLE = table(
    ["Duration", "Windows per cell (range, W=8/H=4)", "Note"],
    [["120 s", "5 -- 48", "already clears the DOF floor"],
     ["300 s", "11 -- 122", "recommended -- balances window count vs server time"],
     ["600 s", "15 -- 244", "most windows, but twice the capture time"]],
    "Window yield by duration, measured across the existing cells.")

SIGFLOOR_TABLE = table(
    ["Independent units n", "Sign / McNemar floor p = 2^(1-n)", "Can reach p < 0.05?"],
    [["5 (LOFO, current families)", "0.0625", "No, ever -- on the current 5 families"],
     ["7+ (LOFO, after capturing IO + IDLE)", "0.016 or lower", "Yes -- the capture lifts it below 0.05"],
     ["11 (LOWO, current workloads)", "0.00098", "Yes (needs >= 6 concordant)"]],
    "Structural significance floor (sign / McNemar). With only **5** families, "
    "leave-one-family-out cannot reach significance regardless of effect size; capturing the "
    "currently-absent IO and IDLE families (>= 7 total) drops the floor below 0.05. Few-unit "
    "comparisons are reported descriptively, with CIs, until the families grow.")

EXPECTED_TABLE = table(
    ["Experiment", "CONFIRM (and its meaning)", "REFUTE (and its meaning)"],
    [["Separability + cohesion (per family, memory only)",
      "within-family tight, between-family distinct -> families have a coherent memory signature; family-normal models are meaningful",
      "families not cohesive -> the semantic taxonomy does not map to memory; sub-family granularity needed (still a finding)"],
     ["Family-normal recogniser (within-family hold-one-out)",
      "held-out, unseen family member accepted; other families rejected -> the recogniser generalises within a family",
      "held-out member rejected, or cross-family accepted -> the family label does not map to one routine"],
     ["Open-set novelty (hold out a whole family)",
      "a never-seen family is flagged NOVEL, not forced into a known one -> open-set recognition works",
      "the novel family is absorbed into a known one -> the novelty boundary is too weak"],
     ["Memory resolves the masquerade (slowburn vs pagefault, NO disk)",
      "separable in memory alone (AUC ~ 1.0) -> the disk channel is not needed for the spine",
      "inseparable out-of-sample -> a second channel is genuinely required after all"],
     ["Detection as a corollary of (b)",
      "threat-shaped families are recognised + flag-on-match works with CIs -> detection rides on recognition",
      "threat families are not recognisable as families -> memory recognises kinds but not threat-ness (an honest negative)"]],
    "Every experiment is win/win. The spine is the per-family recogniser (b); detection (a) is derived from it.")

SECTIONS = [
("summary", "Summary", [
    p("**Plan 07 builds a behaviour recogniser from a single host-side memory signal.** Watching "
      "a VM from the outside -- no agent inside the guest -- the system places a running workload "
      "into a known **behaviour family** (memory-sweep-like, IO-like, file-processing-like, ...) "
      "or flags it as **novel**. That recogniser is the centrepiece and the claim. Threat-flagging "
      "is a *downstream consequence* of it: a behaviour that matches a known threat-shaped family "
      "is flagged for review. We build (b) the recogniser; (a) detection rides on top of it."),
    p("**What already works, and what is open.** We can already identify a workload's behaviour "
      "*when its kind has been seen before* -- instance and family identification reach ~0.97 on "
      "the optimistic (seen-before) split. That much is established. What Plan 07 answers is the "
      "open part: does a per-family 'normal-behaviour' model generalise to an *unseen* member of "
      "its family; do families even have a **coherent memory signature** (or does our taxonomy not "
      "match the data); can the system flag a **novel** family open-set; and does threat-flagging "
      "follow as a corollary."),
    p("The reframe that makes this worth doing: the earlier 'characterisation, not detection' "
      "reading understated the signal. A one-class model already recognises 'normal' at ROC-AUC "
      "**0.925**, and two workloads we had called inseparable in memory separate cleanly (Cohen's "
      "d ~ **5.6**) once the right method is used -- they scored 0.0 only because a *supervised "
      "classifier* failed on them. The memory signal carries more behavioural structure than the "
      "negative implied."),
    p("The signal is **memory only** -- APF plus a new per-page magnitude metric. The disk channel "
      "from Plan 06 is retained as a supporting **ablation**, not part of the spine (we keep the "
      "thesis on its novel contribution, the memory signal). Every decisive experiment runs "
      "**offline on data we already have**, with new captures gated behind an offline pilot. The "
      "outcome is win/win: either a working behaviour recogniser with measured family cohesion, or "
      "a rigorous, error-bounded map of exactly where memory recognises behaviour and where it "
      "does not."),
    callout("new", "Safety and scope (please read first)",
      "This is academic, defensive research. Every reference to 'threat', 'ransomware', "
      "'scanner', or 'encryptor' in this document denotes a SAFE SYNTHETIC SIMULATION -- a "
      "controlled workload generator that imitates a behavioural *shape* using a reversible XOR "
      "on disposable sandbox files. There is no real malware, no real encryption, no network "
      "activity, and no persistence. See RESEARCH_SAFETY_NOTICE.md and "
      "VM_executables_phase2/docs/SAFETY_MODEL.md."),
    callout("info", "Dataset at a glance (current figures; grow with capture)",
      "11 workloads, 5 families, 66 cells = 11 workloads x 3 durations (120 / 300 / 600 s) x 2 "
      "repetitions. Memory snapshot cadence 500 ms; analysis window W=8, hop H=4; 5--244 windows "
      "per cell. Binary majority prior 0.545; family majority prior 0.364."),
]),
("motivation", "Motivation: what led here", [
    p("**APF (Active Page Fraction)** is the fraction of 4 KB guest memory pages that change "
      "between two consecutive snapshots -- one number per snapshot, measured host-side with no "
      "guest agent. Plans 01--04 built and locked the capture pipeline; Plan 05 relieved a "
      "degrees-of-freedom starvation problem and ran a full 66-cell campaign; Plan 06 added a "
      "second, disk-I/O channel."),
    p("The downstream conclusion was sobering and honest: a **supervised** threat/benign "
      "classifier did not generalize to unseen workloads or families (binary leave-one-workload-out "
      "~0.79, leave-one-family-out ~0.58, and 0.0 on the stealth pair), and a second, disk-I/O "
      "channel helped **characterization** (instance identification rose to 0.985) but not "
      "detection. Hence the earlier reading: characterization, not detection. (Plan 07 keeps the "
      "spine **memory-only** and treats that disk channel as a supporting ablation -- see Metrics.)"),
    p("The review then noticed what the supervised framing had hidden. The **same committed "
      "results file** contains a one-class novelty detector at ROC-AUC 0.925 / 0.933, and the "
      "'memory twins' separate cleanly in raw apf_mean. In other words, the negative result was a "
      "property of *how we asked the question* (a supervised boundary across too few workloads), "
      "not of the memory signal. Plan 07 builds on that: a memory-only behaviour recogniser, with "
      "detection as a corollary."),
    fig("results", "The motivating result. A one-class novelty detector already beats the "
        "supervised classifier the thesis used to declare detection dead. The 0.933 is flagged "
        "for a fixable validity issue (see Methodology)."),
    fig("masquerade", "The 'memory twins' are not twins: slowburn and pagefault are "
        "non-overlapping in raw apf_mean, yet the supervised classifier scores both at 0.0 "
        "under leave-one-workload-out -- a labelling-coverage artifact."),
]),
("questions", "Research questions", [
    p("The questions group into three themes, **led by the recogniser** -- the centrepiece. "
      "Detection appears last, as a corollary. None of the eight is assumed; every outcome is a "
      "publishable finding."),
    h3("Theme A -- The recogniser: characterisation, families, novelty (the centrepiece)"),
    ul(["**RQ1 -- Are families cohesive?** Do the workloads of a family resemble each other in "
        "memory enough for a single 'family-normal' model to represent them? Cohesion is "
        "**measured** per family, not assumed.",
        "**RQ2 -- Does a family-normal model generalise?** Trained on some members of a family, "
        "does it accept a held-out, *unseen* member and reject other families?",
        "**RQ3 -- Does memory respect our taxonomy?** With no labels, do data-driven clusters of "
        "the memory signal recover the families we named, or carve the space differently?",
        "**RQ4 -- Can we flag the novel, open-set?** Held out a whole family, does the system "
        "label it **NOVEL** rather than forcing it into a known family -- and does a per-family "
        "one-class *bank* beat a discriminative classifier with a *reject option* (or tie at "
        "this sample size)?"]),
    h3("Theme B -- The signal"),
    ul(["**RQ5 -- Do magnitude-aware metrics catch stealth?** Does mean-Hamming-per-changed-page "
        "expose the simulated encryptor (slowburn) that APF's changed-page count cannot see?",
        "**RQ6 -- How cheap can capture be?** What is the minimum useful trace length, and how "
        "coarsely can we sample, before the signal degrades -- which sets the budget for "
        "expanding the workload set?"]),
    h3("Theme C -- Detection, as a corollary of the recogniser"),
    ul(["**RQ7 -- Does threat-flagging ride on top?** Once behaviour is recognised by family, can "
        "'flag if it matches a known threat-shaped family' be shown to work, with error bars?",
        "**RQ8 -- Is the masquerade resolved in memory alone?** Do slowburn and pagefault separate "
        "in the memory signal *without* the disk channel -- confirming the spine needs only memory?"]),
    p("Theme A is the core. Themes B and C support and extend it; detection (Theme C) is "
      "deliberately downstream of recognition, not the headline."),
]),
("data-model", "How we look at the data", [
    p("The experiment is built on a **nested data model**. Each level is both a statistical unit "
      "and a cross-validation protocol, and the structure enforces the rules that protect us from "
      "false confidence."),
    fig("tree", "The nested data model and how holding out each level maps to a "
        "cross-validation protocol of increasing honesty."),
    p("**Duration and repetition are different axes, and the distinction matters.** A "
      "**repetition** is the *exact same workload at the same length, run again* -- it differs "
      "only through the signal's stochasticity (scheduler, memory layout, cache, host noise), so "
      "repetitions characterise within-workload variance. **Duration** (120 / 300 / 600 s) is a "
      "separate, controlled factor that exists in the current dataset only to answer RQ8 -- how "
      "short a capture can be. The go-forward design is therefore: use the duration sweep *once* "
      "to fix a working length, then capture **many repetitions at that single fixed length**. "
      "The current 3-durations x 2-repetitions structure is the RQ8 probe, not the permanent "
      "shape of the data."),
    p("The binding rule the tree enforces: **nothing below a workload counts as an independent "
      "workload.** A generalization claim must hold out a whole workload (LOWO) or a whole family "
      "(LOFO); repetitions and analysis windows feed variance estimation only. (All cell and "
      "family counts in this document are current figures and grow as we capture.)"),
]),
("overview", "Plan overview", [
    p("Work proceeds on two tracks in parallel -- offline analysis on data in hand, and (gated) "
      "new captures on the server -- across three waves. A single decision gate governs whether "
      "any server time is spent. The whole plan is **memory-only**."),
    fig("overview", "Two tracks, three waves, one gate. Wave 1 (offline) builds the per-family "
        "recogniser from the memory signal; the gate releases server captures only if the offline "
        "pilot passes; either gate outcome leaves a complete, honest thesis."),
    h3("Wave 1 (offline, existing data) -- build the recogniser"),
    ul(["**Separability + cohesion** -- the workload-level matrix, blocked by family: which "
        "behaviours alias, and how cohesive each family is (RQ1, RQ3).",
        "**Family-normal pilot** -- on the families that already have enough workloads "
        "(ransomware, mem_sweep), train a family-normal model and test it on a held-out, unseen "
        "member (RQ2); run the bank vs reject-option bake-off (RQ4).",
        "**Open-set recogniser** -- hold out a whole family, check it is flagged NOVEL; report a "
        "leak-free operating point and per-workload recall.",
        "**Confidence intervals + significance** on every headline number (workload / family unit)."]),
    h3("Wave 2 (offline) and Wave 3 (server, gated)"),
    ul(["Wave 2: the **capture-economy** study -- trajectory decimation and truncation -- fixing "
        "the minimum useful cadence and length (RQ6). The Plan 06 disk experiments are dropped "
        "from the spine.",
        "Wave 3: **capture more workloads** to lift the small families and add the absent IO and "
        "IDLE families, and (if justified) a graded-Hamming re-capture -- only after the gate passes."]),
]),
("metrics", "Metrics: existing and new", [
    p("The features split into level (magnitude) and shape (scale-invariant). The strong "
      "one-class model leans on magnitude features, which is also its weakness: magnitude is "
      "a poor proxy for the threat-shaped class here (the stealth simulation slowburn is *quieter* "
      "than the benign mmap is loud). Plan 07 therefore tests magnitude, scale-invariant, and "
      "combined feature sets, and adds a new magnitude-aware-but-cheap metric. The recogniser uses "
      "**memory features only**; the disk features remain in the inventory as a second-channel "
      "**ablation**, not part of the spine."),
    fig("features", "The feature taxonomy. Existing memory and disk features (left) plus the "
        "new per-page Hamming scalars (right), with the motivation for the new metric."),
    METRIC_TABLE,
    callout("new", "Why the new metric, concretely",
      "APF asks only 'did this page change (yes/no)'. A stealth encryptor changes few pages "
      "(low APF, looks quiet) but fully encrypts each one (~half its bits flip). "
      "mean-Hamming-per-changed-page stays high even when the page count is low -- it can see "
      "what APF cannot. It is stored next to APF (the baseline) so the comparison is built in, "
      "and it costs a few scalars per snapshot, not per-page storage."),
]),
("tests", "Tests: what we have and what to capture", [
    p("The current dataset is 11 workloads across 5 families. The families are small and uneven "
      "in their number of distinct workloads, and that -- not the cell count -- is the binding "
      "constraint on the per-family work. (These figures are provisional and grow with capture.)"),
    FAM_TABLE,
    p("But the 66-cell subset is only a fraction of what is available. The built catalogue is far "
      "larger -- whole families (IO, IDLE) and several app and mem variants were never captured -- "
      "and the capture plan should draw on **all** of it for diversity, not merely fill the thin "
      "families."),
    CATALOG_TABLE,
    p("Because server access is available, we capture broadly. The priority axis is **distinct "
      "workloads** (which test generalisation); **repetitions at a fixed length** are the second "
      "axis (which estimate variance). Most of the expansion is **capture-only** -- the "
      "executables already exist -- so it costs server time, not new code; only the scanner family "
      "needs new programs written."),
    CAPTURE_TABLE,
    p("On duration: even 120 s already clears the analysis-window floor, so short captures are "
      "viable. We recommend **300 s** as the working length (a good window yield at half the cost "
      "of 600 s), confirm the true minimum offline by **truncating** existing 600 s traces, and "
      "thereafter capture repetitions at that single fixed length."),
    fig("windows", "Windows per cell versus capture duration (measured). Even the shortest "
        "duration clears the starvation floor; 300 s is the recommended balance."),
    WINDOWS_TABLE,
]),
("paths", "Two modelling paths", [
    p("Two distinct, complementary modelling paths run in parallel. The **per-family path is the "
      "centrepiece** (the recogniser); the per-workload path supports it with instance-level "
      "separability. They are not competitors."),
    fig("two_paths", "The per-workload path (instance identification) and the per-family path "
        "(learn a family's normal routine). The per-family path both characterises the family "
        "and yields an open-set normality model."),
    p("**Cohesion is a research question, not an assumption.** Pooling a family 'as one' only "
      "describes it well if its workloads resemble each other -- and whether they do is itself a "
      "finding (RQ3). If a family is cohesive, its label maps to a real shared memory signature. "
      "If it is not (an early data point: ransomware's leave-one-family-out recall was only "
      "0.25), then the semantic family does **not** correspond to a single memory routine, which "
      "is a stronger and more interesting result -- it forces sub-family granularity. We "
      "quantify cohesion as a number (a between-family / within-family ratio plus within-family "
      "generalization) and also test it without labels: does clustering the memory signal "
      "recover the families we imposed?"),
]),
("ladder", "The separability ladder", [
    p("Separability is examined as a ladder of increasing difficulty. The master artifact is a "
      "**workload-level separability matrix ordered by family** -- the family-level view is "
      "derivable from it by averaging blocks, but not vice versa."),
    fig("ladder", "From 1:1 pairs (are two behaviours separable) up through N:1 crosses "
        "(does a group generalize to a held-out member / family). The N:1 rungs are exactly the "
        "leave-one-out generalization tests."),
    fig("sepmatrix", "The master artifact (schematic). Diagonal blocks measure within-family "
        "cohesion; off-diagonal blocks measure between-family separation; a cross-family low cell "
        "(green) is a masquerade. Run three times -- magnitude, scale-invariant, and combined -- "
        "to map which feature type resolves which pair."),
    p("A caution carried from the metrics section: per-trace normalization removes the absolute "
      "level, but it was the *level* that separated slowburn from pagefault. So magnitude and "
      "shape are likely **complementary** -- each resolves different pairs -- and the matrix is "
      "the instrument that shows, pair by pair, where each matters."),
]),
("flavors", "Global model, then per-family blocks", [
    p("All the models here are **novelty / one-class** -- learn what 'normal' looks like, flag "
      "what does not fit. They differ in **granularity**, and that granularity is the (a) -> (b) "
      "structure of the whole plan."),
    fig("flavors", "(a) one GLOBAL normal-model (output: normal / anomalous -- the detection "
        "corollary) specialises into (b) one normal-model PER FAMILY (output: which family, or "
        "NOVEL -- the recogniser, and the centrepiece). Within-trace streaming is an optional "
        "online variant, not the spine."),
    p("For the per-family blocks (b), one mechanism question is settled **empirically**, not by "
      "argument: a **per-family one-class bank** (one density model per family) versus a "
      "**discriminative classifier with a reject option** (open-set with two thresholds rather "
      "than five blocks). At the current sample size the two may be statistically "
      "indistinguishable, so the pilot runs **both on the same folds** and lets the data decide (RQ4)."),
    callout("warn", "The sharp failure mode to guard against",
      "The per-family rule 'clear it if it matches a known benign family' can launder the stealth "
      "simulation: slowburn (low APF) is closest to a benign family in a magnitude-driven space, so "
      "a benign block would clear it. Guard: run the pilot on magnitude-invariant shape features "
      "(apf_cov, peak2med, duty_gt05) and hold slowburn out as a probe, with its recall on the "
      "headline table so it can never hide in an average."),
]),
("methodology", "Methodology and statistical protocol", [
    p("The design's credibility rests on a few non-negotiable rules drawn from the review."),
    ul(["**Leak-free fitting.** The imputer, scaler, and any operating-point threshold are fit "
        "**inside each fold** on training-benign only. No global fit survives for any "
        "threshold/PR metric (the committed code fits these globally -- valid for ROC ranking, "
        "but a leak for an operating point).",
        "**Honest units.** Bootstrap confidence intervals resample whole **workloads** (11) for "
        "LOWO/recogniser claims and **families** (5) for LOFO; the (workload, duration) cluster "
        "bootstrap is a labelled sensitivity bound only, never the headline CI.",
        "**Significance by permutation/exact tests**, computed at the generalization unit -- not "
        "asymptotic tests, given n <= 11.",
        "**A regression anchor.** Reproduce the committed 0.925 / 0.933 before changing anything, "
        "then show what each fix moves."]),
    p("A structural limit must be stated honestly: with few independent units, some comparisons "
      "**cannot** reach significance regardless of the effect. These are reported descriptively, "
      "with confidence intervals, and never dressed up with a p-value."),
    SIGFLOOR_TABLE,
    p("Accordingly, claims are sorted into three tiers: a small set of **confirmatory** "
      "hypotheses (the family-normal recogniser vs a permutation null, the slowburn/pagefault "
      "separability in memory, the one-class normal-model vs chance) corrected together with "
      "Holm; **descriptive** results (all leave-one-family-out comparisons) reported as intervals "
      "with no significance verdict; and **effect-size-only** results (the masquerade separation "
      "magnitude). The false-positive "
      "benefit of the per-family scheme is, on current data, demonstrable only as a worked "
      "mechanism -- with at most ~5 honest novel-benign test points, its exact-McNemar floor is "
      "0.0625, so it cannot reach significance until the families grow."),
]),
("expected", "Expected results", [
    p("Each experiment is designed so that both outcomes advance the thesis. The table states, "
      "per experiment, what would confirm versus refute -- and what each means."),
    EXPECTED_TABLE,
    callout("good", "Why this is win/win",
      "If the per-family recogniser generalises and families prove cohesive, the thesis is a "
      "working **memory-only behaviour recogniser** with open-set novelty -- and detection follows "
      "as a corollary. If families prove *not* cohesive, that is itself a finding (the memory "
      "signal carves behaviour differently from our taxonomy), reported with error bars, and it "
      "redirects the capture toward the granularity that does work. Neither path is a dead end."),
]),
("risks", "Risks and honest scope", [
    ul(["**Small n.** 11 workloads / 5 families bound what any analysis can claim; several "
        "current 'findings' are descriptive, not significant, by construction.",
        "**Synthetic workloads.** External validity to real malware is untested; this is a "
        "controlled study of behaviour classes, not a field detector.",
        "**Capture-environment confound.** New captures must use the same host configuration as "
        "the original 66, or family identity could entangle with capture conditions.",
        "**Cohesion may be low.** The per-family path may show that families are not cohesive in "
        "memory -- reported as a finding, not hidden.",
        "**Per-page spatial detail is not retained.** The new Hamming scalars keep aggregate "
        "magnitude, not which pages changed; a future spatial analysis would need a richer "
        "capture.",
        "**Disk is out of scope on the spine.** The Plan 06 disk-channel result is reported only "
        "as an ablation; the recogniser claims nothing from it, keeping the contribution "
        "memory-only."]),
    p("The plan is deliberately staged so that the expensive, irreversible step (server capture) "
      "is the last one, and is taken only after the cheap, offline steps have shown it is worth "
      "taking."),
]),
]

# ---------------------------------------------------------------- inline markup
def _md(s, esc):
    s = esc(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"\*(.+?)\*", r"<i>\1</i>", s)   # single-asterisk italics
    return s

def _esc_html(s): return _html.escape(s, quote=False)
def _esc_rl(s): return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

CALLOUT_COLOR = {"info": P["old"], "new": P["new"], "warn": P["threat"],
                 "good": P["benign"], "neutral": P["muted"]}

# ---------------------------------------------------------------- HTML renderer
def render_html():
    css = f"""
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=Crimson+Pro:wght@400;600;700&display=swap');
:root{{--ink:{P['ink']};--muted:{P['muted']};--line:{P['line']};--paper:{P['paper']};
  --panel:{P['panel']};--old:{P['old']};--new:{P['new']};--threat:{P['threat']};--benign:{P['benign']};}}
*{{box-sizing:border-box}}
body{{margin:0;background:#E8E6DF;color:var(--ink);
  font-family:'Atkinson Hyperlegible',system-ui,Segoe UI,Roboto,sans-serif;line-height:1.65;font-size:16px}}
.page{{max-width:880px;margin:28px auto;background:var(--paper);
  box-shadow:0 2px 18px rgba(0,0,0,.12);padding:54px 64px 72px;border-radius:4px}}
h1{{font-family:'Crimson Pro',Georgia,serif;font-size:32px;line-height:1.15;margin:0 0 6px;font-weight:700}}
h2{{font-family:'Crimson Pro',Georgia,serif;font-size:23px;margin:42px 0 12px;font-weight:700;
  color:var(--ink);border-bottom:2px solid var(--line);padding-bottom:6px}}
h3{{font-family:'Crimson Pro',Georgia,serif;font-size:18px;margin:24px 0 8px;font-weight:600;color:#333}}
.lead{{color:var(--muted);font-size:15px;margin:0 0 4px}}
.secnum{{color:var(--new);font-weight:700;font-family:'Crimson Pro',serif}}
p{{margin:10px 0;max-width:70ch}}
ul{{margin:10px 0;padding-left:22px;max-width:72ch}} li{{margin:7px 0}}
figure{{margin:22px 0;text-align:center}}
figure img{{max-width:100%;border:1px solid var(--line);border-radius:6px;background:#fff}}
figcaption{{color:var(--muted);font-size:13.5px;margin-top:8px;text-align:left;max-width:72ch;margin-left:auto;margin-right:auto;font-style:italic}}
table{{border-collapse:collapse;width:100%;margin:14px 0;font-size:13.5px}}
th,td{{border:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top}}
th{{background:var(--panel);font-weight:700}}
tr:nth-child(even) td{{background:#FAF9F5}}
.tcap{{color:var(--muted);font-size:12.5px;font-style:italic;margin:-6px 0 16px}}
.callout{{border-left:5px solid var(--muted);background:var(--panel);padding:12px 16px;border-radius:0 6px 6px 0;margin:16px 0}}
.callout .ct{{font-weight:700;margin-bottom:3px}}
code{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.92em;background:#F0EEE8;padding:1px 4px;border-radius:3px}}
.toc{{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:14px 20px;margin:22px 0}}
.toc ol{{margin:6px 0;columns:2;font-size:14px}}
.foot{{margin-top:48px;border-top:1px solid var(--line);padding-top:14px;color:var(--muted);font-size:12.5px}}
@media print{{body{{background:#fff}}.page{{box-shadow:none;margin:0;max-width:none;padding:0.5in}}h2{{page-break-after:avoid}}figure,table{{page-break-inside:avoid}}}}
"""
    out = [f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
           "<meta name='viewport' content='width=device-width, initial-scale=1'>",
           "<title>Plan 07 -- Experimental Design Handout</title>",
           f"<style>{css}</style></head><body><div class='page'>"]
    out.append("<p class='lead'>Memory-Signal Behaviour Detection &middot; Thesis Experiment Plan</p>")
    out.append("<h1>Plan 07 &mdash; Is detection real? An offline-first re-test of the memory-signal spine</h1>")
    out.append(f"<p class='lead'>Comprehensive experimental-design handout &middot; {DATE}</p>")
    # TOC
    out.append("<div class='toc'><b>Contents</b><ol>")
    for sid, title, _ in SECTIONS:
        out.append(f"<li><a href='#{sid}' style='color:var(--old);text-decoration:none'>{_esc_html(title)}</a></li>")
    out.append("</ol></div>")
    for i, (sid, title, blocks) in enumerate(SECTIONS, 1):
        out.append(f"<h2 id='{sid}'><span class='secnum'>{i}.</span> {_esc_html(title)}</h2>")
        for b in blocks:
            t = b[0]
            if t == "p":
                out.append(f"<p>{_md(b[1], _esc_html)}</p>")
            elif t == "h3":
                out.append(f"<h3>{_esc_html(b[1])}</h3>")
            elif t == "ul":
                out.append("<ul>" + "".join(f"<li>{_md(it, _esc_html)}</li>" for it in b[1]) + "</ul>")
            elif t == "fig":
                b64 = FIGS[b[1]][0]
                out.append(f"<figure><img alt='{_esc_html(b[2])}' src='data:image/png;base64,{b64}'>"
                           f"<figcaption>{_md(b[2], _esc_html)}</figcaption></figure>")
            elif t == "callout":
                _, kind, title2, text = b
                c = CALLOUT_COLOR[kind]
                out.append(f"<div class='callout' style='border-left-color:{c}'>"
                           f"<div class='ct' style='color:{c}'>{_esc_html(title2)}</div>{_md(text, _esc_html)}</div>")
            elif t == "table":
                _, header, rows, cap = b
                out.append("<table><thead><tr>" + "".join(f"<th>{_md(h, _esc_html)}</th>" for h in header) + "</tr></thead><tbody>")
                for r in rows:
                    out.append("<tr>" + "".join(f"<td>{_md(c, _esc_html)}</td>" for c in r) + "</tr>")
                out.append("</tbody></table>")
                if cap:
                    out.append(f"<div class='tcap'>{_md(cap, _esc_html)}</div>")
    out.append("<div class='foot'>Plan 07 design handout. Data figures use verified project "
               "results; design diagrams are schematic. Generated by "
               "<code>plan07_campaign/make_plan07_handout.py</code>.</div>")
    out.append("</div></body></html>")
    (DOCS / "plan07_handout.html").write_text("\n".join(out))
    print(f"html -> {DOCS / 'plan07_handout.html'}")

# ---------------------------------------------------------------- PDF renderer
def render_pdf():
    ss = getSampleStyleSheet()
    ink = colors.HexColor(P["ink"]); muted = colors.HexColor(P["muted"])
    H1 = ParagraphStyle("H1", parent=ss["Title"], fontName="Times-Bold", fontSize=22,
                        textColor=ink, spaceAfter=4, leading=26)
    LEAD = ParagraphStyle("LEAD", parent=ss["Normal"], fontName="Helvetica", fontSize=10,
                          textColor=muted, spaceAfter=2)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Times-Bold", fontSize=15.5,
                        textColor=ink, spaceBefore=16, spaceAfter=7)
    H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontName="Times-Bold", fontSize=12,
                        textColor=colors.HexColor("#333333"), spaceBefore=9, spaceAfter=3)
    BODY = ParagraphStyle("BODY", parent=ss["Normal"], fontName="Helvetica", fontSize=10,
                          leading=14.5, spaceAfter=6, textColor=ink)
    LI = ParagraphStyle("LI", parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=4)
    CAP = ParagraphStyle("CAP", parent=ss["Normal"], fontName="Helvetica-Oblique", fontSize=8.5,
                         textColor=muted, leading=11, spaceAfter=10, spaceBefore=2)
    TCAP = ParagraphStyle("TCAP", parent=CAP, spaceBefore=3, spaceAfter=12)

    def para(s): return Paragraph(_md(s, _esc_rl), BODY)
    story = []
    story.append(Paragraph("Memory-Signal Behaviour Detection &middot; Thesis Experiment Plan", LEAD))
    story.append(Paragraph("Plan 07 &mdash; Is detection real?", H1))
    story.append(Paragraph("An offline-first re-test of the memory-signal spine", ParagraphStyle(
        "sub", parent=H1, fontSize=14, spaceAfter=4)))
    story.append(Paragraph(f"Comprehensive experimental-design handout &middot; {DATE}", LEAD))
    story.append(Spacer(1, 6))
    # TOC
    toc = [[Paragraph(f"<b>{i}.</b> {_esc_rl(t)}", BODY)] for i, (_, t, _) in enumerate(SECTIONS, 1)]
    half = (len(toc) + 1) // 2
    tdata = [[toc[i][0], toc[i + half][0] if i + half < len(toc) else ""] for i in range(half)]
    tt = Table(tdata, colWidths=[3.3 * inch, 3.3 * inch])
    tt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(P["panel"])),
                            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor(P["line"])),
                            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                            ("LEFTPADDING", (0, 0), (-1, -1), 10)]))
    story += [tt, Spacer(1, 8)]

    avail = 6.6 * inch
    for i, (sid, title, blocks) in enumerate(SECTIONS, 1):
        story.append(Paragraph(f'<font color="{P["new"]}">{i}.</font> {_esc_rl(title)}', H2))
        for b in blocks:
            t = b[0]
            if t == "p":
                story.append(para(b[1]))
            elif t == "h3":
                story.append(Paragraph(_esc_rl(b[1]), H3))
            elif t == "ul":
                for it in b[1]:
                    story.append(Paragraph(_md(it, _esc_rl), LI, bulletText=u"•"))
            elif t == "fig":
                path = FIGS[b[1]][1]
                iw, ih = ImageReader(path).getSize()
                w = min(avail, 6.6 * inch); h = w * ih / iw
                if h > 4.2 * inch:
                    h = 4.2 * inch; w = h * iw / ih
                story.append(KeepTogether([Image(path, width=w, height=h),
                                           Paragraph(_md(b[2], _esc_rl), CAP)]))
            elif t == "callout":
                _, kind, title2, text = b
                c = colors.HexColor(CALLOUT_COLOR[kind])
                inner = Paragraph(f'<b><font color="{CALLOUT_COLOR[kind]}">{_esc_rl(title2)}</font></b><br/>'
                                  + _md(text, _esc_rl), BODY)
                ct = Table([[inner]], colWidths=[avail])
                ct.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(P["panel"])),
                                        ("LINEBEFORE", (0, 0), (0, -1), 3, c),
                                        ("LEFTPADDING", (0, 0), (-1, -1), 12),
                                        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                                        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
                story += [Spacer(1, 4), ct, Spacer(1, 6)]
            elif t == "table":
                _, header, rows, cap = b
                ncol = len(header)
                hdr = [Paragraph(f"<b>{_md(h, _esc_rl)}</b>", ParagraphStyle("th", parent=BODY, fontSize=8.5, textColor=colors.white, leading=11)) for h in header]
                data = [hdr] + [[Paragraph(_md(c, _esc_rl), ParagraphStyle("td", parent=BODY, fontSize=8.3, leading=10.5)) for c in r] for r in rows]
                cw = [avail / ncol] * ncol
                tb = Table(data, colWidths=cw, repeatRows=1)
                tb.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), ink),
                                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAF9F5")]),
                                        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor(P["line"])),
                                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                                        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6)]))
                story += [Spacer(1, 4), tb]
                if cap:
                    story.append(Paragraph(_md(cap, _esc_rl), TCAP))

    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8); canvas.setFillColor(muted)
        canvas.drawString(0.8 * inch, 0.5 * inch, "Plan 07 design handout")
        canvas.drawRightString(7.7 * inch, 0.5 * inch, f"p. {doc.page}")
        canvas.restoreState()

    SimpleDocTemplate(str(DOCS / "plan07_handout.pdf"), pagesize=LETTER,
                      leftMargin=0.8 * inch, rightMargin=0.8 * inch,
                      topMargin=0.7 * inch, bottomMargin=0.7 * inch,
                      title="Plan 07 -- Experimental Design Handout").build(story, onFirstPage=_footer, onLaterPages=_footer)
    print(f"pdf  -> {DOCS / 'plan07_handout.pdf'}")


if __name__ == "__main__":
    render_html()
    render_pdf()
