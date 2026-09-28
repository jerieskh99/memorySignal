#!/usr/bin/env python3
"""latex_skeleton.py -- writes the paper 2 LaTeX skeleton: headings, table shells, figure
placeholders and comment blocks carrying P2_STRUCTURE.md's substance bullets. No body prose.

Builder 3 (report), 2026-09-16. SPEC 6.8 asks for a complete compilable document with the
section headings of P2 Sec. 3 (Abstract; I to X; Artifact and reproducibility), each followed
by a comment block `% - ...` listing P2's substance bullets in abbreviated form,
`\\input{tables/<name>.tex}` where the section's table belongs, and figure placeholders with
empty captions. The build brief (2026-09-16) asks for the IEEEtran conference class and one
table environment per table shell with the column headers and empty rows, so:

  - the document class defaults to `IEEEtran` (`[conference]`) and `--documentclass article`
    gives SPEC 6.8's fallback when IEEEtran.cls is not installed;
  - every generated table is brought in as `\\IfFileExists{tables/<name>.tex}{\\input{...}}{<shell>}`,
    where `<shell>` is the table environment with the column headers and one empty row, so the
    same skeleton compiles beside the generated tables (`<out>/report/`) and standing alone
    (`apf_paper/p2_skeleton.tex`);
  - every figure is `\\IfFileExists{figures/<name>.pdf}{\\includegraphics}{<framed placeholder>}`.

Static tables (P2 Sec. 4): Table 1 as a comment-only shell (III), Table 2 as a static tabular
(IV), Table 3 as a static tabular with the pass table's declared column merged from
`<out>/inputs/pass_table.csv` when it exists (VI), Table 4 as the plan/gate shell (V).
Not one sentence of paper body: the only text outside comments is headings, column headers
and the fixed cells of P2's own shells.

CLI (SPEC 7.1): latex_skeleton.py --out O [--documentclass IEEEtran|article]
                [--standalone PATH]   (also write a copy to PATH; NOT a live paper -- refused if it holds prose
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import argparse
import re
import traceback

from plan11_encoding_ladder._report_common import (  # noqa: E402
    ARCHETYPES, KERNELS, PACKAGE_VERSION, latex_escape, now_iso, read_csv, result_json, write_json,
)
from plan11_encoding_ladder.tables import (  # noqa: E402
    TABLE4_COLUMNS, TABLE5_COLUMNS, TABLE5_G3_COLUMNS, TABLE6_COLUMNS, TABLE7_COLUMNS,
    TABLEGV_COLUMNS, TABLE_COMPARATORS_COLUMNS, WAPF_COLUMNS,
)

CITATION = "P2 Sec. 3 (section skeleton), Sec. 4 (table shells), Sec. VII (figures); SPEC 6.8"

# ----------------------------------------------------------------------------------------------
# P2 Table 2 and Table 3 static cells (P2_STRUCTURE.md section 4, verbatim cells)
# ----------------------------------------------------------------------------------------------
TABLE2_ROWS = [
    ("0", "APF", "breadth", "one scalar $K_t / N$", "amount, identity, position", "none beyond the row count"),
    ("0'", "wAPF", "breadth x amount folded", "one scalar", "the shape of the amount distribution, identity", "one sum"),
    ("1", "Persistence (Jaccard)", "identity over time", "one scalar $J(t)$", "amount, breadth (normalized out)", "one set intersection per pair"),
    ("2", "Content-change distribution", "amount", "three ratios + fixed quantiles", "identity, position", "one pass over rows"),
    ("3", "Combined", "all three", "union of level-normalized features", "position", "the three above"),
]
TABLE3_ROWS = [  # kernel, archetype, base parameters, footprint (pages, INFERRED), steady-state stores change content?, pass period (P2 Table 3)
    ("gemm", "WORKING-SET", "--dim 1024 --block 64", "4,096 rewritten per pass (A, C); B read-only", "yes", "undeclared"),
    ("floyd", "WORKING-SET", "--dim 1024", "2,048", "yes", "undeclared"),
    ("gibbs", "WORKING-SET", "--width 1024 --height 1024 --states 2 --beta-milli 400", "256", "yes", "undeclared"),
    ("nbody", "WORKING-SET", "--particles 262144 --neighbors 16", "2,048 (8 MiB, from metadata)", "yes", "declared: 6,147 steps / 600 s, aliased by design"),
    ("spmm", "WORKING-SET", "--rows 4096 --inner 4096 --cols 64 --nnz-per-row 16", "", "", "undeclared"),
    ("stencil_jacobi", "WORKING-SET", "--grid-n 1024", "", "yes", "undeclared"),
    ("fft", "SCATTER", "--n 1048576", "4,096", "yes", "undeclared"),
    ("histogram", "SCATTER", "--samples 50000000 --bins 1048576 --dist uniform", "2,048 (int64 bins)", "only through pass phase", "undeclared"),
    ("fem_assembly", "SCATTER", "--nodes 2048 --elements 8192 --npe 4", "8,192 + 64", "yes", "undeclared"),
    ("lexer", "SEQUENTIAL-GROW", "--input-mb 32", "tokens of one pass", "no (after the first pass)", "undeclared"),
    ("rmat_gen", "SEQUENTIAL-GROW", "--scale 18 --edge-factor 16 --a-milli 570 --b-milli 190 ...", "", "", "undeclared"),
    ("bnb_tsp", "FRONTIER-CHURN", "--cities 13", "peak frontier x 40 B", "yes", "undeclared"),
    ("idle (to capture)", "control", "sleep 600", "floor", "no", "n/a"),
]
TABLE4_ROWS = [  # plan, fixes, gates, scope, APF at 500 ms, wAPF, persistence, content-change (P2 Table 4)
    ("02", "interval; session validity", "C1-C8 (C1 re-mapped)", "per dataset", "not yet run", "cited", "cited", "cited"),
    ("03", "window, hop", "G1-G5 as amended, G-ORD", "per encoding", "", "", "", ""),
    ("04", "segment count k", "segmenter gates", "per encoding, if kept", "", "", "", ""),
    ("05", "throughput levers", "G-T1, G-T2, G-T3", "per dataset / G-T2 per encoding", "n/a (no lever); G-T2 inherited", "n/a", "n/a", "n/a"),
    ("07", "scale-up", "stage 13", "per dataset", "cited", "cited", "cited", "cited"),
    ("08", "analysis validity", "B1-G1 to G6, restated", "per encoding", "", "", "", ""),
    ("09", "campaign worth", "Gates 0-3", "per dataset", "passed (numbers with the author)", "cited", "cited", "cited"),
    ("council", "preconditions, calibration, readings, comparisons",
     "G-K0, G-F, G-C, G-P, G-J, G-DEC, G-L, G-N, G-X, G-DIM, G-M, G-V", "per rung / per dataset", "", "", "", ""),
]
TABLE1_FIELDS = [  # comment-only shell (SPEC 6.8: III)
    "Labels | dwarfs1 (+_resume, _resume3), sandbox_deepdive_01c, sandbox_deepdive_01c1",
    "Commit | fcc184e (all three)",
    "Dates | 2026-09-06, 2026-09-07, 2026-09-08 (stencil re-diff 2026-09-14/15)",
    "Cells | 96 = 12 kernels x 8 reps",
    "Kernels | bnb_tsp, fem_assembly, fft, floyd, gemm, gibbs, histogram, lexer, nbody, rmat_gen, spmm, stencil_jacobi",
    "Reps per kernel | 8 (rep 0 seed 42; reps 1-7 per-kernel series, one thousand apart)",
    "Scale | 1.0 only",
    "Duration per cell | 600 s guest-running time (kernel's monotonic loop; --duration 600, SUSTAIN_LOOP=1)",
    "Wall time per cell | about 55-60 min (host-stamped capture jobs, median gap 2.6-3.6 s)",
    "Snapshot interval, configured | 500 ms",
    "Realized pairs per cell | 890 to 945 (one cell: 931)  [toolkit: sidecar n_pairs]",
    "Guest spacing per pair | about 645 ms, derived (600 s / pairs); not recorded per snapshot  [toolkit: sidecar dt_est_s]",
    "Guest RAM | 1024 MiB; N = 262,144 pages of 4 KiB",
    "Guest clocksource | kvm-clock",
    "Differ, mode, speed | live_delta_calc_modular, --sparse, speed 2 (from the launch configuration, not recorded per cell)",
    "Feature families | positional, distributional, informational, structure, level, change_location, texture (01c/01c1); the dwarfs1 cells re-diffed to the same set",
    "Retention | combined (zstd delta chains + trajectory); 747.9 GB total",
    "Throughput levers | none; timing log not enabled; dumps to the per-domain directory",
    "Idle cells | to be captured: 8 x 600 s sleep over SSH, same settings",
    "Cell order | OPEN (from the steps files)",
    "Sampling regularity | disclosure with the [0.500, 0.644] s bracket, not a gate (CR 2.3 items 36, 37)",
]

# ----------------------------------------------------------------------------------------------
# substance bullets per section (abbreviated from P2_STRUCTURE.md section 3; comments only)
# ----------------------------------------------------------------------------------------------
BULLETS = {
    "abstract": [
        "About 200 words. Written last, by the author.",
    ],
    "intro": [
        "Memory under a workload is a time signal; the delta between consecutive dumps is its derivative (paper 1's framing, cited).",
        "The field's quantity for the delta is a page count per pair (Law 2010, Savoldi 2010); paper 1 adopts it as APF, the minimal Representation.",
        "Two questions: at what resolution is each encoding of the delta adequately sampled; what does each keep of the workload.",
        "Contributions (one sentence each, final once the tables fill): (1) the gate chain fixing a realization's free parameters from the form, declared before the data, executed for APF and re-run per encoding; (2) the three-axis reading of the delta (breadth, amount, identity over time) and the ladder of reductions computable from one retained artifact; (3) the level-matched tests and the store-predicted vs state-measured taxonomy, with null floors.",
        "What this paper is not: not the form (paper 1); not a detector (the sandbox family is absent); not a spectral study of the byte domain; not the spatial-blocking study (Sec. VIII names it).",
    ],
    "background": [
        "Repeated memory acquisition and page-level differencing: Law 2010, Savoldi 2010.",
        "Atomicity of a dump; hypervisor suspension as the atomic case: Oliveri and Balzarotti 2025, Vomel and Freiling 2012, Pagani 2019.",
        "Time-sequenced hypervisor-level acquisition with a classifier on telemetry: the Purnaye lineage.",
        "The gated-selection lineage for Sec. V: van der Kouwe et al. 2019 (configurations chosen on the test data), Kalibera and Jones 2013, Hoefler and Belli 2015 (al-Nadim's Yes verdicts).",
        "Workload identification from memory behaviour; the Berkeley dwarfs (Asanovic et al. 2006); per-page change magnitude as a feature.",
        "Nearest published line: Hirano and Kobayashi (CSR 2022; RanSMAP, Computers and Security 2025; RanSAP, FSI:DI 2022): first-touch write faults per 30 s flush epoch vs changed pages per 500 ms pair (council/13_).",
        "Every citation and the sentence it supports: council/11_al_nadim_final.md section 4.",
    ],
    "apparatus": [
        "One paragraph: the QEMU pipeline, suspend-dump-resume, producer/consumer queue, sparse differ at speed 2, per-cell substrate trajectory, zstd delta-chain retention; cite paper 1 Sec. VII.",
        "Table 1, campaign identity (run records and the author's answers; the toolkit's sidecars supply n_pairs and dt_est_s).",
        "The time axes: A guest-running interval; B host wall clock; C analysis frame spacing = A; D throughput. Guest clocksource kvm-clock; both guest clocks freeze under suspension (AA S5, A7).",
        "Guest-running time per cell 600 s; guest spacing per pair about 645 ms (600 s over the pair count); wall time per cell about an hour.",
        "Every temporal quantity in pair units; seconds and hertz under both ends of the [0.500, 0.644] s bracket; no sentence mixes the two clocks; realized spacing not recorded per snapshot (disclosure, not a gate).",
        "The retained chains: every encoding recomputable from the raw states (stencil_jacobi trajectories regenerated from chains 2026-09-14/15); paper 1's 'computed, not merely recoverable' rests on this.",
    ],
    "ladder": [
        "The primitive: per pair, one row per changed page with page_index and a 64-metric vector (sparse mode, hamming != 0); the (seq, page_index) support is the changed-page indicator; every rung is a function of this artifact.",
        "Rung 0, APF (breadth): K_t / N, N = 262,144; the root and the named prior method; the footprint-size trajectory is APF's whole-cell reading, not a separate rung. Expectation: mean is level; variance may carry an alias beat of a pass rhythm above Nyquist (alias falsifier). One cell computed 2026-09-14 (bnb_tsp rep 1): 931 pairs, mean 0.0173, min 0.0048, max 0.0917.",
        "Rung 0', wAPF (breadth folded with amount): sum(hamming) / (N x page bits); the negative control for folding an axis into a count; wAPF over APF = mean flipped bits per changed page. Expected to separate floyd from histogram crudely and lose to rung 2 where the shape carries the difference.",
        "Rung 1, persistence (identity over time): J(t) between S_t and S_{t+1}; two nulls (independence K_t K_{t+1} / N; the idle cells' own J, G-J); calibration pulse gemm's re-seed dips J toward one half at pass boundaries; read beside G-P (J near one is 'aliased by design' when every pass fits inside the interval); the per-cell failed/ count is zero by construction (AA A5). Expectation: J near one for fft, histogram, fem_assembly, gibbs, nbody; bnb_tsp high; lexer at the floor; gemm regime-dependent.",
        "Rung 2, content-change distribution (amount): channels l0, l1, hamming; l2 and linf out of the decay reading; mean_abs only as the calibration ordering; fixed summary = three level-free ratios l0/4096, l1/l0, hamming/l0 (near one for a counter bump, near four for a re-randomized double), fixed quantiles secondary, over pages present in both snapshots; decay on l0 first, l1 second, hamming as direction; floyd the exhibit (G-P three or more snapshots per solve), gibbs the control. K is not a feature of this rung (G-L). Expectation: mean_abs gibbs < histogram < gemm; l0 histogram < gibbs < gemm; floyd and histogram separated here and nowhere above.",
        "Rung 3, combined: the union of level-normalized per-rung features keyed by (cell, seq), with G-DIM's dimension-matched comparison and its own null.",
        "Named, seen, not used: the within-page channels (changed_runs, change_span, longest_changed_run, polarity, mean_shift; the blocking paper's); address-shape features (Gate 2); the byte-domain FFT/wavelet/cepstrum (aliased at 2 Hz); spatial blocking (Sec. VIII, parked).",
        "Table 2, the ladder.",
    ],
    "gates": [
        "The bridge to paper 1: the form fixes what every realization must satisfy; a realization still has free parameters (interval, window, hop, segment count, differ speed, throughput levers, retention); this section is how those were fixed for APF without post-hoc selection, plan by plan, and how the chain is re-run per rung.",
        "Al-Farabi's binding condition (F 1.5): every resolution grid declared before the data, every grid point computed and kept, the selection rule declared, the selected point marked; (W, H) fixed per encoding, never per kernel (F question 8).",
        "Principles: metric named before data; one independent variable per pilot row; every decision saved in run metadata; a bounded single revisit; every gate can refuse and a refusal is written to the artifact, never silently absorbed.",
    ],
    "gates_51": [
        "Plan 02, C1-C8 (plan02_validate_session.py; re-mapped for apf_queue in validate_campaign.py); C1 re-mapped so an idle cell is not refused (al-Farabi); toolkit: gates/preconditions.csv (SPEC 3.3.1).",
        "Plan 03, G1-G5 as amended, grid W in {8, 16, 32, 64, whole cell}, hop ratios as Plan 03, per encoding: G1 keeps z 1.0 and the 0.80 floor and gains a surrogate null (a trend hands the cell to the whole-cell reading); G2 in pair units with the rhythm declared per kernel or 'undeclared', floor 2.0, at dt 0.500 and 0.644 s, 'not applicable, rhythm above Nyquist' when T < 2 dt; G3 renamed a per-kernel signal flag off the (W, H) decision (option (a), decided 2026-09-16); G4 unchanged; G5 reported not gated; the Delta-5 guard not applied; plus G-ORD, the time-shuffle null per W.",
        "Plan 04: the change-point view is not in this paper (decided 2026-09-16); cited and skipped.",
        "Plan 05, G-T1/G-T3 'not applicable' (no lever in 01c, AA A6); G-T2 'not applicable, inherited by A1 and A3' with the argument stated once (decided 2026-09-16).",
        "Plan 07, scale-up, the stage-13 gate: cited from the campaign's record.",
        "Plan 08 (B1), B1-G1 to G6 restated at the unit of the split: the label-shuffle null at the unit (LOKO: archetype labels across the 12 kernels; LORO: kernel labels across cells), at least 500 permutations, strict exceedance of p95, rank reported, near_unfalsifiable excluded from every table; B1-G3 as a one-feature threshold model reproducing the full model on all but at most one unit, then quarantine and re-run; B1-G6 at the unit. Recorded as changes to a pre-registered gate (ML question 6).",
        "Plan 09 (dwarf pilot), Gates 0-3: passed; numbers with the author (Table 4 row 09; OPEN whether it authorized 01c or was read afterwards).",
    ],
    "gates_52": [
        "Preconditions of valid observation (per cell): C1-C8 with C1 re-mapped; the per-cell failed/ count (zero or the seq axis corrected; min); G-K0 state-change disclosure (source line per kernel in Table 3; measured: median K over the last 80 percent vs the idle floor's 95th percentile; inside the band -> 'IDLE, measured', removed from the recovery denominator; min); G-F floor (part (i) idle reps mutually inseparable under the rung or the rung is void; part (ii) per kernel the headline reading outside the idle envelope or 'at floor in this lead', a finding; admissibility of the idle control stated; the three floors K, l0 per changed page, J; with no admissible idle cell the tripwire row is 'not run').",
        "The tripwire clause, corrected wording (VME): idle reps must be mutually inseparable under every rung; a kernel inseparable from idle is at floor in that lead, never 'the instrument hears the host'.",
        "Calibration (per rung): G-C, the calibration pulse (min): gemm's re-seed for APF and persistence (a two-to-one jump predicted, detected at 1.5, and a J dip toward one half within one snapshot of the boundary, every rep), the content orderings mean_abs gibbs < histogram < gemm and l0 histogram < gibbs < gemm (eight of eight); refuses the whole rung as a disconnected lead; 'pulse aliased by design' when the full footprint is lit every snapshot in every rep (al-Kindi review 2).",
        "Readings (per rung, per kernel): G-P pass period (min; T >= 4 dt resolvable, 2 dt <= T < 4 dt marginal, T < 2 dt aliased by design, unknown undeclared; five passes per cell for a rhythm feature, three snapshots per pass for a within-pass feature; declared for nbody only, undeclared for the other eleven, AA A7); G-J the persistence null (independence null; idle J as the empirical floor; interpret J only where K_t > 3 x the floor's median K; no floor subtraction, decided 2026-09-16); G-DEC decay validity (l0 first, l1 second; monotone fall over three snapshots inside a pass in seven of eight reps; K not falling more than median l0; block-shuffle surrogate; idle no slope of the same sign; refusals 'no decay beyond breadth', 'no decay beyond floor or host', 'decay not resolved').",
        "Comparisons (per split): G-L level blindness (min; part (i) the normalization rule fixed per rung, the normalized LOKO archetype score above the kernel-level shuffle null's p95 or 'level only'; part (ii) within-window CV regressed on 1/sqrt(K_median), refused when r2 > 0.5 until cov and its relatives are dropped); G-N class support (min; three kernels for a headline row; two 'one training kernel per fold'; one a structural novelty); G-X cross-campaign grouping (the leak measured blind: campaign predicted under LOKO against the unit null; then ML's confound rule; three labels); G-DIM dimension parity (feature count on every row; the combined rung's matched comparison; a declared reduction when d exceeds the cell count); G-M paired margin (spread across seeds on the APF arm; sign test on 12 kernels: six up and none down, or seven up and one down; otherwise a difference with its margin).",
        "The splits: within-trace as a ceiling only; LORO an instrument's reading; LOKO the headline; unit = cell; whole cells held out. Models: random forest; clustering with k fixed to the number of predicted archetypes present, one primary algorithm, ARI and NMI against the unit-level null; no AE bank.",
        "Reports and disclosures: G-V variance decomposition (L0 within-kernel, L2 within-archetype, L3 between-archetype; 'LOKO not estimable' when L0 > L3 for every feature); sampling regularity as a Table 1 disclosure; G3 as a per-kernel flag.",
    ],
    "gates_53": [
        "wAPF, persistence, content-change, combined: the per-encoding gates (G1-G5 as amended, G-ORD, B1-G1 to G6 as restated, G-C, G-P, G-L, G-DEC where a decay is claimed, G-DIM, G-M) re-run; the per-dataset gates (C1-C8, the failed/ count, G-K0, G-F, G-T1/G-T3, stage 13, Gates 0-3, G-X) run once and cited.",
    ],
    "gates_54": [
        "Al-Kindi: none gates the hypothesis; every one is craft below the line; the author may prune. The minimum the blind moves need: the failed/ count, G-K0, G-C, G-P, G-L, G-N.",
    ],
    "design": [
        "Table 3, cells: the twelve kernels, predicted archetypes, base parameters, seeds, the state-change disclosure line, the pass-period status (declared for nbody only).",
        "The aliasing argument, stated once: 500 ms is a 2 Hz stroboscope; at base sizes it aliases every kernel's intra-pass geometry to 'footprint fully lit' unless a pass outlasts the interval; the sizing law cited from the dwarf design.",
        "Variance decomposition rationale (E1, 2026-08-29): L0 within-kernel (reps = noise floor), L2 within-archetype (cohesion), L3 between-archetype (separability).",
        "Reps: rep 0 at seed 42, reps 1-7 on a per-kernel series one thousand apart; every rep at scale x1; the guest rebooted per cell (cold boot); reps are genuine repeats; LORO is an unseen-run test.",
        "Cell order: OPEN, read from the steps files; G-X measures any leak blind.",
        "Warm-up: CAPTURE_WARMUP_SECONDS was 0; the drop-warm-up guard drops by phase marker, not by count; the lexer's first pass kept out of every steady-state statistic (toolkit: inputs/head_drop.csv, default 0).",
        "Level-matched sets (the claim's core): floyd, histogram, nbody at 2,048 pages; fft and gemm's per-pass rewrite at 4,096.",
    ],
    "results": [
        "Table 5: gate verdicts per rung at each grid point, refusals written not filled (toolkit report/tables/table5.tex, table5_g3.tex).",
        "Table 6: APF alone under the three splits, raw and level-normalized, unit-level null and majority on every row, G-N support, G-X result: the measured blind spot.",
        "Table 7: each rung at its gated resolution and the combination, three splits, null and majority on every row, G-C, G-F (i), G-L, G-DIM parity, G-M paired margin.",
        "Table 8: the LOKO assignment per kernel, read as assignments with counts (never as a confusion matrix in the statistical sense), beside the clustering; store-predicted rows against state-measured columns; the physical reason for every off-diagonal (the author's column).",
        "The G-V variance table.",
        "Figures: APF(t) per kernel (eight reps overlaid); the level-matched sets side by side; the fused plane (median l0 over persistent pages against J, one panel per kernel, idle cells if any); J(t) histograms per kernel with both nulls; the three-ratio histograms; floyd's within-solve l0 if G-DEC admits it; one cell's piano roll beside its APF and J; the Table 5 verdict grid.",
    ],
    "discussion": [
        "The blind spot as a property of the instrument, not of any one encoding: at this interval and these sizes, breadth is level; what separates kernels is amount and identity.",
        "Resolution as the organizing idea: what 500 ms can and cannot see; what a faster differ buys and costs (speed 0 at about 39 s per pair, offline only).",
        "The store taxonomy against the state taxonomy: why SCATTER and SEQUENTIAL-GROW empty (the lexer's front exists in the stores and not in the state; scattered stores overwritten or undone before the dump); why FRONTIER-CHURN reads as a working set with a fluctuating level.",
        "The spatial-blocking paragraph (the programme sentence with content): the same ladder per block of pages sliding across the address space gives a space-by-time field per metric; the raw block reading encodes allocator placement below the whole address space; a placement-invariant reading is required; the block size at which placement dominates is a measurable property of the guest (PARKED_spatial_blocking_paper.md).",
        "Contiguity is a property of this guest (buddy allocator, THP opt-out); forcing THP is a lever for the blocking paper.",
        "The time axes: both guest clocks freeze under suspension (measured); the realized spacing is derived, not recorded; stated, not hidden.",
        "What transfers to forensics: detection and masquerade need the sandbox family; the RAID paper.",
    ],
    "limitations": [
        "One guest, one hypervisor, one RAM size, one interval captured; twelve kernels; two archetypes with fewer than three kernels; pass period declared for one kernel; reps within one campaign across three launch labels (G-X); Gate 2 on one kernel at one size; idle cells captured after the kernels (image-state mismatch stated).",
        "The failed/ count of zero is a declared assumption (AA A5; the orchestrator's re-run path at fcc184e to be verified) carried in every persistence reading's params (al-Farabi review, for the author 5).",
    ],
    "conclusions": [
        "One paragraph; the programme sentence names the blocking paper and the RAID paper.",
    ],
    "artifact": [
        "Same repository as paper 1 (FAIR decision ~20 September 2026); every run record carries its git_sha; three labels, one commit.",
        "The retained chains: 96 cells, 747.9 GB, 890 to 945 dumps of 1 GiB each per cell (per cell 767 MB to 23.3 GB, median 8.0 GB); every encoding recomputable from them (PARKED_retention_paper.md).",
        "The gate code at a named commit (plan02_validate_session.py, plan05_campaign/validate_campaign.py, plan03_sweep.py, plan03_aggregate.py, plan05_fidelity.py, plan08_b1/, plan09_dwarf_pilot/), re-pointed at the substrate trajectories through plan11_encoding_ladder/ (the one-pass extractor of move 1 is new code).",
        "A DOI deposit of the trajectories is al-Nadim's first addition for the C&S extension.",
    ],
}

FIGURES = [  # (file stem, label, comment)
    ("apf_per_kernel", "fig:apf_per_kernel", "APF(t) per kernel, eight reps overlaid, one panel per kernel, log y; idle as a 13th panel when present (K2 move 5)"),
    ("level_matched", "fig:level_matched", "the level-matched sets side by side: A floyd, histogram, nbody; B fft, gemm; median line per kernel"),
    ("fused_plane", "fig:fused_plane", "per snapshot (median l0 over persistent pages, J to the next snapshot), one panel per kernel, idle in grey, G-J mask (masked pairs hollow) (K2 move 8)"),
    ("j_hist", "fig:j_hist", "J(t) histograms per kernel with the independence null and the floor null (idle J quantiles)"),
    ("ratio_hist", "fig:ratio_hist", "histograms of the three ratios' per-snapshot medians per kernel"),
    ("floyd_decay", "fig:floyd_decay", "floyd's within-pass median l0 aligned at the boundary, reps overlaid; a placeholder with the G-DEC verdict when not admitted"),
    ("piano_roll", "fig:piano_roll", "one cell's changed-page raster beside its APF and J"),
    ("table5_grid", "fig:table5_grid", "the Table 5 verdict grid as a categorical heat map, one panel per rung"),
    ("dhodapkar_sweep", "fig:dhodapkar_sweep", "the Dhodapkar-Smith threshold sweep per kernel: stability (solid) and mean phase length (dashed, log) against delta_th, the declared default as a dotted line (build epoch 2)"),
]


# ----------------------------------------------------------------------------------------------
# builders
# ----------------------------------------------------------------------------------------------
def _comment_block(lines) -> str:
    return "\n".join(f"% - {ln}" for ln in lines) + "\n"


def _shell(columns, label, *, wide=None, note="", rows=None, size="footnotesize") -> str:
    """A table environment with the column headers and one empty row (or the fixed rows)."""
    if wide is None:
        wide = len(columns) > 7
    env = "table*" if wide else "table"
    out = [f"% columns: {', '.join(columns)}"]
    if note:
        out.append(f"% {note}")
    out += [f"\\begin{{{env}}}[t]", "\\centering", f"\\{size}", "\\caption{}", f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{'l' * len(columns)}}}", "\\toprule",
            " & ".join(latex_escape(c) for c in columns) + " \\\\", "\\midrule"]
    if rows:
        for r in rows:
            out.append(" & ".join(_cell(x) for x in r) + " \\\\")
    else:
        out.append(" & ".join([""] * len(columns)) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}"]
    return "\n".join(out) + "\n"


def _cell(x: str) -> str:
    """Escape a fixed cell; a math span (`$...$`, whole cell or inline) is left as it is (CHECK_2 M13)."""
    x = "" if x is None else str(x)
    return "$".join(seg if i % 2 else latex_escape(seg) for i, seg in enumerate(x.split("$")))


def _generated(name: str, shell: str) -> str:
    """`\\IfFileExists{tables/<name>.tex}{\\input{tables/<name>.tex}}{<shell>}`."""
    return (f"\\IfFileExists{{tables/{name}.tex}}{{\\input{{tables/{name}.tex}}}}{{%\n{shell}}}\n")


def _figure(stem: str, label: str, comment: str, wide: bool = False) -> str:
    env = "figure*" if wide else "figure"
    return (f"% figure: {comment}\n"
            f"\\begin{{{env}}}[t]\n\\centering\n"
            f"\\IfFileExists{{figures/fig_{stem}.pdf}}{{\\includegraphics[width=\\linewidth]{{figures/fig_{stem}.pdf}}}}"
            f"{{\\framebox[\\linewidth]{{\\rule{{0pt}}{{0.3\\linewidth}}}}}}\n"
            f"\\caption{{}}\n\\label{{{label}}}\n\\end{{{env}}}\n")


def _table3_rows(pass_table_csv: Path | None):
    """P2 Table 3 with the pass table's declared column merged when `inputs/pass_table.csv`
    exists (SPEC 3.4.1: `kernel, passes_per_600s, source, notes`)."""
    rows = [list(r) for r in TABLE3_ROWS]
    if pass_table_csv and Path(pass_table_csv).exists():
        pt = {r.get("kernel"): r for r in read_csv(pass_table_csv)}
        for r in rows:
            e = pt.get(r[0])
            if e and str(e.get("passes_per_600s", "")).strip():
                r[5] = f"{e['passes_per_600s']} per 600 s ({e.get('source', '')})"
            elif e:
                r[5] = e.get("source", "undeclared") or "undeclared"
    return rows


def build_skeleton(*, documentclass: str = "IEEEtran", pass_table_csv: Path | None = None) -> str:
    """The skeleton text. `documentclass` is `IEEEtran` (conference) or `article`."""
    L = []
    if documentclass == "IEEEtran":
        L.append("\\documentclass[conference]{IEEEtran}")
    else:
        L.append("\\documentclass[10pt,twocolumn]{article}")
    L += [
        "% paper 2 skeleton, written by plan11_encoding_ladder/latex_skeleton.py (builder 3) on " + now_iso(),
        "% headings, table shells, figure placeholders and substance bullets only; the wording is the author's",
        "% source of truth: apf_paper/P2_STRUCTURE.md sections 3, 4, VII (2026-09-16)",
        "\\usepackage[utf8]{inputenc}",
        "\\usepackage[T1]{fontenc}",
        "\\usepackage{booktabs}",
        "\\usepackage{graphicx}",
        "\\usepackage{amsmath}",
        "\\graphicspath{{./}}",
        "",
        "% working title candidates (P2 Sec. 1; the author chooses):",
        "% 1. Encoding the memory delta: gated parameter selection for the Active Page Fraction and its successors",
        "% 2. What the Active Page Fraction keeps and what it destroys: a gated study of encodings of the memory delta on twelve numerical kernels",
        "% 3. Breadth, amount, identity: three reductions of the whole-memory delta under one gate chain",
        "\\title{}",
        "\\author{}",
        "",
        "\\begin{document}",
        "\\maketitle",
        "",
        "\\begin{abstract}",
        _comment_block(BULLETS["abstract"]).rstrip("\n"),
        "\\end{abstract}",
        "",
        "\\section{Introduction}",
        _comment_block(BULLETS["intro"]),
        "\\section{Background and prior work}",
        _comment_block(BULLETS["background"]),
        "\\section{Apparatus}",
        _comment_block(BULLETS["apparatus"]),
        "% Table 1, campaign identity (comment-only shell; SPEC 6.8):",
        "\n".join(f"%   {ln}" for ln in TABLE1_FIELDS),
        "",
        _generated("preconditions", _shell(["cell_id", "role", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8",
                                            "failed_verdict", "all_hard_pass"], "tab:preconditions", wide=True,
                                           note="Plan 02 C1-C8 as re-mapped, per cell (gates/preconditions.csv)")),
        "\\section{Encodings of the delta: the ladder}",
        _comment_block(BULLETS["ladder"]),
        _shell(["Rung", "Encoding", "Axis kept", "Output per snapshot", "Destroys", "Cost"], "tab:table2",
               wide=False, note="Table 2, the ladder (P2 Sec. 4, static cells)", rows=TABLE2_ROWS, size="scriptsize"),
        "\\section{The gate chain: how a realization's parameters are fixed from the form}",
        _comment_block(BULLETS["gates"]),
        "\\subsection{The chain as executed for APF, and its status on this dataset}",
        _comment_block(BULLETS["gates_51"]),
        _generated("table4_status", _shell(list(TABLE4_COLUMNS), "tab:table4_status", wide=True,
                                           note="Table 4, the gate chain (P2 Sec. 4); the toolkit's status column per rung",
                                           rows=[list(r) + [""] for r in TABLE4_ROWS], size="scriptsize")),
        _generated("table5", _shell(list(TABLE5_COLUMNS), "tab:table5", wide=True,
                                    note="Table 5, gate verdicts per rung at each grid point, 65 rows, refusals written not filled",
                                    size="scriptsize")),
        _generated("table5_g3", _shell(list(TABLE5_G3_COLUMNS), "tab:table5_g3", wide=True,
                                       note="Table 5 companion: G3 as a per-kernel signal flag, off the (W, H) decision")),
        "\\subsection{The per-rung gates the council added}",
        _comment_block(BULLETS["gates_52"]),
        "\\subsection{The same chain for each rung}",
        _comment_block(BULLETS["gates_53"]),
        "\\subsection{Pruning}",
        _comment_block(BULLETS["gates_54"]),
        "\\section{Experimental design}",
        _comment_block(BULLETS["design"]),
        _shell(["Kernel", "Predicted archetype", "Base parameters (scale 1.0)", "Footprint (pages, INFERRED)",
                "Steady-state stores change content?", "Pass period"], "tab:table3", wide=True,
               note="Table 3, cells (P2 Sec. 4; the pass period column merged from inputs/pass_table.csv when present)",
               rows=_table3_rows(pass_table_csv), size="scriptsize"),
        "\\section{Results}",
        _comment_block(BULLETS["results"]),
        _generated("table6", _shell(list(TABLE6_COLUMNS), "tab:table6", wide=True,
                                    note="Table 6, APF alone: rows per kernel, per archetype, all; the measured blind spot",
                                    size="scriptsize")),
        _generated("table7", _shell(list(TABLE7_COLUMNS), "tab:table7", wide=True,
                                    note="Table 7, rungs compared at gated resolution; null and majority on every row",
                                    size="scriptsize")),
        # build epoch 2 (builder A): the comparator rows of Table 7 (C14 candidates 1, 3, 2; P2 Sec. 0 Baseline)
        # and the comparators' own per-kernel table; the same generated-table form as table7
        _generated("table7_comparators", _shell(list(TABLE7_COLUMNS), "tab:table7_comparators", wide=True,
                                                note="Table 7, comparator rows: Savoldi 2010, Dhodapkar-Smith 2003, Law 2010 (raw and level-normalized), appended below Table 7",
                                                size="scriptsize")),
        _generated("table_comparators", _shell(list(TABLE_COMPARATORS_COLUMNS), "tab:table_comparators", wide=True,
                                               note="the comparators' own per-kernel numbers (the per-run form of C14); medians over admissible cells",
                                               size="scriptsize")),
        _generated("table8", _shell(["predicted \\ measured", "IDLE (measured)"] + [a for a in ARCHETYPES if a != "IDLE"]
                                    + ["clusters (k = ...)", "physical reason"], "tab:table8", wide=True,
                                    note="Table 8, store-predicted vs state-measured: assignments with counts, never a confusion matrix",
                                    rows=[[f"{a} ({'0 predicted' if a == 'IDLE' else n})", "", "", "", "", "", "", ""]
                                          for a, n in zip(ARCHETYPES, ("0", "6", "3", "2", "1"))])),
        _generated("tablegv", _shell(list(TABLEGV_COLUMNS), "tab:tablegv", wide=False,
                                     note="The G-V variance table: L0 within-kernel, L2 within-archetype, L3 between-archetype")),
        _generated("table_wapf_over_apf", _shell(list(WAPF_COLUMNS), "tab:wapf_over_apf", wide=False,
                                                 note="wAPF over APF per kernel (K2 move 11)")),
        "".join(_figure(stem, label, comment, wide=(stem in ("apf_per_kernel", "fused_plane", "j_hist", "table5_grid", "ratio_hist", "dhodapkar_sweep")))
                for stem, label, comment in FIGURES),
        "\\section{Discussion}",
        _comment_block(BULLETS["discussion"]),
        "\\section{Limitations}",
        _comment_block(BULLETS["limitations"]),
        "\\section{Conclusions}",
        _comment_block(BULLETS["conclusions"]),
        "\\section*{Artifact and reproducibility}",
        _comment_block(BULLETS["artifact"]),
        "% bibliography: council/11_al_nadim_final.md section 4 lists every citation and the sentence it supports",
        "% \\bibliographystyle{IEEEtran}",
        "% \\bibliography{p2}",
        "\\end{document}",
        "",
    ]
    return "\n".join(L)


def targets(tex: str) -> list[str]:
    """Every `\\input{...}` and `\\includegraphics[...]{...}` target in the skeleton (relative
    paths; the test checks they exist beside the report copy after tables and figures ran)."""
    t = re.findall(r"\\input\{([^}]+)\}", tex)
    t += re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    return t


def prose_lines(tex: str) -> list[str]:
    """Lines outside comments that are not LaTeX commands, tabular rows or environment
    syntax; the test asserts this list is empty (no body prose)."""
    bad = []
    in_tab = False
    for raw in tex.splitlines():
        line = raw.split("%", 1)[0].strip() if not raw.lstrip().startswith("%") else ""
        if not line:
            continue
        if line.startswith("\\begin{tabular}"):
            in_tab = True
        if in_tab:
            if line.startswith("\\end{tabular}"):
                in_tab = False
            continue
        if line.startswith("\\") or line.startswith("}") or line.startswith("{") or line.endswith("\\\\"):
            continue
        bad.append(raw)
    return bad


def write_skeleton(out: Path, *, documentclass: str = "IEEEtran", standalone: Path | None = None,
                   force_standalone: bool = False) -> dict:
    """Write the encoding paper's scaffold, and a standalone copy when `standalone` is given.

    The standalone write REFUSES a target that already holds prose; this builder emits a
    scaffold with none. Pass `force_standalone=True` to discard that prose deliberately.
    """
    out = Path(out)
    rep = out / "report"
    rep.mkdir(parents=True, exist_ok=True)
    tex = build_skeleton(documentclass=documentclass, pass_table_csv=out / "inputs" / "pass_table.csv")
    p = rep / "paper2_skeleton.tex"
    p.write_text(tex, encoding="utf-8")
    written = {"report": p}
    if standalone:
        sp = Path(standalone)
        # GUARD (2026-09-27), the same one latex_skeleton_eusipco.py carries. This builder emits a
        # SCAFFOLD: build_skeleton() returns zero prose_lines() by construction. The live paper at
        # apf_paper/p2_skeleton.tex is hand-maintained and holds author-approved content that this
        # scaffold does not: the separation paragraph (A-B1), Table 2's "Claim expected" column and
        # caption (A-I12, A-I13), the section renamed to "the readings" (A-I11), the EUSIPCO
        # citation and \label{sec:readings}. Overwriting it destroys all of that silently.
        if sp.exists() and not force_standalone:
            existing = prose_lines(sp.read_text(encoding="utf-8"))
            if existing:
                raise RuntimeError(
                    f"refusing to overwrite {sp}: it holds {len(existing)} prose line(s) and this "
                    f"builder emits a scaffold with none. That file is hand-maintained. Pass "
                    f"force_standalone=True only if you mean to discard its prose."
                )
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(tex, encoding="utf-8")
        written["standalone"] = sp
    write_json(rep / "paper2_skeleton.json", result_json(
        "latex_skeleton", {"out": str(out), "documentclass": documentclass, "standalone": str(standalone) if standalone else None,
                           "package_version": PACKAGE_VERSION, "written_at": now_iso()},
        CITATION, {"targets": targets(tex), "n_prose_lines": len(prose_lines(tex)), "written": {k: str(v) for k, v in written.items()}}))
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="plan11 LaTeX skeleton (builder 3)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--documentclass", default="IEEEtran", choices=["IEEEtran", "article"])
    ap.add_argument("--standalone", default=None, help="also write the skeleton to this path")
    ap.add_argument("--force-standalone", action="store_true", dest="force_standalone",
                    help="discard the standalone target's prose and overwrite it with the scaffold")
    a = ap.parse_args(argv)
    try:
        w = write_skeleton(Path(a.out), documentclass=a.documentclass, standalone=Path(a.standalone,
                       force_standalone=getattr(a, "force_standalone", False)) if a.standalone else None)
    except Exception:
        traceback.print_exc()
        return 1
    for k, v in w.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
