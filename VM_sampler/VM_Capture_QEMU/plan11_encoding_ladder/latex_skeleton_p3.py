#!/usr/bin/env python3
"""latex_skeleton_p3.py -- writes the detection paper's LNCS skeleton: headings, table shells,
figure placeholders and comment blocks carrying the substance bullets of P3_RAID_STRUCTURE.md
section 2 and the box contents of al-Nadim's final report section 1. Not one sentence of body
prose (SPEC_DETECTION 5.4; the binding rule NO PAPER PROSE).

Builder B (report), 2026-09-17. The document class is `llncs` with `[runningheads]` (the RAID
2026 call names the LNCS template, N3 preamble); `--documentclass article` is the fallback when
`llncs.cls` is not installed, and a comment line at the top says which was chosen. Every
generated table is brought in as `\\IfFileExists{tables/<name>.tex}{\\input{...}}{<shell>}`
where `<shell>` is the table environment with the column headers and one empty row, so the same
skeleton compiles beside the generated tables (`<out>/report/detection/`) and standing alone
(`apf_paper/p3_skeleton.tex`); every figure is `\\IfFileExists{figures/<name>.pdf}
{\\includegraphics}{<framed placeholder>}`. Static shells: Table 1 (campaign identity, the
stage-1 facts of P3 0a as labels), Table 2 (prior work by observer position, the bib keys of the
six rows named in P3 Sec. 2 in the first column), Table 3 (the ladder, plan11's Table 2 rows).

CLI (SPEC 7.1): latex_skeleton_p3.py --out O [--documentclass llncs|article] [--standalone PATH]
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
    PACKAGE_VERSION, RUNGS, latex_escape, now_iso, result_json, write_json,
)
from plan11_encoding_ladder.latex_skeleton import TABLE2_ROWS as P2_LADDER_ROWS  # noqa: E402  (imported, never changed)
from plan11_encoding_ladder.tables_detection import (  # noqa: E402
    FP_COLUMNS, LADDER_COLUMNS, MISS_COLUMNS, REASON_COLUMN, TABLE4_COLUMNS, TABLE5_COLUMNS, TABLE6_COLUMNS,
    TABLE7_COLUMNS, TABLE9_COLUMNS, TABLE11_COLUMNS,
)

CITATION = "P3 Sec. 2 (the paper's shape); N3 Sec. 1 (the boxes), Sec. 2 (the outline); P3 0a (Table 1 facts); SPEC_DETECTION 5.4"

# P3 0a stage-1 facts as labels (Table 1, campaign identity); `--` where the design leaves a cell open
TABLE1_ROWS = [
    ("commit", "fcc184e"), ("guest RAM", "1024 MiB"), ("capture interval", "500 ms"), ("speed", "speed 2"),
    ("retention", "retention combined"), ("duration per cell", "--duration 600"), ("seed", "seed on every line"),
    ("guest image", "unchanged since the kernels (P3 0a)"), ("stage 1 cells", "96 kernel + 8 idle + 64 sandbox"),
    ("order", "sandbox cells in order by member; idle after; kernels before (P3 0a)"),
    ("realized spacing", "--"), ("wall time per cell", "--"), ("logging", "--"),
]
TABLE2_COLUMNS = ["work", "observer position", "feature", "benign set", "window rule", "idle treatment", "data availability"]
# the six prior-work rows named in P3 Sec. 2 / N3 Sec. 2 (bib keys of p3.bib; the lineage rows cite the keys p2.bib carries)
TABLE2_ROWS = [
    ("\\cite{hirano2022ransomware,hirano2022ransap,hirano2025ransmap}", "hypervisor telemetry", "--", "--", "--", "--", "--"),
    ("\\cite{purnaye2022bishm,purnaye2025dataverse,purnaye2026agent}", "hypervisor telemetry", "--", "--", "--", "--", "--"),
    ("\\cite{law2010volatile,savoldi2010uncertainty}", "dump differencing", "--", "--", "--", "--", "--"),
    ("\\cite{clark2005livemigration}", "hypervisor page-fault tracking", "--", "--", "--", "--", "--"),
    ("\\cite{lindemann2018identification}", "hypervisor page-fault tracking", "--", "--", "--", "--", "--"),
    ("\\cite{oliveri2025inconsistencies}", "dump differencing (atomicity)", "--", "--", "--", "--", "--"),
    ("this paper", "hypervisor dump differencing, process-blind", "--", "--", "--", "--", "--"),
]
TABLE3_COLUMNS = ["rung", "encoding", "axis kept"]
TABLE3_ROWS = [(r[0], r[1], r[2]) for r in P2_LADDER_ROWS] + [("2'", "content channel", "content of what changed (labelled row, not built in stage 1)")]

# ----------------------------------------------------------------------------------------------
# the substance bullets (P3 Sec. 2 "Carries"; N3 Sec. 1 boxes); comments only
# ----------------------------------------------------------------------------------------------
BULLETS = {
    "intro": [
        "the gap: the nearest published hypervisor-side detector was evaluated with an office-class benign world; this paper builds the benign world to contain every mechanism the class uses",
        "whose channel this is: the observer position in one paragraph",
        "the object paragraph: the author's alone",
        "RQ1 to RQ5 listed (RQ6 conditional)",
        "contributions, bulleted: the dataset (whole-memory state sequences at sub-second cadence, cell count, DOI); the measurement at a declared operating point with every workload unseen; the three pitfalls with sizes; the unseen-case readings; the comparator on the same rows",
        "papers 1 and 2 in the third person (double blind)",
    ],
    "background": [
        "a prior-work table by observer position (in-guest agent, hypervisor page-fault tracking, hypervisor telemetry, dump differencing), feature, benign set, window rule, idle treatment, data availability",
        "rows from Hunayn's verified reads: Hirano and Kobayashi 2022; the Purnaye lineage; Law 2010 and Savoldi 2010; Clark 2005; Lindemann and Fischer 2018; Oliveri and Balzarotti 2025 (atomicity)",
        "RAID precedents named in the venue dossier (Ninan 2024 negative finding; Qu 2025 dataset-shift protocol): entries NEEDED in p3.bib",
    ],
    "channel": [
        "what the differ leaves per pair (changed pages, 64 metrics, hamming != 0); the three axes (breadth, amount, identity)",
        "the ladder in one table (Table 3) and one paragraph per rung, self-contained for a reviewer who cannot read paper 2",
        "campaign identity (Table 1); the two time axes stated separately (guest-running 600 s; about 645 ms per pair; about an hour of wall time)",
        "what the channel cannot see: reads of cached data, rhythms above Nyquist, position, page reuse",
    ],
    "dataset": [
        "Table 4: the six tiers with cell counts (stage 1: kernels, idle, the sandbox family; stage 2 proposed: re-launched controls, breadth, harness-idle; stage 3: an external source, test only)",
        "the mechanism rationale for the benign world (M1 to M6), stated as the design rule",
        "the two controls (re-launched, harness-idle) and why they are not optional; absent in stage 1 and stated as a limitation",
        "interleaving and its record: stage 1 ran in order by member; the realized order is released as a class-only letter sequence",
        "the pilot and its exclusion; what was logged and why none of it is a feature; the frozen image and commit",
        "the released artifact and its anonymisation (P3 D8): trajectories with a DOI, the anonymous review copy, chains on request",
        "the sandbox family named by index and letter only; sub-families A (4), B (1), C (3)",
    ],
    "protocol": [
        "unit = cell; the four splits (LOWO headline, LOCO signature ceiling, LOFO, one-class) and their reading order",
        "the workload-level null with null_not_estimable below 20 assignments; 500 permutations or the exhaustive set",
        "the baselines: majority (always benign) once, the random scorer (AUC 0.5, TPR = FPR)",
        "the operating point: five percent per cell declared before the data, threshold in fold, realized FPR beside it; one percent at the resolution limit; G-OP",
        "G-K0's three verdicts (at floor, at harness floor, above floor); at floor leaves the denominator",
        "the pre-registration hash and date",
        "Table 5: the gates in one row each with the appendix pointer",
        "hyperparameters declared before the data: forest 300 trees, sqrt features, min leaf 1, no class weight; row unit cell; threshold source inner_lowo",
    ],
    "validity": [
        "G-C on the calibrated core (gemm's re-seed pulse; the content orderings); the three floors (idle, harness-idle, idle set against idle set); early-against-late idle; the state-change yield per tier; the G-K0 verdict counts per tier; Figure 2",
        "box if it holds: the instrument is connected on the benign side; the floor is stable or drifted by a stated amount; N cells at floor and M at harness floor leave the denominator",
        "box if it fails: G-C fails, every negative that follows is void (F6) and the paper cannot be written on this campaign",
    ],
    "rq1": [
        "Table 7 (rows: apf raw, apf, wapf, persist, content, combined, combined (matched), content channel 2' labelled, comparator; columns: feature count, ROC area, TPR at 5% with the realized FPR, TPR at 1% at the resolution limit, the null's p95 and rank, the random scorer; the majority baseline in a note)",
        "Table 8: per-member recall in eighths, never a mean alone; Figure 5: ROC per rung under LOWO; Figure 6 and the ladder table: TPR at 5% at 30, 60, 120, 300, 600 s, two readings (from pair 1; from the first boundary)",
        "box if it holds: the rung, the S recalls, the realized FPR, the null's rank, the shortest prefix that clears the null; the number stands after level, campaign and harness are removed (RQ3)",
        "box if it fails: no rung clears the workload-level null at the operating point under LOWO once RQ3's controls are applied; the class has no form on this channel at this interval distinct from its mechanism; RQ2's miss table says what each member resembles",
        "a negative finding heads an accepted RAID paper (Ninan): the failing box is a result, not a retraction; voided, not failed, by G-C, G-F (idle reps separable) or the harness clause",
    ],
    "rq2": [
        "the ladder read as an ablation: breadth against amount against identity against their combination, the content channel as a labelled row; G-M's paired margin and G-DIM parity on every comparison",
        "Table 10: the miss table (per missed sandbox cell: nearest benign tier and workload, the resemblance axis and the axis of largest residual, the physical reason M1 to M6 from the author, never from the source; per benign false positive: the nearest member index)",
        "Figure 4: the fused plane (median l0 over persistent pages against J - J_null), one panel per benign tier and one for the class with members as marker shapes; idle and harness-idle clouds in every panel; boundary pairs hollow",
        "levels 2 and 3: the sub-family confusion with one member held out (A and C; B reads one member, no held-out test), the exact member under leave-one-rep-out as the signature ceiling",
        "box if it holds: the axis (amount or identity, not breadth), the rung, where the misses fall; rung A beats rung B only at S of six or more (the sign test)",
        "box if it fails: only breadth separates (F1, level only); only the content channel separates and vanishes under axis matching (F10); the members scatter across the benign clouds with no shared region (a class without a form of its own)",
    ],
    "rq3": [
        "Table 9 with three pitfalls as sizes, never pass marks: level (raw APF against level-normalized APF; G-LM per member), campaign (G-ANCHOR on the kernels and the idle sets; the order test; the cross-campaign row), harness (the margins per feature; the re-launched control's flagged rate; the cepstral line, not built in stage 1)",
        "stage 1: the sandbox cells ran in order by member, so position and identity coincide by construction (P3 0a); the order test is a required row read as a size; the within-workload half rule strips identity",
        "the cadence and active-fraction probes (ML 3.4, 3.5) as rows of the same table",
        "box if it holds: each pitfall's size: what level alone would have detected, whether campaign is audible and by how much, what the harness contributes",
        "box if it fails: which of the three is the whole separation, in its falsifier's words: level only (F1), the detection table void inside the interleaved campaign (F5), re-launch not class (F4); the paper becomes the pitfall paper",
    ],
    "rq4": [
        "Table 11: four splits per rung (LOWO, LOCO, LOFO, one-class) with the LOCO minus LOWO gap (G-SIG), the false-positive attribution per benign family (G-FP), benign recall per family",
        "box if it holds: the one-class model trained on benign only flags the class at rate r at the declared FPR; no held-out benign family fires above the declared rate; the signature gap is small",
        "box if it fails: separable when heard, not flagged when not (F9); a novelty detector of its training set with the firing family and axis named (F8); recognises workload identity, not family behaviour (G-SIG), citable for signature matching only",
    ],
    "rq5": [
        "one row of Table 7, one column of Table 10, the mapping table and placement label in the appendix; decided in another session; placed here only",
        "box if it holds: the paired margin under G-M with the feature count stated; if it fails, the same with the sign reversed, or the label that the comparator's number is a placement reading; one row, never the paper's centre",
    ],
    "rq6": [
        "RQ6 only if a lever or a launch mode exists: evasion by a shaping lever, or the mixture arm (one member and one benign workload in one step); otherwise one Discussion paragraph and the programme sentence for paper 4; the word masquerade withdrawn pending Hunayn's read",
    ],
    "discussion": [
        "what the channel reads and for whom; the evasion paragraph (RQ6 absent); the limitations: one guest, one hypervisor, one RAM size, one interval, S members, the pass period declared only where the marker exists, the two controls absent in stage 1, no extra logging in stage 1",
        "the ethics paragraph: no human data; a test image; what the released trajectories contain and what the chains would",
    ],
    "conclusion": [
        "one paragraph; the programme sentence names paper 4",
    ],
    "appendix": [
        "the gate chain in full with every threshold and every grid point (Table 5's appendix pointer); the per-cell tables; the pilot's numbers by index; the comparator's mapping; hyperparameters; the anonymisation procedure; the harness-rhythm figure if cut from the body",
    ],
}
LEVEL2_COLUMNS = ["true sub-family", "n members", "n cells", "n at floor", "predicted A", "predicted B", "predicted C", "recall", "status", "null p95", "rank", "verdict"]
LEVEL3_COLUMNS = ["true member", "n at floor", "predicted 1", "predicted ...", "predicted S", "recall (eighths)", "label", "null p95", "rank", "verdict"]
TABLE8_COLUMNS = ["rung", "split", "member 1", "...", "member S", "min", "median", "max", "n at floor", "n without score", "note"]
CELLS_COLUMNS = ["cell_id", "class", "member", "sub-family", "rep", "order token", "campaign", "n pairs", "dt (s)", "K median", "G-K0 verdict", "admissible",
                 "LOWO score per rung", "flagged at 5% per rung"]
GV_COLUMNS = ["rung", "feature", "L0", "L2", "L3", "L0_b", "L2_b", "L3_families", "L3_over_L3_families", "note"]


# ----------------------------------------------------------------------------------------------
# builders
# ----------------------------------------------------------------------------------------------
def _comment_block(lines) -> str:
    """`% - <bullet>` lines; the box contents of N3 Sec. 1 as `% box if it holds:` / `% box if it fails:` (SPEC_DETECTION 5.4)."""
    return "\n".join((f"% {ln}" if ln.startswith("box if it ") else f"% - {ln}") for ln in lines) + "\n"


def _cell(x: str) -> str:
    """Escape a fixed cell; a `\\cite{...}` span or a math span is left as it is."""
    x = "" if x is None else str(x)
    if x.startswith("\\cite{") and x.endswith("}"):
        return x
    return "$".join(seg if i % 2 else latex_escape(seg) for i, seg in enumerate(x.split("$")))


def _shell(columns, label, *, wide=None, note="", rows=None, size="footnotesize") -> str:
    """A table environment with the column headers and one empty row (or the fixed rows)."""
    env = "table*" if wide else "table"      # llncs is single column: the shells stay `table` (wide is never set by this module)
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


def _generated(name: str, columns, note: str = "") -> str:
    """`\\IfFileExists{tables/<name>.tex}{\\input{tables/<name>.tex}}{<shell>}`."""
    shell = _shell(columns, f"tab:{name}", note=note)
    return f"\\IfFileExists{{tables/{name}.tex}}{{\\input{{tables/{name}.tex}}}}{{%\n{shell}}}\n"


def _figure(stem: str, label: str, comment: str, wide: bool = False) -> str:
    env = "figure"                            # llncs is single column: never figure*
    return (f"% figure: {comment}\n"
            f"\\begin{{{env}}}[t]\n\\centering\n"
            f"\\IfFileExists{{figures/{stem}.pdf}}{{\\includegraphics[width=\\linewidth]{{figures/{stem}.pdf}}}}"
            f"{{\\framebox[\\linewidth]{{\\rule{{0pt}}{{0.3\\linewidth}}}}}}\n"
            f"\\caption{{}}\n\\label{{{label}}}\n\\end{{{env}}}\n")


def build_skeleton(*, documentclass: str = "llncs") -> str:
    """The skeleton text. `documentclass` is `llncs` (`[runningheads]`, the LNCS template the
    RAID 2026 call names) or `article` (the fallback when llncs.cls is absent)."""
    L = []
    if documentclass == "llncs":
        L += ["\\documentclass[runningheads]{llncs}",
              "% document class: llncs (the LNCS template named by the RAID 2026 call, N3 preamble); pass --documentclass article when llncs.cls is absent"]
    else:
        L += ["\\documentclass[10pt]{article}",
              "% document class: article (fallback; llncs.cls absent); the submission uses llncs [runningheads]"]
    L += ["\\usepackage{booktabs}", "\\usepackage{graphicx}", "\\usepackage{amsmath}", ""]
    L += ["% title candidate 1: (open, P3 Sec. 9; the author's)", "% title candidate 2: (open, P3 Sec. 9; the author's)",
          "% title candidate 3: (open, P3 Sec. 9; the author's)", "\\title{}", "\\author{}"]
    if documentclass == "llncs":
        L += ["\\institute{}"]
    L += ["\\begin{document}", "\\maketitle", ""]
    L += ["\\begin{abstract}", "% - (no prose; the abstract is written in the writing epoch)", "\\end{abstract}", ""]
    # 1 Introduction
    L += ["\\section{Introduction}", _comment_block(BULLETS["intro"])]
    # 2 Background and prior work
    L += ["\\section{Background and prior work}", _comment_block(BULLETS["background"]),
          _shell(TABLE2_COLUMNS, "tab:table2_prior_work", rows=TABLE2_ROWS, note="Table 2: prior work by observer position (P3 Sec. 2); the first column holds the bib keys of p3.bib; -- where the row is the author's to fill", wide=False)]
    # 3 The channel and the instrument
    L += ["\\section{The channel and the instrument}", _comment_block(BULLETS["channel"]),
          _shell(["item", "value"], "tab:table1_campaign_identity", rows=TABLE1_ROWS, note="Table 1: campaign identity, the stage-1 facts of P3 0a as labels", wide=False),
          _shell(TABLE3_COLUMNS, "tab:table3_ladder", rows=TABLE3_ROWS, note="Table 3: the ladder (plan11's Table 2 rows: rung, encoding, axis kept)", wide=False)]
    # 4 Dataset and capture design
    L += ["\\section{Dataset and capture design}", _comment_block(BULLETS["dataset"]),
          _generated("table4_tiers", TABLE4_COLUMNS, note="Table 4: the tiers with counts")]
    # 5 Evaluation protocol
    L += ["\\section{Evaluation protocol}", _comment_block(BULLETS["protocol"]),
          _generated("table5_gates", TABLE5_COLUMNS, note="Table 5: the gates in one row each; the full chain in the appendix")]
    # 6 Validity of the campaign
    L += ["\\section{Validity of the campaign}", _comment_block(BULLETS["validity"]),
          _generated("table6_validity", TABLE6_COLUMNS, note="Table 6: the validity block"),
          _figure("fig2_three_floors", "fig:three_floors", "Figure 2: the three floors (K, l0, J) of the idle cells; harness-idle overlaid when present; the idle sets by campaign", wide=True)]
    # 7 Results
    L += ["\\section{Results}", "% - each block closed by a boxed takeaway written for both outcomes (N3 Sec. 1)", ""]
    L += ["\\subsection{RQ1: detection at a declared operating point with every workload unseen}", _comment_block(BULLETS["rq1"]),
          _generated("table7_detection", TABLE7_COLUMNS, note="Table 7: detection under LOWO, one row per rung display name; the l1 rows after the rungs"),
          _generated("table8_member_recall", TABLE8_COLUMNS, note="Table 8: per-member recall in eighths; the sub-family letters as the first row; no mean"),
          _figure("fig5_roc_lowo", "fig:roc_lowo", "Figure 5: ROC per rung under LOWO with the in-fold operating point"),
          _generated("table_ladder", LADDER_COLUMNS, note="the time-to-detect ladder (K3 move 17)"),
          _figure("fig6_ladder", "fig:ladder", "Figure 6: TPR at 5% against prefix length, two readings")]
    L += ["\\subsection{RQ2: which axis carries the separation, and what the misses resemble}", _comment_block(BULLETS["rq2"]),
          _generated("table10_misses", MISS_COLUMNS + [REASON_COLUMN], note="Table 10: the miss table; assignments with counts, never a confusion matrix"),
          _generated("table10_false_positives", FP_COLUMNS + [REASON_COLUMN], note="Table 10 (continued): the false positives"),
          _figure("fig4_fused_plane_tiers", "fig:fused_plane_tiers", "Figure 4: the fused plane per tier and for the class, members as marker shapes", wide=True),
          _generated("table_level2", LEVEL2_COLUMNS, note="level 2: the sub-family confusion with one member held out, per sub-family, never averaged"),
          _generated("table_level3", LEVEL3_COLUMNS, note="level 3: the exact member under leave-one-rep-out, the signature ceiling")]
    L += ["\\subsection{RQ3: the three pitfalls, measured as sizes}", _comment_block(BULLETS["rq3"]),
          _generated("table9_pitfalls", TABLE9_COLUMNS, note="Table 9: level, campaign or order, harness; sizes, never pass marks")]
    L += ["\\subsection{RQ4: the unseen case}", _comment_block(BULLETS["rq4"]),
          _generated("table11_splits", TABLE11_COLUMNS, note="Table 11: the four splits side by side; LOCO carries the signature ceiling label")]
    L += ["\\subsection{RQ5: the comparator row}", _comment_block(BULLETS["rq5"]),
          "% table shell (comparator row): not run: comparator row from another session (RQ5); its numbers enter Table 7's comparator row and Table 10's comparator column when that session writes them in this layer's schema (SPEC_DETECTION 3.5.13)", ""]
    L += ["% \\subsection{RQ6: evasion or the mixture arm} (only if a lever or a launch mode exists)", _comment_block(BULLETS["rq6"])]
    # 8, 9
    L += ["\\section{Discussion and limitations}", _comment_block(BULLETS["discussion"])]
    L += ["\\section{Conclusion}", _comment_block(BULLETS["conclusion"]), ""]
    L += ["\\bibliographystyle{splncs04}", "\\bibliography{p3}", ""]
    # appendices
    L += ["\\appendix", "\\section{The gate chain}", _comment_block(BULLETS["appendix"]),
          "% - every threshold and every grid point: the params blocks of gates/detection/*.params.json and report/detection/manifest.json", ""]
    L += ["\\section{Per-cell tables}", _generated("table_cells_detection", CELLS_COLUMNS, note="the per-cell appendix; never path, label, test_label, traj_file or order_index"),
          _generated("table_gv_two_class", GV_COLUMNS, note="G-V two-class")]
    for rung in RUNGS:
        L += [_generated(f"table_level2_{rung}", LEVEL2_COLUMNS, note=f"level 2, {rung}"),
              _generated(f"table_level3_{rung}", LEVEL3_COLUMNS, note=f"level 3, {rung}")]
    L += ["\\section{The pilot's numbers}", "% - by index; stage 1 has no pilot (P3 0a); -- until stage 2", ""]
    L += ["\\section{The comparator mapping}", "% - the mapping table and the placement label (RQ5; another session)", ""]
    L += ["\\section{Hyperparameters}", "% - the forest (300 trees, sqrt features, min leaf 1, no class weight, median imputation, per-fold standardization); row unit cell; threshold source inner_lowo; the one-class model (isolation forest, 300 trees, contamination auto); (W, H) per rung inherited from the encoding paper's selection; every constant of SPEC_DETECTION section 7", ""]
    L += ["\\section{The anonymisation procedure}", "% - the class-only letter sequence; public cell ids; the anonymisation map kept outside the deposit (CR3 1.8); the sandbox family by index and letter only", ""]
    L += ["% figure 7 (the harness rhythm; cepstrum against the logged iteration period): not built in stage 1 (K3 move 7; SPEC_DETECTION preamble); appendix if space is short", ""]
    L += ["\\end{document}", ""]
    return "\n".join(L)


def targets(tex: str) -> list[str]:
    """Every `\\input{...}` and `\\includegraphics[...]{...}` target in the skeleton (relative paths)."""
    t = re.findall(r"\\input\{([^}]+)\}", tex)
    t += re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    return t


def prose_lines(tex: str) -> list[str]:
    """SPEC_DETECTION 4.5's prose test: lines outside comments that end in a period and contain a
    space-separated word of more than three letters other than a LaTeX command, tabular rows and
    environment syntax excluded; the test asserts this list is empty."""
    bad = []
    in_tab = False
    for raw in tex.splitlines():
        stripped = raw.strip()
        if stripped.startswith("%"):
            continue
        line = stripped.split("%", 1)[0].strip()
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
        words = [w for w in re.split(r"\s+", line) if len(re.sub(r"[^A-Za-z]", "", w)) > 3 and not w.startswith("\\")]
        if line.endswith(".") and words:
            bad.append(raw)
        elif words:
            bad.append(raw)
    return bad


def brace_balance(tex: str) -> int:
    """Braces outside comments; 0 when balanced."""
    depth = 0
    for raw in tex.splitlines():
        line = raw.split("%", 1)[0] if not raw.lstrip().startswith("%") else ""
        line = line.replace("\\{", "").replace("\\}", "")
        depth += line.count("{") - line.count("}")
    return depth


def write_skeleton(out: Path, *, documentclass: str = "llncs", standalone: Path | None = None) -> dict:
    out = Path(out)
    rep = out / "report" / "detection"
    rep.mkdir(parents=True, exist_ok=True)
    tex = build_skeleton(documentclass=documentclass)
    p = rep / "paper3_skeleton.tex"
    p.write_text(tex, encoding="utf-8")
    written = {"report": p}
    if standalone:
        sp = Path(standalone)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(tex, encoding="utf-8")
        written["standalone"] = sp
    write_json(rep / "paper3_skeleton.json", result_json(
        "detection.latex_skeleton", {"out": str(out), "documentclass": documentclass, "standalone": str(standalone) if standalone else None,
                                     "package_version": PACKAGE_VERSION, "written_at": now_iso()},
        CITATION, {"targets": targets(tex), "n_prose_lines": len(prose_lines(tex)), "brace_balance": brace_balance(tex),
                   "written": {k: str(v) for k, v in written.items()}}))
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="plan11 detection LaTeX skeleton (builder B)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--documentclass", default="llncs", choices=["llncs", "article"])
    ap.add_argument("--standalone", default=None, help="also write the skeleton to this path (apf_paper/p3_skeleton.tex)")
    a = ap.parse_args(argv)
    try:
        w = write_skeleton(Path(a.out), documentclass=a.documentclass, standalone=Path(a.standalone) if a.standalone else None)
    except Exception:
        traceback.print_exc()
        return 1
    for k, v in w.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
