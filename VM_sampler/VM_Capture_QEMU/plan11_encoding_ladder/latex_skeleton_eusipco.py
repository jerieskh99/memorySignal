#!/usr/bin/env python3
"""latex_skeleton_eusipco.py -- the five-page IEEE conference skeleton of the EUSIPCO paper:
headings, four numbered-equation placeholders, table and figure shells, `\\cite` placeholders
keyed to `apf_paper/p2.bib`, and comment blocks carrying `P2E_STRUCTURE.md`'s substance bullets.
Not one sentence of body prose: every text line outside a comment is a heading, a LaTeX command,
a tabular row or environment syntax, and `latex_skeleton.prose_lines(tex) == []` is asserted.

Builder 2 (eusipco) of build epoch 2, 2026-09-17. Source of truth: `apf_paper/P2E_STRUCTURE.md`
sections 0 (the four gates), 1 (what the venue accepts), 2 (the one claim), 3 (the section plan:
I Introduction, II The representation, III Data and protocol, IV Results, V Conclusion, the
reference plan), 5 (the compression map) and 7 (open, for the author); the comparator rows per
`P2_AUTHOR_ANSWERS.md` "Decisions of 2026-09-17" and `council/14_hunayn_exact_input_comparators.md`.
The venue facts that shape the page (P2E sec. 1; 2027 rules UNVERIFIED): five pages, four to six
sections, one to four numbered equations, the representation shown as a figure, two to four result
tables, 19 to 23 references.

Built on `latex_skeleton.py` (builder 3, epoch 1), whose helpers `_comment_block`, `_shell`,
`_figure`, `_generated`, `targets`, `prose_lines` are imported and not edited: every generated
table is `\\IfFileExists{tables/<name>.tex}{\\input{...}}{<shell>}` so the same document compiles
beside `<out>/report/tables/` and standing alone (`apf_paper/p2e_skeleton.tex`); every figure is
`\\IfFileExists{figures/fig_<stem>.pdf}{\\includegraphics}{<framed placeholder>}`. The table shells'
column headers are `tables_eusipco.EUSIPCO_TABLE2_COLUMNS` and `EUSIPCO_TABLE3_COLUMNS`.

The `\\cite` placeholders: `CITE_PLAN` lists each key with the P2E sentence it supports; each is
emitted as its own `\\cite{<key>}` line under a comment naming that sentence, so the count (19 to
23, P2E sec. 1 and sec. 3 "References") and the keys are checkable against `p2.bib`; the
bibliography lines stay commented as in `latex_skeleton.py` (the report copy has no `p2.bib`
beside it). The reference plan's twelve keys of P2E sec. 3 are all in the plan.

CLI (SPEC 7.1 form): latex_skeleton_eusipco.py --out O [--documentclass IEEEtran|article]
                     [--standalone PATH]   (also write a copy to PATH; NOT a live paper -- refused if it holds prose
Exit 0 on success, 1 on an internal error.
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
    PACKAGE_VERSION, now_iso, result_json, write_json,
)
from plan11_encoding_ladder.latex_skeleton import (  # noqa: E402
    _comment_block, _figure, _generated, _shell, prose_lines, targets,
)
from plan11_encoding_ladder.tables_eusipco import (  # noqa: E402
    COMPARATORS_DEFAULT, COMPARATOR_DISPLAY, EUSIPCO_TABLE2_COLUMNS, EUSIPCO_TABLE3_COLUMNS,
)

CITATION = ("P2E sec. 3 (the five sections and the reference plan), sec. 0 (the four gates), sec. 1 (venue "
            "facts), sec. 2 (the one claim), sec. 5 (the compression map), sec. 7 (open items); AA 2026-09-17 "
            "(the comparator rows); C14 (the comparators); latex_skeleton.py (the helpers; SPEC 6.8)")

SECTION_TITLES = ("Introduction", "The representation", "Data and protocol", "Results", "Conclusion")
EQUATIONS = (  # (label, P2E sec. 3.II one-line substance)
    ("set", "Eq. 1: the changed-page set S_t and the per-page byte distance d_t(p) for pair t (the differ's sparse output; 'changed', not 'written')"),
    ("breadth", "Eq. 2: breadth, K_t / N (APF); wAPF as the count weighted by bits flipped, the negative control"),
    ("content", "Eq. 3: the content-change summary: the three ratios l0/4096, l1/l0, hamming/l0 over pages present in both snapshots, with fixed quantiles"),
    ("persistence", "Eq. 4: persistence, J(t) = |S_t and S_{t+1}| / |S_t or S_{t+1}|, its independence null, and the floor null from idle cells"),
)
TABLE1_ROWS = [  # P2E sec. 3.III, the dataset in one compact table; the blanks are the author's
    ("kernels", "12"), ("reps per kernel", "8"), ("guest time per cell", "600 s"),
    ("configured interval", "500 ms"), ("pairs per cell", "about 930"), ("guest memory", "1 GiB"),
    ("N (pages)", "262,144"), ("launch labels", "three, one commit"), ("dataset DOI", ""), ("idle cells", ""),
]
GATES_P2E_SEC0 = (
    "gate 1, the co-chair question: a five-page EUSIPCO paper followed by the encoding paper at IFIP must be a declared extension; one email from the author before anything is written",
    "gate 2, the dataset released: the 01c/01c1 substrate trajectories with a DOI, so the evaluation is on a public benchmark",
    "gate 3, the external comparator in the table: resolved in principle 2026-09-17 (council/14; three exact-input methods; the author's picks in P2_AUTHOR_ANSWERS.md Decisions of 2026-09-17)",
    "gate 4, the track: chosen from the 2027 list when it appears (Hunayn's pending request 9a)",
)
COMPRESSION_MAP = (  # P2E sec. 5, one line per row
    "I Introduction -> I, cut to 0.7 page, related work folded in",
    "II Prior work -> folded into I; 19 to 23 references",
    "III Apparatus, time axes, retained chains -> III, one table and three sentences",
    "IV The ladder -> II, four equations and one figure",
    "V The gate chain -> one paragraph in II; cited",
    "VI Design -> III",
    "VII Results, Tables 5 to 8, G-V -> Table 2 (from Table 7 plus the comparator), Table 3 (the level-matched test); Table 8 optional as a figure; Table 5 and G-V absent",
    "VIII Discussion -> two sentences in IV; the blocking paper not named",
    "IX Limitations -> caveats in the text of IV",
    "X Conclusions -> V, one paragraph",
)
VENUE_FACTS = (  # P2E sec. 1 (2026 rules; 2027 UNVERIFIED)
    "five pages, two columns; four to six sections; method section the longest (1.5 to 2 pages), results about 1.5 pages, introduction about 0.7 page, conclusion one paragraph",
    "related work folded into the introduction or a half-page Preliminaries; 19 to 23 references",
    "formalism light and functional: one to four numbered equations defining the representation; the representation itself shown as a figure; two to four result tables",
    "every accepted paper evaluates on one public benchmark and puts named methods from the literature in the same table; two of three add a cost comparison; one uses statistical testing",
    "noise and small-sample caveats stated in the text, not in a section",
)
BULLETS = {  # abbreviated from P2E_STRUCTURE.md sections 2 and 3; comments only
    "abstract": [
        "The one claim, EUSIPCO reading (P2E sec. 2): the whole-memory delta between consecutive hypervisor snapshots is a sparse field of changed pages, each with a byte distance.",
        "Three reductions keep three axes: breadth (the page count, the field's prior quantity), amount (the distribution of byte distance over the changed pages), identity over time (the overlap of consecutive changed sets).",
        "Twelve numerical kernels at 500 ms: breadth reads only the footprint level, a run parameter.",
        "Kernels at the same level are separated by the amount and identity reductions computed from the same rows, under leave-one-kernel-out, against a null and against a named external feature set.",
        "Extraction cost: one streaming pass per cell.",
        "Falsifiers: P2_STRUCTURE.md section 2, items 1 to 6, one sentence each in the text.",
    ],
    "intro": [
        "The delta as a signal; the field's per-pair count (Law 2010, Savoldi 2010) as the prior representation; the hypervisor as the atomic acquisition point (Oliveri and Balzarotti 2025); three sentences with citations.",
        "Related work folded in: workload identification from memory behaviour and the nearest prior feature set (Hirano and Kobayashi 2022), two sentences; the Purnaye lineage, one sentence.",
        "Contributions, three items (Neri's pattern): (i) a three-axis reading of the memory delta and the reductions that keep each axis; (ii) the comparison of the reductions against each other and against a named external feature set on a released dataset; (iii) the level-matched test that shows what the count cannot separate and which axis does.",
        "One sentence naming the gate chain and pointing to the encoding paper (IFIP 2028) for it.",
    ],
    "representation": [
        "Four numbered equations below define the representation (P2E sec. 3.II; the only formalism).",
        "Figure 1, the representation itself: the fused plane, median l0 over persistent pages against J to the next snapshot, one cloud per kernel, idle cells as the floor.",
    ],
    "resolution": [
        "Resolution paragraph: the window and hop are fixed per reduction by a declared grid with every point kept and the selected point marked; the selected (W, H) per reduction and the count of refused grid points stated here; the chain itself cited from the encoding paper (IFIP 2028).",
        "Extraction paragraph: one streaming pass per cell, two page sets in memory at a time; cost per cell in seconds and feature count per reduction (the cost line).",
    ],
    "data": [
        "Table 1 (compact): 12 kernels, 8 reps, 600 s of guest time per cell, 500 ms configured interval, about 930 pairs per cell, 1 GiB guest, N = 262,144 pages, three launch labels at one commit, the DOI of the release, the idle cells.",
        "Splits: unit = cell; leave-one-kernel-out to archetype as the headline; leave-one-rep-out as the unseen-run test; the label-shuffle null at the unit; the majority baseline; class support stated per archetype (two archetypes with fewer than three kernels are reported, not averaged).",
        "The external comparator (AA 2026-09-17, replacing the Hirano and Kobayashi analogue of P2E sec. 3.III): Savoldi 2010, U = mean +/- SD of the per-pair changed-page count per run; Dhodapkar-Smith 2003, the relative working set distance (one minus the overlap) with a declared threshold sweep, phase boundaries, stability, mean phase length; both computed on the same rows under the same splits and the same unit-level null; one sentence on the mapping (C14: exact input).",
        "Clark 2005 / QEMU calc-dirty-rate cited in one sentence as the systems name for APF (the writable working set, the dirty rate); Law 2010 held for the IFIP version.",
        "The two clocks in one sentence: guest time 600 s, wall time about an hour; every temporal quantity in pair units.",
    ],
    "results": [
        "Table 2, reductions compared: rows APF, wAPF, content-change, persistence, combined, the comparators; columns feature count, LOKO accuracy and macro recall over qualifying archetypes, LORO accuracy, the null's 95th percentile, the majority baseline, the paired margin against APF; refusals printed as words (generated: tables/eusipco_table2.tex).",
        "Table 3, the level-matched test: rows the 2,048-page set (floyd, histogram, nbody) and the 4,096-page pair (fft, gemm); columns separable under APF (no), under content-change, under persistence, under Dhodapkar-Smith 2003, with the pass-period status of gemm stated (generated: tables/eusipco_table3.tex).",
        "Figure 2 (optional if space): the store-predicted versus state-measured assignment, compact.",
        "Text: the blind spot in two sentences; which axis separates which pair, with the physical reason; the floor kernels reported at floor; the comparator's row read honestly (better, worse, or within margin); the cost comparison in one sentence.",
        "Caveats in the text, not a section: one interval, twelve kernels, two archetypes under-supported, pass period declared for one kernel.",
    ],
    "conclusion": [
        "The three axes as the takeaway; the encoding paper (IFIP 2028) as the full account of the gate chain; the released dataset and code.",
    ],
}
# The reference plan (P2E sec. 3 "References: 19 to 23") as \cite placeholders keyed to p2.bib:
# (section, sentence it supports, key). Twenty-one entries; the twelve keys the three-builder
# addendum names (section 3.8 item 9) are all present; the epoch-2 comparator keys come from the
# block appended to p2.bib on 2026-09-17.
CITE_PLAN = (
    ("intro", "the field's per-pair count as the prior representation", "law2010volatile"),
    ("intro", "the field's per-pair count as the prior representation; U = mean +/- SD (the Table 2 comparator)", "savoldi2010uncertainty"),
    ("intro", "the hypervisor as the atomic acquisition point", "oliveri2025inconsistencies"),
    ("intro", "atomicity of a dump (the lineage, one citation)", "vomel2012correctness"),
    ("intro", "workload identification from memory behaviour", "jones2006antfarm"),
    ("intro", "workload identification from memory behaviour (one more, as space allows)", "lindemann2018identification"),
    ("intro", "the nearest prior feature set (Hirano and Kobayashi 2022; no longer the comparator, AA 2026-09-17)", "hirano2022ransomware"),
    ("intro", "the nearest prior line, the RanSMAP dataset", "hirano2025ransmap"),
    ("intro", "the Purnaye lineage, one sentence", "purnaye2022bishm"),
    ("intro", "performance-counter workload identification, one sentence (al-Nadim Yes)", "demme2013feasibility"),
    ("intro", "paper 1, the form (third person if the track is blind)", "khoury2026architecture"),
    ("intro", "the gate chain sentence: configurations chosen on the test data", "vanderkouwe2019sok"),
    ("intro", "the gate chain sentence: rigorous benchmarking", "kalibera2013rigorous"),
    ("representation", "Eq. 2: APF is the writable working set of live migration", "clark2005livemigration"),
    ("representation", "Eq. 2: the page dirty rate per pre-copy iteration", "akoush2010predicting"),
    ("representation", "Eq. 2: QEMU's calc-dirty-rate, the shipping tool", "qemu_calc_dirty_rate"),
    ("representation", "Eq. 3: per-page change magnitude as a feature (the byte distance)", "gupta2008differenceengine"),
    ("representation", "Eq. 4: the relative working set distance is one minus J (the Table 2 comparator)", "dhodapkar2003comparing"),
    ("representation", "Eq. 4: the working set signature lineage (ISCA 2002)", "dhodapkar2002managing"),
    ("data", "the twelve kernels as Berkeley dwarfs", "asanovic2006landscape"),
    ("data", "the dwarf benchmark lineage (Rodinia), one sentence as space allows", "che2009rodinia"),
)


def _cites(section: str) -> str:
    """The `\\cite` placeholder lines of one section: a comment naming the sentence, then the
    command on its own line (a LaTeX command, not prose; `prose_lines` admits it)."""
    out = []
    for sec, sentence, key in CITE_PLAN:
        if sec == section:
            out.append(f"% cite placeholder: {sentence}")
            out.append(f"\\cite{{{key}}}")
    return "\n".join(out) + "\n"


def cite_keys(tex: str) -> list[str]:
    """Every key inside a `\\cite{...}` outside comments, in order (duplicates kept)."""
    keys = []
    for raw in tex.splitlines():
        line = raw.split("%", 1)[0] if not raw.lstrip().startswith("%") else ""
        keys += re.findall(r"\\cite\{([^}]+)\}", line)
    return keys


def _equation(label: str, substance: str) -> str:
    return f"% {substance}\n\\begin{{equation}}\\label{{eq:{label}}}\\phantom{{\\cdot}}\\end{{equation}}\n"


def build_p2e_skeleton(*, documentclass: str = "IEEEtran") -> str:
    """The skeleton text (P2E sec. 3's five sections in order). `documentclass` is `IEEEtran`
    (`[conference]`, the default) or `article` (`[10pt,twocolumn]`, when IEEEtran.cls is absent)."""
    L = []
    if documentclass == "IEEEtran":
        L.append("\\documentclass[conference]{IEEEtran}")
    else:
        L.append("\\documentclass[10pt,twocolumn]{article}")
    comps = ", ".join(f"{COMPARATOR_DISPLAY[k]} [{k}]" for k in COMPARATORS_DEFAULT)
    L += [
        "% paper 2E (EUSIPCO) skeleton, written by plan11_encoding_ladder/latex_skeleton_eusipco.py (builder 2, build epoch 2) on " + now_iso(),
        "% headings, equation placeholders, table shells, figure placeholders, \\cite placeholders and substance bullets only; the wording is the author's",
        "% source of truth: apf_paper/P2E_STRUCTURE.md sections 2, 3, 5 (2026-09-17); comparators: council/14, P2_AUTHOR_ANSWERS.md Decisions of 2026-09-17",
        f"% comparator rows of Table 2 (AA 2026-09-17): {comps}; Law 2010 [law2010volatile] held for the IFIP version",
        "% the four gates of P2E sec. 0, on the critical path:",
        "\n".join(f"%   - {g}" for g in GATES_P2E_SEC0),
        "% the compression map of P2E sec. 5 (encoding paper section -> this paper):",
        "\n".join(f"%   - {m}" for m in COMPRESSION_MAP),
        "% venue facts that shape the page (P2E sec. 1; 2026 rules, 2027 UNVERIFIED):",
        "\n".join(f"%   - {v}" for v in VENUE_FACTS),
        "\\usepackage[utf8]{inputenc}",
        "\\usepackage[T1]{fontenc}",
        "\\usepackage{booktabs}",
        "\\usepackage{graphicx}",
        "\\usepackage{amsmath}",
        "\\graphicspath{{./}}",
        "",
        "% working title: the author's (P2E sec. 7: shorter than paper 2's; the representation in the title)",
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
        f"\\section{{{SECTION_TITLES[0]}}}",
        _comment_block(BULLETS["intro"]),
        _cites("intro"),
        f"\\section{{{SECTION_TITLES[1]}}}",
        _comment_block(BULLETS["representation"]),
        "".join(_equation(lab, sub) for lab, sub in EQUATIONS),
        _cites("representation"),
        _figure("fused_plane", "fig:p2e_plane",
                "Figure 1, the representation itself: the fused plane, median l0 over persistent pages against J to the next snapshot, one cloud per kernel, idle cells as the floor (report/figures/fig_fused_plane.pdf)",
                wide=True),
        _comment_block(BULLETS["resolution"]),
        f"\\section{{{SECTION_TITLES[2]}}}",
        _comment_block(BULLETS["data"]),
        _cites("data"),
        _shell(["field", "value"], "tab:p2e_table1", wide=False, rows=TABLE1_ROWS,
               note="Table 1, the dataset (P2E sec. 3.III; DOI blank until the release; idle cells blank until captured)"),
        f"\\section{{{SECTION_TITLES[3]}}}",
        _comment_block(BULLETS["results"]),
        _generated("eusipco_table2", _shell(list(EUSIPCO_TABLE2_COLUMNS), "tab:p2e_table2", wide=True, size="scriptsize",
                                            note="Table 2, reductions compared (P2E sec. 3.IV; from Table 7 plus the comparators; generated by tables_eusipco.py)")),
        _generated("eusipco_table3", _shell(list(EUSIPCO_TABLE3_COLUMNS), "tab:p2e_table3", wide=True, size="scriptsize",
                                            note="Table 3, the level-matched test (P2E sec. 3.IV; readings, no verdict; generated by tables_eusipco.py)")),
        _figure("table8_assignment", "fig:p2e_assign",
                "optional: the store-predicted versus state-measured assignment, compact (from report/tables/table8.csv); no generator writes this file, the framed placeholder stands",
                wide=False),
        f"\\section{{{SECTION_TITLES[4]}}}",
        _comment_block(BULLETS["conclusion"]),
        "% references: 19 to 23 (P2E sec. 1, sec. 3); the \\cite placeholders above are keyed to apf_paper/p2.bib:",
        "\n".join(f"%   - {key}: {sentence} [{sec}]" for sec, sentence, key in CITE_PLAN),
        "% not cited here, entered in p2.bib on 2026-09-17 for the author's choice (council/14 candidates 5 to 8 and the leads): "
        "bitchebe2020pml, ferreira2011libhashckpt, gioiosa2005tick, svard2011delta, sancho2004incremental, ibrahim2011precopy, nathan2015model",
        "% \\bibliographystyle{IEEEtran}",
        "% \\bibliography{p2}",
        "\\end{document}",
        "",
    ]
    return "\n".join(L)


def write_p2e_skeleton(out: Path, *, documentclass: str = "IEEEtran", standalone: Path | None = None,
                       force_standalone: bool = False) -> dict:
    """Write `<out>/report/p2e_skeleton.tex` (beside `report/tables/` and `report/figures/`),
    `<out>/report/p2e_skeleton.json` (`targets`, `n_prose_lines`, `n_cite`, `cite_keys`,
    `written`) and, when `standalone` is given, an identical copy there.

    The standalone write REFUSES a target that already holds prose, because this builder emits a
    scaffold with none; pass `force_standalone=True` to discard that prose deliberately. See the
    guard's comment below."""
    out = Path(out)
    rep = out / "report"
    rep.mkdir(parents=True, exist_ok=True)
    tex = build_p2e_skeleton(documentclass=documentclass)
    p = rep / "p2e_skeleton.tex"
    p.write_text(tex, encoding="utf-8")
    written = {"report": p}
    if standalone:
        sp = Path(standalone)
        # GUARD (2026-09-27). This builder emits a SCAFFOLD: empty title, empty author, section
        # headings, table shells and planning bullets. It contains no prose -- build_p2e_skeleton()
        # returns zero prose_lines() by construction. apf_paper/p2e_skeleton.tex forked from it on
        # 2026-09-17 and is now hand-maintained; it carries the author-approved wording, including
        # the EUSIPCO/encoding separation his advisor asked for. Overwriting it with the scaffold
        # destroys all of that silently. Refuse when the target already holds prose.
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
    keys = cite_keys(tex)
    write_json(rep / "p2e_skeleton.json", result_json(
        "latex_skeleton_eusipco",
        {"out": str(out), "documentclass": documentclass, "standalone": str(standalone) if standalone else None,
         "epoch": 2, "package_version": PACKAGE_VERSION, "written_at": now_iso()},
        CITATION,
        {"targets": targets(tex), "n_prose_lines": len(prose_lines(tex)), "n_cite": len(keys), "cite_keys": keys,
         "sections": list(SECTION_TITLES), "equations": [f"eq:{lab}" for lab, _ in EQUATIONS],
         "written": {k: str(v) for k, v in written.items()}}))
    return written


def build_parser() -> argparse.ArgumentParser:
    """The CLI parser (exposed so the runbook check can parse the documented commands)."""
    ap = argparse.ArgumentParser(description="plan11 EUSIPCO LaTeX skeleton (builder 2, epoch 2)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--documentclass", default="IEEEtran", choices=["IEEEtran", "article"])
    ap.add_argument("--standalone", default=None, help="also write the skeleton to this path (apf_paper/p2e_skeleton.tex)")
    ap.add_argument("--force-standalone", action="store_true", dest="force_standalone",
                    help="discard the standalone target's prose and overwrite it with the scaffold")
    return ap


def main(argv: list[str] | None = None) -> int:
    """The CLI of the module docstring: exit 0 on success, 1 on an internal error (SPEC 7.1)."""
    a = build_parser().parse_args(argv)
    try:
        w = write_p2e_skeleton(Path(a.out), documentclass=a.documentclass,
                               force_standalone=getattr(a, "force_standalone", False),
                               standalone=Path(a.standalone) if a.standalone else None)
    except Exception:
        traceback.print_exc()
        return 1
    for k, v in w.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
