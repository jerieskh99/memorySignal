#!/usr/bin/env python3
"""Portable PDF of the comprehensive Plan 05 -> Plan 06 data: figures + tables.
Run make_comprehensive_report.py first (it produces the figures + CSV).

    python3 plan06_campaign/make_comprehensive_pdf.py -> docs/plan06_comprehensive.pdf
"""
import csv
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak)

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent / "docs"
FIG = DOCS / "plan06_figs"
OUT = DOCS / "plan06_comprehensive.pdf"

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Title"], fontSize=18, spaceAfter=2)
SUB = ParagraphStyle("SUB", parent=ss["Normal"], fontSize=9.5, textColor=colors.HexColor("#666"), spaceAfter=10)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12.5, textColor=colors.HexColor("#1a3c5e"), spaceBefore=12, spaceAfter=4)
CAP = ParagraphStyle("CAP", parent=ss["Normal"], fontSize=8, textColor=colors.HexColor("#777"), spaceAfter=10)
BODY = ParagraphStyle("BODY", parent=ss["Normal"], fontSize=9.5, leading=14, spaceAfter=6)

cells = list(csv.DictReader(open(DOCS / "plan06_all_data.csv")))
lift = json.load(open(HERE / "downstream" / "diskio_lift_full.json"))["results"]


def fig(name, cap, width=6.4):
    img = Image(str(FIG / name)); ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width * inch; img.drawHeight = width * ratio * inch
    return [img, Paragraph(cap, CAP)]


def styled(t, fs=7.4, header_bg="#1a3c5e"):
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), fs),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f5f8")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cdd5df")),
        ("ALIGN", (3, 0), (-1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


story = [Paragraph("Plan 05 &rarr; Plan 06 &mdash; Comprehensive Data", H1),
         Paragraph("All 66 cells, 11 workloads, both channels. Memory (APF) = Plan 05; disk = Plan 06.", SUB)]

story.append(Paragraph("1. Two-channel behavior space", H2))
story += fig("scatter_2channel.png", "Each point = one cell. X = APF mean (memory), Y = disk write "
             "rate (Plan 06). Memory-aliased workloads (slowburn vs pagefault) separate on the disk "
             "axis; the heaviest writer is benign (mmap).")

story.append(Paragraph("2. Per-workload, both channels", H2))
story += fig("bar_apf.png", "Memory: APF mean per workload.")
story += fig("bar_disk.png", "Disk: write rate per workload (log).")
story += fig("scatter_rdwr.png", "Read vs write rate per cell. Encrypt-in-place threats "
            "(ransom_seq/selective) read AND write; benign mmap is a pure writer (reads ~0).")
story += fig("bar_rdfrac.png", "Read fraction of disk I/O per workload. Only the two "
            "encryptors read meaningfully; everything else is writes-only.")
story += fig("heatmap.png", "Per-workload metric heatmap (rows min-max normalized).")

# per-workload table
WL = {}
for c in cells:
    WL.setdefault(c["workload"], []).append(c)
def mean(rs, k): return sum(float(r[k]) for r in rs) / len(rs)
head = ["family", "workload", "kind", "APF mean", "write MB/s", "write MB",
        "read MB/s", "read frac", "win"]
data = [head]
for wl in sorted(WL, key=lambda w: (WL[w][0]["family"], w)):
    rs = WL[wl]
    data.append([rs[0]["family"], rs[0]["workload"].replace("sandbox_", "").replace("_v2", ""),
                 rs[0]["kind"], f"{mean(rs,'apf_mean'):.4f}",
                 f"{mean(rs,'wr_rate_mbs'):.3f}", f"{mean(rs,'wr_total_mb'):.0f}",
                 f"{mean(rs,'rd_rate_mbs'):.3f}", f"{mean(rs,'rd_frac'):.3f}",
                 f"{mean(rs,'n_windows'):.0f}"])
story.append(styled(Table(data, colWidths=[0.85*inch, 1.6*inch, 0.55*inch, 0.7*inch, 0.7*inch,
                                           0.6*inch, 0.7*inch, 0.6*inch, 0.45*inch]), fs=8))

story.append(PageBreak())
story.append(Paragraph("3. Classification: what each channel buys", H2))
story += fig("bar_classification.png", "Accuracy by feature set across tasks. Disk helps "
             "identification, hurts threat/benign detection.")
tasks = [("binary LOWO", lambda r: r["binary_lowo"]["lowo_acc"]),
         ("binary LOFO", lambda r: r["binary_lofo"]["lofo_acc"]),
         ("family LORO", lambda r: r["family5_lowo"]["loro_acc"]),
         ("family LOWO", lambda r: r["family5_lowo"]["lowo_acc"]),
         ("instance LORO", lambda r: r["instance11_loro"]["loro_acc"])]
cd = [["task", "APF only", "+peak/var", "+disk"]]
for t, g in tasks:
    cd.append([t, f"{g(lift['AGNOSTIC']):.3f}", f"{g(lift['AGNOSTIC+peakvar']):.3f}",
               f"{g(lift['AGNOSTIC+peakvar+diskio']):.3f}"])
story.append(styled(Table(cd, colWidths=[1.6*inch, 1.1*inch, 1.1*inch, 1.1*inch]), fs=9))
story.append(Paragraph("Disk I/O (writes + reads) helps characterization &mdash; instance "
                       "0.924&rarr;0.985 (the read signature fingerprints the two encryptors), "
                       "family LOWO 0.348&rarr;0.439 crossing the 0.364 baseline &mdash; and hurts "
                       "detection (binary LOWO 0.742&rarr;0.636). Reads sharpen instance ID but do "
                       "not detect: 3 of 5 threats read ~0, like the benigns. A characterization "
                       "channel, not a detector.", BODY))

story.append(PageBreak())
story.append(Paragraph("4. Every cell (66)", H2))
ch = ["fam", "workload", "kind", "dur", "rep", "APF", "win", "wMB/s", "wMB", "rMB/s", "rfrac"]
cdata = [ch]
for c in cells:
    cdata.append([c["family"], c["workload"].replace("sandbox_", "").replace("_v2", ""), c["kind"],
                  c["duration_s"], c["rep"], f"{float(c['apf_mean']):.4f}", c["n_windows"],
                  f"{float(c['wr_rate_mbs']):.3f}", f"{float(c['wr_total_mb']):.0f}",
                  f"{float(c['rd_rate_mbs']):.3f}", f"{float(c['rd_frac']):.3f}"])
t = Table(cdata, repeatRows=1, colWidths=[0.6*inch, 1.45*inch, 0.55*inch, 0.38*inch, 0.32*inch,
                                          0.55*inch, 0.42*inch, 0.55*inch, 0.45*inch, 0.55*inch, 0.48*inch])
story.append(styled(t, fs=6.8))

SimpleDocTemplate(str(OUT), pagesize=LETTER, leftMargin=0.7*inch, rightMargin=0.7*inch,
                  topMargin=0.7*inch, bottomMargin=0.6*inch,
                  title="Plan 05->06 comprehensive data").build(story)
print(f"wrote {OUT}")
