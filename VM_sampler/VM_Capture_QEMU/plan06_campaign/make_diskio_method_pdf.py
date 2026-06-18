#!/usr/bin/env python3
"""Generate a well-explained PDF of how Plan 06 measures disk I/O.

    python3 plan06_campaign/make_diskio_method_pdf.py
-> docs/plan06_diskio_method.pdf
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, HRFlowable)

OUT = Path(__file__).resolve().parents[1] / "docs" / "plan06_diskio_method.pdf"

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Title"], fontSize=19, leading=23, spaceAfter=2)
SUB = ParagraphStyle("SUB", parent=ss["Normal"], fontSize=10.5, textColor=colors.HexColor("#666"), spaceAfter=10)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=13, leading=16,
                    textColor=colors.HexColor("#1a3c5e"), spaceBefore=14, spaceAfter=5)
BODY = ParagraphStyle("BODY", parent=ss["Normal"], fontSize=10, leading=15, spaceAfter=6, alignment=TA_LEFT)
CODE = ParagraphStyle("CODE", parent=ss["Normal"], fontName="Courier", fontSize=8.6,
                      leading=11.5, textColor=colors.HexColor("#102a43"),
                      backColor=colors.HexColor("#f1f4f8"), borderColor=colors.HexColor("#d4dbe4"),
                      borderWidth=0.5, borderPadding=6, leftIndent=2, spaceAfter=7, spaceBefore=2)
CAP = ParagraphStyle("CAP", parent=ss["Normal"], fontSize=8.5, textColor=colors.HexColor("#777"), spaceAfter=10)


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def code(t):
    return Paragraph(esc(t).replace("\n", "<br/>"), CODE)


def tbl(rows, widths):
    t = Table(rows, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3c5e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.8),
        ("FONTNAME", (0, 1), (0, -1), "Courier"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7fa")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cdd5df")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def P(t): return Paragraph(t, BODY)


story = []
story.append(Paragraph("Measuring Disk I/O in the VM Memory-Signal Pipeline", H1))
story.append(Paragraph("Plan 06 &mdash; how the second capture channel is sampled and computed", SUB))
story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#cdd5df"), spaceAfter=10))

story.append(Paragraph("1. Why a disk-I/O channel", H2))
story.append(P("The memory signal (APF, the fraction of guest pages that change between "
               "snapshots) cannot separate two workloads that are near-idle in memory with "
               "opposite labels: <b>slowburn</b> (a threat that writes files to disk) and "
               "<b>pagefault</b> (a benign workload that only faults pages in RAM). They alias "
               "in memory. What distinguishes them is <b>disk I/O</b> &mdash; a quantity APF "
               "cannot observe. Plan 06 adds a disk-I/O channel sampled at the same cadence as "
               "the memory snapshots, making each trajectory two-channel (APF + disk rate)."))

story.append(Paragraph("2. Data source &mdash; libvirt domblkstat", H2))
story.append(P("A single host-side command, issued once per snapshot, reads the guest virtual "
               "disk's I/O counters from libvirt:"))
story.append(code('virsh -c qemu:///system domblkstat "Kali Jeries" vda'))
story.append(P("It returns counters that are <b>cumulative since the VM booted</b>:"))
story.append(tbl([["counter", "meaning"],
                  ["rd_bytes", "total bytes read from the virtual disk"],
                  ["wr_bytes", "total bytes written to the virtual disk"],
                  ["rd_req / wr_req", "read / write request counts (not used here)"]],
                 [1.6 * inch, 4.3 * inch]))
story.append(P("We extract the two byte counters with awk:"))
story.append(code('rd=$(awk \'$2=="rd_bytes"{print $3}\' <<<"$stat")\n'
                  'wr=$(awk \'$2=="wr_bytes"{print $3}\' <<<"$stat")'))

story.append(Paragraph("3. When and where it is sampled", H2))
story.append(P("Inside the capture producer (<font face=\"Courier\" size=8>capture_producer_qemu_pmemsave.sh</font>), "
               "once per memory snapshot, right after the pmemsave dump while the <b>VM is "
               "paused</b>. Polling during the pause is cadence-safe: it adds host wall-clock "
               "but not guest-running time, so the memory-snapshot cadence is preserved. The "
               "whole block is gated by a <font face=\"Courier\" size=8>TIMING_DISKIO</font> "
               "flag &mdash; default off means the capture is byte-identical to the existing "
               "pipeline. Each sample appends one line to the cell's "
               "<font face=\"Courier\" size=8>diskio_trajectory.jsonl</font>:"))
story.append(code('{"seq": N, "t_emit_epoch": 1781049060.26, "rd_bytes": 502927360, "wr_bytes": 6521856}'))
story.append(P("<font face=\"Courier\" size=8>t_emit_epoch</font> is wall-clock at the poll "
               "(<font face=\"Courier\" size=8>date +%s.%N</font>). A <b>stride</b> option polls "
               "only every Nth snapshot, because the domblkstat call is the costly part; the "
               "cumulative counters make the per-cell rate unaffected by coarser sampling."))

story.append(Paragraph("4. From counters to rates (the math)", H2))
story.append(P("Because the counters are cumulative, the per-interval rate is the difference "
               "between consecutive samples divided by the elapsed time:"))
story.append(code("wr_rate(i) = ( wr_bytes[i+1] - wr_bytes[i] ) / ( t_emit_epoch[i+1] - t_emit_epoch[i] )\n"
                  "           [ bytes/s, then / 1e6 -> MB/s ]"))
story.append(P("Negative deltas (a counter reset on VM reboot) clamp to zero. The per-interval "
               "rates are then aggregated into per-cell features:"))
story.append(tbl([["feature", "definition"],
                  ["wr_rate_mean", "mean write rate over the cell (MB/s)"],
                  ["wr_rate_max", "peak write rate (MB/s)"],
                  ["wr_rate_p95", "95th-percentile write rate (MB/s)"],
                  ["rd_rate_mean", "mean read rate (MB/s)"],
                  ["wr_total", "total bytes written = last - first (MB)"]],
                 [1.7 * inch, 4.2 * inch]))

story.append(Paragraph("5. What it measures (and what it does not)", H2))
story.append(P("<b>Layer:</b> the <i>virtual</i> block device (vda) &mdash; the I/O the guest "
               "issued to its own disk, measured from the <b>host</b> via libvirt. No agent runs "
               "inside the guest."))
story.append(P("<b>After the guest page cache:</b> writes buffer in guest RAM and reach vda on "
               "writeback, so short bursts lag; cumulative <font face=\"Courier\" size=8>wr_bytes</font> "
               "over a 600&nbsp;s cell is accurate even though the per-interval rate is lumpy."))
story.append(P("<b>It is not</b> the host's physical disk, not the page cache, and not the qcow2 "
               "image file's size."))

story.append(Paragraph("6. Validation", H2))
story.append(P("The host counter was cross-checked against the guest's own view "
               "(<font face=\"Courier\" size=8>/proc/diskstats</font> sectors &times; 512) during "
               "a 4&nbsp;GB sequential writer. They agree:"))
story.append(tbl([["source", "bytes written"],
                  ["host  virsh domblkstat vda", "~7.8 GB"],
                  ["guest /proc/diskstats", "~7.8 GB"]],
                 [3.0 * inch, 2.9 * inch]))
story.append(Paragraph("Host-side libvirt and the guest's own block layer report the same total, "
                       "confirming the measurement is correct.", CAP))

story.append(Paragraph("7. The pipeline (where the numbers flow)", H2))
story.append(code("producer  domblkstat  ->  diskio_trajectory.jsonl   (per snapshot, cumulative bytes)\n"
                  "build_cells_dir.py    ->  pairs each cell's diskio with its apf trajectory\n"
                  "diskio_features.py    ->  deltas / time -> per-cell wr_rate_* , rd_rate, wr_total\n"
                  "diskio_lift.py        ->  feeds the features to the classifier / masquerade test"))

story.append(Paragraph("8. Honest caveats", H2))
story.append(P("&bull; <b>Page-cache lag</b> makes the per-interval rate lumpy; the per-cell "
               "aggregate over a long window is the robust quantity.<br/>"
               "&bull; <b>domblkstat is the costly call</b> (~0.5&ndash;1&nbsp;s each); the stride "
               "option amortizes it for long campaigns.<br/>"
               "&bull; <b>Disk I/O is a behavioral axis, not a threat marker</b> &mdash; the "
               "heaviest writer measured was a benign memory-mapped workload (~140&nbsp;MB/s), and "
               "a threat (memory-capped) wrote almost nothing. A second channel buys behavioral "
               "<i>disambiguation</i> of workloads that alias in the first channel, not detection."))

SimpleDocTemplate(str(OUT), pagesize=LETTER,
                  leftMargin=0.85 * inch, rightMargin=0.85 * inch,
                  topMargin=0.8 * inch, bottomMargin=0.7 * inch,
                  title="Plan 06 - Measuring Disk I/O",
                  author="memorySignal / Plan 06").build(story)
print(f"wrote {OUT}")
