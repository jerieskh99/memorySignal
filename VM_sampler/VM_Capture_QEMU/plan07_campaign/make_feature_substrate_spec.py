#!/usr/bin/env python3
"""Feature-substrate spec -> docs/feature_substrate_spec.md + docs/feature_substrate_spec.pdf.

One content model, two renderers (Markdown + PDF via reportlab) so both documents are
identical. Catalogues every per-page memory-delta metric, organised into the perspective
families A-G (+H appendix). Each family is a full table with columns:
Group | Metric | Measures | What it catches.

Notation is ASCII-safe (abs(), dot(), norm(), std(), var(), cov(), sqrt, sum, log) so it
renders cleanly in both Markdown and the PDF. p = old page, q = new page (4096 bytes each);
P, Q = their 256-bin byte histograms (as probabilities).

Run: python3 plan07_campaign/make_feature_substrate_spec.py
"""
import html as _html
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, KeepTogether)

DOCS = Path(__file__).resolve().parent.parent / "docs"
DATE = "June 2026"
INK = "#1A1A1A"; MUTED = "#5B5B5B"; LINE = "#D8D6CE"; PANEL = "#F5F3EE"; NEW = "#B9822A"

# ---------------------------------------------------------------- content model
# Each family: id, title, intro, catch_header, grouped(bool), group_label, rows.
# Grouped rows: (group, metric, measures, catches).  Ungrouped rows: (metric, measures, catches).

FAM_A = {
 "id": "A", "title": "Amount of change  (alternatives to Hamming)",
 "catch": "What it catches (vs Hamming)", "grouped": True, "group_label": "Space",
 "intro":
   "Amount = how much one page changed. Hamming measures it in bit-position space. The SAME "
   "question has different answers in three representation SPACES: positional (where the bytes "
   "sit), distributional (the byte histogram, position-free), and informational (compressibility). "
   "A memmove is huge positionally but ~zero distributionally; encryption is huge distributionally "
   "and informationally. Each row is a magnitude heat-map channel.",
 "rows": [
  ("Positional", "Hamming (bit)", "popcount(p XOR q)", "bits flipped -- total bit churn (the baseline)"),
  ("Positional", "Byte-Hamming / L0", "count(p_j != q_j)", "how many BYTES changed (vs how many bits) -- spread of change"),
  ("Positional", "L1 / SAD", "sum abs(p_j - q_j)", "total byte-magnitude moved -- weights big byte jumps, not just that a byte changed"),
  ("Positional", "L2 / Euclidean", "sqrt(sum (p_j - q_j)^2)", "Euclidean magnitude -- emphasises a few large byte changes"),
  ("Positional", "L-inf / Chebyshev", "max abs(p_j - q_j)", "the single biggest byte jump -- an outlier change a page-wide sum hides"),
  ("Positional", "Mean abs change", "mean abs(p_j - q_j)", "average per-byte change -- size-normalised magnitude"),
  ("Positional", "Gradient-magnitude change", "energy of abs(grad q) - abs(grad p)", "change in local roughness -- did the page get smoother/jaggier"),
  ("Distributional", "Total Variation", "0.5 * sum abs(P_b - Q_b)", "max probability mass that moved -- coarse histogram shift"),
  ("Distributional", "Chi-square", "sum (P_b - Q_b)^2 / (P_b + Q_b)", "frequency-weighted histogram diff -- sensitive to spikes / rare bins"),
  ("Distributional", "Hellinger", "sqrt(sum (sqrt(P_b) - sqrt(Q_b))^2) / sqrt(2)", "bounded [0,1] metric -- balanced, well-behaved histogram distance"),
  ("Distributional", "Kullback-Leibler", "sum P_b * log(P_b / Q_b)", "info lost approximating one by the other -- raw surprise (asymmetric, unbounded)"),
  ("Distributional", "Jensen-Shannon", "symmetrised, bounded KL", "symmetric bounded version -- survives non-overlapping byte supports"),
  ("Distributional", "Wasserstein / EMD", "min transport cost P->Q", "how FAR byte-mass moved (geometry-aware) -- gradual shift vs spike; order matters"),
  ("Distributional", "Bhattacharyya", "-ln sum sqrt(P_b * Q_b)", "distribution overlap -- close cousin of Hellinger"),
  ("Distributional", "Histogram intersection", "sum min(P_b, Q_b)", "shared mass -- simple overlap similarity"),
  ("Informational", "Entropy delta", "H(q) - H(p)", "change in randomness -- toward/away from ciphertext (signed = also a direction)"),
  ("Informational", "Normalized Compression Distance", "(C(pq) - min) / max", "shared compressible structure -- weak on high-entropy (encrypted) pages"),
  ("Informational", "Compressed-size delta", "len(C(q)) - len(C(p))", "change in raw complexity -- page got simpler / more complex"),
  ("Informational", "Structural-entropy change", "change in entropy-stream / wavelet signature", "byte-granularity structural shift -- the malware-classification feature"),
  ("Informational", "LZ-complexity change", "change in LZ77 factor count", "algorithmic-complexity change -- Kolmogorov proxy"),
 ],
}

FAM_B = {
 "id": "B", "title": "Direction of change  (alternatives to Cosine)",
 "catch": "Direction sense (what it catches)", "grouped": True, "group_label": "Reading",
 "intro":
   "Direction = which WAY a page changed. Cosine reads only the STRUCTURE (pattern) component. "
   "SSIM shows that similarity decomposes into structure + level + spread; add polarity, "
   "distributional-direction, and spatial-shift. Each row's direction sense says which way the "
   "change points.",
 "rows": [
  ("Structure", "Cosine similarity", "dot(p,q) / (norm(p) * norm(q))", "angle between content vectors -- did the pattern stay aligned (the baseline)"),
  ("Structure", "Pearson correlation", "centered cosine (mean removed)", "linear co-variation ignoring level shift -- pattern alignment minus brightness"),
  ("Structure", "Spearman rank corr.", "correlation of byte ranks", "did the ORDERING of byte values persist -- monotonic structure even if rescaled"),
  ("Structure", "Kendall tau", "concordant-pair fraction", "ordinal agreement -- robust ordering persistence"),
  ("Structure", "SSIM structure term", "cov(p,q) / (std(p) * std(q))", "local structural similarity -- the 's' term of SSIM"),
  ("Level", "Mean shift (signed)", "mean(q) - mean(p)", "did bytes get bigger or smaller on average -- net 'brightness' direction"),
  ("Level", "SSIM luminance term", "(2*mu_p*mu_q + c) / (mu_p^2 + mu_q^2 + c)", "closeness of means -- level agreement"),
  ("Level", "Median shift", "median(q) - median(p)", "robust level direction -- central-tendency move resistant to outliers"),
  ("Spread", "Variance ratio", "var(q) / var(p)", "got more or less varied -- densifying vs sparsifying direction"),
  ("Spread", "Std-dev delta (signed)", "std(q) - std(p)", "change in spread -- additive form of the contrast direction"),
  ("Spread", "SSIM contrast term", "(2*std_p*std_q + c) / (var_p + var_q + c)", "spread agreement -- the 'c' term of SSIM"),
  ("Spread", "Range / IQR delta", "change in (max-min) or IQR", "dynamic-range direction -- widened vs narrowed value range"),
  ("Polarity", "Fraction up vs down", "count(q_j > p_j) vs count(q_j < p_j)", "which way the changed bytes moved -- the directional 'current'"),
  ("Polarity", "Net signed drift", "sum (q_j - p_j)", "net byte-value flow -- accumulation vs erasure"),
  ("Polarity", "Sign-of-delta entropy", "entropy of the up/down pattern", "is the direction coherent or a mixed-up wash"),
  ("Distributional dir.", "Entropy delta sign", "sign(H(q) - H(p))", "toward random or toward structured -- encryption vs zeroing direction"),
  ("Distributional dir.", "Histogram-mean shift sign", "sign of byte-value mean move", "toward high or low byte values"),
  ("Distributional dir.", "Move-toward-uniform", "KL(q||unif) - KL(p||unif)", "approaching ciphertext -- uniformisation direction"),
  ("Distributional dir.", "Zero-mass delta", "Q_0 - P_0 (zero-bin mass change)", "toward zeroed / freed memory"),
  ("Spatial-shift", "Cross-correlation lag", "argmax circular xcorr(p, q)", "content SHIFTED within the page (memmove/scroll) -- offset & direction"),
  ("Spatial-shift", "Phase-correlation peak", "peak of normalised cross-power spectrum (FFT)", "translational shift -- robust shift estimate"),
  ("Spatial-shift", "Optimal byte-rotation", "best cyclic alignment offset", "content rotated within the page"),
 ],
}

FAM_C = {
 "id": "C", "title": "Content / character of q  (what it became; reference-free)",
 "catch": "What it catches", "grouped": False, "group_label": "",
 "intro":
   "Not amount, not direction -- describe the NEW page q itself, with no reference to p. "
   "Reference-free fingerprints of what the page became (ciphertext, zeros, text, pointers, ...).",
 "rows": [
  ("Shannon entropy of q", "-sum P_b * log2(P_b)", "randomness / compressibility -- ciphertext (~8) vs structured (4-5) vs sparse (low)"),
  ("Structural entropy", "entropy stream / wavelet energy over the page", "texture of randomness -- localised encrypted regions"),
  ("Distinct byte count", "count(b: count_b > 0)", "palette size -- full 256 = random/dense; few = structured"),
  ("Zero-byte fraction", "count(byte == 0) / 4096", "emptiness -- zeroed / freed / sparse pages"),
  ("Constant-fill fraction", "max_b count(b) / 4096 (e.g. 0xFF)", "fill patterns -- uninitialised / poisoned memory"),
  ("Printable-ASCII fraction", "count(0x20..0x7E) / 4096", "text-likeness -- strings / JSON / code vs binary"),
  ("Byte moments", "mean, var, skew, kurtosis of bytes", "distribution shape -- coarse content fingerprint"),
  ("Bigram entropy", "entropy over adjacent byte pairs", "local structure -- pointers / tables vs random"),
  ("Autocorrelation peak", "max self-correlation at lag > 0", "periodicity within the page -- arrays / records / strides"),
  ("Chi-square uniformity", "sum (count_b - E)^2 / E", "deviation from uniform -- the 'is it random' (encryption) test"),
 ],
}

FAM_D = {
 "id": "D", "title": "Internal structure / texture  (where & how, within the unit)",
 "catch": "What it catches", "grouped": True, "group_label": "Group",
 "intro":
   "WHERE the change sits inside the page, and how textured the content is. Two groups: the "
   "intra-page change mask (location of the change), and the texture of q (its spatial pattern).",
 "rows": [
  ("Change location", "Changed-run count", "number of contiguous changed-byte segments", "clustered vs scattered change -- localised edit vs full rewrite"),
  ("Change location", "Change span", "last - first changed offset", "extent of the touched region -- narrow patch vs whole page"),
  ("Change location", "Change centroid", "center of mass of changed offsets", "where in the page -- header vs body edits"),
  ("Change location", "Longest changed run", "max contiguous changed bytes", "biggest edited block -- block-overwrite size"),
  ("Change location", "Change density in span", "changed bytes / span", "compactness of the edit"),
  ("Texture of q", "Gradient / edge energy", "sum abs(q_{j+1} - q_j)", "local roughness -- smooth (zeros/text) vs jagged (random/binary)"),
  ("Texture of q", "GLCM / Haralick", "contrast, homogeneity, energy, correlation of byte co-occurrence", "texture class -- structured vs noise"),
  ("Texture of q", "LBP histogram", "local binary patterns over the byte sequence", "micro-texture -- repeated motifs"),
  ("Texture of q", "Spatial-frequency energy", "low- vs high-band FFT/DCT energy", "coarse vs fine structure -- periodic tables vs random"),
  ("Texture of q", "Run-length distribution", "lengths of constant-byte runs", "fill / structure -- RLE-compressibility proxy"),
 ],
}

FAM_E = {
 "id": "E", "title": "Spatial field across pages  (the heat map AS an image)",
 "catch": "What it catches", "grouped": False, "group_label": "",
 "intro":
   "Lay any per-page channel (amount, direction, ...) across physical address -> a 2D image "
   "(512x512 for a 1 GB guest). These describe the change FIELD as an image. All are DERIVED "
   "offline from a per-page channel + page_index -- nothing extra is stored.",
 "rows": [
  ("Active-area ratio (APF)", "changed pages / total pages", "overall breadth -- the foreground fraction (the baseline)"),
  ("Moran's I", "spatial autocorrelation of the field", "clustered / dispersed / random layout -- sequential sweep vs scattered random"),
  ("Connected-component count", "number of contiguous changed-page regions", "fragmentation -- one region vs many specks"),
  ("Component size distribution", "sizes of changed-page clusters", "region scale -- big buffers vs scattered touches"),
  ("Address-axis FFT / autocorr.", "periodicity of the changed-page index set", "stride detection -- strided access (every Nth page)"),
  ("GLCM of the magnitude field", "Haralick texture of the heat map", "spatial change pattern / texture"),
  ("Concentration (Gini / entropy)", "inequality of magnitude across pages", "a few hot pages vs uniform spread"),
  ("Field edges / gradient", "boundaries of active regions", "working-set edges -- where activity starts/stops in address space"),
  ("Centroid & dispersion", "center and spread of the active set", "WHERE in RAM the work sits, and how concentrated"),
 ],
}

FAM_F = {
 "id": "F", "title": "Temporal dynamics  (the heat-map movie; needs page identity)",
 "catch": "What it catches", "grouped": True, "group_label": "Level",
 "intro":
   "Per-page over snapshots (requires storing page_index). The heat map as a MOVIE: how pages "
   "and the active region evolve over time. All DERIVED offline from page_index + the per-page "
   "channels.",
 "rows": [
  ("Per-page", "Revisit rate", "snapshots a page changes / total", "page hotness -- hot loop (high) vs touched-once (low)"),
  ("Per-page", "Inter-change interval", "gaps between a page's changes", "rhythm -- steady vs bursty"),
  ("Per-page", "Burstiness", "Fano factor / CV of intervals", "clumping of a page's writes"),
  ("Per-page", "Change-series autocorr.", "periodicity of a page's activity over time", "cyclic access to the same page"),
  ("Per-page", "Page change-point", "when a page's behaviour shifts", "per-page phase transitions"),
  ("Per-page", "Page lifetime", "first-touch to last-touch span", "allocation -> free lifetime"),
  ("Field-level", "Working-set size", "distinct pages active in a window", "footprint dynamics -- growing vs stable"),
  ("Field-level", "Working-set migration", "Jaccard between consecutive active sets", "does the hot region MOVE -- sweep (low overlap) vs hot loop (high)"),
  ("Field-level", "Active-region drift", "displacement of the activity centroid", "sweep direction & speed (optical-flow of the heat map)"),
  ("Field-level", "New-page rate", "newly-touched pages per step", "allocation / discovery rate"),
 ],
}

FAM_G = {
 "id": "G", "title": "Reference basis  (what we diff against; multiplies A-F)",
 "catch": "What it catches", "grouped": False, "group_label": "",
 "intro":
   "Every metric above diffs against SOMETHING. Hamming/cosine use t vs t-1. Changing the "
   "reference is a whole perspective family -- the memory-forensics standard is diff-vs-baseline. "
   "This multiplies every metric in A-F. Chosen offline by picking the reference snapshot. "
   "PHASE 2 (deferred): the non-consecutive references need a NEW capture state model -- a pinned "
   "baseline, an EWMA accumulator, or buffered lagged snapshots (the pipeline deletes prev "
   "immediately today). Only the histogram-space references are offline-derivable; positional "
   "baseline/lag diffs need the raw bytes at both times. May require a capture redesign.",
 "rows": [
  ("Consecutive (t vs t-1)", "adjacent snapshots", "instantaneous change -- the baseline (what Hamming/cosine use)"),
  ("Baseline / idle", "vs a fixed idle snapshot", "drift from rest -- total deviation from 'doing nothing' (forensics standard)"),
  ("Self EWMA", "vs the page's own moving average", "self-anomaly -- unusual relative to that page's own habit"),
  ("N-lag / multi-scale", "vs t-k for several k", "slow drift vs fast churn -- separates timescales"),
  ("Boot / origin", "vs the first snapshot", "cumulative change since start -- lifetime accumulation"),
  ("Canonical page", "vs a reference page (e.g. zero page)", "absolute content type -- zero-distance, fill-distance"),
 ],
}

FAM_H = {
 "id": "H", "title": "Learned / embedding  (appendix; heavier; future work)",
 "catch": "What it catches", "grouped": False, "group_label": "",
 "intro":
   "Representations learned from data rather than computed in closed form. Heavier (training, "
   "and usually need the raw bytes), so out of scope for the first substrate, but listed for "
   "completeness as a distinct perspective family. PHASE 2 (deferred): the learned half "
   "(autoencoder, learned embeddings) needs a trained model + a TWO-PASS capture (store page "
   "samples -> train offline -> run inference at capture) + a model registry -- a new modelling "
   "paradigm. The fixed-hash sketches (SimHash, random projection) are the exception: data-"
   "independent, no training, so they already fit the Phase 1 form.",
 "rows": [
  ("Autoencoder reconstruction error", "distance from a learned 'normal' page manifold", "novelty of a page's content -- one-class page anomaly"),
  ("Learned page embedding distance", "distance in a trained embedding space", "semantic page similarity"),
  ("SimHash / fuzzy hash (ssdeep)", "locality-sensitive hash distance", "cheap near-duplicate detection"),
  ("Random-projection sketch", "L2 in a Johnson-Lindenstrauss sketch", "cheap approximate L2 at low storage"),
 ],
}

FAMILIES = [FAM_A, FAM_B, FAM_C, FAM_D, FAM_E, FAM_F, FAM_G, FAM_H]

INTRO_PARAS = [
 "This spec catalogues the per-page memory-delta metrics for the capture feature-substrate -- the "
 "set of primitives we extract once, in-pass, from each changed 4 KB page before its snapshot is "
 "deleted, so that all later feature engineering is offline with no recapture.",
 "The generative idea: a metric is a choice on four axes -- representation x reading x reference x "
 "unit. Hamming = bit-space x amount x consecutive x page. Cosine = vector-space x direction x "
 "consecutive x page. Varying any axis generates the families below. A = amount (vary the space), "
 "B = direction (vary the reading), C = content of q, D = internal structure, E = the field as an "
 "image, F = the field over time, G = the reference, H = learned representations.",
 "Notation: p = old page, q = new page (4096 bytes each); P, Q = their 256-bin byte histograms as "
 "probabilities; C() = a compressor. Each family is one perspective -- a per-unit heat-map channel.",
]

PHASING = {
 "title": "Phasing: what fits the existing capture form (Phase 1) vs what needs a redesign (Phase 2)",
 "paras": [
   "Phase 1 -- Families A-F. All fit the existing capture FORM: a closed-form function of the local "
   "consecutive diff (prev vs curr), computed in the same page-comparison pass the APF differ already "
   "runs, stored as a small per-page descriptor. This is a small, additive, flag-gated emit (mirroring "
   "the Plan 06 disk block) -- byte-identical when off. It does NOT require redesigning or re-proving "
   "the capture architecture; it is a small change within the proven form.",
   "Phase 2 -- Families G and H (deferred). These break the form. G needs a new reference / state "
   "model (pin a baseline snapshot, keep an EWMA accumulator, or buffer N lagged snapshots -- the "
   "pipeline deletes prev immediately today). H's learned half needs a trained model and a two-pass "
   "capture (store page samples -> train -> run inference at capture) plus a model registry. Both may "
   "require redesigning the capture, so they are research extensions, not the first substrate.",
 ],
}

SYNTHESIS = {
 "title": "Synthesis: channels, and what to store vs derive",
 "paras": [
   "Treat each per-page perspective as an image CHANNEL; the substrate is a sparse (changed pages x "
   "channels) tensor per snapshot. The decisive rule: store only what needs the raw bytes and cannot "
   "be derived. Families E, F, and G are entirely DERIVED offline from stored per-page channels + "
   "page_index. The distributional half of A, the distributional direction of B, and all of C are "
   "derived offline from a stored byte histogram. So the in-pass store list is small; the offline "
   "experiment space is enormous.",
   "The one real decision is the byte histogram (256 B/page, exact, but ~40 GB on a 100%-APF cell) "
   "vs a handful of cheap moments (~10 B, approximate). Likely answer: moments always, full histogram "
   "on quiet cells only. Compute budget (capture must keep up at 500 ms over 1 GB): cheap in-pass = "
   "Hamming, L0, L1, L-inf, mean, var, entropy, zero-frac, histogram; expensive = EMD, NCD, GLCM -- "
   "so those are never in-pass, they are derived offline from the histogram + index.",
 ],
 "store_rows": [
   ("page_index", "u32 (4 B)", "STORE", "unlocks all of E, F, G offline -- highest leverage single field"),
   ("hamming (bit)", "u16 (2 B)", "STORE", "positional amount; cannot come from a histogram"),
   ("L1 / SAD", "u32 (4 B)", "STORE", "positional amount, magnitude-weighted"),
   ("cosine", "u16 (2 B)", "STORE", "structure direction; needs the raw vectors"),
   ("byte histogram of q", "256 B (quiet cells) / moments ~10 B", "STORE (tiered)", "derives all of A-distributional, B-distributional, C"),
   ("n_runs, span", "2 B", "STORE", "intra-page structure (D, change-location)"),
   ("everything in E, F, G", "0 B", "DERIVE", "computed offline from page_index + the stored channels"),
   ("GLCM / FFT texture (D)", "0 B (or raw)", "DERIVE / defer", "needs raw bytes; defer unless a Tier-2 page sample is kept"),
 ],
 "store_cols": ["Channel", "Size", "Store / derive", "Why"],
}

SOURCES = [
 ("Histogram distance overview", "https://safjan.com/metrics-to-compare-histograms/"),
 ("Divergence metrics guide", "https://bookdown.org/mike/data_analysis/divergence-metrics-and-tests-for-comparing-distributions.html"),
 ("SSIM components", "https://metricgate.com/docs/structural-similarity-index/"),
 ("SSIM derivation (Wang et al.)", "https://www.math.uwaterloo.ca/~ervrscay/talks/post206_iciar12.pdf"),
 ("NCD on malware", "https://link.springer.com/article/10.1007/s11416-015-0260-0"),
 ("Detecting malware with information complexity", "https://arxiv.org/pdf/1502.07661"),
 ("Image descriptors (color/shape/texture)", "https://www.oreateai.com/blog/understanding-image-descriptors-a-deep-dive-into-color-shape-and-texture/41b63a70b1c4b0c756c53d0b6296d2ca"),
 ("Intel PML working-set estimation", "https://arxiv.org/pdf/2001.09991"),
 ("eBPF working-set-size estimation", "https://arxiv.org/pdf/2303.05919"),
 ("Moran's I image segmentation", "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11630814/"),
 ("Image biomarker standardisation (texture)", "https://arxiv.org/pdf/1612.07003"),
]

# ---------------------------------------------------------------- markdown renderer
def _md_cell(s):
    return s.replace("|", r"\|")

def render_md():
    out = ["# Feature-Substrate Spec -- per-page memory-delta metrics",
           "", f"*Capture-side primitive catalogue. {DATE}.*", ""]
    for p in INTRO_PARAS:
        out.append(p); out.append("")
    out.append("## How to read this")
    out.append("")
    out.append("Each family below is one **perspective** (a per-unit heat-map channel). Every family "
               "has a full table: the metric, what it **Measures** (the formula/definition), and **What "
               "it catches** that the others miss. Family A is split by representation **Space**; "
               "Family B by **Reading**; the rest by their natural groups.")
    out.append("")
    out.append(f"## {PHASING['title']}")
    out.append("")
    for p in PHASING["paras"]:
        out.append(p); out.append("")
    for fam in FAMILIES:
        out.append(f"## Family {fam['id']} -- {fam['title']}")
        out.append("")
        out.append(fam["intro"]); out.append("")
        if fam["grouped"]:
            header = f"| {fam['group_label']} | Metric | Measures | {fam['catch']} |"
            sep = "|" + "|".join(["---"] * 4) + "|"
            out.append(header); out.append(sep)
            for g, metric, meas, catch in fam["rows"]:
                out.append(f"| {_md_cell(g)} | {_md_cell(metric)} | {_md_cell(meas)} | {_md_cell(catch)} |")
        else:
            header = f"| Metric | Measures | {fam['catch']} |"
            sep = "|" + "|".join(["---"] * 3) + "|"
            out.append(header); out.append(sep)
            for metric, meas, catch in fam["rows"]:
                out.append(f"| {_md_cell(metric)} | {_md_cell(meas)} | {_md_cell(catch)} |")
        out.append("")
    # synthesis
    out.append(f"## {SYNTHESIS['title']}")
    out.append("")
    for p in SYNTHESIS["paras"]:
        out.append(p); out.append("")
    out.append("| " + " | ".join(SYNTHESIS["store_cols"]) + " |")
    out.append("|" + "|".join(["---"] * len(SYNTHESIS["store_cols"])) + "|")
    for r in SYNTHESIS["store_rows"]:
        out.append("| " + " | ".join(_md_cell(c) for c in r) + " |")
    out.append("")
    out.append("## Sources")
    out.append("")
    for name, url in SOURCES:
        out.append(f"- [{name}]({url})")
    out.append("")
    (DOCS / "feature_substrate_spec.md").write_text("\n".join(out))
    print(f"md  -> {DOCS / 'feature_substrate_spec.md'}")

# ---------------------------------------------------------------- pdf renderer
def _esc(s):
    return _html.escape(s, quote=False)

def render_pdf():
    ss = getSampleStyleSheet()
    ink = colors.HexColor(INK); muted = colors.HexColor(MUTED)
    H1 = ParagraphStyle("H1", parent=ss["Title"], fontName="Times-Bold", fontSize=20,
                        textColor=ink, spaceAfter=4, leading=24)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Times-Bold", fontSize=14,
                        textColor=ink, spaceBefore=15, spaceAfter=5, leading=17)
    LEAD = ParagraphStyle("LEAD", parent=ss["Normal"], fontName="Helvetica", fontSize=9.5,
                          textColor=muted, spaceAfter=3)
    BODY = ParagraphStyle("BODY", parent=ss["Normal"], fontName="Helvetica", fontSize=9.5,
                          leading=13.5, spaceAfter=6, textColor=ink)
    CELL = ParagraphStyle("CELL", parent=BODY, fontSize=7.6, leading=9.4, spaceAfter=0)
    CELLB = ParagraphStyle("CELLB", parent=CELL, fontName="Helvetica-Bold")
    TH = ParagraphStyle("TH", parent=CELL, fontName="Helvetica-Bold", textColor=colors.white)

    story = []
    story.append(Paragraph("Memory-Signal Behaviour Detection &middot; Capture substrate", LEAD))
    story.append(Paragraph("Feature-Substrate Spec", H1))
    story.append(Paragraph("Per-page memory-delta metrics &mdash; the families A&ndash;G (+H)", LEAD))
    story.append(Paragraph(f"{DATE}", LEAD))
    story.append(Spacer(1, 6))
    for p in INTRO_PARAS:
        story.append(Paragraph(_esc(p), BODY))
    story.append(Paragraph("How to read this", H2))
    story.append(Paragraph(
        "Each family is one perspective (a per-unit heat-map channel). Every family has a full table: "
        "the metric, what it <b>Measures</b> (formula / definition), and <b>What it catches</b> that the "
        "others miss. Family A is split by representation <b>Space</b>; Family B by <b>Reading</b>; the "
        "rest by their natural groups.", BODY))

    story.append(Paragraph("Phasing: Phase 1 (A&ndash;F) vs Phase 2 (G, H)", H2))
    for p in PHASING["paras"]:
        story.append(Paragraph(_esc(p), BODY))

    avail = 7.0 * inch
    for fam in FAMILIES:
        blocks = [Paragraph(f"Family {fam['id']} &mdash; {_esc(fam['title'])}", H2),
                  Paragraph(_esc(fam["intro"]), BODY)]
        if fam["grouped"]:
            cols = [fam["group_label"], "Metric", "Measures", fam["catch"]]
            widths = [0.92 * inch, 1.45 * inch, 2.05 * inch, 2.58 * inch]
            data = [[Paragraph(_esc(c), TH) for c in cols]]
            for g, metric, meas, catch in fam["rows"]:
                data.append([Paragraph(_esc(g), CELLB), Paragraph(_esc(metric), CELLB),
                             Paragraph(_esc(meas), CELL), Paragraph(_esc(catch), CELL)])
        else:
            cols = ["Metric", "Measures", fam["catch"]]
            widths = [1.7 * inch, 2.1 * inch, 3.2 * inch]
            data = [[Paragraph(_esc(c), TH) for c in cols]]
            for metric, meas, catch in fam["rows"]:
                data.append([Paragraph(_esc(metric), CELLB), Paragraph(_esc(meas), CELL),
                             Paragraph(_esc(catch), CELL)])
        tb = Table(data, colWidths=widths, repeatRows=1)
        tb.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), ink),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAF9F5")]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor(LINE)),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ]))
        # Keep the heading + intro with the first rows where possible.
        story.append(KeepTogether(blocks))
        story.append(Spacer(1, 3))
        story.append(tb)
        story.append(Spacer(1, 4))

    # synthesis
    story.append(Paragraph(_esc(SYNTHESIS["title"]), H2))
    for p in SYNTHESIS["paras"]:
        story.append(Paragraph(_esc(p), BODY))
    sc = SYNTHESIS["store_cols"]
    swid = [1.35 * inch, 1.7 * inch, 1.25 * inch, 2.7 * inch]
    sdata = [[Paragraph(_esc(c), TH) for c in sc]]
    for r in SYNTHESIS["store_rows"]:
        sdata.append([Paragraph(_esc(r[0]), CELLB), Paragraph(_esc(r[1]), CELL),
                      Paragraph(_esc(r[2]), CELLB), Paragraph(_esc(r[3]), CELL)])
    stb = Table(sdata, colWidths=swid, repeatRows=1)
    stb.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ink),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAF9F5")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor(LINE)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(Spacer(1, 3)); story.append(stb)

    story.append(Paragraph("Sources", H2))
    for name, url in SOURCES:
        story.append(Paragraph(f'<link href="{url}" color="#3F6CA8">{_esc(name)}</link>', BODY))

    def _footer(canvas, doc):
        canvas.saveState(); canvas.setFont("Helvetica", 8); canvas.setFillColor(muted)
        canvas.drawString(0.75 * inch, 0.5 * inch, "Feature-substrate spec")
        canvas.drawRightString(7.75 * inch, 0.5 * inch, f"p. {doc.page}")
        canvas.restoreState()

    SimpleDocTemplate(str(DOCS / "feature_substrate_spec.pdf"), pagesize=LETTER,
                      leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                      topMargin=0.7 * inch, bottomMargin=0.7 * inch,
                      title="Feature-Substrate Spec").build(story, onFirstPage=_footer, onLaterPages=_footer)
    print(f"pdf -> {DOCS / 'feature_substrate_spec.pdf'}")


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    render_md()
    render_pdf()
