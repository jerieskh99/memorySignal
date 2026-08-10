#!/usr/bin/env python3
"""Generate workloads_algorithms.html -- one page per campaign workload holding
the VERBATIM algorithm description from that workload's own source file.

Phase 1 (this script): faithful extraction, minimal reformatting. The authors
already wrote numbered ALGORITHM steps and ASCII diagrams in each source header;
this pulls them into one browsable document instead of 101 separate files. The
only transformation is stripping C comment syntax (/*, */, leading " * ") and
Python docstring quotes -- the text itself is unchanged, so there is no drift.

Phase 2 (later, separate): a plain-language rewrite in the explainer's voice,
kept in a different file so the verbatim source of truth stays intact here.

Regenerate after editing any workload source:
    python3 docs/make_workload_algorithms.py
"""
from __future__ import annotations

import html
import re
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    from workload_rewrites import REWRITES, GLOSSARY  # phase-2 plain rewrites
except Exception:
    REWRITES, GLOSSARY = {}, {}
from workload_introspect import extract_params, extract_edges, parse_invocation

# flags that name a scratch/output path rather than tune behavior
_PATH_FLAGS = {"--output-dir", "--sandbox-dir", "--backing-dir", "--inputs-dir",
               "--safe-root", "--child-binary", "--child-args"}


def render_params(params, cmd):
    """HTML table: each flag, its default, and the value chosen this campaign."""
    chosen = parse_invocation(cmd)
    rows = []
    listed = set()
    for flag, default in params:
        listed.add(flag)
        val = chosen.get(flag)
        if val is True:
            camp = '<span class="p-set">on</span>'
        elif val is not None:
            differs = str(val) != str(default).strip('"')
            cls = "p-set" if differs else "p-same"
            camp = f'<span class="{cls}">{html.escape(str(val))}</span>'
        else:
            camp = '<span class="p-def">(default)</span>'
        tag = ""
        if flag in _PATH_FLAGS:
            tag = '<span class="p-tag path">path</span>'
        elif val is not None:
            tag = '<span class="p-tag set">set</span>'
        rows.append(f'<tr><td><code>{html.escape(flag)}</code> {tag}</td>'
                    f'<td>{html.escape(str(default))}</td><td>{camp}</td></tr>')
    # any flag on the command line the source didn't declare (rare)
    for flag, val in chosen.items():
        if flag in listed:
            continue
        shown = "on" if val is True else html.escape(str(val))
        rows.append(f'<tr><td><code>{html.escape(flag)}</code> '
                    f'<span class="p-tag set">set</span></td>'
                    f'<td class="muted">not declared in source</td>'
                    f'<td><span class="p-set">{shown}</span></td></tr>')
    if not rows:
        return '<p class="muted">no tunable parameters found in source</p>'
    return ('<table class="params"><tr><th>Parameter</th><th>Default</th>'
            '<th>This campaign</th></tr>' + "".join(rows) + '</table>')


def render_edges(edges):
    if not edges:
        return '<p class="muted">no explicit guards or caveats found in source</p>'
    guards = [e for k, e in edges if k == "guard"]
    caveats = [e for k, e in edges if k == "caveat"]
    out = []
    if guards:
        out.append('<p class="edge-h">Guard rails (what the workload rejects or fails on):</p><ul class="edges">')
        for g in guards:
            out.append(f'<li>{html.escape(g)}</li>')
        out.append('</ul>')
    if caveats:
        out.append('<p class="edge-h">Author caveats (from the source header):</p><ul class="edges cav">')
        for c in caveats:
            out.append(f'<li>{html.escape(c)}</li>')
        out.append('</ul>')
    return "".join(out)
CAPTURE_ROOT = HERE.parent
REPO_ROOT = CAPTURE_ROOT.parent.parent
STEPS = CAPTURE_ROOT / "plan07_campaign" / "full_campaign_steps.txt"
SRC_ROOT = REPO_ROOT / "VM_executables_phase2"
OUT = HERE / "workloads_algorithms.html"

FAMILY_ORDER = ["cpu", "cache", "mem", "thread", "io", "mixed",
                "app", "sandbox", "kernel", "mp"]
FAMILY_LABEL = {
    "cpu": "CPU — pure computation, near-silent memory",
    "cache": "CACHE — cache-boundary access patterns",
    "mem": "MEM — deliberate large-region memory writes",
    "thread": "THREAD — concurrency, small footprint",
    "io": "IO — disk-facing reads and writes",
    "mixed": "MIXED — blended CPU / memory / I-O",
    "app": "APP — realistic application-shaped work",
    "sandbox": "SANDBOX — ransomware-behavior simulations",
    "kernel": "KERNEL — Berkeley 'dwarf' compute motifs (64)",
    "mp": "METHODOLOGY — analysis steps, not workloads",
}


_TERM_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")


def render_terms(text: str) -> str:
    """Turn [[term]] / [[term|inline def]] into hover-glossary spans.
    Everything outside the markers is HTML-escaped; the definition comes from
    GLOSSARY unless given inline."""
    out = []
    last = 0
    for m in _TERM_RE.finditer(text):
        out.append(html.escape(text[last:m.start()]))
        term = m.group(1)
        definition = m.group(2) or GLOSSARY.get(term, "")
        out.append(
            f'<span class="term" tabindex="0" data-def="{html.escape(definition, quote=True)}">'
            f'{html.escape(term)}</span>'
        )
        last = m.end()
    out.append(html.escape(text[last:]))
    return "".join(out)


def family_of(name: str) -> str:
    stem = name[:-3] if name.endswith(".py") else name
    return stem.split("_", 1)[0] if "_" in stem else stem


def find_src(name: str) -> Path | None:
    stem = name[:-3] if name.endswith(".py") else name
    for ext in (".c", ".py"):
        hits = list(SRC_ROOT.rglob(stem + ext))
        if hits:
            return hits[0]
    return None


def extract_header(src: Path) -> str:
    """Return the leading block comment (C) or module docstring (Python),
    with only the comment syntax stripped -- content byte-for-byte otherwise."""
    txt = src.read_text(errors="ignore")
    if src.suffix == ".c":
        m = re.match(r"\s*/\*(.*?)\*/", txt, re.S)
        if not m:
            return ""
        body = m.group(1)
        lines = []
        for ln in body.splitlines():
            # strip a single leading " * " / " *" continuation marker, nothing else
            stripped = re.sub(r"^\s?\*\s?", "", ln)
            lines.append(stripped.rstrip())
        # trim leading/trailing blank lines only
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        return "\n".join(lines)
    else:
        m = re.search(r'"""(.*?)"""', txt, re.S)
        if not m:
            return ""
        return m.group(1).strip("\n")


def parse_steps():
    """Yield (index, program_name, full_command) for each campaign step."""
    out = []
    for i, ln in enumerate(STEPS.read_text().splitlines()):
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        toks = shlex.split(s)
        prog = None
        for t in toks:
            if "/bin/" in t or "/app_realistic/" in t or "/methodology/" in t:
                prog = Path(t).name
                break
        if prog:
            out.append((len(out) + 1, prog, s))
    return out


def main():
    steps = parse_steps()

    # group unique programs by family, preserving first campaign appearance
    seen = {}
    for idx, prog, cmd in steps:
        if prog not in seen:
            seen[prog] = {"first_step": idx, "cmd": cmd,
                          "steps": [], "family": family_of(prog)}
        seen[prog]["steps"].append(idx)
    for prog, info in seen.items():
        info["src"] = find_src(prog)
        info["algo"] = extract_header(info["src"]) if info["src"] else ""
        info["params"] = extract_params(info["src"]) if info["src"] else []
        info["edges"] = extract_edges(info["src"]) if info["src"] else []

    by_family = {f: [] for f in FAMILY_ORDER}
    for prog, info in sorted(seen.items(), key=lambda kv: kv[1]["first_step"]):
        by_family.setdefault(info["family"], []).append((prog, info))

    def rel(p: Path) -> str:
        try:
            return str(p.relative_to(REPO_ROOT))
        except ValueError:
            return str(p)

    # ---- build HTML ----
    parts = []
    parts.append(HEAD)
    parts.append('<main>\n<header>')
    parts.append('<h1>Workload Algorithms — Explained + Verbatim</h1>')
    n_rw = sum(1 for p in seen if p in REWRITES)
    parts.append('<p class="lede">One page per campaign workload. Each opens with a '
                 '<strong>plain-language explanation</strong> of the algorithm, then three '
                 'expandable tabs: <strong>Parameters</strong> (every flag, its default, and '
                 'the value this campaign chose), <strong>Edge cases &amp; limits</strong> '
                 '(the guard rails and caveats the workload states in its own source), and '
                 '<strong>the verbatim source</strong> it is all based on. Only the plain '
                 'explanation is authored; the tabs are extracted from source, so nothing '
                 'drifts.</p>')
    parts.append(f'<p class="glossary-hint">{len(seen)} workloads across '
                 f'{sum(1 for f in by_family if by_family[f])} families · '
                 f'{n_rw} with plain rewrites · {len(GLOSSARY)} glossary terms (hover the '
                 'underlined words).</p>')
    parts.append('</header>')

    # TOC
    parts.append('<section id="toc-section"><h2>Contents</h2><div class="toc">')
    for fam in FAMILY_ORDER:
        items = by_family.get(fam, [])
        if not items:
            continue
        parts.append(f'<p class="toc-fam"><strong>{html.escape(FAMILY_LABEL.get(fam, fam.upper()))}</strong> '
                     f'<span class="toc-count">{len(items)}</span></p><ul>')
        for prog, info in items:
            parts.append(f'<li><a href="#{html.escape(prog)}">{html.escape(prog)}</a></li>')
        parts.append('</ul>')
    parts.append('</div></section>')

    # sections
    for fam in FAMILY_ORDER:
        items = by_family.get(fam, [])
        if not items:
            continue
        parts.append(f'<section><h2 id="fam-{fam}">{html.escape(FAMILY_LABEL.get(fam, fam.upper()))}</h2>')
        for prog, info in items:
            steps_str = ", ".join(str(s) for s in info["steps"])
            src_disp = rel(info["src"]) if info["src"] else "(source not found)"
            algo = info["algo"] or "(no algorithm header found in source)"
            # rewrite keys are stem-only; the campaign name may carry a .py
            rw = REWRITES.get(prog) or REWRITES.get(prog[:-3] if prog.endswith(".py") else prog)
            parts.append(f'<details id="{html.escape(prog)}">')
            parts.append(f'<summary>{html.escape(prog)} '
                         f'<span class="step-badge">step {steps_str}</span></summary>')
            parts.append('<div class="body">')
            # Phase-2 plain rewrite on top (with hover glossary).
            if rw:
                parts.append('<div class="plain">')
                parts.append(f'<p class="plain-text">{render_terms(rw["plain"])}</p>')
                parts.append(f'<p class="signal"><span class="signal-tag">memory signal</span> '
                             f'{render_terms(rw["signal"])}</p>')
                parts.append('</div>')
            else:
                parts.append('<p class="plain-text muted">(no plain-language rewrite yet)</p>')
            # Tab: parameters (chosen vs default).
            nset = sum(1 for f, _ in info["params"]
                       if f in parse_invocation(info["cmd"]))
            parts.append('<details class="tab tab-params">')
            parts.append(f'<summary>Parameters <span class="tab-hint">'
                         f'{len(info["params"])} flags, {nset} set this campaign</span></summary>')
            parts.append('<div class="body">')
            parts.append(render_params(info["params"], info["cmd"]))
            parts.append('</div></details>')

            # Tab: edge cases & limits.
            ne = len(info["edges"])
            parts.append('<details class="tab tab-edges">')
            parts.append(f'<summary>Edge cases &amp; limits <span class="tab-hint">'
                         f'{ne} from source</span></summary>')
            parts.append('<div class="body">')
            parts.append(render_edges(info["edges"]))
            parts.append('</div></details>')

            # Tab: verbatim source, collapsed beneath the plain version.
            parts.append('<details class="tab verbatim">')
            parts.append('<summary>Show original source (verbatim)</summary>')
            parts.append('<div class="body">')
            parts.append(f'<p class="srcpath">source: <code>{html.escape(src_disp)}</code></p>')
            parts.append(f'<pre class="algo">{html.escape(algo)}</pre>')
            parts.append('</div></details>')
            parts.append('<p class="cmd-label">campaign invocation:</p>')
            parts.append(f'<pre class="cmd">{html.escape(info["cmd"])}</pre>')
            parts.append('</div></details>')
        parts.append('</section>')

    parts.append('</main>\n</body>\n</html>')
    OUT.write_text("\n".join(parts))
    print(f"wrote {OUT}")
    print(f"  {len(seen)} workloads, "
          f"{sum(1 for _,i in seen.items() if i['algo'])} with algorithm text")


HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Workload Algorithms — Verbatim Source — VM_Capture_QEMU</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{
  --bg:#fafafa;--fg:#1c1c1c;--muted:#5a5a5a;--accent:#0a5aa0;
  --card:#fff;--border:#d8d8d8;--good:#2c7a3a;--warn:#a3540a;--bad:#a01b2a;
  --callout:#f1f6fb;--codebg:#f4f4f4;
}
html,body{margin:0;padding:0;background:var(--bg);color:var(--fg);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
  line-height:1.55;font-size:16px;}
main{max-width:1000px;margin:0 auto;padding:32px 24px 80px;}
h1{font-size:1.9rem;margin:0 0 8px;}
h2{margin-top:48px;padding-bottom:8px;border-bottom:2px solid var(--border);font-size:1.35rem;}
header p.lede{color:var(--muted);margin-top:0;font-size:1.05rem;}
a.anchor,.toc a{color:var(--accent);text-decoration:none;}
a.anchor:hover,.toc a:hover{text-decoration:underline;}
.glossary-hint{display:inline-block;font-size:0.9rem;color:var(--muted);
  background:var(--callout);border:1px solid var(--border);border-radius:20px;
  padding:5px 14px;margin-top:8px;}
.toc{border:1px solid var(--border);border-radius:8px;background:var(--card);
  padding:8px 20px 16px;column-count:2;column-gap:28px;}
@media(max-width:640px){.toc{column-count:1;}}
.toc-fam{margin:14px 0 2px;break-after:avoid;}
.toc-count{color:var(--muted);font-weight:400;font-size:0.82rem;}
.toc ul{margin:2px 0 8px;padding-left:18px;}
.toc li{font-size:0.9rem;list-style:none;}
.toc li a{font-family:"SFMono-Regular",Consolas,monospace;font-size:0.85rem;}
details{background:var(--card);border:1px solid var(--border);border-radius:8px;margin:8px 0;}
details summary{cursor:pointer;padding:11px 16px;font-weight:600;list-style:none;
  font-family:"SFMono-Regular",Consolas,monospace;font-size:0.95rem;}
details summary::-webkit-details-marker{display:none;}
details summary::before{content:"▸ ";color:var(--muted);}
details[open] summary::before{content:"▾ ";}
details[open] summary{border-bottom:1px solid var(--border);}
details .body{padding:8px 18px 18px;}
.step-badge{float:right;font-family:-apple-system,sans-serif;font-weight:400;
  font-size:0.75rem;color:var(--muted);background:var(--callout);
  border:1px solid var(--border);border-radius:10px;padding:1px 9px;}
.srcpath{color:var(--muted);font-size:0.85rem;margin:6px 0 10px;}
code{font-family:"SFMono-Regular",Consolas,"Liberation Mono",monospace;font-size:0.9em;}
pre{font-family:"SFMono-Regular",Consolas,"Liberation Mono",monospace;
  background:var(--codebg);border:1px solid var(--border);border-radius:6px;
  padding:12px 14px;overflow-x:auto;font-size:0.85rem;line-height:1.45;}
pre.algo{white-space:pre;}
.cmd-label{margin:14px 0 4px;font-size:0.8rem;color:var(--muted);
  text-transform:uppercase;letter-spacing:.04em;}
pre.cmd{background:#1c2733;color:#e8eef4;border-color:#2a3947;white-space:pre-wrap;
  word-break:break-all;}
.plain{background:#f7fbff;border:1px solid #d5e3f0;border-left:4px solid var(--accent);
  border-radius:6px;padding:12px 16px;margin:4px 0 12px;}
.plain-text{margin:0 0 8px;font-size:1rem;line-height:1.6;}
.plain-text.muted{color:var(--muted);font-style:italic;}
.signal{margin:0;font-size:0.92rem;color:#333;line-height:1.55;}
.signal-tag{display:inline-block;font-size:0.68rem;font-weight:700;letter-spacing:.04em;
  text-transform:uppercase;color:var(--good);background:#eaf6ee;border:1px solid #cfe6d6;
  border-radius:10px;padding:1px 8px;margin-right:6px;vertical-align:1px;}
details.tab{background:#fbfbfb;border:1px solid var(--border);margin:6px 0 0;}
details.tab>summary{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  font-weight:600;font-size:0.85rem;color:#333;padding:8px 14px;}
details.tab .body{padding:6px 14px 14px;}
details.tab-params>summary{border-left:3px solid var(--accent);}
details.tab-edges>summary{border-left:3px solid var(--warn);}
details.verbatim{background:#fbfbfb;border:1px dashed var(--border);}
details.verbatim>summary{font-weight:500;color:var(--muted);border-left:3px solid var(--border);}
.tab-hint{float:right;font-weight:400;font-size:0.75rem;color:var(--muted);}
table.params{width:100%;font-size:0.86rem;}
table.params th{background:#f0f4f8;font-size:0.78rem;text-transform:uppercase;letter-spacing:.03em;}
table.params td{padding:5px 9px;vertical-align:top;}
.p-set{color:var(--good);font-weight:700;}
.p-same{color:#333;}
.p-def{color:var(--muted);}
.p-tag{display:inline-block;font-size:0.62rem;font-weight:700;letter-spacing:.03em;
  padding:0 6px;border-radius:8px;vertical-align:1px;margin-left:4px;}
.p-tag.set{background:#eaf6ee;color:var(--good);}
.p-tag.path{background:#f0eefa;color:#5a3fa0;}
.edge-h{font-size:0.85rem;font-weight:600;margin:8px 0 4px;color:#333;}
ul.edges{margin:2px 0 10px;padding-left:20px;}
ul.edges li{font-size:0.85rem;line-height:1.5;margin:3px 0;}
ul.edges.cav li{color:#555;font-style:italic;}
.muted{color:var(--muted);}
.term{border-bottom:1px dotted var(--accent);cursor:help;position:relative;}
.term::after{content:attr(data-def);position:absolute;left:0;bottom:140%;
  width:max-content;max-width:300px;background:#1c1c1c;color:#fafafa;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  font-size:0.82rem;font-weight:400;line-height:1.45;padding:9px 12px;border-radius:7px;
  box-shadow:0 6px 20px rgba(0,0,0,.28);z-index:80;pointer-events:none;
  opacity:0;transform:translateY(5px);transition:opacity .12s,transform .12s;}
.term:hover::after,.term:focus::after{opacity:1;transform:translateY(0);}
.term:focus{outline:2px solid var(--accent);outline-offset:2px;border-radius:2px;}
</style>
</head>
<body>
"""

if __name__ == "__main__":
    main()
