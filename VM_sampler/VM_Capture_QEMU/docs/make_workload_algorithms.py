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
from pathlib import Path

HERE = Path(__file__).resolve().parent
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
    parts.append('<h1>Workload Algorithms — Verbatim Source</h1>')
    parts.append('<p class="lede">One page per campaign workload, showing the '
                 'algorithm exactly as written in that workload\'s own source '
                 'file. Nothing here is paraphrased — only the C comment markers '
                 'and Python docstring quotes were removed. This is the source of '
                 'truth; a plain-language rewrite lives in '
                 '<a class="anchor" href="./workloads_explained.html">workloads_explained.html</a>.</p>')
    parts.append(f'<p class="glossary-hint">{len(seen)} workloads across '
                 f'{sum(1 for f in by_family if by_family[f])} families. '
                 'Each entry links back to its source path.</p>')
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
            parts.append(f'<details id="{html.escape(prog)}">')
            parts.append(f'<summary>{html.escape(prog)} '
                         f'<span class="step-badge">step {steps_str}</span></summary>')
            parts.append('<div class="body">')
            parts.append(f'<p class="srcpath">source: <code>{html.escape(src_disp)}</code></p>')
            parts.append(f'<pre class="algo">{html.escape(algo)}</pre>')
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
</style>
</head>
<body>
"""

if __name__ == "__main__":
    main()
