#!/usr/bin/env python3
"""Build analysis_console.html by injecting the pipeline's real data into the template.

The UI (analysis_console.template.html) hardcodes NOTHING about the channels,
which of them are live, the recordings, the module ports, or the known-issue
history. This script derives all of it from the pipeline and the filesystem
and injects it, so the console can never silently drift from the code and
never renders data it invented. Same contract as plan07_campaign/ui/build_console.py.

Sources of truth:
  live_delta_calc_modular/src/metrics/*.rs   -> ROSTER   (plan10_analysis/channel_roster.py)
  plan07_campaign/subset_run.py               -> ROSTER   (family / submodule, by name)
  <trace root>/<family>/<workload>/<variant>/rep* -> MANIFEST (plan10_analysis/corpus_manifest.py)
  plan05 artifacts, plan03/04 sources          -> ISSUES   (plan10_analysis/known_issues.py)
  plan08_b1/b1_features.py and friends         -> MODULES  (plan10_analysis/modules.py)
  config_qemu_upc.json                         -> CONFIG   (substrateSpeed, intervalMsec, page count default)

Refuses to build when the corpus is missing or empty: a console that renders
sample recordings is worse than one that does not start. Refuses when the
roster cannot be derived, when the template mentions any channel or workload
by name (that would be a copy to drift from), or when the output carries
network code. External resource references (the font stylesheet) are reported,
not stripped -- see the report.

Run after any change to the sources above:
    python3 plan10_analysis/ui/build_analysis_console.py [--root TRACE_ROOT] [--metrics-root M]
-> rewrites plan10_analysis/ui/analysis_console.html
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent            # .../plan10_analysis/ui
PKG = HERE.parent                                 # .../plan10_analysis
QEMU_DIR = PKG.parent                             # .../VM_Capture_QEMU
sys.path.insert(0, str(QEMU_DIR))

from plan10_analysis import channel_roster, corpus_manifest, known_issues  # noqa: E402
from plan10_analysis.modules import build_modules                          # noqa: E402
from plan10_analysis.scheme import load_config, make_examples              # noqa: E402

TEMPLATE = HERE / "analysis_console.template.html"
OUT = HERE / "analysis_console.html"
OUT_SERVED = HERE / "analysis_console.served.html"
MARKER = "/*@@GENERATED_DATA@@*/"
# The bridge client (source / run / results) lives between these markers, in HTML and in JS.
# The static build STRIPS it: analysis_console.html carries zero network code. --served keeps it.
SERVED_MARKERS = (("<!--@@SERVED_ONLY_START@@-->", "<!--@@SERVED_ONLY_END@@-->"), ("/*@@SERVED_ONLY_START@@*/", "/*@@SERVED_ONLY_END@@*/"))


def strip_served(html: str) -> str:
    for a, b in SERVED_MARKERS:
        while a in html and b in html:
            i = html.index(a)
            j = html.index(b) + len(b)
            html = html[:i] + html[j:]
    return html
NETWORK_CODE = re.compile(r"\b(fetch\s*\(|XMLHttpRequest|WebSocket|EventSource|sendBeacon|importScripts)\b")
EXTERNAL_REF = re.compile(r"https?://[^\s\"'<>)]+")


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=QEMU_DIR, capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def js_const(name: str, value) -> str:
    return f"const {name} = " + json.dumps(value, ensure_ascii=False, separators=(",", ":")) + ";"


def build(root: Path | None, metrics_root: Path | None, out: Path, manifest_path: Path | None, served: bool = False) -> int:
    template = TEMPLATE.read_text()
    if MARKER not in template:
        sys.exit(f"[build_analysis_console] marker {MARKER} not found in {TEMPLATE.name}")
    if not served:
        template = strip_served(template)

    try:
        roster = channel_roster.build_roster()
    except channel_roster.RosterError as e:
        sys.exit(f"[build_analysis_console] REFUSED: channel roster could not be derived: {e}")

    try:
        if manifest_path:
            manifest = json.loads(manifest_path.read_text())
        else:
            manifest = corpus_manifest.scan(root or corpus_manifest.default_root(), metrics_root)
    except corpus_manifest.CorpusMissing as e:
        sys.exit(f"[build_analysis_console] REFUSED: no corpus, and this console does not render sample data. {e}")
    if manifest.get("n_recordings", 0) == 0:
        sys.exit("[build_analysis_console] REFUSED: manifest holds zero recordings")

    issues = known_issues.build_registry()
    modules = build_modules()
    config = load_config()
    examples = make_examples(manifest)

    # no-drift: the template must not name any channel or any workload
    names = [c["name"] for c in roster["channels"]]
    leaked = [n for n in names if re.search(r"\b" + re.escape(n) + r"\b", template)]
    if leaked:
        sys.exit(f"[build_analysis_console] REFUSED: template names channels (a copy to drift from): {leaked}")
    wls = sorted({r["workload"] for r in manifest["recordings"]})
    leaked = [w for w in wls if w in template]
    if leaked:
        sys.exit(f"[build_analysis_console] REFUSED: template names workloads: {leaked}")

    build_meta = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "template_sha256_16": hashlib.sha256(template.encode()).hexdigest()[:16],
        "roster_derivation": roster["derivation"],
        "manifest_root": manifest["root"],
        "manifest_scanned_at": manifest["scanned_at"],
        "n_recordings": manifest["n_recordings"],
        "issues_verified": sum(1 for i in issues if i["verified"]),
        "issues_unverified": [i["id"] for i in issues if not i["verified"]],
        "external_refs": sorted(set(EXTERNAL_REF.findall(template))),
    }

    data = "\n".join([
        js_const("ROSTER", roster),
        js_const("MANIFEST", manifest),
        js_const("ISSUES", issues),
        js_const("MODULES", modules),
        js_const("CONFIG", config),
        js_const("EXAMPLES", examples),
        js_const("BUILD", build_meta),
    ])
    html = template.replace(MARKER, data)

    if not served and NETWORK_CODE.search(html):
        sys.exit("[build_analysis_console] REFUSED: static output contains network code")

    out.write_text(html)
    print(f"[build_analysis_console] wrote {out.name} ({len(html) // 1024} KB) {'(served: bridge client kept)' if served else '(static: bridge client stripped)'}")
    print(f"  ROSTER: {roster['n_total']} columns; computed per level "
          + ", ".join(f"{l['speed']}:{l['computed']}" for l in roster["levels"]))
    print(f"  MANIFEST: {manifest['n_recordings']} recordings under {manifest['root']} "
          f"({manifest['n_with_chain']} chains, {manifest['n_with_substrate_csv']} substrate CSVs, "
          f"{manifest['n_workloads']} workloads)")
    print(f"  ISSUES: {build_meta['issues_verified']} verified, unverified: {build_meta['issues_unverified']}")
    print(f"  MODULES: {len(modules['modules'])}   CONFIG: speed {config['substrateSpeed']}, "
          f"iv {config['intervalMsec']} ms, n_pages default {config['n_pages_default']}")
    print(f"  network code: none   external references: {build_meta['external_refs'] or 'none'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=None, help="trace root (default: console.sh TRACES_LOCAL_DIR)")
    ap.add_argument("--metrics-root", type=Path, default=None, help="optional root holding substrate CSVs")
    ap.add_argument("--manifest", type=Path, default=None, help="use an existing manifest JSON instead of scanning")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--served", action="store_true", help="keep the bridge client (analysis_console.served.html, for analysis_bridge.py)")
    a = ap.parse_args()
    out = a.out or (OUT_SERVED if a.served else OUT)
    return build(a.root, a.metrics_root, out, a.manifest, a.served)


if __name__ == "__main__":
    raise SystemExit(main())
