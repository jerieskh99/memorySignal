#!/usr/bin/env python3
"""scheme.py -- the scheme format and its validator, outside the browser.

A scheme is the graph the console draws: modules with parameters, pipes
between typed ports, and the list of warnings the author knowingly accepted.

    {"schema": "plan10.scheme.v1", "label": "...",
     "nodes": [{"id": "n1", "module": "cells", "params": {...}, "x": 0, "y": 0}, ...],
     "pipes": [{"from": ["n1", "cells"], "to": ["n2", "cells"]}, ...],
     "acknowledged": [{"id": "<constraint id>", "at": "<iso>", "note": "..."}]}

This module is the rule set the console's JavaScript mirrors, kept in Python
so the same scheme is refused for the same reasons when a runner reads it.
The rules and their severities are the ones typed in
plan10_analysis_console_UX.md section 13: `hard` refuses, `soft` warns and
must be acknowledged, `note` informs. The one severity this module adds that
the document does not type is `no_substrate`, below, and it is marked as
such.

Exit codes for `validate`: 0 valid, 1 at least one hard issue, 2 no hard
issue but at least one unacknowledged soft one.

Run:  python3 plan10_analysis/scheme.py validate scheme.json [--roster R] [--manifest M]
      python3 plan10_analysis/scheme.py examples --out-dir D [--manifest M]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
QEMU_DIR = HERE.parent
if str(QEMU_DIR) not in sys.path:
    sys.path.insert(0, str(QEMU_DIR))

from plan10_analysis import channel_roster, corpus_manifest  # noqa: E402
from plan10_analysis.modules import build_modules  # noqa: E402


def stages_wavelet_max_level(fam: str, w: int) -> int:
    """runner.stages.wavelet_max_level, imported lazily: scheme.py must stay numpy-free."""
    try:
        import pywt  # type: ignore
    except ImportError:
        return -1
    try:
        return int(pywt.dwt_max_level(w, pywt.Wavelet(fam)))
    except (ValueError, KeyError):
        return 0

SCHEMA = "plan10.scheme.v1"
CONFIG_PATH = QEMU_DIR / "config_qemu_upc.json"


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def load_config(path: Path = CONFIG_PATH) -> dict:
    c = json.loads(path.read_text())
    page = int(c.get("rawRetention", {}).get("rawBuild", {}).get("pageSize", 4096))
    return {
        "substrateSpeed": int(c["substrateSpeed"]),
        "intervalMsec": int(c["intervalMsec"]),
        "ramSizeMb": int(c["ramSizeMb"]),
        "pageSize": page,
        # the page-count decision is open (UX section 14); this is the config-derived default,
        # used only for a recording whose own n_pages is null
        "n_pages_default": int(c["ramSizeMb"]) * 1024 * 1024 // page,
        "source": str(path.relative_to(QEMU_DIR.parent.parent)),
    }


class Context:
    def __init__(self, roster: dict, manifest: dict, modules: dict, config: dict):
        self.roster = roster
        self.manifest = manifest
        self.modules = modules
        self.config = config
        self.mod_by_id = {m["id"]: m for m in modules["modules"]}
        self.rec_by_id = {r["id"]: r for r in manifest["recordings"]}
        self.drop_at = {c["name"]: c["drop_at"] for c in roster["channels"]}
        self.group_of = {c["name"]: c["group"] for c in roster["channels"]}

    @classmethod
    def default(cls, manifest_path: Path | None = None, roster_path: Path | None = None,
                modules_path: Path | None = None, manifest_root: Path | None = None) -> "Context":
        roster = json.loads(roster_path.read_text()) if roster_path else channel_roster.build_roster()
        if manifest_path:
            manifest = json.loads(manifest_path.read_text())
        else:
            manifest = corpus_manifest.scan(manifest_root or corpus_manifest.default_root())
        modules = json.loads(modules_path.read_text()) if modules_path else build_modules()
        return cls(roster, manifest, modules, load_config())

    def rec_speed(self, rec: dict) -> tuple[int, str]:
        if rec.get("speed") is not None:
            return int(rec["speed"]), "recorded"
        return self.config["substrateSpeed"], "assumed from config"

    def rec_n_pages(self, rec: dict) -> int:
        return int(rec["n_pages"]) if rec.get("n_pages") else self.config["n_pages_default"]


# ---------------------------------------------------------------------------
# graph helpers
# ---------------------------------------------------------------------------

class Graph:
    def __init__(self, scheme: dict, ctx: Context):
        self.s = scheme
        self.ctx = ctx
        self.nodes = {n["id"]: n for n in scheme.get("nodes", [])}
        self.pipes = list(scheme.get("pipes", []))
        self.acks = {a["id"] for a in scheme.get("acknowledged", []) if isinstance(a, dict) and "id" in a}

    def mod(self, nid: str) -> dict | None:
        n = self.nodes.get(nid)
        return self.ctx.mod_by_id.get(n["module"]) if n else None

    def in_pipes(self, nid: str, port: str | None = None) -> list[dict]:
        return [p for p in self.pipes if p["to"][0] == nid and (port is None or p["to"][1] == port)]

    def out_pipes(self, nid: str) -> list[dict]:
        return [p for p in self.pipes if p["from"][0] == nid]

    def port(self, nid: str, k: str, out: bool) -> dict | None:
        m = self.mod(nid)
        if not m:
            return None
        return next((p for p in (m["outputs"] if out else m["inputs"]) if p["k"] == k), None)

    def reaches(self, a: str, b: str) -> bool:
        seen, st = set(), [a]
        while st:
            x = st.pop()
            if x == b:
                return True
            if x in seen:
                continue
            seen.add(x)
            st.extend(p["to"][0] for p in self.out_pipes(x))
        return False

    def params(self, nid: str) -> dict:
        m = self.mod(nid)
        p = {q["k"]: q["default"] for q in m["params"]} if m else {}
        p.update(self.nodes[nid].get("params", {}))
        return p


# ---------------------------------------------------------------------------
# descriptor propagation: what flows through each pipe
# ---------------------------------------------------------------------------

def descriptor(g: Graph, nid: str, memo: dict) -> dict | None:
    if nid in memo:
        return memo[nid]
    n = g.nodes[nid]
    p = g.params(nid)
    ctx = g.ctx

    def up(k):
        e = g.in_pipes(nid, k)
        return descriptor(g, e[0]["from"][0], memo) if e else None

    d: dict | None
    mod = n["module"]
    if mod == "cells":
        recs = [ctx.rec_by_id[r] for r in p.get("sel", []) if r in ctx.rec_by_id]
        kept = [r for r in recs if r["n_pairs"] >= int(p.get("min_pairs", 0))]
        mp = int(p.get("max_pairs") or 0) or None
        d = {"type": "cells", "recs": kept, "dropped_short": len(recs) - len(kept),
             "nmin": min((min(r["n_pairs"], mp) if mp else r["n_pairs"] for r in kept), default=0),
             "max_pairs": mp, "workloads": sorted({r["workload"] for r in kept})}
    elif mod == "channels":
        u = up("cells")
        d = {"type": "field", "channels": list(p.get("chans", [])), "complex": False, "axis": "page",
             "recs": u["recs"] if u else [], "nmin": u["nmin"] if u else 0, "max_pairs": u.get("max_pairs") if u else None,
             "workloads": u["workloads"] if u else []}
    elif mod in ("single", "vectorize"):
        u = up("in")
        d = dict(u, type="field") if u else None
    elif mod == "complex":
        a, b = up("mag"), up("dir")
        base = a or b or {"recs": [], "nmin": 0, "workloads": []}
        d = {"type": "complex", "complex": True, "axis": "page", "phase": p.get("phase") or "",
             "channels": [*(a["channels"] if a else []), *(b["channels"] if b else [])],
             "mag": a, "dir": b, "recs": base["recs"], "nmin": min(a["nmin"] if a else 10**9, b["nmin"] if b else 10**9),
             "workloads": base["workloads"]}
    elif mod == "collapse":
        u = up("in")
        d = dict(u, type="series", axis="collapsed") if u else None
    elif mod == "block":
        u = up("in")
        # n_blocks rides along so it survives the Collapse that must follow: the tile count
        # is windows x blocks, and Collapse used to erase the marker the estimate looked for.
        # n_pages is the config default because no recording records its own (UX 14.1 item 7).
        wp_, hp_ = int(p["wp"]), int(p["hp"])
        npg = ctx.config["n_pages_default"]
        d = dict(u, type="field", axis="blocked", wp=wp_, hp=hp_,
                 n_blocks=(0 if npg < wp_ else (npg - wp_) // hp_ + 1)) if u else None
    elif mod == "window":
        u = up("in")
        d = dict(u, type="tiles", w=int(p["w"]), h=int(p["h"]), taper=p["taper"], edge=p["edge"]) if u else None
    elif mod == "baseline":
        u = up("in")
        d = {"type": "reference", "mode": p.get("mode"), "fitted": u is not None,
             "kind": "envelope" if p.get("mode") == "benign" else "plv",
             "src": (u or {}).get("type"), "complex": (u or {}).get("complex", False),
             "names": (u or {}).get("names")}
    elif mod == "deviation":
        u = up("in")
        d = {"type": "features", "names": ["dev_n_outside", "dev_frac_outside", "dev_max_excess", "dev_total_excess"],
             "up": u, "w": (u or {}).get("w")}
    elif mod == "concat":
        es = g.in_pipes(nid, "in")
        d = {"type": "features", "n": len(es)}
    elif mod == "write":
        d = {"type": "sink"}
    else:  # lenses
        u = up("in")
        names = list(p.get("feats", [])) if mod in ("stats", "deep") else [mod]
        d = {"type": "features", "names": names, "up": u, "w": (u or {}).get("w")}
    memo[nid] = d
    return d


# ---------------------------------------------------------------------------
# constraints
# ---------------------------------------------------------------------------

def _issue(node, sev, msg, src, id=None, fix=None):
    out = {"node": node, "sev": sev, "msg": msg, "src": src}
    if id:
        out["id"] = id
    if fix:
        out["fix"] = fix
    return out


def node_constraints(g: Graph, nid: str, memo: dict) -> list[dict]:
    n = g.nodes[nid]
    mod = n["module"]
    m = g.mod(nid)
    p = g.params(nid)
    ctx = g.ctx
    out: list[dict] = []

    def up(k):
        e = g.in_pipes(nid, k)
        return descriptor(g, e[0]["from"][0], memo) if e else None

    for i in m["inputs"]:
        if i["req"] and not g.in_pipes(nid, i["k"]):
            out.append(_issue(nid, "hard", f"input '{i['lab']}' is not connected", "required port"))

    if mod == "cells":
        sel = [r for r in p.get("sel", []) if r in ctx.rec_by_id]
        unknown = [r for r in p.get("sel", []) if r not in ctx.rec_by_id]
        if unknown:
            out.append(_issue(nid, "hard", f"{len(unknown)} selected recording(s) are not in the manifest: {unknown[:3]}",
                              "corpus_manifest.py"))
        recs = [ctx.rec_by_id[r] for r in sel]
        kept = [r for r in recs if r["n_pairs"] >= int(p.get("min_pairs", 0))]
        if not sel:
            out.append(_issue(nid, "hard", "no recordings selected", "nothing to read"))
        elif not kept:
            out.append(_issue(nid, "hard", f"every selected recording has fewer than min_pairs={p.get('min_pairs')} pairs", "plan08 --min-pairs"))
        elif len(kept) < len(recs):
            out.append(_issue(nid, "note", f"{len(recs) - len(kept)} recording(s) below min_pairs={p.get('min_pairs')} are excluded", "plan08 --min-pairs"))
        no_chain = [r["id"] for r in kept if not r["has"]["chain"]]
        if no_chain:
            out.append(_issue(nid, "hard", f"{len(no_chain)} selected recording(s) hold no usable chain (empty or base only)", "corpus_manifest.py status"))
        ivs = {r["iv_ms"] for r in kept if r.get("iv_ms") is not None}
        if len(ivs) > 1:
            out.append(_issue(nid, "soft", f"selected recordings mix sampling intervals ({sorted(ivs)} ms); one window then spans different wall-clock times across them",
                              "plan10 UX section 11.2", id="mixed_iv"))
        if kept and all(r.get("iv_ms") is None for r in kept):
            out.append(_issue(nid, "note", f"sampling interval is unrecorded per recording; the time-span readout assumes config intervalMsec={ctx.config['intervalMsec']}",
                              "corpus_manifest.py iv_source"))

    elif mod == "channels":
        chans = list(p.get("chans", []))
        u = up("cells")
        if not chans:
            out.append(_issue(nid, "hard", "no channels selected", "nothing to extract"))
        unknown = [c for c in chans if c not in ctx.drop_at]
        if unknown:
            out.append(_issue(nid, "hard", f"not a differ column: {unknown}", "channel_roster.py"))
        recs = u["recs"] if u else []
        if recs and chans:
            dead_by_speed: dict[str, set] = {}
            assumed = False
            for r in recs:
                sp, how = ctx.rec_speed(r)
                assumed |= how != "recorded"
                for c in chans:
                    da = ctx.drop_at.get(c)
                    if da is not None and da <= sp:
                        dead_by_speed.setdefault(c, set()).add(sp)
            if dead_by_speed:
                lv = sorted({s for ss in dead_by_speed.values() for s in ss})
                out.append(_issue(nid, "hard",
                                  f"{sorted(dead_by_speed)} emit zero at substrateSpeed {lv} "
                                  f"({'speed assumed from config, unrecorded per recording' if assumed else 'recorded speed'})",
                                  "channel_roster.py drop_at; config_qemu_upc.json substrateSpeed"))
            no_csv = [r for r in recs if not r["has"]["substrate_csv"]]
            if no_csv:
                out.append(_issue(nid, "soft",
                                  f"{len(no_csv)} of {len(recs)} selected recording(s) carry a raw chain only, no substrate CSV; "
                                  f"the runner will reconstruct each chain and re-run the differ at the run's speed "
                                  f"(about 2 s per pair at speed 2 on a 1 GiB dump, plus reconstruction)",
                                  "corpus_manifest.py has.substrate_csv; runner/extract.py (severity: plan10 UX section 14.1 item 5)",
                                  id="no_substrate"))

    elif mod == "single":
        u = up("in")
        if u and len(u["channels"]) != 1:
            out.append(_issue(nid, "hard", f"Single needs exactly one channel upstream; got {len(u['channels'])}", "the mode is defined by it"))

    elif mod == "vectorize":
        u = up("in")
        if u and len(u["channels"]) > 8:
            out.append(_issue(nid, "soft", f"{len(u['channels'])} channels per page; payload and the effective-n gap both scale with this",
                              "ANALYSIS_PIPELINE_METHODOLOGY.md section 6.4", id="wide_vector"))

    elif mod == "complex":
        a, b = up("mag"), up("dir")
        for d, lab, want in ((a, "magnitude", "amount"), (b, "direction", "direction")):
            if d is None:
                continue
            if len(d["channels"]) != 1:
                out.append(_issue(nid, "hard", f"{lab} input must carry exactly one channel; got {len(d['channels'])}", "one magnitude x one direction"))
            else:
                fam = ctx.group_of.get(d["channels"][0])
                if fam != want:
                    if fam == "content":
                        msg = f"{d['channels'][0]} is family C: it describes the new page (state), not the change; fusing state with a derivative makes a number that does not mean what it looks like"
                    else:
                        msg = f"{d['channels'][0]} is family {fam}, not {want}"
                    out.append(_issue(nid, "soft", msg, "ANALYSIS_PIPELINE_METHODOLOGY.md section 2.1", id=f"{lab}_not_{want}"))
        if not p.get("phase"):
            out.append(_issue(nid, "hard", "choose a phase convention; it cannot be recovered from the output afterwards", "plan10 UX section 13.2"))
        elif p["phase"] == "2pi":
            out.append(_issue(nid, "soft", "2 pi collides: distance 0 and distance 1 land on the same angle. Proceed only to reproduce B1 or the first-generation features",
                              "family_b/structure.rs", id="phase_2pi"))

    elif mod == "block":
        out.append(_issue(nid, "soft", "address blocks are allocator placement, not program structure (Gate 2 measured guest-physical scatter)",
                          "docs/dwarf_pilot_design/GATE2_RESULT.md", id="gate2"))
        wp, hp = int(p["wp"]), int(p["hp"])
        n_pages = ctx.config["n_pages_default"]
        if wp < 1 or hp < 1:
            out.append(_issue(nid, "hard", "block and hop must be at least 1 page", "ANALYSIS_PIPELINE_METHODOLOGY.md section 5"))
        elif wp > n_pages:
            out.append(_issue(nid, "hard", f"a block of {wp} pages does not fit in {n_pages} pages", "runner/stages.py n_blocks"))
        else:
            nb = (n_pages - wp) // hp + 1
            if hp < wp:
                out.append(_issue(nid, "soft",
                                  f"blocks overlap ({wp} pages stepped by {hp}), so each changed page belongs to about {wp / hp:.1f} blocks "
                                  f"and its row is replicated once per block: {nb} blocks, roughly {wp / hp:.1f}x the rows and the cost",
                                  "runner/stages.py block", id=f"block_overlap:{nid}"))
            elif hp > wp:
                gap = hp - wp
                out.append(_issue(nid, "soft",
                                  f"blocks are spaced apart ({wp} pages stepped by {hp}), so {gap} of every {hp} pages fall in no block and are dropped: "
                                  f"{nb} blocks covering {nb * wp} of {n_pages} pages",
                                  "runner/stages.py block", id=f"block_gap:{nid}"))
            tail = n_pages - ((nb - 1) * hp + wp)
            if tail:
                out.append(_issue(nid, "note", f"{tail} trailing page(s) do not fill a whole block and are dropped, as Window's edge=drop does in time",
                                  "runner/stages.py n_blocks"))
            out.append(_issue(nid, "note",
                              f"{nb} blocks, and the tile count with it, assume {n_pages} pages per dump from the capture config; "
                              "no recording records its own page count, and the runner uses each recording's actual one",
                              "config_qemu_upc.json; plan10 UX 14.1 item 7"))

    elif mod == "window":
        u = up("in")
        w, h = int(p["w"]), int(p["h"])
        if w < 1 or h < 1:
            out.append(_issue(nid, "hard", "W and H must be at least 1", "window arithmetic"))
        if u and u.get("axis") == "page":
            out.append(_issue(nid, "hard", "tiles at full page resolution are not implemented in the runner; put Collapse (or Block) before Window", "runner/stages.py"))
        if u and u.get("nmin") and w > u["nmin"]:
            out.append(_issue(nid, "hard", f"W_t={w} exceeds the shortest connected recording ({u['nmin']} pairs): zero windows", "plan10 UX section 13.2"))
        if h > w:
            out.append(_issue(nid, "soft", "hop larger than the window skips frames", "Plan 03 G4 hop validity", id="hop_gt_w"))
        if w == 8 and h == 4:
            out.append(_issue(nid, "note", "8/4 is inherited: tuned on APF alone, smallest-W tiebreak, 128 never swept", "plan05 recommendation.json"))
        spectral_down = any((g.mod(e["to"][0]) or {}).get("spectral") for e in g.out_pipes(nid))
        if spectral_down and w < 4:
            out.append(_issue(nid, "hard", f"a spectral lens downstream needs W_t of at least 4; {w} gives fewer than two bins", "plan10 UX section 13.2"))
        if spectral_down and p["taper"] == "rectangular":
            out.append(_issue(nid, "soft", "a spectral lens downstream with no taper leaks across bins; rectangular is a choice here, not a default",
                              "plan10 UX section 5.2", id=f"no_taper:{nid}", fix="taper = hann"))

    elif mod in ("fft", "cepstrum", "wavelet", "scattering"):
        u = up("in")
        if u and u.get("w") and u["w"] < 4:
            out.append(_issue(nid, "hard", f"upstream window W={u['w']} is too short for a spectrum", "plan10 UX section 13.2"))
        if mod == "scattering" and u and u.get("w"):
            J, Q = int(p["J"]), int(p["Q"])
            lim = ctx.modules["feature_sources"].get("scattering", {}).get("max_J", {}).get(str(u["w"]), {}).get(str(Q))
            if not ctx.modules["feature_sources"].get("scattering", {}).get("available"):
                out.append(_issue(nid, "hard", "scattering needs kymatio in the analysis environment (pip install kymatio; the numpy frontend needs no torch)", "runner/stages.py"))
            elif lim is None:
                out.append(_issue(nid, "note", f"no measured J ceiling for W={u['w']} at Q={Q}; the runner measures it and may refuse", "modules.py scattering_limits"))
            elif lim < 1:
                out.append(_issue(nid, "hard", f"a {u['w']}-sample window is too short for scattering at Q={Q}: kymatio's filters do not fit at any J", "kymatio, measured"))
            elif J < 1 or J > lim:
                out.append(_issue(nid, "hard", f"scattering J={J} on a {u['w']}-sample window at Q={Q} allows 1 to {lim}; beyond that every coefficient is a border effect", "kymatio, measured"))
        if mod == "wavelet" and u and u.get("w"):
            levels, fam = int(p["levels"]), p.get("fam", "")
            lim = stages_wavelet_max_level(fam, int(u["w"]))
            if lim == -1:
                out.append(_issue(nid, "hard", "wavelet needs pywt in the analysis environment (pip install PyWavelets)", "runner/stages.py"))
            elif levels < 1 or levels > lim:
                out.append(_issue(nid, "hard",
                                  f"{fam} on a {u['w']}-sample window allows 1 to {lim} level(s); got {levels}. "
                                  "The limit is the filter length, not log2(W)",
                                  "pywt.dwt_max_level"))

    elif mod == "msc":
        # MSC here is self-coherence between adjacent internal windows, not a channel pair:
        # one channel is enough, and the binding requirement is tile length.
        u = up("in")
        iw, ih, meth = int(p["iw"]), int(p["ih"]), p.get("method", "welch")
        # welch averages over segment PAIRS, so it needs three segments; with one pair the
        # ratio is identically 1 whatever the data. The legacy path needs two segments.
        need = iw + (2 if meth == "welch" else 1) * ih
        if iw < 2 or ih < 1:
            out.append(_issue(nid, "hard", "the internal window must be at least 2 samples and the step at least 1", "runner/stages.py"))
        elif u and u.get("w") and u["w"] < need:
            out.append(_issue(nid, "hard",
                              f"MSC ({meth}) at {iw}/{ih} needs a tile of at least {need} samples; upstream tiles are W={u['w']}",
                              "runner/stages.py msc_min_window"))
        if meth == "legacy_adjacent":
            out.append(_issue(nid, "soft", "the legacy MSC is identically 1 wherever both windows hold power: it measures spectral occupancy, not coherence",
                              "known_issues.py msc_single_segment", id="msc_legacy"))

    elif mod == "baseline":
        u = up("in")
        mode = p.get("mode", "cell")
        if u:
            if mode == "benign" and u.get("type") != "features":
                out.append(_issue(nid, "hard", "the benign envelope is a band per FEATURE: connect a lens (features), not tiles", "plan05_campaign/normal_profile.py"))
            if mode == "cell" and u.get("type") != "tiles":
                out.append(_issue(nid, "hard", "a PLV phase baseline is fitted on tiles: connect a Window, not features", "plv_calcolator.py fit_baseline"))
            if mode == "cell" and u.get("type") == "tiles" and not u.get("complex"):
                out.append(_issue(nid, "hard", "a PLV phase baseline needs complex tiles; put a Complex module before the window", "plv_calcolator.py _getPhaseOfComplexSignal"))
        if mode == "benign":
            want = list(p.get("recordings") or [])
            unknown = [r for r in want if r not in ctx.rec_by_id]
            if unknown:
                out.append(_issue(nid, "hard", f"{len(unknown)} benign recording(s) are not in the manifest: {unknown[:2]}", "corpus_manifest.py"))
            lo, hi = float(p.get("q_lo", 5.0)), float(p.get("q_hi", 95.0))
            if not (0.0 <= lo < hi <= 100.0):
                out.append(_issue(nid, "hard", f"the percentiles must satisfy 0 <= q_lo < q_hi <= 100; got {lo} and {hi}", "numpy.nanpercentile"))
            if not want:
                out.append(_issue(nid, "soft", "no benign set chosen, so the envelope is fitted over EVERY recording reaching this node, threats included; that is a normal region defined partly by what it should flag",
                                  "plan05_campaign/normal_profile.py fits on the benign cells only", id=f"benign_all:{nid}"))
            out.append(_issue(nid, "note", f"'normal' here means these {len(want) or 'all'} recordings, not production traffic; normal_profile.py states the same limit",
                              "plan05_campaign/normal_profile.py scope note"))

    elif mod == "deviation":
        r = up("ref")
        if r and r.get("kind") != "envelope":
            out.append(_issue(nid, "hard", "Deviation needs a benign envelope; this Baseline is in cell mode and fits a PLV phase baseline", "runner/stages.py deviation"))
        if r and not r.get("fitted"):
            out.append(_issue(nid, "hard", "the Baseline module has nothing to fit on; connect features to it", "runner/stages.py baseline_envelope"))
        u = up("in")
        if u and r and u.get("names") and r.get("names") and list(u["names"]) != list(r["names"]):
            out.append(_issue(nid, "hard", "the features and the envelope carry different feature names; the band is per feature, so both sides must come from the same lens", "runner/stages.py deviation"))

    elif mod == "plv":
        u = up("in")
        if u and not u.get("complex"):
            out.append(_issue(nid, "hard", "PLV reads the phase of a complex signal; upstream tiles are real. Put a Complex module before the window", "plv_calcolator.py _getPhaseOfComplexSignal"))
        r = up("ref")
        if r and r.get("kind") != "plv":
            out.append(_issue(nid, "hard", "PLV needs a phase baseline; this Baseline is in benign mode and fits a p5-p95 envelope", "plv_calcolator.py fit_baseline"))
        if r and not r.get("fitted"):
            out.append(_issue(nid, "hard", "the Baseline module has nothing to fit on; connect a clean run to it", "plv_calcolator.py fit_baseline"))

    elif mod in ("stats", "deep"):
        feats = list(p.get("feats", []))
        if not feats:
            out.append(_issue(nid, "hard", "no features selected", "nothing to emit"))
        src = {f["name"]: f for f in ctx.modules["feature_sources"]["simple" if mod == "stats" else "deep"]}
        for f in feats:
            if f not in src:
                out.append(_issue(nid, "hard", f"not a known feature for this lens: {f}", "modules.py feature_sources"))
            elif src[f].get("flag"):
                out.append(_issue(nid, "soft", f"{f} is flagged in the known-issue registry ({src[f]['flag']})", "known_issues.py", id=f"feat:{f}"))

    elif mod == "write":
        if not g.in_pipes(nid, "in"):
            out.append(_issue(nid, "hard", "nothing to write", "required port"))

    return out


def validate(scheme: dict, ctx: Context) -> list[dict]:
    issues: list[dict] = []
    if scheme.get("schema") != SCHEMA:
        issues.append(_issue(None, "hard", f"schema is {scheme.get('schema')!r}, expected {SCHEMA!r}", "scheme.py"))
        return issues
    g = Graph(scheme, ctx)
    for nid, n in g.nodes.items():
        if n["module"] not in ctx.mod_by_id:
            issues.append(_issue(nid, "hard", f"unknown module {n['module']!r}", "modules.py"))
    if any(i["sev"] == "hard" and "unknown module" in i["msg"] for i in issues):
        return issues
    seen_single: set[tuple] = set()
    for pi in g.pipes:
        fn, fp = pi["from"]
        tn, tp = pi["to"]
        o, i = g.port(fn, fp, True), g.port(tn, tp, False)
        if fn not in g.nodes or tn not in g.nodes or not o or not i:
            issues.append(_issue(tn if tn in g.nodes else None, "hard", f"pipe refers to a missing node or port: {pi}", "scheme.py"))
            continue
        if fn == tn:
            issues.append(_issue(tn, "hard", "a module cannot feed itself", "scheme.py"))
        if o["t"] not in i["t"]:
            issues.append(_issue(tn, "hard", f"{ctx.modules['types'][o['t']]['label']} cannot enter port '{i['lab']}' ({' / '.join(i['t'])})", "port types"))
        if not i["multi"]:
            key = (tn, tp)
            if key in seen_single:
                issues.append(_issue(tn, "hard", f"port '{i['lab']}' accepts one pipe; got several", "port arity"))
            seen_single.add(key)
    # cycles
    for pi in g.pipes:
        if g.reaches(pi["to"][0], pi["from"][0]):
            issues.append(_issue(pi["to"][0], "hard", f"cycle through {pi['from'][0]} -> {pi['to'][0]}", "graph must be acyclic"))
            break
    if any(i["sev"] == "hard" for i in issues):
        return issues
    memo: dict = {}
    for nid in g.nodes:
        issues.extend(node_constraints(g, nid, memo))
    if not any(n["module"] == "write" for n in g.nodes.values()):
        issues.append(_issue(None, "hard", "no Write module: the scheme produces nothing", "plan10 UX section 7"))
    if not any(n["module"] == "cells" for n in g.nodes.values()):
        issues.append(_issue(None, "hard", "no Cells module: nothing is read", "plan10 UX section 3"))
    for nid, n in g.nodes.items():
        if len(g.nodes) > 1 and not g.in_pipes(nid) and not g.out_pipes(nid):
            issues.append(_issue(nid, "note", f"{ctx.mod_by_id[n['module']]['name']} is not connected to anything", "orphan"))
    return issues


def verdict(issues: list[dict], scheme: dict) -> tuple[int, dict]:
    acks = {a["id"] for a in scheme.get("acknowledged", []) if isinstance(a, dict) and "id" in a}
    hard = [i for i in issues if i["sev"] == "hard"]
    soft_unacked = [i for i in issues if i["sev"] == "soft" and i.get("id") not in acks]
    soft_acked = [i for i in issues if i["sev"] == "soft" and i.get("id") in acks]
    stale_acks = sorted(acks - {i.get("id") for i in issues if i["sev"] == "soft"})
    code = 1 if hard else (2 if soft_unacked else 0)
    return code, {"hard": hard, "soft_unacknowledged": soft_unacked, "soft_acknowledged": soft_acked,
                  "notes": [i for i in issues if i["sev"] == "note"], "acknowledgments_without_issue": stale_acks}


# ---------------------------------------------------------------------------
# cost readout: effective n is workloads, never windows
# ---------------------------------------------------------------------------

def estimate(scheme: dict, ctx: Context) -> dict:
    g = Graph(scheme, ctx)
    memo: dict = {}
    windows = 0
    blocks = 1
    workloads: set[str] = set()
    for nid, n in g.nodes.items():
        if n["module"] != "window":
            continue
        d = descriptor(g, nid, memo)
        if not d:
            continue
        p = g.params(nid)
        w, h = int(p["w"]), int(p["h"])
        mp = d.get("max_pairs")
        for r in d.get("recs", []):
            workloads.add(r["workload"])
            n = min(r["n_pairs"], mp) if mp else r["n_pairs"]
            windows += 0 if n < w else (n - w) // h + 1
        blocks = max(blocks, int(d.get("n_blocks") or 1))
    return {"windows": windows, "blocks": blocks, "tiles": windows * blocks,
            "effective_n_workloads": len(workloads)}


# ---------------------------------------------------------------------------
# examples: the three worked graphs, over the real manifest
# ---------------------------------------------------------------------------

def _usable_ids(manifest: dict, min_pairs: int = 50) -> list[str]:
    return [r["id"] for r in manifest["recordings"] if r["has"]["chain"] and r["n_pairs"] >= min_pairs]


def make_examples(manifest: dict) -> dict:
    sel = _usable_ids(manifest)

    def node(i, module, x, y, **params):
        return {"id": i, "module": module, "params": params, "x": x, "y": y}

    def pipe(a, ap, b, bp):
        return {"from": [a, ap], "to": [b, bp]}

    b1 = {"schema": SCHEMA, "label": "b1_apf_floor", "acknowledged": [],
          "nodes": [node("n1", "cells", 20, 120, sel=sel, min_pairs=50), node("n2", "channels", 290, 120, chans=["hamming"]),
                    node("n3", "collapse", 560, 120, reduce="changed_fraction"), node("n4", "window", 830, 120, w=8, h=4),
                    node("n5", "stats", 1100, 120), node("n6", "write", 1370, 120)],
          "pipes": [pipe("n1", "cells", "n2", "cells"), pipe("n2", "field", "n3", "in"), pipe("n3", "out", "n4", "in"),
                    pipe("n4", "out", "n5", "in"), pipe("n5", "out", "n6", "in")]}
    # Both complex examples collapse the page axis (mean phasor per pair) before windowing:
    # tiles at full page resolution are not implemented and nothing in the record used them.
    cx = {"schema": SCHEMA, "label": "complex_pi_spectral", "acknowledged": [],
          "nodes": [node("n1", "cells", 20, 200, sel=sel, min_pairs=50), node("n2", "channels", 290, 90, chans=["hamming"]),
                    node("n3", "channels", 290, 320, chans=["cosine"]), node("n4", "complex", 560, 200, phase="pi"),
                    node("n10", "collapse", 830, 200), node("n5", "window", 1100, 200, w=32, h=16, taper="hann"),
                    node("n6", "fft", 1370, 90), node("n7", "cepstrum", 1370, 320), node("n8", "concat", 1640, 200),
                    node("n9", "write", 1910, 200)],
          "pipes": [pipe("n1", "cells", "n2", "cells"), pipe("n1", "cells", "n3", "cells"), pipe("n2", "field", "n4", "mag"),
                    pipe("n3", "field", "n4", "dir"), pipe("n4", "out", "n10", "in"), pipe("n10", "out", "n5", "in"),
                    pipe("n5", "out", "n6", "in"), pipe("n5", "out", "n7", "in"), pipe("n6", "out", "n8", "in"),
                    pipe("n7", "out", "n8", "in"), pipe("n8", "out", "n9", "in")]}
    plv = {"schema": SCHEMA, "label": "complex_plv_baseline", "acknowledged": [],
           # min_pairs 130: a W=128 window needs it, and the migrated corpus holds a 70-pair recording
           "nodes": [node("n1", "cells", 20, 200, sel=sel, min_pairs=130), node("n2", "channels", 290, 90, chans=["hamming"]),
                     node("n3", "channels", 290, 320, chans=["cosine"]), node("n4", "complex", 560, 200, phase="arccos"),
                     node("n10", "collapse", 830, 200), node("n5", "window", 1100, 200, w=128, h=64, taper="hann"),
                     node("n6", "baseline", 1370, 360), node("n7", "plv", 1370, 120), node("n8", "write", 1640, 120)],
           "pipes": [pipe("n1", "cells", "n2", "cells"), pipe("n1", "cells", "n3", "cells"), pipe("n2", "field", "n4", "mag"),
                     pipe("n3", "field", "n4", "dir"), pipe("n4", "out", "n10", "in"), pipe("n10", "out", "n5", "in"),
                     pipe("n5", "out", "n6", "in"), pipe("n5", "out", "n7", "in"), pipe("n6", "out", "n7", "ref"),
                     pipe("n7", "out", "n8", "in")]}
    return {"b1": b1, "complex": cx, "plv": plv}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("scheme", type=Path)
    v.add_argument("--roster", type=Path)
    v.add_argument("--manifest", type=Path)
    v.add_argument("--modules", type=Path)
    v.add_argument("--root", type=Path, help="trace root to scan when --manifest is not given")
    v.add_argument("--json", action="store_true")
    e = sub.add_parser("examples")
    e.add_argument("--manifest", type=Path)
    e.add_argument("--root", type=Path)
    e.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    try:
        ctx = Context.default(manifest_path=a.manifest, roster_path=getattr(a, "roster", None),
                              modules_path=getattr(a, "modules", None), manifest_root=a.root)
    except (corpus_manifest.CorpusMissing, channel_roster.RosterError) as ex:
        print(f"[scheme] REFUSED: {ex}", file=sys.stderr)
        return 3
    if a.cmd == "examples":
        a.out_dir.mkdir(parents=True, exist_ok=True)
        for k, s in make_examples(ctx.manifest).items():
            (a.out_dir / f"{k}.json").write_text(json.dumps(s, indent=1) + "\n")
            code, _ = verdict(validate(s, ctx), s)
            print(f"[scheme] wrote {a.out_dir / (k + '.json')}  (validate -> exit {code})")
        return 0
    s = json.loads(a.scheme.read_text())
    issues = validate(s, ctx)
    code, v = verdict(issues, s)
    est = estimate(s, ctx)
    if a.json:
        print(json.dumps({"exit": code, "verdict": v, "estimate": est}, indent=1))
    else:
        for i in issues:
            who = f"[{i['node']}] " if i.get("node") else ""
            print(f"  {i['sev']:5s} {who}{i['msg']}   ({i['src']})")
        print(f"[scheme] tiles={est['tiles']} windows={est['windows']} blocks={est['blocks']} "
              f"effective_n={est['effective_n_workloads']} workloads")
        print(f"[scheme] exit {code}: " + {0: "valid", 1: "blocked", 2: "unacknowledged warnings"}[code])
    return code


if __name__ == "__main__":
    raise SystemExit(main())
