#!/usr/bin/env python3
"""render_dependency_graph.py -- self-contained HTML view of dependency_graph.json.

Reads the graph the extractor produced (nodes, edges with evidence, violations, blind spots),
computes a layered left-to-right layout in the spirit of a neural-network diagram, and injects
everything into dependency_graph.template.html at the /*@@GENERATED_DATA@@*/ marker. The output,
dependency_graph.html, needs no network: no CDN, no fonts fetched, no fetch() calls.

Layout rules (from the brief):
  * x is the layer index from the JSON; nodes stack vertically inside a layer column.
  * the two true roots sit at layer 0, alone, at the top of the page;
  * rank-1 nodes (secondary roots and the subtrees only they dominate) live in a separate band
    BELOW a divider, never at or left of layer 0;
  * every edge points strictly left to right; the only same-layer edges are inside a marked SCC,
    which is drawn as one banded group. Any other same-layer or right-to-left edge is reported.

No extractor logic lives here and nothing in dependency_graph.json is modified.

Run:  python3 docs/dependency_graph/render_dependency_graph.py [--json PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "dependency_graph.template.html"
DEFAULT_JSON = HERE / "dependency_graph.json"
DEFAULT_OUT = HERE / "dependency_graph.html"
CORE_OUT = HERE / "dependency_graph_core.html"
MARKER = "/*@@GENERATED_DATA@@*/"

# Two views over one graph.
#   full -- every node the extractor found, rank 0 above the divider and rank 1 below.
#   core -- rank 0 only: the two console entry points and everything they actually reach.
# core is a SUBSET and says so on the page. A partial graph presented as whole would license
# deletions a complete one forbids, so the banner carries the global counts, names what dropped
# out, and links to the full view.
VIEWS = ("full", "core")


def filter_view(d: dict, view: str) -> dict:
    """Return a graph dict restricted to `view`. `full` is the input unchanged."""
    if view == "full":
        return d
    keep = {n["path"] for n in d["nodes"] if n["rank"] == 0}
    nodes = [n for n in d["nodes"] if n["path"] in keep]
    edges = [e for e in d["edges"] if e["src"] in keep and e["dst"] in keep]

    def viol_in(v: dict) -> bool:
        ee = v.get("expected_edge")
        if ee:
            return ee["src"] in keep and ee["dst"] in keep
        return v.get("node") in keep

    by_class: dict[str, int] = defaultdict(int)
    by_kind: dict[str, int] = defaultdict(int)
    for n in nodes:
        by_class[n["class"]] += 1
        by_kind[n["kind"]] += 1
    by_type: dict[str, int] = defaultdict(int)
    by_binding: dict[str, int] = defaultdict(int)
    for e in edges:
        by_type[e["type"]] += 1
        by_binding[e["binding"]] += 1
    cycles = [c for c in d.get("cycles", []) if all(m in keep for m in c["members"])]

    out = dict(d)
    out["nodes"] = nodes
    out["edges"] = edges
    out["cycles"] = cycles
    out["violations"] = [v for v in d["violations"] if viol_in(v)]
    out["roots"] = {"true": d["roots"]["true"], "secondary": []}
    out["summary"] = {
        "nodes": len(nodes), "edges": len(edges),
        "by_class": dict(by_class), "by_kind": dict(by_kind),
        "edges_by_type": dict(by_type), "edges_by_binding": dict(by_binding),
        "unresolved_edges": by_binding.get("unresolved", 0),
        "cycles": len(cycles),
        "nodes_edges_incomplete": sum(1 for n in nodes if n.get("edges_incomplete")),
    }
    return out


def view_meta(d: dict, shown: dict, view: str) -> dict:
    """The honesty block the page prints: what is on screen, and what is not."""
    gs, ss = d["summary"], shown["summary"]
    acc = d.get("acceptance", {})
    kept = {n["path"] for n in shown["nodes"]}
    dropped_acc = []
    for key in ("A", "B"):
        a = acc.get(key) or {}
        paths = [p for p in (a.get("node"), a.get("artifact"),
                             (a.get("writes_edge") or [{}])[0].get("dst") if a.get("writes_edge") else None) if p]
        if paths and not any(p in kept for p in paths):
            dropped_acc.append(key)
    return {
        "name": view,
        "is_subset": view != "full",
        "shown_nodes": ss["nodes"], "total_nodes": gs["nodes"],
        "shown_edges": ss["edges"], "total_edges": gs["edges"],
        "hidden_nodes": gs["nodes"] - ss["nodes"],
        "hidden_edges": gs["edges"] - ss["edges"],
        "global_by_class": gs["by_class"],
        "shown_by_class": ss["by_class"],
        "global_violations": len(d["violations"]),
        "shown_violations": len(shown["violations"]),
        "acceptance_not_shown": dropped_acc,
        "counterpart": "dependency_graph.html" if view == "core" else "dependency_graph_core.html",
        "counterpart_label": "full view (all ranks)" if view == "core" else "core view (rank 0 only)",
    }

# geometry (px, unscaled)
COL_W = 340          # x step per layer
NODE_W = 244         # node width
X0 = 60              # left margin
Y0 = 96              # top of the rank-0 band (below column headers)
H_ROOT = 32
H0 = 22              # rank-0 node height
H0_DENSE = 15        # rank-0 height in a column with more than DENSE_THRESHOLD nodes
H1 = 17              # rank-1 node height
H_DOC = 13           # rank-1 doc node height
GAP = 6
GAP_DENSE = 3
DENSE_THRESHOLD = 40
DIVIDER_GAP = 84     # vertical room for the divider label between the two bands
SWEEPS = 10


def build_layout(d: dict) -> dict:
    nodes = d["nodes"]
    N = {n["path"]: n for n in nodes}
    edges = d["edges"]
    scc_of = {m: c["scc_id"] for c in d.get("cycles", []) for m in c["members"]}

    # --- ordering units: an SCC is one unit so its members stay contiguous -------------------
    unit_of: dict[str, str] = {}
    units: dict[str, dict] = {}
    for n in nodes:
        uid = f"scc:{scc_of[n['path']]}" if n["path"] in scc_of else n["path"]
        unit_of[n["path"]] = uid
        u = units.setdefault(uid, {"id": uid, "members": [], "layer": n["layer"], "rank": n["rank"]})
        u["members"].append(n["path"])
        # an SCC member's layer/rank are shared by construction; keep the max as a guard
        u["layer"] = max(u["layer"], n["layer"])
        u["rank"] = min(u["rank"], n["rank"])
    for u in units.values():
        u["members"].sort()

    out_deg: dict[str, int] = defaultdict(int)
    in_deg: dict[str, int] = defaultdict(int)
    preds: dict[str, set] = defaultdict(set)
    succs: dict[str, set] = defaultdict(set)
    inversions = []
    for e in edges:
        out_deg[e["src"]] += 1
        in_deg[e["dst"]] += 1
        a, b = unit_of[e["src"]], unit_of[e["dst"]]
        if a != b:
            preds[b].add(a)
            succs[a].add(b)
            if N[e["dst"]]["layer"] <= N[e["src"]]["layer"]:
                inversions.append({"src": e["src"], "dst": e["dst"], "type": e["type"]})

    def isolated(u: dict) -> bool:
        return all(out_deg[m] == 0 and in_deg[m] == 0 for m in u["members"])

    def is_doc(u: dict) -> bool:
        return all(N[m]["kind"] == "doc" for m in u["members"])

    def has_root(u: dict) -> bool:
        return any(N[m]["class"] == "root" for m in u["members"])

    layers = sorted({u["layer"] for u in units.values()})
    groups: dict[tuple[int, int], list[dict]] = {(L, r): [] for L in layers for r in (0, 1)}
    for u in units.values():
        groups[(u["layer"], u["rank"])].append(u)
    dense_layer = {L: len(groups[(L, 0)]) > DENSE_THRESHOLD for L in layers}

    # --- heights ------------------------------------------------------------------------------
    for u in units.values():
        L, r = u["layer"], u["rank"]
        hs = []
        for m in u["members"]:
            n = N[m]
            if n["class"] == "root":
                h = H_ROOT
            elif r == 0:
                h = H0_DENSE if dense_layer[L] else H0
            elif n["kind"] == "doc":
                h = H_DOC
            else:
                h = H1
            hs.append(h)
        u["heights"] = hs
        u["gap"] = GAP_DENSE if (r == 0 and dense_layer[L]) or (r == 1 and is_doc(u)) else GAP
        u["h"] = sum(hs) + GAP_DENSE * (len(hs) - 1)

    # --- initial order: roots first, connected before isolated, docs last, then by path --------
    for key, lst in groups.items():
        lst.sort(key=lambda u: (0 if has_root(u) else 1, 1 if isolated(u) else 0, 1 if is_doc(u) else 0, u["id"]))

    def assign_centres() -> dict[str, float]:
        """centre y of every unit inside its own band (band offsets are applied at the end)."""
        pos: dict[str, float] = {}
        for key, lst in groups.items():
            y = 0.0
            for u in lst:
                pos[u["id"]] = y + u["h"] / 2
                y += u["h"] + u["gap"]
        return pos

    # --- barycenter sweeps (forward on predecessors, backward on successors) ------------------
    for it in range(SWEEPS):
        pos = assign_centres()
        forward = it % 2 == 0
        order = layers if forward else list(reversed(layers))
        for L in order:
            for r in (0, 1):
                lst = groups[(L, r)]
                movable = [u for u in lst if not isolated(u) and not has_root(u)]
                roots = [u for u in lst if has_root(u)]
                pinned = [u for u in lst if isolated(u)]

                def bary(u: dict):
                    nb = preds[u["id"]] if forward else succs[u["id"]]
                    if r == 0:
                        nb = {v for v in nb if units[v]["rank"] == 0}   # keep the spine coherent
                    vals = [pos[v] for v in nb if v in pos]
                    if not vals:
                        return None
                    return sum(vals) / len(vals)

                keyed = []
                for idx, u in enumerate(movable):
                    b = bary(u)
                    keyed.append((b if b is not None else pos[u["id"]], idx, u))
                keyed.sort(key=lambda t: (t[0], t[1]))
                groups[(L, r)] = roots + [t[2] for t in keyed] + pinned

    # --- final coordinates: rank-0 band, divider, rank-1 band ---------------------------------
    positions: dict[str, dict] = {}
    col_bottom0: dict[int, float] = {}
    for L in layers:
        x = X0 + L * COL_W
        y = float(Y0)
        for u in groups[(L, 0)]:
            for m, h in zip(u["members"], u["heights"]):
                positions[m] = {"x": x, "y": round(y, 1), "w": NODE_W, "h": h, "layer": L, "rank": 0,
                                "dense": dense_layer[L]}
                y += h + GAP_DENSE
            y += u["gap"] - GAP_DENSE
        col_bottom0[L] = y
    divider_y = max(col_bottom0.values()) + 24
    band1_top = divider_y + DIVIDER_GAP
    col_bottom1: dict[int, float] = {}
    for L in layers:
        x = X0 + L * COL_W
        y = float(band1_top)
        for u in groups[(L, 1)]:
            for m, h in zip(u["members"], u["heights"]):
                positions[m] = {"x": x, "y": round(y, 1), "w": NODE_W, "h": h, "layer": L, "rank": 1,
                                "dense": False}
                y += h + GAP_DENSE
            y += u["gap"] - GAP_DENSE
        col_bottom1[L] = y
    has_rank1 = any(groups[(L, 1)] for L in layers)
    if not has_rank1:
        # nothing below the divider in this view: do not reserve a band for it
        divider_y = band1_top = max(col_bottom0.values())
        total_h = divider_y + 60
    else:
        total_h = max(list(col_bottom1.values()) + [band1_top]) + 60
    total_w = X0 + (max(layers) + 1) * COL_W

    # --- SCC bands ----------------------------------------------------------------------------
    scc_boxes = []
    for c in d.get("cycles", []):
        ps = [positions[m] for m in c["members"] if m in positions]
        if not ps:
            continue
        x0 = min(p["x"] for p in ps) - 8
        y0 = min(p["y"] for p in ps) - 18
        x1 = max(p["x"] + p["w"] for p in ps) + 8
        y1 = max(p["y"] + p["h"] for p in ps) + 8
        scc_boxes.append({"scc_id": c["scc_id"], "members": c["members"], "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0})

    columns = [{"layer": L, "x": X0 + L * COL_W, "n0": sum(len(u["members"]) for u in groups[(L, 0)]),
                "n1": sum(len(u["members"]) for u in groups[(L, 1)]), "dense": dense_layer[L]} for L in layers]
    return {
        "nodes": positions,
        "columns": columns,
        "geometry": {"col_w": COL_W, "node_w": NODE_W, "x0": X0, "y0": Y0, "divider_y": divider_y,
                     "band1_top": band1_top, "total_w": total_w, "total_h": total_h},
        "scc_boxes": scc_boxes,
        "inversions": inversions,
        "counts": {"units": len(units), "rank0": sum(1 for n in nodes if n["rank"] == 0),
                   "rank1": sum(1 for n in nodes if n["rank"] == 1),
                   "isolated": sum(1 for n in nodes if out_deg[n["path"]] == 0 and in_deg[n["path"]] == 0)},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--view", choices=VIEWS, default="full",
                    help="full: every node. core: rank 0 only, the two roots and what they reach.")
    a = ap.parse_args()
    if a.out is None:
        a.out = DEFAULT_OUT if a.view == "full" else CORE_OUT
    full = json.loads(a.json.read_text())
    for key in ("nodes", "edges", "violations", "blind_spots", "roots", "cycles", "summary", "acceptance", "meta"):
        if key not in full:
            sys.exit(f"{a.json}: missing top-level key {key!r}")
    d = filter_view(full, a.view)
    vmeta = view_meta(full, d, a.view)
    layout = build_layout(d)
    if layout["inversions"]:
        print(f"[render] WARNING: {len(layout['inversions'])} edge(s) do not point left to right outside an SCC; "
              "they are drawn in red and listed on the page", file=sys.stderr)
    payload = {
        "graph": d,
        "layout": layout,
        "view": vmeta,
        "render": {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "source_json": str(a.json.relative_to(HERE.parent.parent)) if a.json.is_relative_to(HERE.parent.parent) else str(a.json),
                   "source_bytes": a.json.stat().st_size,
                   "renderer": "docs/dependency_graph/render_dependency_graph.py"},
    }
    template = TEMPLATE.read_text()
    if MARKER not in template:
        sys.exit(f"marker {MARKER} not found in {TEMPLATE.name}")
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = template.replace(MARKER, "const DATA = " + blob + ";")
    a.out.write_text(html)
    g = layout["geometry"]
    if vmeta["is_subset"]:
        print(f"[render] view={a.view}: SUBSET -- {vmeta['shown_nodes']}/{vmeta['total_nodes']} nodes, "
              f"{vmeta['shown_edges']}/{vmeta['total_edges']} edges, "
              f"{vmeta['shown_violations']}/{vmeta['global_violations']} violations"
              + (f"; acceptance {', '.join(vmeta['acceptance_not_shown'])} not visible in this view"
                 if vmeta["acceptance_not_shown"] else ""), file=sys.stderr)
    print(f"[render] {a.out.name}: {len(d['nodes'])} nodes, {len(d['edges'])} edges, {len(d['violations'])} violation rows; "
          f"canvas {g['total_w']}x{int(g['total_h'])}px, divider at y={int(g['divider_y'])}, "
          f"{layout['counts']['rank0']} rank-0 / {layout['counts']['rank1']} rank-1 nodes, "
          f"{len(layout['inversions'])} inversions; {a.out.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
