#!/usr/bin/env python3
"""pipeline.py -- the Learn pipeline format, its validator, and the sweep.

A pipeline is a chain of slots in tier order; each slot holds one or more alternatives, and a
run is the product of the alternatives: every configuration, over every split and seed. That
is "execute every possibility", counted and warned about, never silent.

    {"schema": "plan10.learn.v1", "label": "...",
     "slots": [{"tier": "input",      "alts": [{"module": "input", "params": {"run": "...", "source": "features"}}]},
               {"tier": "preprocess", "alts": [{"module": "scale", "params": {"method": "standard"}}], "optional": false},
               {"tier": "model",      "alts": [{"module": "logreg"}, {"module": "rf"}]},
               {"tier": "score",      "alts": [{"module": "score"}]},
               {"tier": "output",     "alts": [{"module": "write"}]}],
     "run": {"splits": ["loro", "lowo"], "seeds": [0], "test_frac": 0.2, "target": "family"},
     "acknowledged": [{"id": "...", "at": "...", "note": "..."}]}

Validation follows the scheme's: hard refuses, soft must be acknowledged by id, notes inform.
The shape flows through each configuration (input -> preprocess -> represent -> model) and a
module that does not accept what reaches it is refused by name; a module whose package is
missing, or that is designed and unbuilt, likewise.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

from plan10_analysis.learn import registry as R
from plan10_analysis.learn.splits import SPLITS

SCHEMA = "plan10.learn.v1"
TIER_ORDER = [t["k"] for t in R.TIERS]
SWEEP_SOFT_AT = 12
TARGETS = ("family", "workload", "archetype")


class Context:
    """What validation needs beyond the pipeline: the registry and what each run offers."""

    def __init__(self, runs: dict, registry: dict | None = None):
        self.registry = registry or R.build_registry()
        self.mod = {m["id"]: m for m in self.registry["modules"]}
        self.runs = runs      # label -> data.available() plus n_families / n_workloads / n_recordings when known


def _issue(slot, alt, sev, msg, src, id=None):
    out = {"slot": slot, "alt": alt, "sev": sev, "msg": msg, "src": src}
    if id:
        out["id"] = id
    return out


def params_of(mod: dict, given: dict | None) -> dict:
    p = {q["k"]: q["default"] for q in mod.get("params", [])}
    p.update(given or {})
    return p


def configurations(pipeline: dict) -> list[dict]:
    """The product over slots; an optional slot contributes a 'skip'. Each configuration lists
    its steps in slot order and carries a deterministic name."""
    slots = pipeline.get("slots") or []
    choices = []
    for si, s in enumerate(slots):
        alts = [dict(a, _slot=si, _alt=ai) for ai, a in enumerate(s.get("alts") or [])]
        if s.get("optional"):
            alts = [None] + alts
        choices.append(alts if alts else [None])
    out = []
    for i, combo in enumerate(itertools.product(*choices)):
        steps = [c for c in combo if c]
        name = " > ".join(_step_name(c) for c in steps if c["module"] not in ("score", "write"))
        out.append({"id": f"c{i + 1:03d}", "name": name, "steps": steps})
    return out


def _step_name(step: dict) -> str:
    p = step.get("params") or {}
    if step["module"] == "input":
        return f"{p.get('run', '?')}/{p.get('source', 'features')}"
    keyp = {k: v for k, v in p.items() if k in ("method", "n_components", "k", "bank", "k_mode", "bottleneck", "latent", "hidden", "epochs", "bins")}
    return step["module"] + (":" + ",".join(f"{k}={v}" for k, v in keyp.items()) if keyp else "")


def input_shape(ctx: Context, params: dict) -> tuple[str | None, str | None]:
    """(shape, why not) for an input step."""
    run = ctx.runs.get(params.get("run") or "")
    if not run:
        return None, f"no run {params.get('run')!r} with results"
    src = params.get("source", "features")
    if src == "features":
        return ("rows", None) if run.get("features") else (None, f"run {run['label']} has no features.npz")
    if src == "tiles":
        return (run.get("tiles_shape") or "path", None) if run.get("tiles") else (None, f"run {run['label']} has no tiles.npz (add Write tiles to its scheme)")
    return None, f"unknown source {src!r}"


def validate(pipeline: dict, ctx: Context) -> list[dict]:
    issues: list[dict] = []
    if pipeline.get("schema") != SCHEMA:
        return [_issue(None, None, "hard", f"schema is {pipeline.get('schema')!r}, expected {SCHEMA!r}", "pipeline.py")]
    slots = pipeline.get("slots") or []
    run = pipeline.get("run") or {}
    # tiers in order, every alternative a known, available module of that tier
    last = -1
    tiers_present = set()
    for si, s in enumerate(slots):
        t = s.get("tier")
        if t not in TIER_ORDER:
            issues.append(_issue(si, None, "hard", f"unknown tier {t!r}", "registry"))
            continue
        tiers_present.add(t)
        if TIER_ORDER.index(t) < last:
            issues.append(_issue(si, None, "hard", f"slot {si + 1} ({t}) comes after a later tier", "tier order"))
        last = max(last, TIER_ORDER.index(t))
        if not s.get("alts") and not s.get("optional"):
            issues.append(_issue(si, None, "hard", f"slot {si + 1} ({t}) has no alternative", "empty slot"))
        for ai, a in enumerate(s.get("alts") or []):
            m = ctx.mod.get(a.get("module"))
            if not m:
                issues.append(_issue(si, ai, "hard", f"unknown module {a.get('module')!r}", "registry"))
                continue
            if m["tier"] != t:
                issues.append(_issue(si, ai, "hard", f"{m['name']} belongs to the {m['tier']} tier, not {t}", "tier"))
            if not m["available"]:
                issues.append(_issue(si, ai, "hard", f"{m['name']}: {m['why']}", "registry"))
            for q in m.get("params", []):
                if q["kind"] == "number" and a.get("params", {}).get(q["k"]) is not None:
                    v = a["params"][q["k"]]
                    try:
                        v = float(v)
                    except (TypeError, ValueError):
                        issues.append(_issue(si, ai, "hard", f"{m['name']}: {q['lab']} is not a number", "params"))
                        continue
                    if q.get("min") is not None and v < q["min"] or q.get("max") is not None and v > q["max"]:
                        issues.append(_issue(si, ai, "hard", f"{m['name']}: {q['lab']} = {v:g} is outside [{q.get('min')}, {q.get('max')}]", "params"))
    for need in ("input", "model", "score", "output"):
        if need not in tiers_present:
            issues.append(_issue(None, None, "hard", f"no {need} slot: " + {"input": "nothing is read", "model": "nothing is fitted",
                                                                          "score": "nothing is measured", "output": "the run produces nothing"}[need], "pipeline"))
    if any(i["sev"] == "hard" for i in issues):
        return issues
    # the run panel
    splits = run.get("splits") or []
    if not splits:
        issues.append(_issue(None, None, "hard", "no split chosen", "run"))
    for sp in splits:
        if sp not in SPLITS:
            issues.append(_issue(None, None, "hard", f"unknown split {sp!r}; one of {', '.join(SPLITS)}", "run"))
    if not run.get("seeds"):
        issues.append(_issue(None, None, "hard", "no seed", "run"))
    if run.get("target", "family") not in TARGETS:
        issues.append(_issue(None, None, "hard", f"target must be one of {', '.join(TARGETS)}", "run"))
    if any(i["sev"] == "hard" for i in issues):
        return issues
    target = run.get("target", "family")
    if splits == ["within_trace"]:
        issues.append(_issue(None, None, "soft", "within-trace is the memorisation ceiling and the only split chosen; it is never the headline", "b1_splits", id="ceiling_only"))
    # every configuration: the shape flows, and what it meets on the way
    confs = configurations(pipeline)
    seen_soft = set()
    for c in confs:
        shape, why = None, None
        for st in c["steps"]:
            m = ctx.mod[st["module"]]
            p = params_of(m, st.get("params"))
            if m["tier"] == "input":
                shape, why = input_shape(ctx, p)
                if why:
                    issues.append(_issue(st["_slot"], st["_alt"], "hard", why, "input"))
                    break
                r = ctx.runs.get(p.get("run"))
                if r and target == "archetype" and ("archetype:" + r["label"]) not in seen_soft:
                    seen_soft.add("archetype:" + r["label"])
                    na, nk = len(r.get("archetypes") or []), r.get("n_kernels") or 0
                    if na < 2:
                        issues.append(_issue(st["_slot"], st["_alt"], "hard", f"{r['label']}: {nk} kernel(s) of the twelve, {na} archetype(s): target archetype needs at least two",
                                             "plan11_encoding_ladder/schema.py ARCHETYPE_OF"))
                    else:
                        dropped = r.get("archetype_rows_dropped") or 0
                        issues.append(_issue(st["_slot"], st["_alt"], "note", f"{r['label']}: {nk} kernels over {na} archetypes (plan11 schema.py ARCHETYPE_OF); "
                                             f"{dropped} row(s) have no archetype (the idle cells and anything not among the twelve kernels) and are dropped from this run"
                                             + ("; leave-one-workload-out is leave one kernel out" if "lowo" in splits else "")
                                             + (f"; {r['single_kernel_archetypes']} archetype(s) with a single kernel cannot be predicted under it" if r.get("single_kernel_archetypes") and "lowo" in splits else ""),
                                             "learn/data.py archetype"))
                if r and r.get("n_workloads") == 1 and any(s in ("lowo",) for s in splits) and ("single_workload:" + r["label"]) not in seen_soft:
                    seen_soft.add("single_workload:" + r["label"])
                    issues.append(_issue(st["_slot"], st["_alt"], "soft", f"{r['label']} holds one workload: leave-one-workload-out leaves nothing to train on; those folds are skipped and named",
                                         "splits", id="single_workload:" + r["label"]))
                if r and r.get("n_recordings") == 1 and any(s in ("loro", "lowo", "loco") for s in splits) and ("single_recording:" + r["label"]) not in seen_soft:
                    seen_soft.add("single_recording:" + r["label"])
                    issues.append(_issue(st["_slot"], st["_alt"], "soft", f"{r['label']} holds one recording: only within-trace can split it", "splits", id="single_recording:" + r["label"]))
                continue
            if m["tier"] in ("score", "output"):
                continue
            if shape not in m["accepts"]:
                issues.append(_issue(st["_slot"], st["_alt"], "hard",
                                     f"{m['name']} reads {' / '.join(m['accepts'])}, but {shape} reaches it in configuration {c['id']} ({c['name']})"
                                     + (" ; add Flatten before it" if shape in ("path", "image") and "rows" in m["accepts"] else ""), "shapes"))
                break
            if m["tier"] == "preprocess":
                shape = shape if m["emits"] == "same" else m["emits"]
                if st["module"] == "per_recording_z" and "level_removed" not in seen_soft:
                    seen_soft.add("level_removed")
                    issues.append(_issue(st["_slot"], st["_alt"], "soft", "per-recording z removes each recording's level, the thing the APF vs wAPF arms differ in; a held-out recording normalises itself (no labels used)",
                                         "preprocess", id="level_removed"))
            elif m["tier"] == "represent":
                shape = "embedding"
            elif m["tier"] == "model":
                if "torch" in m["needs"] and ("deep:" + st["module"]) not in seen_soft:
                    seen_soft.add("deep:" + st["module"])
                    issues.append(_issue(st["_slot"], st["_alt"], "note", f"{m['name']} is trained from scratch; at effective n = workloads read it as a ceiling probe next to the null", "small n"))
                if m["kind"] == "clusterer":
                    issues.append(_issue(st["_slot"], st["_alt"], "note", f"{m['name']}: k is fixed in advance ({p.get('k_mode')}), never swept to fit", "b1"))
    n_fits = len(confs) * len(splits) * len(run.get("seeds") or [1])
    if len(confs) * len(run.get("seeds") or [1]) > SWEEP_SOFT_AT:
        issues.append(_issue(None, None, "soft", f"{len(confs)} configurations x {len(run.get('seeds') or [1])} seed(s) = {len(confs) * len(run.get('seeds') or [1])} fits per split at effective n = workloads: "
                                                 "a multiple-comparisons hazard; the count is recorded in every result", "sweep", id=f"sweep:{len(confs) * len(run.get('seeds') or [1])}"))
    return issues


def verdict(issues: list[dict], pipeline: dict) -> tuple[int, dict]:
    acks = {a["id"] for a in pipeline.get("acknowledged", []) if isinstance(a, dict) and "id" in a}
    hard = [i for i in issues if i["sev"] == "hard"]
    soft_unacked = [i for i in issues if i["sev"] == "soft" and i.get("id") not in acks]
    soft_acked = [i for i in issues if i["sev"] == "soft" and i.get("id") in acks]
    stale = sorted(acks - {i.get("id") for i in issues if i["sev"] == "soft"})
    code = 1 if hard else (2 if soft_unacked else 0)
    return code, {"hard": hard, "soft_unacknowledged": soft_unacked, "soft_acknowledged": soft_acked,
                  "notes": [i for i in issues if i["sev"] == "note"], "acknowledgments_without_issue": stale}


def estimate(pipeline: dict, ctx: Context) -> dict:
    confs = configurations(pipeline)
    run = pipeline.get("run") or {}
    splits = run.get("splits") or []
    seeds = run.get("seeds") or [0]
    runs_used = sorted({params_of(ctx.mod["input"], s.get("params")).get("run") for c in confs for s in c["steps"] if s["module"] == "input"} - {None, ""})
    folds = 0
    for lab in runs_used:
        r = ctx.runs.get(lab) or {}
        for sp in splits:
            n_wl = (r.get("n_kernels") or 0) if run.get("target") == "archetype" else (r.get("n_workloads") or 0)
            folds += {"within_trace": 1, "loro": r.get("n_recordings") or 0, "lowo": n_wl, "loco": r.get("n_campaigns") or 0}.get(sp, 0)
    return {"configurations": len(confs), "splits": len(splits), "seeds": len(seeds), "runs": runs_used,
            "folds_per_run": folds, "fits": len(confs) * len(seeds) * folds if runs_used else 0}


def examples(ctx: Context) -> dict:
    """Worked pipelines over the runs that exist, the way scheme.py's examples are over the manifest."""
    labs = [l for l, r in ctx.runs.items() if r.get("features")]
    tile_labs = [l for l, r in ctx.runs.items() if r.get("tiles")]
    out = {}
    if labs:
        lab = labs[0]
        out["floors"] = {"schema": SCHEMA, "label": "floors", "acknowledged": [],
                         "slots": [{"tier": "input", "alts": [{"module": "input", "params": {"run": lab, "source": "features"}}]},
                                   {"tier": "preprocess", "alts": [{"module": "scale", "params": {"method": "standard"}}]},
                                   {"tier": "model", "alts": [{"module": "logreg"}, {"module": "knn"}, {"module": "rf"}]},
                                   {"tier": "score", "alts": [{"module": "score"}]}, {"tier": "output", "alts": [{"module": "write"}]}],
                         "run": {"splits": ["loro", "lowo"], "seeds": [0], "test_frac": 0.2, "target": "family"}}
        out["b1_bank"] = {"schema": SCHEMA, "label": "b1_bank", "acknowledged": [],
                          "slots": [{"tier": "input", "alts": [{"module": "input", "params": {"run": lab, "source": "features"}}]},
                                    {"tier": "preprocess", "alts": [{"module": "scale", "params": {"method": "standard"}}]},
                                    {"tier": "model", "alts": [{"module": "ae_bank", "params": {"bank": "family", "bottleneck": 3}}, {"module": "kmeans"}, {"module": "rf"}]},
                                    {"tier": "score", "alts": [{"module": "score"}]}, {"tier": "output", "alts": [{"module": "write"}]}],
                          "run": {"splits": ["within_trace", "loro", "lowo"], "seeds": [0], "test_frac": 0.2, "target": "family"}}
    if tile_labs:
        lab = tile_labs[0]
        out["paths"] = {"schema": SCHEMA, "label": "paths", "acknowledged": [],
                        "slots": [{"tier": "input", "alts": [{"module": "input", "params": {"run": lab, "source": "tiles"}}]},
                                  {"tier": "preprocess", "alts": [{"module": "scale", "params": {"method": "standard"}}]},
                                  {"tier": "model", "alts": [{"module": "minirocket"}, {"module": "lstm", "params": {"epochs": 30}}, {"module": "tcn", "params": {"epochs": 30}}]},
                                  {"tier": "score", "alts": [{"module": "score"}]}, {"tier": "output", "alts": [{"module": "write"}]}],
                        "run": {"splits": ["loro", "lowo"], "seeds": [0], "test_frac": 0.2, "target": "family"}}
    return out


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())
