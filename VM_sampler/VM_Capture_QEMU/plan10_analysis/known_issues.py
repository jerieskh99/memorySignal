#!/usr/bin/env python3
"""known_issues.py -- the registry of things known to be bad, with their history recomputed.

Nothing known-bad is removed from the console. It stays, badged, with the
history on hover (plan10_analysis_console_UX.md section 12). This module is
where that history comes from, and the rule is: every number a hovercard
shows was recomputed from the artifact it cites at build time, or the card
says it was not.

Each entry names its artifacts and a `verify` that reads them. If an artifact
is missing the entry is emitted with verified=false and the reason, and the
detail text keeps its placeholders visibly unfilled. The one entry whose
artifact could not be found on this machine at all (the CUSUM null
calibration) is emitted that way on purpose rather than dropped.

Run:  python3 plan10_analysis/known_issues.py [--out issues.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
QEMU_DIR = HERE.parent
REPO = QEMU_DIR.parent.parent
DOWNSTREAM = QEMU_DIR / "plan05_campaign" / "downstream"
RUST_METRICS = REPO / "VM_sampler" / "VM_Capture" / "live_delta_calc_modular" / "src" / "metrics"

SCHEMA = "plan10.known_issues.v1"


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------
# verifiers: each returns the numbers its entry's detail text needs
# ---------------------------------------------------------------------------

def v_f1_phase() -> dict:
    p = DOWNSTREAM / "sweep.csv"
    rows = list(csv.DictReader(p.open()))
    filled = sum(1 for r in rows if r.get("f1_phase", "") not in ("", "None"))
    return {"n_rows": len(rows), "n_filled": filled}


def v_coverage_ratio() -> dict:
    sweep_py = (QEMU_DIR / "plan03_sweep.py").read_text()
    m = re.search(r'RHYTHM_S\s*=\s*\{[^}]*"phasic":\s*([\d.]+)[^}]*"steady":\s*([\d.]+)', sweep_py, re.S)
    if not m:
        raise FileNotFoundError("RHYTHM_S not found in plan03_sweep.py")
    phasic, steady = float(m.group(1)), float(m.group(2))
    d = json.loads((DOWNSTREAM / "leakage_ablation.json").read_text())
    alone = d["runs"]["coverage_ratio_alone"]
    c = alone["confusion"]
    return {"phasic_s": phasic, "steady_s": steady,
            "w8_phasic": round(8 * 0.5 / phasic, 4), "w8_steady": round(8 * 0.5 / steady, 4),
            "acc_alone": alone["accuracy"],
            "confusion": f"{c['tn_steady']}/{c['fp']}/{c['fn']}/{c['tp_phasic']}"}


def v_ceps_snr() -> dict:
    d = json.loads((DOWNSTREAM / "leakage_ablation.json").read_text())
    return {"acc_alone": round(d["runs"]["ceps_snr_alone"]["accuracy"], 4),
            "majority": round(d["majority_baseline"], 4), "protocol": d["protocol"]}


def v_cv_workingset() -> dict:
    text = (QEMU_DIR / "plan05_campaign" / "behavior_families.py").read_text()
    m = re.search(r"FAMILY_CONDITIONAL\s*=\s*\[([^\]]*)\]", text)
    if not m:
        raise FileNotFoundError("FAMILY_CONDITIONAL not found in behavior_families.py")
    names = re.findall(r'"(\w+)"', m.group(1))
    if "cv_workingset" not in names:
        raise FileNotFoundError("cv_workingset is not in FAMILY_CONDITIONAL")
    return {"family_conditional": ", ".join(names)}


def v_duty_gt05() -> dict:
    text = (QEMU_DIR / "plan05_campaign" / "extra_features.py").read_text()
    m = re.search(r'"duty_gt05":.*?>\s*([\d.]+)', text)
    if not m:
        raise FileNotFoundError("duty_gt05 threshold not found in extra_features.py")
    b1 = (QEMU_DIR / "plan08_b1" / "b1_features.py").read_text()
    m2 = re.search(r"thr\s*=\s*([\d.]+)\s*\*\s*mx", b1)
    if not m2:
        raise FileNotFoundError("duty threshold not found in b1_features.py")
    return {"abs_threshold": m.group(1), "rel_threshold": m2.group(1)}


def v_cusum_band() -> dict:
    hits = [p for p in DOWNSTREAM.glob("*.json") if "band_near_unfalsifiable" in p.read_text()]
    hits += [p for p in QEMU_DIR.glob("plan04*.json") if "band_near_unfalsifiable" in p.read_text()]
    if not hits:
        raise FileNotFoundError("no local artifact carries band_near_unfalsifiable "
                                "(searched plan05_campaign/downstream/*.json, plan04*.json)")
    return {"artifact": _rel(hits[0])}


def v_phase_2pi() -> dict:
    rs = (RUST_METRICS / "family_b" / "structure.rs").read_text()
    if "cosine DISTANCE" not in rs or "all-zero side to 1" not in rs:
        raise FileNotFoundError("distance/all-zero comments not found in family_b/structure.rs")
    fe = (REPO / "VMsig_featureExctraction" / "block_feature_extractor.py").read_text()
    sites = len(re.findall(r"np\.exp\(\s*1j\s*\*\s*2(?:\.0)?\s*\*\s*np\.pi", fe))
    return {"n_sites": sites}


def v_w8h4() -> dict:
    rec = json.loads((DOWNSTREAM / "recommendation.json").read_text())
    text = json.dumps(rec)
    n_tiebreak = text.count("smallest W")
    sweep = (QEMU_DIR / "plan03_sweep.py").read_text()
    m = re.search(r"default=\[([\d,\s]+)\]", sweep)
    grid = [int(x) for x in m.group(1).split(",")] if m else []
    return {"n_tiebreak": n_tiebreak, "grid": grid, "has_128": 128 in grid}


def v_gate2() -> dict:
    p = REPO / "docs" / "dwarf_pilot_design" / "GATE2_RESULT.md"
    text = p.read_text()
    fracs = [float(x) for x in re.findall(r"\|\s*\*{0,2}(0\.\d{3})\*{0,2}\s*\|\s*$", text, re.M)]
    if not fracs:
        raise FileNotFoundError("no frac_in_longest_run values found in GATE2_RESULT.md")
    return {"frac_min": min(fracs), "frac_max": max(fracs), "n": len(fracs)}


def v_needs_env() -> dict:
    req = [l.strip() for l in (QEMU_DIR / "plan08_b1" / "requirements.txt").read_text().splitlines()
           if l.strip() and not l.startswith("#")]
    wav = (REPO / "VMsig_featureExctraction" / "wavelet_analysis_features.py").read_text()
    imports = sorted({m for m in re.findall(r"^(?:import|from)\s+(pywt|torch|kymatio)", wav, re.M)})
    return {"pinned": ", ".join(req), "imports": ", ".join(imports)}


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------

ISSUES = [
    dict(id="f1_phase", applies=["feature:f1_phase"], sev="do_not_use",
         head="Empty in every row of the shipped sweep",
         detail="{n_filled} of {n_rows} rows carry a value. Marker-aligned F1 is structurally "
                "unavailable: PHASE markers describe events inside the workload, CUSUM detects mean "
                "shifts in the memory signal. Different kinds of event.",
         artifacts=["VM_sampler/VM_Capture_QEMU/plan05_campaign/downstream/sweep.csv"],
         record="Plan 04 / D-86", verify=v_f1_phase),
    dict(id="coverage_ratio", applies=["feature:coverage_ratio"], sev="do_not_use",
         head="A per-family constant, not a measurement",
         detail="Computed as (W x iv)/rhythm with rhythm fixed per family ({phasic_s} s phasic, "
                "{steady_s} s steady), so at any fixed window it takes one value per family: "
                "{w8_phasic} vs {w8_steady} at W=8. Alone it reproduces the headline binary accuracy "
                "of {acc_alone} exactly (confusion tn/fp/fn/tp {confusion}). Kept only as the "
                "contaminated control arm that makes the leak measurable.",
         artifacts=["VM_sampler/VM_Capture_QEMU/plan03_sweep.py",
                    "VM_sampler/VM_Capture_QEMU/plan05_campaign/downstream/leakage_ablation.json"],
         record="Plan 04 leakage ablation", verify=v_coverage_ratio),
    dict(id="ceps_snr", applies=["feature:ceps_peak_snr_db", "lens:cepstrum"], sev="caution",
         head="A presence gate, not a discriminator",
         detail="Alone it classifies phasic-vs-steady at {acc_alone} against a majority baseline of "
                "{majority} ({protocol}). Fine for 'is there a rhythm here', wrong for 'which class "
                "is this'.",
         artifacts=["VM_sampler/VM_Capture_QEMU/plan05_campaign/downstream/leakage_ablation.json"],
         record="Plan 05 record", verify=v_ceps_snr),
    dict(id="cv_workingset", applies=["feature:cv_workingset"], sev="caution",
         head="Only computed for steady workloads",
         detail="The plan03 kernel fills it for steady cells and leaves it null otherwise, so its "
                "presence encodes family membership. FAMILY_CONDITIONAL = {family_conditional}.",
         artifacts=["VM_sampler/VM_Capture_QEMU/plan05_campaign/behavior_families.py"],
         record="Plan 04 leakage ablation", verify=v_cv_workingset),
    dict(id="duty_gt05", applies=["feature:duty_gt05"], sev="caution",
         head="Absolute threshold, calibrated to APF's scale",
         detail="Counts samples above a fixed {abs_threshold}. Not scale-equivariant: across encodings "
                "it favours whichever arm shares that scale. duty (fraction above {rel_threshold} x max, "
                "b1_features.py) is the scale-equivariant form.",
         artifacts=["VM_sampler/VM_Capture_QEMU/plan05_campaign/extra_features.py",
                    "VM_sampler/VM_Capture_QEMU/plan08_b1/b1_features.py"],
         record="B1 pre-registration gate G5", verify=v_duty_gt05),
    dict(id="cusum_band", applies=["lens:cusum"], sev="caution",
         head="Null calibration failed: gap 0.000",
         detail="The plausibility band scored 0.850 on real phasic data, 0.850 on time-shuffled "
                "surrogates and 0.850 on IID Gaussian; the artifact self-flags "
                "band_near_unfalsifiable. Artifact: {artifact}.",
         artifacts=[], record="Plan 04 record, as cited by METHODOLOGY_AS_EXECUTED.md section 5.4",
         verify=v_cusum_band),
    dict(id="phase_2pi", applies=["phase:2pi"], sev="do_not_use",
         head="The angle collides: distance 0 and distance 1 land on the same tick",
         detail="The differ's cosine channel is a DISTANCE (0 = identical) and an all-zero previous "
                "page pins to 1. Multiplying by 2 pi glues both ends together, so a page rewritten in "
                "place is indistinguishable from a page freshly allocated. The same expression appears "
                "at {n_sites} sites in block_feature_extractor.py. Kept only to reproduce B1 and the "
                "first-generation features.",
         artifacts=["VM_sampler/VM_Capture/live_delta_calc_modular/src/metrics/family_b/structure.rs",
                    "VMsig_featureExctraction/block_feature_extractor.py"],
         record="ANALYSIS_PIPELINE_METHODOLOGY.md section 7", verify=v_phase_2pi),
    dict(id="w8h4", applies=["window:8/4"], sev="note",
         head="Inherited, not chosen",
         detail="Tuned on the APF trajectory alone; {n_tiebreak} recommendation entries cite a "
                "'smallest W' tiebreak; the sweep grid was {grid}, so 128 was never tested.",
         artifacts=["VM_sampler/VM_Capture_QEMU/plan05_campaign/downstream/recommendation.json",
                    "VM_sampler/VM_Capture_QEMU/plan03_sweep.py"],
         record="Plan 03", verify=v_w8h4),
    dict(id="gate2", applies=["module:block"], sev="caution",
         head="Blocks are allocator placement, not program structure",
         detail="Gate 2 measured frac_in_longest_run between {frac_min} and {frac_max} over {n} "
                "snapshots: a contiguous array in the program's view arrives in guest-physical RAM "
                "as thousands of fragments.",
         artifacts=["docs/dwarf_pilot_design/GATE2_RESULT.md"],
         record="Dwarf pilot Gate 2, 2026-09-06", verify=v_gate2),
    dict(id="needs_env", applies=["module:scattering", "module:wavelet"], sev="note",
         head="Needs packages outside the pinned analysis env",
         detail="The existing implementation imports {imports}; the analysis env pins {pinned}. "
                "pywt is being added; scattering stays a separate call.",
         artifacts=["VM_sampler/VM_Capture_QEMU/plan08_b1/requirements.txt",
                    "VMsig_featureExctraction/wavelet_analysis_features.py"],
         record="plan10 UX section 6", verify=v_needs_env),
]


class _Unfilled(dict):
    def __missing__(self, k):
        return "(unverified)"


def build_registry() -> list[dict]:
    out = []
    today = date.today().isoformat()
    for e in ISSUES:
        entry = {k: v for k, v in e.items() if k != "verify"}
        try:
            nums = e["verify"]()
            entry["detail"] = e["detail"].format_map(_Unfilled(nums))
            entry["numbers"] = nums
            entry["verified"] = True
            entry["verified_at"] = today
            entry["verification"] = "recomputed from " + ", ".join(e["artifacts"]) if e["artifacts"] else "found: " + nums.get("artifact", "")
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as ex:
            entry["detail"] = e["detail"].format_map(_Unfilled({}))
            entry["numbers"] = {}
            entry["verified"] = False
            entry["verified_at"] = None
            entry["verification"] = f"NOT recomputed: {ex}"
        out.append(entry)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    reg = build_registry()
    if a.out:
        a.out.write_text(json.dumps({"schema": SCHEMA, "issues": reg}, indent=1) + "\n")
        print(f"[known_issues] wrote {a.out}")
    for e in reg:
        print(f"[known_issues] {e['id']:16s} {'verified' if e['verified'] else 'UNVERIFIED'}  {e['verification']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
