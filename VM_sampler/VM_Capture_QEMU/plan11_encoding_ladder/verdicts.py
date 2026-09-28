#!/usr/bin/env python3
"""verdicts.py -- the verdict vocabulary of the paper 2 toolkit (SPEC section 3.0).

Every verdict column of every result file holds exactly one of the strings below, or one of
the four parameterised forms (``not run: ...``, ``not applicable: ...``, ``refused: ...``,
``pending: ...``). Numbers never appear in a verdict column; a refusal is never a blank.

Citation: SPEC.md section 3.0; P2_STRUCTURE.md section V ("every gate can refuse and a refusal
is written to the artifact, never silently absorbed").

Additions to the SPEC 3.0 list, each from a review marked "must change before build":
  GC_ALIASED_BY_DESIGN      SPEC_review_al_kindi.md item 2 (G-C absence verdict, second cause)
  GORD_ORDER_BLIND_BY_CONSTRUCTION  SPEC_review_al_farabi.md item 2.11 (b) (the whole-cell point)
  GDEC_NO_BOUNDARY          SPEC_review_al_kindi.md item 4 (b): GDEC_NOT_RESOLVED with its reason
"""
from __future__ import annotations

PASS = "pass"
FAIL = "fail"


def not_run(reason: str) -> str:
    return f"not run: {reason}"


def not_applicable(reason: str) -> str:
    return f"not applicable: {reason}"


def refused(reason: str) -> str:
    return f"refused: {reason}"


def pending(reason: str) -> str:
    return f"pending: {reason}"


# G1
TREND_PRESENT = "trend present"
# G2
G2_ABOVE_NYQUIST = "not applicable, rhythm above Nyquist"
G2_UNDETERMINED = "undetermined by the interval calibration"
# G-P
GP_RESOLVABLE = "resolvable"
GP_MARGINAL = "marginal"
GP_ALIASED = "aliased by design at this size"
GP_UNDECLARED = "undeclared"
GP_RHYTHM_UNDERSAMPLED = "rhythm under-sampled"
GP_PASS_ALIASED = "pass aliased"
GP_ADMITTED = "admitted"
# G3 flag
G3_PRESENT = "rhythm flag: present"
G3_ABSENT = "rhythm flag: absent"
# G-ORD
GORD_ORDER_BLIND = "order-blind"
GORD_RESOLUTION = "resolution"
GORD_ORDER_BLIND_BY_CONSTRUCTION = "order-blind (by construction)"
# G-K0
GK0_IDLE_MEASURED = "IDLE, measured"
GK0_ABOVE_FLOOR = "above floor"
# G-F
GF_INSEPARABLE = "inseparable at floor"
GF_VOID = "void: idle reps separable under this rung"
GF_AT_FLOOR = "at floor in this lead"
# G-C
GC_DISCONNECTED = "disconnected lead"
GC_ALIASED_BY_DESIGN = "not applicable: pulse aliased by design (full footprint lit every snapshot)"
# G-J
GJ_INTERPRETABLE = "interpretable"
GJ_FLOOR_OVERLAP = "floor overlap"
GJ_FLOOR_UNMEASURED = "floor unmeasured"
# G-DEC
GDEC_DECAY = "decay"
GDEC_NO_DECAY = "no decay"
GDEC_NO_BEYOND_BREADTH = "no decay beyond breadth"
GDEC_NO_BEYOND_FLOOR = "no decay beyond floor or host"
GDEC_NOT_RESOLVED = "decay not resolved"
GDEC_NO_BOUNDARY = "decay not resolved: no pass boundary"
# B1-G1
NEAR_UNFALSIFIABLE = "near_unfalsifiable"
# G-L
GL_LEVEL_ONLY = "level only"
GL_SHOT_NOISE = "refused: shot noise explains CV"
# G-N
GN_HEADLINE = "headline"
GN_ONE_TRAIN = "one training kernel per fold"
GN_NOVELTY = "structural novelty"
GN_NO_ROW = "no kernel row"
# G-X
GX_POOLING_STANDS = "pooling stands"
GX_LEAK = "campaign predictable"
GX_CONFOUND_NONE = "confound: none"
GX_CONFOUND_PARTIAL = "confound: partial"
GX_CONFOUND_TOTAL = "confound: total"
# G-DIM
GDIM_FULL = "full vector"
GDIM_REDUCED = "declared reduction"
# G-M
GM_BEATS = "beats"
GM_DIFFERENCE = "difference with margin"
# G-V
GV_NOT_ESTIMABLE = "LOKO not estimable"
GV_ESTIMABLE = "estimable"
# Alias falsifier
ALIAS_MOVES = "moves with the interval"
ALIAS_STAYS = "does not move with the interval"

# ---------------------------------------------------------------------------------------------
# The detection layer (SPEC_DETECTION.md section 3.0; build epoch 2, builder A). Appended without
# changing any string above; the new named refusals are added to NAMED_REFUSALS below.
# ---------------------------------------------------------------------------------------------
# the two-class null (CR3 2.1; ML 1.4)
NULL_NOT_ESTIMABLE = "null_not_estimable"
NULL_INSIDE = "inside the workload-level null"
# G-OP (CR3 2.13; ML 2.3, 4.3)
GOP_SUPPORTED = "operating point supported"
GOP_SET_BY_FEW = "set by one workload, family named"
# G-LM (CR3 2.17; ML 4.3; K3 F1)
GLM_SURVIVES = "detection survives level matching"
GLM_LEVEL_ONLY = "detected by level in this lead"
GLM_NOT_DETECTED = "not applicable: not detected at the operating point"
GLM_EMPTY_BAND = "not applicable: no benign workload within the level band"
GLM_LABEL_MEDIAN_K = "median K, stage 1"
# G-ANCHOR and the order test (CR3 2.14, 2.20; ML 3.1, 3.2; K3 F5; P3 0a)
GANCHOR_AUDIBLE = "campaign audible"
GANCHOR_NOT_AUDIBLE = "campaign not audible above the null"
ORDER_AUDIBLE = "position audible"
ORDER_NOT_AUDIBLE = "position not audible above the null"
ORDER_VOID = "void: position audible inside the interleaved campaign"
DRIFT_SLOPE = "drift: slope above the shuffle null"
DRIFT_NONE = "no drift above the shuffle null"
# G-SIG (CR3 2.16; K3 F2)
GSIG_REPORTED = "gap reported"
GSIG_IDENTITY = "recognizes workload identity, not family behaviour"
# G-FP (CR3 2.15; K3 F8)
GFP_ATTRIBUTED = "attributed"
GFP_INSEPARABLE = "inseparable from sandbox under this rung"
# G-1C (CR3 2.19; K3 section 5 point 1)
G1C_PRIMARY = "primary"
G1C_SECONDARY = "secondary, not citable"
G1C_SEARCH = "refused: a second one-class model without a declared primary reads as a search"
# the harness clause (CR3 2.12, 2.21; K3 F4)
HARNESS_STAGE2_ABSENT = "not run: stage 2 absent"
HARNESS_COMPARABLE = "harness's (comparable margins)"
HARNESS_CLASS_EXCEEDS = "class margin exceeds the harness margin"
HARNESS_RELAUNCH_NOT_CLASS = "re-launch, not class"
# G-CAL (CR3 2.18; ML 1.6 item 2)
GCAL_AGREE = "per-fold and pooled agree within the null spread"
GCAL_PERFOLD = "per-fold reported; pooled curve threshold set post hoc on the held-out scores"
POST_HOC_LABEL = "threshold set post hoc on the held-out scores"
# G-K0 two-class (CR3 2.8; K3 F3)
GK0_AT_FLOOR = "at floor"
GK0_AT_HARNESS_FLOOR = "at harness floor"
GK0_MIXED = "cells in more than one verdict"
# G-N two-class (CR3 2.5)
GN_ONE_TRAIN_WORKLOAD = "one training workload per fold"
GN_NO_SUPERVISED = "no supervised headline"
GN_SINGLE_WORKLOAD = "single workload"
# levels 2 and 3 (P3 0a; G-N)
L2_ONE_TRAIN = "one training member per fold"
SIGNATURE_CEILING = "signature ceiling"


def level2_no_heldout(n: int) -> str:
    """The level-2 row of a sub-family too small to hold a member out (P3 0a: sub-family B reads
    "one member, no held-out test")."""
    return "one member, no held-out test" if n == 1 else f"{n} members, no held-out test"


# the time-to-detect ladder (K3 move 17; CR3 2.29)
LADDER_FROM_PAIR1_ONLY = "from pair 1 only"
# the miss table (K3 move 18)
AT_FLOOR_NOT_A_MISS = "at floor, not a miss"
# the leak probes (SPEC_DETECTION_review_ml.md 2.6; ML 3.4, 3.5) and the cadence row (al-Kindi 2.9)
LEAK_AUDIBLE = "leak audible"
LEAK_NOT_AUDIBLE = "leak not audible above the null"
CADENCE_AUDIBLE = "cadence audible"
# the rep-identity disclosure (SPEC_DETECTION_review_ml.md 2.8)
REPS_NEAR_IDENTICAL = "reps near identical: LOCO reads as within-trace"
# stage 1 placeholders (SPEC_DETECTION.md preamble)
CROSS_CAMPAIGN_STAGE1 = "not applicable: stage 1 (the 01c cells are the benign class)"
RUNG2P_NOT_BUILT = "not run: content family columns not in the plan11 extract (rung 2' needs an extract extension)"
COMPARATOR_ELSEWHERE = "not run: comparator row from another session (RQ5)"
YIELD_NOT_RECORDED = "not run: no vmstat record (stage 1)"

NAMED_REFUSALS = frozenset({
    TREND_PRESENT, G2_ABOVE_NYQUIST, G2_UNDETERMINED, GF_VOID, GF_AT_FLOOR,
    GC_DISCONNECTED, GDEC_NO_BEYOND_BREADTH, GDEC_NO_BEYOND_FLOOR, GDEC_NOT_RESOLVED,
    GDEC_NO_BOUNDARY, NEAR_UNFALSIFIABLE, GL_LEVEL_ONLY, GL_SHOT_NOISE, GV_NOT_ESTIMABLE,
    # the detection layer (SPEC_DETECTION.md 3.0)
    NULL_NOT_ESTIMABLE, NULL_INSIDE, GOP_SET_BY_FEW, GLM_LEVEL_ONLY, ORDER_VOID, GSIG_IDENTITY,
    GFP_INSEPARABLE, G1C_SEARCH, HARNESS_RELAUNCH_NOT_CLASS, GK0_AT_FLOOR, GK0_AT_HARNESS_FLOOR,
    GN_NO_SUPERVISED,
})


class Refusal(str):
    """A verdict string that is a refusal. ``str`` subclass so it writes like any verdict."""

    def __new__(cls, s: str):
        return str.__new__(cls, s)


def is_refusal(s) -> bool:
    """True for FAIL, any ``refused:`` / ``not run:`` / ``not applicable:`` string, and the named
    refusals of SPEC 3.0 (``TREND_PRESENT`` ... ``GV_NOT_ESTIMABLE``). A verdict that carries a
    suffix (``" (INFERRED)"``, ``" (floor unmeasured)"``) is judged on its head."""
    if s is None:
        return False
    t = str(s)
    head = t.split(" (", 1)[0]
    if t == FAIL or head == FAIL:
        return True
    if t.startswith("refused:") or t.startswith("not run:") or t.startswith("not applicable"):
        return True
    return t in NAMED_REFUSALS or head in NAMED_REFUSALS
