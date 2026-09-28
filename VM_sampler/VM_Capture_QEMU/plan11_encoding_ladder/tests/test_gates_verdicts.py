"""verdicts.py: the vocabulary and is_refusal (SPEC 3.0)."""
from _b2_common import V


def test_named_refusals_are_refusals():
    for s in (V.FAIL, V.TREND_PRESENT, V.G2_ABOVE_NYQUIST, V.G2_UNDETERMINED, V.GF_VOID, V.GF_AT_FLOOR, V.GC_DISCONNECTED,
              V.GDEC_NO_BEYOND_BREADTH, V.GDEC_NO_BEYOND_FLOOR, V.GDEC_NOT_RESOLVED, V.NEAR_UNFALSIFIABLE, V.GL_LEVEL_ONLY,
              V.GL_SHOT_NOISE, V.GV_NOT_ESTIMABLE, V.not_run("x"), V.not_applicable("x"), V.refused("x")):
        assert V.is_refusal(s), s


def test_passes_and_labels_are_not_refusals():
    for s in (V.PASS, V.GP_RESOLVABLE, V.GP_UNDECLARED, V.G3_PRESENT, V.GORD_ORDER_BLIND, V.GK0_IDLE_MEASURED, V.GJ_INTERPRETABLE,
              V.GDEC_DECAY, V.GDEC_NO_DECAY, V.GN_HEADLINE, V.GX_POOLING_STANDS, V.GM_DIFFERENCE, V.pending("x"), V.ALIAS_MOVES):
        assert not V.is_refusal(s), s


def test_suffix_judged_on_head():
    assert V.is_refusal(V.GDEC_NO_BEYOND_FLOOR + " (floor unmeasured)")
    assert not V.is_refusal(V.GP_RESOLVABLE + " (INFERRED)")
    assert isinstance(V.Refusal(V.GF_VOID), str)
