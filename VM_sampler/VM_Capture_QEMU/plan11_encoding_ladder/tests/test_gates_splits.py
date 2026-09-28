"""splits.py: the copied fold functions against the originals, grouping (SPEC 4.1)."""
import importlib.util
import numpy as np
import pytest

from _b2_common import S, PKG, corpus
from plan11_encoding_ladder import splits as SP


def _lab():
    out = corpus(reps=2, idle=1, kernels=["gemm", "gibbs", "fft"], n_pairs=60)
    f = S.load_features(S.build_features(out, None, "apf", 8, 4, True))
    return SP.make_labels(f, f["role"] == "kernel"), f


def test_folds_equal_original_b1_splits():
    orig = PKG.parent / "plan08_b1" / "b1_splits.py"
    if not orig.is_file():
        pytest.skip("plan08_b1/b1_splits.py not importable here")
    spec = importlib.util.spec_from_file_location("b1_splits_orig", orig)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    lab, _ = _lab()
    olab = {"n": lab["n"], "cell_id": lab["cell_id"], "workload": lab["kernel"], "family": lab["archetype"],
            "rep": lab["rep"], "win_start": lab["win_start"]}
    for mine, theirs in ((SP.fold_within_trace(lab), mod.fold_within_trace(olab)), (SP.fold_loro(lab), mod.fold_loro(olab)),
                         (SP.fold_loko(lab), mod.fold_lowo(olab))):
        assert len(mine) == len(theirs)
        for a, b in zip(mine, theirs):
            assert np.array_equal(a["train"], b["train"]) and np.array_equal(a["test"], b["test"])


def test_no_cell_straddles_train_and_test():
    lab, _ = _lab()
    for split in ("loro", "loko"):
        folds = SP.folds_for(split, lab)
        for f in folds:
            assert not (set(lab["cell_id"][f["train"]]) & set(lab["cell_id"][f["test"]]))
    assert len(SP.folds_for("loro", lab)) == 6 and len(SP.folds_for("loko", lab)) == 3
    assert len(SP.folds_for("loro", lab, loro_mode="rep_index")) == 2
    wt = SP.folds_for("within_trace", lab)[0]
    for c in set(lab["cell_id"]):
        m = lab["cell_id"] == c
        assert lab["win_start"][wt["test"]][lab["cell_id"][wt["test"]] == c].min() > lab["win_start"][wt["train"]][lab["cell_id"][wt["train"]] == c].max()
    nov = [f for f in SP.fold_loko(lab) if f["novelty"]]
    assert {f["held_out"] for f in nov} == {"fft"}
