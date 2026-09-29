#!/usr/bin/env python3
"""splits.py -- grouped train/test folds over a dataset's keys. B1's b1_splits.py, generalised
to the keys every run writes (recording, workload, family, campaign, t_index), model-agnostic:
folds are index arrays, the model decides what to do with the training rows.

  within_trace  per recording, the last `test_frac` of its tiles (by t_index) is test, the rest
                train. ONE fold. A memorisation CEILING (the same recording is on both sides),
                never the headline.
  loro          leave-one-recording-out: one fold per recording. "Unseen run."
  lowo          leave-one-workload-out: one fold per workload. "Unseen workload", the headline.
                A family with a single workload gets no same-family training on its own fold:
                structurally a novelty case, flagged in `novelty`, not hidden.
  loco          leave-one-campaign-out: one fold per campaign (the part of the recording id
                after rep001__). Tests whether a model learned the campaign, since the four
                campaigns were captured at different times.

No group straddles a split; `assert_grouped` proves it and the executor runs it.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

SPLITS = {
    "within_trace": "within-trace tail: the memorisation ceiling, never the headline",
    "loro": "leave one recording out",
    "lowo": "leave one workload out: the honest headline",
    "loco": "leave one campaign out: did the model learn the campaign?",
}


def campaign_of(recording: str) -> str:
    """`family/workload/variant/rep001__campaign` -> `campaign`; the whole tail when there is no __."""
    tail = str(recording).split("/")[-1]
    return tail.split("__", 1)[1] if "__" in tail else tail


def _groups(keys) -> "OrderedDict[str, np.ndarray]":
    out: OrderedDict = OrderedDict()
    for i, k in enumerate(keys):
        out.setdefault(str(k), []).append(i)
    return OrderedDict((k, np.asarray(v, dtype=np.int64)) for k, v in out.items())


def fold_within_trace(keys: dict, test_frac: float = 0.2) -> list[dict]:
    tr, te = [], []
    t_index = np.asarray(keys["t_index"])
    for rec, idx in _groups(keys["recording"]).items():
        order = idx[np.argsort(t_index[idx], kind="stable")]
        n = len(order)
        n_test = max(1, int(round(test_frac * n)))
        n_test = min(n_test, n - 1) if n > 1 else 0
        te.extend(order[n - n_test:].tolist())
        tr.extend(order[:n - n_test].tolist())
    return [{"name": "within_trace", "split": "within_trace", "held_out": None, "novelty": False,
             "train": np.asarray(sorted(tr), dtype=np.int64), "test": np.asarray(sorted(te), dtype=np.int64)}]


def _leave_one_out(keys: dict, group_key: str, split: str, novelty_key: str = "family") -> list[dict]:
    """novelty_key: the label whose absence from the training rows makes a fold a novelty case;
    family by default, the archetype when that is the target (an archetype with one kernel has
    no same-archetype training under leave-one-kernel-out)."""
    n = len(keys["recording"])
    fam = np.asarray(keys[novelty_key])
    fam_groups: dict[str, set] = {}
    for g, idx in _groups(keys[group_key]).items():
        fam_groups.setdefault(str(fam[idx[0]]), set()).add(g)
    folds = []
    for g, idx in _groups(keys[group_key]).items():
        f = str(fam[idx[0]])
        train = np.setdiff1d(np.arange(n), idx, assume_unique=False)
        novelty = not bool((fam[train] == f).any())
        folds.append({"name": f"{split}/{g}", "split": split, "held_out": g, "family": f, "novelty": novelty,
                      "train": train, "test": idx})
    return folds


def fold_loro(keys: dict, novelty_key: str = "family") -> list[dict]:
    return _leave_one_out(keys, "recording", "loro", novelty_key)


def fold_lowo(keys: dict, novelty_key: str = "family") -> list[dict]:
    return _leave_one_out(keys, "workload", "lowo", novelty_key)


def fold_loco(keys: dict, novelty_key: str = "family") -> list[dict]:
    k = dict(keys)
    k["campaign"] = np.asarray([campaign_of(r) for r in keys["recording"]])
    return _leave_one_out(k, "campaign", "loco", novelty_key)


def folds_for(keys: dict, split: str, test_frac: float = 0.2, novelty_key: str = "family") -> list[dict]:
    if split == "within_trace":
        return fold_within_trace(keys, test_frac)
    if split == "loro":
        return fold_loro(keys, novelty_key)
    if split == "lowo":
        return fold_lowo(keys, novelty_key)
    if split == "loco":
        return fold_loco(keys, novelty_key)
    raise ValueError(f"unknown split {split!r}; one of {', '.join(SPLITS)}")


def assert_grouped(folds: list[dict], keys: dict, group_key: str | None):
    """No group straddles train and test, and no row is on both sides."""
    for f in folds:
        assert len(np.intersect1d(f["train"], f["test"])) == 0, f"{f['name']}: train/test row overlap"
        if group_key:
            g = np.asarray(keys[group_key]) if group_key != "campaign" else np.asarray([campaign_of(r) for r in keys["recording"]])
            overlap = set(g[f["test"]].tolist()) & set(g[f["train"]].tolist())
            assert not overlap, f"{f['name']}: {group_key} leaks across the split: {sorted(overlap)[:5]}"


GROUP_OF = {"within_trace": None, "loro": "recording", "lowo": "workload", "loco": "campaign"}


def split_label(split: str, target: str = "family") -> str:
    """What a split means under a target. With target archetype every kernel is one workload, so
    leave-one-workload-out holds out all the runs of one kernel: leave one kernel out (LOKO)."""
    if target == "archetype" and split == "lowo":
        return "leave one kernel out: every run of one kernel held out"
    return SPLITS[split]


def summary(keys: dict, splits: list[str], test_frac: float = 0.2) -> dict:
    out = {}
    for s in splits:
        fs = folds_for(keys, s, test_frac)
        assert_grouped(fs, keys, GROUP_OF[s])
        out[s] = {"n_folds": len(fs), "novelty": [f["held_out"] for f in fs if f.get("novelty")],
                  "train_sizes": [int(f["train"].size) for f in fs], "test_sizes": [int(f["test"].size) for f in fs]}
    return out
