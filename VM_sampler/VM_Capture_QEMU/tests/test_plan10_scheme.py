#!/usr/bin/env python3
"""scheme.py: the rules survive the move out of the browser.

Runs the three worked examples and a set of deliberately broken schemes
through the Python validator over a synthetic manifest (so the test does not
depend on the migrated corpus existing) and checks that everything the UX
document types as hard refuses, everything typed as soft warns and blocks
until acknowledged, and that a scheme in the canvas mockup's format loads.

Run:  python3 tests/test_plan10_scheme.py
      pytest tests/test_plan10_scheme.py
"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

QEMU_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QEMU_DIR))
from plan10_analysis import channel_roster, corpus_manifest, scheme as S  # noqa: E402
from plan10_analysis.modules import build_modules  # noqa: E402


def _ctx(n_pairs=(700, 700, 690), substrate=False):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "zstd_local"
        for i, n in enumerate(n_pairs):
            wl = ["mem_alpha_v2", "mem_alpha_v2", "cpu_beta_v2"][i]
            fam = wl.split("_", 1)[0]
            d = root / fam / wl / f"args_--duration_450_--seed_{i}" / f"rep00{i + 1}__t"
            d.mkdir(parents=True)
            for j in range(n + 1):
                (d / f"{j:06d}.zst").write_bytes(b"x")
        manifest = corpus_manifest.scan(root)
    if substrate:
        for r in manifest["recordings"]:
            r["has"]["substrate_csv"] = True
    return S.Context(channel_roster.build_roster(), manifest, build_modules(), S.load_config())


def _ids(issues, sev=None):
    return [i.get("id") or i["msg"] for i in issues if sev is None or i["sev"] == sev]


def test_examples_over_a_chain_only_corpus():
    ctx = _ctx()
    ex = S.make_examples(ctx.manifest)
    for k in ("b1", "complex", "plv"):
        issues = S.validate(ex[k], ctx)
        code, v = S.verdict(issues, ex[k])
        assert code == 2, (k, issues)
        assert _ids(issues, "hard") == []
        assert set(_ids(issues, "soft")) == {"no_substrate"}, (k, _ids(issues, "soft"))
    # acknowledging the one warning makes each valid
    for k in ("b1", "complex", "plv"):
        s = copy.deepcopy(ex[k])
        s["acknowledged"] = [{"id": "no_substrate", "at": "2026-09-08T00:00:00Z", "note": "chains only on this machine"}]
        assert S.verdict(S.validate(s, ctx), s)[0] == 0


def test_examples_over_a_substrate_corpus_are_clean():
    ctx = _ctx(substrate=True)
    ex = S.make_examples(ctx.manifest)
    for k in ("b1", "complex", "plv"):
        code, _ = S.verdict(S.validate(ex[k], ctx), ex[k])
        assert code == 0, k


def test_effective_n_is_workloads_not_windows():
    ctx = _ctx(substrate=True)
    est = S.estimate(S.make_examples(ctx.manifest)["b1"], ctx)
    assert est["effective_n_workloads"] == 2
    assert est["windows"] > 500 and est["tiles"] == est["windows"]


def _break(ctx, k, mutate):
    s = copy.deepcopy(S.make_examples(ctx.manifest)[k])
    mutate(s)
    return S.validate(s, ctx)


def test_hard_rules():
    ctx = _ctx(substrate=True)

    def node(s, i):
        return next(n for n in s["nodes"] if n["id"] == i)

    # PLV fed by a real (non-complex) window: hard
    def plv_real(s):
        s["pipes"] = [p for p in s["pipes"] if p["to"] != ["n5", "in"]]
        s["pipes"].append({"from": ["n2", "field"], "to": ["n5", "in"]})
    iss = _break(ctx, "plv", plv_real)
    assert any(i["sev"] == "hard" and i["node"] == "n7" and "complex" in i["msg"] for i in iss), iss

    # a channel that is zero at the assumed speed: hard, and the message says the speed is assumed
    iss = _break(ctx, "b1", lambda s: node(s, "n2")["params"].update(chans=["ncd"]))
    h = [i for i in iss if i["sev"] == "hard" and i["node"] == "n2"]
    assert h and "ncd" in h[0]["msg"] and "assumed" in h[0]["msg"], h

    # window longer than the shortest recording: hard
    iss = _break(ctx, "b1", lambda s: node(s, "n4")["params"].update(w=2000, h=1000))
    assert any(i["sev"] == "hard" and "zero windows" in i["msg"] for i in iss)

    # spectral lens on W=2: hard
    def short(s):
        node(s, "n5")["params"].update(w=2, h=1)
    iss = _break(ctx, "complex", short)
    assert any(i["sev"] == "hard" and "fewer than two bins" in i["msg"] for i in iss)

    # complex without a phase convention: hard
    iss = _break(ctx, "complex", lambda s: node(s, "n4")["params"].update(phase=""))
    assert any(i["sev"] == "hard" and "phase convention" in i["msg"] for i in iss)

    # wrong port type and a cycle: hard, before any module rule runs
    iss = _break(ctx, "b1", lambda s: s["pipes"].append({"from": ["n1", "cells"], "to": ["n4", "in"]}))
    assert any(i["sev"] == "hard" and "cannot enter port" in i["msg"] for i in iss)
    iss = _break(ctx, "b1", lambda s: s["pipes"].append({"from": ["n5", "out"], "to": ["n2", "cells"]}))
    assert any(i["sev"] == "hard" for i in iss)

    # an unknown recording id (a mockup-era scheme selected workload names): hard, not a crash
    iss = _break(ctx, "b1", lambda s: node(s, "n1")["params"].update(sel=["mem_workingset_sweep"]))
    assert any(i["sev"] == "hard" and "not in the manifest" in i["msg"] for i in iss)

    # a dangling pipe is caught structurally, before any module rule runs
    iss = _break(ctx, "b1", lambda s: s["nodes"].pop())
    assert any(i["sev"] == "hard" and "missing node" in i["msg"] for i in iss)

    # no Write module (node and its pipe removed): hard
    def no_write(s):
        s["nodes"] = [n for n in s["nodes"] if n["module"] != "write"]
        s["pipes"] = [p for p in s["pipes"] if p["to"][0] != "n6"]
    iss = _break(ctx, "b1", no_write)
    assert any(i["sev"] == "hard" and "Write" in i["msg"] for i in iss)


def test_soft_rules_and_acknowledgment():
    ctx = _ctx(substrate=True)

    def node(s, i):
        return next(n for n in s["nodes"] if n["id"] == i)

    # 2pi: soft, id phase_2pi
    iss = _break(ctx, "complex", lambda s: node(s, "n4")["params"].update(phase="2pi"))
    assert "phase_2pi" in _ids(iss, "soft")

    # a family-C channel as the direction: soft
    iss = _break(ctx, "complex", lambda s: node(s, "n3")["params"].update(chans=["ent_q"]))
    assert "direction_not_direction" in _ids(iss, "soft")

    # MSC needs a tile long enough to hold its internal windows, and ONE channel is enough:
    # it is self-coherence between adjacent windows, not a channel pair (an earlier rule here
    # demanded two channels and was wrong).
    def msc(s, **params):
        s["nodes"].append({"id": "n9", "module": "msc", "params": params, "x": 0, "y": 0})
        s["pipes"].append({"from": ["n4", "out"], "to": ["n9", "in"]})
    mine = lambda iss: [i for i in iss if i["node"] == "n9"]
    # the b1 example windows at 8 frames; the shipped 128/64 needs 256
    hard = [i for i in mine(_break(ctx, "b1", msc)) if i["sev"] == "hard"]
    assert hard and "at least 256 samples" in hard[0]["msg"], hard
    assert not any("pairwise" in i["msg"] for i in mine(_break(ctx, "b1", msc)))
    # internal windows that fit the tile: no issue at all, on a single channel
    assert mine(_break(ctx, "b1", lambda s: msc(s, iw=4, ih=2))) == []
    # a nonsense internal window is refused
    assert any(i["sev"] == "hard" for i in mine(_break(ctx, "b1", lambda s: msc(s, iw=1, ih=0))))

    # a flagged feature: soft, and blocks until acknowledged
    s = copy.deepcopy(S.make_examples(ctx.manifest)["b1"])
    node(s, "n5")["params"]["feats"] = ["mean", "duty_gt05"]
    iss = S.validate(s, ctx)
    assert "feat:duty_gt05" in _ids(iss, "soft")
    assert S.verdict(iss, s)[0] == 2
    s["acknowledged"] = [{"id": "feat:duty_gt05", "at": "x"}]
    assert S.verdict(S.validate(s, ctx), s)[0] == 0

    # a stale acknowledgment is reported, never silently kept
    s["acknowledged"].append({"id": "gate2", "at": "x"})
    assert S.verdict(S.validate(s, ctx), s)[1]["acknowledgments_without_issue"] == ["gate2"]


def test_cli_exit_codes(tmp_path=None):
    ctx = _ctx()
    with tempfile.TemporaryDirectory() as td:
        mp = Path(td) / "m.json"
        mp.write_text(json.dumps(ctx.manifest))
        sp = Path(td) / "s.json"
        s = S.make_examples(ctx.manifest)["b1"]
        sp.write_text(json.dumps(s))
        import subprocess
        r = subprocess.run([sys.executable, str(QEMU_DIR / "plan10_analysis" / "scheme.py"), "validate", str(sp), "--manifest", str(mp)],
                           capture_output=True, text=True)
        assert r.returncode == 2, r.stdout + r.stderr
        s["acknowledged"] = [{"id": "no_substrate", "at": "x"}]
        sp.write_text(json.dumps(s))
        r = subprocess.run([sys.executable, str(QEMU_DIR / "plan10_analysis" / "scheme.py"), "validate", str(sp), "--manifest", str(mp)],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
