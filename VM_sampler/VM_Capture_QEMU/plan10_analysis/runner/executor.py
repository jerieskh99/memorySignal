#!/usr/bin/env python3
"""executor.py -- run a validated scheme over a corpus; write features, status and sidecar.

    python3 -m plan10_analysis.runner.executor run SCHEME.json --out-dir OUT
        [--source-json '{"kind":"local","root":...}'] [--store L1] [--speed N]
        [--max-pairs N] [--acknowledge-all]

Refuses a scheme that scheme.py refuses (hard: exit 1) or that carries unacknowledged
warnings (exit 2). The run then proceeds in three phases:

  extract   walk each recording's chain once and keep the union of every Channels
            module's columns (L1 store; reused across runs)
  per-recording   evaluate every module that does not cross recordings, keep only the
            small outputs (series, tiles, features), drop the field
  cross-recording Baseline, PLV, Concat, Write over the per-recording outputs

status.json is rewritten atomically after every pair and every node; control.json is
read between pairs ({"command": "stop" | "pause" | "run"}). The output directory gets
features.npz (X, feature_names, tile_keys), features.csv, sidecar.json and run.log.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
QEMU_DIR = HERE.parent.parent
if str(QEMU_DIR) not in sys.path:
    sys.path.insert(0, str(QEMU_DIR))

from plan10_analysis import channel_roster, corpus_manifest, scheme as S  # noqa: E402
from plan10_analysis.modules import build_modules                         # noqa: E402
from plan10_analysis.sources import make_source                           # noqa: E402
from plan10_analysis.runner import extract, stages, differ                # noqa: E402

CROSS = {"baseline", "plv", "concat", "write"}


class Stop(Exception):
    pass


class RunRefused(RuntimeError):
    def __init__(self, code: int, verdict: dict):
        super().__init__(f"scheme refused (exit {code})")
        self.code = code
        self.verdict = verdict


# ---------------------------------------------------------------------------
# status + control
# ---------------------------------------------------------------------------

class Status:
    def __init__(self, out_dir: Path, label: str):
        self.path = out_dir / "status.json"
        self.control = out_dir / "control.json"
        self.log = out_dir / "run.log"
        self.d = {"schema": "plan10.run_status.v1", "label": label, "state": "starting", "phase": None,
                  "recording": None, "recording_index": 0, "n_recordings": 0, "pair": 0, "n_pairs": 0,
                  "node": None, "message": "", "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "updated_at": None, "errors": [], "per_recording": {}}
        self.write()

    def write(self, **kw):
        self.d.update(kw)
        self.d["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.d, indent=1))
        os.replace(tmp, self.path)

    def logline(self, msg: str):
        line = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {msg}\n"
        with self.log.open("a") as f:
            f.write(line)
        print(msg, flush=True)

    def check_control(self):
        if not self.control.exists():
            return
        try:
            cmd = json.loads(self.control.read_text()).get("command")
        except (OSError, json.JSONDecodeError):
            return
        if cmd == "stop":
            raise Stop()
        while cmd == "pause":
            self.write(state="paused")
            time.sleep(1.0)
            try:
                cmd = json.loads(self.control.read_text()).get("command")
            except (OSError, json.JSONDecodeError):
                cmd = "run"
            if cmd == "stop":
                raise Stop()
        if self.d["state"] == "paused":
            self.write(state="running")


# ---------------------------------------------------------------------------
# graph helpers
# ---------------------------------------------------------------------------

def topo(nodes: dict, pipes: list[dict]) -> list[str]:
    indeg = {n: 0 for n in nodes}
    for p in pipes:
        indeg[p["to"][0]] += 1
    order, ready = [], sorted(n for n, d in indeg.items() if d == 0)
    while ready:
        n = ready.pop(0)
        order.append(n)
        for p in pipes:
            if p["from"][0] == n:
                indeg[p["to"][0]] -= 1
                if indeg[p["to"][0]] == 0:
                    ready.append(p["to"][0])
                    ready.sort()
    if len(order) != len(nodes):
        raise ValueError("cycle")
    return order


def cross_set(nodes: dict, pipes: list[dict]) -> set[str]:
    """Nodes evaluated over all recordings at once: baseline/plv/write and their descendants."""
    cross = {n for n, v in nodes.items() if v["module"] in ("baseline", "plv", "write")}
    changed = True
    while changed:
        changed = False
        for p in pipes:
            if p["from"][0] in cross and p["to"][0] not in cross:
                cross.add(p["to"][0])
                changed = True
    cross |= {n for n, v in nodes.items() if v["module"] == "concat" and any(p["to"][0] == n and p["from"][0] in cross for p in pipes)}
    return cross


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def run(scheme_path: Path, out_dir: Path, source_spec: dict | None = None, store: Path | None = None,
        speed: int | None = None, max_pairs: int | None = None, acknowledge_all: bool = False,
        manifest_path: Path | None = None) -> int:
    scheme_path = Path(scheme_path)
    sch = json.loads(scheme_path.read_text())
    out_dir = Path(out_dir) / sch.get("label", "scheme")
    out_dir.mkdir(parents=True, exist_ok=True)
    st = Status(out_dir, sch.get("label", "scheme"))
    try:
        return _run(sch, scheme_path, out_dir, st, source_spec, store, speed, max_pairs, acknowledge_all, manifest_path)
    except Stop:
        st.write(state="stopped", message="stopped by control.json")
        st.logline("stopped")
        return 130
    except RunRefused as e:
        st.write(state="refused", message=str(e), verdict=e.verdict)
        st.logline(f"refused: {json.dumps(e.verdict)[:500]}")
        return e.code
    except Exception as e:  # noqa: BLE001  -- the status file must say what happened
        st.write(state="failed", message=f"{type(e).__name__}: {e}")
        st.logline("failed:\n" + traceback.format_exc())
        return 1


def _run(sch, scheme_path, out_dir, st, source_spec, store, speed, max_pairs, acknowledge_all, manifest_path) -> int:
    t0 = time.time()
    # ---- context: roster, manifest (from the source), modules, config
    src = make_source(source_spec or {"kind": "local", "root": str(corpus_manifest.default_root())})
    roster = channel_roster.build_roster()
    manifest = json.loads(Path(manifest_path).read_text()) if manifest_path else corpus_manifest.scan_source(src)
    ctx = S.Context(roster, manifest, build_modules(), S.load_config())
    speed = int(speed if speed is not None else ctx.config["substrateSpeed"])
    st.logline(f"source {src.describe()}; {manifest['n_recordings']} recordings; speed {speed}")

    # ---- validate exactly as the console does
    issues = S.validate(sch, ctx)
    if acknowledge_all:
        sch = dict(sch, acknowledged=list(sch.get("acknowledged", [])) + [
            {"id": i["id"], "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "note": "acknowledge-all at launch"}
            for i in issues if i["sev"] == "soft" and i.get("id") and i["id"] not in {a.get("id") for a in sch.get("acknowledged", [])}])
    code, verdict = S.verdict(issues, sch)
    if code:
        raise RunRefused(code, {k: v for k, v in verdict.items() if k in ("hard", "soft_unacknowledged")})
    st.write(state="running", phase="plan", verdict={"notes": [n["msg"] for n in verdict["notes"]]})

    g = S.Graph(sch, ctx)
    order = topo(g.nodes, g.pipes)
    cross = cross_set(g.nodes, g.pipes)
    mod_of = {n: g.nodes[n]["module"] for n in g.nodes}

    # ---- recordings and the union of channels
    cells_nodes = [n for n in order if mod_of[n] == "cells"]
    recs: dict[str, dict] = {}
    for n in cells_nodes:
        p = g.params(n)
        for rid in p.get("sel", []):
            r = ctx.rec_by_id.get(rid)
            if r and r["has"]["chain"] and r["n_pairs"] >= int(p.get("min_pairs", 0)):
                recs[rid] = r
        mp = int(p.get("max_pairs") or 0)
        if mp:
            max_pairs = min(max_pairs or mp, mp)
    rec_ids = sorted(recs)
    union: set[str] = set()
    for n in order:
        if mod_of[n] == "channels":
            union |= set(g.params(n).get("chans", []))
    union |= {"hamming"}
    st.write(n_recordings=len(rec_ids), channels=sorted(union), max_pairs=max_pairs)
    st.logline(f"{len(rec_ids)} recordings, channels {sorted(union)}, max_pairs {max_pairs}")
    store_dir = extract.store_dir(store)

    # ---- phase 1: extract (L1)
    stores: dict[str, Path] = {}
    for i, rid in enumerate(rec_ids, 1):
        st.check_control()
        st.write(phase="extract", recording=rid, recording_index=i, pair=0, n_pairs=min(recs[rid]["n_pairs"], max_pairs or 10 ** 9))
        hit = extract.existing(store_dir, rid, speed, sorted(union), max_pairs)
        if hit:
            stores[rid] = hit
            st.logline(f"[{i}/{len(rec_ids)}] {rid}: L1 store reused")
            continue
        local = src.fetch(rid)
        st.logline(f"[{i}/{len(rec_ids)}] {rid}: extracting ({'fetched' if src.kind == 'ssh' else 'local'})")

        def prog(d, n, _rid=rid):
            st.write(pair=d, n_pairs=n)
            st.check_control()

        stores[rid] = extract.extract(rid, local, speed, sorted(union), store_dir, max_pairs=max_pairs, progress=prog, work_dir=out_dir / "work")
        st.d["per_recording"][rid] = {"extracted": True, "n_pairs": extract.load(stores[rid])["n_pairs"]}

    # ---- phase 2: per recording
    per_rec: dict[str, dict[str, object]] = {}   # node -> rid -> output
    for i, rid in enumerate(rec_ids, 1):
        st.check_control()
        st.write(phase="analyse", recording=rid, recording_index=i)
        store_d = extract.load(stores[rid])
        memo: dict[str, object] = {}
        for n in order:
            if n in cross:
                continue
            st.write(node=n)
            memo[n] = _eval_local(n, g, mod_of, memo, store_d, rid)
        for n, v in memo.items():
            if mod_of[n] in ("window", "stats", "deep", "fft", "cepstrum", "wavelet", "scattering", "msc", "cusum", "concat"):
                per_rec.setdefault(n, {})[rid] = v
        del store_d, memo
        st.logline(f"[{i}/{len(rec_ids)}] {rid}: analysed")

    # ---- phase 3: cross-recording
    st.write(phase="assemble", recording=None)
    cross_out: dict[str, object] = {}
    written = None
    for n in order:
        if n not in cross:
            continue
        st.write(node=n)
        mod = mod_of[n]
        p = g.params(n)
        ins = {pi["to"][1]: pi["from"][0] for pi in g.in_pipes(n)}
        multi_ins = [pi["from"][0] for pi in g.in_pipes(n, "in")]

        def fetch(src_node):
            return cross_out[src_node] if src_node in cross_out else per_rec.get(src_node, {})

        if mod == "baseline":
            tiles_map = fetch(ins["in"])
            target = p.get("recording") or (rec_ids[0] if rec_ids else None)
            if target not in tiles_map:
                raise ValueError(f"baseline recording {target!r} produced no tiles")
            cross_out[n] = stages.baseline(tiles_map[target], p.get("mode", "cell"), target)
            st.logline(f"baseline fitted on {target}")
        elif mod == "plv":
            tiles_map = fetch(ins["in"])
            ref = cross_out[ins["ref"]]
            cross_out[n] = {rid: stages.plv(t, ref, float(p["drop"]), float(p["normal"])) for rid, t in tiles_map.items()}
        elif mod == "concat":
            maps = [fetch(s) for s in multi_ins]
            cross_out[n] = {rid: stages.concat([m[rid] for m in maps]) for rid in rec_ids if all(rid in m for m in maps)}
        elif mod == "write":
            maps = [fetch(s) for s in multi_ins]
            written = _write(out_dir, sch, scheme_path, maps, rec_ids, p, src, manifest, roster, ctx, speed, max_pairs, differ.differ_version(), st)
        else:
            raise ValueError(f"cross-recording node of kind {mod} not handled")
    st.write(state="done", phase="done", message=f"{written['n_rows']} rows, {written['n_features']} features in {time.time() - t0:.1f}s" if written else "no Write module reached")
    st.logline(st.d["message"])
    return 0


def _eval_local(n, g, mod_of, memo, store_d, rid):
    mod = mod_of[n]
    p = g.params(n)
    ins = {pi["to"][1]: pi["from"][0] for pi in g.in_pipes(n)}
    multi_ins = [pi["from"][0] for pi in g.in_pipes(n, "in")]
    up = lambda k: memo[ins[k]]  # noqa: E731
    if mod == "cells":
        return rid
    if mod == "channels":
        return stages.field_from_store(store_d, list(p.get("chans", [])))
    if mod == "single":
        return stages.single(up("in"))
    if mod == "vectorize":
        return stages.vectorize(up("in"))
    if mod == "complex":
        return stages.complex_field(up("mag"), up("dir"), p.get("phase", ""))
    if mod == "block":
        return stages.block(up("in"), int(p["wp"]), int(p["hp"]))
    if mod == "collapse":
        return stages.collapse(up("in"), p.get("unchanged", "zero"), p.get("reduce", "mean"))
    if mod == "window":
        s = up("in")
        if isinstance(s, dict) and "page_index" in s:
            raise stages.NotImplementedStage("tiles at full page resolution are not implemented; add Collapse or Block before Window")
        if isinstance(s, list):
            tl = [stages.window(x, int(p["w"]), int(p["h"]), p.get("edge", "drop"), p.get("taper", "rectangular")) for x in s]
            return {"X": np.concatenate([t["X"] for t in tl]), "keys": [k for t in tl for k in t["keys"]], "w": tl[0]["w"], "h": tl[0]["h"],
                    "taper": tl[0]["taper"], "channels": tl[0]["channels"], "complex": tl[0]["complex"],
                    "series_mean": float(np.mean([t["series_mean"] for t in tl])), "series_std": float(np.mean([t["series_std"] for t in tl]))}
        return stages.window(s, int(p["w"]), int(p["h"]), p.get("edge", "drop"), p.get("taper", "rectangular"))
    if mod == "stats":
        return stages.stats(up("in"), list(p["feats"]))
    if mod == "deep":
        return stages.deep(up("in"), list(p["feats"]))
    if mod == "fft":
        return stages.fft(up("in"), p.get("out", "bands"), detrend=p.get("detrend", "mean"))
    if mod == "cepstrum":
        return stages.cepstrum(up("in"))
    if mod == "cusum":
        return stages.cusum(up("in"), float(p["k"]), float(p["h"]))
    if mod == "wavelet":
        return stages.wavelet(up("in"), p.get("fam", ""), int(p["levels"]), p.get("mode", "periodization"))
    if mod == "scattering":
        return stages.scattering(up("in"), int(p["J"]), int(p["Q"]))
    if mod == "msc":
        return stages.msc(up("in"), int(p["iw"]), int(p["ih"]), p.get("method", "welch"), p.get("detrend", "mean"))
    if mod == "concat":
        return stages.concat([memo[s] for s in multi_ins])
    raise ValueError(f"module {mod} cannot be evaluated per recording")


def _write(out_dir, sch, scheme_path, maps, rec_ids, p, src, manifest, roster, ctx, speed, max_pairs, dv, st) -> dict:
    rows, keys, names = [], [], None
    for rid in rec_ids:
        blocks = [m[rid] for m in maps if rid in m]
        if len(blocks) != len(maps):
            continue
        f = stages.concat(blocks) if len(blocks) > 1 else blocks[0]
        names = names or list(f["names"])
        if list(f["names"]) != names:
            raise ValueError("feature names differ across recordings")
        for k, r in zip(f["keys"], f["rows"]):
            rows.append(r)
            keys.append((rid, ctx.rec_by_id[rid]["workload"], ctx.rec_by_id[rid]["family"], k[0] if k[0] is not None else -1, k[1], k[2]))
    X = np.asarray(rows, dtype=np.float32).reshape(len(rows), len(names or []))
    key_dt = np.dtype([("recording", "U256"), ("workload", "U128"), ("family", "U32"), ("block", "i4"), ("t_index", "i4"), ("seq_start", "i4")])
    tile_keys = np.array(keys, dtype=key_dt)
    np.savez_compressed(out_dir / "features.npz", X=X, feature_names=np.array(names or [], dtype="U64"), tile_keys=tile_keys)
    with (out_dir / "features.csv").open("w") as f:
        f.write(",".join(["recording", "workload", "family", "block", "t_index", "seq_start"] + (names or [])) + "\n")
        for k, r in zip(keys, X):
            f.write(",".join([str(x) for x in k] + [f"{v:.6g}" for v in r]) + "\n")
    sidecar = {
        "schema": "plan10.sidecar.v1",
        "label": sch.get("label"),
        "scheme": sch,
        "scheme_file": str(scheme_path),
        "acknowledged": sch.get("acknowledged", []),
        "source": src.describe(),
        "manifest": {"root": manifest["root"], "scanned_at": manifest["scanned_at"], "n_recordings": manifest["n_recordings"]},
        "recordings": [{"id": r, "n_pairs": ctx.rec_by_id[r]["n_pairs"], "workload": ctx.rec_by_id[r]["workload"]} for r in rec_ids],
        "speed": speed, "speed_source": "run parameter (config default when unset); unrecorded per recording",
        "max_pairs": max_pairs,
        "n_pages_default": ctx.config["n_pages_default"],
        "differ": dv,
        "roster_sha": roster["derivation"]["source_sha256_16"],
        "python": platform.python_version(), "numpy": np.__version__,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rows": int(X.shape[0]), "n_features": len(names or []), "feature_names": names or [],
        "format": p.get("fmt", "npz+csv"),
    }
    (out_dir / "sidecar.json").write_text(json.dumps(sidecar, indent=1, default=str))
    st.logline(f"wrote {out_dir / 'features.npz'} ({X.shape[0]} x {X.shape[1]})")
    return {"n_rows": int(X.shape[0]), "n_features": len(names or [])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("scheme", type=Path)
    r.add_argument("--out-dir", type=Path, required=True)
    r.add_argument("--source-json", type=str, default=None)
    r.add_argument("--manifest", type=Path, default=None)
    r.add_argument("--store", type=Path, default=None)
    r.add_argument("--speed", type=int, default=None)
    r.add_argument("--max-pairs", type=int, default=None)
    r.add_argument("--acknowledge-all", action="store_true")
    a = ap.parse_args()
    spec = json.loads(a.source_json) if a.source_json else None
    return run(a.scheme, a.out_dir, spec, a.store, a.speed, a.max_pairs, a.acknowledge_all, a.manifest)


if __name__ == "__main__":
    raise SystemExit(main())
