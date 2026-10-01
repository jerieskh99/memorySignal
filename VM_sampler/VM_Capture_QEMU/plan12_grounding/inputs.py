#!/usr/bin/env python3
"""inputs.py -- move 0 of plan12_grounding (SPEC move 0): the inputs and the index.

  python3 -m plan12_grounding.inputs index --out O (--root R | --ssh USER@HOST --remote-root RR)
      [--encoding-out E] [--cut-declared 16] [--cut-measured 112] [--allow-unmatched-declared]
      [--room-removal] [--dry-run]

Reads the source's listing (`LocalSource.listing()` or `SshSource.listing()`, one `find` over ssh,
read only) and indexes it with `plan11_encoding_ladder.extract.build_index` itself: the listing is
mirrored as an empty directory tree (every directory, and every trajectory file as an empty file of
the same name) under a scratch folder, `build_index` runs on that mirror exactly as it runs on the
retention root (roles kernel and idle, the rep index from the seed, the declared seed map), and the
mirror is removed afterwards. The data fixes are the encoding paper's own declared files, read in
place from `plan11_encoding_ladder/declared/`: `seed_map.csv` (through build_index), `keep_first_pairs.csv`
(the three double recordings; applied at move 1, recorded here per recording), `head_drop_values.csv`
(through `series.load_head_drop` / `head_drop_for`; applied at analysis time, recorded here).
Admissibility is the encoding toolkit's rule `series.admissible_cells`: status ok in the index, and
`all_hard_pass` in the named encoding run's `gates/preconditions.csv` when `--encoding-out` (D2) names
that run; that is where lexer seed 6898 is refused, as it was there. Without `--encoding-out` the
index alone decides, and `params.json` says so.

Writes `<out>/cells.csv`, `<out>/inputs/` (copies of the declared files with sha256; the encoding run's
cells.csv and preconditions.csv when given), `<out>/params.json` (both cuts, D2, D3, D4, the move 9
switch) and `<out>/moves/00_index/index.json`. Recordings outside the twelve kernels and idle are
counted, never named (SPEC 1.1): they do not appear in cells.csv, and the mirror that briefly held
their directory names is removed.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import argparse  # noqa: E402
import csv  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import traceback  # noqa: E402

from plan12_grounding import __version__, toolkit_fingerprint  # noqa: E402
from plan12_grounding.run_moves import (  # noqa: E402
    CUT_DECLARED, CUT_MEASURED, DECLARED_DIR, DECLARED_FILES, add_source_args, describe_source, now_iso,
    read_json, sha256_file, source_from_args, write_json,
)
from plan11_encoding_ladder import extract as E11  # noqa: E402  (build_index and the declared-file readers; imported, never copied)
from plan11_encoding_ladder import schema, series as S11  # noqa: E402

CITATION = "plan12_grounding/SPEC.md move 0 and section 2; plan11_encoding_ladder/extract.py build_index (SPEC 2.6, 2.7); series.admissible_cells (SPEC 3.3.1)"
TRAJ_MARKER = "substrate_trajectory"
CELLS_COLUMNS = ("cell_id", "kernel", "role", "archetype_predicted", "seed", "rep", "rep_dir", "label", "campaign",
                 "rec_rel", "traj_file", "index_status", "seed_from_map", "keep_first_pairs", "head_drop_pairs",
                 "admissible", "reason")
CUT_CONVENTION = ("a cut of H pairs drops the first H pairs of a recording's per-pair series in pair-index order "
                  "(plan11_encoding_ladder/series.py rung_series: n_series = n_pairs - 1 - head_drop, the first head_drop "
                  "rows dropped); H per recording = series.head_drop_for(load_head_drop(declared/head_drop_values.csv), "
                  "kernel, role), the idle row for idle cells; every number is computed at both cuts")


def _mirror_listing(entries, mirror: Path) -> dict:
    """The listing as an empty tree: every directory, and every trajectory file as an empty file."""
    n_dirs = n_files = n_traj = 0
    for e in entries:
        rel = e.relpath.strip("/")
        if not rel or rel.startswith("."):
            continue
        if e.is_dir:
            (mirror / rel).mkdir(parents=True, exist_ok=True)
            n_dirs += 1
        else:
            n_files += 1
            if TRAJ_MARKER in Path(rel).name:
                p = mirror / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.touch()
                n_traj += 1
    return {"n_entries": len(entries), "n_dirs": n_dirs, "n_files": n_files, "n_trajectory_files": n_traj}


def _is_corpus(row: dict) -> bool:
    if row.get("role") == "idle":
        return True
    return row.get("role") == "kernel" and row.get("kernel") in schema.KERNEL_NAMES and row.get("family") == "kernel"


def _encoding_admissibility(enc_out: Path) -> tuple[dict, dict, dict]:
    """The named encoding run's records, read only: its cells.csv by cell_id, its preconditions by
    cell_id (series.preconditions_map), and the paths copied. Missing files are reported, not fatal."""
    cells = {}
    p = enc_out / "cells.csv"
    if p.is_file():
        cells = {r["cell_id"]: r for r in S11.read_csv(p)}
    pm = S11.preconditions_map(enc_out)
    return cells, pm, {"cells_csv": str(p) if p.is_file() else None,
                       "preconditions_csv": str(enc_out / "gates" / "preconditions.csv") if (enc_out / "gates" / "preconditions.csv").is_file() else None}


def _precondition_reason(row: dict) -> str:
    bad = []
    for k, v in row.items():
        v = str(v or "")
        if k.startswith("C") and k[1:].isdigit() and v and not v.startswith("pass"):
            bad.append(f"{k}: {v}")
    if str(row.get("failed_verdict", "")).startswith(("refused", "not run", "fail")):
        bad.append(f"failed_verdict: {row['failed_verdict']}")
    return "; ".join(bad) or "all_hard_pass false"


def index(o: argparse.Namespace) -> int:
    out = Path(os.path.expanduser(o.out))
    src = source_from_args(o)
    declared = {name: DECLARED_DIR / name for name in DECLARED_FILES}
    missing = [str(p) for p in declared.values() if not p.is_file()]
    if missing:
        print(f"missing declared input: {', '.join(missing)}", file=sys.stderr)
        return 2
    if o.dry_run:
        print(f"[index] dry run: source {describe_source(src)}")
        if src.kind == "ssh":
            print(f"[index] would run over ssh (read only): {src.listing_cmd()}")
        else:
            print(f"[index] would walk the local root {src.root}")
        print(f"[index] would copy into {out / 'inputs'}: " + ", ".join(f"{k} ({sha256_file(v)[:12]})" for k, v in declared.items()))
        if o.encoding_out:
            print(f"[index] would read the encoding run's cells.csv and gates/preconditions.csv under {o.encoding_out} (read only)")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    entries = src.listing()
    listing_warning = getattr(src, "listing_warning", None)
    # the mirror: a scratch tree of empty files under <out>, built for build_index and removed below
    # (it holds only directory names and empty placeholder files this move created; it also holds the
    # names of the recordings outside the corpus, which must not persist anywhere: SPEC 1.1)
    mirror = Path(tempfile.mkdtemp(prefix=".index_mirror_", dir=str(out)))
    try:
        mstats = _mirror_listing(entries, mirror)
        top = sorted(d for d in os.listdir(mirror) if (mirror / d).is_dir())
        n_other_top = sum(1 for d in top if d not in ("kernel", "sleep"))
        rows = E11.build_index(mirror, mirror / "cells.csv", seed_map=str(declared["seed_map.csv"]))
        index_params = read_json(mirror / "cells.index.json").get("params", {})
        for r in rows:
            r["rec_rel"] = Path(r["path"]).relative_to(mirror).as_posix()
        corpus = [r for r in rows if _is_corpus(r)]
        others = [r for r in rows if not _is_corpus(r)]
        other_status = {}
        for r in others:
            other_status[r["status"]] = other_status.get(r["status"], 0) + 1
        # the data fixes, recorded per recording
        kfp_rows = E11._load_keep_first_pairs(declared["keep_first_pairs.csv"])
        hd = S11.load_head_drop(declared["head_drop_values.csv"])
        matched_kfp = set()
        enc_cells, enc_pm, enc_paths = ({}, {}, {})
        if o.encoding_out:
            enc_cells, enc_pm, enc_paths = _encoding_admissibility(Path(os.path.expanduser(o.encoding_out)))
        seed_from_map = set(index_params.get("seed_from_map") or [])
        for r in corpus:
            kf = E11._keep_first_for(Path(r["path"]), kfp_rows)
            r["keep_first_pairs"] = kf["keep_first_pairs"] if kf else ""
            if kf:
                matched_kfp.add(kf["row"])
            r["head_drop_pairs"] = S11.head_drop_for(hd, r["kernel"], r["role"])
            r["seed_from_map"] = "true" if E11._cell_tail(r["path"]) in seed_from_map or r["rec_rel"] in seed_from_map else "false"
            r["index_status"] = r["status"]
            reasons = []
            if r["status"] != schema.STATUS_OK:
                reasons.append(f"index: {r['status']}")
            ec = enc_cells.get(r["cell_id"])
            if ec is not None and ec.get("status", "ok") != "ok":
                reasons.append(f"encoding run cells.csv: {ec['status']}")
            pr = enc_pm.get(r["cell_id"])
            if pr is not None and str(pr.get("all_hard_pass", "true")).lower() != "true":
                reasons.append(f"encoding run preconditions: all_hard_pass false ({_precondition_reason(pr)})")
            r["admissible"] = "true" if not reasons else "false"
            r["reason"] = "; ".join(reasons) if reasons else ("ok" + ("" if not o.encoding_out else
                                                                   ("; in the encoding run's preconditions" if pr is not None else "; not in the encoding run's preconditions")))
        unmatched_kfp = [{"row": k["row"], "path": k["path"], "keep_first_pairs": k["keep_first_pairs"]} for k in kfp_rows if k["row"] not in matched_kfp]
        if unmatched_kfp and not o.allow_unmatched_declared:
            print("stopped: declared keep-first rows that name no recording of this source (the encoding toolkit stops here too; "
                  "pass --allow-unmatched-declared on a corpus that does not hold them):", file=sys.stderr)
            for u in unmatched_kfp:
                print(f"  row {u['row']}: {u['path']}", file=sys.stderr)
            return 2
    finally:
        shutil.rmtree(mirror, ignore_errors=True)   # the scratch mirror of empty files this move created (see above)
    order = {k: i for i, k in enumerate(schema.KERNEL_NAMES)}
    corpus.sort(key=lambda r: (r["role"] != "kernel", order.get(r["kernel"], 99), r["rep"] if r["rep"] is not None else 999, r["cell_id"]))
    # cells.csv
    with open(out / "cells.csv.tmp", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(CELLS_COLUMNS), extrasaction="ignore")
        w.writeheader()
        for r in corpus:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in CELLS_COLUMNS})
    os.replace(out / "cells.csv.tmp", out / "cells.csv")
    # inputs/: the declared files, copied with sha256 (read in place from plan11's folder; never edited)
    inp = out / "inputs"
    inp.mkdir(parents=True, exist_ok=True)
    shas = {}
    for name, p in declared.items():
        shutil.copyfile(p, inp / name)
        shas[name] = {"sha256": sha256_file(inp / name), "source": str(p), "bytes": p.stat().st_size}
    if o.encoding_out:
        for tag, p in enc_paths.items():
            if p:
                dst = inp / "encoding_run" / Path(p).name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, dst)
                shas[f"encoding_run/{Path(p).name}"] = {"sha256": sha256_file(dst), "source": p, "bytes": Path(p).stat().st_size}
    write_json(inp / "sha256.json", {"schema": "plan12.inputs.v1", "citation": CITATION, "written_at": now_iso(), "files": shas})
    n_adm = sum(1 for r in corpus if r["admissible"] == "true")
    n_kernel = sum(1 for r in corpus if r["role"] == "kernel")
    n_idle = len(corpus) - n_kernel
    fp = toolkit_fingerprint()
    params = {
        "schema": "plan12.params.v1", "citation": CITATION, "package_version": __version__, "written_at": now_iso(),
        "toolkit_fingerprint": fp["sha256"],
        "source": describe_source(src), "listing": {**mstats, "warning": listing_warning},
        "cuts": {"declared_pairs": int(o.cut_declared), "measured_pairs": int(o.cut_measured),
                 "convention": CUT_CONVENTION, "head_drop_source": str(declared["head_drop_values.csv"])},
        "D1_room_removal": {"switch": "the driver's --room-removal at move 9 (moves/09_removed/removal.json records a run); this record only names the default", "on_at_move_0": bool(o.room_removal), "default": "off (SPEC 9: built, off)"},
        "D2_baseline": {"encoding_out": (str(Path(os.path.expanduser(o.encoding_out))) if o.encoding_out else None),
                        "default": "the named encoding run's own per-pair extract, pulled read only (SPEC 2)",
                        "admissibility": ("the encoding run's gates/preconditions.csv through series.admissible_cells" if o.encoding_out
                                          else "index status only: no encoding run named, so no preconditions record was applied")},
        "D3_window": {"grid_id": None, "default": "the encoding run's selected grid point for its combined rung (series.selected_grid_id); fallback W=8, H=4",
                      "resolved_at": "move 6"},
        "D4_out": str(out),
        "declared": {"files": shas, "keep_first_rows_unmatched": unmatched_kfp,
                     "seed_map_rows_unmatched": index_params.get("seed_map_rows_unmatched"),
                     "seed_from_map_n": index_params.get("n_seed_from_map"),
                     "seed_map_name_truncated": index_params.get("seed_map_name_truncated"),
                     "index_params": {k: v for k, v in index_params.items() if k not in ("root",)}},
        "counts": {"recordings_listed": len(rows), "corpus_kernel": n_kernel, "corpus_idle": n_idle,
                   "admissible": n_adm, "not_admissible": len(corpus) - n_adm,
                   "other_recordings_counted_not_named": len(others), "other_recordings_by_index_status": other_status,
                   "other_top_level_dirs_counted_not_named": n_other_top},
    }
    write_json(out / "params.json", params)
    write_json(out / "moves" / "00_index" / "index.json", {
        "schema": "plan12.index.v1", "citation": CITATION, "written_at": now_iso(),
        "counts": params["counts"], "not_admissible": [{"cell_id": r["cell_id"], "reason": r["reason"]} for r in corpus if r["admissible"] != "true"],
        "keep_first": [{"cell_id": r["cell_id"], "keep_first_pairs": r["keep_first_pairs"]} for r in corpus if r["keep_first_pairs"] != ""],
        "head_drop": sorted({(r["kernel"] if r["role"] == "kernel" else "idle", r["head_drop_pairs"]) for r in corpus}),
    })
    print(f"[index] {len(rows)} recordings listed: {n_kernel} kernel + {n_idle} idle in the corpus, {len(others)} other "
          f"(counted, not named), {n_other_top} other top-level directories; admissible {n_adm} of {len(corpus)}")
    for r in corpus:
        if r["admissible"] != "true":
            print(f"[index]   not admissible: {r['cell_id']}: {r['reason']}")
    if unmatched_kfp:
        print(f"[index] declared keep-first rows matching no recording here: {len(unmatched_kfp)} (allowed by --allow-unmatched-declared)")
    if listing_warning:
        print(f"[index] listing warning: {listing_warning}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="plan12_grounding.inputs", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("index", help="move 0: the inputs and the index")
    p.add_argument("--out", required=True)
    add_source_args(p)
    p.add_argument("--encoding-out", default=None)
    p.add_argument("--cut-declared", type=int, default=CUT_DECLARED)
    p.add_argument("--cut-measured", type=int, default=CUT_MEASURED)
    p.add_argument("--allow-unmatched-declared", action="store_true")
    p.add_argument("--room-removal", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    o = ap.parse_args(argv)
    try:
        return index(o)
    except (E11.SeedMapError, E11.KeepFirstError) as exc:
        print(f"declared input: {exc}", file=sys.stderr)
        return 2
    except SystemExit as exc:
        if exc.code not in (None, 0):
            print(str(exc), file=sys.stderr)
        return 2 if exc.code not in (None, 0) else 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
