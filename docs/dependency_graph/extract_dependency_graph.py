#!/usr/bin/env python3
"""extract_dependency_graph.py -- read-only, file-level dependency graph of this repository.

Deletes nothing, moves nothing, proposes no deletions. Writes two files next to itself:
dependency_graph.json and DEPENDENCY_GRAPH.md.

Why import analysis is not enough here: the pipeline couples through files on disk. Shell
invokes Python, Python shells out to a Rust binary, the binary writes a CSV a later stage reads.
So the graph has four edge types (imports, invokes, writes, reads), artifacts are nodes, and
every edge cites file:line plus the literal text it matched.

Extraction has two halves:
  * mechanical  Python imports via ast; Rust `mod`/`use`; C `#include`; Makefile sources;
                shell `source`/`bash x.sh`/`python3 x.py`; artifact-name tokens on code lines,
                classified read/write by the verbs on that line, with module-level path
                constants folded (HERE / "x.py", os.environ.get("K", default), ...).
  * curated     semantic hops the regexes cannot see (a config-valued binary resolved to its
                crate through Cargo.toml, a launch line assembled by another script, a Popen
                whose target arrives as a parameter). Each curated entry names the literal
                text it relies on; the script locates that text to obtain the line number and
                downgrades the edge to `unresolved` if the text is gone.

Run:  python3 docs/dependency_graph/extract_dependency_graph.py
"""
from __future__ import annotations

import ast
import io
import json
import os
import re
import subprocess
import sys
import tokenize
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
Q = "VM_sampler/VM_Capture_QEMU"
VC = "VM_sampler/VM_Capture"
P7 = f"{Q}/plan07_campaign"
P8 = f"{Q}/plan08_b1"
P10 = f"{Q}/plan10_analysis"

TRUE_ROOTS = [f"{P7}/ui/console.sh", f"{P10}/ui/analysis.sh"]
AUTHORITY = f"{P7}/database_steps_manifest.csv"

EXCLUDE_RE = re.compile(
    r"^(\.claude/worktrees/|utils/venv/|data/|\.git/)"
    r"|(^|/)(target|__pycache__)/"
    r"|\.lock$"
    r"|(^|/)file_structure\.txt$"
    r"|\.(pyc|png|pdf|zip|npy|zst)$"
)
MAX_SCAN_BYTES = 100 * 1024

CODE_EXT = {".py", ".sh", ".rs", ".c", ".h", ".ps1", ".toml"}

# --------------------------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------------------------
_text_cache: dict[str, str] = {}


def read(rel: str) -> str:
    if rel not in _text_cache:
        try:
            _text_cache[rel] = (REPO / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            _text_cache[rel] = ""
    return _text_cache[rel]


def lines_of(rel: str) -> list[str]:
    return read(rel).splitlines()


def find_line(rel: str, needle: str, nth: int = 1) -> tuple[int, str]:
    """1-based line number of the nth line containing `needle`, or (0, '') if absent."""
    cl = code_lines(rel) if (REPO / rel).is_file() else []
    hits = [(i + 1, l) for i, l in enumerate(cl) if needle in l]
    if len(hits) >= nth:
        ln = hits[nth - 1][0]
        return ln, lines_of(rel)[ln - 1]
    hits = [(i + 1, l) for i, l in enumerate(lines_of(rel)) if needle in l]
    if len(hits) >= nth:
        return hits[nth - 1]
    return 0, ""


def git_files() -> list[str]:
    tracked = subprocess.check_output(["git", "ls-files"], cwd=REPO, text=True).splitlines()
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"],
                                        cwd=REPO, text=True).splitlines()
    out = set()
    for f in tracked + untracked:
        if EXCLUDE_RE.search(f):
            continue
        if not (REPO / f).is_file():
            continue
        out.add(f)
    return sorted(out)


def git_exec_bits() -> set[str]:
    out = set()
    for l in subprocess.check_output(["git", "ls-files", "-s"], cwd=REPO, text=True).splitlines():
        parts = l.split("\t")
        if parts[0].split()[0] == "100755":
            out.add(parts[1])
    return out


# --------------------------------------------------------------------------------------------
# graph containers
# --------------------------------------------------------------------------------------------
class Graph:
    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.edges: list[dict] = []
        self._edge_index: dict = {}
        self.blind: dict[str, list] = defaultdict(list)
        self.notes: dict[str, list[str]] = defaultdict(list)
        self.incomplete: set[str] = set()

    def node(self, path: str, kind: str | None = None, **flags) -> dict:
        n = self.nodes.get(path)
        if n is None:
            n = {"path": path, "kind": kind or "module"}
            self.nodes[path] = n
        elif kind and n.get("kind") == "module":
            n["kind"] = kind
        for k, v in flags.items():
            n[k] = v
        return n

    def edge(self, src: str, dst: str, typ: str, file: str, line: int, text: str,
             binding: str, **extra) -> dict | None:
        assert typ in ("imports", "invokes", "writes", "reads"), typ
        assert binding in ("literal", "interpolated", "unresolved"), binding
        if src == dst:
            return None
        for end in (src, dst):
            if end not in self.nodes:
                if end.startswith("artifact:"):
                    self.node(end, "artifact")
                elif end in FILESET or end in COLLAPSED_NODES:
                    self.node(end, "module")
                else:
                    self.blind["edge_to_unknown_node"].append({"src": src, "dst": dst, "type": typ, "file": file, "line": line})
                    return None
        key = (src, dst, typ)
        ev = {"file": file, "line": line, "text": text.strip()[:240]}
        if key in self._edge_index:
            e = self._edge_index[key]
            if ev not in e.get("also_evidence", []) and ev != e["evidence"] and len(e.get("also_evidence", [])) < 8:
                e.setdefault("also_evidence", []).append(ev)
            rank = {"literal": 2, "interpolated": 1, "unresolved": 0}
            if rank[binding] > rank[e["binding"]]:
                e["binding"] = binding
                e["evidence"], ev = ev, e["evidence"]
                e.setdefault("also_evidence", []).insert(0, ev)
            if not extra.get("alternative") and e.get("alternative"):
                e.pop("alternative", None)
            for k, v in extra.items():
                if v is not None and k not in e and k != "alternative":
                    e[k] = v
            return e
        e = {"src": src, "dst": dst, "type": typ, "evidence": ev, "binding": binding}
        e.update({k: v for k, v in extra.items() if v is not None})
        self.edges.append(e)
        self._edge_index[key] = e
        return e

    def mark_incomplete(self, path: str, why: str):
        self.incomplete.add(path)
        if why not in self.notes[path]:
            self.notes[path].append(why)


G = Graph()
FILES = git_files()
FILESET = set(FILES)
EXEC_BITS = git_exec_bits()
BY_BASENAME: dict[str, list[str]] = defaultdict(list)
for _f in FILES:
    BY_BASENAME[os.path.basename(_f)].append(_f)


def resolve_by_basename(base: str, hints: list[str], src: str) -> tuple[str | None, str]:
    """Repo file for a basename. Returns (path|None, reason)."""
    cands = BY_BASENAME.get(base, [])
    if not cands:
        return None, "no-such-basename"
    if len(cands) == 1:
        return cands[0], "unique"
    src_dir = os.path.dirname(src)
    same = [c for c in cands if os.path.dirname(c) == src_dir]
    if len(same) == 1:
        return same[0], "same-dir"
    for h in hints:
        hit = [c for c in cands if h and h in c]
        if len(hit) == 1:
            return hit[0], f"hint:{h}"
    return None, f"ambiguous:{len(cands)}"


# --------------------------------------------------------------------------------------------
# run-instance collapsing (one artifact-class node per instance tree)
# --------------------------------------------------------------------------------------------
COLLAPSE = [
    (re.compile(rf"^{re.escape(Q)}/plan06_campaign/downstream/cells_full/work/[0-9a-f]+/apf_trajectory\.jsonl$"),
     f"{Q}/plan06_campaign/downstream/cells_full/work/<cell>/apf_trajectory.jsonl"),
    (re.compile(rf"^{re.escape(Q)}/plan06_campaign/downstream/cells_full/work/[0-9a-f]+/diskio_trajectory\.jsonl$"),
     f"{Q}/plan06_campaign/downstream/cells_full/work/<cell>/diskio_trajectory.jsonl"),
    (re.compile(rf"^{re.escape(Q)}/timing_runs/exp2c_[^/]+/.+$"),
     f"{Q}/timing_runs/exp2c_<run>/<file>"),
]
FILE_NODES: list[str] = []
COLLAPSED_NODES: set[str] = set()
collapsed_count: dict[str, int] = defaultdict(int)
for f in FILES:
    for rx, cls in COLLAPSE:
        if rx.match(f):
            collapsed_count[cls] += 1
            break
    else:
        FILE_NODES.append(f)
for cls, n in collapsed_count.items():
    COLLAPSED_NODES.add(cls)
    G.node(cls, "artifact", instances=n, collapsed=True)
for f in FILE_NODES:
    G.node(f, "module")

# --------------------------------------------------------------------------------------------
# code-only line views (comments and docstrings blanked) so that string matches are real
# --------------------------------------------------------------------------------------------
_code_cache: dict[str, list[str]] = {}


def code_lines(rel: str) -> list[str]:
    """Lines with comments (and Python docstrings) blanked out. Same line numbering."""
    if rel in _code_cache:
        return _code_cache[rel]
    text = read(rel)
    ls = text.splitlines()
    ext = os.path.splitext(rel)[1]
    out = list(ls)
    if ext == ".py":
        try:
            toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
            # blank comments
            for t in toks:
                if t.type == tokenize.COMMENT:
                    r, c = t.start
                    out[r - 1] = out[r - 1][:c]
            # blank docstrings
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    body = getattr(node, "body", [])
                    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                            and isinstance(body[0].value.value, str):
                        for r in range(body[0].lineno, (body[0].end_lineno or body[0].lineno) + 1):
                            out[r - 1] = ""
        except (SyntaxError, tokenize.TokenError):
            pass
    elif ext in (".sh", ".ps1", ".toml", "") or rel.endswith("Makefile"):
        for i, l in enumerate(out):
            if l.lstrip().startswith("#"):
                out[i] = ""
    elif ext in (".rs", ".c", ".h"):
        for i, l in enumerate(out):
            s = l.lstrip()
            if s.startswith("//") or s.startswith("*") or s.startswith("/*"):
                out[i] = ""
    _code_cache[rel] = out
    return out


# --------------------------------------------------------------------------------------------
# Python: module-level path constant folding
# --------------------------------------------------------------------------------------------
def _fold(node: ast.AST, consts: dict[str, str | None], src: str) -> tuple[str | None, str]:
    """Return (repo-relative-or-abs path string | None, binding). Very small evaluator."""
    src_dir = os.path.dirname(src)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value, "literal"
    if isinstance(node, ast.Name):
        v = consts.get(node.id)
        if v is None:
            return None, "unresolved"
        return v, "literal"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        l, bl = _fold(node.left, consts, src)
        r, br = _fold(node.right, consts, src)
        if l is None or r is None:
            return None, "unresolved"
        return os.path.normpath(os.path.join(l, r)), "interpolated" if "interpolated" in (bl, br) else "literal"
    if isinstance(node, ast.Attribute):
        base, b = _fold(node.value, consts, src)
        if node.attr == "parent" and base is not None:
            return os.path.dirname(base), b
        if node.attr in ("resolve",):
            return base, b
        return None, "unresolved"
    if isinstance(node, ast.Call):
        fn = node.func
        # Path(__file__)
        if isinstance(fn, ast.Name) and fn.id == "Path" and node.args:
            a = node.args[0]
            if isinstance(a, ast.Name) and a.id == "__file__":
                return src, "literal"
            return _fold(a, consts, src)
        if isinstance(fn, ast.Attribute):
            if fn.attr == "resolve":
                return _fold(fn.value, consts, src)
            if fn.attr == "with_name" and node.args:
                base, b = _fold(fn.value, consts, src)
                nm, _ = _fold(node.args[0], consts, src)
                if base is not None and nm is not None:
                    return os.path.join(os.path.dirname(base), nm), b
            if fn.attr == "expanduser":
                return _fold(fn.value, consts, src)
            if fn.attr == "join" and isinstance(fn.value, ast.Attribute) and fn.value.attr == "path":
                parts = [_fold(a, consts, src) for a in node.args]
                if all(p[0] is not None for p in parts):
                    return os.path.normpath(os.path.join(*[p[0] for p in parts])), \
                        "interpolated" if any(p[1] == "interpolated" for p in parts) else "literal"
                return None, "unresolved"
            if fn.attr in ("dirname", "abspath"):
                base, b = _fold(node.args[0], consts, src) if node.args else (None, "unresolved")
                if base is None:
                    return None, "unresolved"
                return (os.path.dirname(base) if fn.attr == "dirname" else base), b
            if fn.attr == "get" and isinstance(fn.value, ast.Attribute) and fn.value.attr == "environ":
                if len(node.args) >= 2:
                    v, _ = _fold(node.args[1], consts, src)
                    return v, "interpolated"        # env override possible; default is the edge
                return None, "unresolved"
        if isinstance(fn, ast.Name) and fn.id == "str" and node.args:
            return _fold(node.args[0], consts, src)
    if isinstance(node, ast.Subscript):
        # Path(__file__).resolve().parents[N]
        if isinstance(node.value, ast.Attribute) and node.value.attr == "parents":
            base, b = _fold(node.value.value, consts, src)
            idx = node.slice
            if isinstance(idx, ast.Constant) and base is not None:
                p = os.path.dirname(base)
                for _ in range(int(idx.value)):
                    p = os.path.dirname(p)
                return p, b
        return None, "unresolved"
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        # a.out or (a.root / "x")  -> take the last alternative (the default)
        return _fold(node.values[-1], consts, src)
    return None, "unresolved"


_logical_cache: dict[str, list[tuple[int, str]]] = {}


def logical_lines(rel: str) -> list[tuple[int, str]]:
    """(first physical line number, joined code text) per logical statement. Python via tokenize,
    shell via backslash continuation; other languages one physical line each."""
    if rel in _logical_cache:
        return _logical_cache[rel]
    cl = code_lines(rel)
    ext = os.path.splitext(rel)[1]
    out: list[tuple[int, str]] = []
    if ext == ".py":
        try:
            toks = list(tokenize.generate_tokens(io.StringIO(read(rel)).readline))
            start = None
            for t in toks:
                if t.type in (tokenize.NL, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING):
                    continue
                if start is None and t.type not in (tokenize.NEWLINE, tokenize.ENDMARKER):
                    start = t.start[0]
                if t.type == tokenize.NEWLINE and start is not None:
                    end = t.end[0]
                    out.append((start, " ".join(x.strip() for x in cl[start - 1:end])))
                    start = None
            if start is not None:
                out.append((start, " ".join(x.strip() for x in cl[start - 1:])))
        except (SyntaxError, tokenize.TokenError):
            out = [(i, l) for i, l in enumerate(cl, 1)]
    elif ext == ".sh":
        i = 0
        while i < len(cl):
            j = i
            buf = cl[i]
            while buf.rstrip().endswith("\\") and j + 1 < len(cl):
                j += 1
                buf = buf.rstrip()[:-1] + " " + cl[j].strip()
            out.append((i + 1, buf))
            i = j + 1
    else:
        out = [(i, l) for i, l in enumerate(cl, 1)]
    _logical_cache[rel] = out
    return out


def py_consts(rel: str) -> dict[str, tuple[str, str, int]]:
    """NAME -> (folded path, binding, line) for module-level assignments."""
    out: dict[str, tuple[str, str, int]] = {}
    consts: dict[str, str | None] = {"__file__": rel}
    try:
        tree = ast.parse(read(rel))
    except SyntaxError:
        return out
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            v, b = _fold(node.value, consts, rel)
            if v is not None:
                consts[name] = v
                out[name] = (v, b, node.lineno)
    return out


def norm_repo(p: str) -> str | None:
    """Normalise a folded path to a repo-relative path if it is inside the repo."""
    if p.startswith("/"):
        try:
            return str(Path(p).relative_to(REPO))
        except ValueError:
            return None
    return os.path.normpath(p)


# --------------------------------------------------------------------------------------------
# Python imports (ast)
# --------------------------------------------------------------------------------------------
PY_ROOTS = ["", Q, P7, f"{Q}/plan05_campaign", f"{Q}/plan06_campaign", P8, P10,
            "coherence_temp_spec_stability", "VM_executables_phase2", "utils"]
STDLIB = set(getattr(sys, "stdlib_module_names", set())) | {"numpy", "np", "sklearn", "scipy", "pandas",
           "matplotlib", "pywt", "reportlab", "h5py", "plotly", "kymatio", "dask", "torch", "pyeeg",
           "borgmatic", "zstandard", "seaborn", "tqdm", "PIL", "joblib", "yaml", "requests", "pytest"}


def resolve_module(modname: str, src: str) -> list[tuple[str, str]]:
    """[(repo path, how)] for a dotted module; package __init__ first, then module file."""
    parts = modname.split(".")
    roots = [os.path.dirname(src)] + PY_ROOTS
    for root in roots:
        base = os.path.join(root, *parts) if root else os.path.join(*parts)
        base = os.path.normpath(base)
        hits: list[tuple[str, str]] = []
        # package inits along the chain
        for i in range(1, len(parts)):
            init = os.path.normpath(os.path.join(root, *parts[:i], "__init__.py") if root else os.path.join(*parts[:i], "__init__.py"))
            if init in FILESET:
                hits.append((init, "package-init"))
        if base + ".py" in FILESET:
            hits.append((base + ".py", "module"))
            return hits
        if os.path.join(base, "__init__.py") in FILESET:
            hits.append((os.path.join(base, "__init__.py"), "package"))
            return hits
        if os.path.isdir(REPO / base) and hits:
            return hits  # namespace package with resolved parents
    return []


def extract_python_imports(rel: str):
    try:
        tree = ast.parse(read(rel))
    except SyntaxError as e:
        G.mark_incomplete(rel, f"python parse failed: {e}")
        return
    src_lines = lines_of(rel)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [(a.name, None) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import
                pkg_dir = os.path.dirname(rel)
                for _ in range(node.level - 1):
                    pkg_dir = os.path.dirname(pkg_dir)
                mod = (pkg_dir.replace("/", ".") + ("." + node.module if node.module else ""))
                names = [(mod, [a.name for a in node.names])]
            else:
                names = [(node.module, [a.name for a in node.names])]
        else:
            continue
        nested = node.col_offset > 0
        binding = "interpolated" if nested else "literal"
        text = src_lines[node.lineno - 1]
        for mod, subnames in names:
            if not mod:
                continue
            top = mod.split(".")[0]
            hits = resolve_module(mod, rel)
            targets = set()
            for p, how in hits:
                targets.add(p)
            if subnames:
                # from pkg import name -> name may be a submodule
                for sn in subnames:
                    for p, how in resolve_module(mod + "." + sn, rel):
                        if how in ("module", "package"):
                            targets.add(p)
            if not targets:
                if top in STDLIB or top in ("__future__",):
                    continue
                cand = BY_BASENAME.get(mod.split(".")[-1] + ".py", [])
                if cand:
                    for c in cand:
                        G.edge(rel, c, "imports", rel, node.lineno, text, "unresolved",
                               note=f"module {mod!r} not on any statically visible root; basename matches")
                    G.mark_incomplete(rel, f"import {mod!r} resolved only by basename")
                elif top not in STDLIB:
                    # third-party or truly missing: not a repo file, not an edge
                    pass
                continue
            for t in sorted(targets):
                G.edge(rel, t, "imports", rel, node.lineno, text, binding)


# --------------------------------------------------------------------------------------------
# Rust / Cargo / C / Makefile
# --------------------------------------------------------------------------------------------
CRATES: dict[str, dict] = {}   # binary name -> {"crate_dir", "main", "lib", "cargo"}


def extract_rust():
    for rel in FILE_NODES:
        if os.path.basename(rel) == "Cargo.toml":
            text = read(rel)
            crate_dir = os.path.dirname(rel)
            m = re.search(r'^\[package\]\s*\nname\s*=\s*"([^"]+)"', text, re.M)
            pkg = m.group(1) if m else os.path.basename(crate_dir)
            main = f"{crate_dir}/src/main.rs"
            lib = f"{crate_dir}/src/lib.rs"
            bins: list[tuple[str, str, int]] = []
            for bm in re.finditer(r'\[\[bin\]\]\s*\nname\s*=\s*"([^"]+)"\s*\npath\s*=\s*"([^"]+)"', text):
                bins.append((bm.group(1), os.path.normpath(os.path.join(crate_dir, bm.group(2))),
                             text[:bm.start()].count("\n") + 1))
            if not bins and main in FILESET:
                ln, _ = find_line(rel, "[package]")
                bins.append((pkg, main, ln))
            for name, path, ln in bins:
                CRATES[name] = {"crate_dir": crate_dir, "main": path, "lib": lib if lib in FILESET else None,
                                "cargo": rel, "package": pkg}
                if path in FILESET:
                    G.edge(rel, path, "imports", rel, ln, lines_of(rel)[ln - 1] if ln else "[package]", "literal",
                           note="bin target (explicit [[bin]] or default src/main.rs)")
            if lib in FILESET:
                ln, t = find_line(rel, "[package]")
                G.edge(rel, lib, "imports", rel, ln, t, "literal", note="default lib target src/lib.rs")
            # cargo auto-discovers examples/ and tests/
            for sub in ("examples", "tests", "benches"):
                d = f"{crate_dir}/{sub}"
                for f in FILE_NODES:
                    if f.startswith(d + "/") and f.endswith(".rs"):
                        ln, t = find_line(rel, "[package]")
                        G.edge(rel, f, "imports", rel, ln, t, "interpolated",
                               note=f"cargo auto-discovery of {sub}/")
    for rel in FILE_NODES:
        if not rel.endswith(".rs"):
            continue
        cl = code_lines(rel)
        base = os.path.basename(rel)
        d = os.path.dirname(rel)
        mod_base = d if base in ("mod.rs", "lib.rs", "main.rs") else os.path.join(d, base[:-3])
        for i, l in enumerate(cl, 1):
            m = re.match(r"^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*;", l)
            if m:
                name = m.group(1)
                for cand in (f"{mod_base}/{name}.rs", f"{mod_base}/{name}/mod.rs"):
                    if cand in FILESET:
                        G.edge(rel, cand, "imports", rel, i, l, "literal")
                        break
                else:
                    G.mark_incomplete(rel, f"mod {name}; has no file under {mod_base}")
            m = re.match(r"^\s*use\s+([A-Za-z_][A-Za-z0-9_]*)::", l)
            if m:
                crate = m.group(1)
                for binname, info in CRATES.items():
                    if info["package"] == crate and info["lib"] and info["lib"] != rel:
                        G.edge(rel, info["lib"], "imports", rel, i, l, "literal", note="use <crate>:: -> src/lib.rs")


def extract_c_and_make():
    for rel in FILE_NODES:
        if rel.endswith((".c", ".h")):
            for i, l in enumerate(code_lines(rel), 1):
                m = re.search(r'#include\s*"([^"]+)"', l)
                if m:
                    tgt = os.path.normpath(os.path.join(os.path.dirname(rel), m.group(1)))
                    if tgt in FILESET:
                        G.edge(rel, tgt, "imports", rel, i, l, "literal")
                    else:
                        G.mark_incomplete(rel, f"include {m.group(1)} not found")
        elif os.path.basename(rel) == "Makefile":
            d = os.path.dirname(rel)
            for i, l in enumerate(code_lines(rel), 1):
                for m in re.finditer(r"([A-Za-z0-9_./-]+\.c)\b", l):
                    tgt = os.path.normpath(os.path.join(d, m.group(1)))
                    if tgt in FILESET:
                        G.edge(rel, tgt, "imports", rel, i, l, "literal", note="compiled into bin/<stem>")
                m = re.search(r"\$\(wildcard\s+([^)]+)\)", l)
                if m:
                    pat = m.group(1).strip()
                    rx = re.compile("^" + re.escape(os.path.normpath(os.path.join(d, pat))).replace(r"\*", "[^/]*") + "$")
                    for f in FILE_NODES:
                        if rx.match(f):
                            G.edge(rel, f, "imports", rel, i, l, "interpolated", note="wildcard source list")


# --------------------------------------------------------------------------------------------
# workload steps files -> guest binaries -> their sources (over ssh, executed by run_files_controlled.py)
# --------------------------------------------------------------------------------------------
PHASE2_BIN: dict[str, str] = {}
for _f in FILE_NODES:
    if _f.startswith("VM_executables_phase2/") and _f.endswith(".c"):
        PHASE2_BIN[os.path.basename(_f)[:-2]] = _f
STEPS_FILES = [f for f in FILE_NODES if re.search(r"(steps[^/]*\.txt|steps_[^/]+\.txt|full_steps\.txt)$", f)
               and "plan05_campaign/README" not in f]


def extract_steps():
    for rel in STEPS_FILES:
        seen: dict[str, tuple[int, str]] = {}
        for i, l in enumerate(lines_of(rel), 1):
            if not l.strip() or l.lstrip().startswith("#"):
                continue
            for m in re.finditer(r"/bin/([A-Za-z0-9_]+)\b", l):
                stem = m.group(1)
                tgt = PHASE2_BIN.get(stem)
                if tgt and tgt not in seen:
                    seen[tgt] = (i, l)
                elif not tgt:
                    py = BY_BASENAME.get(stem + ".py")
                    if py and len(py) == 1 and py[0] not in seen:
                        seen[py[0]] = (i, l)
            for m in re.finditer(r"VM_executables/([A-Za-z0-9_]+\.(?:py|sh))\b", l):
                cand = f"VM_executables/{m.group(1)}"
                if cand in FILESET and cand not in seen:
                    seen[cand] = (i, l)
        for tgt, (i, l) in seen.items():
            G.edge(rel, tgt, "invokes", rel, i, l, "interpolated",
                   remote="guest", note="guest binary bin/<stem> built by VM_executables_phase2/Makefile from this source; "
                                        "executed over ssh by run_files_controlled.py")


# --------------------------------------------------------------------------------------------
# artifact catalogue
# --------------------------------------------------------------------------------------------
A = "artifact:"
CATALOGUE: list[dict] = []


def cat(rx: str, node: str, kind: str = "artifact", scope: list[str] | None = None,
        exclude: list[str] | None = None, desc: str = "", **flags):
    CATALOGUE.append({"rx": re.compile(rx), "node": node, "kind": kind, "scope": scope,
                      "exclude": exclude, "desc": desc, "flags": flags})


# --- steps / manifests in the repo (plan07 lineage) ---
cat(r"full_campaign_steps\.txt\.bak", f"{P7}/full_campaign_steps.txt.bak", "config")
cat(r"full_campaign_steps\.txt", f"{P7}/full_campaign_steps.txt", "config")
cat(r"database_steps_manifest\.csv|database_manifest\.csv", AUTHORITY, "generated",
    desc="authoritative for step_index, family, workload, rep, scale, command")
cat(r"database_steps_rawsubset\.txt", f"{P7}/database_steps_rawsubset.txt", "generated")
cat(r"database_steps\.txt", f"{P7}/database_steps.txt", "generated")
cat(r"newfamilies_manifest\.csv", f"{P7}/newfamilies_manifest.csv", "generated")
cat(r"newfamilies_steps\.txt", f"{P7}/newfamilies_steps.txt", "generated")
cat(r"repfill_manifest\.csv", f"{P7}/repfill_manifest.csv", "generated")
cat(r"repfill_steps\.txt", f"{P7}/repfill_steps.txt", "generated")
cat(r"pilot_baseline\.json", f"{P7}/pilot_baseline.json", "config")
cat(r"diskio_subset_manifest\.csv", f"{Q}/plan06_campaign/diskio_subset_manifest.csv", "generated")
cat(r"diskio_subset_steps\.txt", f"{Q}/plan06_campaign/diskio_subset_steps.txt", "generated")
cat(r"full_manifest\.csv", f"{Q}/plan05_campaign/full_manifest.csv", "generated")
cat(r"full_steps\.txt", f"{Q}/plan05_campaign/full_steps.txt", "generated")
cat(r"subset_steps\.txt", f"{Q}/plan05_campaign/subset_steps.txt", "config")
# --- consoles ---
cat(r"capture_console\.template\.html", f"{P7}/ui/capture_console.template.html", "template")
cat(r"capture_console\.served\.html", f"{P7}/ui/capture_console.served.html", "generated")
cat(r"capture_console\.html", f"{P7}/ui/capture_console.html", "generated")
cat(r"analysis_console\.template\.html", f"{P10}/ui/analysis_console.template.html", "template")
cat(r"analysis_console\.served\.html", f"{P10}/ui/analysis_console.served.html", "generated")
cat(r"analysis_console\.html", f"{P10}/ui/analysis_console.html", "generated")
cat(r"manifest\.current\.json", f"{A}plan10/runs/manifest.current.json", scope=[P10])
# --- capture configs ---
cat(r"config_qemu_upc\.json", f"{Q}/config_qemu_upc.json", "config")
cat(r"config_qemu_mac\.json\.example", f"{Q}_MAC/config_qemu_mac.json.example", "config")
cat(r"config_qemu_mac\.json", f"{A}config_qemu_mac.json", desc="runtime copy of config_qemu_mac.json.example; not in the repo")
cat(r"config_qemu\.json", f"{A}config_qemu.json", desc="default CONFIG of the QEMU shell scripts; not in the repo (config_qemu_upc.json is passed by run_files_controlled.py)")
cat(r"config_gate2\.json", f"{A}plan09/config_gate2.json")
cat(r"config_gate3\.json", f"{A}plan09/config_gate3.json")
cat(r"config_timing_experiment\.json", f"{A}plan05/config_timing_experiment.json")
cat(r"\bconfig\.json", f"{VC}/config.json", "config", scope=[VC + "/"])
# --- queue-dir control files ---
cat(r"capture_status\.json", f"{A}queue_dir/capture_status.json", desc="producer writes {captured,state,workload}; bridge reads")
cat(r"capture_control\.json", f"{A}queue_dir/capture_control.json", desc="UI/orchestrator write {command}; producer and orchestrator read")
cat(r"vm_state\.txt", f"{A}queue_dir/vm_state.txt")
cat(r"capture_pids\.txt", f"{A}capture_pids.txt")
cat(r"\{jobId\}\.json|\$jobId\.json|-name '\*\.json'|glob\(\"\*\.json\"\)|-Filter \*\.json", f"{A}queue_dir/pending/<jobId>.json",
    scope=[f"{Q}/capture_", f"{Q}/run_files_controlled.py", f"{Q}_MAC/capture_", f"{VC}/capture_", f"{Q}/plan02_run.py",
           f"{Q}/run_timing_instrumentation_experiment.py", f"{Q}/run_exp2a_consumer_isolation.py"],
    desc="one job file per snapshot pair {prev,curr,output}")
cat(r'/cosine/|/hamming/|memory_dump_(hamming|cosine)_results', f"{A}output_dir/{{cosine,hamming}}/<ts>.txt",
    desc="one frame per pair, written by live_delta_calc (and, legacy, live_delta_calc_modular)")
cat(r"memory_dump(?!_(hamming|cosine))", f"{A}imageDir/memory_dump-<ts>.raw", desc="raw guest RAM dump written by QEMU pmemsave on the producer's request")
cat(r"producer\.log", f"{A}producer.log")
cat(r"consumer\.log", f"{A}consumer.log")
cat(r"migrate_agent\.log", f"{A}migrate_agent.log")
# --- per-pair binary outputs (glob-mediated, class-valued) ---
cat(r"page_metrics-?\*?\.csv|page_metrics-", f"{A}output_dir/metrics/page_metrics-<ts>.csv",
    desc="one sparse per-page feature CSV per snapshot pair, written by live_delta_calc_modular")
cat(r'"\$output/apf"/\*\.txt|/apf/\*\.txt', f"{A}output_dir/apf/<ts>.txt", desc="APF value file per pair, written by apf_calc")
cat(r"seq_\*?\.apf_done|\.apf_done|seq_\{record\.seq:07d\}\.apf_done", f"{A}<step>.apf_acks/seq_<N>.apf_done")
# --- trajectories ---
cat(r"apf_trajectory\.jsonl", f"{Q}/plan06_campaign/downstream/cells_full/work/<cell>/apf_trajectory.jsonl",
    scope=[f"{Q}/plan06_campaign/make_comprehensive_report.py"])
cat(r"diskio_trajectory\.jsonl", f"{Q}/plan06_campaign/downstream/cells_full/work/<cell>/diskio_trajectory.jsonl",
    scope=[f"{Q}/plan06_campaign/make_comprehensive_report.py"])
cat(r"manifest\.csv", f"{A}plan05/cells/manifest.csv", scope=[f"{Q}/plan05_campaign/build_cells_dir.py"],
    desc="cells-dir manifest written by build_cells_dir.py --out; plan06 cells_full/manifest.csv is presumably an instance (not provable statically)")
cat(r"apf_trajectory\.jsonl", f"{A}apf_trajectory.jsonl", desc="per-step APF trajectory (consumer or apf helper appends)")
cat(r"diskio_trajectory\.jsonl", f"{A}diskio_trajectory.jsonl")
cat(r"substrate_trajectory\.csv", f"{A}substrate_trajectory.csv",
    desc="per-step per-changed-page feature rows, seq-stamped; the fat B1 source")
cat(r"snapshot_timings\.jsonl", f"{A}snapshot_timings.jsonl")
cat(r"run_matrix[A-Za-z_{}$0-9.-]*\.npy|\.npy\.zst|RUN_MATRIX=|run_matrix\.npy", f"{A}run_matrix_<step>.npy[.zst]",
    scope=[Q + "/"], exclude=[P10, P8])
cat(r"raw_matrix\.npy", f"{A}raw_matrix.npy")
cat(r"streaming_f\$\{numFrames\}|\{output_prefix\}\.npz|\{output_prefix\}\.json", f"{A}streaming_results/<prefix>.npz|.json")
# --- zstd retention chain ---
cat(r"000000\.zst|\{n\}\.zst|\\d\{6\}\)?\\\.zst|\*\.zst|NNNNNN\.zst|\.zst\b(?!\S)", f"{A}zstd_local/<family>/<workload>/<variant>/rep<N>/<NNNNNN>.zst",
    exclude=[P8, f"{Q}/tests/test_plan08", f"{Q}/batch_offline"],
    desc="delta chain: 000000.zst full base, later files zstd --patch-from deltas")
cat(r"ledger\.jsonl|\.archived\.jsonl", f"{A}.migration/ledger.jsonl")
cat(r"control\.json", f"{A}.migration/control.json", scope=[f"{P7}/ui/"])
# --- plan08 B1 artifacts ---
cat(r"hc_field\.csv", f"{A}hc_field.csv.zst", desc="seq,page_index,hamming,cosine sparse field; written by b1_extract_hamming.py")
cat(r"b1_trajectory\.jsonl", f"{A}b1_trajectory.jsonl")
cat(r"provenance\.json", f"{A}provenance.json", scope=[P8, f"{Q}/tests/test_plan08"])
cat(r"features_long\.csv", f"{A}features_long.csv")
cat(r"b1_windows_meta\.csv", f"{A}b1_windows_meta.csv")
cat(r"b1_windows\.npz", f"{A}b1_windows.npz")
cat(r"b1_ae_results\.json", f"{A}b1_ae_results.json")
# --- plan07 runs/ (gitignored) ---
cat(r'CONFIGS / f"\{label\}\.json"', f"{A}plan07_campaign/configs/<label>.json", scope=[f"{P7}/ui/console_bridge.py"])
cat(r"\{label\}_steps\.txt|subset_run\.txt|\{cfg\['label'\]\}_steps", f"{A}plan07_campaign/runs/<label>_steps.txt", scope=[P7])
cat(r"\{label\}\.json|\{cfg\['label'\]\}\.json", f"{A}plan07_campaign/runs/<label>.json", scope=[P7])
cat(r"\{label\}\.log", f"{A}plan07_campaign/runs/<label>.log", scope=[P7])
cat(r"preflight\.json", f"{A}preflight.json")
# --- plan10 runs/ and L1 store ---
for _n in ("scheme.json", "manifest.json", "source.json", "status.json", "run.log", "bridge.log",
           "features.npz", "features.csv", "sidecar.json"):
    cat(re.escape(_n).replace(r"\.", r"\."), f"{A}plan10/runs/<label>/{_n}", scope=[P10, f"{Q}/tests/test_plan10"])
cat(r"control\.json", f"{A}plan10/runs/<label>/control.json", scope=[P10, f"{Q}/tests/test_plan10"])
cat(r"meta\.json", f"{A}plan10/l1/<key>/meta.json", scope=[f"{P10}/runner/extract.py"])
# --- plan02..05 session artifacts (runtime) ---
_P2SCOPE = [r"re:^VM_sampler/VM_Capture_QEMU/(plan0[2-5]_[^/]+\.py|run_[^/]+\.py|migrate_schema[^/]+|offline_step[^/]+|batch_offline[^/]+|rebuild_matrices[^/]+|combine_run[^/]+|tests/test_plan0[2-5][^/]+|tests/test_run_files[^/]+)$"]
cat(r"session_sentinel\.json", f"{A}plan02/session_sentinel.json", scope=_P2SCOPE)
cat(r'cell_\{[a-z_]+\}\.json|cell_[a-z]\.json|"cell_"|warmup_block0\.json', f"{A}plan02/cells/cell_<id>.json", scope=_P2SCOPE)
cat(r"manifest\.csv", f"{A}plan02/manifest.csv", scope=_P2SCOPE)
cat(r"heartbeat\.json", f"{A}plan02/heartbeat.json", scope=_P2SCOPE)
cat(r"metrics\.json", f"{A}plan02/metrics.json", scope=_P2SCOPE)
cat(r"validate_report\.json", f"{A}plan02/validate_report.json", scope=_P2SCOPE)
cat(r"plan03_recommendation\.json", f"{A}plan03/plan03_recommendation.json", scope=_P2SCOPE)
cat(r"plan04_segmenter_results\.json", f"{A}plan04/plan04_segmenter_results.json", scope=_P2SCOPE)
cat(r"plan04_classification\.json|plan04\*\.json", f"{A}plan04/plan04_classification.json", scope=_P2SCOPE + [P10])
cat(r"tidy\.csv", f"{A}plan02/tidy.csv", scope=_P2SCOPE)
cat(r"acceptance_thresholds\.json", f"{A}plan02/acceptance_thresholds.json", scope=_P2SCOPE)
cat(r"iv_recommendations\.json", f"{A}plan02/iv_recommendations.json", scope=_P2SCOPE)
cat(r"plan05_summary\.json", f"{A}plan05/plan05_summary.json", scope=_P2SCOPE)
cat(r"summary\.json", f"{A}plan03/summary.json", scope=_P2SCOPE)
cat(r"plan05_run_records\.json", f"{A}plan05/plan05_run_records.json", scope=_P2SCOPE)
cat(r"run_record\.json", f"{A}plan05/run_record.json", scope=_P2SCOPE)
cat(r"sweep_full\.csv", f"{Q}/plan06_campaign/downstream/sweep_full.csv", "artifact")
cat(r"diskio_lift_full\.json", f"{Q}/plan06_campaign/downstream/diskio_lift_full.json", "artifact")
cat(r"plan05_campaign/downstream/sweep\.csv|DOWNSTREAM / \"sweep\.csv\"", f"{Q}/plan05_campaign/downstream/sweep.csv", "artifact")
cat(r"sweep\.csv", f"{Q}/plan06_campaign/downstream/sweep.csv", "artifact", scope=[f"{Q}/plan06_campaign/"])
cat(r"sweep\.csv", f"{Q}/plan05_campaign/downstream/sweep.csv", "artifact",
    scope=[f"{Q}/plan05_campaign/", f"{Q}/plan07_campaign/separability", P10])
cat(r"sweep\.csv", f"{A}plan03/sweep.csv", scope=_P2SCOPE,
    desc="plan03_sweep.py --output-csv; whether plan05_campaign/downstream/sweep.csv is an instance of it is not statically provable")
cat(r"leakage_ablation\.json", f"{Q}/plan05_campaign/downstream/leakage_ablation.json", "artifact")
cat(r"recommendation\.json", f"{Q}/plan05_campaign/downstream/recommendation.json", "artifact", scope=[P10])
cat(r'DOWNSTREAM\.glob\("\*\.json"\)', f"{Q}/plan05_campaign/downstream/summary.json", "artifact", scope=[P10])
# --- offline metrics ---
cat(r"plv_baseline_aware\.json", f"{A}offline/<step>/plv_baseline_aware.json")
cat(r"baseline_plv\.npy", f"{A}offline/baseline_plv.npy")
cat(r"segment_meta\.json", f"{A}offline/<step>/segment_meta.json")
cat(r"streaming\.json", f"{A}offline/<step>/streaming.json", scope=[f"{Q}/offline_step_metrics.py", f"{Q}/batch_offline", f"{Q}/rebuild_matrices"])
cat(r"meta\.json", f"{A}offline/<step>/meta.json", scope=[f"{Q}/offline_step_metrics.py"])
cat(r"timestamps\.log", f"{A}timestamps.log", desc="run_files_controlled.py per-step timestamps")
# --- phase2 workload metadata ---
cat(r"_metadata\.json", f"{A}<workload>_metadata.json", desc="phase2 workload counters; written by phase2_common, read by mp_* and gate1")
cat(r"gate1_results\.csv", f"{A}plan09/gate1_results.csv")
cat(r"keep_dumps_compression_report\.json", f"{A}keep_dumps_compression_report.json")
cat(r"keep_dumps_compression_report\.txt", f"{A}keep_dumps_compression_report.txt")
# --- generated docs ---
for _n in ("plan06_all_data.csv", "plan06_comprehensive.html", "plan07_execution_log.html", "feature_substrate_spec.md",
           "plan07_handout.html", "substrate_progress.html", "test_families_spec.md", "test_families_spec.html"):
    cat(re.escape(_n), f"{Q}/docs/{_n}", "generated")
for _n in ("plan06_comprehensive.pdf", "plan06_diskio_method.pdf", "feature_substrate_spec.pdf", "plan07_handout.pdf",
           "test_families_spec.pdf"):
    cat(re.escape(_n), f"{Q}/docs/{_n}", "generated", binary_instance_excluded=True)
cat(r"substrate_progress\.json", f"{A}substrate_progress.json", desc="browser-side download from substrate_progress.html")

TOKEN_RX = re.compile(
    r"""[A-Za-z0-9_./*{}$'\[\]<>:-]+\.(?:json|jsonl|csv|txt|zst|npy|npz|log|html|md|pdf|sh|py|rs|toml|raw|apf_done|example|bak)\b(?:\.(?:zst|gz|tmp|bak|example))*"""
)

WRITE_PY = re.compile(r"write_text\(|write_bytes\(|json\.dump\(|\.to_csv\(|np\.save\(|savez|\.open\([\"'][wa]|open\([^)]*[\"'][wa]b?[\"']|\.write\(|mkstemp|\.replace\(|os\.replace\(|shutil\.(?:copy|move|copyfile)|savefig|SimpleDocTemplate|os\.fdopen\(fd, [\"']w|with_suffix|\.tmp\b|_field_writer\(|\btee\b|\.rename\(|\b(write|save|dump|emit)_\w*\(")
READ_PY = re.compile(r"read_text\(|read_bytes\(|json\.load\(|json\.loads\(|\.open\(\)|\.open\([\"']r|(?<![\w.])open\((?![^)]*[\"'][wa]b?[\"'])|np\.load\(|DictReader|read_csv|loadtxt|\.read\(\)|readlines|splitlines|csv\.reader|--manifest|--steps|--facts|zstd\W+-dc?\b|\b(parse|load|read|cell)_\w*\(")
STAT_PY = re.compile(r"\.stat\(\)|\.exists\(\)|is_file\(\)|is_dir\(\)|os\.path\.isfile|os\.path\.exists|\.unlink\(|os\.remove\(|rmtree|missing_ok|\.glob\(|rglob\(|\.name ==|endswith|startswith|\.stem|\.with_name\(|\.parent\b")
WRITE_FLAGS = ("--out", "--output", "--output-csv", "--output-json", "--json", "-o", "--summary-json", "--report")
READ_FLAGS = ("--config", "--manifest", "--steps", "--facts", "--scheme", "--matrix", "--baseline-dir", "--input", "--curation", "--root", "--source-json", "--sources")


def classify_line(rel: str, line: str, tok_start: int, tok: str) -> tuple[str, str]:
    """('writes'|'reads'|'stat'|'mention', reason) for an artifact token on a code line."""
    ext = os.path.splitext(rel)[1]
    before = line[:tok_start]
    after = line[tok_start + len(tok):]
    if ext == ".ps1":
        if re.search(r"Get-Content|Import-Csv|ConvertFrom-Json", line):
            return "reads", "Get-Content"
        if re.search(r"Set-Content|Out-File|Add-Content|Export-Csv|ConvertTo-Json", line):
            return "writes", "Set-Content"
        if re.search(r"^\s*\$\w+\s*=", line):
            return "assign", "var"
        if re.search(r"\b(throw|Write-Host|Write-Output)\b", line):
            return "mention", "message text"
        return "mention", "no verb"
    if ext in (".sh", "") or rel.endswith("Makefile"):
        # classify within the command segment holding the token (split on && || ; | outside quotes, roughly)
        seg_start = 0
        for sm in re.finditer(r"\s(?:&&|\|\||;|\|)\s", line):
            if sm.end() <= tok_start:
                seg_start = sm.end()
            else:
                break
        seg_end = len(line)
        sm2 = re.search(r"\s(?:&&|\|\||;|\|)\s", line[tok_start + len(tok):])
        if sm2:
            seg_end = tok_start + len(tok) + sm2.start()
        line = line[seg_start:seg_end]
        tok_start -= seg_start
        before = line[:tok_start]
        after = line[tok_start + len(tok):]
        stripped = line.lstrip()
        is_msg = re.match(r"(?:local\s+\w+=\$\()?(echo|printf|log|warn|fail|pass|die|section|info)\b", stripped) is not None
        if is_msg and not re.search(r"(?<![-=<])>>?\s*[\"']?\$?\{?[A-Za-z_]*[\"']?\s*$", before):
            return "mention", "message text"
        # redirection into the token or a var holding it
        if re.search(r"(?<![-=<])>>?\s*[\"']?\$?\{?[A-Za-z_]*[\"']?\s*$", before) or re.search(r"(?:^|\s)(?:-o|--output|--out|--json|-O|--report)\s+\"?$", before):
            return "writes", "redirect/flag"
        if re.search(r"\b(tee|mv|cp)\b", before) and not re.search(r"\S", after.strip().strip('"').replace("2>&1", "").replace("||", "").replace("&", "")[:1] or ""):
            return "writes", "mv/cp/tee target"
        if re.search(r"\b(mv|cp)\s+\"?$", before):
            return "reads", "mv/cp source"
        if re.search(r"\b(jq|cat|grep|awk|head|tail|sed|wc|sort|cut|zstd -d|zstd -dc|du|python3 -c|source|\.)\s+[^|>]*$", before) or re.search(r"<\s*\"?$", before):
            if re.search(r"\bdu\b", before):
                return "stat", "du"
            return "reads", "read tool"
        if re.search(r"\b(find|ls|rm|test -f|\[\[? -f|\[\[? -d|\[\[? -s)\b", before):
            return "stat", "existence/listing"
        if re.search(r"\b(bash|sh|python3|python|nohup|exec|\$PYTHON|\$\{PYTHON\}|\$PYTHON_BIN|\$\{PYTHON_BIN\})\b", before) and tok.endswith((".py", ".sh")):
            return "invokes", "shell verb"
        if re.search(r"^\s*(local\s+)?[A-Za-z_][A-Za-z0-9_]*=\S*$", before + tok) or re.search(r"^\s*(local\s+)?[A-Za-z_][A-Za-z0-9_]*=", before):
            return "assign", "var"
        if re.search(r"\b(echo|printf|log|die|warn|fail|pass|section|info)\b", before):
            return "mention", "message text"
        for fl in WRITE_FLAGS:
            if before.rstrip().endswith(fl):
                return "writes", "flag"
        for fl in READ_FLAGS:
            if before.rstrip().endswith(fl):
                return "reads", "flag"
        return "mention", "no verb"
    if ext == ".py":
        w = WRITE_PY.search(line)
        r = READ_PY.search(line)
        s = STAT_PY.search(line)
        if re.search(r"^\s*(self\.)?[A-Za-z_][A-Za-z0-9_]*\s*(?::\s*[\w\[\], |]+\s*)?=[^=]", line) and not (w or r):
            if s and re.search(r"\.glob\(|rglob\(", line):
                return "stat", s.group(0)
            return "assign", "var"
        if re.search(r"(subprocess\.(run|Popen|check_output|check_call|call)\(|\b_run\(|\bargv\s*(=|\+=)\s*\[|\bcmd(_parts)?\s*(=|\+=)\s*\[|\bargv\.(append|extend)\(|\bcmd(_parts)?\.(append|extend)\()", line):
            # inside a command line: flags decide
            if tok.endswith((".py", ".sh")):
                return "invokes", "subprocess arg"
            return "passthrough", "argv of a child process"
        if w and not r:
            return "writes", w.group(0)
        if r and not w:
            return "reads", r.group(0)
        if w and r:
            # both: position of the token relative to the verbs
            if w.start() < tok_start and (r.start() > tok_start or r.start() < w.start()):
                return "writes", w.group(0)
            return "reads", r.group(0)
        if s:
            return "stat", s.group(0)
        return "mention", "no verb"
    if ext == ".rs":
        if re.search(r"File::create|write_all|create_dir|BufWriter|OpenOptions", line):
            return "writes", "rust write"
        if re.search(r"File::open|read_to_string|BufReader|fs::read", line):
            return "reads", "rust read"
        return "mention", "no verb"
    return "mention", "no verb"


def _scope_hit(p: str, rel: str) -> bool:
    if p.startswith("re:"):
        return re.search(p[3:], rel) is not None
    if p.endswith("/"):
        return rel.startswith(p)
    if (REPO / p).is_dir():
        return rel == p or rel.startswith(p + "/")
    return rel.startswith(p)


def enclosing_literal_has_space(line: str, a: int, b: int) -> bool:
    """True when the token at [a,b) sits inside a quoted literal that also contains whitespace (a message)."""
    for q in ('"', "'"):
        i = line.rfind(q, 0, a)
        j = line.find(q, b)
        if i != -1 and j != -1:
            lit = line[i + 1:j]
            if re.search(r"\s", lit) and not re.search(r"^\S*\{[^}]*\}\S*$", lit):
                return True
    return False


def in_scope(entry: dict, rel: str) -> bool:
    if entry["exclude"] and any(_scope_hit(p, rel) for p in entry["exclude"]):
        return False
    if entry["scope"] is None:
        return True
    return any(_scope_hit(p, rel) for p in entry["scope"])


def lookup_artifact(tok: str, rel: str, line: str) -> dict | None:
    for e in CATALOGUE:
        if not in_scope(e, rel):
            continue
        if e["rx"].search(tok) or (e["rx"].pattern.startswith(("CONFIGS", "DOWNSTREAM", '"\\$output', 'DOWNSTREAM')) and e["rx"].search(line)):
            return e
    return None


def token_binding(tok: str) -> str:
    return "interpolated" if re.search(r"[$*{}<>]|\bf\"", tok) else "literal"


def scan_artifacts(rel: str):
    ext = os.path.splitext(rel)[1]
    if ext not in CODE_EXT and not rel.endswith("Makefile"):
        return
    if rel.endswith(".toml"):
        return
    cl = code_lines(rel)
    raw = lines_of(rel)
    consts = py_consts(rel) if ext == ".py" else {}
    # NAME -> (artifact entry, def line, token) for assignment lines, then uses are classified
    var_defs: dict[str, tuple[dict, int, str, tuple[int, int]]] = {}
    pending_mentions: list[tuple[int, str, str]] = []

    func_ranges: list[tuple[int, int]] = []
    if ext == ".py":
        try:
            for nd in ast.walk(ast.parse(read(rel))):
                if isinstance(nd, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_ranges.append((nd.lineno, nd.end_lineno or nd.lineno))
        except SyntaxError:
            pass
    elif ext == ".sh":
        start = None
        for i, l in enumerate(cl, 1):
            if re.match(r"^\s*(function\s+)?[A-Za-z_][A-Za-z0-9_]*\s*\(\)\s*\{", l) or re.match(r"^\s*function\s+\w+", l):
                start = i
            elif start is not None and re.match(r"^\}", l):
                func_ranges.append((start, i))
                start = None

    def scope_of(line_no: int, local: bool) -> tuple[int, int]:
        inner = [r for r in func_ranges if r[0] <= line_no <= r[1]]
        if inner and (local or ext == ".py"):
            return min(inner, key=lambda r: r[1] - r[0])
        return (1, len(cl))

    def emit(entry: dict, i: int, action: str, tok: str, reason: str, via: str | None = None):
        node = entry["node"]
        if node in FILESET or node in G.nodes:
            G.node(node, entry["kind"] if entry["kind"] != "artifact" or node.startswith(A) else "artifact", **entry["flags"])
        else:
            G.node(node, entry["kind"], **entry["flags"])
            if entry["desc"]:
                G.nodes[node]["desc"] = entry["desc"]
        if node == rel:
            return
        binding = token_binding(tok)
        if via:
            binding = "interpolated"
        extra = {"class_valued": True} if ("*" in tok or "<" in node and node.startswith(A) and re.search(r"[*{]", tok)) else {}
        if via:
            extra["via"] = via
        if "tests/" in rel and "/tests/" in "/" + rel:
            extra["role"] = "test"
        G.edge(rel, node, action, rel, i, raw[i - 1], binding, **extra)

    for i, l in logical_lines(rel):
        if not l.strip():
            continue
        for m in TOKEN_RX.finditer(l):
            tok = m.group(0).strip("'\"[]")
            if ext in (".py", ".rs") and enclosing_literal_has_space(l, m.start(), m.end()):
                continue  # prose inside a message string, not a path
            if not tok or tok.endswith((".py", ".sh", ".rs", ".toml")) and not re.search(r"read_text|open\(|sed -i|dos2unix|\.write", l):
                # scripts referenced as programs are handled by the invocation scanners;
                # scripts read as DATA fall through (read_text / open / sed -i)
                if tok.endswith((".py", ".sh", ".rs", ".toml")):
                    continue
            entry = lookup_artifact(tok, rel, l)
            if entry is None:
                # a source file read as data?
                if tok.endswith((".py", ".sh", ".rs", ".toml", ".c", ".h")) and re.search(r"read_text|open\(|sed -i|dos2unix", l) \
                        and (ext != ".py" or re.search(r'(/\s*"[^"]*' + re.escape(os.path.basename(tok)) + r'"|(open|Path)\(\s*"[^"]*' + re.escape(os.path.basename(tok)) + r'")', l)):
                    base = os.path.basename(tok)
                    hints = re.findall(r'"([A-Za-z0-9_]+)"', l)
                    # fold constants on the line: NAME / "a" / "b.rs"
                    cm = re.search(r"([A-Z_][A-Z0-9_]*)((?:\s*/\s*\"[^\"]+\")+)", l)
                    tgt = None
                    if cm and cm.group(1) in consts:
                        segs = re.findall(r'"([^"]+)"', cm.group(2))
                        folded = norm_repo(os.path.join(consts[cm.group(1)][0], *segs))
                        if folded in FILESET:
                            tgt = folded
                    if tgt is None:
                        tgt, why = resolve_by_basename(base, hints, rel)
                        if tgt is None:
                            G.mark_incomplete(rel, f"line {i}: source-as-data {tok!r} unresolved ({why})")
                            continue
                    action = "writes" if re.search(r"sed -i|dos2unix|\.write", l) else "reads"
                    G.edge(rel, tgt, action, rel, i, raw[i - 1], token_binding(tok), note="source file consumed as data")
                continue
            action, reason = classify_line(rel, l, m.start(), m.group(0))
            if action == "assign":
                vm = re.match(r"^\s*(local\s+)?((?:self\.)?[A-Za-z_][A-Za-z0-9_]*)\s*(?::\s*[\w\[\], |]+)?\s*=", l)
                if vm:
                    var_defs[vm.group(2)] = (entry, i, tok, (1, len(cl)) if vm.group(2).startswith("self.") else scope_of(i, vm.group(1) is not None))
                    continue
                action = "mention"
            if action in ("writes", "reads"):
                emit(entry, i, action, tok, reason)
            elif action == "invokes":
                pass  # handled by invocation scanner
            elif action == "stat":
                G.blind["metadata_only_touches"].append({"file": rel, "line": i, "text": raw[i - 1].strip()[:160], "artifact": entry["node"]})
            elif action == "passthrough":
                G.blind["argv_pass_through"].append({"file": rel, "line": i, "text": raw[i - 1].strip()[:160], "artifact": entry["node"],
                                                     "note": "path handed to a child process; the child's own edge carries the read/write"})
            elif reason != "message text":
                pending_mentions.append((i, tok, entry["node"]))
    # variable uses (worklist: a use may define a second variable)
    queue = list(var_defs.keys())
    done_vars: set[str] = set()
    while queue:
        var = queue.pop(0)
        if var in done_vars:
            continue
        done_vars.add(var)
        entry, defline, tok, (lo, hi) = var_defs[var]
        used = False
        if len(var) == 1:
            rx = re.compile(r"(?<![A-Za-z0-9_.$\"'])" + re.escape(var) + r"(?=\.|\s*[,)])")
        else:
            rx = re.compile(r"(?<![A-Za-z0-9_.])\$?\{?" + re.escape(var) + r"\}?(?![A-Za-z0-9_])")
            if var.startswith("self."):
                rx = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(var) + r"(?![A-Za-z0-9_])")
        for j, l in logical_lines(rel):
            if j == defline or j < lo or j > hi or not rx.search(l):
                continue
            for um in rx.finditer(l):
                action, reason = classify_line(rel, l, um.start(), um.group(0))
                if action in ("writes", "reads"):
                    emit(entry, j, action, tok, reason, via=f"{var} defined at {rel}:{defline}")
                    used = True
                elif action == "passthrough":
                    G.blind["argv_pass_through"].append({"file": rel, "line": j, "text": raw[j - 1].strip()[:160], "artifact": entry["node"],
                                                         "note": "path handed to a child process; the child's own edge carries the read/write"})
                    used = True
                elif action == "stat":
                    G.blind["metadata_only_touches"].append({"file": rel, "line": j, "text": raw[j - 1].strip()[:160], "artifact": entry["node"]})
                    used = True
                elif action == "assign":
                    # re-assignment through another variable: chase only a plain alias
                    vm = re.match(r"^\s*(local\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", l)
                    if vm and vm.group(2) != var and vm.group(2) not in var_defs:
                        rhs = vm.group(3).strip()
                        alias = re.fullmatch(r'"?\$\{?' + re.escape(var) + r'\}?"?|"?\$\{' + re.escape(var) + r':-[^}]*\}"?|' + re.escape(var)
                                             + r'|str\(' + re.escape(var) + r'\)|Path\(' + re.escape(var) + r'\)(\.resolve\(\))?', rhs)
                        if alias:
                            var_defs[vm.group(2)] = (entry, j, tok, scope_of(j, vm.group(1) is not None))
                            queue.append(vm.group(2))
        if not used:
            pending_mentions.append((defline, tok, entry["node"]))
    for i, tok, node in pending_mentions:
        G.blind["unclassified_mentions"].append({"file": rel, "line": i, "text": raw[i - 1].strip()[:160], "artifact": node})
        G.mark_incomplete(rel, f"line {i}: artifact name {tok!r} on a code line with no read/write verb")


# --------------------------------------------------------------------------------------------
# invocation scanners
# --------------------------------------------------------------------------------------------
def shell_var_default(rel: str, var: str) -> tuple[str | None, int]:
    """Default value of VAR from VAR="${VAR:-...}" / VAR="$X/y.sh" / VAR="$(dirname "$0")/y.sh"."""
    for i, l in enumerate(code_lines(rel), 1):
        m = re.match(r"^\s*(?:local\s+)?" + re.escape(var) + r"=\"?\$\{" + re.escape(var) + r":-([^}]+)\}\"?", l)
        if m:
            return m.group(1), i
        m = re.match(r"^\s*(?:local\s+)?" + re.escape(var) + r"=\"?([^\"]+)\"?\s*$", l)
        if m:
            return m.group(1), i
        m = re.match(r"^\s*(?:local\s+)?" + re.escape(var) + r"=\$\(jq -r '[^']*// \"([^\"]+)\"'", l)
        if m:
            return m.group(1), i
        m = re.match(r"^\s*(?:local\s+)?" + re.escape(var) + r"=(.*?/([A-Za-z0-9_.-]+\.(?:sh|py)))\"?\s*$", l)
        if m:
            return m.group(1), i
    return None, 0


def scan_shell_invocations(rel: str):
    cl = code_lines(rel)
    raw = lines_of(rel)
    for i, l in enumerate(cl, 1):
        if not l.strip() or re.match(r"^\s*(echo|printf)\b", l):
            continue
        # source / .
        m = re.search(r"(?:^|[;&|(]\s*|\s)(?:source|\.)\s+\"?([^\s\"]+\.sh)\"?", l)
        if m:
            tok = m.group(1)
            tgt = resolve_shell_target(rel, tok)
            if tgt:
                G.edge(rel, tgt[0], "imports", rel, i, raw[i - 1], tgt[1], via=tgt[2])
            else:
                G.mark_incomplete(rel, f"line {i}: source target {tok!r} unresolved")
        # verbs
        for m in re.finditer(r"(?:^|[;&|(]|\s)(bash|sh|python3|python|\"?\$\{?PYTHON(?:_BIN)?\}?\"?|\"?\$streamingPython\"?)\s+((?:-[a-zA-Z]+\s+)*)(\"?[^\s\"|;&)]+\"?)", l):
            verb, opts, tok = m.group(1), m.group(2), m.group(3).strip('"')
            if "-m" in opts:
                # python -m module
                mod = tok
                if mod.startswith("$"):
                    val, dl = shell_var_default(rel, mod.strip("${}"))
                    if val and "//" in val:
                        val = val.split("//")[-1].strip().strip("\"' ")
                    mod = val or mod
                    binding = "interpolated"
                else:
                    binding = "literal"
                hits = resolve_module(mod.strip('"'), rel)
                if hits:
                    for p, how in hits:
                        if how in ("module", "package"):
                            G.edge(rel, p, "invokes", rel, i, raw[i - 1], binding, note="python -m module")
                else:
                    G.mark_incomplete(rel, f"line {i}: python -m {mod!r} unresolved")
                continue
            if tok in ("-c", "-", "--version") or tok.startswith("-"):
                continue
            if not re.search(r"\.(sh|py|ps1)$", tok) and "$" not in tok:
                continue
            tgt = resolve_shell_target(rel, tok)
            if tgt:
                G.edge(rel, tgt[0], "invokes", rel, i, raw[i - 1], tgt[1], via=tgt[2])
            elif re.search(r"\.(sh|py)$", tok) or tok.startswith("$"):
                if tok.startswith("$") and not re.search(r"SCRIPT|AGENT|PRODUCER|CONSUMER|HERE|ROOT|QEMU_DIR|BIN", tok):
                    continue
                G.blind["unresolved_invocations"].append({"file": rel, "line": i, "token": tok, "text": raw[i - 1].strip()[:160]})
                G.mark_incomplete(rel, f"line {i}: invocation target {tok!r} unresolved")
        # ./x.sh or $HERE/x.py executed directly
        for m in re.finditer(r"(?:^|[;&|(]|\s)(\./[A-Za-z0-9_./-]+\.(?:sh|py))\b", l):
            tok = m.group(1)
            tgt = resolve_shell_target(rel, tok)
            if tgt:
                G.edge(rel, tgt[0], "invokes", rel, i, raw[i - 1], tgt[1], via=tgt[2])
        # ssh remote commands: declare
        if re.search(r"(?:^|\s)ssh\s", l):
            G.blind["remote_execution_sites"].append({"file": rel, "line": i, "text": raw[i - 1].strip()[:160]})


def resolve_shell_target(rel: str, tok: str) -> tuple[str, str, str | None] | None:
    """(repo path, binding, via) for a shell token naming a script."""
    via = None
    binding = "literal"
    t = tok.strip('"\'')
    if t.startswith("$"):
        var = re.match(r"^\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?(.*)$", t)
        if not var:
            return None
        val, dl = shell_var_default(rel, var.group(1))
        if not val:
            if re.search(r"/[A-Za-z0-9_.-]+\.(sh|py)$", var.group(2)):
                t = var.group(2)
                via = f"{var.group(1)} is a directory variable (unfolded)"
                binding = "interpolated"
            else:
                return None
        else:
            t = val + var.group(2)
            via = f"{var.group(1)} default at {rel}:{dl}"
            binding = "interpolated"
    if "$" in t or "(" in t:
        binding = "interpolated"
    base = os.path.basename(t)
    if not re.search(r"\.(sh|py|ps1)$", base):
        return None
    # relative path from the script's own directory first
    if not t.startswith(("$", "/")):
        cand = os.path.normpath(os.path.join(os.path.dirname(rel), t.lstrip("./") if t.startswith("./") else t))
        if cand in FILESET:
            return cand, binding, via
        cand2 = os.path.normpath(os.path.join(Q, t))
        if cand2 in FILESET:
            return cand2, binding, via
    hints = [h for h in re.findall(r"[A-Za-z0-9_]+", t) if h not in ("bin", "HOME")]
    tgt, why = resolve_by_basename(base, hints, rel)
    if tgt:
        return tgt, binding, via
    return None


def scan_python_invocations(rel: str):
    """subprocess targets: str(CONST) / CONST / literal script names inside command lists."""
    try:
        tree = ast.parse(read(rel))
    except SyntaxError:
        return
    consts = py_consts(rel)
    raw = lines_of(rel)
    call_names = {"run", "Popen", "check_output", "check_call", "call", "_run", "system", "popen"}
    # gather per-function list assignments (argv = [...]; argv += [...]; cmd_parts = [...])
    list_defs: dict[tuple[int, str], list[ast.AST]] = defaultdict(list)

    def owner(node):
        return id(node)

    def local_walk(fn):
        """Nodes of fn's own body, not descending into nested function/class definitions."""
        out = []
        stack = list(fn.body) if hasattr(fn, "body") else []
        while stack:
            n = stack.pop()
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            out.append(n)
            stack.extend(ast.iter_child_nodes(n))
        return out

    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module))]:
        body_nodes = local_walk(fn)
        for n in body_nodes:
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and isinstance(n.value, (ast.List, ast.Tuple)):
                list_defs[(owner(fn), n.targets[0].id)].extend(n.value.elts)
            if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name) and isinstance(n.value, (ast.List, ast.Tuple)):
                list_defs[(owner(fn), n.target.id)].extend(n.value.elts)
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) \
                    and n.value.func.attr in ("append", "extend") and isinstance(n.value.func.value, ast.Name):
                list_defs[(owner(fn), n.value.func.value.id)].extend(n.value.args)
        for n in body_nodes:
            if not isinstance(n, ast.Call):
                continue
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
            if name not in call_names:
                continue
            elems: list[ast.AST] = []
            for a in n.args:
                if isinstance(a, (ast.List, ast.Tuple)):
                    elems.extend(a.elts)
                elif isinstance(a, ast.Name) and (owner(fn), a.id) in list_defs:
                    elems.extend(list_defs[(owner(fn), a.id)])
                else:
                    elems.append(a)
            for e in elems:
                target_expr = e
                if isinstance(e, ast.Call) and isinstance(e.func, ast.Name) and e.func.id == "str" and e.args:
                    target_expr = e.args[0]
                folded, binding = _fold(target_expr, {k: v[0] for k, v in consts.items()}, rel)
                if folded is None and isinstance(target_expr, ast.Name) and target_expr.id in consts:
                    folded, binding, _ = consts[target_expr.id]
                if folded is None:
                    if isinstance(target_expr, ast.Constant) and isinstance(target_expr.value, str) and re.search(r"\.(py|sh)$", target_expr.value):
                        folded, binding = target_expr.value, "literal"
                    else:
                        continue
                if not re.search(r"\.(py|sh|ps1)$", folded) and "target/release" not in folded:
                    continue
                if isinstance(target_expr, ast.Name) and target_expr.id in consts:
                    binding = consts[target_expr.id][1]
                line = n.lineno
                text = raw[line - 1]
                if "target/release" in folded:
                    binname = os.path.basename(folded)
                    info = CRATES.get(binname)
                    if info:
                        G.edge(rel, info["main"], "invokes", rel, line, text, binding,
                               via=f"release binary {binname} -> crate {info['crate_dir']} (Cargo.toml bin target)",
                               binary=binname)
                    else:
                        G.blind["unresolved_invocations"].append({"file": rel, "line": line, "token": folded, "text": text.strip()[:160]})
                    continue
                tgt = norm_repo(folded)
                if tgt not in FILESET:
                    base = os.path.basename(folded)
                    tgt, why = resolve_by_basename(base, [], rel)
                    if tgt is None:
                        G.blind["unresolved_invocations"].append({"file": rel, "line": line, "token": folded, "text": text.strip()[:160]})
                        G.mark_incomplete(rel, f"line {line}: subprocess target {folded!r} unresolved")
                        continue
                    binding = "interpolated"
                extra = {}
                if isinstance(target_expr, ast.Name):
                    extra["via"] = f"module constant {target_expr.id} at {rel}:{consts.get(target_expr.id, ('', '', 0))[2]}"
                if "/tests/" in "/" + rel:
                    extra["role"] = "test"
                G.edge(rel, tgt, "invokes", rel, line, text, binding, **extra)


def scan_ps1(rel: str):
    raw = lines_of(rel)
    for i, l in enumerate(code_lines(rel), 1):
        for m in re.finditer(r"([A-Za-z0-9_]+\.ps1)", l):
            tok = m.group(1)
            if tok == os.path.basename(rel):
                continue
            tgt, why = resolve_by_basename(tok, [], rel)
            if tgt and re.search(r"\$Prod|\$Cons|Start-Process|&\s|\.\\|Join-Path", l) or (tgt and re.search(r"\[string\]\$\w+\s*=", l)):
                G.edge(rel, tgt, "invokes", rel, i, raw[i - 1], "interpolated", note="PowerShell parameter default / launch")
        if "config.json" in l and re.search(r"Get-Content|ConvertFrom-Json", l):
            if "C:\\" in l:
                G.blind["unresolved_reads"].append({"file": rel, "line": i, "text": raw[i - 1].strip()[:160], "why": "absolute Windows path outside the repo"})
                G.mark_incomplete(rel, f"line {i}: absolute Windows config path")


# --------------------------------------------------------------------------------------------
# HTML templates: <script src>, fetch( call sites
# --------------------------------------------------------------------------------------------
def scan_html(rel: str):
    raw = lines_of(rel)
    for i, l in enumerate(raw, 1):
        for m in re.finditer(r"<script[^>]*\bsrc=[\"']([^\"']+)[\"']", l):
            src = m.group(1)
            if src.startswith(("http:", "https:", "//")):
                G.blind["external_resources"].append({"file": rel, "line": i, "url": src})
                continue
            tgt = os.path.normpath(os.path.join(os.path.dirname(rel), src))
            if tgt in FILESET:
                G.edge(rel, tgt, "imports", rel, i, l, "literal")
            else:
                G.mark_incomplete(rel, f"line {i}: script src {src!r} not in repo")
        if "fetch(" in l:
            G.blind["browser_to_bridge_http"].append({"file": rel, "line": i, "text": l.strip()[:160]})


# --------------------------------------------------------------------------------------------
# curated semantic edges (text needles; the script finds the line numbers)
# --------------------------------------------------------------------------------------------
def curated(src: str, dst: str, typ: str, needle: str, binding: str, file: str | None = None, nth: int = 1, **extra):
    file = file or src
    ln, text = find_line(file, needle, nth)
    if ln == 0:
        G.blind["stale_curated_evidence"].append({"src": src, "dst": dst, "type": typ, "file": file, "needle": needle})
        G.edge(src, dst, typ, file, 0, needle, "unresolved", **extra)
        G.mark_incomplete(src, f"curated evidence {needle!r} not found in {file}")
        return
    G.edge(src, dst, typ, file, ln, text, binding, **extra)


def curated_edges():
    CB = f"{P7}/ui/console_bridge.py"
    SR = f"{P7}/subset_run.py"
    RFC = f"{Q}/run_files_controlled.py"
    RQC = f"{Q}/run_qemu_capture.sh"
    PROD = f"{Q}/capture_producer_qemu_pmemsave.sh"
    PROD2 = f"{Q}/capture_producer_qemu.sh"
    PROD3 = f"{Q}/capture_producer_qemu_user_raw.sh"
    CONS = f"{Q}/capture_consumer_qemu.sh"
    CFG = f"{Q}/config_qemu_upc.json"
    AB = f"{P10}/ui/analysis_bridge.py"
    DIFF = f"{P10}/runner/differ.py"
    LDC = f"{VC}/live_delta_calc/src/main.rs"
    LDCM = f"{VC}/live_delta_calc_modular/src/main.rs"
    APF = f"{VC}/apf_calc/src/main.rs"

    # console.sh -> server-side build + bridge (over ssh), laptop-side agent
    curated(TRUE_ROOTS[0], f"{P7}/ui/build_console.py", "invokes", "python3 plan07_campaign/ui/build_console.py --served", "literal",
            remote="server", note="runs on the research server inside the ssh command string")
    curated(TRUE_ROOTS[0], CB, "invokes", "exec python3 plan07_campaign/ui/console_bridge.py --port", "literal",
            remote="server", note="runs on the research server inside the ssh command string")
    # console_bridge: the launch line comes back from subset_run's metadata and is exec'd under screen
    curated(CB, RFC, "invokes", '_run(["screen", "-dmS", session, "bash", "-c", script]', "interpolated",
            via=f"launch_line read from runs/<label>.json; composed at {SR}: add_constant_envvars(\"./run_files_controlled.py\")",
            note="the bridge never names run_files_controlled.py in the argv; subset_run.py does")
    G.blind["composition_sites"].append({"file": SR, "line": find_line(SR, 'launch = add_constant_envvars("./run_files_controlled.py")')[0],
                                        "text": 'launch = add_constant_envvars("./run_files_controlled.py")',
                                        "note": "subset_run.py composes the launch line but does not execute it; not drawn as an edge"})
    curated(SR, f"{A}plan07_campaign/runs/<label>_steps.txt", "writes", 'out_path.write_text("\\n".join(steps) + "\\n")', "interpolated",
            note="--out path (default subset_run.txt; the bridge passes runs/<label>_steps.txt)")
    # runtime-alternative producers named in the bridge (pkill/pgrep coupling, not a launch site)
    for p in (PROD, PROD2):
        curated(CB, p, "invokes", 'PRODUCER_SCRIPTS = ("capture_producer_qemu_pmemsave.sh", "capture_producer_qemu.sh")', "unresolved",
                alternative=True, note="PRODUCER_SCRIPTS tuple is consumed by pkill -f (stop) and pgrep -f (health), i.e. process-name coupling; "
                                       "the launch path reaches the producer through run_files_controlled.py -> run_qemu_capture.sh")
    curated(CB, CONS, "invokes", 'CONSUMER_SCRIPT = "capture_consumer_qemu.sh"', "unresolved", alternative=True,
            note="same process-name coupling as PRODUCER_SCRIPTS")
    # run_files_controlled -> run_qemu_capture.sh -> producer/consumer
    curated(RFC, RQC, "invokes", 'BACKGROUND=1 ./run_qemu_capture.sh"', "literal",
            note="shell string run via os.system-style run(); PRODUCER_SCRIPT and CONFIG are passed in the environment")
    curated(RFC, PROD, "invokes", 'PRODUCER_SCRIPT={producer_q}', "interpolated",
            via=f"CAPTURE_PRODUCER_SCRIPT env default at {RFC}:67, forwarded as PRODUCER_SCRIPT to run_qemu_capture.sh which nohups it", alternative=True)
    curated(RFC, f"{Q}/offline_step_metrics.py", "invokes", '"python3", OFFLINE_METRICS_SCRIPT,', "interpolated",
            via="OFFLINE_METRICS_SCRIPT env default (offline_step_metrics.py); only when OFFLINE_METRICS_MODE=1")
    curated(RFC, CFG, "reads", 'with open(CAPTURE_CONFIG, "r", encoding="utf-8") as fh:', "interpolated",
            via=f"CAPTURE_CONFIG env default at {RFC}:64")
    for step in ("run_idle.sh", "mem_stream.py", "mem_pointer_chase.py", "mem_alloc_touch_pages.py", "io_seq_fsync.py", "io_rand_rw.py", "io_many_files.py"):
        curated(RFC, f"VM_executables/{step}", "invokes", f"VM_executables/{step}", "literal", remote="guest",
                note="default step sequence when no STEPS_FILE is given; executed on the guest over ssh")
    # run_qemu_capture.sh alternatives (env-selected producer); default already found by the shell scanner
    for p in (PROD2, PROD3):
        curated(RQC, p, "invokes", 'PRODUCER_SCRIPT="${PRODUCER_SCRIPT:-$ROOT/capture_producer_qemu_pmemsave.sh}"', "unresolved",
                alternative=True, note="PRODUCER_SCRIPT is an env override; these siblings are the other producers in the tree "
                                       "(capture_producer_qemu_user_raw.sh:5 states it uses the same queue format and consumer)")
    # producer: pmemsave dump, apf helper
    curated(PROD, f"{A}imageDir/memory_dump-<ts>.raw", "writes", '{"execute":"pmemsave"', "interpolated",
            note="QEMU writes the dump on the producer's monitor command; path from imageDir in CONFIG")
    curated(PROD, f"{A}snapshot_timings.jsonl", "writes", '>> "$TIMING_JSONL_PATH"', "interpolated",
            via=f"TIMING_JSONL_PATH env set by {Q}/run_timing_instrumentation_experiment.py:343 (workdir/snapshot_timings.jsonl)")
    curated(PROD, f"{A}diskio_trajectory.jsonl", "writes", '>> "$TIMING_DISKIO_JSONL"', "interpolated",
            via=f"TIMING_DISKIO_JSONL env set by {Q}/run_files_controlled.py:426 (<base>.diskio_trajectory.jsonl) when CAPTURE_DISKIO=1")
    curated(CONS, f"{A}queue_dir/pending/<jobId>.json", "reads", "jobFile=$(find \"$qPending\" -maxdepth 1 -name '*.json' -print 2>/dev/null | sort | head -1)", "interpolated",
            class_valued=True, note="oldest pending job; prev/curr/output read from it with jq")
    EXE = f"{P10}/runner/executor.py"
    curated(EXE, f"{A}plan10/runs/<label>/scheme.json", "reads", "sch = json.loads(scheme_path.read_text())", "interpolated", note="scheme path is argv (bridge passes runs/<label>/scheme.json)")
    curated(EXE, f"{A}plan10/runs/<label>/manifest.json", "reads", "manifest = json.loads(Path(manifest_path).read_text()) if manifest_path else corpus_manifest.scan_source(src)", "interpolated",
            note="--manifest argv from the bridge; otherwise rescans the source")
    curated(PROD, f"{Q}/plan02_apf_helper.py", "invokes", 'python3 "${SCRIPT_DIR}/plan02_apf_helper.py"', "interpolated",
            note="only when TIMING_APF_STREAM is set (CAPTURE_METRIC=apf)")
    BX = f"{P8}/b1_extract_hamming.py"
    curated(BX, f"{A}hc_field.csv.zst", "writes", 'field_path = out_dir / ("hc_field.csv.zst" if write_field else "hc_field.csv")', "interpolated",
            via=f"_field_writer(field_path, compress=write_field) -> subprocess.Popen([\"zstd\", \"-3\", \"-q\", \"-f\", \"-o\", str(path)]) at {BX}:78")
    curated(BX, f"{A}b1_trajectory.jsonl", "writes", 'with traj_path.open("w") as ftraj:', "interpolated", via=f"traj_path = out_dir / \"b1_trajectory.jsonl\" at {BX}:100")
    curated(BX, f"{A}provenance.json", "writes", 'prov_path.write_text(json.dumps(prov, indent=2) + "\\n")', "interpolated", via=f"prov_path = out_dir / \"provenance.json\" at {BX}:101")
    curated(BX, f"{A}substrate_trajectory.csv", "reads", 'proc = subprocess.Popen(["zstd", "-dc", "-q", path], stdout=subprocess.PIPE)', "interpolated",
            note="src positional argument: plain, .gz or .zst substrate_trajectory.csv")
    GD = f"{P7}/generate_database_steps.py"
    curated(GD, f"{P7}/database_steps.txt", "writes", 'out_p.write_text("\\n".join(c[0] for c in cells) + "\\n")', "interpolated",
            via=f"--out default \"database_steps.txt\" at {GD}:160")
    curated(GD, AUTHORITY, "writes", 'man_p = HERE / (Path(a.out).stem + "_manifest.csv")', "interpolated",
            note="<stem>_manifest.csv next to the steps file; header step_index,family,workload,rep,scale,command")
    curated(GD, f"{P7}/database_steps_rawsubset.txt", "writes", 'sp = HERE / (Path(a.out).stem + "_rawsubset.txt")', "interpolated")
    BAC = f"{P10}/ui/build_analysis_console.py"
    curated(BAC, f"{P10}/ui/analysis_console.served.html", "writes", "out.write_text(html)", "interpolated", via=f"OUT_SERVED at {BAC}:52 when --served", alternative=True)
    curated(BAC, f"{P10}/ui/analysis_console.html", "writes", "out.write_text(html)", "interpolated", via=f"OUT at {BAC}:51 (static build)", alternative=True)
    curated(BAC, f"{Q}/config_qemu_upc.json", "reads", "load_config", "interpolated", via=f"scheme.load_config() reads CONFIG_PATH = QEMU_DIR / \"config_qemu_upc.json\" ({P10}/scheme.py:54)")
    PF = f"{Q}/qa/preflight.sh"
    for chk in sorted(f for f in FILE_NODES if f.startswith(f"{Q}/qa/checks/") and f.endswith(".sh")):
        curated(PF, chk, "imports", '. "$chk"', "interpolated", via=f'for chk in "$QA_ROOT"/checks/[0-9][0-9]_*.sh (glob) at {PF}:75')
    curated(PF, f"{A}preflight.json", "writes", '} > "$QA_JSON"', "interpolated", via="--json REPORT; console_bridge.py passes staging/preflight.json")
    curated(f"{Q}/qa/checks/10_host.sh", CFG, "reads", 'json.load(open(sys.argv[1]))" "$CAPTURE_CONFIG"', "interpolated",
            via=f"CAPTURE_CONFIG default at {PF}:31 (env-overridable, exported to the sourced checks)")
    curated(f"{Q}/qa/lib/common.sh", CFG, "reads", 'python3 - "$CAPTURE_CONFIG" "$key"', "interpolated", via=f"CAPTURE_CONFIG default at {PF}:31")
    for chk, needle in ((f"{Q}/qa/checks/30_steps.sh", 'python3 "$QA_ROOT/analyze_steps.py" "$STEPS_FILE"'),
                        (f"{Q}/qa/checks/60_source_audit.sh", '"$STEPS_FILE" 2>/dev/null')):
        curated(chk, f"{P7}/full_campaign_steps.txt", "reads", needle, "interpolated", alternative=True,
                via=f"STEPS_FILE default at {PF}:32; console_bridge.py passes --steps runs/<label>_steps.txt instead")
        curated(chk, f"{A}plan07_campaign/runs/<label>_steps.txt", "reads", needle, "interpolated", alternative=True,
                via="STEPS_FILE from console_bridge.py --steps (console launch path)")
    curated(RFC, f"{A}queue_dir/capture_control.json", "reads", 'return json.loads(p.read_text()).get("command", "run") or "run"', "interpolated",
            via=f"capture_control_path() = queueDir/capture_control.json at {RFC}:604")
    curated(RFC, f"{A}queue_dir/capture_control.json", "writes", 'tmp.write_text(json.dumps({"command": command}))', "interpolated",
            via=f"capture_control_path() at {RFC}:604; write_capture_control('run') resets the file at step start")
    curated(RFC, f"{A}timestamps.log", "writes", 'with open(log_path, "a", encoding="utf-8") as fh:', "interpolated",
            via=f"TIMESTAMPS_LOG env default timestamps.log at {RFC}:69-72")
    OSM = f"{Q}/offline_step_metrics.py"
    curated(OSM, f"{A}run_matrix_<step>.npy[.zst]", "reads", 'mat = np.load(matrix_path, mmap_mode="r")', "interpolated",
            note="matrix path is argv; run_files_controlled.py passes the per-step run_matrix_<step>.npy")
    curated(OSM, f"{A}offline/baseline_plv.npy", "writes", "np.save(baseline_path, baseline)", "interpolated")
    curated(OSM, f"{A}offline/baseline_plv.npy", "reads", "baseline: np.ndarray = np.load(baseline_path)", "interpolated")
    curated(f"{Q}/plan03_sweep.py", f"{A}plan03/sweep.csv", "writes", "out_csv = Path(args.output_csv).expanduser().resolve()", "interpolated",
            note="--output-csv; whether plan05_campaign/downstream/sweep.csv is this file's committed instance is not statically provable")
    curated(LDCM, f"{A}output_dir/{{cosine,hamming}}/<ts>.txt", "writes", 'format!("{}/hamming/memory_dump_hamming_results_par-{}.txt", output_dir, timestamp);', "interpolated",
            note="legacy per-pair frame files kept by the modular binary")
    curated(f"{Q}/reconstruct_zstd_chain.sh", f"{A}zstd_local/<family>/<workload>/<variant>/rep<N>/<NNNNNN>.zst", "reads",
            'zstd -d -q --long=31 --patch-from="$prev" "$f" -o "$tgt"', "interpolated", class_valued=True)
    curated(f"{P7}/ui/migrate_agent.sh", f"{A}zstd_local/<family>/<workload>/<variant>/rep<N>/<NNNNNN>.zst", "reads",
            'until rsync -a --partial $rmflag -e "ssh $SSH_OPTS"', "interpolated", class_valued=True, remote="server",
            note="moves completed chains from the server to TRACES_LOCAL_DIR on the laptop; same artifact class")
    curated(f"{P7}/ui/migrate_agent.sh", f"{A}.migration/ledger.jsonl", "writes", "cat >> '$LEDGER'", "interpolated", remote="server")
    curated(f"{Q}/pull_traces.sh", f"{A}zstd_local/<family>/<workload>/<variant>/rep<N>/<NNNNNN>.zst", "reads",
            'until rsync -a --partial --remove-source-files -e "ssh $SSH_OPTS"', "interpolated", class_valued=True, remote="server")
    MCR = f"{Q}/plan06_campaign/make_comprehensive_report.py"
    for cls in ("apf_trajectory.jsonl", "diskio_trajectory.jsonl"):
        curated(MCR, f"{Q}/plan06_campaign/downstream/cells_full/work/<cell>/{cls}", "reads", 'CELLS = HERE / "downstream" / "cells_full"', "interpolated",
                via="load_extra(CELLS) at :68 / load_diskio(CELLS) at :69 read <cell>/" + cls + " through plan05_campaign/extra_features.py and plan06_campaign/diskio_features.py")
    # consumer: config-valued binaries resolved through Cargo.toml
    curated(CONS, APF, "invokes", '"$apfCalculationProgram" "$prev" "$curr" "$output"', "interpolated",
            via=f"apfCalculationProgram in {CFG} -> .../apf_calc/target/release/apf_calc -> crate apf_calc (Cargo.toml package)", binary="apf_calc")
    curated(CONS, LDCM, "invokes", '"$substrateProgram" --speed "$substrateSpeed" --sparse', "interpolated",
            via=f"substrateProgram in {CFG} -> .../live_delta_calc_modular/target/release/live_delta_calc_modular -> crate live_delta_calc_modular",
            binary="live_delta_calc_modular")
    curated(CONS, LDC, "invokes", '"$rustDeltaCalculationProgram" "$prev" "$curr" "$output"', "interpolated",
            via=f"rustDeltaCalculationProgram in {CFG} -> .../live_delta_calc/target/release/live_delta_calc -> crate live_delta_calc",
            binary="live_delta_calc")
    curated(CONS, f"{A}output_dir/apf/<ts>.txt", "reads", 'apfFile=$(ls -t "$output/apf"/*.txt 2>/dev/null | head -1)', "interpolated", class_valued=True,
            note="glob-mediated: newest file wins")
    curated(CONS, f"{A}output_dir/metrics/page_metrics-<ts>.csv", "reads", 'metFile=$(ls -t "$output/metrics"/page_metrics-*.csv 2>/dev/null | head -1)', "interpolated",
            class_valued=True, note="glob-mediated: newest file wins")
    curated(CONS, f"{A}output_dir/{{cosine,hamming}}/<ts>.txt", "reads", 'latestFrame=$(ls -t "$output/$subdir"/*.txt 2>/dev/null | head -1)', "interpolated",
            class_valued=True, note="glob-mediated: newest file wins")
    curated(CONS, f"{A}imageDir/memory_dump-<ts>.raw", "reads", 'prev=$(jq -r \'.prev\' "$jobPath")', "interpolated",
            note="prev/curr dump paths come from the job file")
    curated(CONS, f"{A}run_matrix_<step>.npy[.zst]", "writes", "np.save(out_path, new_mat)", "interpolated",
            via="RUN_MATRIX default $qPath/run_matrix.npy; run_files_controlled.py passes a per-step RUN_MATRIX")
    curated(CONS, f"{A}zstd_local/<family>/<workload>/<variant>/rep<N>/<NNNNNN>.zst", "writes",
            'zstd -q -"$ZSTD_LEVEL" --long=31 --patch-from="$prev" "$curr" -o "$dir/${n}.zst"', "interpolated")
    curated(CONS, "coherence_temp_spec_stability/streaming_metrics.py", "invokes",
            '"$streamingPython" -m "$streamingModule" --input "$runNpy" --output "$outPrefix"', "interpolated",
            via="streaming.streamingMetricsModule default coherence_temp_spec_stability.streaming_metrics (jq // fallback and config_qemu_upc.json)")
    curated(CONS, "coherence_temp_spec_stability/raw_matrix_builder.py", "invokes",
            'if eval "$builderProgram --input-dir', "interpolated",
            via="rawRetention.rawBuild.builderProgram default `python3 -m coherence_temp_spec_stability.raw_matrix_builder`")
    curated(CONS, "coherence_temp_spec_stability/stability_validator.py", "invokes",
            'python3 -m "$validatorModule" "$absMatrix"', "interpolated",
            via="rawRetention.rawMetrics.validatorModule default coherence_temp_spec_stability.stability_validator")
    curated(CONS, f"{A}streaming_results/<prefix>.npz|.json", "writes", 'local outPrefix="$streamingOutputDir/streaming_f${numFrames}_', "interpolated",
            note="prefix passed to streaming_metrics.py --output, which writes <prefix>.npz and <prefix>.json")
    # Rust binaries write their per-pair outputs
    curated(LDCM, f"{A}output_dir/metrics/page_metrics-<ts>.csv", "writes", "page_metrics-", "interpolated")
    curated(APF, f"{A}output_dir/apf/<ts>.txt", "writes", 'let path = format!("{}/apf_results_par-{}.txt", apf_dir, ms);', "interpolated")
    curated(LDC, f"{A}output_dir/{{cosine,hamming}}/<ts>.txt", "writes", "cosine", "interpolated")
    for rs in (LDC, LDCM, APF):
        curated(rs, f"{A}imageDir/memory_dump-<ts>.raw", "reads", "File::open", "interpolated", note="prev/curr dump paths are argv")
    # plan02 apf helper writes acks + trajectory, reads dumps
    H = f"{Q}/plan02_apf_helper.py"
    curated(H, f"{A}<step>.apf_acks/seq_<N>.apf_done", "writes", 'target = ack_dir / f"seq_{record.seq:07d}.apf_done"', "interpolated")
    curated(H, f"{A}imageDir/memory_dump-<ts>.raw", "reads", "--prev", "interpolated", note="dump paths are argv")
    # plan10 bridge -> executor / build (module constants) -- also found by the ast scanner; keep the binary hop explicit
    curated(DIFF, LDCM, "invokes", 'r = subprocess.run([str(b), "--speed", str(int(speed)), "--sparse", str(prev), str(curr), str(out_dir)]', "interpolated",
            via=f"DEFAULT_BINARY at {DIFF}:31 -> .../live_delta_calc_modular/target/release/live_delta_calc_modular -> crate live_delta_calc_modular",
            binary="live_delta_calc_modular")
    curated(DIFF, f"{A}output_dir/metrics/page_metrics-<ts>.csv", "reads", "with open(path, newline=\"\") as f:", "interpolated",
            class_valued=True, note="newest metrics CSV the differ wrote into its own out_dir")
    curated(f"{P10}/runner/chain.py", f"{A}zstd_local/<family>/<workload>/<variant>/rep<N>/<NNNNNN>.zst", "reads",
            'cmd = ["zstd", "-d", "-q", "-f"]', "interpolated", class_valued=True)
    curated(f"{P10}/sources.py", f"{A}zstd_local/<family>/<workload>/<variant>/rep<N>/<NNNNNN>.zst", "reads",
            '_RE_SNAP = re.compile(r"^\\d{6}\\.zst$")', "interpolated", class_valued=True, note="LocalSource lists the chain tree; SshSource lists it remotely (find -printf) and fetches with rsync")
    curated(f"{P10}/corpus_manifest.py", f"{A}substrate_trajectory.csv", "reads",
            '_RE_SUBSTRATE = re.compile(r"substrate_trajectory\\.csv(\\.zst|\\.gz)?$")', "interpolated", class_valued=True,
            note="optional metrics root scan: recognises the file by name to record its presence")
    curated(f"{P10}/testing/synth.py", f"{A}zstd_local/<family>/<workload>/<variant>/rep<N>/<NNNNNN>.zst", "writes",
            'subprocess.run(["zstd", "-q", "-3", "-f", "--long=31", f"--patch-from={prev_raw}", str(raw), "-o", str(out)], check=True)', "interpolated",
            role="test", note="synthetic chain for tests, same zstd invocations as the consumer")
    # plan09 gates invoke the config-valued substrate binary too
    for g in ("gate2_contiguity_check.sh", "gate3_pilot_capture.sh"):
        curated(f"{Q}/plan09_dwarf_pilot/{g}", LDCM, "invokes", "SUBSTRATE_PROGRAM=\"$(jq -r '.substrateProgram'", "interpolated",
                via="substrateProgram read from the scratch copy of config_qemu_upc.json", binary="live_delta_calc_modular")
    # delegated Popen: plan02_run/exp2b/exp2c start the producer through e1.start_producer(...)
    E1 = f"{Q}/run_timing_instrumentation_experiment.py"
    curated(E1, PROD, "invokes", '["bash", str(producer_script)]', "interpolated",
            via=f"producer_script parameter; argparse default DEFAULT_PRODUCER at {E1}:76")
    curated(f"{Q}/plan02_run.py", PROD, "invokes", "producer_proc = e1.start_producer(", "interpolated",
            via=f"e1.start_producer(...) with DEFAULT_PRODUCER at {Q}/plan02_run.py:62")
    curated(f"{Q}/run_exp2c_flush_sensitivity.py", PROD, "invokes", '["bash", str(producer)], env=env, stdout=log_f, stderr=log_f,', "interpolated",
            via="producer path is a parameter (e1 defaults)")
    curated(f"{Q}/run_exp2a_consumer_isolation.py", CONS, "invokes", 'DEFAULT_CONSUMER_SCRIPT = Path(__file__).resolve().parent / "capture_consumer_qemu.sh"', "interpolated",
            note="--consumer-script default; the consumer is started/killed around the isolation passes")
    curated(f"{Q}/tests/apf_calc_equivalence.py", APF, "invokes", "subprocess.run([binary, str(a), str(b), str(out)], check=True)", "interpolated",
            role="test", via="binary path is argv[1]; the docstring names ../VM_Capture/apf_calc/target/release/apf_calc", binary="apf_calc")
    # batch/rebuild offline runners
    curated(f"{Q}/batch_offline_step_metrics_zst.py", f"{Q}/offline_step_metrics.py", "invokes", "str(offline_script),", "interpolated",
            via='argparse default Path(__file__).with_name("offline_step_metrics.py")')
    curated(f"{Q}/rebuild_matrices_and_rerun_offline.py", f"{Q}/offline_step_metrics.py", "invokes", "str(offline_script),", "interpolated",
            via='argparse default Path(__file__).with_name("offline_step_metrics.py")')
    # VM_executables/run_files.sh -> its workloads (variables)
    for v, s in (("MEM_A1", "mem_stream.py"), ("MEM_A2", "mem_pointer_chase.py"), ("MEM_A3", "mem_alloc_touch_pages.py"),
                 ("IO_B1", "io_seq_fsync.py"), ("IO_B2", "io_rand_rw.py"), ("IO_B3", "io_many_files.py")):
        curated("VM_executables/run_files.sh", f"VM_executables/{s}", "invokes", f'{v}="{s}"', "interpolated", note="run via $PYTHON_BIN later in the script")
    # plan06 comprehensive report reads the collapsed cells_full tree
    curated(f"{Q}/plan06_campaign/make_comprehensive_report.py", f"{Q}/plan06_campaign/downstream/cells_full/manifest.csv", "reads",
            'CELLS = HERE / "downstream" / "cells_full"', "interpolated", note="cells_full/ root; per-cell trajectories under work/<cell>/")
    # steps files are what run_files_controlled.py executes (STEPS_FILE env)
    curated(RFC, f"{A}plan07_campaign/runs/<label>_steps.txt", "reads", 'STEPS_FILE = os.environ.get("STEPS_FILE", "")', "interpolated",
            alternative=True, note="console launch path: subset_run.py writes runs/<label>_steps.txt and the launch line sets STEPS_FILE")
    curated(RFC, f"{P7}/database_steps.txt", "reads", 'STEPS_FILE = os.environ.get("STEPS_FILE", "")', "interpolated",
            alternative=True, note="campaign launch path (STEPS_FILE=plan07_campaign/database_steps.txt per plan07 docs); env-selected")
    G.blind["comment_only_mentions"].append({"file": RFC, "line": find_line(RFC, "plan07_campaign/repfill_steps.txt")[0],
                                            "text": "plan07_campaign/repfill_steps.txt named in a comment as a STEPS_FILE example; not drawn as an edge"})


# --------------------------------------------------------------------------------------------
# run all extractors
# --------------------------------------------------------------------------------------------
def extract_all():
    extract_rust()
    extract_c_and_make()
    extract_steps()
    for rel in FILE_NODES:
        size = (REPO / rel).stat().st_size
        ext = os.path.splitext(rel)[1]
        if size > MAX_SCAN_BYTES and ext not in (".txt", ".csv"):
            G.blind["skipped_over_100kb"].append({"file": rel, "bytes": size})
            G.mark_incomplete(rel, f"not scanned: {size} bytes > 100 KB")
            continue
        if ext == ".py":
            extract_python_imports(rel)
            scan_python_invocations(rel)
            scan_artifacts(rel)
        elif ext == ".sh":
            scan_shell_invocations(rel)
            scan_artifacts(rel)
        elif ext == ".ps1":
            scan_ps1(rel)
            scan_artifacts(rel)
        elif ext in (".rs", ".c", ".h") or rel.endswith("Makefile"):
            scan_artifacts(rel)
        elif ext == ".html" and (".template." in rel or ".mockup." in rel):
            scan_html(rel)
    curated_edges()


# --------------------------------------------------------------------------------------------
# entry markers, kinds
# --------------------------------------------------------------------------------------------
DOC_CMD_RX = re.compile(r"(?:python3?|bash|sh|\./|screen\s+-\S+\s+\S+\s+)\s*[\w./~$-]*?([A-Za-z0-9_]+\.(?:py|sh))\b")
DOC_FILES = [f for f in FILE_NODES if f.endswith((".md", ".txt")) or os.path.basename(f).startswith("README")]
CMD_NAMED: dict[str, list[str]] = defaultdict(list)
for _d in DOC_FILES:
    for _m in DOC_CMD_RX.finditer(read(_d)):
        CMD_NAMED[_m.group(1)].append(_d)
CORPUS = {f: read(f) for f in FILE_NODES}


def entry_marker(rel: str) -> str | None:
    if rel in EXEC_BITS or os.access(REPO / rel, os.X_OK):
        marker = "exec-bit"
    else:
        marker = None
    text = CORPUS.get(rel, "")
    if text.startswith("#!"):
        marker = marker or "shebang"
    if rel.endswith(".py") and '__name__ == "__main__"' in text or "__name__ == '__main__'" in text:
        marker = marker or "__main__"
    base = os.path.basename(rel)
    if base in CMD_NAMED and any(d != rel for d in CMD_NAMED[base]):
        marker = marker or f"named-as-command-in:{CMD_NAMED[base][0]}"
    return marker


def basename_named_elsewhere(rel: str) -> str | None:
    base = os.path.basename(rel)
    if len(base) < 6:
        return None
    for f, t in CORPUS.items():
        if f != rel and base in t:
            return f
    return None


def assign_kinds():
    incoming_writes = {e["dst"] for e in G.edges if e["type"] == "writes"}
    for p, n in G.nodes.items():
        if p.startswith(A) or n.get("collapsed"):
            n["kind"] = "artifact"
            continue
        if p in TRUE_ROOTS:
            n["kind"] = "entry"
            continue
        ext = os.path.splitext(p)[1]
        base = os.path.basename(p)
        if n["kind"] in ("template", "generated", "config") and n["kind"] != "module":
            if n["kind"] == "config" and p in incoming_writes:
                n["kind"] = "generated"
            continue
        if ".template." in p or ".mockup." in p:
            n["kind"] = "template"
        elif p in incoming_writes and ext in (".html", ".csv", ".txt", ".json", ".md", ".pdf"):
            n["kind"] = "generated"
        elif ext in (".json", ".toml", ".txt", ".csv", ".example", ".gitignore", ".python-version") or base in ("Makefile", "requirements.txt", ".gitignore"):
            n["kind"] = "config" if ext != ".csv" or "manifest" in base or "steps" in base else "artifact"
        elif ext in (".md", ".tex", ".html"):
            n["kind"] = "doc"
        elif ext in (".py", ".sh", ".ps1"):
            m = entry_marker(p) or ""
            if m in ("exec-bit", "shebang", "__main__"):
                n["kind"] = "entry"
            else:
                n["kind"] = "module" if ext == ".py" else "script"
        elif ext in (".rs", ".c", ".h"):
            n["kind"] = "module"
        else:
            n["kind"] = "module"


# --------------------------------------------------------------------------------------------
# SCC condensation, roots, layers, classes
# --------------------------------------------------------------------------------------------
def tarjan(nodes: list[str], adj: dict[str, set[str]]) -> list[list[str]]:
    index = {}
    low = {}
    on = set()
    stack = []
    sccs = []
    counter = [0]
    sys.setrecursionlimit(max(10000, len(nodes) * 4))

    def strong(v):
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on.add(v)
        for w in adj.get(v, ()):
            if w not in index:
                strong(w)
                low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on.discard(w)
                comp.append(w)
                if w == v:
                    break
            sccs.append(sorted(comp))

    for v in nodes:
        if v not in index:
            strong(v)
    return sccs


def analyse():
    nodes = sorted(G.nodes)
    adj: dict[str, set[str]] = defaultdict(set)
    radj: dict[str, set[str]] = defaultdict(set)
    for e in G.edges:
        adj[e["src"]].add(e["dst"])
        radj[e["dst"]].add(e["src"])
    sccs = tarjan(nodes, adj)
    scc_of: dict[str, int] = {}
    for i, comp in enumerate(sccs):
        for v in comp:
            scc_of[v] = i
    cadj: dict[int, set[int]] = defaultdict(set)
    cradj: dict[int, set[int]] = defaultdict(set)
    for e in G.edges:
        a, b = scc_of[e["src"]], scc_of[e["dst"]]
        if a != b:
            cadj[a].add(b)
            cradj[b].add(a)
    # reachability from true roots
    rank0: set[int] = set()
    stack = [scc_of[r] for r in TRUE_ROOTS if r in scc_of]
    while stack:
        c = stack.pop()
        if c in rank0:
            continue
        rank0.add(c)
        stack.extend(cadj[c])
    root_sccs = {scc_of[r] for r in TRUE_ROOTS if r in scc_of}
    # secondary roots: condensation in-degree 0, not reachable from a true root
    secondary = [c for c in range(len(sccs)) if not cradj[c] and c not in rank0]
    # layers via topological order (Kahn)
    indeg = {c: len(cradj[c]) for c in range(len(sccs))}
    layer: dict[int, int] = {}
    order = []
    ready = [c for c in range(len(sccs)) if indeg[c] == 0]
    while ready:
        c = ready.pop()
        order.append(c)
        for d in cadj[c]:
            indeg[d] -= 1
            if indeg[d] == 0:
                ready.append(d)
    for c in order:
        if c in root_sccs:
            layer[c] = 0
        elif c in secondary:
            layer[c] = 1
        else:
            preds = [layer[p] for p in cradj[c] if p in layer]
            layer[c] = 1 + max(preds) if preds else (0 if c in rank0 else 1)
    # back edges into roots (a root with predecessors): report, never break
    back_edges = []
    for e in G.edges:
        if e["dst"] in TRUE_ROOTS and scc_of[e["src"]] != scc_of[e["dst"]]:
            back_edges.append(e)
    # dominated subtree size for secondary roots
    def reach(c: int) -> set[int]:
        seen = set()
        st = [c]
        while st:
            x = st.pop()
            if x in seen:
                continue
            seen.add(x)
            st.extend(cadj[x])
        return seen

    sec_info = []
    for c in secondary:
        r = reach(c) - rank0 - {c}
        size = sum(len(sccs[x]) for x in r)
        sec_info.append((size, sccs[c][0], c))
    sec_info.sort(key=lambda t: (-t[0], t[1]))
    # per node fields
    write_in: dict[str, int] = defaultdict(int)
    read_in: dict[str, int] = defaultdict(int)
    read_in_nontest: dict[str, int] = defaultdict(int)
    indeg_all: dict[str, int] = defaultdict(int)
    for e in G.edges:
        indeg_all[e["dst"]] += 1
        if e["type"] == "writes":
            write_in[e["dst"]] += 1
        if e["type"] == "reads":
            read_in[e["dst"]] += 1
            if e.get("role") != "test":
                read_in_nontest[e["dst"]] += 1
    unresolved_total = sum(1 for e in G.edges if e["binding"] == "unresolved")
    for p, n in G.nodes.items():
        c = scc_of[p]
        n["layer"] = layer.get(c, 1)
        n["rank"] = 0 if c in rank0 else 1
        if len(sccs[c]) > 1:
            n["scc_id"] = c
            n["cycle"] = True
        n["in_degree"] = indeg_all[p]
        n["write_in_degree"] = write_in[p]
        n["read_in_degree"] = read_in[p]
        n["read_in_degree_nontest"] = read_in_nontest[p]
        n["edges_incomplete"] = p in G.incomplete
        if p in G.notes:
            n["incomplete_reasons"] = G.notes[p]
        marker = entry_marker(p) if not p.startswith(A) else None
        if marker:
            n["entry_marker"] = marker
        # class
        if p in TRUE_ROOTS:
            n["class"] = "root"
        elif write_in[p] > 0 and read_in_nontest[p] == 0:
            n["class"] = "unconsumed-artifact"
        elif indeg_all[p] == 0:
            if marker:
                n["class"] = "orphaned-but-live"
            else:
                named = basename_named_elsewhere(p) if not p.startswith(A) else None
                if named:
                    n["class"] = "named-but-unlinked"
                    n["named_in"] = named
                else:
                    n["class"] = "unreachable"
                    # 'dead' is assertable only with zero unresolved bindings anywhere
                    if unresolved_total == 0:
                        n["class"] = "dead"
        else:
            n["class"] = "reachable"
    return {
        "sccs": [comp for comp in sccs if len(comp) > 1],
        "scc_index": {i: comp for i, comp in enumerate(sccs) if len(comp) > 1},
        "secondary_roots": [{"path": p, "dominated_subtree_size": s, "scc_id": c if len(sccs[c]) > 1 else None} for s, p, c in sec_info],
        "back_edges_into_roots": back_edges,
        "unresolved_total": unresolved_total,
    }


# --------------------------------------------------------------------------------------------
# expected-edge check (violations)
# --------------------------------------------------------------------------------------------
FACTS = ("step_index", "family", "workload", "rep", "scale", "command")
DERIVATION_HINTS = {
    f"{P8}/b1_features.py": "clean_workload() derives the workload by regex on the source path basename (b1_features.py:70-82); family_of() maps a name prefix",
    f"{P8}/b1_windows.py": "imports clean_workload/family_of/RUN_RE from b1_features.py; labels come from provenance.json source paths",
    f"{P8}/b1_splits.py": "reads workload/family labels stored in b1_windows.npz",
    f"{P8}/b1_ae.py": "reads labels via b1_splits.load_labels(b1_windows.npz)",
    f"{P10}/corpus_manifest.py": "family/workload/rep parsed from the zstd_local directory layout <family>/<workload>/<variant>/rep<NNN> (corpus_manifest.py:44-49, 84-87, 157-159)",
    f"{P10}/runner/executor.py": "workload/family copied from the corpus manifest built by corpus_manifest.py (executor.py:363-369)",
    f"{P7}/subset_run.py": "cells derived from full_campaign_steps.txt plus generate_database_steps.py functions; writes its own per-run record runs/<label>.json",
    f"{P7}/ui/console_bridge.py": "relays cells/launch_line from subset_run.py's runs/<label>.json",
    f"{P7}/ui/build_console.py": "workload catalogue derived from full_campaign_steps.txt and generate_database_steps.py constants",
    f"{P7}/separability_matrix.py": "plan05-lineage input (plan05_campaign/downstream/sweep.csv) with family_of from behavior_families.py",
    f"{P7}/make_execution_log.py": "reads pilot_baseline.json and hard-coded campaign facts",
}


def violations() -> list[dict]:
    out = []
    reads_manifest = {e["src"] for e in G.edges if e["dst"] == AUTHORITY and e["type"] == "reads"}
    for rel in FILE_NODES:
        if not rel.endswith(".py"):
            continue
        if not (rel.startswith(P7) or rel.startswith(P8) or rel.startswith(P10)):
            continue
        if "/tests/" in "/" + rel or rel.endswith("test_subset_run.py"):
            continue
        if rel == f"{P7}/generate_database_steps.py":
            continue  # writes the authority
        code = "\n".join(code_lines(rel))
        emitted = [f for f in FACTS if re.search(r"[\"']" + f + r"[\"']", code)]
        if "family" not in emitted or "workload" not in emitted:
            continue
        produces = any(e["src"] == rel and e["type"] == "writes" for e in G.edges) or "json.dumps" in code or "print(" in code
        if not produces:
            continue
        if rel in reads_manifest:
            continue
        lines = [i for i, l in enumerate(code_lines(rel), 1) if re.search(r"[\"'](family|workload)[\"']", l)]
        out.append({
            "node": rel,
            "facts_emitted": emitted,
            "expected_edge": {"src": rel, "dst": AUTHORITY, "type": "reads"},
            "observed": "no reads edge to the authority",
            "derives_from": DERIVATION_HINTS.get(rel, "see fact_literal_lines"),
            "fact_literal_lines": lines[:12],
        })
    # the authority itself: who reads it at all?
    readers = sorted(reads_manifest)
    out.insert(0, {
        "node": AUTHORITY,
        "observed": f"read in-degree {len(readers)}; readers: {readers}",
        "expected": "every plan07+ node that emits step_index/family/workload/rep/scale/command reads this file",
        "note": "generate_database_steps.py writes it (writer, exempt)",
    })
    return out


# --------------------------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------------------------
def main() -> int:
    extract_all()
    assign_kinds()
    meta = analyse()
    viol = violations()

    nodes_out = []
    for p in sorted(G.nodes):
        n = G.nodes[p]
        row = {"path": p, "kind": n["kind"], "class": n["class"], "layer": n["layer"], "rank": n["rank"],
               "edges_incomplete": n["edges_incomplete"]}
        for k in ("scc_id", "cycle", "entry_marker", "in_degree", "write_in_degree", "read_in_degree", "read_in_degree_nontest",
                  "instances", "collapsed", "desc", "incomplete_reasons", "named_in", "binary_instance_excluded"):
            if k in n:
                row[k] = n[k]
        nodes_out.append(row)
    edges_out = sorted(G.edges, key=lambda e: (e["src"], e["type"], e["dst"], e["evidence"]["line"]))

    by_class = defaultdict(int)
    by_kind = defaultdict(int)
    for n in nodes_out:
        by_class[n["class"]] += 1
        by_kind[n["kind"]] += 1
    by_type = defaultdict(int)
    by_binding = defaultdict(int)
    for e in edges_out:
        by_type[e["type"]] += 1
        by_binding[e["binding"]] += 1

    # acceptance
    HC = f"{A}hc_field.csv.zst"
    a_edge = [e for e in edges_out if e["src"] == f"{P8}/b1_extract_hamming.py" and e["dst"] == HC and e["type"] == "writes"]
    hc = G.nodes.get(HC, {})
    hc_readers = [e for e in edges_out if e["dst"] == HC and e["type"] == "reads"]
    acc_a = bool(a_edge) and hc.get("read_in_degree_nontest", 1) == 0 and hc.get("class") == "unconsumed-artifact"
    bf = f"{P8}/b1_features.py"
    bf_reads = [e for e in edges_out if e["src"] == bf and e["dst"] == AUTHORITY and e["type"] == "reads"]
    bf_viol = [v for v in viol if v.get("node") == bf]
    acc_b = (not bf_reads) and bool(bf_viol) and "workload" in (bf_viol[0]["facts_emitted"] if bf_viol else []) \
        and "family" in (bf_viol[0]["facts_emitted"] if bf_viol else []) and "rep" in (bf_viol[0]["facts_emitted"] if bf_viol else [])
    acceptance = {
        "A": {"pass": acc_a, "writes_edge": a_edge[:1], "hc_field_read_in_degree_all": hc.get("read_in_degree"),
              "hc_field_read_in_degree_nontest": hc.get("read_in_degree_nontest"), "hc_field_class": hc.get("class"),
              "readers": hc_readers},
        "B": {"pass": acc_b, "b1_features_reads_manifest": bf_reads, "violation": bf_viol[:1]},
        "extractor_is_import_only": not (acc_a and acc_b),
    }

    blind_spots = {
        "construction": [
            "The graph is laptop-local by construction: console.sh builds and runs console_bridge.py on the research server over ssh, "
            "analysis_bridge.py runs its executor against a remote_root over ssh via sources.SshSource, and every guest workload runs "
            "inside the VM over ssh. The server and guest halves are not on this machine; their edges are drawn from this tree's copies "
            "of the scripts and marked remote where known.",
            "Artifacts whose instances live outside the repository (queue dir, output dir, zstd_local, runs/) are nodes named "
            "'artifact:<class>'; run-instance trees inside the repo are collapsed to one class node each (cells_full/work/<cell>/...).",
            "For a node with edges_incomplete=true, 'no edge to B' reads as UNKNOWN, never as NO.",
        ],
        "declared": [
            {"id": 1, "topic": "module-level path constants feeding subprocess",
             "resolution": "console_bridge.py:53 SUBSET_RUN, :54 PREFLIGHT, :382 CLEANUP_SCRIPT and analysis_bridge.py:53 BUILD, :54 EXECUTOR are folded by a "
                           "small ast evaluator (HERE / 'x', Path(__file__).parent, os.path.join, os.environ.get default) and matched to the "
                           "argv of subprocess.run/Popen/_run; edges carry via='module constant ...'. Constants built from os.environ.get are "
                           "bound 'interpolated' because the default, not the override, is the edge."},
            {"id": 2, "topic": "shell env-var indirection",
             "resolution": "capture_producer_qemu_pmemsave.sh:21 CONFIG=\"${CONFIG:-$ROOT/config_qemu.json}\" and the PRODUCER_SCRIPT/CONSUMER_SCRIPT "
                           "defaults in run_qemu_capture.sh are edges to the DEFAULT target (binding interpolated). Caller overrides are not statically "
                           "knowable in general; the one visible override chain (run_files_controlled.py:64/:557 sets CONFIG=config_qemu_upc.json and "
                           "PRODUCER_SCRIPT=capture_producer_qemu_pmemsave.sh) is recorded as run_files_controlled.py's own edges, not re-attributed to the producer."},
            {"id": 3, "topic": "glob-mediated reads (ls -t | head -1)",
             "resolution": "capture_consumer_qemu.sh:347 (apf/*.txt), :373 (metrics/page_metrics-*.csv), :403 (cosine|hamming/*.txt) are reads of an artifact "
                           "CLASS node, marked class_valued=true, binding interpolated. Which instance is consumed is decided at runtime by mtime."},
            {"id": 4, "topic": "runtime alternatives",
             "resolution": "console_bridge.py:65 PRODUCER_SCRIPTS is consumed only by pkill -f (stop) and pgrep -f (health): process-name coupling, not a launch "
                           "site. Both producers are drawn from the bridge as invokes with binding unresolved and alternative=true so the tuple is visible; "
                           "the real launch chain is console_bridge -> (screen) run_files_controlled.py -> run_qemu_capture.sh -> producer, where the producer "
                           "is an env-selected alternative (pmemsave default; capture_producer_qemu.sh and _user_raw.sh drawn alternative/unresolved)."},
            {"id": 5, "topic": "remote execution",
             "resolution": "analysis_bridge.py delegates to sources.SshSource (ssh listing, rsync fetch) and console.sh runs its build+bridge command on the "
                           "server. The server-side process tree is not observable here. Sites are listed under remote_execution_sites; edges are drawn "
                           "from this tree's copy of each script."},
            {"id": 6, "topic": "compiled-binary hop",
             "resolution": "Binaries are resolved to crates through Cargo.toml: apf_calc, live_delta_calc and live_delta_calc_modular have no [[bin]] section, "
                           "so the package name is the bin and src/main.rs its root; matrix_builder declares [[bin]] name=matrix_builder path=src/main.rs. "
                           "capture_consumer_qemu.sh reaches the binaries only through config values (rustDeltaCalculationProgram, apfCalculationProgram, "
                           "substrateProgram in config_qemu_upc.json); differ.py through DEFAULT_BINARY. matrix_builder is invoked by nothing in this tree "
                           "(the consumer's raw builder default is the Python coherence_temp_spec_stability.raw_matrix_builder)."},
            {"id": 7, "topic": "browser-to-bridge HTTP endpoints",
             "resolution": "capture_console.template.html and analysis_console.template.html couple to console_bridge.py / analysis_bridge.py routes over "
                           "fetch(...) URLs. That is not a file edge and is declared MISSING (not absent): see browser_to_bridge_http for the call sites "
                           "and the bridges' ROUTES_GET/ROUTES_POST tables."},
        ],
        "resolved_forms_by_language": {
            "python": ["import X / from X import Y (ast, any nesting; nested imports bound interpolated)",
                       "sys.path roots assumed: own dir, repo root, VM_Capture_QEMU, plan05/06/07/08/10 dirs, coherence_temp_spec_stability, VM_executables_phase2, utils",
                       "subprocess.run/Popen/check_output/check_call/call/_run/system with list argv, str(CONST), module constants, argv/cmd_parts lists built by =, +=, append, extend",
                       "artifact names as string literals on code lines (comments and docstrings blanked with tokenize/ast), classified by verbs on the same line "
                       "(write_text/json.dump/to_csv/np.save/savez/open(...,'w') vs read_text/json.load/np.load/DictReader/open()), variable chasing one level",
                       "NOT resolved: importlib/__import__ by string, getattr dispatch, paths assembled across functions, f-strings whose fields are not label/cfg['label']"],
            "shell": ["source X / . X", "bash|sh|python3|python|$PYTHON|$PYTHON_BIN <path>.sh|.py, ./x.sh, python -m module",
                      "VAR=\"${VAR:-default}\" and VAR=\"$DIR/x\" defaults chased one level for \"$VAR\" invocations",
                      "redirections > >> tee, -o/--out/--output/--json flags as writes; jq/cat/grep/awk/head/tail/sed/wc/< as reads; find/ls/rm/test -f/du as metadata-only",
                      "NOT resolved: eval'd strings beyond the curated builderProgram, xterm -e strings, pkill/pgrep name coupling (declared, not drawn)"],
            "rust": ["mod x; -> x.rs | x/mod.rs (2018 paths)", "use <package>:: -> src/lib.rs", "Cargo.toml -> bin/lib targets; examples/tests/benches auto-discovery (interpolated)",
                     "File::open/create literals on the same line only (curated for the three binaries' output dirs)"],
            "c_make": ["#include \"x\" relative to the file", "Makefile listed .c sources (literal) and $(wildcard ...) (interpolated); bin/<stem> <- <stem>.c"],
            "html": ["<script src> local (none found; only Google Fonts links, listed under external_resources)", "fetch( call sites declared as blind spot 7"],
            "powershell": ["*.ps1 parameter defaults and config.json Get-Content"],
            "steps_txt": ["/bin/<stem> -> VM_executables_phase2/**/<stem>.c via Makefile bin rule; VM_executables/<name>.py|.sh; edges are invokes, remote=guest"],
        },
        "test_role_edges": "edges from tests/ carry role='test'; read_in_degree_nontest excludes them and is what the artifact class uses. Both counts are on every node.",
        "sites": {k: v for k, v in G.blind.items()},
    }

    result = {
        "meta": {
            "repo": str(REPO), "generated_by": "docs/dependency_graph/extract_dependency_graph.py",
            "true_roots": TRUE_ROOTS, "authority": AUTHORITY,
            "excluded": [".claude/worktrees/**", "**/target/**", "**/__pycache__/**", "utils/venv/**", ".git/**", "data/**", "*.lock",
                         "file_structure.txt", "*.pyc *.png *.pdf *.zip *.npy *.zst (binary instances; schemas keep nodes)"],
            "collapsed_instance_trees": collapsed_count,
            "class_vocabulary_note": "unconsumed-artifact | orphaned-but-live | unreachable | dead (never asserted while unresolved edges exist) "
                                     "plus three residual classes the four do not cover: root (the two declared roots), reachable (in-degree > 0) and "
                                     "named-but-unlinked (in-degree 0, no entry marker, basename appears as text in another file).",
            "layer_note": "layer = longest path from the dominating root on the SCC condensation; true roots 0, secondary roots 1. "
                          "Edges INTO a true root from a deeper node are reported under back_edges_into_roots and are not broken.",
        },
        "summary": {
            "nodes": len(nodes_out), "edges": len(edges_out),
            "by_class": dict(sorted(by_class.items())), "by_kind": dict(sorted(by_kind.items())),
            "edges_by_type": dict(sorted(by_type.items())), "edges_by_binding": dict(sorted(by_binding.items())),
            "unresolved_edges": meta["unresolved_total"],
            "cycles": len(meta["sccs"]),
            "nodes_edges_incomplete": sum(1 for n in nodes_out if n["edges_incomplete"]),
        },
        "acceptance": acceptance,
        "roots": {"true": TRUE_ROOTS, "secondary": meta["secondary_roots"]},
        "cycles": [{"scc_id": i, "members": comp} for i, comp in meta["scc_index"].items()],
        "back_edges_into_roots": meta["back_edges_into_roots"],
        "nodes": nodes_out,
        "edges": edges_out,
        "violations": viol,
        "blind_spots": blind_spots,
    }
    (OUT_DIR / "dependency_graph.json").write_text(json.dumps(result, indent=1) + "\n")
    (OUT_DIR / "DEPENDENCY_GRAPH.md").write_text(render_markdown(result))
    print(json.dumps({"summary": result["summary"], "acceptance": {"A": acc_a, "B": acc_b}}, indent=1))
    return 0


def render_markdown(r: dict) -> str:
    N = {n["path"]: n for n in r["nodes"]}
    E = r["edges"]
    s = r["summary"]
    acc = r["acceptance"]

    def short(p: str) -> str:
        return p.replace("VM_sampler/VM_Capture_QEMU/", "").replace("VM_sampler/VM_Capture/", "VM_Capture/")

    def writers(a):
        return sorted({short(e["src"]) for e in E if e["dst"] == a and e["type"] == "writes"})

    def readers(a):
        return sorted({short(e["src"]) + (" (test)" if e.get("role") == "test" else "") for e in E if e["dst"] == a and e["type"] == "reads"})

    L = []
    L.append("# Dependency graph: file-level, read-only\n")
    L.append("Generated by `docs/dependency_graph/extract_dependency_graph.py`; the full graph is `dependency_graph.json` next to this file. "
             "Nothing was deleted or moved and no deletion is proposed.\n")
    L.append("## Acceptance\n")
    L.append(f"- **A** ({'PASS' if acc['A']['pass'] else 'FAIL'}): `plan08_b1/b1_extract_hamming.py --writes--> artifact:hc_field.csv.zst` exists "
             f"(evidence {short(acc['A']['writes_edge'][0]['evidence']['file'])}:{acc['A']['writes_edge'][0]['evidence']['line']}); "
             f"read in-degree {acc['A']['hc_field_read_in_degree_all']} (tests included); class `{acc['A']['hc_field_class']}`. "
             "The only touch besides the writer is tests/test_plan08_b1_extract.py handing the path to `zstd -dc` as argv (recorded under argv_pass_through, not a reads edge).")
    v = acc["B"]["violation"][0] if acc["B"]["violation"] else {}
    L.append(f"- **B** ({'PASS' if acc['B']['pass'] else 'FAIL'}): `plan08_b1/b1_features.py` emits {v.get('facts_emitted')} and has no reads edge to "
             f"`plan07_campaign/database_steps_manifest.csv`; listed in violations[] with derivation: {v.get('derives_from')}.")
    t = s["edges_by_type"]
    L.append(f"- The extractor is not import-only: {t.get('invokes', 0)} invokes / {t.get('reads', 0)} reads / {t.get('writes', 0)} writes edges accompany the {t.get('imports', 0)} imports.\n")
    L.append("## Counts\n")
    L.append(f"- nodes {s['nodes']}, edges {s['edges']}; edges by type {s['edges_by_type']}; by binding {s['edges_by_binding']}")
    L.append(f"- nodes by class {s['by_class']}")
    L.append(f"- nodes by kind {s['by_kind']}")
    L.append(f"- cycles {s['cycles']}; nodes edges-incomplete {s['nodes_edges_incomplete']}; unresolved edges {s['unresolved_edges']} "
             "(therefore no node is asserted `dead`).\n")
    L.append("## Roots and layers\n")
    L.append("True roots (layer 0, rank 0): `plan07_campaign/ui/console.sh`, `plan10_analysis/ui/analysis.sh`. "
             "Layer = longest path from the dominating root on the SCC condensation. Selected nodes:\n")
    for k in ["plan07_campaign/ui/console_bridge.py", "plan07_campaign/ui/build_console.py", "run_files_controlled.py", "run_qemu_capture.sh",
              "capture_producer_qemu_pmemsave.sh", "capture_consumer_qemu.sh", "plan07_campaign/subset_run.py", "plan07_campaign/generate_database_steps.py",
              "plan07_campaign/database_steps_manifest.csv", "plan10_analysis/ui/analysis_bridge.py", "plan10_analysis/runner/executor.py",
              "plan10_analysis/runner/differ.py", "plan08_b1/b1_extract_hamming.py", "plan08_b1/b1_features.py"]:
        n = N.get(f"VM_sampler/VM_Capture_QEMU/{k}")
        if n:
            L.append(f"- L{n['layer']} r{n['rank']} `{k}` ({n['class']}, in-degree {n['in_degree']})")
    for k in ["VM_sampler/VM_Capture/live_delta_calc_modular/src/main.rs", "VM_sampler/VM_Capture/apf_calc/src/main.rs",
              "VM_sampler/VM_Capture/live_delta_calc/src/main.rs", "VM_sampler/VM_Capture/matrix_builder/src/main.rs",
              "VM_executables_phase2/kernel/D1_visible_dense_linear_algebra/kernel_gemm_v2.c", "coherence_temp_spec_stability/streaming_metrics.py"]:
        n = N.get(k)
        if n:
            L.append(f"- L{n['layer']} r{n['rank']} `{k}` ({n['class']}, in-degree {n['in_degree']})")
    L.append("")
    cyc = r["cycles"]
    L.append("Cycles (contracted, banded): " + ("; ".join("{" + ", ".join(short(m) for m in c["members"]) + "}" for c in cyc) if cyc else "none") + ".")
    L.append("Back-edges into a true root: " + (", ".join(f"{short(e['src'])} -> {short(e['dst'])}" for e in r["back_edges_into_roots"]) or "none") + ".\n")
    L.append(f"Secondary roots (in-degree 0, claims of orphanhood to adjudicate): {len(r['roots']['secondary'])}. Largest dominated subtrees:\n")
    for sr in r["roots"]["secondary"][:12]:
        L.append(f"- {sr['dominated_subtree_size']:3d} `{short(sr['path'])}`")
    L.append("\nMost of the secondary roots are pytest files (discovered by pytest, not invoked by any file), one-off make_* report generators, "
             "and the 160 docs. `VM_executables_phase2/Makefile` and `VM_Capture/matrix_builder/Cargo.toml` are genuine in-degree-0 build roots: "
             "nothing in the tree runs `make` or the matrix_builder binary (the consumer's raw builder default is the Python raw_matrix_builder).\n")
    L.append("## The capture chain, as coupled through files\n")
    L.append("console.sh -> (ssh) build_console.py + console_bridge.py -> subset_run.py (writes runs/<label>_steps.txt, runs/<label>.json) -> "
             "console_bridge execs the launch line under screen -> run_files_controlled.py -> run_qemu_capture.sh -> capture_producer_qemu_pmemsave.sh "
             "(writes memory_dump raw via pmemsave, queue job json, capture_status.json, vm_state.txt) and capture_consumer_qemu.sh (reads job json, "
             "invokes apf_calc / live_delta_calc / live_delta_calc_modular by config-valued binary path, reads their per-pair outputs by `ls -t | head -1`, "
             "writes substrate_trajectory.csv, apf_trajectory.jsonl, run_matrix npy, the zstd chain). Control flows back through capture_control.json "
             "(bridge and orchestrator write, producer and orchestrator read) and capture_status.json (producer writes, bridge reads).\n")
    L.append("## Unconsumed artifacts (written, read in-degree 0 outside tests)\n")
    for a in sorted(p for p, n in N.items() if n["class"] == "unconsumed-artifact"):
        L.append(f"- `{short(a)}` W={writers(a)} R={readers(a)}")
    L.append("\nNotable: `plan07_campaign/database_steps_manifest.csv` (the authority for step_index/family/workload/rep/scale/command) is written by "
             "generate_database_steps.py and read by nothing. `hc_field.csv.zst`, `features_long.csv`, `b1_windows_meta.csv`, `b1_ae_results.json` end the B1 chain. "
             "`capture_console.html` and `analysis_console.html` are the published static builds (no reader by design).\n")
    L.append("## Expected-edge violations\n")
    for v in r["violations"]:
        if "facts_emitted" in v:
            L.append(f"- `{short(v['node'])}` emits {v['facts_emitted']}; no reads edge to the manifest; derives from: {v['derives_from']}")
        else:
            L.append(f"- authority `{short(v['node'])}`: {v['observed']}")
    L.append("")
    L.append("## Blind spots (declared)\n")
    for b in r["blind_spots"]["declared"]:
        L.append(f"- **{b['id']}. {b['topic']}** -- {b['resolution']}")
    L.append("")
    sites = r["blind_spots"]["sites"]
    L.append("Site counts: " + ", ".join(f"{k} {len(v)}" for k, v in sorted(sites.items())) + ". "
             "Files over 100 KB were not scanned (listed under skipped_over_100kb; all are generated HTML reports or large docs). "
             "For any node marked edges_incomplete, 'no edge' means unknown, not no.\n")
    L.append("Resolved reference forms per language are listed in `blind_spots.resolved_forms_by_language` in the JSON.\n")
    L.append("## Deviations from the brief\n")
    L.append("- Two extra classes were needed because the four given ones do not cover reachable nodes: `reachable` (in-degree > 0, not an unconsumed artifact) "
             "and `named-but-unlinked` (in-degree 0, no entry marker, but the basename appears as text in another file). The two roots carry class `root`.")
    L.append("- plan05_campaign/downstream/*.json were kept as individual nodes (eleven named result files, each read by name), not collapsed; "
             "cells_full/work/<cell>/ and timing_runs/exp2c_* were collapsed.")
    L.append("- Edges are deduplicated per (src, dst, type); additional evidence lines are kept in `also_evidence`. Test-file edges carry role=test and are "
             "excluded from the read in-degree that decides `unconsumed-artifact` (both counts are stored on every node).")
    L.append("- Generated PDFs (excluded as binary instances) keep a node flagged binary_instance_excluded so their writers have a target.\n")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
