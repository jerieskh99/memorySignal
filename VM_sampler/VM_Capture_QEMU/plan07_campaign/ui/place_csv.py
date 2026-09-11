"""Place substrate trajectory CSVs next to the chain they belong to.

usage: place_csv.py <steps.txt> <chain_root> <csv_dir> <rep_dirname> <MM-DD>[,<MM-DD>...]|any [--move] [--skip-missing]

test<N> in a CSV filename is the step index; step N of the steps file yields the
retention signature via the same algorithm as run_files_controlled.py
(param_signature_from_command). The workload parsed from the CSV name is
cross-checked against the steps line, so a misaligned index is caught, not moved.
Refuses to act unless every file maps one-to-one onto an existing chain dir.
"""
import sys, os, re, glob, shlex, hashlib, shutil, datetime as dt

FLAGS = ("--output-dir", "--sandbox-dir")

def signature(cmd):
    try: toks = shlex.split(cmd)
    except Exception: toks = cmd.split()
    kept, skip = [], False
    for t in toks:
        if skip: skip = False; continue
        if t in FLAGS: skip = True; continue
        if "/" in t: continue
        kept.append(t)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", "_".join(kept)).strip("._-")
    if not s: return "default"
    return s if len(s) <= 60 else s[:60] + "_" + hashlib.sha1(s.encode()).hexdigest()[:8]

steps_p, root, csvdir, repdir, days = sys.argv[1:6]
do_move = "--move" in sys.argv
skip_missing = "--skip-missing" in sys.argv   # chain not on NFS yet -> skip, not fail
days = None if days == "any" else set(days.split(","))
steps = [l for l in open(steps_p).read().splitlines() if l.strip()]
os.chdir(csvdir)

rows, problems, skipped = [], [], []
for f in sorted(glob.glob("*.substrate_trajectory.csv.zst")):
    d = dt.datetime.fromtimestamp(os.path.getmtime(f))
    if days is not None and d.strftime("%m-%d") not in days: continue
    m = re.match(r"run_matrix_test(\d+)_(.+)\.npy\.substrate_trajectory\.csv\.zst$", f)
    if not m: problems.append((f, "unparseable name")); continue
    n, wl = int(m.group(1)), m.group(2)
    if n > len(steps): problems.append((f, "step %d beyond steps file" % n)); continue
    cmd = steps[n - 1]
    if wl not in cmd:                       # index/workload cross-check
        problems.append((f, "workload %r not in step %d" % (wl, n))); continue
    dest = os.path.join(root, wl.split("_")[0], wl, signature(cmd), repdir)
    if not os.path.isdir(dest):
        (skipped if skip_missing else problems).append((f, "no chain dir yet")); continue
    rows.append((n, f, dest))

seen = {}
for n, f, dest in rows: seen.setdefault(dest, []).append(n)
dups = {k: v for k, v in seen.items() if len(v) > 1}

for n, f, dest in sorted(rows):
    print("test%-3d %-52s -> %s" % (n, f[:52], os.path.relpath(dest, root)))
for f, why in problems:
    print("PROBLEM %-52s : %s" % (f[:52], why))
for f, why in skipped:
    print("skip    %-52s : %s" % (f[:52], why))
print("\nmapped=%d  skipped=%d  problems=%d  distinct=%d  duplicates=%s"
      % (len(rows), len(skipped), len(problems), len(seen), dups or "none"))

if problems or dups:
    print("\nREFUSING: mapping is not clean."); sys.exit(1)
if not do_move:
    print("\nDry run. Add --move to perform the moves."); sys.exit(0)
# The archive keeps its own manifest; a recording that just gained its trajectory is re-registered
# so readers learn it can serve those columns without a re-diff. Registration failing never undoes
# a move (the file is in place either way; `rebuild` reconciles later).
_REGISTER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "plan10_analysis", "archive_manifest.py")
def _register(dest):
    rel = os.path.relpath(dest, root)
    if not os.path.isfile(_REGISTER):
        print("  WARN: no archive_manifest.py at %s; %s not registered" % (_REGISTER, rel)); return
    import subprocess
    r = subprocess.run([sys.executable, _REGISTER, "register", root, rel], capture_output=True, text=True)
    print("  manifest: %s" % ((r.stdout or r.stderr).strip().splitlines() or ["?"])[-1])

moved_dests = []
for n, f, dest in sorted(rows):
    t = os.path.join(dest, f)
    if os.path.exists(t): print("skip (present): %s" % f); continue
    shutil.move(f, t); print("moved: %s" % f); moved_dests.append(dest)
for dest in sorted(set(moved_dests)):
    _register(dest)
print("\ndone: %d file(s)" % len(rows))
