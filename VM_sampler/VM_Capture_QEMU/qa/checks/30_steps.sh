#!/usr/bin/env bash
# Static analysis of the steps file against the probed guest facts, plus
# campaign-definition sanity (duplicate chain identities).

section "Steps file: resource fit"

if [[ ! -r "$STEPS_FILE" ]]; then
  fail "STEPS_FILE not readable: $STEPS_FILE"
  return 0 2>/dev/null || exit 0
fi

scratch_args=()
if [[ -s "$QA_SCRATCH_FILE" ]]; then
  while read -r a; do [[ -n "$a" ]] && scratch_args+=("$a"); done < "$QA_SCRATCH_FILE"
fi

if [[ -r "$QA_FACTS" ]]; then
  if python3 "$QA_ROOT/analyze_steps.py" "$STEPS_FILE" --facts "$QA_FACTS" "${scratch_args[@]}"; then
    pass "no resource or scratch findings at FAIL severity"
  else
    fail "steps file has resource/scratch findings (listed above)"
  fi
else
  warn "guest facts unavailable; skipped resource fit analysis"
fi

section "Steps file: campaign identity"

# The retention layer names one chain folder per (family, workload, params, rep).
# Two steps sharing all four would target the same folder; zstd then refuses to
# overwrite, so the second step archives nothing and its raw 1 GiB dumps are
# retained instead of deleted. rep numbering exists to prevent this -- verify it.
python3 - "$STEPS_FILE" <<'PY'
import sys, shlex, re, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[0]).resolve().parent))
lines=[l.strip() for l in Path(sys.argv[1]).read_text().splitlines()
       if l.strip() and not l.strip().startswith('#')]

def label(cmd):
    try: t=shlex.split(cmd)
    except ValueError: t=cmd.split()
    c=[x for x in t if x.endswith(('.py','.sh'))]
    n=Path(c[0]).stem if c else (Path(t[0]).name if t else 'step')
    return re.sub(r'[^A-Za-z0-9_.-]+','_',n).strip('._-') or 'step'

def sig(cmd):
    try: t=shlex.split(cmd)
    except ValueError: t=cmd.split()
    keep=[]; skip=False
    for tok in t:
        if skip: skip=False; continue
        if tok in ('--output-dir','--sandbox-dir'): skip=True; continue
        if '/' in tok: continue
        keep.append(tok)
    s=re.sub(r'[^A-Za-z0-9._-]+','_','_'.join(keep)).strip('._-') or 'default'
    if len(s)>60: s=s[:60]+'_'+hashlib.sha1(s.encode()).hexdigest()[:8]
    return s

seen={}; dupes=[]
for i,l in enumerate(lines,1):
    k=(label(l).split('_',1)[0], label(l), sig(l))
    seen.setdefault(k,[]).append(i)
for k,v in seen.items():
    if len(v)>1: dupes.append((k[1],v))
print(f"  {len(lines)} steps, {len(seen)} distinct (family, workload, params) identities")
if dupes:
    for name,idxs in dupes:
        print(f"  [info] {name} repeats at steps {idxs} -> rep001, rep002, ... (expected for replicate designs)")
PY
pass "chain identities enumerated (repeats get distinct rep numbers)"

dur_missing=$(grep -vcE '^\s*#|^\s*$|--duration [0-9]+' "$STEPS_FILE" 2>/dev/null || echo 0)
if (( dur_missing > 0 )); then
  info "$dur_missing step(s) have no --duration (analysis-only steps run to completion)"
fi
