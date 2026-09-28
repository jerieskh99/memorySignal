#!/usr/bin/env python3
"""Paper B, D11 part 1: READ-ONLY facts from the server. Changes nothing.

Collects what cycle 1 of Paper B needs and the code alone cannot answer:
  1. environment      host OS/kernel, CPU, RAM; libvirt and QEMU versions; the guest disk image (path,
                      format, backing chain); guest OS/kernel only if --guest is given and the VM is up
  2. run records      every <label>.json that plan07_campaign/subset_run.py wrote (runs dir): commit,
                      cell count, reps, curated, steps file (exists? lines, lines with --duration, sha256)
  3. replicate index  per run, how many chain folders are rep001, rep002, ...; and how many workload
                      keys have a rep001 in more than one run (the counter restarting per run)
  4. chain check      per run, every chain folder: base 000000.zst present, members contiguous,
                      no empty member (a failed delta write leaves a GAP in the numbering)
  5. consumer.log     counts of "zstd base write failed" / "zstd delta write failed" /
                      "zstd delta did not succeed", with the image timestamps of each hit
  6. failed queue     number of job records in <queueDir>/failed and their time range
  7. libvirt folder   whether any per-domain folder /var/lib/libvirt/qemu/domain-* exists right now

OUTPUT. The screen shows a summary with NO workload names and NO run labels (runs are R1, R2, ... in
creation order), safe to paste back. The full detail, names included, goes to
~/paperB_facts_<UTC timestamp>.txt on the server and stays there.

Run from the repo checkout on the server:
    cd ~/memorySignal/VM_sampler/VM_Capture_QEMU
    python3 paperB_facts_readonly.py
Options:
    --root DIR       VM_Capture_QEMU checkout (default: this file's directory)
    --runs-dir DIR   run records (default: <root>/plan07_campaign/runs)
    --only L1,L2,..  restrict to these run labels (typed on the server; not printed on screen)
    --guest USER@IP  also read the guest's OS/kernel over SSH (only if the VM is already running;
                     this script never starts, stops or suspends anything)
"""
import argparse, glob, hashlib, json, os, re, shlex, subprocess, sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))


def sh(cmd, timeout=60):
    """Run a read-only command; return its stdout (stripped) or an ERROR string. Never raises."""
    try:
        r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout)
        out = (r.stdout or '').strip()
        return out if r.returncode == 0 else f'ERROR rc={r.returncode}: {(r.stderr or out).strip()[:200]}'
    except Exception as e:  # noqa: BLE001
        return f'ERROR: {e}'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def launch_env(launch_line):
    """NAME=value tokens before the command in subset_run.py's launch line (values are shlex-quoted)."""
    env = {}
    try:
        toks = shlex.split(launch_line or '')
    except ValueError:
        toks = (launch_line or '').split()
    for t in toks:
        m = re.fullmatch(r'([A-Z_][A-Z0-9_]*)=(.*)', t)
        if not m:
            break
        env[m.group(1)] = m.group(2)
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=HERE)
    ap.add_argument('--runs-dir', default=None)
    ap.add_argument('--only', default=None)
    ap.add_argument('--guest', default=None)
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    runs_dir = a.runs_dir or os.path.join(root, 'plan07_campaign', 'runs')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    detail_path = os.path.expanduser(f'~/paperB_facts_{stamp}.txt')
    detail = open(detail_path, 'w', encoding='utf-8')

    def say(line=''):          # screen AND detail file (no names ever go through here)
        print(line); detail.write(line + '\n')

    def note(line=''):         # detail file only (may contain names)
        detail.write(line + '\n')

    say(f'# Paper B read-only facts, {stamp}')
    say(f'# detail file (names included, stays on the server): {detail_path}')

    # ---- 1. environment -------------------------------------------------------------------------
    say('\n## 1. Environment')
    say('host kernel:    ' + sh('uname -srvm'))
    say('host OS:        ' + sh("grep '^PRETTY_NAME=' /etc/os-release | cut -d= -f2- | tr -d '\"'"))
    say('host CPU:       ' + sh("lscpu | grep -E '^(Model name|Socket\\(s\\)|Core\\(s\\) per socket|Thread\\(s\\) per core)' | tr -s ' ' | paste -sd ';'"))
    say('host RAM:       ' + sh("grep MemTotal /proc/meminfo"))
    say('virsh version:  ' + sh('virsh -c qemu:///system version | paste -sd ";"').replace('\n', '; '))
    cfg_path = os.path.join(root, 'config_qemu_upc.json')
    try:
        cfg = json.load(open(cfg_path))
    except Exception as e:  # noqa: BLE001
        cfg = {}
        say(f'config:         ERROR reading {cfg_path}: {e}')
    domain = cfg.get('domain', '')
    if domain:
        xml = sh(['virsh', '-c', 'qemu:///system', 'dumpxml', domain])
        emu = re.search(r'<emulator>([^<]+)</emulator>', xml or '')
        say('QEMU (emulator):' + (' ' + sh([emu.group(1), '--version']).splitlines()[0] if emu else ' not found in domain XML'))
        disks = re.findall(r"<disk type='file' device='disk'>.*?<source file='([^']+)'", xml or '', re.S)
        say('domain state:   ' + sh(['virsh', '-c', 'qemu:///system', 'domstate', domain]))
        for d in disks:
            info = sh(['qemu-img', 'info', '-U', '--backing-chain', d], timeout=120)
            fmt = ', '.join(re.findall(r'^file format: (\S+)', info, re.M)) or info[:120]
            say(f'guest disk:     format(s) along the backing chain: {fmt}; backing files: '
                f'{len(re.findall(r"^backing file:", info, re.M))}')
            note(f'guest disk path: {d}\n{info}')
        say('guest image SHA-256: NOT computed (the image changes whenever the VM runs). Decide which file the'
            ' paper means (e.g. a read-only backing file above); the detail file lists the chain.')
    if a.guest:
        up = sh(['virsh', '-c', 'qemu:///system', 'domstate', domain]) if domain else ''
        if up.strip() == 'running':
            g = sh(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', a.guest,
                    "uname -srm; grep '^PRETTY_NAME=' /etc/os-release"])
            say('guest OS/kernel: ' + g.replace('\n', '; '))
        else:
            say(f'guest OS/kernel: skipped (VM state: {up or "unknown"}; this script never starts it)')
    else:
        say('guest OS/kernel: not requested (pass --guest user@ip while the VM is running)')

    # ---- 2. run records -------------------------------------------------------------------------
    say('\n## 2. Run records')
    only = set(x for x in (a.only or '').split(',') if x)
    recs = []
    for p in sorted(glob.glob(os.path.join(runs_dir, '*.json'))):
        try:
            r = json.load(open(p))
        except Exception:  # noqa: BLE001
            continue
        if not {'label', 'steps_file', 'launch_line'} <= set(r):
            continue
        if only and r['label'] not in only:
            continue
        recs.append(r)
    recs.sort(key=lambda r: r.get('created_at', ''))
    say(f'runs dir: {runs_dir}   records found: {len(recs)}' + (f' (restricted by --only to {len(only)} labels)' if only else ''))
    say('run | created (UTC)        | commit  | cells | reps | curated | steps file: exists, lines, --duration lines, sha256[:12]')
    run_zdir = {}
    for i, r in enumerate(recs, 1):
        rid = f'R{i}'
        sf = r.get('steps_file', '')
        if sf and os.path.isfile(sf):
            lines = [l for l in open(sf, encoding='utf-8', errors='replace').read().splitlines() if l.strip()]
            dur = sum(1 for l in lines if '--duration' in l)
            sfi = f'yes, {len(lines)}, {dur}, {sha256(sf)[:12]}'
        else:
            sfi = 'MISSING'
        env = launch_env(r.get('launch_line', ''))
        run_zdir[rid] = (env.get('ZSTD_DIR', ''), env.get('ZSTD_RUN_ID', r['label']))
        say(f"{rid:<3} | {r.get('created_at', '')[:19]:<20} | {str(r.get('git_sha'))[:7]:<7} | {r.get('cells'):>5} | "
            f"{r.get('reps'):>4} | {str(r.get('curated')):<7} | {sfi}")
        note(f"{rid} = label {r['label']}; steps_file {sf}; ZSTD_DIR {env.get('ZSTD_DIR', '')}; "
             f"retention {r.get('config', {}).get('retention')}; launch_line: {r.get('launch_line', '')}")
    say('(label for each R is in the detail file)')

    # ---- 3 and 4. replicate indices and chain contiguity ---------------------------------------------
    say('\n## 3. Replicate indices   ## 4. Chain check')
    key_runs = defaultdict(set)   # workload key -> runs that have a rep001 for it
    for rid, (zdir, run_id) in run_zdir.items():
        if not zdir or not os.path.isdir(zdir):
            say(f'{rid}: ZSTD_DIR not found ({"unset" if not zdir else "missing on disk"}); skipped')
            continue
        chains = sorted(glob.glob(os.path.join(zdir, '*', '*', '*', f'rep*__{run_id}')))
        reps = Counter(os.path.basename(c).split('__', 1)[0] for c in chains)
        ok = gap = nobase = empty = 0
        bad = []
        for c in chains:
            key = os.path.relpath(os.path.dirname(c), zdir)
            if os.path.basename(c).startswith('rep001__'):
                key_runs[key].add(rid)
            nums = sorted(int(m.group(1)) for f in os.listdir(c) if (m := re.fullmatch(r'(\d{6})\.zst', f)))
            sizes0 = [f for f in os.listdir(c) if f.endswith('.zst') and os.path.getsize(os.path.join(c, f)) == 0]
            missing = sorted(set(range(0, (nums[-1] + 1) if nums else 0)) - set(nums))
            problems = []
            if not nums or nums[0] != 0:
                nobase += 1; problems.append('no base 000000.zst')
            if missing:
                gap += 1; problems.append(f'gap at member(s) {missing[:10]}{"..." if len(missing) > 10 else ""}')
            if sizes0:
                empty += 1; problems.append(f'{len(sizes0)} empty member file(s)')
            if problems:
                bad.append((c, problems))
            else:
                ok += 1
        say(f'{rid}: {len(chains)} chain folders; replicate indices: '
            + ', '.join(f'{k}={v}' for k, v in sorted(reps.items())))
        say(f'{rid}: chains contiguous from 000000 with no empty member: {ok}; with a gap: {gap}; '
            f'without a base: {nobase}; with an empty member: {empty}')
        for c, p in bad:
            note(f'{rid} PROBLEM {c}: {"; ".join(p)}')
    multi = sum(1 for runs in key_runs.values() if len(runs) > 1)
    say(f'workload keys with a rep001 in more than one run: {multi} of {len(key_runs)}'
        ' (non-zero means the replicate counter restarted per run)')

    # ---- 5. consumer.log --------------------------------------------------------------------------
    say('\n## 5. consumer.log')
    clog = os.path.join(root, 'consumer.log')
    if os.path.isfile(clog):
        pats = {'zstd base write failed': 0, 'zstd delta write failed': 0, 'zstd delta did not succeed': 0}
        hits = []
        with open(clog, encoding='utf-8', errors='replace') as f:
            for line in f:
                for k in pats:
                    if k in line:
                        pats[k] += 1
                        ts = re.search(r'memory_dump-(\d{17})', line)
                        hits.append((k, ts.group(1) if ts else '?'))
        for k, v in pats.items():
            say(f'"{k}": {v} line(s)')
        for k, t in hits[:30]:
            say(f'  {k} at image timestamp {t}')
        if len(hits) > 30:
            say(f'  ... {len(hits) - 30} more in the detail file')
            for k, t in hits[30:]:
                note(f'  {k} at image timestamp {t}')
    else:
        say(f'consumer.log not found at {clog}')

    # ---- 6. failed queue ---------------------------------------------------------------------------
    say('\n## 6. Failed queue')
    qdir = cfg.get('queueDir', '')
    fdir = os.path.join(qdir, 'failed') if qdir else ''
    if fdir and os.path.isdir(fdir):
        fs = [os.path.join(fdir, f) for f in os.listdir(fdir) if f.endswith('.json')]
        if fs:
            mt = sorted(datetime.fromtimestamp(os.path.getmtime(p), timezone.utc) for p in fs)
            say(f'{len(fs)} job record(s); modified from {mt[0]:%Y-%m-%d %H:%M} to {mt[-1]:%Y-%m-%d %H:%M} UTC')
        else:
            say('0 job records')
    else:
        say(f'failed queue directory not found ({fdir or "queueDir unset in config"})')

    # ---- 7. libvirt per-domain folder -------------------------------------------------------------------
    say('\n## 7. libvirt per-domain folder (optional check)')
    ds = glob.glob('/var/lib/libvirt/qemu/domain-*')
    state = sh(['virsh', '-c', 'qemu:///system', 'domstate', domain]) if domain else 'unknown'
    say(f'VM state now: {state}; per-domain folders present: {len(ds)}')
    say('(run once with the VM up and once after it stops: expect >=1 then 0)')

    detail.close()
    print(f'\nDone. Nothing was changed. Paste the screen output above; the detail file stays at {detail_path}')


if __name__ == '__main__':
    sys.exit(main())
