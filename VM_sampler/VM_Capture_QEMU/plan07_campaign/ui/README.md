# Capture Console

A visual planner for memory-signal capture subsets, plus an optional server-side
bridge to launch, monitor, and stop captures from the browser.

There are two ways to use it, sharing one codebase:

| | how you launch |
|---|---|
| **Static console** (`capture_console.html`) | builds the plan + prints the exact command; you paste it into `screen` yourself. Zero network code. Publishable as an Artifact. |
| **Served console** + **bridge** (`console_bridge.py`) | same UI plus a *Launch on server* panel: Preflight → Launch → live log → Stop, driven over a localhost bridge. |

## Files

```
capture_console.template.html   UI source (logic + design). Hand-edit this.
build_console.py                injects pipeline data; emits the two variants.
capture_console.html            GENERATED static build (network-free; the Artifact).
capture_console.served.html     GENERATED served build (has the launch panel).
console_bridge.py               localhost server that the served console talks to.
console.sh                      laptop launcher: tunnel + bridge + migration agent.
migrate_agent.sh                laptop-side migration loop (pulls chains here).
```

Nothing in the HTML is hand-authored data. `build_console.py` reads the single
sources of truth (`full_campaign_steps.txt`, `generate_database_steps.py`,
`subset_run.py`) and injects the workloads, scaling rules, and metric taxonomy,
so the UI can never drift from the pipeline.

## Build

```bash
python3 ui/build_console.py            # -> capture_console.html   (static, network-free)
python3 ui/build_console.py --served   # -> capture_console.served.html (launch panel kept)
```

Run these after any change to the template or to the Python pipeline.

## Launch from the browser (the bridge)

The bridge runs **on the capture server** (where `run_files_controlled.py` and
the QEMU guest live). It never opens SSH itself — it shells out locally and wraps
the orchestrator in `screen`. Your browser reaches it through an SSH tunnel.
Because the bridge and the capture's `screen` both live on the server, a launched
capture keeps running even if your laptop sleeps or disconnects — reconnect and
you rejoin it.

### One command (recommended)

From your **laptop**:

```bash
plan07_campaign/ui/console.sh <you>@<server>
```

That SSHes in, builds the served page, starts the bridge, forwards the port, and
opens the console in your browser. `Ctrl-C` stops the bridge and closes the
tunnel. (`<you>@<server>` is your **research server** login — not the guest VM;
the guest `kali@…` goes in the console's Host identity fields.)

Ports: the launcher uses **`LPORT` on your laptop (default `8765`)** and
**`RPORT` on the server (default `8000`)**; the tunnel maps `laptop:LPORT →
server:RPORT`, so they're independent. If `8765` is also taken locally, pick
another:

```bash
LPORT=9123 plan07_campaign/ui/console.sh <you>@<server>
```

### Or the two steps by hand

```bash
# on the server
cd .../VM_sampler/VM_Capture_QEMU
python3 plan07_campaign/ui/build_console.py --served      # once, or after edits
python3 plan07_campaign/ui/console_bridge.py --port 8000
# it prints a URL that contains a one-time token

# on your laptop
ssh -L 8000:localhost:8000 <user>@<server>
# open the printed  http://localhost:8000/?token=<token>  in your browser
```

Then in the *Launch on server* panel: **Preflight** runs `qa/preflight.sh` on the
generated steps and shows the verdict; **Launch capture** starts the orchestrator
in a detached `screen` (`mem_console_<label>`) and tails its log live; **Stop**
ends it.

## Data migration (server -> laptop)

The console can also pull completed capture chains down to your laptop and free
them on the server. The hard constraint: **the copy must be initiated by the
laptop** — the server can't open a connection back to a NAT'd, sleeping laptop,
and the browser is sandboxed (no `rsync`). So the UI does not copy anything
itself; it drives a laptop-side agent through a control file on the server.

- **`migrate_agent.sh`** (laptop) is auto-started by `console.sh` next to the
  tunnel (set `MIGRATE=0` to skip). Each cycle it reads
  `<parent-of-ZSTD_DIR>/.migration/control.json` and:
  - pulls any **on-demand requested** chains promptly (checked every `POLL_SEC`),
  - if **auto** is on, sweeps every *stable* chain each `<interval>` minutes.
  It moves each chain with `rsync --partial --remove-source-files` (the transfer
  from `pull_traces.sh`, freeing the server), then appends the chain to
  `.migration/ledger.jsonl` so the UI can mark it migrated.
- A **migratable unit** is one retention chain leaf
  (`<family>/<workload>/<param_sig>/repNNN__<runid>`). It is **ready** once idle
  for the stable window (mtime older than `stable_min`, default 10 min) — the
  same "no longer growing" signal `pull_traces.sh` uses, so an in-progress chain
  is never pulled mid-write.
- **Per chain, three actions:** **Migrate** = move (pull, then free the server);
  **Copy** = pull but keep the chain on the server; **Delete** = remove it from
  the server *without* pulling (permanent, for traces you don't want). Migrate
  and Copy queue for the agent (each ledgered with its `action`, so the UI shows
  `copied` for a kept chain and `migrated` for a moved one). Delete is done by
  the bridge itself — no agent needed — and **refuses a chain still being
  written** (younger than `stable_min`) so it can't nuke the active run.
- **Bridge endpoints:** `GET /migration/list` (per-chain status
  growing/ready/queued/copied/migrated/deleted, from rep-leaf mtimes + the
  ledger), `POST /migration/request` (queue one **validated** rep leaf, `mode`
  move|copy), `POST /migration/delete` (server-side remove of one validated,
  idle rep leaf), `POST /migration/config` (set interval + auto). All chain
  paths are validated as existing rep leaves under `ZSTD_DIR`, so only real
  relpaths reach `rsync` or `rmtree`.
- **UI:** the *Data migration* card — per-chain badges and **Migrate / Copy /
  Delete** buttons (Delete behind a confirm), an interval field + **auto** toggle
  (**Save schedule**), polled every 10 s.

The console's **ZSTD dir** (panel 07) and the agent's `ZSTD_REMOTE_DIR` must be
the same server path (both default to
`/project/homes/jeries/memory_traces/zstd_local`) — that's what makes the bridge
and the agent agree on where `.migration/` lives.

`pull_traces.sh` still exists as the standalone, no-console puller; the agent is
the console-driven equivalent and leaves it untouched.

## Security model

The bridge runs shell on a shared, no-root research server as *your* user. It is
built to be safe under that reality:

- **Localhost only.** Binds `127.0.0.1` — never network-exposed. Reachable solely
  through your own `ssh -L` tunnel.
- **Token gate (Jupyter-style).** A random token is printed to the server
  terminal on startup and required both in the `GET /?token=…` URL and in the
  `X-Console-Token` header on every API call. `GET /` without it returns 403 and
  never leaks the token — so a co-resident process that can also reach
  `127.0.0.1:port` still cannot drive the bridge. Keep the token private.
- **Trusted re-derivation.** The browser sends *structured intent* — the config
  and which `workload×duration×scale×rep` cells to keep, in order — **never raw
  commands.** The bridge regenerates the actual guest commands from
  `full_campaign_steps.txt` via `subset_run.py`, so nothing arbitrary can reach
  the guest. The launch line is composed by `subset_run.py`'s own env functions,
  so only known env keys can appear.
- **Namespaced screens.** Sessions are `mem_console_<label>`; status and stop only
  ever touch `mem_console_*`.
- **The SSH key never touches the bridge.** `run_files_controlled.py` uses it on
  the server exactly as before; the bridge only passes its *path* through.

## Known rough edges

- **Stop is best-effort.** Quitting the `screen` stops the orchestrator between
  SSH steps, but the current step's `nohup`'d producer/consumer are cleared with
  `pkill -f <script>` (process-name-global; fine on this single-VM/single-user
  host). After a mid-run stop, check `virsh list --all` / `virsh domstate` and for
  leftover guest scratch before the next launch. A clean mid-run stop is a
  follow-up.
- **`metrics` retention** still lands flat rather than in the retention tree
  (pre-existing; see `DATABASE_DESIGN.md`). Unaffected by the bridge.
- The bridge is single-process and unauthenticated beyond the token + tunnel.
  Run one per user; do not expose the port beyond localhost.
