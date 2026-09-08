# Plan 10, part 2: the runner, the bridge, and a shippable app

Status: plan, 2026-09-08. Written before the code; each phase names what it ships and what
test proves it. Nothing below runs a model; the output is a feature file with its sidecar.

Goal, in JK's words: a ready-to-go web app that works on data, local or over SSH, and runs
at least the three examples the console can load. Open for upgrades: every component
separable and exchangeable.

## 0 · What is fixed by facts on this machine

- `zstd` 1.5.7 CLI present; chains reconstruct with `zstd -d --long=31 --patch-from` exactly
  as `reconstruct_zstd_chain.sh` does.
- `numpy` 2.2.6 present (the analysis env pins numpy + scikit-learn). The runner uses numpy;
  the console's data layer stays stdlib.
- The differ binary is a build artifact (`target/` is untracked). The runner resolves it from
  `PLAN10_DIFFER`, then `<repo>/VM_sampler/VM_Capture/live_delta_calc_modular/target/release/`,
  and refuses with the `cargo build --release` line if neither exists.
- `ssh`, `scp`, `rsync` present. The server is Linux (GNU `find -printf`).
- Cost: the differ is ~2.1 s/pair at speed 2 on a 1 GB pair, plus ~1-2 s to reconstruct
  each snapshot. 700 pairs is ~40 min per recording. The examples select every usable
  recording; a full run is a background job of many hours. A `max_pairs` parameter on Cells
  bounds a run for a smoke test and is recorded in the sidecar like any other choice.

## 1 · Components, each one file, each with its test

```
plan10_analysis/
  sources.py          LocalSource / SshSource: list the tree, fetch a recording to a local
                      cache. Same interface; the executor never knows which it has.
  corpus_manifest.py  refactored: scan_listing(entries) builds recordings from a listing;
                      local walk and ssh `find` both produce listings.
  runner/
    chain.py          walk one chain: yields (seq, prev_raw, curr_raw) with a two-file
                      rolling window; deletes as it goes; resumable by seq.
    differ.py         run the differ on one pair (--speed N --sparse); parse the sparse CSV
                      into {page_index, <column>: float32[]}; select columns.
    extract.py        L1 store: per recording, per (speed, channel set) -> one npz holding
                      seq, page_index and the requested columns. Skips what exists.
    stages.py         one function per module kind, pure: field -> field / series / tiles /
                      features. Stats, cepstrum, PLV, CUSUM reuse the code the project has.
    executor.py       validate (scheme.py), topologically order, run per recording, write
                      status.json atomically, honour control.json (pause/stop), write the
                      output npz + csv + sidecar.
  ui/
    analysis_bridge.py  stdlib HTTP on 127.0.0.1 + token: /scan, /validate, /run, /status,
                        /log, /stop, /results, /source/test. Serves the built console.
    analysis.sh         launcher: build, bridge, browser.
    analysis_console.template.html  gains a Source panel (local path | ssh fields), a Run
                        panel (launch, progress, log tail, stop), and a Results list.
tests/
  test_plan10_sources.py    listing parsing, ssh command construction, local fetch
  test_plan10_runner.py     synthetic corpus through chain -> differ -> extract -> stages
                            -> executor for all three examples; APF equality against
                            plan02_apf_helper on the same pair
  test_plan10_bridge.py     endpoints against a live bridge on a random port
```

## 2 · Phases and their exit tests

| Phase | Ships | Proven by |
|---|---|---|
| 1 | `sources.py`; manifest over listings | listing round-trip; ssh command lines; manifest over a local walk equals manifest over the same tree's listing |
| 2 | `chain.py`, `differ.py`, `extract.py`; synthetic corpus fixture | a synthetic chain reconstructs byte-exact; the differ's `hamming` on a pair equals popcount(xor); derived APF equals `plan02_apf_helper` on the same pair |
| 3 | `stages.py`, `executor.py` | the three examples run end to end on the synthetic corpus; output npz has one row per tile; sidecar carries the scheme, acknowledgments, source, differ version, per-recording status |
| 4 | bridge, launcher, UI wiring | each endpoint answers; a run started from the page reaches `done`; a stop is honoured |
| 5 | a bounded real run | B1 example over 3 real recordings with `max_pairs` 40 finishes; the console shows it |
| 6 | README, commit | someone else can start it with one command |

## 3 · Decisions taken here, each reversible

- **Extraction unit is the recording, not the pair.** One L1 npz per (recording, speed,
  channel set). Two Channels modules in one scheme share the same differ pass: the executor
  extracts the union of channels once.
- **Speed for extraction is a run parameter** (`--speed`, default the config value the
  console assumes). Recorded in the sidecar. A recording carrying its own speed would win.
- **Complex on the differ's own columns.** Both inputs come from the same sparse rows of
  the same pair, so the join is by construction.
- **Collapse with unchanged = zero divides by the page count** (the config default,
  262144 for 1 GiB, until a recording carries its own).
- **PLV baseline "cell" mode uses the first recording of the input in manifest order** unless
  the Baseline module names one (`recording` param, added, optional).
- **Not implemented in this pass, refused with a message, still on the palette:** wavelet
  (needs pywt), scattering (needs kymatio/torch), MSC. Vectorize, Single, Block, Deep, CUSUM
  are implemented because their pieces exist.
- **SSH fetches, it does not execute remotely.** Recordings are rsynced into a local cache
  and the executor runs locally. Remote execution is the capture console's model and can be
  added behind the same `Source` interface later.

## 4 · What "works on the examples" means, testably

B1: `stats` over collapsed magnitude windows, 8 features per tile, one row per (recording,
window). Complex: `fft` band energies and `cepstrum` peak/snr over complex tiles, concatenated.
PLV: baseline fitted on one recording's tiles, PLV category counts per tile for the rest.
Each output is a `.npz` with `X`, `feature_names`, `tile_keys`, and a csv twin, plus
`<label>.sidecar.json`.

## 5 · Findings while building (2026-09-08)

- **The differ mis-indexes pages on dumps that are not a multiple of 4 MiB.** `main.rs` splits
  each file into 16 segments and reads 256 KB chunks from each segment's start, so when
  `file_size / 16 < 256 KB` the segments overlap and the running `page_index` counter is wrong
  (a 256 KB pair reported one change three times, at pages 10, 70 and 126). A 1 GiB guest dump
  is unaffected. `runner/differ.py` refuses any other size that is not a multiple of 4 MiB and
  the synthetic fixture uses 4 MiB dumps. `[traced: main.rs THREAD_COUNT, CHUNK_SIZE; probed at
  64, 256, 1024, 4096 pages]`
- **numpy appends `.npz` to any save name without it**, so an atomic write must name its temp
  file `*.tmp.npz`, not `*.npz.tmp`.
- **The B1 example needed a reduction the graph did not have.** Collapse averaged channel
  values; APF is `K/N`. Collapse gained `reduce = changed_fraction`, and the test asserts the
  fixture's APF (7 changed pages per pair over 1024) comes out exactly.
- **Two examples piped a complex field straight into Window.** That is a tile at full page
  resolution, which nothing in the record ever computed. The examples now collapse to the mean
  phasor per pair first, and Window refuses a page-resolution input in both engines.
- **The "no runner implements yet" wording became false the moment the runner existed**; the
  warning now states the cost instead. The severity stays soft (UX section 14.1 item 5).
- **Bounded real run** (B1, three recordings from mem / cpu / io, `max_pairs` 40, speed 2):
  see the status recorded in the report for this session.
