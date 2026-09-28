NO BLOCKING FINDINGS

# CHECK_3.md: checker's report on the paper 2 analysis toolkit, cycle 3 (2026-09-16)

Rules kept: no server, no path under a server mount, no remote command of any kind; the sandbox
family's sources and workload names were not read, grepped or named; no paper prose was written;
no file outside `plan11_encoding_ladder/` was edited and the only file written inside it is this
report; nothing was committed; `P2_STRUCTURE.md` was not touched. Every test and every driver run
below used synthetic data generated on this machine with builder 1's generator (`synth.py`).
Environment: Python 3.10.12, numpy 2.2.6, scikit-learn 1.7.2, scipy 1.15.3, matplotlib 3.10.7,
joblib 1.5.2, pytest 9.1.1, the `zstd` binary on PATH (1.5.7), the `zstandard` module absent, no
TeX compiler, eight cores.

Summary. The test suite passes (158 passed, 1 skipped, the skip being the `zstandard` reader test on a
machine without that module). The driver runs end to end with exit 0 twice: once with the runbook's
standard command as written (which, at the default C1 threshold, admits only the gemm cells and the two
idle cells of the corpus), and once with the preconditions re-run by hand at `--c1-activity-min 0.001`,
the pass table and the idle admissibility record filled, so that every gate exercises its real path. No
gate deviates from its definition in P2_STRUCTURE.md section V or council report section 2: every
threshold value, every verdict string and every refusal string match, every gate function cites its
definition, and every choice the definitions leave open is a parameter recorded in `params`. The three
builders' modules agree on every interface file (headers and keys listed in record (4)). All 64 command
lines of RUNBOOK.md parse against the module CLIs and the same subcommands ran under the driver. The
skeleton `apf_paper/p2_skeleton.tex` carries no body prose. The cycle-2 fix B1 (one score after a B1-G3
quarantine) was re-verified in situ: G-L, G-DIM, G-M, Table 7, Table 8 and the exclusion list all read
the re-run.

Thirteen minor findings follow, most severe first. The first three matter for the real run (the C1
threshold decides which kernels exist for the paper; a plain resume re-runs four split stages that cost
hours; three gates judge on a five-permutation null that B1-G1 refuses); the rest are carried items,
interface gaps that name what is missing, and cost claims.

---

## BLOCKING

None.

---

## MINOR

### M1. At the default C1 threshold the runbook's smoke run admits three of its twelve synthetic kernels and the real run would refuse at least six of the twelve; on this corpus it admitted one of four, and the run completed with exit 0 (CHECK_1 M1, CHECK_2 M1; refused twice as more than one line)

`gates_precondition.py:28` (`C1_ACTIVITY_MIN = 0.02`, the apf_queue re-map of CR 2.1 item 1, 5,243 pages)
and `:150` (`apf_max >= c1_activity_min` for every kernel cell). `run_moves.py:481-500` (`_add_run_args`)
has no `--c1-activity-min` and `run_moves.py:118-124` does not pass one, so the runbook's standard
command (RUNBOOK.md:84-86) and its smoke run (RUNBOOK.md:66-68) run C1 at 0.02. Run 1 below (the
runbook's command as written on 4 kernels x 3 reps + 2 idle cells): `preconditions.json` lists 9
excluded cells (every floyd, gibbs and histogram cell; `apf_max` 0.0165, 0.0016, 0.0088), 5 of 14 admissible,
G-C content `not run: no admissible cell for gibbs,histogram`, G-DEC `not run: no admissible floyd cell`,
Table 6 `--` in every score cell of the excluded kernels, and exit 0. By P2 Table 3's footprints the
same threshold on the real corpus refuses floyd, histogram and nbody (2,048 pages, `apf_max` about
0.008), gibbs (256 pages), fft (4,096 pages, about 0.016 unless a pass lights more) and the lexer (at
floor), so at least six of the twelve kernels never reach a gate; gemm (the re-seed lights about
8,192 pages), fem_assembly (8,256) and bnb_tsp (measured maximum 0.0917) pass; on the runbook's
synthetic smoke corpus the same three presets (gemm, fem_assembly, bnb_tsp) are the only ones whose
`K_max` clears 5,243 pages. This is not a deviation (CR 2.1 item 1 re-maps C1 for the idle cell
only), and the toolkit's own chain test (`tests/test_gates_chain.py:36`) already runs at
`c1_activity_min=0.0005` to get past it. The decision is the author's ("For the author" item 1).
Fix (builder 3, three lines plus one runbook sentence): add
`ap.add_argument("--c1-activity-min", type=float, default=None)` to `_add_run_args`, append
`["--c1-activity-min", o.c1_activity_min]` to the `preconditions` args at `run_moves.py:118-124` when
given, and state in RUNBOOK.md move 2 that the default excludes every kernel whose `K_max` is below
5,243 pages and which real kernels that is. The staleness side already holds (al-Farabi 7.1: a
by-hand re-run of the preconditions marks every later move stale; verified on run 2).

### M2. A plain resume with nothing changed re-runs the four split stages, both G-X runs per rung, G-L, Table 6 and the two alias steps (FIX_2 section 5, recorded there for this check)

`run_moves.py:314-320` (`_stale_reason`) hashes each declared input file whole; `run_moves.py:181`,
`:183`, `:190`, `:195` declare `gates/selection.json` as an input of `splits <rung>`, `gx <rung>`, `gl`
and `tables table6`, and `run_moves.py:187`, `:193` declare `gates/g3_flags.csv` as an input of the two
`alias` steps. `select <rung>` (`gates_temporal.py:555-650`) adds one rung's entry to the shared
`selection.json` at moves 6, 9, 10, 11 and 12, and `g3 <rung>` rewrites `g3_flags.csv` at each of those
moves, so a step that ran at move 7 recorded the one-rung file and finds the five-rung file on resume.
Verified on run 2: `run --moves 2-13 --dry-run` with nothing changed marks 12 steps stale (`alias` x2 on
`g3_flags.csv`; `splits apf`, `gx apf`, `gl`, `tables table6`, `splits persist`, `gx persist`, `splits
content`, `gx content`, `splits wapf`, `gx wapf` on `selection.json`) and skips the other 58 (record
(2c)). On the real corpus a resume of moves 7 to 13 would re-run four split stages at 500 permutations
(RUNBOOK.md:112-113: LORO's null about 48,000 forest fits per rung) for no change, producing the same
files at the same seeds. Fix (builder 3): let `inputs` use the per-key syntax the outputs already use
(`_output_exists`, `run_moves.py:284-305`): declare `json:gates/selection.json:<rung>` for the rung's
own steps and `csv:gates/g3_flags.csv:rung=<rung>` for `alias`, and make `_stale_reason` hash the named
JSON entry (`json.dumps(j[key], sort_keys=True)`) or the matching CSV rows instead of the file; for
`gl`, `tables table6` and the move-12 consumers that read every rung, keep the whole file. A test:
after a full run, `--dry-run` marks zero steps stale.

### M3. G-F part (i), G-X and the clustering null judge on any permutation count, while B1-G1 refuses below 500; on the smoke run a five-permutation null voided APF and persistence and every score cell of Tables 6 and 7 for those rungs

`gates_precondition.py:341-353` (`_gf_part1`) draws `n_perm` window-level label shuffles and
`gates_precondition.py:411-415` writes `GF_VOID` on strict exceedance of their p95 whatever `n_perm` is;
the driver passes `--n-perm o.null_perm` (`run_moves.py:146`, `:223`). `gates_comparison.py:247-251`
(`gate_gx`) and `models.py:611-624` (`run_clustering`, `exceeds_ari`, `exceeds_nmi`) do the same.
`models.py:215-223` (`b1_g1_verdict`) instead writes `not run: N permutations < 500` below
`B1G1_MIN_PERM`, because CR 2.1 item 9 defines the label-shuffle null with at least 500 permutations;
CR 2.2 item 21 (G-F (i)) and item 26 (G-X) name that same null ("the label-shuffle null's 95th
percentile"). Verified on run 2 (`--null-perm 5`): `gf.csv` part (i) reads `void: idle reps separable
under this rung` for `apf` and `persist` (score 0.5 against a p95 of 0.4167 from five draws over two idle
reps) and `inseparable at floor` for the other three; `report/tables/table6.csv` then prints the void
string in every score, null, majority and rank cell of every row, and Table 7 does the same for the
`apf` and `persist` rows (record (2b)). B1-G1's cells on the same run read `not run: 5 permutations <
500`. On the real run at 500 permutations nothing deviates; on a smoke run the runbook's promise
(RUNBOOK.md:61-62, "every table and figure exists once with every not run: cell named") is not kept,
because a five-draw refusal prints as a finding. Fix (builder 2, one guard in three places): in
`_gf_part1`, `gate_gx` and `run_clustering`, when `n_perm < B1G1_MIN_PERM` write `not run: N
permutations < 500` (the G-F row's verdict, the leak verdict, and `exceeds_*` as that string) exactly as
`b1_g1_verdict` does, and record `n_perm` in the row; `rung_override` in `_report_common.py` already
treats only `GF_VOID` as an override, so the tables then print the numbers with the `not run` string in
the G-F (i) column.

### M4. Table 7's `combined (matched)` rows for LORO and within-trace name a file no gate writes

`gates_comparison.py:289-298` (`gate_gdim`) runs the dimension-matched comparison for the LOKO/archetype
split only and writes it under `gates/splits_matched/combined/<gid>/loko__archetype/`;
`tables.py:510-511` (`table7`) plans `combined (matched)` rows for all three splits and both label
spaces, and `tables.py:473-489` (`_matched_scores`) then prints `not run:
gates/splits/combined_matched/W16_H8/loro__kernel/scores.json missing (and gates/gdim.json has no
matched[loro__kernel])` in the feature-count, accuracy, macro-recall, null and rank cells of the four
extra rows (run 2, record (2b)). CR 2.2 item 32 asks for one feature-count-matched comparison beside the
combined rung; SPEC 3.7.7 calls it "the `combined (matched)` row" (one row); SPEC 6.3 asks for one row
per (rung, split). Nothing is silent (the cell names the file), but the two builders disagree on the
matched row's splits. Fix, one of: (a) builder 2 runs the same `run_split_stage(..., reduce_to=d_star,
base_dir="splits_matched")` for `("loro", "kernel")`, `("within_trace", "kernel")` and the two
archetype-space rows, five more calls at `gates_comparison.py:294`; or (b) builder 3 restricts the
`combined (matched)` plan to `("loko", "archetype")` at `tables.py:510-511` and its `note_comment` says
the matched comparison is LOKO only. The author chooses (item 4 below).

### M5. Table 7's `feature count` for `combined (matched)` prints the pre-reduction width (CHECK_2 M3, refused as three lines)

`tables.py:542`, `:544` print `scores.json["feature_count"]`, which `models.py:524` writes as the width
before the per-fold reduction (`d_full`, 60, or 40 after a quarantine) while the forest saw `d_target`
(`models.py:143-149`, returned per unit as `d_used` and never written). Run 2: the row reads `feature
count 40` beside `G-DIM: declared reduction (d = 40, matched 6, train_importance)`. Fix: write
`feature_count_used = min(v["d_used"] for v in preds.values())` into `scores.json` and
`with_quarantine` (`models.py:520-529`, `:486-488`), and print it at `tables.py:542-544` when present.

### M6. `gates_temporal gord` and `grid` accept `--n-jobs` and ignore it; the runbook's smoke-run heading says "minutes" (CHECK_1 M4 and M9, CHECK_2 M7; the cost statement was added, the code was not changed)

`gates_temporal.py:426-427` (`gate_gord`) and `:290-291` (`gate_grid`) take `n_jobs` and never use it;
`run_moves.py:71-72` still lists both in `NJOBS_COMMANDS`. Measured on run 2 (12 kernel cells x 120
pairs, `--n-jobs 4`, 300 trees): G-ORD 327 s (apf), 324 s (persist), 641 s (content), 318 s (wapf),
643 s (combined), 2,253 s of the 2,708 s run, single process. RUNBOOK.md:59 still heads the section "The
smoke run (minutes, synthetic corpus)" while its own paragraph at :72-76 says the better part of an hour
per rung at the 104-cell corpus. Fix (builder 2): run the `n_order_perm` and `null_perm` loops of
`gate_gord` through `joblib.Parallel(n_jobs=n_jobs)` as `_gf_part1` (`gates_precondition.py:347-349`)
does, or drop the flag from the two CLIs and from `NJOBS_COMMANDS`; (builder 3) change the heading to
"The smoke run (synthetic corpus; about an hour at the full corpus)".

### M7. The roll-up counts "no applicable kernel for G1" as a failed gate (CHECK_1 M8, CHECK_2 M4, al-Farabi 7.3; the author's choice)

`gates_temporal.py:597` (`applicable = {"G1": g1, "G4": g4}`) always includes G1; `_rollup`
(`gates_temporal.py:542-543`) returns `not applicable: no applicable kernel` when every kernel is
`TREND_PRESENT` or relabelled IDLE, which is not `pass`, so `_all` is false and the selection reads
`best-feasible` for a reason unrelated to the grid point, while G2 in the same situation is dropped
from the count (`:598-599`). Did not arise on either run (every kernel passed G1). Fix, one of: drop G1
from `applicable` when its roll-up starts with `not applicable` (two lines, the form G2 has), or refuse
the selection outright; the author says which (item 6 below).

### M8. Table 6 prints `--` in the rows of kernels excluded by the preconditions and counts them in `n` (CHECK_2 M16, refused as more than one line)

`tables.py:329-330`, `:365-374` build the kernel rows and `n` from every `status = ok` kernel cell of
`cells.csv`, not from the admissible set (`gates/preconditions.json` `excluded_cells`). Run 1: floyd,
gibbs and histogram rows print `--` in every score cell with a `G-N status` copied from `gn.csv`
(`structural novelty`, `no kernel row`), and the `all` row reads `n = 12` while 3 kernel cells entered
the chain (record (2a)). SPEC section 7 asks that a cell that is not a number names what is missing.
Fix: read `preconditions.json` `excluded_cells`, print `not run: excluded by C1-C8 (all_hard_pass
false)` in those rows, and count `n` over admissible cells. The same read fixes `wapf_over_apf`
(`tables.py:932`), which also averages every `ok` cell (run 1's table carries floyd, gibbs and
histogram rows although the chain excluded them).

### M9. The idle cells' head drop is keyed two ways (CHECK_2 M10, refused as more than one line)

`gates_precondition.py:419` (G-F part (ii)) reads the idle cells' drop as `head_drop_for(hd, "idle")`,
the key `series.write_head_drop_template` writes (`series.py:220-226`); `gates_temporal.py:309`,
`:383-384`, `:446`, `gates_readings.py:84`, `gates_comparison.py:126` and `series.py:610`
(`build_features`) read it as `head_drop_for(hd, c["kernel"])`, where an idle cell's `kernel` is its
label-derived name (`sleep` on this corpus). Harmless at the default 0. Fix: resolve the key as
`"idle" if c["role"] == "idle" else c["kernel"]` at the seven call sites, or give `head_drop_for` a
`role` argument.

### M10. The G-L (ii) refusal's consequence is the author's manual step (CHECK_1 M5, CHECK_2 M8; the runbook sentence was added, the driver command was not)

SPEC 3.7.4 says that on `GL_SHOT_NOISE` "the driver then re-runs the split stage with `feature_drop =
("cov", "std", "peak2med")` and reports both"; `gates_comparison.py:30` declares `GL_FEATURE_DROP` and
nothing in `run_moves.py` reads `gl.csv`. RUNBOOK.md:270-274 now says the re-run is by hand. Did not
arise on run 2 (part (ii) `pass`, r2 0.21 over 4 kernels). Fix as CHECK_2 M8, or leave the runbook
sentence and record the choice (item 7 below).

### M11. G-DEC's `pass_frac` is a function parameter without a CLI flag, and the idle clause refused floyd on this run through one idle cell's slope of -0.002

`gates_readings.py:34` (`GDEC_PASS_FRAC = 0.5`) is a threshold the definition does not name (CR 2.2 item
34 (b) says "in at least seven of eight reps at the same pass phase" and nothing about the fraction of a
cell's passes); it is recorded in `params` (verified: `gdec.params.json` `pass_frac 0.5`) but
`gates_readings.py:379-385` (`main`) exposes `--min-run`, `--min-reps`, `--idle-slope-rule` and not
`--pass-frac`. On run 2 the exhibit read `no decay beyond floor or host` in all three cells because
`idle__rep00` had a whole-cell slope of `l0_q50_per` of -0.0023 against a phase-randomized p95 of 0.0021
(`idle_slope_rule = "any"`, CHECK_1 item 4), while (b) held (10 passes, phase 0), (c) held (K drop
0.477 against l0 drop 0.516) and (d) held (slope -49.6 against a surrogate p05 of -8.5); the kernel row
then reads `no decay` because 3 reps cannot reach `min_reps = 7`. Fix: add `--pass-frac` to `main`;
the idle-clause rule stays the author's (item 8 below).

### M12. The synthetic corpus declares 600 s for 120 pairs, so G2's seconds columns can never pass and every selection is `best-feasible`

`synth.py` writes 120-pair cells whose sidecar `dt_est_s` is 600 / 120 = 5 s, outside the paper's
[0.500, 0.644] s bracket; `gates_temporal.py:214-233` (`g2_kernel`) computes the seconds coverage `W dt /
T_seconds` with `T_seconds = 600 / passes`, so on run 2 (gemm declared 5 passes, floyd 12) every
`G2_0500` and `G2_0644` cell reads `fail` (coverage at W = 8 is 0.033) while `G2_pairs` reads `pass` at
W = 64 (2.67), and all five selections are `best-feasible` (record (2b)). On the real corpus the two
representations agree at the median cell (600 / 931 = 0.644 s), so nothing deviates; the smoke run
simply never exercises a passing G2 or the selection rule's first branch. Fix (builder 1, one
parameter): a `duration_s` field on `SynthSpec` defaulting to `n_pairs * 0.644` (or `n_pairs * 5` when
the author prefers), written into the trajectory directory's `truth.json` and read by `extract` from
`--duration`, so the synthetic `dt_est_s` sits in the bracket; or a runbook sentence that the smoke
corpus's G2 seconds columns are expected to fail.

### M13. The smoke corpus has 3 reps and 4 kernels, so G3's kernel flag, G-DEC's roll-up and G-M's sign test read their fixed minima (7 of 8 cells, 7 of 8 reps, 6 of 12 kernels) and cannot pass

Not a code defect: `G3_MIN_CELLS = 7` (`gates_temporal.py:38`), `GDEC_MIN_REPS = 7`
(`gates_readings.py:32`) and `GM_SIGN_RULE` (`gates_comparison.py:34`) are the definitions' numbers for
the 96-cell corpus. On run 2 every G3 kernel flag reads `rhythm flag: absent` although gemm's three
cells each flag `present` at the true period (peak 24, SNR 12.1 dB against a null p95 of 8.8), and every
G-M pair reads `difference with margin` with at most 2 kernels improving. The runbook's smoke corpus
(12 kernels x 8 reps) has the right counts once M1 lets the kernels through. Fix (runbook, one sentence
under 0b): a smoke corpus with fewer than 8 reps or 12 kernels reads those three gates at their minima
by construction; pass `--min-cells`, `--min-reps` by hand when a smaller corpus is used.

---

## For the author

Choices the definitions leave open that this check found consequential. I decide none of them.

1. **C1 for the kernel cells** (M1). CR 2.1 item 1 re-maps `apf_max >= 0.02` for the idle cell only; at
   that value the chain refuses at least six of the twelve kernels before any gate runs (floyd,
   histogram, nbody, gibbs, fft, lexer by P2 Table 3's footprints). Either re-declare C1 for kernel cells
   (a lower activity floor, or a rule on `K_max` against the idle band once the idle cells exist, which
   would make C1 a consequence of G-K0), and record it as a change to a pre-registered gate as Plan 08's
   restatements are; or keep 0.02 and state that the paper's chain runs on the kernels it admits. The
   toolkit runs either way once the driver carries the flag.
2. **The null's permutation floor on G-F (i), G-X and the clustering** (M3): apply B1-G1's 500 rule to
   every gate that cites the label-shuffle null, or leave them judging on whatever `--n-perm` gives and
   never read a smoke run's void.
3. **G-F part (i)'s executable form** (unchanged from SPEC 3.3.4, section 8 item 15): a within-trace
   window-level test with a window-level label shuffle, `part1_consequence = "void"`. On run 2 two idle
   reps at 120 pairs scored 0.5 (chance for two classes) and were still voided by a five-draw null; with
   eight idle cells at 500 draws the same design tests whether the forest can tell one idle rep's last
   windows from another's, which any slow within-cell drift satisfies. Al-Farabi's `"report"`
   alternative exists on the CLI (`--part1-consequence report`).
4. **The matched comparison's splits** (M4): LOKO only (one row, CR 2.2 item 32's reading) or every
   split (SPEC 6.3's reading).
5. **The staleness rule's granularity** (M2): per-key hashing of the shared files, or `--force` on a
   resume with the cost stated.
6. **The roll-up with no applicable kernel for G1** (M7): drop it from the count as G2 is, or refuse the
   selection outright.
7. **The G-L (ii) re-run** (M10): a conditional driver command, or the runbook's manual step.
8. **G-DEC's idle clause** (M11): `idle_slope_rule = "any"` refuses the exhibit when any one idle cell has
   a significant same-sign whole-cell slope; at a 95th-percentile surrogate test over eight idle cells
   the chance of one false refusal is about one in three even when no idle cell drifts. The alternative
   `"min_reps"` is on the CLI. `pass_frac = 0.5` (the fraction of a cell's passes that must fall) is a
   threshold the definition does not name.
9. **The synthetic corpus's duration** (M12) and **its size** (M13): the smoke run reads G2, G3, G-DEC
   and G-M at their minima by construction unless the corpus has 8 reps, 12 kernels and a `dt_est_s`
   inside the bracket.
10. **Which score after a B1-G3 quarantine**: FIX_2 applied SPEC 3.7.2's reading (the re-run) everywhere;
    re-verified on run 2 (`gl.csv` score_norm 0.583 = Table 7's APF LOKO accuracy after the void is
    lifted, `gdim.csv` d = 6, `predictions_with_quarantine.csv` beside `predictions.csv` in every
    LOKO/archetype directory). Unchanged unless the author prefers the full model.
11. **The remaining exposed defaults** stand as SPEC section 8 and CHECK_2 "For the author" item 8 list
    them; every value used is in the `params` block of the file that used it (237 `params` blocks in
    run 2's `report/manifest.json`).

---

## Checks that passed

- Every gate against its definition (record (3)): threshold values, verdict strings, refusal strings and
  citations all match P2 Sec. V 5.1 and 5.2 and CR section 2 items 1 to 11, 20 to 27, 31 to 35, with the
  four documented amendments from the SPEC reviews (G-C's detection ratio 1.5 with the prediction 2.0
  recorded, G-C's aliased-regime not-applicable verdict, G3's order-shuffle null and half-tail search,
  G2's pair-unit column). No undeclared threshold was found beyond `pass_frac` (M11, already in
  `params`).
- The verbatim copies: `stationarity_per_window`, `features`, `_pct`, `fold_within_trace`, `fold_loro`,
  `fold_loko`, `_assert_grouped`, `g4_pass`, `open_text` carry the `# copied from` comment and the
  equality tests against the originals pass.
- The streaming rule: `extract.py:252-351` holds two snapshots and the row buffer of the snapshot being
  read, reads `.csv.zst` through `zstd -dc -q`, then the `zstandard` module, then gzip, then plain text;
  14 cells of 120 pairs extracted in 4 s under `--jobs 4`.
- No server path in any `.py` file (`grep -n "/mnt/nfs\|/project/" *.py tests/*.py` returns nothing);
  the runbook names the retention root once as the path the author passes.
- The cycle-2 fixes hold in situ: B1 (one score after a quarantine, see item 10 above), M2 (the decay
  figure compares the verdict's head), M5 (the three runbook lines replaced by `run_moves plan`), M6
  (pytest required, stated in three places), M9 (`--seed-offset` reaches `gdim`), M11 (one-label G-X
  writes the not-applicable string in both columns; verified on both runs), M12 (no duplicated lead in
  Table 5's refusal cell), M13 (the two Table 2 math cells), M14 (the three helper citations), M15 (G-X's
  majority in the campaign space), al-Farabi 7.1 (the preconditions re-run by hand at move 2 made every
  later move run; `gp` re-ran on the edited pass table with `stale: pass_table.csv changed`) and 7.2
  (`gord`'s inputs a superset of `grid`'s).
- `gates/gf_check.json` (move 13) reads `pass` on run 2: every Table 7 row carries its G-F (i) verdict.
- `report/manifest.json`: 509 files with sha256 and size, 237 `params` blocks, the ledger.

---

## Records

### (1) The test suite, verbatim

Command, from `plan11_encoding_ladder/`: `python3 -m pytest -q -rs tests`

```
.............................s.......................................... [ 45%]
........................................................................ [ 90%]
...............                                                          [100%]
=========================== short test summary info ============================
SKIPPED [1] tests/test_extract.py:515: zstandard module not installed
158 passed, 1 skipped in 255.18s (0:04:15)
exit=0
```

### (2) The driver end to end

Corpus: builder 1's presets (`synth.corpus_specs(n_pairs=120, reps=3, idle=2)`) filtered to gemm (K0
4,096, pulse period 24, pulse extra 4,096, double), floyd (2,048, period 10, extra 2,048, decay), gibbs
(256, spin, no pulse), histogram (2,048, counter) and the two idle cells (floor 150, `label = idle`,
test label `kernel_sleep_v2`), written by `synth.write_cell` with `compress=True` (the `zstd` binary):
14 cells under `<scratchpad>/check3/root/kernel/...`, rep 0 at seed 42, reps 1 and 2 at 1000 r + the
kernel's index (AA A4).

**(2a) Run 1, the runbook's command as written.** `run_moves run --out <out> --root <root> --null-perm 5
--n-jobs 4 --assume-failed-zero --assume-reason "checker cycle 3 run"` (RUNBOOK.md:84-86 with the smoke
run's permutation count). Exit 0 after 2 min 01 s; 78 ledger entries, all `done`; `gf_check` `pass`.
What it wrote: `cells.csv` (14 rows, `status = ok`, roles and predicted archetypes right, `rep 0` = seed
42); `extract/<cell>/extract.csv` (58 columns, 120 rows) and `sidecar.json` (`n_pairs 120`, `header_ncols
66`, `n_rows_skipped 0`, `dt_est_s 5.0`, `status ok`) for every cell; `gates/preconditions.csv` with 5 of
14 `all_hard_pass` (M1: every floyd, gibbs and histogram cell `C1 = fail`; the idle cells `not
applicable: control (C1 re-mapped)`); the four `inputs/` templates; `gp.csv` (gemm `undeclared`);
`gc.csv` (`apf`, `persist`, `wapf` `pass` on gemm's three reps, 4 events each at seq 25, 49, 73, 97,
`stat_a` 1.98, `j_at_event` 0.48 to 0.49; `content` and `combined` `not run: no admissible cell for
gibbs,histogram`); `gk0.csv` (gemm `above floor`, idle band edge 150); `gf.csv` every row `not run: no
admissible idle cell; admissibility record missing`; the 13 grid points, `g3_flags.csv`, `gord.json`,
`table5_*`, `selection.json` (every rung `passes_acceptance true` at W8_H4 or W16_H8, because no pass
period was declared and G2 was not applicable), `alias.csv`; `gates/splits/<rung>/...` for the five
rungs (raw and norm for APF); `gl.csv`, `gn.csv`, `gx.*` (`not applicable: one campaign label` in both
columns), `gj.*`, `gdec.csv` (`not run: no admissible floyd cell`), `gdim.csv`, `gm.csv`, `gv.*`,
`clustering.*`; `report/tables/*` (ten tables as csv, md, tex, json), `report/figures/*` (eight figures
as pdf and png, the decay figure a placeholder), `report/paper2_skeleton.tex`, `report/manifest.json`,
`driver_state.json`. Table 6 on this run: `--` in the score cells of the three excluded kernels and `n =
12` on the `all` row (M8); Table 7's LORO and within-trace rows `not run: every feature quarantined`
(one kernel, three cells: every one-feature model reproduces a constant label).

**(2b) Run 2, every gate on its real path.** `run --moves 0-2` through the driver; then by hand
`gates_precondition preconditions --c1-activity-min 0.001 --assume-failed-zero --assume-reason ...` (14 of
14 admissible); `inputs/pass_table.csv` edited (gemm 5, floyd 12 passes per 600 s, `declared: synth
spec`); `inputs/idle_admissibility.json` filled; then `run --moves 2-13 --null-perm 5 --n-jobs 4
--assume-failed-zero --assume-reason ...`. Exit 0 after 45 min 08 s (3,196 s user); 76 ledger entries
(71 `done`, 4 `kept: author input exists`, 1 `skipped` = the preconditions, whose declared inputs had
not changed, so the by-hand run was kept); `gp` re-ran with `stale: pass_table.csv changed since move
2`; `gf_check` `pass`. Where the time went: G-ORD 327, 324, 641, 318, 643 s (apf, persist, content, wapf,
combined); the split stages 81, 56, 86, 50, 96 s; G-F at the selected points 13 s; `gdim` 23 s; `gm` 6 s;
`figures (all)` 7 s; everything else under 2 s each. Results: `gc.csv` every rung `pass` (content orderings hold in 3 of 3
rep pairs: `mean_abs` 0.0012 < 0.0120 < 10.68, `l0` histogram 2 < gibbs 5 < gemm 512 in the extracts);
`gk0.csv` every kernel `above floor` (tail medians 4,247, 2,211, 406, 2,213 against an idle edge of 150);
`gp.csv` gemm and floyd `resolvable` / `admitted` / `admitted` (T_pairs 24 and 10), gibbs and histogram
`undeclared`; `gf.csv` part (i) at W8_H4: apf and persist `void` (M3), wapf, content, combined
`inseparable at floor`; part (ii) every kernel `pass` except histogram under content `at floor in this
lead` (a counter's `l0` of 1 to 2 bytes per page equals the idle preset's; a finding, as defined); the
three floors in `gf_floors.json`; `table5_grid.csv`: G1 `pass` everywhere, G2 seconds `fail` everywhere
and `G2_pairs` `pass` at W = 64 (M12), G4 and G5 as defined, G-ORD `resolution` for APF at W 8 to 32 and
`order-blind` at 64, `order-blind` for persistence everywhere, `order-blind (by construction)` at the
whole-cell point; every selection `best-feasible` at `2 of 3`; `g3_flags.csv`: gemm's cells `present` at
peak 24 (the true period), floyd's at 20 (a rahmonic of 10), gibbs's and histogram's `absent`, every
kernel flag `absent` (M13); `gj.csv`: gibbs `floor overlap` (K 406 below 3 x 150), the rest
`interpretable`, `frac_below_null 0`; `gdec.csv` (M11); `gl.csv` part (i) `not run: 5 permutations <
500` for APF with score_norm 0.583, `no selection` for the other four at move 7 and filled at move 12,
part (ii) `pass` (r2 0.21, 4 points); `gn.csv` WORKING-SET (3) `headline`, SCATTER (1) `structural
novelty`; `gx.csv` `not applicable: one campaign label` in both columns for every rung; `gdim.csv` apf d
6, wapf 3, persist 7, content 28 (`declared reduction`), combined 40, `combined (matched)` matched to apf
at 6; `gm.csv` every ordered pair `difference with margin` (spread 0 over five seeds; at most 2 of 4
kernels improving); `gv_summary.csv` every rung `estimable`; `clustering.csv` k = 2, ARI 0.195 = its own
null p95, `exceeds false`; Table 7 as M3, M4 and M5 describe; Table 8: WORKING-SET (3) -> 2 (gemm,
floyd) under WORKING-SET and 1 (gibbs) under SCATTER, SCATTER (1) -> 1 (histogram) under WORKING-SET,
clusters `c0:6 c1:3` and `c1:3`; the eight figures written, the decay figure a placeholder carrying `G-DEC
(floyd): no decay`.

**(2c) Resume probe.** `run --moves 2-13 --dry-run` on run 2's output with nothing changed: 58 steps
`skipped: outputs exist and inputs unchanged`, 4 templates `kept`, and 12 steps `dry-run (stale: ...)`,
listed in M2.

### (3) Every gate against its definition

| Gate | Where | Definition | Threshold, verdicts, refusal, citation as read in the code |
|---|---|---|---|
| C1-C8 re-mapped | `gates_precondition.py:104-202` | P2 V 5.1 Plan 02; CR 2.1 item 1 | C1 `apf_max >= 0.02` for kernels, the not-applicable string for idle (never `fail`); C2 `n_pairs >= 8`; C3 informational `> 3` windows; C4, C5, C8 fixed strings; C6 status, 66 columns, modal header, `n_rows_skipped 0`, gaps reported not failing; C7 pending then pass/fail from `passes_acceptance`; `all_hard_pass` as SPEC 3.3.1; cited |
| `failed/` count | `:74-88` | P2 V 5.2; CR 2.1 item 2; AA A5 | 0 pass; > 0 `refused: failed count n > 0, seq axis uncorrected`; null `refused: failed count not recorded` unless `--assume-failed-zero --assume-reason`; pair rungs exclude refused cells; cited |
| G-K0 | `:235-294` | P2 V 5.2 G-K0; CR 2.2 item 20 | median K over the last 80 percent against the pooled idle 95th percentile; `IDLE, measured` / `above floor`; `not run: no admissible idle cell`; cited |
| G-F | `:356-462` | P2 V 5.2 G-F and the tripwire clause; CR 2.2 item 21, 2.1 item 15 | admissibility record first (`not run` without it); part (i) strict exceedance of the null p95 -> `void: idle reps separable under this rung` else `inseparable at floor` (M3 on the permutation floor); part (ii) envelope of idle medians -> `at floor in this lead`; three floors written; cited |
| G-P | `gates_calibration.py:106-186` | P2 V 5.2 G-P; CR 2.2 item 23 | T_pairs >= 4 `resolvable`, 2 to 4 `marginal`, < 2 `aliased by design at this size`, none `undeclared`; passes >= 5 for rhythm, T_pairs >= 3 within pass; both dt; `(INFERRED)` suffix; cited |
| G-C | `:199-375` | P2 V 5.2 G-C; CR 2.2 item 22; al-Kindi review 1, 2 | gemm K jump detected at 1.5 with 2.0 recorded, J dip <= 0.75 within 1 pair, every rep; content orderings 8 of 8 by rep index; `disconnected lead` refuses the rung; the aliased-regime not-applicable verdict when no rep shows the event; combined from the four rows; cited |
| alias falsifier | `:380-399`, `:441-470` | P2 Sec. 2 falsifier (2); CR 2.2 item 23 | OLS of the per-cell feature on `dt_est_s`, r2 > 0.5 -> `moves with the interval`; run on G3 peaks and on Table 6's separating features; cited |
| G1 | `gates_temporal.py:69-167` | P2 V 5.1 Plan 03; CR 2.1 item 3 | verbatim `stationarity_per_window` (z 1.0), floor 0.80 and not below the surrogates' 5th percentile (200 phase-randomized); `trend present` when more than half the cells drift beyond 1.0 sd; cited |
| G2 | `:170-200` | CR 2.1 item 4; al-Kindi review 7 | `W dt / T` at 0.500 and 0.644, floor 2.0; `undeclared`; `not applicable, rhythm above Nyquist` when T < 2 dt; `undetermined by the interval calibration` when the two disagree; pairs column beside; cited |
| G3 flag | `:203-265`, `:368-423` | P2 V 5.1 and Sec. 6 item 7 (a); CR 2.1 item 5; al-Kindi review 3 | cepstral peak over [n/8, n/2], order-shuffle null p95 (strict), 7 of the kernel's cells; CV reported beside `1/sqrt(K_median)` with no verdict; off the grid decision; cited |
| G4, G5 | `:268-270`, `:340-341` | CR 2.1 items 6, 7 | `2H <= W`; `n_windows_min >= 5` reported, not in the rule; Delta-5 `not applied` in params; cited |
| G-ORD | `:426-512` | P2 V 5.1; CR 2.2 item 27 | per W at H = W/2, LOKO/archetype, 20 order shuffles, 100 label shuffles for the spread; `order-blind` if the ordered score is within p95 - p05 of the shuffled mean, else `resolution`; whole cell `order-blind (by construction)`; refuses nothing; cited |
| roll-up and selection | `:518-650` | P2 V (the binding condition); SPEC 3.5.7 | every point kept, all-kernels roll-up with not-applicable kernels out, smallest W passing G1, G2, G4 with hop nearest 0.5, else `best-feasible` with `passes_acceptance false`; `grid_complete.json`; cited (M7 on the G1 not-applicable case) |
| G-J | `gates_readings.py:44-117` | P2 V 5.2 G-J and Sec. 6 item 6; CR 2.2 item 31 | independence null and the idle J quantiles; mask `K > 3 x floor median K`; `interpretable` / `floor overlap` / `floor unmeasured`; no floor subtraction; cited |
| G-DEC | `:122-370` | P2 V 5.2 G-DEC; CR 2.2 item 34; al-Kindi review 4 | admission on G-P's within-pass verdict else `decay not resolved`; (a) `l0_q50_per`, `l1` second, hamming sign; (b) 3-snapshot fall at a common phase in 7 of 8 reps; (c) K drop not larger -> else `no decay beyond breadth`; (d) within-pass order surrogates, slope below p05 -> else `no decay`; (e) idle same-sign slope -> `no decay beyond floor or host`; `decay not resolved: no pass boundary`; cited (M11 on `pass_frac`) |
| B1-G1 | `models.py:215-223`, `:301-319` | P2 V 5.1 Plan 08; CR 2.1 item 9 | unit-level shuffle (archetype labels across kernels; kernel labels across cells), strict exceedance of p95, rank reported, `near_unfalsifiable` excluded from every table, `not run: N permutations < 500`; cited |
| B1-G3 | `:322-338` | CR 2.1 item 10 | one-feature tree per feature on the training fold, quarantine when it disagrees with the full model on at most 1 unit, re-run without it, both scores kept, the re-run is the rung's score; cited |
| B1-G6 | `:190-212` | CR 2.1 item 11 | most populous archetype by kernel count among training kernels under LOKO; by cell count otherwise; cited |
| G-L | `gates_comparison.py:63-144` | P2 V 5.2 G-L; CR 2.2 item 24 | (i) normalized LOKO score > null p95 else `level only`; (ii) OLS of within-window CV on `1/sqrt(K_median)` over kernels, r2 > 0.5 -> `refused: shot noise explains CV`; cited |
| G-N | `:149-184` | CR 2.2 item 25 | >= 3 `headline`, 2 `one training kernel per fold`, 1 `structural novelty`, 0 `no kernel row`; macro recall over headline rows only; cited |
| G-X | `:189-261` | CR 2.2 item 26; K2 Sec. 4 item 3 | campaign label under LOKO, strict exceedance -> `campaign predictable` else `pooling stands`; confound none/partial/total; `refused: campaign leak with total confound` on the headline; one label -> not applicable in both columns; cited |
| G-DIM | `:266-307` | CR 2.2 item 32 | feature count per row; matched comparison at the strongest single rung's d by train-fold importance; reduction to `n_train_cells` when d exceeds it (`declared reduction` / `full vector`); cited (M4, M5) |
| G-M | `:312-368` | CR 2.2 item 33 | spread = max - min of the APF LOKO score over 5 forest seeds; `beats` iff diff > spread and (>= 6 up, 0 down or >= 7 up, <= 1 down) else `difference with margin`; cited |
| G-V | `variance.py:23-82` | CR 2.3 item 35 | population L0, L2, L3 on normalized per-cell means; `LOKO not estimable` iff L0 > L3 for every non-constant feature; cited |
| clustering | `models.py:553-633` | P2 V 'Models'; SPEC 4.4 | k = archetypes present after G-K0, KMeans primary, GMM and ward beside, ARI and NMI against 500 archetype-label permutations; cited |

### (4) Interfaces between the three builders

Read from run 1's and run 2's output directories. `extract/<cell>/extract.csv`: 58 columns in
`schema.EXTRACT_COLUMNS` order (SPEC 2.2). `sidecar.json`: every SPEC 2.3 key plus `rep_source`,
`source_sha256` and the `schema`/`params`/`citation` triple. `cells.csv`: the 12 SPEC 2.7 columns.
`gates/preconditions.csv`: the SPEC 3.3 columns plus `failed_source`. `gk0.csv`, `gf.csv`, `gc.csv`,
`gp.csv`, `alias.csv`, `g3_flags.csv`, `gj.csv` (plus the two `mask_persist` columns), `gdec.csv`,
`gl.csv` (plus `grid_id`), `gn.csv`, `gx.csv` (plus `grid_id`, `headline_mark`, `n_labels`), `gdim.csv`
(plus `loko_score`, `matched_to`), `gm.csv`, `gv.csv` (plus `constant`), `gv_summary.csv`,
`clustering.csv` (plus `primary`, `grid_id`, `status`), `table5_grid.csv` (plus the pairs columns and the
G-C/G-F columns), `temporal_per_kernel.csv`, `predictions.csv`: as SPEC sections 3.3 to 3.8, 4.4, 4.5,
with the documented additions. `scores.json`: `accuracy`, `macro_recall`, `recall_per_class`,
`recall_per_kernel`, `majority`, `b1_g1`, `b1_g1_rank`, `null_p95`, `with_quarantine`, `feature_count`,
`dim_status`, `seed`, `n_perm`, plus `score_source`, `predictions_file`, `excluded_row`,
`headline_classes`, `null_summary`, and the `params` block with the 29 keys SPEC 4.5 and the reviews
name. `_report_common.effective_scores` delegates to `models.effective_scores` and its fallback copy is
identical (the cycle-2 test). Builder 3 reads every file by column name and recomputes no verdict.
The one disagreement is M4 (the matched row's splits).

### (5) The runbook

All 64 `python3 -m plan11_encoding_ladder.<module> ...` lines of RUNBOOK.md (with `<out>` and `<root>`
substituted, the move-3 loop expanded once) were parsed against each module's argparse without
execution: 64 parsed, 0 rejected. The same subcommands with the same flags (other than the permutation
counts) ran under the driver in runs 1 and 2. `run_moves plan --out <out> --moves 10` prints the four
`gates_temporal` lines the M5 replacement refers to; `run_moves status` prints the ledger; `run_moves
plan --moves 6-7 --root` prints the commands. Not run as written: the `pip install` line (nothing was
installed), `pdflatex` (not on this machine, as in the two earlier cycles). The claims checked and true:
`unittest` sees 96 of 159 tests (FIX_2's count; the suite has 159); the `zstd` binary is read first; the
smoke run's B1-G1 cells read `not run: 20 permutations < 500` by design (they read `not run: 5
permutations < 500` here). The claims not borne out: the 0b heading's "minutes" (M6) and "every not
run: cell named" (M3).

### (6) The skeleton

`apf_paper/p2_skeleton.tex` (432 lines, the FIX_2 regeneration) and run 2's
`report/paper2_skeleton.tex` (432 lines) were read whole. Outside comments there are only
`\documentclass`, packages, `\title{}`, `\author{}`, the section and subsection headings of P2 Sec. 3
(Abstract, I to X, Artifact and reproducibility), `\IfFileExists{tables/...}{\input{...}}{...}` shells
with empty `\caption{}` and the header rows and P2's static cells (Table 2, Table 3, the Table 4 shell,
Table 8's row and column labels), and eight `figure` environments with `\includegraphics` or a framed
placeholder and empty captions. Every bullet of substance is a `%` comment. The two differ only in the
timestamp comment and Table 3's nbody pass-period cell, which run 2's copy fills from the pass table.
No sentence of paper body exists in either file.

---

Files read: every `.py` under `plan11_encoding_ladder/` and `tests/`, `SPEC.md`, `RUNBOOK.md`,
`requirements.txt`, `CHECK_2.md`, `FIX_2.md`, `CERTIFY_al_farabi.md`, `SPEC_review_al_kindi.md`;
`apf_paper/P2_STRUCTURE.md`, `apf_paper/P2_AUTHOR_ANSWERS.md`, `apf_paper/council/12_P2_COUNCIL_REPORT.md`
section 2, `apf_paper/council/10_al_kindi_revised.md` sections 2 and 5, `apf_paper/p2_skeleton.tex`.
This file:
`/Users/jeries/Desktop/projects/thesis/memorySignal/mem_sig/VM_sampler/VM_Capture_QEMU/plan11_encoding_ladder/CHECK_3.md`.
Scratch artifacts of the runs are under
`/private/tmp/claude-501/-Users-jeries-Desktop-projects-thesis-memorySignal-mem-sig/14810c8c-2535-466d-a296-d2aae9739c16/scratchpad/check3/`
(`root/` the corpus, `out/` run 1, `out2/` run 2, `pytest_output.txt`, `driver_run1.log`,
`driver_run2.log`).
