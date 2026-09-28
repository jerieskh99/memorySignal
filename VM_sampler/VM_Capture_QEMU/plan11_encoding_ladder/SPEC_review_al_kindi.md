# SPEC review, al-Kindi (2026-09-16)

Scope: the extract schema against the fourteen blind moves (K2 section 5), label and level
leakage into the feature path, the fused plane, the calibration pulses, and the move order in
the driver. Verified against code where the SPEC cites it: the differ header is 66 columns with
`hamming` at index 2, `l0` at 4, `l1` at 5, and `mean_abs = l1 / 4096` (`metrics/mod.rs`,
`family_a/positional.rs`); the consumer increments `seq` only after a successful differ and an
empty differ output leaves a numbered gap (`capture_consumer_qemu.sh` lines 367 to 388); the
cepstral tail starts at `n // 8` (`plan03_metric_kernel.py` line 145); the eight B1 features and
the `0.1 * max` duty rule are as stated. Two checks were simulated on synthetic series in the
scratch directory; no data and no server were touched.

## 1. Passes as specified

- The extract (SPEC 2.2) carries everything the moves read after move 1: `K`; sums and the five
  quantiles of `hamming`, `l0`, `l1` over all rows and over the persistent pages; the three
  ratios as quantiles on the persistent pages; `J` with the independence null in both forms;
  `n_persist` and `n_union`; and in the sidecar the pair count and `600 / n_pairs`. K2 move 1
  also kept the sorted `page_index` set; nothing in this paper reads it (the piano roll
  re-streams the trajectory), so its absence is right (section 3, item 3).
- Nothing in the feature path carries the label or the level. Extract rows carry no identity;
  identity lives in the sidecar and `cells.csv`. The feature matrix `X` excludes `cell_id`,
  `kernel`, `rep`, `campaign` and `n_series_cell`, which are side arrays. Normalization uses
  per-cell statistics only; `K` is not a channel of the content rung; the label shuffles are at
  the unit; imputer and scaler are fitted per training fold. `J - J_null` subtracts a
  level-dependent quantity of order 1e-3, declared and harmless.
- The fused plane is computable as specified: `r_l0_q50_per[t]` and `J[t]` sit on one row, both
  about the pair `(t, t+1)`, the persistent-side values taken at `t` under the declared
  `persist_side`. The G-J mask is a per-row boolean on the same axis.
- The streaming pass (SPEC 2.4) is right and matches the consumer: two snapshots in memory, a
  missing `seq` is a `K = 0` snapshot, a decreasing `seq` refuses, a failed differ job is
  invisible in the file and is the `failed/` count's business (move 2).
- The move order is preserved. Driver moves 0 to 13 match P2 section 5 one to one: G-C before
  any reading, the floor before any split, the temporal gate fixing (W, H) before Table 6, the
  fused plane at move 8, the tripwire re-run at move 13. Every grid point is computed and kept,
  the selection rule is declared, nothing is deleted from `gates/grid/`.
- G-C's content orderings are computable from `l1_q50_per / 4096` and `l0_q50_per`, and the
  summary statistic is declared (section 8 item 16). G-P is in pair units from each cell's own
  pair count. Every null runs at the unit of the split.

## 2. Must change before build

1. **SPEC 3.4.3, G-C, the K-jump threshold.** What is wrong: `K_t >= 2.0 * median(K)` uses the
   source prediction (two to one) as the detection threshold. The floor `F` sits in both levels,
   so the observed ratio is `(2K + F) / (K + F)`, below 2 always. On the SPEC's own synthetic
   gemm preset (`K0 = 4096`, `pulse_extra = 4096`, `floor_F = 150`, `k_noise = 0.02`) the ratio
   is 1.97 and the must-pass case of 5.3 refuses in eight of eight reps (simulated); under
   `--cv-case random` it is found by luck of the jitter, zero to three events per rep.
   Correction: separate prediction from detection exactly as the J dip already does (one half
   predicted, `j_dip_max = 0.75` the midpoint as detection): `jump_predicted = 2.0` recorded,
   `jump_detect_ratio = 1.5` (the midpoint between 1 and 2 on the ratio scale) as the event
   rule `K_t >= jump_detect_ratio * reference`. Write the observed per-rep maximum ratio into
   `gc.csv` as `stat_a` so the author sees how near two it lands. Section 8 item 16 lists both.

2. **SPEC 3.4.3, G-C, the absence verdict.** What is wrong: the pulse is absent for two reasons,
   a disconnected lead or a gemm pass that fits inside the interval (P2 Sec. IV rung 1
   "regime-dependent"; K2 rung 1 (d)), and the SPEC gives both one name, `GC_DISCONNECTED`. It
   would void APF and persistence on a healthy instrument if gemm's pass is faster than 0.64 s.
   gemm is undeclared under G-P, so the regime cannot be read from the pass table. Correction:
   when no rep shows the event, read the regime blind from the same extract with the same
   declared tolerance. If in every rep the cell's median `K` lies inside the midpoint band of
   the source-predicted full footprint (`pulse_full_footprint_pages = 4096`, Table 3 gemm row;
   `K_median / 4096` in `[1 / 1.5, 1.5]`) and `J_q50 >= 0.75`, the verdict is
   `not applicable: pulse aliased by design (full footprint lit every snapshot)`; the rung is
   neither passed nor voided and its G-C row says the level pulse alone was seen. Mixed reps
   stay `GC_DISCONNECTED` (the definition says every rep). The new constant goes to section 8;
   the docstring carries K2's INFERRED tag on the pulse shape.

3. **SPEC 3.5.3, G3, the surrogate null.** What is wrong: the cepstrum as copied is
   `irfft(log |rfft(x)|)`, a function of the amplitude spectrum alone, and the phase-randomized
   surrogate preserves that spectrum exactly (SPEC 3.2 says so). Therefore
   `snr_surrogate_p95 == snr_db` to machine precision (simulated spread 4e-14) and the strict
   exceedance is false for every cell of every kernel: the flag can never read "present". The
   SPEC saw this trap for CV and missed it for the cepstrum. Correction: G3's null is the
   order-shuffle surrogate (`nulls.order_shuffle`, the operator G-ORD already uses; the iid null
   that keeps the marginal distribution and destroys rhythm), `g3_null = "order_shuffle"`,
   alternative `"block_bootstrap"` with block length below the quefrency floor; section 8 item
   11 updated. Two consequences seen in the same simulation: (a) the search over the whole tail
   returns the mirror index `n - P` (the cepstrum of a real series is symmetric; at `n = 931`,
   `P = 24` it reports 907), so the search stops at `n // 2` or the period is reported as
   `min(q, n - q)`, otherwise alias falsifier (a) regresses a mirror on `dt`; (b) the must-pass
   case of 5.3 at 120 pairs flags five of eight cells under the shuffle null against the
   seven-of-eight rule; the G3 test cell needs 240 pairs (eight of eight there).

4. **SPEC 3.6.2 G-DEC with 5.2 and 5.3, the boundary and the synthetic case.** What is wrong:
   the floyd preset has `pulse_period = 10` and `pulse_extra = 0`, so `K` never jumps;
   `boundary_source = "k_jump"` finds no boundary and `"period"` is anchored "from the first
   jump", so the must-pass case is unreachable under either default; and the verdict for a cell
   with fewer than two boundaries is undefined (gibbs, the control, never has one). Correction:
   (a) `"period"` anchors at `seq_first + phase_pairs` with `phase_pairs = 0` declared in
   section 8, not at a jump; (b) a cell with fewer than two detected boundaries writes
   `GDEC_NOT_RESOLVED` with the reason `no pass boundary`; (c) the control, and the idle clause
   (e), are evaluated as a whole-cell slope against phase-randomized surrogates when no
   boundary exists, written in the docstring; (d) the synthetic floyd preset gets
   `pulse_extra = 2048` so the `k_jump` path is exercised, and the test names which boundary
   source it exercises.

5. **SPEC 3.4.4 and section 7 (moves 6 and 7), alias falsifier (b).** What is wrong: "any
   feature that separates a level-matched pair under Table 6" has no operational definition in
   the SPEC (3.7 does not give one), and the driver runs `alias` at move 6, before Table 6
   exists at move 7. Correction: define separation with the envelope rule G-F (ii) already
   uses: at APF's selected point, per cell the window mean of each normalized APF feature; a
   feature separates a level-matched pair when the two kernels' eight cell means have disjoint
   ranges (no threshold). Run the falsifier on every such feature per set, and run
   `gates_calibration.py alias` again at the end of move 7 (idempotent; `kind = table6_feature`
   rows appended).

6. **Section 7 (driver) and 3.4.2, G-P's place.** G-P needs only the pass table and `n_pairs`,
   both present at move 2, yet it runs at move 6; G-C (item 2) and G-DEC read it. Correction:
   `gates_calibration.py gp` runs at move 2 right after the pass-table template is filled; move
   6 keeps `alias`.

7. **SPEC 3.5.2, G2 in pair units.** The rhythm is declared in pair units (CR 2.1 item 4 last
   sentence; K2 rung 0 (b)); the SPEC computes coverage only in seconds at the two bracket ends.
   Correction: add per cell `coverage_pairs = W * passes / n_pairs` (the cell's own count, no
   `dt` anywhere), its median per kernel as `coverage_pairs` in `temporal_per_kernel.csv` and
   Table 5, with `G2_pairs` by the same floor 2.0. The two seconds columns stay as the bracket
   representation; at the median cell the 0.644 column equals this number.

8. **SPEC 3.7.6 and section 7 (moves 9 to 12), G-X per rung.** 3.7.6 says "at the selected
   point of each rung" and Table 7 marks a per-rung refusal, but the driver runs `gx` only for
   `apf`. Correction: "the same for <rung>" in moves 9 to 12 includes
   `gates_comparison.py gx --rung <rung>`; Table 6's G-X cell is APF's.

9. **SPEC 3.6.1 and 6.7, the fused plane's mask.** What is wrong: the G-J mask tests `K_t`
   (breadth) against three times the floor's median `K`, but the plane's x coordinate is a
   median over the persistent pages, and for a marching kernel the persistent pages are the
   floor's pages even when `K_t` is large (`n_persist` near `F` while `K_t` is ten `F`). A
   floor point would print as a kernel point. Correction: `gj_mask` writes two boolean columns,
   `mask_K` (the definition, applied by default) and `mask_persist`
   (`n_persist > k_factor * the idle cells' median n_persist`, the same `k_factor`); the figure
   applies `fused_plane_mask = "K"` by default and the alternative is listed in section 8.

## 3. For the author

1. G-C's pulse shape (two to one and one half) presumes the whole of C is rewritten inside every
   interval and A is re-seeded at the boundary; K2 marked this INFERRED from
   `kernel_gemm_v2.c` lines 142 to 143. If the pass outlasts the interval, K2 move 5 expects
   `K` alternating between about 4,096 and a few hundred with `J` near the floor mid-pass, so
   the jump exceeds two to one and the dip goes below one half; item 1's rule still fires. The
   600 s gemm run without capture (P2 section 6 item 5) settles the regime and makes item 2's
   blind branch unnecessary.
2. `head_drop.csv` is a per-kernel parameter (the lexer's first pass). Al-Farabi's condition (CR
   2.1 item 19) is that no parameter carries the label; a drop applied to one kernel changes
   that kernel's `K_median` and window count. Either one drop for every kernel, or the exception
   recorded in `params`. The default of 0 is safe.
3. The extract does not keep `S_t` (K2 move 1 did). Nothing in this paper needs it; the blocking
   paper's per-block counts will need a second streaming pass. Fine to leave.
4. G-F's floor of "`l0` per changed page" is pooled from per-snapshot quantiles (a quantile of
   quantiles), not from pages. Exact pooling needs a page-level pass over the idle cells only
   (eight cells, cheap). Choose whether the toolkit adds it.
5. G3's reported quefrency may be a rahmonic (2P or 3P) rather than P (seen in simulation); the
   alias falsifier tolerates it if the eight cells report the same multiple. `g3_peak_report`
   as a choice: `argmax` (default) or the smallest local maximum above the null.
6. G-C's `jump_reference`: when the pass period is between one and two pairs the boundary
   snapshots are not a minority and the cell median sits on the upper level; the lower quartile
   of `K` is the more robust reference. `cell_q25_K` as an alternative to list.
7. The synthetic corpus at 120 pairs is short for G3 (item 3b). If the smoke run must stay
   fast, lengthen only the G3 test cell.
