# Validation Semantics — Cosine-Only Offline Path (Opus Spec)

## 1. Executive Summary
Treat the finding as **primarily a documentation + acceptance-criteria issue**, not an implementation issue. The offline-step path is single-channel by design (driven by `deltaMetric=cosine`). Combined hamming+cosine analysis already exists elsewhere in the repo and is a distinct path. Resolve ambiguity by (a) restating what current offline-step results prove, (b) tightening acceptance-criteria wording to name the channel explicitly, and (c) adding a short README/docstring note in the offline-step entrypoints. Defer combined-channel offline-step support.

## 2. Core Interpretation Invariants
- Offline-step outputs reflect **only the selected `deltaMetric` channel** (currently cosine).
- Hamming `.txt` files are persisted but are **not** ingested into the offline-step matrix.
- Combined hamming+cosine complex representation is a **separate downstream path**, not the offline-step path.
- Any clustering/classification claim must name the channel used.
- No metric, plot, or conclusion may aggregate "offline-step" results with "combined-channel" results without an explicit channel qualifier.

## 3. Scope Decision
- **Documentation caveat?** Yes — required.
- **Acceptance-criteria change?** Yes — wording must name channel (`cosine-only`) explicitly.
- **Implementation required now?** No. Combined-channel offline-step support is **deferred**.

Justification: active code already behaves correctly for the single-channel contract. Ambiguity is purely semantic. Minimal-churn fix = wording + small inline notes. Redesigning the offline-step matrix to carry both channels is out of scope and would expand storage, metric math, and validator semantics unnecessarily.

## 4. Proposed Change Set
- `VM_sampler/VM_Capture_QEMU/offline_step_metrics.py` -> add module docstring note stating single-channel contract and that channel is taken from `deltaMetric`.
- `VM_sampler/VM_Capture_QEMU/capture_consumer_qemu.sh` -> add comment above the frame-selection block making channel selection explicit.
- `VM_sampler/VM_Capture_QEMU/config_qemu_upc.json` -> no code change; add a short comment-style sibling doc entry (below).
- `docs/controlled-qemu-pipeline/03-consumer-and-run-matrix.md` -> add "Channel Semantics" subsection: matrix is single-channel, driven by `deltaMetric`.
- `docs/controlled-qemu-pipeline/05-offline-metrics.md` -> add "Validation Scope" subsection stating offline-step results validate cosine channel only.
- `docs/controlled-qemu-pipeline/07-ambiguities-and-out-of-scope.md` -> add explicit entry: combined hamming+cosine offline-step is deferred.
- `docs/controlled-qemu-pipeline/issue/plans/validation-semantics-opus-spec.md` -> this spec (acceptance anchor).
- **No changes** to `wavelet_analysis_features.py` or `streaming_metrics.py`.

## 5. Implementation Plan (Ordered)
1. **Clarify offline-step channel contract** — Required now
   - Objective: make single-channel nature explicit at entrypoint.
   - Change: add 3–5 line docstring block at top of `offline_step_metrics.py` stating: matrix is derived from one `deltaMetric` channel; hamming not ingested; combined-channel analysis is a separate path.
   - Affected: `offline_step_metrics.py`.

2. **Annotate consumer frame selection** — Required now
   - Objective: make `deltaMetric`-driven selection explicit in the consumer.
   - Change: 2-line comment above the `subdir="cosine" ... [[ "$deltaMetric" == "hamming" ]] && subdir="hamming"` block: "Single-channel run_matrix: only the selected deltaMetric is appended. Hamming files remain on disk but are not part of the offline-step matrix."
   - Affected: `capture_consumer_qemu.sh`.

3. **Update booklet: consumer/run-matrix** — Required now
   - Objective: document channel semantics in the active booklet.
   - Change: add subsection "Channel Semantics" in `03-consumer-and-run-matrix.md`: run matrix is 1-channel; channel selected by `deltaMetric`; hamming persisted but excluded.
   - Affected: `docs/controlled-qemu-pipeline/03-consumer-and-run-matrix.md`.

4. **Update booklet: offline metrics** — Required now
   - Objective: bound validation claims.
   - Change: add subsection "Validation Scope" in `05-offline-metrics.md`: clustering/classification outputs from this path validate separability on the selected channel only; combined-channel claims require the downstream path.
   - Affected: `docs/controlled-qemu-pipeline/05-offline-metrics.md`.

5. **Update booklet: ambiguities** — Required now
   - Objective: record deferred scope.
   - Change: append entry "Combined hamming+cosine offline-step path" under Ambiguities: not implemented; downstream combined-channel analysis (`VMsig_featureExctraction/`) is the valid path for that claim.
   - Affected: `docs/controlled-qemu-pipeline/07-ambiguities-and-out-of-scope.md`.

6. **Audit reporting wording** — Required now
   - Objective: prevent overclaiming in existing reports.
   - Change: grep reports/plots/README strings for phrases like "hamming+cosine", "combined delta", "complex representation" co-occurring with "offline" / "step matrix"; where found, rewrite to "cosine-only offline-step" or move claim to the downstream-path section.
   - Affected: any doc/report artifact found; no code paths.

7. **Combined-channel offline-step support** — Deferred
   - Objective (future): ingest both channels into a 2-plane or complex step matrix.
   - Not implemented now. Tracked as open item.

## 6. Interpretation / Validation Analysis
- **Cosine-only offline-step outputs prove:** step classes are separable via PLV / MSC / cepstrum computed over the cosine delta channel across time.
- **They do not prove:** separability under a combined hamming+cosine representation; robustness when cosine signal is degraded; equivalence to the downstream combined-data path's feature space.
- **Difference from downstream combined path:** downstream path (`VMsig_featureExctraction/wavelet_analysis_features.py` etc.) constructs a joint representation from both delta streams; its results are not interchangeable with offline-step results and must be cited separately.
- **Wording rule:** always qualify as "cosine-channel offline-step" in figures, abstracts, and acceptance statements. Never write "offline-step validates the combined representation."

## 7. Test / Validation Plan
- **Acceptance-criteria wording checks**
  - Grep docs, plan, and reports for unqualified "offline-step" paired with "combined" / "hamming+cosine" / "complex" — must be zero hits after update.
  - Each offline-step claim must contain the token `cosine-only` or `cosine-channel`.
- **Regression checks (behavior)**
  - No code path changes: run one controller step; confirm `offline/<step>/` artifacts (`meta.json`, `streaming.*`, `plv_baseline_aware.json`) byte-equivalent (aside from timestamps) to pre-change run.
- **Documentation checks**
  - Booklet files (`03`, `05`, `07`) contain new subsections verbatim.
  - `offline_step_metrics.py` module docstring present and mentions `deltaMetric`.
- **Implementation tests for combined-channel (deferred)**
  - Not executed this cycle. Placeholder note added to deferred item.

## 8. Acceptance Criteria
- Offline-step channel contract stated in code docstring (`offline_step_metrics.py`) and inline consumer comment (`capture_consumer_qemu.sh`).
- Booklet sections 03, 05, 07 updated per Plan steps 3–5.
- No surviving text in the repo asserts that offline-step results validate a combined hamming+cosine representation.
- No behavioral/code changes to metric computation, producer, consumer queue logic, or storage layout.
- Combined-channel offline-step explicitly marked deferred in `07-ambiguities-and-out-of-scope.md`.

## 9. Rollout / Rollback
- **Rollout:** single docs+comments commit. No service restart. No data migration.
- **Verify:** run one controlled step end-to-end; confirm artifacts unchanged.
- **Rollback:** `git revert` the commit. Zero runtime impact either way.

## 10. Executor Handoff Prompt
```
Implement ONLY the changes in docs/controlled-qemu-pipeline/issue/plans/validation-semantics-opus-spec.md.

Scope: documentation + inline comments only. No behavioral code changes. Do not modify metric math, queue logic, producer, storage, or run_matrix shape.

Tasks:
1. Add a 3–5 line module docstring to VM_sampler/VM_Capture_QEMU/offline_step_metrics.py stating: the offline-step matrix is single-channel, channel is determined by config `deltaMetric` (currently cosine); hamming outputs are persisted but not ingested here; combined hamming+cosine analysis is a separate downstream path.
2. In VM_sampler/VM_Capture_QEMU/capture_consumer_qemu.sh, add a 2-line comment directly above the block that sets `subdir="cosine"` / switches to `hamming`, stating the run_matrix is single-channel and the non-selected channel is not ingested.
3. In docs/controlled-qemu-pipeline/03-consumer-and-run-matrix.md, append a "Channel Semantics" subsection: run matrix holds one channel chosen by `deltaMetric`; hamming files persist on disk but are excluded from the matrix.
4. In docs/controlled-qemu-pipeline/05-offline-metrics.md, append a "Validation Scope" subsection: offline-step metrics validate separability on the selected channel only (currently cosine); combined-channel claims require the downstream path in VMsig_featureExctraction/.
5. In docs/controlled-qemu-pipeline/07-ambiguities-and-out-of-scope.md, append an entry "Combined hamming+cosine offline-step path": deferred; not supported by current offline_step_metrics.py; combined analysis lives in VMsig_featureExctraction/.
6. Grep the repo for text that pairs "offline-step" (or offline_step_metrics) with "combined", "hamming+cosine", "complex representation". Where found, rewrite to "cosine-only offline-step" or relocate the claim to the downstream-path context. Report every edit.

Do NOT:
- Change deltaMetric, config values, or any code that produces or consumes run_matrix.
- Add combined-channel support to offline_step_metrics.py.
- Touch wavelet_analysis_features.py or streaming_metrics.py.
- Modify queue, producer, or consumer logic beyond the single comment in step 2.

Deliverables: one commit containing the above edits. Confirm a controlled capture step still produces the same artifacts as before (no behavior change expected).
```
