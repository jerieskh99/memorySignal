# Prompt 3: Validation Semantics / Cosine-Only Offline Path

```text
You are acting as a senior systems architect and implementation planner.

Important context:
Initial repo exploration has already been completed collaboratively. Do NOT spend effort rediscovering the codebase unless a missing fact is truly blocking a safe plan. Assume the findings below are materially correct unless they contradict each other.

Your job is to convert the findings into a concrete, implementation-grade specification that another coding agent (Cursor / Codex / Sonnet) can execute with minimal ambiguity.

Your priorities, in order:
1. Correct interpretation of results
2. Minimal code churn
3. Scope clarity
4. Maintainability
5. Performance

You must be decisive. Do not present multiple equal-weight options unless a real unresolved blocker forces it. Prefer the smallest safe change that resolves ambiguity.

==================================================
PROJECT CONTEXT

This repository computes page-wise delta outputs from RAM snapshots and then performs downstream analysis.
Different analysis paths exist in the repo, and evaluation semantics depend on which data path is actually used.

==================================================
TECH STACK

- Bash
- Python
- NumPy
- QEMU capture pipeline
- step matrices
- offline metric scripts
- downstream combined-data feature extraction

==================================================
PLANNING SCOPE

This planning task is intentionally scoped to validation semantics and acceptance-criteria clarity.

Out of scope unless strictly required:
- Rust optimization
- queue deletion coordination
- producer redesign
- run-matrix storage redesign
- full pipeline redesign

==================================================
CURRENT OBJECTIVE

Resolve the implications of the following finding:

- `capture_consumer_qemu.sh` builds the step matrix from one selected delta stream (`deltaMetric`), not both
- in active config (`config_qemu_upc.json`), `deltaMetric=cosine`
- therefore `offline_step_metrics.py` currently computes PLV / MSC / cepstrum on cosine-only step matrices
- hamming outputs are still generated to files, but are not consumed by the offline-step matrix path
- downstream scripts elsewhere in the repo can combine hamming + cosine into a complex representation, but that is a separate path

Implication:
Current offline-step clustering/classification results demonstrate separability from the cosine channel alone.
They do NOT yet validate the combined hamming+cosine representation in the offline-step path.

You must decide whether this is:
1. only a documentation/interpretation constraint,
2. a blocker for acceptance criteria,
3. or a required implementation task now.

==================================================
KNOWN RELEVANT FILES

- `VM_sampler/VM_Capture_QEMU/capture_consumer_qemu.sh`
- `VM_sampler/VM_Capture_QEMU/offline_step_metrics.py`
- `VM_sampler/VM_Capture_QEMU/config_qemu_upc.json`
- `VMsig_featureExctraction/wavelet_analysis_features.py`
- `coherence_temp_spec_stability/streaming_metrics.py`

==================================================
KNOWN CONSTRAINTS

- Avoid overstating what current metrics prove
- Prefer the smallest change that restores semantic clarity
- Do not force a major redesign unless truly needed
- Acceptance criteria must match actual data paths

==================================================
NON-GOALS

- Do not optimize runtime in this task
- Do not redesign the full pipeline
- Do not assume combined-channel support exists in the offline-step path unless explicitly added
- Do not broaden into unrelated capture changes

==================================================
REPO FINDINGS

- Offline-step metrics currently consume a single-channel step matrix
- Active config selects cosine
- Hamming exists but is excluded from the offline-step matrix in current configuration
- Combined hamming+cosine analysis exists elsewhere in the repo as a separate downstream path
- This affects what current clustering/classification results actually validate

==================================================
YOUR TASK

Produce a concrete implementation/specification document that resolves this validation ambiguity.

You must:
- decide whether this is only a documentation issue, an acceptance-criteria issue, or an implementation issue
- define clearly what the current offline-step outputs do and do not prove
- state whether combined-channel offline-step support is in scope now or deferred
- identify exact files likely to change if any changes are needed
- provide a concise executor handoff prompt for a coding agent

==================================================
OUTPUT FORMAT (MANDATORY)

## 1. Executive Summary
One paragraph describing the chosen approach.

## 2. Core Interpretation Invariants
List the validation and interpretation rules that must remain true.

## 3. Scope Decision
Explicitly answer:
- Is this only a documentation caveat?
- Does it change acceptance criteria?
- Is any implementation required now?
Justify the decision.

## 4. Proposed Change Set
List exact files/modules/components likely to change and why.

Format:
- path/or/module -> purpose of change

## 5. Implementation Plan (Ordered)
Provide numbered steps.

For each step include:
- objective
- exact logic or documentation change to add
- affected components
- whether it is required now or deferred

## 6. Interpretation / Validation Analysis
Be explicit about:
- what cosine-only offline-step outputs prove
- what they do not prove
- how this differs from the downstream combined-data path
- how future results should be worded to avoid overclaiming

## 7. Test / Validation Plan
Include:
- acceptance-criteria wording checks
- regression checks if behavior changes
- documentation or reporting checks
- any implementation tests if combined-channel support is brought in-scope

## 8. Acceptance Criteria
Write precise criteria for declaring this work complete.

## 9. Rollout / Rollback
How to deploy safely and revert if needed.

## 10. Executor Handoff Prompt
Write a concise prompt for a coding agent to implement ONLY this approved plan.

==================================================
STRICT RULES

- Be decisive.
- Prefer minimal clarification work over architectural expansion.
- Do not recommend rewrites unless unavoidable.
- Do not redo repo discovery unless a missing fact truly blocks safe planning.
- If assumptions are required, list them explicitly.
- Keep the plan implementation-focused, not essay-like.
- Another model will code directly from your output.

Return only the structured spec.
```
