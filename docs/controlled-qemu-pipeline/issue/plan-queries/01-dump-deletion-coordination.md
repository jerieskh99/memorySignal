# Prompt 1: Dump Deletion Coordination

```text
You are acting as a senior systems architect and implementation planner.

Important context:
Initial repo exploration has already been completed collaboratively. Do NOT spend effort rediscovering the codebase unless a missing fact is truly blocking a safe plan. Assume the findings below are materially correct unless they contradict each other.

Your job is to convert the findings into a concrete, implementation-grade specification that another coding agent (Cursor / Codex / Sonnet) can execute with minimal ambiguity.

Your priorities, in order:
1. Correctness
2. Concurrency safety
3. Minimal code churn
4. Performance
5. Maintainability

You must be decisive. Do not present multiple equal-weight options unless a real unresolved blocker forces it. Prefer the smallest safe change that solves the problem.

==================================================
PROJECT CONTEXT

This repository studies volatile memory as a temporal behavioral signal under controlled guest workloads.
The active QEMU pipeline captures RAM snapshots, enqueues adjacent dump-pairs as jobs, and processes them through a queue-based consumer pipeline.

==================================================
TECH STACK

- Bash
- Python
- JSON job files
- QEMU / libvirt / virsh
- Queue directories: pending / processing / done / failed

==================================================
PLANNING SCOPE

This planning task is intentionally scoped to dump lifecycle coordination and safe deletion behavior.

Out of scope unless strictly required:
- Rust compute optimization
- offline metrics redesign
- run-matrix redesign
- producer redesign
- full queue architecture rewrite

Assume the current active pipeline is effectively single-consumer unless findings prove otherwise.
The goal is to design deletion coordination that remains safe if multiple consumers are introduced, without requiring a major architectural rewrite.

==================================================
CURRENT OBJECTIVE

Design a minimal, safe deletion strategy for RAW dump files.

Facts:
- Each dump can be referenced twice:
  - as `curr` in job N
  - as `prev` in job N+1
- A dump must not be deleted while any active queue state still references it.
- Candidate mechanism:
  - after job completion, scan `pending/` + `processing/` for references to the dump
  - if zero references remain, the dump becomes eligible for deletion

Treat this scan-based mechanism as a candidate, not a fixed truth.
Decide whether it is the best minimal safe solution at expected queue scale.
If retained, justify why it is acceptable.
If rejected, propose the smallest safer alternative.

==================================================
KNOWN RELEVANT FILES

- `VM_sampler/VM_Capture_QEMU/capture_consumer_qemu.sh`
- `VM_sampler/VM_Capture_QEMU/capture_producer_qemu_pmemsave.sh`
- `VM_sampler/VM_Capture_QEMU/run_qemu_capture.sh`
- `VM_sampler/VM_Capture_QEMU/run_files_controlled.py`

==================================================
KNOWN CONSTRAINTS

- Preserve current producer/consumer architecture
- Keep code churn minimal
- Prioritize deletion safety over cleanup aggressiveness
- Deletion behavior must remain correct even if multiple consumers are added later
- Idempotent cleanup is preferred

==================================================
NON-GOALS

- Do not redesign the queue format unless unavoidable
- Do not replace the file-based queue system
- Do not optimize Rust compute in this task
- Do not broaden this into full end-to-end pipeline optimization

==================================================
REPO FINDINGS

- The consumer currently performs snapshot cleanup
- Queue lifecycle uses `pending`, `processing`, `done`, and `failed`
- Adjacent jobs share one dump as `curr` then `prev`
- Deletion correctness depends on queue state, not just local job completion
- Future parallel-consumer safety is desired even if current pipeline is effectively single-consumer

==================================================
YOUR TASK

Produce a concrete implementation specification for safe dump deletion coordination.

You must:
- choose the best minimal implementation path
- define the deletion eligibility rule precisely
- analyze race conditions and idempotency
- state whether scanning queue JSON files is acceptable or whether a slightly different mechanism is safer
- identify exact files likely to change
- provide a concise executor handoff prompt for a coding agent

==================================================
OUTPUT FORMAT (MANDATORY)

## 1. Executive Summary
One paragraph describing the chosen approach.

## 2. Core Invariants
List the dump lifecycle and queue-state invariants that must remain true.

## 3. Proposed Change Set
List exact files/modules/components likely to change and why.

Format:
- path/or/module -> purpose of change

## 4. Implementation Plan (Ordered)
Provide numbered execution steps.

For each step include:
- objective
- exact logic to add/change
- affected components
- migration concerns, if any

## 5. Algorithms / Pseudocode
Provide precise pseudocode for:
- deletion eligibility check
- reference scan logic
- post-job cleanup flow
- idempotent deletion behavior

## 6. Concurrency / Failure Analysis
Explicitly analyze:
- two workers completing adjacent jobs simultaneously
- double-delete attempts
- crash between marking job done and cleanup
- retries / duplicate processing
- malformed or stale job metadata
- idempotency requirements

## 7. Performance Impact
Estimate overhead of the chosen coordination mechanism and why it is acceptable or not.

## 8. Test Plan
Include:
- unit tests
- integration tests
- concurrency tests
- regression tests
- edge cases

## 9. Acceptance Criteria
Write precise criteria for declaring this work complete.

## 10. Rollout / Rollback
How to deploy safely and revert if needed.

## 11. Executor Handoff Prompt
Write a concise prompt for a coding agent to implement ONLY this approved plan.

==================================================
STRICT RULES

- Be decisive.
- Prefer simple mechanisms over elegant complex systems.
- Do not recommend rewrites unless unavoidable.
- Do not redo repo discovery unless a missing fact truly blocks safe planning.
- If assumptions are required, list them explicitly.
- Keep the plan implementation-focused, not essay-like.
- Another model will code directly from your output.

Return only the structured spec.
```
