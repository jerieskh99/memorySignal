# Prompt 2: Rust Hot-Loop Optimization

```text
You are acting as a senior systems architect and implementation planner.

Important context:
Initial repo exploration has already been completed collaboratively. Do NOT spend effort rediscovering the codebase unless a missing fact is truly blocking a safe plan. Assume the findings below are materially correct unless they contradict each other.

Your job is to convert the findings into a concrete, implementation-grade specification that another coding agent (Cursor / Codex / Sonnet) can execute with minimal ambiguity.

Your priorities, in order:
1. Correctness
2. Minimal code churn
3. Performance
4. Safety
5. Maintainability

You must be decisive. Do not present multiple equal-weight options unless a real unresolved blocker forces it. Prefer the smallest safe change that solves the problem.

==================================================
PROJECT CONTEXT

This repository captures adjacent RAM snapshots and computes per-page delta outputs using a Rust binary.
The Rust delta program is in the hot path of the capture pipeline and likely contributes materially to runtime.

==================================================
TECH STACK

- Rust
- Tokio
- Rayon
- large RAW snapshot files
- page-wise delta computation
- outputs: cosine + hamming values per page

==================================================
PLANNING SCOPE

This planning task is intentionally scoped to Rust hot-loop optimization in the delta binary.

Out of scope unless strictly required:
- queue redesign
- dump deletion coordination
- offline metrics redesign
- run-matrix redesign
- producer redesign

==================================================
CURRENT OBJECTIVE

Design a minimal, safe optimization plan for the Rust hot loop.

Known findings:
- For 2GB RAM / 4KB pages, there are 524,288 pages per job
- Current per-page logic appears to allocate fresh vectors for cosine computation:
  - bytes -> Vec<f32>
  - bytes -> Vec<f32>
  - cosine(vec1, vec2)
  - vectors dropped
- This implies very high allocator pressure across a single job
- Candidate optimization:
  - allocate reusable float buffers outside the page loop
  - reuse them per page instead of allocating every iteration
- Candidate fast path:
  - if page bytes are identical, emit:
    - cosine = 1
    - hamming = 0
  - skip conversion and cosine computation entirely

You must evaluate whether reusable stack buffers, heap-backed reusable buffers, slices, or another minimal mechanism is the safest choice.

==================================================
KNOWN RELEVANT FILES

- `VM_sampler/VM_Capture/live_delta_calc/src/main.rs`
- `VM_sampler/VM_Capture/live_delta_calc/Cargo.toml`

==================================================
KNOWN CONSTRAINTS

- Preserve output semantics
- Keep the optimization local if possible
- Avoid introducing unsafe code unless clearly justified
- Minimize architectural churn
- Correctness of cosine/hamming outputs is more important than maximum speed

==================================================
NON-GOALS

- Do not redesign the surrounding capture pipeline
- Do not redesign file formats
- Do not solve queue deletion in this task
- Do not broaden into full pipeline benchmarking

==================================================
REPO FINDINGS

- The delta binary reads adjacent snapshot files
- It computes hamming and cosine per page
- The current implementation likely performs costly per-page allocations/conversions
- There is an opportunity for a correctness-preserving identical-page fast path

==================================================
YOUR TASK

Produce a concrete implementation specification for optimizing the Rust hot loop.

You must:
- choose the best minimal implementation path
- decide on the right reusable-buffer strategy
- define the identical-page fast path precisely
- identify exact files likely to change
- analyze correctness risks and output equivalence
- provide a concise executor handoff prompt for a coding agent

==================================================
OUTPUT FORMAT (MANDATORY)

## 1. Executive Summary
One paragraph describing the chosen approach.

## 2. Core Invariants
List the computational and output invariants that must remain true.

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
- per-page processing loop
- reusable buffer flow
- identical-page fast path
- output writing behavior

## 6. Correctness / Failure Analysis
Explicitly analyze:
- numerical equivalence risks
- precision or conversion differences
- identical-page fast-path correctness
- edge pages / partial chunks
- thread-safety considerations
- any allocator or memory-layout risks

## 7. Performance Impact
Estimate likely gains and explain where they come from.

## 8. Test Plan
Include:
- unit tests
- integration tests
- regression tests
- benchmark or profiling checks
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
- Prefer local changes over broad rewrites.
- Do not recommend rewrites unless unavoidable.
- Do not redo repo discovery unless a missing fact truly blocks safe planning.
- If assumptions are required, list them explicitly.
- Keep the plan implementation-focused, not essay-like.
- Another model will code directly from your output.

Return only the structured spec.
```
