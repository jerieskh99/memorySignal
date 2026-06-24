# `security_like_safe/` — safe behavioural simulations (NOT malware)

The programs in this directory are **safe, synthetic workload simulators** used by an academic
PhD thesis on **defensive** memory-signal research. They exist to drive a virtual machine with
**recognisable, reproducible behavioural patterns** so that a host-side memory-signal sampler
can study whether such behaviour can be detected and characterised. **They are not malware.**

## What these programs are

- `sandbox_ransom_seq`, `sandbox_ransom_batched`, `sandbox_ransom_slowburn`,
  `sandbox_ransom_selective` — simulate the *shape* of a per-file processing pipeline
  (stat → read → transform → write → rename), in sequential / batched / paced / selective
  variants.
- `sandbox_scanner_metadata` — simulates staged file-metadata enumeration.
- `sandbox_stealth_microwrite`, `sandbox_stealth_scattered`, `sandbox_stealth_paced` — the
  **STEALTH subfamily**: single-page (4096 B) rewrites in a steady / spatially-scattered /
  time-jittered pattern, producing a **low page-change count but a full per-page bit-flip**.
  These are **defensive test cases** for the detector — a quiet-but-intense writer that the
  Active Page Fraction (which only counts *how many* pages changed) is blind to but the
  per-page magnitude (Hamming) metric should catch. They do **not** evade anything; they are
  named for the memory *footprint* they produce, and obey the same reversible-XOR / sandbox
  rules as every other program here.

They are named after the **benchmark pattern they imitate**, not after any real capability.

## What they explicitly do NOT do

No real encryption (the only transform is a **reversible XOR** with a fixed key). No network,
sockets, exfiltration, or command-and-control. No persistence. No evasion, stealth, or
anti-analysis. No privilege escalation or credential access. No access to real user data — they
operate **only on disposable, synthetic files inside a validated sandbox directory**, are
resource-capped and time-bounded, and run only with the launching user's privileges.

Each source file carries a header banner restating this.

## Authoritative documents

- Formal safety contract (whitelist / blacklist + enforcement):
  [`../docs/SAFETY_MODEL.md`](../docs/SAFETY_MODEL.md)
- Repository-wide notice for reviewers / scanners:
  [`../../RESEARCH_SAFETY_NOTICE.md`](../../RESEARCH_SAFETY_NOTICE.md)
