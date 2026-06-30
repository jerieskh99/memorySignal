# Task C spec -- magnitude + entropy capture features (2 axes x 2 granularities)

Status: SPEC (not yet implemented in capture). Additive, flag-gated; the APF path stays
byte-identical when the flag is off. Built to compute everything in ONE pass over the pages
the APF comparison already touches.

## Intent

APF answers only "how many pages changed". We add two more axes that see what APF is blind
to -- a quiet-but-intense rewriter (a stealth encryptor) is low APF but loud on both:

- **Magnitude (Hamming)** -- HOW MUCH each page changed (bits flipped).
- **Entropy (byte)** -- HOW RANDOM the changed content looks (the encryption tell).

Each axis is emitted at two granularities, because they catch different failure modes:

- **per-page-aggregate** (local): compute the metric on each changed page, then aggregate
  over the changed pages (mean / max / p95 / std). The MAX catches a single encrypted page
  buried among quiet ones.
- **per-snapshot** (global): one number for the whole snapshot -- the aggregate composition
  of everything that changed this interval. Dilutes a single hot page but captures the bulk.

This mirrors a pattern already in the Hamming design (`ham_mean` is local, `ham_sum` is global)
and the principle behind it: capture richly (capture is the expensive, irreversible part),
let the offline feature selection decide which scalars carry signal.

## Constants

```
B      = 4096            # page size in bytes
BITS_P = 8 * B = 32768   # bits per page
LOG2B  = log2(B) = 12
```

## The 2 x 2 feature matrix (per snapshot pair, prev vs curr)

Notation: N pages total; for changed page i, h_i = popcount(prev_i XOR curr_i) (bits flipped);
c_b^(i) = count of byte value b in curr_i; per-page entropy H_byte_i in bits/byte in [0, 8].

Two columns = two granularities. LEFT (distribution): compute the metric on each changed page,
then summarize the spread OVER pages (mean/max/p95/std). RIGHT (global): pool all changed content
into one blob and compute ONE number -- inherently single, nothing to average within a snapshot.

| way | DISTRIBUTION over the K changed pages | GLOBAL -- one pooled blob |
|------|-----------------------------------------------|------------------------|
| **count (baseline)** | -- | `n_changed` K, `apf` = K/N |
| **magnitude (Hamming)** = "magnitude distribution" | `ham_mean` = (Sigma h_i)/K, `ham_max`, `ham_p95`, `ham_std` | `ham_sum` = Sigma h_i, `ham_frac` = Sigma h_i / (N*BITS_P) |
| **entropy (byte)** = "entropy distribution" | `ent_mean`, `ent_max`, `ent_p95`, `ent_std` (over H_byte_i) | `ent_pooled` = entropy of the POOLED histogram |

The entropy mean/max are `ent_mean`/`ent_max` on the LEFT (over pages). `ent_pooled` on the RIGHT is
a different, global view -- the entropy twin of `ham_sum`. Mean/max for the RIGHT column are not
emitted per snapshot (a single blob has none); they reappear OFFLINE, taken over snapshots (time) --
see the per-cell section. Same for `ham_sum`/`ent_pooled`: single per snapshot, a series over a cell.

Read this as: **APF (the naive baseline) is always co-emitted**, and next to it sits the DEPTH
block = 2 ways (magnitude + byte entropy) x 2 granularities (per-page-aggregate + per-snapshot)
= 4 feature groups. The naive Hamming scalars we already had (mean/max/p95/std) ARE the
magnitude/per-page group -- kept, not replaced. Every snapshot carries APF and all 4 groups, so
the depth axes are always measured against the APF baseline on identical data.

Per-page byte entropy (constant-denominator form -- B factors out, no division inside the log):
```
H_byte_i = LOG2B - (1/B) * Sigma_b ( c_b^(i) * log2(c_b^(i)) )       # over b with c_b > 0
```
Pooled entropy (sum the per-page histograms; total = K*B, NOT constant):
```
C_b      = Sigma_i c_b^(i)
ent_pooled = log2(K*B) - (1/(K*B)) * Sigma_b ( C_b * log2(C_b) )
```
All magnitude/entropy fields are 0 when K = 0 (nothing changed).

## Per-snapshot record (magent_trajectory.jsonl -- one line per snapshot, keyed by seq)

```json
{
  "seq": 0, "t_emit_epoch": 0.0,
  "n_pages": 0, "n_changed": 0, "apf": 0.0,
  "ham_mean": 0.0, "ham_max": 0, "ham_p95": 0, "ham_std": 0.0,
  "ham_sum": 0, "ham_frac": 0.0,
  "ent_mean": 0.0, "ent_max": 0.0, "ent_p95": 0.0, "ent_std": 0.0,
  "ent_pooled": 0.0
}
```
Sibling to `apf_trajectory.jsonl`, joined by `seq` (exactly like the Plan 06 diskio trajectory).
Hamming fields are in BITS; a normalised fraction = field / BITS_P (do this offline if wanted).

## Single-pass algorithm (the shared computation / reuse)

Precompute ONCE, reuse for every page in every snapshot of every cell:
```
POP[256]       = popcount of each byte value
T[k] = k*log2(k) for k = 0 .. B   (T[0] = 0)      # entropy without logs in the hot loop
LOG2B = 12
```
Per snapshot pair (prev, curr), one pass:
```
K = 0; ham_sum = 0
hams = []; ents = []                     # for per-page p95 / std (size K, transient)
C = int[256] = 0                         # pooled histogram
for i in 0..N:
    if pages_equal(prev_i, curr_i): continue     # cheap memcmp/SIMD; unchanged pages stop here
    K += 1
    d   = prev_i XOR curr_i
    h_i = Sigma_byte POP[d_byte]                  # Hamming (reuses the XOR)
    hist = histogram256(curr_i)                   # byte counts of CURRENT page content
    ent_i = LOG2B - (1.0/B) * Sigma_b T[hist[b]]  # per-page entropy (table lookups)
    hams.append(h_i); ents.append(ent_i)
    for b: C[b] += hist[b]                         # accumulate pooled histogram
    ham_sum += h_i
# aggregates
ham_mean/max/p95/std  <- hams
ent_mean/max/p95/std  <- ents
ham_frac = ham_sum / (N * BITS_P)
ent_pooled <- C, total = K*B   (0 if K == 0)
emit record
```
Reuse summary: the **page-diff is shared** (APF "changed?" + Hamming XOR + gates the entropy);
the per-page **histograms are summed** into the pooled histogram (per-page layer + global layer
from the same histograms); `POP[]`, `T[k]`, `LOG2B` are computed once and reused everywhere.
Expensive work (popcount + histogram) runs ONLY on the K changed pages, not all N.

Note on p95/std: storing `hams`/`ents` is O(K) transient per snapshot. If K is very large,
replace with a small fixed histogram of h_i (bins over 0..BITS_P) and of ent_i (bins over 0..8)
to get percentiles in O(1) memory.

## Integration (additive, flag-gated)

- New env flag `TIMING_MAGENT` (default unset). The whole block above runs only when set;
  unset => the APF/diff path is byte-identical to today. Mirrors the Plan 06 `TIMING_DISKIO`
  additive block in `capture_producer_qemu_pmemsave.sh` / the apf_calc consumer.
- The consumer that computes APF already does the page diff -- the MAGENT block attaches the
  popcount + histogram in that same loop. No second pass over memory.
- Emits `<base>.magent_trajectory.jsonl` keyed by `seq`, joinable to the APF trajectory.

## Offline per-cell features (plan07_campaign/magent_features.py -- to write against this schema)

Per cell: read `magent_trajectory.jsonl`, and for each of the 14 snapshot scalars take the
across-snapshot aggregate (mean, max, p95) -> the per-cell magnitude+entropy feature block,
which joins the AGNOSTIC + peakvar matrix as new columns. Keyed by cell_id, mirroring
`extra_features.load_extra` / `diskio_features.load_diskio`.
```
wr feature examples per cell:
  ham_mean_mean, ham_max_max, ham_frac_mean, ent_mean_mean, ent_max_max, ent_pooled_mean, ...
```

## What to capture with it

Hamming + entropy only matter where APF goes blind -- the quiet-but-intense writers. So the
MAGENT capture targets the threats (slowburn) and the stealth family (sandbox_stealth_*), not a
blind re-run of the loud mem/app workloads. The stealth signature to confirm:
**low apf  +  high ham_mean / ham_max  +  high ent_max (and ent_pooled)** = few pages, each
fully and randomly rewritten.
