# Gate 2 result — guest-physical contiguity: SCATTERED

Date: 2026-09-06. Guest: `Kali Jeries` (libvirt, 1 GiB RAM). Kernel: `kernel_gemm_v2`
at default `--dim 1024`. Differ: `live_delta_calc_modular --speed 0 --sparse`.
Script: [`gate2_contiguity_check.sh`](../../VM_sampler/VM_Capture_QEMU/plan09_dwarf_pilot/gate2_contiguity_check.sh).

## Question

Every "address-shape" feature in the pilot (barnes_hut vs nbody occupying different
address regions; the LU/QR front's spatial position/shape) assumes that an array
which is contiguous in the program's **guest-virtual** address space (one `mmap()`)
stays contiguous in **guest-physical** RAM — which is what `pmemsave` dumps and the
differ diffs. The memory-systems and DSP experts predicted it would not, because the
guest buddy allocator places pages and every phase-2 kernel calls `MADV_NOHUGEPAGE`
(disabling the transparent-huge-page path that would otherwise force physical
contiguity). The DSP expert withdrew his own address-shape feature over this risk
during the design debate. Gate 2 measures it directly.

Metric: `frac_in_longest_run` = (pages in the single longest contiguous run of
`page_index`) / (total changed pages). `> ~0.8` contiguous; `< ~0.2` scattered.

## Result: SCATTERED (confirmed across two runs, three snapshots)

Run 1 (single snapshot):

| metric | value |
|---|---|
| changed pages | 6464 (≈6144 for A+B+C mmap + ~300 OS floor) |
| distinct contiguous runs | 3210 (avg ~2 pages each) |
| longest run | 608 pages (page_index 61792..62399) |
| **frac_in_longest_run** | **0.094** |

Top runs after the longest: 98, 56, 38, 37, 35, 26, 25, 24, 24 — i.e. rubble.

Run 2 (three snapshots, to rule out a timing fluke):

| seq | changed pages | runs | longest run | frac_in_longest_run |
|---|---|---|---|---|
| 1 | 6661 | 2847 | 578 | 0.087 |
| 2 | 14382 | 3953 | 2329 | 0.162 |

(seq=2's ~2x page count is gemm re-seeding A and B on top of rewriting C — a snapshot
catching a re-seed phase, still shattered.)

Every snapshot lands `frac` in **0.087–0.162** — all below the 0.2 scatter threshold,
none near the 0.8 contiguous bar. A matrix that is one contiguous block in the
program's own view arrives in physical RAM as ~3000 fragments. The experts' warning
was correct.

## Implications for Gate 3 (pre-registered contingency, not a surprise)

- **Drop** the address-shape features: barnes_hut-vs-nbody (needed different address
  *regions*) and the LU/QR-front *spatial shape/position* claim.
- **Keep** (untouched by this result — they never depended on page position):
  per-page **magnitude distribution** (l1/l2/hamming) and **footprint size** (count of
  changed pages over time).
- **Nuance**: the resized QR/LU fronts from Gate 1 survive as a **footprint-size
  trajectory** — the *count* of changed pages growing (QR) / shrinking (LU) across
  snapshots is placement-invariant. Only the front's *shape* dies, not its
  *size-over-time*. The temporal arm survives in a weaker, count-based form.

## Caveats

- One kernel (gemm), one size (dim 1024). Strong evidence the allocator scatters large
  mmap'd arrays; a formal claim would repeat it on 1–2 more kernels/sizes.
- This is a property of **this** guest (buddy allocator + `MADV_NOHUGEPAGE`), not
  physics. Forcing transparent huge pages back on, or a different guest, could restore
  contiguity — a possible **future lever** to recover address-shape features, out of
  scope for this pilot.

## Reproduce / raw data

Scratch trajectory CSV on the server (not committed; regenerable):
`/var/tmp/gate2_dwarf_pilot/output/substrate_trajectory.csv`. Regenerate with:

```bash
SSH_TARGET=kali@<guest-ip> bash VM_sampler/VM_Capture_QEMU/plan09_dwarf_pilot/gate2_contiguity_check.sh
```

Contiguity summary on any such CSV:

```bash
awk -F, 'NR>1{print $2+0}' <substrate_trajectory.csv> | sort -n | uniq | awk '
NR==1{prev=$1;start=$1;run=1;next}
{if($1==prev+1)run++;else{n++;rl[n]=run;rs[n]=start;re[n]=prev;run=1;start=$1}prev=$1}
END{n++;rl[n]=run;rs[n]=start;re[n]=prev;for(i=1;i<=n;i++){tot+=rl[i];if(rl[i]>mx){mx=rl[i];mxi=i}}
printf "changed=%d runs=%d longest=%d frac=%.3f\n",tot,n,mx,mx/tot}'
```
