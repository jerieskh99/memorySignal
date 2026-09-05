# Understanding the B1 result — in plain terms

**Why this file exists.** This is the plain-language companion to the first analysis-stage
result (Plan 08 / B1, the encoding floor). It explains, from the ground up and with no jargon,
what the experiment does and what the numbers mean — especially **recall**, the **APF vs wAPF**
comparison, and the three test **splits**. It is written to be read by a human (JK) *and* by any
future chat or agent that needs to understand these results without re-deriving them.

If you are an agent picking up this project: read this before interpreting anything in
`plan08_overview.html`, `b1_ae_results.json`, or `docs/EXPERIMENT_B1_ENCODING_FLOOR.md`.

The pretty version of this same story is the diary lesson
`docs/research-diary/part_b1_result.html` ("The first thing the analysis saw").

---

## Step 1 — What are we even looking at?

Every half-second we photograph the computer's memory and compare it to the previous photo. For
each 4 KB chunk (a **page**) we note how much it changed. Then we squash that whole photo-diff
into **one number**: "how busy was memory this instant?"

Two flavors of that number:
- **APF** (Active Page Fraction) = what *fraction of pages* changed (just did-it-change, yes/no,
  averaged over all pages).
- **wAPF** (weighted APF) = same idea, but each page counts by *how much* of it changed, not just
  yes/no.

Do that every half-second and you get a wiggly line over time — a heartbeat monitor for memory.
One line per recording.

## Step 2 — What is a "window"?

We chop that heartbeat line into pieces of **8 readings in a row** (stepping 4 each time, so they
overlap). Each 8-number piece is one **window** — think of an 8-second clip of a song. One
recording gives ~170 clips. Every clip is one thing we classify.

## Step 3 — What are we trying to do with a clip?

Sort it into its **family**: was this memory activity made by a *memory* program, a *cpu* program,
an *app*, a *disk (io)* program, etc.? There are 7 families. We guess the family from the clip
alone — we never see the actual program.

## Step 4 — How do we guess? (the "autoencoder bank")

We train one **expert per family**. Each expert studies *only its own family's clips* until it
knows them by heart. Then, to classify a new clip, we ask **all 7 experts** "how familiar is this
to you?" and give the clip to whichever expert is *most comfortable* (least surprised).

Seven specialists, each recognizes its own kind, and the clip goes to whoever recognizes it best.
(The "expert" is a small neural net called an **autoencoder**: it learns to squeeze a clip into 3
numbers and rebuild it. It rebuilds its own family's clips well and unfamiliar ones badly, so
"badly rebuilt" = "not my family.")

## Step 5 — What is "recall"? (the number that carries the story)

Recall answers **one specific question, per family**:

> "Of all the clips that *truly* belong to this family, what fraction did we correctly catch?"

Example: suppose 100 clips are truly **mem**. If 75 of them get correctly labeled "mem", then
**mem recall = 75/100 = 0.75.**

So recall = "did we **find** the ones that really belong here?" — a per-family batting average.
- **app recall 0.84** → of all real app clips, we caught 84%. We can find app.
- **cpu recall 0.02** → of all real cpu clips, we caught 2%. We almost never recognize cpu; those
  clips got labeled as something else.

Recall is reported per family because it shows, family by family, *which families the instrument
can actually find and which it is blind to.* (Its sister number, **precision**, asks the reverse:
"of the clips we *called* mem, how many really were mem." We report recall because the question
here is "can we find each family.")

## Step 6 — So what do our recall numbers say? (unseen-workload test)

```
family     APF recall   wAPF recall
app          0.84         0.54       <- found
mem          0.75         0.64       <- found
io           0.02         0.02       <- missed
cpu          0.02         0.00       <- missed
thread       0.01         0.01       <- missed
cache        0.00         0.00       <- missed (too little data)
sandbox      0.00         0.00       <- missed (special case)
```

Two families we find; five we basically cannot.

## Step 7 — *Why?* (the meaning)

The instrument only sees memory **writes**. Programs that mostly *compute* or *read* — a cpu
hashing loop, threads fighting over a lock — barely write memory, so their clips are almost all
**near-zero** (flat lines). And one flat line looks like every other flat line: the experts for
cpu/io/thread cannot tell them apart, so they mislabel each other and recall goes to ~0.

app and mem programs **write a lot** → their clips have real shape → they are recognized.

This "loud vs quiet" split is the instrument's **write-only blind spot**. It was predicted years
ago in this project ("only writes are visible"); this is the first time it has been *measured*. We
can even see it is *mutual*, not random: cpu is mislabeled as thread, thread as io — the quiet
families pour into one indistinguishable pool.

## Step 8 — The three "splits" (how hard we make the test)

| split | the test | difficulty |
|---|---|---|
| **within_trace** | test on clips from the *same recording* we trained on | easy (memorizing) — the "ceiling" |
| **loro** | test on a *different run* of the same program | medium |
| **lowo** | test on a program we have *never seen*, only others of its family | hard — real generalization; **the honest headline** |

Overall **accuracy** (all families lumped) on each, APF vs wAPF (majority-guess baseline 0.234):

```
                 APF acc  wAPF acc
within_trace      0.576    0.480     (ceiling)
loro              0.623    0.427     (unseen run)
lowo              0.342    0.254     (unseen workload — the headline)
```

`lowo 0.342` beats blind guessing (0.234) but not by much — because 5 of 7 families are invisible
and drag the average down. The important detail: at the ceiling, the *visible* families are
near-perfect (mem 1.00, app 0.98, io 0.97), so the model is **not broken** — it separates families
fine where there is a sound to hear.

## Step 9 — APF vs wAPF (the actual experiment)

We ran the *entire thing twice* — once with APF clips, once with wAPF clips — and compared.
**APF won everywhere.**

Why: the families we *can* see (memory sweeps) change *many* pages a *tiny* bit each. wAPF weights
by "how much," which shrinks those clips toward zero — into the same flat-line blind spot as the
quiet families. APF just counts pages, keeping them visible. For telling families apart, the plain
**count** beats the **weighted amount**. Clearest at the ceiling: mem falls from 1.00 (APF) to
0.66 (wAPF).

## Step 10 — The full picture in one breath

We turn memory into a busyness-heartbeat, slice it into 8-beat clips, and ask a panel of
family-experts which family each clip belongs to. **Recall** tells us, per family, how many of
that family's clips we actually caught. The result: the instrument catches the two families that
write heavily (**app, mem**), is deaf to the three that mostly compute (**cpu, io, thread**), and
the plain **count** of changed pages (APF) beats the **weighted** version (wAPF). That is the
**floor** everything else gets measured against.

---

## The one open question that changes the verdict

wAPF is APF **times** an intensity — a *product*, and a product hides its factors. So wAPF losing
is ambiguous: maybe intensity carries nothing, or maybe *multiplying* it into one number is what
destroyed it. The next experiment (**E2**) keeps breadth and intensity as *two separate channels*
(16 numbers per clip instead of 8). If E2 beats APF, the loss was the product; if E2 matches APF,
intensity genuinely adds nothing for family classification. That single test turns "weighting
lost" into the real verdict.

## The numbers, in one place

- Corpus: 55 recordings, **8971 clips**, 19 workloads, **7 families**. Windows: 8 long, hop 4.
- Model: one autoencoder per family (8 -> 3 -> 8), predicted family = best reconstruction.
- Headline (unseen workload, APF): only **app 0.84** and **mem 0.75** generalize; the rest ~0.
- APF > wAPF on every split. Majority-guess baseline: 0.234.
- Blind spot: cpu/io/thread collapse into one mutually-confused pool (write-only, predicted).

**Takeaway to remember:** the instrument hears writers and is deaf to thinkers; among the writers,
counting beats weighting; and the honest score rests on two families, not seven. Everything the
next experiments do is measured against that.
