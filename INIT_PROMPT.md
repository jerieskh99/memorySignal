# Init prompt for a new session

Paste the block below into a fresh chat to bring it onto this project. It deliberately does not
restate the content of `START_HERE.md`: the point is to make the session read it.

---

```
Repo: /Users/jeries/Desktop/projects/thesis/memorySignal/mem_sig
Branch: fullv5 (make sure you're on it, other branches are stale or partial)

Read START_HERE.md at the repo root first, then follow its ramp-up in order
before proposing anything. It's written for you specifically: it covers what
this project is, the way of thinking you're inheriting, what to read in what
order, and what you're being asked to build.

Short version so you know where you're headed: this is a doctoral project that
identifies what a VM is doing by watching only its memory. The capture half is
built, validated, gated, and driven from a console UI. The analysis half is not
built. That's your work.

Two things START_HERE.md will tell you but that are worth knowing up front:

1. There are two documents already proposing the analysis stage
   (docs/ANALYSIS_PIPELINE_METHODOLOGY.md, docs/METHODOLOGY_AS_EXECUTED.md).
   They contain real, correct, load-bearing work and also several claims that
   do not survive checking. Read them together with
   docs/RAJA_REVIEW_ANALYSIS_METHODOLOGY.md, which is a code-verified critique
   of both. Do not build on either without re-verifying.

2. Verify before asserting. Don't state that a file, function, flag, threshold,
   or channel exists without opening it. If you couldn't verify something, say
   so rather than smoothing over it. This project's record deliberately keeps
   its own failed claims visible, and that standard applies to you.

Start by working through the ramp-up. Tell me what you've understood before you
start designing, so I can correct you early if we're not on the same page.
```

---

## Why it is shaped this way

- **It sends the session to the file rather than summarising it.** A summary in the prompt would
  compete with `START_HERE.md` and drift from it as that file changes.
- **It front-loads the two failure modes that are expensive to discover late:** inheriting the
  draft methodology documents uncritically, and asserting unverified facts.
- **It asks for a read-back before design.** Cheaper to correct a misunderstanding early than to
  unwind a design built on one.

If `START_HERE.md` moves or is renamed, update the path in the block above.
