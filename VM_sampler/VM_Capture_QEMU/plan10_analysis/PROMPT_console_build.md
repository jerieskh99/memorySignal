# Plan 10 · Analysis Console — the Fable prompt

Drafted by envoy, 2026-09-08. **This is a prompt, not a change.** Nothing here has run.

**Before pasting: substitute the branch.** `init_prompt.md` says `fullv5`. The repo's current
branch is `b1-encoding-floor`, and `main` and `main-512mb` also exist. All of them are real.
Pick one, replace `<BRANCH>`, and do not leave the choice to the agent.

---

```
Work in a NEW git worktree off <BRANCH>.

WHERE YOU ARE

This is a doctoral project that identifies what a virtual machine is doing by watching only
its memory. A differ samples the guest's whole physical RAM on a cadence and writes, per page
pair, sixty-four metric channels describing how that page changed. Everything downstream is an
attempt to read workload identity out of that signal.

The project is in two halves and they are not in the same condition. The capture half is
finished work: it composes, launches, watches and gates a campaign, driven from a console UI
that is one hand-edited vanilla HTML file, a Python build script, a stdlib bridge, a launcher.
A full 101-workload campaign has run through it. The analysis half is still five scripts
invoked by hand over an ad-hoc list of sources, which is exactly the condition the capture
half was in before its console existed.

The two consoles are meant to be a pair. The capture console composes how memory is recorded;
the analysis console composes how the recording is read. They should feel like one instrument,
and they share the pipeline's real definitions by importing them rather than restating them.

Building the analysis console is your job.

THE STANDARD THIS PROJECT HOLDS, WHICH APPLIES TO YOU

Verify before asserting. Do not state that a file, function, flag, threshold or channel exists
without opening it. Where you could not verify something, say so plainly rather than smoothing
over it. This project's written record deliberately keeps its own failed claims visible instead
of editing them away, and that standard extends to your work and to how you report it. An
honest "I could not determine this" is worth more here than a confident guess, and there is at
least one question below where that is the correct answer.

WHAT EXISTS, AND HOW MUCH TO TRUST IT

The console's design is settled and there is a working mockup:
plan10_analysis/ui/analysis_canvas.mockup.html. A single vanilla file with no dependencies, in
which analysis modules are dropped on a canvas and piped together. Ports are typed, so a signal
can only enter a port that fits it. A constraint layer decides what is refused outright, what
is warned about, and what must be explicitly acknowledged before a run is allowed. It emits a
scheme as JSON. Treat it as the specification rather than a draft; it encodes decisions that
were argued over. Where you think it is wrong, say so, but do not quietly improve it.

It is also entirely fake. The channel taxonomy is typed into the file, the set of dead channels
is a hand-written constant, the known-issue registry is prose, and the list of available
recordings is invented and completely convincing. Every number in it was made up to make the
design legible.

Two design documents describe the console, docs/plan10_analysis_console_proposal.md and
docs/plan10_analysis_console_UX.md, the second amending the first. They are load-bearing and
mostly right. They are also older than the mockup, and the mockup silently overturned one of
their recommendations. A third, docs/RAJA_REVIEW_ANALYSIS_METHODOLOGY.md, is a code-verified
critique of the project's earlier methodology documents; the habit it represents matters more
than its contents, which is that claims in this project's documents get checked against code,
and several have not survived.

WHAT YOU ARE BEING ASKED FOR

The console, for real: the mockup wired to the pipeline, with nothing invented left in it.

Someone should be able to open it, see the recordings that actually exist on this machine with
their real properties, compose a scheme over the channels the differ actually wrote, be refused
when the scheme is invalid and warned when it is merely questionable, and come away with a
scheme file a runner could execute.

The runner itself is not in scope, and neither is reproducing an earlier experiment through it.
Those are the next piece of work, excluded here because the thing they consume has to exist and
be trustworthy first.

WHY THIS IS NOT PLUMBING

Two things make this harder than wiring, and they are why the console is worth building.

The constraint layer is load-bearing rather than advisory. A canvas will happily let someone
compose a scheme that is perfectly well-typed and scientifically meaningless. Fusing a channel
that describes a page's state with one that describes a page's change produces a number that
does not mean what it looks like. A window longer than the shortest selected recording produces
zero windows. A spectral transform on a two-sample window has no spectrum. Nothing about the
graph's shape catches any of that; the constraints are the only thing that does. The design
documents type each rule as hard or soft and cite where it comes from, and getting those
severities right is most of the console's value.

And this console's whole claim is that a scheme's provenance is complete: which recordings,
which channels, which transforms, what was warned about and knowingly accepted. That claim is
worth nothing if the console will display plausible data it invented. A console that renders
sample recordings when the real corpus is missing is worse than one that refuses to start. The
capture side already solved the general version of this problem, and its build script's
docstring states the contract in the pipeline's own words; read it early.

One related habit, unusual enough to state: this project does not hide what is known to be bad.
A feature that misled someone in an earlier experiment stays in the UI, badged, with its history
attached, because a feature deleted from a console is one someone re-derives in six months.
Proceeding past a warning is allowed, and the acknowledgment is written into the output's
record, so a run that ignored a warning is never indistinguishable from one that never faced it.
The acknowledgment is part of the artifact, not a dismissed dialog.

A DECISION THAT NEEDS RECORDING

The UX document recommends a fixed linear spine over a free-form graph editor, on two premises:
that the analysis topology is fixed with branching only at the transform stage, and that a
drag-and-drop graph engine could not fit the project's one-vanilla-file constraint. The mockup
built afterwards is a free-form typed-port graph editor, and both premises look false against
it. JK has decided the canvas stands.

Verify both premises yourself rather than taking that on my word, then record the reversal in
the document in this project's house style: the superseded recommendation stays visible with
its reasoning intact, and a dated amendment says what evidence overturned it and what the
canvas costs that the spine would not have.

WHAT DONE LOOKS LIKE

The console displays only things that are true. The channel taxonomy, which channels are live,
the issue history and the corpus all come from the pipeline or the filesystem, and the template
keeps no copy of its own to drift from. A missing corpus is a loud failure rather than a
fallback to sample data.

The rules survive the move out of the browser. Everything the documents type as hard refuses,
with a message naming what to change; everything typed as soft is selectable, reported, blocks
until acknowledged, and the acknowledgment lands in the emitted scheme. The mockup's three
worked examples still load and behave the same way against real recordings, and a scheme file
the mockup emits today still loads, because that format is already a contract the runner will
read.

Cost is reported in a way that keeps this project's discipline intact: effective n is workloads,
not windows, and the two numbers must not be conflatable.

The built console actually renders and behaves, confirmed by loading it rather than by reasoning
about it. And someone who knows the capture console can read this one without relearning
anything.

WHAT WILL BITE

Things I checked this session that cost real time to find. Facts, not instructions.

Which channels are dead is not a fixed list: it follows from the speed level a recording was
captured at, and the differ drops channels cumulatively as that level rises. The mockup froze
one level's answer into a global constant. Whether the level each recording was captured at is
recorded anywhere is something you will have to find out, and if it is not, that is a finding to
report rather than a gap to fill with a default.

The differ's help text says the highest speed level drops something called "struct_entropy", and
no column by that name exists. Two columns have names close to it, in different families.
Resolve it from the code if you can; if you cannot resolve it with certainty, say so and leave
that level marked unverified.

The Rust's column order and the Python taxonomy's order are not the same, so any check that the
two agree has to compare names rather than positions. A positional check passes today and lies
later.

The mockup pulls fonts over the network, so a build advertised as network-free would not be
literally network-free. The capture console may have the same caveat. Check, and report what you
find rather than quietly fixing it; it may be a finding about both.

NOT YOURS TO SETTLE

The UX document lists open decisions and at least one will touch this work: whether the page
count is a single constant or a per-recording value. Leave it open, in a shape that can become
per-recording later without forcing a rewrite. If you hit others, name them in your report
instead of deciding them.

BOUNDARIES

Pure stdlib for anything you write; the analysis environment pins numpy and scikit-learn only
and none of this should need either. The UI stays one hand-edited vanilla file: no framework, no
bundler, no dependency beyond python3, no external reference beyond what the mockup already
carries. Start no server, open no browser, launch no analysis run; any network path should be
visibly unimplemented rather than half-wired, because the bridge is later work. Leave
analysis_canvas.mockup.html in place and untouched as the record of the design step. Tests
belong in VM_sampler/VM_Capture_QEMU/tests/, which has a convention already, including how tests
are run and what they may depend on; read one before writing one. No emojis, no em-dashes.

WHEN DONE

Report what you built and why it took that shape; what you verified and what you could not;
where the documents and the code disagree and how you called it; anything you introduced that no
source supports; the answers to the four things above; and a diff summary.

Do NOT merge.
```

---

## What this leaves out, and why

**The runner, and reproducing the B1 experiment through the console.** The real acceptance test for
Plan 10 is driving the console's own reproduction of an earlier experiment and asserting the
features come out identical to what `plan08_b1/b1_features.py` already produced. That is the proof
and the obvious next prompt, but it needs the thing it consumes to exist first.

**The bridge.** The capture side's is roughly a thousand lines. The UX document says the analysis
one binds localhost with a token and needs no ssh, which makes it simpler, but it is still its own
piece of work.

**A measurement the UX document flags as never taken:** whether the guest's physical address range
contains permanently dead rows from reserved or memory-mapped regions, which would inflate every
"unchanged" count identically across all workloads. The document calls it cheap to measure from any
existing chain and worth doing before the address axis is used seriously. Capture-side measurement,
not a console change.
