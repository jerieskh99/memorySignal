"""plan10_analysis.learn -- the Learn view's engine: pipelines of typed modules over what runs
wrote, run per split and fold with every fitted step inside the fold, scored deeper than one
number. See LEARN_DESIGN_BRIEF.md. Each file is separable and tested on its own:

  registry.py    the palette: tiers, shapes, modules, what each needs, what is unbuilt
  pipeline.py    the pipeline format, its validator (hard / soft / note) and the sweep
  data.py        rows from features.npz, paths and images from tiles.npz, labels from keys
  splits.py      grouped folds: within-trace ceiling, leave-one-recording/workload/campaign-out
  preprocess.py  fitted transforms, fit on train only
  models.py      one wrapper interface over sklearn, numpy and torch models
  scores.py      the scores, the null, the bootstrap, the explanations
  executor.py    run every configuration; status, control, outputs, sidecar
  results.py     what a Learn run wrote, as frames the console's views can draw
"""
