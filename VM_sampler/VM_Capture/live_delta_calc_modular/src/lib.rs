//! live_delta_calc_modular -- the per-page memory-delta feature substrate, as a library.
//!
//! The binary (`main.rs`) handles IO, threading and CSV output; this library is the
//! pure compute: one `PageMetrics` per (prev, curr) 4 KB page pair, organised family
//! by family. See docs/feature_substrate_spec.{md,pdf}.
//!
//! Behaviour is identical to the original ../live_delta_calc (verified by a byte-exact
//! golden diff of the metrics CSV).

pub mod metrics;
