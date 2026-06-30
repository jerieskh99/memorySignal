//! Family B / polarity: the directional "current" of the change (up vs down).

use crate::metrics::common::Shared;
use crate::metrics::PAGE_SIZE;

#[derive(Clone, Default, Debug)]
pub struct Polarity {
    pub polarity: f32,       // (n_up - n_down) / n  (signed)
    pub sign_delta_ent: f32, // entropy of the up/down/same pattern (direction coherence)
}

pub fn compute(sh: &Shared) -> Polarity {
    let nf = PAGE_SIZE as f64;
    let pu = sh.n_up as f64 / nf;
    let pd = sh.n_down as f64 / nf;
    let psame = 1.0 - pu - pd;
    let mut sde = 0.0;
    for &pr in &[pu, pd, psame] {
        if pr > 0.0 {
            sde -= pr * pr.log2();
        }
    }
    Polarity {
        polarity: ((sh.n_up as f64 - sh.n_down as f64) / nf) as f32,
        sign_delta_ent: sde as f32,
    }
}
