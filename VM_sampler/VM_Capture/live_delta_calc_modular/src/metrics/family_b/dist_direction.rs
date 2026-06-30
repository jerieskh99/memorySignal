//! Family B / distributional-direction: where the histogram moved (zero-mass change).

use crate::metrics::common::Shared;
use crate::metrics::PAGE_SIZE;

#[derive(Clone, Default, Debug)]
pub struct DistDirection {
    pub zero_mass_delta: f32, // Q_0 - P_0  (zero-byte mass change; signed)
}

pub fn compute(sh: &Shared) -> DistDirection {
    let nf = PAGE_SIZE as f64;
    DistDirection {
        zero_mass_delta: ((sh.hq[0] as f64 - sh.hp[0] as f64) / nf) as f32,
    }
}
