//! Family A -- amount of change (alternatives to Hamming), split by representation space.

pub mod distributional;
pub mod informational;
pub mod positional;

use crate::metrics::common::Shared;

#[derive(Clone, Default, Debug)]
pub struct Amount {
    pub positional: positional::Positional,
    pub distributional: distributional::Distributional,
    pub informational: informational::Informational,
}

pub fn compute(sh: &Shared, p: &[u8], q: &[u8], speed: u8) -> Amount {
    Amount {
        positional: positional::compute(sh, p, q),
        distributional: distributional::compute(sh),
        informational: informational::compute(sh, p, q, speed),
    }
}
