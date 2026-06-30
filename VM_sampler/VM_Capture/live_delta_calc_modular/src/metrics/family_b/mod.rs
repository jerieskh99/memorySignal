//! Family B -- direction of change (alternatives to cosine), split by reading.

pub mod dist_direction;
pub mod level;
pub mod polarity;
pub mod spatial_shift;
pub mod spread;
pub mod structure;

use crate::metrics::common::Shared;

#[derive(Clone, Default, Debug)]
pub struct Direction {
    pub structure: structure::Structure,
    pub level: level::Level,
    pub spread: spread::Spread,
    pub polarity: polarity::Polarity,
    pub dist_direction: dist_direction::DistDirection,
    pub spatial: spatial_shift::Spatial,
}

pub fn compute(sh: &Shared, p: &[u8], q: &[u8]) -> Direction {
    Direction {
        structure: structure::compute(sh, p, q),
        level: level::compute(sh),
        spread: spread::compute(sh),
        polarity: polarity::compute(sh),
        dist_direction: dist_direction::compute(sh),
        spatial: spatial_shift::compute(p, q),
    }
}
