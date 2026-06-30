//! Family B / level: did the values move up or down (mean / luminance / median).

use crate::metrics::common::{pctl, ssim_consts, Shared};
use crate::metrics::PAGE_SIZE;

#[derive(Clone, Default, Debug)]
pub struct Level {
    pub mean_shift: f32,   // mean(q) - mean(p)  (signed)
    pub ssim_lum: f32,     // SSIM luminance term
    pub median_shift: f32, // median(q) - median(p)  (signed)
}

pub fn compute(sh: &Shared) -> Level {
    let mo = sh.moments();
    let (c1, _c2, _c3) = ssim_consts();
    Level {
        mean_shift: (mo.mq - mo.mp) as f32,
        ssim_lum: ((2.0 * mo.mp * mo.mq + c1) / (mo.mp * mo.mp + mo.mq * mo.mq + c1)) as f32,
        median_shift: (pctl(&sh.hq, PAGE_SIZE, 0.5) as f64 - pctl(&sh.hp, PAGE_SIZE, 0.5) as f64)
            as f32,
    }
}
