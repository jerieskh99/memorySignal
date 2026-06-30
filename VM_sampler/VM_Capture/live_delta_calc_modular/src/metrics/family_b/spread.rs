//! Family B / spread: did the page get more or less varied (variance / contrast / range).

use crate::metrics::common::{hi_bin, lo_bin, pctl, ssim_consts, Shared};
use crate::metrics::PAGE_SIZE;

#[derive(Clone, Default, Debug)]
pub struct Spread {
    pub var_ratio: f32,     // var(q) / var(p)
    pub std_delta: f32,     // std(q) - std(p)  (signed)
    pub ssim_contrast: f32, // SSIM contrast term
    pub range_delta: f32,   // (max-min)(q) - (max-min)(p)  (signed)
    pub iqr_delta: f32,     // IQR(q) - IQR(p)  (signed)
}

pub fn compute(sh: &Shared) -> Spread {
    let mo = sh.moments();
    let (_c1, c2, _c3) = ssim_consts();
    let var_ratio = if mo.vp > 0.0 { mo.vq / mo.vp } else { 0.0 };
    let range_p = hi_bin(&sh.hp) as f64 - lo_bin(&sh.hp) as f64;
    let range_q = hi_bin(&sh.hq) as f64 - lo_bin(&sh.hq) as f64;
    let iqr_p = pctl(&sh.hp, PAGE_SIZE, 0.75) as f64 - pctl(&sh.hp, PAGE_SIZE, 0.25) as f64;
    let iqr_q = pctl(&sh.hq, PAGE_SIZE, 0.75) as f64 - pctl(&sh.hq, PAGE_SIZE, 0.25) as f64;
    Spread {
        var_ratio: var_ratio as f32,
        std_delta: (mo.sdq - mo.sdp) as f32,
        ssim_contrast: ((2.0 * mo.sdp * mo.sdq + c2) / (mo.vp + mo.vq + c2)) as f32,
        range_delta: (range_q - range_p) as f32,
        iqr_delta: (iqr_q - iqr_p) as f32,
    }
}
