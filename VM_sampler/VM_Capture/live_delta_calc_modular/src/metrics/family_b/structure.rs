//! Family B / structure: did the pattern persist (cosine, Pearson, rank correlations, SSIM-s).

use crate::metrics::common::{spearman_kendall, ssim_consts, Shared};
use distances::vectors::cosine;

#[derive(Clone, Default, Debug)]
pub struct Structure {
    pub cosine: f32,      // cosine DISTANCE (0 = identical)
    pub pearson: f32,     // centered cosine of the byte vectors
    pub ssim_struct: f32, // SSIM structure term
    pub spearman: f32,    // Spearman rank correlation
    pub kendall: f32,     // Kendall tau-a (ties not corrected)
}

pub fn compute(sh: &Shared, p: &[u8], q: &[u8]) -> Structure {
    // Page differs here (orchestrator early-returns identical pages), so cosine needs
    // no guard -- the `distances` crate maps an all-zero side to 1 = max.
    let p_f32: Vec<f32> = p.iter().map(|&x| x as f32).collect();
    let q_f32: Vec<f32> = q.iter().map(|&x| x as f32).collect();
    let cos = cosine(&p_f32, &q_f32);

    let mo = sh.moments();
    let (_c1, _c2, c3) = ssim_consts();
    let pearson = if mo.vp > 0.0 && mo.vq > 0.0 {
        mo.cov / (mo.sdp * mo.sdq)
    } else {
        0.0
    };
    let (spearman, kendall) = spearman_kendall(p, q, &sh.hp, &sh.hq);

    Structure {
        cosine: cos,
        pearson: pearson as f32,
        ssim_struct: ((mo.cov + c3) / (mo.sdp * mo.sdq + c3)) as f32,
        spearman: spearman as f32,
        kendall: kendall as f32,
    }
}
