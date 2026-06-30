//! Family A / distributional: byte-histogram distances (0 = identical histogram).

use crate::metrics::common::Shared;
use crate::metrics::PAGE_SIZE;

#[derive(Clone, Default, Debug)]
pub struct Distributional {
    pub tv: f32,            // Total Variation: 0.5 * sum |P_b - Q_b|
    pub chi2: f32,          // symmetric chi-square: sum (P_b - Q_b)^2 / (P_b + Q_b)
    pub hellinger: f32,     // (1/sqrt2) * sqrt(sum (sqrt(P_b) - sqrt(Q_b))^2)
    pub kl: f32,            // Kullback-Leibler: sum P_b * log2(P_b / Q_b)  (eps-smoothed)
    pub js: f32,            // Jensen-Shannon divergence (bounded [0,1])
    pub wasserstein: f32,   // 1D Wasserstein-1 = sum |cumsum(P) - cumsum(Q)| over bins
    pub bhattacharyya: f32, // -ln( sum sqrt(P_b * Q_b) )  (eps-floored)
    pub hist_inter_dist: f32, // 1 - sum min(P_b, Q_b)
}

pub fn compute(sh: &Shared) -> Distributional {
    let n = PAGE_SIZE as f64;
    let eps = 1e-12_f64;

    let mut tv = 0.0;
    let mut chi2 = 0.0;
    let mut hell = 0.0;
    let mut kl = 0.0;
    let mut js = 0.0;
    let mut bc = 0.0;
    let mut inter = 0.0;
    let mut cp = 0.0;
    let mut cq = 0.0;
    let mut wass = 0.0;

    for b in 0..256 {
        let pb = sh.hp[b] as f64 / n;
        let qb = sh.hq[b] as f64 / n;
        let m = 0.5 * (pb + qb);

        tv += (pb - qb).abs();
        if pb + qb > 0.0 {
            chi2 += (pb - qb) * (pb - qb) / (pb + qb);
        }
        let s = pb.sqrt() - qb.sqrt();
        hell += s * s;
        if pb > 0.0 {
            kl += pb * (pb / qb.max(eps)).log2();
            js += 0.5 * pb * (pb / m).log2();
        }
        if qb > 0.0 {
            js += 0.5 * qb * (qb / m).log2();
        }
        bc += (pb * qb).sqrt();
        inter += pb.min(qb);

        cp += pb;
        cq += qb;
        wass += (cp - cq).abs();
    }

    Distributional {
        tv: (0.5 * tv) as f32,
        chi2: chi2 as f32,
        hellinger: (hell.sqrt() / std::f64::consts::SQRT_2) as f32,
        kl: kl as f32,
        js: js as f32,
        wasserstein: wass as f32,
        bhattacharyya: (-(bc.max(eps).ln())) as f32,
        hist_inter_dist: (1.0 - inter) as f32,
    }
}
