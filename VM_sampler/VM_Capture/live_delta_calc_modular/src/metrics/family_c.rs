//! Family C -- content / character of the NEW page q (reference-free; describes q itself).
//! Gated behind the changed-page short-circuit like every family: an unchanged page is
//! all-zeros. These describe the content of pages that changed.

use crate::metrics::common::{autocorr_peak, bigram_entropy, struct_entropy, Shared};
use crate::metrics::PAGE_SIZE;

#[derive(Clone, Default, Debug)]
pub struct Content {
    pub ent_q: f32,          // Shannon entropy of q (bits/byte): ~8 random, ~0 constant
    pub struct_ent_q: f32,   // windowed-entropy lumpiness of q
    pub distinct_bytes: u16, // palette size (distinct byte values present)
    pub zero_frac: f32,      // fraction of 0x00 bytes
    pub fill_frac: f32,      // fraction of the most common byte value (constant-fill tell)
    pub printable_frac: f32, // fraction of printable ASCII (0x20..=0x7E): text-likeness
    pub mean_q: f32,         // byte mean
    pub var_q: f32,          // byte variance
    pub skew_q: f32,         // byte skewness
    pub kurt_q: f32,         // byte excess kurtosis
    pub chi2_uniform: f32,   // chi-square deviation from a uniform histogram (encryption tell)
    pub bigram_ent: f32,     // adjacent-byte-pair entropy (local structure)
    pub autocorr_peak: f32,  // within-page periodicity / self-similarity
}

pub fn compute(sh: &Shared, q: &[u8]) -> Content {
    let n = PAGE_SIZE as f64;
    let hq = &sh.hq;

    // Single histogram pass: entropy, distinct, fill, printable, mean.
    let mut ent = 0.0;
    let mut distinct: u16 = 0;
    let mut fill_max: u32 = 0;
    let mut printable: u64 = 0;
    let mut mean = 0.0;
    for b in 0..256 {
        let c = hq[b];
        if c > 0 {
            distinct += 1;
            if c > fill_max {
                fill_max = c;
            }
            let pb = c as f64 / n;
            ent -= pb * pb.log2();
        }
        if (0x20..=0x7E).contains(&b) {
            printable += hq[b] as u64;
        }
        mean += b as f64 * hq[b] as f64;
    }
    mean /= n;

    // Central moments (variance / skewness / kurtosis) from the histogram.
    let (mut m2, mut m3, mut m4) = (0.0, 0.0, 0.0);
    for b in 0..256 {
        let d = b as f64 - mean;
        let w = hq[b] as f64;
        m2 += w * d * d;
        m3 += w * d * d * d;
        m4 += w * d * d * d * d;
    }
    m2 /= n;
    m3 /= n;
    m4 /= n;
    let skew = if m2 > 0.0 { m3 / m2.powf(1.5) } else { 0.0 };
    let kurt = if m2 > 0.0 { m4 / (m2 * m2) - 3.0 } else { 0.0 };

    // Chi-square vs uniform (expected n/256 per bin).
    let e = n / 256.0;
    let mut chi2u = 0.0;
    for b in 0..256 {
        let dd = hq[b] as f64 - e;
        chi2u += dd * dd / e;
    }

    Content {
        ent_q: ent as f32,
        struct_ent_q: struct_entropy(q) as f32,
        distinct_bytes: distinct,
        zero_frac: (hq[0] as f64 / n) as f32,
        fill_frac: (fill_max as f64 / n) as f32,
        printable_frac: (printable as f64 / n) as f32,
        mean_q: mean as f32,
        var_q: m2 as f32,
        skew_q: skew as f32,
        kurt_q: kurt as f32,
        chi2_uniform: chi2u as f32,
        bigram_ent: bigram_entropy(q) as f32,
        autocorr_peak: autocorr_peak(q),
    }
}
