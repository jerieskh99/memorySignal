//! Shared kernel: the precompute (`Shared`) used by every family, plus the stateless
//! helpers (histograms, compression, entropy, ranks, FFT) the families build on.

use flate2::write::ZlibEncoder;
use flate2::Compression;
use std::io::Write;

use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync::OnceLock;

use crate::metrics::PAGE_SIZE;

/// Precomputed-once-per-page primitives shared across families: the single byte-loop
/// outputs, the moment accumulators, and the two byte histograms.
#[derive(Clone, Debug)]
pub struct Shared {
    pub hamming: u32,
    // positional byte-loop outputs
    pub l0: u16,
    pub l1: u64,
    pub sse: u64,
    pub linf: u8,
    // moment accumulators
    pub sp: u64,
    pub sq: u64,
    pub spp: u64,
    pub sqq: u64,
    pub spq: u64,
    pub n_up: u32,
    pub n_down: u32,
    // byte histograms
    pub hp: [u32; 256],
    pub hq: [u32; 256],
}

/// Derived byte moments (means, variances, covariance, std-devs).
pub struct Moments {
    pub mp: f64,
    pub mq: f64,
    pub vp: f64,
    pub vq: f64,
    pub cov: f64,
    pub sdp: f64,
    pub sdq: f64,
}

impl Shared {
    /// One byte-wise pass (positional + moments) and the two histograms. `hamming` is
    /// passed in (already computed for the identical-page guard).
    pub fn new(p: &[u8], q: &[u8], hamming: u32) -> Shared {
        let mut l0: u16 = 0;
        let mut l1: u64 = 0;
        let mut sse: u64 = 0;
        let mut linf: u8 = 0;
        let (mut sp, mut sq, mut spp, mut sqq, mut spq) = (0u64, 0u64, 0u64, 0u64, 0u64);
        let (mut n_up, mut n_down) = (0u32, 0u32);
        for (&a, &b) in p.iter().zip(q.iter()) {
            let d = (a as i16 - b as i16).unsigned_abs(); // 0..=255
            if d != 0 {
                l0 += 1;
            }
            l1 += d as u64;
            sse += (d as u64) * (d as u64);
            if d as u8 > linf {
                linf = d as u8;
            }
            sp += a as u64;
            sq += b as u64;
            spp += (a as u64) * (a as u64);
            sqq += (b as u64) * (b as u64);
            spq += (a as u64) * (b as u64);
            if b > a {
                n_up += 1;
            } else if b < a {
                n_down += 1;
            }
        }
        Shared {
            hamming,
            l0,
            l1,
            sse,
            linf,
            sp,
            sq,
            spp,
            sqq,
            spq,
            n_up,
            n_down,
            hp: hist(p),
            hq: hist(q),
        }
    }

    pub fn moments(&self) -> Moments {
        let nf = PAGE_SIZE as f64;
        let mp = self.sp as f64 / nf;
        let mq = self.sq as f64 / nf;
        let vp = (self.spp as f64 / nf - mp * mp).max(0.0);
        let vq = (self.sqq as f64 / nf - mq * mq).max(0.0);
        let cov = self.spq as f64 / nf - mp * mq;
        Moments {
            mp,
            mq,
            vp,
            vq,
            cov,
            sdp: vp.sqrt(),
            sdq: vq.sqrt(),
        }
    }
}

/// SSIM stabilising constants for 8-bit data (dynamic range L = 255): (c1, c2, c3).
pub fn ssim_consts() -> (f64, f64, f64) {
    let c2 = (0.03 * 255.0f64).powi(2);
    ((0.01 * 255.0f64).powi(2), c2, c2 / 2.0)
}

/// 256-bin byte histogram of a page (counts; sum = page length).
pub fn hist(page: &[u8]) -> [u32; 256] {
    let mut h = [0u32; 256];
    for &b in page {
        h[b as usize] += 1;
    }
    h
}

/// Total within-page byte gradient energy: sum of |page[j+1] - page[j]|.
pub fn grad_energy(page: &[u8]) -> u64 {
    page.windows(2)
        .map(|w| (w[0] as i16 - w[1] as i16).unsigned_abs() as u64)
        .sum()
}

/// Compressed byte-length of a buffer (zlib/deflate).
pub fn csize(data: &[u8]) -> usize {
    let mut e = ZlibEncoder::new(Vec::new(), Compression::default());
    e.write_all(data).unwrap();
    e.finish().unwrap().len()
}

/// Windowed-entropy "structural entropy" proxy: std-dev of per-256-byte-window byte
/// entropies (spatial lumpiness of randomness).
pub fn struct_entropy(page: &[u8]) -> f64 {
    let mut es: Vec<f64> = Vec::new();
    for chunk in page.chunks(256) {
        let mut h = [0u32; 256];
        for &b in chunk {
            h[b as usize] += 1;
        }
        let m = chunk.len() as f64;
        let mut e = 0.0;
        for &cnt in h.iter() {
            if cnt > 0 {
                let pr = cnt as f64 / m;
                e -= pr * pr.log2();
            }
        }
        es.push(e);
    }
    if es.is_empty() {
        return 0.0;
    }
    let mean = es.iter().sum::<f64>() / es.len() as f64;
    let var = es.iter().map(|e| (e - mean) * (e - mean)).sum::<f64>() / es.len() as f64;
    var.sqrt()
}

/// Lempel-Ziv (LZ76 / Kaspar-Schuster) complexity: number of distinct factors.
pub fn lz_complexity(s: &[u8]) -> u32 {
    let n = s.len();
    if n == 0 {
        return 0;
    }
    let (mut c, mut l, mut i, mut k, mut k_max) = (1u32, 1usize, 0usize, 1usize, 1usize);
    loop {
        if l + k - 1 >= n {
            if k > 1 {
                c += 1;
            }
            break;
        }
        if s[i + k - 1] == s[l + k - 1] {
            k += 1;
        } else {
            if k > k_max {
                k_max = k;
            }
            i += 1;
            if i == l {
                c += 1;
                l += k_max;
                if l >= n {
                    break;
                }
                i = 0;
                k = 1;
                k_max = 1;
            } else {
                k = 1;
            }
        }
    }
    c
}

/// Smallest byte value whose cumulative histogram mass reaches `frac` of total.
pub fn pctl(h: &[u32; 256], total: usize, frac: f64) -> u8 {
    let target = (frac * total as f64).ceil() as u64;
    let mut cum: u64 = 0;
    for (v, &cnt) in h.iter().enumerate() {
        cum += cnt as u64;
        if cum >= target {
            return v as u8;
        }
    }
    255
}

pub fn lo_bin(h: &[u32; 256]) -> u8 {
    (0..256).find(|&v| h[v] > 0).unwrap_or(0) as u8
}
pub fn hi_bin(h: &[u32; 256]) -> u8 {
    (0..256).rev().find(|&v| h[v] > 0).unwrap_or(0) as u8
}

/// Average (tie-corrected) rank of each byte value, from its histogram.
pub fn ranks(h: &[u32; 256]) -> [f64; 256] {
    let mut r = [0.0f64; 256];
    let mut cum: u64 = 0;
    for v in 0..256 {
        let c = h[v] as u64;
        if c > 0 {
            r[v] = cum as f64 + (c as f64 + 1.0) / 2.0;
        }
        cum += c;
    }
    r
}

/// Spearman rank correlation + Kendall tau-a, from the byte histograms and a joint
/// histogram of (p,q) value pairs (2D prefix sums over the 256x256 table).
pub fn spearman_kendall(p: &[u8], q: &[u8], hp: &[u32; 256], hq: &[u32; 256]) -> (f64, f64) {
    let n = p.len();
    let nf = n as f64;
    let rp = ranks(hp);
    let rq = ranks(hq);
    let mut j = vec![0u32; 256 * 256];
    for (&a, &b) in p.iter().zip(q.iter()) {
        j[a as usize * 256 + b as usize] += 1;
    }
    let (mut srp, mut srq, mut srpp, mut srqq, mut srpq) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for v in 0..256 {
        srp += hp[v] as f64 * rp[v];
        srq += hq[v] as f64 * rq[v];
        srpp += hp[v] as f64 * rp[v] * rp[v];
        srqq += hq[v] as f64 * rq[v] * rq[v];
    }
    for a in 0..256 {
        for b in 0..256 {
            let c = j[a * 256 + b];
            if c != 0 {
                srpq += c as f64 * rp[a] * rq[b];
            }
        }
    }
    let (mrp, mrq) = (srp / nf, srq / nf);
    let vrp = (srpp / nf - mrp * mrp).max(0.0);
    let vrq = (srqq / nf - mrq * mrq).max(0.0);
    let covr = srpq / nf - mrp * mrq;
    let spearman = if vrp > 0.0 && vrq > 0.0 {
        covr / (vrp.sqrt() * vrq.sqrt())
    } else {
        0.0
    };
    let mut pre = vec![0u64; 256 * 256];
    for a in 0..256 {
        for b in 0..256 {
            let cur = j[a * 256 + b] as u64;
            let up = if a > 0 { pre[(a - 1) * 256 + b] } else { 0 };
            let left = if b > 0 { pre[a * 256 + (b - 1)] } else { 0 };
            let diag = if a > 0 && b > 0 { pre[(a - 1) * 256 + (b - 1)] } else { 0 };
            pre[a * 256 + b] = cur + up + left - diag;
        }
    }
    let nn = n as i64;
    let (mut conc, mut disc): (i64, i64) = (0, 0);
    for a in 0..256 {
        for b in 0..256 {
            let jab = j[a * 256 + b] as i64;
            if jab == 0 {
                continue;
            }
            let p_a_last = pre[a * 256 + 255] as i64;
            let p_last_b = pre[255 * 256 + b] as i64;
            let p_a_b = pre[a * 256 + b] as i64;
            let gt_gt = nn - p_a_last - p_last_b + p_a_b;
            conc += jab * gt_gt;
            let gt_lt = if b > 0 {
                pre[255 * 256 + (b - 1)] as i64 - pre[a * 256 + (b - 1)] as i64
            } else {
                0
            };
            disc += jab * gt_lt;
        }
    }
    let denom = nn * (nn - 1) / 2;
    let kendall = if denom > 0 {
        (conc - disc) as f64 / denom as f64
    } else {
        0.0
    };
    (spearman, kendall)
}

/// Spearman rank correlation only, computed in O(n) with no joint histogram and no
/// Kendall (the `--fast` path). Returns the same value as `spearman_kendall().0`.
pub fn spearman_fast(p: &[u8], q: &[u8], hp: &[u32; 256], hq: &[u32; 256]) -> f64 {
    let n = p.len();
    let nf = n as f64;
    let rp = ranks(hp);
    let rq = ranks(hq);
    let (mut srp, mut srq, mut srpp, mut srqq) = (0.0, 0.0, 0.0, 0.0);
    for v in 0..256 {
        srp += hp[v] as f64 * rp[v];
        srq += hq[v] as f64 * rq[v];
        srpp += hp[v] as f64 * rp[v] * rp[v];
        srqq += hq[v] as f64 * rq[v] * rq[v];
    }
    let mut srpq = 0.0;
    for (&a, &b) in p.iter().zip(q.iter()) {
        srpq += rp[a as usize] * rq[b as usize];
    }
    let (mrp, mrq) = (srp / nf, srq / nf);
    let vrp = (srpp / nf - mrp * mrp).max(0.0);
    let vrq = (srqq / nf - mrq * mrq).max(0.0);
    let covr = srpq / nf - mrp * mrq;
    if vrp > 0.0 && vrq > 0.0 {
        covr / (vrp.sqrt() * vrq.sqrt())
    } else {
        0.0
    }
}

/// Shared 4096-point FFT plans (forward + inverse), built once, Send+Sync.
fn ffts() -> &'static (std::sync::Arc<dyn Fft<f32>>, std::sync::Arc<dyn Fft<f32>>) {
    static F: OnceLock<(std::sync::Arc<dyn Fft<f32>>, std::sync::Arc<dyn Fft<f32>>)> =
        OnceLock::new();
    F.get_or_init(|| {
        let mut pl = FftPlanner::<f32>::new();
        (pl.plan_fft_forward(PAGE_SIZE), pl.plan_fft_inverse(PAGE_SIZE))
    })
}

/// Spatial-shift metrics via FFT cross-correlation: (cross_corr_lag, phase_corr, byte_rotation).
pub fn xcorr_metrics(p: &[u8], q: &[u8]) -> (f32, f32, f32) {
    let n = p.len();
    let (fwd, inv) = ffts();
    let mut fp: Vec<Complex<f32>> = p.iter().map(|&x| Complex::new(x as f32, 0.0)).collect();
    let mut fq: Vec<Complex<f32>> = q.iter().map(|&x| Complex::new(x as f32, 0.0)).collect();
    fwd.process(&mut fp);
    fwd.process(&mut fq);
    let mut raw: Vec<Complex<f32>> = (0..n).map(|i| fp[i] * fq[i].conj()).collect();
    let mut phase: Vec<Complex<f32>> = raw
        .iter()
        .map(|&c| {
            let m = c.norm();
            if m > 1e-12 {
                c / m
            } else {
                Complex::new(0.0, 0.0)
            }
        })
        .collect();
    inv.process(&mut raw);
    inv.process(&mut phase);
    let signed = |k: usize| -> i64 {
        if k > n / 2 {
            k as i64 - n as i64
        } else {
            k as i64
        }
    };
    let (mut best_raw, mut lag_raw) = (f32::MIN, 0usize);
    let (mut best_ph, mut lag_ph) = (f32::MIN, 0usize);
    for i in 0..n {
        if raw[i].re > best_raw {
            best_raw = raw[i].re;
            lag_raw = i;
        }
        if phase[i].re > best_ph {
            best_ph = phase[i].re;
            lag_ph = i;
        }
    }
    (
        signed(lag_raw) as f32,
        best_ph / n as f32,
        signed(lag_ph) as f32,
    )
}

/// Shannon entropy (bits) of the adjacent-byte-pair distribution of a page -- local
/// structure (pointers / tables vs random).
pub fn bigram_entropy(page: &[u8]) -> f64 {
    let mut bg = vec![0u32; 256 * 256];
    for w in page.windows(2) {
        bg[w[0] as usize * 256 + w[1] as usize] += 1;
    }
    let total = page.len().saturating_sub(1) as f64;
    if total <= 0.0 {
        return 0.0;
    }
    let mut e = 0.0;
    for &c in bg.iter() {
        if c > 0 {
            let pr = c as f64 / total;
            e -= pr * pr.log2();
        }
    }
    e
}

/// Peak of the (normalised) circular autocorrelation of a page at lag in [1, n/2) --
/// within-page periodicity / self-similarity (0 = aperiodic, ~1 = periodic/constant).
pub fn autocorr_peak(page: &[u8]) -> f32 {
    let n = page.len();
    let (fwd, inv) = ffts();
    let mut f: Vec<Complex<f32>> = page.iter().map(|&x| Complex::new(x as f32, 0.0)).collect();
    fwd.process(&mut f);
    let mut power: Vec<Complex<f32>> = f.iter().map(|&c| Complex::new(c.norm_sqr(), 0.0)).collect();
    inv.process(&mut power);
    let ac0 = power[0].re;
    if ac0 <= 0.0 {
        return 0.0;
    }
    let mut best = 0.0f32;
    for k in 1..n / 2 {
        let v = power[k].re / ac0;
        if v > best {
            best = v;
        }
    }
    best
}

/// Haralick texture features from the adjacent-byte co-occurrence matrix of a page:
/// (contrast, homogeneity, energy/ASM, correlation).
pub fn glcm_haralick(page: &[u8]) -> (f64, f64, f64, f64) {
    let mut g = vec![0u32; 256 * 256];
    let mut total: u64 = 0;
    for w in page.windows(2) {
        g[w[0] as usize * 256 + w[1] as usize] += 1;
        total += 1;
    }
    if total == 0 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let t = total as f64;
    let (mut contrast, mut homog, mut energy, mut mu_i, mut mu_j) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 0..256 {
        for j in 0..256 {
            let c = g[i * 256 + j];
            if c == 0 {
                continue;
            }
            let pij = c as f64 / t;
            let di = i as f64 - j as f64;
            contrast += di * di * pij;
            homog += pij / (1.0 + di.abs());
            energy += pij * pij;
            mu_i += i as f64 * pij;
            mu_j += j as f64 * pij;
        }
    }
    let (mut si, mut sj, mut cor) = (0.0, 0.0, 0.0);
    for i in 0..256 {
        for j in 0..256 {
            let c = g[i * 256 + j];
            if c == 0 {
                continue;
            }
            let pij = c as f64 / t;
            si += (i as f64 - mu_i).powi(2) * pij;
            sj += (j as f64 - mu_j).powi(2) * pij;
            cor += (i as f64 - mu_i) * (j as f64 - mu_j) * pij;
        }
    }
    let (si, sj) = (si.sqrt(), sj.sqrt());
    let correlation = if si > 0.0 && sj > 0.0 { cor / (si * sj) } else { 0.0 };
    (contrast, homog, energy, correlation)
}

/// Fraction of a page's spectral energy (excluding DC) in the high-frequency band
/// (k in [n/4, 3n/4)). Smooth pages low, fine-textured / high-frequency pages high.
pub fn high_freq_frac(page: &[u8]) -> f32 {
    let n = page.len();
    let (fwd, _) = ffts();
    let mut f: Vec<Complex<f32>> = page.iter().map(|&x| Complex::new(x as f32, 0.0)).collect();
    fwd.process(&mut f);
    let (mut total, mut high) = (0.0f64, 0.0f64);
    for k in 1..n {
        let p = f[k].norm_sqr() as f64;
        total += p;
        if k >= n / 4 && k < 3 * n / 4 {
            high += p;
        }
    }
    if total > 0.0 {
        (high / total) as f32
    } else {
        0.0
    }
}
