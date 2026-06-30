use std::env;
use chrono::Local;

use tokio::fs::File; // For asynchronous file operations.
use tokio::io::{self, AsyncReadExt, AsyncSeekExt, AsyncWriteExt}; // For asynchronous I/O traits.

use hamming::distance;
use distances::vectors::cosine;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use std::io::Write; // sync Write for the compressor (tokio's AsyncWriteExt is used elsewhere)
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync::OnceLock;

use std::sync::{Arc, Mutex}; // To enable shared access to file handles across threads
use rayon::prelude::*; // For parallel iterator

const CHUNK_SIZE: usize = 262144; // 256KB
const PAGE_SIZE: usize = 4096; // 4KB
const THREAD_COUNT: usize = 16; // Number of threads to be used for parallel processing

// Asynchronously read a chunk from a file at the given offset
async fn read_chunk(file: &mut File, offset: u64) -> io::Result<Vec<u8>> {
    let mut buffer = vec![0; CHUNK_SIZE];
    file.seek(io::SeekFrom::Start(offset)).await?;
    let n = file.read(&mut buffer).await?;
    buffer.truncate(n); // Adjust buffer size to actual bytes read
    Ok(buffer)
}

/// 256-bin byte histogram of a page (counts; sum = page length).
fn hist(page: &[u8]) -> [u32; 256] {
    let mut h = [0u32; 256];
    for &b in page {
        h[b as usize] += 1;
    }
    h
}

/// Total within-page byte gradient energy: sum of |page[j+1] - page[j]|.
/// A roughness proxy -- smooth pages (zeros / text) are low, jagged (random) high.
fn grad_energy(page: &[u8]) -> u64 {
    page.windows(2)
        .map(|w| (w[0] as i16 - w[1] as i16).unsigned_abs() as u64)
        .sum()
}

/// Compressed byte-length of a buffer (zlib/deflate). Basis of the informational metrics.
fn csize(data: &[u8]) -> usize {
    let mut e = ZlibEncoder::new(Vec::new(), Compression::default());
    e.write_all(data).unwrap();
    e.finish().unwrap().len()
}

/// Windowed-entropy "structural entropy" proxy: split the page into 256-byte windows,
/// take each window's byte entropy, return the spread (std-dev) of those window
/// entropies -- i.e. how spatially LUMPY the randomness is (a page with one encrypted
/// region scores high; uniform random or uniform text scores low). A lightweight
/// stand-in for the full wavelet structural entropy.
fn struct_entropy(page: &[u8]) -> f64 {
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

/// Lempel-Ziv (LZ76 / Kaspar-Schuster) complexity: the number of distinct factors in
/// the greedy LZ parsing. A direct algorithmic-complexity measure (no compressor).
fn lz_complexity(s: &[u8]) -> u32 {
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
fn pctl(h: &[u32; 256], total: usize, frac: f64) -> u8 {
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

/// Lowest / highest byte value present in the histogram (for range).
fn lo_bin(h: &[u32; 256]) -> u8 {
    (0..256).find(|&v| h[v] > 0).unwrap_or(0) as u8
}
fn hi_bin(h: &[u32; 256]) -> u8 {
    (0..256).rev().find(|&v| h[v] > 0).unwrap_or(0) as u8
}

/// Average (tie-corrected) rank of each byte value, from its histogram.
fn ranks(h: &[u32; 256]) -> [f64; 256] {
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
/// histogram of (p,q) value pairs. Kendall is counted via 2D prefix sums over the
/// 256x256 joint table (O(256^2)); ties are not corrected (tau-a).
fn spearman_kendall(p: &[u8], q: &[u8], hp: &[u32; 256], hq: &[u32; 256]) -> (f64, f64) {
    let n = p.len();
    let nf = n as f64;
    let rp = ranks(hp);
    let rq = ranks(hq);
    let mut j = vec![0u32; 256 * 256];
    for (&a, &b) in p.iter().zip(q.iter()) {
        j[a as usize * 256 + b as usize] += 1;
    }
    // Spearman = Pearson of the rank-transformed sequences.
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
    // Kendall tau-a: concordant - discordant over the joint table.
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
            let gt_gt = nn - p_a_last - p_last_b + p_a_b; // strictly greater in both
            conc += jab * gt_gt;
            let gt_lt = if b > 0 {
                pre[255 * 256 + (b - 1)] as i64 - pre[a * 256 + (b - 1)] as i64
            } else {
                0
            }; // greater in p, smaller in q
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

/// Shared 4096-point FFT plans (forward + inverse), built once, Send+Sync.
fn ffts() -> &'static (std::sync::Arc<dyn Fft<f32>>, std::sync::Arc<dyn Fft<f32>>) {
    static F: OnceLock<(std::sync::Arc<dyn Fft<f32>>, std::sync::Arc<dyn Fft<f32>>)> =
        OnceLock::new();
    F.get_or_init(|| {
        let mut pl = FftPlanner::<f32>::new();
        (pl.plan_fft_forward(PAGE_SIZE), pl.plan_fft_inverse(PAGE_SIZE))
    })
}

/// Spatial-shift metrics via FFT cross-correlation: the lag of the best raw
/// alignment, the phase-correlation peak (normalised cross-power), and the lag of
/// that peak. Detects content SHIFTED within the page (e.g. a memmove).
fn xcorr_metrics(p: &[u8], q: &[u8]) -> (f32, f32, f32) {
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
        best_ph / n as f32, // inverse FFT is unnormalised (scales by n)
        signed(lag_ph) as f32,
    )
}

/// Per-page feature substrate: one struct per (prev, curr) page pair.
///
/// CONVENTION: every channel is a CHANGE reading -- 0 = no change, larger = more
/// change. Identical pages short-circuit to all-zeros (see `page_metrics`).
/// Extend this family by family -- each new metric is one field here plus a few
/// lines in `page_metrics`. See docs/feature_substrate_spec.{md,pdf} and the
/// progress tracker docs/substrate_progress.html.
#[derive(Clone, Default, Debug)]
struct PageMetrics {
    hamming: u32,   // existing -- Family A / positional: bits flipped (popcount of p XOR q)
    cosine: f32,    // existing -- Family B / structure: cosine DISTANCE (0 = identical)
    // --- Family A: positional group (alternatives to Hamming) ---
    l0: u16,        // byte-Hamming: number of CHANGED bytes (0..=4096)
    l1: u32,        // SAD: sum of absolute byte differences
    l2: f32,        // Euclidean: sqrt(sum of squared byte differences)
    linf: u8,       // Chebyshev: the single biggest byte jump (0..=255)
    mean_abs: f32,  // L1 / 4096: average per-byte change
    gradient_mag: f32, // change in within-page roughness: grad_energy(q) - grad_energy(p)
    // --- Family A: distributional group (byte-histogram distances; 0 = identical histogram) ---
    tv: f32,            // Total Variation: 0.5 * sum |P_b - Q_b|
    chi2: f32,          // symmetric chi-square: sum (P_b - Q_b)^2 / (P_b + Q_b)
    hellinger: f32,     // (1/sqrt2) * sqrt(sum (sqrt(P_b) - sqrt(Q_b))^2)
    kl: f32,            // Kullback-Leibler: sum P_b * log2(P_b / Q_b)  (eps-smoothed)
    js: f32,            // Jensen-Shannon divergence (bounded [0,1])
    wasserstein: f32,   // 1D Wasserstein-1 = sum |cumsum(P) - cumsum(Q)| over bins
    bhattacharyya: f32, // -ln( sum sqrt(P_b * Q_b) )  (eps-floored)
    hist_inter_dist: f32, // 1 - sum min(P_b, Q_b)  (distance form of histogram intersection)
    // --- Family A: informational ---
    ent_delta: f32,         // entropy delta: H(q) - H(p)  (signed; bits/byte)
    csize_delta: f32,       // compressed-size delta: len(C(q)) - len(C(p))  (signed)
    ncd: f32,               // Normalized Compression Distance (0 = identical)
    struct_ent_change: f32, // change in windowed-entropy lumpiness (structural-entropy proxy)
    lz_change: f32,         // LZ76 factor-count change: lz(q) - lz(p)  (signed)
    // === Family B: direction of change (alternatives to cosine; moment + histogram part) ===
    // structure
    pearson: f32,        // centered cosine of the byte vectors
    ssim_struct: f32,    // SSIM structure term: (cov + c3) / (std_p*std_q + c3)
    // level
    mean_shift: f32,     // mean(q) - mean(p)  (signed)
    ssim_lum: f32,       // SSIM luminance term
    median_shift: f32,   // median(q) - median(p)  (signed)
    // spread
    var_ratio: f32,      // var(q) / var(p)
    std_delta: f32,      // std(q) - std(p)  (signed)
    ssim_contrast: f32,  // SSIM contrast term
    range_delta: f32,    // (max-min)(q) - (max-min)(p)  (signed)
    iqr_delta: f32,      // IQR(q) - IQR(p)  (signed)
    // polarity
    polarity: f32,       // (n_up - n_down) / n  (signed; direction of the byte current)
    sign_delta_ent: f32, // entropy of the up/down/same pattern (direction coherence)
    // distributional-direction
    zero_mass_delta: f32, // Q_0 - P_0  (zero-byte mass change; signed)
    // structure (rank-based) + spatial-shift (FFT)
    spearman: f32,        // Spearman rank correlation
    kendall: f32,         // Kendall tau-a (ties not corrected)
    cross_corr_lag: f32,  // lag (signed) of max raw circular cross-correlation
    phase_corr: f32,      // peak of the phase-correlation (normalised cross-power)
    byte_rotation: f32,   // lag (signed) of the phase-correlation peak
    // NOTE: ent_delta_sign = sign(ent_delta), hist_mean_shift_sign = sign(mean_shift),
    // move_toward_uniform = -ent_delta, net_drift = mean_shift * 4096 are EXACT functions of
    // stored columns -> derived offline, not stored.
}

/// Compute every per-page metric for one page pair in a single set of passes.
fn page_metrics(p: &[u8], q: &[u8]) -> PageMetrics {
    let hamming = distance(p, q) as u32;
    if hamming == 0 {
        // Identical page = no change. Every channel is 0; skip all the work
        // (unchanged pages are the bulk of a real dump).
        return PageMetrics::default();
    }

    // Family B -- cosine DISTANCE (the page differs here, so norms are well defined
    // unless one side is all-zero, which the `distances` crate maps to 1 = max).
    let p_f32: Vec<f32> = p.iter().map(|&x| x as f32).collect();
    let q_f32: Vec<f32> = q.iter().map(|&x| x as f32).collect();
    let cos = cosine(&p_f32, &q_f32);

    // Family A -- positional group: one byte-wise pass.
    let mut l0: u16 = 0;
    let mut l1: u64 = 0;
    let mut sse: u64 = 0;
    let mut linf: u8 = 0;
    // Family B moment accumulators (means, variances, covariance, polarity).
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
    let gradient_mag = grad_energy(q) as f32 - grad_energy(p) as f32;

    // Family A -- distributional group + entropy: from the two byte histograms.
    let hp = hist(p);
    let hq = hist(q);
    let n = PAGE_SIZE as f64;
    let eps = 1e-12_f64;

    let mut tv = 0.0;
    let mut chi2 = 0.0;
    let mut hell = 0.0; // sum of (sqrt(p)-sqrt(q))^2, rooted at the end
    let mut kl = 0.0;
    let mut js = 0.0;
    let mut bc = 0.0; // Bhattacharyya coefficient
    let mut inter = 0.0;
    let mut hp_ent = 0.0;
    let mut hq_ent = 0.0;
    let mut cp = 0.0; // running CDFs for Wasserstein
    let mut cq = 0.0;
    let mut wass = 0.0;

    for b in 0..256 {
        let pb = hp[b] as f64 / n;
        let qb = hq[b] as f64 / n;
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
            hp_ent -= pb * pb.log2();
        }
        if qb > 0.0 {
            js += 0.5 * qb * (qb / m).log2();
            hq_ent -= qb * qb.log2();
        }
        bc += (pb * qb).sqrt();
        inter += pb.min(qb);

        cp += pb;
        cq += qb;
        wass += (cp - cq).abs();
    }

    // Family A -- informational (compression-based). Only changed pages reach here.
    let csz_p = csize(p);
    let csz_q = csize(q);
    let mut pq = Vec::with_capacity(p.len() + q.len());
    pq.extend_from_slice(p);
    pq.extend_from_slice(q);
    let csz_pq = csize(&pq);
    let ncd = (csz_pq as f64 - csz_p.min(csz_q) as f64) / csz_p.max(csz_q).max(1) as f64;

    // === Family B: direction -- moments (from the byte loop) + histogram percentiles ===
    let nf = PAGE_SIZE as f64;
    let mp = sp as f64 / nf;
    let mq = sq as f64 / nf;
    let vp = (spp as f64 / nf - mp * mp).max(0.0);
    let vq = (sqq as f64 / nf - mq * mq).max(0.0);
    let cov = spq as f64 / nf - mp * mq;
    let sdp = vp.sqrt();
    let sdq = vq.sqrt();
    // SSIM stabilising constants for 8-bit data (dynamic range L = 255).
    let c1 = (0.01 * 255.0f64).powi(2);
    let c2 = (0.03 * 255.0f64).powi(2);
    let c3 = c2 / 2.0;
    let pearson = if vp > 0.0 && vq > 0.0 { cov / (sdp * sdq) } else { 0.0 };
    let var_ratio = if vp > 0.0 { vq / vp } else { 0.0 };
    let pu = n_up as f64 / nf;
    let pd = n_down as f64 / nf;
    let psame = 1.0 - pu - pd;
    let mut sde = 0.0;
    for &pr in &[pu, pd, psame] {
        if pr > 0.0 {
            sde -= pr * pr.log2();
        }
    }
    let range_p = hi_bin(&hp) as f64 - lo_bin(&hp) as f64;
    let range_q = hi_bin(&hq) as f64 - lo_bin(&hq) as f64;
    let iqr_p = pctl(&hp, PAGE_SIZE, 0.75) as f64 - pctl(&hp, PAGE_SIZE, 0.25) as f64;
    let iqr_q = pctl(&hq, PAGE_SIZE, 0.75) as f64 - pctl(&hq, PAGE_SIZE, 0.25) as f64;
    let (spearman, kendall) = spearman_kendall(p, q, &hp, &hq);
    let (cross_corr_lag, phase_corr, byte_rotation) = xcorr_metrics(p, q);

    PageMetrics {
        hamming,
        cosine: cos,
        l0,
        l1: l1 as u32,
        l2: (sse as f32).sqrt(),
        linf,
        mean_abs: l1 as f32 / PAGE_SIZE as f32,
        gradient_mag,
        tv: (0.5 * tv) as f32,
        chi2: chi2 as f32,
        hellinger: (hell.sqrt() / std::f64::consts::SQRT_2) as f32,
        kl: kl as f32,
        js: js as f32,
        wasserstein: wass as f32,
        bhattacharyya: (-(bc.max(eps).ln())) as f32,
        hist_inter_dist: (1.0 - inter) as f32,
        ent_delta: (hq_ent - hp_ent) as f32,
        csize_delta: csz_q as f32 - csz_p as f32,
        ncd: ncd as f32,
        struct_ent_change: (struct_entropy(q) - struct_entropy(p)) as f32,
        lz_change: lz_complexity(q) as f32 - lz_complexity(p) as f32,
        // Family B
        pearson: pearson as f32,
        ssim_struct: ((cov + c3) / (sdp * sdq + c3)) as f32,
        mean_shift: (mq - mp) as f32,
        ssim_lum: ((2.0 * mp * mq + c1) / (mp * mp + mq * mq + c1)) as f32,
        median_shift: (pctl(&hq, PAGE_SIZE, 0.5) as f64 - pctl(&hp, PAGE_SIZE, 0.5) as f64) as f32,
        var_ratio: var_ratio as f32,
        std_delta: (sdq - sdp) as f32,
        ssim_contrast: ((2.0 * sdp * sdq + c2) / (vp + vq + c2)) as f32,
        range_delta: (range_q - range_p) as f32,
        iqr_delta: (iqr_q - iqr_p) as f32,
        polarity: ((n_up as f64 - n_down as f64) / nf) as f32,
        sign_delta_ent: sde as f32,
        zero_mass_delta: ((hq[0] as f64 - hp[0] as f64) / nf) as f32,
        spearman: spearman as f32,
        kendall: kendall as f32,
        cross_corr_lag,
        phase_corr,
        byte_rotation,
    }
}

/// One PageMetrics per page in the chunk.
fn process_chunk(chunk1: &[u8], chunk2: &[u8]) -> Vec<PageMetrics> {
    let num_pages = chunk1.len() / PAGE_SIZE;
    (0..num_pages)
        .map(|i| {
            let start = i * PAGE_SIZE;
            let end = start + PAGE_SIZE;
            page_metrics(&chunk1[start..end], &chunk2[start..end])
        })
        .collect()
}

#[tokio::main]
async fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4  {
        eprintln!("Usage: {} <prev_image> <new_image> <output_dir>", args[0]);
        std::process::exit(1);
    }

    let prev_image_path = &args[1];
    let new_image_path = &args[2];
    let output_dir = &args[3];

    let file1_path = prev_image_path;
    let file2_path = new_image_path;

    let timestamp = Local::now().format("%Y%m%d%H%M%S").to_string();

    //let hamming_result_file_path = format!("C:\\Users\\jeries\\Desktop\\thesis\\results\\1\\hamming\\memory_dump_hamming_results_par-{}.txt", timestamp);
    //let cosine_result_file_path = format!("C:\\Users\\jeries\\Desktop\\thesis\\results\\1\\cosine\\memory_dump_cosine_results_par-{}.txt", timestamp);
    let hamming_result_file_path = format!("{}/hamming/memory_dump_hamming_results_par-{}.txt", output_dir, timestamp);
    let cosine_result_file_path = format!("{}/cosine/memory_dump_cosine_results_par-{}.txt", output_dir, timestamp);
    // Combined per-page substrate: one CSV, one row per page, one column per metric.
    let metrics_csv_path = format!("{}/metrics/page_metrics-{}.csv", output_dir, timestamp);

    // Ensure output subdirs exist (additive; harmless if already present).
    for sub in ["hamming", "cosine", "metrics"] {
        let _ = std::fs::create_dir_all(format!("{}/{}", output_dir, sub));
    }

    let file1 = Arc::new(Mutex::new(File::open(file1_path).await?));
    let file2 = Arc::new(Mutex::new(File::open(file2_path).await?));
    let hamming_result_file = Arc::new(Mutex::new(File::create(hamming_result_file_path).await?));
    let cosine_result_file = Arc::new(Mutex::new(File::create(cosine_result_file_path).await?));
    let metrics_result_file = Arc::new(Mutex::new(File::create(metrics_csv_path).await?));

    // Calculate the total size of the files
    let file1_size = file1.lock().unwrap().metadata().await?.len();
    let file2_size = file2.lock().unwrap().metadata().await?.len();

    assert_eq!(file1_size, file2_size, "Files should be of the same size");

    // Calculate the segment size for each thread
    let segment_size = file1_size / THREAD_COUNT as u64;

    let result_vecs: Arc<Mutex<Vec<Vec<PageMetrics>>>> =
        Arc::new(Mutex::new(vec![Vec::new(); THREAD_COUNT]));

    // Spawn multiple threads for parallel processing using Rayon
    (0..THREAD_COUNT).into_par_iter().for_each(|thread_id| {
        let file1 = Arc::clone(&file1);
        let file2 = Arc::clone(&file2);
        let result_vecs = Arc::clone(&result_vecs);

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async move {
            let start_offset = thread_id as u64 * segment_size;
            let end_offset = if thread_id == THREAD_COUNT - 1 {
                file1_size // Last thread processes the remaining part
            } else {
                start_offset + segment_size
            };

            let mut offset = start_offset;
            let mut local_results: Vec<PageMetrics> = Vec::new(); // per-thread, reduces lock contention

            while offset < end_offset {
                let chunk1 = read_chunk(&mut file1.lock().unwrap(), offset).await.unwrap_or_else(|_| vec![]);
                let chunk2 = read_chunk(&mut file2.lock().unwrap(), offset).await.unwrap_or_else(|_| vec![]);

                if chunk1.is_empty() || chunk2.is_empty() {
                    break; // Exit loop if either file has no more data
                }

                local_results.extend(process_chunk(&chunk1, &chunk2));

                offset += CHUNK_SIZE  as u64;
            }

            // Write local results to shared vector
            result_vecs.lock().unwrap()[thread_id] = local_results;
        });
    });

    // Write the accumulated results to the output files
    let mut hamming_result_file = hamming_result_file.lock().unwrap();
    let mut cosine_result_file = cosine_result_file.lock().unwrap();
    let mut metrics_result_file = metrics_result_file.lock().unwrap();

    let result_vecs = Arc::try_unwrap(result_vecs).unwrap().into_inner().unwrap();

    let mut hamming_buffer = String::new();
    let mut cosine_buffer = String::new();
    // Header row for the combined substrate CSV (one column per metric).
    let mut metrics_buffer = String::from(
        "hamming,cosine,l0,l1,l2,linf,mean_abs,gradient_mag,tv,chi2,hellinger,kl,js,wasserstein,bhattacharyya,hist_inter_dist,ent_delta,csize_delta,ncd,struct_ent_change,lz_change,pearson,ssim_struct,mean_shift,ssim_lum,median_shift,var_ratio,std_delta,ssim_contrast,range_delta,iqr_delta,polarity,sign_delta_ent,zero_mass_delta,spearman,kendall,cross_corr_lag,phase_corr,byte_rotation\n",
    );

    for result_vec in &result_vecs {
        for m in result_vec.iter() {
            // Backward-compatible single-metric files (unchanged format).
            hamming_buffer.push_str(&format!("{}\n", m.hamming));
            cosine_buffer.push_str(&format!("{}\n", m.cosine));
            // Combined substrate row: all metrics for this page.
            metrics_buffer.push_str(&format!(
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},\
                 {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                m.hamming, m.cosine, m.l0, m.l1, m.l2, m.linf, m.mean_abs, m.gradient_mag,
                m.tv, m.chi2, m.hellinger, m.kl, m.js, m.wasserstein, m.bhattacharyya,
                m.hist_inter_dist, m.ent_delta, m.csize_delta, m.ncd, m.struct_ent_change,
                m.lz_change, m.pearson, m.ssim_struct, m.mean_shift, m.ssim_lum, m.median_shift,
                m.var_ratio, m.std_delta, m.ssim_contrast, m.range_delta, m.iqr_delta,
                m.polarity, m.sign_delta_ent, m.zero_mass_delta, m.spearman, m.kendall,
                m.cross_corr_lag, m.phase_corr, m.byte_rotation
            ));
        }
    }

    hamming_result_file.write_all(hamming_buffer.as_bytes()).await?;
    cosine_result_file.write_all(cosine_buffer.as_bytes()).await?;
    metrics_result_file.write_all(metrics_buffer.as_bytes()).await?;

    Ok(())
}
