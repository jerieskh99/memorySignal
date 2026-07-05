//! Per-page feature substrate. Composition root: builds the shared precompute once,
//! then asks each family to compute its slice, and assembles `PageMetrics`.
//!
//! CONVENTION: every channel is a CHANGE reading -- 0 = no change, larger = more
//! change. Identical pages short-circuit to all-zeros (see `page_metrics`).

pub mod common;
pub mod family_a;
pub mod family_b;
pub mod family_c;
pub mod family_d;

use common::Shared;
use hamming::distance;

pub const PAGE_SIZE: usize = 4096; // 4KB

/// One row of the substrate: Family A (amount) + Family B (direction), composed.
#[derive(Clone, Default, Debug)]
pub struct PageMetrics {
    pub a: family_a::Amount,
    pub b: family_b::Direction,
    pub c: family_c::Content,
    pub d: family_d::Internal,
}

impl PageMetrics {
    /// Backward-compatible single-metric accessors (for the legacy hamming/cosine files).
    pub fn hamming(&self) -> u32 {
        self.a.positional.hamming
    }
    pub fn cosine(&self) -> f32 {
        self.b.structure.cosine
    }
}

/// Compute every per-page metric for one page pair (full 64-column substrate = speed 0).
pub fn page_metrics(p: &[u8], q: &[u8]) -> PageMetrics {
    page_metrics_mode(p, q, 0)
}

/// Compute the per-page metrics at a completeness/speed level (0 = full, higher = faster,
/// fewer metrics). Dropped metrics emit 0; the 64-column schema is identical at every
/// level. Cumulative:
///   >= 1: drop lz_change   (O(n^2) LZ76, the dominant cost)
///   >= 2: drop the heavy 12 (FFT spatial/autocorr/high-freq, GLCM x4, Kendall, bigram,
///         ncd); Spearman falls back to its O(n) path
///   >= 3: drop csize_delta (2 deflates/page)
///   >= 4: drop struct_entropy (windowed-entropy lumpiness)
pub fn page_metrics_mode(p: &[u8], q: &[u8], speed: u8) -> PageMetrics {
    let hamming = distance(p, q) as u32;
    if hamming == 0 {
        // Identical page = no change. Every channel is 0; skip all the work
        // (unchanged pages are the bulk of a real dump).
        return PageMetrics::default();
    }
    let sh = Shared::new(p, q, hamming);
    PageMetrics {
        a: family_a::compute(&sh, p, q, speed),
        b: family_b::compute(&sh, p, q, speed),
        c: family_c::compute(&sh, q, speed),
        d: family_d::compute(p, q, speed),
    }
}

/// One PageMetrics per page in the chunk, at speed `speed` (see `page_metrics_mode`).
pub fn process_chunk(chunk1: &[u8], chunk2: &[u8], speed: u8) -> Vec<PageMetrics> {
    let num_pages = chunk1.len() / PAGE_SIZE;
    (0..num_pages)
        .map(|i| {
            let start = i * PAGE_SIZE;
            let end = start + PAGE_SIZE;
            page_metrics_mode(&chunk1[start..end], &chunk2[start..end], speed)
        })
        .collect()
}

/// CSV column names, in row order (no trailing newline).
pub fn csv_header() -> &'static str {
    "hamming,cosine,l0,l1,l2,linf,mean_abs,gradient_mag,tv,chi2,hellinger,kl,js,wasserstein,bhattacharyya,hist_inter_dist,ent_delta,csize_delta,ncd,struct_ent_change,lz_change,pearson,ssim_struct,mean_shift,ssim_lum,median_shift,var_ratio,std_delta,ssim_contrast,range_delta,iqr_delta,polarity,sign_delta_ent,zero_mass_delta,spearman,kendall,cross_corr_lag,phase_corr,byte_rotation,ent_q,struct_ent_q,distinct_bytes,zero_frac,fill_frac,printable_frac,mean_q,var_q,skew_q,kurt_q,chi2_uniform,bigram_ent,autocorr_peak,changed_runs,change_span,change_centroid,longest_changed_run,change_density,edge_energy,glcm_contrast,glcm_homogeneity,glcm_energy,glcm_correlation,high_freq_frac,max_run_len"
}

/// One CSV row for a page (no trailing newline). Column order matches `csv_header`.
pub fn csv_row(m: &PageMetrics) -> String {
    let p = &m.a.positional;
    let d = &m.a.distributional;
    let i = &m.a.informational;
    let s = &m.b.structure;
    let l = &m.b.level;
    let sp = &m.b.spread;
    let po = &m.b.polarity;
    let dd = &m.b.dist_direction;
    let xc = &m.b.spatial;
    let c = &m.c;
    let cl = &m.d.change_location;
    let tx = &m.d.texture;
    format!(
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},\
         {},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},\
         {},{},{},{},{},{},{},{},{},{},{},{},{},\
         {},{},{},{},{},{},{},{},{},{},{},{}",
        p.hamming, s.cosine, p.l0, p.l1, p.l2, p.linf, p.mean_abs, p.gradient_mag,
        d.tv, d.chi2, d.hellinger, d.kl, d.js, d.wasserstein, d.bhattacharyya,
        d.hist_inter_dist, i.ent_delta, i.csize_delta, i.ncd, i.struct_ent_change,
        i.lz_change, s.pearson, s.ssim_struct, l.mean_shift, l.ssim_lum, l.median_shift,
        sp.var_ratio, sp.std_delta, sp.ssim_contrast, sp.range_delta, sp.iqr_delta,
        po.polarity, po.sign_delta_ent, dd.zero_mass_delta, s.spearman, s.kendall,
        xc.cross_corr_lag, xc.phase_corr, xc.byte_rotation,
        c.ent_q, c.struct_ent_q, c.distinct_bytes, c.zero_frac, c.fill_frac,
        c.printable_frac, c.mean_q, c.var_q, c.skew_q, c.kurt_q, c.chi2_uniform,
        c.bigram_ent, c.autocorr_peak,
        cl.changed_runs, cl.change_span, cl.change_centroid, cl.longest_changed_run,
        cl.change_density, tx.edge_energy, tx.glcm_contrast, tx.glcm_homogeneity,
        tx.glcm_energy, tx.glcm_correlation, tx.high_freq_frac, tx.max_run_len
    )
}
