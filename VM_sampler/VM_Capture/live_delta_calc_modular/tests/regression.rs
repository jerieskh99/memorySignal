//! Permanent regression tests for the metric values (the cases previously checked by
//! hand-run smoke). Run with `cargo test`.

use live_delta_calc_modular::metrics::{self, page_metrics, PageMetrics};

fn page(byte: u8) -> Vec<u8> {
    vec![byte; 4096]
}

#[test]
fn header_column_count() {
    // 20 (A) + 19 (cosine + B) + 13 (C) + 12 (D) = 64
    assert_eq!(metrics::csv_header().split(',').count(), 64);
    // header and a row must have the same number of fields
    let m = page_metrics(&page(0x10), &page(0x20));
    assert_eq!(
        metrics::csv_row(&m).split(',').count(),
        metrics::csv_header().split(',').count()
    );
}

#[test]
fn identical_page_is_all_zero() {
    let p = page(0x42);
    let m = page_metrics(&p, &p);
    assert_eq!(m.hamming(), 0);
    assert_eq!(m.a.positional.l1, 0);
    // identical page short-circuits to the default (every channel 0)
    assert_eq!(metrics::csv_row(&m), metrics::csv_row(&PageMetrics::default()));
}

#[test]
fn const_shift_0x10_to_0x20() {
    let m = page_metrics(&page(0x10), &page(0x20));
    // Family A / positional (integer-exact)
    assert_eq!(m.a.positional.l0, 4096);
    assert_eq!(m.a.positional.l1, 65536);
    assert_eq!(m.a.positional.l2, 1024.0);
    assert_eq!(m.a.positional.linf, 16);
    assert_eq!(m.a.positional.mean_abs, 16.0);
    assert_eq!(m.a.positional.hamming, 8192); // popcount(0x10 ^ 0x20)=2 per byte * 4096
    // Family B / level + polarity
    assert_eq!(m.b.level.mean_shift, 16.0);
    assert_eq!(m.b.level.median_shift, 16.0);
    assert_eq!(m.b.polarity.polarity, 1.0); // every byte moved up
    assert_eq!(m.b.polarity.sign_delta_ent, 0.0); // coherent direction
    assert_eq!(m.b.structure.cosine, 0.0); // identical direction -> distance 0
    assert_eq!(m.b.spread.ssim_contrast, 1.0); // both constant
}

#[test]
fn distributional_0x00_to_0x01() {
    let m = page_metrics(&page(0x00), &page(0x01));
    let d = &m.a.distributional;
    assert!((d.tv - 1.0).abs() < 1e-6);
    assert!((d.chi2 - 2.0).abs() < 1e-6);
    assert!((d.hellinger - 1.0).abs() < 1e-6);
    assert!((d.js - 1.0).abs() < 1e-6);
    assert!((d.wasserstein - 1.0).abs() < 1e-6);
    assert!((d.hist_inter_dist - 1.0).abs() < 1e-6);
}

#[test]
fn ramp_to_reversed_is_anticorrelated() {
    let p: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
    let q: Vec<u8> = (0..4096).map(|i| (255 - (i % 256)) as u8).collect();
    let m = page_metrics(&p, &q);
    assert!((m.b.structure.pearson - (-1.0)).abs() < 1e-6, "pearson={}", m.b.structure.pearson);
    assert!((m.b.structure.kendall - (-0.996337)).abs() < 1e-4, "kendall={}", m.b.structure.kendall);
    assert!((m.b.structure.spearman - (-1.0)).abs() < 1e-6, "spearman={}", m.b.structure.spearman);
    assert_eq!(m.b.spread.var_ratio, 1.0);
    assert_eq!(m.b.polarity.sign_delta_ent, 1.0);
    assert_eq!(m.b.level.mean_shift, 0.0);
}

#[test]
fn informational_struct_entropy_and_lz() {
    // p = all zero; q = 2048 zeros then range(256)*8 (8 zero-windows + 8 random-windows)
    let p = page(0x00);
    let mut q: Vec<u8> = vec![0u8; 2048];
    for _ in 0..8 {
        q.extend(0..=255u8);
    }
    let m = page_metrics(&p, &q);
    assert_eq!(m.a.informational.struct_ent_change, 4.0); // std of {0 x8, 8 x8} = 4
    assert_eq!(m.a.informational.lz_change, 255.0); // lz(q)=257, lz(p)=2
    assert!(m.a.informational.ncd > 0.0 && m.a.informational.ncd <= 1.0);
    assert!(m.a.informational.csize_delta > 0.0);
}

#[test]
fn spatial_shift_detected() {
    // aperiodic page (LCG), circularly shifted right by 100
    let mut x: u32 = 1234567;
    let p: Vec<u8> = (0..4096)
        .map(|_| {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            (x >> 16) as u8
        })
        .collect();
    let q: Vec<u8> = (0..4096).map(|i| p[(i + 4096 - 100) % 4096]).collect();
    let m = page_metrics(&p, &q);
    assert_eq!(m.b.spatial.cross_corr_lag.abs(), 100.0, "lag={}", m.b.spatial.cross_corr_lag);
    assert_eq!(m.b.spatial.byte_rotation.abs(), 100.0, "rot={}", m.b.spatial.byte_rotation);
    assert!((m.b.spatial.phase_corr - 1.0).abs() < 1e-3, "phase_corr={}", m.b.spatial.phase_corr);
}

#[test]
fn content_all_zero_q() {
    // changed page (p != q) whose new content q is all 0x00
    let m = page_metrics(&page(0x01), &page(0x00));
    let c = &m.c;
    assert_eq!(c.distinct_bytes, 1);
    assert_eq!(c.zero_frac, 1.0);
    assert_eq!(c.fill_frac, 1.0);
    assert_eq!(c.ent_q, 0.0);
    assert_eq!(c.printable_frac, 0.0); // 0x00 is not printable
    assert_eq!(c.mean_q, 0.0);
    assert_eq!(c.var_q, 0.0);
    assert_eq!(c.bigram_ent, 0.0);
    assert_eq!(c.autocorr_peak, 0.0); // zero-energy guard
}

#[test]
fn content_uniform_q() {
    // q = every byte value 16x (uniform histogram, periodic period 256)
    let q: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
    let m = page_metrics(&page(0x00), &q);
    let c = &m.c;
    assert_eq!(c.distinct_bytes, 256);
    assert!((c.ent_q - 8.0).abs() < 1e-5, "ent_q={}", c.ent_q); // uniform -> 8 bits/byte
    assert!(c.chi2_uniform < 1e-3, "chi2_uniform={}", c.chi2_uniform); // perfectly uniform -> 0
    assert!((c.fill_frac - 16.0 / 4096.0).abs() < 1e-6);
    assert!((c.printable_frac - 95.0 / 256.0).abs() < 1e-4); // 0x20..=0x7E = 95 values
    assert!(c.autocorr_peak > 0.99, "autocorr_peak={}", c.autocorr_peak); // period 256
}

#[test]
fn change_location_full_and_local() {
    // full change: every byte differs
    let m = page_metrics(&page(0x10), &page(0x20));
    let cl = &m.d.change_location;
    assert_eq!(cl.changed_runs, 1);
    assert_eq!(cl.change_span, 4096);
    assert_eq!(cl.longest_changed_run, 4096);
    assert_eq!(cl.change_density, 1.0);
    assert!((cl.change_centroid - 2047.5).abs() < 1e-1);

    // localized: bytes 100..104 differ
    let p = page(0x00);
    let mut q = page(0x00);
    for x in q.iter_mut().take(104).skip(100) {
        *x = 0xFF;
    }
    let cl2 = page_metrics(&p, &q).d.change_location;
    assert_eq!(cl2.changed_runs, 1);
    assert_eq!(cl2.change_span, 4);
    assert_eq!(cl2.longest_changed_run, 4);
    assert!((cl2.change_centroid - 101.5).abs() < 1e-3);
}

#[test]
fn change_location_two_runs() {
    let p = page(0x00);
    let mut q = page(0x00);
    q[10] = 1;
    q[11] = 1; // run 1 (len 2)
    q[100] = 1;
    q[101] = 1;
    q[102] = 1; // run 2 (len 3)
    let cl = page_metrics(&p, &q).d.change_location;
    assert_eq!(cl.changed_runs, 2);
    assert_eq!(cl.longest_changed_run, 3);
    assert_eq!(cl.change_span, 93); // 102 - 10 + 1
}

#[test]
fn texture_constant_vs_alternating() {
    // constant q: max run = whole page, no edges, no high-freq, zero GLCM contrast
    let tx = page_metrics(&page(0x01), &page(0x10)).d.texture;
    assert_eq!(tx.max_run_len, 4096);
    assert_eq!(tx.edge_energy, 0.0);
    assert_eq!(tx.high_freq_frac, 0.0);
    assert_eq!(tx.glcm_contrast, 0.0);

    // alternating 0,255,0,255...: max edges, all energy at Nyquist, run length 1
    let p = page(0x00);
    let q: Vec<u8> = (0..4096).map(|i| if i % 2 == 0 { 0u8 } else { 255u8 }).collect();
    let tx2 = page_metrics(&p, &q).d.texture;
    assert_eq!(tx2.max_run_len, 1);
    assert!(tx2.edge_energy > 0.0);
    assert!((tx2.high_freq_frac - 1.0).abs() < 1e-3, "high_freq={}", tx2.high_freq_frac);
}
