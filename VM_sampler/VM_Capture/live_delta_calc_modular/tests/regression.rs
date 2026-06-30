//! Permanent regression tests for the metric values (the cases previously checked by
//! hand-run smoke). Run with `cargo test`.

use live_delta_calc_modular::metrics::{self, page_metrics, PageMetrics};

fn page(byte: u8) -> Vec<u8> {
    vec![byte; 4096]
}

#[test]
fn header_has_39_columns() {
    assert_eq!(metrics::csv_header().split(',').count(), 39);
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
