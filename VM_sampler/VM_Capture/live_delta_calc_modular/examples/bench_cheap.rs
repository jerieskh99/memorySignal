//! Attribute the per-changed-page "cheap" cost: which supposedly-cheap metric is the
//! O(n^2) long pole? Run: cargo run --release --example bench_cheap

use live_delta_calc_modular::metrics::common::{csize, lz_complexity, struct_entropy};
use std::time::Instant;

fn main() {
    let n = 13107usize; // 5% of 262144 (1 GB)
    let mut x: u64 = 0x123456789;
    let pages: Vec<Vec<u8>> = (0..n)
        .map(|_| {
            (0..4096)
                .map(|_| {
                    x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    (x >> 33) as u8
                })
                .collect()
        })
        .collect();

    println!("cost per RANDOM 4 KB page, single-thread ({} pages):", n);

    let t = Instant::now();
    let mut s = 0u64;
    for p in &pages {
        s = s.wrapping_add(lz_complexity(p) as u64);
    }
    let d = t.elapsed();
    println!(
        "  lz_complexity   {:>9.1} us/page   ({:?} total)   [runs 2x/page]",
        d.as_micros() as f64 / n as f64,
        d
    );

    let t = Instant::now();
    for p in &pages {
        s = s.wrapping_add(csize(p) as u64);
    }
    let d = t.elapsed();
    println!(
        "  csize           {:>9.1} us/page   ({:?} total)   [runs 2x/page]",
        d.as_micros() as f64 / n as f64,
        d
    );

    let t = Instant::now();
    for p in &pages {
        s = s.wrapping_add(struct_entropy(p) as u64);
    }
    let d = t.elapsed();
    println!(
        "  struct_entropy  {:>9.1} us/page   ({:?} total)   [runs 2x/page]",
        d.as_micros() as f64 / n as f64,
        d
    );

    println!("(sink {})", s);
}
