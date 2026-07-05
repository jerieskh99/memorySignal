//! Micro-benchmark: deflate cost for the compression metrics, to decide whether
//! csize (1 deflate/page) and/or ncd (3 deflates/page) can ride along in the live
//! differ. Run: cargo run --release --example bench_compress

use flate2::write::ZlibEncoder;
use flate2::Compression;
use std::io::Write;
use std::time::Instant;

fn csize(data: &[u8]) -> usize {
    let mut e = ZlibEncoder::new(Vec::new(), Compression::default());
    e.write_all(data).unwrap();
    e.finish().unwrap().len()
}

fn main() {
    let n = 13107usize; // ~5% of 262144 pages (1 GB)
    // deterministic pseudo-random 4 KB pages (incompressible ~ worst case for deflate work)
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

    // 1 deflate per page (csize)
    let t = Instant::now();
    let mut s1 = 0usize;
    for p in &pages {
        s1 += csize(p);
    }
    let d1 = t.elapsed();

    // 3 deflates per page (ncd: csize(p) + csize(q) + csize(p||q)), p,q = consecutive pages
    let t = Instant::now();
    let mut s3 = 0usize;
    for w in pages.windows(2) {
        s3 += csize(&w[0]);
        s3 += csize(&w[1]);
        let mut pq = Vec::with_capacity(8192);
        pq.extend_from_slice(&w[0]);
        pq.extend_from_slice(&w[1]);
        s3 += csize(&pq);
    }
    let d3 = t.elapsed();

    println!("{} changed pages (5% of 1 GB), single-threaded:", n);
    println!(
        "  csize  (1 deflate):  {:?}   ({:.1} us/page)",
        d1,
        d1.as_micros() as f64 / n as f64
    );
    println!(
        "  ncd    (3 deflates): {:?}   ({:.1} us/page)",
        d3,
        d3.as_micros() as f64 / (n - 1) as f64
    );
    println!();
    println!("  Live budget per snapshot pair = 500 ms (the cadence).");
    println!("  Estimated wall time on ~16 threads (optimistic /16):");
    println!("    csize ~ {:?}   |   ncd ~ {:?}", d1 / 16, d3 / 16);
    println!("  (sinks {} {})", s1, s3);
}
