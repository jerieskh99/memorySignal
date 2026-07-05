//! Break down the remaining --fast cost per changed page: which of the 51 kept metrics
//! are the next poles after lz? Run: cargo run --release --example bench_fast_breakdown

use distances::vectors::cosine;
use hamming::distance;
use live_delta_calc_modular::metrics::common::{csize, struct_entropy, Shared};
use live_delta_calc_modular::metrics::page_metrics_mode;
use std::time::Instant;

fn rand_pages(n: usize, seed: u64) -> Vec<Vec<u8>> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            (0..4096)
                .map(|_| {
                    x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    (x >> 33) as u8
                })
                .collect()
        })
        .collect()
}

fn main() {
    let n = 13107usize;
    let pp = rand_pages(n, 0x1111);
    let qq = rand_pages(n, 0x9999);

    macro_rules! bench {
        ($name:expr, $body:expr) => {{
            let t = Instant::now();
            let mut s = 0u64;
            for i in 0..n {
                s = s.wrapping_add($body(&pp[i], &qq[i]));
            }
            let d = t.elapsed();
            println!(
                "  {:22} {:>8.1} us/page   ({:?})",
                $name,
                d.as_micros() as f64 / n as f64,
                d
            );
            s
        }};
    }

    println!("--fast path, per changed page, single-thread ({} pages):", n);
    let _ = bench!("FULL fast page_metrics", |p: &Vec<u8>, q: &Vec<u8>| {
        let m = page_metrics_mode(p, q, 2);
        m.hamming() as u64
    });
    println!("  --- components ---");
    let _ = bench!("csize x2 (p,q)", |p: &Vec<u8>, q: &Vec<u8>| {
        (csize(p) + csize(q)) as u64
    });
    let _ = bench!("struct_entropy x3", |p: &Vec<u8>, q: &Vec<u8>| {
        (struct_entropy(q) + struct_entropy(p) + struct_entropy(q)) as u64
    });
    let _ = bench!("cosine (alloc+convert)", |p: &Vec<u8>, q: &Vec<u8>| {
        let pf: Vec<f32> = p.iter().map(|&x| x as f32).collect();
        let qf: Vec<f32> = q.iter().map(|&x| x as f32).collect();
        let c: f32 = cosine(&pf, &qf);
        c as u64
    });
    let _ = bench!("Shared::new (loop+hist)", |p: &Vec<u8>, q: &Vec<u8>| {
        let h = distance(p, q) as u32;
        Shared::new(p, q, h).l1
    });
}
