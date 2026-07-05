//! Family A / informational: compressibility / complexity change.

use crate::metrics::common::{csize, lz_complexity, struct_entropy, Shared};
use crate::metrics::PAGE_SIZE;

#[derive(Clone, Default, Debug)]
pub struct Informational {
    pub ent_delta: f32,         // H(q) - H(p)  (signed; bits/byte)
    pub csize_delta: f32,       // len(C(q)) - len(C(p))  (signed)
    pub ncd: f32,               // Normalized Compression Distance (0 = identical)
    pub struct_ent_change: f32, // change in windowed-entropy lumpiness
    pub lz_change: f32,         // lz(q) - lz(p)  (signed)
}

/// Shannon entropy (bits/byte) of a byte histogram. Same accumulation order (b = 0..256)
/// as the original combined loop, so `ent_delta` is bit-identical.
fn entropy(h: &[u32; 256]) -> f64 {
    let n = PAGE_SIZE as f64;
    let mut e = 0.0;
    for b in 0..256 {
        let pb = h[b] as f64 / n;
        if pb > 0.0 {
            e -= pb * pb.log2();
        }
    }
    e
}

pub fn compute(sh: &Shared, p: &[u8], q: &[u8], speed: u8) -> Informational {
    let hp_ent = entropy(&sh.hp);
    let hq_ent = entropy(&sh.hq);

    // csize_delta = 2 deflates (p, q), dropped at speed >= 3. ncd needs a 3rd deflate
    // (the p||q concat) and is dropped at speed >= 2.
    let (csize_delta, ncd) = if speed >= 3 {
        (0.0, 0.0)
    } else {
        let csz_p = csize(p);
        let csz_q = csize(q);
        let ncd = if speed >= 2 {
            0.0
        } else {
            let mut pq = Vec::with_capacity(p.len() + q.len());
            pq.extend_from_slice(p);
            pq.extend_from_slice(q);
            let csz_pq = csize(&pq);
            (csz_pq as f64 - csz_p.min(csz_q) as f64) / csz_p.max(csz_q).max(1) as f64
        };
        (csz_q as f64 - csz_p as f64, ncd)
    };

    // struct_ent_change = 2 struct_entropy passes, dropped at speed >= 4.
    let struct_ent_change = if speed >= 4 {
        0.0
    } else {
        (struct_entropy(q) - struct_entropy(p)) as f32
    };

    // LZ76 complexity is O(n^2) on random/incompressible pages (~5 ms/page, x2) -- the
    // dominant live cost. Dropped at speed >= 1; entropy + csize carry the complexity
    // signal. Kept only at speed 0 (offline, not time-constrained).
    let lz_change = if speed >= 1 {
        0.0
    } else {
        lz_complexity(q) as f32 - lz_complexity(p) as f32
    };

    Informational {
        ent_delta: (hq_ent - hp_ent) as f32,
        csize_delta: csize_delta as f32,
        ncd: ncd as f32,
        struct_ent_change,
        lz_change,
    }
}
