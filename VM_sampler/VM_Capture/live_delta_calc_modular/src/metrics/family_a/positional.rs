//! Family A / positional: magnitude in byte/bit-position space (alternatives to Hamming).

use crate::metrics::common::{grad_energy, Shared};
use crate::metrics::PAGE_SIZE;

#[derive(Clone, Default, Debug)]
pub struct Positional {
    pub hamming: u32,  // bits flipped (popcount of p XOR q)
    pub l0: u16,       // byte-Hamming: number of CHANGED bytes (0..=4096)
    pub l1: u32,       // SAD: sum of absolute byte differences
    pub l2: f32,       // Euclidean: sqrt(sum of squared byte differences)
    pub linf: u8,      // Chebyshev: the single biggest byte jump (0..=255)
    pub mean_abs: f32, // L1 / 4096: average per-byte change
    pub gradient_mag: f32, // grad_energy(q) - grad_energy(p): change in within-page roughness
}

pub fn compute(sh: &Shared, p: &[u8], q: &[u8]) -> Positional {
    Positional {
        hamming: sh.hamming,
        l0: sh.l0,
        l1: sh.l1 as u32,
        l2: (sh.sse as f32).sqrt(),
        linf: sh.linf,
        mean_abs: sh.l1 as f32 / PAGE_SIZE as f32,
        gradient_mag: grad_energy(q) as f32 - grad_energy(p) as f32,
    }
}
