//! Family D / texture: the spatial texture of the new page q.

use crate::metrics::common::{glcm_haralick, grad_energy, high_freq_frac};

#[derive(Clone, Default, Debug)]
pub struct Texture {
    pub edge_energy: f32,       // sum |q[j+1]-q[j]|: local roughness
    pub glcm_contrast: f32,     // Haralick contrast
    pub glcm_homogeneity: f32,  // Haralick homogeneity
    pub glcm_energy: f32,       // Haralick energy (angular second moment)
    pub glcm_correlation: f32,  // Haralick correlation
    pub high_freq_frac: f32,    // fraction of spectral energy in the high band
    pub max_run_len: u16,       // longest constant-byte run
}

pub fn compute(q: &[u8], speed: u8) -> Texture {
    // GLCM (256x256 co-occurrence, 2 sweeps) and the FFT high-frequency fraction are
    // dropped at speed >= 2; edge_energy and max_run_len stay (both O(n), cheap).
    let (contrast, homog, energy, corr) = if speed >= 2 {
        (0.0, 0.0, 0.0, 0.0)
    } else {
        glcm_haralick(q)
    };

    let mut max_run: u16 = if q.is_empty() { 0 } else { 1 };
    let mut cur: u16 = 1;
    for w in q.windows(2) {
        if w[0] == w[1] {
            cur += 1;
            if cur > max_run {
                max_run = cur;
            }
        } else {
            cur = 1;
        }
    }

    Texture {
        edge_energy: grad_energy(q) as f32,
        glcm_contrast: contrast as f32,
        glcm_homogeneity: homog as f32,
        glcm_energy: energy as f32,
        glcm_correlation: corr as f32,
        high_freq_frac: if speed >= 2 { 0.0 } else { high_freq_frac(q) },
        max_run_len: max_run,
    }
}
